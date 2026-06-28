"""Adaptive (error-controlled step-doubling) MuJoCo solver: ``SolverMuJoCoAdaptive``.

The manager hands the solver a state and a control boundary period ``dt_outer``; it
marches every world to that boundary with a per-world adaptive inner timestep and
writes the advanced state back. Accuracy comes from **step doubling**: each attempt
integrates one full step at ``dt`` and two half steps at ``dt/2`` and uses their
difference as a local-error estimate, which a Drake step-size controller turns into a
per-world accept/reject + grow/shrink decision -- entirely on the GPU.

The ragged ``step`` machine (the per-iteration body is captured once and replayed; the
loop stops the moment every world reaches its boundary)::

    next_time[w] = sim_time[w] + dt_outer                # boundary target per world
    for _ in range(max_substeps):                        # max_substeps is a SAFETY cap
        clamp_dt_to_boundary(dt, sim_time, next_time)    # done worlds -> dt=0; never overshoot
        snapshot state_cur                               # rollback target on reject
        full   = substep(state_cur, dt)                  # \
        mid    = substep(state_cur, dt/2)                #  } step doubling (3 MuJoCo evals)
        double = substep(mid,       dt/2)                # /
        e = infnorm(full, double)                        # per-world local error = max|Δq|
        _calc_adjusted_step(e, ...):                     # per-thread Drake controller:
            ACCEPT (e<=tol): commit=double; sim_time+=dt; grow dt
            REJECT (e>tol):  hold state_cur; shrink dt; retry
        state_cur = commit ? double : state_saved        # masked commit (NaN-safe via _commit)
        apply_dt_cap(ideal_dt -> dt)                     # clamp next attempt to [dt_min, dt_max]
        boundary_flag = any(sim_time < next_time)        # ONE 4-byte host read per iteration
        if boundary_flag == 0:  break
    write state_cur back

dt is ALWAYS per-world: each world adapts its OWN dt from its OWN error, so ``P(s'|s,a)``
for one world never depends on another (the Markov property the RL gradient requires). A
shared/global worst-case dt is deliberately NOT supported -- it would couple worlds.

The per-world dt tiles the control interval via the ``"ragged"`` adaptive boundary loop
sketched above, with a clamped-remainder landing (:meth:`_run_iteration_body`).

The ONLY host sync in the ragged step path is the single 4-byte boundary-flag read per
iteration (~3 iters/frame). The controller
kernels (Drake step sizing, the inf-norm error metric, the masked select, the time
rebase/clamp) live in the shared :mod:`..adaptive.controller_kernels` and are re-imported
below so this file's tests resolve them unchanged.

Note: true CENIC = this adaptive controller + convex ICF contact; the ICF contact model
is not yet built, so this is the adaptive (pseudo-CENIC) MuJoCo solver.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...utils.benchmark import event_scope
from ..adaptive.controller_kernels import (  # noqa: F401  (re-exported for tests + reuse)
    _DRAKE_HYSTERESIS_HIGH,
    _DRAKE_HYSTERESIS_LOW,
    _DRAKE_MAX_GROW,
    _DRAKE_MIN_SHRINK,
    _DRAKE_SAFETY,
    _advance_sim_time,
    _apply_dt_cap,
    _boundary_advance,
    _boundary_check,
    _boundary_reset,
    _calc_adjusted_step,
    _clamp_dt_to_boundary,
    _finish_diverged_worlds,
    _inf_norm_state_error_kernel,
    _iter_count_increment,
    _rebase_time,
    _reset_worlds,
    _select_float_kernel,
    _select_spatial_vector_kernel,
    _select_transform_kernel,
    _status_sentinel_reset,
    _status_summary_kernel,
)
from .solver_mujoco import SolverMuJoCo


class SolverMuJoCoAdaptive(SolverMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The ragged boundary
    loop replays one captured iteration body on CUDA when possible, checking a
    4-byte flag via ``.numpy()`` to detect when all worlds have reached the
    target time.

    Timesteps are managed internally by the error controller.  Set the
    initial value via ``dt_inner_init`` and query current values via
    :attr:`dt`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoAdaptive(model, tol=1e-3)
        state_0, state_1 = model.state(), model.state()

        while viewer.is_running():
            state_0, state_1 = solver.step_dt(DT, state_0, state_1, control, apply_forces=viewer.apply_forces)
            viewer.render(state_0, solver.sim_time.numpy().min())
    """

    def __init__(
        self,
        model: Model,
        *,
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-6,
        dt_inner_max: float | None = None,
        dt_mode: str = "per_world",
        tiling: str = "ragged",
        max_substeps: int = 256,
        **kwargs,
    ):
        """
        Args:
            model: The model to simulate.
            tol: Inf-norm error tolerance on joint_q per world [m or rad, depending on joint type].
                Error is ``max|Δq|`` between the full step and the doubled half-step.
                Worlds with error > tol are rejected and retry with a smaller dt.
            dt_inner_init: Initial inner (adaptive physics) timestep [s].
            dt_inner_min: Minimum allowed inner timestep [s].
            dt_inner_max: Maximum allowed inner timestep [s].  If None, clamped
                to the ``dt_outer`` argument of each :meth:`step_dt` call
                automatically so the inner step never overshoots the boundary.
            dt_mode: ``"per_world"`` -- each world picks its own dt from its own error.
                The only supported mode: per-world dt keeps each world's transition
                independent of the others, which the RL gradient requires (Markov).
            tiling: ``"ragged"`` (the only supported mode; adaptive dt with a clamped
                remainder landing). ``"even"`` tiling was removed.
            max_substeps: Hard upper bound on inner adaptive attempts per control interval. The
                loop stops after this many iterations and exposes any lag through ``sim_time``.
                Bounds worst-case work when a world's ideal_dt collapses to the dt_min floor.
            **kwargs: Forwarded to :class:`SolverMuJoCo`.
        """
        if dt_mode != "per_world":
            raise ValueError(
                f"dt_mode must be 'per_world' (the only supported mode; 'global' was removed -- a "
                f"shared worst-case dt makes one world's transition depend on others, breaking the "
                f"per-world Markov property the RL gradient requires), got {dt_mode!r}"
            )
        if tiling != "ragged":
            raise ValueError(
                f"tiling must be 'ragged' (the only supported mode; 'even' tiling was removed), got {tiling!r}"
            )
        if int(max_substeps) < 1:
            raise ValueError(f"max_substeps must be >= 1, got {max_substeps!r}")
        # Contacts come from MuJoCo's native collision pipeline (run_collision_detection=True);
        # each step-doubling substep re-collides via mujoco_warp, so MuJoCo sizes its own contact
        # buffers and there is no separate Newton collision pass to feed in.
        super().__init__(model, separate_worlds=True, use_mujoco_cpu=False, use_mujoco_contacts=True, **kwargs)

        world_count = model.world_count
        device = model.device

        # ---- per-world controller clocks + timestep (the dt VECTOR is the primitive) ----
        self._dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(world_count, dt_inner_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(world_count, dtype=wp.bool, device=device)
        # _commit: write the doubled state (vs hold last good). _diverged: latch for
        # worlds that hit the dt_min floor still non-finite -- read by the env to reset them.
        self._commit = wp.zeros(world_count, dtype=wp.bool, device=device)
        self._diverged = wp.zeros(world_count, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(world_count, dtype=wp.float32, device=device)

        # ---- controller scalars / bounds ----
        self._tol = float(tol)
        # Fix C: construction default for the per-world controller reset, plus a
        # reusable all-True mask for a full reset (world_mask=None path).
        self._dt_inner_init = float(dt_inner_init)
        self._full_world_mask = wp.full(world_count, True, dtype=wp.bool, device=device)
        self._dt_min = float(dt_inner_min)
        # Error sentinel for NaN/inf states is 1e10 (see _inf_norm_state_error_kernel);
        # anything at/above this threshold is treated as a diverged world.
        self._divergence_threshold = float(1.0e9)
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")

        # ---- configuration (see module docstring) ----
        self._dt_mode = dt_mode  # "per_world" only (global removed: a shared dt couples worlds)
        self._tiling = tiling  # "ragged" only ("even" removed)
        self._max_substeps = int(max_substeps)

        # ---- step-doubling scratch states: full @ dt, and mid -> double @ dt/2 ----
        self._scratch_full = model.state()
        self._scratch_mid = model.state()
        self._scratch_double = model.state()

        # Working state marched across iterations + its rollback snapshot.
        self._state_cur = model.state()
        self._state_saved = model.state()

        self._coords_per_world = model.joint_coord_count // world_count
        self._dofs_per_world = model.joint_dof_count // world_count
        self._bodies_per_world = model.body_count // world_count

        # ---- boundary-loop bookkeeping ----
        self._next_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)
        # Non-resetting cumulative boundary-loop iteration count (NOT zeroed per
        # step_dt, unlike _iteration_count_buf). Each iteration runs the 3-eval
        # step-doubling attempt, so total MuJoCo opt-steps = iterations * 3, and
        # rejected attempts are counted (a rejection is just another iteration).
        # Used as the compute axis for work-precision (V1). Reset with
        # reset_compute_counter().
        self._cum_iters = wp.zeros(1, dtype=wp.int32, device=device)

        # Stable buffer for opt.timestep; updated via wp.copy() per substep.
        self._timestep_buf = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self.mjw_model.opt.timestep = self._timestep_buf

        # Adaptive-controller accuracy-metric scaling S (Sec. V-E): e = || S (q - q̂) ||_inf. The paper
        # gives no formula for S and specifies NO mass weighting, clipping, or
        # normalization -- "S can be estimated from knowledge of coordinate types or
        # specified by expert users." Per PI directive (project_s_removed_identity),
        # S = identity. To use expert per-coordinate scales, overwrite self._state_scale
        # (shape [world_count, coords_per_world]) after construction.
        self._state_scale = wp.array(
            np.ones((world_count, self._coords_per_world), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        # ---- solver-internal CUDA-graph capture ----
        # Ragged tiling captures one adaptive iteration body keyed by effective dt_max and still
        # reads the 4-byte boundary flag between iterations. Gated by NEWTON_MJ_ADAPTIVE_GRAPH
        # (default on) + CUDA device; capture is warmed on the first frame.
        import os as _os

        try:
            _is_cuda = bool(wp.get_device(device).is_cuda)
        except Exception:
            _is_cuda = False
        self._graph_enabled = _is_cuda and _os.environ.get("NEWTON_MJ_ADAPTIVE_GRAPH", "1") != "0"
        self._ragged_graph_cache: dict = {}
        self._ragged_graph_warmed = False

    # =====================================================================
    # Phase: one inner MuJoCo physics step
    # =====================================================================
    def substep(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt_array: wp.array,
    ) -> None:
        """ONE inner MuJoCo physics step (= CENIC ``ComputeNextContinuousState``): sync state_in,
        set timestep, step, write state_out. Already NON-conditional (no ``wp.capture_*``), so it
        records as a flat launch stream and is safe inside :meth:`step`'s single-level capture.

        ``control`` / ``contacts`` are accepted for the unified Newton substep signature but are
        UNUSED: control is pre-applied to ``mjw_data`` once per boundary, and MuJoCo runs its own
        collision detection inside each ``_mujoco_warp_step`` (``use_mujoco_contacts=True``).
        ``dt_array`` is the per-world timestep ``wp.array``.
        """
        self._update_mjc_data(self.mjw_data, self.model, state_in)
        wp.copy(self.mjw_model.opt.timestep, dt_array)

        with wp.ScopedDevice(self.model.device):
            self._mujoco_warp_step()

        self._update_newton_state(self.model, state_out, self.mjw_data, state_prev=state_in)

    # =====================================================================
    # Adaptive-core helpers (shared by every iteration body and the diagnostic step)
    # =====================================================================
    @staticmethod
    def _copy_state(dst: State, src: State) -> None:
        """Copy joint_q/qd (and body_q/qd when both states carry them) src -> dst.

        Used to load the incoming state, snapshot for rollback, and write the result back.
        """
        wp.copy(dst.joint_q, src.joint_q)
        wp.copy(dst.joint_qd, src.joint_qd)
        if src.body_q is not None and dst.body_q is not None:
            wp.copy(dst.body_q, src.body_q)
        if src.body_qd is not None and dst.body_qd is not None:
            wp.copy(dst.body_qd, src.body_qd)

    def _step_double(self, state_in: State) -> None:
        """Step doubling -- the 3 MuJoCo evals: one full step at ``dt`` into ``_scratch_full``,
        then two half steps at ``dt/2`` (``state_in -> _scratch_mid -> _scratch_double``).
        ``_scratch_full`` vs ``_scratch_double`` is the Richardson pair the error kernel differences.
        """
        self.substep(state_in, self._scratch_full, None, None, self._dt)
        self.substep(state_in, self._scratch_mid, None, None, self._dt_half)
        self.substep(self._scratch_mid, self._scratch_double, None, None, self._dt_half)

    def _estimate_error(self) -> None:
        """Per-world local error: inf-norm ``e = max|Δq|`` between the full step and the doubled
        half-step (NaN/inf collapse to a 1e10 sentinel). Writes ``_last_error``."""
        wp.launch(
            _inf_norm_state_error_kernel,
            dim=self.model.world_count,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._state_scale,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=self.model.device,
        )

    def _select_committed_state(self, candidate: State, fallback: State, out: State) -> None:
        """Masked commit: write ``candidate`` into ``out`` for committed worlds, ``fallback`` for
        the rest. The ``_commit`` mask (NOT ``_accepted``) gates the write so a floor-diverged
        world that still advances time holds its last good state instead of writing NaN.
        """
        model = self.model
        dev = model.device
        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[candidate.joint_q, fallback.joint_q, self._commit, self._coords_per_world],
            outputs=[out.joint_q],
            device=dev,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[candidate.joint_qd, fallback.joint_qd, self._commit, self._dofs_per_world],
            outputs=[out.joint_qd],
            device=dev,
        )
        if out.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[candidate.body_q, fallback.body_q, self._commit, self._bodies_per_world],
                outputs=[out.body_q],
                device=dev,
            )
        if out.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[candidate.body_qd, fallback.body_qd, self._commit, self._bodies_per_world],
                outputs=[out.body_qd],
                device=dev,
            )

    # =====================================================================
    # Per-frame iteration bodies (the captured/replayed substep bodies)
    # =====================================================================
    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """ONE ragged adaptive iteration: clamp -> step-double -> error -> Drake controller ->
        masked commit -> advance -> dt cap -> boundary check.

        This is the body captured once and replayed per iteration of the ragged boundary loop
        (see :meth:`step`). Every phase is a flat kernel-launch sequence, so the whole body records
        as a single CUDA graph; the only host sync is the 4-byte boundary-flag read between replays.
        """
        n = self.model.world_count
        dev = self.model.device

        # Count this attempt (per-step + cumulative). A rejection is just another iteration.
        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        # Never overshoot the boundary; worlds already at it get dt=0 (no-op step).
        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
            device=dev,
        )

        # Snapshot for rollback on rejection.
        self._copy_state(self._state_saved, self._state_cur)

        # --- adaptive core: step double, estimate error, run the controller ---
        self._step_double(self._state_cur)
        self._estimate_error()
        wp.launch(
            _calc_adjusted_step,
            dim=n,
            inputs=[
                self._last_error,
                self._dt,
                self._ideal_dt,
                self._accepted,
                self._commit,
                self._diverged,
                self._tol,
                self._dt_min,
                self._divergence_threshold,
            ],
            device=dev,
        )

        # Commit the doubled state for committed worlds; hold last good for the rest.
        self._select_committed_state(self._scratch_double, self._state_saved, self._state_cur)

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=dev,
        )
        # Persistently-diverged worlds (held at the floor) jump straight to their boundary so
        # the loop terminates instead of grinding dt_min steps on a world that can't progress.
        wp.launch(
            _finish_diverged_worlds,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._diverged],
            device=dev,
        )

        # Size dt for the next attempt, then test whether all worlds have landed.
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=dev,
        )
        wp.launch(_boundary_reset, dim=1, inputs=[self._boundary_flag], device=dev)
        wp.launch(
            _boundary_check,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )

    # =====================================================================
    # Diagnostic single-attempt step (NOT the boundary call)
    # =====================================================================
    @event_scope
    def _step_once(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
    ) -> State:
        """Advance each world by one adaptive attempt (test/diagnostic helper, NOT the boundary call).

        Single-iteration path: one 3-eval attempt, controller update, select. Does not loop to a
        boundary -- use :meth:`step` for a real march.

        Renamed from the old single-attempt ``step()`` so the boundary call (formerly
        ``step_dt``) can take the canonical ``step()`` name without clashing.

        Args:
            state_in: Input state.
            state_out: Output state (written in place).
            control: Control inputs.
            contacts: Unused. MuJoCo runs its own collision detection each substep.

        Returns:
            state_out
        """
        model = self.model
        device = model.device
        n = model.world_count

        self._diverged.fill_(False)
        self._apply_mjc_control(model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        self._step_double(state_in)
        self._estimate_error()
        wp.launch(
            _calc_adjusted_step,
            dim=n,
            inputs=[
                self._last_error,
                self._dt,
                self._ideal_dt,
                self._accepted,
                self._commit,
                self._diverged,
                self._tol,
                self._dt_min,
                self._divergence_threshold,
            ],
            device=device,
        )
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, self._dt_max, self._dt, self._dt_half],
            device=device,
        )

        self._select_committed_state(self._scratch_double, state_in, state_out)

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=device,
        )

        self._step += 1
        return state_out

    # =====================================================================
    # The boundary call: march every world to dt_outer
    # =====================================================================
    @event_scope
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts | None = None,
        dt: float | None = None,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Advance every world by exactly ``dt`` (= ``dt_outer``) seconds of sim time (= CENIC
        ``DoStep`` + the N-substep march): the boundary call.

        Newton solver signature ``(state_in, state_out, control, contacts, dt)``. ``state_in`` is
        read and written in place; ``state_out`` is returned unchanged (scratch). ``contacts`` is
        accepted for signature uniformity but UNUSED -- MuJoCo runs its own collision detection
        inside each step-doubling substep (``use_mujoco_contacts=True``).

        Ragged tiling: adaptive boundary loop with a graph-captured iteration body when available
        and a 4-byte ``.numpy()`` boundary-flag read-back per iteration.
        """
        if dt is None:
            raise ValueError("SolverMuJoCoAdaptive.step requires dt (the outer boundary period).")
        state_0 = state_in
        state_1 = state_out
        dt_outer = float(dt)
        device = self.model.device
        n = self.model.world_count

        effective_dt_max = min(self._dt_max, dt_outer)

        # Seed this frame's per-world working dt from the carried ideal_dt, clamped to bounds.
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=device,
        )

        # Load the incoming state into the working buffer.
        self._copy_state(self._state_cur, state_0)

        self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
        if apply_forces is not None:
            apply_forces(state_0)

        self._enable_rne_postconstraint(self._state_cur)

        # Fix B: rebase both clocks by the per-world boundary so float32 magnitude stays bounded
        # (prevents landing-remainder precision loss / dt jitter that grows over a run). The
        # subtract-baseline preserves the remaining time exactly; do this BEFORE advancing next_time.
        wp.launch(_rebase_time, dim=n, inputs=[self._sim_time, self._next_time], device=device)
        wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)
        # Latch reset per outer step: _diverged reflects worlds that hit the floor non-finite
        # during THIS step; the env reads it afterward to reset them.
        self._diverged.fill_(False)

        self._march_ragged(effective_dt_max)

        # Write the advanced working state back into the caller's state.
        self._copy_state(state_0, self._state_cur)

        return state_0, state_1

    def _march_ragged(self, effective_dt_max: float) -> None:
        """Ragged tiling (default): replay the adaptive iteration body until the 4-byte boundary
        flag reports every world has landed, capped at ``max_substeps``."""
        for _ in range(self._max_substeps):
            self._run_ragged_iteration(effective_dt_max)
            if self._boundary_flag.numpy()[0] == 0:
                break

    # --------------------------------------------------------------- step_dt (alias)
    def step_dt(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Backward-compatible alias for :meth:`step` (old ``(dt, s0, s1, control)`` order).

        ``step`` is the canonical boundary call (Newton ``(state_in, state_out, control,
        contacts, dt)`` signature); this thin wrapper preserves the legacy call sites/tests.
        """
        return self.step(state_0, state_1, control, None, dt_outer, apply_forces=apply_forces)

    # =====================================================================
    # CUDA-graph capture of the iteration bodies
    # =====================================================================
    def _ragged_iteration_graph(self, effective_dt_max: float):
        """Return the captured ragged-iteration-body graph (keyed by effective_dt_max), or
        ``None`` to run eagerly. The first iteration runs eagerly so MuJoCo/Warp can lazily
        initialize allocations OUTSIDE capture; capture failures disable capture permanently."""
        if not self._graph_enabled:
            return None

        if not self._ragged_graph_warmed:
            self._ragged_graph_warmed = True
            return None

        key = round(float(effective_dt_max), 12)
        graph = self._ragged_graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._run_iteration_body(effective_dt_max)
                graph = cap.graph
                self._ragged_graph_cache[key] = graph
            except Exception:
                self._graph_enabled = False
                self._ragged_graph_cache.clear()
                return None
        return graph

    def _run_ragged_iteration(self, effective_dt_max: float) -> None:
        """Run one ragged iteration: replay the captured body if available, else run eagerly.
        A capture/launch failure falls back to eager so a run never crashes on a graph error."""
        graph = self._ragged_iteration_graph(effective_dt_max)
        if graph is None:
            self._run_iteration_body(effective_dt_max)
            return

        try:
            wp.capture_launch(graph)
        except Exception:
            self._graph_enabled = False
            self._ragged_graph_cache.clear()
            self._run_iteration_body(effective_dt_max)

    # =====================================================================
    # Reset
    # =====================================================================
    @override
    def reset(
        self,
        state,
        world_mask: wp.array | None = None,
        flags=None,
    ) -> None:
        """Restore per-world adaptive-controller state for reset worlds (Fix C).

        Overrides :meth:`SolverMuJoCo.reset` (which clears MuJoCo warm-start
        buffers and, per ``flags``, resets joint state to model defaults) and
        ADDITIONALLY restores this controller's persistent per-world buffers
        (ideal_dt/dt/dt_half/sim_time/next_time + the accepted/diverged latches)
        to construction defaults, so pre-reset controller state never leaks into
        the post-reset (s,a)->s' map. Also the consumer of Fix A's ``diverged``
        latch: passing ``world_mask=self.diverged`` clears flagged worlds.

        Pass ``flags=0`` (StateFlags none) to keep the env's randomized post-reset
        joint state instead of resetting joint_q/joint_qd to model defaults.
        """
        super().reset(state, world_mask=world_mask, flags=flags)
        mask = self._full_world_mask if world_mask is None else world_mask
        wp.launch(
            _reset_worlds,
            dim=self.model.world_count,
            inputs=[
                mask,
                self._dt_inner_init,
                self._ideal_dt,
                self._dt,
                self._dt_half,
                self._sim_time,
                self._next_time,
                self._diverged,
                self._accepted,
            ],
            device=self.model.device,
        )

    # =====================================================================
    # Telemetry / properties
    # =====================================================================
    @property
    def diverged(self) -> wp.array:
        """Per-world divergence latch from the most recent step, shape ``[world_count]``, bool, on device.

        ``True`` for a world that hit the ``dt_min`` floor with a non-finite state: the solver
        held its last good state instead of writing NaN. The env should reset these worlds.
        """
        return self._diverged

    @property
    def iteration_count(self) -> wp.array:
        """Iteration count from the most recent ``step_dt``, shape ``[1]``, int32, on device."""
        return self._iteration_count_buf

    @property
    def cumulative_iterations(self) -> wp.array:
        """Boundary-loop iterations accumulated since the last :meth:`reset_compute_counter`,
        shape ``[1]``, int32, on device. Includes rejected attempts. Read with ``.numpy()``
        OUTSIDE the inner loop only (it is a device sync)."""
        return self._cum_iters

    def cumulative_substeps(self) -> int:
        """Total MuJoCo opt-steps since the last :meth:`reset_compute_counter` (= iterations * 3
        for the step-doubling 3-eval). Compute axis for work-precision. Host sync; call outside
        the hot path."""
        return int(self._cum_iters.numpy()[0]) * 3

    def reset_compute_counter(self) -> None:
        """Zero the cumulative iteration/substep counter."""
        self._cum_iters.fill_(0)

    @property
    def sim_time(self) -> wp.array:
        """Per-world simulation time [s], shape ``[world_count]``, float32, on device.

        Only advances for accepted steps. Rebased to its outer boundary at the start of
        each :meth:`step_dt` (Fix B float32 time-rebase), so this is the time WITHIN the
        current outer interval (``~[0, dt_outer]``), not absolute cumulative time. A
        consumer needing absolute time must accumulate ``dt_outer`` itself.
        """
        return self._sim_time

    @property
    def dt(self) -> wp.array:
        """Current per-world timestep [s], shape ``[world_count]``, float32, on device."""
        return self._dt

    @property
    def tiling(self) -> str:
        """Substep tiling mode: always ``"ragged"`` (``"even"`` tiling was removed)."""
        return self._tiling

    @property
    def last_error(self) -> wp.array:
        """Inf-norm state error from the most recent accepted step, shape ``[world_count]``, float32, on device."""
        return self._accepted_error

    @property
    def last_raw_error(self) -> wp.array:
        """Inf-norm state error from the most recent attempt (accepted or rejected), shape ``[world_count]``, float32, on device."""
        return self._last_error

    @property
    def accepted(self) -> wp.array:
        """Per-world accept flags from the most recent step, shape ``[world_count]``, bool, on device."""
        return self._accepted

    def get_status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary via one GPU transfer."""
        device = self.model.device
        n = self.model.world_count

        wp.launch(_status_sentinel_reset, dim=1, inputs=[self._status_scalars], device=device)
        wp.launch(
            _status_summary_kernel,
            dim=n,
            inputs=[self._sim_time, self._accepted_error, self._dt, self._accepted, self._status_scalars],
            device=device,
        )

        scalars = self._status_scalars.numpy()
        return {
            "sim_time_min": float(scalars[0]),
            "sim_time_max": float(scalars[1]),
            "error_max": float(scalars[2]),
            "accept_count": int(scalars[3]),
            "dt_min": float(scalars[4]),
            "dt_max": float(scalars[5]),
        }
