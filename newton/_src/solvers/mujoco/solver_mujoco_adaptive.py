"""Adaptive (error-controlled step-doubling) MuJoCo solver.

Per-world adaptive time-stepping via step doubling.  The boundary loop
issues direct ``wp.launch()`` calls each iteration; no CUDA-graph capture.
Step controller follows Drake's CalcAdjustedStepSize.

Note: true CENIC = this adaptive controller + convex ICF contact; the ICF
contact model is not yet built, so this is the adaptive (pseudo-CENIC) solver.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton

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
    _broadcast_error,
    _calc_adjusted_step,
    _calc_even_step,
    _clamp_dt_to_boundary,
    _finish_diverged_worlds,
    _inf_norm_state_error_kernel,
    _iter_count_increment,
    _rebase_time,
    _reduce_max_error,
    _reset_error_scalar,
    _reset_worlds,
    _select_even_dt,
    _select_float_kernel,
    _select_spatial_vector_kernel,
    _select_transform_kernel,
    _set_uniform_even_dt,
    _status_sentinel_reset,
    _status_summary_kernel,
)
from .solver_mujoco import SolverMuJoCo



class SolverMuJoCoAdaptive(SolverMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The boundary loop
    launches kernels directly via ``wp.launch()`` each iteration, checking
    a 4-byte flag via ``.numpy()`` to detect when all worlds have reached
    the target time.

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
            dt_mode: ``"per_world"`` (default) lets each world pick its own dt
                based on its own error.  ``"global"`` reduces error to the
                worst-case (max) across worlds before the Drake controller,
                forcing all worlds to march at a single shared dt.  Used to
                measure the value of per-world adaptivity vs naive batched
                adaptive stepping.
            tiling: ``"ragged"`` (default; legacy adaptive dt with a clamped remainder
                landing) or ``"even"`` (Fix 4: choose the substep COUNT N from the carried
                ideal_dt, then tile each control interval with a uniform inner dt = dt_outer/N
                so substeps are uniform and the landing is clean; N still adapts across steps).
            max_substeps: Hard upper bound on the even-tiling substep count N per control
                interval (``"even"`` mode only). Bounds worst-case work when a world's ideal_dt
                collapses to the dt_min floor (uncapped N = ceil(dt_outer/dt_min) ~ 1e4 would make
                the per-world fixed loop launch ~1e4 kernels/frame and effectively hang). Default
                256 is far above the N needed for normal motion (~1-60) so it only bounds runaway.
            **kwargs: Forwarded to :class:`SolverMuJoCo`.
        """
        if dt_mode not in ("per_world", "global"):
            raise ValueError(f"dt_mode must be 'per_world' or 'global', got {dt_mode!r}")
        if tiling not in ("ragged", "even"):
            raise ValueError(f"tiling must be 'ragged' or 'even', got {tiling!r}")
        if int(max_substeps) < 1:
            raise ValueError(f"max_substeps must be >= 1, got {max_substeps!r}")
        super().__init__(model, separate_worlds=True, use_mujoco_cpu=False, use_mujoco_contacts=False, **kwargs)

        world_count = model.world_count
        device = model.device

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
        self._dt_mode = dt_mode
        # Fix 4: "ragged" = legacy boundary-loop + _clamp_dt_to_boundary remainder landing.
        # "even" = pick N once per interval, uniform inner dt = dt_outer/N, fixed-count loop.
        self._tiling = tiling
        self._max_substeps = int(max_substeps)
        self._N = wp.zeros(world_count, dtype=wp.int32, device=device)
        self._error_scalar = wp.zeros(1, dtype=wp.float32, device=device)

        self._scratch_full = model.state()
        self._scratch_mid = model.state()
        self._scratch_double = model.state()

        # Internal state buffers for the iteration body.
        self._state_cur = model.state()
        self._state_saved = model.state()

        self._coords_per_world = model.joint_coord_count // world_count
        self._dofs_per_world = model.joint_dof_count // world_count
        self._bodies_per_world = model.body_count // world_count

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

        self._pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=self.mjw_data.naconmax,
        )
        self._contacts_start = self._pipeline.contacts()

    def _run_substep(
        self,
        state_in: State,
        state_out: State,
        dt_array: wp.array,
    ) -> None:
        """Run one MuJoCo step: sync state_in, set timestep, step, write state_out.

        Contacts must already be written to ``mjw_data`` before calling this
        (via :meth:`_convert_contacts_to_mjwarp`).  Converting once per
        iteration and reusing across the three step-doubling substeps ensures
        the error estimate only reflects integration error, not contact-set
        discrepancy from differing body transforms.
        """
        self._update_mjc_data(self.mjw_data, self.model, state_in)
        wp.copy(self.mjw_model.opt.timestep, dt_array)

        with wp.ScopedDevice(self.model.device):
            self._mujoco_warp_step()

        self._update_newton_state(self.model, state_out, self.mjw_data, state_prev=state_in)

    def _run_even_iteration_body(self) -> None:
        """One even-tiling substep: clamp-to-boundary (zeros finished worlds), 3-eval +
        error + even controller + commit-gated select + advance. NO _apply_dt_cap (dt is
        held at dt_outer/N) and NO boundary flag (the outer loop is fixed-count = N_max)."""
        model = self.model
        n = model.world_count
        dev = model.device

        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        # Zero dt for per-world worlds that already reached next_time; also absorbs the
        # bounded float32 residual on the final substep so even tiling lands exactly.
        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
            device=dev,
        )

        wp.copy(self._state_saved.joint_q, self._state_cur.joint_q)
        wp.copy(self._state_saved.joint_qd, self._state_cur.joint_qd)
        if self._state_cur.body_q is not None and self._state_saved.body_q is not None:
            wp.copy(self._state_saved.body_q, self._state_cur.body_q)
        if self._state_cur.body_qd is not None and self._state_saved.body_qd is not None:
            wp.copy(self._state_saved.body_qd, self._state_cur.body_qd)

        if not self.mjw_model.opt.run_collision_detection:
            self._convert_contacts_to_mjwarp(self.model, self._state_cur, self._contacts_start)

        self._run_substep(self._state_cur, self._scratch_full, self._dt)
        self._run_substep(self._state_cur, self._scratch_mid, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, self._dt_half)

        wp.launch(
            _inf_norm_state_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._state_scale,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=dev,
        )

        if self._dt_mode == "global":
            wp.launch(_reset_error_scalar, dim=1, inputs=[self._error_scalar], device=dev)
            wp.launch(_reduce_max_error, dim=n, inputs=[self._last_error, self._error_scalar], device=dev)
            wp.launch(_broadcast_error, dim=n, inputs=[self._error_scalar, self._last_error], device=dev)

        wp.launch(
            _calc_even_step,
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

        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, self._state_saved.joint_q, self._commit, self._coords_per_world],
            outputs=[self._state_cur.joint_q],
            device=dev,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, self._state_saved.joint_qd, self._commit, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd],
            device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_q, self._state_saved.body_q, self._commit, self._bodies_per_world],
                outputs=[self._state_cur.body_q],
                device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_qd,
                    self._state_saved.body_qd,
                    self._commit,
                    self._bodies_per_world,
                ],
                outputs=[self._state_cur.body_qd],
                device=dev,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=dev,
        )
        wp.launch(
            _finish_diverged_worlds,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._diverged],
            device=dev,
        )

    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """One step-doubling iteration: 3-eval + error control + dt cap + boundary check."""
        model = self.model
        n = model.world_count
        dev = model.device

        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        # Clamp dt so no world overshoots its boundary target.
        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
            device=dev,
        )

        # Snapshot for rollback on rejection.
        wp.copy(self._state_saved.joint_q, self._state_cur.joint_q)
        wp.copy(self._state_saved.joint_qd, self._state_cur.joint_qd)
        if self._state_cur.body_q is not None and self._state_saved.body_q is not None:
            wp.copy(self._state_saved.body_q, self._state_cur.body_q)
        if self._state_cur.body_qd is not None and self._state_saved.body_qd is not None:
            wp.copy(self._state_saved.body_qd, self._state_cur.body_qd)

        # Convert contacts once using state_cur transforms so all 3 substeps
        # see identical MuJoCo contacts (avoids error-estimate corruption from
        # the third substep using scratch_mid's different body transforms).
        if not self.mjw_model.opt.run_collision_detection:
            self._convert_contacts_to_mjwarp(self.model, self._state_cur, self._contacts_start)

        # 3 MuJoCo evals: full dt, half dt, half dt.
        self._run_substep(self._state_cur, self._scratch_full, self._dt)
        self._run_substep(self._state_cur, self._scratch_mid, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, self._dt_half)

        wp.launch(
            _inf_norm_state_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._state_scale,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=dev,
        )

        # Global dt mode: collapse per-world error to the worst case, broadcast
        # back so the Drake controller produces one shared dt and one shared
        # accept/reject decision for every world.
        if self._dt_mode == "global":
            wp.launch(_reset_error_scalar, dim=1, inputs=[self._error_scalar], device=dev)
            wp.launch(_reduce_max_error, dim=n, inputs=[self._last_error, self._error_scalar], device=dev)
            wp.launch(_broadcast_error, dim=n, inputs=[self._error_scalar, self._last_error], device=dev)

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

        # State select: committed worlds get scratch_double, the rest hold saved.
        # commit (not accepted) gates the write so a floor-diverged world that
        # advances time still holds its last good state instead of writing NaN.
        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, self._state_saved.joint_q, self._commit, self._coords_per_world],
            outputs=[self._state_cur.joint_q],
            device=dev,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, self._state_saved.joint_qd, self._commit, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd],
            device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_q, self._state_saved.body_q, self._commit, self._bodies_per_world],
                outputs=[self._state_cur.body_q],
                device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[
                    self._scratch_double.body_qd,
                    self._state_saved.body_qd,
                    self._commit,
                    self._bodies_per_world,
                ],
                outputs=[self._state_cur.body_qd],
                device=dev,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=dev,
        )

        # Persistently-diverged worlds (held at the floor) jump straight to their
        # boundary so the loop terminates instead of grinding dt_min steps.
        wp.launch(
            _finish_diverged_worlds,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._diverged],
            device=dev,
        )

        # Cap dt for the next iteration.
        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=dev,
        )

        # Boundary check: sets _boundary_flag to 0 (done) or 1 (continue).
        wp.launch(_boundary_reset, dim=1, inputs=[self._boundary_flag], device=dev)
        wp.launch(
            _boundary_check,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )

    @event_scope
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
    ) -> State:
        """Advance each world by one adaptive step.

        Single-iteration path: one 3-eval attempt, controller update, select.
        Does not loop to a boundary — use :meth:`step_dt` for that.

        Args:
            state_in: Input state.
            state_out: Output state (written in place).
            control: Control inputs.
            contacts: Unused. Contacts are generated internally via :class:`CollisionPipeline`.

        Returns:
            state_out
        """
        model = self.model
        device = model.device
        n = model.world_count

        self._diverged.fill_(False)
        self._apply_mjc_control(model, state_in, control, self.mjw_data)
        self._enable_rne_postconstraint(state_out)

        self._pipeline.collide(state_in, self._contacts_start)

        if not self.mjw_model.opt.run_collision_detection:
            self._convert_contacts_to_mjwarp(self.model, state_in, self._contacts_start)

        self._run_substep(state_in, self._scratch_full, self._dt)
        self._run_substep(state_in, self._scratch_mid, self._dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, self._dt_half)

        wp.launch(
            _inf_norm_state_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._state_scale,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=device,
        )
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

        wp.launch(
            _select_float_kernel,
            dim=model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, state_in.joint_q, self._commit, self._coords_per_world],
            outputs=[state_out.joint_q],
            device=device,
        )
        wp.launch(
            _select_float_kernel,
            dim=model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, state_in.joint_qd, self._commit, self._dofs_per_world],
            outputs=[state_out.joint_qd],
            device=device,
        )
        if state_out.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_q, state_in.body_q, self._commit, self._bodies_per_world],
                outputs=[state_out.body_q],
                device=device,
            )
        if state_out.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=model.body_count,
                inputs=[self._scratch_double.body_qd, state_in.body_qd, self._commit, self._bodies_per_world],
                outputs=[state_out.body_qd],
                device=device,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=device,
        )

        self._step += 1
        return state_out

    @event_scope
    @override
    def step_dt(
        self,
        dt_outer: float,
        state_0: State,
        state_1: State,
        control: Control,
        apply_forces=None,
    ) -> tuple[State, State]:
        """Advance all worlds by exactly ``dt_outer`` seconds of simulation time.

        Loops the 3-eval step-doubling attempt, controller, and state-select
        via direct ``wp.launch()`` calls until every world's ``sim_time``
        reaches the boundary.  A single ``.numpy()`` read-back (4 bytes) per
        iteration checks the boundary flag for termination.

        Args:
            dt_outer: Outer control/render period [s].
            state_0: Current state (input/output).
            state_1: Scratch state (unused; returned unchanged).
            control: Control inputs (applied once, persists across substeps).
            apply_forces: Optional ``fn(state)`` for external forces.

        Returns:
            ``(state_0, state_1)`` with ``state_0`` updated.
        """
        device = self.model.device
        n = self.model.world_count

        effective_dt_max = min(self._dt_max, dt_outer)

        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=device,
        )

        wp.copy(self._state_cur.joint_q, state_0.joint_q)
        wp.copy(self._state_cur.joint_qd, state_0.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(self._state_cur.body_q, state_0.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(self._state_cur.body_qd, state_0.body_qd)

        self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
        if apply_forces is not None:
            apply_forces(state_0)

        self._enable_rne_postconstraint(self._state_cur)

        # Fix B: rebase both clocks by the per-world boundary so float32 magnitude
        # stays bounded (prevents landing-remainder precision loss / dt jitter that
        # grows over a run). Subtract-baseline preserves remaining exactly; do this
        # BEFORE advancing next_time.
        wp.launch(_rebase_time, dim=n, inputs=[self._sim_time, self._next_time], device=device)
        wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)
        # Latch reset per outer step: _diverged reflects worlds that hit the floor
        # non-finite during THIS step_dt; the env reads it afterward to reset them.
        self._diverged.fill_(False)

        # Detect contacts once per boundary.  Body-frame contact points are
        # converted to world-frame in _run_iteration_body using the current
        # state transforms, so they track body motion across iterations.
        if not self.mjw_model.opt.run_collision_detection:
            self._pipeline.collide(self._state_cur, self._contacts_start)

        if self._tiling == "even":
            # Fix 4: choose N once (per world) from the carried ideal_dt, uniform dt=dt_outer/N.
            wp.launch(
                _select_even_dt,
                dim=n,
                inputs=[self._ideal_dt, dt_outer, self._dt_min, effective_dt_max,
                        self._max_substeps, self._N, self._dt, self._dt_half],
                device=device,
            )
            if self._dt_mode == "global":
                n_max = int(self._N.numpy().max())  # one host read; shared worst-case N
                wp.launch(
                    _set_uniform_even_dt,
                    dim=n,
                    inputs=[n_max, dt_outer, self._N, self._dt, self._dt_half],
                    device=device,
                )
            else:
                n_max = int(self._N.numpy().max())  # loop bound; per-world short worlds no-op
            # Fixed-count loop: no per-iteration host sync (FPS win over the boundary flag).
            for _ in range(n_max):
                self._run_even_iteration_body()
        else:
            while True:
                self._run_iteration_body(effective_dt_max)
                if self._boundary_flag.numpy()[0] == 0:
                    break

        wp.copy(state_0.joint_q, self._state_cur.joint_q)
        wp.copy(state_0.joint_qd, self._state_cur.joint_qd)
        if state_0.body_q is not None and self._state_cur.body_q is not None:
            wp.copy(state_0.body_q, self._state_cur.body_q)
        if state_0.body_qd is not None and self._state_cur.body_qd is not None:
            wp.copy(state_0.body_qd, self._state_cur.body_qd)

        return state_0, state_1

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
        """Substep tiling mode: ``"ragged"`` (legacy) or ``"even"`` (Fix 4 uniform tiling)."""
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

    @property
    def contacts(self) -> Contacts:
        """Contacts from the most recent :meth:`step_dt` boundary.

        Populated once per outer step by the solver's internal
        :class:`~newton.CollisionPipeline` and reused across all inner
        iterations.  Pass to ``viewer.log_contacts`` for rendering without
        duplicating the collision pass.
        """
        return self._contacts_start

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
