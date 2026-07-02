"""Adaptive (error-controlled step-doubling) MuJoCo solver: ``SolverMuJoCoAdaptive``.

The manager hands the solver a state and a control boundary period ``dt_outer``; it
marches every world to that boundary with a per-world adaptive inner timestep and
writes the advanced state back. Accuracy comes from **step doubling**: each attempt
integrates one full step at ``dt`` and two half steps at ``dt/2`` and uses their
difference as a local-error estimate, which a Drake step-size controller turns into a
per-world accept/reject + grow/shrink decision -- entirely on the GPU.

The inner loop runs ENTIRELY in MuJoCo coordinates (``mjw_data.qpos``/``qvel``): the
Newton state is converted in ONCE at boundary entry (``_update_mjc_data``) and back out
ONCE at boundary exit (``_update_newton_state``, incl. the FK pass for body poses). No
per-substep Newton<->MuJoCo conversion or FK happens inside the loop.

The ragged ``step`` machine (one iteration = :meth:`_run_iteration_body`)::

    update_mjc_data(state_0)                             # Newton -> qpos/qvel, ONCE
    next_time[w] = sim_time[w] + dt_outer                # boundary target per world
    for _ in range(max_substeps):                        # max_substeps is a SAFETY cap
        clamp_dt_to_boundary(dt, sim_time, next_time)    # done worlds -> dt=0; never overshoot
        snapshot qpos/qvel                               # rollback target on reject
        full   = mjw_eval(dt);   save qpos_full          # \
        restore qpos/qvel; mjw_eval(dt/2)                #  } step doubling (3 MuJoCo evals)
        mjw_eval(dt/2)                                   # /  doubled state now in mjw_data
        e = infnorm(qpos_full, qpos)                     # per-world local error = max|Δq|
        _calc_adjusted_step(e, ...):                     # per-thread Drake controller:
            ACCEPT (e<=tol): commit; sim_time+=dt; grow dt
            REJECT (e>tol):  restore snapshot; shrink dt; retry
        apply_dt_cap(ideal_dt -> dt)                     # clamp next attempt to [dt_min, dt_max]
        boundary_flag = any(sim_time < next_time)        # ONE 4-byte host read per iteration
        if boundary_flag == 0:  break
    update_newton_state(state_0)                         # qpos/qvel -> Newton + FK, ONCE

dt is ALWAYS per-world: each world adapts its OWN dt from its OWN error, so ``P(s'|s,a)``
for one world never depends on another (the Markov property the RL gradient requires). A
shared/global worst-case dt is deliberately NOT supported -- it would couple worlds.

Each iteration body replays as ONE self-captured REGULAR CUDA graph by default; the
only host sync is the 4-byte boundary-flag read between replays. The opt-in
conditional tier (``NEWTON_MJ_ADAPTIVE_CONDITIONAL=1``) instead records the whole
march as a ``wp.capture_while`` conditional while-node -- zero host syncs, and an
outer manager-level capture may wrap the full decimation loop -- by hiding
mujoco_warp's per-step scratch allocations behind a per-call-site buffer cache
(CUDA forbids allocation nodes inside conditional bodies; see
:mod:`.mjw_alloc_cache`). The controller kernels (Drake step sizing, the inf-norm
error metric, the masked restore, the time rebase/clamp) are defined inline below,
so this solver is self-contained: open this one file to see all of its logic.

Note: true CENIC = this adaptive controller + convex ICF contact; the ICF contact model
is not yet built, so this is the adaptive (pseudo-CENIC) MuJoCo solver.
"""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ...utils.benchmark import event_scope
from .mjw_alloc_cache import MjwStepAllocCache
from .solver_mujoco import SolverMuJoCo


# =====================================================================
# Adaptive step-doubling controller kernels (inlined so this solver is
# self-contained: open this file to see all of its logic).
# =====================================================================
@wp.kernel
def _apply_dt_cap(
    ideal_dt: wp.array[wp.float32],
    dt_min: float,
    dt_max: float,
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
):
    """Clamp ideal_dt to [dt_min, dt_max], preserving ideal_dt for controller recovery."""
    i = wp.tid()
    actual = wp.clamp(ideal_dt[i], dt_min, dt_max)
    dt[i] = actual
    dt_half[i] = actual * wp.float32(0.5)


@wp.kernel
def _inf_norm_state_error_kernel(
    qpos_full: wp.array2d[wp.float32],
    qpos_double: wp.array2d[wp.float32],
    state_scale: wp.array2d[wp.float32],
    nq: int,
    error_out: wp.array[wp.float32],
):
    """Adaptive-controller accuracy metric (Kurtz & Castro, Sec. V-E)::

        e^{n+1} = || S (q^{n+1} - q̂^{n+1}) ||_∞

    Position-only inf-norm of the difference between the doubled half-step ``q`` and the
    full step ``q̂``, computed directly on MuJoCo ``qpos`` (the loop's native space),
    scaled by the diagonal ``S`` that "maps each component to a dimensionless unit."
    Velocity and contact impulses are excluded from the controller, exactly as the paper
    specifies. The paper gives no formula for ``S`` and mandates NO mass weighting,
    clipping, or normalization ("S can be estimated from knowledge of coordinate types
    or specified by expert users"); here ``S = identity`` per PI directive (free-joint
    quaternions therefore enter in MuJoCo's wxyz convention -- same components as the
    Newton joint_q metric, so the inf-norm is unchanged for hinge/slide/ball/free
    layouts). Diverged sims get error = 1e10.
    """
    world = wp.tid()

    max_err = float(0.0)
    has_nan = int(0)
    for i in range(nq):
        d = wp.abs(qpos_double[world, i] - qpos_full[world, i])
        # NaN must be flagged per component: wp.max is fmaxf on CUDA, which RETURNS THE
        # NON-NAN OPERAND, so a NaN d would be silently dropped from the running max and
        # a fully-NaN world would report error 0 and be committed.
        if wp.isnan(d):
            has_nan = 1
        else:
            max_err = wp.max(max_err, state_scale[world, i] * d)

    if has_nan != 0 or wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


# Drake CalcAdjustedStepSize constants (err_order=2 for step doubling).
_DRAKE_SAFETY = wp.constant(wp.float32(0.9))
_DRAKE_MIN_SHRINK = wp.constant(wp.float32(0.1))
_DRAKE_MAX_GROW = wp.constant(wp.float32(5.0))
_DRAKE_HYSTERESIS_HIGH = wp.constant(wp.float32(1.2))
_DRAKE_HYSTERESIS_LOW = wp.constant(wp.float32(0.9))


@wp.kernel
def _calc_adjusted_step(
    err: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    ideal_dt: wp.array[wp.float32],
    accepted: wp.array[wp.bool],
    commit: wp.array[wp.bool],
    diverged: wp.array[wp.bool],
    tol: float,
    dt_min: float,
    divergence_threshold: float,
):
    """Per-world Drake CalcAdjustedStepSize for step doubling (err_order=2).

    Writes three decisions per world:
      * ``accepted`` -- advance ``sim_time`` (progress; avoids a boundary-loop hang).
      * ``commit``   -- write the doubled state. ``False`` => hold the last good state
        (used to refuse a non-finite step instead of poisoning the batch with NaN).
      * ``diverged`` -- latch: the world hit the ``dt_min`` floor still non-finite, so it
        cannot be salvaged by subdivision; the env should reset it.

    The error kernel emits a large sentinel (``1e10``) for NaN/inf states, so
    ``e >= divergence_threshold`` (or a literal NaN/inf) means "diverged".
    dt_max clamping is deferred to _apply_dt_cap so ideal_dt is preserved.
    """
    world = wp.tid()
    e = err[world]
    step = dt[world]
    # NOTE (measured, 2026-07-01): ``step`` here is the post-boundary-clamp dt, so an
    # accepted landing sliver rewrites ideal_dt relative to the remainder (<= 5x it),
    # collapsing the carried dt. This is DELIBERATELY RETAINED. A Drake-style fix
    # ("artificially limited" steps keep the carried ideal_dt) was implemented and
    # A/B-benchmarked on Allegro reorient (2048 envs, tol=1e-3): iterations were
    # UNCHANGED (the boundary loop is gated by the max-substep world, whose dt is
    # error-limited, not landing-limited) while wall time rose 5.2% because the
    # non-binding worlds ran larger, costlier constraint solves for zero iteration
    # savings. At batch scale the landing collapse acts as a free dt-limiter on
    # non-binding worlds. Revisit for small world counts or much tighter tolerances,
    # where a landing-poisoned world CAN be the binding one.
    is_diverged = wp.isnan(e) or wp.isinf(e) or e >= divergence_threshold

    # Boundary-stalled worlds (dt clamped to 0): no-op step; accept+commit the
    # (unchanged) state without touching ideal_dt so the next interval inherits a
    # good dt instead of ramping from dt_min.
    if step <= wp.float32(0.0):
        accepted[world] = True
        commit[world] = True
        return

    # At the floor we cannot subdivide any further.
    if step <= dt_min * wp.float32(1.001):
        # Divergence latch removed (per request): a world that is still non-finite at
        # the dt_min floor now COMMITS through the e > tol path below, exactly like a
        # finite-but-can't-meet-tol world, instead of holding its last-good state and
        # flagging the env to reset it. Rationale: dt_min (~1e-6 s) is ~1000x below the
        # stable fixed step, so if the fixed step does not produce NaN, the floor will
        # not either; the latch was dormant in that regime. (`diverged` is therefore
        # never set -> stays all-False; the manager's reset(world_mask=diverged) and the
        # `diverged` property are left in place as dormant no-ops.)
        if e > tol:
            # Finite but can't meet tol at the floor: accept progress and commit.
            accepted[world] = True
            commit[world] = True
            ideal_dt[world] = dt_min
            return
        # e <= tol at the floor: fall through to the normal accept path.

    # Above the floor and diverged: reject and shrink hard for a smaller retry.
    if is_diverged:
        accepted[world] = False
        commit[world] = False
        ideal_dt[world] = _DRAKE_MIN_SHRINK * step
        return

    new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))

    # Symmetric deadband (paper Alg 1): keep dt unchanged when new_step lands
    # in [k_Low * dt, k_High * dt]. Prevents dt thrash from small error spikes
    # (lower edge) and suppresses tiny grows (upper edge).
    if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
        new_step = step

    new_step = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)

    acc = e <= tol or new_step >= step
    accepted[world] = acc
    commit[world] = acc
    ideal_dt[world] = new_step


@wp.kernel
def _advance_sim_time(
    sim_time: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    accepted: wp.array[wp.bool],
    error: wp.array[wp.float32],
    accepted_error: wp.array[wp.float32],
):
    """Advance sim_time[i] by dt[i] and snapshot error for accepted worlds only."""
    i = wp.tid()
    if accepted[i]:
        sim_time[i] = sim_time[i] + dt[i]
        accepted_error[i] = error[i]


@wp.kernel
def _reset_worlds(
    mask: wp.array[wp.bool],
    dt_init: float,
    ideal_dt: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    diverged: wp.array[wp.bool],
    accepted: wp.array[wp.bool],
):
    """Restore the step-doubling controller's persistent per-world state to
    construction defaults for worlds flagged in ``mask``; leave others untouched.

    Fix C (per-world controller reset on env/episode reset). sim_time and next_time
    are reset TOGETHER to 0 so the world restarts a clean boundary interval (the next
    step_dt advances next_time by dt_outer from 0); this also drops the float32
    unbounded-growth of a long-lived world."""
    i = wp.tid()
    if mask[i]:
        ideal_dt[i] = dt_init
        dt[i] = dt_init
        dt_half[i] = dt_init * wp.float32(0.5)
        sim_time[i] = wp.float32(0.0)
        next_time[i] = wp.float32(0.0)
        diverged[i] = False
        accepted[i] = False


@wp.kernel
def _restore_uncommitted_rows_kernel(
    saved: wp.array2d[wp.float32],
    commit: wp.array[wp.bool],
    dt: wp.array[wp.float32],
    out: wp.array2d[wp.float32],
):
    """Masked rollback in MuJoCo space: restore the pre-attempt snapshot row for worlds
    that did NOT commit a real step; committed worlds keep the doubled state already in
    ``out`` (= ``mjw_data`` fields). The ``_commit`` mask (NOT ``_accepted``) gates the
    write so a floor-diverged world that still advances time holds its last good state.

    Boundary-stalled worlds (dt clamped to 0) are restored as well, even though they
    "commit": their no-op evals still run the full MuJoCo pipeline with timestep 0,
    whose constraint scaling divides by dt -- a NaN/garbage qacc there would poison
    qvel via ``qvel += 0 * NaN`` and corrupt the warm start. Restoring the snapshot
    makes the stalled iterations true no-ops regardless."""
    world, j = wp.tid()
    if (not commit[world]) or dt[world] <= wp.float32(0.0):
        out[world, j] = saved[world, j]


@wp.kernel
def _boundary_reset(flag: wp.array[wp.int32]):
    """Set flag[0] = 0 (assume all worlds reached the boundary)."""
    flag[0] = 0


@wp.kernel
def _boundary_check(
    sim_time: wp.array[wp.float32],
    target: wp.array[wp.float32],
    iter_count: wp.array[wp.int32],
    max_iters: int,
    flag: wp.array[wp.int32],
):
    """Set flag to 1 if any world has not yet reached target.

    The flag stays 0 once ``max_iters`` attempts ran this boundary, so the
    device-side conditional loop (``wp.capture_while``) honors the
    ``max_substeps`` safety cap without any host involvement.
    """
    i = wp.tid()
    if iter_count[0] >= max_iters:
        return
    if sim_time[i] < target[i]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _boundary_advance(arr: wp.array[wp.float32], delta: float):
    """Increment arr[i] by delta."""
    i = wp.tid()
    arr[i] = arr[i] + delta


@wp.kernel
def _rebase_time(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
):
    """Rebase both per-world clocks by subtracting each world's boundary baseline.

    Fix B (float32 time-rebase). ``_sim_time`` and ``_next_time`` are never reset and
    grow unbounded across a training run; the landing remainder ``next_time - sim_time``
    then loses float32 precision as magnitude grows, causing dt jitter that worsens over
    time. Subtracting the per-world baseline ``next_time[i]`` (NOT zeroing) keeps both
    clocks small while preserving the remainder bit-exactly: ``next_time`` -> 0 and
    ``sim_time`` -> the (>= 0) residual overshoot, which is carried forward instead of
    dropped. Called once at the top of ``step_dt`` before ``_boundary_advance``.
    """
    i = wp.tid()
    base = next_time[i]
    sim_time[i] = sim_time[i] - base
    next_time[i] = next_time[i] - base


@wp.kernel
def _clamp_dt_to_boundary(
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
):
    """Clamp dt so worlds don't overshoot their boundary target.

    Worlds already at or past the boundary get dt=0 (no-op step).
    """
    i = wp.tid()
    remaining = next_time[i] - sim_time[i]
    if remaining <= wp.float32(0.0):
        dt[i] = wp.float32(0.0)
        dt_half[i] = wp.float32(0.0)
    elif dt[i] > remaining:
        dt[i] = remaining
        dt_half[i] = remaining * wp.float32(0.5)


@wp.kernel
def _iter_count_increment(count: wp.array[wp.int32]):
    """Increment iteration counter (dim=1, single thread)."""
    count[0] = count[0] + 1


@wp.kernel
def _status_sentinel_reset(out: wp.array[wp.float32]):
    """Reset 6-element summary buffer: [min_sim_time, max_sim_time, max_error, accept_count, min_dt, max_dt]."""
    out[0] = float(1.0e38)
    out[1] = float(0.0)
    out[2] = float(0.0)
    out[3] = float(0.0)
    out[4] = float(1.0e38)
    out[5] = float(0.0)


@wp.kernel
def _status_summary_kernel(
    sim_time: wp.array[wp.float32],
    last_error: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    accepted: wp.array[wp.bool],
    out: wp.array[wp.float32],
):
    """Reduce per-world arrays to 6 summary scalars via atomics."""
    i = wp.tid()
    wp.atomic_min(out, 0, sim_time[i])
    wp.atomic_max(out, 1, sim_time[i])
    wp.atomic_max(out, 2, last_error[i])
    if accepted[i]:
        wp.atomic_add(out, 3, wp.float32(1.0))
    wp.atomic_min(out, 4, dt[i])
    wp.atomic_max(out, 5, dt[i])


class SolverMuJoCoAdaptive(SolverMuJoCo):
    """Adaptive-step MuJoCo solver for high-accuracy dataset generation.

    Uses step doubling (3 MuJoCo evals per attempt) to estimate per-world
    integration error and adapt the timestep on the GPU.  The ragged boundary
    loop replays one captured iteration body on CUDA when possible, checking a
    4-byte flag via ``.numpy()`` to detect when all worlds have reached the
    target time. The inner loop marches ``mjw_data.qpos``/``qvel`` directly;
    Newton state conversion (incl. FK) happens once per boundary in each
    direction.

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

        # ---- step-doubling scratch buffers, in MuJoCo space ([nworld, nq]/[nworld, nv]) ----
        # The inner loop marches mjw_data.qpos/qvel directly; these hold the rollback
        # snapshot and the full-step result (the Richardson pair is _qpos_full vs the
        # doubled state left in mjw_data.qpos).
        self._nq = int(self.mjw_data.qpos.shape[1])
        self._nv = int(self.mjw_data.qvel.shape[1])
        self._qpos_saved = wp.zeros_like(self.mjw_data.qpos)
        self._qvel_saved = wp.zeros_like(self.mjw_data.qvel)
        self._qpos_full = wp.zeros_like(self.mjw_data.qpos)
        # The snapshot also covers the solver warm start and actuator activations so a
        # rejected attempt is a TRUE rollback: act must not integrate on rejects (and the
        # full eval's act must not leak into the half evals -- the Richardson pair needs
        # both estimates to start from identical internal state), and a rejected/stalled
        # eval's qacc must not seed the next attempt's warm start.
        self._warmstart_saved = wp.zeros_like(self.mjw_data.qacc_warmstart)
        _act = getattr(self.mjw_data, "act", None)
        self._na = int(_act.shape[1]) if _act is not None and len(_act.shape) == 2 else 0
        self._act_saved = wp.zeros_like(_act) if self._na > 0 else None

        # Boundary output buffer: _update_newton_state writes the final committed state
        # (+ FK'd body poses) here once per boundary, then it is copied into the caller's
        # state. Kept separate from state_0 so state==state_prev aliasing never occurs.
        self._state_cur = model.state()

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
        # (shape [world_count, nq]; scales MuJoCo qpos components) after construction.
        self._state_scale = wp.array(
            np.ones((world_count, self._nq), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        # ---- solver-internal CUDA-graph capture ----
        # Default tier: one REGULAR graph per iteration body (keyed by effective dt_max),
        # replayed with a 4-byte boundary-flag poll between iterations. mujoco_warp's step
        # allocates per call, and CUDA forbids allocation nodes inside conditional body
        # graphs -- fine for regular graphs (alloc nodes). Gated by
        # NEWTON_MJ_ADAPTIVE_GRAPH (default on) + CUDA device; capture is warmed on the
        # first boundary.
        try:
            _is_cuda = bool(wp.get_device(device).is_cuda)
        except Exception:
            _is_cuda = False
        self._is_cuda = _is_cuda
        self._graph_enabled = _is_cuda and os.environ.get("NEWTON_MJ_ADAPTIVE_GRAPH", "1") != "0"
        self._march_graph_cache: dict = {}
        self._march_warmed = False

        # ---- conditional-march tier (opt-in; targets wall time on fast GPUs) ----
        # NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 runs the WHOLE boundary loop as one CUDA
        # conditional while-node (wp.capture_while): zero host syncs per boundary, and an
        # outer manager-level capture may wrap the full decimation loop around it. This
        # requires the mjw step to be allocation-free at record time, which the
        # MjwStepAllocCache shim provides (per-call-site buffer reuse; see that module).
        # Warmup runs on the per-iteration tier so module loads / mjw lazy init /
        # alloc-cache population all happen OUTSIDE capture. Any capture failure
        # permanently downgrades back to the per-iteration tier (never crashes a run).
        # NEWTON_MJW_ALLOC_CACHE=1 enables just the shim without the conditional tier.
        self._conditional_enabled = self._graph_enabled and os.environ.get("NEWTON_MJ_ADAPTIVE_CONDITIONAL", "0") == "1"
        alloc_cache_on = self._conditional_enabled or os.environ.get("NEWTON_MJW_ALLOC_CACHE", "0") == "1"
        self._alloc_cache = MjwStepAllocCache() if alloc_cache_on else None
        self._conditional_graph_cache: dict = {}
        self._conditional_warm_boundaries = 0

    # =====================================================================
    # Adaptive-core helpers (the pieces of one iteration body)
    # =====================================================================
    @staticmethod
    def _copy_state(dst: State, src: State) -> None:
        """Copy joint_q/qd (and body_q/qd when both states carry them) src -> dst.

        Used to load the incoming state and write the result back at the boundary.
        """
        wp.copy(dst.joint_q, src.joint_q)
        wp.copy(dst.joint_qd, src.joint_qd)
        if src.body_q is not None and dst.body_q is not None:
            wp.copy(dst.body_q, src.body_q)
        if src.body_qd is not None and dst.body_qd is not None:
            wp.copy(dst.body_qd, src.body_qd)

    def _mjw_eval(self, dt_array: wp.array) -> None:
        """ONE MuJoCo-Warp eval at the per-world timesteps in ``dt_array``: sets
        ``opt.timestep`` and steps ``mjw_data`` in place (no Newton conversion).

        With the alloc cache enabled, the step's scratch allocations resolve to cached
        per-call-site buffers, so the eval records with zero allocation nodes (required
        inside conditional body graphs; see :mod:`.mjw_alloc_cache`)."""
        wp.copy(self.mjw_model.opt.timestep, dt_array)
        with wp.ScopedDevice(self.model.device):
            if self._alloc_cache is not None:
                with self._alloc_cache.scope():
                    self._mujoco_warp_step()
            else:
                self._mujoco_warp_step()

    def _step_double(self) -> None:
        """Step doubling -- the 3 MuJoCo evals, entirely in qpos/qvel space: snapshot,
        one full step at ``dt`` (result saved to ``_qpos_full``), restore, then two half
        steps at ``dt/2``. ``mjw_data`` ends holding the doubled state; ``_qpos_full``
        vs ``mjw_data.qpos`` is the Richardson pair the error kernel differences.

        The snapshot/restore includes ``qacc_warmstart`` and ``act`` so the two
        Richardson estimates start from identical internal state (the full eval's
        warm start and activations must not leak into the half evals).
        """
        d = self.mjw_data
        wp.copy(self._qpos_saved, d.qpos)
        wp.copy(self._qvel_saved, d.qvel)
        wp.copy(self._warmstart_saved, d.qacc_warmstart)
        if self._na > 0:
            wp.copy(self._act_saved, d.act)
        self._mjw_eval(self._dt)
        wp.copy(self._qpos_full, d.qpos)
        wp.copy(d.qpos, self._qpos_saved)
        wp.copy(d.qvel, self._qvel_saved)
        wp.copy(d.qacc_warmstart, self._warmstart_saved)
        if self._na > 0:
            wp.copy(d.act, self._act_saved)
        self._mjw_eval(self._dt_half)
        self._mjw_eval(self._dt_half)

    def _estimate_error(self) -> None:
        """Per-world local error: inf-norm ``e = max|Δq|`` between the full step and the doubled
        half-step (NaN/inf collapse to a 1e10 sentinel). Writes ``_last_error``."""
        wp.launch(
            _inf_norm_state_error_kernel,
            dim=self.model.world_count,
            inputs=[
                self._qpos_full,
                self.mjw_data.qpos,
                self._state_scale,
                self._nq,
            ],
            outputs=[self._last_error],
            device=self.model.device,
        )

    def _commit_or_restore(self) -> None:
        """Masked rollback: worlds that did NOT commit a real step (rejected, or stalled
        at the boundary with dt=0) restore the pre-attempt snapshot -- qpos, qvel, the
        solver warm start, and actuator activations; committed worlds keep the doubled
        state already in ``mjw_data``."""
        n = self.model.world_count
        dev = self.model.device
        for saved, out, width in (
            (self._qpos_saved, self.mjw_data.qpos, self._nq),
            (self._qvel_saved, self.mjw_data.qvel, self._nv),
            (self._warmstart_saved, self.mjw_data.qacc_warmstart, self._nv),
        ):
            wp.launch(
                _restore_uncommitted_rows_kernel,
                dim=(n, width),
                inputs=[saved, self._commit, self._dt],
                outputs=[out],
                device=dev,
            )
        if self._na > 0:
            wp.launch(
                _restore_uncommitted_rows_kernel,
                dim=(n, self._na),
                inputs=[self._act_saved, self._commit, self._dt],
                outputs=[self.mjw_data.act],
                device=dev,
            )

    # =====================================================================
    # Per-frame iteration bodies (the captured/replayed substep bodies)
    # =====================================================================
    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """ONE ragged adaptive iteration: clamp -> step-double -> error -> Drake controller ->
        masked rollback -> advance -> dt cap -> boundary check. All in MuJoCo qpos/qvel space.

        This is the body the device-side conditional loop replays (see :meth:`_march_ragged`).
        Every phase is a flat kernel-launch sequence, so it records cleanly inside a CUDA
        graph / conditional while-node.
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

        # --- adaptive core: step double, estimate error, run the controller ---
        self._step_double()
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

        # Rejected worlds roll back to the snapshot; committed worlds keep the doubled state.
        self._commit_or_restore()

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
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
            inputs=[
                self._sim_time,
                self._next_time,
                self._iteration_count_buf,
                self._max_substeps,
                self._boundary_flag,
            ],
            device=dev,
        )

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

        Ragged tiling: adaptive boundary loop with a graph-captured iteration body when
        available and a 4-byte ``.numpy()`` boundary-flag read-back per iteration;
        Newton<->MuJoCo state conversion happens once per boundary in each direction.
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

        self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
        if apply_forces is not None:
            apply_forces(state_0)

        self._enable_rne_postconstraint(self._state_cur)

        # Load the incoming Newton state into MuJoCo coordinates ONCE per boundary;
        # the whole inner loop then marches mjw_data.qpos/qvel directly.
        self._update_mjc_data(self.mjw_data, self.model, state_0)

        # Fix B: rebase both clocks by the per-world boundary so float32 magnitude stays bounded
        # (prevents landing-remainder precision loss / dt jitter that grows over a run). The
        # subtract-baseline preserves the remaining time exactly; do this BEFORE advancing next_time.
        wp.launch(_rebase_time, dim=n, inputs=[self._sim_time, self._next_time], device=device)
        wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)

        self._march_ragged(effective_dt_max)

        # Convert the final committed state back to Newton coordinates ONCE per boundary
        # (incl. the FK pass for body_q/body_qd), then hand it to the caller. _state_cur
        # buffers the write so state and state_prev never alias in the convert kernel.
        self._update_newton_state(self.model, self._state_cur, self.mjw_data, state_prev=state_0)
        self._copy_state(state_0, self._state_cur)

        return state_0, state_1

    def _iteration_graph(self, effective_dt_max: float):
        """Return the captured iteration-body graph (keyed by effective_dt_max), or ``None``
        to run eagerly. The first iteration runs eagerly so MuJoCo/Warp can lazily initialize
        allocations OUTSIDE capture; capture failures disable capture permanently.

        NOTE: this default tier replays ONE REGULAR graph per iteration with a 4-byte
        host flag poll in between, NOT a ``wp.capture_while`` conditional while-node over
        the whole march: mujoco_warp's ``step`` performs per-call memory allocations, and
        CUDA forbids allocation nodes inside conditional body graphs ("Conditional body
        graph contains an unsupported operation (memory allocation)"). Regular captured
        graphs allow alloc nodes, so the per-iteration replay works everywhere. The
        opt-in conditional tier (``NEWTON_MJ_ADAPTIVE_CONDITIONAL=1``) removes the polls
        by hiding those allocations behind :class:`.mjw_alloc_cache.MjwStepAllocCache`.
        """
        if not self._graph_enabled:
            return None

        if not self._march_warmed:
            self._march_warmed = True
            return None

        key = round(float(effective_dt_max), 12)
        graph = self._march_graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._run_iteration_body(effective_dt_max)
                graph = cap.graph
                self._march_graph_cache[key] = graph
            except Exception:
                self._graph_enabled = False
                self._march_graph_cache.clear()
                return None
        return graph

    def _run_ragged_iteration(self, effective_dt_max: float) -> None:
        """Run one ragged iteration: replay the captured body if available, else run eagerly.
        A capture/launch failure falls back to eager so a run never crashes on a graph error."""
        graph = self._iteration_graph(effective_dt_max)
        if graph is None:
            self._run_iteration_body(effective_dt_max)
            return

        try:
            wp.capture_launch(graph)
        except Exception:
            self._graph_enabled = False
            self._march_graph_cache.clear()
            self._run_iteration_body(effective_dt_max)

    _CONDITIONAL_WARM_BOUNDARIES = 2
    """Boundaries to run on the per-iteration tier before attempting conditional capture:
    kernel-module loads, mjw lazy init, and alloc-cache population must all happen
    OUTSIDE the capture (a conditional body may not record allocations)."""

    def _external_capture_active(self) -> bool:
        """True when an OUTER CUDA-graph capture (e.g. the Isaac Lab manager's) is
        recording on the current stream, so the march must record its conditional node
        into that graph instead of launching/replaying its own."""
        if not self._is_cuda:
            return False
        try:
            dev = wp.get_device(self.model.device)
            return dev.captures.get(wp.get_stream(dev)) is not None
        except Exception:
            return False

    def _march_conditional(self, effective_dt_max: float) -> None:
        """Ragged march as a device-side loop: ``wp.capture_while`` replays the iteration
        body while the boundary flag is nonzero (the flag kernel also enforces the
        ``max_substeps`` cap on device). Inside a capture this records ONE conditional
        while-node; the body must be allocation-free (alloc cache active)."""
        wp.capture_while(self._boundary_flag, lambda: self._run_iteration_body(effective_dt_max))

    def _abort_active_capture(self) -> None:
        """Best-effort: never leave the stream in capture mode after a failed capture.

        A stream stuck mid-capture silently records every subsequent launch into an
        orphan graph, which manifests later as bogus OOMs -- observed on the first
        conditional-capture experiment. Belt-and-braces alongside ScopedCapture's own
        cleanup."""
        try:
            dev = wp.get_device(self.model.device)
            stream = wp.get_stream(dev)
            if dev.captures.get(stream) is not None:
                with contextlib.suppress(Exception):
                    wp.capture_end(stream=stream)
        except Exception:
            pass

    def _launch_conditional_march(self, effective_dt_max: float) -> bool:
        """Replay (capturing on first use) the whole-march conditional graph.
        Returns False after permanently downgrading on any capture/launch failure."""
        key = round(float(effective_dt_max), 12)
        graph = self._conditional_graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._march_conditional(effective_dt_max)
                graph = cap.graph
                self._conditional_graph_cache[key] = graph
            except Exception as exc:
                self._abort_active_capture()
                self._conditional_enabled = False
                self._conditional_graph_cache.clear()
                warnings.warn(
                    f"SolverMuJoCoAdaptive: conditional-march capture failed ({exc}); "
                    "downgrading permanently to per-iteration graph replay.",
                    stacklevel=2,
                )
                return False
        try:
            wp.capture_launch(graph)
            return True
        except Exception as exc:
            self._conditional_enabled = False
            self._conditional_graph_cache.clear()
            warnings.warn(
                f"SolverMuJoCoAdaptive: conditional-march launch failed ({exc}); "
                "downgrading permanently to per-iteration graph replay.",
                stacklevel=2,
            )
            return False

    def _march_ragged(self, effective_dt_max: float) -> None:
        """March every world to its boundary.

        Tiers (first applicable wins):
        1. Conditional mode + outer capture recording -> contribute the while-node.
        2. Conditional mode, warmed -> replay the self-captured whole-march graph
           (zero host syncs per boundary).
        3. Default / warmup / post-failure: replay the per-iteration graph with a
           4-byte boundary-flag poll between iterations, capped at ``max_substeps``.
        """
        if self._conditional_enabled:
            if self._external_capture_active():
                self._march_conditional(effective_dt_max)
                return
            if self._conditional_warm_boundaries >= self._CONDITIONAL_WARM_BOUNDARIES:
                if self._launch_conditional_march(effective_dt_max):
                    return
            else:
                self._conditional_warm_boundaries += 1

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
