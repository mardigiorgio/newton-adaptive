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
        dt_histogram_accum(dt, ...)                      # optional; MUST precede the clamp below (dt is
                                                          # still the controller's chosen step here)
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
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import BodyFlags, Contacts, Control, Model, State
from ...sim.articulation import eval_articulation_fk
from ...utils.benchmark import event_scope
from ..adaptive_boundary import mark_unfinished
from .kernels import convert_mj_coords_to_warp_kernel
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


def _dt_hist_layout(dt_min: float, dt_init: float, bins_per_decade: int) -> tuple[int, float]:
    """Bin count and low edge for the dt-occupancy histogram.

    One extra decade above ``dt_init`` is included as headroom for configurations where
    ``dt_outer > dt_init`` (the controller can then grow the step past ``dt_init``, up to
    ``effective_dt_max = min(dt_max, dt_outer)``). ``dt_max`` defaults to ``inf``, so for
    the common case ``dt_outer <= dt_init`` the step can never exceed ``dt_init`` and the
    bins above ``min(dt_max, dt_outer)`` are simply never populated.

    Args:
        dt_min: Adaptive timestep floor [s].
        dt_init: Initial adaptive timestep [s].
        bins_per_decade: Log-spaced bins per decade.

    Returns:
        ``(n_bins, lo_log10)`` -- total bin count (floor bin + log bins + overflow bin)
        and ``log10(dt_min)``.
    """
    lo_log10 = math.log10(dt_min)
    n_decades = math.ceil(math.log10(dt_init / dt_min)) + 1
    return 1 + n_decades * bins_per_decade + 1, lo_log10


def _dt_hist_edges(dt_min: float, n_bins: int, bins_per_decade: int) -> np.ndarray:
    """Edges [s] of the log-spaced bins, i.e. bins ``1 .. n_bins - 2``.

    Length is ``n_bins - 1``: the floor bin (0) and the overflow bin (``n_bins - 1``)
    are open-ended and contribute no finite edge of their own.
    """
    return dt_min * 10.0 ** (np.arange(n_bins - 1) / float(bins_per_decade))


_DT_HIST_SENTINEL = 1.0e38
"""Initial value of the saturation-depth accumulator; means "floor never hit".

Not exactly representable in float32 (rounds to ``9.9999997e37``), so any comparison
against the untouched accumulator must go through ``float(np.float32(_DT_HIST_SENTINEL))``
rather than the raw double literal -- see :meth:`SolverMuJoCoAdaptive.dt_histogram_stats`.
"""


@wp.kernel
def _dt_histogram_accum(
    dt: wp.array[wp.float32],
    ideal_dt: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt_min: float,
    lo_log10: float,
    bins_per_decade: float,
    n_bins: int,
    counts: wp.array[wp.int64],
    saturation: wp.array[wp.float32],
):
    """Bin the timestep this iteration is about to attempt.

    Launched at the TOP of the iteration body, before ``_clamp_dt_to_boundary``: at that
    point ``dt`` still holds the controller's chosen step, not a landing sliver. Worlds
    already at their boundary are skipped -- they take no further step this interval.

    Bin 0 counts exact floor hits; ``_apply_dt_cap`` produces bitwise ``dt_min`` on clamp,
    so the equality test is reliable. ``saturation`` accumulates ``min(ideal_dt)`` over
    floor-clamped worlds, showing how far below the floor the controller wanted to go.

    Precondition: ``dt`` must be finite here. ``NaN`` and ``+inf`` both fail the ``h <=
    dt_min`` test (NaN comparisons are always false; ``+inf > dt_min``), so they fall
    through to the ``b = ...`` branch below -- but casting an ``inf``/``NaN``-derived
    ``wp.log10`` result to ``int`` does not saturate to a large positive index the way the
    overflow bin needs; it lands ``b`` near or below the valid range, which ``wp.clamp``
    then floors up to bin 1 (near-floor) instead of the last (overflow) bin -- the
    INVERTING direction. This is safe only because the caller
    (:meth:`SolverMuJoCoAdaptive._run_iteration_body`) launches this kernel on ``self._dt``
    AFTER ``_apply_dt_cap`` has already sanitized it that same iteration (NaN -> ``dt_min``,
    ``+inf`` -> ``dt_max``), so a non-finite ``dt`` never actually reaches this kernel. No
    runtime finiteness branch is added here to avoid the per-iteration cost of guarding an
    otherwise-unreachable case.
    """
    i = wp.tid()
    if sim_time[i] >= next_time[i]:
        return
    h = dt[i]
    if h <= dt_min:
        wp.atomic_add(counts, 0, wp.int64(1))
        wp.atomic_min(saturation, 0, ideal_dt[i])
        return
    b = 1 + int(wp.floor((wp.log10(h) - lo_log10) * bins_per_decade))
    wp.atomic_add(counts, wp.clamp(b, 1, n_bins - 1), wp.int64(1))


@wp.kernel
def _inf_norm_state_error_kernel(
    qpos_full: wp.array2d[wp.float32],
    qpos_double: wp.array2d[wp.float32],
    state_scale: wp.array2d[wp.float32],
    nq: int,
    qvel_double: wp.array2d[wp.float32],
    nv: int,
    rtol_over_atol: float,
    error_out: wp.array[wp.float32],
):
    """Adaptive-controller accuracy metric::

        e^{n+1} = || S (q^{n+1} - q̂^{n+1}) ||_∞

    Position-only inf-norm of the difference between the doubled half-step ``q`` and the
    full step ``q̂``, computed directly on MuJoCo ``qpos`` (the loop's native space),
    scaled by the diagonal ``S`` (default identity; free-joint quaternions enter in
    MuJoCo's wxyz convention). Velocity and contact impulses are excluded from the
    error norm. Diverged sims get error = 1e10.
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
            # Filtered error estimate (state_scale weighting; externally writable
            # otherwise, weight == 1): discount each coordinate by the damping factor of
            # the implicit solve, w_i = 1/(1 + h*kd_i/M_ii). Error in a fast-decaying
            # actuator mode is damped in the ESTIMATE by the same factor the integrator
            # damps it in the SOLUTION; undamped coords (kd=0) keep w=1, and w -> 1 as
            # h -> 0, restoring full sensitivity when the step resolves the mode.
            if rtol_over_atol > 0.0:
                # Normalized (mixed-tolerance) error: dividing by
                # 1 + (rtol/atol)*|q| makes the fixed-tol accept test
                # equivalent to |d| <= atol + rtol*|q|, so the budget scales
                # with coordinate magnitude exactly as the float grid does --
                # a large coordinate's representation noise can never pin the
                # estimate near tol and freeze the controller's deadband.
                m = wp.max(wp.abs(qpos_double[world, i]), wp.abs(qpos_full[world, i]))
                d = d / (wp.float32(1.0) + rtol_over_atol * m)
            max_err = wp.max(max_err, state_scale[world, i] * d)

    # Full-state divergence detection: the error NORM stays position-only, but the
    # committed state includes qvel, so finiteness must be checked there too. A
    # non-finite velocity with still-finite positions would otherwise commit
    # (position error looks small) and poison subsequent steps via forces ~ v.
    for i in range(nv):
        v = qvel_double[world, i]
        if wp.isnan(v) or wp.isinf(v):
            has_nan = 1

    if has_nan != 0 or wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


@wp.kernel
def _inf_norm_state_error_indexed_kernel(
    qpos_full: wp.array2d[wp.float32],
    qpos_double: wp.array2d[wp.float32],
    state_scale: wp.array2d[wp.float32],
    nq: int,
    qvel_double: wp.array2d[wp.float32],
    nv: int,
    idx: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slot: int,
    rtol_over_atol: float,
    error_out: wp.array[wp.float32],
):
    """:func:`_inf_norm_state_error_kernel` over a compacted world list: only
    worlds in the index list get a fresh error; landed worlds keep their last
    written value. Safe because a landed world's error never reaches a
    decision: the controller's dt <= 0 branch returns before its value can
    affect any write. Fixed launch dim with a device-side count guard so the
    launch records cleanly under graph capture; the loop body must stay
    identical to the full-dim kernel's (same fp operation order per world)."""
    i = wp.tid()
    if i >= counts[slot]:
        return
    world = idx[i]

    max_err = float(0.0)
    has_nan = int(0)
    for k in range(nq):
        d = wp.abs(qpos_double[world, k] - qpos_full[world, k])
        # NaN must be flagged per component: wp.max is fmaxf on CUDA, which
        # RETURNS THE NON-NAN OPERAND (see the full-dim kernel).
        if wp.isnan(d):
            has_nan = 1
        else:
            if rtol_over_atol > 0.0:
                # Same normalization as the full-dim kernel; the loop body
                # must stay identical (same fp operation order).
                m = wp.max(wp.abs(qpos_double[world, k]), wp.abs(qpos_full[world, k]))
                d = d / (wp.float32(1.0) + rtol_over_atol * m)
            max_err = wp.max(max_err, state_scale[world, k] * d)

    for k in range(nv):
        v = qvel_double[world, k]
        if wp.isnan(v) or wp.isinf(v):
            has_nan = 1

    if has_nan != 0 or wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


# Drake CalcAdjustedStepSize constants (err_order=2 for step doubling).
_DRAKE_SAFETY = wp.constant(wp.float32(0.9))
_DRAKE_MIN_SHRINK = wp.constant(wp.float32(0.1))
_DRAKE_MAX_GROW = wp.constant(wp.float32(5.0))
_DRAKE_HYSTERESIS_HIGH = wp.constant(wp.float32(1.2))
_DRAKE_HYSTERESIS_LOW = wp.constant(wp.float32(0.9))
# Ceiling memory: a rejection at step h records dt_ceiling = 0.9*h; growth is
# clamped to the ceiling, which relaxes per accepted step (default 1.1x,
# override NEWTON_ADAPTIVE_CEILING_RELAX). Handles error landscapes with a knee
# (contact regimes) where order-2 growth sizing otherwise oscillates
# accept-grow-reject around the acceptance boundary; the relax rate sets how
# often a world in sustained contact re-probes its knee.
_CEILING_MARGIN = wp.constant(wp.float32(0.9))
_CEILING_RELAX = wp.constant(wp.float32(float(os.environ.get("NEWTON_ADAPTIVE_CEILING_RELAX", "1.1"))))


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
    dt_max: float,
    divergence_threshold: float,
    dt_ceiling: wp.array[wp.float32],
    limited: wp.array[wp.int32],
    consec_rej: wp.array[wp.int32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
):
    """Per-world Drake CalcAdjustedStepSize for step doubling (err_order=2).

    Writes two decisions per world:
      * ``accepted`` -- advance ``sim_time`` (progress; avoids a boundary-loop hang).
      * ``commit``   -- write the doubled state. ``False`` => hold the last good state
        (used to refuse a non-finite step instead of poisoning the batch with NaN).

    ``diverged`` is set only by the NaN-guard floor path (``nan_guard == 1``);
    no other path writes it.

    The error kernel emits a large sentinel (``1e10``) for NaN/inf states, so
    ``e >= divergence_threshold`` (or a literal NaN/inf) means "diverged".
    dt_max clamping is deferred to _apply_dt_cap so ideal_dt is preserved.
    """
    world = wp.tid()
    e = err[world]
    step = dt[world]
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
        if e > tol:
            accepted[world] = True
            ideal_dt[world] = dt_min
            consec_rej[world] = 0
            # NaN guard: never commit a non-finite state at the floor -- hold
            # the last-good (finite) state, freeze this world to its frame
            # boundary, and latch `diverged` so env-side terminations can
            # reset it. Only genuine non-finiteness (or the error kernel's
            # divergence sentinel) triggers this: a large-but-finite error is
            # a legitimately hard step, not a blow-up.
            if wp.isnan(e) or wp.isinf(e) or e >= divergence_threshold:
                commit[world] = False
                diverged[world] = True
                sim_time[world] = next_time[world]
                return
            # Finite but can't meet tol at the floor: accept progress and commit.
            commit[world] = True
            return
        # e <= tol at the floor: fall through to the normal accept path.

    # Above the floor and diverged: reject and shrink hard for a smaller retry.
    if is_diverged:
        accepted[world] = False
        commit[world] = False
        dt_ceiling[world] = wp.min(dt_ceiling[world], _CEILING_MARGIN * step)
        ideal_dt[world] = _DRAKE_MIN_SHRINK * step
        consec_rej[world] = consec_rej[world] + 1
        return

    new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))

    # Symmetric deadband: keep dt unchanged when new_step lands
    # in [k_Low * dt, k_High * dt]. Prevents dt thrash from small error spikes
    # (lower edge) and suppresses tiny grows (upper edge).
    if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
        new_step = step

    new_step = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)

    acc = e <= tol or new_step >= step
    accepted[world] = acc
    commit[world] = acc
    if acc:
        consec_rej[world] = 0
        # Sliver exemption: an accepted step that was artificially shortened
        # by the boundary clamp says nothing about the error-limited step
        # size, so keep the carried ideal_dt instead of collapsing it to ~the
        # sliver (Drake's artificially-limited rule). It also carries no
        # ceiling information.
        if limited[world] == 1:
            return
        dt_ceiling[world] = wp.min(dt_ceiling[world] * _CEILING_RELAX, dt_max)
    else:
        dt_ceiling[world] = wp.min(dt_ceiling[world], _CEILING_MARGIN * step)
        consec_rej[world] = consec_rej[world] + 1
    new_step = wp.min(new_step, dt_ceiling[world])
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
def _count_rejects(accepted: wp.array[wp.bool], rejects: wp.array[wp.int32]):
    """Accumulate this iteration's rejected attempts (diagnostic only)."""
    i = wp.tid()
    if not accepted[i]:
        wp.atomic_add(rejects, 0, 1)


@wp.kernel
def _debt_guard(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt_outer: float,
    dt_init: float,
    ceiling_init: float,
    ideal_dt: wp.array[wp.float32],
    dt_ceiling: wp.array[wp.float32],
    consec_rej: wp.array[wp.int32],
    guard_hits: wp.array[wp.int32],
):
    """Bound the damage of a truncated boundary.

    A world that cannot land within max_substeps would otherwise carry
    unbounded time debt and a collapsed controller into every later
    boundary: the debt compounds and recovery never outruns it. Cap the
    carried debt at one boundary and reissue a fresh controller so the
    next boundary attacks the shortfall at full growth.
    """
    i = wp.tid()
    if sim_time[i] < next_time[i]:
        floor_t = next_time[i] - dt_outer
        if sim_time[i] < floor_t:
            sim_time[i] = floor_t
        ideal_dt[i] = dt_init
        dt_ceiling[i] = ceiling_init
        consec_rej[i] = 0
        wp.atomic_add(guard_hits, 0, 1)


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
    dt_ceiling: wp.array[wp.float32],
    ceiling_init: float,
):
    """Restore the step-doubling controller's persistent per-world state to
    construction defaults for worlds flagged in ``mask``; leave others untouched.

    sim_time and next_time are reset TOGETHER to 0 so the world restarts a clean
    boundary interval (the next step_dt advances next_time by dt_outer from 0); this
    also bounds the float32 growth of a long-lived world."""
    i = wp.tid()
    if mask[i]:
        ideal_dt[i] = dt_init
        dt[i] = dt_init
        dt_half[i] = dt_init * wp.float32(0.5)
        sim_time[i] = wp.float32(0.0)
        next_time[i] = wp.float32(0.0)
        diverged[i] = False
        accepted[i] = False
        dt_ceiling[i] = ceiling_init


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
def _copy_rows_indexed(
    src: wp.array2d[wp.float32],
    idx: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slot: int,
    src_row_off: int,
    dst: wp.array2d[wp.float32],
):
    """Row copy ``dst[w] = src[src_row_off + w]`` for every world ``w`` in the
    index list. The (idx, counts, slot) triple exists so a device-built
    compacted world list can bound the work later; the count is read on device
    to keep the launch dim fixed under graph capture."""
    i, j = wp.tid()
    if i >= counts[slot]:
        return
    w = idx[i]
    dst[w, j] = src[src_row_off + w, j]


@wp.kernel
def _reset_active_counts(counts: wp.array[wp.int32]):
    """Zero both compaction counters (dim=1): slot 0 = active set, slot 1 =
    snapshot set. Must run before :func:`_build_active_worlds` each iteration
    (stream order guarantees it)."""
    counts[0] = 0
    counts[1] = 0


@wp.kernel
def _build_active_worlds(
    dt: wp.array[wp.float32],
    prev_active: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    active_idx: wp.array[wp.int32],
    snap_idx: wp.array[wp.int32],
):
    """Compact the unfinished worlds into index lists. Runs AFTER the boundary
    clamp so post-clamp ``dt > 0`` is the single source of truth for "this
    world attempts a step" -- the same predicate the masked restore keys on.
    The snapshot set keeps each world for ONE extra iteration after it lands so
    its saved rows capture the committed state; the full-dim restore then
    rewrites landed rows from those frozen bits every iteration, keeping them a
    true no-op. List order comes from atomics and is nondeterministic; every
    consumer writes world-private rows only, so ordering cannot perturb any
    floating-point result."""
    i = wp.tid()
    act = dt[i] > wp.float32(0.0)
    if act or prev_active[i] != 0:
        snap_idx[wp.atomic_add(counts, 1, 1)] = i
    if act:
        active_idx[wp.atomic_add(counts, 0, 1)] = i
        prev_active[i] = 1
    else:
        prev_active[i] = 0


@wp.kernel
def _boundary_reset(flag: wp.array[wp.int32]):
    """Set flag[0] = 0 (assume all worlds reached the boundary)."""
    flag[0] = 0


@wp.kernel
def _write_coupled_solref(
    dt: wp.array[wp.float32],
    mask: wp.array2d[wp.bool],
    solref: wp.array2d[wp.vec2],
):
    """Write timeconst = 2 * dt_attempt per world onto every coupled contact.

    Runs once per attempt AFTER the boundary clamp so both step-double
    trials integrate the same contact stiffness. Landed worlds (dt = 0) are
    skipped; dampratio is preserved.
    """
    w, g = wp.tid()
    if not mask[w, g]:
        return
    d = dt[w]
    if d <= wp.float32(0.0):
        return
    solref[w, g] = wp.vec2(wp.float32(2.0) * d, solref[w, g][1])


@wp.kernel
def _iters_exhausted_stop(iter_count: wp.array[wp.int32], max_iters: int, flag: wp.array[wp.int32]):
    """Latch the boundary loop closed once ``max_substeps`` attempts have run.

    Runs AFTER the boundary-exit check so the cap wins: the safety bound must
    stop the loop regardless of how many worlds are still short.
    """
    if iter_count[0] >= max_iters:
        flag[0] = 0


@wp.kernel
def _count_boundary_truncation(
    iter_count: wp.array[wp.int32],
    max_iters: int,
    out: wp.array[wp.int64],
):
    """Record one boundary, and whether it used the full ``max_substeps`` budget (dim=1).

    ``out[0]`` counts boundaries. ``out[1]`` counts boundaries that consumed the entire
    ``max_substeps`` budget -- this INCLUDES the case where every world happened to land
    exactly on the final permitted iteration, so it is not on its own proof that any world
    is short of its target time. It is still a genuine saturation signal: the boundary had
    no iterations to spare. For actual under-advance, see ``out[2]`` (``unfinished_worlds``,
    written by :func:`_count_unfinished_worlds`).
    """
    out[0] = out[0] + wp.int64(1)
    if iter_count[0] >= max_iters:
        out[1] = out[1] + wp.int64(1)


@wp.kernel
def _count_unfinished_worlds(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    out: wp.array[wp.int64],
):
    """Accumulate world-boundaries that ended short of the target time into ``out[2]``."""
    i = wp.tid()
    if sim_time[i] < next_time[i]:
        wp.atomic_add(out, 2, wp.int64(1))


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

    ``_sim_time`` and ``_next_time`` otherwise grow unbounded across a long run; the
    landing remainder ``next_time - sim_time`` then loses float32 precision as magnitude
    grows, causing dt jitter that worsens over time.
    Subtracting the per-world baseline ``next_time[i]`` (NOT zeroing) keeps both
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
    limited: wp.array[wp.int32],
):
    """Clamp dt so worlds don't overshoot their boundary target.

    Worlds already at or past the boundary get dt=0 (no-op step). ``limited[i]=1``
    marks a step artificially shortened by the boundary (a "landing sliver"), so the
    controller can exempt it from step-size resizing (Drake's artificially-limited
    rule, always on).
    """
    i = wp.tid()
    remaining = next_time[i] - sim_time[i]
    limited[i] = 0
    if remaining <= wp.float32(0.0):
        dt[i] = wp.float32(0.0)
        dt_half[i] = wp.float32(0.0)
    elif dt[i] > remaining:
        dt[i] = remaining
        dt_half[i] = remaining * wp.float32(0.5)
        limited[i] = 1


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

        t = 0.0
        while viewer.is_running():
            state_0, state_1 = solver.step_dt(DT, state_0, state_1, control, apply_forces=viewer.apply_forces)
            t += DT  # solver.sim_time is REBASED per boundary (~[0, DT]); accumulate absolute time here
            viewer.render(state_0, t)
    """

    def __init__(
        self,
        model: Model,
        *,
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-12,
        dt_inner_max: float | None = None,
        dt_mode: str = "per_world",
        tiling: str = "ragged",
        max_substeps: int = 256,
        use_newton_contacts: bool = False,
        dt_histogram: bool = False,
        dt_histogram_bins_per_decade: int = 4,
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
            dt_histogram: Accumulate a per-iteration histogram of the inner timestep so
                floor occupancy is measurable. Off by default: it adds one kernel launch
                per adaptive iteration, which would perturb benchmark timings. Must be set
                at construction — the flag is baked into the captured iteration graph.
            dt_histogram_bins_per_decade: Log-spaced bins per decade when
                ``dt_histogram`` is enabled.
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
        if int(dt_histogram_bins_per_decade) < 1:
            raise ValueError(f"dt_histogram_bins_per_decade must be >= 1, got {dt_histogram_bins_per_decade!r}")
        # Contact source is selectable: with ``use_newton_contacts=True`` the externally
        # provided Newton CollisionPipeline contacts (full non-convex mesh/SDF fidelity)
        # are injected ONCE per boundary and held across the step-doubling march —
        # identical contacts for every substep keep the error estimate consistent.
        # Otherwise MuJoCo re-collides each substep on ITS OWN geometry, which
        # convexifies meshes and only instantiates env_0's static geoms — unusable
        # for multi-env non-convex scenes.
        self._use_newton_contacts = bool(use_newton_contacts)
        # Sleeping cannot coexist with the adaptive march: the step-doubling
        # rollback restores only qpos/qvel/warmstart/act, so sleep bookkeeping
        # mutated by a trial eval would survive a reject, and the shared-forward
        # suffix eval skips the wake/sleep stages entirely. Force it off here --
        # the base otherwise inherits model.mujoco.enable_sleeping.
        if kwargs.get("enable_sleeping"):
            raise ValueError("SolverMuJoCoAdaptive does not support enable_sleeping=True.")
        kwargs["enable_sleeping"] = False
        # Single data-world per Newton world (the base solver supports replica
        # multiples; this solver no longer uses them).
        self._mjw_world_replicas = 1
        # Tail compaction (default ON; NEWTON_ADAPTIVE_TAIL_COMPACT=0 opts out):
        # late march iterations run full-batch kernels while only a few worlds
        # are still unfinished; a device-built index list of active worlds
        # bounds the solver-owned snapshot/seed/gather/error work to those
        # worlds while every launch keeps its fixed dim (graph-capture safe).
        # A pure scheduling change — march_equivalence_probe.py asserts bitwise
        # identity with the full-batch path and must keep passing for this
        # default to remain ON. Read at construction; the choice is baked into
        # the captured iteration graphs.
        self._tail_compact = os.environ.get("NEWTON_ADAPTIVE_TAIL_COMPACT", "1") != "0"
        super().__init__(
            model,
            separate_worlds=True,
            use_mujoco_cpu=False,
            use_mujoco_contacts=not self._use_newton_contacts,
            **kwargs,
        )

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
        # Construction default for the per-world controller reset, plus a
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
        # Explicit per-world shapes (NOT zeros_like the mjw_data fields):
        # controller state, snapshots, and the Richardson pair are per
        # world.
        self._qpos_saved = wp.zeros((world_count, self._nq), dtype=wp.float32, device=device)
        self._qvel_saved = wp.zeros((world_count, self._nv), dtype=wp.float32, device=device)
        self._qpos_full = wp.zeros((world_count, self._nq), dtype=wp.float32, device=device)
        # The snapshot also covers the solver warm start and actuator activations so a
        # rejected attempt is a TRUE rollback: act must not integrate on rejects (and the
        # full eval's act must not leak into the half evals -- the Richardson pair needs
        # both estimates to start from identical internal state), and a rejected/stalled
        # eval's qacc must not seed the next attempt's warm start.
        self._warmstart_saved = wp.zeros((world_count, self._nv), dtype=wp.float32, device=device)
        _act = getattr(self.mjw_data, "act", None)
        self._na = int(_act.shape[1]) if _act is not None and len(_act.shape) == 2 else 0
        self._act_saved = wp.zeros((world_count, self._na), dtype=wp.float32, device=device) if self._na > 0 else None

        # Boundary output buffer: _update_newton_state writes the final committed state
        # (+ FK'd body poses) here once per boundary, then it is copied into the caller's
        # state. Kept separate from state_0 so state==state_prev aliasing never occurs.
        self._state_cur = model.state()

        # ---- boundary-loop bookkeeping ----
        self._next_time = wp.zeros(world_count, dtype=wp.float32, device=device)
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)
        # Controller-fix state (see _clamp_dt_to_boundary / _calc_adjusted_step):
        # per-world boundary-limited flag + consecutive-rejection counter, and the two
        # opt-in fix switches (baked into the captured graph at construction).
        self._limited = wp.zeros(world_count, dtype=wp.int32, device=device)
        self._consec_rej = wp.zeros(world_count, dtype=wp.int32, device=device)
        # Ceiling memory (always on): per-world upper
        # bound on growth, recorded at rejections, relaxed on accepts. Init above any
        # reachable dt so it never binds until a rejection writes it.
        _ceiling_init = self._dt_max if self._dt_max != float("inf") else 1.0e6
        self._dt_ceiling = wp.full(world_count, wp.float32(_ceiling_init), dtype=wp.float32, device=device)
        # Non-finite (or catastrophic) state at the dt floor: NEVER commit it. Hold
        # the last valid state for this ONE frame and latch ``diverged``; the env
        # consumes the latch as a termination and resets the world the same step.
        # Non-resetting cumulative boundary-loop iteration count (NOT zeroed per
        # step_dt, unlike _iteration_count_buf). Each iteration runs the 3-eval
        # step-doubling attempt, so total MuJoCo opt-steps = iterations * 3, and
        # rejected attempts are counted (a rejection is just another iteration).
        # Used as the compute axis for work-precision. Reset with
        # reset_compute_counter().
        self._cum_iters = wp.zeros(1, dtype=wp.int32, device=device)

        # Opt-in per-boundary march telemetry (see _log_march_boundary). The
        # env gate keeps the hot path to a single attribute check when unset.
        self._march_log_path = os.environ.get("NEWTON_ADAPTIVE_MARCH_LOG") or None
        self._march_log_file = None
        self._march_log_boundary = 0
        self._march_log_hist_every = 48
        self._reject_count_buf = wp.zeros(1, dtype=wp.int32, device=device)

        # Opt-in truncation guard (see _debt_guard). Off until validated.
        self._dt_init = float(dt_inner_init)
        self._guard_hits = wp.zeros(1, dtype=wp.int32, device=device)
        # Opt-in mixed-tolerance normalization: accept test becomes
        # |d| <= atol + rtol*|q| with atol = tol. Zero disables (bit-identical
        # legacy path).
        self._err_rtol = float(os.environ.get("NEWTON_ADAPTIVE_RTOL", "2e-6") or 0.0)
        self._err_rtol_over_atol = self._err_rtol / self._tol if self._err_rtol > 0.0 else 0.0

        # dt-occupancy histogram (opt-in). Allocated here, never inside the captured
        # iteration body. int64 because a long run at 4096 worlds overflows int32.
        self._dt_hist: wp.array | None = None
        self._dt_hist_sat: wp.array | None = None
        self._dt_hist_trunc: wp.array | None = None
        self._dt_hist_bpd = int(dt_histogram_bins_per_decade)
        self._dt_hist_n_bins = 0
        self._dt_hist_lo_log10 = 0.0
        if dt_histogram:
            self._dt_hist_n_bins, self._dt_hist_lo_log10 = _dt_hist_layout(
                self._dt_min, self._dt_inner_init, self._dt_hist_bpd
            )
            self._dt_hist = wp.zeros(self._dt_hist_n_bins, dtype=wp.int64, device=device)
            self._dt_hist_sat = wp.full(1, _DT_HIST_SENTINEL, dtype=wp.float32, device=device)
            # [0] boundaries, [1] truncated by max_substeps, [2] world-boundaries short of target
            self._dt_hist_trunc = wp.zeros(3, dtype=wp.int64, device=device)

        # Stable buffer for opt.timestep; updated via wp.copy() per substep.
        self._timestep_buf = wp.full(world_count, dt_inner_init, dtype=wp.float32, device=device)
        self.mjw_model.opt.timestep = self._timestep_buf

        # dt-coupled contact regularization (NEWTON_ADAPTIVE_CONTACT_COUPLING,
        # default ON; set 0 to keep authored materials): every positive-
        # timeconst contact runs at timeconst = 2 * THE ATTEMPT'S dt, written
        # per world per iteration BEFORE the step-double trials. The write is
        # per-ATTEMPT, not per-evaluation, because the error estimator's
        # foundation requires the full step and the two half steps to
        # integrate IDENTICAL contact parameters — letting refsafe re-derive
        # the timeconst per sub-evaluation hands the half steps 4x the full
        # step's stiffness and the full-vs-half difference then converges to
        # a constant (~ the standing penetration depth) instead of vanishing
        # with dt. With 2*dt_attempt >= 2*dt_eval the refsafe clamp stays
        # inactive for both evals. Constraint enforcement still tightens as
        # dt^-2 wherever the controller refines. Non-positive entries are
        # direct k/b specifications and are never touched; dampratio is
        # preserved; landed worlds (dt = 0) keep their previous values, which
        # nothing commits.
        self._coupled_solref_targets: list = []

        # ---- tail-compaction scratch (see _build_active_worlds) ----
        if self._tail_compact:
            # counts[0] = active-set size (worlds attempting a step this
            # iteration, post-boundary-clamp dt > 0); counts[1] = snapshot-set
            # size (active OR active last iteration: a world landing at
            # iteration k is snapshotted once more at k+1 so its saved rows
            # freeze at the committed state, which the full-dim restore then
            # rewrites bit-identically on every landed iteration). Rebuilt on
            # device each iteration; every consumer keeps its FIXED launch dim
            # and early-exits on the device-read count, so the launches record
            # cleanly inside captured graphs.
            self._active_counts = wp.zeros(2, dtype=wp.int32, device=device)
            self._active_idx = wp.zeros(world_count, dtype=wp.int32, device=device)
            self._snap_idx = wp.zeros(world_count, dtype=wp.int32, device=device)
            self._prev_active = wp.zeros(world_count, dtype=wp.int32, device=device)

        # Accuracy-metric scaling S: e = || S (q - q̂) ||_inf. Default S = identity.
        # To use expert per-coordinate scales, overwrite self._state_scale
        # (shape [world_count, nq]; scales MuJoCo qpos components) after construction.
        self._state_scale = wp.array(
            np.ones((world_count, self._nq), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        # ---- injected-contact refresh ----
        # With use_newton_contacts, the boundary call converts the CollisionPipeline
        # contacts ONCE with boundary-entry transforms. The inner march must NOT reuse
        # the boundary ``dist``: a stale negative dist keeps firing penalty force into
        # a body that has already separated — energy injection the step-doubling
        # controller CANNOT see (both trial solutions share the snapshot, so the error
        # cancels in the comparison). So dist/pos are re-anchored to the CURRENT
        # marching state at every iteration via the conversion kernel's fast path
        # (frame/solref/friction stay from the boundary full pass).
        # NEWTON_ADAPTIVE_CONTACT_REFRESH=0 is a DIAGNOSTIC-ONLY escape hatch for
        # cost attribution: without the refresh the error estimate is blind to
        # contact (the energy-pumping defect returns), so it must never run as
        # an experiment arm.
        self._contact_refresh_enabled = (
            self._use_newton_contacts and os.environ.get("NEWTON_ADAPTIVE_CONTACT_REFRESH", "1") != "0"
        )
        self._refresh_state = self.model.state() if self._contact_refresh_enabled else None
        self._refresh_contacts = None
        self._refresh_contacts_key = None

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

        # ---- conditional-march tier (opt-in) ----
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

        # ---- shared forward prefix (DEFAULT ON; cuts one prefix pass of three per iteration) ----
        # The full eval and the FIRST half eval of every step-doubling attempt start from
        # the identical snapshot state, so the state-dependent pipeline prefix
        # (kinematics, collision, mass matrix + factorization, bias/passive/actuator
        # forces, qacc_smooth) is computed twice on identical inputs. The first half
        # eval therefore runs only the timestep-dependent suffix -- constraint assembly
        # (KBI terms scale with dt), constraint solve, integrate -- reusing the prefix
        # the full eval left in mjw_data (its integrator advanced only qpos/qvel/act,
        # which the rollback restores; smooth.py has no timestep dependence). Bonus:
        # both Richardson estimates then judge the IDENTICAL contact set.
        # NEWTON_MJ_ADAPTIVE_SHARED_FWD=0 opts out. Euler/implicitfast only: RK4
        # re-runs forward inside its integrator and falls back to full evals
        # (silently, unless the flag was set explicitly).
        _shared_fwd_env = os.environ.get("NEWTON_MJ_ADAPTIVE_SHARED_FWD")
        self._shared_fwd = (_shared_fwd_env or "1") == "1"
        self._mjw_suffix_integrate = None
        if self._shared_fwd:
            mjw = self._mujoco_warp
            integrator = self.mjw_model.opt.integrator
            if integrator == mjw.IntegratorType.EULER:
                self._mjw_suffix_integrate = mjw.euler
            elif integrator == mjw.IntegratorType.IMPLICITFAST:
                self._mjw_suffix_integrate = mjw.implicit
            else:
                self._shared_fwd = False
                if _shared_fwd_env is not None:
                    warnings.warn(
                        f"NEWTON_MJ_ADAPTIVE_SHARED_FWD: integrator {integrator!r} does not support the "
                        "shared forward prefix (RK4 re-runs forward internally); using full evals.",
                        stacklevel=2,
                    )

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

    def _mjw_eval_suffix(self, dt_array: wp.array) -> None:
        """Reduced eval for the FIRST half step of the Richardson pair (shared forward
        prefix): only the timestep-dependent stages run at ``dt_array`` -- constraint
        assembly (``make_constraint``: KBI/regularizer terms scale with dt), the
        constraint solve, and integration.

        Every state-dependent prefix output in ``mjw_data`` -- kinematics, contacts,
        mass matrix + factorization, bias/passive/actuator forces, ``qacc_smooth``,
        actuator moments -- was computed by the immediately preceding FULL eval from
        the SAME snapshot state and is still valid: that eval's integrator advanced
        only qpos/qvel/act (which the caller restores from the snapshot first), never
        the derived quantities. ``make_constraint`` reassembles efc from those stored
        contacts/kinematics and the restored qpos, so the constraint set matches the
        full eval's exactly. Sensors are skipped (nothing consumes mid-attempt sensor
        values; the final committed eval computes them).
        """
        mjw = self._mujoco_warp
        m, d = self.mjw_model, self.mjw_data
        wp.copy(m.opt.timestep, dt_array)
        with wp.ScopedDevice(self.model.device):
            if self._alloc_cache is not None:
                with self._alloc_cache.scope():
                    mjw.make_constraint(m, d)
                    mjw.solve(m, d)
                    self._mjw_suffix_integrate(m, d)
            else:
                mjw.make_constraint(m, d)
                mjw.solve(m, d)
                self._mjw_suffix_integrate(m, d)

    def _step_double(self) -> None:
        """Step doubling -- the 3 MuJoCo evals, entirely in qpos/qvel space: snapshot,
        one full step at ``dt`` (result saved to ``_qpos_full``), restore, then two half
        steps at ``dt/2``. ``mjw_data`` ends holding the doubled state; ``_qpos_full``
        vs ``mjw_data.qpos`` is the Richardson pair the error kernel differences.

        The snapshot/restore includes ``qacc_warmstart`` and ``act`` so the two
        Richardson estimates start from identical internal state (the full eval's
        warm start and activations must not leak into the half evals).

        With the shared forward prefix (default on; ``NEWTON_MJ_ADAPTIVE_SHARED_FWD=0``
        opts out), the FIRST half eval reuses the full eval's state-dependent prefix and
        runs only the dt-dependent suffix (see :meth:`_mjw_eval_suffix`); the second half
        eval starts from a different state and always runs the full pipeline.
        """
        d = self.mjw_data
        if self._tail_compact:
            # Compacted snapshot (slot 1 = snapshot set): landed worlds past
            # their one snapshot-freeze iteration stop paying the copy
            # bandwidth; their saved rows already hold the committed bits the
            # full restores below rewrite them from.
            n = self.model.world_count
            dev = self.model.device
            for src, dst, width in (
                (d.qpos, self._qpos_saved, self._nq),
                (d.qvel, self._qvel_saved, self._nv),
                (d.qacc_warmstart, self._warmstart_saved, self._nv),
            ):
                wp.launch(
                    _copy_rows_indexed,
                    dim=(n, width),
                    inputs=[src, self._snap_idx, self._active_counts, 1, 0],
                    outputs=[dst],
                    device=dev,
                )
            if self._na > 0:
                wp.launch(
                    _copy_rows_indexed,
                    dim=(n, self._na),
                    inputs=[d.act, self._snap_idx, self._active_counts, 1, 0],
                    outputs=[self._act_saved],
                    device=dev,
                )
        else:
            wp.copy(self._qpos_saved, d.qpos)
            wp.copy(self._qvel_saved, d.qvel)
            wp.copy(self._warmstart_saved, d.qacc_warmstart)
            if self._na > 0:
                wp.copy(self._act_saved, d.act)
        self._mjw_eval(self._dt)
        if self._tail_compact:
            # Gather the full-step result for active worlds only (slot 0):
            # landed worlds' _qpos_full rows go stale, which is safe because
            # only the (equally compacted) error kernel reads them.
            wp.launch(
                _copy_rows_indexed,
                dim=(n, self._nq),
                inputs=[d.qpos, self._active_idx, self._active_counts, 0, 0],
                outputs=[self._qpos_full],
                device=dev,
            )
        else:
            wp.copy(self._qpos_full, d.qpos)
        # The mid-double restore stays a FULL copy in every mode: it must
        # rewrite the real rows the full eval just advanced, and for landed
        # worlds the frozen saved rows equal the committed bits, so the full
        # rewrite is exact.
        wp.copy(d.qpos, self._qpos_saved)
        wp.copy(d.qvel, self._qvel_saved)
        wp.copy(d.qacc_warmstart, self._warmstart_saved)
        if self._na > 0:
            wp.copy(d.act, self._act_saved)
        if self._shared_fwd:
            self._mjw_eval_suffix(self._dt_half)
        else:
            self._mjw_eval(self._dt_half)
        # Injected contacts: re-anchor dist/pos to the half-step state before the
        # second half eval. All three evals sharing one contact snapshot makes the
        # Richardson pair's contact response identical, so contact stiffness
        # cancels out of the error estimate and the controller grows dt straight
        # through contact (energy-pumping resting objects). The mid-double refresh
        # restores the estimator's contact sensitivity; MuJoCo-native contacts get
        # the same effect from re-detection inside every eval.
        if self._contact_refresh_enabled and self._refresh_contacts is not None:
            self._refresh_injected_contacts()
        self._mjw_eval(self._dt_half)

    def _estimate_error(self) -> None:
        """Per-world local error: inf-norm ``e = max|Δq|`` between the full step and the doubled
        half-step (NaN/inf collapse to a 1e10 sentinel). Writes ``_last_error``.

        Under tail compaction only active worlds get a fresh error; landed
        worlds keep the last written value, which never reaches a controller
        decision (the dt <= 0 branch returns first)."""
        if self._tail_compact:
            wp.launch(
                _inf_norm_state_error_indexed_kernel,
                dim=self.model.world_count,
                inputs=[
                    self._qpos_full,
                    self.mjw_data.qpos,
                    self._state_scale,
                    self._nq,
                    self.mjw_data.qvel,
                    self._nv,
                    self._active_idx,
                    self._active_counts,
                    0,
                    self._err_rtol_over_atol,
                ],
                outputs=[self._last_error],
                device=self.model.device,
            )
            return
        wp.launch(
            _inf_norm_state_error_kernel,
            dim=self.model.world_count,
            inputs=[
                self._qpos_full,
                self.mjw_data.qpos,
                self._state_scale,
                self._nq,
                self.mjw_data.qvel,
                self._nv,
                self._err_rtol_over_atol,
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
    def _refresh_injected_contacts(self) -> None:
        """Re-anchor injected Newton contacts to the CURRENT marching state.

        mjw qpos/qvel -> Newton joint coords -> FK body poses -> conversion-kernel
        FAST path (recomputes contact ``dist``/``pos`` only; frame, solref, friction
        and the compacted layout stay from the boundary full pass). Flat kernel
        launches on buffers preallocated in ``__init__`` — records cleanly inside
        the captured iteration graph.
        """
        model = self.model
        st = self._refresh_state
        mjw = self.mjw_data
        # The refresh scratch state and the Newton joint
        # layout are per-world sized.
        joints_per_world = model.joint_count // model.world_count
        mujoco_attrs = getattr(model, "mujoco", None)
        dof_ref = getattr(mujoco_attrs, "dof_ref", None) if mujoco_attrs is not None else None
        # st.joint_q/qd double as state_prev: the kernel copies kinematic dofs
        # prev -> out element-wise, which is alias-safe.
        wp.launch(
            convert_mj_coords_to_warp_kernel,
            dim=(model.world_count, joints_per_world),
            inputs=[
                mjw.qpos,
                mjw.qvel,
                joints_per_world,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.body_com,
                dof_ref,
                model.body_flags,
                st.joint_q,
                st.joint_qd,
                self.mj_q_start,
                self.mj_qd_start,
            ],
            outputs=[st.joint_q, st.joint_qd],
            device=model.device,
        )
        wp.launch(
            kernel=eval_articulation_fk,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.articulation_end,
                model.articulation_count,
                None,
                None,
                model.joint_articulation,
                st.joint_q,
                st.joint_qd,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_flags,
                int(BodyFlags.ALL),
            ],
            outputs=[st.body_q, st.body_qd],
            device=model.device,
        )
        self._convert_contacts_to_mjwarp(model, st, self._refresh_contacts)

    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """ONE ragged adaptive iteration: histogram sample (if enabled) -> clamp -> step-double
        -> error -> Drake controller -> masked rollback -> advance -> dt cap -> boundary check.
        All in MuJoCo qpos/qvel space.

        The histogram sample MUST run before the clamp: it bins the controller's chosen
        step, and after ``_clamp_dt_to_boundary`` runs, ``dt`` may instead hold a
        boundary-landing sliver, binning the sample into the wrong bucket (see
        :func:`_dt_histogram_accum`).

        This is the body the device-side conditional loop replays (see :meth:`_march_ragged`).
        Every phase is a flat kernel-launch sequence, so it records cleanly inside a CUDA
        graph / conditional while-node.
        """
        n = self.model.world_count
        dev = self.model.device

        # Sample BEFORE _clamp_dt_to_boundary: dt still holds the controller's chosen
        # step here, not a landing sliver.
        if self._dt_hist is not None:
            wp.launch(
                _dt_histogram_accum,
                dim=n,
                inputs=[
                    self._dt,
                    self._ideal_dt,
                    self._sim_time,
                    self._next_time,
                    self._dt_min,
                    self._dt_hist_lo_log10,
                    float(self._dt_hist_bpd),
                    self._dt_hist_n_bins,
                    self._dt_hist,
                    self._dt_hist_sat,
                ],
                device=dev,
            )

        # Count this attempt (per-step + cumulative). A rejection is just another iteration.
        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        # Never overshoot the boundary; worlds already at it get dt=0 (no-op step).
        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time, self._limited],
            device=dev,
        )

        # Compact the unfinished worlds AFTER the clamp (post-clamp dt > 0 is
        # the single "attempts a step" predicate) and before anything consumes
        # the lists. Fixed-dim launches; the counts live on device.
        if self._tail_compact:
            wp.launch(_reset_active_counts, dim=1, inputs=[self._active_counts], device=dev)
            wp.launch(
                _build_active_worlds,
                dim=n,
                inputs=[self._dt, self._prev_active, self._active_counts, self._active_idx, self._snap_idx],
                device=dev,
            )

        # Attempt-consistent contact coupling: both step-double trials must see
        # identical contact parameters (see the constructor block).
        for _arr, _mask in self._coupled_solref_targets:
            wp.launch(
                _write_coupled_solref,
                dim=(n, _arr.shape[1]),
                inputs=[self._dt, _mask, _arr],
                device=dev,
            )

        # Re-anchor injected contacts to the current committed state BEFORE the
        # trials: they must not push against boundary-stale penetration (energy
        # injection at footfall — the "flying robots" failure).
        if self._contact_refresh_enabled and self._refresh_contacts is not None:
            self._refresh_injected_contacts()

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
                self._dt_max if self._dt_max != float("inf") else 1.0e6,
                self._divergence_threshold,
                self._dt_ceiling,
                self._limited,
                self._consec_rej,
                self._sim_time,
                self._next_time,
            ],
            device=dev,
        )
        if self._march_log_path is not None:
            wp.launch(_count_rejects, dim=n, inputs=[self._accepted, self._reject_count_buf], device=dev)

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
        # Full-integrity boundary exit: every world marches to its boundary —
        # no abandonment, no forced completion.
        wp.launch(
            mark_unfinished,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )
        wp.launch(
            _iters_exhausted_stop,
            dim=1,
            inputs=[self._iteration_count_buf, self._max_substeps, self._boundary_flag],
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
        """Advance every world by exactly ``dt`` (= ``dt_outer``) seconds of sim time:
        the boundary call.

        Newton solver signature ``(state_in, state_out, control, contacts, dt)``. ``state_in`` is
        read and written in place; ``state_out`` is returned unchanged (scratch). ``contacts`` is
        UNUSED when constructed with MuJoCo-native contacts (the default), and REQUIRED
        when ``use_newton_contacts=True`` — injected once per boundary, with dist/pos
        re-anchored per iteration while the contact refresh is enabled.

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

        # The whole boundary call executes MJWarp code, so it runs under the base
        # solver's MJWarp compilation/execution options (module options, generated-
        # kernel invalidation, deterministic config), exactly like SolverMuJoCo.step.
        with wp.ScopedDevice(device), self._scoped_mujoco_warp_execution():
            self._apply_mjc_control(self.model, state_0, control, self.mjw_data)
            if apply_forces is not None:
                apply_forces(state_0)

            self._enable_rne_postconstraint(self._state_cur)

            # Load the incoming Newton state into MuJoCo coordinates ONCE per boundary;
            # the whole inner loop then marches mjw_data.qpos/qvel directly.
            self._update_mjc_data(self.mjw_data, self.model, state_0)
            if self._use_newton_contacts:
                if contacts is None:
                    raise ValueError(
                        "SolverMuJoCoAdaptive(use_newton_contacts=True) requires contacts= at step(); "
                        "feed the Newton CollisionPipeline contacts from the caller."
                    )
                # Convert once with boundary-entry transforms (FULL path:
                # frame/solref/friction/layout); the inner march refreshes dist/pos per
                # iteration via _refresh_injected_contacts (fast path).
                self._convert_contacts_to_mjwarp(self.model, state_0, contacts)
                if self._contact_refresh_enabled:
                    # Seed the refresh scratch (kinematic dofs are copied from prev).
                    self._copy_state(self._refresh_state, state_0)
                    self._refresh_contacts = contacts
                    # A different Contacts object would leave stale device pointers
                    # baked into the captured iteration graphs — drop the caches.
                    key = id(contacts.contact_generation)
                    if key != self._refresh_contacts_key:
                        self._refresh_contacts_key = key
                        self._march_graph_cache.clear()
                        self._conditional_graph_cache.clear()

            # Rebase both clocks by the per-world boundary so float32 magnitude stays bounded
            # (prevents landing-remainder precision loss / dt jitter that grows over a run). The
            # subtract-baseline preserves the remaining time exactly; do this BEFORE advancing next_time.
            wp.launch(_rebase_time, dim=n, inputs=[self._sim_time, self._next_time], device=device)
            wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

            self._iteration_count_buf.fill_(0)
            self._boundary_flag.fill_(1)
            if self._march_log_path is not None:
                self._reject_count_buf.fill_(0)
            self._guard_hits.fill_(0)
            if self._tail_compact:
                # Every world starts the boundary active; the fill also forces
                # the first iteration's snapshot set to cover all worlds, so
                # saved rows are valid before any restore reads them.
                self._prev_active.fill_(1)

            self._march_ragged(effective_dt_max)

            # Once per boundary, outside the captured body: no per-iteration cost.
            if self._dt_hist_trunc is not None:
                wp.launch(
                    _count_boundary_truncation,
                    dim=1,
                    inputs=[self._iteration_count_buf, self._max_substeps, self._dt_hist_trunc],
                    device=device,
                )
                wp.launch(
                    _count_unfinished_worlds,
                    dim=n,
                    inputs=[self._sim_time, self._next_time, self._dt_hist_trunc],
                    device=device,
                )

            # Bound truncation damage BEFORE the next boundary consumes the
            # carried state; device-only work, legal graphs on or off.
            wp.launch(
                _debt_guard,
                dim=n,
                inputs=[
                    self._sim_time,
                    self._next_time,
                    dt_outer,
                    self._dt_init,
                    self._dt_max if self._dt_max != float("inf") else 1.0e6,
                    self._ideal_dt,
                    self._dt_ceiling,
                    self._consec_rej,
                    self._guard_hits,
                ],
                device=device,
            )

            # Opt-in telemetry readout: host reads of post-march state, which
            # are only legal outside an active capture -- hence gated, never
            # unconditional. Runs after the guard, so resid reflects the
            # bounded carry while n_guard reports the worlds it touched.
            if self._march_log_path is not None:
                self._log_march_boundary()

            # Convert the final committed state back to Newton coordinates ONCE per boundary
            # (incl. the FK pass for body_q/body_qd), then hand it to the caller. _state_cur
            # buffers the write so state and state_prev never alias in the convert kernel.
            self._update_newton_state(self.model, self._state_cur, self.mjw_data, state_prev=state_0)
            self._copy_state(state_0, self._state_cur)

        return state_0, state_1

    def _log_march_boundary(self) -> None:
        """Append one CSV row of post-march telemetry for the finished boundary.

        Pure observer: reads the boundary's counters and the controller-carry
        arrays the march has already committed; writes nothing back to solver
        state. Host reads are illegal while a capture is recording, so the
        caller gates this on the opt-in env var instead of running it
        unconditionally.
        """
        if self._march_log_file is None:
            # Line-buffered so a killed run keeps its tail.
            self._march_log_file = open(self._march_log_path, "a", buffering=1)
            self._march_log_file.write(
                "boundary,iters,cum_iters,ideal_min,ideal_mean,ideal_max,"
                "resid_min,resid_max,rejects,err_max,n_debt,n_subfloor,n_guard,"
                "eworld,qmax_i,qmax,ncon\n"
            )
        iters = int(self._iteration_count_buf.numpy()[0])
        cum = int(self._cum_iters.numpy()[0])
        ideal = self._ideal_dt.numpy()
        resid = self._next_time.numpy() - self._sim_time.numpy()
        rejects = int(self._reject_count_buf.numpy()[0])
        err_arr = self._last_error.numpy()
        err_max = float(err_arr.max())
        # Name the worst world's largest-magnitude coordinate: which qpos
        # slot, and how big. Identifies WHAT the error norm is reading when
        # err_max pins at a grid constant.
        eworld = int(err_arr.argmax())
        q_abs = np.abs(self.mjw_data.qpos.numpy()[eworld])
        qmax_i = int(q_abs.argmax())
        qmax = float(q_abs[qmax_i])
        ncon = int(self.mjw_data.nacon.numpy()[0])
        n_debt = int((resid > 0.0).sum())
        n_subfloor = int((ideal < self._dt_min).sum())
        n_guard = int(self._guard_hits.numpy()[0])
        self._march_log_file.write(
            f"{self._march_log_boundary},{iters},{cum},"
            f"{ideal.min():.6e},{ideal.mean():.6e},{ideal.max():.6e},"
            f"{resid.min():.6e},{resid.max():.6e},"
            f"{rejects},{err_max:.6e},{n_debt},{n_subfloor},{n_guard},"
            f"{eworld},{qmax_i},{qmax:.6e},{ncon}\n"
        )
        self._march_log_boundary += 1
        if self._dt_hist is not None and self._march_log_boundary % self._march_log_hist_every == 0:
            self._march_log_file.write(f"HIST {self.dt_histogram_stats()}\n")

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
        orphan graph, which manifests later as bogus OOMs. Belt-and-braces alongside
        ScopedCapture's own cleanup."""
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
        """Restore per-world adaptive-controller state for reset worlds.

        Overrides :meth:`SolverMuJoCo.reset` (which clears MuJoCo warm-start
        buffers and, per ``flags``, resets joint state to model defaults) and
        ADDITIONALLY restores this controller's persistent per-world buffers
        (ideal_dt/dt/dt_half/sim_time/next_time + the accepted/diverged latches)
        to construction defaults, so pre-reset controller state never leaks into
        the post-reset (s,a)->s' map. Also the consumer of the ``diverged``
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
                self._dt_ceiling,
                self._dt_max if self._dt_max != float("inf") else 1.0e6,
            ],
            device=self.model.device,
        )

    # =====================================================================
    # Telemetry / properties
    # =====================================================================
    @property
    def diverged(self) -> wp.array:
        """Per-world divergence latch, shape ``[world_count]``, bool, on device.

        Set by the NaN-guard floor path in :func:`_calc_adjusted_step` (on by
        default; ``NEWTON_ADAPTIVE_NAN_GUARD=0`` disables it, leaving this
        all-False). Cleared per world by :meth:`reset`.
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
    def dt_histogram(self) -> wp.array | None:
        """Per-bin counts of the inner timestep SELECTED per iteration, shape ``[n_bins]``,
        int64, on device.

        ``None`` unless the solver was constructed with ``dt_histogram=True``. Each sample is
        the controller's chosen step for that iteration, taken BEFORE the boundary-landing
        clamp (see :func:`_dt_histogram_accum`); on a landing iteration the step actually
        integrated is smaller than what is binned here. Bin 0 counts iterations where the
        selected step was already at/below the ``dt_inner_min`` floor, bins ``1 .. n_bins - 2``
        are log-spaced (see :attr:`dt_histogram_edges`), and the last bin absorbs everything
        above the range. Read with ``.numpy()`` OUTSIDE the inner loop only (it is a device
        sync).
        """
        return self._dt_hist

    @property
    def dt_histogram_edges(self) -> np.ndarray | None:
        """Edges [s] of the log-spaced histogram bins, shape ``[n_bins - 1]``.

        ``None`` unless ``dt_histogram=True``. The floor and overflow bins are open-ended
        and contribute no finite edge, so this is one shorter than :attr:`dt_histogram`.
        """
        if self._dt_hist is None:
            return None
        return _dt_hist_edges(self._dt_min, self._dt_hist_n_bins, self._dt_hist_bpd)

    def reset_dt_histogram(self) -> None:
        """Zero the histogram and saturation accumulators.

        Call after warmup so graph capture and initial dt settling stay out of the counts.
        No-op when the histogram is disabled.
        """
        if self._dt_hist is None:
            return
        self._dt_hist.zero_()
        self._dt_hist_sat.fill_(_DT_HIST_SENTINEL)
        self._dt_hist_trunc.zero_()

    def dt_histogram_stats(self) -> dict[str, float | int]:
        """Scalar summary of floor occupancy. Host sync; call outside the hot path.

        Returns:
            ``total_samples`` (attempted inner steps counted), ``floor_samples``,
            ``floor_fraction`` (0..1), ``saturation_depth`` -- the smallest
            ``ideal_dt`` [s] the controller asked for while clamped to the floor, or
            ``0.0`` if the floor was never hit -- ``boundaries`` (``step`` calls
            counted), ``capped_boundaries`` (boundaries that used the full
            ``max_substeps`` budget; this includes boundaries where every world
            happened to land exactly on the final permitted iteration, so it is not
            on its own proof of under-advance), and ``unfinished_worlds``
            (world-boundaries that ended with ``sim_time < next_time``, i.e. the
            actual under-advance measure).

        Raises:
            RuntimeError: If the solver was not constructed with ``dt_histogram=True``.
        """
        if self._dt_hist is None:
            raise RuntimeError("dt_histogram_stats() requires SolverMuJoCoAdaptive(dt_histogram=True)")
        counts = self._dt_hist.numpy()
        total = int(counts.sum())
        floor = int(counts[0])
        sat = float(self._dt_hist_sat.numpy()[0])
        # _DT_HIST_SENTINEL (1e38) is not exactly representable in float32: the untouched
        # accumulator reads back as 9.9999997e37, which is < 1.0e38. Comparing against the
        # raw double literal would never fire, so "floor never hit" must compare against
        # the SAME float32 round-trip the accumulator itself went through.
        never_hit = sat >= float(np.float32(_DT_HIST_SENTINEL))
        trunc = self._dt_hist_trunc.numpy()
        return {
            "total_samples": total,
            "floor_samples": floor,
            "floor_fraction": (floor / total) if total else 0.0,
            "saturation_depth": 0.0 if never_hit else sat,
            "boundaries": int(trunc[0]),
            "capped_boundaries": int(trunc[1]),
            "unfinished_worlds": int(trunc[2]),
        }

    @property
    def sim_time(self) -> wp.array:
        """Per-world simulation time [s], shape ``[world_count]``, float32, on device.

        Only advances for accepted steps. Rebased to its outer boundary at the start of
        each :meth:`step_dt` (float32 time-rebase), so this is the time WITHIN the
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
        """Inf-norm state error from the most recent attempt (accepted or rejected),
        shape ``[world_count]``, float32, on device."""
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
