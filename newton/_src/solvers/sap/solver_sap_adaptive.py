# SPDX-License-Identifier: Apache-2.0
"""Per-world error-controlled (step-doubling) SAP solver: ``SolverSAPAdaptive``.

This is the CENIC integrator for the convex SAP contact solver. The manager hands
it a state and a control boundary period ``dt_outer``; it advances every world to
that boundary **entirely on the GPU** using a TRUE per-world adaptive step and
returns the advanced state. The returned state has controlled LOCAL error per step
(unlike a fixed step, where accuracy is whatever the step lands on). The research
HYPOTHESIS -- not yet validated -- is that error-controlled ``(s, a, s')`` transitions
improve policy transfer to hardware.

The per-world primitive is a **dt vector** -- ``dt[world]`` -- never a substep count
``N``. Each world adapts ITS OWN dt from ITS OWN step-doubling error estimate; there
is no shared/global/batch-max dt and no cross-world reduction, so ``P(s'|s,a)`` for
one world never depends on another (the MDP stays Markov and per-world).

The machine (the per-substep body is one flat, capturable launch stream; the loop
stops the instant every world has reached its boundary)::

    next_time[w] = sim_time[w] + dt_outer
    for _ in range(max_substeps):                  # max_substeps is a SAFETY cap, not the work
        clamp_dt_to_boundary(dt, sim_time, next_time)   # done worlds -> dt=0; never overshoot
        substep(state_cur -> full,   dt)                # one inner SAP step at the per-world dt
        substep(state_cur -> mid,    dt/2)              # (adaptive only: step-doubling)
        substep(mid       -> double, dt/2)
        err = infnorm(full, double)                     # per-world local error estimate
        adapt_dt(err, ...):                             # ALL per-thread, in-kernel (data branches):
            DONE   (sim_time>=next_time): no-op
            ACCEPT (err<=tol):  state_cur=double; sim_time+=dt; grow dt
            REJECT (err>tol):   hold state; shrink dt; retry next iteration
        if no world is unfinished:  break              # ONE 4-byte host flag read per iteration
    write state_cur back

The ONLY host sync in the step path is a 4-byte boundary-status read: once per
iteration on the per-iteration tier (it lets the loop stop as soon as every world
lands instead of grinding a fixed count of wasted no-op substeps -- a ``dt=0`` no-op
still runs the full batched SAP solve, so wasted iterations are NOT free), or ONCE
PER BOUNDARY on the whole-march conditional tier (default ON;
``NEWTON_SAP_ADAPTIVE_CONDITIONAL=0`` opts out), which records the entire march as
one ``wp.capture_while`` conditional while-node -- the solve's own device
conditionals nest inside it -- and keeps only the post-march converge-or-throw
status check on the host. Reject is a masked
state-hold (a data branch), not control flow. The inner SAP solve is run CONVERGENT (to a
fixed ``optimality_rel_tol``, independent of ``tol``) so its residual
cannot pollute the step-doubling error estimate at any integration tolerance;
the target is coupled to the selected solve precision (``1e-8`` on the fp64
default; the fp32-achievable analogue when ``solve_precision="fp32"`` is
opted into -- see ``_FP32_OPTIMALITY_K``).
A per-world inner-solve failure is CONTAINED by default: the failing world's
attempt rejects (its error is forced to the divergence sentinel, so the
unconverged result is never committed) and retries at a shrunken dt, latching
``diverged`` at the dt floor -- the same per-world path a NaN state takes,
while every other world marches on. A floor-latch freezes the world's
committed state and force-advances its clock to the boundary, so the latch
(:attr:`diverged`, read after the boundary call) is the CONSUMER'S only signal
that the world's trajectory is broken; recovering the world -- resetting it
and terminating its episode -- is the consumer's responsibility, the solver
only reports.
``NEWTON_SAP_CONTAINMENT=0`` restores the strict batch-fatal converge-or-throw. The substep body -- including the solve's
``wp.capture_while`` -- captures as a flat launch stream, replayed per iteration on the
per-iteration tier and re-recorded as the while-node body on the conditional tier.

Two modes (one machine; mode only changes how ``dt`` evolves, set per solver) -- these are
the only two compared:

  * ``"fixed"``      -- constant dt, error control off; commits the single full step
    (the baseline being beaten on accuracy). Skips the doubling it does not use.
  * ``"adaptive"``   -- dt grows on accept / shrinks on reject WITHIN the frame from
    the step-doubling error, targeting local error <= tol per step per world.

This file is self-contained: every controller kernel is inlined below (no shared
``adaptive.controller_kernels`` import) and there is no global/even-tiling code.
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings

import numpy as np
import warp as wp

# sys.path is configured by the package __init__ before this module is imported.
from sim.contact_solve import _env_grid_capacity_guard
from sim.sap_runtime import (
    SapTargetRemap,
    sap_contacts_from_newton,
    sap_control_from_newton,
    sap_model_from_newton,
    sap_state_from_newton,
)
from sim.solver_sap import SolverSAP

import newton

from ..adaptive_boundary import (
    mark_unfinished_contained,
    mark_unfinished_contained_target,
    mark_unfinished_with_status,
    mark_unfinished_with_status_target,
)

# ---- step-evolution mode codes (passed to _adapt_dt as a uniform kernel arg) ----
_MODE_FIXED = wp.constant(0)
_MODE_CODES = {"fixed": 0, "adaptive": 1}

# ---- Drake CalcAdjustedStepSize constants (err_order=2 for step doubling) ----
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

# ---- fp32 solve-precision optimality target ----
# The inner solve's convergence norms are evaluated in the solve dtype, so
# gradient cancellation (two O(norm)-sized impulse terms subtracting to ~0 at
# convergence) floors the achievable relative residual near that dtype's
# epsilon. fp64 reaches the pinned 1e-8 target. In fp32 that target sits BELOW
# the floor: every solve would cap out unconverged, contained rejection would
# shrink dt to the floor, and every contacting world would latch diverged. The
# fp32-selected configuration therefore couples its target to
# max(1e-8, _FP32_OPTIMALITY_K * eps_fp32). K bounds the measured stagnation
# floor of the fp32 residual evaluation with margin for population tails the
# probe scene cannot sample; it is re-derived by
# tools/probes/sap_fp32_floor_probe.py and must never be lowered below what
# that probe reports. The fp64 path never reads this constant.
_FP32_OPTIMALITY_K = 16.0


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
rather than the raw double literal -- see :meth:`SolverSAPAdaptive.dt_histogram_stats`.
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

    Launched at the TOP of the substep body, before ``_clamp_dt_to_boundary``: at that
    point ``dt`` still holds the controller's chosen step, not a landing sliver -- the
    ``_adapt_dt`` accept paths restore ``dt`` from the carried step after a
    boundary-limited accept precisely so this holds. Worlds already at their boundary
    are skipped -- they take no further step this interval.

    Bin 0 counts exact floor hits; the ``wp.clamp`` that writes ``dt`` (in ``_seed_dt``
    and ``_adapt_dt``) produces bitwise ``dt_min`` on clamp, so the equality test is
    reliable. ``saturation`` accumulates ``min(ideal_dt)`` over floor-clamped worlds,
    showing how far below the floor the controller wanted to go.

    Precondition: ``dt`` must be finite here. ``NaN`` and ``+inf`` both fail the ``h <=
    dt_min`` test (NaN comparisons are always false; ``+inf > dt_min``), so they fall
    through to the ``b = ...`` branch below -- but casting an ``inf``/``NaN``-derived
    ``wp.log10`` result to ``int`` does not saturate to a large positive index the way the
    overflow bin needs; it lands ``b`` near or below the valid range, which ``wp.clamp``
    then floors up to bin 1 (near-floor) instead of the last (overflow) bin -- the
    INVERTING direction. This is safe because every writer of ``dt`` produces a finite
    value: ``_seed_dt`` and ``_adapt_dt`` write through ``wp.clamp`` into
    ``[dt_min, cap]`` (which sanitizes NaN -> ``dt_min`` and ``+inf`` -> the cap), while
    ``_clamp_dt_to_boundary`` writes boundary remainders of the finite clocks (or 0)
    and ``_reset_worlds`` writes the finite ``dt_init``; so a non-finite ``dt`` never
    actually reaches this kernel. No runtime
    finiteness branch is added here to avoid the per-iteration cost of guarding an
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


# ============================================================================
# Inlined controller kernels (all per-world / per-element; no cross-world reduce)
# ============================================================================
@wp.kernel
def _open_frame(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt_outer: float,
):
    """Rebase the per-world clocks (Fix B) and set the new boundary to ``dt_outer``.

    ``sim_time`` and ``next_time`` are never zeroed across a run, so the landing
    remainder ``next_time - sim_time`` would lose float32 precision as the magnitude
    grows. Subtract each world's previous boundary ``next_time[i]`` (not zero): the
    residual overshoot is preserved bit-exactly in ``sim_time`` and carried forward,
    while ``next_time`` resets to ``dt_outer``.
    """
    i = wp.tid()
    base = next_time[i]
    sim_time[i] = sim_time[i] - base
    next_time[i] = dt_outer


@wp.kernel
def _seed_dt(
    mode: int,
    ideal_dt: wp.array[wp.float32],
    dt_fixed: float,
    dt_min: float,
    dt_max: float,
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
):
    """Seed this frame's per-world working dt.

    ``fixed`` mode pins dt to the (clamped) constant ``dt_fixed``; ``adaptive`` seeds from
    the carried controller estimate ``ideal_dt`` (which holds the Drake step sized from the
    last accepted error). ``ideal_dt`` is preserved unclamped
    so a world parked at ``dt_max`` can still recover a large step next frame.
    """
    i = wp.tid()
    if mode == _MODE_FIXED:
        d = wp.clamp(dt_fixed, dt_min, dt_max)
    else:
        d = wp.clamp(ideal_dt[i], dt_min, dt_max)
    dt[i] = d
    dt_half[i] = d * wp.float32(0.5)


@wp.kernel
def _clamp_dt_to_boundary(
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    limited: wp.array[wp.int32],
):
    """Clamp dt so no world oversteps its boundary; worlds at/past it get dt=0 (no-op).

    ``limited[i]=1`` marks a step artificially shortened by the boundary (a "landing
    sliver"), so the controller can exempt it from step-size resizing (Drake's
    artificially-limited rule, always on).
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
def _inf_norm_state_error_kernel(
    joint_q_full: wp.array[wp.float32],
    joint_q_double: wp.array[wp.float32],
    state_scale: wp.array2d[wp.float32],
    coords_per_world: int,
    joint_qd_commit: wp.array[wp.float32],
    dofs_per_world: int,
    rtol_over_atol: float,
    error_out: wp.array[wp.float32],
):
    """Per-world step-doubling accuracy metric (Kurtz & Castro, Sec. V-E)::

        e = || S (q_double - q_full) ||_inf

    Position-only inf-norm of the doubled-half-step vs. full-step ``q``, scaled by the
    diagonal ``S`` (here identity). NaN is flagged PER COMPONENT: ``wp.max`` is fmaxf
    on CUDA, which returns the non-NaN operand, so a NaN difference would otherwise be
    silently dropped from the running max and a non-finite world would report error 0
    and be committed. NaN/inf collapse to a large sentinel so the controller treats them as
    divergence -- including in ``fixed`` mode, where ``joint_q_full == joint_q_double``
    gives ``e == 0`` for finite states (always accept) and the sentinel for NaN ones.

    With ``rtol_over_atol > 0``, each coordinate's difference is normalized by
    ``1 + (rtol/atol)*|q|`` so the fixed-tol accept test is equivalent to
    ``|d| <= atol + rtol*|q|`` -- the budget scales with coordinate magnitude exactly
    as the float grid does, so a large coordinate's representation noise can never pin
    the estimate near tol and freeze the controller's deadband. Zero disables the
    branch (bit-identical legacy path).

    The error NORM stays position-only, but the committed state includes ``joint_qd``,
    so finiteness must be checked there too: a non-finite velocity with still-finite
    positions would otherwise commit (position error looks small) and poison
    subsequent steps via forces ~ v. ``joint_qd_commit`` is the commit candidate's
    velocity.
    """
    world = wp.tid()
    q_start = world * coords_per_world

    max_err = float(0.0)
    has_nan = int(0)
    for i in range(coords_per_world):
        d = wp.abs(joint_q_double[q_start + i] - joint_q_full[q_start + i])
        if wp.isnan(d):
            has_nan = 1
        else:
            if rtol_over_atol > 0.0:
                m = wp.max(wp.abs(joint_q_double[q_start + i]), wp.abs(joint_q_full[q_start + i]))
                d = d / (wp.float32(1.0) + rtol_over_atol * m)
            max_err = wp.max(max_err, state_scale[world, i] * d)

    for i in range(dofs_per_world):
        v = joint_qd_commit[world * dofs_per_world + i]
        if wp.isnan(v) or wp.isinf(v):
            has_nan = 1

    if has_nan != 0 or wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


@wp.kernel
def _inf_norm_state_error_indexed_kernel(
    joint_q_full: wp.array[wp.float32],
    joint_q_double: wp.array[wp.float32],
    state_scale: wp.array2d[wp.float32],
    coords_per_world: int,
    joint_qd_commit: wp.array[wp.float32],
    dofs_per_world: int,
    idx: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slot: int,
    rtol_over_atol: float,
    error_out: wp.array[wp.float32],
):
    """:func:`_inf_norm_state_error_kernel` over a compacted world list: only
    worlds in the index list get a fresh error; landed worlds keep their last
    written value. Safe because a landed world's error never reaches a
    decision: ``_adapt_dt``'s DONE branch returns before reading ``err``.
    Fixed launch dim with a device-side count guard so the launch records
    cleanly under graph capture; the loop body must stay identical to the
    full-dim kernel's (same fp operation order per world)."""
    i = wp.tid()
    if i >= counts[slot]:
        return
    world = idx[i]
    q_start = world * coords_per_world

    max_err = float(0.0)
    has_nan = int(0)
    for k in range(coords_per_world):
        d = wp.abs(joint_q_double[q_start + k] - joint_q_full[q_start + k])
        # NaN must be flagged per component: wp.max is fmaxf on CUDA, which
        # RETURNS THE NON-NAN OPERAND (see the full-dim kernel).
        if wp.isnan(d):
            has_nan = 1
        else:
            if rtol_over_atol > 0.0:
                m = wp.max(wp.abs(joint_q_double[q_start + k]), wp.abs(joint_q_full[q_start + k]))
                d = d / (wp.float32(1.0) + rtol_over_atol * m)
            max_err = wp.max(max_err, state_scale[world, k] * d)

    for k in range(dofs_per_world):
        v = joint_qd_commit[world * dofs_per_world + k]
        if wp.isnan(v) or wp.isinf(v):
            has_nan = 1

    if has_nan != 0 or wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


@wp.kernel
def _reset_active_counts(counts: wp.array[wp.int32]):
    """Zero the compaction counter (dim=1): slot 0 = active set. Must run
    before :func:`_build_active_worlds` each iteration (stream order
    guarantees it)."""
    counts[0] = 0


@wp.kernel
def _build_active_worlds(
    dt: wp.array[wp.float32],
    counts: wp.array[wp.int32],
    active_idx: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
):
    """Compact the unfinished worlds into an index list plus a per-world gate
    mask. Runs AFTER the boundary clamp so post-clamp ``dt > 0`` is the single
    source of truth for "this world attempts a step" -- the same predicate the
    controller's DONE branch keys on. Unlike the MuJoCo sibling there is no
    snapshot set: SAP's rollback is "hold ``state_cur``" (the evals write
    out-of-place scratch and the commit is accept-gated), so a landing world
    has no saved rows to freeze. List order comes from atomics and is
    nondeterministic; every consumer writes world-private rows only, so
    ordering cannot perturb any floating-point result."""
    i = wp.tid()
    if dt[i] > wp.float32(0.0):
        active_idx[wp.atomic_add(counts, 0, 1)] = i
        world_active[i] = 1
    else:
        world_active[i] = 0


# Canonical (ascending) active-list build, used in place of the atomic
# _build_active_worlds when march compaction runs deterministic: the narrow
# tail body makes the list an ITERATION SPACE, so its order is pinned to a
# pure function of the dt vector (chunked count -> serial chunk scan ->
# in-order scatter; every launch has fixed dims and no atomics). Results are
# bitwise-invariant to list order either way (consumers write world-private
# rows), so the two builds are interchangeable for physics; the ordered one
# additionally makes the list itself run-to-run stable.
_MC_CHUNK = 64


@wp.kernel
def _build_active_worlds_chunk_count(
    dt: wp.array[wp.float32],
    n: int,
    chunk: int,
    chunk_counts: wp.array[wp.int32],
):
    b = wp.tid()
    lo = b * chunk
    hi = wp.min(lo + chunk, n)
    c = int(0)
    for i in range(lo, hi):
        if dt[i] > wp.float32(0.0):
            c = c + 1
    chunk_counts[b] = c


@wp.kernel
def _build_active_worlds_chunk_scan(
    chunk_counts: wp.array[wp.int32],
    n_chunks: int,
    chunk_offsets: wp.array[wp.int32],
    counts: wp.array[wp.int32],
):
    """Serial exclusive scan over the (small) per-chunk counts (dim=1)."""
    acc = int(0)
    for b in range(n_chunks):
        chunk_offsets[b] = acc
        acc = acc + chunk_counts[b]
    counts[0] = acc


@wp.kernel
def _build_active_worlds_ordered_scatter(
    dt: wp.array[wp.float32],
    n: int,
    chunk: int,
    chunk_offsets: wp.array[wp.int32],
    active_idx: wp.array[wp.int32],
    world_active: wp.array[wp.int32],
):
    b = wp.tid()
    lo = b * chunk
    hi = wp.min(lo + chunk, n)
    c = chunk_offsets[b]
    for i in range(lo, hi):
        if dt[i] > wp.float32(0.0):
            active_idx[c] = i
            world_active[i] = 1
            c = c + 1
        else:
            world_active[i] = 0


@wp.kernel
def _derive_narrow_cond(
    counts: wp.array[wp.int32],
    cap: int,
    cond: wp.array[wp.int32],
):
    """Select the march-compact branch (dim=1): narrow iff the active-world
    count fits the narrow grid budget. The narrow body's launch dims are
    sized to ``cap``, so this predicate is the ONLY thing that may route an
    iteration into it; the in-branch capacity guard re-checks the same
    invariant so an edit that skews cond vs dims cannot pass silently."""
    if counts[0] <= cap:
        cond[0] = 1
    else:
        cond[0] = 0


@wp.kernel
def _average_velocity_guess_f64(
    a: wp.array[wp.float64],
    b: wp.array[wp.float64],
    out: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = wp.float64(0.5) * (a[i] + b[i])


@wp.kernel
def _guess_stage_f64_to_f32(
    src: wp.array[wp.float64],
    dst: wp.array[wp.float32],
):
    i = wp.tid()
    dst[i] = wp.float32(src[i])


@wp.kernel
def _average_velocity_guess_f32(
    a: wp.array[wp.float32],
    b: wp.array[wp.float32],
    out: wp.array[wp.float32],
):
    i = wp.tid()
    out[i] = wp.float32(0.5) * (a[i] + b[i])


@wp.kernel
def _set_scalar_i32(value: wp.array[int], new_value: int):
    value[0] = new_value


@wp.kernel
def _reset_solve_convergence(ok: wp.array[int]):
    i = wp.tid()
    ok[i] = 1


@wp.kernel
def _accumulate_solve_convergence(
    converged_env: wp.array[int],
    ok: wp.array[int],
):
    i = wp.tid()
    if converged_env[i] == 0:
        ok[i] = 0


@wp.kernel
def _apply_solve_convergence_to_error(
    ok: wp.array[int],
    err: wp.array[wp.float32],
    divergence_threshold: float,
):
    i = wp.tid()
    if ok[i] == 0:
        err[i] = divergence_threshold


@wp.kernel
def _adapt_dt(
    err: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    ideal_dt: wp.array[wp.float32],
    diverged: wp.array[wp.bool],
    accept: wp.array[wp.bool],
    accepted_error: wp.array[wp.float32],
    substeps_frame: wp.array[wp.int32],
    cum_accepted: wp.array[wp.int32],
    mode: int,
    tol: float,
    dt_min: float,
    dt_max: float,
    divergence_threshold: float,
    dt_ceiling: wp.array[wp.float32],
    limited: wp.array[wp.int32],
    consec_rej: wp.array[wp.int32],
    ceiling_cap: float,
    dt_fixed: float,
):
    """The per-world step-doubling controller -- the whole accept/reject/done decision.

    Writes ``accept[w]`` (gates the state commit), advances ``sim_time`` on accept, and
    evolves ``dt``/``ideal_dt`` per ``mode``. Every branch is per-thread data flow (no
    device control flow), which is what makes the enclosing substep loop a flat graph.

    ``dt_max`` is the effective per-boundary cap (``min(ctor dt_max, dt_outer)``) that
    clamps the working ``dt``; ``ceiling_cap`` is the ctor ``dt_max`` (sanitized finite)
    that bounds the ceiling relax -- the ceiling is cross-boundary memory and must not
    be pulled down to the current boundary's cap.
    """
    w = wp.tid()
    step = dt[w]

    # DONE: world reached its boundary (or clamp zeroed its step) -> commit nothing.
    # Out-of-place scratch keeps a stalled world's state untouched by construction
    # (state_cur is read-only through the evals), so there is nothing to restore and
    # nothing to accept: committing here would copy dt=0 scratch garbage over good
    # state. Returning before the ideal_dt writes also carries the good ideal_dt
    # across the boundary unchanged.
    if sim_time[w] >= next_time[w] or step <= wp.float32(0.0):
        accept[w] = False
        return

    e = err[w]
    is_div = wp.isnan(e) or wp.isinf(e) or e >= divergence_threshold

    # ---------- FIXED: constant dt, error control off (NaN guard only) ----------
    if mode == _MODE_FIXED:
        if is_div:
            # Refuse the non-finite step; finish the frame holding the last good state.
            accept[w] = False
            sim_time[w] = next_time[w]
            diverged[w] = True
            return
        accept[w] = True
        sim_time[w] = sim_time[w] + step
        accepted_error[w] = e
        substeps_frame[w] = substeps_frame[w] + 1
        wp.atomic_add(cum_accepted, 0, 1)
        # A boundary-limited (sliver) accept leaves the clamped remainder in dt;
        # restore the constant step so a rounding-residue iteration attempts (and
        # the histogram bins) the configured dt, not the sliver. The next clamp
        # re-derives dt=0 (DONE) or the true remainder either way.
        if limited[w] == 1:
            d = wp.clamp(dt_fixed, dt_min, dt_max)
            dt[w] = d
            dt_half[w] = d * wp.float32(0.5)
        return

    # ---------- ADAPTIVE: within-frame grow on accept / shrink+retry on reject ----------
    # At the floor we cannot subdivide further.
    if step <= dt_min * wp.float32(1.001):
        if is_div:
            accept[w] = False
            sim_time[w] = next_time[w]
            diverged[w] = True
            ideal_dt[w] = dt_min
            consec_rej[w] = 0
            return
        # Accept progress (cannot subdivide further). CRUCIAL: still size ideal_dt by the
        # Drake rule so a world RECOVERS once its step is good again -- e <= tol grows ideal_dt
        # (lifts it off the floor next frame); e > tol leaves it ~floor. Pinning ideal_dt =
        # dt_min here is a TRAP: a world driven to the floor by any transient stays pinned
        # there forever even after the difficulty passes, which is the per-world dt collapse
        # seen on the steady shadow-hand task.
        accept[w] = True
        sim_time[w] = sim_time[w] + step
        accepted_error[w] = e
        substeps_frame[w] = substeps_frame[w] + 1
        wp.atomic_add(cum_accepted, 0, 1)
        consec_rej[w] = 0
        new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
        if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
            new_step = step
        ideal_dt[w] = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)
        # A boundary-limited accept (remainder at/below the floor) leaves the sliver
        # in dt; restore from the resized ideal_dt so a rounding-residue iteration
        # attempts (and the histogram bins) the controller's step, not the sliver.
        if limited[w] == 1:
            d = wp.clamp(ideal_dt[w], dt_min, dt_max)
            dt[w] = d
            dt_half[w] = d * wp.float32(0.5)
        return

    # Above the floor and diverged: reject, shrink hard, hold state, retry.
    if is_div:
        accept[w] = False
        dt_ceiling[w] = wp.min(dt_ceiling[w], _CEILING_MARGIN * step)
        consec_rej[w] = consec_rej[w] + 1
        new_step = _DRAKE_MIN_SHRINK * step
        ideal_dt[w] = new_step
        d = wp.clamp(new_step, dt_min, dt_max)
        dt[w] = d
        dt_half[w] = d * wp.float32(0.5)
        return

    new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
    # Symmetric deadband (paper Alg 1): hold dt when new_step lands in [k_low, k_high]*dt
    # to suppress thrash from small error spikes and tiny grows.
    if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
        new_step = step
    new_step = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)

    # Accept when within tol, or when the controller still wants to grow (avoids
    # rejecting a marginally-over-tol step the controller would enlarge anyway).
    acc = e <= tol or new_step >= step
    if acc:
        accept[w] = True
        sim_time[w] = sim_time[w] + step
        accepted_error[w] = e
        substeps_frame[w] = substeps_frame[w] + 1
        wp.atomic_add(cum_accepted, 0, 1)
        consec_rej[w] = 0
        # Sliver exemption: an accepted step that was artificially shortened by the
        # boundary clamp says nothing about the error-limited step size, so keep the
        # carried ideal_dt instead of collapsing it to ~the sliver (Drake's
        # artificially-limited rule). It also carries no ceiling information, so the
        # relax is skipped too. dt is restored from the carried ideal_dt so a
        # rounding-residue iteration attempts (and the histogram bins) the
        # controller's step, not the sliver; the next iteration's clamp re-derives
        # dt=0 (DONE) or the true remainder either way.
        if limited[w] == 1:
            d = wp.clamp(ideal_dt[w], dt_min, dt_max)
            dt[w] = d
            dt_half[w] = d * wp.float32(0.5)
            return
        dt_ceiling[w] = wp.min(dt_ceiling[w] * _CEILING_RELAX, ceiling_cap)
    else:
        accept[w] = False
        dt_ceiling[w] = wp.min(dt_ceiling[w], _CEILING_MARGIN * step)
        consec_rej[w] = consec_rej[w] + 1
    new_step = wp.min(new_step, dt_ceiling[w])
    ideal_dt[w] = new_step
    d = wp.clamp(new_step, dt_min, dt_max)
    dt[w] = d
    dt_half[w] = d * wp.float32(0.5)


@wp.kernel
def _commit_float(
    src: wp.array[wp.float32],
    accept: wp.array[wp.bool],
    stride: int,
    state: wp.array[wp.float32],
):
    """Commit the stepped result into the working state for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


@wp.kernel
def _commit_transform(
    src: wp.array[wp.transform],
    accept: wp.array[wp.bool],
    stride: int,
    state: wp.array[wp.transform],
):
    """Commit body poses for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


@wp.kernel
def _commit_spatial_vector(
    src: wp.array[wp.spatial_vector],
    accept: wp.array[wp.bool],
    stride: int,
    state: wp.array[wp.spatial_vector],
):
    """Commit body velocities for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


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
    """Restore the per-world controller state to construction defaults for masked worlds.

    Called on env/episode reset so pre-reset controller state (dt / clocks / latches)
    does not leak into post-reset dynamics. ``sim_time`` and ``next_time`` reset together
    to 0 so the world restarts a clean boundary interval. ``consec_rej`` is not reset
    here: it is inert outside the debt guard (which zeroes it itself), and no branch
    reads its value.
    """
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
def _boundary_reset(flag: wp.array[wp.int32]):
    """Set flag[0] = 0 (assume all worlds reached the boundary)."""
    flag[0] = 0


@wp.kernel
def _iter_count_increment(count: wp.array[wp.int32]):
    """Increment iteration counter (dim=1, single thread)."""
    count[0] = count[0] + 1


@wp.kernel
def _iters_exhausted_stop(iter_count: wp.array[wp.int32], max_iters: int, flag: wp.array[wp.int32]):
    """Latch the boundary loop closed once ``max_substeps`` attempts have run.

    Runs AFTER the boundary-exit check so the cap wins: the safety bound must stop the loop
    regardless of how many worlds are still short. A solve-failure status
    (``flag[0] >= 2``) is preserved -- exhausting the attempt budget on the final
    permitted iteration must never mask the converge-or-throw contract.
    """
    if iter_count[0] >= max_iters and flag[0] < 2:
        flag[0] = 0


@wp.kernel
def _march_continue_set(cont: wp.array[wp.int32]):
    """Open the device-side march loop (dim=1).

    The conditional while-node tests its condition BEFORE the first trip, and
    the march's contract is that the substep body runs at least once per
    boundary, so the condition must enter the loop open.
    """
    cont[0] = 1


@wp.kernel
def _march_continue_from_status(status: wp.array[wp.int32], cont: wp.array[wp.int32]):
    """Derive the march loop condition from the boundary status word (dim=1).

    Only status 1 (unfinished, budget remaining) continues the loop; 0 (done,
    or budget exhausted) and 2 (inner solve failed) both stop it. The loop
    condition is a SEPARATE word from the status because the failure status
    must survive for the caller's post-march converge-or-throw check -- and a
    nonzero status used directly as the loop condition would spin a failed
    solve forever on device.
    """
    if status[0] == 1:
        cont[0] = 1
    else:
        cont[0] = 0


@wp.kernel
def _march_continue_quantile(
    status: wp.array[wp.int32],
    active: wp.array[wp.int32],
    cutoff: int,
    cont: wp.array[wp.int32],
    gate: wp.array[wp.int32],
):
    """Loop condition with a straggler cutoff (dim=1).

    Identical to :func:`_march_continue_from_status` except that the loop also
    stops once the number of worlds still marching has fallen to ``cutoff`` or
    below. The boundary loop otherwise runs until the LAST world lands, so a
    handful of worlds whose local error forces a very small dt keep the whole
    batch iterating while the rest sit finished -- the batch pays full launch
    width for a vanishing active set.

    Worlds still unfinished when the cutoff fires have NOT reached the boundary
    and their state must not be consumed; :attr:`boundary_cut_mask` records them
    so the caller can drop those environments rather than integrate them
    inaccurately.
    """
    if status[0] != 1:
        cont[0] = 0
        gate[0] = 0
    elif active[0] <= cutoff:
        cont[0] = 0
        gate[0] = 1
    else:
        cont[0] = 1
        gate[0] = 0


@wp.kernel
def _mark_cut_worlds(
    gate: wp.array[wp.int32],
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    cut: wp.array[wp.int32],
):
    """Flag worlds left short of the boundary when the quantile stop fired.

    Runs inside the march, on the trip that closes the loop: after the loop the
    remaining worlds are landed on the boundary, and ``sim_time < next_time`` no
    longer distinguishes them.
    """
    w = wp.tid()
    if gate[0] != 0 and sim_time[w] < next_time[w]:
        cut[w] = 1


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
def _debt_guard_target(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    t_call: float,
    dt_outer: float,
    dt_init: float,
    ceiling_init: float,
    ideal_dt: wp.array[wp.float32],
    dt_ceiling: wp.array[wp.float32],
    consec_rej: wp.array[wp.int32],
    guard_hits: wp.array[wp.int32],
):
    """:func:`_debt_guard` for the run-ahead march: "in debt" is measured
    against the CALL target ``t_call``, not the per-world boundary clock -- a
    run-ahead world legitimately sits mid-boundary (``sim_time < next_time``)
    at call exit and must not have its controller reissued. Debt requires
    BOTH clocks short: a throttle-held world parked AT its boundary
    (``sim_time == next_time``) completed its boundary and carries no debt
    -- the per-boundary guard would not touch it, so neither does this one.
    For worlds genuinely mid-boundary the carry bound and controller reissue
    below are the original kernel's, verbatim."""
    i = wp.tid()
    if sim_time[i] < t_call and sim_time[i] < next_time[i]:
        floor_t = next_time[i] - dt_outer
        if sim_time[i] < floor_t:
            sim_time[i] = floor_t
        ideal_dt[i] = dt_init
        dt_ceiling[i] = ceiling_init
        consec_rej[i] = 0
        wp.atomic_add(guard_hits, 0, 1)


@wp.kernel
def _ra_advance_boundary(
    mode: int,
    dt_outer: float,
    window_end: float,
    dt_fixed: float,
    dt_min: float,
    dt_max: float,
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    ideal_dt: wp.array[wp.float32],
    diverged: wp.array[wp.bool],
    substeps_frame: wp.array[wp.int32],
    crossed: wp.array[wp.int32],
    crossed_any: wp.array[wp.int32],
    crossings: wp.array[wp.int32],
    fire: wp.array[wp.int32],
):
    """The run-ahead crossing: a world that reached its boundary target inside
    the action window does not park -- its boundary bookkeeping is applied
    in-place and it marches on toward the next boundary, capped at the window
    end. Runs after the commit; per-world data flow only.

    ``fire`` is the throttle gate written by :func:`_ra_throttle_decide`:
    while it is closed an eligible world HOLDS at its boundary (state, clock
    and controller carry untouched -- the parked-world state the per-boundary
    march produces between calls), so the gate batches crossings without
    skipping or reordering any world's adoption. The diverged latch-park is
    NOT gated: it is a containment freeze, not a crossing, and must stay
    visible the same call it latched.

    The bookkeeping mirrors what integrate() applies between boundary calls:
    the :func:`_seed_dt` clamp of the carried ``ideal_dt`` into ``dt`` /
    ``dt_half`` (a landing accept already left dt at exactly this value, so
    the reseed is idempotent -- kept for exactness against the per-boundary
    path) and the per-world ``substeps_frame`` rollover. ``next_time``
    advances by one ``dt_outer`` add, the same fp operation the host-side
    target chain uses, so boundary targets stay bit-comparable everywhere.

    A ``diverged`` world does NOT run ahead: its floor-latch froze its state
    and clock at its boundary, and this kernel parks it at the window end
    (clock force-advanced, latch left visible for the caller's post-call
    read) so it stays isolated for the rest of the window -- the run-ahead
    analogue of the per-boundary latch's freeze-to-boundary.

    ``crossed``/``crossed_any`` arm the crossing-batched collide+adopt node
    at the top of the next march iteration (the world's contact set refreshes
    at ITS boundary-entry state, preserving per-world contact cadence);
    ``crossings`` is the cumulative engagement counter probes read.
    """
    w = wp.tid()
    if sim_time[w] < next_time[w]:
        return
    if next_time[w] >= window_end:
        return
    if diverged[w]:
        sim_time[w] = window_end
        next_time[w] = window_end
        return
    if fire[0] == 0:
        return
    nt = next_time[w] + dt_outer
    if nt > window_end:
        nt = window_end
    next_time[w] = nt
    if mode == _MODE_FIXED:
        d = wp.clamp(dt_fixed, dt_min, dt_max)
    else:
        d = wp.clamp(ideal_dt[w], dt_min, dt_max)
    dt[w] = d
    dt_half[w] = d * wp.float32(0.5)
    substeps_frame[w] = 0
    crossed[w] = 1
    wp.atomic_max(crossed_any, 0, 1)
    wp.atomic_add(crossings, 0, 1)


@wp.kernel
def _ra_resync_reset_worlds(
    t_prev: float,
    t_call: float,
    mode: int,
    dt_fixed: float,
    dt_min: float,
    dt_max: float,
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
    ideal_dt: wp.array[wp.float32],
    substeps_frame: wp.array[wp.int32],
    crossed: wp.array[wp.int32],
    crossed_any: wp.array[wp.int32],
):
    """Mid-window re-entry for reset worlds (run-ahead mode only).

    ``reset()`` zeroes a world's clocks; between windows the normal window-open
    rebase absorbs that, but a mid-window reset (the manager resets diverged
    worlds after every boundary call) would otherwise leave the world at local
    time 0 and re-march the whole window. ``next_time == 0`` is an exact
    reset signature (every live target is a positive dt_outer chain value), so
    this kernel re-seats exactly those worlds at the current call's start:
    one ``dt_outer`` of marching this call, run-ahead from there. The seed
    clamp mirrors :func:`_seed_dt` (reset wrote a raw ``dt_init``). The world
    is flagged ``crossed`` so the march's first collide+adopt node refreshes
    its contact set at its post-reset state."""
    w = wp.tid()
    if next_time[w] != wp.float32(0.0):
        return
    sim_time[w] = t_prev
    next_time[w] = t_call
    if mode == _MODE_FIXED:
        d = wp.clamp(dt_fixed, dt_min, dt_max)
    else:
        d = wp.clamp(ideal_dt[w], dt_min, dt_max)
    dt[w] = d
    dt_half[w] = d * wp.float32(0.5)
    substeps_frame[w] = 0
    crossed[w] = 1
    wp.atomic_max(crossed_any, 0, 1)


@wp.kernel
def _ra_clear_crossed(
    crossed: wp.array[wp.int32],
    crossed_any: wp.array[wp.int32],
    adopts: wp.array[wp.int32],
):
    """Disarm the crossing-batch flags after a collide+adopt node consumed
    them (stream-ordered after the masked collide and the adopt, which read
    ``crossed`` as their mask). ``adopts`` counts consumed batches -- the
    device engagement counter for the conditional node itself (replays
    included)."""
    i = wp.tid()
    crossed[i] = 0
    if i == 0:
        crossed_any[0] = 0
        adopts[0] = adopts[0] + 1


@wp.kernel
def _ra_throttle_count(
    window_end: float,
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    diverged: wp.array[wp.bool],
    counts: wp.array[wp.int32],
):
    """Classify every world for this iteration's crossing-gate decision.

    ``counts[0]`` (pending): parked at a boundary target short of the window
    end, not diverged -- eligible to cross. ``counts[1]`` (marching): still
    integrating toward its current target. Worlds at the window end and
    diverged latch-parks fall in neither class. Runs after the commit, so
    the classification reflects this iteration's accepted steps; the decide
    kernel consumes and resets the counts.
    """
    w = wp.tid()
    if sim_time[w] < next_time[w]:
        wp.atomic_add(counts, 1, 1)
        return
    if next_time[w] < window_end and not diverged[w]:
        wp.atomic_add(counts, 0, 1)


@wp.kernel
def _ra_throttle_decide(
    bound: int,
    age: int,
    counts: wp.array[wp.int32],
    wait: wp.array[wp.int32],
    fire: wp.array[wp.int32],
):
    """Open the crossing gate (dim=1) when any of three rules trips:
    (1) COUNT -- the pending batch reached the bound (wide-regime batching:
    fire exactly when a large batch is ready); (2) AGE -- some world has
    been held ``age`` iterations (``wait`` counts consecutive iterations
    with a non-empty pending set, so the FIRST lander bounds every holder's
    delay): without it a sub-bound trailing set would wait on rule (3) for
    the deepest straggler to land, a global barrier that serializes half
    the fleet behind one world; (3) LIVENESS -- no world can advance
    without crossing (once every unheld world is parked at the window end,
    a diverged latch, or pending, the batch must fire or the march would
    spin). Consumes and resets the counts so the next iteration classifies
    afresh; ``wait`` persists across iterations and resets on fire or on an
    empty pending set."""
    pending = counts[0]
    marching = counts[1]
    w = wait[0]
    if pending > 0:
        w = w + 1
    else:
        w = 0
    f = 0
    if pending >= bound or (pending > 0 and (marching == 0 or w >= age)):
        f = 1
        w = 0
    fire[0] = f
    wait[0] = w
    counts[0] = 0
    counts[1] = 0


@wp.kernel
def _count_rejects(
    accepted: wp.array[wp.bool],
    dt: wp.array[wp.float32],
    diverged: wp.array[wp.bool],
    rejects: wp.array[wp.int32],
):
    """Accumulate this iteration's rejected attempts (diagnostic only).

    A world at its boundary makes no attempt (its dt was clamped to 0 and the
    controller's DONE branch leaves ``accepted`` False), so a reject is
    not-accepted AND attempting (post-controller ``dt > 0``). A divergence
    LATCH (the floor / fixed-mode / force-accept non-finite paths, which
    freeze the world to its boundary) is a refusal that will NOT be retried,
    not a controller rejection, so latched worlds are excluded -- the same
    semantics as the MuJoCo counter, whose floor path never lowers
    ``accepted``. The above-floor diverged path (shrink and retry) does not
    latch and still counts.
    """
    i = wp.tid()
    if (not accepted[i]) and (not diverged[i]) and dt[i] > wp.float32(0.0):
        wp.atomic_add(rejects, 0, 1)


@wp.kernel
def _count_boundary_truncation(
    iter_count: wp.array[wp.int32],
    max_iters: int,
    out: wp.array[wp.int64],
):
    """Record one boundary, and whether it used the full ``max_substeps`` budget (dim=1).

    ``out[0]`` counts boundaries. ``out[1]`` counts boundaries whose MARCH consumed the
    entire ``max_substeps`` budget (``iter_count`` is the caller's march-only
    snapshot) -- this INCLUDES the case where
    every world happened to land exactly on the final permitted iteration, so it is not
    on its own proof that any world is short of its target time. It is still a genuine saturation signal: the boundary had
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


class SolverSAPAdaptive:
    """Per-world adaptive (step-doubling) SAP integrator.

    Drop-in mirror of ``SolverMuJoCoAdaptive(model, ...)``: takes the Newton ``Model``,
    builds the ``SapModel`` + inner ``SolverSAP`` internally, and exposes the Newton
    solver surface (``step`` / ``step_dt`` / ``reset``) plus per-world telemetry
    (``dt`` / ``sim_time`` / ``diverged`` / ``substeps``).

    ``max_substeps`` defaults to 16 here (each attempt costs three SAP solves), unlike
    ``SolverMuJoCoAdaptive``'s 256; callers that need the MuJoCo-adaptive budget must
    pass their own value.
    """

    def __init__(
        self,
        model,
        *,
        mode: str = "adaptive",
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-12,
        dt_inner_max: float | None = None,
        max_substeps: int = 16,
        dt_histogram: bool = False,
        dt_histogram_bins_per_decade: int = 4,
        max_rigid_contact: int = 128,
        max_triangle_pairs: int = 1_000_000,
        max_iterations: int = 30,
        contact_preset_variant: str = "drake",
        line_search_variant: str = "armijo_decay",
        contact_tau_d: float = 0.01,
        solve_precision: str | None = None,
        **kwargs,
    ):
        if mode not in _MODE_CODES:
            raise ValueError(f"mode must be one of {tuple(_MODE_CODES)}, got {mode!r}.")
        if float(tol) <= 0.0:
            raise ValueError(f"tol must be > 0, got {tol!r}.")
        if float(dt_inner_init) <= 0.0:
            raise ValueError(f"dt_inner_init must be > 0, got {dt_inner_init!r}.")
        if float(dt_inner_min) <= 0.0:
            raise ValueError(f"dt_inner_min must be > 0, got {dt_inner_min!r}.")
        if dt_inner_max is not None and float(dt_inner_max) <= 0.0:
            raise ValueError(f"dt_inner_max must be > 0 when provided, got {dt_inner_max!r}.")
        # The controller invariant: the seed must sit strictly above the floor and
        # within the cap, else the floor/deadband branches degenerate.
        if not float(dt_inner_min) < float(dt_inner_init):
            raise ValueError(
                f"dt_inner_min must be < dt_inner_init, got dt_inner_min={dt_inner_min!r}, "
                f"dt_inner_init={dt_inner_init!r}."
            )
        if dt_inner_max is not None and not float(dt_inner_init) <= float(dt_inner_max):
            raise ValueError(
                f"dt_inner_init must be <= dt_inner_max when provided, got "
                f"dt_inner_init={dt_inner_init!r}, dt_inner_max={dt_inner_max!r}."
            )
        if int(max_substeps) < 1:
            raise ValueError(f"max_substeps must be >= 1, got {max_substeps!r}.")
        if int(dt_histogram_bins_per_decade) < 1:
            raise ValueError(f"dt_histogram_bins_per_decade must be >= 1, got {dt_histogram_bins_per_decade!r}.")
        self.model = model
        device = model.device
        wc = int(model.world_count)

        # ---- inner SAP solver + model ----
        # Solve-precision selection (opt-in; fp64 is the default). "fp64"
        # passes NO precision overrides, leaving every knob to the contact
        # preset (whose contact solve is fp64 in all shipped presets) -- the
        # exact pre-option construction. "fp32" overrides the full solve stack
        # -- free motion, contact weights, contact solve, contact linear solve
        # -- to float32 on top of the preset's MODE choices (weights, contact
        # points, boundary-pose handling, position integration), so precision
        # is the only thing that changes. Committed state, the error metric,
        # the controller, and the march are float32/float64 exactly as before
        # in both settings. Resolution: explicit argument, else
        # NEWTON_SAP_SOLVE_PRECISION, else fp64.
        if solve_precision is None:
            solve_precision = os.environ.get("NEWTON_SAP_SOLVE_PRECISION", "fp64")
        _sp = str(solve_precision).strip().lower()
        if _sp == "f32":
            _sp = "fp32"
        elif _sp == "f64":
            _sp = "fp64"
        if _sp not in ("fp32", "fp64"):
            raise ValueError(f"solve_precision must be 'fp32'/'f32' or 'fp64'/'f64', got {solve_precision!r}.")
        self._solve_precision = _sp

        # The convex SAP solve runs CONVERGENT (conditional, per-env early exit) to a fixed
        # optimality_rel_tol, independent of the integration tolerance: the solver residual
        # must sit far below tol at ANY tol setting, else it pollutes the step-doubling
        # error estimate and the controller subdivides spuriously. The target is COUPLED
        # to the solve dtype: fp64 keeps the pinned 1e-8; fp32 uses the tightest
        # achievable analogue (see _FP32_OPTIMALITY_K), because the convergence norms are
        # evaluated in the solve dtype and a target below the dtype's residual floor can
        # never be met by any iteration budget.
        if self._solve_precision == "fp32":
            self._optimality_rel_tol = max(1.0e-8, _FP32_OPTIMALITY_K * float(np.finfo(np.float32).eps))
        else:
            self._optimality_rel_tol = 1.0e-8
        self._sap_model = sap_model_from_newton(model)
        # Coord-layout newtons store control position targets per coordinate;
        # the remap gathers them into the dof layout SAP indexes. None when the
        # layouts already coincide (control arrays then pass through unchanged).
        self._target_remap = SapTargetRemap.from_newton_model(model)
        # fp64 passes an EMPTY override dict so the construction is argument-
        # for-argument identical to the pre-option solver (bitwise-clean
        # default path); fp32 overrides the four precision knobs only.
        _precision_overrides: dict[str, str] = {}
        if self._solve_precision == "fp32":
            _precision_overrides = {
                "free_motion_solve_precision": "fp32",
                "contact_solve_precision": "fp32",
                "contact_linear_solve_precision": "fp32",
                "sap_contact_weight_precision": "fp32",
            }
        self._sap = SolverSAP(
            self._sap_model,
            max_rigid_contact=int(max_rigid_contact),
            max_iterations=int(max_iterations),
            optimality_rel_tol=self._optimality_rel_tol,
            cost_abs_tol=0.0,
            cost_rel_tol=0.0,
            static_substep=False,
            contact_tau_d=float(contact_tau_d),
            contact_preset_variant=str(contact_preset_variant),
            line_search_variant=str(line_search_variant),
            **_precision_overrides,
        )

        # ---- scratch SapStates (independent backing arrays) ----
        # state_cur is read-only through all three evals (full/mid read it, double reads
        # mid), so it IS the natural rollback fallback -- a rejected world simply keeps it.
        self._scratch_full = self._sap_model.state()
        self._scratch_mid = self._sap_model.state()
        self._scratch_double = self._sap_model.state()
        self._state_cur = self._sap_model.state()

        # ---- physical warm-start buffers (Drake CENIC reference) ----
        # full: v_t; half-1: (v_t + v_full) / 2; half-2: v_full.
        self._vt = wp.clone(self._sap.contact_solve.v_flat)
        self._vhalf1 = wp.clone(self._sap.contact_solve.v_flat)
        self._vfull = wp.clone(self._sap.contact_solve.v_flat)
        if self._sap.contact_solve.v_flat.dtype == wp.float64:
            self._average_velocity_guess_kernel = _average_velocity_guess_f64
            self._guess_stage_f64 = None
        elif self._sap.contact_solve.v_flat.dtype == wp.float32:
            self._average_velocity_guess_kernel = _average_velocity_guess_f32
            # The public->SAP boundary conversion kernels write CANONICAL f64
            # SAP-order velocities; a float32 solve stack needs a staging
            # buffer between that contract and the f32 guess buffers.
            self._guess_stage_f64 = wp.zeros(int(model.joint_dof_count), dtype=wp.float64, device=device)
        else:
            raise TypeError(f"Unsupported SAP velocity dtype {self._sap.contact_solve.v_flat.dtype!r}.")
        self._solve_ok = wp.ones(wc, dtype=int, device=device)

        # Per-world solve-failure CONTAINMENT (default ON; NEWTON_SAP_CONTAINMENT=0
        # opts into the strict batch-fatal converge-or-throw, for probes/debugging).
        # Contained semantics: a failing world's error is already forced to the
        # divergence sentinel, so the controller rejects the attempt (never
        # consuming the unconverged result) and retries at a shrunken dt; a
        # failure that persists to the dt floor latches the world ``diverged``
        # exactly like a NaN state -- the world holds its last committed finite
        # state while its clock is forced to the boundary, and the latch
        # (.diverged, read after the boundary call) is the consumer's only
        # signal to reset the world and terminate its episode. The batch
        # keeps marching either way. The committed-step contract is unchanged:
        # every committed step still passed the full converge-to-tolerance
        # solve; failures only ever reject or latch.
        self._containment = os.environ.get("NEWTON_SAP_CONTAINMENT", "1") != "0"
        # Cumulative per-world failure-event counts (device; written by the
        # contained boundary kernel, read host-side only on the rare path when
        # the sticky per-boundary latch says a failure happened).
        self._solve_fail_world = wp.zeros(wc, dtype=wp.int32, device=device)
        # Host-side cumulative accounting + rate-limit state for the warning.
        self._solve_failure_events = 0
        self._solve_failure_boundaries = 0
        self._fail_world_prev = np.zeros(wc, dtype=np.int64)
        self._fail_warn_emitted = 0
        # Host count of boundaries integrated with containment active (probe
        # engagement tripwire; no device work).
        self._containment_boundaries = 0

        # Tail compaction (default ON; NEWTON_ADAPTIVE_TAIL_COMPACT=0 opts out):
        # late march iterations run full-batch kernels while only a few worlds
        # are still unfinished; a device-built index list + per-world gate mask
        # of active worlds restricts the per-world SAP pipeline work (free
        # motion, Jacobian/dynamics assembly, contact-solve entry, integration,
        # FK, the error metric) to those worlds while every launch keeps its
        # fixed dim (graph-capture safe). Landed worlds enter each solve
        # pre-converged, so the Newton loop's trip count is set by active
        # worlds only. A pure scheduling change -- the tail-compact arm of
        # sap_flag_equivalence_probe.py asserts bitwise identity with the
        # full-batch path and must keep passing for this default to remain ON.
        # Read at construction; the choice is baked into the captured body.
        # NOT covered: the two CollisionPipeline passes and the contact
        # scatter, which stay full-width (their global atomic slot assignment
        # must see an identical thread set, else active worlds' contact-slot
        # order -- and downstream fp summation order -- could change).
        self._tail_compact = os.environ.get("NEWTON_ADAPTIVE_TAIL_COMPACT", "1") != "0"
        if self._tail_compact:
            # counts[0] = active-set size (worlds attempting a step this
            # iteration, post-boundary-clamp dt > 0). Rebuilt on device each
            # iteration; consumers keep FIXED launch dims and early-exit on
            # the device-read count or the per-world gate mask.
            self._active_counts = wp.zeros(1, dtype=wp.int32, device=device)
            self._active_idx = wp.zeros(wc, dtype=wp.int32, device=device)
            self._world_active = wp.ones(wc, dtype=wp.int32, device=device)

        # ---- per-world controller buffers (the dt VECTOR is the primitive; no N) ----
        self._dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(wc, dt_inner_init * 0.5, dtype=wp.float32, device=device)

        # Attempt-consistent constitutive law (default ON;
        # NEWTON_SAP_ATTEMPT_CONSISTENT_R=0 disables): pin every trial solve's
        # near-rigid clamps (contact rn_hard, joint-limit r_nr, PD gain clamp)
        # to THIS attempt's dt instead of each solve's own dt, so the
        # step-doubling comparison measures truncation error of one fixed
        # constitutive law rather than the dt-coupled law difference between
        # the full step and its halves.  The full solve's transform is exactly
        # 1 (committed-step laws unchanged; attempts still tighten as the
        # attempted dt falls), and the twin precedent is the MuJoCo solver's
        # NEWTON_ADAPTIVE_CONTACT_COUPLING (coupled solref with
        # timeconst = 2*dt_attempt for both trials).
        self._attempt_consistent_r = os.environ.get("NEWTON_SAP_ATTEMPT_CONSISTENT_R") != "0"
        if self._attempt_consistent_r:
            self._sap.contact_solve.set_constitutive_dt(self._dt)
        self._ideal_dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._next_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(wc, dtype=wp.bool, device=device)
        self._diverged = wp.zeros(wc, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._substeps_frame = wp.zeros(wc, dtype=wp.int32, device=device)
        self._cum_accepted = wp.zeros(1, dtype=wp.int32, device=device)
        # Boundary status word. Slot 0: per-iteration boundary flag (0 done,
        # 1 unfinished, 2 solve failed -- strict mode only), reset in-body each
        # iteration and read back once per iteration / once per march (the
        # single accepted host sync) to break the loop early. Slot 1: sticky
        # per-boundary solve-failure latch (containment mode only), zeroed at
        # boundary open so the same post-march read reports failures that
        # recovered or latched before the final iteration.
        self._unfinished = wp.zeros(2, dtype=wp.int32, device=device)

        self._mode = str(mode)
        self._mode_code = _MODE_CODES[self._mode]
        self._tol = float(tol)
        self._dt_min = float(dt_inner_min)
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")
        self._dt_inner_init = float(dt_inner_init)
        self._max_substeps = int(max_substeps)
        self._divergence_threshold = float(1.0e9)
        self._full_world_mask = wp.full(wc, True, dtype=wp.bool, device=device)

        # Boundary-limited flag (landing slivers, see _clamp_dt_to_boundary) and
        # consecutive-rejection counter (debt-guard input; no branch reads it).
        self._limited = wp.zeros(wc, dtype=wp.int32, device=device)
        self._consec_rej = wp.zeros(wc, dtype=wp.int32, device=device)
        # Ceiling memory (always on): per-world upper bound on growth, recorded at
        # rejections, relaxed on accepts. Init above any reachable dt so it never
        # binds until a rejection writes it; also the finite stand-in for an inf
        # dt_max wherever a kernel arg must stay finite.
        self._ceiling_init = self._dt_max if self._dt_max != float("inf") else 1.0e6
        self._dt_ceiling = wp.full(wc, wp.float32(self._ceiling_init), dtype=wp.float32, device=device)
        self._guard_hits = wp.zeros(1, dtype=wp.int32, device=device)
        # Mixed-tolerance normalization: accept test becomes |d| <= atol + rtol*|q|
        # with atol = tol. Zero disables (bit-identical legacy path).
        self._err_rtol = float(os.environ.get("NEWTON_ADAPTIVE_RTOL", "2e-6") or 0.0)
        self._err_rtol_over_atol = self._err_rtol / self._tol if self._err_rtol > 0.0 else 0.0

        # Per-boundary device iteration counter (also latches the loop closed at
        # max_substeps, see _iters_exhausted_stop) and the non-resetting cumulative
        # attempt counter: each iteration runs the 3-eval step-doubling attempt, so
        # total SAP evals = iterations * 3, and rejected attempts are counted (a
        # rejection is just another iteration). Compute axis for work-precision;
        # reset with reset_compute_counter().
        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)
        self._cum_iters = wp.zeros(1, dtype=wp.int32, device=device)
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

        # Opt-in per-boundary march telemetry (see _log_march_boundary). The env
        # gate keeps the hot path to a single attribute check when unset.
        self._march_log_path = os.environ.get("NEWTON_ADAPTIVE_MARCH_LOG") or None
        self._march_log_file = None
        self._march_log_boundary = 0
        self._march_log_hist_every = 48
        self._reject_count_buf = wp.zeros(1, dtype=wp.int32, device=device)

        # dt-occupancy histogram (opt-in). Allocated here, never inside the captured
        # body. int64 because a long run at large world counts overflows int32.
        self._dt_hist: wp.array | None = None
        self._dt_hist_sat: wp.array | None = None
        self._dt_hist_trunc: wp.array | None = None
        self._march_iters: wp.array | None = None
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
            # March-only iteration count snapshot for the truncation label.
            self._march_iters = wp.zeros(1, dtype=wp.int32, device=device)

        self._coords_per_world = int(model.joint_coord_count) // wc
        self._dofs_per_world = int(model.joint_dof_count) // wc
        self._bodies_per_world = int(model.body_count) // wc
        self._world_count = wc
        self._max_rigid_contact = int(max_rigid_contact)

        # Accuracy-metric scaling S = identity (per PI directive); overwrite after
        # construction for expert per-coordinate scales.
        self._state_scale = wp.array(
            np.ones((wc, self._coords_per_world), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        # fixed mode commits the single full step (an honest fixed-dt baseline) and skips
        # the doubling it does not need; adaptive commits the doubled (more accurate) state
        # and feeds the step-doubling error to the controller.
        self._do_doubling = self._mode != "fixed"
        self._commit_src = self._scratch_full if self._mode == "fixed" else self._scratch_double
        self._err_lhs = self._scratch_full
        self._err_rhs = self._scratch_full if self._mode == "fixed" else self._scratch_double

        # ---- own collision pipeline ----
        # The pair cap is GLOBAL across worlds and overflow drops mesh contacts
        # silently, so it must be scene-sized by the caller — the default only
        # fits small scenes.
        # NEWTON_SAP_DETERMINISTIC (default OFF; "1" opts in): run-to-run
        # reproducibility for the whole SAP-adaptive stack. Here it makes the
        # pipeline sort contacts by a canonical key after the narrow phase,
        # so the global contact buffer ORDER (which every downstream per-env
        # slot assignment and fp summation inherits) is a pure function of
        # the state, not of atomic arrival order. sap_warp reads the same
        # variable for its own order-bound stages (per-env contact slots,
        # cost reductions, tree-force accumulation). Deterministic only while
        # capacity caps are unsaturated: an overflowing triangle-pair or
        # contact buffer drops entries by arrival order, which no sort can
        # canonicalize -- caps must stay scene-sized.
        self._deterministic = os.environ.get("NEWTON_SAP_DETERMINISTIC", "0") == "1"
        # Deterministic contact packing indexes every buffered candidate
        # contact with CONTACT_ID_BITS id bits, so the reducer capacity (=
        # the triangle-pair cap) must fit that budget; clamp oversized caps
        # to it. The determinism caveat is unchanged either way: an
        # overflowing cap drops entries by arrival order, so caps must stay
        # above the scene's live demand.
        _tri_pairs = int(max_triangle_pairs)
        if self._deterministic:
            from ...geometry.contact_reduction_global import CONTACT_ID_BITS  # noqa: PLC0415

            _tri_pairs = min(_tri_pairs, 1 << int(CONTACT_ID_BITS))
        self._pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=int(max_rigid_contact) * wc,
            max_triangle_pairs=_tri_pairs,
            deterministic=self._deterministic,
        )
        self._contacts = self._pipeline.contacts()
        self._collide_state = model.state()
        self._sap_contacts = sap_contacts_from_newton(self._contacts)
        self._sap_control = None
        # NEWTON_ADAPTIVE_CONTACT_REFRESH selects the intra-boundary contact
        # cadence (the same gate SolverMuJoCoAdaptive reads):
        #   unset/"1" (default) -- collide ONCE per boundary at the entry
        #     state; every attempt reuses that contact SET. The per-attempt
        #     Jacobian rebuild re-derives gap/points/Jacobian from the state
        #     each eval runs from (q_t, then q_{t+h/2}), which is this
        #     solver's intrinsic analog of the MuJoCo fast-path dist/pos
        #     refresh -- the error estimator keeps its contact sensitivity
        #     while frames/pairs/materials stay from the boundary pass.
        #   "attempt"/"2" -- re-collide at q_t and q_{t+h/2} on EVERY attempt
        #     (diagnostic-only cost/behavior attribution).
        #   "0" -- parsed for parity with the MuJoCo gate but identical to
        #     the default here: the re-anchor is intrinsic to the Jacobian
        #     rebuild and has no separate off switch.
        # Read at construction; the choice shapes the captured substep body
        # (per-attempt keeps the collide inside it, the default hoists it out).
        self._contact_refresh_per_attempt = os.environ.get("NEWTON_ADAPTIVE_CONTACT_REFRESH", "1") in (
            "attempt",
            "2",
        )
        # Boundary cadence + determinism: the contact SET is frozen per
        # boundary, so the canonical slot ranks are too -- compute them once
        # per boundary (outside the captured body) instead of on every
        # jacobian rebuild; per-attempt cadence keeps in-compute ranks.
        if self._deterministic and not self._contact_refresh_per_attempt:
            self._sap.contact_jacobian.det_slots_external = True
        # Host-side count of CollisionPipeline invocations. In the default
        # cadence every collide happens OUTSIDE the captured body, so this is
        # exact; in per-attempt mode it undercounts under graph replay (the
        # replay does not re-enter Python) -- probes that assert on it must
        # pin the eager path.
        self._collide_calls = 0

        # ---- world-level march compaction (narrow-grid tail body) ----
        # Default ON; NEWTON_SAP_MARCH_COMPACT=0 opts out (any value except
        # "0" enables, so partial/typo values fail toward the default). Late
        # march iterations carry only a few still-marching worlds, yet every
        # launch keeps its full fixed grid (graph-capture requirement); this
        # records the eval core (the three SAP solves + the error metric) as
        # a device-side conditional with TWO bodies -- the wide body is
        # today's launch stream verbatim, the narrow body is the SAME kernel
        # sequence with every list-indexed launch's env axis sized to a small
        # budget -- and routes each iteration by the device-read active-world
        # count. Bitwise by construction: list-indexed kernels exit at the
        # device count, so grid slots beyond it never did work; the subset
        # chain (world_active -> stage2 -> newton -> LS lists) bounds every
        # list by the active-world count, which the branch predicate bounds
        # by the budget; capacity guards latch a poison word on any
        # violation and the post-march reader raises. Requires the whole
        # compaction chain (tail/solve/LS) and the boundary contact cadence
        # (per-attempt collide inside the eval core must stay full-width and
        # is excluded rather than special-cased); needs >= 2 worlds so both
        # branches are reachable. Mask-gated (non-listed) kernels keep full
        # grids in BOTH bodies -- the five status kernels, collision, the
        # contact scatter, and the controller/commit/clamp kernels among
        # them, which is what the hold-semantics argument requires.
        _mc_requested = os.environ.get("NEWTON_SAP_MARCH_COMPACT", "1") != "0"
        self._march_compact = bool(
            _mc_requested
            and self._tail_compact
            and self._sap.contact_solve._solve_compact
            and self._sap.contact_solve._ls_compact
            and not self._contact_refresh_per_attempt
            and wc >= 2
        )
        # Narrow-grid env budget: small enough to shed most idle grid slots,
        # large enough that mid-march active counts still route wide only
        # while genuinely wide; capped at wc//2 so the wide branch stays
        # reachable at any world count (both-branch tripwires need it).
        _mc_width = os.environ.get("NEWTON_SAP_MARCH_COMPACT_WIDTH")
        if _mc_width is not None:
            _mc_width = int(_mc_width)
        else:
            _mc_width = max(64, wc // 16)
        self._mc_width = max(1, min(_mc_width, max(1, wc // 2)))
        # NEWTON_SAP_SHARED_ASSEMBLY (default ON; "0" opts out): within one
        # step-doubling attempt the full solve and the first half solve
        # anchor at the same state, contact set, control and world mask, so
        # the dt-independent assembly (rigid ID, tau accumulation, mass
        # matrix + factorization, body/contact Jacobians, Delassus weights)
        # would rewrite byte-identical buffers; the half solve reuses the
        # full solve's assembly and re-runs only the dt-dependent work (the
        # per-world dt fill, v_star assembly, and the contact solve itself).
        # Only the first half solve qualifies -- the second anchors at the
        # midpoint state and keeps its own assembly.
        self._shared_assembly = os.environ.get("NEWTON_SAP_SHARED_ASSEMBLY", "1") != "0" and self._do_doubling
        self._sa_execs = wp.zeros(1, dtype=wp.int32, device=device)

        # ---- cross-boundary overlap: RUN-AHEAD single march (default OFF) ----
        # NEWTON_SAP_RUNAHEAD=1 opts in: a world reaching its boundary target
        # inside the action window does not park -- a device crossing kernel
        # applies its boundary bookkeeping in place and it marches on, capped
        # at the window end (NEWTON_SAP_RUNAHEAD_WINDOW boundary calls per
        # action window; must equal the env's decimation). The call-return
        # predicate compares sim_time against a per-call device scalar target,
        # so run-ahead worlds count as finished for the call and the LAST call
        # of the window returns only when every world sits at the window end
        # (action-edge state stays batch-synchronized). Boundary collides
        # become crossing-batched conditional nodes inside the march body
        # (masked collide + per-env set ADOPT); per-world contact cadence and
        # anchoring are semantically identical to the per-boundary march.
        # Per-world dt control, the step-doubling estimator and accept/reject
        # are untouched. BATCH-VISIBLE semantic change: state written back at
        # mid-window calls shows run-ahead worlds at mixed boundary times
        # (action-edge reads are unaffected); shipping this ON requires the
        # consumer to have no sub-action-cadence state reader and no
        # per-physics-step control variation. Engagement counters are
        # allocated unconditionally so an OFF-configuration read is a real
        # observation (leak tripwire), mirroring the march-compact counters.
        self._ra_crossings = wp.zeros(1, dtype=wp.int32, device=device)
        self._ra_adopts = wp.zeros(1, dtype=wp.int32, device=device)
        self._runahead = os.environ.get("NEWTON_SAP_RUNAHEAD", "0") == "1"
        if self._runahead:
            if self._contact_refresh_per_attempt:
                raise ValueError(
                    "NEWTON_SAP_RUNAHEAD=1 requires the boundary contact cadence "
                    "(NEWTON_ADAPTIVE_CONTACT_REFRESH=attempt re-collides inside the eval core; "
                    "the run-ahead crossing node owns the collide cadence instead)."
                )
            if self._solve_precision != "fp64":
                raise ValueError(
                    "NEWTON_SAP_RUNAHEAD=1 supports the fp64 solve stack only "
                    "(the fp32 stack has no certified adopt/anchor coverage)."
                )
            self._runahead_window = int(os.environ.get("NEWTON_SAP_RUNAHEAD_WINDOW", "4"))
            if self._runahead_window < 1:
                raise ValueError(f"NEWTON_SAP_RUNAHEAD_WINDOW must be >= 1, got {self._runahead_window}.")
            # Call-phase offset for callers whose boundary-call stream does not
            # start on an action-window edge (e.g. startup settle steps).
            self._ra_phase = int(os.environ.get("NEWTON_SAP_RUNAHEAD_PHASE", "0")) % self._runahead_window
            self._ra_call_idx = 0
            self._ra_window_pos = 0
            self._ra_t_call = wp.zeros(1, dtype=wp.float32, device=device)
            self._ra_t_call_host = 0.0
            self._ra_crossed = wp.zeros(wc, dtype=wp.int32, device=device)
            self._ra_crossed_any = wp.zeros(1, dtype=wp.int32, device=device)
            self._ra_all_worlds = wp.ones(wc, dtype=wp.int32, device=device)
            self._ra_target_cache: dict = {}
            # Crossing-batch throttle: a world reaching its boundary is HELD
            # parked there (making no attempts -- exactly the per-boundary
            # march's parked-world state) until the pending batch reaches the
            # bound, or until no world can advance without crossing (the
            # liveness rule: the march must never spin with only parked
            # worlds). Batching fires trades run-ahead depth for fewer
            # masked collide+adopt passes; it can only DELAY a world's
            # crossing, never skip or reorder it, and a held world's contact
            # set still refreshes at exactly its boundary-entry state, so
            # per-world contact cadence and anchoring are unchanged. Bound
            # semantics: values >= 1 are an absolute pending-world count;
            # values in (0, 1) are a fraction of the world count (scale-
            # invariant across env counts). Bound 1 fires every crossing
            # immediately (the unthrottled schedule); a bound at/above the
            # world count degenerates to lockstep generations (fires only
            # via the liveness rule), which forfeits the merge value -- the
            # default sits between the two failure edges.
            raw_bound = float(os.environ.get("NEWTON_SAP_RUNAHEAD_BATCH", "0.5"))
            if raw_bound <= 0.0:
                raise ValueError(f"NEWTON_SAP_RUNAHEAD_BATCH must be > 0, got {raw_bound}.")
            if raw_bound < 1.0:
                self._ra_batch_bound = max(1, int(round(raw_bound * wc)))
            else:
                self._ra_batch_bound = int(round(raw_bound))
            # Max-hold age (march iterations a non-empty pending set may be
            # held before the gate opens regardless of the count bound).
            # Small values cap every holder's delay at a few cheap trailing
            # iterations -- the plateau regime's escape hatch from the
            # liveness barrier; the count bound still fires large batches
            # immediately in the wide regime.
            self._ra_batch_age = int(os.environ.get("NEWTON_SAP_RUNAHEAD_BATCH_AGE", "2"))
            if self._ra_batch_age < 1:
                raise ValueError(f"NEWTON_SAP_RUNAHEAD_BATCH_AGE must be >= 1, got {self._ra_batch_age}.")
            self._ra_pending_counts = wp.zeros(2, dtype=wp.int32, device=device)
            self._ra_wait = wp.zeros(1, dtype=wp.int32, device=device)
            self._ra_fire = wp.zeros(1, dtype=wp.int32, device=device)
            # Per-world contact persistence: the global buffer only ever holds
            # the latest crossing batch, so per-env sets own the anchored
            # rows; the per-attempt scatter becomes the ANCHOR re-derivation.
            self._sap.contact_jacobian.enable_runahead_set_store()
            # Deterministic slot ranks are computed per crossing batch inside
            # adopt_contact_set; the per-boundary external-slot walk is
            # superseded.
            self._sap.contact_jacobian.det_slots_external = False

        # Engagement counters (dim-1 increments recorded FIRST inside each
        # branch body). Allocated UNCONDITIONALLY so an OFF-configuration read
        # is a real observation: any branch-body emission in an OFF cell
        # advances a counter instead of being masked by an early return.
        self._mc_narrow_execs = wp.zeros(1, dtype=wp.int32, device=device)
        self._mc_wide_execs = wp.zeros(1, dtype=wp.int32, device=device)
        if self._march_compact:
            # Branch condition word (int32[1], read by the conditional node)
            # and the chunked canonical-build scratch.
            self._mc_cond = wp.zeros(1, dtype=wp.int32, device=device)
            _n_chunks = (wc + _MC_CHUNK - 1) // _MC_CHUNK
            self._mc_n_chunks = _n_chunks
            self._mc_chunk_counts = wp.zeros(_n_chunks, dtype=wp.int32, device=device)
            self._mc_chunk_offsets = wp.zeros(_n_chunks, dtype=wp.int32, device=device)

        # ---- solver-internal CUDA-graph capture of the substep BODY ----
        # The per-substep body is a flat kernel sequence, so it captures once and replays at
        # driver speed; the boundary loop reads a 4-byte flag between replays to stop early.
        # Gated by NEWTON_SAP_ADAPTIVE_GRAPH (default on) and CUDA; CPU unit tests run the
        # eager loop. Cached per dt_outer.
        try:
            _is_cuda = bool(wp.get_device(device).is_cuda)
        except Exception:
            _is_cuda = False
        # Body-graph capture. The convergent solve uses wp.capture_while; capturing ONE substep
        # body that contains it is the intended use of conditional graph nodes, and its node
        # count is constant in env count (never one body per substep, whose node count would
        # scale with the loop). On any capture/instantiate failure the loop falls back to eager.
        self._is_cuda = _is_cuda
        self._graph_enabled = _is_cuda and os.environ.get("NEWTON_SAP_ADAPTIVE_GRAPH", "1") != "0"
        self._graph_cache: dict = {}
        # Modules/allocations must be warm before capture (a launch that triggers a lazy
        # module load syncs the stream and aborts capture). Run the first frame eagerly.
        self._graph_warmed = False

        # ---- whole-march conditional tier (default ON; NEWTON_SAP_ADAPTIVE_CONDITIONAL=0 opts out) ----
        # The march loop itself records as ONE wp.capture_while conditional
        # while-node whose body is the substep body: the per-iteration 4-byte
        # status poll collapses to a single post-march status read per boundary
        # (the converge-or-throw check), and an outer manager-level capture may
        # wrap the boundary call around the while-node. The solve's own device
        # conditionals (capture_if / capture_while) nest inside the while-node
        # body; conditional bodies may not record allocations or host work, so
        # warmup boundaries run on the per-iteration tier first and any
        # capture/launch failure downgrades permanently to that tier (never
        # crashes a run). A pure scheduling change: the while-node body is the
        # SAME launch stream the per-iteration tier replays.
        self._conditional_enabled = (
            self._graph_enabled and os.environ.get("NEWTON_SAP_ADAPTIVE_CONDITIONAL", "1") != "0"
        )
        self._conditional_graph_cache: dict = {}
        self._conditional_warm_boundaries = 0
        # Host-side count of whole-march conditional replays: the engagement
        # tripwire probes read to prove the tier actually executed.
        self._conditional_launches = 0
        # While-node loop condition, separate from the status word (see
        # _march_continue_from_status).
        self._march_continue = wp.zeros(1, dtype=wp.int32, device=device)

        # Quantile stop: end the boundary once the active set has fallen to this
        # many worlds, instead of waiting for the last straggler. The cutoff is
        # a WORLD COUNT derived from a fraction of the batch, so it scales with
        # env count. 0 (the default) reproduces march-to-the-last-world exactly.
        _q = float(os.environ.get("NEWTON_SAP_QUANTILE_STOP", "0.0"))
        self._quantile_stop = min(max(_q, 0.0), 1.0)
        self._quantile_cutoff = int(wc * (1.0 - self._quantile_stop)) if self._quantile_stop > 0.0 else 0
        # Sticky per-world record of environments abandoned short of the
        # boundary. Their state is mid-boundary and must be dropped, never
        # consumed as a transition.
        self._boundary_cut = wp.zeros(wc, dtype=wp.int32, device=device)
        # Raised on the trip where the cutoff closes the loop, so the marking
        # kernel can identify the abandoned worlds before they are landed.
        self._cut_gate = wp.zeros(1, dtype=wp.int32, device=device)

    # ---------------------------------------------------------------- properties
    @property
    def diverged(self) -> wp.array:
        return self._diverged

    @property
    def containment(self) -> bool:
        """True when per-world solve-failure containment is active (default);
        False in strict converge-or-throw mode (``NEWTON_SAP_CONTAINMENT=0``)."""
        return self._containment

    @property
    def march_compact(self) -> bool:
        """True when the narrow-grid tail body (world-level march compaction)
        is active: default ON, disabled by ``NEWTON_SAP_MARCH_COMPACT=0`` or
        by a missing prerequisite (tail/solve/LS compaction, boundary contact
        cadence, >= 2 worlds)."""
        return self._march_compact

    @property
    def march_compact_width(self) -> int:
        """Env budget of the narrow branch's list-indexed grids."""
        return self._mc_width

    def shared_assembly_execs(self) -> int:
        """Replay count of half-1 solves that reused the full solve's assembly
        (host read; probe-side engagement tripwire, not consumed by physics)."""
        return int(self._sa_execs.numpy()[0])

    def march_compact_execs(self) -> tuple[int, int]:
        """(narrow, wide) branch execution counts -- the engagement
        tripwires. Host sync; call outside the hot path. Always a real
        device read (the counters exist in every configuration), so an OFF
        configuration's (0, 0) observes that no branch body ever emitted
        rather than restating the flag."""
        return (
            int(self._mc_narrow_execs.numpy()[0]),
            int(self._mc_wide_execs.numpy()[0]),
        )

    @property
    def runahead(self) -> bool:
        """True when the run-ahead single march is active (``NEWTON_SAP_RUNAHEAD=1``;
        default OFF -- the OFF path is the per-boundary march, byte-untouched)."""
        return self._runahead

    @property
    def runahead_batch_bound(self) -> int:
        """Resolved crossing-batch throttle bound (worlds pending before the
        gate opens; ``NEWTON_SAP_RUNAHEAD_BATCH``, fractions resolved against
        the world count). 0 when run-ahead is inactive."""
        return self._ra_batch_bound if self._runahead else 0

    @property
    def runahead_batch_age(self) -> int:
        """Max-hold age of the crossing-batch throttle (march iterations a
        non-empty pending set may be held; ``NEWTON_SAP_RUNAHEAD_BATCH_AGE``).
        0 when run-ahead is inactive."""
        return self._ra_batch_age if self._runahead else 0

    @property
    def runahead_window(self) -> int:
        """Boundary calls per action window in run-ahead mode (0 when off)."""
        return self._runahead_window if self._runahead else 0

    def runahead_engagement(self) -> tuple[int, int]:
        """(boundary crossings, consumed collide+adopt batches) -- the
        run-ahead engagement tripwires. Host sync; call outside the hot path.
        Always a real device read (the counters exist in every
        configuration), so an OFF configuration's (0, 0) observes that the
        crossing kernel and the conditional collide node never fired rather
        than restating the flag."""
        return (
            int(self._ra_crossings.numpy()[0]),
            int(self._ra_adopts.numpy()[0]),
        )

    def _raise_if_march_compact_poisoned(self) -> None:
        """Fail loudly if any narrowed grid's capacity invariant was violated
        (a device list outgrew the narrow budget -- envs would have been
        silently skipped). Host read; callers invoke it only at post-march /
        eager points, never inside a capture."""
        if not self._march_compact:
            return
        if int(self._sap.contact_solve._env_grid_poison.numpy()[0]) != 0:
            raise RuntimeError(
                "SolverSAPAdaptive march compaction: a device env list exceeded "
                f"the narrow grid budget ({self._mc_width}); results from the "
                "poisoned march are unsafe. This invariant is structural "
                "(branch cond and subset chain bound every list by the budget) "
                "-- a violation means the routing or the chain was edited "
                "inconsistently."
            )

    @property
    def solve_failure_worlds(self) -> wp.array:
        """Cumulative per-world count of contained inner-solve failure events,
        shape ``[world_count]``, int32, on device. Read with ``.numpy()``
        OUTSIDE the inner loop only (it is a device sync)."""
        return self._solve_fail_world

    @property
    def solve_failure_events(self) -> int:
        """Host-side cumulative count of contained inner-solve failure events
        (updated on the rare post-march path of a failing boundary)."""
        return self._solve_failure_events

    @property
    def dt(self) -> wp.array:
        return self._dt

    @property
    def sim_time(self) -> wp.array:
        return self._sim_time

    @property
    def last_error(self) -> wp.array:
        return self._accepted_error

    @property
    def accepted(self) -> wp.array:
        return self._accepted

    @property
    def substeps(self) -> wp.array:
        """Per-world accepted-substep count for the most recent frame (per-world work)."""
        return self._substeps_frame

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def solve_precision(self) -> str:
        """Resolved solve-stack precision, ``"fp64"`` (default) or ``"fp32"`` (opt-in).

        fp32 also couples the inner optimality target to the fp32-achievable
        analogue (``optimality_rel_tol`` reflects the coupled value); the
        integration tolerance ``tol`` is the physics contract and is identical
        in both settings.
        """
        return self._solve_precision

    @property
    def optimality_rel_tol(self) -> float:
        """The inner solve's optimality target actually in force (dtype-coupled)."""
        return float(self._optimality_rel_tol)

    @property
    def tiling(self) -> str:
        # Retained for back-compat; the per-world dt vector is no longer an even tiling.
        return "adaptive"

    @property
    def contacts(self):
        return self._contacts

    @property
    def iteration_count(self) -> wp.array:
        """Iteration count from the most recent boundary call, shape ``[1]``, int32, on device."""
        return self._iteration_count_buf

    @property
    def cumulative_iterations(self) -> wp.array:
        """Boundary-loop iterations accumulated since the last :meth:`reset_compute_counter`,
        shape ``[1]``, int32, on device. Includes rejected attempts. Read with ``.numpy()``
        OUTSIDE the inner loop only (it is a device sync)."""
        return self._cum_iters

    def cumulative_substeps(self) -> int:
        """Total SAP evals since the last :meth:`reset_compute_counter` (= boundary-loop
        attempts * 3 in adaptive mode; rejected attempts included -- a rejection is just
        another iteration). Compute axis for work-precision. The three evals share no
        forward prefix, and the two half-step solves converge cheaper than the full one,
        so this counts evals, not equal-cost work units. Fixed mode runs one eval per
        iteration and is counted as such. Host sync; call outside the hot path."""
        return int(self._cum_iters.numpy()[0]) * (3 if self._do_doubling else 1)

    def cumulative_accepted_steps(self) -> int:
        """Total ACCEPTED per-world substeps since the last
        :meth:`reset_compute_counter` (each accepting world counts one per
        accepted attempt; rejections excluded). This is the DEMAND axis:
        unlike :meth:`cumulative_substeps`, which counts batch march
        iterations and therefore depends on how the march schedules worlds
        into shared iterations, this counts per-world integration work and
        is schedule-invariant -- equal demand between two runs makes their
        wall ratio a pure speed comparison. Host sync; call outside the hot
        path."""
        return int(self._cum_accepted.numpy()[0])

    def get_status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary via one GPU transfer."""
        device = self.model.device
        n = self._world_count

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
            ``0.0`` if the floor was never hit -- ``boundaries`` (boundary calls
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
            raise RuntimeError("dt_histogram_stats() requires SolverSAPAdaptive(dt_histogram=True)")
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

    def get_max_contact_count(self) -> int:
        """Per-batch rigid-contact capacity (for manager-level sensor buffer sizing)."""
        return self._max_rigid_contact * self._world_count

    def update_contacts(self, contacts, state) -> None:
        """No-op: SAP-adaptive owns its internal contact set; contact-sensor writeback
        from SAP is not yet wired (documented limitation for v1)."""
        return None

    def reset_compute_counter(self) -> None:
        """Zero the cumulative iteration/substep counters."""
        self._cum_iters.fill_(0)
        self._cum_accepted.fill_(0)

    def notify_model_changed(self, flags: int) -> None:
        """Forward model-change notifications to the inner SAP solver.

        The controller's own state is per-world scalars (dt / clocks / latches) unaffected
        by model-array changes, so only the inner ``SolverSAP``'s topology caches refresh.
        """
        self._sap.notify_model_changed(flags)

    # ----------------------------------------------------------- state copy utils
    @staticmethod
    def _copy_state(dst, src) -> None:
        wp.copy(dst.joint_q, src.joint_q)
        wp.copy(dst.joint_qd, src.joint_qd)
        if src.body_q is not None and dst.body_q is not None:
            wp.copy(dst.body_q, src.body_q)
        if src.body_qd is not None and dst.body_qd is not None:
            wp.copy(dst.body_qd, src.body_qd)
        if src.body_f is not None and dst.body_f is not None:
            wp.copy(dst.body_f, src.body_f)

    # ---------------------------------------------------------- warm-start seam
    def _set_solver_guess(self, guess) -> None:
        if guess is None:
            wp.launch(
                _set_scalar_i32,
                dim=1,
                inputs=[self._sap._contact_solve_v_guess_active, 0],
                device=self.model.device,
            )
            return
        wp.copy(self._sap.contact_solve.v_flat, guess)
        wp.launch(
            _set_scalar_i32,
            dim=1,
            inputs=[self._sap._contact_solve_v_guess_active, 1],
            device=self.model.device,
        )

    def _copy_state_velocity_to_sap_guess(self, state_in, guess) -> None:
        if getattr(state_in, "joint_qd_order", "sap") == "public":
            if guess.dtype == wp.float32:
                # The boundary conversion writes canonical f64 SAP velocities;
                # stage there, then downcast into the f32 guess buffer.
                self._sap._copy_public_joint_velocity_to_sap(state_in, self._guess_stage_f64)
                wp.launch(
                    _guess_stage_f64_to_f32,
                    dim=int(self.model.joint_dof_count),
                    inputs=[self._guess_stage_f64, guess],
                    device=self.model.device,
                )
            else:
                self._sap._copy_public_joint_velocity_to_sap(state_in, guess)
        else:
            wp.copy(guess, state_in.joint_qd)

    def _average_velocity_guess(self, a, b, out) -> None:
        wp.launch(
            self._average_velocity_guess_kernel,
            dim=int(self.model.joint_dof_count),
            inputs=[a, b, out],
            device=self.model.device,
        )

    def _collide_from(self, state_in) -> None:
        self._collide_calls += 1
        wp.copy(self._collide_state.body_q, state_in.body_q)
        self._pipeline.collide(self._collide_state, self._contacts)

    def substep(
        self, state_in, state_out, control, contacts, dt: wp.array, guess=None, world_active=None, reuse_assembly=False
    ) -> None:
        """ONE inner physics step at the per-world ``dt`` vector (= CENIC ``ComputeNextContinuousState``).

        ``guess`` is an explicit SAP-order velocity seed. Passing ``None`` disables the
        solver's persisted warm-start so the solve starts from its physical boundary
        velocity ``v0``. This mirrors Drake's CENIC warm-starts instead of accidentally
        reusing a rejected attempt's terminal velocity.

        ``world_active`` (per-world int gate) restricts the SAP pipeline to
        still-marching worlds; landed worlds' ``state_out`` rows go stale,
        which is safe because the accept-gated commit never reads them.
        """
        self._set_solver_guess(guess)
        self._sap.step(
            state_in, state_out, control, contacts, dt, world_active=world_active, reuse_assembly=reuse_assembly
        )
        wp.launch(
            _accumulate_solve_convergence,
            dim=self._world_count,
            inputs=[self._sap.contact_solve.converged_env, self._solve_ok],
            device=self.model.device,
        )

    # ----------------------------------------------------------- eval core
    def _substep_evals(self, wa, env_grid: int, narrow: bool) -> None:
        """Emit the eval core once: the three SAP solves plus the error metric.

        ``env_grid`` sizes the ENV AXIS of every list-indexed launch (threaded
        into the contact-solve interior via ``set_env_grid``; consumed
        directly by the indexed error kernel); the subset chain
        world_active -> stage2 -> newton -> LS bounds every list by the
        active-world count, which the caller's branch predicate bounds by
        ``env_grid``. ``narrow`` marks the march-compact narrow branch, which
        records its engagement counter FIRST (execution proof for the
        tripwires) and then the branch-capacity guard re-checking the routing
        invariant on device. Everything mask-gated or full-width-by-design
        (the solve's status/init kernels, collision, the world-gated contact
        scatter, the v-guess copies/averages) emits identically in both
        branches: any restriction there comes from the mask VALUES read per
        replay, not from branch structure.
        """
        n = self._world_count
        dev = self.model.device

        if self._march_compact:
            if narrow:
                wp.launch(_iter_count_increment, dim=1, inputs=[self._mc_narrow_execs], device=dev)
                wp.launch(
                    _env_grid_capacity_guard,
                    dim=1,
                    inputs=[self._active_counts, self._mc_width, self._sap.contact_solve._env_grid_poison],
                    device=dev,
                )
            else:
                wp.launch(_iter_count_increment, dim=1, inputs=[self._mc_wide_execs], device=dev)
            self._sap.contact_solve.set_env_grid(env_grid)

        try:
            # Per-attempt cadence only: rebuild the contact set at q_t for the
            # full step and first half-step. The default (boundary) cadence
            # collided once in integrate(), outside this (captured) body, and
            # every attempt reuses that set with per-attempt re-anchoring via
            # the Jacobian rebuild. When collision does run here it stays
            # FULL-WIDTH in every mode: the pipeline compacts all worlds'
            # contacts into one global buffer via atomic slot assignment, so
            # restricting its thread set could permute active worlds' contact
            # order (and downstream fp summation order). March compaction
            # excludes the per-attempt cadence at construction, so this
            # branch never emits inside a conditional body.
            if self._contact_refresh_per_attempt:
                self._collide_from(self._state_cur)

            # Drake CENIC warm-starts: full from v_t.
            self._copy_state_velocity_to_sap_guess(self._state_cur, self._vt)
            self.substep(
                self._state_cur,
                self._scratch_full,
                self._sap_control,
                self._sap_contacts,
                self._dt,
                guess=self._vt,
                world_active=wa,
            )
            if self._do_doubling:
                # half-1 from (v_t + v_full) / 2, reusing the q_t contact model.
                wp.copy(self._vfull, self._sap.contact_solve.v_flat)
                self._average_velocity_guess(self._vt, self._vfull, self._vhalf1)
                if self._shared_assembly:
                    # Counter records inside the captured body so replays
                    # count: the tripwire proving the reuse path executes.
                    wp.launch(_iter_count_increment, dim=1, inputs=[self._sa_execs], device=dev)
                self.substep(
                    self._state_cur,
                    self._scratch_mid,
                    self._sap_control,
                    self._sap_contacts,
                    self._dt_half,
                    guess=self._vhalf1,
                    world_active=wa,
                    reuse_assembly=self._shared_assembly,
                )

                # half-2 starts from q_{t+h/2}; warm-start from v_full. In the
                # default (boundary) cadence the boundary contact SET is
                # reused and this solve's Jacobian rebuild re-anchors it at
                # the midpoint state (the mid-double refresh that keeps the
                # Richardson pair contact-sensitive); per-attempt mode
                # re-collides instead.
                if self._contact_refresh_per_attempt:
                    self._collide_from(self._scratch_mid)
                self.substep(
                    self._scratch_mid,
                    self._scratch_double,
                    self._sap_control,
                    self._sap_contacts,
                    self._dt_half,
                    guess=self._vfull,
                    world_active=wa,
                )

            if self._tail_compact:
                # Compacted error metric: landed worlds keep their last
                # written error, which never reaches a decision (the
                # controller's DONE branch returns before reading it). The
                # env axis runs at env_grid; the kernel exits at the
                # device-read active count.
                wp.launch(
                    _inf_norm_state_error_indexed_kernel,
                    dim=env_grid,
                    inputs=[
                        self._err_lhs.joint_q,
                        self._err_rhs.joint_q,
                        self._state_scale,
                        self._coords_per_world,
                        self._commit_src.joint_qd,
                        self._dofs_per_world,
                        self._active_idx,
                        self._active_counts,
                        0,
                        self._err_rtol_over_atol,
                    ],
                    outputs=[self._last_error],
                    device=dev,
                )
            else:
                wp.launch(
                    _inf_norm_state_error_kernel,
                    dim=n,
                    inputs=[
                        self._err_lhs.joint_q,
                        self._err_rhs.joint_q,
                        self._state_scale,
                        self._coords_per_world,
                        self._commit_src.joint_qd,
                        self._dofs_per_world,
                        self._err_rtol_over_atol,
                    ],
                    outputs=[self._last_error],
                    device=dev,
                )
        finally:
            if self._march_compact:
                self._sap.contact_solve.set_env_grid(None)

    # ------------------------------------------------------- run-ahead helpers
    def _ra_targets(self, dt_outer: float) -> np.ndarray:
        """Boundary-target chain for one action window: ``[T_0 .. T_W]`` with
        ``T_k`` built by k successive float32 adds of ``dt_outer`` -- the SAME
        fp operation the device-side crossing bump performs -- so host targets
        and per-world ``next_time`` values are bit-comparable (the window-end
        park test and the reset-resync signature rely on exact equality).
        Cached per dt_outer."""
        key = round(float(dt_outer), 12)
        targets = self._ra_target_cache.get(key)
        if targets is None:
            w = self._runahead_window
            step = np.float32(dt_outer)
            targets = np.zeros(w + 1, dtype=np.float32)
            acc = np.float32(0.0)
            for k in range(1, w + 1):
                acc = np.float32(acc + step)
                targets[k] = acc
            self._ra_target_cache[key] = targets
        return targets

    def _ra_crossing_refresh(self) -> None:
        """Body of the crossing-batched conditional node: masked collide at
        the crossed worlds' boundary-entry states (sentinel AABBs keep every
        other world out of the broad phase), per-env set ADOPT for exactly
        those worlds, then flag disarm. Device-only launches with no
        allocations -- records under both capture tiers and runs eagerly
        during warmup. In deterministic mode the pass skips the global
        contact sort (its native implementation allocates temp memory, which
        a conditional graph body cannot contain) and the adopt derives the
        SAME canonical per-env slot order from the per-contact sort keys
        directly."""
        wp.copy(self._collide_state.body_q, self._state_cur.body_q)
        self._pipeline.collide(self._collide_state, self._contacts, world_mask=self._ra_crossed, sort_contacts=False)
        self._sap.contact_jacobian.adopt_contact_set(
            self._sap_contacts,
            self._ra_crossed,
            sort_keys=self._pipeline._sort_key_array if self._deterministic else None,
        )
        wp.launch(
            _ra_clear_crossed,
            dim=self._world_count,
            inputs=[self._ra_crossed, self._ra_crossed_any, self._ra_adopts],
            device=self.model.device,
        )

    # ----------------------------------------------------------- substep body
    def _substep_body(self, eff_dt_max: float, dt_outer: float) -> None:
        """One masked substep iteration: clamp -> evals -> error -> adapt -> commit -> mark.

        Identical flat kernel sequence every iteration, so it captures ONCE and replays per
        iteration. Per-world accept/reject/done is decided in ``_adapt_dt`` and applied by
        the gated ``_commit_*`` launches (a rejected or done world holds ``state_cur``). The
        final boundary-exit kernel (``mark_unfinished_contained`` by default,
        ``mark_unfinished_with_status`` in strict mode) sets the boundary flag the loop reads to stop early; the
        flag is reset by the caller before each iteration so it reflects this step only.

        Run-ahead mode adds three fixed nodes to the same flat stream: a
        crossing-batched conditional collide+adopt at the top (fires only when
        some world crossed a boundary since the last consume), the crossing
        kernel after the commit, and the scalar-target boundary-exit kernels
        in place of the per-world ones. ``dt_outer`` shapes only these nodes
        (the graph cache is keyed per dt_outer either way).
        """
        n = self._world_count
        dev = self.model.device

        # Crossing-batched contact refresh: worlds flagged by the crossing
        # kernel (or the mid-window reset resync) get a masked collide at
        # their boundary-entry state plus a per-env set ADOPT, then the flags
        # disarm. Recorded as one conditional node; the whole subgraph is
        # skipped while no world has crossed. Placed FIRST so a world that
        # crossed on the previous iteration (or previous call) refreshes
        # before its next attempt consumes the contact set.
        if self._runahead:
            wp.capture_if(self._ra_crossed_any, on_true=self._ra_crossing_refresh)

        # Sample BEFORE _clamp_dt_to_boundary: dt still holds the controller's chosen
        # step here, not a landing sliver (worlds already landed are skipped in-kernel).
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

        # Count this attempt (per-boundary + cumulative). A rejection is just another iteration.
        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time, self._limited],
            device=dev,
        )

        # Compact the unfinished worlds AFTER the clamp (post-clamp dt > 0 is
        # the single "attempts a step" predicate) and before anything consumes
        # the list/mask. Fixed-dim launches; the count lives on device. Under
        # deterministic march compaction the list becomes an iteration space
        # (the narrow body sizes grids to it), so it is built canonically
        # ascending instead of in atomic arrival order; both builds produce
        # the same set/count and every consumer writes world-private rows,
        # so the choice cannot perturb any floating-point result.
        wa = None
        if self._tail_compact:
            if self._march_compact and self._deterministic:
                wp.launch(
                    _build_active_worlds_chunk_count,
                    dim=self._mc_n_chunks,
                    inputs=[self._dt, n, _MC_CHUNK, self._mc_chunk_counts],
                    device=dev,
                )
                wp.launch(
                    _build_active_worlds_chunk_scan,
                    dim=1,
                    inputs=[self._mc_chunk_counts, self._mc_n_chunks, self._mc_chunk_offsets, self._active_counts],
                    device=dev,
                )
                wp.launch(
                    _build_active_worlds_ordered_scatter,
                    dim=self._mc_n_chunks,
                    inputs=[self._dt, n, _MC_CHUNK, self._mc_chunk_offsets, self._active_idx, self._world_active],
                    device=dev,
                )
            else:
                wp.launch(_reset_active_counts, dim=1, inputs=[self._active_counts], device=dev)
                wp.launch(
                    _build_active_worlds,
                    dim=n,
                    inputs=[self._dt, self._active_counts, self._active_idx, self._world_active],
                    device=dev,
                )
            wa = self._world_active

        wp.launch(_reset_solve_convergence, dim=n, inputs=[self._solve_ok], device=dev)

        # Eval core: the three SAP solves plus the error metric. With march
        # compaction the core records as a device-side conditional -- one
        # wide body (today's stream verbatim) and one narrow body (the same
        # stream with list-indexed env grids sized to the budget) -- routed
        # per iteration by the active-world count written just above. Both
        # branch bodies are pure pre-allocated launch streams (no host work,
        # no allocations), so the conditional records under both capture
        # tiers and executes eagerly during warmup.
        if self._march_compact:
            wp.launch(
                _derive_narrow_cond,
                dim=1,
                inputs=[self._active_counts, self._mc_width, self._mc_cond],
                device=dev,
            )
            wp.capture_if(
                self._mc_cond,
                on_true=lambda: self._substep_evals(wa, self._mc_width, True),
                on_false=lambda: self._substep_evals(wa, n, False),
            )
        else:
            self._substep_evals(wa, n, False)

        wp.launch(
            _apply_solve_convergence_to_error,
            dim=n,
            inputs=[self._solve_ok, self._last_error, self._divergence_threshold],
            device=dev,
        )

        wp.launch(
            _adapt_dt,
            dim=n,
            inputs=[
                self._last_error,
                self._sim_time,
                self._next_time,
                self._dt,
                self._dt_half,
                self._ideal_dt,
                self._diverged,
                self._accepted,
                self._accepted_error,
                self._substeps_frame,
                self._cum_accepted,
                self._mode_code,
                self._tol,
                self._dt_min,
                eff_dt_max,
                self._divergence_threshold,
                self._dt_ceiling,
                self._limited,
                self._consec_rej,
                self._ceiling_init,
                self._dt_inner_init,
            ],
            device=dev,
        )
        if self._march_log_path is not None:
            wp.launch(
                _count_rejects,
                dim=n,
                inputs=[self._accepted, self._dt, self._diverged, self._reject_count_buf],
                device=dev,
            )

        src = self._commit_src
        wp.launch(
            _commit_float,
            dim=self.model.joint_coord_count,
            inputs=[src.joint_q, self._accepted, self._coords_per_world],
            outputs=[self._state_cur.joint_q],
            device=dev,
        )
        wp.launch(
            _commit_float,
            dim=self.model.joint_dof_count,
            inputs=[src.joint_qd, self._accepted, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd],
            device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                _commit_transform,
                dim=self.model.body_count,
                inputs=[src.body_q, self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_q],
                device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                _commit_spatial_vector,
                dim=self.model.body_count,
                inputs=[src.body_qd, self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_qd],
                device=dev,
            )

        # Run-ahead crossing: worlds that just landed on their boundary target
        # bump it (capped at the window end), apply their boundary bookkeeping
        # in place, and arm the next iteration's collide+adopt node. Runs
        # after the commit so the crossing state IS the committed boundary
        # state; per-world data flow only. The count/decide pair ahead of it
        # is the crossing-batch throttle: the gate opens only when the
        # pending batch reaches the bound or nothing else can march, so
        # scattered landings batch into few collide+adopt fires instead of
        # arming the conditional node one world at a time.
        if self._runahead:
            window_end = float(self._ra_targets(dt_outer)[self._runahead_window])
            wp.launch(
                _ra_throttle_count,
                dim=n,
                inputs=[window_end, self._sim_time, self._next_time, self._diverged, self._ra_pending_counts],
                device=dev,
            )
            wp.launch(
                _ra_throttle_decide,
                dim=1,
                inputs=[
                    self._ra_batch_bound,
                    self._ra_batch_age,
                    self._ra_pending_counts,
                    self._ra_wait,
                    self._ra_fire,
                ],
                device=dev,
            )
            wp.launch(
                _ra_advance_boundary,
                dim=n,
                inputs=[
                    self._mode_code,
                    dt_outer,
                    window_end,
                    self._dt_inner_init,
                    self._dt_min,
                    eff_dt_max,
                    self._sim_time,
                    self._next_time,
                    self._dt,
                    self._dt_half,
                    self._ideal_dt,
                    self._diverged,
                    self._substeps_frame,
                    self._ra_crossed,
                    self._ra_crossed_any,
                    self._ra_crossings,
                    self._ra_fire,
                ],
                device=dev,
            )

        # Boundary flag for early termination: reset in-body (so the flag reflects
        # this step only, with no host write between replays), then set to 1 if any
        # world is still unfinished. Containment (default) keeps a solve failure
        # per-world -- the controller's reject/shrink/floor-latch absorbs it --
        # and records it in the sticky slot-1 latch + per-world counter; strict
        # mode folds it into the batch-fatal status 2 instead. Run-ahead mode
        # judges "finished" against the per-call scalar target (run-ahead
        # worlds count as finished for the call).
        wp.launch(_boundary_reset, dim=1, inputs=[self._unfinished], device=dev)
        if self._runahead:
            if self._containment:
                wp.launch(
                    mark_unfinished_contained_target,
                    dim=n,
                    inputs=[self._sim_time, self._ra_t_call, self._solve_ok, self._solve_fail_world, self._unfinished],
                    device=dev,
                )
            else:
                wp.launch(
                    mark_unfinished_with_status_target,
                    dim=n,
                    inputs=[self._sim_time, self._ra_t_call, self._solve_ok, self._unfinished],
                    device=dev,
                )
        elif self._containment:
            wp.launch(
                mark_unfinished_contained,
                dim=n,
                inputs=[self._sim_time, self._next_time, self._solve_ok, self._solve_fail_world, self._unfinished],
                device=dev,
            )
        else:
            wp.launch(
                mark_unfinished_with_status,
                dim=n,
                inputs=[self._sim_time, self._next_time, self._solve_ok, self._unfinished],
                device=dev,
            )
        # The max_substeps cap runs LAST so it wins over the boundary-exit check;
        # the kernel preserves a status-2 flag so the cap can never mask a solve
        # failure.
        wp.launch(
            _iters_exhausted_stop,
            dim=1,
            inputs=[self._iteration_count_buf, self._max_substeps, self._unfinished],
            device=dev,
        )

    def _body_graph(self, eff_dt_max: float, dt_outer: float):
        """Return the captured single-substep-body graph, or ``None`` to run eagerly.

        The first frame runs eagerly so any lazy module load completes (a launch that
        triggers one aborts capture); from the second frame on the flat body is captured
        ONCE per ``dt_outer`` and replayed per iteration. On capture failure, capture is
        disabled and the loop falls back to eager launches (correct, just slower).
        """
        if not self._graph_enabled:
            return None
        if not self._graph_warmed:
            self._graph_warmed = True
            return None
        # The contact cadence shapes the captured body (per-attempt keeps the
        # collide inside it), so it is part of the key: a cached graph must
        # never replay the other cadence's launch stream. The LS-compaction
        # switch shapes it too (list-rebuild launches inside the line-search
        # trips), so it is keyed for the same reason. The line-search variant
        # selects entirely different launch streams inside the Newton loop
        # (fused single-launch ladder vs conditional backtracking trips), so
        # it is keyed as well.
        key = (
            round(float(dt_outer), 12),
            self._contact_refresh_per_attempt,
            bool(getattr(self._sap.contact_solve, "_ls_compact", False)),
            bool(getattr(self._sap.contact_solve, "_gemm_reshape", False)),
            # The fused armijo ladder replaces the conditional backtracking
            # trip subgraph with a single-kernel walk, so the flag selects a
            # different launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_ls", False)),
            # The folded alpha-max rung deletes the per-trip trial launch
            # chain and swaps the ladder kernel, so the flag selects a
            # different launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_alphamax", False)),
            # The per-contact pack swaps the bounded pack kernel and adds a
            # per-solve j_flat build launch, so the flag selects a different
            # launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_pack_percontact", False)),
            # The fused update evaluation collapses the per-trip committed-
            # point launch chain into one kernel and deletes the trip-opening
            # hessian projection, so the flag selects a different launch
            # stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_update", False)),
            # The narrow-v3 routing swaps the full-width trip-cadence
            # kernels for their list-indexed launches (fused update via the
            # prepare list, the serial LS direction/init/accumulate chain
            # via the newton list, the world-gated contact scatter), so the
            # flag selects a different launch stream inside the captured
            # solves.
            bool(getattr(self._sap.contact_solve, "_narrow_v3", False)),
            bool(getattr(self._sap.contact_jacobian, "_narrow_v3", False)),
            str(getattr(self._sap, "line_search_variant", "")),
            self._tail_compact,
            self._march_compact,
            self._mc_width,
            self._shared_assembly,
            # The attempt-consistent constitutive scale kernels record inside
            # the captured solves, so the flag selects a different launch
            # stream.
            self._attempt_consistent_r,
            # Construction-constant today (precision is baked into the buffer
            # dtypes and kernel table at __init__), keyed defensively so a
            # future mutable-precision refactor cannot replay the other
            # dtype's launch stream.
            self._solve_precision,
            # The run-ahead march adds the crossing-batched collide+adopt
            # node, the crossing kernel and the scalar-target boundary-exit
            # kernels to the body, so the flag selects a different launch
            # stream. The throttle bound is a captured kernel scalar, so a
            # cached graph must never replay another bound's value.
            self._runahead,
            self._ra_batch_bound if self._runahead else 0,
            self._ra_batch_age if self._runahead else 0,
        )
        graph = self._graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._substep_body(eff_dt_max, dt_outer)
                graph = cap.graph
                self._graph_cache[key] = graph
            except Exception:
                self._graph_enabled = False
                return None
        return graph

    _CONDITIONAL_WARM_BOUNDARIES = 2
    """Boundaries to run on the per-iteration tier before attempting whole-march
    conditional capture: kernel-module loads and any lazy allocation must happen
    OUTSIDE the capture (a conditional body may not record them)."""

    def _external_capture_active(self) -> bool:
        """True when an OUTER CUDA-graph capture is recording on the current stream,
        so the march must record its conditional while-node into that graph instead
        of launching/replaying its own (host reads, including the post-march status
        check, are illegal there -- the outer graph's owner reads ``_unfinished``
        after replay if it needs the converge-or-throw contract)."""
        if not self._is_cuda:
            return False
        try:
            dev = wp.get_device(self.model.device)
            return dev.captures.get(wp.get_stream(dev)) is not None
        except Exception:
            return False

    def _conditional_march_body(self, eff_dt_max: float, dt_outer: float) -> None:
        """One while-node trip: the substep body, then the loop-condition update.

        The update runs AFTER ``_iters_exhausted_stop`` (the last launch of the
        body) so the budget cap and a solve-failure status both close the loop."""
        self._substep_body(eff_dt_max, dt_outer)
        if self._quantile_cutoff > 0 and self._tail_compact:
            wp.launch(
                _march_continue_quantile,
                dim=1,
                inputs=[
                    self._unfinished,
                    self._active_counts,
                    self._quantile_cutoff,
                    self._march_continue,
                    self._cut_gate,
                ],
                device=self.model.device,
            )
            wp.launch(
                _mark_cut_worlds,
                dim=self.model.world_count,
                inputs=[self._cut_gate, self._sim_time, self._next_time, self._boundary_cut],
                device=self.model.device,
            )
        else:
            wp.launch(
                _march_continue_from_status,
                dim=1,
                inputs=[self._unfinished, self._march_continue],
                device=self.model.device,
            )

    def _record_boundary_cuts(self) -> None:
        """Latch worlds abandoned short of the boundary by the quantile stop."""
        if self._quantile_cutoff <= 0:
            return
        wp.launch(
            _mark_cut_worlds,
            dim=self.model.world_count,
            inputs=[self._sim_time, self._next_time, self._boundary_cut],
            device=self.model.device,
        )

    @property
    def boundary_cut_mask(self) -> wp.array:
        """Per-world flag, shape ``[world_count]``, int32, on device: 1 where the
        quantile stop ended the boundary before that world reached it.

        Those worlds hold mid-boundary state that never met the integration
        tolerance, so the consumer must drop them (terminate the environment)
        rather than treat the state as a completed transition. Sticky until
        :meth:`clear_boundary_cuts`. ``None`` when the quantile stop is off, so
        a consumer can tell "nothing was cut" from "cutting is not in play"."""
        if self._quantile_cutoff <= 0:
            return None
        return self._boundary_cut

    def clear_boundary_cuts(self) -> None:
        """Zero the cut mask, after the consumer has acted on it."""
        self._boundary_cut.zero_()

    def _march_conditional(self, eff_dt_max: float, dt_outer: float) -> None:
        """Record the whole march as a device-side loop: seed the loop condition
        open, then ``wp.capture_while`` over the substep body. The seed records
        INSIDE the graph so every replay is self-contained (the body always runs
        at least once per boundary, matching the per-iteration tier)."""
        wp.launch(_march_continue_set, dim=1, inputs=[self._march_continue], device=self.model.device)
        wp.capture_while(self._march_continue, lambda: self._conditional_march_body(eff_dt_max, dt_outer))

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

    def _conditional_graph_key(self, dt_outer: float):
        """Cache key for the whole-march conditional graph: every switch that
        shapes the recorded launch stream (the same determinants as
        :meth:`_body_graph`'s key -- a cached graph must never replay another
        configuration's launch stream)."""
        return (
            round(float(dt_outer), 12),
            self._contact_refresh_per_attempt,
            bool(getattr(self._sap.contact_solve, "_ls_compact", False)),
            bool(getattr(self._sap.contact_solve, "_gemm_reshape", False)),
            # The fused armijo ladder replaces the conditional backtracking
            # trip subgraph with a single-kernel walk, so the flag selects a
            # different launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_ls", False)),
            # The folded alpha-max rung deletes the per-trip trial launch
            # chain and swaps the ladder kernel, so the flag selects a
            # different launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_alphamax", False)),
            # The per-contact pack swaps the bounded pack kernel and adds a
            # per-solve j_flat build launch, so the flag selects a different
            # launch stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_pack_percontact", False)),
            # The fused update evaluation collapses the per-trip committed-
            # point launch chain into one kernel and deletes the trip-opening
            # hessian projection, so the flag selects a different launch
            # stream inside the captured solves.
            bool(getattr(self._sap.contact_solve, "_fused_update", False)),
            # The narrow-v3 routing swaps the full-width trip-cadence
            # kernels for their list-indexed launches (fused update via the
            # prepare list, the serial LS direction/init/accumulate chain
            # via the newton list, the world-gated contact scatter), so the
            # flag selects a different launch stream inside the captured
            # solves.
            bool(getattr(self._sap.contact_solve, "_narrow_v3", False)),
            bool(getattr(self._sap.contact_jacobian, "_narrow_v3", False)),
            str(getattr(self._sap, "line_search_variant", "")),
            self._tail_compact,
            self._march_compact,
            self._mc_width,
            self._shared_assembly,
            # The attempt-consistent constitutive scale kernels record inside
            # the captured solves, so the flag selects a different launch
            # stream.
            self._attempt_consistent_r,
            # The run-ahead march adds the crossing-batched collide+adopt
            # node, the crossing kernel and the scalar-target boundary-exit
            # kernels to the body, so the flag selects a different launch
            # stream. The throttle bound is a captured kernel scalar, so a
            # cached graph must never replay another bound's value.
            self._runahead,
            self._ra_batch_bound if self._runahead else 0,
            self._ra_batch_age if self._runahead else 0,
        )

    def _launch_conditional_march(self, eff_dt_max: float, dt_outer: float) -> bool:
        """Replay (capturing on first use) the whole-march conditional graph.
        Returns False after permanently downgrading on any capture/launch failure."""
        key = self._conditional_graph_key(dt_outer)
        graph = self._conditional_graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._march_conditional(eff_dt_max, dt_outer)
                graph = cap.graph
                self._conditional_graph_cache[key] = graph
            except Exception as exc:
                self._abort_active_capture()
                self._conditional_enabled = False
                self._conditional_graph_cache.clear()
                warnings.warn(
                    f"SolverSAPAdaptive: conditional-march capture failed ({exc}); "
                    "downgrading permanently to per-iteration graph replay.",
                    stacklevel=2,
                )
                return False
        try:
            wp.capture_launch(graph)
        except Exception as exc:
            self._conditional_enabled = False
            self._conditional_graph_cache.clear()
            warnings.warn(
                f"SolverSAPAdaptive: conditional-march launch failed ({exc}); "
                "downgrading permanently to per-iteration graph replay.",
                stacklevel=2,
            )
            return False
        self._conditional_launches += 1
        return True

    # ------------------------------------------------------- failure forensics
    def _failure_dump_world_record(self, w: int, arrays: dict) -> dict:
        """One failing world's record for the forensic dump: raw buffer values
        only; every derived field carries its derivation rule in the JSON so
        the dump stays interpretable without this source."""
        cpw = self._coords_per_world
        dpw = self._dofs_per_world
        ncon_w = int(arrays["ncon"][w])
        phi0_w = arrays["phi0"][w, : max(ncon_w, 0)]
        # Attempted-step reconstruction: the controller's reject path already
        # ran on the failing attempt (solve failure forces err to the
        # divergence threshold), so dt was shrunk by _DRAKE_MIN_SHRINK and
        # ideal_dt holds the unclamped shrunken step; a diverged-latch world
        # (floor / fixed mode) keeps dt untouched instead.
        if bool(arrays["diverged"][w]):
            dt_attempted = float(arrays["dt"][w])
        else:
            dt_attempted = float(arrays["ideal_dt"][w]) / float(np.float32(_DRAKE_MIN_SHRINK))
        return {
            "world": int(w),
            "solve_ok": int(arrays["solve_ok"][w]),
            "last_solve": {
                "converged": int(arrays["converged"][w]),
                "optimality_reached": int(arrays["opt_reached"][w]),
                "cost_reached": int(arrays["cost_reached"][w]),
                "newton_iterations": int(arrays["n_iters"][w]),
                "grad_norm": float(arrays["grad_norm"][w]),
                "p_norm": float(arrays["p_norm"][w]),
                "jc_norm": float(arrays["jc_norm"][w]),
                "opt_tol": float(arrays["opt_tol"][w]),
                "alpha": float(arrays["alpha"][w]),
                "ls_status": int(arrays["ls_status"][w]),
                "ls_iterations_total": int(arrays["ls_total"][w]),
                "cost": float(arrays["cost"][w]),
                "previous_cost": float(arrays["prev_cost"][w]),
            },
            "contact": {
                "count": ncon_w,
                "phi0_min": float(phi0_w.min()) if ncon_w > 0 else None,
                "n_penetrating": int((phi0_w < 0.0).sum()) if ncon_w > 0 else 0,
                "phi0": [float(x) for x in phi0_w[:256]],
            },
            "controller": {
                "dt_post_adapt": float(arrays["dt"][w]),
                "dt_attempted_est": dt_attempted,
                "ideal_dt": float(arrays["ideal_dt"][w]),
                "dt_ceiling": float(arrays["dt_ceiling"][w]),
                "sim_time": float(arrays["sim_time"][w]),
                "next_time": float(arrays["next_time"][w]),
                "boundary_remaining": float(arrays["next_time"][w] - arrays["sim_time"][w]),
                "diverged": bool(arrays["diverged"][w]),
                "limited": int(arrays["limited"][w]),
                "consec_rej": int(arrays["consec_rej"][w]),
                "last_error": float(arrays["last_error"][w]),
            },
            "joint_q": [float(x) for x in arrays["joint_q"][w * cpw : (w + 1) * cpw]],
            "joint_qd": [float(x) for x in arrays["joint_qd"][w * dpw : (w + 1) * dpw]],
        }

    def _note_contained_failures(self) -> None:
        """Rare-path host accounting for a boundary whose march contained at
        least one per-world solve failure (the sticky slot-1 latch fired).

        Reads the per-world event counters, updates the cumulative host
        counters, and emits a rate-limited warning naming the world count.
        Runs strictly post-march with no capture active, so the host read is
        legal; the hot loop itself never syncs for this. The failure's
        dynamics were already handled on device: the failing attempt was
        rejected (or the world floor-latched ``diverged``, reported via
        :attr:`diverged` for the consumer's reset/termination path), so the
        unconverged result was never consumed.
        """
        counts = self._solve_fail_world.numpy().astype(np.int64)
        total = int(counts.sum())
        new_events = total - self._solve_failure_events
        new_worlds = int((counts > self._fail_world_prev).sum())
        self._solve_failure_events = total
        self._fail_world_prev = counts
        self._solve_failure_boundaries += 1
        if self._fail_warn_emitted < 5 or self._solve_failure_boundaries % 100 == 0:
            self._fail_warn_emitted += 1
            warnings.warn(
                "SolverSAPAdaptive: contained inner-solve failure(s) this boundary: "
                f"{new_worlds} world(s), {new_events} rejected attempt(s) "
                f"(cumulative {self._solve_failure_events} events over "
                f"{self._solve_failure_boundaries} failing boundaries). Failing worlds "
                "retry at a shrunken dt and latch diverged at the dt floor (read "
                "solver.diverged after the boundary call to reset/terminate them); "
                "no unconverged result is ever committed. "
                "NEWTON_SAP_CONTAINMENT=0 restores strict converge-or-throw.",
                stacklevel=3,
            )

    def _raise_if_solve_failed(self, status: int) -> None:
        """Converge-or-throw (strict mode): a status-2 boundary word means some
        world's inner solve did not reach its optimality tolerance and its
        result must never be consumed. Containment mode never folds status 2
        (the failure stays per-world), so this is a no-op there."""
        if status >= 2:
            raise RuntimeError(
                "SolverSAPAdaptive inner SAP solve failed to converge to "
                f"optimality_rel_tol={self._optimality_rel_tol:.3e}."
            )

    def _run_substep_loop(self, eff_dt_max: float, dt_outer: float) -> None:
        """March substeps until every world reaches its boundary, capped at ``max_substeps``.

        Tiers (first applicable wins):
        1. Conditional mode + outer capture recording -> contribute the while-node
           (no host reads are possible there; see :meth:`_external_capture_active`).
        2. Conditional mode, warmed -> replay the whole-march conditional graph,
           then ONE 4-byte post-march status read (the converge-or-throw check).
        3. Default / warmup / post-failure: replay the per-iteration body graph
           (or run eagerly while warming / if capture fails), reading the 4-byte
           ``_unfinished`` status between iterations to stop as soon as all
           worlds land instead of grinding fixed no-op substeps.
        """
        if self._conditional_enabled:
            if self._external_capture_active():
                self._march_conditional(eff_dt_max, dt_outer)
                if self._march_iters is not None:
                    wp.copy(self._march_iters, self._iteration_count_buf)
                return
            if self._conditional_warm_boundaries >= self._CONDITIONAL_WARM_BOUNDARIES:
                if self._launch_conditional_march(eff_dt_max, dt_outer):
                    # ONE post-march read of the 2-int32 status word: slot 0 is
                    # the boundary status (strict converge-or-throw check),
                    # slot 1 the sticky containment failure latch (rare path).
                    st = self._unfinished.numpy()
                    self._raise_if_solve_failed(int(st[0]))
                    self._raise_if_march_compact_poisoned()
                    if self._containment and int(st[1]) != 0:
                        self._note_contained_failures()
                    if self._march_iters is not None:
                        wp.copy(self._march_iters, self._iteration_count_buf)
                    return
            else:
                self._conditional_warm_boundaries += 1

        graph = self._body_graph(eff_dt_max, dt_outer)
        fail_latched = 0
        for _ in range(self._max_substeps):
            if graph is not None:
                try:
                    wp.capture_launch(graph)
                except Exception:
                    # cudaGraphInstantiate can OOM here (outside capture); drop it and finish
                    # this frame eagerly so the boundary still advances.
                    self._graph_cache.clear()
                    self._graph_enabled = False
                    graph = None
                    self._substep_body(eff_dt_max, dt_outer)
            else:
                self._substep_body(eff_dt_max, dt_outer)
            st = self._unfinished.numpy()
            status = int(st[0])
            # Slot 1 is sticky across the boundary's iterations, so the last
            # read carries every failure this march contained.
            fail_latched = int(st[1])
            self._raise_if_solve_failed(status)
            if status == 0:
                break
        self._raise_if_march_compact_poisoned()
        if self._containment and fail_latched != 0:
            self._note_contained_failures()

        if self._march_iters is not None:
            wp.copy(self._march_iters, self._iteration_count_buf)

    # ------------------------------------------------------------------- integrate
    def integrate(self, state, control, dt_outer: float):
        """Advance every world by exactly ``dt_outer`` of sim time on the GPU.

        The integrator owns WHEN and HOW-BIG the inner steps are (per-world adaptive dt);
        it calls :meth:`substep` for the physics of one step. ``state`` (Newton State) is
        read and written in place and returned.
        """
        device = self.model.device
        n = self._world_count
        dt_outer = float(dt_outer)
        eff_dt_max = min(self._dt_max, dt_outer)

        self._sap_control = sap_control_from_newton(control, target_remap=self._target_remap)

        # Load the incoming Newton state into the internal working buffer.
        self._copy_state(self._state_cur, sap_state_from_newton(state))

        # Open the frame. Per-boundary (default): rebase clocks (Fix B), set
        # the new boundary, seed per-world dt, clear per-frame work counters
        # and the divergence latch. Run-ahead: the same opening happens once
        # per WINDOW (all worlds sit batch-synchronized at the previous
        # window's end there); mid-window calls only advance the device call
        # target and re-seat any world the manager reset since the last call
        # -- every other world already carries its own run-ahead clock and
        # bookkeeping from the crossing kernel.
        if self._runahead:
            targets = self._ra_targets(dt_outer)
            pos = (self._ra_call_idx + self._ra_phase) % self._runahead_window
            self._ra_window_pos = pos
            t_call = float(targets[pos + 1])
            self._ra_t_call_host = t_call
            self._ra_t_call.fill_(t_call)
            if pos == 0:
                wp.launch(_open_frame, dim=n, inputs=[self._sim_time, self._next_time, dt_outer], device=device)
                wp.launch(
                    _seed_dt,
                    dim=n,
                    inputs=[
                        self._mode_code,
                        self._ideal_dt,
                        self._dt_inner_init,
                        self._dt_min,
                        eff_dt_max,
                        self._dt,
                        self._dt_half,
                    ],
                    device=device,
                )
                self._substeps_frame.zero_()
                self._diverged.zero_()
                # The window-open full collide+adopt below covers every world;
                # disarm any flags left by final-iteration crossings.
                self._ra_crossed.zero_()
                self._ra_crossed_any.zero_()
            else:
                wp.launch(
                    _ra_resync_reset_worlds,
                    dim=n,
                    inputs=[
                        float(targets[pos]),
                        t_call,
                        self._mode_code,
                        self._dt_inner_init,
                        self._dt_min,
                        eff_dt_max,
                        self._sim_time,
                        self._next_time,
                        self._dt,
                        self._dt_half,
                        self._ideal_dt,
                        self._substeps_frame,
                        self._ra_crossed,
                        self._ra_crossed_any,
                    ],
                    device=device,
                )
        else:
            wp.launch(_open_frame, dim=n, inputs=[self._sim_time, self._next_time, dt_outer], device=device)
            wp.launch(
                _seed_dt,
                dim=n,
                inputs=[
                    self._mode_code,
                    self._ideal_dt,
                    self._dt_inner_init,
                    self._dt_min,
                    eff_dt_max,
                    self._dt,
                    self._dt_half,
                ],
                device=device,
            )
            self._substeps_frame.zero_()
            self._diverged.zero_()
        self._iteration_count_buf.fill_(0)
        self._guard_hits.fill_(0)
        if self._containment:
            # Clear the sticky slot-1 failure latch for the new boundary (slot 0
            # is reset in-body every iteration regardless). A device fill, so it
            # records correctly when an outer capture wraps this boundary.
            self._unfinished.zero_()
            self._containment_boundaries += 1
        if self._march_log_path is not None:
            self._reject_count_buf.fill_(0)

        # Boundary contact pass (default cadence, mirroring
        # SolverMuJoCoAdaptive's once-per-boundary contact injection): ONE
        # CollisionPipeline pass at the boundary-entry state; every attempt
        # reuses this contact SET, re-anchored to its own eval state by the
        # per-attempt Jacobian rebuild. Runs OUTSIDE the captured substep
        # body by construction. Per-attempt mode collides inside the body
        # instead (diagnostic-only). Run-ahead: the full pass runs at the
        # WINDOW open only (every world is crossing into its first boundary
        # there); interior boundary collides are the crossing-batched
        # conditional nodes inside the march body.
        if self._runahead:
            if self._ra_window_pos == 0:
                self._collide_from(self._state_cur)
                self._sap.contact_jacobian.adopt_contact_set(self._sap_contacts, self._ra_all_worlds)
        elif not self._contact_refresh_per_attempt:
            self._collide_from(self._state_cur)
            if self._sap.contact_jacobian.det_slots_external:
                self._sap.contact_jacobian.compute_deterministic_contact_slots(self._sap_contacts)

        # Masked substep march: the loop stops as soon as every world reaches
        # its boundary (one 4-byte flag read per iteration; count is scene-dependent).
        self._run_substep_loop(eff_dt_max, dt_outer)

        # Once per boundary, outside the captured body: no per-iteration cost.
        if self._dt_hist_trunc is not None:
            wp.launch(
                _count_boundary_truncation,
                dim=1,
                inputs=[self._march_iters, self._max_substeps, self._dt_hist_trunc],
                device=device,
            )
            wp.launch(
                _count_unfinished_worlds,
                dim=n,
                inputs=[self._sim_time, self._next_time, self._dt_hist_trunc],
                device=device,
            )

        # Bound truncation damage BEFORE the next boundary consumes the carried
        # state; device-only work, legal graphs on or off. A completed march
        # makes this a no-op. Run-ahead: debt is judged against the CALL
        # target (a run-ahead world sits mid-boundary legitimately and must
        # not have its controller reissued); the per-world carry bound is
        # unchanged.
        if self._runahead:
            wp.launch(
                _debt_guard_target,
                dim=n,
                inputs=[
                    self._sim_time,
                    self._next_time,
                    self._ra_t_call_host,
                    dt_outer,
                    self._dt_inner_init,
                    self._ceiling_init,
                    self._ideal_dt,
                    self._dt_ceiling,
                    self._consec_rej,
                    self._guard_hits,
                ],
                device=device,
            )
            self._ra_call_idx += 1
        else:
            wp.launch(
                _debt_guard,
                dim=n,
                inputs=[
                    self._sim_time,
                    self._next_time,
                    dt_outer,
                    self._dt_inner_init,
                    self._ceiling_init,
                    self._ideal_dt,
                    self._dt_ceiling,
                    self._consec_rej,
                    self._guard_hits,
                ],
                device=device,
            )

        # Opt-in telemetry readout: host reads of post-march state, which are only
        # legal outside an active capture -- hence gated, never unconditional. Runs
        # after the guard, so resid reflects the bounded carry while n_guard reports
        # the worlds it touched.
        if self._march_log_path is not None:
            self._log_march_boundary()

        # Write the advanced state back into the Newton state.
        self._copy_state(sap_state_from_newton(state), self._state_cur)
        return state

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
                "eworld,qmax_i,qmax,ncon,cum_acc,ra_cross,ra_fires\n"
            )
        iters = int(self._iteration_count_buf.numpy()[0])
        cum = int(self._cum_iters.numpy()[0])
        ideal = self._ideal_dt.numpy()
        resid = self._next_time.numpy() - self._sim_time.numpy()
        rejects = int(self._reject_count_buf.numpy()[0])
        err_arr = self._last_error.numpy()
        err_max = float(err_arr.max())
        # Name the worst world's largest-magnitude coordinate: which joint_q
        # slot, and how big. Identifies WHAT the error norm is reading when
        # err_max pins at a grid constant.
        eworld = int(err_arr.argmax())
        q_abs = np.abs(
            self._state_cur.joint_q.numpy()[eworld * self._coords_per_world : (eworld + 1) * self._coords_per_world]
        )
        qmax_i = int(q_abs.argmax())
        qmax = float(q_abs[qmax_i])
        # The solver-side contact count is a fixed capacity, so the ncon column
        # reports the collision pipeline's live rigid-contact count instead.
        ncon = int(self._contacts.rigid_contact_count.numpy()[0])
        n_debt = int((resid > 0.0).sum())
        n_subfloor = int((ideal < self._dt_min).sum())
        n_guard = int(self._guard_hits.numpy()[0])
        # Demand + run-ahead engagement, cumulative: cum_acc is the accepted
        # per-world substep count (the schedule-invariant work axis next to
        # the batch-iteration cum column); ra_cross/ra_fires are the crossing
        # and consumed-adopt-batch counters (zero outside run-ahead mode --
        # the buffers exist unconditionally, so an OFF read is a real
        # observation).
        cum_acc = int(self._cum_accepted.numpy()[0])
        ra_cross = int(self._ra_crossings.numpy()[0])
        ra_fires = int(self._ra_adopts.numpy()[0])
        self._march_log_file.write(
            f"{self._march_log_boundary},{iters},{cum},"
            f"{ideal.min():.6e},{ideal.mean():.6e},{ideal.max():.6e},"
            f"{resid.min():.6e},{resid.max():.6e},"
            f"{rejects},{err_max:.6e},{n_debt},{n_subfloor},{n_guard},"
            f"{eworld},{qmax_i},{qmax:.6e},{ncon},{cum_acc},{ra_cross},{ra_fires}\n"
        )
        self._march_log_boundary += 1
        if self._dt_hist is not None and self._march_log_boundary % self._march_log_hist_every == 0:
            self._march_log_file.write(f"HIST {self.dt_histogram_stats()}\n")

    # ------------------------------------------------------------------- step
    def step(self, state_in, state_out, control, contacts=None, dt=None, apply_forces=None):
        """Newton-signature boundary call ``(state_in, state_out, control, contacts, dt)``.

        Thin adapter over :meth:`integrate`: ``state_in`` is advanced in place by ``dt``
        (= ``dt_outer``) and returned; ``state_out`` is accepted for signature uniformity
        and returned unchanged. ``contacts`` is accepted but UNUSED; the integrator builds
        its internal contact set once per boundary at the entry state (default; each
        attempt re-anchors it via the Jacobian rebuild) or, in the diagnostic
        per-attempt mode (``NEWTON_ADAPTIVE_CONTACT_REFRESH=attempt``), at each
        attempt's start and midpoint state.
        """
        if dt is None:
            raise ValueError("SolverSAPAdaptive.step requires dt (the outer boundary period).")
        if apply_forces is not None:
            apply_forces(state_in)
        self.integrate(state_in, control, float(dt))
        return state_in, state_out

    def step_dt(self, dt_outer: float, state_0, state_1, control, apply_forces=None):
        """Backward-compatible alias for :meth:`step` (legacy ``(dt, s0, s1, control)`` order)."""
        return self.step(state_0, state_1, control, None, dt_outer, apply_forces=apply_forces)

    # ------------------------------------------------------------------- reset
    def reset(self, state, world_mask: wp.array | None = None, flags=0) -> None:
        """Restore per-world controller state for reset worlds and clear the SAP warm-start."""
        mask = self._full_world_mask if world_mask is None else world_mask
        self._sap.reset_runtime_state()
        wp.launch(
            _reset_worlds,
            dim=self._world_count,
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
                self._ceiling_init,
            ],
            device=self.model.device,
        )
