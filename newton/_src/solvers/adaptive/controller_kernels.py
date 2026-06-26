# SPDX-License-Identifier: Apache-2.0
"""Leaf ``@wp.kernel`` primitives for the adaptive step-doubling controller.

Extracted verbatim from ``solver_mujoco_adaptive.py`` (lines 24-494) so the
controller logic is single-sourced and shared by both ``SolverMuJoCoAdaptive``
and ``SolverSAPAdaptive``. These are pure free functions at module scope that
touch no solver-internal symbol; they carry all the subtle logic:

  * Fix A  -> ``_calc_adjusted_step`` (per-world Drake step + NaN/floor guard)
  * Fix B  -> ``_rebase_time`` (float32 time-rebase)
  * Fix C  -> ``_reset_worlds`` (per-world controller reset on env reset)
  * metric -> ``_inf_norm_state_error_kernel`` (Kurtz & Castro Sec. V-E)
  * rollback-> ``_select_{float,transform,spatial_vector}_kernel``
  * even   -> ``_select_even_dt`` / ``_set_uniform_even_dt`` / ``_calc_even_step``

Moving them here is a pure relocation: ``solver_mujoco_adaptive`` re-imports
every name, so its existing unit tests (which import these symbols from that
module) keep resolving them unchanged.
"""

import warp as wp


@wp.kernel
def _apply_dt_cap(
    ideal_dt: wp.array(dtype=wp.float32),
    dt_min: float,
    dt_max: float,
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
):
    """Clamp ideal_dt to [dt_min, dt_max], preserving ideal_dt for controller recovery."""
    i = wp.tid()
    actual = wp.clamp(ideal_dt[i], dt_min, dt_max)
    dt[i] = actual
    dt_half[i] = actual * wp.float32(0.5)


@wp.kernel
def _select_even_dt(
    ideal_dt: wp.array(dtype=wp.float32),
    dt_outer: float,
    dt_min: float,
    dt_max: float,
    max_substeps: int,
    N_out: wp.array(dtype=wp.int32),
    dt_out: wp.array(dtype=wp.float32),
    dt_half_out: wp.array(dtype=wp.float32),
):
    """Even-tiling (Fix 4) N-selection: N = clamp(ceil(dt_outer / clamp(ideal_dt, dt_min, dt_max)),
    1, max_substeps), uniform inner dt = dt_outer / N. Chosen ONCE per outer interval (from the
    carried controller ideal_dt) so the control step tiles evenly with no ragged remainder, while
    N still adapts across control steps.

    ``max_substeps`` is a HARD upper bound on N. Without it a world whose ideal_dt collapses to the
    dt_min floor demands N = ceil(dt_outer / dt_min) ~ 1e4 substeps; because the per-world fixed
    loop runs max_i(N) substeps for the whole batch, one such world would force ~1e4 eager kernel
    launches per frame (CPU-launch-bound -> effective hang). The cap bounds worst-case work and
    lets a capped world self-recover (its capped dt is small enough that the error -- hence next
    interval's N -- drops back down)."""
    i = wp.tid()
    d = wp.clamp(ideal_dt[i], dt_min, dt_max)
    n = wp.max(wp.int32(1), wp.int32(wp.ceil(dt_outer / d)))
    n = wp.min(n, max_substeps)
    N_out[i] = n
    step = dt_outer / wp.float32(n)
    dt_out[i] = step
    dt_half_out[i] = step * wp.float32(0.5)


@wp.kernel
def _set_uniform_even_dt(
    n_shared: int,
    dt_outer: float,
    N_out: wp.array(dtype=wp.int32),
    dt_out: wp.array(dtype=wp.float32),
    dt_half_out: wp.array(dtype=wp.float32),
):
    """Global even mode: overwrite every world with the shared worst-case N = max_i N_i."""
    i = wp.tid()
    N_out[i] = n_shared
    step = dt_outer / wp.float32(n_shared)
    dt_out[i] = step
    dt_half_out[i] = step * wp.float32(0.5)


@wp.kernel
def _inf_norm_state_error_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    state_scale: wp.array2d(dtype=wp.float32),
    coords_per_world: int,
    error_out: wp.array(dtype=wp.float32),
):
    """Adaptive-controller accuracy metric (Kurtz & Castro, Sec. V-E)::

        e^{n+1} = || S (q^{n+1} - q̂^{n+1}) ||_∞

    Position-only inf-norm of the difference between the doubled half-step ``q`` and the
    full step ``q̂``, scaled by the diagonal ``S`` that "maps each component to a
    dimensionless unit." Velocity and contact impulses are excluded from the controller,
    exactly as the paper specifies. The paper gives no formula for ``S`` and mandates NO
    mass weighting, clipping, or normalization ("S can be estimated from knowledge of
    coordinate types or specified by expert users"); here ``S = identity`` per PI
    directive. Diverged sims get error = 1e10.
    """
    world = wp.tid()
    q_start = world * coords_per_world

    max_err = float(0.0)
    for i in range(coords_per_world):
        d = wp.abs(joint_q_double[q_start + i] - joint_q_full[q_start + i])
        max_err = wp.max(max_err, state_scale[world, i] * d)

    if wp.isnan(max_err) or wp.isinf(max_err):
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
    err: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    commit: wp.array(dtype=wp.bool),
    diverged: wp.array(dtype=wp.bool),
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
        if is_diverged:
            # Refuse to write the NaN/garbage state: advance time so the boundary
            # loop terminates, but HOLD the last good state and flag the world so
            # the env resets it. This is the fix for NaN propagation out of the solver.
            accepted[world] = True
            commit[world] = False
            diverged[world] = True
            ideal_dt[world] = dt_min
            return
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
def _calc_even_step(
    err: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    commit: wp.array(dtype=wp.bool),
    diverged: wp.array(dtype=wp.bool),
    tol: float,
    dt_min: float,
    divergence_threshold: float,
):
    """Even-tiling controller (Fix 4): NO within-interval retry.

    Always advances (``accepted=True``) so the fixed N-substep loop lands deterministically.
    ``commit`` keeps Fix A's NaN guard (hold last-good on non-finite); ``diverged`` keeps A's
    floor latch. ``ideal_dt`` is still updated by the Drake formula, but ONLY to size the NEXT
    interval's substep count N -- ``dt`` is held at ``dt_outer/N`` for the whole interval.
    """
    world = wp.tid()
    e = err[world]
    step = dt[world]
    is_diverged = wp.isnan(e) or wp.isinf(e) or e >= divergence_threshold

    # Boundary-stalled world (dt clamped to 0 because it already reached next_time): no-op.
    if step <= wp.float32(0.0):
        accepted[world] = True
        commit[world] = True
        return

    if is_diverged:
        # Refuse to write the non-finite state (A's NaN guard); still advance time so the
        # fixed-count loop stays in lockstep. Latch divergence at the floor.
        accepted[world] = True
        commit[world] = False
        if step <= dt_min * wp.float32(1.001):
            diverged[world] = True
            ideal_dt[world] = dt_min
        else:
            ideal_dt[world] = _DRAKE_MIN_SHRINK * step
        return

    # Finite: commit and advance. Update ideal_dt (Drake) to size the NEXT interval's N.
    new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
    if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
        new_step = step
    new_step = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)
    accepted[world] = True
    commit[world] = True
    ideal_dt[world] = new_step


@wp.kernel
def _advance_sim_time(
    sim_time: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    error: wp.array(dtype=wp.float32),
    accepted_error: wp.array(dtype=wp.float32),
):
    """Advance sim_time[i] by dt[i] and snapshot error for accepted worlds only."""
    i = wp.tid()
    if accepted[i]:
        sim_time[i] = sim_time[i] + dt[i]
        accepted_error[i] = error[i]


@wp.kernel
def _finish_diverged_worlds(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    diverged: wp.array(dtype=wp.bool),
):
    """Jump diverged worlds straight to their boundary target.

    A world flagged diverged at the floor holds its last good state; stepping it
    again would just re-diverge, so finish its outer interval in one shot. This
    keeps the boundary loop from grinding ``remaining / dt_min`` extra iterations
    on a world that cannot make progress.
    """
    i = wp.tid()
    if diverged[i]:
        sim_time[i] = next_time[i]


@wp.kernel
def _reset_worlds(
    mask: wp.array(dtype=wp.bool),
    dt_init: float,
    ideal_dt: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    diverged: wp.array(dtype=wp.bool),
    accepted: wp.array(dtype=wp.bool),
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
def _select_float_kernel(
    candidate: wp.array(dtype=wp.float32),
    fallback: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.float32),
):
    """Select candidate for accepted worlds, fallback for rejected worlds."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _select_transform_kernel(
    candidate: wp.array(dtype=wp.transform),
    fallback: wp.array(dtype=wp.transform),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.transform),
):
    """Select body pose from accepted or fallback state."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _select_spatial_vector_kernel(
    candidate: wp.array(dtype=wp.spatial_vector),
    fallback: wp.array(dtype=wp.spatial_vector),
    accepted: wp.array(dtype=wp.bool),
    stride: int,
    out: wp.array(dtype=wp.spatial_vector),
):
    """Select body velocity from accepted or fallback state."""
    i = wp.tid()
    world = i // stride
    if accepted[world]:
        out[i] = candidate[i]
    else:
        out[i] = fallback[i]


@wp.kernel
def _boundary_reset(flag: wp.array(dtype=wp.int32)):
    """Set flag[0] = 0 (assume all worlds reached the boundary)."""
    flag[0] = 0


@wp.kernel
def _boundary_check(
    sim_time: wp.array(dtype=wp.float32),
    target: wp.array(dtype=wp.float32),
    flag: wp.array(dtype=wp.int32),
):
    """Set flag to 1 if any world has not yet reached target."""
    i = wp.tid()
    if sim_time[i] < target[i]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def _boundary_advance(arr: wp.array(dtype=wp.float32), delta: float):
    """Increment arr[i] by delta."""
    i = wp.tid()
    arr[i] = arr[i] + delta


@wp.kernel
def _rebase_time(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
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
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
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
def _iter_count_increment(count: wp.array(dtype=wp.int32)):
    """Increment iteration counter (dim=1, single thread)."""
    count[0] = count[0] + 1


@wp.kernel
def _status_sentinel_reset(out: wp.array(dtype=wp.float32)):
    """Reset 6-element summary buffer: [min_sim_time, max_sim_time, max_error, accept_count, min_dt, max_dt]."""
    out[0] = float(1.0e38)
    out[1] = float(0.0)
    out[2] = float(0.0)
    out[3] = float(0.0)
    out[4] = float(1.0e38)
    out[5] = float(0.0)


@wp.kernel
def _reset_error_scalar(out: wp.array(dtype=wp.float32)):
    out[0] = wp.float32(0.0)


@wp.kernel
def _reduce_max_error(src: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    wp.atomic_max(out, 0, src[i])


@wp.kernel
def _broadcast_error(scalar: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i] = scalar[0]


@wp.kernel
def _status_summary_kernel(
    sim_time: wp.array(dtype=wp.float32),
    last_error: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.float32),
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
