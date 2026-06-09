# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Solver-agnostic Warp kernels for the adaptive step wrapper.

Lifted from newton/_src/solvers/mujoco/solver_mujoco_cenic.py — see that
file's lines 20-284 for the originals. These operate on plain Newton State
arrays and have no MuJoCo coupling.
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
def _inf_norm_state_error_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    coords_per_world: int,
    error_out: wp.array(dtype=wp.float32),
):
    """Position-only inf-norm error between full-step and doubled half-step.

    Error = ``max_i |Δq_i|`` across the world's joint coordinates, i.e. the
    unweighted ``|| q - q̂ ||_∞`` (the error-norm weighting is the identity).
    Diverged sims get error = 1e10.
    """
    world = wp.tid()
    q_start = world * coords_per_world

    max_err = float(0.0)
    for i in range(coords_per_world):
        d = wp.abs(joint_q_double[q_start + i] - joint_q_full[q_start + i])
        max_err = wp.max(max_err, d)

    if wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)

    error_out[world] = max_err


@wp.kernel
def _calc_adjusted_step(
    err: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    tol: float,
    dt_min: float,
    safety: float,
    min_shrink: float,
    max_grow: float,
    hyst_high: float,
    hyst_low: float,
):
    """Per-world Drake CalcAdjustedStepSize for step doubling (err_order=2).

    dt_max clamping is deferred to _apply_dt_cap so ideal_dt is preserved.
    The 5 controller constants (safety, min_shrink, max_grow, hyst_high,
    hyst_low) are passed as per-launch float scalars from ControllerConfig,
    replacing the former wp.constant values.
    """
    world = wp.tid()
    e = err[world]
    step = dt[world]

    # Boundary-stalled worlds (dt clamped to 0): accept without touching ideal_dt
    # so the next interval inherits a good dt instead of ramping from dt_min.
    if step <= wp.float32(0.0):
        accepted[world] = True
        return

    if wp.isnan(e) or wp.isinf(e):
        accepted[world] = False
        ideal_dt[world] = min_shrink * step
        return

    # At the floor we must accept to avoid stalling.
    if step <= dt_min * wp.float32(1.001) and e > tol:
        accepted[world] = True
        ideal_dt[world] = dt_min
        return

    new_step = safety * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))

    # Symmetric deadband (paper Alg 1): keep dt unchanged when new_step lands
    # in [k_Low * dt, k_High * dt]. Prevents dt thrash from small error spikes
    # (lower edge) and suppresses tiny grows (upper edge).
    if new_step > hyst_low * step and new_step < hyst_high * step:
        new_step = step

    new_step = wp.clamp(new_step, min_shrink * step, max_grow * step)

    accepted[world] = e <= tol or new_step >= step
    ideal_dt[world] = new_step


@wp.kernel
def _advance_sim_time(
    sim_time: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    error: wp.array(dtype=wp.float32),
    accepted_error: wp.array(dtype=wp.float32),
    accepted_error_max: wp.array(dtype=wp.float32),
    accepted_dt_max: wp.array(dtype=wp.float32),
):
    """Advance sim_time and snapshot per-boundary error/dt for accepted worlds.

    ``accepted_error`` keeps the LAST accepted step's error (legacy). The last
    accepted step of a boundary is the fill-to-target substep clamped by
    ``_clamp_dt_to_boundary`` to a tiny dt, so its error is tiny and unrepresentative.
    ``accepted_error_max`` / ``accepted_dt_max`` instead keep the running max over
    the boundary's REAL accepted steps (dt > 0): the worst per-step error actually
    committed and the largest step size ridden. These are reset each boundary.
    """
    i = wp.tid()
    if accepted[i]:
        sim_time[i] = sim_time[i] + dt[i]
        accepted_error[i] = error[i]
        if dt[i] > wp.float32(1.0e-9):
            accepted_error_max[i] = wp.max(accepted_error_max[i], error[i])
            accepted_dt_max[i] = wp.max(accepted_dt_max[i], dt[i])


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


@wp.kernel
def _update_active_mask(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    world_active: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    world_active[i] = sim_time[i] < next_time[i]


@wp.kernel
def _scalar_max_dt_reset(
    out: wp.array(dtype=wp.float32),
):
    """Reset output to a very small number before _scalar_max_dt runs."""
    out[0] = wp.float32(-1.0e30)


@wp.kernel
def _scalar_max_dt(
    dt: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    """Parallel max via atomic_max. Launch with dim=N. Each thread writes
    its world's dt to out[0] via atomic max. Caller must call
    _scalar_max_dt_reset first to initialize out[0] to -inf."""
    tid = wp.tid()
    wp.atomic_max(out, 0, dt[tid])


@wp.kernel
def _inf_norm_q_qd_kernel(
    q_full: wp.array(dtype=wp.float32),
    q_double: wp.array(dtype=wp.float32),
    qd_full: wp.array(dtype=wp.float32),
    qd_double: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    coords_per_world: int,
    dofs_per_world: int,
    last_error: wp.array(dtype=wp.float32),
):
    """Unweighted L-inf norm on joint_q AND dt*joint_qd combined.

    Captures position-equivalent error from velocity divergence -- useful for
    solvers (XPBD, SemiImplicit) whose position step is too smooth for a
    pure-q norm to detect step-doubling differences. The error-norm weighting
    is the identity.
    """
    world = wp.tid()
    err = wp.float32(0.0)
    dt_w = dt[world]

    # q contribution
    q_base = world * coords_per_world
    for i in range(coords_per_world):
        d = wp.abs(q_full[q_base + i] - q_double[q_base + i])
        err = wp.max(err, d)

    # dt * qd contribution (position-equivalent)
    qd_base = world * dofs_per_world
    for i in range(dofs_per_world):
        d = wp.abs(qd_full[qd_base + i] - qd_double[qd_base + i])
        err = wp.max(err, dt_w * d)

    last_error[world] = err


@wp.kernel
def _inf_norm_body_kernel(
    body_q_full: wp.array(dtype=wp.transform),
    body_q_double: wp.array(dtype=wp.transform),
    body_qd_full: wp.array(dtype=wp.spatial_vector),
    body_qd_double: wp.array(dtype=wp.spatial_vector),
    dt: wp.array(dtype=wp.float32),
    bodies_per_world: int,
    last_error: wp.array(dtype=wp.float32),
):
    """Weighted L-inf norm on body transforms + spatial velocities.

    Used by maximal-coord solvers (XPBD, SemiImplicit). Their canonical state
    is body_q (wp.transform [px,py,pz,qx,qy,qz,qw]) and body_qd
    (wp.spatial_vector [wx,wy,wz,vx,vy,vz]); joint_q/joint_qd are stale because
    the solvers don't write them back. Using joint_q here would always read 0.
    """
    world = wp.tid()
    err = wp.float32(0.0)
    dt_w = dt[world]

    base = world * bodies_per_world
    for i in range(bodies_per_world):
        tf_full = body_q_full[base + i]
        tf_double = body_q_double[base + i]
        for j in range(7):
            err = wp.max(err, wp.abs(tf_full[j] - tf_double[j]))

        v_full = body_qd_full[base + i]
        v_double = body_qd_double[base + i]
        for j in range(6):
            err = wp.max(err, dt_w * wp.abs(v_full[j] - v_double[j]))

    last_error[world] = err
