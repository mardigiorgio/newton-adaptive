# SPDX-License-Identifier: Apache-2.0
"""Per-world error-controlled (step-doubling) SAP solver: ``SolverSAPAdaptive``.

This is the CENIC integrator for the convex SAP contact solver. The manager hands
it a state and a control boundary period ``dt_outer``; it advances every world to
that boundary **entirely on the GPU** using a TRUE per-world adaptive step and
returns the advanced state. The returned state is more accurate than a fixed step
(local error is controlled, not whatever the step lands on), so the RL ``(s, a, s')``
transitions are faithful to the real dynamics and the policy transfers to hardware.

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
        substep(state_cur -> mid,    dt/2)              # (feedforward/adaptive only: step-doubling)
        substep(mid       -> double, dt/2)
        err = infnorm(full, double)                     # per-world local error estimate
        adapt_dt(err, ...):                             # ALL per-thread, in-kernel (data branches):
            DONE   (sim_time>=next_time): no-op
            ACCEPT (err<=tol):  state_cur=double; sim_time+=dt; grow dt
            REJECT (err>tol):   hold state; shrink dt; retry next iteration
        if no world is unfinished:  break              # ONE 4-byte host flag read per iteration
    write state_cur back

The ONLY host sync in the step path is that single 4-byte boundary-flag read per
iteration (typically ~3 iterations/frame): it lets the loop stop as soon as every world
lands instead of grinding a fixed count of wasted no-op substeps (a ``dt=0`` no-op still
runs the full batched SAP solve, so wasted iterations are NOT free). Reject is a masked
state-hold (a data branch), not control flow. The inner SAP solve is run CONVERGENT (to
``optimality_rel_tol = max(kappa*tol, 1e-8)``, kappa=1e-3; CENIC Sec. VI-B) so its residual
cannot pollute the step-doubling error estimate. The substep body -- including the solve's
``wp.capture_while`` -- is captured ONCE and replayed per iteration: capturing ONE such body
is the intended use of conditional graph nodes (the old >=1024-env SIGABRT was from capturing
3N of them in the old per-N loop), validated clean to 4096 envs.

Three modes (one machine; mode only changes how ``dt`` evolves, set per solver):

  * ``"fixed"``      -- constant dt, error control off; commits the single full step
    (the baseline being beaten on accuracy). Skips the doubling it does not use.
  * ``"feedforward"`` -- dt set once per frame from the carried controller estimate
    (built from last frame's error), held constant within the frame; always accepts.
  * ``"adaptive"``   -- dt grows on accept / shrinks on reject WITHIN the frame from
    the step-doubling error, targeting local error <= tol per step per world.

This file is self-contained: every controller kernel is inlined below (no shared
``adaptive.controller_kernels`` import) and there is no global/even-tiling code.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton

# sys.path is configured by the package __init__ before this module is imported.
from sim.solver_sap import SolverSAP  # noqa: E402
from sim.sap_runtime import (  # noqa: E402
    sap_contacts_from_newton,
    sap_control_from_newton,
    sap_model_from_newton,
    sap_state_from_newton,
)

# ---- step-evolution mode codes (passed to _adapt_dt as a uniform kernel arg) ----
_MODE_FIXED = wp.constant(0)
_MODE_FEEDFORWARD = wp.constant(1)
_MODE_ADAPTIVE = wp.constant(2)
_MODE_CODES = {"fixed": 0, "feedforward": 1, "adaptive": 2}

# ---- Drake CalcAdjustedStepSize constants (err_order=2 for step doubling) ----
_DRAKE_SAFETY = wp.constant(wp.float32(0.9))
_DRAKE_MIN_SHRINK = wp.constant(wp.float32(0.1))
_DRAKE_MAX_GROW = wp.constant(wp.float32(5.0))
_DRAKE_HYSTERESIS_HIGH = wp.constant(wp.float32(1.2))
_DRAKE_HYSTERESIS_LOW = wp.constant(wp.float32(0.9))


# ============================================================================
# Inlined controller kernels (all per-world / per-element; no cross-world reduce)
# ============================================================================
@wp.kernel
def _open_frame(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
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
    ideal_dt: wp.array(dtype=wp.float32),
    dt_fixed: float,
    dt_min: float,
    dt_max: float,
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
):
    """Seed this frame's per-world working dt.

    ``fixed`` mode pins dt to the (clamped) constant ``dt_fixed``; ``feedforward`` and
    ``adaptive`` seed from the carried controller estimate ``ideal_dt`` (which holds the
    Drake step sized from the last accepted error). ``ideal_dt`` is preserved unclamped
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
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
):
    """Clamp dt so no world oversteps its boundary; worlds at/past it get dt=0 (no-op)."""
    i = wp.tid()
    remaining = next_time[i] - sim_time[i]
    if remaining <= wp.float32(0.0):
        dt[i] = wp.float32(0.0)
        dt_half[i] = wp.float32(0.0)
    elif dt[i] > remaining:
        dt[i] = remaining
        dt_half[i] = remaining * wp.float32(0.5)


@wp.kernel
def _inf_norm_state_error_kernel(
    joint_q_full: wp.array(dtype=wp.float32),
    joint_q_double: wp.array(dtype=wp.float32),
    state_scale: wp.array2d(dtype=wp.float32),
    coords_per_world: int,
    error_out: wp.array(dtype=wp.float32),
):
    """Per-world step-doubling accuracy metric (Kurtz & Castro, Sec. V-E)::

        e = || S (q_double - q_full) ||_inf

    Position-only inf-norm of the doubled-half-step vs. full-step ``q``, scaled by the
    diagonal ``S`` (here identity). NaN/inf collapse to a large sentinel so the
    controller treats them as divergence. In ``fixed`` mode this is called with
    ``joint_q_full == joint_q_double`` so ``e == 0`` for finite states (always accept)
    and ``NaN`` propagates for non-finite ones (the fixed-mode NaN guard).
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


@wp.kernel
def _average_velocity_guess_f64(
    a: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
    out: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    out[i] = wp.float64(0.5) * (a[i] + b[i])


@wp.kernel
def _average_velocity_guess_f32(
    a: wp.array(dtype=wp.float32),
    b: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    out[i] = wp.float32(0.5) * (a[i] + b[i])


@wp.kernel
def _set_scalar_i32(value: wp.array(dtype=int), new_value: int):
    value[0] = new_value


@wp.kernel
def _reset_solve_convergence(ok: wp.array(dtype=int)):
    i = wp.tid()
    ok[i] = 1


@wp.kernel
def _accumulate_solve_convergence(
    converged_env: wp.array(dtype=int),
    ok: wp.array(dtype=int),
):
    i = wp.tid()
    if converged_env[i] == 0:
        ok[i] = 0


@wp.kernel
def _apply_solve_convergence_to_error(
    ok: wp.array(dtype=int),
    err: wp.array(dtype=wp.float32),
    divergence_threshold: float,
):
    i = wp.tid()
    if ok[i] == 0:
        err[i] = divergence_threshold


@wp.kernel
def _adapt_dt(
    err: wp.array(dtype=wp.float32),
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    dt_half: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    diverged: wp.array(dtype=wp.bool),
    accept: wp.array(dtype=wp.bool),
    accepted_error: wp.array(dtype=wp.float32),
    substeps_frame: wp.array(dtype=wp.int32),
    cum_accepted: wp.array(dtype=wp.int32),
    mode: int,
    tol: float,
    dt_min: float,
    dt_max: float,
    divergence_threshold: float,
):
    """The per-world step-doubling controller -- the whole accept/reject/done decision.

    Writes ``accept[w]`` (gates the state commit), advances ``sim_time`` on accept, and
    evolves ``dt``/``ideal_dt`` per ``mode``. Every branch is per-thread data flow (no
    device control flow), which is what makes the enclosing substep loop a flat graph.
    """
    w = wp.tid()
    step = dt[w]

    # DONE: world reached its boundary (or clamp zeroed its step) -> commit nothing.
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
        return

    # ---------- FEEDFORWARD: dt frozen in-frame; refine ideal_dt for next frame ----------
    if mode == _MODE_FEEDFORWARD:
        if is_div:
            accept[w] = False
            sim_time[w] = next_time[w]
            diverged[w] = True
            ideal_dt[w] = dt_min
            return
        accept[w] = True
        sim_time[w] = sim_time[w] + step
        accepted_error[w] = e
        substeps_frame[w] = substeps_frame[w] + 1
        wp.atomic_add(cum_accepted, 0, 1)
        new_ideal = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
        if new_ideal > _DRAKE_HYSTERESIS_LOW * step and new_ideal < _DRAKE_HYSTERESIS_HIGH * step:
            new_ideal = step
        ideal_dt[w] = wp.clamp(new_ideal, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)
        return

    # ---------- ADAPTIVE: within-frame grow on accept / shrink+retry on reject ----------
    # At the floor we cannot subdivide further.
    if step <= dt_min * wp.float32(1.001):
        if is_div:
            accept[w] = False
            sim_time[w] = next_time[w]
            diverged[w] = True
            ideal_dt[w] = dt_min
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
        new_step = _DRAKE_SAFETY * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
        if new_step > _DRAKE_HYSTERESIS_LOW * step and new_step < _DRAKE_HYSTERESIS_HIGH * step:
            new_step = step
        ideal_dt[w] = wp.clamp(new_step, _DRAKE_MIN_SHRINK * step, _DRAKE_MAX_GROW * step)
        return

    # Above the floor and diverged: reject, shrink hard, hold state, retry.
    if is_div:
        accept[w] = False
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
    d = wp.clamp(new_step, dt_min, dt_max)
    if acc:
        accept[w] = True
        sim_time[w] = sim_time[w] + step
        accepted_error[w] = e
        substeps_frame[w] = substeps_frame[w] + 1
        wp.atomic_add(cum_accepted, 0, 1)
    else:
        accept[w] = False
    ideal_dt[w] = new_step
    dt[w] = d
    dt_half[w] = d * wp.float32(0.5)


@wp.kernel
def _commit_float(
    src: wp.array(dtype=wp.float32),
    accept: wp.array(dtype=wp.bool),
    stride: int,
    state: wp.array(dtype=wp.float32),
):
    """Commit the stepped result into the working state for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


@wp.kernel
def _commit_transform(
    src: wp.array(dtype=wp.transform),
    accept: wp.array(dtype=wp.bool),
    stride: int,
    state: wp.array(dtype=wp.transform),
):
    """Commit body poses for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


@wp.kernel
def _commit_spatial_vector(
    src: wp.array(dtype=wp.spatial_vector),
    accept: wp.array(dtype=wp.bool),
    stride: int,
    state: wp.array(dtype=wp.spatial_vector),
):
    """Commit body velocities for accepted worlds; hold otherwise."""
    i = wp.tid()
    if accept[i // stride]:
        state[i] = src[i]


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
    """Restore the per-world controller state to construction defaults for masked worlds.

    Called on env/episode reset so pre-reset controller state (dt / clocks / latches)
    does not leak into post-reset dynamics. ``sim_time`` and ``next_time`` reset together
    to 0 so the world restarts a clean boundary interval.
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


@wp.kernel
def _mark_unfinished(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    solve_ok: wp.array(dtype=int),
    flag: wp.array(dtype=wp.int32),
):
    """Set ``flag[0]`` to the loop status: 0 done, 1 unfinished, 2 solve failed.

    Read back (one int32) after each substep to decide whether the boundary loop can stop:
    once every world has landed (``sim_time >= next_time``) the flag is 0 and the loop
    breaks. A non-converged inner SAP solve uses the same status read to enforce the
    CENIC/Drake "converge or throw" contract without adding another host sync.
    """
    i = wp.tid()
    if solve_ok[i] == 0:
        wp.atomic_max(flag, 0, 2)
    elif sim_time[i] < next_time[i]:
        wp.atomic_max(flag, 0, 1)


class SolverSAPAdaptive:
    """Per-world adaptive (step-doubling) SAP integrator.

    Drop-in mirror of ``SolverMuJoCoAdaptive(model, ...)``: takes the Newton ``Model``,
    builds the ``SapModel`` + inner ``SolverSAP`` internally, and exposes the Newton
    solver surface (``step`` / ``step_dt`` / ``reset``) plus per-world telemetry
    (``dt`` / ``sim_time`` / ``diverged`` / ``substeps``).
    """

    def __init__(
        self,
        model,
        *,
        mode: str = "adaptive",
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-6,
        dt_inner_max: float | None = None,
        max_substeps: int = 16,
        max_rigid_contact: int = 128,
        max_iterations: int = 30,
        contact_preset_variant: str = "drake",
        line_search_variant: str = "armijo_decay",
        contact_tau_d: float = 0.01,
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
        if int(max_substeps) < 1:
            raise ValueError(f"max_substeps must be >= 1, got {max_substeps!r}.")
        self.model = model
        device = model.device
        wc = int(model.world_count)

        # Optional per-world spread telemetry (host sync; throttled; off unless the env var is set).
        self._spread_log = os.environ.get("NEWTON_SAP_SPREAD_LOG")
        self._spread_every = int(os.environ.get("NEWTON_SAP_SPREAD_EVERY", "10"))
        self._frame_counter = 0

        # ---- inner SAP solver + model ----
        # The convex SAP solve runs CONVERGENT (conditional, per-env early exit) to
        # optimality_rel_tol = max(kappa*tol, 1e-8), kappa = 1e-3 (CENIC, Kurtz & Castro
        # Sec. VI-B): the solver residual must sit far below the integration tolerance, else
        # it pollutes the step-doubling error estimate and the controller subdivides spuriously.
        self._sap_model = sap_model_from_newton(model)
        self._sap = SolverSAP(
            self._sap_model,
            max_rigid_contact=int(max_rigid_contact),
            max_iterations=int(max_iterations),
            optimality_rel_tol=max(1.0e-3 * float(tol), 1.0e-8),  # CENIC kappa*acc (Sec. VI-B)
            cost_abs_tol=0.0,
            cost_rel_tol=0.0,
            static_substep=False,
            contact_tau_d=float(contact_tau_d),
            contact_preset_variant=str(contact_preset_variant),
            line_search_variant=str(line_search_variant),
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
        elif self._sap.contact_solve.v_flat.dtype == wp.float32:
            self._average_velocity_guess_kernel = _average_velocity_guess_f32
        else:
            raise TypeError(f"Unsupported SAP velocity dtype {self._sap.contact_solve.v_flat.dtype!r}.")
        self._solve_ok = wp.ones(wc, dtype=int, device=device)

        # ---- per-world controller buffers (the dt VECTOR is the primitive; no N) ----
        self._dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(wc, dt_inner_init * 0.5, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._next_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(wc, dtype=wp.bool, device=device)
        self._diverged = wp.zeros(wc, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._substeps_frame = wp.zeros(wc, dtype=wp.int32, device=device)
        self._cum_accepted = wp.zeros(1, dtype=wp.int32, device=device)
        # Boundary flag: 1 if any world is still short of its boundary after a substep.
        # Read back once per iteration (the single accepted host sync) to break the loop early.
        self._unfinished = wp.zeros(1, dtype=wp.int32, device=device)

        self._mode = str(mode)
        self._mode_code = _MODE_CODES[self._mode]
        self._tol = float(tol)
        self._dt_min = float(dt_inner_min)
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")
        self._dt_inner_init = float(dt_inner_init)
        self._max_substeps = int(max_substeps)
        self._divergence_threshold = float(1.0e9)
        self._full_world_mask = wp.full(wc, True, dtype=wp.bool, device=device)

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
        # the doubling it does not need; feedforward/adaptive commit the doubled (more
        # accurate) state and feed the step-doubling error to the controller.
        self._do_doubling = self._mode != "fixed"
        self._commit_src = self._scratch_full if self._mode == "fixed" else self._scratch_double
        self._err_lhs = self._scratch_full
        self._err_rhs = self._scratch_full if self._mode == "fixed" else self._scratch_double

        # ---- own collision pipeline (refreshed at q_t and q_{t+h/2} per attempt) ----
        self._pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=int(max_rigid_contact) * wc,
        )
        self._contacts = self._pipeline.contacts()
        self._collide_state = model.state()
        self._sap_contacts = sap_contacts_from_newton(self._contacts)
        self._sap_control = None

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
        # count is constant in env count -- validated clean (no SIGABRT, tight spread) at 1024
        # and 4096 envs. The old >=1024-env SIGABRT was from capturing 3N such bodies in the old
        # per-N loop, not one. On any capture/instantiate failure the loop falls back to eager.
        self._graph_enabled = _is_cuda and os.environ.get("NEWTON_SAP_ADAPTIVE_GRAPH", "1") != "0"
        self._graph_cache: dict = {}
        # Modules/allocations must be warm before capture (a launch that triggers a lazy
        # module load syncs the stream and aborts capture). Run the first frame eagerly.
        self._graph_warmed = False

    # ---------------------------------------------------------------- properties
    @property
    def diverged(self) -> wp.array:
        return self._diverged

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
    def tiling(self) -> str:
        # Retained for back-compat; the per-world dt vector is no longer an even tiling.
        return "adaptive"

    @property
    def contacts(self):
        return self._contacts

    def cumulative_substeps(self) -> int:
        """Total SAP opt-steps since the last reset (= accepted inner steps * 3 evals)."""
        return int(self._cum_accepted.numpy()[0]) * 3

    def get_max_contact_count(self) -> int:
        """Per-batch rigid-contact capacity (for manager-level sensor buffer sizing)."""
        return self._max_rigid_contact * self._world_count

    def update_contacts(self, contacts, state) -> None:
        """No-op: SAP-adaptive owns its internal contact set; contact-sensor writeback
        from SAP is not yet wired (documented limitation for v1)."""
        return None

    def reset_compute_counter(self) -> None:
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
        wp.copy(self._collide_state.body_q, state_in.body_q)
        self._pipeline.collide(self._collide_state, self._contacts)

    def substep(self, state_in, state_out, control, contacts, dt: wp.array, guess=None) -> None:
        """ONE inner physics step at the per-world ``dt`` vector (= CENIC ``ComputeNextContinuousState``).

        ``guess`` is an explicit SAP-order velocity seed. Passing ``None`` disables the
        solver's persisted warm-start so the solve starts from its physical boundary
        velocity ``v0``. This mirrors Drake's CENIC warm-starts instead of accidentally
        reusing a rejected attempt's terminal velocity.
        """
        self._set_solver_guess(guess)
        self._sap.step(state_in, state_out, control, contacts, dt)
        wp.launch(
            _accumulate_solve_convergence,
            dim=self._world_count,
            inputs=[self._sap.contact_solve.converged_env, self._solve_ok],
            device=self.model.device,
        )

    # ----------------------------------------------------------- substep body
    def _substep_body(self, eff_dt_max: float) -> None:
        """One masked substep iteration: clamp -> evals -> error -> adapt -> commit -> mark.

        Identical flat kernel sequence every iteration, so it captures ONCE and replays per
        iteration. Per-world accept/reject/done is decided in ``_adapt_dt`` and applied by
        the gated ``_commit_*`` launches (a rejected or done world holds ``state_cur``). The
        final ``_mark_unfinished`` sets the boundary flag the loop reads to stop early; the
        flag is reset by the caller before each iteration so it reflects this step only.
        """
        n = self._world_count
        dev = self.model.device

        wp.launch(
            _clamp_dt_to_boundary,
            dim=n,
            inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
            device=dev,
        )

        wp.launch(_reset_solve_convergence, dim=n, inputs=[self._solve_ok], device=dev)

        # Build the contact model around q_t for the full step and first half-step.
        self._collide_from(self._state_cur)

        # Drake CENIC warm-starts: full from v_t.
        self._copy_state_velocity_to_sap_guess(self._state_cur, self._vt)
        self.substep(self._state_cur, self._scratch_full, self._sap_control, self._sap_contacts, self._dt,
                     guess=self._vt)
        if self._do_doubling:
            # half-1 from (v_t + v_full) / 2, reusing the q_t contact model.
            wp.copy(self._vfull, self._sap.contact_solve.v_flat)
            self._average_velocity_guess(self._vt, self._vfull, self._vhalf1)
            self.substep(self._state_cur, self._scratch_mid, self._sap_control, self._sap_contacts, self._dt_half,
                         guess=self._vhalf1)

            # half-2 starts from q_{t+h/2}, so rebuild contacts at the midpoint state and
            # warm-start from v_full.
            self._collide_from(self._scratch_mid)
            self.substep(self._scratch_mid, self._scratch_double, self._sap_control, self._sap_contacts,
                         self._dt_half, guess=self._vfull)

        wp.launch(
            _inf_norm_state_error_kernel,
            dim=n,
            inputs=[self._err_lhs.joint_q, self._err_rhs.joint_q, self._state_scale, self._coords_per_world],
            outputs=[self._last_error],
            device=dev,
        )
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
            ],
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

        # Boundary flag for early termination: set to 1 if any world is still unfinished.
        wp.launch(
            _mark_unfinished,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._solve_ok, self._unfinished],
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
        key = round(float(dt_outer), 12)
        graph = self._graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    self._substep_body(eff_dt_max)
                graph = cap.graph
                self._graph_cache[key] = graph
            except Exception:
                self._graph_enabled = False
                return None
        return graph

    def _run_substep_loop(self, eff_dt_max: float, dt_outer: float) -> None:
        """March substeps until every world reaches its boundary, capped at ``max_substeps``.

        The body is captured once and replayed per iteration (or run eagerly while warming /
        if capture fails). After each iteration the 4-byte ``_unfinished`` flag is read back
        -- the single accepted host sync in the step path -- to stop as soon as all worlds
        land instead of grinding fixed no-op substeps.
        """
        graph = self._body_graph(eff_dt_max, dt_outer)
        for _ in range(self._max_substeps):
            self._unfinished.zero_()
            if graph is not None:
                try:
                    wp.capture_launch(graph)
                except Exception:
                    # cudaGraphInstantiate can OOM here (outside capture); drop it and finish
                    # this frame eagerly so the boundary still advances.
                    self._graph_cache.clear()
                    self._graph_enabled = False
                    graph = None
                    self._substep_body(eff_dt_max)
            else:
                self._substep_body(eff_dt_max)
            status = int(self._unfinished.numpy()[0])
            if status >= 2:
                raise RuntimeError(
                    "SolverSAPAdaptive inner SAP solve failed to converge to "
                    f"optimality_rel_tol={max(1.0e-3 * self._tol, 1.0e-8):.3e}."
                )
            if status == 0:
                break

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

        self._sap_control = sap_control_from_newton(control)

        # Load the incoming Newton state into the internal working buffer.
        self._copy_state(self._state_cur, sap_state_from_newton(state))

        # Open the frame: rebase clocks (Fix B), set the new boundary, seed per-world dt,
        # clear per-frame work counters and the divergence latch.
        wp.launch(_open_frame, dim=n, inputs=[self._sim_time, self._next_time, dt_outer], device=device)
        wp.launch(
            _seed_dt,
            dim=n,
            inputs=[self._mode_code, self._ideal_dt, self._dt_inner_init, self._dt_min, eff_dt_max,
                    self._dt, self._dt_half],
            device=device,
        )
        self._substeps_frame.zero_()
        self._diverged.zero_()

        # Masked substep march: each attempt rebuilds contacts at q_t and, for
        # step-doubling, at q_{t+h/2}. The loop stops as soon as every world reaches
        # its boundary (one 4-byte flag read per iteration; ~3 iters typically).
        self._run_substep_loop(eff_dt_max, dt_outer)

        # Optional per-world spread telemetry (diagnostic; one host sync, throttled).
        self._frame_counter += 1
        if self._spread_log and self._frame_counter % self._spread_every == 0:
            s = self._substeps_frame.numpy()
            pct = np.percentile(s, [50, 90, 99, 100]).astype(int)
            with open(self._spread_log, "a") as f:
                f.write(
                    f"frame={self._frame_counter} substeps_per_world[min={int(s.min())} "
                    f"p50={pct[0]} p90={pct[1]} p99={pct[2]} max={pct[3]}] "
                    f"saturated@{self._max_substeps}={int((s >= self._max_substeps).sum())}/{s.size}\n"
                )

        # Write the advanced state back into the Newton state.
        self._copy_state(sap_state_from_newton(state), self._state_cur)
        return state

    # ------------------------------------------------------------------- step
    def step(self, state_in, state_out, control, contacts=None, dt=None, apply_forces=None):
        """Newton-signature boundary call ``(state_in, state_out, control, contacts, dt)``.

        Thin adapter over :meth:`integrate`: ``state_in`` is advanced in place by ``dt``
        (= ``dt_outer``) and returned; ``state_out`` is accepted for signature uniformity
        and returned unchanged. ``contacts`` is accepted but UNUSED; the integrator rebuilds
        its internal contact set at each adaptive attempt's start and midpoint state.
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
            ],
            device=self.model.device,
        )
