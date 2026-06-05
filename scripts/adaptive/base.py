# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""AdaptiveWrapper core class.

Wraps an arbitrary step_fn with step-doubling adaptive stepping and a
Drake PI controller. Solver-agnostic -- translate solver specifics in the
step_fn shim built by scripts.adaptive.factories.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import warp as wp

import newton

from scripts.adaptive import kernels as K
from scripts.adaptive.controller import ControllerConfig

StepFn = Callable[
    [newton.Model, newton.State, newton.State, newton.Control,
     newton.Contacts, wp.array, wp.array],
    None,
]
# Args: model, state_in, state_out, ctrl, contacts, dt_array (per-world wp.array(N, float32)),
#       dt_scalar_buf (1-elem wp.array(1, float32), pre-reduced max).

# Called once at the top of each step_dt boundary, before any iteration.
# Hook receives (model, state_0, control, contacts).
PreBoundaryHook = Callable[
    [newton.Model, newton.State, newton.Control, newton.Contacts],
    None,
]

# Called once per iteration, after state_saved snapshot but before the 3
# substeps. Receives (model, state_cur, contacts). MuJoCo uses this to
# re-transform body-frame contacts to world frame using state_cur transforms,
# matching SolverMuJoCoAdaptive._run_iteration_body.
PreIterHook = Callable[
    [newton.Model, newton.State, newton.Contacts],
    None,
]


class AdaptiveWrapper:
    """Step-doubling adaptive wrapper around an arbitrary step_fn."""

    def __init__(
        self,
        model: newton.Model,
        step_fn: StepFn,
        *,
        tol: float,
        dt_init: float,
        dt_min: float,
        dt_max: float,
        dt_outer: float,
        needs_collide: bool,
        contacts: newton.Contacts | None = None,
        stuck_policy: Literal["freeze", "raise"] = "freeze",
        max_iters: int = 500,
        controller: ControllerConfig | None = None,
        q_weights: wp.array | None = None,
        qd_weights: wp.array | None = None,
        error_norm_fn=None,
        pre_boundary_hook: PreBoundaryHook | None = None,
        pre_iter_hook: PreIterHook | None = None,
        needs_scalar_dt: bool = False,
    ):
        self.model = model
        self.step_fn = step_fn
        self.tol = tol
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_outer = dt_outer
        self.needs_collide = needs_collide
        self.contacts = contacts if contacts is not None else model.contacts()
        self.stuck_policy = stuck_policy
        self.max_iters = max_iters
        self.controller = controller or ControllerConfig()
        self.pre_boundary_hook = pre_boundary_hook
        self.pre_iter_hook = pre_iter_hook
        self.needs_scalar_dt = needs_scalar_dt

        device = model.device
        n = model.world_count

        # Per-world arrays.
        self._dt = wp.full(n, dt_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(n, dt_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(n, dt_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(n, dtype=wp.float32, device=device)
        self._next_time = wp.zeros(n, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(n, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(n, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(n, dtype=wp.float32, device=device)

        # 1-element host-sync arrays.
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)

        # Pre-reduced scalar dt (for solvers that take scalar, not per-world).
        # Used by XPBD/Semi step_fn shims; MuJoCo ignores it (uses opt.timestep array).
        self._dt_scalar = wp.zeros(1, dtype=wp.float32, device=device)
        # Host-side cache of the scalar, updated once per iteration in _run_iteration_body.
        self._scalar_dt_host = float(dt_init)
        self._scalar_dt_half_host = float(dt_init) * 0.5

        # Scratch states.
        self._state_saved = model.state()
        self._state_full = model.state()
        self._state_mid = model.state()
        self._state_double = model.state()

        # Per-coord weights for error norm (q-only, weighted). Caller may
        # supply pre-computed weights (MuJoCo factory does this from
        # dof_invweight0, matching SolverMuJoCoAdaptive); otherwise fall back
        # to all-ones (equivalent to unweighted inf-norm).
        if q_weights is not None:
            self._q_weights = q_weights
        else:
            self._q_weights = self._build_q_weights(model, device)

        # Geometry constants used by select kernels.
        self._coords_per_world = model.joint_coord_count // n
        self._dofs_per_world = model.joint_dof_count // n
        self._bodies_per_world = (model.body_count // n) if model.body_count > 0 else 0

        # qd weights default to all-ones (same shape as a (N, dofs_per_world) wp.array2d).
        if qd_weights is None:
            dofs_per_world = self._dofs_per_world
            qd_w_np = np.ones((n, dofs_per_world), dtype=np.float32)
            self._qd_weights = wp.from_numpy(qd_w_np, dtype=wp.float32, device=device)
        else:
            self._qd_weights = qd_weights

        # Error-norm callable: default = q-only (current behavior).
        if error_norm_fn is None:
            self.error_norm_fn = self._default_error_norm_q_only
        else:
            self.error_norm_fn = error_norm_fn

        # Status summary scratch (6 scalars).
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

    def _build_q_weights(self, model: newton.Model, device) -> wp.array:
        """Build default all-ones per-coord weights for the error norm.

        Equivalent to an unweighted inf-norm. The MuJoCo factory supplies
        a pre-computed q_weights array (built from ``dof_invweight0``,
        matching :class:`SolverMuJoCoAdaptive` lines 382-413) via the
        ``q_weights`` constructor argument.

        Returns a wp.array2d(dtype=wp.float32) with shape (world_count, coords_per_world).
        """
        n = model.world_count
        coords_per_world = model.joint_coord_count // n
        q_weights_np = np.ones((n, coords_per_world), dtype=np.float32)
        return wp.from_numpy(q_weights_np, dtype=wp.float32, device=device)

    def _default_error_norm_q_only(self):
        """v1/default error norm: weighted L-inf on joint_q only."""
        n = self.model.world_count
        wp.launch(
            K._inf_norm_state_error_kernel, dim=n,
            inputs=[self._state_full.joint_q, self._state_double.joint_q,
                    self._q_weights, self._coords_per_world],
            outputs=[self._last_error], device=self.model.device,
        )

    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """One step-doubling iteration: 3 evals + error + accept/reject + advance."""
        model = self.model
        n = model.world_count
        dev = model.device

        wp.launch(K._iter_count_increment, dim=1,
                  inputs=[self._iteration_count_buf], device=dev)

        # Clamp dt so no world overshoots its boundary target.
        wp.launch(K._clamp_dt_to_boundary, dim=n,
                  inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
                  device=dev)

        # Save state for rejection rollback.
        wp.copy(self._state_saved.joint_q, self._state_cur.joint_q)
        wp.copy(self._state_saved.joint_qd, self._state_cur.joint_qd)
        if self._state_cur.body_q is not None and self._state_saved.body_q is not None:
            wp.copy(self._state_saved.body_q, self._state_cur.body_q)
        if self._state_cur.body_qd is not None and self._state_saved.body_qd is not None:
            wp.copy(self._state_saved.body_qd, self._state_cur.body_qd)

        # Collide if solver doesn't (MuJoCo handles internally).
        if self.needs_collide:
            self.model.collide(self._state_cur, self.contacts)

        # Per-iteration setup (MuJoCo: collide via its own pipeline + convert
        # contacts to MJWarp format using state_cur transforms, matching
        # SolverMuJoCoAdaptive._run_iteration_body lines 469-471).
        if self.pre_iter_hook is not None:
            self.pre_iter_hook(model, self._state_cur, self.contacts)

        # Reduce per-world dt to a scalar (only for solvers that need it).
        # MuJoCo reads per-world dt from opt.timestep directly -- skip for it.
        if self.needs_scalar_dt:
            wp.launch(K._scalar_max_dt_reset, dim=1,
                      inputs=[self._dt_scalar], device=dev)
            wp.launch(K._scalar_max_dt, dim=n,
                      inputs=[self._dt, self._dt_scalar], device=dev)
            self._scalar_dt_host = float(self._dt_scalar.numpy()[0])
            self._scalar_dt_half_host = self._scalar_dt_host * 0.5

        # 3 step_fn calls: full dt, half dt, half dt.
        self.step_fn(model, self._state_cur, self._state_full, None,
                     self.contacts, self._dt, self._dt_scalar)
        self.step_fn(model, self._state_cur, self._state_mid, None,
                     self.contacts, self._dt_half, self._dt_scalar)
        self.step_fn(model, self._state_mid, self._state_double, None,
                     self.contacts, self._dt_half, self._dt_scalar)

        # Error norm -- pluggable (q-only by default, q+qd for XPBD/Semi).
        self.error_norm_fn()

        # Per-world accept/reject + new ideal_dt.
        wp.launch(
            K._calc_adjusted_step, dim=n,
            inputs=[
                self._last_error, self._dt, self._ideal_dt,
                self._accepted, self.tol, self.dt_min,
                self.controller.safety_factor,
                self.controller.shrink_cap,
                self.controller.growth_cap,
                self.controller.hysteresis_high,
                self.controller.hysteresis_low,
            ],
            device=dev,
        )

        # State select: cur := accepted ? double : saved.
        wp.launch(
            K._select_float_kernel, dim=model.joint_coord_count,
            inputs=[self._state_double.joint_q, self._state_saved.joint_q,
                    self._accepted, self._coords_per_world],
            outputs=[self._state_cur.joint_q], device=dev,
        )
        wp.launch(
            K._select_float_kernel, dim=model.joint_dof_count,
            inputs=[self._state_double.joint_qd, self._state_saved.joint_qd,
                    self._accepted, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd], device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                K._select_transform_kernel, dim=model.body_count,
                inputs=[self._state_double.body_q, self._state_saved.body_q,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_q], device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                K._select_spatial_vector_kernel, dim=model.body_count,
                inputs=[self._state_double.body_qd, self._state_saved.body_qd,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_qd], device=dev,
            )

        # Advance sim_time for accepted worlds.
        wp.launch(
            K._advance_sim_time, dim=n,
            inputs=[self._sim_time, self._dt, self._accepted,
                    self._last_error, self._accepted_error],
            device=dev,
        )

        # Cap dt for the next iteration.
        wp.launch(
            K._apply_dt_cap, dim=n,
            inputs=[self._ideal_dt, self.dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=dev,
        )

        # Boundary check.
        wp.launch(K._boundary_reset, dim=1, inputs=[self._boundary_flag], device=dev)
        wp.launch(
            K._boundary_check, dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )

    def step_dt(
        self,
        dt_outer: float,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
    ) -> tuple[newton.State, newton.State]:
        """Advance every world by exactly dt_outer seconds of sim time.

        Loops _run_iteration_body until every world's sim_time reaches the
        boundary. One 4-byte boundary-flag sync per iteration.
        """
        model = self.model
        n = model.world_count
        device = model.device

        # state_cur is the "live" state; we point it at state_0 for in-place use.
        self._state_cur = state_0

        effective_dt_max = min(self.dt_max, dt_outer)

        # Initial dt = ideal_dt clamped to [dt_min, effective_dt_max].
        wp.launch(
            K._apply_dt_cap, dim=n,
            inputs=[self._ideal_dt, self.dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=device,
        )

        # Set per-world next_time = sim_time + dt_outer (in place, via _boundary_advance which adds).
        wp.launch(K._boundary_advance, dim=n,
                  inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)

        # Per-boundary setup (MuJoCo: apply control once, enable RNE, run
        # broad-phase collision once -- matching SolverMuJoCoAdaptive.step_dt
        # lines 730-746).
        if self.pre_boundary_hook is not None:
            self.pre_boundary_hook(model, state_0, control, self.contacts)

        # Boundary loop: one PCIe sync per iter (matches v1 CENIC).
        # max_iters is tracked on host (free) -- no extra GPU launches needed.
        iter_host_count = 0
        while True:
            self._run_iteration_body(effective_dt_max)
            iter_host_count += 1
            if iter_host_count >= self.max_iters:
                raise RuntimeError(
                    f"AdaptiveWrapper: max_iters={self.max_iters} exceeded "
                    f"in step_dt(dt_outer={dt_outer})"
                )
            if int(self._boundary_flag.numpy()[0]) == 0:
                break

        return state_0, state_1

    @property
    def iteration_count(self) -> wp.array:
        """Most-recent K count (shape [1], int32, on device)."""
        return self._iteration_count_buf

    @property
    def dt(self) -> wp.array:
        """Per-world current dt (shape [N], float32, on device)."""
        return self._dt

    def status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary (one PCIe sync)."""
        device = self.model.device
        n = self.model.world_count

        wp.launch(K._status_sentinel_reset, dim=1,
                  inputs=[self._status_scalars], device=device)
        wp.launch(
            K._status_summary_kernel, dim=n,
            inputs=[self._sim_time, self._accepted_error, self._dt,
                    self._accepted, self._status_scalars],
            device=device,
        )

        s = self._status_scalars.numpy()
        return {
            "sim_time_min": float(s[0]),
            "sim_time_max": float(s[1]),
            "error_max":    float(s[2]),
            "accept_count": int(s[3]),
            "dt_min":       float(s[4]),
            "dt_max":       float(s[5]),
        }
