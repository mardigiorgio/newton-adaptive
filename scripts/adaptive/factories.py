# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Factory builders for adaptive solver variants."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.solvers

from scripts.adaptive import kernels as K
from scripts.adaptive.base import AdaptiveWrapper
from scripts.adaptive.compaction_mixin import CompactionMixin
from scripts.adaptive.masking_mixin import ActiveSetMaskingMixin


def _make_q_qd_error_norm(wrapper_ref):
    """Build an error_norm_fn that uses q + dt*qd norm.

    Takes wrapper_ref (the same forward-reference box used by step_fn closures)
    so we can read wrapper._dt at call time.
    """
    def error_norm_fn():
        wrapper = wrapper_ref[0]
        n = wrapper.model.world_count
        wp.launch(
            K._inf_norm_q_qd_kernel, dim=n,
            inputs=[
                wrapper._state_full.joint_q, wrapper._state_double.joint_q,
                wrapper._state_full.joint_qd, wrapper._state_double.joint_qd,
                wrapper._q_weights, wrapper._qd_weights, wrapper._dt,
                wrapper._coords_per_world, wrapper._dofs_per_world,
            ],
            outputs=[wrapper._last_error], device=wrapper.model.device,
        )
    return error_norm_fn


def _build_mujoco_q_weights(model, mjw_model) -> wp.array:
    """Build CENIC-style per-coord error weights from MuJoCo dof_invweight0.

    Mirrors :class:`SolverMuJoCoAdaptive` lines 386-413: for each joint take the
    max invweight across its DoFs, sqrt to get the qpos weight, broadcast to
    all qpos coords of that joint, normalize per world so the heaviest coord
    has weight 1, then clip to ``[1, 10]``.
    """
    n = model.world_count
    coords_per_world = model.joint_coord_count // n
    dofs_per_world = model.joint_dof_count // n

    jnt_qposadr = mjw_model.jnt_qposadr.numpy()
    jnt_dofadr = mjw_model.jnt_dofadr.numpy()
    invweight = mjw_model.dof_invweight0.numpy()  # [world_count, nv]
    njnt = len(jnt_qposadr)

    q_weights = np.ones((n, coords_per_world), dtype=np.float32)
    for w in range(n):
        dof_w = np.clip(invweight[w], 1.0e-30, None)
        for j in range(njnt):
            q_s = int(jnt_qposadr[j])
            q_e = int(jnt_qposadr[j + 1]) if j + 1 < njnt else coords_per_world
            qd_s = int(jnt_dofadr[j])
            qd_e = int(jnt_dofadr[j + 1]) if j + 1 < njnt else dofs_per_world
            joint_max_invweight = float(dof_w[qd_s:qd_e].max())
            q_weights[w, q_s:q_e] = np.sqrt(joint_max_invweight)
        q_weights[w] /= q_weights[w].min()
    q_weights = np.clip(q_weights, 1.0, 10.0)
    return wp.array(q_weights, dtype=wp.float32, device=model.device)


class AdaptiveCompactionWrapper(CompactionMixin, AdaptiveWrapper):
    """Adaptive wrapper + compaction mixin (for MuJoCo)."""


class AdaptiveMaskingWrapper(ActiveSetMaskingMixin, AdaptiveWrapper):
    """Adaptive wrapper + active-set masking (for XPBD / SemiImplicit)."""


def adaptive_mujoco_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
    nconmax: int,
    njmax: int,
    compaction_sizes: tuple[float, ...] = (1.0, 0.25, 0.0625, 0.015625),
):
    """Build (solver, step_fn) for MuJoCo adaptive.

    Per-world dt is passed via mjw_model.opt.timestep (copied each step).
    The wrapper's CompactionMixin tracks active worlds (v1: bookkeeping only).
    """

    def build(model):
        underlying = newton.solvers.SolverMuJoCo(
            model, separate_worlds=True, use_mujoco_contacts=False,
            nconmax=nconmax, njmax=njmax,
        )

        # Replace mjw_model.opt.timestep with a stable per-world buffer so
        # wp.copy() targets a fixed warp array (same pattern as SolverMuJoCoAdaptive
        # lines 382-384 in solver_mujoco_cenic.py).
        n = model.world_count
        device = model.device
        timestep_buf = wp.full(n, dt_init, dtype=wp.float32, device=device)
        underlying.mjw_model.opt.timestep = timestep_buf

        # Per-coord error weights from MuJoCo's dof_invweight0, matching
        # SolverMuJoCoAdaptive's weighted position-only norm (paper Sec V-E).
        q_weights = _build_mujoco_q_weights(model, underlying.mjw_model)

        # Dedicated SAP collision pipeline sized to MJWarp's max contact count,
        # matching SolverMuJoCoAdaptive line 415-418. Default model.contacts()
        # uses explicit broad phase + different buffer size -> different
        # contact set -> divergence from CENIC.
        pipeline = newton.CollisionPipeline(
            model, broad_phase="sap",
            rigid_contact_max=underlying.mjw_data.naconmax,
        )
        contacts = pipeline.contacts()

        def pre_boundary_hook(model_arg, state_0, control, contacts_arg):
            """Apply control + enable RNE + run broad-phase collide, once per
            outer step. Matches SolverMuJoCoAdaptive.step_dt lines 730-746."""
            underlying._apply_mjc_control(model_arg, state_0, control, underlying.mjw_data)
            underlying._enable_rne_postconstraint(state_0)
            if not underlying.mjw_model.opt.run_collision_detection:
                pipeline.collide(state_0, contacts_arg)

        def pre_iter_hook(model_arg, state_cur, contacts_arg):
            """Re-transform contacts to world frame using state_cur transforms,
            once per iteration (not per substep). Matches
            SolverMuJoCoAdaptive._run_iteration_body lines 469-471."""
            if not underlying.mjw_model.opt.run_collision_detection:
                underlying._convert_contacts_to_mjwarp(model_arg, state_cur, contacts_arg)

        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array, dt_scalar_buf):
            """MuJoCo per-substep shim: sync state, set per-world dt, step, read back.

            Matches SolverMuJoCoAdaptive._run_substep exactly. Contacts are
            already populated in MJWarp format by pre_iter_hook; this shim
            must not touch them (re-converting per substep with mid-substep
            transforms corrupts the step-doubling error estimate).
            dt_scalar_buf is unused for MuJoCo (it reads per-world from opt.timestep).
            """
            underlying._update_mjc_data(underlying.mjw_data, model_arg, state_in)
            wp.copy(underlying.mjw_model.opt.timestep, dt_array)
            with wp.ScopedDevice(model_arg.device):
                underlying._mujoco_warp_step()
            underlying._update_newton_state(model_arg, state_out, underlying.mjw_data)

        wrapper = AdaptiveCompactionWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False,
            contacts=contacts,
            q_weights=q_weights,
            pre_boundary_hook=pre_boundary_hook,
            pre_iter_hook=pre_iter_hook,
            compaction_sizes=compaction_sizes,
        )

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build


def adaptive_xpbd_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
):
    """Build (solver, step_fn) for XPBD adaptive (effectively-global dt).

    XPBD takes scalar dt; we pass max(dt_array) and rely on world_active
    to skip finished worlds at the kernel level (Task 9 modifications).
    """

    def build(model):
        underlying = newton.solvers.SolverXPBD(model)
        contacts = model.contacts()

        # Forward declaration: the shim needs to read wrapper._world_active,
        # but wrapper is built AFTER step_fn. Box it via a list.
        wrapper_ref = []
        error_norm = _make_q_qd_error_norm(wrapper_ref)

        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array, dt_scalar_buf):
            wrapper = wrapper_ref[0]
            # Detect half-step by comparing array refs.
            scalar_dt = (wrapper._scalar_dt_half_host
                         if dt_array is wrapper._dt_half
                         else wrapper._scalar_dt_host)
            model_arg.collide(state_in, contacts)
            mask = wrapper._world_active
            underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)

        wrapper = AdaptiveMaskingWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False, contacts=contacts,
            needs_scalar_dt=True,
            error_norm_fn=error_norm,
        )
        wrapper_ref.append(wrapper)  # close the forward-reference loop

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build


def adaptive_semi_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
):
    """Build (solver, step_fn) for SemiImplicit adaptive (effectively-global dt)."""

    def build(model):
        underlying = newton.solvers.SolverSemiImplicit(model)
        contacts = model.contacts()
        wrapper_ref = []
        error_norm = _make_q_qd_error_norm(wrapper_ref)

        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array, dt_scalar_buf):
            wrapper = wrapper_ref[0]
            # Detect half-step by comparing array refs.
            scalar_dt = (wrapper._scalar_dt_half_host
                         if dt_array is wrapper._dt_half
                         else wrapper._scalar_dt_host)
            model_arg.collide(state_in, contacts)
            mask = wrapper._world_active
            underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)

        wrapper = AdaptiveMaskingWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False, contacts=contacts,
            needs_scalar_dt=True,
            error_norm_fn=error_norm,
        )
        wrapper_ref.append(wrapper)

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build
