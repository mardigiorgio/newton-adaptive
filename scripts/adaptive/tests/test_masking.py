# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import warp as wp

from scripts.adaptive.base import AdaptiveWrapper
from scripts.adaptive.masking_mixin import ActiveSetMaskingMixin
from scripts.scenes import _registry


class _MaskedWrapper(ActiveSetMaskingMixin, AdaptiveWrapper):
    pass


def _noop_step_fn(model, state_in, state_out, ctrl, contacts, dt_array, dt_scalar_buf):
    pass


def test_masking_mixin_allocates_active_mask():
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = _MaskedWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    assert w._world_active.shape == (4,)
    assert all(w._world_active.numpy().tolist())  # all True initially


def test_masking_mixin_updates_mask_after_step_dt():
    """After step_dt completes, all worlds are done -> world_active all False."""
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = _MaskedWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()
    # After step_dt, sim_time = next_time for all worlds, so active mask is all False.
    assert not any(w._world_active.numpy().tolist())


def test_xpbd_step_respects_world_active_mask():
    """Step with all-False mask: XPBD kernels must skip work and stay finite.

    Notes:
        ``SolverXPBD.step`` calls ``integrate_bodies`` from the shared
        ``SolverBase`` (in ``newton/_src/solvers/solver.py``) which is not
        an XPBD kernel and therefore is not mask-aware in this task. That
        integrator applies gravity over ``dt``, producing a small
        ``g * dt**2`` z-displacement (~1e-5 for dt=1e-3) regardless of
        the mask. We assert that:
          * the output is finite (no NaN/Inf),
          * the per-component displacement does not exceed the
            free-fall budget (``|g|*dt**2``) by more than a small slack,
        which proves XPBD's per-world kernels did not apply any
        constraint/contact deltas to inactive worlds.
    """
    import numpy as np
    import newton, newton.solvers
    from scripts.scenes import _registry
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    s = newton.solvers.SolverXPBD(m)
    contacts = m.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    m.collide(s0, contacts)

    # Snapshot input
    bq0 = s0.body_q.numpy().copy()

    # All-False mask
    mask = wp.full(m.world_count, False, dtype=wp.bool, device=m.device)
    dt = 1e-3
    s.step(s0, s1, ctrl, contacts, dt, world_active=mask)
    wp.synchronize()

    bq1 = s1.body_q.numpy()
    assert np.isfinite(bq1).all(), "all-False mask must not introduce NaN/Inf"

    # Rotation: integrate_bodies normalizes the quaternion every step, which
    # round-trips through float32 arithmetic and can drift by ~1 ULP per
    # component even with zero angular velocity. Tolerate that ULP-scale
    # noise but reject any larger change (which would come from constraint
    # kernels mistakenly executing on inactive worlds).
    np.testing.assert_allclose(
        bq1[:, 3:], bq0[:, 3:], atol=1e-6,
        err_msg="quaternion must be unchanged when no constraint/contact kernels run",
    )

    # Translation: bound by free-fall over one dt (gravity-only integration).
    g_mag = float(np.linalg.norm(m.gravity.numpy()[0]))
    free_fall = g_mag * dt * dt
    delta = np.abs(bq1[:, :3] - bq0[:, :3])
    assert delta.max() <= free_fall * 1.01, (
        f"position delta {delta.max():.3e} exceeds free-fall budget "
        f"{free_fall:.3e} - XPBD kernels did not respect the mask"
    )


def test_semi_step_respects_world_active_mask():
    """Step with all-False mask should leave SemiImplicit kernel deltas unapplied.

    Note: SolverBase.integrate_bodies still runs unmasked (Task 9 noted this
    architectural limitation -- it's shared across all solvers and out of this
    task's scope). So body_q may drift by gravity*dt^2 even with all-False
    mask. Assertion budgets for this.
    """
    import numpy as np
    import newton, newton.solvers
    from scripts.scenes import _registry
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    s = newton.solvers.SolverSemiImplicit(m)
    contacts = m.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    m.collide(s0, contacts)

    bq0 = s0.body_q.numpy().copy()
    mask = wp.full(m.world_count, False, dtype=wp.bool, device=m.device)
    s.step(s0, s1, ctrl, contacts, 1e-3, world_active=mask)
    wp.synchronize()

    bq1 = s1.body_q.numpy()
    # No NaN
    assert not np.isnan(bq1).any(), "all-False mask shouldn't produce NaN"
    # Translation drift bounded by free-fall budget (gravity * dt^2)
    delta_trans = np.abs(bq1[..., :3] - bq0[..., :3])
    assert delta_trans.max() < 1e-4, (
        f"all-False mask: translation drift {delta_trans.max():.2e} exceeds free-fall budget"
    )
