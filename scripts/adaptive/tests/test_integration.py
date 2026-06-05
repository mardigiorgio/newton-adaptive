# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for AdaptiveWrapper."""

import pytest
import warp as wp

from scripts.adaptive.base import AdaptiveWrapper
from scripts.scenes import _registry


def _noop_step_fn(model, state_in, state_out, ctrl, contacts, dt_array, dt_scalar_buf):
    """No-op step_fn for testing allocation paths."""
    pass


def test_adaptive_wrapper_allocates_scratch_states():
    """Constructor allocates 4 scratch states + per-world arrays."""
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = AdaptiveWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    # Per-world arrays sized N
    assert w._dt.shape == (4,)
    assert w._sim_time.shape == (4,)
    assert w._next_time.shape == (4,)
    assert w._accepted.shape == (4,)
    assert w._last_error.shape == (4,)
    # Scratch states
    assert w._state_saved is not None
    assert w._state_full is not None
    assert w._state_mid is not None
    assert w._state_double is not None
    # Boundary / iter scalars
    assert w._boundary_flag.shape == (1,)
    assert w._iteration_count_buf.shape == (1,)
    # Per-coord weights (2D: world_count x coords_per_world)
    assert w._q_weights.shape[0] == model.world_count


def test_step_dt_with_noop_step_fn_terminates_immediately():
    """With a no-op step_fn, error is 0 everywhere, all worlds accept on iter 1.

    The no-op step_fn writes nothing to state_full / state_mid / state_double,
    so the error_norm sees max|0 - 0| = 0, which is <= tol -> all worlds accept,
    sim_time += dt -> boundary reached after 1 iteration.
    """
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = AdaptiveWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()

    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()

    K_done = int(w.iteration_count.numpy()[0])
    assert K_done == 1, f"no-op step_fn should accept in 1 iter, got K={K_done}"


def test_adaptive_mujoco_factory_runs():
    """adaptive_mujoco_factory produces a working (solver, step_fn) pair."""
    import numpy as np
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()

    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()

    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), "NaN in body_q after 10 outer steps"


def test_adaptive_xpbd_factory_runs():
    """adaptive_xpbd_factory produces a working (solver, step_fn) pair."""
    import numpy as np
    from scripts.adaptive.factories import adaptive_xpbd_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    assert not np.isnan(s0.body_q.numpy()).any()


def test_adaptive_semi_factory_runs():
    import numpy as np
    from scripts.adaptive.factories import adaptive_semi_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_semi_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    assert not np.isnan(s0.body_q.numpy()).any()


def test_adaptive_mujoco_factory_matches_shim_oracle():
    """adaptive_mujoco_factory produces same state as SolverMuJoCoAdaptive after 100 outer
    steps. This is the v1 correctness gate."""
    import numpy as np
    import newton, newton.solvers
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    N = 4
    n_outer = 100

    # Run CENIC oracle.
    m_oracle = scene.build_model_randomized(N, seed=42)
    cenic = newton.solvers.SolverMuJoCoAdaptive(
        m_oracle, tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6,
        dt_max=DT_OUTER, nconmax=8, njmax=32,
    )
    s0_o, s1_o, ctrl_o = m_oracle.state(), m_oracle.state(), m_oracle.control()
    for _ in range(n_outer):
        cenic.step(s0_o, s1_o, ctrl_o, None, DT_OUTER)
    wp.synchronize()
    oracle_bq = s0_o.body_q.numpy()

    # Run adaptive wrapper.
    m_wrap = scene.build_model_randomized(N, seed=42)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
    )
    _, step_fn = builder(m_wrap)
    s0_w, s1_w, ctrl_w = m_wrap.state(), m_wrap.state(), m_wrap.control()
    for _ in range(n_outer):
        s0_w, s1_w = step_fn(m_wrap, s0_w, s1_w, ctrl_w)
    wp.synchronize()
    wrap_bq = s0_w.body_q.numpy()

    # Match within float32 tolerance.
    np.testing.assert_allclose(
        wrap_bq, oracle_bq, rtol=1e-4, atol=1e-5,
        err_msg="adaptive_mujoco diverged from CENIC oracle after 100 outer steps",
    )


@pytest.mark.parametrize("factory_name,kwargs", [
    ("adaptive_mujoco_factory", dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01, nconmax=8, njmax=32)),
    ("adaptive_xpbd_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
    ("adaptive_semi_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
])
def test_adaptive_variant_stable_on_falling_cylinder(factory_name, kwargs):
    """200 outer steps at N=4, no NaN, pos_max < 5m. Small-N smoke."""
    import importlib
    import numpy as np

    factories_mod = importlib.import_module("scripts.adaptive.factories")
    factory = getattr(factories_mod, factory_name)

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    _, step_fn = factory(**kwargs)(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(200):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), f"{factory_name}: NaN in body_q"
    pos_max = float(np.abs(bq[..., :3]).max())
    assert pos_max < 5.0, f"{factory_name}: pos_max={pos_max:.2f}m (scene bound ~1m)"


@pytest.mark.parametrize("use_scalar_dt,expected_dt_scalar_reads", [
    (False, 0),  # MuJoCo-like: no _dt_scalar read per iter (1 PCIe sync)
    (True, "K_done"),  # XPBD/Semi-like: one _dt_scalar read per iter (2 PCIe syncs)
])
def test_step_dt_does_one_pcie_sync_per_iter_on_boundary_flag(
    monkeypatch, use_scalar_dt, expected_dt_scalar_reads
):
    """Each inner iter does exactly 1 host sync (boundary_flag). Matches v1
    CENIC's sync profile. The _pack_loop_status kernel was removed in v2
    Task 13 -- its launch cost outweighed the sync savings.

    With needs_scalar_dt=False (MuJoCo path): 1 sync/iter (_boundary_flag only).
    With needs_scalar_dt=True (XPBD/Semi path): 2 syncs/iter (_dt_scalar + _boundary_flag).
    """
    from scripts.adaptive.base import AdaptiveWrapper

    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()

    def _stepfn(model, sin, sout, ctrl, contacts, dt_array, dt_scalar_buf):
        pass

    w = AdaptiveWrapper(
        model=model, step_fn=_stepfn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
        needs_scalar_dt=use_scalar_dt,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()

    real_numpy = wp.array.numpy
    counts = {"boundary_flag": 0, "dt_scalar": 0}
    def _counting_numpy(self, *a, **kw):
        if self is w._boundary_flag:
            counts["boundary_flag"] += 1
        if hasattr(w, "_dt_scalar") and self is w._dt_scalar:
            counts["dt_scalar"] += 1
        return real_numpy(self, *a, **kw)
    monkeypatch.setattr(wp.array, "numpy", _counting_numpy)

    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()

    K_done = int(w.iteration_count.numpy()[0])  # reads _iteration_count_buf, a different array
    assert counts["boundary_flag"] == K_done, (
        f"expected {K_done} numpy() on _boundary_flag; got {counts['boundary_flag']}"
    )
    # _dt_scalar read: 0 when needs_scalar_dt=False, K_done when True.
    expected = K_done if expected_dt_scalar_reads == "K_done" else 0
    assert counts["dt_scalar"] == expected, (
        f"needs_scalar_dt={use_scalar_dt}: expected {expected} numpy() on _dt_scalar; "
        f"got {counts['dt_scalar']}"
    )


@pytest.mark.parametrize("factory_name,kwargs", [
    ("adaptive_mujoco_factory", dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01, nconmax=8, njmax=32)),
    ("adaptive_xpbd_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
    ("adaptive_semi_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
])
def test_adaptive_variant_stable_at_n_1024(factory_name, kwargs):
    """HARD GATE: each adaptive variant runs 50 outer steps at N=1024 on
    falling_cylinder with no NaN and pos_max < 5m. v1 ships only when this
    passes for all 3 factories."""
    import importlib
    import numpy as np

    factories_mod = importlib.import_module("scripts.adaptive.factories")
    factory = getattr(factories_mod, factory_name)

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(1024)
    _, step_fn = factory(**kwargs)(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(50):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), f"{factory_name}: NaN at N=1024"
    pos_max = float(np.abs(bq[..., :3]).max())
    assert pos_max < 5.0, f"{factory_name}: pos_max={pos_max:.2f}m at N=1024"


def test_adaptive_public_api_after_rename():
    """SolverMuJoCoAdaptive public API uses canonical dt_init/dt_min/dt_max and step()."""
    import newton, newton.solvers
    import warp as wp
    from scripts.scenes import _registry
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)

    solver = newton.solvers.SolverMuJoCoAdaptive(
        m, tol=1e-3, dt_init=0.01, dt_min=1e-6,
        dt_max=0.01, nconmax=8, njmax=32,
    )
    s0, s1, ctrl = m.state(), m.state(), m.control()

    # Method: step (canonical, updates s0 in place)
    solver.step(s0, s1, ctrl, None, 0.01)
    wp.synchronize()

    # Method: get_status_summary
    summary = solver.get_status_summary()
    assert set(summary.keys()) >= {"sim_time_min", "sim_time_max", "error_max",
                                    "accept_count", "dt_min", "dt_max"}

    # Properties
    assert solver.iteration_count.shape == (1,)
    assert solver.dt.shape == (4,)
    assert solver.sim_time.shape == (4,)
    assert solver.last_error.shape == (4,)
    assert solver.accepted.shape == (4,)
    # contacts is a newton.Contacts (not a wp.array); just confirm it exists.
    assert solver.contacts is not None

    # Inherited from SolverMuJoCo
    assert solver.mjw_data is not None
    assert solver.mjw_model is not None

    # Private attrs accessed by scripts (display-only)
    assert hasattr(solver, "_tol")
    assert hasattr(solver, "_dt")
    assert hasattr(solver, "_dt_max")

    # step_dt must NOT be present on the public class
    assert not hasattr(solver, "step_dt"), "step_dt should not be a public method after rename"



def test_xpbd_adaptive_q_qd_norm_is_active():
    """With q+qd norm wired in, XPBD adaptive on falling_cylinder should still
    run cleanly. K may be > 1 now during contact moments (the whole point).
    Verify: 50 outer steps run without NaN or crash, and the wrapper's
    error_norm_fn is the q+qd variant (not the default)."""
    import numpy as np
    from scripts.adaptive.factories import adaptive_xpbd_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    )
    wrapper, step_fn = builder(m)

    # The wrapper's error_norm_fn must NOT be the default q-only.
    assert wrapper.error_norm_fn is not wrapper._default_error_norm_q_only, (
        "adaptive_xpbd_factory should install the q+qd norm, not the q-only default"
    )

    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(50):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    assert not np.isnan(s0.body_q.numpy()).any()
