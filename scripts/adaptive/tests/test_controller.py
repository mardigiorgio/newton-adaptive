# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the controller config dataclass."""

from scripts.adaptive.controller import ControllerConfig


def test_default_controller_matches_cenic_drake_constants():
    """Default config matches CENIC's hard-coded constants in
    _calc_adjusted_step (see scripts/adaptive/kernels.py _DRAKE_* constants
    lifted from newton/_src/solvers/mujoco/solver_mujoco_cenic.py)."""
    cfg = ControllerConfig()
    assert cfg.safety_factor == 0.9
    assert cfg.growth_cap == 5.0
    assert cfg.shrink_cap == 0.1
    assert cfg.kp == 0.0  # CENIC's controller is I-only at the moment


def test_controller_config_overridable():
    cfg = ControllerConfig(safety_factor=0.95, growth_cap=2.0)
    assert cfg.safety_factor == 0.95
    assert cfg.growth_cap == 2.0
    assert cfg.shrink_cap == 0.1  # default preserved


import warp as wp


def test_controller_config_overrides_take_effect():
    """A custom ControllerConfig with growth_cap=2.0 limits dt growth to 2x
    per acceptance, while default growth_cap=5.0 allows 5x. Detect by running
    a no-op step_fn (error=0, always accept) and watching dt grow."""
    from scripts.adaptive.base import AdaptiveWrapper
    from scripts.adaptive.controller import ControllerConfig
    from scripts.scenes import _registry

    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()

    def _noop(m, si, so, ctrl, c, dta, dts):
        pass

    # Default: growth_cap=5.0 → first acceptance grows dt by 5x
    w_def = AdaptiveWrapper(
        model=model, step_fn=_noop, tol=1e-3, dt_init=1e-6,
        dt_min=1e-6, dt_max=1.0, dt_outer=1.0,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0, s1 = w_def.step_dt(1.0, s0, s1, ctrl)
    wp.synchronize()
    dt_default_iter1 = float(w_def._ideal_dt.numpy().max())

    # Restrictive: growth_cap=2.0 → first acceptance grows dt by only 2x
    model2 = scene.build_model_randomized(4)
    contacts2 = model2.contacts()
    w_restr = AdaptiveWrapper(
        model=model2, step_fn=_noop, tol=1e-3, dt_init=1e-6,
        dt_min=1e-6, dt_max=1.0, dt_outer=1.0,
        needs_collide=False, contacts=contacts2,
        controller=ControllerConfig(growth_cap=2.0),
    )
    s0r, s1r, ctrlr = model2.state(), model2.state(), model2.control()
    s0r, s1r = w_restr.step_dt(1.0, s0r, s1r, ctrlr)
    wp.synchronize()
    dt_restr_iter1 = float(w_restr._ideal_dt.numpy().max())

    # default config should grow dt > 2x faster than the restrictive one
    assert dt_default_iter1 > dt_restr_iter1 * 2.0, (
        f"default ({dt_default_iter1}) should be > 2x restrictive ({dt_restr_iter1})"
    )
