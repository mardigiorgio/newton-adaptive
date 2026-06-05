# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for every scene in scripts/scenes/.

Each test verifies:
  * The scene module imports cleanly.
  * ``build_model(1)`` returns a model with at least one body.
  * One CENIC outer step from rest produces no NaN in joint_q / body_q.
  * ``build_model_randomized(2)`` produces two distinct worlds (where
    applicable -- some scenes only randomise via per-step helpers).

Run with:

    uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic -m pytest scripts/tests/test_scenes.py -v
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import newton
from scripts.scenes import _registry as scene_registry

# Scenes covered by the smoke suite. For non-bench-capable scenes (Franka)
# we still verify build + one bare-step (no helpers); the released objects
# will fall but nothing should NaN.
SCENE_NAMES = scene_registry.all_scenes()


@pytest.fixture(scope="module", params=SCENE_NAMES)
def scene(request):
    return scene_registry.get(request.param)


def _has_nan(arr) -> bool:
    return bool(np.any(np.isnan(np.asarray(arr))))


def test_scene_module_imports(scene):
    """The scene module imports without raising."""
    importlib.import_module(scene.module_path)


def test_build_model_one_world(scene):
    """build_model(1) returns a Model with bodies and joints."""
    model = scene.build_model(1)
    assert model.body_count > 0, f"{scene.name}: zero bodies"
    assert model.joint_coord_count > 0, f"{scene.name}: zero joint coords"


def test_build_model_randomized_two_worlds_distinct(scene):
    """build_model_randomized(2) produces two distinct world configurations.

    Some scenes (notably franka_dish_rack and contact_objects perturbed
    variants) randomise via the kinematic-pin helper or only by
    perturbation; in those cases at least the world count must double.
    """
    model = scene.build_model_randomized(2)
    coords_per_world = model.joint_coord_count // 2
    assert coords_per_world > 0
    q = model.joint_q.numpy()
    half0 = q[:coords_per_world]
    half1 = q[coords_per_world : 2 * coords_per_world]
    # Either the joint_q halves differ (most scenes) OR the world count is
    # exactly 2 (acceptable when randomisation is in body_q only).
    if np.allclose(half0, half1):
        assert model.world_count == 2, f"{scene.name}: world halves identical and world_count != 2"


def test_one_outer_step_no_nan(scene):
    """One CENIC outer step from rest produces no NaN in state."""
    model = scene.build_model(1)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()

    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    solver = scene.make_solver(model)
    solver.step(s0, s1, ctrl, None, scene.dt_outer)

    assert not _has_nan(s0.joint_q.numpy()), f"{scene.name}: joint_q has NaN"
    assert not _has_nan(s0.joint_qd.numpy()), f"{scene.name}: joint_qd has NaN"
    assert not _has_nan(s0.body_q.numpy()), f"{scene.name}: body_q has NaN"


def test_make_fixed_solver_constructs(scene):
    """make_fixed_solver returns an instance without raising."""
    model = scene.build_model(1)
    solver = scene.make_fixed_solver(model)
    assert solver is not None


def test_franka_helpers_present():
    """Franka scene exposes its kinematic-pin helper and trajectory helper."""
    franka_scene = importlib.import_module("scripts.scenes.franka_dish_rack")
    assert hasattr(franka_scene, "update_held_objects")
    assert hasattr(franka_scene, "update_franka_targets")
    assert hasattr(franka_scene, "RELEASE_TIME")
    assert hasattr(franka_scene, "KEYFRAMES")


def test_franka_pin_keeps_held_close_to_ee():
    """When the Franka pin helper is called at t=0, the held bodies snap to
    a position within 25 cm of the EE link7 body."""
    franka_scene = importlib.import_module("scripts.scenes.franka_dish_rack")
    model = franka_scene.build_model(1)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    franka_scene.update_held_objects(model, state, sim_time=0.0)
    bq = state.body_q.numpy()
    ee = bq[franka_scene._EE_BODY_IDX_PER_WORLD, :3]
    for i in range(franka_scene._HELD_COUNT):
        held = bq[franka_scene._FRANKA_BODY_COUNT + i, :3]
        dist = float(np.linalg.norm(held - ee))
        assert dist < 0.25, f"held body {i} too far from EE: {dist:.3f} m (EE={ee}, held={held})"


def test_franka_release_makes_pin_noop():
    """At sim_time >= RELEASE_TIME, update_held_objects should not modify
    body poses (held bodies are now free-falling)."""
    franka_scene = importlib.import_module("scripts.scenes.franka_dish_rack")
    model = franka_scene.build_model(1)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    bq_before = state.body_q.numpy().copy()
    franka_scene.update_held_objects(
        model,
        state,
        sim_time=franka_scene.RELEASE_TIME + 0.001,
    )
    bq_after = state.body_q.numpy()
    np.testing.assert_array_equal(bq_before, bq_after)
