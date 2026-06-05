# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Falling cylinder scene: a single cylinder dropped onto the ground.

Tests CENIC adaptive stepping during edge-rolling contact for a curved
primitive (geometrically harder than a sphere because the contact point
sweeps along an edge as the cylinder rolls).

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01  # 100 Hz control / render cadence [s]
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

CYL_RADIUS = 0.05
CYL_HALF_HEIGHT = 0.10
DROP_Z_DEFAULT = 0.5
TILT_RAD = math.radians(15.0)


def _tilt_quat(angle_rad: float) -> wp.quat:
    """Rotation about +x by angle_rad so the cylinder lands on its rim."""
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle_rad)


def build_template() -> newton.ModelBuilder:
    """Single-world template: one tilted cylinder dropped above the ground."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)

    cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.4, margin=5e-3)
    body = template.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, DROP_Z_DEFAULT), q=_tilt_quat(TILT_RAD)),
    )
    template.add_shape_cylinder(
        body,
        radius=CYL_RADIUS,
        half_height=CYL_HALF_HEIGHT,
        cfg=cfg,
    )
    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    # Required by SolverVBD; harmless for other solvers.
    builder.color()
    return builder.finalize()


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    """Shoemake's uniform random quaternion."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (s1 * math.sin(a1), s1 * math.cos(a1), s2 * math.sin(a2), s2 * math.cos(a2))


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds with per-world random xy, z, and orientation. Seeded per world."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    body_q_np = model.body_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        for b in range(bodies_per_world):
            x = rng.uniform(-0.2, 0.2)
            y = rng.uniform(-0.2, 0.2)
            z = rng.uniform(0.4, 0.8)
            qx, qy, qz, qw = _random_unit_quaternion(rng)

            base = w * coords_per_world + b * 7
            joint_q_np[base + 0] = x
            joint_q_np[base + 1] = y
            joint_q_np[base + 2] = z
            joint_q_np[base + 3] = qx
            joint_q_np[base + 4] = qy
            joint_q_np[base + 5] = qz
            joint_q_np[base + 6] = qw

            body_idx = w * bodies_per_world + b
            body_q_np[body_idx] = (x, y, z, qx, qy, qz, qw)

    model.joint_q.assign(joint_q_np)
    model.body_q.assign(body_q_np)
    return model


# mjwarp multiplies nconmax/njmax by nworld internally — pass per-world values.
_NCON = 8
_NJM = 32


def make_solver(
    model: newton.Model,
    tol: float = TOL,
) -> newton.solvers.SolverMuJoCoAdaptive:
    """CENIC solver with falling-cylinder defaults."""
    return newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=tol,
        dt_init=DT_OUTER,
        dt_min=DT_INNER_MIN,
        dt_max=DT_OUTER,
        nconmax=_NCON,
        njmax=_NJM,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=_NCON, njmax=_NJM,
    )


from scripts.scenes import _solvers as _s  # noqa: E402
from scripts.adaptive import factories as _af  # noqa: E402

SOLVER_FACTORIES: dict = {
    "mujoco_adaptive_1e-3": _af.adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=_NCON, njmax=_NJM,
    ),
    "mujoco_adaptive_1e-2": _s.mujoco_adaptive_factory(
        tol=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    "mujoco_fixed_1ms": _s.mujoco_fixed_factory(
        dt=1e-3, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    "mujoco_fixed_10ms": _s.mujoco_fixed_factory(
        dt=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    # Featherstone NaNs at first ground contact regardless of dt or contact
    # stiffness -- explicit integration can't handle the tilted-cylinder edge
    # impact in this scene. Dropped.
    "semi_implicit_1ms": _s.semi_implicit_factory(dt=1e-3, dt_outer=DT_OUTER),
    "xpbd_1ms": _s.xpbd_factory(dt=1e-3, dt_outer=DT_OUTER),
    "vbd_1ms": _s.vbd_factory(dt=1e-3, dt_outer=DT_OUTER),
    "xpbd_adaptive_1e-3": _af.adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
    "semi_adaptive_1e-3": _af.adaptive_semi_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
}
