# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The CENIC paper's benchmark scenes (Kurtz & Castro, arXiv:2511.08771),
rebuilt on Newton for the four-arm Part-1 benchmarks.

Reproduced from the paper's Sec. VII and Fig. 6/8 text:

* ``soft-clutter`` — "dropping 20 objects into a bin ... only spheres, a
  contact stiffness of 10^3 N/m, and a large friction regularization
  (v_s = 1 cm/s)".
* ``hard-clutter`` — "spheres and cubes, with stiffer contact parameters
  approximating rigid Coulomb friction: contact stiffness 10^5 N/m and
  stiction tolerance 0.1 mm/s".
* ``ball`` — Fig. 8: "a 0.1 kg bouncing ball with zero dissipation ...
  contact stiffness 10^3 N/m ... dropped from an initial height of 1 m ...
  bounces 11 times in the 10 second simulation. Potential energy is defined
  such that total energy is zero when the ball is at rest on the ground."

Quantities the paper does NOT state are marked ASSUMED below (object size,
bin size, friction coefficient, Hunt & Crossley dissipation for clutter,
initial lattice) — confirm with the authors before publication.

Stiffness/dissipation/stiction reach the two backends differently:
MuJoCo reads per-shape ``ke``/``kd`` (Newton converts them to solref);
ICF takes global ``IcfParams`` (``contact_stiffness``,
``contact_hc_dissipation``, ``contact_stiction_tolerance``) — each scene
therefore carries both, and ``make_arm`` applies the ICF overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import warp as wp

import newton

DT_OUTER = 0.01  # maximum step / control boundary [s]

# Point contact: force only at penetration (phi < 0). A shape margin would
# inflate every surface -- ICF applies its law at (distance - margin), so
# bodies would settle ON the margin skin and read "zero penetration" by
# construction; MuJoCo's contact activation would shift likewise.
CONTACT_MARGIN = 0.0

# ASSUMED geometry (not stated in the paper)
CLUTTER_SPHERE_R = 0.025
CLUTTER_CUBE_HALF = 0.025
BIN_HALF = 0.15
BIN_WALL_T = 0.02
BIN_WALL_H = 0.30
CLUTTER_MU = 0.5
BALL_R = 0.05
BALL_MASS = 0.1
BALL_DROP = 1.0
LATTICE_SEED = 7


@dataclass
class SceneSpec:
    name: str
    build: Callable[[int], newton.Model]
    icf: dict = field(default_factory=dict)  # IcfParams overrides
    horizon_s: float = 2.0  # simulated seconds for work-precision runs
    note: str = ""


def _finish(builder: newton.ModelBuilder) -> newton.Model:
    builder.add_ground_plane()
    return builder.finalize()


def _add_bin(builder: newton.ModelBuilder, cfg: newton.ModelBuilder.ShapeConfig) -> None:
    hw, t, h = BIN_HALF, BIN_WALL_T, BIN_WALL_H
    for px, py, hx, hy in [
        (-(hw + t), 0.0, t, hw + t),
        (hw + t, 0.0, t, hw + t),
        (0.0, -(hw + t), hw + t, t),
        (0.0, hw + t, hw + t, t),
    ]:
        builder.add_shape_box(
            body=-1, xform=wp.transform(p=wp.vec3(px, py, h), q=wp.quat_identity()), hx=hx, hy=hy, hz=h, cfg=cfg
        )


def _clutter_template(hard: bool) -> newton.ModelBuilder:
    # ke: the paper's k. kd: ASSUMED at kd = 0.02 * ke (the repo's demo
    # scene ratio); the paper states no dissipation for clutter.
    ke = 1.0e5 if hard else 1.0e3
    cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=0.02 * ke, mu=CLUTTER_MU, margin=CONTACT_MARGIN, density=1000.0)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    # Initial arrangement: 4 columns x 5 layers above the bin, alternate
    # layers staggered by half the column spacing, every body jittered
    # (+-1.5 cm in xy, +-5 mm in z) and every cube tilted by a random
    # rotation -- a fixed seed, so the scene is one deterministic drop.
    # Perfectly aligned columns would land as columns, not as clutter.
    import math
    import random

    rng = random.Random(LATTICE_SEED)
    i = 0
    for layer in range(5):
        shift = 0.03 if layer % 2 else 0.0
        for cx, cy in ((-0.06, -0.06), (0.06, -0.06), (-0.06, 0.06), (0.06, 0.06)):
            x = cx + shift + rng.uniform(-0.015, 0.015)
            y = cy + shift + rng.uniform(-0.015, 0.015)
            z = 0.12 + 0.07 * layer + rng.uniform(-0.005, 0.005)
            q = wp.quat_identity()
            if hard and i % 2 == 1:
                ax = wp.vec3(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))
                q = wp.quat_from_axis_angle(wp.normalize(ax), rng.uniform(0.0, math.pi))
            b = t.add_body(xform=wp.transform(p=wp.vec3(x, y, z), q=q))
            if hard and i % 2 == 1:
                t.add_shape_box(b, hx=CLUTTER_CUBE_HALF, hy=CLUTTER_CUBE_HALF, hz=CLUTTER_CUBE_HALF, cfg=cfg)
            else:
                t.add_shape_sphere(b, radius=CLUTTER_SPHERE_R, cfg=cfg)
            i += 1
    return t


def build_clutter(n_worlds: int, hard: bool) -> newton.Model:
    template = _clutter_template(hard)
    ke = 1.0e5 if hard else 1.0e3
    wall_cfg = newton.ModelBuilder.ShapeConfig(ke=ke, kd=0.02 * ke, mu=CLUTTER_MU, margin=CONTACT_MARGIN, is_visible=False)
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    _add_bin(builder, wall_cfg)
    return _finish(builder)


def build_ball(n_worlds: int) -> newton.Model:
    density = BALL_MASS / (4.0 / 3.0 * 3.141592653589793 * BALL_R**3)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e3, kd=0.0, mu=0.0, margin=CONTACT_MARGIN, density=density)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    b = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, BALL_DROP + BALL_R), q=wp.quat_identity()))
    t.add_shape_sphere(b, radius=BALL_R, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(t, n_worlds)
    return _finish(builder)


SCENES: dict[str, SceneSpec] = {
    "soft-clutter": SceneSpec(
        "soft-clutter",
        lambda n: build_clutter(n, hard=False),
        icf={"contact_stiffness": 1.0e3, "contact_stiction_tolerance": 1.0e-2},
        note="20 spheres in a bin, k=1e3 N/m, v_s=1 cm/s",
    ),
    "hard-clutter": SceneSpec(
        "hard-clutter",
        lambda n: build_clutter(n, hard=True),
        icf={"contact_stiffness": 1.0e5, "contact_stiction_tolerance": 1.0e-4},
        note="10 spheres + 10 cubes in a bin, k=1e5 N/m, v_s=0.1 mm/s",
    ),
    "ball": SceneSpec(
        "ball",
        build_ball,
        icf={"contact_stiffness": 1.0e3, "contact_hc_dissipation": 0.0},
        horizon_s=10.0,
        note="0.1 kg ball, k=1e3 N/m, zero dissipation, 1 m drop",
    ),
}


def ball_energy(model: newton.Model, state) -> "list[float]":
    """Per-world total energy [J] of the ball scene, zero at rest on the
    ground: 0.5 m |v|^2 + m g (z - r)."""
    import numpy as np

    g = float(-model.gravity.numpy()[0][2])
    m = model.body_mass.numpy()
    q = state.body_q.numpy().reshape(-1, 7)
    qd = state.joint_qd.numpy().reshape(-1, 6)  # free joint: [linear, angular]
    ke = 0.5 * m * (qd[:, :3] ** 2).sum(axis=1)
    pe = m * g * (q[:, 2] - BALL_R)
    return (ke + pe).tolist()


def ball_initial_energy(model: newton.Model) -> float:
    g = float(-model.gravity.numpy()[0][2])
    return float(model.body_mass.numpy()[0]) * g * BALL_DROP
