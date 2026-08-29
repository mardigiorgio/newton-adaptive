# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actuated stiff-contact scene: a PD-driven prismatic gantry (x, z) carrying
a light fingertip presses a box into a table and slides it.

Design from the literature review (scripts/bench/results/PART1_LITERATURE.md,
Theme E): the stiffness comes from the controller (CENIC Fig. 12's axis,
K_p swept, K_d = 2 sqrt(K_p m) critically damped as Drake's k_d = 2 sqrt(k_p))
and from a light mass under high gain (Drake #14694); the box is SAP's
clutter body (1 kg, 10 cm); mu = 0.5; the press-and-slide program follows the
pushing datasets (quasi-static 50 mm/s, dynamic 300 mm/s).

Fingertip: 0.1 kg sphere, r = 1 cm, on the z carriage, starting beside the
box at the box's mid-height plus a clearance. Program (per boundary the
caller sets ``control.joint_target_q``): descend to the box's mid-height,
push the box from the side SLIDE_LEN on a trapezoidal profile at
SLIDE_SPEED (the pushing datasets' side push), stop, settle SETTLE_S. The
push closes the GAP first, so the tip meets the box at speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

import newton
from scripts.scenes.cenic_scenes import CLUTTER_HC_DISSIPATION, CONTACT_MARGIN, MUJOCO_TAU_K1E5

BOX_HALF = 0.05
BOX_MASS = 1.0
TIP_R = 0.01
TIP_MASS = 0.1
MU = 0.5
GAP = 0.02  # tip surface to box face at the start
SLIDE_LEN = 0.30
SETTLE_S = 1.0
T_DESCEND = 0.5
Z_CLEAR = 0.05  # fingertip start height above the box's mid-height
X0 = -(BOX_HALF + TIP_R + GAP)  # fingertip centre x at the start (box centred at 0)


@dataclass
class ActuatedSpec:
    k: float  # contact stiffness [N/m]
    kp: float  # PD position gain [N/m]
    slide_speed: float  # [m/s]

    @property
    def kd(self) -> float:
        return 2.0 * math.sqrt(self.kp * TIP_MASS)

    @property
    def icf(self) -> dict:
        return {"contact_stiffness": self.k, "contact_stiction_tolerance": 1e-4, "contact_hc_dissipation": CLUTTER_HC_DISSIPATION}

    @property
    def mujoco_solref(self) -> tuple[float, float]:
        return (MUJOCO_TAU_K1E5 * math.sqrt(1e5 / self.k), 1.0)

    @property
    def horizon_s(self) -> float:
        return T_DESCEND + 0.2 + SLIDE_LEN / self.slide_speed + 0.2 + SETTLE_S


def build(n_worlds: int, spec: ActuatedSpec) -> newton.Model:
    density_box = BOX_MASS / (2 * BOX_HALF) ** 3
    density_tip = TIP_MASS / (4.0 / 3.0 * math.pi * TIP_R**3)
    cfg = newton.ModelBuilder.ShapeConfig(ke=spec.k, kd=0.02 * spec.k, mu=MU, margin=CONTACT_MARGIN)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    # the box, free on the table
    box = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, BOX_HALF), q=wp.quat_identity()))
    t.add_shape_box(
        box,
        hx=BOX_HALF,
        hy=BOX_HALF,
        hz=BOX_HALF,
        cfg=newton.ModelBuilder.ShapeConfig(
            ke=spec.k, kd=0.02 * spec.k, mu=MU, margin=CONTACT_MARGIN, density=density_box
        ),
    )
    # gantry: world -> prismatic x (carriage) -> prismatic z (fingertip); one
    # articulation, links added without the free joint add_body would create
    z_start = BOX_HALF + Z_CLEAR
    carriage = t.add_link(xform=wp.transform(p=wp.vec3(X0, 0.0, z_start), q=wp.quat_identity()), mass=0.5)
    jx = t.add_joint_prismatic(
        -1,
        carriage,
        axis=newton.Axis.X,
        parent_xform=wp.transform(p=wp.vec3(X0, 0.0, z_start), q=wp.quat_identity()),
        target_ke=spec.kp,
        target_kd=spec.kd,
        limit_lower=-1.0,
        limit_upper=1.0,
    )
    tip = t.add_link(xform=wp.transform(p=wp.vec3(X0, 0.0, z_start), q=wp.quat_identity()))
    t.add_shape_sphere(
        tip,
        radius=TIP_R,
        cfg=newton.ModelBuilder.ShapeConfig(
            ke=spec.k, kd=0.02 * spec.k, mu=MU, margin=CONTACT_MARGIN, density=density_tip
        ),
    )
    jz = t.add_joint_prismatic(
        carriage, tip, axis=newton.Axis.Z, target_ke=spec.kp, target_kd=spec.kd, limit_lower=-1.0, limit_upper=1.0
    )
    t.add_articulation([jx, jz])
    b = newton.ModelBuilder()
    b.replicate(t, n_worlds)
    b.add_ground_plane()
    return b.finalize()


def program(t: float, spec: ActuatedSpec) -> tuple[float, float]:
    """Joint targets (x displacement, z displacement) of the gantry at time t.
    Descend to the box's mid-height first, then a trapezoidal push (accelerate 0.1 s, cruise, decelerate 0.1 s), then hold."""
    z_press = -Z_CLEAR
    if t < T_DESCEND:
        return 0.0, z_press * (t / T_DESCEND)
    t1 = t - T_DESCEND - 0.2
    if t1 < 0:
        return 0.0, z_press
    ta = 0.1
    v = spec.slide_speed
    x_acc = 0.5 * v * ta
    t_cruise = (SLIDE_LEN - 2 * x_acc) / v
    if t1 < ta:
        x = 0.5 * v / ta * t1 * t1
    elif t1 < ta + t_cruise:
        x = x_acc + v * (t1 - ta)
    elif t1 < 2 * ta + t_cruise:
        td = t1 - ta - t_cruise
        x = x_acc + v * t_cruise + v * td - 0.5 * v / ta * td * td
    else:
        x = SLIDE_LEN
    return x, z_press
