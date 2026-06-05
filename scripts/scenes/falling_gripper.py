# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Falling gripper scene: parallel-jaw gripper on a vertical rail, holding a box.

Geometry (all primitives):
  * Rail (static, body=-1): tall thin box, visual + reference geometry only.
    Filtered out of carriage collision so the rail does not push the carriage.
  * Carriage ("wrist"): small box on a vertical prismatic joint to world.
  * Two finger boxes: thin boxes attached to the carriage by mirrored
    prismatic joints along +/-x. Joint targets driven by target_ke / target_kd
    so the fingers maintain a constant pinch on the held box.
  * Held box (free 6-DOF body): clamped between the fingers by friction +
    pinch normal force, no kinematic attachment.

Tests frictional grasping under acceleration (gravity), prismatic joint
dynamics, and impact at the rail bottom.

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import numpy as np
import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01
TOL = 1e-3
DT_INNER_MIN = 1e-7
LOG_EVERY = 250

# --- Geometry [m] -----------------------------------------------------------

RAIL_HX = 0.05
RAIL_HY = 0.05
RAIL_HZ = 0.6
RAIL_CENTER = wp.vec3(0.0, -0.10, RAIL_HZ)  # base on ground

CARRIAGE_HALF = 0.04  # cube body
CARRIAGE_Q_INIT = 0.8
CARRIAGE_Q_MIN = 0.0
CARRIAGE_Q_MAX = 1.0

# Held cube: the prominent "object being held" -- sized so the fingers
# visibly wrap around it rather than fusing with it.
HELD_HX = 0.025
HELD_HY = 0.025
HELD_HZ = 0.025

# Fingers: visibly thick parallel-jaw arms extending down from the carriage.
# FINGER_HY kept small enough that the finger's -y face sits clear of the
# rail's +y face (rail is at y=-0.10 with hy=0.05, so its +y face is at
# y=-0.05). At FINGER_HY=0.03 the finger -y face is at y=-0.03, leaving a
# 2 cm gap so the fingers don't scrape the rail.
FINGER_HX = 0.010  # 2 cm thick fingers -- read clearly as "arms"
FINGER_HY = 0.03
FINGER_HZ = 0.06   # 12 cm tall, fully bracket the 5 cm tall held cube

# Vertical offset of the finger pivot below the carriage center [m].  Must
# satisfy FINGER_Z_OFFSET >= CARRIAGE_HALF + FINGER_HZ so the fingers sit
# cleanly below the carriage bottom face -- otherwise the finger top edge
# overlaps the carriage and the contact constraint holds the carriage up
# against gravity.  Adds 1 cm clearance.
FINGER_Z_OFFSET = CARRIAGE_HALF + 0.06 + 0.01  # 0.11

# Finger joint q (slide distance from carriage center along its axis [m]).
# Held cube outer face is at HELD_HX; finger inner face at q - FINGER_HX.
# Target value gives 1 mm pinch overlap; init value gives 1 mm clearance so
# the scene starts with no penetration.
FINGER_Q_INIT = HELD_HX + FINGER_HX + 0.001    # 0.036 -- 1 mm clear at startup
FINGER_Q_TARGET = HELD_HX + FINGER_HX - 0.001  # 0.034 -- 1 mm pinch overlap
FINGER_Q_LIMIT_UPPER = 0.06                    # fully open

# Pinch controller gains: stiff enough to hold against gravity and impact,
# soft enough not to explode the held box. dampratio kd/(2 sqrt(ke)) ~ 1.0.
FINGER_TARGET_KE = 1e4
FINGER_TARGET_KD = 2.0e2

# Collision groups: rail (-2) is filtered from everything; carriage / fingers
# / held box share group +1 so they collide with each other and the ground.
GROUP_RAIL = -2
GROUP_DYN = 1


def build_template() -> newton.ModelBuilder:
    """Single-world template: rail + carriage + 2 fingers + held box.

    Joint layout (10 coords per world):
      [0]   carriage prismatic q  (z-slide from world)
      [1]   finger +x prismatic q (x-slide from carriage)
      [2]   finger -x prismatic q (-x-slide from carriage)
      [3]   held box pos x
      [4]   held box pos y
      [5]   held box pos z
      [6]   held box quat x
      [7]   held box quat y
      [8]   held box quat z
      [9]   held box quat w
    """
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)

    cfg_rail = newton.ModelBuilder.ShapeConfig(
        ke=1e4,
        kd=200,
        mu=0.0,
        margin=5e-3,
        collision_group=GROUP_RAIL,
    )
    cfg_carriage = newton.ModelBuilder.ShapeConfig(
        ke=1e4,
        kd=200,
        mu=0.3,
        margin=5e-3,
        collision_group=GROUP_DYN,
    )
    cfg_finger = newton.ModelBuilder.ShapeConfig(
        ke=1e4,
        kd=200,
        mu=1.5,
        margin=1e-3,
        collision_group=GROUP_DYN,
    )
    cfg_held = newton.ModelBuilder.ShapeConfig(
        ke=1e4,
        kd=200,
        mu=1.5,
        margin=1e-3,
        collision_group=GROUP_DYN,
    )

    # Rail: visual / reference only (body=-1 -> world/static).
    template.add_shape_box(
        body=-1,
        xform=wp.transform(p=RAIL_CENTER, q=wp.quat_identity()),
        hx=RAIL_HX,
        hy=RAIL_HY,
        hz=RAIL_HZ,
        cfg=cfg_rail,
    )

    # --- Gripper articulation: carriage + 2 fingers -------------------------
    # Use add_link (not add_body) to defer joint creation and group them into
    # one articulation.  add_body would auto-create a free joint per body.
    # NOTE: link xform is ignored once a joint is attached -- body pose is
    # determined entirely by parent_world * parent_xform * (q * axis) *
    # inv(child_xform). Initial poses are set via joint_q below.

    carriage = template.add_link(xform=wp.transform_identity())
    template.add_shape_box(
        carriage,
        hx=CARRIAGE_HALF,
        hy=CARRIAGE_HALF,
        hz=CARRIAGE_HALF,
        cfg=cfg_carriage,
    )

    # Carriage slides on world along +z.
    j_carriage = template.add_joint_prismatic(
        parent=-1,
        child=carriage,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(0.0, 0.0, 1.0),
        limit_lower=CARRIAGE_Q_MIN,
        limit_upper=CARRIAGE_Q_MAX,
    )

    finger_joints = []
    for sign in (+1.0, -1.0):
        finger = template.add_link(xform=wp.transform_identity())
        template.add_shape_box(
            finger,
            hx=FINGER_HX,
            hy=FINGER_HY,
            hz=FINGER_HZ,
            cfg=cfg_finger,
        )
        j = template.add_joint_prismatic(
            parent=carriage,
            child=finger,
            parent_xform=wp.transform(
                p=wp.vec3(0.0, 0.0, -FINGER_Z_OFFSET), q=wp.quat_identity(),
            ),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(sign, 0.0, 0.0),
            limit_lower=0.0,
            limit_upper=FINGER_Q_LIMIT_UPPER,
            target_pos=FINGER_Q_TARGET,
            target_ke=FINGER_TARGET_KE,
            target_kd=FINGER_TARGET_KD,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        finger_joints.append(j)

    # Register all three joints as one articulation.
    template.add_articulation([j_carriage, *finger_joints])

    # --- Held box: free-floating body (7 DOF) --------------------------------
    held = template.add_body(xform=wp.transform_identity())
    template.add_shape_box(
        held,
        hx=HELD_HX,
        hy=HELD_HY,
        hz=HELD_HZ,
        cfg=cfg_held,
    )

    # --- Initial joint q ----------------------------------------------------
    # Joint coord layout (set by builder in joint creation order):
    #   [0]      carriage prismatic q  -> world z
    #   [1]      finger +x prismatic q
    #   [2]      finger -x prismatic q
    #   [3..9]   held box free joint (px py pz qx qy qz qw)
    template.joint_q[0] = CARRIAGE_Q_INIT
    template.joint_q[1] = FINGER_Q_INIT
    template.joint_q[2] = FINGER_Q_INIT
    template.joint_q[3] = 0.0
    template.joint_q[4] = 0.0
    template.joint_q[5] = CARRIAGE_Q_INIT - FINGER_Z_OFFSET  # held box hangs below carriage
    template.joint_q[6] = 0.0
    template.joint_q[7] = 0.0
    template.joint_q[8] = 0.0
    template.joint_q[9] = 1.0

    return template


def build_model(n_worlds: int) -> newton.Model:
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    return builder.finalize()


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds. Randomizes carriage starting q and held-box yaw per world."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds

    # Joint layout per world (matches build_template docstring):
    #   [0]      carriage prismatic q
    #   [1]      finger +x prismatic q
    #   [2]      finger -x prismatic q
    #   [3..9]   held box free-joint (3 pos + 4 quat)
    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        base = w * coords_per_world

        carriage_q = rng.uniform(0.6, 0.95)
        joint_q_np[base + 0] = carriage_q

        # Fingers always start at the same just-clear-of-box position.
        joint_q_np[base + 1] = FINGER_Q_INIT
        joint_q_np[base + 2] = FINGER_Q_INIT

        # Held box centered between the fingers, axis-aligned. Random yaw
        # creates off-axis pinch forces that slowly squeeze the box out
        # laterally, so we keep yaw at zero -- per-world variation comes
        # from carriage_q only.
        joint_q_np[base + 3] = 0.0
        joint_q_np[base + 4] = 0.0
        joint_q_np[base + 5] = carriage_q - FINGER_Z_OFFSET  # held box hangs below carriage
        joint_q_np[base + 6] = 0.0
        joint_q_np[base + 7] = 0.0
        joint_q_np[base + 8] = 0.0
        joint_q_np[base + 9] = 1.0

    model.joint_q.assign(joint_q_np)

    # body_q follows from forward kinematics; rebuild it via newton.eval_fk.
    state = model.state()
    state.joint_q.assign(joint_q_np)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    model.body_q.assign(state.body_q.numpy())

    return model


# mjwarp multiplies nconmax/njmax by nworld internally — pass per-world values.
_NCON = 30
_NJM = 80


def make_solver(
    model: newton.Model,
    tol: float = TOL,
) -> newton.solvers.SolverMuJoCoAdaptive:
    return newton.solvers.SolverMuJoCoAdaptive(
        model, tol=tol, dt_init=0.005, dt_min=DT_INNER_MIN,
        dt_max=DT_OUTER, nconmax=_NCON, njmax=_NJM,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=_NCON, njmax=_NJM,
    )


from scripts.scenes import _solvers as _s  # noqa: E402

# Featherstone NaNs on this scene (gripper articulation + box contact).
# XPBD works but accuracy is poor on the articulated rail+carriage chain.
SOLVER_FACTORIES: dict = {
    "mujoco_adaptive_1e-3": _s.mujoco_adaptive_factory(
        tol=1e-3, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
        dt_inner_max=DT_OUTER,
    ),
    "mujoco_adaptive_1e-2": _s.mujoco_adaptive_factory(
        tol=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
        dt_inner_max=DT_OUTER,
    ),
    "mujoco_fixed_1ms": _s.mujoco_fixed_factory(
        dt=1e-3, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    "mujoco_fixed_10ms": _s.mujoco_fixed_factory(
        dt=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    "xpbd_1ms": _s.xpbd_factory(dt=1e-3, dt_outer=DT_OUTER),
}
