"""Greenfield bimanual articulation cfg for the Trossen Stationary AI rig.

Authored from ``stationary_ai.usd`` (joint/body names dumped directly from the USD)
following the single-arm ``WXAI_BASE_CFG`` convention
(``trossen_ai_isaac .../manipulation/assets/wxai.py``):

- The active ``left_arm`` uses the Isaac Lab manipulation reference gains (stiffness=80, damping=4,
  as in Franka/OpenArm lift); the USD-baked gains are ~500x stiffer and cause the hold jitter (see
  the actuator comment below). The parked right arm + grippers keep their baked gains (``None``).
- Each gripper actuates only its LEFT carriage joint; the matching RIGHT carriage is a USD
  ``physxMimicJoint`` (gearing -1.0, referenceJoint = left carriage) that mirrors it
  automatically (same gripper hardware as the single-arm WXAI). This leaves 14 actuated DOF
  out of 16 -- the "14 != 16 actuators" warning at load is expected and benign.

This module imports ``isaaclab.*`` at load time, so it can only be imported AFTER
``AppLauncher``/``SimulationApp`` has started (USD ``pxr`` runtime).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ..paths import STATIONARY_AI_USD  # rig USD path; override via $STATIONARY_AI_USD (see trossen_cube/paths.py)

STATIONARY_AI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=STATIONARY_AI_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "follower_left_joint_0": 0.0,
            "follower_left_joint_1": 0.0,
            "follower_left_joint_2": 0.0,
            "follower_left_joint_3": 0.0,
            "follower_left_joint_4": 0.0,
            "follower_left_joint_5": 0.0,
            "follower_left_left_carriage_joint": 0.0,
            "follower_right_joint_0": 0.0,
            "follower_right_joint_1": 0.0,
            "follower_right_joint_2": 0.0,
            "follower_right_joint_3": 0.0,
            "follower_right_joint_4": 0.0,
            "follower_right_joint_5": 0.0,
            "follower_right_left_carriage_joint": 0.0,
        },
    ),
    actuators={
        # Isaac Lab manipulation reference gains (Franka & OpenArm lift both use arm stiffness=80,
        # damping=4). The USD-baked gains are ~40000/340 -- ~500x too stiff, a ~30-100 Hz PD bandwidth
        # that faithfully reproduces the policy's ~18 Hz command micro-oscillation as visible hold
        # jitter (diagnosed in diag_jitter.py). The soft 80/4 PD (~1.5-4.5 Hz bandwidth) cannot follow
        # that chatter, so it smooths it -- which is why the reference arms hold steadily.
        "left_arm": ImplicitActuatorCfg(joint_names_expr=["follower_left_joint_[0-5]"], stiffness=80.0, damping=4.0),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_left_carriage_joint"], stiffness=None, damping=None
        ),
        "right_arm": ImplicitActuatorCfg(joint_names_expr=["follower_right_joint_[0-5]"], stiffness=None, damping=None),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_left_carriage_joint"], stiffness=None, damping=None
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
