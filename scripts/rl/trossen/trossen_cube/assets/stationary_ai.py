"""Greenfield bimanual articulation cfg for the Trossen Stationary AI rig.

Authored from ``stationary_ai.usd`` (joint/body names dumped directly from the USD)
following the single-arm ``WXAI_BASE_CFG`` convention
(``trossen_ai_isaac .../manipulation/assets/wxai.py``):

- PD gains are baked into the USD, so actuator ``stiffness``/``damping`` are ``None``.
- Each gripper actuates only its LEFT carriage joint; the matching RIGHT carriage is
  a USD mimic joint (same gripper hardware as the single-arm WXAI).

This module imports ``isaaclab.*`` at load time, so it can only be imported AFTER
``AppLauncher``/``SimulationApp`` has started (USD ``pxr`` runtime).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Execution happens inside the `isaaclab` container where the trossen repo is mounted
# at /isaac. Override with $STATIONARY_AI_USD if the asset path differs.
STATIONARY_AI_USD = os.environ.get(
    "STATIONARY_AI_USD",
    "/isaac/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd",
)

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
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_joint_[0-5]"], stiffness=None, damping=None
        ),
        "left_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_left_left_carriage_joint"], stiffness=None, damping=None
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_joint_[0-5]"], stiffness=None, damping=None
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["follower_right_left_carriage_joint"], stiffness=None, damping=None
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
