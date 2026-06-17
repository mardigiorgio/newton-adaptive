"""Stationary AI cube-lift manager env (teacher variant).

Subclasses Isaac Lab's ``LiftEnvCfg`` and wires the greenfield Stationary AI rig with
a single active LEFT arm (right arm parked). Observations are split into two groups so
the privileged-state teacher and (Phase 4) vision student train on the same env:

- ``policy``     : proprioception (joint pos/vel + last action) -- shared
- ``privileged`` : exact cube pose + commanded target -- teacher only

All reward/termination/event/command/timing values are inherited from ``LiftEnvCfg``
(see ``scripts/rl/trossen/IMPL_GROUND_TRUTH.md``). Must be imported AFTER ``AppLauncher``.
"""

from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.sim.schemas import RigidBodyPropertiesCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from trossen_cube.assets import STATIONARY_AI_CFG

# Active (controlled) arm = LEFT; the right arm holds its default pose.
ARM_JOINTS = "follower_left_joint_[0-5]"
GRIPPER_JOINT = "follower_left_left_carriage_joint"  # right carriage is a USD mimic
# `follower_left_ee_gripper_link` is a USD frame merged out of the articulation, so the
# closest real articulation body is the wrist link_6. TODO(grasp): refine to the gripper
# tool-center (offset toward `follower_left_gripper_{left,right}`) before full teacher train.
EE_LINK = "follower_left_link_6"
BASE_LINK = "follower_left_base_link"


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception shared by teacher and student."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Teacher-only ground-truth state (the student must not see this)."""

        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class StationaryAiCubeLiftEnvCfg(LiftEnvCfg):
    observations: ObservationsCfg = ObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # robot
        self.scene.robot = STATIONARY_AI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # manipuland: DexCube (~57 mm at scale 0.8), in front of the left arm
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0.0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # actions: active left arm + its (left-carriage) gripper
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[ARM_JOINTS], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=[GRIPPER_JOINT],
            open_command_expr={GRIPPER_JOINT: 0.044},
            close_command_expr={GRIPPER_JOINT: 0.0},
        )

        # command target tied to the active ee; ranges tuned to the left-arm workspace
        self.commands.object_pose.body_name = EE_LINK
        self.commands.object_pose.ranges.pos_x = (0.2, 0.4)
        self.commands.object_pose.ranges.pos_y = (-0.15, 0.15)
        self.commands.object_pose.ranges.pos_z = (0.1, 0.3)

        # ee frame (source = left arm base, target = left ee); sensor MUST be named "ee_frame"
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + BASE_LINK,
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + EE_LINK,
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class StationaryAiCubeLiftEnvCfg_PLAY(StationaryAiCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
