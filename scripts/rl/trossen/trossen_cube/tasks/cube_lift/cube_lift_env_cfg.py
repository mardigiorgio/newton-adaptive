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

import os

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
# Gripper: actuate only the LEFT carriage. The RIGHT carriage is a PhysX mimic joint baked
# into the USD (``physxMimicJoint``, gearing -1.0, referenceJoint = left carriage), so it
# mirrors the left finger automatically. The benign "14 != 16 actuators" warning is exactly
# these two mimic carriages (one per arm) -- this matches the single-arm WXAI gripper.
GRIPPER_JOINT = "follower_left_left_carriage_joint"
# End-effector. ``follower_left_ee_gripper_link`` exists in the USD but is NOT an enumerated
# articulation body (no joint links it into the articulation), so it cannot be a command
# ``body_name`` nor resolved by ``robot.find_bodies`` -- only the wrist ``follower_left_link_6``
# is available there. We reference link_6 for the articulation body and place the real grasp
# frame via a fixed offset: the TCP (== follower_left_ee_gripper_link) sits 0.1561 m ahead of
# link_6 along its local x (measured from the USD). The reach reward uses link_6 + this offset,
# so it targets the true tool-center, not the wrist (which would bias the reward ~16 cm high).
EE_LINK = "follower_left_link_6"
# 0.1561 (= `ee_gripper_link`) put the reach TCP ~7cm IN FRONT of the fingers (measured
# TCP->finger_mid = 6.96cm by diag_grasp_geom.py) -> the reach reward landed the cube 7cm short of
# the fingers, so the gripper could never close. The real grasp point is the finger MIDPOINT:
# link_6 -> finger_mid is ~0.087 along link_6's local x.
EE_TCP_OFFSET = (0.087, 0.0, 0.0)
BASE_LINK = "follower_left_base_link"
# Rails removed. The rig's ``frame_link`` (perimeter rail border + camera gantry) is one collision
# body that a lift policy exploits as a crutch -- it learns to jam the cube against the rail instead
# of grasping cleanly. ``stationary_ai_norails.usda`` is a thin override (make_norails_usd.py) that
# deactivates frame_link's collision and hides its visual; the arms and tabletop are untouched.
NORAILS_USD = os.environ.get(
    "STATIONARY_AI_NORAILS_USD",
    "/isaac/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai_norails.usda",
)


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

        # The bimanual rig (2 arms + grippers + frame + tabletop, self-collisions on) at high env
        # counts overflows the default GPU contact buffers -> PhysX silently drops contacts -> the
        # cube tunnels through the tabletop and ~all episodes end via object_dropping. Raise the
        # GPU collision-buffer capacities so contacts survive at 1-2k envs. (Default patch count
        # 5*2**15=163840 < the ~256k this scene needs.)
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26

        # robot (no-rails override -- see NORAILS_USD)
        self.scene.robot = STATIONARY_AI_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=STATIONARY_AI_CFG.spawn.replace(usd_path=NORAILS_USD),
        )

        # manipuland: DexCube. CRITICAL: it must be WIDER than the gripper's CLOSED finger gap
        # (4.83 cm, measured by diag_grasp_geom.py) or the fingers close right past it without touching.
        # DexCube native is 6 cm; scale 0.8 = 4.8 cm was TOO SMALL (0.3 mm under the closed gap). scale
        # 0.9 = ~5.4 cm -> the gripper seats firmly on it. Rests on the rig tabletop (top z=0.02); z=0.05
        # settles it on the slab. Spawns OUT IN FRONT of the left arm (base at y~0.46) and within reach --
        # not back under the arm's shoulder. See the cube reset + goal ranges below.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.13, 0.05], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.9, 0.9, 0.9),
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

        # Goal target: random, OUT IN FRONT of the arm and WITHIN ITS REACH -- not back under the base
        # (y~0.46), and not pushed so far toward platform centre that the arm can't reach it. z is free to
        # be HIGH: the arm may hold the cube up in the air; the constraint is reachability, not height.
        # The x/y band stays inside the region the previous run already grasped + tracked across.
        # Goal randomized OUT IN FRONT of the arm. The cube spawns near (y~0.13, see below) and the goal
        # sits forward of it (y in [-0.10, 0.05]) so the task is grasp-near -> carry forward -> place out
        # front, not lift-in-place. Base is at y~0.457; the measured graspable footprint (reach_map.json)
        # reaches to ~ -0.29, so this band stays inside reach. Goal is uniform-random over this range.
        self.commands.object_pose.body_name = EE_LINK
        self.commands.object_pose.ranges.pos_x = (-0.12, 0.12)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.05)
        self.commands.object_pose.ranges.pos_z = (0.08, 0.25)

        # Cube reset: a delta around the init pose [0, 0.13, 0.05] keeping it OUT IN FRONT of the arm and
        # WITHIN REACH -- never back under the base (y~0.46), never pushed past the arm's reach toward
        # platform centre. y delta (-0.075, 0.075) -> y in [0.055, 0.205]; matches the goal x/y band.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.12, 0.12),
            "y": (-0.075, 0.075),
            "z": (0.0, 0.0),
        }

        # The Stationary AI rig carries its own collision tabletop (slab top z=0.02), exactly as
        # Trossen's pick-place scene does. Drop the base lift task's foreign SeattleLabTable
        # (which sits at +x[0.5] under the WXAI workspace) and rest the ground at the rig base.
        self.scene.table = None
        self.scene.plane.init_state.pos = (0.0, 0.0, 0.0)

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
                    offset=OffsetCfg(pos=list(EE_TCP_OFFSET)),
                ),
            ],
        )

        # Reward = the stock reference LiftEnvCfg 4-term reward (reaching / lifting / goal-tracking /
        # fine-grained + action & velocity penalties), unchanged. The ONLY deviation: minimal_height
        # 0.04 -> 0.09 on the lift + both goal-tracking terms, because our cube rests at ~0.048 on the
        # raised rig tabletop (the reference's table puts the cube below 0.04, so it needs no re-base).
        LIFT_HEIGHT = 0.09
        self.rewards.lifting_object.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = LIFT_HEIGHT

        # Placement was weak (goal_tracking ~8.8/16 on the stiff/soft runs); raise the goal-tracking
        # weights so the policy invests in nailing the target pose, not just lifting. (Stock: 16 / 5.)
        self.rewards.object_goal_tracking.weight = 24.0
        self.rewards.object_goal_tracking_fine_grained.weight = 8.0


@configclass
class StationaryAiCubeLiftEnvCfg_PLAY(StationaryAiCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
