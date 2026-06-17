"""ANYmal-C velocity-tracking task: model build + vectorized obs/reward/termination.

Obs/action layout and PD/contact parameters follow the Isaac Lab / rsl_rl
ANYmal-C locomotion convention (48-dim observation, 12-dim action), matching
the pretrained policy shipped with the ``anybotics_anymal_c`` asset.

All per-step functions are fully vectorized (all worlds at once) over zero-copy
``wp.to_torch`` views — no per-world Python loop, no ``.numpy()`` in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # noqa: TID253
import warp as wp

import newton
import newton.utils
from newton import GeoType

# --- task dimensions / conventions (from the reference policy) ---
OBS_DIM = 48
ACT_DIM = 12
ACTION_SCALE = 0.5

# Joint index remap between Isaac-Lab ("lab") order and Newton/MuJoCo order.
LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

# Standing pose targets (joint name -> angle [rad]), Newton/MuJoCo joint order.
INITIAL_Q = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}

JOINT_TARGET_KE = 150.0  # PD position gain (Kp)
JOINT_TARGET_KD = 5.0  # PD damping gain (Kd)


@dataclass
class TaskMeta:
    """Per-world model layout and constants needed by the env."""

    num_worlds: int
    coords_per_world: int  # 19 = 7 (free) + 12 (actuated)
    dofs_per_world: int  # 18 = 6 (free) + 12 (actuated)
    bodies_per_world: int
    default_joint_q: list[float]  # 12 actuated standing angles, Newton order
    trunk_local_index: int = 0  # trunk/base body index within a world


def build_anymal_model(num_worlds: int, *, spacing=(1.5, 0.0, 0.0), device=None):
    """Build a replicated ANYmal-C model. Returns ``(model, TaskMeta)``.

    ``register_custom_attributes`` must run before ``add_urdf``.
    """
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(robot)
    robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1)
    robot.default_shape_cfg.ke = 5.0e4
    robot.default_shape_cfg.kd = 5.0e2
    robot.default_shape_cfg.kf = 1.0e3
    robot.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    robot.add_urdf(
        str(asset_path / "urdf" / "anymal.urdf"),
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.62),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )

    # MuJoCo wants sphere collision geoms expressed as (radius, 0, 0).
    for i in range(len(robot.shape_type)):
        if robot.shape_type[i] == GeoType.SPHERE:
            r = robot.shape_scale[i][0]
            robot.shape_scale[i] = (r * 2.0, 0.0, 0.0)

    for name, value in INITIAL_Q.items():
        idx = next((i for i, lbl in enumerate(robot.joint_label) if lbl.endswith(f"/{name}")), None)
        if idx is None:
            raise ValueError(f"Joint '{name}' not found in anymal model")
        robot.joint_q[idx + 6] = value

    for i in range(len(robot.joint_target_ke)):
        robot.joint_target_ke[i] = JOINT_TARGET_KE
        robot.joint_target_kd[i] = JOINT_TARGET_KD

    # 12 actuated standing angles live at coords [7:19] (after the 7-coord free joint).
    default_joint_q = [float(v) for v in robot.joint_q[7:19]]

    scene = newton.ModelBuilder()
    scene.replicate(robot, num_worlds, spacing=spacing)
    scene.add_ground_plane()
    model = scene.finalize()

    meta = TaskMeta(
        num_worlds=num_worlds,
        coords_per_world=model.joint_coord_count // num_worlds,
        dofs_per_world=model.joint_dof_count // num_worlds,
        bodies_per_world=model.body_count // num_worlds,
        default_joint_q=default_joint_q,
    )
    return model, meta


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate ``v`` by the inverse of quaternion ``q`` (xyzw, w last). Batched (N,*)."""
    q_w = q[:, 3]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


@dataclass
class BaseFrame:
    """Root state expressed in the base frame (all (N,*) tensors)."""

    vel_b: torch.Tensor  # base linear velocity, body frame (N,3)
    ang_b: torch.Tensor  # base angular velocity, body frame (N,3)
    grav: torch.Tensor  # projected gravity unit vector, body frame (N,3)
    qpos: torch.Tensor  # actuated joint positions, Newton order (N,12)
    qvel: torch.Tensor  # actuated joint velocities, Newton order (N,12)
    base_height: torch.Tensor  # world-frame root height z (N,)


def compute_base_frame(jq: torch.Tensor, jqd: torch.Tensor, gravity_vec: torch.Tensor) -> BaseFrame:
    """Derive base-frame quantities from joint state views.

    Args:
        jq: ``(N, coords_per_world)`` view of ``state.joint_q``. Layout:
            [0:3]=root pos, [3:7]=root quat (xyzw), [7:]=actuated qpos.
        jqd: ``(N, dofs_per_world)`` view of ``state.joint_qd``. Layout:
            [0:3]=root lin vel, [3:6]=root ang vel, [6:]=actuated qvel.
        gravity_vec: ``(1,3)`` or ``(N,3)`` world-down unit vector [0,0,-1].
    """
    root_quat = jq[:, 3:7]
    lin = jqd[:, 0:3]
    ang = jqd[:, 3:6]
    return BaseFrame(
        vel_b=quat_rotate_inverse(root_quat, lin),
        ang_b=quat_rotate_inverse(root_quat, ang),
        grav=quat_rotate_inverse(root_quat, gravity_vec.expand(jq.shape[0], 3)),
        qpos=jq[:, 7:],
        qvel=jqd[:, 6:],
        base_height=jq[:, 2],
    )


def compute_obs(
    bf: BaseFrame,
    command: torch.Tensor,
    last_action: torch.Tensor,
    default_q: torch.Tensor,
    lab_idx: torch.Tensor,
) -> torch.Tensor:
    """Build the 48-dim observation (joint blocks reordered to lab/policy order)."""
    joint_pos_rel = torch.index_select(bf.qpos - default_q, 1, lab_idx)
    joint_vel = torch.index_select(bf.qvel, 1, lab_idx)
    return torch.cat([bf.vel_b, bf.ang_b, bf.grav, command, joint_pos_rel, joint_vel, last_action], dim=1)


def action_to_targets(action: torch.Tensor, default_q: torch.Tensor, mujoco_idx: torch.Tensor) -> torch.Tensor:
    """Map a (N,12) policy action (lab order) to actuated PD targets (Newton order)."""
    action_newton = torch.index_select(action, 1, mujoco_idx)
    return default_q + ACTION_SCALE * action_newton


# Reward weights (legged_gym / Isaac Lab ANYmal-flat lineage). Contact-dependent
# terms (feet_air_time, collision) are disabled in v1 — see TERMINATION note.
REWARD_WEIGHTS = {
    "lin_vel_xy": 1.0,
    "ang_vel_z": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "torque": -2.5e-5,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "flat_orientation": -5.0,
}
TRACKING_SIGMA = 0.25


def compute_reward(
    bf: BaseFrame,
    command: torch.Tensor,
    action: torch.Tensor,
    last_action: torch.Tensor,
    targets_actuated: torch.Tensor,
    last_qvel: torch.Tensor,
    dt: float,
    ke: float = JOINT_TARGET_KE,
    kd: float = JOINT_TARGET_KD,
) -> torch.Tensor:
    """Velocity-tracking reward, clipped to >= 0 (only-positive-rewards)."""
    w = REWARD_WEIGHTS
    lin_err = (command[:, :2] - bf.vel_b[:, :2]).pow(2).sum(dim=1)
    ang_err = (command[:, 2] - bf.ang_b[:, 2]).pow(2)
    tau = ke * (targets_actuated - bf.qpos) - kd * bf.qvel

    r = (
        w["lin_vel_xy"] * torch.exp(-lin_err / TRACKING_SIGMA)
        + w["ang_vel_z"] * torch.exp(-ang_err / TRACKING_SIGMA)
        + w["lin_vel_z"] * bf.vel_b[:, 2].pow(2)
        + w["ang_vel_xy"] * bf.ang_b[:, :2].pow(2).sum(dim=1)
        + w["torque"] * tau.pow(2).sum(dim=1)
        + w["dof_acc"] * ((last_qvel - bf.qvel) / dt).pow(2).sum(dim=1)
        + w["action_rate"] * (last_action - action).pow(2).sum(dim=1)
        + w["flat_orientation"] * bf.grav[:, :2].pow(2).sum(dim=1)
    )
    return r.clamp_min(0.0)


# v1 termination is purely kinematic (no contact array needed): the robot has
# fallen if the base drops below 0.25 m or tilts past ~60 deg (projected gravity
# z-component rises above -0.5). Contact-based termination is a v1.1 refinement.
TERMINATION_HEIGHT = 0.25
TERMINATION_GRAVITY_Z = -0.5


def compute_termination(bf: BaseFrame) -> torch.Tensor:
    """Boolean (N,) failure mask (fall). Time-out is tracked separately by the env."""
    fallen = (bf.base_height < TERMINATION_HEIGHT) | (bf.grav[:, 2] > TERMINATION_GRAVITY_Z)
    return fallen


class VelocityCommand:
    """Per-world [vx, vy, wz] velocity command with periodic resampling + deadband."""

    def __init__(
        self,
        num_worlds: int,
        device,
        vx=(-1.0, 1.0),
        vy=(-1.0, 1.0),
        wz=(-1.0, 1.0),
        deadband: float = 0.2,
        generator: torch.Generator | None = None,
    ):
        self.device = device
        self.ranges = torch.tensor([vx, vy, wz], device=device, dtype=torch.float32)  # (3,2)
        self.deadband = deadband
        self.cmd = torch.zeros(num_worlds, 3, device=device, dtype=torch.float32)
        # Dedicated stream so command draws stay paired across backends in eval,
        # independent of how many domain-randomization randoms a reference consumes.
        self.generator = generator

    def resample(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        lo = self.ranges[:, 0]
        hi = self.ranges[:, 1]
        u = torch.rand(len(env_ids), 3, device=self.device, generator=self.generator)
        c = lo + u * (hi - lo)
        # zero tiny planar commands so the policy learns a clean standstill gait
        small = c[:, :2].norm(dim=1) < self.deadband
        c[small, :2] = 0.0
        self.cmd[env_ids] = c
