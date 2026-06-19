"""Measure the LEFT arm's graspable table footprint by forward-kinematics sampling.

Samples thousands of random left-arm joint configs (within limits) across many parallel envs,
reads the resulting grasp-TCP pose, and keeps the ones that can grasp on the table (TCP near
table height with the gripper pointing down). Writes every sampled TCP (robot-root frame) + flags
to JSON for plot_reach_map.py. This answers, objectively, how much of the table the arm can reach
-- so the cube/goal spawn can cover the whole REACHABLE area instead of a guessed box.

    podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python \
      scripts/rl/trossen/reach_map.py --headless --out /isaac/reach_map.json"
"""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--batches", type=int, default=16)
parser.add_argument("--out", default="/isaac/reach_map.json")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

TCP_OFFSET = 0.087  # link_6 local +x to grasp point (matches EE_TCP_OFFSET)

cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
cfg.scene.num_envs = args.num_envs
# Pure-FK reachability: we only read kinematic TCP poses, never simulate contact. Self-collisions
# ON made wild random configs explode on self/table contact (garbage TCPs out past the table). Off.
cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
env = gym.make(args.task, cfg=cfg)
u = env.unwrapped
robot = u.scene["robot"]
sim = u.sim
dev = u.device
dt = sim.get_physics_dt()
env.reset()
# park the cube off-platform so the arm never contacts it during sampling
u.scene["object"].write_root_pose_to_sim(
    torch.tensor([5.0, 5.0, 0.05, 1.0, 0.0, 0.0, 0.0], device=dev).repeat(args.num_envs, 1))

arm_ids = torch.tensor(robot.find_joints("follower_left_joint_[0-5]")[0], device=dev)
link6 = robot.find_bodies("follower_left_link_6")[0][0]
lim = robot.data.soft_joint_pos_limits  # [N, J, 2]
lo, hi = lim[:, arm_ids, 0], lim[:, arm_ids, 1]
default_q = robot.data.default_joint_pos.clone()

pts = []
for _ in range(args.batches):
    q = default_q.clone()
    q[:, arm_ids] = lo + torch.rand_like(lo) * (hi - lo)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    robot.write_data_to_sim()
    sim.step(render=False)
    robot.update(dt)

    p = robot.data.body_pos_w[:, link6]          # wrist pos, world
    qt = robot.data.body_quat_w[:, link6]         # (w,x,y,z)
    w_, x_, y_, z_ = qt[:, 0], qt[:, 1], qt[:, 2], qt[:, 3]
    # link_6 local +x expressed in world (the tool/approach axis)
    ax_x = 1 - 2 * (y_ * y_ + z_ * z_)
    ax_y = 2 * (x_ * y_ + w_ * z_)
    ax_z = 2 * (x_ * z_ - w_ * y_)
    tcp = torch.stack([p[:, 0] + TCP_OFFSET * ax_x,
                       p[:, 1] + TCP_OFFSET * ax_y,
                       p[:, 2] + TCP_OFFSET * ax_z], dim=1)
    tcp_r = tcp - robot.data.root_pos_w           # robot-root frame (spawn-range frame)
    for i in range(args.num_envs):
        pts.append([round(float(tcp_r[i, 0]), 4), round(float(tcp_r[i, 1]), 4),
                    round(float(tcp_r[i, 2]), 4), round(float(ax_z[i]), 3)])

# table extent (robot-root frame) from the USD dump; base + rest from reach_probe.
meta = {"table": {"x": [-0.380, 0.369], "y": [-0.610, 0.610], "z_top": 0.020},
        "base_y": 0.457, "rest_tcp": [-0.02, 0.275, 0.218], "n": len(pts),
        "grasp_z": [0.03, 0.14], "down_thresh": -0.4}
with open(args.out, "w") as f:
    json.dump({"pts": pts, "meta": meta}, f)
print(f"[reach_map] wrote {len(pts)} samples -> {args.out}")

import sys  # noqa: E402

sys.stdout.flush()
app.close()
import os  # noqa: E402

os._exit(0)
