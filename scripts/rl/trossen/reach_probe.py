"""Measure the LEFT arm's rest-pose geometry, in the robot-root frame the spawn ranges use.

Prints the base, wrist (link_6), and grasp-TCP positions at the default joint pose, plus the
current cube pose -- so the cube/goal spawn band can be anchored on where the gripper actually
rests instead of a guess. Run in the container:
    podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python \
      scripts/rl/trossen/reach_probe.py --headless"
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
obj = u.scene["object"]
ee = u.scene["ee_frame"]

# settle at the default (rest) joint pose
act = torch.zeros((1, u.action_manager.total_action_dim), device=u.device)
for _ in range(30):
    env.step(act)

root = robot.data.root_pos_w[0]  # robot-root world pos; ranges are expressed relative to this


def rel(name):
    i = robot.find_bodies(name)[0][0]
    p = robot.data.body_pos_w[0, i] - root
    return f"x={p[0]:+.3f} y={p[1]:+.3f} z={p[2]:+.3f}"


tcp = ee.data.target_pos_w[0, 0] - root
cube = obj.data.root_pos_w[0, :3] - root
print("=== REST POSE (robot-root frame; same frame as cube/goal spawn ranges) ===")
print(f"  left base   : {rel('follower_left_base_link')}")
print(f"  wrist link_6: {rel('follower_left_link_6')}")
print(f"  grasp TCP   : x={tcp[0]:+.3f} y={tcp[1]:+.3f} z={tcp[2]:+.3f}   <-- where the gripper rests")
print(f"  cube now    : x={cube[0]:+.3f} y={cube[1]:+.3f} z={cube[2]:+.3f}")

import sys  # noqa: E402

sys.stdout.flush()
app.close()
import os  # noqa: E402

os._exit(0)
