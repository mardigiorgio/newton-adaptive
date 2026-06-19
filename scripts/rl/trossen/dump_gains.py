"""Dump the Trossen arm's actual (USD-baked) PD gains, to compare against the Isaac Lab
reference arms (Franka/OpenArm lift use arm stiffness=80 damping=4, gripper 2e3/1e2).

Writes JSON (Kit eats stdout, so a file is the reliable channel).

    scripts/rl/trossen/docker/run.sh scripts/rl/trossen/dump_gains.py --headless
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", default="/isaac/artifacts/diag/gains.json")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg)
robot = env.unwrapped.scene["robot"]
names = list(robot.joint_names)
k = robot.data.joint_stiffness[0].cpu().numpy().tolist()
d = robot.data.joint_damping[0].cpu().numpy().tolist()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"joint_names": names,
               "stiffness": [round(x, 3) for x in k],
               "damping": [round(x, 4) for x in d]}, f, indent=2)
env.close()
os._exit(0)
