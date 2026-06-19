"""Visualize the randomized cube + goal spawn: reset the env N times and tile the frames.

Proves the cube AND goal command are randomized (the training timelapse hid this by re-seeding
to a fixed draw). No policy -- arm holds default pose; the cube sits at its random reset pose and
the goal marker (RGB axes) at its random commanded pose.

    scripts/rl/trossen/docker/run.sh scripts/rl/trossen/show_spawn.py --headless --enable_cameras \
      --num 12 --out /isaac/artifacts/frames/spawn_demo.png
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=12, help="number of resets to show")
parser.add_argument("--cols", type=int, default=4)
parser.add_argument("--settle", type=int, default=3, help="zero-action steps after reset before capture")
parser.add_argument("--seed", type=int, default=0, help="base seed (resets advance the RNG -> varied draws)")
parser.add_argument("--out", default="/isaac/artifacts/frames/spawn_demo.png")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app = AppLauncher(args).app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.scene.num_envs = 1
cfg.seed = args.seed
# top-down-ish 3/4 view framing the table in front of the arm
cfg.viewer.resolution = (480, 360)
cfg.viewer.eye = (1.2, -0.9, 1.0)
cfg.viewer.lookat = (0.0, 0.05, 0.12)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
adim = env.action_space.shape[-1]
zero = torch.zeros((1, adim), device=env.unwrapped.device)

frames = []
env.reset()
for k in range(args.num):
    env.reset()  # NOT re-seeded -> each draw differs
    frame = None
    for _ in range(args.settle):
        env.step(zero)
        frame = env.render()
    if frame is None:
        continue
    bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, f"reset {k+1}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    frames.append(bgr)

cols = args.cols
rows = [np.hstack(frames[i:i + cols]) for i in range(0, len(frames), cols)]
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
os.makedirs(os.path.dirname(args.out), exist_ok=True)
cv2.imwrite(args.out, np.vstack(rows))
with open(args.out + ".done", "w") as f:
    f.write(f"resets={len(frames)}\n")

env.close()
os._exit(0)
