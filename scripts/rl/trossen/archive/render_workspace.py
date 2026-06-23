"""Show the cube-spawn workspace: composite the cube at a grid of positions onto one image.

Moves the single cube through a 3x3 grid spanning the proposed spawn box (x/y in robot-root
frame), snapshots each placement from a fixed camera, and overlays them (diff-mask vs a
cube-free baseline) so every spawn position shows on one platform. Lets you eyeball the spread
and whether the arm can reach the corners BEFORE committing to a training run. Markers off.

    podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python \
      scripts/rl/trossen/render_workspace.py --headless --enable_cameras --out_dir /isaac/ws_check"
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--x", type=float, nargs=2, default=[-0.15, 0.15], help="x range (robot-root)")
parser.add_argument("--y", type=float, nargs=2, default=[0.12, 0.32], help="y range (robot-root)")
parser.add_argument("--z", type=float, default=0.05, help="cube spawn z (table)")
parser.add_argument("--n", type=int, default=3, help="grid points per axis")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--out_dir", default="/isaac/ws_check")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trossen_cube  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# (label, eye, target) -- top-down maps x/y spread; three-quarter shows depth + reach.
VIEWS = [
    ("workspace_top_down", (0.0, 0.27, 1.25), (0.0, 0.27, 0.02)),
    ("workspace_three_quarter", (0.95, -0.55, 0.75), (0.0, 0.22, 0.08)),
]


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = parse_env_cfg(args.task, num_envs=1)
    cfg.scene.num_envs = 1
    cfg.viewer.resolution = (args.width, args.height)
    cfg.scene.ee_frame.debug_vis = False
    cfg.commands.object_pose.debug_vis = False  # no goal marker -- only the cube should be colourful

    env = gym.make(args.task, cfg=cfg, render_mode="rgb_array")
    sim = env.unwrapped.sim
    u = env.unwrapped
    obj = u.scene["object"]
    dev = u.device
    env.reset()

    zero = torch.zeros((1, u.action_manager.total_action_dim), device=dev)
    for _ in range(15):
        env.step(zero)

    # Calibrate the robot-root -> world offset from where the cube actually settled after reset
    # (its cfg init is [0, 0.225, 0.05] in robot-root frame).
    offset = obj.data.root_pos_w[0].clone() - torch.tensor([0.0, 0.225, 0.05], device=dev)

    def place(x, y, z):
        world = offset + torch.tensor([x, y, z], device=dev)
        pose = torch.cat([world, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev)]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        env.step(zero)
        frame = None
        for _ in range(3):
            frame = env.render()
        return np.asarray(frame, dtype=np.uint8)

    xs = np.linspace(args.x[0], args.x[1], args.n)
    ys = np.linspace(args.y[0], args.y[1], args.n)

    for label, eye, tgt in VIEWS:
        try:
            sim.set_camera_view(eye=eye, target=tgt)
        except Exception as exc:
            print(f"[ws] camera set failed for {label}: {exc}")
        baseline = place(5.0, 5.0, args.z)  # cube parked off-platform -> clean background
        comp = baseline.copy()
        for x in xs:
            for y in ys:
                f = place(float(x), float(y), args.z)
                diff = np.abs(f.astype(int) - baseline.astype(int)).sum(axis=2)
                mask = diff > 35
                comp[mask] = f[mask]
        bgr = cv2.cvtColor(comp, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            f"cube spawn box  x[{args.x[0]},{args.x[1]}]  y[{args.y[0]},{args.y[1]}]  ({args.n}x{args.n})",
            (24, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        path = os.path.join(args.out_dir, f"{label}.png")
        cv2.imwrite(path, bgr)
        print(f"[ws] wrote {path}")

    env.close()


main()
os._exit(0)
