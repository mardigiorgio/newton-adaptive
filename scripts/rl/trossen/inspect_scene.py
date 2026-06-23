"""Render labeled stills of the Stationary AI cube-lift scene for visual geometry checks.

This is the "look before you train" pre-flight: it spawns the corrected env with a SINGLE
env, turns the ee_frame debug marker ON, lets the cube settle, and dumps a handful of fixed
camera views to PNG so you can eyeball -- in seconds -- the geometry that reward curves hide:

  - cube sits on the rig tabletop, in front of the LEFT arm (+y), not floating / not in +x
  - the ee_frame marker sits between the gripper fingers (the TCP), not back at the wrist
  - exactly ONE table (the rig's own tabletop), rig resting on the ground
  - at the default pose the left arm can plausibly reach the cube

No trained checkpoint needed -- this inspects the env itself. Rendering needs --enable_cameras.
Run natively:
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/inspect_scene.py --headless --enable_cameras
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--settle", type=int, default=30, help="zero-action steps to let the cube settle")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--out_dir", default=None, help="PNG output dir (default: <ARTIFACT_ROOT>/scene_check)")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
parser.add_argument("--no_marker", action="store_true", help="hide the ee_frame debug marker")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trossen_cube  # noqa: F401,E402  (registers gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from trossen_cube.paths import ARTIFACT_ROOT  # noqa: E402

args.out_dir = args.out_dir or os.path.join(ARTIFACT_ROOT, "scene_check")

# Fixed viewpoints framing the LEFT arm (base ~y=0.46) + cube (~[0, 0.25, 0.05]) + tabletop.
# (eye, lookat) in robot-root/world meters; labels become the PNG filenames.
VIEWS = [
    ("top_down", (0.02, 0.30, 1.05), (0.0, 0.30, 0.02)),
    ("left_arm_side", (1.15, 0.30, 0.45), (-0.05, 0.30, 0.12)),
    ("three_quarter", (0.95, -0.70, 0.80), (0.0, 0.25, 0.12)),
    ("ee_closeup", (0.50, 0.02, 0.38), (-0.02, 0.21, 0.19)),
]


def _save(frame, path, label):
    bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, label, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(path, bgr)


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.viewer.resolution = (args.width, args.height)
    # Show where the env thinks the end-effector frame is -- the whole point of the check.
    if not args.no_marker:
        env_cfg.scene.ee_frame.debug_vis = True

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    sim = env.unwrapped.sim
    env.reset()

    # Let the cube settle on the tabletop under a held (zero-delta) action.
    act = torch.zeros(
        (args.num_envs, env.unwrapped.action_manager.total_action_dim),
        device=env.unwrapped.device,
    )
    for _ in range(args.settle):
        env.step(act)

    saved = []
    for label, eye, lookat in VIEWS:
        try:
            sim.set_camera_view(eye=eye, target=lookat)
        except Exception as exc:  # camera API differences shouldn't kill the whole run
            print(f"[inspect_scene] camera set failed for {label}: {exc}")
        # Render a few times so the new view + draw settles before grabbing the frame.
        frame = None
        for _ in range(4):
            frame = env.render()
        if frame is None:
            print(f"[inspect_scene] no frame for {label} (need --enable_cameras)")
            continue
        path = os.path.join(args.out_dir, f"{label}.png")
        _save(frame, path, f"{label}   cube={env_cfg.scene.object.init_state.pos}")
        saved.append(path)

    with open(os.path.join(args.out_dir, "scene_check.done"), "w") as f:
        f.write("views=" + ",".join(os.path.basename(p) for p in saved) + "\n")
    print(f"[inspect_scene] wrote {len(saved)} views to {args.out_dir}")
    env.close()


main()
os._exit(0)
