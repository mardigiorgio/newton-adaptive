"""Render a teacher training timelapse.

Rolls out each saved checkpoint in the PLAY env (single env, single app session),
overlays the iteration number, and writes ONE mp4 showing the policy improving.

Run in the container (rendering needs --enable_cameras):
    podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python \
      scripts/rl/trossen/render_timelapse.py --headless --enable_cameras \
      --steps 120 --out /isaac/teacher_timelapse.mp4"
"""

import argparse
import glob
import importlib.metadata as metadata
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120, help="rollout steps per checkpoint")
parser.add_argument("--every", type=int, default=1, help="use every Nth checkpoint")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=360)
parser.add_argument("--out", default="/isaac/teacher_timelapse.mp4")
parser.add_argument("--log_dir", default="/isaac/logs/trossen/stationary_ai_lift_teacher")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402  (registers gym ids)


def _iter_of(path: str) -> int:
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def main():
    ver = metadata.version("rsl-rl-lib")
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg.scene.num_envs = args.num_envs
    # Smaller viewport -> much faster per-frame host copy. Frame a 3/4 view on the
    # active (left) arm + cube (cube spawns near x=0.3).
    env_cfg.viewer.resolution = (args.width, args.height)
    env_cfg.viewer.eye = (1.4, -1.4, 1.1)
    env_cfg.viewer.lookat = (0.3, 0.0, 0.2)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, ver)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    ckpts = sorted(
        glob.glob(os.path.join(args.log_dir, "**", "model_*.pt"), recursive=True), key=_iter_of
    )[:: args.every]

    writer = None
    n_frames = 0
    for ckpt in ckpts:
        it = _iter_of(ckpt)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=agent_cfg.device)
        obs, _ = wrapped.reset()
        # NOTE: torch.no_grad (NOT inference_mode) -- inference_mode marks the env's
        # state tensors as inference tensors, which breaks the next checkpoint's reset().
        with torch.no_grad():
            for _ in range(args.steps):
                obs, _, _, _ = wrapped.step(policy(obs))
                frame = env.render()
                if frame is None:
                    continue
                bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                cv2.putText(bgr, f"teacher  iter {it}", (24, 44), cv2.FONT_HERSHEY_SIMPLEX,
                            1.1, (255, 255, 255), 2, cv2.LINE_AA)
                if writer is None:
                    h, w = bgr.shape[:2]
                    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
                writer.write(bgr)
                n_frames += 1

    if writer is not None:
        writer.release()
    with open(args.out + ".done", "w") as f:
        f.write(f"frames={n_frames} checkpoints={[_iter_of(c) for c in ckpts]}\n")
    wrapped.close()


main()
os._exit(0)
