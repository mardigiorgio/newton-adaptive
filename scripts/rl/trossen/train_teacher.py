"""Stage-1 PPO teacher training for the Stationary AI cube-lift task.

Self-contained (mirrors Isaac Lab's rsl_rl train.py flow + imports trossen_cube to
register the gym id). Run inside the container:

    scripts/rl/trossen/docker/run.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 4096
    # smoke:
    scripts/rl/trossen/docker/run.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 64 --max_iterations 3
"""

import argparse
import importlib.metadata as metadata
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--resume_from", default=None, help="checkpoint .pt to resume from")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402  (registers gym ids)

TASK = "Isaac-Lift-Cube-StationaryAI-Teacher-v0"
LOG_ROOT = os.environ.get("TROSSEN_LOG_ROOT", "/isaac/logs/trossen")


def main():
    env_cfg = parse_env_cfg(TASK, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    # Migrate legacy cfg fields (e.g. `stochastic`) to match the installed rsl-rl-lib, as
    # Isaac Lab's own train.py does -- otherwise rsl-rl's MLPModel rejects the extra kwarg.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    log_dir = os.path.join(LOG_ROOT, agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume_from:
        # Loads model + optimizer + iteration counter; learn() then runs `max_iterations` MORE.
        runner.load(args.resume_from)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # Marker survives Kit's stdout capture + any hanging app.close(); used to detect success.
    import json

    with open(os.path.join(log_dir, "train_done.json"), "w") as f:
        json.dump({"iterations": args.max_iterations, "task": TASK, "log_dir": log_dir}, f)

    env.close()


main()
simulation_app.close()
