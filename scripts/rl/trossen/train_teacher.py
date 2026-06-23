"""Stage-1 PPO teacher training for the Stationary AI cube-lift task.

Self-contained (mirrors Isaac Lab's rsl_rl train.py flow + imports trossen_cube to
register the gym id). Run natively via the launcher wrapper:

    scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 2048
    # smoke:
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 64 --max_iterations 3
"""

import argparse
import importlib.metadata as metadata
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_envs",
    type=int,
    default=2048,
    help="2048 is validated (banked teacher + GPU buffers sized for 1-2k). 4096 is "
    "UNVERIFIED -- contact-drop there is silent (shows as object_dropping, not an "
    "error); boot-smoke and watch object_dropping before trusting it.",
)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--video", action="store_true")
parser.add_argument("--resume_from", default=None, help="checkpoint .pt to resume from")
parser.add_argument(
    "--entropy_coef",
    type=float,
    default=None,
    help="override algorithm.entropy_coef. Use a low value (e.g. 0.001) with "
    "--resume_from for the precision/still-hold anneal continuation; rsl-rl has no "
    "built-in entropy schedule, so this is the reference-faithful fine-tune lever.",
)
parser.add_argument(
    "--run_label",
    default=None,
    help="suffix appended to experiment_name (e.g. 'jv03') so one-var-per-run jitter "
    "experiments each write their own log dir instead of clobbering the canonical one.",
)
parser.add_argument(
    "--joint_vel_weight",
    type=float,
    default=None,
    help="override the joint_vel still-hold penalty weight (stock -1e-1 terminal). Sets "
    "BOTH the base reward weight AND the curriculum terminal weight, so the penalty "
    "is active from step 0 -- a --resume_from RESETS common_step_counter, so the "
    "iter-~417 curriculum would otherwise leave it at the base -1e-4 for most of a "
    "short fine-tune.",
)
parser.add_argument(
    "--action_rate_weight",
    type=float,
    default=None,
    help="override the action_rate still-hold penalty weight (see --joint_vel_weight).",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import trossen_cube  # noqa: F401,E402  (registers gym ids)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from trossen_cube.paths import LOG_ROOT  # noqa: E402  (post-launch; default ~/Documents/code/isaac-rl/logs/trossen)

TASK = "Isaac-Lift-Cube-StationaryAI-Teacher-v0"


def main():
    env_cfg = parse_env_cfg(TASK, num_envs=args.num_envs)
    env_cfg.seed = args.seed

    # Still-hold (jitter) lever: strengthen a motion penalty (one var per run). Override the base
    # reward weight AND the curriculum terminal weight together so the penalty bites immediately --
    # crucial on --resume_from, where common_step_counter resets and the stock iter-~417 curriculum
    # would otherwise keep the weight at -1e-4 for most of a short fine-tune. See JITTER_FIX_GUIDE.md.
    for term, weight in (("joint_vel", args.joint_vel_weight), ("action_rate", args.action_rate_weight)):
        if weight is not None:
            getattr(env_cfg.rewards, term).weight = weight
            getattr(env_cfg.curriculum, term).params["weight"] = weight
            print(f"[expt] {term} penalty weight -> {weight} (base + curriculum, active from step 0)")

    agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.seed = args.seed
    if args.entropy_coef is not None:
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.run_label:
        agent_cfg.experiment_name = f"{agent_cfg.experiment_name}.{args.run_label}"
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
