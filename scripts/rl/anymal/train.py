"""Training entrypoint: rsl_rl PPO over the CENIC/fixed locomotion env.

Examples::

    uv run -m scripts.rl.train --backend cenic --dr off --num-envs 4096 --seed 1
    uv run -m scripts.rl.train --backend fixed --dr off --num-envs 4096 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch  # noqa: TID253

from .cenic_env import CenicLocomotionEnv
from .config import add_common_args, train_config_from_args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="Train ANYmal-C locomotion (CENIC vs fixed backend).")
    add_common_args(p)
    p.add_argument("--max-iterations", type=int, default=1500)
    p.add_argument("--logdir", default=None)
    args = p.parse_args()

    cfg = train_config_from_args(args)
    os.makedirs(cfg.logdir, exist_ok=True)
    set_seed(cfg.seed)

    # Imported here so the env/smoke modules don't hard-depend on rsl_rl.
    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    env = CenicLocomotionEnv(cfg.env, cfg.backend, cfg.dr, device=cfg.device, headless=cfg.headless)
    runner = OnPolicyRunner(env, cfg.ppo, log_dir=cfg.logdir, device=cfg.device)

    # Record the run config + a budget log for fair-compute comparison (§5).
    with open(os.path.join(cfg.logdir, "run_config.json"), "w") as f:
        json.dump(
            {
                "backend": cfg.backend.__dict__,
                "dr_enabled": cfg.dr.enable,
                "num_envs": cfg.env.num_envs,
                "seed": cfg.seed,
                "max_iterations": cfg.max_iterations,
            },
            f,
            indent=2,
        )

    t0 = time.perf_counter()
    runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    wall = time.perf_counter() - t0

    final = os.path.join(cfg.logdir, f"model_{cfg.max_iterations}.pt")
    runner.save(final)
    total_steps = cfg.max_iterations * cfg.ppo["runner"]["num_steps_per_env"] * cfg.env.num_envs
    with open(os.path.join(cfg.logdir, "budget.json"), "w") as f:
        json.dump({"wall_seconds": wall, "total_env_steps": total_steps}, f, indent=2)
    print(f"done: {final}  ({wall:.0f}s, {total_steps:,} env-steps)")


if __name__ == "__main__":
    main()
