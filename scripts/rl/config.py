"""Env / PPO / run configuration and CLI assembly for the RL workflow."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .backends import BackendSpec
from .domain_rand import DRConfig


@dataclass
class EnvConfig:
    num_envs: int = 4096
    control_dt: float = 0.02  # 50 Hz policy/control rate [s]
    episode_length_s: float = 20.0
    command_resample_s: float = 4.0
    spacing: tuple[float, float, float] = (1.5, 0.0, 0.0)
    command_vx: tuple[float, float] = (-1.0, 1.0)
    command_vy: tuple[float, float] = (-1.0, 1.0)
    command_wz: tuple[float, float] = (-1.0, 1.0)
    ic_joint_noise: float = 0.1  # uniform +/- rad on actuated joints at reset
    ic_yaw: bool = True  # randomize base yaw at reset
    eval_seed: int | None = None  # if set, IC+command draws use dedicated paired-eval generators

    @property
    def max_episode_length(self) -> int:
        return round(self.episode_length_s / self.control_dt)

    @property
    def command_resample_steps(self) -> int:
        return round(self.command_resample_s / self.control_dt)


# PPO config (rsl_rl v1.x dict schema), ANYmal-flat MLP sizes.
def make_ppo_cfg(max_iterations: int, seed: int, experiment_name: str = "anymal_cenic") -> dict:
    return {
        "runner": {
            "policy_class_name": "ActorCritic",
            "algorithm_class_name": "PPO",
            "num_steps_per_env": 24,
            "max_iterations": max_iterations,
            "save_interval": 50,
            "experiment_name": experiment_name,
            "run_name": "",
            "resume": False,
        },
        "policy": {
            "init_noise_std": 1.0,
            "actor_hidden_dims": [128, 64, 32],
            "critic_hidden_dims": [128, 64, 32],
            "activation": "elu",
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "seed": seed,
    }


# Backend presets selectable from the CLI.
def backend_from_name(name: str) -> BackendSpec:
    if name == "cenic":
        return BackendSpec(kind="cenic")
    if name == "fixed":
        return BackendSpec(kind="fixed", fixed_dt=5e-3)
    if name == "fixed_fine":
        return BackendSpec(kind="fixed", fixed_dt=2e-3)
    raise ValueError(f"unknown backend {name!r}")


@dataclass
class TrainConfig:
    env: EnvConfig
    backend: BackendSpec
    dr: DRConfig
    ppo: dict
    seed: int = 1
    logdir: str = "runs/dev"
    device: str = "cuda"
    headless: bool = True
    max_iterations: int = 1500


def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--backend", choices=["cenic", "fixed", "fixed_fine"], default="cenic")
    p.add_argument("--dr", choices=["off", "on"], default="off")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--no-headless", dest="headless", action="store_false")


def train_config_from_args(args) -> TrainConfig:
    env = EnvConfig(num_envs=args.num_envs)
    backend = backend_from_name(args.backend)
    dr = DRConfig.preset(args.dr)
    logdir = getattr(args, "logdir", None) or f"runs/{args.backend}_dr-{args.dr}_s{args.seed}"
    return TrainConfig(
        env=env,
        backend=backend,
        dr=dr,
        ppo=make_ppo_cfg(args.max_iterations, args.seed),
        seed=args.seed,
        logdir=logdir,
        device=args.device,
        headless=args.headless,
        max_iterations=args.max_iterations,
    )
