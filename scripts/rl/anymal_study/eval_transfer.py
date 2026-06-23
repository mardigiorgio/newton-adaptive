"""Cross-physics transfer evaluation: roll out a frozen policy on multiple
physics backends and record physical metrics for the hypothesis test.

This is the sim-to-sim proxy for the sim-to-real claim (build spec §5): a policy
trained on one backend is evaluated zero-shot on higher-fidelity reference
backends; the degradation is the transfer gap.

    uv run -m scripts.rl.anymal_study.eval_transfer --checkpoint runs/adaptive_dr-off_s1/model_1500.pt \\
        --eval-backends id ref_tol ref_dt --episodes 50 --out results/adaptive_dr-off_s1.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch  # noqa: TID253

from .adaptive_env import AdaptiveLocomotionEnv
from .anymal import compute_base_frame
from .backends import REF_SPECS, BackendSpec
from .config import EnvConfig, make_ppo_cfg
from .domain_rand import DRConfig


def eval_backend_spec(name: str, train_spec: BackendSpec) -> tuple[BackendSpec, DRConfig | None]:
    """Resolve an --eval-backends token to (BackendSpec, DRConfig).

    Single-term tokens (``ref_friction``/``ref_mass``/``ref_kp``/``ref_kd``/
    ``ref_push``/``ref_obsnoise``) hold the integrator IDENTICAL to ``id`` and
    perturb exactly one physical/sensing channel, so the gap vs ``id`` isolates that
    channel (not channel + integrator) for the error-budget ranking; ``ref_tol``/
    ``ref_dt`` change only the integrator and isolate it.
    """
    if name == "id":
        return train_spec, None
    if name in REF_SPECS:
        return REF_SPECS[name], None
    if name == "ref_perturbed":
        return REF_SPECS["ref_tol"], DRConfig.preset("ood")
    if name.startswith("ref_") and name[len("ref_") :] in DRConfig.SINGLE_TERMS:
        return train_spec, DRConfig.single(name[len("ref_") :])
    raise ValueError(f"unknown eval backend {name!r}")


@torch.no_grad()
def run_suite(env: AdaptiveLocomotionEnv, policy, episodes: int, horizon: int) -> dict:
    """Roll out ``episodes`` worlds in parallel for ``horizon`` steps; return metrics.

    Uses the env's vectorization: each of the N worlds is one episode. Metrics are
    accumulated on-device and moved to host once at the end.
    """
    obs, _ = env.reset()
    n = env.num_envs
    device = env.device

    lin_err = torch.zeros(n, device=device)
    ang_err = torch.zeros(n, device=device)
    steps_alive = torch.zeros(n, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)

    for _ in range(horizon):
        actions = policy(obs)
        obs, _, _, dones, extras = env.step(actions)
        # Exclude any world reset on this frame (fall OR timeout): env.jq/jqd now
        # hold its post-reset standing pose, which would bias the tracking error.
        fell = dones & ~extras["time_outs"]
        contrib = alive & ~dones
        cmd = env.command.cmd
        bf = compute_base_frame(env.jq, env.jqd, env.gravity_vec)
        lin_err += contrib * (cmd[:, :2] - bf.vel_b[:, :2]).norm(dim=1)
        ang_err += contrib * (cmd[:, 2] - bf.ang_b[:, 2]).abs()
        steps_alive += contrib.float()
        alive = alive & ~fell

    denom = steps_alive.clamp_min(1.0)
    return {
        "lvte": (lin_err / denom).cpu().numpy(),  # mean lin-vel tracking error [m/s]
        "avte": (ang_err / denom).cpu().numpy(),  # mean ang-vel tracking error [rad/s]
        "survival": alive.float().cpu().numpy(),  # reached horizon without falling
        "steps_alive": steps_alive.cpu().numpy(),
    }


def main():
    p = argparse.ArgumentParser(description="Cross-physics transfer evaluation of a frozen policy.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-backend", choices=["adaptive", "fixed", "fixed_fine"], default="adaptive")
    p.add_argument("--eval-backends", nargs="+", default=["id", "ref_tol", "ref_dt"])
    p.add_argument("--episodes", type=int, default=64, help="parallel worlds = episodes")
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0, help="paired-eval seed: identical ICs+commands across backends")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

    from .config import backend_from_name  # noqa: PLC0415

    train_spec = backend_from_name(args.train_backend)
    # Freeze the command for the whole horizon (resample period >> horizon) so each
    # world tracks one paired command until it falls; combined with eval_seed this
    # makes world w's IC+command byte-identical across backends and the gap isolates
    # the backend. episode length covers the horizon so no mid-rollout timeout reset.
    env_cfg = EnvConfig(
        num_envs=args.episodes,
        eval_seed=args.seed,
        command_resample_s=1.0e9,
        episode_length_s=max(20.0, (args.horizon + 10) * EnvConfig().control_dt),
    )
    results = {}
    for name in args.eval_backends:
        spec, dr = eval_backend_spec(name, train_spec)
        # Seed the global stream too so per-reference DR draws are reproducible run-to-run.
        torch.manual_seed(args.seed)
        if str(args.device).startswith("cuda"):
            torch.cuda.manual_seed_all(args.seed)
        env = AdaptiveLocomotionEnv(env_cfg, spec, dr, device=args.device, headless=True)
        runner = OnPolicyRunner(env, make_ppo_cfg(1, 0), log_dir=None, device=args.device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=args.device)
        results[name] = run_suite(env, policy, args.episodes, args.horizon)
        m = results[name]
        print(f"[{name}] LVTE={m['lvte'].mean():.3f}  AVTE={m['avte'].mean():.3f}  survival={m['survival'].mean():.2f}")
        del env, runner

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **{f"{b}__{k}": v for b, m in results.items() for k, v in m.items()})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
