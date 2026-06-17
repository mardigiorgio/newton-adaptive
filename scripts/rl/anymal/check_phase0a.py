"""Phase 0a validation: paired-eval RNG makes ICs+commands byte-identical across
backends, while a single-term reference perturbs exactly its one channel.

No policy or rsl_rl needed -- this checks env construction only.

    uv run --extra rl -m scripts.rl.check_phase0a
"""

from __future__ import annotations

import sys

import torch  # noqa: TID253
import warp as wp

from .backends import REF_SPECS
from .cenic_env import CenicLocomotionEnv
from .config import EnvConfig, backend_from_name
from .domain_rand import DRConfig


def _build(spec, dr, seed: int, n: int, device: str):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    cfg = EnvConfig(num_envs=n, eval_seed=seed, command_resample_s=1.0e9)
    return CenicLocomotionEnv(cfg, spec, dr, device=device, headless=True)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n, seed = 8, 0
    cenic = backend_from_name("cenic")
    fixed = backend_from_name("fixed")

    env_a = _build(cenic, None, seed, n, device)
    env_b = _build(fixed, None, seed, n, device)
    env_f = _build(REF_SPECS["ref_tol"], DRConfig.single("friction"), seed, n, device)

    ok = True

    def check(name: str, a: torch.Tensor, b: torch.Tensor, want_equal: bool):
        nonlocal ok
        d = (a - b).abs().max().item()
        passed = (d == 0.0) if want_equal else (d > 0.0)
        ok = ok and passed
        rel = "==" if want_equal else "!="
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: max|d|={d:.3e} (want {rel}0)")

    # Pairing: cenic-id vs fixed-id must share ICs and commands.
    check("ic_qpos   cenic==fixed", env_a.jq, env_b.jq, True)
    check("command   cenic==fixed", env_a.command.cmd, env_b.command.cmd, True)
    # Pairing survives DR draws: friction ref still shares ICs+commands with id.
    check("ic_qpos   id==ref_friction", env_a.jq, env_f.jq, True)
    check("command   id==ref_friction", env_a.command.cmd, env_f.command.cmd, True)
    # Single-term isolation: friction ref perturbs friction, and only friction.
    mu_a = wp.to_torch(env_a.model.shape_material_mu)
    mu_f = wp.to_torch(env_f.model.shape_material_mu)
    check("friction  id!=ref_friction", mu_a, mu_f, False)
    mass_a = wp.to_torch(env_a.model.body_mass)
    mass_f = wp.to_torch(env_f.model.body_mass)
    check("mass      id==ref_friction", mass_a, mass_f, True)

    print("\nPhase 0a pairing:", "PASS -- gap isolates the backend" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
