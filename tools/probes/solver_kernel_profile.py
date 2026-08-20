"""Attribute solver wall time to named GPU kernels.

Wall-clock A/B of config knobs cannot say WHERE time goes -- it only says that
a knob moved a total. Warp's activity timing records every kernel launch with
its name and GPU duration, so this probe reports the actual breakdown: which
kernels dominate, how many times each is launched, and how that changes between
two configurations.

Method: ``wp.timing_begin(wp.TIMING_KERNEL)`` / ``wp.timing_end()`` around a
fixed number of stepped control intervals, after a warm-up that excludes module
load and graph capture. Results are aggregated by kernel name into total GPU
milliseconds and launch count.

Read it as: a few kernels at the top with large TOTAL and modest COUNT are
doing genuinely heavy work per launch; a kernel with an enormous COUNT and
small per-launch cost is being called too often, which for an error-controlled
solver means the step controller is subdividing. The launch COUNT of the
integrator kernels is therefore the substep signal, and the ratio between two
configurations is what attributes a slowdown.

Scope: this measures GPU kernel time only. Time spent on the host -- Python,
graph replay overhead, synchronization stalls -- appears as the gap between the
summed kernel total and the measured wall time, which the probe prints.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/solver_kernel_profile.py

Environment: SOLVER (sap-adaptive | mujoco-adaptive | mujoco), N_ENVS, STEPS,
WARMUP, TROSSEN_KE, TOP.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"
N_ENVS = int(os.environ.get("N_ENVS", "2048"))
STEPS = int(os.environ.get("STEPS", "40"))
WARMUP = int(os.environ.get("WARMUP", "25"))
TOP = int(os.environ.get("TOP", "22"))
SOLVER = os.environ.get("SOLVER", "sap-adaptive")

_SOLVER_FLAGS = {
    "mujoco": {},
    "mujoco-adaptive": {"adaptive": True},
    "sap": {"backend": "sap"},
    "sap-adaptive": {"backend": "sap", "sap_adaptive": True},
}


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    import warp as wp
    from isaaclab_tasks.utils import parse_env_cfg

    if SOLVER not in _SOLVER_FLAGS:
        print(f"FAIL: SOLVER must be one of {sorted(_SOLVER_FLAGS)}")
        return 1

    env_cfg = parse_env_cfg(TASK, num_envs=N_ENVS)
    for field, value in _SOLVER_FLAGS[SOLVER].items():
        setattr(env_cfg.sim.physics.solver_cfg, field, value)

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)

    torch.manual_seed(7)

    # Warm-up: module load, graph capture and the first contact events are
    # one-off costs that would otherwise be attributed to steady-state work.
    with torch.inference_mode():
        for _ in range(WARMUP):
            act.uniform_(-1.0, 1.0)
            env.step(act)

    wp.synchronize()
    wp.timing_begin(wp.TIMING_KERNEL)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(STEPS):
            act.uniform_(-1.0, 1.0)
            env.step(act)
    wp.synchronize()
    wall = time.perf_counter() - t0
    results = wp.timing_end()

    total_ms = collections.defaultdict(float)
    counts = collections.Counter()
    for r in results:
        total_ms[r.name] += float(r.elapsed)
        counts[r.name] += 1

    gpu_ms = sum(total_ms.values())
    print(f"\nsolver={SOLVER}  envs={N_ENVS}  steps={STEPS}  ke={os.environ.get('TROSSEN_KE', 'default 4.6e7')}")
    print(f"wall {wall*1000:.0f} ms   summed kernel GPU {gpu_ms:.0f} ms "
          f"({100*gpu_ms/(wall*1000):.0f}% of wall)   {len(results)} launches, {len(total_ms)} distinct kernels")
    print(f"per control step: {1000*wall/STEPS:.1f} ms wall, {gpu_ms/STEPS:.1f} ms GPU, {len(results)/STEPS:.0f} launches\n")

    print(f"{'kernel':<52} {'GPU ms':>10} {'%':>6} {'launches':>10} {'us/launch':>10}")
    print("-" * 92)
    for name, ms in sorted(total_ms.items(), key=lambda kv: -kv[1])[:TOP]:
        n = counts[name]
        print(f"{name[:52]:<52} {ms:>10.1f} {100*ms/gpu_ms:>5.1f}% {n:>10} {1000*ms/n:>10.1f}")

    shown = sum(ms for _, ms in sorted(total_ms.items(), key=lambda kv: -kv[1])[:TOP])
    print("-" * 92)
    print(f"{'(remaining kernels)':<52} {gpu_ms-shown:>10.1f} {100*(gpu_ms-shown)/gpu_ms:>5.1f}%")
    print(f"\nhost-side / non-kernel time: {wall*1000-gpu_ms:.0f} ms "
          f"({100*(wall*1000-gpu_ms)/(wall*1000):.0f}% of wall)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
