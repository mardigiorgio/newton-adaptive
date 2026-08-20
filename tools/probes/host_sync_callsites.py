"""Name the code lines that force a device-to-host synchronization each step.

An nsys trace shows WHAT the cost is (blocking scalar readbacks) but not WHERE
it is issued. This probe attributes each synchronization to a Python call site.

Two detectors, because two runtimes can stall the pipeline:

  torch   ``torch.cuda.set_sync_debug_mode("warn")`` -- PyTorch's own detector,
          which warns at every operation that forces a CPU-GPU sync (``.item()``,
          ``.cpu()``, ``.numpy()``, ``bool(tensor)``, ``nonzero()``, indexing by
          a device tensor, etc.). The warning carries the Python stack.
  warp    ``wp.array.numpy()`` / ``.list()`` are wrapped, since Warp's own
          readbacks are a full device sync and do not go through torch.

Both are attributed to the first stack frame OUTSIDE torch/warp internals, so
the reported line is the caller in Isaac Lab, Newton, or the task -- the line
that would have to change.

Counts are per stepped control interval, after a warm-up, so a site that fires
once at startup does not masquerade as a per-step cost. A site with a high
per-step count is a pipeline stall repeated every step; that is the thing to
remove.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/host_sync_callsites.py

Environment: SOLVER, N_ENVS, STEPS, WARMUP, TOP.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import traceback
import warnings

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"
N_ENVS = int(os.environ.get("N_ENVS", "2048"))
STEPS = int(os.environ.get("STEPS", "10"))
WARMUP = int(os.environ.get("WARMUP", "20"))
TOP = int(os.environ.get("TOP", "25"))
SOLVER = os.environ.get("SOLVER", "sap-adaptive")

_SOLVER_FLAGS = {
    "mujoco": {},
    "mujoco-adaptive": {"adaptive": True},
    "sap": {"backend": "sap"},
    "sap-adaptive": {"backend": "sap", "sap_adaptive": True},
}

_INTERNAL = ("/torch/", "/warp/", "host_sync_callsites.py", "<frozen", "/warnings.py")

hits: collections.Counter = collections.Counter()
recording = False


def _attribute(skip_last: int = 0) -> str:
    """First stack frame outside torch/warp internals: the line to change."""
    stack = traceback.extract_stack()[:-1]
    if skip_last:
        stack = stack[:-skip_last]
    for frame in reversed(stack):
        if not any(marker in frame.filename for marker in _INTERNAL):
            short = "/".join(frame.filename.split("/")[-3:])
            return f"{short}:{frame.lineno} in {frame.name}()"
    return "<all frames internal>"


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

    # Warp readbacks are a full device sync and never reach torch's detector.
    for attr in ("numpy", "list"):
        original = getattr(wp.array, attr, None)
        if original is None:
            continue

        def make(orig):
            def wrapper(self, *a, **kw):
                if recording:
                    hits[f"[warp .{attr}()] " + _attribute()] += 1
                return orig(self, *a, **kw)
            return wrapper

        setattr(wp.array, attr, make(original))

    env_cfg = parse_env_cfg(TASK, num_envs=N_ENVS)
    for field, value in _SOLVER_FLAGS[SOLVER].items():
        setattr(env_cfg.sim.physics.solver_cfg, field, value)

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)

    torch.manual_seed(7)
    with torch.inference_mode():
        for _ in range(WARMUP):
            act.uniform_(-1.0, 1.0)
            env.step(act)

    global recording
    recording = True
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.inference_mode():
            for _ in range(STEPS):
                act.uniform_(-1.0, 1.0)
                env.step(act)
    torch.cuda.set_sync_debug_mode("default")
    recording = False

    for w in caught:
        if "synchron" in str(w.message).lower():
            short = "/".join(str(w.filename).split("/")[-3:])
            hits[f"[torch sync] {short}:{w.lineno}"] += 1

    total = sum(hits.values())
    print(f"\nsolver={SOLVER}  envs={N_ENVS}  steps={STEPS}")
    print(f"{total} host synchronizations over {STEPS} control steps = {total/STEPS:.1f} per step\n")
    if not total:
        print("no synchronizing call sites detected in the stepped window")
        return 0
    print(f"{'per step':>9} {'total':>7}  call site")
    print("-" * 100)
    for site, n in hits.most_common(TOP):
        print(f"{n/STEPS:>9.1f} {n:>7}  {site}")
    shown = sum(n for _, n in hits.most_common(TOP))
    if total > shown:
        print(f"{(total-shown)/STEPS:>9.1f} {total-shown:>7}  (remaining sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
