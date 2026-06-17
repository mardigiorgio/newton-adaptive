"""Build the teacher env, reset, step, assert the policy + privileged obs groups.

Writes its result to JSON (Kit swallows stdout). Run inside the container:
    podman exec isaaclab bash -lc \
      "cd /repo && /opt/venv/bin/python scripts/rl/trossen/trossen_cube/tests/test_env_smoke.py"
"""

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402  (registers gym ids)

OUT = os.environ.get("ENV_SMOKE_OUT", "/isaac/env_smoke.json")
TASK = "Isaac-Lift-Cube-StationaryAI-Teacher-v0"

res: dict = {}
try:
    env_cfg = parse_env_cfg(TASK, num_envs=4)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    obs, _ = env.reset()
    groups = sorted(obs.keys())
    res["obs_groups"] = groups
    res["policy_shape"] = list(obs["policy"].shape)
    res["privileged_shape"] = list(obs["privileged"].shape)
    res["action_dim"] = int(u.action_manager.total_action_dim)
    act = torch.zeros((u.num_envs, res["action_dim"]), device=u.device)
    rew_finite = True
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(act)
        rew_finite = rew_finite and bool(torch.isfinite(rew).all())
    res["rew_finite"] = rew_finite
    res["ok"] = "policy" in groups and "privileged" in groups and rew_finite
    env.close()
except Exception as e:  # noqa: BLE001
    res = {"ok": False, "err": repr(e), "tb": traceback.format_exc()[-1500:]}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

os._exit(0 if res.get("ok") else 1)
