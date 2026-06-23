"""Smoke test: spawn STATIONARY_AI_CFG, assert 16 DOF, step, dump joint names.

Validates the greenfield articulation loads and the carriage-mimic assumption holds
(no error about driving a mimic / unactuated joints). Writes its result to a JSON
file because Kit swallows stdout once the app launches.

Run natively:
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/trossen_cube/tests/test_articulation_smoke.py
"""

import json
import os
import traceback

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from trossen_cube.assets import STATIONARY_AI_CFG  # noqa: E402
from trossen_cube.paths import ARTIFACT_ROOT  # noqa: E402

OUT = os.environ.get("SMOKE_OUT") or os.path.join(ARTIFACT_ROOT, "articulation_smoke.json")

res: dict = {}
try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(STATIONARY_AI_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    res["num_joints"] = int(robot.num_joints)
    res["joint_names"] = list(robot.joint_names)
    for _ in range(5):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.01)
    res["finite_after_steps"] = bool(torch.isfinite(robot.data.joint_pos).all())
    res["ok"] = res["num_joints"] == 16 and res["finite_after_steps"]
except Exception as e:
    res = {"ok": False, "err": repr(e), "tb": traceback.format_exc()[-1000:]}

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)

# SimulationContext.app.close() can hang on shutdown for standalone scripts; force-exit
# with a result-based status so the test's exit code is meaningful (and it doesn't time out).
os._exit(0 if res.get("ok") else 1)
