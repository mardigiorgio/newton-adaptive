# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Work-precision, CENIC Fig. 9/10 definition, four arms.

    x = requested accuracy eps_acc (adaptive arms),  y = wall time per
    simulated second.  "Missing data points indicate solver failure or
    timeout after 100 seconds (real-time rate < 1%)."  (Fig. 10 caption)

The adaptive arms sweep eps_acc over 1e-1 .. 1e-6; the fixed arms have no
accuracy knob and are reported at a ladder of time steps dt as reference
levels (the paper's Fig. 11 does the same with delta t = 10 ms / 1 ms).
Every row states its accuracy or its time step explicitly.

Scenes are the paper's (scripts/scenes/cenic_scenes.py); the default is
hard clutter. Wall time excludes the first two boundaries (eager module
load, graph capture) and is scaled to one simulated second. One
subprocess per configuration.

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_workprecision --scene hard-clutter
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import warp as wp

from scripts.bench.four_arms import build_model, make_arm
from scripts.scenes.cenic_scenes import DT_OUTER, SCENES

ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
FIXED_N_SUB = [1, 2, 5, 10]  # dt = 10, 5, 2, 1 ms
TIMEOUT_PER_SIM_S = 100.0  # the paper's criterion, per simulated second


def _run(scene: str, arm_name: str, knob, n: int, horizon: float) -> dict:
    kwargs = {"n_sub": knob} if arm_name in ("mujoco", "icf") else {"tol": knob}
    model = build_model(n, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    boundaries = int(round(horizon / DT_OUTER))
    for _ in range(2):  # eager load + capture, untimed
        s0, s1 = arm.boundary(s0, s1, ctrl)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(boundaries - 2):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    wp.synchronize()
    wall = time.perf_counter() - t0
    sim_s = (boundaries - 2) * DT_OUTER
    return {"wall_s_per_sim_s": wall / sim_s}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=1, help="worlds; the paper's plots are single-scene")
    p.add_argument("--horizon", type=float, default=None, help="simulated seconds (default: the scene's)")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--single", nargs=2, metavar=("ARM", "KNOB"), default=None)
    args = p.parse_args()
    horizon = args.horizon or SCENES[args.scene].horizon_s
    out = args.out or f"scripts/bench/results/part1_workprecision_{args.scene}.csv"

    if args.single is not None:
        arm_name, knob_s = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        print("ROW " + json.dumps(_run(args.scene, arm_name, knob, args.n, horizon)), flush=True)
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    configs = [(a, k) for a in ("mujoco-adaptive", "icf-adaptive") for k in ACCURACIES]
    configs += [(a, k) for a in ("mujoco", "icf") for k in FIXED_N_SUB]
    for arm_name, knob in configs:
        fixed = arm_name in ("mujoco", "icf")
        row = {
            "scene": args.scene,
            "arm": arm_name,
            "accuracy": "" if fixed else knob,
            "dt_s": DT_OUTER / knob if fixed else "",
            "n_worlds": args.n,
            "horizon_s": horizon,
            "wall_s_per_sim_s": "",
            "status": "ok",
        }
        try:
            r = subprocess.run(
                [
                    sys.executable, "-m", "scripts.bench.benchmarks.part1_workprecision",
                    "--scene", args.scene, "--single", arm_name, str(knob),
                    "--n", str(args.n), "--horizon", str(horizon),
                ],
                capture_output=True, text=True, timeout=TIMEOUT_PER_SIM_S * horizon + 120,
            )
            got = None
            for line in r.stdout.splitlines():
                if line.startswith("ROW "):
                    got = json.loads(line[4:])
            if got is None:
                row["status"] = "fail"
                print(f"FAIL {arm_name} {knob}: {r.stderr[-300:]}", flush=True)
            else:
                row["wall_s_per_sim_s"] = got["wall_s_per_sim_s"]
                if got["wall_s_per_sim_s"] > TIMEOUT_PER_SIM_S:
                    row["status"] = "timeout"
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        rows.append(row)
        print(row, flush=True)

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
