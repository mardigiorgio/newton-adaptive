# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Work-precision, CENIC Fig. 9/10 definition, four arms.

    x = requested accuracy eps_acc (adaptive arms),  y = wall time per
    simulated second.  "Missing data points indicate solver failure or
    timeout after 100 seconds (real-time rate < 1%)."  (Fig. 10 caption)

The paper's timeout is per simulated second OF ONE SCENE; a batch of N
worlds simulates N scenes, so the criterion here is wall / (N * simulated
seconds) > 100 s -- unchanged at N = 1, and at N = 1024 a batch is only a
timeout past 100 s per world-second. Separately, every run is bounded by
a practical wall budget (``--wall-budget-s``); a run killed by it is
reported ``budget`` (not the paper's timeout) and drawn as a distinct
cross.

The adaptive arms sweep eps_acc over 1e-1 .. 1e-6 with a march budget of
``--max-substeps`` (default 4096, dt floor ~2.4 us) so the accuracy is
genuinely pursued; a run in which ANY world ever exhausted the budget or
latched diverged is reported ``budget-exhausted`` and treated as a
failure — a budget-limited point is not an accuracy-achieved point. The
fixed arms have no accuracy knob and are reported at a ladder of time
steps as reference levels (the paper's Fig. 11 does the same). Every row
states its accuracy or its time step explicitly.

Wall time excludes the first two boundaries (eager module load, graph
capture) and is scaled to one simulated second; ``--trials`` independent
subprocess runs per configuration, median reported (single-scene GPU
timing at N=1 sits near the launch-latency floor and needs repeats).

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_workprecision --scene hard-clutter --n 1 --trials 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time

import warp as wp

from scripts.bench.four_arms import ExhaustionTracker, IterationTracker, build_model, make_arm, scene_dt_outer
from scripts.scenes.cenic_scenes import SCENES

ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
FIXED_DTS = [1e-2, 5e-3, 2e-3, 1e-3]  # fixed-step ladder [s]; n_sub = dt_outer / dt
TIMEOUT_PER_SIM_S = 100.0  # the paper's criterion, per simulated second of ONE scene (per world)


def _run(scene: str, arm_name: str, knob, n: int, horizon: float, max_substeps: int) -> dict:
    fixed = arm_name in ("mujoco", "icf")
    kwargs = {"n_sub": knob} if fixed else {"tol": knob, "max_substeps": max_substeps}
    model = build_model(n, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    tracker = ExhaustionTracker(arm) if not fixed else None
    s0, s1, ctrl = model.state(), model.state(), model.control()
    dt_outer = arm.dt_outer
    boundaries = int(round(horizon / dt_outer))
    for _ in range(2):  # eager load + capture, untimed
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
    iters = IterationTracker(arm) if not fixed else None
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(boundaries - 2):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
        if iters:
            iters.tick()
    wp.synchronize()
    wall = time.perf_counter() - t0
    sim_s = (boundaries - 2) * dt_outer
    return {
        "wall_s_per_sim_s": wall / sim_s,
        "iters_per_boundary": (iters.total() / (boundaries - 2)) if iters else kwargs.get("n_sub", ""),
        "exhausted_frac": tracker.fraction() if tracker else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=1, help="worlds; the paper's plots are single-scene")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=float, default=None, help="simulated seconds (default: the scene's)")
    p.add_argument("--max-substeps", type=int, default=4096)
    p.add_argument("--wall-budget-s", type=float, default=3600.0, help="practical cap per run; exceeding it is 'budget', not the paper's timeout")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--single", nargs=2, metavar=("ARM", "KNOB"), default=None)
    args = p.parse_args()
    horizon = args.horizon or SCENES[args.scene].horizon_s
    out = args.out or f"scripts/bench/results/part1_workprecision_{args.scene}_n{args.n}.csv"

    if args.single is not None:
        arm_name, knob_s = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        print("ROW " + json.dumps(_run(args.scene, arm_name, knob, args.n, horizon, args.max_substeps)), flush=True)
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    dt_outer = scene_dt_outer(args.scene)
    configs = [(a, k) for a in ("mujoco-adaptive", "icf-adaptive") for k in ACCURACIES]
    configs += [(a, int(round(dt_outer / d))) for a in ("mujoco", "icf") for d in FIXED_DTS]
    for arm_name, knob in configs:
        fixed = arm_name in ("mujoco", "icf")
        row = {
            "scene": args.scene,
            "arm": arm_name,
            "accuracy": "" if fixed else knob,
            "dt_s": dt_outer / knob if fixed else "",
            "dt_outer_s": dt_outer,
            "max_substeps": "" if fixed else args.max_substeps,
            "n_worlds": args.n,
            "horizon_s": horizon,
            "trials": args.trials,
            "wall_s_per_sim_s": "",
            "wall_s_per_world_sim_s": "",
            "iters_per_boundary": "",
            "exhausted_frac": "",
            "status": "ok",
        }
        walls, exhausted = [], []
        for _ in range(args.trials):
            try:
                r = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.bench.benchmarks.part1_workprecision",
                        "--scene", args.scene, "--single", arm_name, str(knob),
                        "--n", str(args.n), "--horizon", str(horizon), "--max-substeps", str(args.max_substeps),
                    ],
                    capture_output=True, text=True,
                    timeout=min(args.wall_budget_s, TIMEOUT_PER_SIM_S * horizon * args.n) + 120,
                )
            except subprocess.TimeoutExpired:
                row["status"] = "timeout" if TIMEOUT_PER_SIM_S * horizon * args.n <= args.wall_budget_s else "budget"
                break
            got = None
            if "over the scannable budget" in r.stderr:
                row["status"] = "contact-overflow"
                print(f"CONTACT OVERFLOW {arm_name} {knob}: contacts dropped -- raise the budgets in four_arms.py", flush=True)
                break
            for line in r.stdout.splitlines():
                if line.startswith("ROW "):
                    got = json.loads(line[4:])
            if got is None:
                row["status"] = "fail"
                print(f"FAIL {arm_name} {knob}: {r.stderr[-300:]}", flush=True)
                break
            walls.append(got["wall_s_per_sim_s"])
            exhausted.append(got["exhausted_frac"])
            row["iters_per_boundary"] = got.get("iters_per_boundary", "")
        if walls and row["status"] == "ok":
            row["wall_s_per_sim_s"] = statistics.median(walls)
            row["wall_s_per_world_sim_s"] = row["wall_s_per_sim_s"] / args.n
            row["exhausted_frac"] = max(exhausted)
            if row["wall_s_per_world_sim_s"] > TIMEOUT_PER_SIM_S:
                row["status"] = "timeout"
            elif row["exhausted_frac"] > 0.0:
                row["status"] = "budget-exhausted"
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
