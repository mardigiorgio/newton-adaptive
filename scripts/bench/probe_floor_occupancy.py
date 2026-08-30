# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Floor occupancy of the two error-controlled arms at the tightest accuracies.

Both controllers carry a dt_inner_min floor the paper's Alg. 1 does not have.
An accepted step AT the floor is committed without meeting the accuracy test,
so a run that touched the floor is not a clean error-controlled result. This
probe counts floor hits per pass (MuJoCo: the solver's dt histogram, bin 0;
ICF: the solver's floor_count) on the work-precision protocol (one world,
budget 65536, 2 s) for the accuracies where the step is smallest.

    uv run python -m scripts.bench.probe_floor_occupancy
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import newton
import warp as wp

SCENES = ["hard-clutter", "soft-clutter"]
ARMS = ["mujoco-adaptive", "icf-adaptive"]
TOLS = [1e-4, 1e-5, 1e-6]
SIM_S = 2.0
BUDGET = 65536
OUT = "scripts/bench/results/tables/floor_occupancy.md"


def _single(scene: str, arm_name: str, tol: float) -> dict:
    orig = newton.solvers.SolverMuJoCoAdaptive.__init__

    def init(self, *a, **k):
        k["dt_histogram"] = True
        orig(self, *a, **k)

    newton.solvers.SolverMuJoCoAdaptive.__init__ = init
    from scripts.bench.four_arms import ExhaustionTracker, build_model, make_arm

    model = build_model(1, seed=42, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, tol=tol, max_substeps=BUDGET)
    tracker = ExhaustionTracker(arm)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(int(round(SIM_S / arm.dt_outer))):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        tracker.tick()
    wp.synchronize()
    if arm_name == "mujoco-adaptive":
        st = arm.solver.dt_histogram_stats()
        total, floor = int(st["total_samples"]), int(st["floor_samples"])
    else:
        sol = arm.solver
        floor = int(sol.floor_count.numpy().sum())
        total = int(sol.accepted_count.numpy().sum() + sol.rejected_count.numpy().sum())
    return {"attempts": total, "floor_hits": floor, "exhausted_frac": tracker.fraction()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", nargs=3, metavar=("SCENE", "ARM", "TOL"), default=None)
    args = p.parse_args()
    if args.single:
        print("ROW " + json.dumps(_single(args.single[0], args.single[1], float(args.single[2]))), flush=True)
        return 0
    lines = [
        "# Floor occupancy of the error-controlled arms (dt_inner_min = 1e-6 s, budget 65536, one world, 2 s)",
        "",
        "A floor hit is an inner step selected at the floor; an accepted floor step skips the accuracy test.",
        "",
        "| scene | arm | eps_acc | attempts | floor hits | budget-exhausted |",
        "|---|---|---|---|---|---|",
    ]
    for scene in SCENES:
        for arm in ARMS:
            for tol in TOLS:
                r = subprocess.run(
                    [sys.executable, "-m", "scripts.bench.probe_floor_occupancy", "--single", scene, arm, str(tol)],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                row = None
                for line in r.stdout.splitlines():
                    if line.startswith("ROW "):
                        row = json.loads(line[4:])
                if row is None:
                    lines.append(f"| {scene} | {arm} | {tol:g} | FAIL | | |")
                    print(f"FAIL {scene} {arm} {tol}: {r.stderr[-400:]}", flush=True)
                    continue
                lines.append(
                    f"| {scene} | {arm} | {tol:g} | {row['attempts']} | {row['floor_hits']} | {row['exhausted_frac']:.2f} |"
                )
                print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
