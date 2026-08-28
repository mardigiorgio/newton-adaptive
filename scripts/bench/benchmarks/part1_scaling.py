# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Wall time vs world count, four solver arms.

Median wall-clock per outer boundary on the shared contact scene at a
ladder of world counts, one subprocess per (arm, N) configuration (GPU
state isolation, as everywhere in the Part-1 suite). Fixed arms run at
their paper operating point (n_sub from --n-sub, default 1); adaptive
arms at the paper tolerance (--tol, default 1e-3).

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_scaling   # 2^6 .. 2^13 worlds
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import warp as wp

from scripts.bench.four_arms import ARMS, ExhaustionTracker, build_model, make_arm
from scripts.scenes.cenic_scenes import DT_OUTER, SCENES


MAX_SUBSTEPS = 4096


def _run(scene: str, arm_name: str, n: int, steps: int, warmup: int, seed: int, n_sub: int, tol: float) -> dict:
    fixed = arm_name in ("mujoco", "icf")
    kwargs = {"n_sub": n_sub} if fixed else {"tol": tol, "max_substeps": MAX_SUBSTEPS}
    model = build_model(n, seed=seed, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    tracker = ExhaustionTracker(arm) if not fixed else None
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(warmup):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
    wp.synchronize()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "scene": scene,
        "arm": arm_name,
        "accuracy": "" if fixed else tol,
        "dt_s": DT_OUTER / n_sub if fixed else "",
        "max_substeps": "" if fixed else MAX_SUBSTEPS,
        "n_worlds": n,
        "wall_ms_median": float(np.median(times) * 1e3),
        "wall_ms_p90": float(np.quantile(times, 0.9) * 1e3),
        "exhausted_frac": tracker.fraction() if tracker else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--ns", nargs="*", type=int, default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-sub", type=int, default=1)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--trials", type=int, default=3, help="independent subprocess runs; median of per-run medians, band = min..max")
    p.add_argument("--single", nargs=2, metavar=("ARM", "N"), default=None)
    args = p.parse_args()
    out = args.out or f"scripts/bench/results/part1_scaling_{args.scene}.csv"

    if args.single is not None:
        arm_name, n_s = args.single
        row = _run(args.scene, arm_name, int(n_s), args.steps, args.warmup, args.seed, args.n_sub, args.tol)
        print("ROW " + json.dumps(row), flush=True)
        return 0

    rows = []
    for arm_name in args.arms:
        for n in args.ns:
            trials = []
            for _ in range(args.trials):
                r = subprocess.run(
                    [
                        sys.executable, "-m", "scripts.bench.benchmarks.part1_scaling",
                        "--scene", args.scene, "--single", arm_name, str(n),
                        "--steps", str(args.steps), "--warmup", str(args.warmup),
                        "--seed", str(args.seed), "--n-sub", str(args.n_sub), "--tol", str(args.tol),
                    ],
                    capture_output=True, text=True,
                )
                got = None
                for line in r.stdout.splitlines():
                    if line.startswith("ROW "):
                        got = json.loads(line[4:])
                if got is None:
                    print(f"CONFIG FAILED {arm_name} n={n}:\n{r.stderr[-600:]}", flush=True)
                    break
                trials.append(got)
            if not trials:
                continue
            # the timed window spans the chaotic impact phase, so independent
            # runs scatter: report the median of per-run medians and the
            # spread of those medians across runs
            meds = sorted(t["wall_ms_median"] for t in trials)
            row = dict(trials[0])
            row["wall_ms_median"] = float(np.median(meds))
            row["wall_ms_p90"] = float(max(t["wall_ms_p90"] for t in trials))
            row["wall_ms_trial_min"] = meds[0]
            row["wall_ms_trial_max"] = meds[-1]
            row["trials"] = len(trials)
            row["exhausted_frac"] = max(t["exhausted_frac"] for t in trials)
            rows.append(row)
            print(row, flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
