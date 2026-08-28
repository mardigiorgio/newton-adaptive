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

from scripts.bench.four_arms import ARMS, build_model, make_arm
from scripts.scenes.cenic_scenes import DT_OUTER, SCENES


def _run(scene: str, arm_name: str, n: int, steps: int, warmup: int, seed: int, n_sub: int, tol: float) -> dict:
    kwargs = {"n_sub": n_sub} if arm_name in ("mujoco", "icf") else {"tol": tol}
    model = build_model(n, seed=seed, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(warmup):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    wp.synchronize()
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        s0, s1 = arm.boundary(s0, s1, ctrl)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    fixed = arm_name in ("mujoco", "icf")
    return {
        "scene": scene,
        "arm": arm_name,
        "accuracy": "" if fixed else tol,
        "dt_s": DT_OUTER / n_sub if fixed else "",
        "n_worlds": n,
        "wall_ms_median": float(np.median(times) * 1e3),
        "wall_ms_p90": float(np.quantile(times, 0.9) * 1e3),
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
            r = subprocess.run(
                [
                    sys.executable, "-m", "scripts.bench.benchmarks.part1_scaling",
                    "--scene", args.scene, "--single", arm_name, str(n),
                    "--steps", str(args.steps), "--warmup", str(args.warmup),
                    "--seed", str(args.seed), "--n-sub", str(args.n_sub), "--tol", str(args.tol),
                ],
                capture_output=True, text=True,
            )
            row = None
            for line in r.stdout.splitlines():
                if line.startswith("ROW "):
                    row = json.loads(line[4:])
            if row is None:
                print(f"CONFIG FAILED {arm_name} n={n}:\n{r.stderr[-600:]}", flush=True)
                continue
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
