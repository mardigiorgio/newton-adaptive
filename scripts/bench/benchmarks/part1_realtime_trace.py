# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Real-time rate over simulation time: per-boundary wall for each arm
along one long drop (default 5 s) on hard clutter, N worlds. Fixed step
pays the same every boundary; error control pays for impacts and coasts
at dt_max once the pile settles. The companion figure integrates the
trace into cumulative wall, so "cost to simulate T seconds at artifact-
free quality" is read directly.

One subprocess per arm; every boundary is timed with a device sync
(this trace is about the shape of the cost, not its floor).

    uv run python -m scripts.bench.benchmarks.part1_realtime_trace --scene hard-clutter --n 64 --horizon 5
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

from scripts.bench.four_arms import ExhaustionTracker, build_model, make_arm, scene_dt_outer
from scripts.scenes.cenic_scenes import SCENES

FIXED_DTS = [1e-2, 1e-3]
CONFIGS = [("icf", "fixed"), ("icf-adaptive", 1e-2), ("icf-adaptive", 1e-3), ("mujoco", "fixed"), ("mujoco-adaptive", 1e-3)]


def _run(scene: str, arm_name: str, knob, n: int, horizon: float) -> dict:
    fixed = arm_name in ("mujoco", "icf")
    kwargs = {"n_sub": knob} if fixed else {"tol": knob, "max_substeps": 4096}
    model = build_model(n, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    tracker = ExhaustionTracker(arm) if not fixed else None
    cum = getattr(arm.solver, "_march_iters", None)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(2):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
    wp.synchronize()
    walls, iters = [], []
    last = int(cum.numpy()[0]) if cum is not None else 0
    for _ in range(int(round(horizon / arm.dt_outer)) - 2):
        t0 = time.perf_counter()
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
        wp.synchronize()
        walls.append(time.perf_counter() - t0)
        if cum is not None:
            now = int(cum.numpy()[0])
            iters.append(now - last)
            last = now
        elif not fixed:
            iters.append(arm.iteration_count())
        else:
            iters.append(knob)
    return {"wall_s": walls, "iters": iters, "exhausted_frac": tracker.fraction() if tracker else 0.0}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--horizon", type=float, default=5.0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--single", nargs=2, metavar=("ARM", "KNOB"), default=None)
    args = p.parse_args()
    out = args.out or f"scripts/bench/results/part1_realtime_trace_{args.scene}_n{args.n}.csv"

    if args.single is not None:
        arm_name, knob_s = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        print("ROW " + json.dumps(_run(args.scene, arm_name, knob, args.n, args.horizon)), flush=True)
        return 0

    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = []
    dt_outer = scene_dt_outer(args.scene)
    expanded = []
    for arm_name, knob in CONFIGS:
        if knob == "fixed":
            expanded += [(arm_name, int(round(dt_outer / d))) for d in FIXED_DTS]
        else:
            expanded.append((arm_name, knob))
    for arm_name, knob in expanded:
        r = subprocess.run(
            [sys.executable, "-m", "scripts.bench.benchmarks.part1_realtime_trace", "--scene", args.scene,
             "--single", arm_name, str(knob), "--n", str(args.n), "--horizon", str(args.horizon)],
            capture_output=True, text=True, timeout=7200,
        )
        if "over the scannable budget" in r.stderr:
            print(f"CONTACT OVERFLOW {arm_name} {knob}", flush=True)
            continue
        got = None
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                got = json.loads(line[4:])
        if got is None:
            print(f"FAIL {arm_name} {knob}: {r.stderr[-300:]}", flush=True)
            continue
        fixed = arm_name in ("mujoco", "icf")
        for i, (w, it) in enumerate(zip(got["wall_s"], got["iters"])):
            rows.append({"scene": args.scene, "arm": arm_name, "accuracy": "" if fixed else knob,
                         "dt_s": dt_outer / knob if fixed else "", "dt_outer_s": dt_outer, "n_worlds": args.n,
                         "t_s": (i + 2) * dt_outer, "wall_ms": w * 1e3, "iters": it,
                         "exhausted_frac": got["exhausted_frac"]})
        print(f"{arm_name} {knob}: {len(got['wall_s'])} boundaries, total wall {sum(got['wall_s']):.2f} s, exhausted {got['exhausted_frac']:.2f}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
