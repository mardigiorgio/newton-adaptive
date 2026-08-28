# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""March cost of the error-controlled arms vs requested accuracy: march
iterations per simulated second and wall per iteration, one world, hard
clutter. Separates "the controller takes many steps" from "each step is
expensive" -- the split that turned a 46 s/sim-s timeout into 0.7 s once
dropped contacts (the real cause of the step blow-up) were fixed. The
paper's Table II gives ~4k..96k Newton iterations for eps 1e-1..1e-5 on
its CPU hard clutter (~500 steps at 1e-3).

    uv run python scripts/bench/probe_march_cost.py [--scene hard-clutter] [--n 1]
"""

from __future__ import annotations

import argparse
import time

import warp as wp

from scripts.bench.four_arms import ExhaustionTracker, build_model, make_arm, scene_dt_outer
from scripts.scenes.cenic_scenes import SCENES


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--accuracies", nargs="*", type=float, default=[1e-2, 1e-3, 1e-4, 1e-5])
    args = p.parse_args()
    dt_outer = scene_dt_outer(args.scene)
    B = int(round(args.horizon / dt_outer))
    print(f"{'arm':16s} {'eps':>6} {'iters/sim_s':>11} {'wall_s/sim_s':>12} {'us/iter':>8} {'exhausted':>9}")
    for arm_name in ("icf-adaptive", "mujoco-adaptive"):
        for eps in args.accuracies:
            m = build_model(args.n, scene=args.scene)
            a = make_arm(m, arm_name, scene=args.scene, tol=eps, max_substeps=4096)
            tr = ExhaustionTracker(a)
            s0, s1, c = m.state(), m.state(), m.control()
            for _ in range(2):
                s0, s1 = a.boundary(s0, s1, c)
                tr.tick()
            cum = getattr(a.solver, "_march_iters", None)
            it0 = int(cum.numpy()[0]) if cum is not None else 0
            per_boundary = 0
            wp.synchronize()
            t0 = time.perf_counter()
            for _ in range(B - 2):
                s0, s1 = a.boundary(s0, s1, c)
                tr.tick()
                if cum is None:  # MuJoCo-adaptive exposes a per-boundary count only
                    per_boundary += a.iteration_count()
            wp.synchronize()
            wall = time.perf_counter() - t0
            iters = (int(cum.numpy()[0]) - it0) if cum is not None else per_boundary
            sim = (B - 2) * dt_outer
            print(f"{arm_name:16s} {eps:>6.0e} {iters / sim:>11.0f} {wall / sim:>12.2f} {1e6 * wall / max(iters, 1):>8.0f} {tr.fraction():>9.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
