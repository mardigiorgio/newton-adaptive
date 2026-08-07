# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Where the adaptive solver's wall-clock goes, versus the fixed-step solver.

Runs the SAME Allegro in-hand scene through three configurations for the same number
of control boundaries and reports wall time, so the slowdown decomposes into:

  1. fixed          -- SolverMuJoCo, num_substeps=2 (the IsaacLab baseline): 2 evals/tick
  2. adaptive       -- default tier: per-iteration graph replay + a boundary-flag
                       ``.numpy()`` poll BETWEEN iterations, i.e. one full device sync
                       per adaptive iteration (solver_mujoco_adaptive.py:1478)
  3. adaptive+cond  -- NEWTON_MJ_ADAPTIVE_CONDITIONAL=1: the whole march becomes one
                       conditional CUDA-graph node, removing every per-iteration sync

(2) vs (3) isolates the host-sync tax. (3) vs (1) is the intrinsic cost: step doubling
runs 3 MuJoCo evals per iteration, and the ragged loop runs until the SLOWEST world in
the batch lands, so cost tracks max-over-worlds iterations, not the mean.

Usage:
    uv run python scripts/rl/adaptive_expts/dt_cost_breakdown.py --worlds 16 --frames 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import warp as wp

import newton
from newton import ModelFlags

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dt_trace_allegro import _move_hand, build_scene

wp.init()


def run_case(label: str, adaptive: bool, args, dt_outer: float) -> dict:
    model, hand_rotation = build_scene(args.worlds)
    common = {
        "solver": "newton",
        "integrator": "implicitfast",
        "njmax": 200,
        "nconmax": 300,
        "impratio": 20.0,
        "cone": "elliptic",
        "iterations": 100,
        "ls_iterations": 50,
    }
    if adaptive:
        solver = newton.solvers.SolverMuJoCoAdaptive(
            model,
            tol=args.tol,
            dt_inner_init=args.dt_init,
            dt_inner_min=args.dt_min,
            max_substeps=args.max_substeps,
            dt_histogram=True,
            **common,
        )
    else:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, **common)

    s0, s1 = model.state(), model.state()
    control = model.control()
    world_time = wp.zeros(args.worlds, dtype=wp.float32)
    fixed_substeps = 2
    fixed_dt = dt_outer / fixed_substeps

    def drive():
        wp.launch(
            _move_hand,
            dim=args.worlds,
            inputs=[
                model.joint_q_start,
                model.joint_limit_lower,
                model.joint_limit_upper,
                world_time,
                dt_outer,
                hand_rotation,
            ],
            outputs=[control.joint_target_q, model.joint_X_p],
        )
        solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    def one_tick(a, b):
        if adaptive:
            return solver.step_dt(dt_outer, a, b, control)
        for _ in range(fixed_substeps):
            solver.step(a, b, control, None, fixed_dt)
            a, b = b, a
        return a, b

    for _ in range(args.warmup):
        drive()
        s0, s1 = one_tick(s0, s1)
    wp.synchronize_device()

    t0 = time.perf_counter()
    for _ in range(args.frames):
        drive()
        s0, s1 = one_tick(s0, s1)
    wp.synchronize_device()
    wall = time.perf_counter() - t0

    rec = {
        "case": label,
        "wall_s": round(wall, 4),
        "ms_per_tick": round(1e3 * wall / args.frames, 3),
    }
    if adaptive:
        st = solver.dt_histogram_stats()
        iters = int(solver.cumulative_iterations.numpy()[0])
        rec["adaptive_iterations"] = iters
        rec["iters_per_boundary"] = round(iters / max(st["boundaries"], 1), 2)
        rec["mujoco_evals"] = iters * 3
        rec["host_syncs"] = iters  # one boundary-flag poll per iteration in the default tier
    else:
        rec["mujoco_evals"] = args.frames * fixed_substeps
        rec["host_syncs"] = 0
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=16)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--dt-min", type=float, default=1e-6)
    p.add_argument("--dt-init", type=float, default=1e-2)
    p.add_argument("--max-substeps", type=int, default=256)
    p.add_argument("--dt-outer", type=float, default=1.0 / 120.0)
    args = p.parse_args()

    results = [run_case("fixed (num_substeps=2)", False, args, args.dt_outer)]

    os.environ["NEWTON_MJ_ADAPTIVE_CONDITIONAL"] = "0"
    results.append(run_case("adaptive (default: per-iteration host poll)", True, args, args.dt_outer))

    os.environ["NEWTON_MJ_ADAPTIVE_CONDITIONAL"] = "1"
    results.append(run_case("adaptive (NEWTON_MJ_ADAPTIVE_CONDITIONAL=1)", True, args, args.dt_outer))

    base = results[0]["ms_per_tick"]
    for r in results:
        r["slowdown_vs_fixed"] = round(r["ms_per_tick"] / base, 2)
        r["eval_ratio_vs_fixed"] = round(r["mujoco_evals"] / results[0]["mujoco_evals"], 2)

    print(json.dumps({"worlds": args.worlds, "frames": args.frames, "tol": args.tol, "cases": results}, indent=2))
    print()
    print(f"{'case':<46}{'ms/tick':>10}{'slowdown':>10}{'evals x':>9}{'syncs':>9}")
    for r in results:
        print(
            f"{r['case']:<46}{r['ms_per_tick']:>10.3f}{r['slowdown_vs_fixed']:>10.2f}"
            f"{r['eval_ratio_vs_fixed']:>9.2f}{r['host_syncs']:>9}"
        )
    a_def = results[1]["ms_per_tick"]
    a_con = results[2]["ms_per_tick"]
    print()
    print(f"host-sync tax (default vs conditional): {a_def / a_con:.2f}x  ({a_def - a_con:.3f} ms/tick)")
    print(f"intrinsic cost (conditional vs fixed):  {a_con / base:.2f}x")
    np.savez("dt_cost_breakdown.npz", results=json.dumps(results))


if __name__ == "__main__":
    main()
