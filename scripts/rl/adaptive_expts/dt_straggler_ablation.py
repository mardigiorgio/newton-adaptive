# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Why the ragged adaptive loop anti-scales, and which knobs actually buy wall-clock.

Two experiments on the same Allegro in-hand scene.

**A. Does an inactive world cost a full eval?**
``_clamp_dt_to_boundary`` zeroes ``dt`` for worlds that already landed, but ``_step_double``
is one batched kernel over every world. If a ``dt=0`` world still pays a full MuJoCo eval,
then the straggler waste (max-over-worlds iterations vs the mean) is real wall-clock. If
mujoco_warp early-exits its constraint solve for a world that does not move, the waste is
mostly free and capping ``max_substeps`` buys far less than the iteration counts suggest.

Measured by comparing ms-per-iteration in two regimes at identical world count:
  * ALL-ACTIVE  -- dt_inner_init = dt_outer/K with a loose tol, so every world takes the
    same K accepted steps and no world lands early. Active fraction = 100%.
  * RAGGED      -- the normal configuration, where the active fraction decays within each
    boundary as worlds land.

**B. Knob ablation.** Wall-clock, iterations, and the accuracy cost of each mitigation.
``unfinished_worlds`` is the honest damage counter for ``max_substeps`` capping: it counts
worlds that exited a boundary with ``sim_time < next_time``, i.e. silently under-advanced.

Usage:
    uv run python scripts/rl/adaptive_expts/dt_straggler_ablation.py --worlds 256
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

_MJ = {
    "solver": "newton",
    "integrator": "implicitfast",
    "njmax": 200,
    "nconmax": 300,
    "impratio": 20.0,
    "cone": "elliptic",
    "iterations": 100,
    "ls_iterations": 50,
}


def make(worlds: int, **kw):
    """Build the scene + an adaptive solver, returning everything the loop needs."""
    model, rot = build_scene(worlds)
    solver = newton.solvers.SolverMuJoCoAdaptive(model, dt_histogram=True, **_MJ, **kw)
    s0, s1 = model.state(), model.state()
    control = model.control()
    wtime = wp.zeros(worlds, dtype=wp.float32)

    def drive(dt_outer: float):
        wp.launch(
            _move_hand,
            dim=worlds,
            inputs=[
                model.joint_q_start,
                model.joint_limit_lower,
                model.joint_limit_upper,
                wtime,
                dt_outer,
                rot,
            ],
            outputs=[control.joint_target_q, model.joint_X_p],
        )
        solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    return model, solver, s0, s1, control, drive


def timed_run(solver, s0, s1, control, drive, dt_outer, frames, warmup):
    """Run ``frames`` boundaries after ``warmup``; return (wall_s, stats, final joint_q)."""
    for _ in range(warmup):
        drive(dt_outer)
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)
    wp.synchronize_device()
    solver.reset_dt_histogram()
    solver.reset_compute_counter()

    t0 = time.perf_counter()
    for _ in range(frames):
        drive(dt_outer)
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)
    wp.synchronize_device()
    wall = time.perf_counter() - t0

    stats = solver.dt_histogram_stats()
    stats["adaptive_iterations"] = int(solver.cumulative_iterations.numpy()[0])
    return wall, stats, s0.joint_q.numpy().copy()


def experiment_a(args) -> dict:
    """Is a landed (dt=0) world free, or does it pay a full eval?"""
    dt_outer = args.dt_outer
    K = 4  # forced steps per boundary in the all-active regime

    # ALL-ACTIVE: dt pinned to dt_outer/K by dt_inner_max, loose tol so nothing rejects.
    _, sol_a, a0, a1, ca, da = make(
        args.worlds,
        tol=1e9,
        dt_inner_init=dt_outer / K,
        dt_inner_max=dt_outer / K,
        dt_inner_min=dt_outer / K / 10.0,
        max_substeps=256,
    )
    wall_a, st_a, _ = timed_run(sol_a, a0, a1, ca, da, dt_outer, args.frames, args.warmup)
    iters_a = st_a["adaptive_iterations"]
    active_a = st_a["total_samples"] / max(iters_a * args.worlds, 1)

    # RAGGED: the normal configuration.
    _, sol_r, r0, r1, cr, dr = make(args.worlds, tol=args.tol, dt_inner_init=1e-2, dt_inner_min=1e-6, max_substeps=256)
    wall_r, st_r, _ = timed_run(sol_r, r0, r1, cr, dr, dt_outer, args.frames, args.warmup)
    iters_r = st_r["adaptive_iterations"]
    active_r = st_r["total_samples"] / max(iters_r * args.worlds, 1)

    ms_it_a = 1e3 * wall_a / max(iters_a, 1)
    ms_it_r = 1e3 * wall_r / max(iters_r, 1)
    return {
        "all_active": {
            "iterations": iters_a,
            "active_fraction": round(active_a, 4),
            "ms_per_iteration": round(ms_it_a, 4),
        },
        "ragged": {
            "iterations": iters_r,
            "active_fraction": round(active_r, 4),
            "ms_per_iteration": round(ms_it_r, 4),
        },
        # 1.0 => an inactive world costs exactly as much as an active one (waste is real).
        # ~active_fraction => inactive worlds are nearly free (waste is mostly an illusion).
        "cost_ratio_ragged_over_all_active": round(ms_it_r / ms_it_a, 4),
        "ragged_active_fraction": round(active_r, 4),
    }


def experiment_b(args) -> list[dict]:
    """Wall-clock and accuracy cost of each mitigation."""
    dt_outer = args.dt_outer
    variants = [
        ("baseline (defaults)", {}, {}),
        ("CONDITIONAL=1", {}, {"NEWTON_MJ_ADAPTIVE_CONDITIONAL": "1"}),
        ("max_substeps=8", {"max_substeps": 8}, {}),
        ("ORDER_AWARE=1", {}, {"NEWTON_ADAPTIVE_ORDER_AWARE": "1"}),
        ("FILTERED_ERR=1", {}, {"NEWTON_ADAPTIVE_FILTERED_ERR": "1"}),
        (
            "all combined (cap=8)",
            {"max_substeps": 8},
            {
                "NEWTON_MJ_ADAPTIVE_CONDITIONAL": "1",
                "NEWTON_ADAPTIVE_ORDER_AWARE": "1",
                "NEWTON_ADAPTIVE_FILTERED_ERR": "1",
            },
        ),
    ]
    keys = [
        "NEWTON_MJ_ADAPTIVE_CONDITIONAL",
        "NEWTON_ADAPTIVE_ORDER_AWARE",
        "NEWTON_ADAPTIVE_FILTERED_ERR",
    ]
    out, ref_q = [], None
    for label, kw, env in variants:
        for k in keys:
            os.environ.pop(k, None)
        os.environ.update(env)
        cfg = {"tol": args.tol, "dt_inner_init": 1e-2, "dt_inner_min": 1e-6, "max_substeps": 256}
        cfg.update(kw)
        _, solver, s0, s1, control, drive = make(args.worlds, **cfg)
        wall, st, q = timed_run(solver, s0, s1, control, drive, dt_outer, args.frames, args.warmup)
        if ref_q is None:
            ref_q = q
        rec = {
            "variant": label,
            "ms_per_tick": round(1e3 * wall / args.frames, 3),
            "iters_per_boundary": round(st["adaptive_iterations"] / max(st["boundaries"], 1), 2),
            "unfinished_worlds": st["unfinished_worlds"],
            "capped_boundaries": st["capped_boundaries"],
            "floor_pct": round(100.0 * st["floor_fraction"], 4),
            # Divergence from the baseline's final state: the accuracy price of the knob.
            "max_abs_dq_vs_baseline": float(np.abs(q - ref_q).max()),
        }
        out.append(rec)
    for k in keys:
        os.environ.pop(k, None)
    base = out[0]["ms_per_tick"]
    for r in out:
        r["speedup_vs_baseline"] = round(base / r["ms_per_tick"], 2)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--dt-outer", type=float, default=1.0 / 120.0)
    p.add_argument("--skip-a", action="store_true")
    args = p.parse_args()

    report: dict = {"worlds": args.worlds, "frames": args.frames, "tol": args.tol}

    if not args.skip_a:
        a = experiment_a(args)
        report["experiment_a_inactive_world_cost"] = a
        print("\n=== A. Does a landed (dt=0) world still cost a full eval? ===")
        print(
            f"  all-active : {a['all_active']['iterations']:>5} iters, "
            f"active={100 * a['all_active']['active_fraction']:.0f}%, "
            f"{a['all_active']['ms_per_iteration']:.3f} ms/iter"
        )
        print(
            f"  ragged     : {a['ragged']['iterations']:>5} iters, "
            f"active={100 * a['ragged']['active_fraction']:.0f}%, "
            f"{a['ragged']['ms_per_iteration']:.3f} ms/iter"
        )
        r = a["cost_ratio_ragged_over_all_active"]
        af = a["ragged_active_fraction"]
        print(f"  cost ratio = {r:.2f}  (1.00 => inactive worlds pay FULL price; {af:.2f} => nearly free)")

    b = experiment_b(args)
    report["experiment_b_knob_ablation"] = b
    print(f"\n=== B. Knob ablation at {args.worlds} worlds ===")
    print(f"{'variant':<24}{'ms/tick':>9}{'speedup':>9}{'it/bnd':>8}{'unfin':>8}{'capped':>8}{'max|dq|':>11}")
    for r in b:
        print(
            f"{r['variant']:<24}{r['ms_per_tick']:>9.2f}{r['speedup_vs_baseline']:>9.2f}"
            f"{r['iters_per_boundary']:>8.2f}{r['unfinished_worlds']:>8}{r['capped_boundaries']:>8}"
            f"{r['max_abs_dq_vs_baseline']:>11.2e}"
        )

    with open("dt_straggler_ablation.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nwrote dt_straggler_ablation.json")


if __name__ == "__main__":
    main()
