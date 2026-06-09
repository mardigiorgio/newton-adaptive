# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Episode speed gate for SolverMuJoCoAdaptive on contact_objects.

Reusable measurement harness for the autonomous solver-optimization loop. For
each world count N it runs an "episode" (objects drop and settle) and reports
the per-outer-step wall time as both median (steady-state, low-noise -- the
accept/reject signal) and mean (includes the violent drop). It compares the
adaptive solver against the fixed-1ms MuJoCo baseline (ratio > 1 = adaptive
faster) and surfaces the worst-world step-doubling error and any NaN.

This is measurement infrastructure, not a tuning experiment: it is never the
target of a revert. The canonical accuracy gate remains
``scripts.bench.benchmarks.tol_trace``; the ``max_err`` printed here is a coarse
sanity signal at the DT_OUTER=0.02 cadence, not the gate.

Example::

    uv run python -m scripts.bench.episode --ns 512 1024 2048 --steps 120
    uv run python -m scripts.bench.episode --ns 512 1024 2048 --solvers adaptive
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import warp as wp

import newton
import newton.solvers

from scripts.scenes import contact_objects as co


def _build_adaptive(model, tol: float):
    """Adaptive solver, exactly per the optimization handoff construction."""
    return newton.solvers.SolverMuJoCoAdaptive(
        model, tol=tol, dt_init=co.DT_OUTER, dt_min=1e-6, dt_max=co.DT_OUTER,
        use_mujoco_contacts=True, nconmax=co._NCON, njmax=co._NJM,
    )


def _episode_adaptive(n: int, steps: int, warmup: int, seed: int, tol: float) -> dict:
    model = co.build_model_randomized(n, seed=seed)
    solver = _build_adaptive(model, tol)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    for _ in range(warmup):
        solver.step(s0, s1, ctrl, None, co.DT_OUTER)
    wp.synchronize()

    times = np.empty(steps, dtype=np.float64)
    ks = np.empty(steps, dtype=np.float64)
    max_err = 0.0
    for i in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        solver.step(s0, s1, ctrl, None, co.DT_OUTER)
        wp.synchronize()
        times[i] = time.perf_counter() - t0
        # Boundary reads (outside the inner physics loop -> allowed).
        max_err = max(max_err, float(solver.last_error.numpy().max()))
        ks[i] = float(solver.iteration_count.numpy()[0])

    nan = bool(np.isnan(s0.joint_q.numpy()).any())
    return {
        "median_ms": float(np.median(times)) * 1e3,
        "mean_ms": float(np.mean(times)) * 1e3,
        "p25_ms": float(np.percentile(times, 25)) * 1e3,
        "p75_ms": float(np.percentile(times, 75)) * 1e3,
        "k_median": float(np.median(ks)),
        "k_max": float(np.max(ks)),
        "max_err": max_err,
        "nan": nan,
    }


def _episode_fixed(n: int, steps: int, warmup: int, seed: int, dt_sub: float = 1e-3) -> dict:
    model = co.build_model_randomized(n, seed=seed)
    solver = newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=co._NCON, njmax=co._NJM,
    )
    contacts = model.contacts()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    n_sub = max(1, round(co.DT_OUTER / dt_sub))

    def outer(a, b):
        for _ in range(n_sub):
            solver.step(a, b, ctrl, contacts, dt_sub)
            a, b = b, a
        return a, b

    for _ in range(warmup):
        s0, s1 = outer(s0, s1)
    wp.synchronize()

    times = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        s0, s1 = outer(s0, s1)
        wp.synchronize()
        times[i] = time.perf_counter() - t0

    nan = bool(np.isnan(s0.joint_q.numpy()).any())
    return {
        "median_ms": float(np.median(times)) * 1e3,
        "mean_ms": float(np.mean(times)) * 1e3,
        "nan": nan,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--solvers", type=str, default="adaptive,fixed",
                   help="Comma list: adaptive,fixed")
    p.add_argument("--out", type=str, default=None, help="Optional JSON dump path")
    args = p.parse_args()

    wp.init()
    kinds = [s.strip() for s in args.solvers.split(",") if s.strip()]

    results: dict = {"steps": args.steps, "warmup": args.warmup, "seed": args.seed,
                     "tol": args.tol, "ns": args.ns, "data": {}}

    hdr = (f"{'N':>6}  {'adapt_med':>10}  {'adapt_mean':>10}  {'fixed_med':>10}  "
           f"{'ratio_med':>9}  {'K_med':>6}  {'K_max':>6}  {'max_err':>9}  {'NaN':>4}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for n in args.ns:
        entry: dict = {}
        a = _episode_adaptive(n, args.steps, args.warmup, args.seed, args.tol) if "adaptive" in kinds else None
        f = _episode_fixed(n, args.steps, args.warmup, args.seed) if "fixed" in kinds else None
        if a is not None:
            entry["adaptive"] = a
        if f is not None:
            entry["fixed"] = f
        results["data"][str(n)] = entry

        am = a["median_ms"] if a else float("nan")
        amean = a["mean_ms"] if a else float("nan")
        fm = f["median_ms"] if f else float("nan")
        ratio = (fm / am) if (a and f and am > 0) else float("nan")
        kmed = a["k_median"] if a else float("nan")
        kmax = a["k_max"] if a else float("nan")
        merr = a["max_err"] if a else float("nan")
        nan = (a["nan"] if a else False) or (f["nan"] if f else False)
        print(f"{n:>6}  {am:>10.2f}  {amean:>10.2f}  {fm:>10.2f}  "
              f"{ratio:>9.3f}  {kmed:>6.0f}  {kmax:>6.0f}  {merr:>9.2e}  {str(nan):>4}",
              flush=True)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nJSON -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
