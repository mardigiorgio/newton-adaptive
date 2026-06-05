# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""One-off bench: scaling at multiple CENIC tols vs fixed-step baselines.

Diagnostic for why CENIC underperforms fixed_1ms at strict tol on dense-contact
scenes: K_global grows with N (max-world drag), so CENIC pays K*3 mujoco_warp
steps per outer step. Looser tol => smaller K => competitive.

Run:
    uv run python -m scripts.bench.tol_sweep --ns 1 4 16 64 256 1024 --steps 80 --warmup 20

Output: scripts/bench/results/<commit>/plots/scaling_tol_sweep.png
"""
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton, newton.solvers
from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized


def _git_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def measure_cenic(n, tol, steps, warmup):
    model = build_model_randomized(n, seed=42)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model, tol=tol, dt_init=DT_OUTER, dt_min=1e-6,
        dt_max=DT_OUTER, nconmax=128, njmax=640,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(warmup):
        solver.step(s0, s1, ctrl, None, DT_OUTER)
    wp.synchronize()
    times, Ks = [], []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        solver.step(s0, s1, ctrl, None, DT_OUTER)
        wp.synchronize()
        times.append(time.perf_counter() - t0)
        Ks.append(int(solver.iteration_count.numpy()[0]))
    return float(np.median(times)), float(np.mean(Ks))


def measure_fixed(n, fixed_dt, steps, warmup):
    model = build_model_randomized(n, seed=42)
    solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True, nconmax=128, njmax=640)
    contacts = model.contacts()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    n_sub = max(1, round(DT_OUTER / fixed_dt))
    for _ in range(warmup):
        for _ in range(n_sub):
            s1 = solver.step(s0, s1, ctrl, contacts, fixed_dt)
            s0, s1 = s1, s0
    wp.synchronize()
    times = []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_sub):
            s1 = solver.step(s0, s1, ctrl, contacts, fixed_dt)
            s0, s1 = s1, s0
        wp.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), n_sub


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--tols", type=float, nargs="+", default=[1e-3, 1e-2, 5e-2])
    args = p.parse_args()
    ns = sorted(args.ns)

    out = {"ns": ns, "tols": args.tols, "cenic": {}, "fixed_1ms": [], "fixed_10ms": []}

    for tol in args.tols:
        out["cenic"][f"tol={tol:.0e}"] = {"wall": [], "K": []}

    for n in ns:
        print(f"\n=== N={n} ===", flush=True)
        for tol in args.tols:
            wall, K = measure_cenic(n, tol, args.steps, args.warmup)
            out["cenic"][f"tol={tol:.0e}"]["wall"].append(wall * 1e3)
            out["cenic"][f"tol={tol:.0e}"]["K"].append(K)
            print(f"  cenic tol={tol:.0e}: {wall*1e3:7.2f} ms  K={K:5.2f}", flush=True)
        w1, _ = measure_fixed(n, 1e-3, args.steps, args.warmup)
        out["fixed_1ms"].append(w1 * 1e3)
        print(f"  fixed dt=1ms:   {w1*1e3:7.2f} ms", flush=True)
        w10, _ = measure_fixed(n, 1e-2, args.steps, args.warmup)
        out["fixed_10ms"].append(w10 * 1e3)
        print(f"  fixed dt=10ms:  {w10*1e3:7.2f} ms", flush=True)

    commit = _git_short()
    out_dir = Path(f"scripts/bench/results/{commit}/plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(f"scripts/bench/results/{commit}/scaling_tol_sweep.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\ndata -> {json_path}", flush=True)

    cenic_colors = ["#1f77b4", "#2ca02c", "#17becf"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, tol in enumerate(args.tols):
        key = f"tol={tol:.0e}"
        ax.plot(ns, out["cenic"][key]["wall"], "o-", color=cenic_colors[i],
                lw=2, ms=6, label=f"CENIC adaptive  tol={tol:.0e}")
    ax.plot(ns, out["fixed_1ms"], "s-", color="#d62728", lw=2, ms=6, label="Fixed dt=1 ms (best case)")
    ax.plot(ns, out["fixed_10ms"], "D-", color="#ff7f0e", lw=2, ms=6, label="Fixed dt=10 ms (worst case)")
    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per outer step [ms]", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Scaling: CENIC at multiple tolerances vs fixed-step baselines  (contact_objects)", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    out_path = out_dir / "scaling_tol_sweep.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"plot -> {out_path}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, tol in enumerate(args.tols):
        key = f"tol={tol:.0e}"
        ax.plot(ns, out["cenic"][key]["K"], "o-", color=cenic_colors[i],
                lw=2, ms=6, label=f"K_mean  tol={tol:.0e}")
    ax.axhline(10, color="#d62728", ls="--", lw=1.5, label="fixed_1ms substeps (10)")
    ax.axhline(1, color="#ff7f0e", ls="--", lw=1.5, label="fixed_10ms substeps (1)")
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Mean iterations K per outer step")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_title("Why CENIC scales differently: K grows with N (max-world drag in batch)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_k = out_dir / "scaling_tol_K.png"
    fig.savefig(out_k, dpi=150)
    plt.close(fig)
    print(f"plot -> {out_k}", flush=True)


if __name__ == "__main__":
    main()
