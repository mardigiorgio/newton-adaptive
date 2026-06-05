# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""N-scaling benchmark: wall time vs world count, one curve per solver kind.

Each scene declares its supported solvers via ``SOLVER_FACTORIES`` in the
scene module. The bench iterates over that dict and produces one curve per
kind. Per-N subprocess isolation is handled by ``scripts/bench/runner.py``.

Standalone:
    uv run python -m scripts.bench.benchmarks.scaling --ns 1 4 16 64 256
    uv run python -m scripts.bench.benchmarks.scaling --scene falling_cylinder --ns 1 4 16

Produces 1 plot: wall_time vs N, all solver kinds overlaid.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.bench.infra import MeasureResult, _suppress_kernel_noise, power_law_exponent
from scripts.bench.plotting import SeriesData, log_log_plot, save_fig
from scripts.scenes import _registry as _scene_registry


def _measure_kind(scene_entry, kind: str, n: int, steps: int, warmup: int) -> MeasureResult:
    """Build a fresh model + solver for `kind`, time `steps` outer steps."""
    builder = scene_entry.solver_factories[kind]
    model = scene_entry.build_model_randomized(n)
    solver, step_fn = builder(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # K reader: only CENIC has a meaningful iteration_count.
    get_k = None
    if hasattr(solver, "iteration_count"):
        def get_k():
            return int(solver.iteration_count.numpy()[0])

    times = np.empty(steps, dtype=np.float64)
    ks = np.ones(steps, dtype=np.int32)

    with _suppress_kernel_noise():
        for _ in range(warmup):
            s0, s1 = step_fn(model, s0, s1, ctrl)
        wp.synchronize()

        for i in range(steps):
            if get_k is not None and i > 0:
                ks[i - 1] = get_k()
            wp.synchronize()
            t0 = time.perf_counter()
            s0, s1 = step_fn(model, s0, s1, ctrl)
            wp.synchronize()
            times[i] = time.perf_counter() - t0
        if get_k is not None:
            ks[steps - 1] = get_k()

    per_iter = times / np.maximum(ks, 1)
    return MeasureResult(
        times=times, ks=ks,
        median=float(np.median(times)),
        p25=float(np.percentile(times, 25)),
        p75=float(np.percentile(times, 75)),
        k_mean=float(np.mean(ks)), k_max=int(np.max(ks)),
        k_p25=float(np.percentile(ks, 25)),
        k_p75=float(np.percentile(ks, 75)),
        per_iter_median=float(np.median(per_iter)),
    )


def run(scene_entry, ns: list[int], steps: int, warmup: int) -> dict:
    """Run every kind in the scene's solver_factories at every N. Failures
    for one kind don't kill the others — they're recorded as missing data."""
    kinds = scene_entry.solver_kinds()
    data: dict = {
        "ns": ns, "steps": steps, "warmup": warmup,
        "scene": scene_entry.name, "kinds": kinds,
        "modes": {kind: {"medians": [], "p25": [], "p75": [],
                         "k_means": [], "k_maxes": []} for kind in kinds},
        "failures": {kind: [] for kind in kinds},
    }
    for kind in kinds:
        for n in ns:
            t_start = time.perf_counter()
            print(f"  N={n:>5}  {kind:>22}  starting...", flush=True)
            try:
                r = _measure_kind(scene_entry, kind, n, steps, warmup)
            except Exception as e:
                dt = time.perf_counter() - t_start
                print(f"  N={n:>5}  {kind:>22}  FAILED after {dt:.1f}s: {type(e).__name__}: {e}", flush=True)
                data["failures"][kind].append({"n": n, "error": f"{type(e).__name__}: {e}"})
                continue
            data["modes"][kind]["medians"].append(r.median)
            data["modes"][kind]["p25"].append(r.p25)
            data["modes"][kind]["p75"].append(r.p75)
            data["modes"][kind]["k_means"].append(r.k_mean)
            data["modes"][kind]["k_maxes"].append(r.k_max)
            wall = time.perf_counter() - t_start
            print(
                f"  N={n:>5}  {kind:>22}  median={r.median*1e3:7.2f} ms  "
                f"K_mean={r.k_mean:5.2f}  K_max={r.k_max}  (took {wall:.1f}s)",
                flush=True,
            )

    data["exponents"] = {
        kind: power_law_exponent(ns, data["modes"][kind]["medians"])
        for kind in kinds if data["modes"][kind]["medians"]
    }
    return data


def plot(data: dict, out_dir: Path) -> None:
    """One line per solver kind. Each kind plotted against its own surviving Ns."""
    ns_global = data["ns"]
    modes_data = data["modes"]
    # When merged by runner.py with per-N isolation, kinds_ns has per-kind Ns.
    # When run directly via main(), all kinds share data["ns"].
    kinds_ns = data.get("kinds_ns", {k: ns_global for k in modes_data})
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.bench.plotting import STYLES
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_any = False
    for kind, md in modes_data.items():
        if not md["medians"]:
            continue
        ns_k = kinds_ns.get(kind, ns_global)
        style = STYLES.get(kind)
        if style is None:
            continue
        medians_ms = [m * 1e3 for m in md["medians"]]
        p25_ms = [m * 1e3 for m in md["p25"]]
        p75_ms = [m * 1e3 for m in md["p75"]]
        exp = data.get("exponents", {}).get(kind)
        label = f"{style.label}  $N^{{{exp:.2f}}}$" if exp is not None else style.label
        ax.plot(ns_k, medians_ms, color=style.color, marker=style.marker,
                ls=style.linestyle, lw=2, ms=5, label=label)
        ax.fill_between(ns_k, p25_ms, p75_ms, color=style.color, alpha=0.10)
        plotted_any = True

    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel("Wall time per outer step [ms]", fontsize=11)
    ax.set_title(f"Scaling: wall time vs N  (scene={data.get('scene','?')})", fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    if plotted_any:
        # Legend OUTSIDE plot area so it never covers data.
        ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  borderaxespad=0.0, frameon=True)
    save_fig(fig, out_dir / "scaling_wall_time.png")

    print(f"\n{'=' * 78}\nSCALING SUMMARY  scene={data.get('scene','?')}\n{'=' * 78}")
    hdr = (f"{'kind':>22}  {'exponent':>10}  {'N_min':>8}  {'N_max':>8}  "
           f"{'t@min':>10}  {'t@max':>10}  {'ratio':>6}")
    print(hdr)
    print("-" * len(hdr))
    for kind, md in modes_data.items():
        if not md["medians"]:
            print(f"{kind:>22}  (no data)")
            continue
        ns_k = kinds_ns.get(kind, ns_global)
        exp = data.get("exponents", {}).get(kind, float("nan"))
        t1 = md["medians"][0] * 1e3
        tN = md["medians"][-1] * 1e3
        ratio = tN / t1 if t1 > 0 else float("nan")
        print(f"{kind:>22}  N^{exp:<7.3f}   {ns_k[0]:>8}  {ns_k[-1]:>8}  "
              f"{t1:9.2f}  {tN:10.2f}  {ratio:5.1f}x")


def main():
    parser = argparse.ArgumentParser(description="N-scaling benchmark")
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 64, 256])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    parser.add_argument("--scene", type=str, default="contact_objects",
                        help="Scene to benchmark (see scripts.scenes._registry.bench_scenes()).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_entry = _scene_registry.get(args.scene)
    print(f"=== Scaling benchmark: scene={args.scene} kinds={scene_entry.solver_kinds()} ===",
          flush=True)

    data = run(scene_entry, sorted(args.ns), args.steps, args.warmup)

    suffix = "" if args.scene == "contact_objects" else f"_{args.scene}"
    json_path = out_dir / f"scaling{suffix}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {json_path}", flush=True)

    # Always put plots in per-scene subdir so multi-scene bench output stays organized.
    plot_dir = out_dir / "plots" / args.scene
    plot(data, plot_dir)
    print(json.dumps(data))


if __name__ == "__main__":
    main()
