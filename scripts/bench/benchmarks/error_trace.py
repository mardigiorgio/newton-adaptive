# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-step error / dt / K trace over simulation time for adaptive solvers.

Diagnostic, not a benchmark. Builds a model + adaptive solver, runs N outer
steps, and records per-step state (error, dt, K, sim_time). Plots three
stacked panels: error vs sim_time, dt vs sim_time, K vs sim_time.

Only works on adaptive kinds (solvers that expose ``last_error``-equivalent).
Fixed solvers are skipped with a message.

Examples::

    # Compare XPBD and Semi adaptive on falling_cylinder at N=4
    uv run python -m scripts.bench.benchmarks.error_trace \\
        --scene falling_cylinder \\
        --kinds xpbd_adaptive_1e-3 semi_adaptive_1e-3 mujoco_adaptive_1e-3 \\
        --n 4 --steps 60

    # Single solver at high N
    uv run python -m scripts.bench.benchmarks.error_trace \\
        --scene contact_objects --kinds mujoco_adaptive_1e-2 --n 256 --steps 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from scripts.bench.plotting import STYLES, save_fig
from scripts.scenes import _registry as _scene_registry


def _get_attr(solver, names):
    """Return the first attribute from `names` that exists on `solver`."""
    for n in names:
        if hasattr(solver, n):
            return getattr(solver, n)
    return None


def trace_kind(scene_entry, kind: str, n: int, steps: int) -> dict | None:
    """Run `steps` outer steps; record per-step stats. Returns None if the
    kind isn't adaptive (no last_error attribute)."""
    builder = scene_entry.solver_factories[kind]
    model = scene_entry.build_model_randomized(n)
    solver, step_fn = builder(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # SolverMuJoCoAdaptive exposes public properties; raw wrappers (XPBD/Semi)
    # only have the private buffer names. Try both.
    sim_time_arr = _get_attr(solver, ["sim_time", "_sim_time"])
    error_arr = _get_attr(solver, ["last_error", "_last_error"])
    dt_arr = _get_attr(solver, ["dt", "_dt"])
    k_arr = _get_attr(solver, ["iteration_count", "_iteration_count_buf"])

    if error_arr is None or dt_arr is None:
        print(f"  {kind}: no error/dt buffers — not adaptive, skipping")
        return None

    trace = {
        "step": [], "sim_time": [],
        "err_max": [], "err_p50": [], "err_min": [],
        "dt_max_ms": [], "dt_p50_ms": [], "dt_min_ms": [],
        "K": [],
    }
    for i in range(steps):
        try:
            s0, s1 = step_fn(model, s0, s1, ctrl)
        except Exception as e:
            print(f"  {kind}: step {i} raised {type(e).__name__}: {e}")
            break
        wp.synchronize()
        st = sim_time_arr.numpy()
        err = error_arr.numpy()
        dt_ms = dt_arr.numpy() * 1e3
        K = int(k_arr.numpy()[0]) if k_arr is not None else 1
        trace["step"].append(i)
        trace["sim_time"].append(float(st.mean()))
        trace["err_max"].append(float(err.max()))
        trace["err_p50"].append(float(np.median(err)))
        trace["err_min"].append(float(err.min()))
        trace["dt_max_ms"].append(float(dt_ms.max()))
        trace["dt_p50_ms"].append(float(np.median(dt_ms)))
        trace["dt_min_ms"].append(float(dt_ms.min()))
        trace["K"].append(K)
        if i % 10 == 0 or i < 5:
            print(
                f"    step={i:3d}  sim_t={trace['sim_time'][-1]:.4f}s  "
                f"err_max={trace['err_max'][-1]:.2e}  "
                f"dt_p50={trace['dt_p50_ms'][-1]:.3f}ms  K={K}",
                flush=True,
            )
    return trace


def plot_traces(traces: dict[str, dict | None], scene: str, n: int, out_path: Path) -> None:
    """Three stacked panels: error / dt / K over simulation time."""
    fig, (ax_err, ax_dt, ax_k) = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for kind, t in traces.items():
        if t is None or not t["step"]:
            continue
        style = STYLES.get(kind)
        color = style.color if style else None
        label = style.label if style else kind

        x = t["sim_time"]
        ax_err.plot(x, t["err_max"], color=color, lw=1.6, label=label)
        ax_err.fill_between(x, t["err_min"], t["err_max"], color=color, alpha=0.15)

        ax_dt.plot(x, t["dt_p50_ms"], color=color, lw=1.6, label=label)
        ax_dt.fill_between(x, t["dt_min_ms"], t["dt_max_ms"], color=color, alpha=0.15)

        ax_k.plot(x, t["K"], color=color, lw=1.6, label=label)

    ax_err.set_ylabel("Per-step error\n(max over worlds)")
    ax_err.set_yscale("log")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax_err.set_title(f"Error / dt / K traces  (scene={scene}, N={n})", fontsize=11)

    ax_dt.set_ylabel("Inner dt [ms]\n(median + min/max band)")
    ax_dt.set_yscale("log")
    ax_dt.grid(True, which="both", alpha=0.3)

    ax_k.set_xlabel("Simulation time [s]")
    ax_k.set_ylabel("Boundary iters K\n(substeps per outer)")
    ax_k.set_yscale("log")
    ax_k.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    save_fig(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="contact_objects",
                   help="Scene name from scripts.scenes._registry")
    p.add_argument("--kinds", nargs="+", required=True,
                   help="Solver kinds from the scene's SOLVER_FACTORIES")
    p.add_argument("--n", type=int, default=4, help="World count")
    p.add_argument("--steps", type=int, default=100, help="Outer steps to record")
    p.add_argument("--out-dir", default="scripts/bench/results/error_trace",
                   help="Output dir for plot + JSON")
    args = p.parse_args()

    scene = _scene_registry.get(args.scene)
    available = scene.solver_kinds()
    bad = [k for k in args.kinds if k not in available]
    if bad:
        print(f"Unknown kinds {bad}.\nAvailable for scene '{args.scene}': {available}")
        return 1

    traces: dict[str, dict | None] = {}
    for k in args.kinds:
        print(f"=== {k} ===", flush=True)
        try:
            traces[k] = trace_kind(scene, k, args.n, args.steps)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            traces[k] = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scene}_N{args.n}_steps{args.steps}"
    json_path = out_dir / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump({"scene": args.scene, "n": args.n, "steps": args.steps,
                   "traces": traces}, f, indent=2)
    plot_path = out_dir / f"{stem}.png"
    plot_traces(traces, args.scene, args.n, plot_path)
    print(f"\nData: {json_path}\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
