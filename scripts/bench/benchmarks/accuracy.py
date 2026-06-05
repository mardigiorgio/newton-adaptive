# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Work-precision benchmark: wall time vs tolerance.

Each measurement runs in a subprocess for GPU state isolation.

Standalone:
    uv run python -m scripts.bench.benchmarks.accuracy
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_INNER_MIN, DT_OUTER


def _run_in_subprocess(code: str) -> str:
    """Run Python code in a fresh subprocess, return last stdout line."""
    env = {**__import__("os").environ, "WARP_LOG_LEVEL": "error"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    if result.returncode != 0:
        print(f"  SUBPROCESS STDERR:\n{result.stderr[-500:]}", flush=True)
        raise RuntimeError(f"Subprocess failed (exit {result.returncode})")
    lines = [l for l in result.stdout.strip().splitlines() if l.strip()]
    return lines[-1]


def _measure_cenic(n: int, tol: float, sim_duration: float, trials: int) -> float:
    """Measure CENIC wall time (ms/sim-s), median of `trials` fresh processes."""
    code = textwrap.dedent(f"""\
        import time, warp as wp
        from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized, make_solver
        model = build_model_randomized({n})
        solver = make_solver(model, tol={tol})
        s0, s1 = model.state(), model.state()
        ctrl = model.control()
        for _ in range(5):
            solver.step(s0, s1, ctrl, None, DT_OUTER)
        wp.synchronize()
        n_steps = round({sim_duration} / DT_OUTER)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            solver.step(s0, s1, ctrl, None, DT_OUTER)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        print(elapsed / {sim_duration} * 1e3)
    """)
    samples = [float(_run_in_subprocess(code)) for _ in range(trials)]
    return float(np.median(samples))


def run(
    tols: list[float] | None = None,
    n_worlds: int = 16,
    sim_duration: float = 2.0,
    trials: int = 3,
) -> dict:
    """Run the work-precision sweep.

    Args:
        tols: Tolerances to sweep.  Defaults to a log-spaced grid.
        n_worlds: World count for the wall-time sweep.
        sim_duration: Simulated seconds per measurement.
        trials: Number of independent subprocess measurements per point
            (median-reduced).
    """
    if tols is None:
        tols = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    data: dict = {
        "tols": tols,
        "n_worlds": n_worlds,
        "sim_duration": sim_duration,
        "dt_outer": DT_OUTER,
        "dt_min": DT_INNER_MIN,
    }

    print(
        f"\nWall time vs tol  N={n_worlds}  sim={sim_duration}s  trials={trials}",
        flush=True,
    )
    wall_ms: list[float] = []
    for tol in tols:
        ms = _measure_cenic(n_worlds, tol, sim_duration, trials)
        print(f"  tol={tol:.0e}  {ms:.1f} ms/sim-s", flush=True)
        wall_ms.append(ms)
    data["wall_vs_tol"] = wall_ms

    return data


def plot(data: dict, out_dir: Path) -> None:
    """Generate the work-precision plot."""
    tols = data["tols"]
    n_worlds = data.get("n_worlds", 1)
    sim_duration = data.get("sim_duration", 2.0)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_style = {"linewidth": 1.8, "marker": "o", "markersize": 4, "color": "tab:blue",
                  "label": "CENIC adaptive"}

    wall = data["wall_vs_tol"]
    # Back-compat: older runs stored a dict keyed by mode.
    if isinstance(wall, dict):
        wall = list(wall.values())[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tols, wall, **base_style)
    ax.set_xlabel("Error tolerance")
    ax.set_ylabel("Wall time per sim-second [ms/sim-s]")
    ax.set_title(f"Work-precision  (N={n_worlds}, {sim_duration:.0f} s sim with contact)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig, out_dir / "accuracy_wall_vs_tol.png")


def main():
    parser = argparse.ArgumentParser(description="Accuracy benchmark")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sim-duration", type=float, default=2.0)
    parser.add_argument("--n-worlds", type=int, default=16,
                        help="World count for the wall-time sweep.")
    parser.add_argument("--out-dir", type=str, default="scripts/bench/results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    data = run(
        trials=args.trials,
        sim_duration=args.sim_duration,
        n_worlds=args.n_worlds,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "accuracy.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nData saved -> {out_dir / 'accuracy.json'}", flush=True)

    plot(data, out_dir / "plots")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
