# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-step error trace: adaptive vs fixed 1 ms vs fixed 10 ms, on the
``contact_objects`` scene with randomized ICs.

All three solvers report the IDENTICAL step-doubling error estimate
:math:`e_{n+1} = \\lVert q-\\hat q\\rVert_\\infty` each step, so the
comparison is apples-to-apples. Each fixed solver runs at its own outer cadence
equal to its inner ``dt`` (one clean step per boundary, ``K=1``) to avoid the
no-op-substep error floor; the adaptive solver runs at a 10 ms outer cadence.

Per timestep the error is aggregated over the ``N`` worlds by ``--stat``
(``max`` = worst-world control guarantee, ``mean``/``median`` = population
central tendency). Worst-world is N-dependent; mean/median are stable
population statistics. Writes ``<stem>.png`` and ``<stem>.json`` under
``--out-dir`` (default ``scripts/bench/results/error_trace``).

Example::

    uv run -m scripts.bench.error_trace --n 1024 --stat median --t-end 1.6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton  # noqa: F401
from scripts.bench.benchmarks.tol_trace import _ADAPT_COLOR, _FIXED_COLOR, _series
from scripts.scenes import _solvers as _s
from scripts.scenes import contact_objects as co

_FIXED1_COLOR = "#d62728"  # matches the fixed-1ms curve in the scaling plot


def _fixed_factory(tol, dt):
    """Fixed-step solver clamped to ``dt`` inner and outer (K=1 per boundary)."""
    return _s.mujoco_adaptive_factory(
        tol=tol,
        nconmax=co._NCON,
        njmax=co._NJM,
        dt_outer=dt,
        dt_inner_min=dt,
        dt_inner_max=dt,
        use_mujoco_contacts=True,
    )


def _adaptive_factory(tol, dt_outer):
    return _s.mujoco_adaptive_factory(
        tol=tol,
        nconmax=co._NCON,
        njmax=co._NJM,
        dt_outer=dt_outer,
        dt_inner_min=1e-6,
        dt_inner_max=dt_outer,
        use_mujoco_contacts=True,
    )


def _agg(e, stat):
    if stat == "max":
        return np.max(e, axis=1)
    if stat == "mean":
        return np.mean(e, axis=1)
    if stat == "median":
        return np.median(e, axis=1)
    raise ValueError(stat)


def _smooth(y, k):
    if len(y) < k:
        return y
    return np.convolve(np.pad(y, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--t-end", type=float, default=1.6, help="trajectory length [s]")
    p.add_argument("--stat", choices=("max", "mean", "median"), default="median")
    p.add_argument("--out-dir", default="scripts/bench/results/error_trace")
    args = p.parse_args()

    wp.init()
    n, tol = args.n, args.tol

    print(f"=== adaptive (N={n}) ===", flush=True)
    e_ad = _series(_adaptive_factory(tol, 0.01), n, int(args.t_end / 0.01), n)
    print("=== fixed 10 ms ===", flush=True)
    e_f10 = _series(_fixed_factory(tol, 0.01), n, int(args.t_end / 0.01), n)
    print("=== fixed 1 ms ===", flush=True)
    e_f1 = _series(_fixed_factory(tol, 0.001), n, int(args.t_end / 0.001), n)

    t_out = np.arange(1, e_ad.shape[0] + 1) * 0.01
    t_f1 = np.arange(1, e_f1.shape[0] + 1) * 0.001
    ad, f10, f1 = _agg(e_ad, args.stat), _agg(e_f10, args.stat), _agg(e_f1, args.stat)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"error_trace_N{n}_{args.stat}_tol{tol:.0e}"
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(
            {
                "n": n,
                "tol": tol,
                "stat": args.stat,
                "t_out": t_out.tolist(),
                "t_f1": t_f1.tolist(),
                "adaptive": ad.tolist(),
                "fixed1": f1.tolist(),
                "fixed10": f10.tolist(),
            },
            f,
        )

    for nm, y in (("adaptive", ad), ("fixed1ms", f1), ("fixed10ms", f10)):
        print(
            f"{nm:9s} {args.stat} range [{y.min():.2e},{y.max():.2e}]  above-tol {(y > tol).mean() * 100:.0f}%",
            flush=True,
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12}
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.axhspan(1e-8, tol, color="#2e7d32", alpha=0.07, zorder=0)
    ax.axhspan(tol, 1e1, color="#c62828", alpha=0.07, zorder=0)
    eps = r"\varepsilon_{\mathrm{acc}}"
    ax.plot(t_out, _smooth(ad, 7), color=_ADAPT_COLOR, lw=2.8, label=rf"per-world adaptive (${eps}=10^{{-3}}$)")
    ax.plot(t_f1, _smooth(f1, 15), color=_FIXED1_COLOR, lw=2.2, label=r"global fixed  $\delta t = 1$ ms")
    ax.plot(t_out, _smooth(f10, 7), color=_FIXED_COLOR, lw=2.8, label=r"global fixed  $\delta t = 10$ ms")
    ax.axhline(tol, ls="--", color="k", lw=1.6, label=rf"tolerance ${eps}=10^{{-3}}$")
    statlabel = {"max": "Worst-world", "mean": "Mean", "median": "Median"}[args.stat]
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(rf"{statlabel} error  $e_{{n+1}}=\|q-\hat q\|_\infty$  ($N={n}$)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2.0)
    ax.set_xlim(0, args.t_end)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    print(f"\nwrote {out_dir / stem}.png and .json", flush=True)


if __name__ == "__main__":
    main()
