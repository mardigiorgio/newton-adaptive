# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Overlay two scaling JSONs on the same axes to track perf changes across versions.

Usage:
    uv run python -m scripts.bench.plot_comparison \\
        --baseline /tmp/v1_bench_baseline.json \\
        --current scripts/bench/results/2cb5a1d/scaling_falling_cylinder.json \\
        --out scripts/bench/results/2cb5a1d/plots/v1_vs_v2.png

Produces two plots side-by-side:
  - Left: log-log scaling, baseline (dashed) + current (solid), one color per kind
  - Right: percent change per kind, per N (v2/v1 - 1)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bench.plotting import STYLES


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _shared_ns(a: dict, b: dict) -> list[int]:
    """Ns present in both JSONs."""
    return sorted(set(a["ns"]) & set(b["ns"]))


def _series_at_ns(data: dict, kind: str, ns: list[int]) -> list[float] | None:
    """Return medians (ms) for kind at the given ns, or None if kind missing."""
    if kind not in data["modes"]:
        return None
    src_ns = data["ns"]
    medians = data["modes"][kind]["medians"]
    return [medians[src_ns.index(n)] * 1e3 for n in ns]


def main():
    p = argparse.ArgumentParser(description="v1 vs v2 bench comparison plot")
    p.add_argument("--baseline", type=Path, required=True, help="Older bench JSON")
    p.add_argument("--current", type=Path, required=True, help="Newer bench JSON")
    p.add_argument("--out", type=Path, required=True, help="Output PNG path")
    p.add_argument("--baseline-label", default="v1", help="Legend label for baseline")
    p.add_argument("--current-label", default="v2", help="Legend label for current")
    args = p.parse_args()

    base = _load(args.baseline)
    curr = _load(args.current)
    ns = _shared_ns(base, curr)
    if not ns:
        raise SystemExit("No shared N values between baseline and current JSONs.")

    # Kinds present in both runs (preserve current order)
    kinds = [k for k in curr["modes"] if k in base["modes"]]

    fig, (ax_abs, ax_pct) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: absolute scaling, baseline dashed + current solid -----------
    for kind in kinds:
        style = STYLES.get(kind)
        if style is None:
            continue
        b_med = _series_at_ns(base, kind, ns)
        c_med = _series_at_ns(curr, kind, ns)
        if b_med is None or c_med is None:
            continue
        ax_abs.plot(ns, b_med, color=style.color, marker=style.marker,
                    ls="--", lw=1.5, ms=4, alpha=0.6,
                    label=f"{style.label} ({args.baseline_label})")
        ax_abs.plot(ns, c_med, color=style.color, marker=style.marker,
                    ls="-", lw=2, ms=5,
                    label=f"{style.label} ({args.current_label})")

    ax_abs.set_xlabel("N worlds")
    ax_abs.set_ylabel("Wall time per outer step [ms]")
    ax_abs.set_xscale("log", base=2)
    ax_abs.set_yscale("log")
    ax_abs.set_title(f"Scaling: {args.baseline_label} (dashed) vs {args.current_label} (solid)")
    ax_abs.grid(True, which="both", alpha=0.3)
    ax_abs.legend(fontsize=7, loc="upper left", ncol=2)

    # --- Right: percent change per kind per N ------------------------------
    for kind in kinds:
        style = STYLES.get(kind)
        if style is None:
            continue
        b_med = _series_at_ns(base, kind, ns)
        c_med = _series_at_ns(curr, kind, ns)
        if b_med is None or c_med is None:
            continue
        pct = [(c / b - 1.0) * 100.0 for b, c in zip(b_med, c_med)]
        ax_pct.plot(ns, pct, color=style.color, marker=style.marker,
                    ls="-", lw=2, ms=5, label=style.label)

    ax_pct.axhline(0, color="black", lw=1, alpha=0.5)
    ax_pct.set_xlabel("N worlds")
    ax_pct.set_ylabel(f"% change ({args.current_label} / {args.baseline_label} - 1)")
    ax_pct.set_xscale("log", base=2)
    ax_pct.set_title(f"Per-kind delta: {args.current_label} vs {args.baseline_label}")
    ax_pct.grid(True, which="both", alpha=0.3)
    ax_pct.legend(fontsize=8, loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"saved -> {args.out}", flush=True)

    # Console summary table
    print(f"\n{'kind':<28} " + " ".join(f"N={n}" for n in ns))
    for kind in kinds:
        b_med = _series_at_ns(base, kind, ns)
        c_med = _series_at_ns(curr, kind, ns)
        if b_med is None or c_med is None:
            continue
        pcts = [(c / b - 1.0) * 100.0 for b, c in zip(b_med, c_med)]
        row = f"{kind:<28} " + " ".join(f"{p:+5.1f}%" for p in pcts)
        print(row)


if __name__ == "__main__":
    main()
