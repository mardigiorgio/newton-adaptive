# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Focused scaling plot: subset of curves grouped for readability.

Three plot variants, pick one via --view:
  - "headline": 4 curves (CENIC, MuJoCo fixed dt=1ms / dt=10ms, XPBD fixed)
  - "adaptive": 5 adaptive curves only (CENIC variants + adaptive XPBD/Semi)
  - "fixed":    5 fixed curves only (MuJoCo 1ms/10ms, Semi, XPBD, VBD)

Usage:
    uv run python -m scripts.bench.plot_focused \\
        --json scripts/bench/results/2cb5a1d/scaling_falling_cylinder.json \\
        --view headline \\
        --out scripts/bench/results/2cb5a1d/plots/scaling_headline.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.plotting import STYLES

VIEWS = {
    "headline": [
        # The adaptive_1e-2 variant is the right pitch — beats fixed_1ms while
        # maintaining a per-world tolerance guarantee that fixed can't.
        # adaptive_1e-3 is in the "adaptive" view for the strict-tol story.
        "mujoco_adaptive_1e-2",
        "mujoco_fixed_1ms",
        "mujoco_fixed_10ms",
        "xpbd_1ms",
    ],
    "adaptive": [
        "mujoco_adaptive_1e-3",
        "mujoco_adaptive_1e-2",
        "xpbd_adaptive_1e-3",
        "semi_adaptive_1e-3",
    ],
    "fixed": [
        "mujoco_fixed_1ms",
        "mujoco_fixed_10ms",
        "semi_implicit_1ms",
        "xpbd_1ms",
        "vbd_1ms",
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--view", choices=list(VIEWS), default="headline")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Defaults to <json-dir>/plots/<scene>/scaling_<view>.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    with open(args.json) as f:
        data = json.load(f)

    ns = data["ns"]
    kinds = VIEWS[args.view]
    scene = data.get("scene", "unknown_scene")

    if args.out is None:
        args.out = args.json.parent / "plots" / scene / f"scaling_{args.view}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    for kind in kinds:
        if kind not in data["modes"]:
            continue
        style = STYLES.get(kind)
        if style is None:
            continue
        medians_ms = [m * 1e3 for m in data["modes"][kind]["medians"]]
        p25 = [m * 1e3 for m in data["modes"][kind].get("p25", medians_ms)]
        p75 = [m * 1e3 for m in data["modes"][kind].get("p75", medians_ms)]
        exp = data.get("exponents", {}).get(kind)
        label = f"{style.label}  $N^{{{exp:.2f}}}$" if exp is not None else style.label
        ax.plot(ns, medians_ms, color=style.color, marker=style.marker,
                ls=style.linestyle, lw=2.5, ms=6, label=label)
        ax.fill_between(ns, p25, p75, color=style.color, alpha=0.12)

    ax.set_xlabel("N worlds", fontsize=12)
    ax.set_ylabel("Wall time per outer step [ms]", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    title = args.title or f"Scaling [{args.view}]: {scene}"
    ax.set_title(title, fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    # Legend OUTSIDE plot (top-right of axes) so it doesn't cover data.
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches="tight" includes the outside-axes legend in the saved area.
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
