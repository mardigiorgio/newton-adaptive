# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Re-render a scaling_sweep.json in the poster's house style (matches
poster/SCALING.png): no title, framed legend inside, large fonts, blue/red/orange
with circle/square/diamond markers, log-log over Parallel worlds N. CPU-only
(reads the cached JSON; no GPU).

Example::

    uv run python -m scripts.bench.plot_scaling_poster \\
      scripts/bench/results/poster_2026-06-05/scaling_sweep.json \\
      --kinds mujoco_adaptive_1e-3 mujoco_adaptive_1e-2 mujoco_fixed_1ms mujoco_fixed_10ms
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, color, marker) per kind, matching poster/SCALING.png conventions.
POSTER_STYLE = {
    "mujoco_adaptive_1e-3": (r"per-world adaptive ($\varepsilon_{\mathrm{acc}}=10^{-3}$)", "#1f77b4", "o"),
    "mujoco_adaptive_1e-2": (r"per-world adaptive ($\varepsilon_{\mathrm{acc}}=10^{-2}$)", "#5fa8e0", "*"),
    "mujoco_fixed_1ms":     (r"global fixed  $\delta t = 1$ ms",  "#d62728", "s"),
    "mujoco_fixed_10ms":    (r"global fixed  $\delta t = 10$ ms", "#ff7f0e", "D"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json", help="path to scaling_sweep.json")
    p.add_argument("--kinds", type=str, nargs="+", default=list(POSTER_STYLE),
                   help="kinds to draw, in z-order (later = on top)")
    p.add_argument("--bands", action="store_true", help="draw faint p25-p75 bands (poster omits them)")
    p.add_argument("--out", type=str, default=None, help="output PNG (default: SCALING.png next to the json)")
    args = p.parse_args()

    data = json.loads(Path(args.json).read_text())
    modes = data["modes"]

    plt.rcParams.update({
        "font.size": 17, "axes.labelsize": 20,
        "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
    })
    fig, ax = plt.subplots(figsize=(8.2, 5.7))

    for kind in args.kinds:
        m = modes.get(kind)
        if not m or not m.get("medians_s"):
            continue
        label, color, marker = POSTER_STYLE[kind]
        ns = m["ns"]
        ms = [v * 1e3 for v in m["medians_s"]]
        ax.plot(ns, ms, color=color, marker=marker, lw=2.6, ms=8, label=label)
        if args.bands and m.get("p25_s"):
            ax.fill_between(ns, [v * 1e3 for v in m["p25_s"]], [v * 1e3 for v in m["p75_s"]],
                            color=color, alpha=0.10)

    ax.set_xlabel("Parallel worlds  N")
    ax.set_ylabel("Wall-time per outer step [ms]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="0.6")
    fig.tight_layout()

    out = Path(args.out) if args.out else Path(args.json).with_name("SCALING.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"poster-style scaling -> {out}")


if __name__ == "__main__":
    main()
