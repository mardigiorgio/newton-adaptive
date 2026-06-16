"""Phase 0b figure: transfer-gap contribution by channel, with bootstrap CIs.

A horizontal bar per reference (gap vs id), colored by channel
(integrator / physical / sensing). The visual claim: the integrator sits at the
bottom of the budget, so adaptive stepping has no sim-to-real gap to close here.

    uv run --extra rl -m scripts.rl.plot_error_budget --glob 'results/p0b_cenic_s1_seed*.npz'
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.bench.plotting import save_fig

from .analysis import _channel, bootstrap_ci, iqm, load_pooled

_COLOR = {"integrator": "tab:red", "physical": "tab:blue", "sensing": "tab:gray"}


def main():
    p = argparse.ArgumentParser(description="Phase 0b error-budget bar chart.")
    p.add_argument("--glob", required=True)
    p.add_argument("--metric", default="lvte")
    p.add_argument("--id-backend", default="id")
    p.add_argument("--out", default="results/plots/p0b_error_budget.png")
    args = p.parse_args()

    import glob as _glob  # noqa: PLC0415

    pooled = load_pooled(sorted(_glob.glob(args.glob)))
    m_id = pooled[args.id_backend][args.metric]
    rows = []
    for backend, md in pooled.items():
        if backend == args.id_backend or args.metric not in md:
            continue
        n = min(len(m_id), len(md[args.metric]))
        gap = md[args.metric][:n] - m_id[:n]
        lo, hi = bootstrap_ci(gap)
        rows.append((iqm(gap), lo, hi, backend))
    rows.sort()

    labels = [r[3] for r in rows]
    gaps = np.array([r[0] for r in rows])
    los = np.array([r[1] for r in rows])
    his = np.array([r[2] for r in rows])
    colors = [_COLOR[_channel(b)] for b in labels]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y, gaps, color=colors, alpha=0.85,
            xerr=[gaps - los, his - gaps], error_kw={"ecolor": "k", "elinewidth": 1, "capsize": 3})
    ax.axvline(0.0, color="k", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"transfer gap vs id  ({args.metric})  [m/s]  -- positive = degradation")
    ax.set_title("Error budget: contribution to the policy's transfer gap  (ANYmal, closed-loop metric)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _COLOR.values()]
    ax.legend(handles, _COLOR.keys(), fontsize=8, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    n_seeds = len(sorted(_glob.glob(args.glob)))
    ax.annotate(f"{n_seeds} paired seeds x 64 worlds", xy=(0.02, 0.02),
                xycoords="axes fraction", fontsize=7, color="0.4")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_fig(fig, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
