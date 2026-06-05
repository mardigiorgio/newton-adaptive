# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared plotting utilities for the benchmark platform.

Enforces log-log axes (per CLAUDE.md convention), consistent styling,
IQR bands, and power-law exponent annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scripts.bench.infra import power_law_exponent


@dataclass
class PlotStyle:
    color: str
    marker: str
    linestyle: str
    label: str


# Consistent style registry for stepping modes.
# Every label is tagged "(adaptive)" or "(fixed)" so the legend reads cleanly
# even on busy multi-curve plots.
STYLES: dict[str, PlotStyle] = {
    # Adaptive (current keys) — stars, MuJoCo blue, others distinct.
    "mujoco_adaptive_1e-3": PlotStyle("#0066cc", "*", "-", "MuJoCo tol=1e-3 (adaptive)"),
    "mujoco_adaptive_1e-2": PlotStyle("#3399ff", "*", "-", "MuJoCo tol=1e-2 (adaptive)"),
    "xpbd_adaptive_1e-3":   PlotStyle("#cc6600", "*", "-", "XPBD tol=1e-3 (adaptive)"),
    "semi_adaptive_1e-3":   PlotStyle("#660066", "*", "-", "Semi-implicit tol=1e-3 (adaptive)"),
    # Adaptive (legacy keys — old JSONs).
    "mujoco_cenic_1e-3": PlotStyle("#1f77b4", "o", "-", "MuJoCo tol=1e-3 (adaptive, legacy)"),
    "mujoco_cenic_1e-2": PlotStyle("#17becf", "o", "-", "MuJoCo tol=1e-2 (adaptive, legacy)"),
    "cenic":             PlotStyle("#1f77b4", "o", "-", "MuJoCo (adaptive, legacy)"),
    # Fixed-step (current keys) — diamonds/squares, color by family.
    "mujoco_fixed_10ms":  PlotStyle("#ff7f0e", "D", "-", "MuJoCo dt=10 ms (fixed)"),
    "mujoco_fixed_1ms":   PlotStyle("#d62728", "s", "-", "MuJoCo dt=1 ms (fixed)"),
    "featherstone_1ms":   PlotStyle("#2ca02c", "^", "-", "Featherstone dt=1 ms (fixed)"),
    "semi_implicit_1ms":  PlotStyle("#9467bd", "v", "-", "Semi-implicit dt=1 ms (fixed)"),
    "xpbd_1ms":           PlotStyle("#8c564b", "P", "-", "XPBD dt=1 ms (fixed)"),
    "vbd_1ms":            PlotStyle("#e377c2", "X", "-", "VBD dt=1 ms (fixed)"),
    # Fixed-step (legacy keys).
    "fixed_10ms": PlotStyle("#ff7f0e", "D", "-", "dt=10 ms (fixed, legacy)"),
    "fixed_1ms":  PlotStyle("#d62728", "s", "-", "dt=1 ms (fixed, legacy)"),
    "fixed":      PlotStyle("#ff7f0e", "D", "-", "dt=10 ms (fixed, legacy)"),
    # Other legacy keys kept so older result JSONs still plot.
    "single_iter": PlotStyle("#d62728", "s", "--", "Single iteration (legacy)"),
    "identical":   PlotStyle("#2ca02c", "^", "-", "Identical ICs (legacy)"),
    "perturbed":   PlotStyle("#9467bd", "v", "-", "Perturbed ICs (legacy)"),
    "randomized":  PlotStyle("#d62728", "v", "-", "Randomized ICs (legacy)"),
}


@dataclass
class SeriesData:
    medians: list[float]
    p25: list[float] | None = None
    p75: list[float] | None = None


def log_log_plot(
    ax: Axes,
    ns: list[int],
    series: dict[str, SeriesData],
    ylabel: str,
    title: str,
    show_exponents: bool = True,
    show_iqr: bool = True,
) -> dict[str, float]:
    """Standard log-log scaling plot. Returns {mode: exponent}."""
    exponents = {}
    for mode, sd in series.items():
        style = STYLES.get(mode)
        if style is None:
            continue
        exp = power_law_exponent(ns, sd.medians)
        exponents[mode] = exp
        label = f'{style.label}  $N^{{{exp:.2f}}}$' if show_exponents else style.label
        ax.plot(
            ns, sd.medians,
            color=style.color, marker=style.marker, ls=style.linestyle,
            lw=2, ms=5, label=label,
        )
        if show_iqr and sd.p25 is not None and sd.p75 is not None:
            ax.fill_between(ns, sd.p25, sd.p75, color=style.color, alpha=0.10)

    ax.set_xlabel("N worlds", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Legend OUTSIDE plot area (top-right of axes) so it never covers data.
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0.0, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    return exponents


def save_fig(fig: Figure, path: str | Path, dpi: int = 150) -> None:
    """tight_layout + savefig + close. bbox_inches='tight' captures the
    outside-axes legend without clipping."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {path}", flush=True)
