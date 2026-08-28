# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Part-1 figures from the committed CSVs, in the CENIC paper's conventions:
every point states its accuracy eps_acc (adaptive) or time step dt (fixed).
CPU only; re-run after any sweep.

    uv run python scripts/bench/part1_plots.py
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

STYLE = {
    "mujoco": dict(color="#c0392b", marker="s", ls="--", label="MuJoCo, fixed step"),
    "mujoco-adaptive": dict(color="#e67e22", marker="^", ls="-", label="MuJoCo, error control"),
    "icf": dict(color="#2980b9", marker="o", ls="--", label="ICF, fixed step"),
    "icf-adaptive": dict(color="#27ae60", marker="D", ls="-", label="ICF, error control (CENIC)"),
}
SCENE_NOTE = {
    "hard-clutter": "hard clutter: 10 spheres + 10 cubes in a bin, k = 10⁵ N/m, v_s = 0.1 mm/s",
    "soft-clutter": "soft clutter: 20 spheres in a bin, k = 10³ N/m, v_s = 1 cm/s",
    "ball": "0.1 kg ball, k = 10³ N/m, zero dissipation, 1 m drop, 10 s",
}


def _rows(name: str) -> list[dict]:
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    row[k] = v
            out.append(row)
    return out


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.png/.pdf")


def _dt_label(dt: float) -> str:
    return f"δt = {dt * 1e3:g} ms" if dt >= 1e-3 else f"δt = {dt * 1e6:g} µs"


def workprecision() -> None:
    """One row per world count (N=1 is the paper's single-scene semantics;
    N=1024 the GPU regime), one column per scene. Points whose run timed
    out or exhausted its march budget are drawn as crosses at the top edge:
    the paper omits them; we show the gap."""
    variants = [(n, s) for n in (1, 1024) for s in ("hard-clutter", "soft-clutter")
                if _rows(f"part1_workprecision_{s}_n{n}.csv")]
    if not variants:
        return
    ns = sorted({n for n, _ in variants})
    scenes = [s for s in ("hard-clutter", "soft-clutter") if any(sc == s for _, sc in variants)]
    fig, axes = plt.subplots(len(ns), len(scenes), figsize=(5.2 * len(scenes), 3.9 * len(ns)),
                             constrained_layout=True, squeeze=False)
    for i, n in enumerate(ns):
        for j, scene in enumerate(scenes):
            ax = axes[i][j]
            rows = _rows(f"part1_workprecision_{scene}_n{n}.csv")
            if not rows:
                ax.set_visible(False)
                continue
            ok = [r for r in rows if r["status"] == "ok"]
            bad = [r for r in rows if r["status"] != "ok" and r["accuracy"] != ""]
            for arm in ("mujoco-adaptive", "icf-adaptive"):
                pts = sorted((r["accuracy"], r["wall_s_per_sim_s"]) for r in ok if r["arm"] == arm and r["accuracy"] != "")
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], **STYLE[arm])
            for arm in ("mujoco", "icf"):
                for r in sorted((r for r in ok if r["arm"] == arm and r["dt_s"] != ""), key=lambda r: -r["dt_s"]):
                    ax.axhline(r["wall_s_per_sim_s"], color=STYLE[arm]["color"], ls=":", lw=1.0, alpha=0.8)
                    ax.text(1.4e-1, r["wall_s_per_sim_s"], f"{STYLE[arm]['label'].split(',')[0]} {_dt_label(r['dt_s'])}",
                            fontsize=6.5, color=STYLE[arm]["color"], va="bottom", ha="left")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ymax = max([r["wall_s_per_sim_s"] for r in ok] + [1.0]) * 3.0
            ax.set_ylim(top=ymax)
            for r in bad:
                ax.plot(r["accuracy"], ymax / 1.5, marker="x", ms=8, mew=2, ls="none", color=STYLE[r["arm"]]["color"])
                ax.annotate(r["status"], (r["accuracy"], ymax / 1.5), textcoords="offset points", xytext=(0, -10),
                            ha="center", fontsize=5.5, color=STYLE[r["arm"]]["color"])
            ax.set_xlabel("requested accuracy ε_acc  (error control on positions)")
            ax.set_ylabel("wall time per simulated second [s]")
            trials = int(rows[0].get("trials", 1) or 1)
            budget = rows[0].get("max_substeps", "")
            ax.set_title(f"{SCENE_NOTE[scene]}\n{n} world(s), median of {trials} trial(s), march budget {budget:g} substeps", fontsize=7.5)
            ax.grid(True, which="both", alpha=0.3)
    axes[0][0].legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0.0, 0.05))
    fig.suptitle("Work-precision (CENIC Fig. 9/10 definition): lower is better; × = timeout (>100 s per simulated second) or march budget exhausted",
                 fontsize=8.5)
    _save(fig, "workprecision")


def ball_energy() -> None:
    rows = [r for r in _rows("part1_ball_energy.csv") if r["status"] == "ok"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    for arm in ("icf", "mujoco"):
        pts = sorted((r["dt_s"], abs(r["energy_change_pct"])) for r in rows if r["arm"] == arm and r["dt_s"] != "")
        if pts:
            axes[0].plot([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], **STYLE[arm])
    dts = sorted({r["dt_s"] for r in rows if r["dt_s"] != ""})
    if dts:
        ref = [abs(r["energy_change_pct"]) for r in rows if r["arm"] == "icf" and r["dt_s"] == dts[-1]]
        if ref:
            axes[0].plot(dts, [d / dts[-1] * ref[0] for d in dts], color="gray", ls=":", label="O(δt)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("time step δt [s]")
    axes[0].set_ylabel("|energy change after 10 s| [%]")
    axes[0].set_title("fixed step", fontsize=9)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(True, which="both", alpha=0.3)
    for arm in ("icf-adaptive", "mujoco-adaptive"):
        pts = sorted((r["accuracy"], abs(r["energy_change_pct"])) for r in rows if r["arm"] == arm and r["accuracy"] != "")
        if pts:
            axes[1].plot([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], **STYLE[arm])
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel("requested accuracy ε_acc")
    axes[1].set_ylabel("|energy change after 10 s| [%]")
    axes[1].set_title("error control", fontsize=9)
    axes[1].legend(fontsize=7.5)
    axes[1].grid(True, which="both", alpha=0.3)
    fig.suptitle(f"Energy conservation (CENIC Fig. 8 definition) — {SCENE_NOTE['ball']}", fontsize=9)
    _save(fig, "ball_energy")


def _knob_label(r: dict) -> str:
    return f"ε={r['accuracy']:g}" if r.get("accuracy", "") != "" else _dt_label(r["dt_s"]).replace("δt = ", "")


def penetration() -> None:
    for scene in ("hard-clutter", "soft-clutter"):
        rows = _rows(f"part1_penetration_{scene}.csv")
        if not rows:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
        floor = 1e-10
        for ax, col, title in (
            (axes[0], "pen_mean_m", "mean ground penetration [m]"),
            (axes[1], "pen_max_m", "max ground penetration [m]"),
            (axes[2], "out_of_bin_frac", "fraction of bodies ejected from the bin"),
        ):
            for arm in STYLE:
                pts = sorted((r["wall_ms_per_boundary"], r[col], _knob_label(r)) for r in rows if r["arm"] == arm)
                if not pts:
                    continue
                st = STYLE[arm]
                ys = [p[1] if (p[1] > 0 or col == "out_of_bin_frac") else floor for p in pts]
                ax.plot([p[0] for p in pts], ys, **st)
                dy = {"mujoco": 6, "mujoco-adaptive": -9, "icf": 6, "icf-adaptive": -9}[arm]
                for x, y, lab in zip([p[0] for p in pts], ys, [p[2] for p in pts]):
                    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, dy), fontsize=5.5, color=st["color"])
                if col != "out_of_bin_frac":
                    zx = [p[0] for p in pts if p[1] == 0.0]
                    if zx:
                        ax.plot(zx, [floor] * len(zx), ls="none", marker=st["marker"], markersize=9,
                                markerfacecolor="white", markeredgecolor=st["color"], markeredgewidth=1.6, zorder=5)
            ax.set_xscale("log")
            if col != "out_of_bin_frac":
                ax.set_yscale("log")
                ax.set_ylim(floor / 3, 0.2)
                ax.axhline(floor, color="gray", lw=0.6, ls=":")
                ax.text(0.01, 0.06, "open markers: exactly 0 (drawn at axis floor)", transform=ax.transAxes,
                        ha="left", va="bottom", fontsize=6.5, color="gray")
            ax.set_xlabel("wall time per 10 ms boundary [ms]")
            ax.set_ylabel(title)
            ax.grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=7.5)
        n = int(rows[0]["n_worlds"])
        fig.suptitle(f"Penetration vs wall time — {SCENE_NOTE[scene]}, {n} worlds; labels: δt (fixed) / ε_acc (error control)", fontsize=8.5)
        _save(fig, f"penetration_{scene}")


def scaling() -> None:
    for scene in ("hard-clutter", "soft-clutter"):
        rows = _rows(f"part1_scaling_{scene}.csv")
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
        for arm in STYLE:
            pts = sorted((r["n_worlds"], r["wall_ms_median"], r["wall_ms_p90"], _knob_label(r)) for r in rows if r["arm"] == arm)
            if not pts:
                continue
            xs = [p[0] for p in pts]
            st = dict(STYLE[arm])
            st["label"] = f"{st['label']}, {pts[0][3]}"
            ax.plot(xs, [p[1] for p in pts], **st)
            ax.fill_between(xs, [p[1] for p in pts], [p[2] for p in pts], color=STYLE[arm]["color"], alpha=0.12, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("parallel worlds")
        ax.set_ylabel("wall per 10 ms boundary [ms]  (median → p90)")
        ax.set_title(f"wall time vs world count — {SCENE_NOTE[scene]}", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, f"scaling_{scene}")


if __name__ == "__main__":
    workprecision()
    ball_energy()
    penetration()
    scaling()
