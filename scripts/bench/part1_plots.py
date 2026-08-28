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


SCENE_TITLE = {"soft-clutter": "Soft Clutter", "hard-clutter": "Hard Clutter"}
SCENE_ORDER = ("soft-clutter", "hard-clutter")  # the paper's order: least to most complex


def _wp_rows(scene: str, n: int) -> list[dict]:
    return _rows(f"part1_workprecision_{scene}_n{n}.csv")


def workprecision() -> None:
    """CENIC Fig. 10 layout: one column per scene, one row per world count
    (N=1 is the paper's single-scene semantics, N=1024 the GPU regime).
    x = requested accuracy, y = wall time per simulated second. A run that
    timed out (>100 s per simulated second) or exhausted its march budget
    is a cross at the top edge -- the paper omits such points; we show
    the gap and say why."""
    ns = [n for n in (1, 1024) if any(_wp_rows(sc, n) for sc in SCENE_ORDER)]
    scenes = [sc for sc in SCENE_ORDER if any(_wp_rows(sc, n) for n in ns)]
    if not ns or not scenes:
        return
    fig, axes = plt.subplots(len(ns), len(scenes), figsize=(3.6 * len(scenes), 2.9 * len(ns)),
                             constrained_layout=True, squeeze=False, sharex=True)
    for i, n in enumerate(ns):
        for j, scene in enumerate(scenes):
            ax = axes[i][j]
            rows = _wp_rows(scene, n)
            ok = [r for r in rows if r["status"] == "ok"]
            bad = [r for r in rows if r["status"] != "ok" and r["accuracy"] != ""]
            for arm in ("icf-adaptive", "mujoco-adaptive"):
                pts = sorted((r["accuracy"], r["wall_s_per_sim_s"]) for r in ok if r["arm"] == arm and r["accuracy"] != "")
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=4, lw=1.2, **STYLE[arm])
            for arm in ("icf", "mujoco"):
                for r in sorted((r for r in ok if r["arm"] == arm and r["dt_s"] != ""), key=lambda r: -r["dt_s"]):
                    if r["dt_s"] not in (1e-2, 1e-3):
                        continue  # the paper's Fig. 11 reference steps: 10 ms and 1 ms
                    ax.axhline(r["wall_s_per_sim_s"], color=STYLE[arm]["color"], ls=":", lw=0.9, alpha=0.9)
                    ax.text(1.3e-1, r["wall_s_per_sim_s"], f"{STYLE[arm]['label'].split(',')[0]} fixed {_dt_label(r['dt_s'])}",
                            fontsize=5.5, color=STYLE[arm]["color"], va="bottom", ha="left")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if not ax.xaxis_inverted():  # shared x: invert exactly once (the paper tightens accuracy to the right)
                ax.invert_xaxis()
            top = max([r["wall_s_per_sim_s"] for r in ok if r["wall_s_per_sim_s"] != ""] + [1.0]) * 4.0
            ax.set_ylim(top=top)
            for r in bad:
                ax.plot(r["accuracy"], top / 1.8, marker="x", ms=6, mew=1.6, ls="none", color=STYLE[r["arm"]]["color"])
            if bad:
                ax.text(0.99, 0.97, "× = timeout (>100 s / sim s)", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="gray")
            ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
            ax.text(1.3e-1, 1.0, "real time", fontsize=5.5, color="k", alpha=0.6, va="bottom", ha="left")
            if i == 0:
                ax.set_title(SCENE_TITLE[scene], fontsize=9)
            if i == len(ns) - 1:
                ax.set_xlabel("Accuracy ε_acc")
            if j == 0:
                ax.set_ylabel(f"Wall Time (s) per simulated s\nN = {n} world{'s' if n > 1 else ''}")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6.5, loc="upper left")
    _save(fig, "workprecision")


def speed_bars() -> None:
    """CENIC Fig. 11 format: wall time per simulated second as bars, fixed
    step at δt = 10 ms / 1 ms and error control at ε = 1e-1 / 1e-3 / 1e-5,
    single scene (N=1). A missing bar is a timeout."""
    scenes = [sc for sc in SCENE_ORDER if _wp_rows(sc, 1)]
    if not scenes:
        return
    fig, axes = plt.subplots(len(scenes), 1, figsize=(6.4, 2.6 * len(scenes)), constrained_layout=True, squeeze=False)
    settings = [("fixed", 1e-2, "#1f77b4", "Fixed Step, δt = 10 ms"), ("fixed", 1e-3, "#ff7f0e", "Fixed Step, δt = 1 ms"),
                ("ec", 1e-1, "#2ca02c", "Error Control, ε = 10⁻¹"), ("ec", 1e-3, "#d62728", "Error Control, ε = 10⁻³"),
                ("ec", 1e-5, "#9467bd", "Error Control, ε = 10⁻⁵")]
    groups = [("icf-adaptive", "ICF\n(error control)", "ec"), ("icf", "ICF\n(fixed step)", "fixed"),
              ("mujoco-adaptive", "MuJoCo\n(error control)", "ec"), ("mujoco", "MuJoCo\n(fixed step)", "fixed")]
    for ax, scene in zip(axes[:, 0], scenes):
        rows = _wp_rows(scene, 1)
        x = 0.0
        ticks, labels = [], []
        for arm, name, kind in groups:
            xs0 = x
            for skind, val, color, lab in settings:
                if skind != kind:
                    continue
                r = next((r for r in rows if r["arm"] == arm and (r["accuracy"] == val if kind == "ec" else r["dt_s"] == val)), None)
                if r is not None and r["status"] == "ok":
                    ax.bar(x, r["wall_s_per_sim_s"], width=0.8, color=color, label=lab)
                else:
                    ax.bar(x, 1e-3, width=0.8, color=color, alpha=0.25, hatch="//", label=lab)
                    ax.text(x, 1.2e-3, r["status"] if r else "n/a", rotation=90, fontsize=5.5, ha="center", va="bottom", color=color)
                x += 1.0
            ticks.append((xs0 + x - 1.0) / 2.0)
            labels.append(name)
            x += 0.8
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Wall Time (s)\nper simulated s", fontsize=8)
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
        ax.set_title(SCENE_TITLE[scene] + "  (N = 1, GPU)", fontsize=8.5)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
    h, l = axes[0][0].get_legend_handles_labels()
    uniq = dict(zip(l, h))
    fig.legend(uniq.values(), uniq.keys(), loc="upper center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, 1.06), frameon=False)
    _save(fig, "speed_bars")


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
    speed_bars()
    ball_energy()
    penetration()
    scaling()
