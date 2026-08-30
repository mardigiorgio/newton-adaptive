# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Part-1 figures from the committed CSVs, with self-contained figure text:
every point states its accuracy eps_acc (adaptive) or time step dt (fixed).
CPU only; re-run after any sweep.

    uv run python scripts/bench/part1_plots.py
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

STYLE = {
    "mujoco": dict(color="#c0392b", marker="s", ls="--", label="MuJoCo"),
    "mujoco-adaptive": dict(color="#e67e22", marker="^", ls="-", label="MuJoCo EC"),
    "icf": dict(color="#2980b9", marker="o", ls="--", label="ICF"),
    "icf-adaptive": dict(color="#27ae60", marker="D", ls="-", label="ICF EC (CENIC)"),
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
SCENE_ORDER = ("soft-clutter", "hard-clutter")  # least to most complex


def _wp_rows(scene: str, n: int) -> list[dict]:
    return _rows(f"part1_workprecision_{scene}_n{n}.csv")


def workprecision() -> None:
    """One column per scene, one row per world count
    (N=1 is the single-scene setting, N=1024 the GPU regime).
    x = requested accuracy, y = wall time per simulated second. A run that
    timed out (>100 s per simulated second) or exhausted its march budget
    is a cross at the top edge."""
    ns = [n for n in (1, 1024) if any(_wp_rows(sc, n) for sc in SCENE_ORDER)]
    scenes = [sc for sc in SCENE_ORDER if any(_wp_rows(sc, n) for n in ns)]
    if not ns or not scenes:
        return
    fig, axes = plt.subplots(
        len(ns),
        len(scenes),
        figsize=(3.6 * len(scenes), 2.9 * len(ns)),
        constrained_layout=True,
        squeeze=False,
        sharex=True,
    )
    for i, n in enumerate(ns):
        for j, scene in enumerate(scenes):
            ax = axes[i][j]
            rows = _wp_rows(scene, n)
            ok = [r for r in rows if r["status"] == "ok"]
            bad = [r for r in rows if r["status"] != "ok" and r["accuracy"] != ""]
            for arm in ("icf-adaptive", "mujoco-adaptive"):
                pts = sorted(
                    (r["accuracy"], r["wall_s_per_sim_s"]) for r in ok if r["arm"] == arm and r["accuracy"] != ""
                )
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=4, lw=1.2, **STYLE[arm])
            for arm in ("icf", "mujoco"):
                for r in sorted((r for r in ok if r["arm"] == arm and r["dt_s"] != ""), key=lambda r: -r["dt_s"]):
                    if r["dt_s"] not in (1e-2, 1e-3):
                        continue  # reference steps: 10 ms and 1 ms
                    ax.axhline(r["wall_s_per_sim_s"], color=STYLE[arm]["color"], ls=":", lw=0.9, alpha=0.9)
                    ax.text(
                        0.995,
                        r["wall_s_per_sim_s"],
                        f"{STYLE[arm]['label'].split(',')[0]} fixed {_dt_label(r['dt_s'])}",
                        fontsize=5.5,
                        color=STYLE[arm]["color"],
                        va="bottom",
                        ha="right",
                        transform=ax.get_yaxis_transform(),
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            if not ax.xaxis_inverted():  # shared x: invert exactly once (accuracy tightens to the right)
                ax.invert_xaxis()
            thresh = 100.0 * n  # timeout: 100 s per simulated second of one scene
            top = max([r["wall_s_per_sim_s"] for r in ok if r["wall_s_per_sim_s"] != ""] + [1.0]) * 4.0
            if any(r["status"] == "timeout" for r in bad):
                top = max(top, thresh * 2.5)
            ax.set_ylim(top=top)
            if thresh < top:
                ax.axhline(thresh, color="gray", lw=0.8, ls="-.")
                ax.text(
                    0.005,
                    thresh,
                    f"timeout: 100 s per scene-second (×{n})" if n > 1 else "timeout: 100 s per simulated second",
                    fontsize=5.5,
                    color="gray",
                    va="bottom",
                    ha="left",
                    transform=ax.get_yaxis_transform(),
                )
            for r in bad:
                if r["status"] == "timeout":
                    ax.plot(r["accuracy"], thresh, marker="x", ms=6, mew=1.6, ls="none", color=STYLE[r["arm"]]["color"])
                else:  # 'budget' (practical wall cap) or 'budget-exhausted' / 'fail'
                    ax.plot(
                        r["accuracy"], top / 1.6, marker="+", ms=7, mew=1.6, ls="none", color=STYLE[r["arm"]]["color"]
                    )
                    ax.annotate(
                        r["status"],
                        (r["accuracy"], top / 1.6),
                        textcoords="offset points",
                        xytext=(0, -9),
                        ha="center",
                        fontsize=5,
                        color=STYLE[r["arm"]]["color"],
                    )
            ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
            ax.text(
                0.005,
                1.0,
                "real time",
                fontsize=5.5,
                color="k",
                alpha=0.6,
                va="bottom",
                ha="left",
                transform=ax.get_yaxis_transform(),
            )
            if i == 0:
                ax.set_title(SCENE_TITLE[scene], fontsize=9)
            if i == len(ns) - 1:
                ax.set_xlabel("Accuracy")
            if j == 0:
                ax.set_ylabel(f"Wall Time (s)\nN = {n}")
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
    axes[0][0].legend(fontsize=6.5, loc="upper left")
    _save(fig, "workprecision")


def speed_bars() -> None:
    """Wall time per simulated second as bars, single scene: fixed step at
    dt = 10 ms / 1 ms and error control at eps = 1e-1 / 1e-3 / 1e-5, in the
    SAME arm colors as every other figure (settings differ by lightness)."""
    import matplotlib.colors as mcolors

    scenes = [sc for sc in SCENE_ORDER if _wp_rows(sc, 1)]
    if not scenes:
        return
    fig, axes = plt.subplots(len(scenes), 1, figsize=(6.4, 2.6 * len(scenes)), constrained_layout=True, squeeze=False)
    groups = [
        ("icf", "ICF\nfixed step", [("dt_s", 1e-2, "δt = 10 ms"), ("dt_s", 1e-3, "δt = 1 ms")]),
        (
            "icf-adaptive",
            "ICF\nerror control",
            [("accuracy", 1e-1, "ε = 10⁻¹"), ("accuracy", 1e-3, "ε = 10⁻³"), ("accuracy", 1e-5, "ε = 10⁻⁵")],
        ),
        ("mujoco", "MuJoCo\nfixed step", [("dt_s", 1e-2, "δt = 10 ms"), ("dt_s", 1e-3, "δt = 1 ms")]),
        (
            "mujoco-adaptive",
            "MuJoCo\nerror control",
            [("accuracy", 1e-1, "ε = 10⁻¹"), ("accuracy", 1e-3, "ε = 10⁻³"), ("accuracy", 1e-5, "ε = 10⁻⁵")],
        ),
    ]
    for ax, scene in zip(axes[:, 0], scenes):
        rows = _wp_rows(scene, 1)
        x = 0.0
        ticks, labels = [], []
        for arm, name, settings in groups:
            base = mcolors.to_rgb(STYLE[arm]["color"])
            xs0 = x
            for i, (key, val, lab) in enumerate(settings):
                shade = 1.0 - 0.28 * i  # coarser/looser setting lighter, finer/tighter darker
                col = tuple(1 - (1 - c) * shade for c in base)
                r = next((r for r in rows if r["arm"] == arm and r[key] == val), None)
                if r is not None and r["status"] == "ok":
                    ax.bar(x, r["wall_s_per_sim_s"], width=0.8, color=col, edgecolor=STYLE[arm]["color"], lw=0.8)
                    ax.text(
                        x,
                        r["wall_s_per_sim_s"] * 1.08,
                        lab,
                        rotation=90,
                        fontsize=5.5,
                        ha="center",
                        va="bottom",
                        color=STYLE[arm]["color"],
                    )
                else:
                    ax.bar(x, 1e-3, width=0.8, color=col, alpha=0.3, hatch="//", edgecolor=STYLE[arm]["color"])
                    ax.text(
                        x,
                        1.2e-3,
                        f"{lab}: {r['status'] if r else 'n/a'}",
                        rotation=90,
                        fontsize=5.5,
                        ha="center",
                        va="bottom",
                        color=STYLE[arm]["color"],
                    )
                x += 1.0
            ticks.append((xs0 + x - 1.0) / 2.0)
            labels.append(name)
            x += 0.8
        ax.set_yscale("log")
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("Wall Time (s)", fontsize=8)
        ax.axhline(1.0, color="k", lw=0.6, alpha=0.35)
        ax.text(
            0.005, 1.0, "real time", fontsize=6, color="k", alpha=0.6, va="bottom", transform=ax.get_yaxis_transform()
        )
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.set_ylim(top=max([r["wall_s_per_sim_s"] for r in rows if r["wall_s_per_sim_s"] != ""] + [1.0]) * 6)
    _save(fig, "speed_bars")


def ball_energy() -> None:
    """Fig. 8: percent energy change after 10 s vs fixed time step. Error
    control is reported in the text and in ball_workprecision (position-only
    error control does not see the energy a soft impact loses until eps is
    very tight)."""
    rows = [r for r in _rows("part1_ball_energy.csv") if r["status"] == "ok"]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    for arm in ("icf", "mujoco"):
        pts = sorted((r["dt_s"], abs(r["energy_change_pct"])) for r in rows if r["arm"] == arm and r["dt_s"] != "")
        if pts:
            ax.plot([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], **STYLE[arm])
    dts = sorted({r["dt_s"] for r in rows if r["dt_s"] != ""})
    if dts:
        ref = [abs(r["energy_change_pct"]) for r in rows if r["arm"] == "icf" and r["dt_s"] == dts[-1]]
        if ref:
            ax.plot(dts, [d / dts[-1] * ref[0] for d in dts], color="gray", ls=":", label="O(δt)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Time Step (s)")
    ax.set_ylabel("Change in Energy (%)")
    ax.legend(fontsize=7.5)
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, "ball_energy")


def _knob_label(r: dict) -> str:
    return f"ε={r['accuracy']:g}" if r.get("accuracy", "") != "" else _dt_label(r["dt_s"]).replace("δt = ", "")


def penetration() -> None:
    """Mean and max ground penetration vs wall time; the ejection panel is
    drawn only if any configuration ejected a body, otherwise the title
    states that none did. The ``_margin5mm`` variant (contact activated
    5 mm before touching) is drawn as its own figure when present."""
    for scene, suffix in [(sc, sf) for sc in ("hard-clutter", "soft-clutter") for sf in ("", "_margin5mm")]:
        rows = _rows(f"part1_penetration_{scene}{suffix}.csv")
        if not rows:
            continue
        any_eject = any(r["out_of_bin_frac"] > 0 for r in rows)
        panels = [("pen_mean_m", "Mean penetration (m)"), ("pen_max_m", "Max penetration (m)")]
        if any_eject:
            panels.append(("out_of_bin_frac", "Fraction of bodies ejected"))
        fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 3.8), constrained_layout=True)
        floor = 1e-10
        for ax, (col, title) in zip(axes, panels):
            for arm in STYLE:
                pts = sorted((r["wall_ms_per_boundary"], r[col], _knob_label(r)) for r in rows if r["arm"] == arm)
                if not pts:
                    continue
                st = STYLE[arm]
                ys = [p[1] if (p[1] > 0 or col == "out_of_bin_frac") else floor for p in pts]
                ax.plot([p[0] for p in pts], ys, **st)
                dy = {"mujoco": 6, "mujoco-adaptive": -9, "icf": 6, "icf-adaptive": -9}[arm]
                for x, y, lab in zip([p[0] for p in pts], ys, [p[2] for p in pts]):
                    ax.annotate(
                        lab, (x, y), textcoords="offset points", xytext=(4, dy), fontsize=5.5, color=st["color"]
                    )
                if col != "out_of_bin_frac":
                    zx = [p[0] for p in pts if p[1] == 0.0]
                    if zx:
                        ax.plot(
                            zx,
                            [floor] * len(zx),
                            ls="none",
                            marker=st["marker"],
                            markersize=9,
                            markerfacecolor="white",
                            markeredgecolor=st["color"],
                            markeredgewidth=1.6,
                            zorder=5,
                        )
            ax.set_xscale("log")
            if col != "out_of_bin_frac":
                ax.set_yscale("log")
                vals = [r[col] for r in rows]
                ymax = max(vals + [1e-6])
                pos = [v for v in vals if v > 0]
                has_zero = any(v == 0.0 for v in vals)
                ymin = (floor / 3) if has_zero else (min(pos) / 4.0 if pos else 1e-7)
                ax.set_ylim(ymin, ymax * 4.0)
                if ymax > 0.1:
                    ax.axhline(0.025, color="gray", lw=0.6, ls="--")
                    ax.text(
                        0.01,
                        0.025,
                        "object radius: deeper = tunnelled",
                        fontsize=6,
                        color="gray",
                        va="bottom",
                        transform=ax.get_yaxis_transform(),
                    )
                if has_zero:
                    ax.axhline(floor, color="gray", lw=0.6, ls=":")
                    ax.text(
                        0.01,
                        0.06,
                        "open markers: exactly 0 (drawn at axis floor)",
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=6.5,
                        color="gray",
                    )
            ax.set_xlabel(f"Wall Time per {rows[0].get('dt_outer_s', 0.01) * 1e3:g} ms step (ms)")
            ax.set_ylabel(title)
            ax.grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=7.5)
        n = int(rows[0]["n_worlds"])
        eject_note = "" if any_eject else "; no body left the bin in any configuration"
        margin_note = "; 5 mm collision margin" if suffix else ""
        fig.suptitle(SCENE_TITLE[scene], fontsize=9)
        _save(fig, f"penetration_{scene}{suffix}")


def scaling() -> None:
    for scene in ("hard-clutter", "soft-clutter"):
        rows = _rows(f"part1_scaling_{scene}.csv")
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
        for arm in STYLE:
            has_trials = rows and rows[0].get("wall_ms_trial_min", "") != ""
            lo_key, hi_key = (
                ("wall_ms_trial_min", "wall_ms_trial_max") if has_trials else ("wall_ms_median", "wall_ms_p90")
            )
            pts = sorted(
                (r["n_worlds"], r["wall_ms_median"], r[lo_key], r[hi_key], _knob_label(r))
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            st = dict(STYLE[arm])
            st["label"] = f"{st['label']}, {pts[0][4]}"
            ax.plot(xs, [p[1] for p in pts], **st)
            ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=STYLE[arm]["color"], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Parallel worlds")
        trials = int(rows[0].get("trials", 1) or 1)
        band = f"band: spread of {trials} independent runs" if trials > 1 else "band: median → p90"
        ax.set_ylabel(f"Wall Time per {rows[0].get('dt_outer_s', 0.01) * 1e3:g} ms step (ms)")
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, f"scaling_{scene}")


def realtime_trace() -> None:
    """Real-time rate and solver work along a 5 s drop (top: real-time rate
    = 10 ms / wall per boundary; middle: march iterations per boundary;
    bottom: cumulative wall). Fixed step pays the same every boundary;
    error control pays for impacts and coasts at dt_max once settled."""
    import numpy as np

    for n in (64, 1):
        rows = _rows(f"part1_realtime_trace_hard-clutter_n{n}.csv")
        if not rows:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), constrained_layout=True, sharex=True)
        series = {}
        for r in rows:
            key = (r["arm"], r["accuracy"] if r["accuracy"] != "" else r["dt_s"])
            series.setdefault(key, []).append((r["t_s"], r["wall_ms"], r["iters"]))
        for (arm, knob), pts in series.items():
            pts.sort()
            t = np.array([p[0] for p in pts])
            w = np.array([p[1] for p in pts])
            it = np.array([p[2] for p in pts])
            st = dict(STYLE[arm])
            st["marker"] = None
            lab = f"{st['label']}, " + (f"ε = {knob:g}" if arm.endswith("adaptive") else _dt_label(knob))
            st["label"] = lab
            # two settings per arm share a color: the coarser / looser one is dotted, the finer solid/dashed
            settings = sorted({k for (a2, k) in series if a2 == arm}, reverse=True)
            st["ls"] = (
                (":" if knob == settings[0] else "--")
                if not arm.endswith("adaptive")
                else ("-" if knob == settings[-1] else "-.")
            )
            # smooth the rate over 10 boundaries (0.1 s) for legibility; cumulative is exact
            k = 10
            w_s = np.convolve(w, np.ones(k) / k, mode="same")
            dto_ms = rows[0].get("dt_outer_s", 0.01) * 1e3
            axes[0].plot(t, 100.0 * dto_ms / w_s, lw=1.1, **st)
            axes[1].plot(t, np.convolve(it, np.ones(k) / k, mode="same"), lw=1.1, **st)
            axes[2].plot(t, np.cumsum(w) / 1e3, lw=1.1, **st)
        axes[0].axhline(100.0, color="k", lw=0.6, ls="--", alpha=0.5)
        axes[0].text(0.005, 100.0, "100% RTR", fontsize=6, va="bottom", transform=axes[0].get_yaxis_transform())
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Real-Time Rate (%)")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(f"Solver steps per {rows[0].get('dt_outer_s', 0.01) * 1e3:g} ms")
        axes[2].set_ylabel("Cumulative Wall Time (s)")
        axes[2].set_xlabel("Simulation Time (s)")
        for ax in axes:
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=7)
        axes[0].set_title("Hard Clutter", fontsize=9)
        h, l = axes[2].get_legend_handles_labels()
        fig.legend(h, l, fontsize=6.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.11), frameon=False)
        _save(fig, f"realtime_trace_n{n}")


# ---------------------------------------------------------------- story figures
import math as _math

_OBJECT_MASS = 1000.0 * 4.0 / 3.0 * _math.pi * 0.025**3  # water-density 2.5 cm sphere [kg]
_SCENE_K = {"soft-clutter": 1.0e3, "hard-clutter": 1.0e5}


def _static_pen(scene: str) -> float:
    """Single-object static penetration m*g/k [m] -- the compliance the contact model prescribes."""
    return _OBJECT_MASS * 9.81 / _SCENE_K[scene]


_DROP_HEIGHT = 0.40  # top drop layer [m] -> impact speed sqrt(2 g h)


def _impact_pen(scene: str) -> float:
    """Deepest penetration the contact model itself produces for the drop's
    impact speed: v * sqrt(m/k) (Hertz-free linear spring, single body)."""
    v = _math.sqrt(2.0 * 9.81 * _DROP_HEIGHT)
    return v * _math.sqrt(_OBJECT_MASS / _SCENE_K[scene])


def artifacts() -> None:
    """Claim 1. Top row: max penetration relative to the model's own impact
    depth v*sqrt(m/k) -- above 1 the step, not the model, made the depth
    (artifact); ejections ringed; cheapest artifact-free setting starred.
    Bottom row: mean penetration relative to the resting depth m*g/k --
    how faithfully each arm reproduces the model at rest."""
    scenes = [sc for sc in SCENE_ORDER if _rows(f"part1_penetration_{sc}.csv")]
    if not scenes:
        return
    fig, axes = plt.subplots(2, len(scenes), figsize=(5.2 * len(scenes), 7.4), constrained_layout=True, squeeze=False)
    for j, scene in enumerate(scenes):
        rows = _rows(f"part1_penetration_{scene}.csv")
        dto = rows[0].get("dt_outer_s", 0.01) or 0.01
        n = int(rows[0]["n_worlds"])
        d_imp, d_stat = _impact_pen(scene), _static_pen(scene)
        # ---- top: artifacts
        ax = axes[0][j]
        ax.axhspan(1.0, 1e6, color="#c0392b", alpha=0.06, lw=0)
        ax.axhline(1.0, color="#c0392b", lw=0.9, ls="--")
        ax.text(
            0.005,
            1.0,
            "impact depth",
            fontsize=6.5,
            color="#c0392b",
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        cheapest = {}
        for arm in STYLE:
            pts = sorted(
                (
                    (r["wall_ms_per_boundary"] / 1e3) / dto,
                    r["pen_max_m"] / d_imp,
                    r["out_of_bin_frac"] > 0,
                    _knob_label(r),
                )
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            st = STYLE[arm]
            xs = [p[0] for p in pts]
            ys = [max(p[1], 1e-3) for p in pts]
            ax.plot(xs, ys, lw=1.2, ms=6, **st)
            for x, y, ej, lab in zip(xs, ys, [p[2] for p in pts], [p[3] for p in pts]):
                if ej:
                    ax.plot(x, y, marker="o", ms=13, mfc="none", mec="#c0392b", mew=1.6, ls="none", zorder=4)
                    ax.annotate(
                        "ejects",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 9),
                        ha="center",
                        fontsize=6,
                        color="#c0392b",
                    )
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, -9), fontsize=5.5, color=st["color"])
            clean = [
                (x, y, lab)
                for x, y, ej, lab in zip(xs, ys, [p[2] for p in pts], [p[3] for p in pts])
                if y <= 1.0 and not ej
            ]
            # A star means "cheapest artifact-free setting". It is only drawn
            # when the criterion separates this arm's settings on this scene:
            # if every point sits within 25 % of the bound or below and none
            # ejects, the flag is straddling the bound's own approximation and
            # a star would rank noise (soft clutter).
            separated = any(y > 1.25 or ej for y, ej in zip(ys, [p[2] for p in pts]))
            if clean and separated:
                x, y, lab = min(clean)
                ax.plot(x, y, marker="*", ms=15, color=st["color"], mec="k", mew=0.6, ls="none", zorder=5)
                cheapest[arm] = (x, lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ymax = max([r["pen_max_m"] / d_imp for r in rows] + [2.0])
        ax.set_ylim(top=ymax * 30.0)
        ax.set_ylabel("max penetration / impact depth")
        ax.grid(True, which="both", alpha=0.3)
        parts = [
            f"{name}: {'never' if arm not in cheapest else f'{cheapest[arm][1]} ({cheapest[arm][0]:.2g} s)'}"
            for arm, name in (
                ("mujoco", "MuJoCo fixed"),
                ("mujoco-adaptive", "MuJoCo error control"),
                ("icf", "ICF fixed"),
                ("icf-adaptive", "ICF error control"),
            )
        ]
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        if j == 0:
            ax.plot([], [], marker="*", ms=12, color="gray", mec="k", ls="none", label="cheapest artifact-free")
            ax.plot([], [], marker="o", ms=10, mfc="none", mec="#c0392b", ls="none", label="ejection")
            ax.legend(fontsize=6.5, loc="upper right")
        # ---- bottom: fidelity at rest
        ax = axes[1][j]
        ax.axhline(1.0, color="gray", lw=0.9, ls="--")
        ax.text(
            0.005,
            1.0,
            "resting depth",
            fontsize=6.5,
            color="gray",
            va="bottom",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
        for arm in STYLE:
            pts = sorted(
                ((r["wall_ms_per_boundary"] / 1e3) / dto, r["pen_mean_m"] / d_stat, _knob_label(r))
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            st = STYLE[arm]
            xs = [p[0] for p in pts]
            ys = [max(p[1], 1e-3) for p in pts]
            ax.plot(xs, ys, lw=1.2, ms=6, **st)
            for x, y, lab in zip(xs, ys, [p[2] for p in pts]):
                ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, -9), fontsize=5.5, color=st["color"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"Wall Time (s) per simulated second, {n} scenes")
        ax.set_ylabel("mean penetration / resting depth")
        ax.grid(True, which="both", alpha=0.3)
    _save(fig, "artifacts")


def scaling_per_world() -> None:
    """Batching: wall per world per boundary [us] vs world count; the
    curve falling means the GPU is still amortizing; flat means saturated."""
    for scene in SCENE_ORDER:
        rows = _rows(f"part1_scaling_{scene}.csv")
        if not rows:
            continue
        dto = rows[0].get("dt_outer_s", 0.01) or 0.01
        fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
        for arm in STYLE:
            pts = sorted(
                (r["n_worlds"], 1e3 * r["wall_ms_median"] / r["n_worlds"], _knob_label(r))
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            st = dict(STYLE[arm])
            st["label"] = f"{st['label']}, {pts[0][2]}"
            ax.plot([p[0] for p in pts], [p[1] for p in pts], **st)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Parallel worlds")
        ax.set_ylabel(f"Wall Time per world per {dto * 1e3:g} ms step (µs)")
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        _save(fig, f"scaling_per_world_{scene}")


def ball_workprecision() -> None:
    """Claim 3 on the same axes: energy error vs cost for fixed step (dt
    sweep) and error control (eps sweep)."""
    rows = [r for r in _rows("part1_ball_energy.csv") if r["status"] == "ok" and r.get("wall_s_per_sim_s", "") != ""]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
    for arm in STYLE:
        pts = sorted(
            (r["wall_s_per_sim_s"], max(abs(r["energy_change_pct"]), 1e-3), _knob_label(r))
            for r in rows
            if r["arm"] == arm
        )
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **STYLE[arm])
        moves = max(p[1] for p in pts) / max(min(p[1] for p in pts), 1e-9) > 1.5
        if moves:
            for x, y, lab in pts:
                ax.annotate(
                    lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=STYLE[arm]["color"]
                )
        else:
            x, y, _ = pts[-1]
            first, last = pts[0][2], pts[-1][2]
            ax.annotate(
                f"{first} … {last}: all ≈ 100 % lost",
                (x, y),
                textcoords="offset points",
                xytext=({"icf-adaptive": -60}.get(arm, 4), {"mujoco": 10, "mujoco-adaptive": -12, "icf-adaptive": -14}.get(arm, 0)),
                fontsize=5.5,
                color=STYLE[arm]["color"],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Wall Time (s) per simulated second")
    ax.set_ylabel("|energy change after 10 s| (%)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    _save(fig, "ball_workprecision")


def stiffness_sweep() -> None:
    """Realized contact stiffness: resting penetration of one sphere over the
    model's m g / k as the requested k rises (ICF Fig. 18's axis), per arm at
    fixed dt and under error control."""
    rows = [r for r in _rows("part1_stiffness_sweep.csv") if r.get("finite") in (True, "True")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
    variants = {}
    for r in rows:
        variants.setdefault((r["arm"], _knob_label(r)), []).append((r["k_N_per_m"], r["ratio"]))
    for (arm, lab), pts in sorted(variants.items()):
        pts.sort()
        st = dict(STYLE[arm])
        st["label"] = f"{STYLE[arm]['label']}, {lab}"
        if "10 ms" in lab or "0.001" in lab:  # the coarser setting of each pair is hollow
            st["mfc"] = "none"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **st)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Requested contact stiffness k (N/m)")
    ax.set_ylabel("Resting penetration / (m g / k)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=6)
    _save(fig, "stiffness_sweep")


def consistency() -> None:
    """Measured error vs cost: deviation from a dt = 0.1 ms reference after
    0.1 s windows restarted from the reference (Erez 2015's short-window
    self-consistency) -- max over bodies (the L-inf norm of the error
    controller) averaged over windows -- one panel per scene, every point
    labeled."""
    fig, axes = plt.subplots(
        1, len(SCENE_ORDER), figsize=(5.4 * len(SCENE_ORDER), 3.9), constrained_layout=True, squeeze=False
    )
    any_data = False
    for ax, scene in zip(axes[0], SCENE_ORDER):
        rows = [
            r
            for r in _rows(f"part1_consistency_{scene}.csv")
            if r.get("wall_s_per_sim_s", "") != "" and r.get("dev_mean_m", "") != ""
        ]
        if not rows:
            ax.set_visible(False)
            continue
        any_data = True
        for arm in STYLE:
            pts = sorted(
                (
                    r["wall_s_per_sim_s"],
                    r["dev_mean_m"] * 1e3,
                    _knob_label(r),
                    r.get("dt_s", "") != "" and abs(r["dt_s"] - 1e-4) < 1e-12,
                )
                for r in rows
                if r["arm"] == arm
            )
            if not pts:
                continue
            ax.plot([p[0] for p in pts if not p[3]], [p[1] for p in pts if not p[3]], ms=5, **STYLE[arm])
            for x, y, lab, is_floor in pts:
                if is_floor:  # the reference restarted against itself: the instrument's floor
                    ax.plot([x], [y], marker=STYLE[arm]["marker"], mfc="none", color=STYLE[arm]["color"], ls="none", ms=6)
                    ax.axhline(y, color=STYLE[arm]["color"], lw=0.6, ls=":", alpha=0.7)
                    lab = "floor"
                # fixed arms label above the point, EC arms below: EC points can
                # land on a fixed-arm point (same wall, same deviation) and the
                # labels must not collide
                dy = 4 if not arm.endswith("adaptive") else -9
                ax.annotate(
                    lab, (x, y), textcoords="offset points", xytext=(4, dy), fontsize=5.5, color=STYLE[arm]["color"]
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Wall Time (s) per simulated second")
        ax.set_ylabel("Deviation (mm)")
        ax.set_title(SCENE_TITLE[scene], fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    if any_data:
        _save(fig, "consistency")
    else:
        plt.close(fig)


def actuated() -> None:
    """Actuated push: box lift, box pitch rate, max tip penetration and
    tip-box relative velocity vs the controller gain K_p, per arm at one
    fixed dt and one accuracy; unstable cells are drawn as crosses on the
    top edge."""
    rows = _rows("part1_actuated.csv")
    if not rows or "box_lift_max_m" not in rows[0]:
        return
    picks = {"icf": ("dt_s", 1e-3), "mujoco": ("dt_s", 1e-3), "icf-adaptive": ("accuracy", 1e-3), "mujoco-adaptive": ("accuracy", 1e-3)}
    metrics = [("box_lift_max_m", 1e3, "Box lift (mm)"), ("box_pitch_rate_rms", 1.0, "Box pitch rate (rad/s)"),
               ("pen_tip_max_m", 1e3, "Tip penetration (mm)"), ("rel_vx_rms_m_s", 1.0, "Tip–box relative velocity (m/s)")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)
    for ax, (key, scale, ylab) in zip(axes, metrics):
        floor = 1e-3 if scale == 1e3 else 1e-4
        for arm, (col, val) in picks.items():
            sel = [r for r in rows if r["arm"] == arm and r.get(col, "") != "" and abs(r[col] - val) < 1e-12]
            good = sorted((r["kp"], max(r[key] * scale, floor)) for r in sel if r.get("unstable") in (False, "False") and r.get(key, "") != "")
            bad = sorted(r["kp"] for r in sel if r.get("unstable") in (True, "True"))
            st = dict(STYLE[arm]); st["label"] = f"{STYLE[arm]['label']}, {_dt_label(val).replace('δt = ', '') if col == 'dt_s' else f'ε={val:g}'}"
            if good:
                ax.plot([p[0] for p in good], [p[1] for p in good], ms=5, **st)
            if bad:
                ax.plot(bad, [1.0] * len(bad), ls="none", marker="x", ms=8, mew=2, color=STYLE[arm]["color"], transform=ax.get_xaxis_transform(), clip_on=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("K_p (N/m)"); ax.set_ylabel(ylab); ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=6)
    k = rows[0]["k"]; v = rows[0]["slide_speed"]
    _save(fig, "actuated")



def actuated_chatter() -> None:
    """Tip-box relative velocity in the cruise vs cost at K_p = 1e5: the
    chatter a held 100 Hz target excites in a stiff PD on a light tip
    appears as the step is refined; every point labeled."""
    rows = [r for r in _rows("part1_actuated.csv") if r.get("unstable") in (False, "False") and abs(r["kp"] - 1e5) < 1.0 and r.get("rel_vx_rms_m_s", "") != ""]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9), constrained_layout=True)
    for arm in STYLE:
        pts = sorted((r["wall_s_per_sim_s"], max(r["rel_vx_rms_m_s"], 1e-4), _knob_label(r)) for r in rows if r["arm"] == arm)
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **STYLE[arm])
        merged = []  # coincident points (same result at several settings) share one label
        for x, y, lab in pts:
            if merged and abs(merged[-1][1] - y) < 1e-12 and abs(merged[-1][0] - x) / x < 0.2:
                merged[-1] = (merged[-1][0], y, merged[-1][2].split(" … ")[0] + " … " + lab)
            else:
                merged.append((x, y, lab))
        for x, y, lab in merged:
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5.5, color=STYLE[arm]["color"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Wall Time (s)"); ax.set_ylabel("Tip–box relative velocity (m/s)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    _save(fig, "actuated_chatter")



def actuated_scaling() -> None:
    """Throughput in the actuated regime: wall per world per boundary and
    inner attempts per world per boundary vs number of heterogeneous worlds."""
    rows = [r for r in _rows("part1_actuated_scaling.csv") if r.get("unstable") in (False, "False")]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for arm in STYLE:
        pts = sorted((r["n_worlds"], r["wall_ms_per_boundary"] / r["n_worlds"] * 1e3, r["steps_per_boundary"]) for r in rows if r["arm"] == arm)
        if not pts:
            continue
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **STYLE[arm])
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], ms=5, **STYLE[arm])
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("Worlds"); ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Wall Time per world (µs)"); axes[1].set_ylabel("Inner steps per boundary")
    axes[0].set_title("(a)", loc="left", fontsize=10); axes[1].set_title("(b)", loc="left", fontsize=10); axes[0].legend(fontsize=7)
    _save(fig, "actuated_scaling")


if __name__ == "__main__":
    workprecision()
    speed_bars()
    ball_energy()
    penetration()
    artifacts()
    scaling()
    scaling_per_world()
    ball_workprecision()
    realtime_trace()
    stiffness_sweep()
    consistency()
    actuated()
    actuated_chatter()
    actuated_scaling()
