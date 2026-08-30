# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Part-1 story figures: three composites built from the committed CSVs,
each carrying one sentence of the argument in its title.

  story_step.pdf        At the step a learner uses, fixed stepping makes
                        artifacts; error control removes them.
  story_cost.pdf        Being artifact-free costs fixed stepping every step
                        and error control only the impacts.
  story_convergence.pdf Both solvers converge; ICF realizes the model at
                        any step, MuJoCo's constraint conserves an impact.

    uv run python scripts/bench/part1_story.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import part1_plots as P  # noqa: E402

STYLE = P.STYLE
ARMS = ("mujoco", "mujoco-adaptive", "icf", "icf-adaptive")
SHORT = {"mujoco": "MuJoCo", "mujoco-adaptive": "MuJoCo EC", "icf": "ICF", "icf-adaptive": "ICF EC"}


def _is(r, arm, col, val):
    return r["arm"] == arm and r.get(col, "") != "" and abs(r[col] - val) < 1e-12


def story_step() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), constrained_layout=True)
    # (a) realized stiffness
    ax = axes[0, 0]
    rows = [r for r in P._rows("part1_stiffness_sweep.csv") if r.get("finite") in (True, "True")]
    for arm, col, val, extra in (("icf", "dt_s", 1e-2, {}), ("mujoco", "dt_s", 1e-2, {"mfc": "none"}), ("mujoco", "dt_s", 1e-3, {}),
                                 ("icf-adaptive", "accuracy", 1e-3, {}), ("mujoco-adaptive", "accuracy", 1e-3, {})):
        pts = sorted((r["k_N_per_m"], r["ratio"]) for r in rows if _is(r, arm, col, val))
        st = dict(STYLE[arm]); st.update(extra); st["label"] = f"{SHORT[arm]}, {P._knob_label({'dt_s': val} if col == 'dt_s' else {'accuracy': val})}"
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **st)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Requested stiffness k (N/m)"); ax.set_ylabel("Penetration / (mg/k)")
    ax.set_title("(a)", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    # (b) hard clutter at the learner's step
    ax = axes[0, 1]
    pen = [r for r in P._rows("part1_penetration_hard-clutter.csv") if r.get("status") == "ok"]
    impact = P._impact_pen("hard-clutter")
    cells = [("mujoco", "dt_s", 1e-2, "10 ms"), ("icf", "dt_s", 1e-2, "10 ms"), ("mujoco-adaptive", "accuracy", 1e-2, "ε=1e-2"), ("icf-adaptive", "accuracy", 1e-2, "ε=1e-2"),
             ("mujoco", "dt_s", 2e-3, "2 ms"), ("icf", "dt_s", 2e-3, "2 ms")]
    xs, hs, cs, labs, ej = [], [], [], [], []
    for i, (arm, col, val, lab) in enumerate(cells):
        r = next((r for r in pen if _is(r, arm, col, val)), None)
        if r is None:
            continue
        xs.append(i); hs.append(r["pen_max_m"] / impact); cs.append(STYLE[arm]["color"]); labs.append(f"{SHORT[arm]}\n{lab}"); ej.append(r["out_of_bin_frac"])
    bars = ax.bar(xs, hs, color=cs, alpha=0.85)
    for b, e, h in zip(bars, ej, hs):
        ax.annotate(f"ejects {e*100:.1f} %" if e > 0 else "", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=7, color="k")
    ax.axhline(1.0, color="k", lw=0.8, ls=":"); 
    ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7); ax.set_yscale("log")
    ax.set_ylabel("Max penetration / impact depth")
    ax.set_title("(b)", fontsize=10, loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # (c) actuated stability + box lift map (MuJoCo) vs ICF
    ax = axes[1, 0]
    act = P._rows("part1_actuated.csv")
    kps = sorted({r["kp"] for r in act})
    cols = [("dt_s", 1e-2, "10 ms"), ("dt_s", 5e-3, "5 ms"), ("dt_s", 2e-3, "2 ms"), ("dt_s", 1e-3, "1 ms"), ("accuracy", 1e-1, "ε=0.1"), ("accuracy", 1e-2, "ε=1e-2"), ("accuracy", 1e-3, "ε=1e-3"), ("accuracy", 1e-4, "ε=1e-4")]
    grid = np.full((len(kps), len(cols)), np.nan); unstable = np.zeros_like(grid, dtype=bool); icf_max = 0.0
    for i, kp in enumerate(kps):
        for j, (col, val, _) in enumerate(cols):
            arm = "mujoco" if col == "dt_s" else "mujoco-adaptive"
            r = next((r for r in act if _is(r, arm, col, val) and abs(r["kp"] - kp) < 1e-9), None)
            if r is None:
                continue
            if r.get("unstable") in (True, "True"):
                unstable[i, j] = True
            else:
                grid[i, j] = max(r["box_lift_max_m"] * 1e3, 1e-3)
            arm_i = "icf" if col == "dt_s" else "icf-adaptive"
            ri = next((r for r in act if _is(r, arm_i, col, val) and abs(r["kp"] - kp) < 1e-9), None)
            if ri is not None and ri.get("unstable") not in (True, "True"):
                icf_max = max(icf_max, ri["box_lift_max_m"] * 1e3)
    im = ax.imshow(np.log10(grid), cmap="Reds", aspect="auto", vmin=-0.5, vmax=2.5)
    for i in range(len(kps)):
        for j in range(len(cols)):
            if unstable[i, j]:
                ax.text(j, i, "✗", ha="center", va="center", fontsize=13, color="k", fontweight="bold")
            elif not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0f}" if grid[i, j] >= 1 else f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=7, color="k")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2] for c in cols], fontsize=7); ax.set_yticks(range(len(kps))); ax.set_yticklabels([f"{k:.0e}" for k in kps], fontsize=7)
    ax.set_xlabel("MuJoCo: δt  |  ε"); ax.set_ylabel("K_p (N/m)")
    cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("box lift, log10(mm)", fontsize=7)
    ax.set_title("(c)", fontsize=10, loc="left")
    # (d) hidden chatter at K_p = 1e5
    ax = axes[1, 1]
    xt = list(range(len(cols)))
    for arm_f, arm_e in (("icf", "icf-adaptive"), ("mujoco", "mujoco-adaptive")):
        ys, xs_, bad = [], [], []
        for j, (col, val, _) in enumerate(cols):
            arm = arm_f if col == "dt_s" else arm_e
            r = next((r for r in act if _is(r, arm, col, val) and abs(r["kp"] - 1e5) < 1e-9), None)
            if r is None:
                continue
            if r.get("unstable") in (True, "True"):
                bad.append(j); continue
            xs_.append(j); ys.append(max(r["rel_vx_rms_m_s"], 1e-4))
        st = dict(STYLE[arm_f]); st["label"] = SHORT[arm_f]; st["ls"] = "-"
        ax.plot(xs_, ys, ms=6, **st)
        for j in bad:
            ax.text(j, 1.5, "✗", ha="center", color=STYLE[arm_f]["color"], fontsize=12, fontweight="bold")
    ax.axvline(3.5, color="k", lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_xticks(xt); ax.set_xticklabels([c[2] for c in cols], fontsize=7)
    ax.set_ylabel("Tip–box relative velocity (m/s)")
    ax.set_title("(d)", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7, loc="lower right")
    P._save(fig, "story_step")


def story_cost() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    # (a) cost at the learner's step vs cheapest artifact-free
    ax = axes[0]
    pen = [r for r in P._rows("part1_penetration_hard-clutter.csv") if r.get("status") == "ok"]
    impact = P._impact_pen("hard-clutter")
    def wall(r):
        return r["wall_ms_per_boundary"] / 1e3 / r["dt_outer_s"]  # s per simulated second
    def free(r):
        return r["pen_max_m"] <= impact and r["out_of_bin_frac"] == 0
    xs, labels = [], []
    for i, arm in enumerate(ARMS):
        rs = [r for r in pen if r["arm"] == arm]
        coarse = next((r for r in rs if (_is(r, arm, "dt_s", 1e-2) or _is(r, arm, "accuracy", 1e-1))), None)
        cheapest = min((r for r in rs if free(r)), key=wall, default=None)
        x0 = i * 3
        if coarse:
            ax.bar(x0, wall(coarse), color=STYLE[arm]["color"], alpha=0.35, hatch="//", edgecolor=STYLE[arm]["color"])
            ax.text(x0, wall(coarse) * 1.1, f"{P._knob_label(coarse)}\n{'artifact' if not free(coarse) else 'ok'}", ha="center", fontsize=6.5)
        if cheapest:
            ax.bar(x0 + 1, wall(cheapest), color=STYLE[arm]["color"])
            ax.text(x0 + 1, wall(cheapest) * 1.1, f"{P._knob_label(cheapest)}\n{wall(cheapest):.2f} s", ha="center", fontsize=6.5)
        xs.append(x0 + 0.5); labels.append(SHORT[arm])
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.set_yscale("log"); ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.set_ylabel("Wall Time (s)")
    ax.set_title("(a)", fontsize=10, loc="left")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    # (b) cumulative wall along a drop
    ax = axes[1]
    rows = P._rows("part1_realtime_trace_hard-clutter_n64.csv")
    series = {}
    for r in rows:
        key = (r["arm"], r["accuracy"] if r["accuracy"] != "" else r["dt_s"])
        series.setdefault(key, []).append((r["t_s"], r["wall_ms"], r["iters"]))
    want = [("icf", 1e-3), ("icf-adaptive", 1e-2), ("mujoco", 1e-3), ("mujoco-adaptive", 1e-3)]
    for arm, knob in want:
        pts = sorted(series.get((arm, knob), []))
        if not pts:
            continue
        t = [p[0] for p in pts]; cum = np.cumsum([p[1] for p in pts]) / 1e3
        st = dict(STYLE[arm]); st["marker"] = None; st["label"] = f"{SHORT[arm]} {P._knob_label({'dt_s': knob} if 'adaptive' not in arm else {'accuracy': knob})}"
        ax.plot(t, cum, lw=1.8, **st)
    ax.axvspan(0, 1.0, color="gray", alpha=0.12); 
    ax.set_xlabel("Simulated time (s)"); ax.set_ylabel("Cumulative Wall Time (s)")
    ax.set_title("(b)", fontsize=10, loc="left")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    P._save(fig, "story_cost")


def story_convergence() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    ax = axes[0]
    rows = [r for r in P._rows("part1_ball_energy.csv") if r["status"] == "ok" and r["dt_s"] != ""]
    for arm in ("icf", "mujoco"):
        pts = sorted((r["dt_s"], max(abs(r["energy_change_pct"]), 1e-3)) for r in rows if r["arm"] == arm)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ms=5, **STYLE[arm])
    d = np.array([2e-4, 1e-5]); ax.plot(d, 100 * d / 1e-4 * 0.3, color="gray", ls=":", label="O(δt)")
    ax.invert_xaxis(); ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Time Step (s)"); ax.set_ylabel("Change in Energy (%)")
    ax.set_title("(a)", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    ax = axes[1]
    rows = [r for r in P._rows("part1_consistency_soft-clutter.csv") if r.get("wall_s_per_sim_s", "") != ""]
    for arm in STYLE:
        pts = sorted((r["wall_s_per_sim_s"], r["dev_mean_m"] * 1e3, P._knob_label(r), r.get("dt_s", "") != "" and abs(r["dt_s"] - 1e-4) < 1e-12) for r in rows if r["arm"] == arm)
        if not pts:
            continue
        ax.plot([p[0] for p in pts if not p[3]], [p[1] for p in pts if not p[3]], ms=5, **STYLE[arm])
        for x, y, lab, fl in pts:
            if fl:
                ax.plot([x], [y], marker=STYLE[arm]["marker"], mfc="none", color=STYLE[arm]["color"], ls="none", ms=6); ax.axhline(y, color=STYLE[arm]["color"], lw=0.6, ls=":", alpha=0.6); lab = "floor"
            dy = 4 if not arm.endswith("adaptive") else -9  # EC labels below: EC points can land on fixed-arm points
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(4, dy), fontsize=5.5, color=STYLE[arm]["color"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Wall Time (s)"); ax.set_ylabel("Deviation (mm)")
    ax.set_title("(b)", fontsize=10, loc="left")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    P._save(fig, "story_convergence")


if __name__ == "__main__":
    story_step()
    story_cost()
    story_convergence()
