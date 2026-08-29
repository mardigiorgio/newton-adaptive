# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Headline numbers of Part 1, read from the committed CSVs, printed as
markdown so PART1.md and the notebook quote them instead of retyping.

    uv run python scripts/bench/part1_keynumbers.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import part1_plots as P  # noqa: E402

ARMS = ("mujoco", "mujoco-adaptive", "icf", "icf-adaptive")


def _lab(r):
    return P._knob_label(r)


def penetration(scene):
    rows = [r for r in P._rows(f"part1_penetration_{scene}.csv") if r.get("status") == "ok"]
    imp = P._impact_pen(scene); rest = P._static_pen(scene)
    print(f"\n### {scene}: penetration, 64 scenes (impact depth {imp*1e3:.2f} mm, resting depth {rest*1e6:.0f} µm)\n")
    print("| arm | setting | mean µm | max mm | p95 µm | ejected | wall s/sim-s | artifact-free |\n|---|---|---|---|---|---|---|---|")
    cheapest = {}
    for arm in ARMS:
        for r in sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["wall_ms_per_boundary"]):
            free = r["pen_max_m"] <= imp and r["out_of_bin_frac"] == 0
            wall = r["wall_ms_per_boundary"] / 1e3 / r["dt_outer_s"]
            if free and arm not in cheapest:
                cheapest[arm] = (_lab(r), wall)
            print(f"| {arm} | {_lab(r)} | {r['pen_mean_m']*1e6:.1f} | {r['pen_max_m']*1e3:.2f} | {r['pen_p95_m']*1e6:.1f} | {r['out_of_bin_frac']*100:.1f} % | {wall:.2f} | {'yes' if free else 'no'} |")
    print("\nCheapest artifact-free: " + "; ".join(f"{a} {v[0]} ({v[1]:.2f} s)" for a, v in cheapest.items()))


def workprecision(scene):
    for n in (1, 1024):
        rows = P._wp_rows(scene, n)
        if not rows:
            continue
        print(f"\n### {scene}: work-precision, {n} scene(s) (wall s per simulated second)\n")
        for arm in ARMS:
            pts = sorted((r.get("accuracy") if r.get("accuracy", "") != "" else r["dt_s"], r["wall_s_per_sim_s"], r.get("status", "")) for r in rows if r["arm"] == arm)
            print(f"- {arm}: " + ", ".join(f"{_lab({'accuracy': k} if 'adaptive' in arm else {'dt_s': k})} {w:.3g}{'' if st in ('ok', '') else ' [' + st + ']'}" for k, w, st in pts))


def ball():
    rows = [r for r in P._rows("part1_ball_energy.csv")]
    print("\n### ball: energy change after 10 s (last apex), rebounds\n")
    for arm in ARMS:
        pts = [r for r in rows if r["arm"] == arm]
        print(f"- {arm}: " + ", ".join(f"{_lab(r)} {r['energy_change_pct']:+.2f} % ({int(r['bounces'])}{'' if r['status']=='ok' else ', ' + r['status']})" for r in pts))


def consistency(scene):
    rows = [r for r in P._rows(f"part1_consistency_{scene}.csv")]
    if not rows:
        return
    print(f"\n### {scene}: measured deviation after 0.1 s (mm) @ wall s/sim-s\n")
    for arm in ARMS:
        pts = [r for r in rows if r["arm"] == arm]
        print(f"- {arm}: " + ", ".join(f"{_lab(r)} {r['dev_mean_m']*1e3:.3g} @ {r['wall_s_per_sim_s']:.2f}" for r in pts))


def scaling(scene):
    rows = [r for r in P._rows(f"part1_scaling_{scene}.csv")]
    if not rows:
        return
    print(f"\n### {scene}: wall ms per boundary vs worlds (median)\n")
    for arm in ARMS:
        pts = sorted((r["n_worlds"], r["wall_ms_median"]) for r in rows if r["arm"] == arm)
        print(f"- {arm}: " + ", ".join(f"{int(n)}: {w:.3g}" for n, w in pts) + (f" → per world at {int(pts[-1][0])}: {pts[-1][1]/pts[-1][0]*1e3:.0f} µs" if pts else ""))


def stiffness():
    rows = [r for r in P._rows("part1_stiffness_sweep.csv") if r.get("finite") in (True, "True")]
    print("\n### realized stiffness: penetration / (m g/k) vs requested k\n")
    keys = sorted({(r["arm"], _lab(r)) for r in rows})
    for arm, lab in keys:
        pts = sorted((r["k_N_per_m"], r["ratio"]) for r in rows if r["arm"] == arm and _lab(r) == lab)
        print(f"- {arm} {lab}: " + ", ".join(f"{k:.0e}: {v:.2f}" for k, v in pts))


def actuated():
    rows = P._rows("part1_actuated.csv")
    print("\n### actuated push (k = 1e5, 300 mm/s): unstable cells, box lift (mm), tip–box rel. velocity (m/s)\n")
    bad = [(r["arm"], _lab(r), r["kp"]) for r in rows if r.get("unstable") in (True, "True")]
    print("- unstable: " + ("; ".join(f"{a} {l} K_p={k:.0e}" for a, l, k in bad) if bad else "none"))
    for arm in ARMS:
        pts = [r for r in rows if r["arm"] == arm and r.get("unstable") not in (True, "True")]
        lifts = [r["box_lift_max_m"] * 1e3 for r in pts]
        print(f"- {arm}: lift {min(lifts):.2f}…{max(lifts):.2f} mm; rel. vel at K_p=1e5: " + ", ".join(f"{_lab(r)} {r['rel_vx_rms_m_s']:.3f}" for r in pts if abs(r['kp'] - 1e5) < 1))
    rows = P._rows("part1_actuated_scaling.csv")
    if rows:
        print("\n### actuated regime throughput (heterogeneous worlds)\n")
        for arm in ARMS:
            pts = sorted((r["n_worlds"], r["wall_ms_per_boundary"], r["steps_per_boundary"], r.get("unstable")) for r in rows if r["arm"] == arm)
            print(f"- {arm}: " + ", ".join(f"N={int(n)}: {w:.3g} ms/boundary, {s:.3g} steps{'' if u in (False, 'False') else ' UNSTABLE'}" for n, w, s, u in pts))


if __name__ == "__main__":
    for sc in ("hard-clutter", "soft-clutter"):
        penetration(sc); workprecision(sc); consistency(sc); scaling(sc)
    ball(); stiffness(); actuated()
