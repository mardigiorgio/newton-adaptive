# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Emit the Part-1 results tables (Markdown) from the committed CSVs, so the
numbers in the document are generated, never retyped.

    uv run python scripts/bench/part1_results_md.py
Writes scripts/bench/results/tables/results_tables.md
"""

from __future__ import annotations

import csv
import os

RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "tables", "results_tables.md")
ARMS = [("icf-adaptive", "ICF error control"), ("mujoco-adaptive", "MuJoCo error control"),
        ("icf", "ICF fixed step"), ("mujoco", "MuJoCo fixed step")]
EPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
DTS = [1e-2, 5e-3, 2e-3, 1e-3]
NS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


def rows(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    out = []
    for r in csv.DictReader(open(p)):
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = v
        out.append(d)
    return out


def g(v, nd=3):
    return "—" if v in ("", None) else f"{v:.{nd}g}"


def workprecision():
    out = ["## Work-precision — wall time [s] per simulated second (δt_max = 10 ms)\n",
           "| scene, N | arm | " + " | ".join(f"ε = 10^{int(round(__import__('math').log10(e)))}" for e in EPS) + " |",
           "|---|---|" + "---|" * len(EPS)]
    for scene in ("soft-clutter", "hard-clutter"):
        for n in (1, 1024):
            rs = rows(f"part1_workprecision_{scene}_n{n}.csv")
            if not rs:
                continue
            for arm, name in ARMS[:2]:
                cells = []
                for e in EPS:
                    r = next((r for r in rs if r["arm"] == arm and r["accuracy"] == e), None)
                    cells.append("—" if r is None else (g(r["wall_s_per_sim_s"]) if r["status"] == "ok" else r["status"]))
                out.append(f"| {scene.split('-')[0]}, {n} | {name} | " + " | ".join(cells) + " |")
    out += ["", "| scene, N | arm | " + " | ".join(f"δt = {d * 1e3:g} ms" for d in DTS) + " |", "|---|---|" + "---|" * len(DTS)]
    for scene in ("soft-clutter", "hard-clutter"):
        for n in (1, 1024):
            rs = rows(f"part1_workprecision_{scene}_n{n}.csv")
            if not rs:
                continue
            for arm, name in ARMS[2:]:
                cells = []
                for d in DTS:
                    r = next((r for r in rs if r["arm"] == arm and r["dt_s"] == d), None)
                    cells.append("—" if r is None else g(r["wall_s_per_sim_s"]))
                out.append(f"| {scene.split('-')[0]}, {n} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def penetration():
    out = ["\n## Penetration and ejections — 64 worlds, 200 boundaries\n",
           "| scene | arm | setting | mean [µm] | max [mm] | p95 [µm] | ejected | wall/boundary [ms] |", "|---|---|---|---|---|---|---|---|"]
    for scene in ("soft-clutter", "hard-clutter"):
        rs = rows(f"part1_penetration_{scene}.csv")
        for arm, name in ARMS:
            for r in [r for r in rs if r["arm"] == arm]:
                k = f"ε = {r['accuracy']:.0e}" if r["accuracy"] != "" else f"δt = {r['dt_s'] * 1e3:g} ms"
                out.append(f"| {scene.split('-')[0]} | {name} | {k} | {r['pen_mean_m'] * 1e6:.2f} | {r['pen_max_m'] * 1e3:.3f} | {r['pen_p95_m'] * 1e6:.1f} | {100 * r['out_of_bin_frac']:.1f}% | {r['wall_ms_per_boundary']:.2f} |")
    return "\n".join(out) + "\n"


def scaling():
    out = ["\n## Wall time per 10 ms boundary [ms] vs parallel worlds — median of 3 runs (spread in brackets)\n"]
    for scene in ("soft-clutter", "hard-clutter"):
        rs = rows(f"part1_scaling_{scene}.csv")
        if not rs:
            continue
        out += [f"**{scene}**", "", "| arm | " + " | ".join(f"2^{n.bit_length() - 1}" for n in NS) + " |", "|---|" + "---|" * len(NS)]
        for arm, name in ARMS:
            cells = []
            for n in NS:
                r = next((r for r in rs if r["arm"] == arm and r["n_worlds"] == n), None)
                if r is None:
                    cells.append("—")
                else:
                    lo, hi = r.get("wall_ms_trial_min", ""), r.get("wall_ms_trial_max", "")
                    cells.append(f"{g(r['wall_ms_median'])}" + (f" [{g(lo, 2)}–{g(hi, 2)}]" if lo != "" else ""))
            out.append(f"| {name} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out) + "\n"


def ball():
    rs = rows("part1_ball_energy.csv")
    if not rs:
        return ""
    out = ["\n## Bouncing ball — energy change after 10 s and rebounds\n",
           "| arm | setting | energy change [%] | rebounds | status |", "|---|---|---|---|---|"]
    for arm, name in [("icf", "ICF fixed step"), ("mujoco", "MuJoCo fixed step"), ("icf-adaptive", "ICF error control"), ("mujoco-adaptive", "MuJoCo error control")]:
        for r in [r for r in rs if r["arm"] == arm]:
            k = f"δt = {r['dt_s']:g} s" if r["dt_s"] != "" else f"ε = {r['accuracy']:.0e}"
            out.append(f"| {name} | {k} | {r['energy_change_pct']:+.2f} | {int(r['bounces']) if r.get('bounces', '') != '' else '—'} | {r['status']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md = "# Part-1 results tables (generated from the CSVs — do not edit)\n\n" + workprecision() + penetration() + scaling() + ball()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(md)
    print(f"wrote {OUT} ({len(md.splitlines())} lines)")
