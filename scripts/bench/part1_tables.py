# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Part-1 tables in the CENIC paper's Table I format, from the committed
CSVs: real-time rate (%) and an artifact verdict per accuracy (error
control) and per time step (fixed step), one block per scene, ICF and
MuJoCo side by side. Also the bouncing-ball table (energy change, bounces).

Artifact criterion (stated, since the paper's Table I is visual): any body
ejected from the bin, or max ground penetration above 10x the scene's
single-object static penetration m*g/k -- the compliance the contact model
itself prescribes (6.5 um on hard clutter at k = 1e5 N/m, 0.65 mm on soft
clutter at k = 1e3 N/m, for the 65 g objects) -- over the 64-world run.
Penetration of a few static depths is the model; tens of them is the step.

    uv run python scripts/bench/part1_tables.py
Writes scripts/bench/results/tables/part1_table1.md and .tex
"""

from __future__ import annotations

import csv
import os

RES = os.path.join(os.path.dirname(__file__), "results")
OUT = os.path.join(RES, "tables")
os.makedirs(OUT, exist_ok=True)
import math

OBJECT_MASS_KG = 1000.0 * 4.0 / 3.0 * math.pi * 0.025**3  # water-density 2.5 cm sphere
SCENE_K = {"soft-clutter": 1.0e3, "hard-clutter": 1.0e5}
ARTIFACT_FACTOR = 10.0


def pen_artifact_m(scene: str) -> float:
    return ARTIFACT_FACTOR * OBJECT_MASS_KG * 9.81 / SCENE_K[scene]


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


def _rtr(rows, arm, key, val):
    r = next((r for r in rows if r["arm"] == arm and r[key] == val), None)
    if r is None:
        return "—"
    if r["status"] != "ok":
        return r["status"]
    return f"{100.0 / r['wall_s_per_sim_s']:.0f}%"


def _artifact(rows, arm, key, val, scene):
    r = next((r for r in rows if r["arm"] == arm and r[key] == val), None)
    if r is None:
        return "—"
    bad = (r.get("out_of_bin_frac", 0.0) or 0.0) > 0.0 or r["pen_max_m"] > pen_artifact_m(scene)
    detail = f"pen {r['pen_max_m'] * 1e3:.1f} mm, eject {100 * (r.get('out_of_bin_frac', 0.0) or 0.0):.1f}%"
    return ("Yes" if bad else "No") + f" ({detail})"


def table1() -> tuple[str, str]:
    md, tex = [], []
    for scene, title in (("soft-clutter", "Soft clutter"), ("hard-clutter", "Hard clutter")):
        wp = _rows(f"part1_workprecision_{scene}_n1.csv")
        pen = _rows(f"part1_penetration_{scene}.csv")
        if not wp:
            continue
        md.append(f"\n### {title}  (artifact if max penetration > {pen_artifact_m(scene) * 1e3:.3g} mm = 10× m·g/k, or any ejection)\n")
        md.append("| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |")
        md.append("|---|---|---|---|---|")
        for arm, name in (("icf-adaptive", "ICF"), ("mujoco-adaptive", "MuJoCo")):
            md.append(f"| {name} real-time rate | " + " | ".join(_rtr(wp, arm, "accuracy", e) for e in (1e-1, 1e-2, 1e-3, 1e-4)) + " |")
            md.append(f"| {name} artifacts | " + " | ".join(_artifact(pen, arm, "accuracy", e, scene) for e in (1e-1, 1e-2, 1e-3, 1e-4)) + " |")
        md.append("")
        md.append("| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |")
        md.append("|---|---|---|---|---|")
        for arm, name in (("icf", "ICF"), ("mujoco", "MuJoCo")):
            md.append(f"| {name} real-time rate | " + " | ".join(_rtr(wp, arm, "dt_s", d) for d in (1e-2, 5e-3, 2e-3, 1e-3)) + " |")
            md.append(f"| {name} artifacts | " + " | ".join(_artifact(pen, arm, "dt_s", d, scene) for d in (1e-2, 5e-3, 2e-3, 1e-3)) + " |")
        tex.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{title}}}}}\\\\\\hline")
        tex.append("Accuracy $\\varepsilon_{acc}$ & $10^{-1}$ & $10^{-2}$ & $10^{-3}$ & $10^{-4}$\\\\\\hline")
        for arm, name in (("icf-adaptive", "ICF"), ("mujoco-adaptive", "MuJoCo")):
            tex.append(f"{name} real-time rate & " + " & ".join(_rtr(wp, arm, "accuracy", e) for e in (1e-1, 1e-2, 1e-3, 1e-4)) + "\\\\")
            tex.append(f"{name} artifacts & " + " & ".join(_artifact(pen, arm, "accuracy", e, scene).split(" (")[0] for e in (1e-1, 1e-2, 1e-3, 1e-4)) + "\\\\")
        tex.append("\\hline Time step $\\delta t$ & 10 ms & 5 ms & 2 ms & 1 ms\\\\\\hline")
        for arm, name in (("icf", "ICF"), ("mujoco", "MuJoCo")):
            tex.append(f"{name} real-time rate & " + " & ".join(_rtr(wp, arm, "dt_s", d) for d in (1e-2, 5e-3, 2e-3, 1e-3)) + "\\\\")
            tex.append(f"{name} artifacts & " + " & ".join(_artifact(pen, arm, "dt_s", d, scene).split(" (")[0] for d in (1e-2, 5e-3, 2e-3, 1e-3)) + "\\\\")
        tex.append("\\hline")
    head = ("# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world "
            "penetration run: any ejection, or max penetration > 10× the scene's static penetration m·g/k)\n")
    tex_doc = "\\begin{tabular}{lcccc}\\hline\n" + "\n".join(tex) + "\n\\end{tabular}\n"
    return head + "\n".join(md) + "\n", tex_doc


def fixed_levels_table() -> str:
    """Fixed-step reference levels (wall s per simulated s) at N = 1 and 1024."""
    out = ["\n# Fixed-step reference levels — wall time [s] per simulated second\n",
           "| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |", "|---|---|---|---|---|---|---|"]
    for scene in ("soft-clutter", "hard-clutter"):
        for n in (1, 1024):
            rows = _rows(f"part1_workprecision_{scene}_n{n}.csv")
            if not rows:
                continue
            for arm, name in (("icf", "ICF fixed"), ("mujoco", "MuJoCo fixed")):
                vals = []
                for d in (1e-2, 5e-3, 2e-3, 1e-3):
                    r = next((r for r in rows if r["arm"] == arm and r["dt_s"] == d), None)
                    vals.append("—" if r is None or r["status"] != "ok" else f"{r['wall_s_per_sim_s']:.3g}")
                out.append(f"| {scene} | {name} | {n} | " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def ball_table() -> str:
    rows = _rows("part1_ball_energy.csv")
    if not rows:
        return ""
    out = ["\n# Bouncing ball (Fig. 8 scene): energy change after 10 s and rebounds (paper: 11)\n",
           "| arm | δt or ε_acc | energy change [%] | bounces | status |", "|---|---|---|---|---|"]
    for r in rows:
        knob = f"δt = {r['dt_s']:g} s" if r["dt_s"] != "" else f"ε = {r['accuracy']:g}"
        e = f"{r['energy_change_pct']:+.2f}" if r["energy_change_pct"] != "" else "—"
        b = f"{int(r['bounces'])}" if r.get("bounces", "") != "" else "—"
        out.append(f"| {r['arm']} | {knob} | {e} | {b} | {r['status']} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    md, tex = table1()
    md += fixed_levels_table()
    md += ball_table()
    with open(os.path.join(OUT, "part1_table1.md"), "w") as f:
        f.write(md)
    with open(os.path.join(OUT, "part1_table1.tex"), "w") as f:
        f.write(tex)
    print(md)
