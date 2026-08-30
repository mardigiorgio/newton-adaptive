# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sensitivity of the MuJoCo arms' hard-clutter penetration to the friction
cone: the benches run MuJoCo's default (pyramidal, impratio 1); the training
scenes run elliptic with impratio 10. Same protocol as part1_penetration
(64 scenes, 2 s from t = 0, seed 42, margin 0), MuJoCo fixed 10 ms and 1 ms
and error control 1e-2 and 1e-3, with the cone forced to elliptic/impratio 10
through the solver constructor. The pyramidal rows come from the bench's CSV.

    uv run python -m scripts.bench.probe_mujoco_cone
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import newton

PASSES = [("mujoco", 10), ("mujoco", 100), ("mujoco-adaptive", 1e-2), ("mujoco-adaptive", 1e-3)]
BENCH_CSV = "scripts/bench/results/part1_penetration_hard-clutter.csv"
OUT = "scripts/bench/results/tables/mujoco_cone_probe.md"
COLS = ["pen_mean_m", "pen_max_m", "out_of_bin_frac"]


def _single(arm_name: str, knob: float) -> dict:
    orig = newton.solvers.SolverMuJoCo.__init__

    def init(self, *a, **k):
        if k.get("cone") is None:
            k["cone"] = "elliptic"
        if k.get("impratio") is None:
            k["impratio"] = 10.0
        orig(self, *a, **k)

    newton.solvers.SolverMuJoCo.__init__ = init
    import scripts.bench.benchmarks.part1_penetration as P

    kn = int(knob) if arm_name == "mujoco" else float(knob)
    return P._run_pass("hard-clutter", arm_name, kn, 64, 2.0, 0.0, 42, "geometry", 0.0)


def _bench_rows() -> dict:
    rows = {}
    if not os.path.exists(BENCH_CSV):
        return rows
    with open(BENCH_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["arm"], r.get("dt_s") or r.get("accuracy"))
            rows[key] = r
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", nargs=2, metavar=("ARM", "KNOB"), default=None)
    args = p.parse_args()
    if args.single:
        print("ROW " + json.dumps(_single(args.single[0], float(args.single[1]))), flush=True)
        return 0
    bench = _bench_rows()
    lines = [
        "# MuJoCo friction-cone sensitivity, hard clutter (64 scenes, 2 s, seed 42, margin 0)",
        "",
        "| arm | knob | cone | " + " | ".join(COLS) + " |",
        "|---|---|---|" + "---|" * len(COLS),
    ]
    for arm, knob in PASSES:
        knob_s = f"{0.1 / knob:g}" if arm == "mujoco" else f"{knob:g}"
        for key, r in bench.items():
            if key[0] == arm and key[1] and abs(float(key[1]) - float(knob_s)) < 1e-12:
                lines.append(f"| {arm} | {knob_s} | pyramidal, impratio 1 | " + " | ".join(f"{float(r[c]):.3e}" for c in COLS) + " |")
        res = subprocess.run(
            [sys.executable, "-m", "scripts.bench.probe_mujoco_cone", "--single", arm, str(knob)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        row = None
        for line in res.stdout.splitlines():
            if line.startswith("ROW "):
                row = json.loads(line[4:])
        if row is None:
            lines.append(f"| {arm} | {knob_s} | elliptic, impratio 10 | FAIL | | |")
            print(f"FAIL {arm} {knob}: {res.stderr[-400:]}", flush=True)
            continue
        lines.append(f"| {arm} | {knob_s} | elliptic, impratio 10 | " + " | ".join(f"{float(row[c]):.3e}" for c in COLS) + " |")
        print(lines[-1], flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
