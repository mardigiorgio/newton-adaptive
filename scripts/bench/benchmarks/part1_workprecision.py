# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Work-precision, four solver arms, on the well-posed oracle scene.

For each arm and accuracy knob: wall-clock per boundary (timed pass) and
final-state error against the SAME BACKEND's fine-dt reference (n_sub=128
fixed-step), on the sphere-drop + friction-slide scene of the convergence
bench. Precision is self-convergence, per backend: with two different
contact models there is no common ground truth, and pretending one
backend's limit is "the truth" for the other would smuggle the model-bias
question (measured separately by the convergence bench's analytic
stopping-distance oracle) into the work-precision plot. One subprocess
per configuration.

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_workprecision
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import warp as wp

from scripts.bench.benchmarks.part1_convergence import DT_OUTER, build_simple_model
from scripts.bench.four_arms import make_arm

FIXED_SUBS = [1, 2, 4, 8, 16]
ADAPTIVE_TOLS = [1e-2, 1e-3, 1e-4]
REFERENCE_N_SUB = 128
BACKEND_OF = {
    "mujoco": "mujoco",
    "mujoco-adaptive": "mujoco",
    "icf": "icf",
    "icf-adaptive": "icf",
}


def _final_and_wall(arm_name: str, knob, n: int, horizon: float) -> tuple[np.ndarray, float]:
    kwargs = {"n_sub": knob} if arm_name in ("mujoco", "icf") else {"tol": knob}
    model = build_simple_model(n)
    arm = make_arm(model, arm_name, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    boundaries = int(round(horizon / DT_OUTER))
    # warm once (compile), rebuild states for the measured trajectory
    s0, s1 = arm.boundary(s0, s1, ctrl)
    model2 = build_simple_model(n)
    arm2 = make_arm(model2, arm_name, **kwargs)
    s0, s1, ctrl = model2.state(), model2.state(), model2.control()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(boundaries):
        s0, s1 = arm2.boundary(s0, s1, ctrl)
    wp.synchronize()
    wall_ms = (time.perf_counter() - t0) / boundaries * 1e3
    return s0.body_q.numpy().reshape(-1, 7)[:, :3].copy(), wall_ms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_workprecision.csv")
    p.add_argument("--single", nargs=3, metavar=("ARM", "KNOB", "OUT_NPY"), default=None)
    args = p.parse_args()

    if args.single is not None:
        arm_name, knob_s, out_npy = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        final, wall_ms = _final_and_wall(arm_name, knob, args.n, args.horizon)
        np.save(out_npy, final)
        print("WALL " + json.dumps({"wall_ms": wall_ms}), flush=True)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(args.out), "part1_wp_tmp")
    os.makedirs(tmp, exist_ok=True)

    def run_cfg(arm_name: str, knob) -> tuple[np.ndarray | None, float]:
        out_npy = os.path.join(tmp, f"{arm_name}_{knob}.npy")
        r = subprocess.run(
            [
                sys.executable, "-m", "scripts.bench.benchmarks.part1_workprecision",
                "--single", arm_name, str(knob), out_npy,
                "--n", str(args.n), "--horizon", str(args.horizon),
            ],
            capture_output=True, text=True,
        )
        wall = float("nan")
        for line in r.stdout.splitlines():
            if line.startswith("WALL "):
                wall = json.loads(line[5:])["wall_ms"]
        if r.returncode != 0:
            print(f"CONFIG FAILED {arm_name} knob={knob}:\n{r.stderr[-600:]}", flush=True)
            return None, wall
        return np.load(out_npy), wall

    refs = {}
    for backend in ("mujoco", "icf"):
        ref, _ = run_cfg(backend, REFERENCE_N_SUB)
        refs[backend] = ref

    rows = []
    for arm_name in BACKEND_OF:
        knobs = FIXED_SUBS if arm_name in ("mujoco", "icf") else ADAPTIVE_TOLS
        for knob in knobs:
            final, wall_ms = run_cfg(arm_name, knob)
            ref = refs[BACKEND_OF[arm_name]]
            if final is None or ref is None:
                continue
            rows.append({
                "arm": arm_name,
                "knob": knob,
                "wall_ms_per_boundary": round(wall_ms, 4),
                "err_vs_ref_m": float(np.abs(final - ref).max()),
            })
            print(rows[-1], flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
