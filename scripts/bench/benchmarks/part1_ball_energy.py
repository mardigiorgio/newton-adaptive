# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""dt -> 0 convergence, CENIC Fig. 8 definition: energy conservation of a
bouncing ball with zero dissipation.

    "Energy conservation error (percent of energy lost after 10 seconds)
    for a 0.1 kg bouncing ball with zero dissipation. The ball is
    relatively soft (contact stiffness 10^3 N/m) and bounces 11 times in
    the 10 second simulation. Potential energy is defined such that total
    energy is zero when the ball is at rest on the ground."

Fixed ICF and fixed MuJoCo sweep the time step (dt = 10 ms / n_sub); the
adaptive arms sweep eps_acc and are reported on a second axis. A
convergent integrator on a conservative model drives the energy error to
zero as dt -> 0; a contact model whose dissipation does not vanish with
dt cannot. Every row states its dt or its accuracy. One subprocess per
configuration.

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_ball_energy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

from scripts.bench.four_arms import ExhaustionTracker, build_model, make_arm
from scripts.scenes.cenic_scenes import DT_OUTER, ball_energy, ball_initial_energy

N_SUB_LADDER = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # dt 10 ms .. 10 us
ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
HORIZON_S = 10.0


MAX_SUBSTEPS = 4096


def _run(arm_name: str, knob) -> dict:
    fixed = arm_name in ("mujoco", "icf")
    kwargs = {"n_sub": knob} if fixed else {"tol": knob, "max_substeps": MAX_SUBSTEPS}
    model = build_model(1, scene="ball")
    arm = make_arm(model, arm_name, scene="ball", **kwargs)
    tracker = ExhaustionTracker(arm) if not fixed else None
    s0, s1, ctrl = model.state(), model.state(), model.control()
    e0 = ball_initial_energy(model)
    # The paper's stated fact for this scene -- "bounces 11 times in the 10
    # second simulation" -- is an independent reference: count rebounds as
    # boundary-sampled upward vertical-velocity sign flips (contact lasts
    # ~pi*sqrt(m/k) = 31 ms, resolvable at 10 ms sampling).
    bounces = 0
    prev_vz = 0.0
    for _ in range(int(round(HORIZON_S / DT_OUTER))):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
        vz = float(s0.joint_qd.numpy()[2])
        if prev_vz < -0.05 and vz > 0.05:
            bounces += 1
        prev_vz = vz
    e_end = ball_energy(model, s0)[0]
    return {
        "energy_change_pct": 100.0 * (e_end - e0) / e0,
        "final_z": float(s0.body_q.numpy().reshape(-1, 7)[0, 2]),
        "bounces": bounces,
        "exhausted_frac": tracker.fraction() if tracker else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_ball_energy.csv")
    p.add_argument("--single", nargs=2, metavar=("ARM", "KNOB"), default=None)
    args = p.parse_args()

    if args.single is not None:
        arm_name, knob_s = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        print("ROW " + json.dumps(_run(arm_name, knob)), flush=True)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    configs = [(a, k) for a in ("icf", "mujoco") for k in N_SUB_LADDER]
    configs += [(a, k) for a in ("icf-adaptive", "mujoco-adaptive") for k in ACCURACIES]
    for arm_name, knob in configs:
        fixed = arm_name in ("mujoco", "icf")
        row = {
            "arm": arm_name,
            "dt_s": DT_OUTER / knob if fixed else "",
            "accuracy": "" if fixed else knob,
            "max_substeps": "" if fixed else MAX_SUBSTEPS,
            "energy_change_pct": "",
            "final_z": "",
            "bounces": "",
            "exhausted_frac": "",
            "status": "ok",
        }
        try:
            r = subprocess.run(
                [sys.executable, "-m", "scripts.bench.benchmarks.part1_ball_energy", "--single", arm_name, str(knob)],
                capture_output=True, text=True, timeout=1500,
            )
            got = None
            if "over the scannable budget" in r.stderr:
                row["status"] = "contact-overflow"
                print(f"CONTACT OVERFLOW {arm_name} {knob}: contacts dropped -- raise the budgets in four_arms.py", flush=True)
                break
            for line in r.stdout.splitlines():
                if line.startswith("ROW "):
                    got = json.loads(line[4:])
            if got is None:
                row["status"] = "fail"
                print(f"FAIL {arm_name} {knob}: {r.stderr[-300:]}", flush=True)
            else:
                row.update(got)
                if got["exhausted_frac"] > 0.0:
                    row["status"] = "budget-exhausted"
        except subprocess.TimeoutExpired:
            row["status"] = "timeout"
        rows.append(row)
        print(row, flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
