# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Throughput in the actuated regime: the PD-gantry push scene
(scripts/scenes/actuated_press.py) at 64..4096 heterogeneous worlds.

The regime of robot learning: stiff PD gains against stiff contact, many
worlds that do not move in lockstep. Per world the push speed, the start
delay and the gain K_p are drawn from fixed ranges (seeded), so the
impacts of different worlds fall at different times and a batch pays for
its stragglers the way a training batch does. Arms: both error-controlled
solvers at eps = 1e-3 and both fixed solvers at dt = 1 ms (artifact-free
in part1_actuated.py at K_p <= 1e5). Recorded per configuration: wall per
boundary (median over boundaries after the first two), inner attempts per
world per boundary (IterationTracker), exhausted and unstable fractions.

    uv run python -m scripts.bench.benchmarks.part1_actuated_scaling
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import warp as wp

from scripts.bench.four_arms import ExhaustionTracker, IterationTracker, _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive
from scripts.scenes.actuated_press import TIP_MASS, ActuatedSpec, build, program

BOUNDARY_S = 0.01
NS = [64, 256, 1024, 4096]
KP_RANGE = (1e4, 1e5)
SPEED_RANGE = (0.15, 0.30)
DELAY_RANGE = (0.0, 0.3)
SEED = 42
MAX_SUBSTEPS = 65536


def _arm(model, spec, backend, kind, knob):
    if backend == "icf":
        return _make_icf(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, spec.icf) if kind == "fixed" else _make_icf_adaptive(model, knob, BOUNDARY_S, spec.icf, MAX_SUBSTEPS)
    return _make_mujoco(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, spec.mujoco_solref) if kind == "fixed" else _make_mujoco_adaptive(model, knob, BOUNDARY_S, MAX_SUBSTEPS, spec.mujoco_solref)


def _run(backend, kind, knob, n):
    rng = np.random.default_rng(SEED)
    kps = np.exp(rng.uniform(math.log(KP_RANGE[0]), math.log(KP_RANGE[1]), n))
    speeds = rng.uniform(*SPEED_RANGE, n)
    delays = rng.uniform(*DELAY_RANGE, n)
    spec = ActuatedSpec(k=1e5, kp=float(KP_RANGE[1]), slide_speed=float(SPEED_RANGE[0]))  # template; per-world gains below
    model = build(n, spec)
    # per-world PD gains on the two prismatic joints (last two DOFs of each world's block)
    ndof = model.joint_dof_count // n
    ke = model.joint_target_ke.numpy().reshape(n, ndof); kd = model.joint_target_kd.numpy().reshape(n, ndof)
    ke[:, -2:] = kps[:, None]; kd[:, -2:] = (2.0 * np.sqrt(kps * TIP_MASS))[:, None]
    model.joint_target_ke.assign(ke.reshape(-1)); model.joint_target_kd.assign(kd.reshape(-1))
    arm = _arm(model, spec, backend, kind, knob)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    tq = ctrl.joint_target_q.numpy(); stride = tq.shape[0] // n; tq = tq.reshape(n, stride)
    horizon = float(DELAY_RANGE[1] + ActuatedSpec(1e5, 1e5, SPEED_RANGE[0]).horizon_s)
    nb = int(round(horizon / BOUNDARY_S))
    ex = ExhaustionTracker(arm) if kind == "adaptive" else None
    it = IterationTracker(arm)
    walls = []; unstable = False
    specs = [ActuatedSpec(k=1e5, kp=float(kps[w]), slide_speed=float(speeds[w])) for w in range(n)]
    for b in range(nb):
        t = b * BOUNDARY_S
        for w in range(n):
            x_t, z_t = program(max(0.0, t - delays[w]), specs[w])
            tq[w, -2] = x_t; tq[w, -1] = z_t
        ctrl.joint_target_q.assign(tq.reshape(-1))
        wp.synchronize(); t0 = time.perf_counter()
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if ex: ex.tick()
        it.tick()
        wp.synchronize(); walls.append(time.perf_counter() - t0)
        if b % 20 == 19:
            bqd = s0.body_qd.numpy().reshape(-1, 6)
            if not np.isfinite(bqd).all() or np.abs(bqd[:, :3]).max() > 10.0:
                unstable = True
    walls = np.array(walls[2:])
    # batch iterations per boundary: every iteration is one attempt for every still-active world,
    # so this is the number of inner steps the batch pays for (its slowest world's count)
    attempts = it.total() / nb if kind == "adaptive" else BOUNDARY_S / knob
    return {"wall_ms_per_boundary": float(np.median(walls) * 1e3), "wall_ms_per_boundary_p90": float(np.percentile(walls, 90) * 1e3),
            "steps_per_boundary": float(attempts), "exhausted_frac": ex.fraction() if ex else 0.0, "unstable": bool(unstable), "boundaries": nb}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ns", nargs="*", type=int, default=NS)
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_actuated_scaling.csv")
    p.add_argument("--single", nargs=4, metavar=("BACKEND", "KIND", "KNOB", "N"), default=None)
    args = p.parse_args()
    if args.single:
        backend, kind, knob, n = args.single
        print("ROW " + json.dumps(_run(backend, kind, float(knob), int(n))), flush=True)
        return 0
    rows = []
    for n in args.ns:
        for backend in ("icf", "mujoco"):
            for kind, knob in (("adaptive", 1e-3), ("fixed", 1e-3)):
                r = subprocess.run([sys.executable, "-m", "scripts.bench.benchmarks.part1_actuated_scaling", "--single", backend, kind, str(knob), str(n)], capture_output=True, text=True, timeout=7200)
                got = None
                for line in r.stdout.splitlines():
                    if line.startswith("ROW "):
                        got = json.loads(line[4:])
                if got is None:
                    print(f"FAIL {backend} {kind} {knob} N={n}: {r.stderr[-400:]}", flush=True); continue
                if "over the scannable budget" in r.stderr:
                    print(f"CONTACT OVERFLOW {backend} {kind} N={n}", flush=True); continue
                arm = backend if kind == "fixed" else f"{backend}-adaptive"
                row = {"arm": arm, "dt_s": knob if kind == "fixed" else "", "accuracy": knob if kind == "adaptive" else "", "n_worlds": n, **got}
                rows.append(row); print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
