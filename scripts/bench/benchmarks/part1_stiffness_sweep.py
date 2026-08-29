# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resting penetration vs requested contact stiffness (the protocol of ICF,
Castro et al. T-RO 2025, Fig. 18): one 65 g sphere resting on the plane,
k from 1e3 to 1e8 N/m, each arm at fixed dt = 10 ms and 1 ms and under
error control at eps = 1e-3. Reported as penetration / (m*g/k): 1 means the
arm realizes the requested model at rest. MuJoCo's solref is scaled from
the calibrated tau at 1e5 as tau ~ 1/sqrt(k); refsafe clamps tau >= 2 dt,
so its representable stiffness is capped at fixed dt (MuJoCo docs).

    uv run python -m scripts.bench.benchmarks.part1_stiffness_sweep
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys

import numpy as np
import warp as wp

import newton

from scripts.bench.four_arms import _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive
from scripts.scenes.cenic_scenes import MUJOCO_TAU_K1E5

KS = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
R = 0.025
MASS = 1000.0 * 4.0 / 3.0 * math.pi * R**3
BOUNDARY_S = 0.01


def _model(k):
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=k, kd=0.02 * k, mu=0.5, margin=0.0, density=1000.0)
    b = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, R + 1e-4), q=wp.quat_identity()))
    t.add_shape_sphere(b, radius=R, cfg=cfg)
    bb = newton.ModelBuilder(); bb.replicate(t, 2); bb.add_ground_plane()
    return bb.finalize()


def _run(backend, kind, knob, k):
    m = _model(k)
    icf = {"contact_stiffness": k, "contact_stiction_tolerance": 1e-4}
    solref = (MUJOCO_TAU_K1E5 * math.sqrt(1e5 / k), 1.0)
    if backend == "icf":
        a = _make_icf(m, int(round(BOUNDARY_S / knob)), BOUNDARY_S, icf) if kind == "fixed" else _make_icf_adaptive(m, knob, BOUNDARY_S, icf, 4096)
    else:
        a = _make_mujoco(m, int(round(BOUNDARY_S / knob)), BOUNDARY_S, solref) if kind == "fixed" else _make_mujoco_adaptive(m, knob, BOUNDARY_S, 4096, solref)
    s0, s1, c = m.state(), m.state(), m.control()
    for _ in range(int(round(3.0 / BOUNDARY_S))):
        s0, s1 = a.boundary(s0, s1, c)
    z = s0.body_q.numpy().reshape(-1, 7)[:, 2]
    pen = float(R - z.mean())
    return {"pen_m": pen, "static_m": MASS * 9.81 / k, "ratio": pen / (MASS * 9.81 / k), "finite": bool(np.isfinite(z).all()),
            "mujoco_tau_s": solref[0] if backend == "mujoco" else ""}


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == "--single":
        backend, kind, knob, k = sys.argv[2], sys.argv[3], float(sys.argv[4].split(",")[0]), float(sys.argv[4].split(",")[1])
        print("ROW " + json.dumps(_run(backend, kind, knob, k)), flush=True)
        return 0
    out = "scripts/bench/results/part1_stiffness_sweep.csv"
    rows = []
    for backend in ("icf", "mujoco"):
        for kind, knob in (("fixed", 1e-2), ("fixed", 1e-3), ("adaptive", 1e-3), ("adaptive", 1e-5)):
            for k in KS:
                r = subprocess.run([sys.executable, "-m", "scripts.bench.benchmarks.part1_stiffness_sweep", "--single", backend, kind, f"{knob},{k}"],
                                   capture_output=True, text=True, timeout=1800)
                got = None
                for line in r.stdout.splitlines():
                    if line.startswith("ROW "):
                        got = json.loads(line[4:])
                if got is None:
                    print(f"FAIL {backend} {kind} {knob} k={k:g}: {r.stderr[-300:]}", flush=True); continue
                arm = backend if kind == "fixed" else f"{backend}-adaptive"
                row = {"arm": arm, "dt_s": knob if kind == "fixed" else "", "accuracy": knob if kind == "adaptive" else "", "k_N_per_m": k, **got}
                rows.append(row); print(row, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
