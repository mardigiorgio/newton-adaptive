# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penetration vs wall time, four solver arms.

Each configuration (arm x accuracy knob) runs the shared contact scene
twice from identical initial states: a TIMED pass with no readbacks
(wall-clock per outer boundary, synced), and a METRIC pass that reads the
state back every boundary and measures ground-plane penetration
analytically from the state — spheres by center height against their
radius, boxes by their deepest transformed corner. Analytic state-side
penetration is solver-agnostic: it cannot flatter a backend the way a
solver's own contact report can.

Accuracy knobs: fixed arms sweep substeps-per-boundary (dt ladder);
adaptive arms sweep tolerance.

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_penetration --n 64 --steps 200
Emits CSV rows (arm, knob, wall_ms_per_boundary, pen_mean_m, pen_max_m,
pen_p95_m) and a penetration-vs-wall plot.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import warp as wp

from scripts.bench.four_arms import ARMS, build_model, make_arm
from scripts.scenes.contact_objects import BOX_HALF, SPHERE_RADIUS

FIXED_SUBS = [1, 2, 4, 8, 16]
ADAPTIVE_TOLS = [1e-2, 1e-3, 1e-4]

_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float32,
) * BOX_HALF


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate corner offsets v [8,3] by quaternions q [B,4] (x,y,z,w)."""
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    qv = q[:, None, :3]
    uv = np.cross(qv, v[None, :, :])
    uuv = np.cross(qv, uv)
    return v[None, :, :] + 2.0 * (w[:, :, None] * uv + uuv)


def _penetrations(model, state) -> np.ndarray:
    """Per-body ground penetration [m] from body_q; zeros where separated.

    Bodies alternate 9 spheres then 9 boxes per world (the scene's build
    order). The ground plane is z = 0.
    """
    body_q = state.body_q.numpy().reshape(-1, 7)
    n_bodies = body_q.shape[0]
    per_world = 18
    pos = body_q[:, :3]
    quat = body_q[:, 3:]
    is_sphere = (np.arange(n_bodies) % per_world) < 9
    pen = np.zeros(n_bodies, dtype=np.float32)
    pen[is_sphere] = np.maximum(0.0, SPHERE_RADIUS - pos[is_sphere, 2])
    box_idx = ~is_sphere
    corners = _quat_rotate(quat[box_idx], _CORNERS) + pos[box_idx, None, :]
    pen[box_idx] = np.maximum(0.0, -corners[:, :, 2].min(axis=1))
    return pen


def _run(arm_name: str, knob, n: int, steps: int, warmup: int, seed: int) -> dict:
    kwargs = {"n_sub": knob} if arm_name in ("mujoco", "icf") else {"tol": knob}

    # timed pass: no readbacks
    model = build_model(n, seed=seed)
    arm = make_arm(model, arm_name, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(warmup):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    wp.synchronize()
    wall_ms = (time.perf_counter() - t0) / steps * 1e3

    # metric pass: fresh identical build, read back every boundary
    model = build_model(n, seed=seed)
    arm = make_arm(model, arm_name, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    pens = []
    for _ in range(warmup + steps):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        pens.append(_penetrations(model, s0))
    pen = np.concatenate(pens[warmup:])
    return {
        "arm": arm_name,
        "knob": knob,
        "wall_ms_per_boundary": round(wall_ms, 4),
        "pen_mean_m": float(pen.mean()),
        "pen_max_m": float(pen.max()),
        "pen_p95_m": float(np.quantile(pen, 0.95)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_penetration.csv")
    args = p.parse_args()

    rows = []
    for arm_name in args.arms:
        knobs = FIXED_SUBS if arm_name in ("mujoco", "icf") else ADAPTIVE_TOLS
        for knob in knobs:
            row = _run(arm_name, knob, args.n, args.steps, args.warmup, args.seed)
            rows.append(row)
            print(row, flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
