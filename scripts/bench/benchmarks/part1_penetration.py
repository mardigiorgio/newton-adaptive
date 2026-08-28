# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ground penetration vs wall time, four solver arms, on a CENIC scene
(default: hard clutter). Every row states its accuracy (adaptive) or
time step (fixed); ejections past the bin walls are counted as the
paper's passthrough artifact class.

Each configuration (arm x accuracy knob) runs the scene
twice from identical initial states: a TIMED pass with no readbacks
(per-boundary median wall-clock, synced each boundary), and a METRIC pass that reads the
state back every boundary and measures ground-plane penetration
analytically from the state — spheres by center height against their
radius, boxes by their deepest transformed corner. Analytic state-side
penetration is solver-agnostic: it cannot flatter a backend the way a
solver's own contact report can.

Accuracy knobs: fixed arms sweep substeps-per-boundary (dt ladder);
adaptive arms sweep tolerance.

Each configuration runs in its own SUBPROCESS: sequential in-process
solver builds corrupt GPU state (observed CUDA 700 here and in the
convergence bench; the existing accuracy bench isolates the same way).

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_penetration --n 64 --steps 200
Emits CSV rows (arm, knob, wall_ms_per_boundary, pen_mean_m, pen_max_m,
pen_p95_m) and a penetration-vs-wall plot.
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

from scripts.bench.four_arms import ARMS, ExhaustionTracker, build_model, make_arm
from scripts.scenes.cenic_scenes import BIN_HALF, BIN_WALL_H, DT_OUTER, SCENES

FIXED_SUBS = [1, 2, 5, 10]  # dt = 10, 5, 2, 1 ms
ADAPTIVE_TOLS = [1e-1, 1e-2, 1e-3, 1e-4]

_UNIT_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
    dtype=np.float32,
)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate corner offsets v [8,3] by quaternions q [B,4] (x,y,z,w)."""
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    qv = q[:, None, :3]
    uv = np.cross(qv, v[None, :, :])
    uuv = np.cross(qv, uv)
    return v[None, :, :] + 2.0 * (w[:, :, None] * uv + uuv)


class _Geometry:
    """Per-body collision geometry read from the MODEL (not assumed): sphere
    radius or box half-extents, one shape per dynamic body at its origin."""

    def __init__(self, model):
        from newton._src.geometry.types import GeoType

        st = model.shape_type.numpy()
        sb = model.shape_body.numpy()
        sc = model.shape_scale.numpy()
        sx = model.shape_transform.numpy()
        n = model.body_count
        self.radius = np.full(n, np.nan, dtype=np.float32)
        self.half = np.full((n, 3), np.nan, dtype=np.float32)
        for i, b in enumerate(sb):
            if b < 0:
                continue
            assert np.abs(sx[i][:3]).max() < 1e-7, "shape offset from body origin: geometry read would be wrong"
            if GeoType(int(st[i])) == GeoType.SPHERE:
                self.radius[b] = sc[i][0]
            elif GeoType(int(st[i])) == GeoType.BOX:
                self.half[b] = sc[i]
            else:
                raise ValueError(f"unsupported dynamic shape {GeoType(int(st[i])).name}")
        self.is_sphere = ~np.isnan(self.radius)
        self.is_box = ~np.isnan(self.half[:, 0])

    def penetrations(self, state) -> np.ndarray:
        """Per-body ground-plane (z = 0) penetration [m]; zero where separated."""
        bq = state.body_q.numpy().reshape(-1, 7)
        pos, quat = bq[:, :3], bq[:, 3:]
        pen = np.zeros(bq.shape[0], dtype=np.float32)
        sp = self.is_sphere
        pen[sp] = np.maximum(0.0, self.radius[sp] - pos[sp, 2])
        bx = self.is_box
        if bx.any():
            corners = _quat_rotate(quat[bx], _UNIT_CORNERS) * self.half[bx][:, None, :] + pos[bx, None, :]
            pen[bx] = np.maximum(0.0, -corners[:, :, 2].min(axis=1))
        return pen

    @staticmethod
    def out_of_bin(state) -> np.ndarray:
        """Bool per body: ejected past the bin's inner walls or over them
        (the paper's passthrough/ejection artifact class)."""
        bq = state.body_q.numpy().reshape(-1, 7)
        return (np.abs(bq[:, 0]) > BIN_HALF) | (np.abs(bq[:, 1]) > BIN_HALF) | (bq[:, 2] > BIN_WALL_H)

    def ejection_breakdown(self, state) -> dict:
        """Where and what escaped: a body beyond the inner walls while BELOW
        the rim passed THROUGH a wall (a collision failure); one above the
        rim went OVER it (dynamics). Split by shape so a cube-vs-wall
        narrowphase problem is distinguishable from bounce physics."""
        bq = state.body_q.numpy().reshape(-1, 7)
        lateral = (np.abs(bq[:, 0]) > BIN_HALF) | (np.abs(bq[:, 1]) > BIN_HALF)
        over = bq[:, 2] > BIN_WALL_H
        through = lateral & ~over
        n = max(bq.shape[0], 1)
        return {
            "eject_through_wall_frac": float(through.mean()),
            "eject_over_rim_frac": float(over.mean()),
            "eject_spheres_frac": float((through | over)[self.is_sphere].mean()) if self.is_sphere.any() else 0.0,
            "eject_cubes_frac": float((through | over)[self.is_box].mean()) if self.is_box.any() else 0.0,
        }


def _penetrations(model, state) -> np.ndarray:
    return _Geometry(model).penetrations(state)


MAX_SUBSTEPS = 4096


def _run_pass(scene: str, arm_name: str, knob, n: int, steps: int, warmup: int, seed: int, which: str) -> dict:
    """One measurement pass — ONE model build per process (a second build
    in-process reproduces the CUDA 700 on the MuJoCo arms)."""
    fixed = arm_name in ("mujoco", "icf")
    kwargs = {"n_sub": knob} if fixed else {"tol": knob, "max_substeps": MAX_SUBSTEPS}
    model = build_model(n, seed=seed, scene=scene)
    arm = make_arm(model, arm_name, scene=scene, **kwargs)
    tracker = ExhaustionTracker(arm) if not fixed else None
    s0, s1, ctrl = model.state(), model.state(), model.control()
    if which == "timed":
        for _ in range(warmup):
            s0, s1 = arm.boundary(s0, s1, ctrl)
        wp.synchronize()
        # per-boundary MEDIAN, synced each boundary: a mean over the run is
        # hostage to any transient GPU contention (measured 3x inflation
        # when another sweep overlapped).
        times = []
        for _ in range(steps):
            t0 = time.perf_counter()
            s0, s1 = arm.boundary(s0, s1, ctrl)
            if tracker:
                tracker.tick()
            wp.synchronize()
            times.append(time.perf_counter() - t0)
        return {
            "wall_ms_per_boundary": round(float(np.median(times)) * 1e3, 4),
            "exhausted_frac": tracker.fraction() if tracker else 0.0,
        }
    geom = _Geometry(model)
    pens, outs = [], []
    for _ in range(warmup + steps):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        pens.append(geom.penetrations(s0))
        outs.append(geom.out_of_bin(s0))
    pen = np.concatenate(pens[warmup:])
    return {
        "pen_mean_m": float(pen.mean()),
        "pen_max_m": float(pen.max()),
        "pen_p95_m": float(np.quantile(pen, 0.95)),
        "out_of_bin_frac": float(outs[-1].mean()),
        **geom.ejection_breakdown(s0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arms", nargs="*", default=list(ARMS))
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--single", nargs=3, metavar=("ARM", "KNOB", "PASS"), default=None)
    args = p.parse_args()
    out = args.out or f"scripts/bench/results/part1_penetration_{args.scene}.csv"

    if args.single is not None:
        arm_name, knob_s, which = args.single
        knob = int(knob_s) if arm_name in ("mujoco", "icf") else float(knob_s)
        row = _run_pass(args.scene, arm_name, knob, args.n, args.steps, args.warmup, args.seed, which)
        print("ROW " + json.dumps(row), flush=True)
        return 0

    def run_pass(arm_name, knob, which):
        r = subprocess.run(
            [
                sys.executable, "-m", "scripts.bench.benchmarks.part1_penetration",
                "--scene", args.scene, "--single", arm_name, str(knob), which,
                "--n", str(args.n), "--steps", str(args.steps),
                "--warmup", str(args.warmup), "--seed", str(args.seed),
            ],
            capture_output=True, text=True,
        )
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                return json.loads(line[4:])
        print(f"PASS FAILED {arm_name} knob={knob} {which}:\n{r.stderr[-500:]}", flush=True)
        return None

    rows = []
    for arm_name in args.arms:
        knobs = FIXED_SUBS if arm_name in ("mujoco", "icf") else ADAPTIVE_TOLS
        for knob in knobs:
            timed = run_pass(arm_name, knob, "timed")
            metric = run_pass(arm_name, knob, "metric")
            if timed is None or metric is None:
                continue
            fixed = arm_name in ("mujoco", "icf")
            row = {
                "scene": args.scene,
                "arm": arm_name,
                "accuracy": "" if fixed else knob,
                "dt_s": DT_OUTER / knob if fixed else "",
                "max_substeps": "" if fixed else MAX_SUBSTEPS,
                "n_worlds": args.n,
                **timed,
                **metric,
                "status": "budget-exhausted" if timed.get("exhausted_frac", 0.0) > 0.0 else "ok",
            }
            rows.append(row)
            print(row, flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
