# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Actuated stiff contact, four arms: a PD-driven gantry presses a light
fingertip into a box and slides it (scripts/scenes/actuated_press.py).

Design and metrics from the literature review (PART1_LITERATURE.md, Theme
E): controller stiffness K_p swept as in CENIC Fig. 12 (K_d = 2 sqrt(K_p m));
fixed dt in {10, 5, 2, 1} ms and eps in {1e-1 .. 1e-4} as in CENIC Table I.
Per world, from the state only (no host sync in the loop except the
per-boundary target update every arm shares):

* penetration: box into the table (vs the resting depth m g / k) and
  fingertip into the box face (vs the quasi-static push depth mu m g / k),
  max and time-mean;
* chatter: RMS of the box's vertical velocity during the push (a sliding
  box should have none; ICF Fig. 2-3's normal-velocity artifact) and RMS of
  the tip's velocity relative to the box during the cruise (a steady push
  has none);
* instability: non-finite state or |v| > 10 m/s at any boundary;
* tracking: RMS fingertip error against the commanded x target during the
  push; box displacement at the end vs the commanded push.

Wall time per simulated second is recorded; per the matched-accuracy rule it
is only comparable across artifact-free cells.

    uv run python -m scripts.bench.benchmarks.part1_actuated --n 8
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

from scripts.bench.four_arms import (
    ExhaustionTracker,
    _make_icf,
    _make_icf_adaptive,
    _make_mujoco,
    _make_mujoco_adaptive,
)
from scripts.scenes.actuated_press import (
    BOX_HALF,
    BOX_MASS,
    GAP,
    MU,
    SLIDE_LEN,
    TIP_R,
    X0,
    ActuatedSpec,
    build,
    program,
)

BOUNDARY_S = 0.01
KPS = [1e2, 1e3, 1e4, 1e5, 1e6]
KS = [1e5, 1e7]
SPEEDS = [0.05, 0.3]
FIXED_DTS = [1e-2, 5e-3, 2e-3, 1e-3]
ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4]


def _arm(model, spec, backend, kind, knob):
    if backend == "icf":
        return (
            _make_icf(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, spec.icf)
            if kind == "fixed"
            else _make_icf_adaptive(model, knob, BOUNDARY_S, spec.icf, 4096)
        )
    return (
        _make_mujoco(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, spec.mujoco_solref)
        if kind == "fixed"
        else _make_mujoco_adaptive(model, knob, BOUNDARY_S, 4096, spec.mujoco_solref)
    )


def _run(backend, kind, knob, kp, k, speed, n):
    spec = ActuatedSpec(k=k, kp=kp, slide_speed=speed)
    model = build(n, spec)
    arm = _arm(model, spec, backend, kind, knob)
    tracker = ExhaustionTracker(arm) if kind == "adaptive" else None
    s0, s1, ctrl = model.state(), model.state(), model.control()
    nb = int(round(spec.horizon_s / BOUNDARY_S))
    # joint layout per world: box free joint (7 coords / 6 dofs), then prismatic x, prismatic z;
    # targets are DOF- or coord-shaped by newton.use_coord_layout_targets -- the prismatics are last either way
    tq = ctrl.joint_target_q.numpy()
    per_world_target_stride = tq.shape[0] // n
    assert per_world_target_stride in (8, 9), per_world_target_stride
    box_ids = np.arange(n) * 3  # bodies per world: box, carriage, tip
    tip_ids = np.arange(n) * 3 + 2
    static_box = BOX_MASS * 9.81 / k
    static_tip = MU * BOX_MASS * 9.81 / k
    pen_box, pen_tip, chatter, rel, track, unstable = [], [], [], [], [], False
    t_slide0 = 0.5 + 0.2
    t_slide1 = t_slide0 + SLIDE_LEN / speed + 0.2
    t_cruise0, t_cruise1 = t_slide0 + 0.1 + GAP / speed + 0.1, t_slide1 - 0.2 - 0.1
    wp.synchronize()
    t0 = time.perf_counter()
    for b in range(nb):
        t = b * BOUNDARY_S
        x_t, z_t = program(t, spec)
        tq = tq.reshape(n, per_world_target_stride)
        tq[:, -2] = x_t
        tq[:, -1] = z_t
        ctrl.joint_target_q.assign(tq.reshape(-1))
        s0, s1 = arm.boundary(s0, s1, ctrl)
        if tracker:
            tracker.tick()
        bq = s0.body_q.numpy().reshape(-1, 7)
        bqd = s0.body_qd.numpy().reshape(-1, 6)
        if not np.isfinite(bq).all() or np.abs(bqd[:, :3]).max() > 10.0:
            unstable = True
            break
        zb = bq[box_ids, 2]
        pen_box.append(np.maximum(0.0, BOX_HALF - zb))
        beside = np.abs(bq[tip_ids, 2] - zb) < BOX_HALF
        pen_tip.append(np.where(beside, np.maximum(0.0, (bq[tip_ids, 0] + TIP_R) - (bq[box_ids, 0] - BOX_HALF)), 0.0))
        if t_slide0 <= t <= t_slide1:
            chatter.append(bqd[box_ids, 2])
            track.append(bq[tip_ids, 0] - (X0 + x_t))
        if t_cruise0 <= t <= t_cruise1:
            rel.append(bqd[tip_ids, 0] - bqd[box_ids, 0])
    wp.synchronize()
    wall = time.perf_counter() - t0
    bq = s0.body_q.numpy().reshape(-1, 7)
    out = {
        "unstable": bool(unstable),
        "wall_s_per_sim_s": wall / (nb * BOUNDARY_S),
        "exhausted_frac": tracker.fraction() if tracker else 0.0,
    }
    if pen_box:
        pb = np.concatenate(pen_box)
        pt = np.concatenate(pen_tip)
        out.update(
            {
                "pen_box_mean_over_static": float(pb.mean() / static_box),
                "pen_box_max_m": float(pb.max()),
                "pen_tip_mean_over_static": float(pt.mean() / static_tip),
                "pen_tip_max_m": float(pt.max()),
                "chatter_vz_rms_m_s": float(np.sqrt(np.mean(np.concatenate(chatter) ** 2))) if chatter else "",
                "rel_vx_rms_m_s": float(np.sqrt(np.mean(np.concatenate(rel) ** 2))) if rel else "",
                "track_rms_m": float(np.sqrt(np.mean(np.concatenate(track) ** 2))) if track else "",
                "box_displacement_m": float((bq[box_ids, 0]).mean()),
                "box_displacement_commanded_m": SLIDE_LEN - GAP,
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--kps", nargs="*", type=float, default=KPS)
    p.add_argument("--ks", nargs="*", type=float, default=KS)
    p.add_argument("--speeds", nargs="*", type=float, default=SPEEDS)
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_actuated.csv")
    p.add_argument("--single", nargs=6, metavar=("BACKEND", "KIND", "KNOB", "KP", "K", "SPEED"), default=None)
    args = p.parse_args()
    if args.single is not None:
        backend, kind, knob, kp, k, speed = args.single
        print(
            "ROW " + json.dumps(_run(backend, kind, float(knob), float(kp), float(k), float(speed), args.n)), flush=True
        )
        return 0
    rows = []
    for k in args.ks:
        for speed in args.speeds:
            for kp in args.kps:
                for backend in ("icf", "mujoco"):
                    for kind, knobs in (("fixed", FIXED_DTS), ("adaptive", ACCURACIES)):
                        for knob in knobs:
                            r = subprocess.run(
                                [
                                    sys.executable,
                                    "-m",
                                    "scripts.bench.benchmarks.part1_actuated",
                                    "--n",
                                    str(args.n),
                                    "--single",
                                    backend,
                                    kind,
                                    str(knob),
                                    str(kp),
                                    str(k),
                                    str(speed),
                                ],
                                capture_output=True,
                                text=True,
                                timeout=3600,
                            )
                            if "over the scannable budget" in r.stderr:
                                print(f"CONTACT OVERFLOW {backend} {kind} {knob} kp={kp:g} k={k:g}", flush=True)
                                continue
                            got = None
                            for line in r.stdout.splitlines():
                                if line.startswith("ROW "):
                                    got = json.loads(line[4:])
                            if got is None:
                                print(
                                    f"FAIL {backend} {kind} {knob} kp={kp:g} k={k:g} v={speed:g}: {r.stderr[-300:]}",
                                    flush=True,
                                )
                                continue
                            arm = backend if kind == "fixed" else f"{backend}-adaptive"
                            row = {
                                "arm": arm,
                                "dt_s": knob if kind == "fixed" else "",
                                "accuracy": knob if kind == "adaptive" else "",
                                "kp": kp,
                                "kd": ActuatedSpec(k, kp, speed).kd,
                                "k": k,
                                "slide_speed": speed,
                                "n_worlds": args.n,
                                **got,
                            }
                            rows.append(row)
                            print(row, flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = []
    for r in rows:
        for kk in r:
            if kk not in keys:
                keys.append(kk)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
