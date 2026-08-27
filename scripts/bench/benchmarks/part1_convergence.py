# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""dt -> 0 convergence: ICF converges, MuJoCo struggles.

For each fixed-step backend, run an identical WELL-POSED contact scene from
an identical initial state to the same horizon at a ladder of substep
counts (dt = DT_OUTER / n_sub), snapshot the final body positions, and
report the Cauchy differences between consecutive rungs:

    err(n) = max_bodies | pos(n_sub = n) - pos(n_sub = 2n) |

The scene is deliberately well-posed — per world, one dropped sphere and
one FLAT box sliding to rest under friction from an initial push, laterally
separated so they never touch each other. The chaotic 18-body pile is
useless here (body-body cascades measure chaos, not the contact model),
and a TILTED falling box is nearly as bad: which corner lands first is
itself dt-sensitive, and its bounce roulette drowned both backends in
mm-to-cm noise (measured). The sliding box adds an analytic oracle on
top of the Cauchy ladder: a face-sliding box stops after v^2/(2 mu g)
— 0.170 m at v=1, mu=0.3 — so the bench reports absolute stopping-
distance error too, in exactly the sustained-friction regime the paper
targets.

Each rung runs in a SUBPROCESS: repeated in-process solver builds corrupt
GPU state (observed CUDA 700 on the 7th sequential build; the existing
accuracy bench isolates for the same reason).

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_convergence
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import numpy as np

DT_OUTER = 0.01
LADDER = [1, 2, 4, 8, 16, 32, 64]
BOX_PUSH_VX = 1.0
MU = 0.3
STOP_DISTANCE = BOX_PUSH_VX**2 / (2 * MU * 9.81)


def build_simple_model(n_worlds: int):
    """One dropped sphere + one friction-sliding flat box per world."""
    import warp as wp

    import newton

    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005)

    b = template.add_body(xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.30), q=wp.quat_identity()))
    template.add_shape_sphere(b, radius=0.05, cfg=cfg)

    # Flat box resting on the plane (margin-high), pushed along +x below.
    b = template.add_body(xform=wp.transform(p=wp.vec3(0.15, 0.3, 0.0555), q=wp.quat_identity()))
    template.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)

    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    model = builder.finalize()

    # Push the boxes: vx = 1.0 m/s. Free-joint qd is [LINEAR, angular]
    # (slot-probed: writing slot 0 moves the body, slot 3 spins it); the
    # box is the SECOND body of each world's two.
    qd = model.joint_qd.numpy()
    qd = qd.reshape(n_worlds, 2, 6)
    qd[:, 1, 0] = BOX_PUSH_VX
    model.joint_qd.assign(qd.reshape(-1))
    bqd = model.body_qd.numpy().reshape(n_worlds, 2, 6)
    bqd[:, 1, 0] = BOX_PUSH_VX
    model.body_qd.assign(bqd.reshape(-1, 6))
    return model


def _run_single(backend: str, n_sub: int, n: int, horizon: float, out_path: str) -> None:
    from scripts.bench.four_arms import make_arm

    model = build_simple_model(n)
    arm = make_arm(model, backend, n_sub=n_sub)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(int(round(horizon / DT_OUTER))):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    np.save(out_path, s0.body_q.numpy().reshape(-1, 7)[:, :3])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--backends", nargs="*", default=["icf", "mujoco"])
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_convergence.csv")
    p.add_argument("--single", nargs=3, metavar=("BACKEND", "N_SUB", "OUT_NPY"), default=None)
    args = p.parse_args()

    if args.single is not None:
        backend, n_sub, out_npy = args.single
        _run_single(backend, int(n_sub), args.n, args.horizon, out_npy)
        return 0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = os.path.join(os.path.dirname(args.out), "part1_convergence_tmp")
    os.makedirs(tmp, exist_ok=True)

    rows = []
    for backend in args.backends:
        finals = {}
        for n_sub in LADDER:
            out_npy = os.path.join(tmp, f"{backend}_{n_sub}.npy")
            r = subprocess.run(
                [
                    sys.executable, "-m", "scripts.bench.benchmarks.part1_convergence",
                    "--single", backend, str(n_sub), out_npy,
                    "--n", str(args.n), "--horizon", str(args.horizon),
                ],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"RUNG FAILED {backend} n_sub={n_sub}:\n{r.stderr[-800:]}", flush=True)
                continue
            finals[n_sub] = np.load(out_npy)
        for a, b in zip(LADDER[:-1], LADDER[1:]):
            if a not in finals or b not in finals:
                continue
            d = np.abs(finals[a] - finals[b]).reshape(-1, 2, 3)
            box_x = finals[a].reshape(-1, 2, 3)[:, 1, 0]
            stop_err = float(np.abs((box_x - 0.15) - STOP_DISTANCE).max())
            rows.append({
                "backend": backend,
                "n_sub": a,
                "dt": DT_OUTER / a,
                "cauchy_sphere_m": float(d[:, 0, :].max()),
                "cauchy_box_m": float(d[:, 1, :].max()),
                "box_stop_err_m": stop_err,
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
