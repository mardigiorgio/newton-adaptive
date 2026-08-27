# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""dt -> 0 convergence: ICF converges, MuJoCo struggles.

For each fixed-step backend, run the identical contact scene from the
identical initial state to the same horizon at a ladder of substep counts
(dt = DT_OUTER / n_sub), snapshot the final body positions, and report the
Cauchy differences between consecutive ladder rungs:

    err(n) = max_bodies | pos(n_sub = n) - pos(n_sub = 2n) |

A convergent integrator/contact model drives err(n) -> 0 as dt shrinks; a
contact model whose solution depends irreducibly on dt stalls or
oscillates. No reference solution is assumed — the ladder is its own
oracle (Cauchy, not comparison-to-truth), which is exactly the claim the
paper makes: ICF converges to *a* solution as dt -> 0.

Standalone:
    uv run python -m scripts.bench.benchmarks.part1_convergence --n 4 --horizon 1.0
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from scripts.bench.four_arms import build_model, make_arm
from scripts.scenes.contact_objects import DT_OUTER

LADDER = [1, 2, 4, 8, 16, 32, 64]


def _final_positions(backend: str, n_sub: int, n: int, horizon: float, seed: int) -> np.ndarray:
    model = build_model(n, seed=seed)
    arm = make_arm(model, backend, n_sub=n_sub)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    boundaries = int(round(horizon / DT_OUTER))
    for _ in range(boundaries):
        s0, s1 = arm.boundary(s0, s1, ctrl)
    return s0.body_q.numpy().reshape(-1, 7)[:, :3].copy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backends", nargs="*", default=["icf", "mujoco"])
    p.add_argument("--out", type=str, default="scripts/bench/results/part1_convergence.csv")
    args = p.parse_args()

    rows = []
    for backend in args.backends:
        finals = {n_sub: _final_positions(backend, n_sub, args.n, args.horizon, args.seed) for n_sub in LADDER}
        for a, b in zip(LADDER[:-1], LADDER[1:]):
            err = float(np.abs(finals[a] - finals[b]).max())
            rows.append({
                "backend": backend,
                "n_sub": a,
                "dt": DT_OUTER / a,
                "cauchy_err_m": err,
            })
            print(rows[-1], flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
