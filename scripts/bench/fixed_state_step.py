# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reliable, chaos-free timing of a single MuJoCo Warp step.

THE measurement lesson from the solver-optimization sweep (see
docs/superpowers/solver-optimization-log.md): episode wall-time swings +-30% and
NaNs intermittently from chaotic per-world divergence, and even the tier
micro-bench is confounded because its warmup reaches a chaos-dependent contact
state. The ONLY reliable speed signal is to time ``mjw.step`` on ONE FIXED dense
state, reloading the identical ``qpos``/``qvel``/``qacc_warmstart`` before each
rep so every measurement sees the exact same physics. This gives +-0.1%
repeatability and isolates whatever knob you are changing.

Use this to evaluate any per-step change (opt params, integrator, contact
budgets, a step-pipeline refactor). ALWAYS pair a speed delta with the printed
physics-correctness delta: a faster step that changes ``max|dqpos|`` above the
solver tolerance is doing different/cheaper work, not a real speedup (two such
artifacts -- ls_iterations and a naive collision-share -- passed the
self-referential step-doubling error check while being wrong).

Example::

    uv run python -m scripts.bench.fixed_state_step --n 512
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp
import mujoco_warp as mjw

import newton
import newton.solvers

from scripts.scenes import contact_objects as co


def build_fixed_state(n: int, warmsteps: int, seed: int):
    """Step a solver into dense contact and snapshot one reproducible state."""
    model = co.build_model_randomized(n, seed=seed)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model, tol=1e-3, dt_init=co.DT_OUTER, dt_min=1e-6, dt_max=co.DT_OUTER,
        use_mujoco_contacts=True, nconmax=co._NCON, njmax=co._NJM,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(warmsteps):
        solver.step(s0, s1, ctrl, None, co.DT_OUTER)
    wp.synchronize()
    d = solver.mjw_data
    snapshot = (d.qpos.numpy().copy(), d.qvel.numpy().copy(),
                d.qacc_warmstart.numpy().copy())
    return solver, snapshot


def time_step(solver, snapshot, *, dt: float, reps: int = 60, warm: int = 8) -> tuple[float, float, float]:
    """Median ms for one eager ``mjw.step`` on the fixed snapshot at ``dt``.

    Returns (median_ms, max_abs_dqpos_vs_baseline_within_run, n_contacts). The
    caller compares median_ms across configs and qpos across configs.
    """
    m, d, dev = solver.mjw_model, solver.mjw_data, solver.model.device
    qpos0, qvel0, ws0 = snapshot
    nw = d.nworld

    def reload():
        d.qpos.assign(qpos0)
        d.qvel.assign(qvel0)
        d.qacc_warmstart.assign(ws0)

    m.opt.timestep = wp.full(nw, dt, dtype=wp.float32, device=dev)
    with wp.ScopedDevice(dev):
        for _ in range(warm):
            reload(); mjw.step(m, d)
        wp.synchronize()
        tt = []
        for _ in range(reps):
            reload()
            wp.synchronize()
            t0 = time.perf_counter()
            mjw.step(m, d)
            wp.synchronize()
            tt.append(time.perf_counter() - t0)
        reload(); mjw.step(m, d); wp.synchronize()
        out_qpos = d.qpos.numpy().copy()
        ncon = int(d.nacon.numpy()[0])
    return float(np.median(tt)) * 1e3, out_qpos, ncon


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--warmsteps", type=int, default=15)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dt", type=float, default=co.DT_OUTER / 30.0)
    args = p.parse_args()

    wp.init()
    solver, snapshot = build_fixed_state(args.n, args.warmsteps, args.seed)
    ms, qpos, ncon = time_step(solver, snapshot, dt=args.dt)
    print(f"N={args.n}  dt={args.dt:.2e}  fixed-state eager mjw.step: "
          f"{ms:.4f} ms  (ncontacts={ncon})")
    print("To evaluate a knob: change opt before calling time_step again, compare")
    print("ms AND max|dqpos| of the returned qpos vs this baseline qpos. A real")
    print("speedup keeps max|dqpos| at/below the solver tolerance (~1e-6..1e-5).")


if __name__ == "__main__":
    main()
