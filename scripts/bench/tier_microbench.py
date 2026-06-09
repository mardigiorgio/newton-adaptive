# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Deterministic micro-benchmark of the compaction tier step cost.

The episode speed gate is confounded by chaotic per-world divergence (a single
stuck world inflates wall-time +-30%), which drowns out small accuracy-safe perf
changes. This harness instead times the captured CUDA-graph tier replay on a
FIXED dense-contact state, with no episode and no chaos. It answers: is the tier
step compute-bound (time ~ linear in tier size -> finer tiers / lower floor cut
wasted padding compute) or launch-bound (time ~ flat -> granularity cannot help)?

Example::

    uv run python -m scripts.bench.tier_microbench --n 1024
"""

from __future__ import annotations

import argparse
import time

import warp as wp

import newton
import newton.solvers

from scripts.scenes import contact_objects as co

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _gather_rows_2d


def _time_graph(solver, data, reps=200, warm=20):
    """Time solver._run_step(data) (graph replay) over reps, return ms/step."""
    dev = solver.model.device
    with wp.ScopedDevice(dev):
        for _ in range(warm):
            solver._run_step(data)
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            solver._run_step(data)
        wp.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def _fill_tier_from_data(solver, tier, size):
    """Gather the first `size` worlds of the full mjw_data into the tier so it
    holds a realistic dense-contact configuration (not stale zeros)."""
    nq = int(solver.mjw_data.qpos.shape[1])
    nv = int(solver.mjw_data.qvel.shape[1])
    idx = wp.array(list(range(size)), dtype=wp.int32, device=solver.model.device)
    wp.launch(_gather_rows_2d, dim=(size, nq),
              inputs=[solver.mjw_data.qpos, idx], outputs=[tier.qpos])
    wp.launch(_gather_rows_2d, dim=(size, nv),
              inputs=[solver.mjw_data.qvel, idx], outputs=[tier.qvel])
    wp.launch(_gather_rows_2d, dim=(size, nv),
              inputs=[solver.mjw_data.qacc_warmstart, idx], outputs=[tier.qacc_warmstart])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--warmsteps", type=int, default=15)
    args = p.parse_args()

    wp.init()
    model = co.build_model_randomized(args.n, seed=7)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model, tol=1e-3, dt_init=co.DT_OUTER, dt_min=1e-6, dt_max=co.DT_OUTER,
        use_mujoco_contacts=True, nconmax=co._NCON, njmax=co._NJM,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(args.warmsteps):
        solver.step(s0, s1, ctrl, None, co.DT_OUTER)
    wp.synchronize()

    # Tiers (ascending) + the full mjw_data.
    entries = [(size, data) for size, data in solver._tiers]
    entries.append((args.n, solver.mjw_data))

    print(f"N={args.n}  tiers={[s for s, _ in solver._tiers]}  (graph replay, dense contact)")
    print(f"{'tier_size':>10}  {'ms/step':>9}  {'us/world':>9}")
    print("-" * 34)
    prev = None
    for size, data in entries:
        if data is not solver.mjw_data:
            _fill_tier_from_data(solver, data, size)
            wp.synchronize()
        ms = _time_graph(solver, data)
        print(f"{size:>10}  {ms:>9.4f}  {ms/size*1e3:>9.2f}")
        prev = (size, ms)

    # Linearity probe: if ms ~ a + b*size, a large constant 'a' => launch-bound.
    print("\nInterpretation: if us/world is roughly constant across sizes, the tier")
    print("step is COMPUTE-bound (finer tiers cut padding waste). If ms/step is")
    print("roughly flat while size grows, it is LAUNCH-bound (granularity won't help).")


if __name__ == "__main__":
    main()
