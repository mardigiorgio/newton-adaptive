"""Contract tests for the adaptive step controller (`_calc_adjusted_step`).

ICF-parity contract (mirrors SolverICFAdaptive._adapt_dt):

  * Divergence latch: a non-finite/sentinel error latches ``diverged`` when the
    step is at the ``dt_min`` floor OR has shrunk to the floorless sanity depth
    (1e-3 of the boundary seed). The latch refuses the commit, snaps the clock
    to the boundary, and reseeds ``ideal_dt``; the env-side termination
    consumes it. Above both depths, a diverged attempt REJECTS and retries
    smaller -- containment, not escalation.
  * Floor accept: a FINITE error at the floor accepts and commits (the floor
    cannot be subdivided; the accuracy guarantee is suspended for that step),
    and ``ideal_dt`` stays rule-sized rather than pinned to the floor, so the
    world lifts off once the difficulty passes.
  * Elementary Drake law otherwise: safety 0.9, sqrt sizing, deadband
    (0.9, 1.2), clamp [0.1, 5.0]x.

Pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _calc_adjusted_step

wp.init()

DEV = "cpu"
TOL = 1.0e-3
DT_SEED = 1.0e-2
DIVERGENCE = 1.0e9  # threshold; the error kernel emits 1e10 for NaN/inf states
SENTINEL = 1.0e10


def _run(err_vals, dt_vals, dt_min=1.0e-6, dt_seed=DT_SEED):
    n = len(err_vals)
    err = wp.array(np.asarray(err_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    dt = wp.array(np.asarray(dt_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    ideal = wp.zeros(n, dtype=wp.float32, device=DEV)
    accepted = wp.zeros(n, dtype=wp.bool, device=DEV)
    commit = wp.zeros(n, dtype=wp.bool, device=DEV)
    diverged = wp.zeros(n, dtype=wp.bool, device=DEV)
    limited = wp.zeros(n, dtype=wp.int32, device=DEV)
    sim_time = wp.zeros(n, dtype=wp.float32, device=DEV)
    next_time = wp.full(n, 1.0, dtype=wp.float32, device=DEV)
    wp.launch(
        _calc_adjusted_step,
        dim=n,
        inputs=[
            err,
            dt,
            ideal,
            accepted,
            commit,
            diverged,
            TOL,
            dt_min,
            1.0e6,
            dt_seed,
            DIVERGENCE,
            limited,
            sim_time,
            next_time,
        ],
        device=DEV,
    )
    return (
        accepted.numpy(),
        commit.numpy(),
        diverged.numpy(),
        ideal.numpy(),
        sim_time.numpy(),
        next_time.numpy(),
    )


def test_floor_diverged_latches_and_snaps():
    acc, com, div, ideal, st, nt = _run([SENTINEL], [1.0e-6])
    assert bool(div[0]) is True, "non-finite at the floor must latch"
    assert bool(com[0]) is False, "a non-finite state must never be committed"
    assert float(st[0]) == float(nt[0]), "latch snaps the clock to the boundary"
    assert abs(float(ideal[0]) - DT_SEED) < 1e-8, "latch reseeds ideal_dt"


def test_floorless_sanity_depth_latches():
    """dt_min = 0 (floorless): the sanity depth (1e-3 of the seed) still latches."""
    acc, com, div, ideal, st, nt = _run([SENTINEL], [0.5e-5], dt_min=0.0)
    assert bool(div[0]) is True, "sentinel at 1e-3 of the seed must latch floorlessly"
    assert bool(com[0]) is False
    assert float(st[0]) == float(nt[0])


def test_above_sanity_depth_diverged_rejects():
    acc, com, div, ideal, _, _ = _run([SENTINEL], [10.0 * DT_SEED], dt_min=0.0)
    assert bool(acc[0]) is False, "diverged above the sanity depth rejects and retries"
    assert bool(com[0]) is False
    assert bool(div[0]) is False, "containment: no latch while shrinking can still help"
    assert abs(float(ideal[0]) - 1.0 * DT_SEED) < 1e-8, "hard shrink is 0.1x"


def test_floor_finite_over_tol_commits_progress():
    acc, com, div, ideal, _, _ = _run([10.0 * TOL], [1.0e-6])
    assert bool(acc[0]) is True, "the floor cannot be subdivided: accept"
    assert bool(com[0]) is True, "finite floor step must make committed progress"
    assert bool(div[0]) is False
    assert float(ideal[0]) < 1.0e-6, "ideal_dt stays rule-sized, not pinned to the floor"


def test_normal_within_tol_commits_and_grows():
    acc, com, div, ideal, _, _ = _run([0.01 * TOL], [10.0 * DT_SEED])
    assert bool(acc[0]) is True and bool(com[0]) is True and bool(div[0]) is False
    assert float(ideal[0]) > 10.0 * DT_SEED, "well under tol must grow the step"


def test_finite_over_tol_above_floor_rejects_and_shrinks():
    acc, com, div, ideal, _, _ = _run([100.0 * TOL], [10.0 * DT_SEED])
    assert bool(acc[0]) is False and bool(com[0]) is False and bool(div[0]) is False
    assert float(ideal[0]) < 10.0 * DT_SEED, "over-tol reject must shrink"


def test_deadband_holds_step():
    """new_step inside (0.9, 1.2)x holds dt exactly (hysteresis)."""
    # e = 0.7*tol -> 0.9*sqrt(1/0.7) = 1.076x -> inside the deadband.
    acc, com, div, ideal, _, _ = _run([0.7 * TOL], [10.0 * DT_SEED])
    assert bool(acc[0]) is True
    assert abs(float(ideal[0]) - 10.0 * DT_SEED) < 1e-7, "deadband must hold the step"


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
