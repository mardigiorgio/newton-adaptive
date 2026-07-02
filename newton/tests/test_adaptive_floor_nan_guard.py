"""Contract tests for the adaptive step controller (`_calc_adjusted_step`).

Current contract (the dt_min-floor divergence latch was REMOVED by design):
  * ``accepted`` -- advance ``sim_time`` (progress / no hang)
  * ``commit``   -- write the new (doubled) state; FALSE => hold the last good state
  * ``diverged`` -- accepted for signature compatibility but never written (all-False)

Above the floor, a diverged (sentinel-error) world REJECTS and retries smaller --
that is the NaN containment path (the error kernel flags NaN per component; see the
fmaxf note there). AT the floor, any world with e > tol -- including a sentinel one --
accepts AND commits: dt_min (~1e-6 s) sits ~1000x below the stable fixed step, so a
state that is non-finite there would have been non-finite for the fixed solver too;
the old hold-last-good latch was dormant in that regime and was removed.

This is a pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _calc_adjusted_step

wp.init()

DEV = "cpu"
TOL = 1.0e-3
DT_MIN = 1.0e-6
DIVERGENCE = 1.0e9  # threshold; the error kernel emits 1e10 for NaN/inf states
SENTINEL = 1.0e10  # what _inf_norm_state_error_kernel writes for a diverged world


def _run(err_vals, dt_vals):
    n = len(err_vals)
    err = wp.array(np.asarray(err_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    dt = wp.array(np.asarray(dt_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    ideal = wp.zeros(n, dtype=wp.float32, device=DEV)
    accepted = wp.zeros(n, dtype=wp.bool, device=DEV)
    commit = wp.zeros(n, dtype=wp.bool, device=DEV)
    diverged = wp.zeros(n, dtype=wp.bool, device=DEV)
    wp.launch(
        _calc_adjusted_step,
        dim=n,
        inputs=[err, dt, ideal, accepted, commit, diverged, TOL, DT_MIN, DIVERGENCE],
        device=DEV,
    )
    return (
        accepted.numpy(),
        commit.numpy(),
        diverged.numpy(),
        ideal.numpy(),
    )


def test_floor_diverged_commits_progress_without_latch():
    """At the floor with a diverged (sentinel) error: accept AND commit through the
    e > tol path, and do NOT set the (removed) diverged latch."""
    accepted, commit, diverged, ideal = _run([SENTINEL], [DT_MIN])
    assert bool(accepted[0]) is True, "must advance to avoid a boundary-loop hang"
    assert bool(commit[0]) is True, "floor worlds commit like any can't-meet-tol world"
    assert bool(diverged[0]) is False, "latch was removed; must stay all-False"
    assert abs(float(ideal[0]) - DT_MIN) < 1e-12, "floor step pins ideal_dt to dt_min"


def test_floor_finite_over_tol_commits_progress():
    """At the floor with a finite error above tol: accept AND commit (preserve prior progress behavior)."""
    accepted, commit, diverged, _ = _run([10.0 * TOL], [DT_MIN])
    assert bool(accepted[0]) is True
    assert bool(commit[0]) is True, "finite floor step must still make committed progress"
    assert bool(diverged[0]) is False


def test_normal_within_tol_commits():
    """A normal within-tolerance step accepts and commits."""
    accepted, commit, diverged, _ = _run([0.5 * TOL], [10.0 * DT_MIN])
    assert bool(accepted[0]) is True
    assert bool(commit[0]) is True
    assert bool(diverged[0]) is False


def test_above_floor_diverged_rejects_and_retries():
    """Above the floor, a diverged step is rejected (retry smaller) -- not given up, not committed."""
    accepted, commit, diverged, ideal = _run([SENTINEL], [10.0 * DT_MIN])
    assert bool(accepted[0]) is False, "should reject and retry with a smaller dt"
    assert bool(commit[0]) is False
    assert bool(diverged[0]) is False, "not at the floor yet -> not given up"
    assert ideal[0] < 10.0 * DT_MIN, "should shrink the step for the retry"


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
