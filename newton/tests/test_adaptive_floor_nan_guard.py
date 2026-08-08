"""Contract tests for the adaptive step controller (`_calc_adjusted_step`).

Current contract at the ``dt_min`` floor, where the step can no longer be subdivided.
A world there with ``e > tol`` always ACCEPTS (advancing ``sim_time`` avoids a
boundary-loop hang) and pins ``ideal_dt`` to ``dt_min``; what differs is whether the
state is committed:

  * ``nan_guard == 1`` (default) -- a non-finite error, or one at/above the divergence
    sentinel, refuses the commit, latches ``diverged`` for the env to consume as a
    termination, and freezes the world at its boundary (``sim_time = next_time``).
    A large-but-FINITE error does not trigger it: that is a legitimately hard step, not
    a blow-up, and the constraint solve already runs at MuJoCo's default 1e-8 residual
    tolerance rather than leaning on a heuristic bound.
  * ``nan_guard == 0`` -- the legacy path: commit anyway, never latch.

Above the floor, a diverged (sentinel-error) world REJECTS and retries smaller -- the
NaN containment path (the error kernel flags NaN per component).

``force_accept`` is the quantile-stop escape hatch: it makes an otherwise-rejected step
accept unconditionally, so a world abandoned by the boundary loop still lands on its
boundary. It must never force a non-finite state through.

This is a pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.solver_mujoco_adaptive import (
    _calc_adjusted_step,
    _forced_scan_kernel,
    _restore_bad_rows_kernel,
)

wp.init()

DEV = "cpu"
TOL = 1.0e-3
DT_MIN = 1.0e-6
DIVERGENCE = 1.0e9  # threshold; the error kernel emits 1e10 for NaN/inf states
SENTINEL = 1.0e10  # what _inf_norm_state_error_kernel writes for a diverged world


def _run(err_vals, dt_vals, nan_guard=1, force_accept=0, order_aware=0, sliver_fix=0):
    """Launch the controller kernel over one synthetic batch at solver defaults."""
    n = len(err_vals)
    err = wp.array(np.asarray(err_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    dt = wp.array(np.asarray(dt_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    ideal = wp.zeros(n, dtype=wp.float32, device=DEV)
    accepted = wp.zeros(n, dtype=wp.bool, device=DEV)
    commit = wp.zeros(n, dtype=wp.bool, device=DEV)
    diverged = wp.zeros(n, dtype=wp.bool, device=DEV)
    limited = wp.zeros(n, dtype=wp.int32, device=DEV)
    consec_rej = wp.zeros(n, dtype=wp.int32, device=DEV)
    sim_time = wp.zeros(n, dtype=wp.float32, device=DEV)
    # next_time > sim_time: these worlds have NOT reached their boundary.
    next_time = wp.full(n, 1.0, dtype=wp.float32, device=DEV)
    force = wp.full(1, int(force_accept), dtype=wp.int32, device=DEV)
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
            DT_MIN,
            DIVERGENCE,
            limited,
            consec_rej,
            order_aware,
            sliver_fix,
            nan_guard,
            sim_time,
            next_time,
            force,
        ],
        device=DEV,
    )
    return accepted.numpy(), commit.numpy(), diverged.numpy(), ideal.numpy()


def test_floor_diverged_latches_under_nan_guard():
    """At the floor with a sentinel error and the NaN guard on (default): accept for
    progress, but REFUSE the commit and latch ``diverged`` so the env can reset it."""
    accepted, commit, diverged, ideal = _run([SENTINEL], [DT_MIN], nan_guard=1)
    assert bool(accepted[0]) is True, "must advance to avoid a boundary-loop hang"
    assert bool(commit[0]) is False, "a non-finite floor state must never be committed"
    assert bool(diverged[0]) is True, "the NaN guard must latch for env-side termination"
    assert abs(float(ideal[0]) - DT_MIN) < 1e-12, "floor step pins ideal_dt to dt_min"


def test_floor_diverged_commits_without_nan_guard():
    """The legacy path (NEWTON_ADAPTIVE_NAN_GUARD=0): commit anyway, never latch."""
    accepted, commit, diverged, _ = _run([SENTINEL], [DT_MIN], nan_guard=0)
    assert bool(accepted[0]) is True
    assert bool(commit[0]) is True, "with the guard off, floor worlds commit like any over-tol world"
    assert bool(diverged[0]) is False, "the latch is written only by the guard path"


def test_floor_large_finite_error_still_commits():
    """A large but FINITE error at the floor is a hard step, not a divergence: it must
    still commit. Only non-finiteness (or the divergence sentinel) trips the guard."""
    accepted, commit, diverged, _ = _run([1.0e4 * TOL], [DT_MIN], nan_guard=1)
    assert bool(accepted[0]) is True
    assert bool(commit[0]) is True, "a finite floor step must make committed progress"
    assert bool(diverged[0]) is False, "finite error is not a blow-up; do not latch"


def test_floor_finite_over_tol_commits_progress():
    """A merely over-tolerance finite error at the floor still commits real progress."""
    accepted, commit, diverged, _ = _run([10.0 * TOL], [DT_MIN], nan_guard=1)
    assert bool(accepted[0]) is True
    assert bool(commit[0]) is True, "finite floor step must still make committed progress"
    assert bool(diverged[0]) is False, "a finite error never latches the guard"


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


def test_force_accept_turns_a_rejection_into_an_accept():
    """The quantile stop abandons a world mid-flight; force_accept lands it anyway."""
    rej_a, _, _, _ = _run([100.0 * TOL], [10.0 * DT_MIN], force_accept=0)
    assert bool(rej_a[0]) is False, "baseline: this step is rejected"
    acc_a, acc_c, acc_d, _ = _run([100.0 * TOL], [10.0 * DT_MIN], force_accept=1)
    assert bool(acc_a[0]) is True, "force_accept must accept the step"
    assert bool(acc_c[0]) is True, "and commit it, so the world advances"
    assert bool(acc_d[0]) is False, "a finite forced step is not a divergence"


def test_force_accept_never_commits_a_non_finite_state():
    """force_accept must not launder a diverged world into the committed state."""
    _, commit, _, _ = _run([SENTINEL], [10.0 * DT_MIN], force_accept=1)
    assert bool(commit[0]) is False, "a non-finite step must never be force-committed"


def test_forced_completion_containment_restores_and_latches():
    """The production forced-completion path: a world whose forced eval went non-finite
    must be restored to its pre-forced snapshot and latched diverged; clean worlds must
    keep their forced state untouched. (The force_accept kernel tests above cover the
    controller branch; THIS is the path the quantile stop actually takes.)"""
    n, nq, nv = 4, 3, 2
    rng = np.random.default_rng(0)
    saved_q = rng.standard_normal((n, nq)).astype(np.float32)
    saved_v = rng.standard_normal((n, nv)).astype(np.float32)
    forced_q = saved_q + 0.5
    forced_v = saved_v + 0.5
    forced_q[2, 1] = np.nan  # world 2: forced eval blew up in qpos
    forced_v[3, 0] = np.inf  # world 3: blew up in qvel

    qpos = wp.array(forced_q, dtype=wp.float32)
    qvel = wp.array(forced_v, dtype=wp.float32)
    qpos_saved = wp.array(saved_q, dtype=wp.float32)
    qvel_saved = wp.array(saved_v, dtype=wp.float32)
    diverged = wp.zeros(n, dtype=wp.bool)
    bad = wp.zeros(n, dtype=wp.int32)

    wp.launch(_forced_scan_kernel, dim=n, inputs=[qpos, qvel, nq, nv, diverged, bad])
    for sv, out, width in ((qpos_saved, qpos, nq), (qvel_saved, qvel, nv)):
        wp.launch(_restore_bad_rows_kernel, dim=(n, width), inputs=[sv, bad], outputs=[out])
    wp.synchronize()

    q, v = qpos.numpy(), qvel.numpy()
    d, b = diverged.numpy(), bad.numpy()
    assert list(b) == [0, 0, 1, 1], f"bad mask wrong: {list(b)}"
    assert list(d) == [False, False, True, True], f"diverged latch wrong: {list(d)}"
    # poisoned worlds: fully restored (both fields, even if only one was non-finite)
    assert np.allclose(q[2], saved_q[2]) and np.allclose(v[2], saved_v[2])
    assert np.allclose(q[3], saved_q[3]) and np.allclose(v[3], saved_v[3])
    # clean worlds: forced state kept, snapshot NOT leaked back
    assert np.allclose(q[:2], forced_q[:2]) and np.allclose(v[:2], forced_v[:2])
    assert np.isfinite(q).all() and np.isfinite(v).all()


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
