"""Fix B: float32 time-rebase for the adaptive step-doubling controller.

``_sim_time`` and ``_next_time`` are never reset and grow unbounded across a
training run; the landing remainder ``next_time - sim_time`` then loses float32
precision as the magnitude grows, causing dt jitter that worsens over time.

The fix is a per-world rebase kernel ``_rebase_time`` that subtracts each
world's boundary baseline ``next_time[i]`` from BOTH clocks (NOT a zero of
sim_time, which would silently drop the >= 0 residual overshoot). This keeps
both clocks small while preserving ``remaining = next_time - sim_time``
bit-exactly and carrying the residual overshoot forward.

Pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

wp.init()

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _rebase_time

DEV = "cpu"


def _run(sim_vals, next_vals):
    n = len(sim_vals)
    sim = wp.array(np.asarray(sim_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    nxt = wp.array(np.asarray(next_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    wp.launch(_rebase_time, dim=n, inputs=[sim, nxt], device=DEV)
    return sim.numpy(), nxt.numpy()


def test_next_time_zeroed():
    """After rebase, next_time is exactly 0 for every world."""
    _, nxt_after = _run([1.0e6, 5.0], [1.0e6, 5.0])
    assert np.all(nxt_after == 0.0), f"next_time not zeroed: {nxt_after}"


def test_remaining_preserved_exactly():
    """remaining = next_time - sim_time is preserved bit-exactly across the rebase."""
    nxt_before = np.float32(1048576.0)  # 2^20
    sim_before = np.float32(1048576.0) - np.float32(1.0e-3)
    sim_after, nxt_after = _run([sim_before], [nxt_before])
    remaining_before = np.float32(nxt_before) - np.float32(sim_before)
    remaining_after = np.float32(nxt_after[0]) - np.float32(sim_after[0])
    assert remaining_after == remaining_before, (
        f"remaining changed: before={remaining_before!r} after={remaining_after!r}"
    )


def test_sim_becomes_residual():
    """sim_time becomes the residual overshoot sim - next (carried forward, not dropped)."""
    sim_vals = [1.0e6, 5.0 + 1.0e-4]
    next_vals = [1.0e6, 5.0]
    sim_after, _ = _run(sim_vals, next_vals)
    for i in range(len(sim_vals)):
        expected = np.float32(np.float32(sim_vals[i]) - np.float32(next_vals[i]))
        assert sim_after[i] == expected, f"world {i}: {sim_after[i]!r} != {expected!r}"


def test_first_step_noop():
    """sim=next=0 -> both stay exactly 0 (first step_dt is a no-op rebase)."""
    sim_after, nxt_after = _run([0.0], [0.0])
    assert sim_after[0] == 0.0 and nxt_after[0] == 0.0


def test_overshoot_carried_forward():
    """A world slightly past its boundary keeps the >0 overshoot in sim_time."""
    sim_after, nxt_after = _run([10.0 + 2.0e-6], [10.0])
    assert nxt_after[0] == 0.0
    expected = np.float32(np.float32(10.0 + 2.0e-6) - np.float32(10.0))
    assert sim_after[0] == expected, f"{sim_after[0]!r} != {expected!r}"
    assert sim_after[0] > 0.0


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
