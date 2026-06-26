"""Fix C: per-world controller reset for the adaptive step-doubling solver.

The adaptive solver keeps persistent per-world buffers (ideal_dt/dt/dt_half,
sim_time/next_time, accepted/diverged latches) that are NEVER restored on an
env/episode reset, so pre-reset controller state leaks into the post-reset
(s,a)->s' map and the diverged latch never clears.

``_reset_worlds`` restores those buffers to construction defaults for the
worlds flagged in ``mask`` and leaves the others untouched.

Pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

wp.init()

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _reset_worlds

DEV = "cpu"


def _run(mask_vals, dt_init, seeds):
    """seeds: dict of field -> list[val] for the N worlds (pre-launch buffers)."""
    n = len(mask_vals)
    mask = wp.array(np.asarray(mask_vals, dtype=bool), dtype=wp.bool, device=DEV)
    ideal_dt = wp.array(np.asarray(seeds["ideal_dt"], dtype=np.float32), dtype=wp.float32, device=DEV)
    dt = wp.array(np.asarray(seeds["dt"], dtype=np.float32), dtype=wp.float32, device=DEV)
    dt_half = wp.array(np.asarray(seeds["dt_half"], dtype=np.float32), dtype=wp.float32, device=DEV)
    sim_time = wp.array(np.asarray(seeds["sim_time"], dtype=np.float32), dtype=wp.float32, device=DEV)
    next_time = wp.array(np.asarray(seeds["next_time"], dtype=np.float32), dtype=wp.float32, device=DEV)
    diverged = wp.array(np.asarray(seeds["diverged"], dtype=bool), dtype=wp.bool, device=DEV)
    accepted = wp.array(np.asarray(seeds["accepted"], dtype=bool), dtype=wp.bool, device=DEV)
    wp.launch(
        _reset_worlds,
        dim=n,
        inputs=[mask, float(dt_init), ideal_dt, dt, dt_half, sim_time, next_time, diverged, accepted],
        device=DEV,
    )
    return {
        "ideal_dt": ideal_dt.numpy(),
        "dt": dt.numpy(),
        "dt_half": dt_half.numpy(),
        "sim_time": sim_time.numpy(),
        "next_time": next_time.numpy(),
        "diverged": diverged.numpy(),
        "accepted": accepted.numpy(),
    }


def _seed2():
    return {
        "ideal_dt": [0.005, 0.005],
        "dt": [0.005, 0.005],
        "dt_half": [0.0025, 0.0025],
        "sim_time": [12.3, 12.3],
        "next_time": [12.3, 12.3],
        "diverged": [True, True],
        "accepted": [True, True],
    }


def test_masked_world_restored_to_defaults():
    dt_init = 0.01
    out = _run([True, False], dt_init, _seed2())
    assert out["ideal_dt"][0] == np.float32(dt_init)
    assert out["dt"][0] == np.float32(dt_init)
    assert out["dt_half"][0] == np.float32(dt_init) * np.float32(0.5)
    assert out["sim_time"][0] == 0.0
    assert out["next_time"][0] == 0.0
    assert bool(out["diverged"][0]) is False
    assert bool(out["accepted"][0]) is False


def test_unmasked_world_untouched():
    s = _seed2()
    out = _run([True, False], 0.01, s)
    assert out["ideal_dt"][1] == np.float32(0.005)
    assert out["dt"][1] == np.float32(0.005)
    assert out["dt_half"][1] == np.float32(0.0025)
    assert out["sim_time"][1] == np.float32(12.3)
    assert out["next_time"][1] == np.float32(12.3)
    assert bool(out["diverged"][1]) is True
    assert bool(out["accepted"][1]) is True


def test_all_false_mask_is_noop():
    s = _seed2()
    out = _run([False, False], 0.01, s)
    for i in range(2):
        assert out["ideal_dt"][i] == np.float32(0.005)
        assert out["dt"][i] == np.float32(0.005)
        assert out["dt_half"][i] == np.float32(0.0025)
        assert out["sim_time"][i] == np.float32(12.3)
        assert out["next_time"][i] == np.float32(12.3)
        assert bool(out["diverged"][i]) is True
        assert bool(out["accepted"][i]) is True


def test_all_true_mask_resets_all():
    dt_init = 0.02
    seeds = {
        "ideal_dt": [0.005, 0.007, 0.009],
        "dt": [0.005, 0.007, 0.009],
        "dt_half": [0.0025, 0.0035, 0.0045],
        "sim_time": [1.0, 2.0, 3.0],
        "next_time": [1.0, 2.0, 3.0],
        "diverged": [True, True, True],
        "accepted": [True, True, True],
    }
    out = _run([True, True, True], dt_init, seeds)
    for i in range(3):
        assert out["ideal_dt"][i] == np.float32(dt_init)
        assert out["dt"][i] == np.float32(dt_init)
        assert out["dt_half"][i] == np.float32(dt_init) * np.float32(0.5)
        assert out["sim_time"][i] == 0.0
        assert out["next_time"][i] == 0.0
        assert bool(out["diverged"][i]) is False
        assert bool(out["accepted"][i]) is False


def test_dt_half_is_half_dt_init():
    dt_init = 0.01  # not a power of two in float32
    out = _run([True], dt_init, {k: [v[0]] for k, v in _seed2().items()})
    assert out["dt_half"][0] == np.float32(dt_init) * np.float32(0.5)


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
