"""Fix 4: even-tiling N-selection for the adaptive step-doubling controller.

Pick the substep COUNT N adaptively from the carried controller dt, then tile the
control interval with a UNIFORM inner dt = dt_outer / N (no ragged remainder).
Pure-kernel contract test: warp on CPU, no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

wp.init()

from newton._src.solvers.mujoco.solver_mujoco_adaptive import _select_even_dt, _set_uniform_even_dt

DEV = "cpu"
DT_OUTER = 1.0 / 60.0
DT_MIN = 1e-6
DT_MAX = DT_OUTER  # effective_dt_max = min(dt_max, dt_outer)


# A cap large enough to never bind for the legacy math tests (preserves old expectations).
NO_CAP = 1_000_000_000


def _select(ideal_vals, dt_outer=DT_OUTER, dt_min=DT_MIN, dt_max=DT_MAX, max_substeps=NO_CAP):
    n = len(ideal_vals)
    ideal = wp.array(np.asarray(ideal_vals, dtype=np.float32), dtype=wp.float32, device=DEV)
    N = wp.zeros(n, dtype=wp.int32, device=DEV)
    dt = wp.zeros(n, dtype=wp.float32, device=DEV)
    dt_half = wp.zeros(n, dtype=wp.float32, device=DEV)
    wp.launch(
        _select_even_dt,
        dim=n,
        inputs=[ideal, dt_outer, dt_min, dt_max, max_substeps, N, dt, dt_half],
        device=DEV,
    )
    return N.numpy(), dt.numpy(), dt_half.numpy()


def test_N_is_ceil_of_ratio():
    # ideal_dt chosen so dt_outer/ideal_dt = 3.5 -> ceil -> N = 4
    ideal = DT_OUTER / 3.5
    N, dt, _ = _select([ideal])
    assert N[0] == 4, f"expected N=4, got {N[0]}"
    assert np.isclose(dt[0], DT_OUTER / 4, rtol=1e-6)


def test_dt_equals_dt_outer_over_N():
    N, dt, _ = _select([DT_OUTER / 3.5, DT_OUTER / 10.2, DT_OUTER / 1.1])
    for i in range(len(N)):
        assert np.isclose(dt[i], DT_OUTER / float(N[i]), rtol=1e-6)
        assert np.isclose(dt[i] * N[i], DT_OUTER, rtol=1e-5)  # tiles to dt_outer


def test_N_at_least_one():
    N, dt, _ = _select([1.0e9, DT_OUTER, DT_OUTER * 0.5])  # all >= dt_outer after clamp
    assert np.all(N >= 1)


def test_free_motion_N_is_one():
    # large ideal_dt (free motion) -> clamp to dt_max=dt_outer -> N=1, dt=dt_outer
    N, dt, _ = _select([1.0e9])
    assert N[0] == 1
    assert np.isclose(dt[0], DT_OUTER, rtol=1e-6)


def test_stiff_motion_N_large():
    # tiny ideal_dt (stiff) -> clamp to dt_min -> N = ceil(dt_outer / dt_min)
    N, dt, _ = _select([1.0e-12])
    expected = int(np.ceil(DT_OUTER / DT_MIN))
    assert N[0] == expected, f"expected N={expected}, got {N[0]}"


def test_clamp_below_min_and_above_max():
    N, _, _ = _select([1.0e-12, 1.0e9])
    assert N[0] == int(np.ceil(DT_OUTER / DT_MIN))
    assert N[1] == 1


def test_dt_half_is_half_dt():
    N, dt, dt_half = _select([DT_OUTER / 3.5, DT_OUTER / 7.0])
    assert np.allclose(dt_half, dt * 0.5, rtol=1e-6)


def test_N_capped_at_max_substeps():
    # A world driven to the dt_min floor would need N = ceil(dt_outer/dt_min) ~ 16667;
    # the cap MUST bound it (else the per-world fixed loop grinds ~16k launches/frame -> hang).
    cap = 256
    N, dt, _ = _select([1.0e-12], max_substeps=cap)
    assert N[0] == cap, f"expected N capped at {cap}, got {N[0]}"
    assert np.isclose(dt[0], DT_OUTER / cap, rtol=1e-6)


def test_cap_does_not_bind_when_N_small():
    # When the natural N is below the cap, the cap is inert (adaptivity preserved).
    cap = 256
    N, dt, _ = _select([DT_OUTER / 3.5], max_substeps=cap)
    assert N[0] == 4, f"cap must not lower a small N; got {N[0]}"
    assert np.isclose(dt[0], DT_OUTER / 4, rtol=1e-6)


def test_global_uses_max_N():
    # per-world N differ; global broadcasts the worst-case (max) N to all worlds
    ideal_vals = [DT_OUTER / 2.0, DT_OUTER / 9.3, DT_OUTER / 4.0]
    N, _, _ = _select(ideal_vals)
    n_max = int(N.max())
    nworlds = len(ideal_vals)
    N2 = wp.array(N, dtype=wp.int32, device=DEV)
    dt2 = wp.zeros(nworlds, dtype=wp.float32, device=DEV)
    dt_half2 = wp.zeros(nworlds, dtype=wp.float32, device=DEV)
    wp.launch(_set_uniform_even_dt, dim=nworlds, inputs=[n_max, DT_OUTER, N2, dt2, dt_half2], device=DEV)
    assert np.all(N2.numpy() == n_max)
    assert np.allclose(dt2.numpy(), DT_OUTER / n_max, rtol=1e-6)


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
