"""Contract tests for the adaptive dt-occupancy histogram.

The bin-logic tests are pure-kernel: warp on CPU, no GPU / MuJoCo / scene needed,
mirroring test_adaptive_floor_nan_guard.py. The solver-integration tests added later
DO need CUDA (SolverMuJoCoAdaptive forces use_mujoco_cpu=False) and self-skip without it.

Contract:
  * bin 0 counts EXACT floor hits (dt <= dt_min); _apply_dt_cap's wp.clamp
    returns bitwise dt_min on clamp, so equality is reliable.
  * bins 1..n_bins-1 are log10-spaced, bins_per_decade per decade, with the
    last bin absorbing everything above the range.
  * worlds that already landed (sim_time >= next_time) are NOT counted -- they
    will not take the step being sampled.
  * the saturation scalar tracks min(ideal_dt) over floor-clamped worlds only.
"""

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.solver_mujoco_adaptive import (
    _dt_hist_edges,
    _dt_hist_layout,
    _dt_histogram_accum,
)
from newton.solvers import SolverMuJoCoAdaptive

wp.init()

DEV = "cpu"
DT_MIN = 1.0e-6
DT_INIT = 1.0e-2
BPD = 4
SENTINEL = 1.0e38


def _run(dt_vals, ideal_vals=None, landed=None):
    """Launch the kernel over one synthetic batch; return (counts, saturation)."""
    n = len(dt_vals)
    if ideal_vals is None:
        ideal_vals = dt_vals
    if landed is None:
        landed = [False] * n
    n_bins, lo_log10 = _dt_hist_layout(DT_MIN, DT_INIT, BPD)
    # A landed world has sim_time >= next_time.
    sim_time = np.zeros(n, dtype=np.float32)
    next_time = np.array([0.0 if x else 1.0 for x in landed], dtype=np.float32)
    counts = wp.zeros(n_bins, dtype=wp.int64, device=DEV)
    sat = wp.array(np.array([SENTINEL], dtype=np.float32), dtype=wp.float32, device=DEV)
    wp.launch(
        _dt_histogram_accum,
        dim=n,
        inputs=[
            wp.array(np.asarray(dt_vals, dtype=np.float32), dtype=wp.float32, device=DEV),
            wp.array(np.asarray(ideal_vals, dtype=np.float32), dtype=wp.float32, device=DEV),
            wp.array(sim_time, dtype=wp.float32, device=DEV),
            wp.array(next_time, dtype=wp.float32, device=DEV),
            DT_MIN,
            lo_log10,
            float(BPD),
            n_bins,
            counts,
            sat,
        ],
        device=DEV,
    )
    return counts.numpy(), float(sat.numpy()[0])


def test_layout_matches_defaults():
    """IsaacLab defaults give 5 decades -> 1 floor + 20 log + 1 overflow = 22 bins."""
    n_bins, lo_log10 = _dt_hist_layout(DT_MIN, DT_INIT, BPD)
    assert n_bins == 22, f"expected 22 bins, got {n_bins}"
    assert abs(lo_log10 - (-6.0)) < 1e-12, f"expected lo_log10=-6, got {lo_log10}"


def test_exact_floor_lands_in_bin_zero():
    counts, _ = _run([DT_MIN, DT_MIN, 5.0e-3])
    assert counts[0] == 2, f"expected 2 floor hits, got {counts[0]}"
    assert counts.sum() == 3, f"expected 3 total, got {counts.sum()}"


def test_above_floor_never_lands_in_bin_zero():
    """A dt one ulp above the floor is NOT a floor hit."""
    just_above = np.nextafter(np.float32(DT_MIN), np.float32(1.0))
    counts, _ = _run([just_above])
    assert counts[0] == 0, "value above the floor must not count as a floor hit"
    assert counts[1] == 1, f"expected it in bin 1, got bins {np.nonzero(counts)[0]}"


def test_landed_worlds_are_not_counted():
    """A world whose sim_time has reached next_time will not take this step."""
    counts, _ = _run([DT_MIN, 5.0e-3], landed=[False, True])
    assert counts.sum() == 1, f"landed world must be skipped, got total {counts.sum()}"
    assert counts[0] == 1


def test_bin_centers_round_trip():
    """Every log bin is hit exactly once when sampled at its geometric center.

    Bin CENTERS, not edges: a float32 value equal to an edge can fall either
    side of the float64 edge (np.float32(1e-5) < 1.0000000000000002e-05), which
    makes edge-valued samples ambiguous by construction.
    """
    n_bins, _ = _dt_hist_layout(DT_MIN, DT_INIT, BPD)
    edges = _dt_hist_edges(DT_MIN, n_bins, BPD)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean of each bin
    counts, _ = _run(centers.astype(np.float32))
    for b in range(1, len(centers) + 1):
        assert counts[b] == 1, f"bin {b} got {counts[b]}, expected 1"


def test_overflow_bin_absorbs_large_dt():
    n_bins, _ = _dt_hist_layout(DT_MIN, DT_INIT, BPD)
    counts, _ = _run([1.0, 1.0e6])
    assert counts[n_bins - 1] == 2, f"expected 2 in overflow bin, got {counts[n_bins - 1]}"


def test_saturation_tracks_min_ideal_dt_at_floor_only():
    """Saturation depth reads ideal_dt (which is preserved below the floor), and
    only for worlds actually clamped to the floor."""
    counts, sat = _run(
        dt_vals=[DT_MIN, DT_MIN, 5.0e-3],
        ideal_vals=[1.0e-9, 3.0e-8, 1.0e-12],  # the third is NOT at the floor
    )
    assert counts[0] == 2
    assert abs(sat - 1.0e-9) < 1e-15, f"expected saturation 1e-9, got {sat}"


def test_saturation_stays_sentinel_when_floor_never_hit():
    """SENTINEL is compared via its float32 round-trip, not the double literal:
    1.0e38 is not exactly representable in float32 (unlike test_adaptive_floor_nan_guard's
    SENTINEL = 1.0e10, where 5**10 fits the 24-bit mantissa), so the wp.array construction
    in _run already rounds it -- the untouched value is that rounded value, not 1.0e38."""
    _, sat = _run([5.0e-3, 8.0e-3])
    expected = float(np.float32(SENTINEL))
    assert sat == expected, f"expected untouched sentinel, got {sat}"


def test_bins_per_decade_below_one_raises():
    """dt_histogram_bins_per_decade <= 0 would ZeroDivisionError in _dt_hist_edges and
    produce degenerate binning; reject it at construction, mirroring how max_substeps
    is already validated. This check runs before the GPU-only MuJoCo-Warp setup in
    __init__, so it needs no CUDA device."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.1)
    builder.end_world()
    builder.add_ground_plane()
    model = builder.finalize()
    try:
        SolverMuJoCoAdaptive(model, dt_histogram=True, dt_histogram_bins_per_decade=0)
    except ValueError:
        return
    raise AssertionError("dt_histogram_bins_per_decade=0 should raise ValueError")


_GPU = wp.get_cuda_device_count() > 0


def _skip_without_gpu(name: str) -> bool:
    """SolverMuJoCoAdaptive forces use_mujoco_cpu=False, so these need a CUDA device."""
    if not _GPU:
        print(f"SKIP {name}: no CUDA device")
        return True
    return False


def _one_sphere_solver(**solver_kwargs):
    """Minimal falling-sphere scene + adaptive solver; returns (solver, s0, s1, control).

    Bodies must be added inside a ``begin_world()``/``end_world()`` context -- the
    global world (-1) may not contain bodies. ``add_body()`` already creates the
    free joint connecting the body to the world, so no separate ``add_joint_free()``
    call is needed (that would double up into an unsupported loop joint).
    """
    builder = newton.ModelBuilder()
    builder.begin_world()
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(b, radius=0.1)
    builder.end_world()
    builder.add_ground_plane()
    model = builder.finalize()

    solver = SolverMuJoCoAdaptive(model, **solver_kwargs)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    return solver, s0, s1, control


def test_disabled_by_default_exposes_nothing():
    """Off by default so benchmark timings stay uncontaminated."""
    if _skip_without_gpu("test_disabled_by_default_exposes_nothing"):
        return
    solver, _, _, _ = _one_sphere_solver(dt_inner_init=1e-3, dt_inner_min=1e-5)
    assert solver.dt_histogram is None
    assert solver.dt_histogram_edges is None
    try:
        solver.dt_histogram_stats()
    except RuntimeError:
        pass
    else:
        raise AssertionError("dt_histogram_stats() must raise RuntimeError when disabled")


def test_enabled_accumulates_over_a_boundary():
    """With the histogram on, a real boundary call records one sample per attempt."""
    if _skip_without_gpu("test_enabled_accumulates_over_a_boundary"):
        return
    solver, s0, s1, control = _one_sphere_solver(dt_inner_init=1e-3, dt_inner_min=1e-5, dt_histogram=True)
    solver.reset_dt_histogram()
    solver.reset_compute_counter()
    s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)

    counts = solver.dt_histogram.numpy()
    assert counts.sum() > 0, "histogram recorded nothing over a boundary call"
    edges = solver.dt_histogram_edges
    assert len(edges) == len(counts) - 1, "edges must be one shorter than bins"
    stats = solver.dt_histogram_stats()
    assert stats["total_samples"] == int(counts.sum())
    # Cross-check against the solver's own iteration counter: for a single-world scene
    # every counted iteration attempts exactly one step, and the histogram kernel's
    # sim_time >= next_time skip predicate should agree exactly with what the boundary
    # loop itself counted as an iteration (_iter_count_increment / _cum_iters).
    assert stats["total_samples"] == int(solver.cumulative_iterations.numpy()[0]), (
        f"total_samples ({stats['total_samples']}) must equal cumulative_iterations "
        f"({int(solver.cumulative_iterations.numpy()[0])}) for a single-world scene"
    )


def test_histogram_samples_before_boundary_clamp():
    """Load-bearing placement: ``_dt_histogram_accum`` must launch BEFORE
    ``_clamp_dt_to_boundary`` in ``_run_iteration_body``. After the clamp, ``dt`` may
    instead hold a boundary-landing sliver, binning the sample into the wrong bucket.

    Pin the controller's step by setting ``dt_inner_max == dt_inner_init``:
    ``effective_dt_max = min(dt_inner_max, dt_outer) == dt_inner_init``, so every
    iteration's ``_apply_dt_cap`` clamps ``dt`` back to exactly ``dt_inner_init``
    regardless of how the controller wants to grow it, and a smooth free-fall at the
    default tol never rejects (so dt never shrinks below the pin either). Every
    iteration therefore ATTEMPTS the identical dt -- except the final iteration of each
    boundary, whose dt is instead a landing sliver truncated by the boundary target,
    strictly smaller than the pin.

    With the launch correctly BEFORE the clamp, every sample records the pinned
    dt_inner_init, landing all samples in exactly one bin. With the launch moved to
    AFTER the clamp (mutation), the landing-sliver iterations record a smaller value,
    splitting the samples across (at least) two bins -- see task-45-report.md for the
    measured counts both ways.
    """
    if _skip_without_gpu("test_histogram_samples_before_boundary_clamp"):
        return
    solver, s0, s1, control = _one_sphere_solver(
        dt_inner_init=1e-3, dt_inner_min=1e-5, dt_inner_max=1e-3, dt_histogram=True
    )
    solver.reset_dt_histogram()
    solver.reset_compute_counter()
    for _ in range(10):
        s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)

    counts = solver.dt_histogram.numpy()
    nonzero_bins = np.count_nonzero(counts)
    assert nonzero_bins == 1, (
        f"expected all samples in exactly one bin (dt pinned at dt_inner_init via "
        f"dt_inner_max == dt_inner_init), got {nonzero_bins} nonzero bins: {counts}"
    )
    stats = solver.dt_histogram_stats()
    assert stats["total_samples"] == int(solver.cumulative_iterations.numpy()[0]), (
        f"total_samples ({stats['total_samples']}) must equal cumulative_iterations "
        f"({int(solver.cumulative_iterations.numpy()[0])}) for a single-world scene"
    )


def test_reset_zeroes_the_accumulators():
    if _skip_without_gpu("test_reset_zeroes_the_accumulators"):
        return
    solver, s0, s1, control = _one_sphere_solver(dt_inner_init=1e-3, dt_inner_min=1e-5, dt_histogram=True)
    s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)
    assert solver.dt_histogram.numpy().sum() > 0

    solver.reset_dt_histogram()
    assert solver.dt_histogram.numpy().sum() == 0
    assert solver.dt_histogram_stats()["saturation_depth"] == 0.0


def test_truncation_counters_fire_when_capped():
    """max_substeps=1 cannot cross a 1/120 s boundary at dt=1e-3, so every
    boundary truncates and every world ends short of its target time."""
    if _skip_without_gpu("test_truncation_counters_fire_when_capped"):
        return
    solver, s0, s1, control = _one_sphere_solver(
        dt_inner_init=1e-3, dt_inner_min=1e-5, max_substeps=1, dt_histogram=True
    )
    solver.reset_dt_histogram()
    for _ in range(3):
        s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)

    stats = solver.dt_histogram_stats()
    assert stats["boundaries"] == 3, f"expected 3 boundaries, got {stats['boundaries']}"
    assert stats["capped_boundaries"] == 3, f"expected 3 capped, got {stats['capped_boundaries']}"
    assert stats["unfinished_worlds"] == 3, f"expected 3 world-boundaries short, got {stats['unfinished_worlds']}"


def test_truncation_counters_stay_zero_when_uncapped():
    """A generous cap lets every boundary complete."""
    if _skip_without_gpu("test_truncation_counters_stay_zero_when_uncapped"):
        return
    solver, s0, s1, control = _one_sphere_solver(
        dt_inner_init=1e-3, dt_inner_min=1e-5, max_substeps=256, dt_histogram=True
    )
    solver.reset_dt_histogram()
    for _ in range(3):
        s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)

    stats = solver.dt_histogram_stats()
    assert stats["boundaries"] == 3
    assert stats["capped_boundaries"] == 0, f"unexpected truncation: {stats}"
    assert stats["unfinished_worlds"] == 0, f"worlds fell short: {stats}"


def test_saturation_depth_nonzero_when_floor_is_hit():
    """Public-API coverage for the non-zero saturation_depth branch of
    dt_histogram_stats(): a very tight tol (1e-8) rejects nearly every attempt even
    during smooth free fall, and dt_inner_min close to dt_inner_init leaves almost no
    room to shrink into before the controller clamps to the floor. Verified empirically
    (see task-3-report.md) -- floor_samples > 0 within the very first boundary, with
    ideal_dt shrinking well below dt_inner_min once clamped. saturation_depth must
    report that true minimum ideal_dt, not 0.0 (which means "floor never hit")."""
    if _skip_without_gpu("test_saturation_depth_nonzero_when_floor_is_hit"):
        return
    solver, s0, s1, control = _one_sphere_solver(
        dt_inner_init=1e-3, dt_inner_min=9.9e-4, max_substeps=256, dt_histogram=True, tol=1e-8
    )
    solver.reset_dt_histogram()
    for _ in range(3):
        s0, s1 = solver.step_dt(1.0 / 120.0, s0, s1, control)

    stats = solver.dt_histogram_stats()
    assert stats["floor_samples"] > 0, f"expected the floor to be hit, got {stats}"
    assert 0.0 < stats["saturation_depth"] < 9.9e-4, (
        f"expected a positive saturation_depth below dt_inner_min, got {stats['saturation_depth']}"
    )


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
