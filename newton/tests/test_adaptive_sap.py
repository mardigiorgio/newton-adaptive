"""SolverSAPAdaptive integration smoke (CPU): the step-doubling SAP solver.

Builds the Newton cartpole, constructs SolverSAPAdaptive, and checks:
  (1) constructs on cartpole;
  (2) step_dt advances sim_time to the dt_outer boundary;
  (3) result is finite and step_dt is deterministic (warm-start save/restore
      makes the 3 evals independent -> repeating step_dt from the same state
      reproduces joint_q bit-identically);
  (4) a tighter tol uses >= as many cumulative substeps (more subdivision).

Warp runs device="cpu"; no GPU / MuJoCo needed. Mirrors the style of the other
test_adaptive_*.py files (prints PASS/FAIL, exits nonzero on failure).
"""

import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.environ.get("SAP_WARP_PATH", "/home/mdigiorgio/Documents/code/sap_warp"))

wp.init()

import newton
import newton.examples
from newton.solvers import SolverSAPAdaptive

DEV = "cpu"
DT_OUTER = 1.0 / 60.0


def build_cartpole(device, worlds=1):
    cartpole = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(cartpole)
    cartpole.default_shape_cfg.density = 100.0
    cartpole.default_joint_cfg.armature = 0.1
    cartpole.add_usd(
        newton.examples.get_asset("cartpole.usda"), enable_self_collisions=False, collapse_fixed_joints=True
    )
    for body in range(cartpole.body_count):
        inertia = np.asarray(cartpole.body_inertia[body], dtype=np.float32).reshape(3, 3)
        inertia += np.eye(3, dtype=np.float32) * 0.1
        cartpole.body_inertia[body] = wp.mat33(inertia)
    cartpole.joint_q[-3:] = [0.0, 0.3, 0.0]
    builder = newton.ModelBuilder()
    builder.replicate(cartpole, worlds, spacing=(1.0, 2.0, 0.0))
    return builder.finalize(device=device)


def _fresh(device, worlds=1, **kw):
    model = build_cartpole(device, worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    state_1.assign(state_0)
    solver = SolverSAPAdaptive(model, **kw)
    return model, state_0, state_1, control, solver


def test_constructs():
    device = wp.get_device(DEV)
    _, _, _, _, solver = _fresh(device, tol=1e-3)
    assert solver.diverged is not None
    assert solver.dt is not None
    assert solver.tiling == "adaptive"
    assert solver.mode == "adaptive"


def test_step_dt_reaches_boundary():
    device = wp.get_device(DEV)
    _, s0, s1, control, solver = _fresh(device, tol=1e-3)
    solver.step_dt(DT_OUTER, s0, s1, control)
    st = solver.sim_time.numpy()
    assert np.allclose(st, DT_OUTER, rtol=1e-4, atol=1e-7), f"sim_time={st} != {DT_OUTER}"


def test_finite_and_deterministic():
    device = wp.get_device(DEV)
    _, s0, s1, control, solver = _fresh(device, tol=1e-3)
    q_before = s0.joint_q.numpy().copy()
    solver.step_dt(DT_OUTER, s0, s1, control)
    q_a = s0.joint_q.numpy().copy()
    assert np.all(np.isfinite(q_a)), "joint_q not finite after step_dt"
    assert not np.array_equal(q_a, q_before), "joint_q did not advance"

    # Re-run step_dt from the SAME fresh state -> bit-identical (warm-start
    # save/restore makes the trio independent and the step reproducible).
    _, s0b, s1b, controlb, solverb = _fresh(device, tol=1e-3)
    solverb.step_dt(DT_OUTER, s0b, s1b, controlb)
    q_b = s0b.joint_q.numpy().copy()
    assert np.array_equal(q_a, q_b), f"step_dt not deterministic; max diff={np.max(np.abs(q_a - q_b)):.3e}"


def test_tighter_tol_more_substeps():
    device = wp.get_device(DEV)

    # Drive a few frames so the controller settles, count substeps for loose vs tight tol.
    def run(tol):
        _, s0, s1, control, solver = _fresh(device, tol=tol, dt_inner_init=DT_OUTER)
        solver.reset_compute_counter()
        for _ in range(5):
            solver.step_dt(DT_OUTER, s0, s1, control)
        return solver.cumulative_substeps()

    loose = run(1e-2)
    tight = run(1e-6)
    assert tight >= loose, f"tighter tol used fewer substeps: tight={tight} < loose={loose}"


def test_invalid_mode_raises():
    device = wp.get_device(DEV)
    try:
        _fresh(device, mode="bogus")
    except ValueError:
        return
    raise AssertionError("mode='bogus' should raise ValueError")


def test_fixed_mode_reaches_boundary_finite():
    # fixed: constant dt, error control off; still marches to the boundary and stays finite.
    device = wp.get_device(DEV)
    _, s0, s1, control, solver = _fresh(device, mode="fixed", dt_inner_init=DT_OUTER / 4)
    assert solver.mode == "fixed"
    solver.step_dt(DT_OUTER, s0, s1, control)
    st = solver.sim_time.numpy()
    q = s0.joint_q.numpy()
    assert np.all(np.isfinite(q)), "fixed mode produced non-finite joint_q"
    assert np.allclose(st, DT_OUTER, rtol=1e-4, atol=1e-7), f"fixed sim_time={st} != {DT_OUTER}"


def test_fixed_mode_work_constant_per_frame():
    # fixed dt does NOT adapt: the per-frame work (accepted substeps) is identical frame to frame.
    device = wp.get_device(DEV)
    _, s0, s1, control, solver = _fresh(device, mode="fixed", dt_inner_init=DT_OUTER / 4)
    solver.reset_compute_counter()
    solver.step_dt(DT_OUTER, s0, s1, control)
    w1 = int(solver.substeps.numpy().sum())
    solver.step_dt(DT_OUTER, s0, s1, control)
    w2 = int(solver.substeps.numpy().sum())
    assert w1 > 0, "fixed mode did no work"
    assert w2 == w1, f"fixed mode work changed across frames: {w1} then {w2}"


def test_landed_fraction_validation():
    """landed_fraction must lie in (0, 1]."""
    device = wp.get_device(DEV)
    for bad in (0.0, -0.1, 1.5):
        try:
            _fresh(device, tol=1e-3, landed_fraction=bad)
        except ValueError:
            continue
        raise AssertionError(f"landed_fraction={bad} should have raised ValueError")


def test_quantile_stop_leaves_no_world_behind():
    """The quantile stop abandons the slow tail, but forced completion must still land
    EVERY world exactly on the boundary -- degraded accuracy is acceptable, a world at the
    wrong simulation time is not."""
    device = wp.get_device(DEV)
    for frac in (1.0, 0.5):
        _, s0, s1, control, solver = _fresh(device, worlds=8, tol=1e-3, landed_fraction=frac)
        for _ in range(3):
            solver.step_dt(DT_OUTER, s0, s1, control)
        # sim_time is rebased per boundary, so compare against next_time rather than
        # against an accumulated wall time.
        behind = solver.sim_time.numpy() - solver._next_time.numpy()
        assert np.all(behind >= -1e-6), (
            f"landed_fraction={frac}: worlds left short of the boundary by {behind.min():.3e} s"
        )


def test_quantile_stop_default_is_inactive():
    """landed_fraction defaults to 1.0, i.e. the mechanism is off unless asked for."""
    device = wp.get_device(DEV)
    _, _, _, _, solver = _fresh(device, tol=1e-3)
    assert solver._quantile_stop.enabled is False
    assert solver._quantile_stop.max_unfinished == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback

            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
