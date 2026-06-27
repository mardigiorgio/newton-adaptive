"""CENIC (sec VI-D) cubic initial guess for the SAP exact-root line search.

The exact-root line search (Newton-Raphson root find of ell'(alpha)=0) seeds its
bracket with an initial alpha. Refs [16],[17] use a QUADRATIC model
(ell(0), ell'(0), ell''(0)); CENIC instead uses a CUBIC Hermite model from
(ell(0), ell'(0), ell(alpha_max), ell'(alpha_max)) -- Nocedal & Wright eq 3.59 on
[0, alpha_max]. Implemented in sim/contact_solve.py:_init_sap_exact_root_state,
gated by SAP_CUBIC_INIT (default cubic).

Two checks:
  (1) the cubic-minimizer ALGEBRA on a known cubic whose minimizer is analytic;
  (2) exact_root + cubic converges to a finite state on the Allegro-grasp oracle.

Standalone (mirrors the other test_adaptive_*.py): prints PASS/FAIL, exits nonzero
on failure. The oracle test prefers cuda:0 and skips if the asset is unavailable.
"""

import math
import os
import sys

import numpy as np
import warp as wp

sys.path.insert(0, os.environ.get("SAP_WARP_PATH", "/home/mdigiorgio/Documents/code/sap_warp"))

wp.init()

import newton
import newton.examples
from newton.solvers import SolverSAPAdaptive

DT_OUTER = 1.0 / 60.0


def cubic_min(c0, d0, c1, d1, h):
    """Minimizer in (0, h] of the Hermite cubic through (0,c0,d0) and (h,c1,d1).

    Pure-Python mirror of the Warp kernel arithmetic (N&W eq 3.59). Returns None on
    the same fallback conditions the kernel uses (disc<0 / non-finite / out-of-range),
    where the kernel falls back to the quadratic guess.
    """
    theta = d0 + d1 - 3.0 * (c1 - c0) / h
    disc = theta * theta - d0 * d1
    if not math.isfinite(disc) or disc < 0.0:
        return None
    gamma = math.sqrt(disc)
    denom = d1 - d0 + 2.0 * gamma
    if denom == 0.0:
        return None
    alpha = h - h * (d1 + gamma - theta) / denom
    if not math.isfinite(alpha) or alpha <= 0.0 or alpha > h:
        return None
    return alpha


def test_cubic_minimizer_known_cubic():
    # q(a) = a^3 - 1.05 a^2 + 0.3 a, q'(a) = 3(a-0.2)(a-0.5):
    # local max at 0.2, local min at 0.5; on [0,1] (and [0,1.5]) the minimizer is 0.5.
    q = lambda a: a**3 - 1.05 * a**2 + 0.3 * a
    qp = lambda a: 3.0 * a**2 - 2.1 * a + 0.3
    for h in (1.0, 1.5):
        got = cubic_min(q(0.0), qp(0.0), q(h), qp(h), h)
        assert got is not None and abs(got - 0.5) < 1e-12, f"h={h}: expected 0.5, got {got}"


def test_cubic_minimizer_degenerate_falls_back():
    # d0=d1=-1, c0=0, c1=-1, h=1 -> theta=1, disc=0, denom=0 -> fallback (None).
    assert cubic_min(0.0, -1.0, -1.0, -1.0, 1.0) is None


def build_allegro_grasp(device):
    from newton import JointTargetMode
    asset_path = newton.utils.download_asset("wonik_allegro")
    f = str(asset_path / "usd" / "allegro_left_hand_with_cube.usda")
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke = 1.0e3
    b.default_shape_cfg.kd = 1.0e2
    b.default_shape_cfg.margin = 0.005
    b.default_shape_cfg.gap = 0.015
    b.add_usd(f, xform=wp.transform(wp.vec3(0, 0, 0.5)), enable_self_collisions=False,
              ignore_paths=[".*Dummy", ".*CollisionPlane"], hide_collision_shapes=True)
    for i in range(b.joint_dof_count - 6):
        b.joint_target_ke[i] = 150
        b.joint_target_kd[i] = 5
        b.joint_q[i] = 0.3
        b.joint_target_q[i] = 0.3
        if b.joint_label[i][-2:] == "_0":
            b.joint_q[i] = 0.6
            b.joint_target_q[i] = 0.6
        b.joint_target_mode[i] = int(JointTargetMode.POSITION)
        if b.joint_type[i] == newton.JointType.REVOLUTE:
            b.joint_armature[i] = 1e-2
    q = np.array(b.joint_q)
    q[-7:-4] += np.array([0.0, 0.0, 0.05])
    q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
    b.joint_q = q.tolist()
    builder = newton.ModelBuilder()
    builder.replicate(b, 2)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize(device=device), np.array(b.joint_target_q, dtype=np.float32)


def test_exact_root_cubic_converges_finite_on_oracle():
    if os.environ.get("SAP_CUBIC_INIT", "1") == "0":
        print("  (SAP_CUBIC_INIT=0: quadratic seed -- still expect finite convergence)")
    device = wp.get_device("cuda:0") if wp.is_cuda_available() else wp.get_device("cpu")
    try:
        model, tgt = build_allegro_grasp(device)
    except Exception as e:  # asset unavailable (offline CI): skip, do not fail
        print(f"  (skip oracle: {e})")
        return
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s1.assign(s0)
    control.joint_target_q = wp.array(np.tile(tgt, 2), dtype=wp.float32, device=device)
    solver = SolverSAPAdaptive(
        model, mode="adaptive", tol=1e-3, dt_inner_init=0.01, dt_inner_min=1e-6,
        max_substeps=256, max_iterations=200, line_search_variant="exact_root",
    )
    cs = solver._sap.contact_solve
    for _ in range(12):
        solver.step_dt(DT_OUTER, s0, s1, control)
    q = s0.joint_q.numpy()
    assert np.all(np.isfinite(q)), "exact_root+cubic produced non-finite joint_q"
    peak_newton = int(cs.newton_iterations_env.numpy().max())
    assert peak_newton >= 1, f"contact solve did not run the Newton loop (peak={peak_newton})"
    print(f"  oracle OK on {device}: peak_newton={peak_newton}, q finite")


if __name__ == "__main__":
    failures = 0
    for name in sorted(k for k in dict(globals()) if k.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {name}: {e}")
    if failures:
        print(f"{failures} FAILED")
        sys.exit(1)
    print("ALL PASS")
