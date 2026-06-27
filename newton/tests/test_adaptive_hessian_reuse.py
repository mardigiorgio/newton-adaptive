"""TDD: Hessian-factorization reuse in the inner SAP solve (CENIC Sec. V-C).

CENIC reuses a single Hessian factorization across Newton iterations (chord
steps, linear convergence) and refactorizes only when eq. (35) predicts the
iterate would stall. Our SAP currently refactorizes from scratch every Newton
iteration (contact_solve.py: _run_unit_conditional_newton_step -> factorize_masked).

Cycle 1 (this file): the inner SAP solver must COUNT every Hessian factorization,
so reuse can be measured. With reuse OFF, a single fixed-mode convex solve
factorizes exactly once per Newton iteration, so the factorization count equals
the env's Newton-iteration count.

Standalone script (CPU; no GPU/MuJoCo). Mirrors the other test_adaptive_*.py:
prints PASS/FAIL per test, exits nonzero on any failure.
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


def build_contact_scene(device, n=4, radius=0.1, penetration=0.02, impact_vz=-5.0):
    """Several spheres impacting a ground plane -> persistent normal contacts so the
    inner SAP convex solve actually runs its Newton loop (the cartpole has no contact,
    so its solve does zero iterations and factorizes nothing)."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.density = 1000.0
    for i in range(n):
        body = b.add_body()
        b.add_joint_free(child=body)
        b.add_shape_sphere(body, radius=radius)
        b.joint_q[-7:] = [0.3 * i, 0.0, radius - penetration, 0.0, 0.0, 0.0, 1.0]
        b.joint_qd[-6:] = [0.0, 0.0, 0.0, 0.0, 0.0, impact_vz]
    b.add_ground_plane()
    return b.finalize(device=device)


def _single_solve(device, **solver_kw):
    model = build_contact_scene(device)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s1.assign(s0)
    # fixed mode + dt_inner_init == dt_outer + max_substeps=1 => exactly ONE inner
    # convex solve, so the factorization count is for a single solve, not a sum.
    solver_kw.setdefault("max_iterations", 80)
    solver = SolverSAPAdaptive(model, mode="fixed", dt_inner_init=DT_OUTER,
                               max_substeps=1, **solver_kw)
    cs = solver._sap.contact_solve
    cs.factorization_count.zero_()
    solver.step_dt(DT_OUTER, s0, s1, control)
    nf = int(cs.factorization_count.numpy()[0])
    iters = int(cs.newton_iterations_env.numpy().sum())
    return nf, iters


def test_factorization_count_equals_iterations_when_reuse_off():
    device = wp.get_device(DEV)
    nf, iters = _single_solve(device)
    assert iters >= 1, f"contact scene must drive the Newton loop, got {iters} iterations"
    assert nf == iters, f"reuse off: factorizations {nf} must equal Newton iterations {iters}"


def build_allegro_grasp(device):
    """Allegro hand grasping a cube (high-gain PD, coupled fingertip contacts) -- the
    contact-rich shadow-hand difficulty class. The oracle for how many Newton iterations
    the inner convex SAP solve actually takes on a hard manipulation contact problem."""
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
    builder.replicate(b, 1)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize(device=device), np.array(b.joint_target_q, dtype=np.float32)


def test_allegro_grasp_iteration_oracle():
    """ORACLE: on a contact-rich shadow-hand-class grasp, the inner convex SAP solve
    converges in O(few) Newton iterations -- NOT the ~150 a prior investigation claimed.
    This bounds the headroom for within-solve Hessian reuse (CENIC Sec. V-C)."""
    device = wp.get_device(DEV)
    try:
        model, tgt = build_allegro_grasp(device)
    except Exception as e:  # asset unavailable (offline CI): skip, do not fail
        print(f"  (skip allegro oracle: {e})")
        return
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s1.assign(s0)
    control.joint_target_q = wp.array(tgt, dtype=wp.float32, device=device)
    solver = SolverSAPAdaptive(model, mode="adaptive", tol=1e-3, dt_inner_init=DT_OUTER,
                               dt_inner_min=1e-6, max_substeps=16, max_iterations=200)
    cs = solver._sap.contact_solve
    peak_iters = 0
    for _ in range(15):  # settle into the grasp so fingertip contacts develop
        solver.step_dt(DT_OUTER, s0, s1, control)
        peak_iters = max(peak_iters, int(cs.newton_iterations_env.numpy().max()))
    assert peak_iters >= 2, f"contact-rich solve should run the Newton loop, got {peak_iters}"
    assert peak_iters <= 30, (
        f"ORACLE VIOLATED: hardest SAP solve took {peak_iters} Newton iterations; the prior "
        f"'~150 iterations' premise predicted >>30. SAP converges in O(few) iters."
    )
    print(f"  [oracle] hardest single SAP solve on allegro grasp: {peak_iters} Newton iterations")


def _run(test):
    name = test.__name__
    try:
        test()
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {name}: {e}")
        return False
    print(f"PASS {name}")
    return True


if __name__ == "__main__":
    ok = True
    for t in [test_factorization_count_equals_iterations_when_reuse_off,
              test_allegro_grasp_iteration_oracle]:
        ok = _run(t) and ok
    print(f"\n{'all passed' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
