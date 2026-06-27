"""TDD: eq-32 warm-start iteration-0 gate in the inner SAP solve (CENIC Sec. VI-A).

CENIC evaluates the SAP optimality condition (eq. 32) on the warm-started iterate
v0 BEFORE doing any linear-algebra work. An env whose warm start already satisfies
the in-loop tolerance ``max(p_norm, jc_norm)`` is converged at "iteration 0" and must
do ZERO Hessian factorizations -- it never pays the fp64 J^T G J GEMM + blocked
Cholesky.

In our solver this is the pre-loop ``_solver_update_active`` at
contact_solve.py (in ``_run_unit_conditional_newton_loop``), which evaluates the
contact gradient with ``include_hessian=False`` (no linear solve) and runs the exact
eq-32 norm/optimality kernel using the SAME tolerance as the in-loop convergence test
(``optimality_abs_tol + optimality_rel_tol * max(p_norm, jc_norm)``). When it fires,
``newton_active`` is zeroed and ``active_count`` drops to 0, so the captured
``wp.capture_while`` Newton body never runs and ``factorization_count`` stays 0.

This test PINS that behavior so it cannot silently regress:

  * GATE (optimal warm start): a solve warm-started AT its own converged solution does
    ZERO factorizations, while staying finite and reproducing the solution to tol.
  * POSITIVE CONTROL (perturbed warm start): the same problem warm-started off the
    solution factorizes >= 1 time and converges back to the SAME solution -- proving
    the scene genuinely exercises the Newton loop and the gate does not corrupt the
    result.

Verified discriminating: with the pre-loop gate removed (newton_active forced to the
constrained set), the GATE case factorizes once instead of zero -- this test then fails.

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
from newton.solvers import SolverSAPAdaptive

DEV = "cpu"
DT_OUTER = 1.0 / 60.0


def build_contact_scene(device, n=4, radius=0.1, penetration=0.02, impact_vz=-1.0):
    """Several spheres resting on a ground plane -> persistent normal contacts so the
    inner SAP convex solve actually runs its Newton loop."""
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


def _settle_capture_resolve(device, perturb):
    """Settle the contact scene, capture one inner ``solve`` invocation's arguments and
    its converged velocity, then re-run that exact solve warm-started at
    ``solution + perturb``. Returns (factorizations, max|out - solution|, finite)."""
    model = build_contact_scene(device)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s1.assign(s0)
    solver = SolverSAPAdaptive(model, mode="fixed", dt_inner_init=DT_OUTER,
                               max_substeps=1, max_iterations=80)
    cs = solver._sap.contact_solve

    for _ in range(8):  # settle into a near-static resting contact
        solver.step_dt(DT_OUTER, s0, s1, control)
        s0, s1 = s1, s0

    # Capture the arguments of one inner SAP solve, plus its converged solution.
    orig_solve = cs.solve
    cap = {}

    def wrap(contact_result, state, control_, dt, v_star, **kw):
        cap["args"] = (contact_result, state, control_, dt, v_star)
        cap["kw"] = kw
        return orig_solve(contact_result, state, control_, dt, v_star, **kw)

    cs.solve = wrap
    solver.step_dt(DT_OUTER, s0, s1, control)
    cs.solve = orig_solve
    solution = cs.v_flat.numpy().copy()

    # Re-run the identical solve warm-started at solution + perturb. v0 (free-motion
    # velocity) is unchanged, so `solution` is the exact optimum; v_guess seeds iterate 0.
    contact_result, state, control_, dt, v_star = cap["args"]
    kw = dict(cap["kw"])
    kw.pop("v_guess_active", None)
    guess = solution + perturb
    kw["v_guess"] = wp.array(guess, dtype=cs.v_flat.dtype, device=device)

    cs.factorization_count.zero_()
    cs.solve(contact_result, state, control_, dt, v_star, **kw)
    nf = int(cs.factorization_count.numpy()[0])
    out = cs.v_flat.numpy().copy()
    err = float(np.max(np.abs(out - solution)))
    finite = bool(np.all(np.isfinite(out)))
    return nf, err, finite


def test_warmstart_at_solution_skips_all_factorizations():
    """GATE: an inner solve whose warm start already satisfies eq. 32 does ZERO
    factorizations (no fp64 GEMM + Cholesky) and reproduces the solution to tol."""
    device = wp.get_device(DEV)
    nf, err, finite = _settle_capture_resolve(device, perturb=0.0)
    assert finite, "gated solve produced non-finite velocities"
    assert nf == 0, f"warm start at the solution must factorize 0 times, got {nf}"
    assert err <= 1e-9, f"gated solve must reproduce the solution, got max|out-sol|={err:.3e}"


def test_perturbed_warmstart_factorizes_and_converges():
    """POSITIVE CONTROL: the same problem warm-started OFF the solution factorizes
    (Newton loop runs) and converges back to the SAME solution -- the gate neither
    fires spuriously nor corrupts convergence."""
    device = wp.get_device(DEV)
    nf, err, finite = _settle_capture_resolve(device, perturb=0.2)
    assert finite, "perturbed solve produced non-finite velocities"
    assert nf >= 1, f"a non-optimal warm start must run the Newton loop, got {nf} factorizations"
    assert err <= 1e-6, f"perturbed solve must reconverge to the solution, got max|out-sol|={err:.3e}"


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
    for t in [test_warmstart_at_solution_skips_all_factorizations,
              test_perturbed_warmstart_factorizes_and_converges]:
        ok = _run(t) and ok
    print(f"\n{'all passed' if ok else 'FAILURES'}")
    sys.exit(0 if ok else 1)
