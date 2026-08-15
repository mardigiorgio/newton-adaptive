"""Shim-free construction + stepping probe for SolverSAPAdaptive on newton 1.5.

What this probe establishes:
    * sap_warp's Newton-boundary readers construct a SolverSAPAdaptive against
      a newton tree whose Model/Control tombstone the pre-1.5 target names
      (joint_target_pos / joint_target_vel raise on read). No aliases, no
      monkey-patching: a tripwire asserts the tombstones are live in this
      process before the solver is built, so a pass cannot come from an
      accidentally shimmed environment (contrast: sap_flag_equivalence_probe.py
      installs exactly such aliases).
    * The solver marches K_BOUNDARIES step_dt boundaries on a scene that
      guarantees impacts and persistent resting contact; committed state stays
      finite at every boundary and the collision pipeline reports live rigid
      contacts (vacuity guard: a scene without contacts cannot fail the
      contact-path plumbing).
    * Per-world dt reaches the contact solve: the contact-solve kernels read a
      single per-world dt buffer (contact_solve._dt_world), which the
      near-rigid regularization indexes as dt[env]; with per-world impact
      severities the controller's dts must spread, and the guard asserts the
      buffer is non-uniform across worlds after the impact boundary.

Deliberately NOT established here:
    * Numerical correctness of the SAP solve (equivalence probes own that).
    * The coord-layout (use_coord_layout_targets=True) gather path: this scene
      builds under the default dof layout, where the boundary readers pass
      arrays through untouched. The gather map is exercised by construction
      only if the ambient newton flips the default.
    * Actuated control: the scene has no drives, so the target arrays cross
      the boundary as allocated-but-zero (still exercising the renamed reads
      and the float32 placeholder contract when arrays are absent).

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_newton15_construct_probe.py
"""

from __future__ import annotations

import os
import pathlib
import sys

# sap_warp is a sibling checkout of this repo; SAP_WARP_PATH overrides.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("SAP_WARP_PATH", str(_REPO.parent / "sap_warp")))

import numpy as np
import warp as wp

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

N_WORLDS = 4
DT_OUTER = 2.0e-3
K_BOUNDARIES = 50
SEED = 20260814
TOL = 1e-5
R = 0.05
KE = 1.0e8
# Same selector the IsaacLab manager honors, so this gate can certify any
# line-search variant end-to-end without touching the probe.
LINE_SEARCH = os.environ.get("NEWTON_SAP_LINE_SEARCH", "armijo_decay")

_FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tail = f" ({detail})" if detail else ""
    print(f"{'PASS' if ok else 'FAIL'}: {label}{tail}")
    if not ok:
        _FAILURES.append(label)


def build_model():
    """One sub-world (free sphere over an authored-stiffness ground plane),
    replicated into N_WORLDS worlds -- same scene as the flag-equivalence
    probe, minus every compatibility alias."""
    t = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def main() -> int:
    dev = wp.get_device()
    print(f"device: {dev} (cuda={bool(dev.is_cuda)})")
    print(f"newton: {newton.__version__} at {pathlib.Path(newton.__file__).parent}")
    print(f"line_search_variant: {LINE_SEARCH}")

    model = build_model()

    # ---- tombstone tripwire: this process must be UNshimmed newton 1.5 ------
    for obj, name in ((model, "Model"), (model.control(), "Control")):
        for attr in ("joint_target_pos", "joint_target_vel"):
            try:
                getattr(obj, attr)
                removed = False
            except AttributeError:
                removed = True
            check(f"{name}.{attr} raises on read (1.5 tombstone live, no shim)", removed)
    if _FAILURES:
        print("SAP-NEWTON15-CONSTRUCT: VACUOUS (environment is shimmed or pre-1.5)")
        return 3

    # ---- per-world initial conditions: every sphere impacts inside boundary 0
    rng = np.random.default_rng(SEED)
    gap = rng.uniform(0.001, 0.003, N_WORLDS)
    vz = rng.uniform(-4.0, -2.0, N_WORLDS)
    coords = model.joint_coord_count // N_WORLDS
    dofs = model.joint_dof_count // N_WORLDS
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    q = state_0.joint_q.numpy()
    qd = state_0.joint_qd.numpy()
    for w in range(N_WORLDS):
        q[w * coords + 2] = R + gap[w]
        qd[w * dofs + 2] = vz[w]
    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    state_1.assign(state_0)

    # ---- construction (the 1.5 breakage point) ------------------------------
    try:
        solver = newton.solvers.SolverSAPAdaptive(
            model,
            mode="adaptive",
            tol=TOL,
            dt_inner_init=DT_OUTER,
            dt_inner_min=1e-7,
            dt_inner_max=DT_OUTER,
            max_substeps=64,
            line_search_variant=LINE_SEARCH,
        )
    except Exception as exc:  # noqa: BLE001 -- the probe reports, not raises
        check("SolverSAPAdaptive constructs against newton 1.5", False, f"{type(exc).__name__}: {exc}")
        print("SAP-NEWTON15-CONSTRUCT: FAIL")
        return 1
    check("SolverSAPAdaptive constructs against newton 1.5", True)

    # ---- march K_BOUNDARIES step_dt boundaries ------------------------------
    ncon_max = 0
    nonuniform_dt_seen = False
    finite_all = True
    first_bad = None
    for k in range(K_BOUNDARIES):
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        # Post-boundary host reads (never inside the inner loop).
        ncon_max = max(ncon_max, int(solver._contacts.rigid_contact_count.numpy()[0]))
        dt_world = solver._sap.contact_solve._dt_world.numpy()
        if len(np.unique(dt_world)) > 1:
            nonuniform_dt_seen = True
        for name, arr in (
            ("joint_q", state_0.joint_q),
            ("joint_qd", state_0.joint_qd),
            ("body_q", state_0.body_q),
            ("body_qd", state_0.body_qd),
        ):
            if not np.all(np.isfinite(arr.numpy())):
                finite_all = False
                first_bad = first_bad or (k, name)

    check(f"state finite in joint_q/joint_qd/body_q/body_qd over {K_BOUNDARIES} boundaries",
          finite_all, f"first non-finite at {first_bad}" if first_bad else "")
    check("collision pipeline produced rigid contacts (> 0)", ncon_max > 0, f"max count {ncon_max}")
    check("no world latched diverged", int(solver.diverged.numpy().sum()) == 0)
    check(
        "per-world dt vector reached the contact solve non-uniform "
        "(contact_solve._dt_world, the buffer the near-rigid regularization reads as dt[env])",
        nonuniform_dt_seen,
        f"last dt_world={solver._sap.contact_solve._dt_world.numpy().tolist()}",
    )

    z = state_0.joint_q.numpy()[2::coords]
    print(f"final sphere heights: {np.array2string(z, precision=5)} (radius {R})")
    print(f"cumulative substeps: {solver.cumulative_substeps()}")

    if _FAILURES:
        print("SAP-NEWTON15-CONSTRUCT: FAIL")
        return 1
    print("SAP-NEWTON15-CONSTRUCT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
