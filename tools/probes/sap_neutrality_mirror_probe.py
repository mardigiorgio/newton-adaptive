"""Pass-37: env-privacy and per-world-dt threading certificate (audit item C1).

WHAT IS UNDER TEST
    C1 claims that converting ~10 SAP kernels from a scalar ``dt`` to a
    per-world ``dt[env]`` array is physics-neutral. That claim has two halves:

    (a) THREADING. Every world reads ITS OWN dt, and no per-world quantity
        leaks across the env axis.  This half is MEASURABLE and is what this
        probe measures.
    (b) UNIFORM REDUCTION. A scalar dt fills the array uniformly, so the
        array path reproduces the deleted scalar path.  The scalar kernels no
        longer exist at HEAD, so no run-level A/B is possible; that half is a
        source-level argument, not a measurement, and this probe does not
        claim it.

ORACLE ARGUMENT (not a tautology, not a snapshot, not structural-only)
    The scene is built as MIRROR PAIRS: world i and world (N-1-i) get
    byte-identical initial conditions, but they occupy different env indices
    and have different neighbours in the batch.  Physics is env-local, so the
    two members of a pair must produce byte-identical committed trajectories.
    The probe never computes what the trajectory should be -- it constrains
    two independently-indexed copies of the same physical world to agree.

    That constraint is violated by exactly the failure modes C1 could have:
      * a wrongly indexed dt read (dt[env plus k], dt[world of something else]) hands
        world i a neighbour's step and world N-1-i a DIFFERENT neighbour's
        step, because the mirror maps neighbourhoods to different ones;
      * any cross-env leak in the compacted/list-indexed kernels (C2, C3, C7)
        makes a world's bytes depend on which OTHER worlds are in its list
        slot, and the mirror puts different worlds in those slots;
      * a global-instead-of-per-world reduction shows up as pair inequality
        as soon as the pair members' dt histories differ from each other's
        neighbours.

    A single differing byte in any pair refutes env-privacy.

    NOT proven by a pass: that a globally-broadcast dt (every world reading
    dt[0]) would be caught -- that failure is mirror-symmetric.  The vacuity
    guards below require the recorded per-world dt to actually SPREAD, and
    the source-level check (every kernel reads dt[env] with the same ``env``
    it uses for every other per-env array) covers that mode; it is named as
    residual risk rather than claimed.

VACUITY GUARDS (a probe that cannot fail is not a test)
    * live pipeline contacts,
    * at least one rejection (a boundary needing >= 2 march iterations),
    * per-world accepted dt genuinely spread (> 1 distinct value) -- without
      spread, dt heterogeneity is never exercised and the threading claim is
      untested,
    * distinct per-world substep counts,
    * no world diverged.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_neutrality_mirror_probe.py
"""

from __future__ import annotations

import os
import pathlib
import sys

# Pinned BEFORE newton/sap_warp import: several flags resolve at import time.
os.environ["NEWTON_SAP_DETERMINISTIC"] = "1"

# sap_warp is a sibling checkout of this repo; SAP_WARP_PATH overrides.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if not (_REPO / "newton").is_dir():  # running from the pass scratchpad
    _REPO = pathlib.Path("/home/mdigiorgio/Documents/code/newton-adaptive")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("SAP_WARP_PATH", str(_REPO.parent / "sap_warp")))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

import newton  # noqa: E402
from newton._src.sim.control import Control as _Control  # noqa: E402
from newton._src.sim.model import Model as _Model  # noqa: E402

_Model.joint_target_pos = property(lambda self: self.joint_target_q)
_Model.joint_target_vel = property(lambda self: self.joint_target_qd)
_Control.joint_target_pos = property(lambda self: self.joint_target_q)
_Control.joint_target_vel = property(lambda self: self.joint_target_qd)

N_PAIRS = 8
N_WORLDS = 2 * N_PAIRS
DT_OUTER = 2.0e-3
K_BOUNDARIES = 8
SEED = 20260817
# The accuracy budget must be tight enough that the impact boundary's
# step-doubling error EXCEEDS it, because a scene where every world accepts at
# the cap has no per-world dt spread and therefore does not exercise the
# threading this probe exists to test (the vacuity guards enforce that).
# Under the attempt-consistent law (ACR, the shipped default) the trial pair
# discretizes ONE contact model, so the residual is pure truncation error and
# is much smaller than with ACR off -- hence a tighter budget here than the
# flag-equivalence probe's ACR-off cells need.
TOL = float(os.environ.get("P37_TOL", "1e-9"))
# Floor well below anything the controller should need, so a rejection cascade
# shrinks the step instead of latching the floor (a floor latch accepts
# regardless of the error and would silence the spread the guards require).
DT_MIN = float(os.environ.get("P37_DT_MIN", "1e-12"))
R = 0.05
KE = 1.0e8


def build_model():
    t = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def mirror_initial_conditions():
    """N_PAIRS distinct (gap, vz) draws, then the SAME draws in REVERSED order.

    World i and world (N_WORLDS-1-i) are physically identical; every other
    world in their respective neighbourhoods differs.
    """
    rng = np.random.default_rng(SEED)
    gap = rng.uniform(0.001, 0.003, N_PAIRS)
    vz = rng.uniform(-4.0, -2.0, N_PAIRS)
    gap_full = np.concatenate([gap, gap[::-1]])
    vz_full = np.concatenate([vz, vz[::-1]])
    return (R + gap_full).astype(np.float64), vz_full.astype(np.float64)


def fresh(z0, vz):
    model = build_model()
    coords = model.joint_coord_count // N_WORLDS
    dofs = model.joint_dof_count // N_WORLDS
    s0, s1 = model.state(), model.state()
    control = model.control()
    q = s0.joint_q.numpy()
    qd = s0.joint_qd.numpy()
    for w in range(N_WORLDS):
        q[w * coords + 2] = z0[w]
        qd[w * dofs + 2] = vz[w]
    s0.joint_q.assign(q)
    s0.joint_qd.assign(qd)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    s1.assign(s0)
    solver = newton.solvers.SolverSAPAdaptive(
        model,
        mode="adaptive",
        tol=TOL,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_MIN,
        dt_inner_max=DT_OUTER,
        max_substeps=64,
    )
    return model, s0, s1, control, solver, coords, dofs


def main() -> int:
    dev = wp.get_device()
    print(f"device: {dev} (cuda={bool(dev.is_cuda)})")
    print(f"scene: {N_WORLDS} worlds = {N_PAIRS} mirror pairs, {K_BOUNDARIES} boundaries, det=1")
    print(f"tol={TOL:g}  ACR={os.environ.get('NEWTON_SAP_ATTEMPT_CONSISTENT_R', '(unset -> shipped default ON)')}")

    z0, vz = mirror_initial_conditions()
    model, s0, s1, control, solver, coords, dofs = fresh(z0, vz)
    bodies = model.body_count // N_WORLDS

    frames = []
    iters = []
    ncon_seen = 0
    for _ in range(K_BOUNDARIES):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, control)
        ncon_seen = max(ncon_seen, int(solver._contacts.rigid_contact_count.numpy()[0]))
        frames.append(
            {
                "joint_q": s0.joint_q.numpy().reshape(N_WORLDS, coords).copy(),
                "joint_qd": s0.joint_qd.numpy().reshape(N_WORLDS, dofs).copy(),
                "body_q": s0.body_q.numpy().reshape(N_WORLDS, bodies, -1).copy(),
                "body_qd": s0.body_qd.numpy().reshape(N_WORLDS, bodies, -1).copy(),
                "dt": solver._dt.numpy().copy(),
                "ideal_dt": solver._ideal_dt.numpy().copy(),
                "dt_ceiling": solver._dt_ceiling.numpy().copy(),
                "consec_rej": solver._consec_rej.numpy().copy(),
                "substeps": solver.substeps.numpy().copy(),
                "diverged": solver.diverged.numpy().copy(),
            }
        )
        iters.append(int(solver.iteration_count.numpy()[0]))

    # ---- vacuity guards -------------------------------------------------
    dts = np.concatenate([f["dt"] for f in frames])
    spread = len(np.unique(np.concatenate([f["dt"] for f in frames]))) > 1
    per_boundary_spread = any(len(np.unique(f["dt"])) > 1 for f in frames)
    substep_spread = any(len(np.unique(f["substeps"])) > 1 for f in frames)
    diverged = sum(int(f["diverged"].sum()) for f in frames)
    guards = {
        "pipeline produced contacts": ncon_seen > 0,
        "per-world dt spread WITHIN a boundary (heterogeneous dt exercised)": per_boundary_spread,
        "per-world dt takes more than one value over the run": bool(spread),
        "per-world accepted-substep counts differ": substep_spread,
        "rejection exercised (a boundary needed >= 2 march iterations)": max(iters) >= 2,
        "no world diverged": diverged == 0,
    }
    print("\n--- vacuity guards ---")
    ok = True
    for k, v in guards.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
        ok = ok and bool(v)
    print(f"  dt range over run: [{dts.min():.6e}, {dts.max():.6e}], {len(np.unique(dts))} distinct values")
    if not ok:
        print("MIRROR-PAIR: VACUOUS (exit 3)")
        return 3

    # ---- mirror-pair bitwise equality -----------------------------------
    print("\n--- mirror-pair bitwise equality ---")
    failed = False
    for k, f in enumerate(frames):
        for field, arr in f.items():
            a = np.ascontiguousarray(arr)
            lo = a[:N_PAIRS]
            hi = a[N_PAIRS:][::-1]  # mirror: world N-1-i
            if lo.tobytes() != hi.tobytes():
                failed = True
                d = np.flatnonzero(
                    (lo.reshape(N_PAIRS, -1).view(np.uint8) != hi.reshape(N_PAIRS, -1).view(np.uint8)).any(axis=1)
                )
                p = int(d[0])
                x = lo.reshape(N_PAIRS, -1)[p]
                y = hi.reshape(N_PAIRS, -1)[p]
                j = int(np.flatnonzero(x != y)[0])
                print(
                    f"FAIL[{field}] boundary {k}: pair ({p}, {N_WORLDS - 1 - p}) differs at element {j}: "
                    f"{x[j]!r} vs {y[j]!r}"
                )
                break
        if failed:
            break
    if failed:
        print("MIRROR-PAIR: FAIL (exit 1) -- env-privacy / dt-threading violated")
        return 1

    nfields = len(frames[0])
    print(f"PASS: {N_PAIRS} mirror pairs x {K_BOUNDARIES} boundaries x {nfields} fields, all bitwise identical")
    print("MIRROR-PAIR: PASS")
    print(
        "\nRESIDUAL RISK (what this pass does NOT establish): a dt broadcast that is "
        "identical for every world (e.g. every kernel reading dt[0]) is mirror-symmetric "
        "and would pass; the uniform-dt reduction to the deleted scalar path is a "
        "source-level argument, not measured here; and the scene is a single "
        "sphere-plane pair per world with no actuators."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
