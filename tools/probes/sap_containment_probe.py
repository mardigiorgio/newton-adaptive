"""Forced-failure containment probe for SolverSAPAdaptive.

Establishes the per-world solve-failure CONTAINMENT contract (default ON;
``NEWTON_SAP_CONTAINMENT=0`` = strict converge-or-throw): a world whose inner
SAP solve cannot converge must never kill the batch, must never have its
unconverged result committed, and must not perturb any other world by a single
bit.

Oracle argument (why this is not a tautology or a snapshot):
    The failure is FORCED by construction, not simulated: the inner Newton
    iteration cap is set to 0, so any world with live contact rows cannot ever
    satisfy the optimality test, at any dt -- the dt-insensitive stall class.
    Exactly one world (FAIL_WORLD) is dropped onto the ground; every other
    world free-falls high above it (zero contact rows -> its solve is marked
    converged as unconstrained, independent of the cap). The correct healthy-
    world output is computed by an INDEPENDENT route: a control run of the
    identical scene where the failing world converges (default cap), executed
    in the same process, same build, same device. Worlds are independent by
    design, so the healthy rows of the two runs must be BYTE-identical; a
    containment-vs-containment repeat guards the oracle itself (if the
    platform reorders fp work run-to-run, bitwise judgments are void and the
    probe says so). The never-consumed contract is a within-run invariant, not
    a snapshot: from the first boundary that latches the failing world, its
    committed state must be byte-identical to its state at the END of the
    previous boundary (the failing boundary accepted nothing) and must stay
    frozen for every later boundary (hold semantics).

Assertion tiers:
    1. Vacuity guards (exit 3): the containment arm must actually contain
       failures (cumulative failure events > 0, exactly FAIL_WORLD named, the
       diverged latch raised) and the control arm must actually converge
       (zero failure events, zero latches). A probe that cannot fail is not a
       test.
    2. Oracle validity (exit 2, ORACLE-DEGRADED): containment arm vs its
       repeat must match bitwise on every recorded field of every world.
    3. Containment contract (exit 1):
       (a) the containment arm COMPLETES all boundaries with no raise, with
           the env var UNSET (certifying the default);
       (b) the failing world latches ``diverged``, freezes its clock to the
           boundary (sim_time == next_time), and its post-failure state is
           bitwise-frozen at the last committed pre-failure state -- the
           unconverged result is never consumed;
       (c) every OTHER world's committed trajectory and controller carry are
           bitwise identical to the control run's -- exact cross-world
           isolation;
       (d) strict mode (NEWTON_SAP_CONTAINMENT=0) still raises RuntimeError
           on the same scene.

Residual risk (what a clean pass does NOT establish):
    * The forced stall is dt-insensitive (cap 0). The measured production
      failure class (a line-search limit cycle at one dt that converges at
      dt/10) exercises the same reject path but RECOVERS via the shrink-retry
      before latching; this probe pins the latch endpoint, not the recovery
      trajectory.
    * One sphere-on-plane contact per failing world; multi-contact manifolds
      and mesh collisions revalidate separately.
    * The manager-side consumption of the diverged latch (env termination and
      reset) is outside this probe; it drives the solver directly.

Run (single line, from the newton-adaptive repo root; GPU exercises the
graph + whole-march conditional tiers, CPU runs eagerly):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_containment_probe.py
"""

from __future__ import annotations

import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("SAP_WARP_PATH", str(_REPO.parent / "sap_warp")))

import numpy as np
import warp as wp

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

N_WORLDS = 6
FAIL_WORLD = 2
DT_OUTER = 2.0e-3
K_BOUNDARIES = 30
SEED = 20260814
R = 0.05
KE = 1.0e8
FAIL_CAP = 0  # no Newton iteration can run -> a contacting world can never converge
HEALTHY_CAP = 30  # the shipping default; the same scene converges everywhere

# Pinned so the certificate names the configuration it certifies (the shipping
# defaults). NEWTON_SAP_CONTAINMENT is deliberately NOT pinned in the
# containment arms: the run must certify the DEFAULT.
PINNED_ENV = {
    "NEWTON_SAP_ADAPTIVE_GRAPH": "1",
    "NEWTON_SAP_ADAPTIVE_CONDITIONAL": "1",
    "NEWTON_ADAPTIVE_TAIL_COMPACT": "1",
    "NEWTON_ADAPTIVE_CONTACT_REFRESH": "1",
    "NEWTON_SAP_DETERMINISTIC": "1",
}
CLEARED_ENV = ("NEWTON_SAP_CONTAINMENT", "NEWTON_SAP_FAILURE_DUMP", "NEWTON_SAP_FAILURE_REPLAY")

# Controller carry compared across runs. ``dt`` is EXCLUDED by design: after a
# world lands, later march iterations (whose count legitimately differs between
# the containment and control runs -- the failing world's shrink ladder adds
# iterations) clamp its dt to 0, so the post-march value depends on the
# iteration count, and it is DEAD state either way -- _seed_dt re-derives dt
# from ideal_dt at every boundary open in adaptive mode. The containment-vs-
# repeat oracle (identical iteration counts) still covers dt bitwise.
PER_WORLD_FIELDS = ("ideal_dt", "dt_ceiling", "consec_rej", "sim_time", "next_time", "substeps_frame")
ORACLE_ONLY_FIELDS = ("dt",)
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def build_model():
    t = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def initial_conditions():
    """FAIL_WORLD is dropped from 2 cm above the plane (several clean free-fall
    boundaries, then contact -- so a pre-failure committed state is always on
    record); every other world free-falls from ~1 m and cannot reach the ground
    within K_BOUNDARIES * DT_OUTER (0.06 s of fall = ~1.8 cm), so its contact
    count stays zero and its solve converges as unconstrained at ANY iteration
    cap."""
    rng = np.random.default_rng(SEED)
    z0 = R + 1.0 + rng.uniform(0.0, 0.05, N_WORLDS)
    vz = np.full(N_WORLDS, -1.0)
    z0[FAIL_WORLD] = R + 0.02
    vz[FAIL_WORLD] = -3.0
    return z0, vz


def _apply_env(containment: str | None):
    for k in CLEARED_ENV:
        os.environ.pop(k, None)
    os.environ.update(PINNED_ENV)
    if containment is not None:
        os.environ["NEWTON_SAP_CONTAINMENT"] = containment


def run_arm(name: str, max_iterations: int, containment: str | None):
    """March K_BOUNDARIES boundaries; record every world's committed state and
    controller carry per boundary. Returns (records, solver, raised_message)."""
    _apply_env(containment)
    model = build_model()
    coords = model.joint_coord_count // N_WORLDS
    dofs = model.joint_dof_count // N_WORLDS
    z0, vz = initial_conditions()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    q = state_0.joint_q.numpy()
    qd = state_0.joint_qd.numpy()
    for w in range(N_WORLDS):
        q[w * coords + 2] = z0[w]
        qd[w * dofs + 2] = vz[w]
    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    state_1.assign(state_0)

    solver = newton.solvers.SolverSAPAdaptive(
        model,
        mode="adaptive",
        tol=1e-5,
        dt_inner_init=DT_OUTER,
        dt_inner_min=1e-7,
        dt_inner_max=DT_OUTER,
        max_substeps=64,
        max_iterations=max_iterations,
        # Default "drake" = the ctor default (construction unchanged when the
        # env var is unset). SAP_CONTAINMENT_PRESET selects another preset so
        # the certificate can name the configuration under test (e.g. the
        # production approx32 modes when certifying the fp32 solve stack,
        # whose drake-mode control arm needs > the shipping cap on this
        # stiff-impact scene and would trip the zero-failure vacuity guard).
        contact_preset_variant=os.environ.get("SAP_CONTAINMENT_PRESET", "drake"),
    )
    # Initial committed state (post-fk): the pre-failure reference when the
    # failure fires on boundary 0 (contact rows can exist at a standoff, so
    # the first boundary may already be the failing one).
    init_state = {
        "joint_q": state_0.joint_q.numpy().copy(),
        "joint_qd": state_0.joint_qd.numpy().copy(),
        "body_q": state_0.body_q.numpy().copy(),
        "body_qd": state_0.body_qd.numpy().copy(),
    }
    records = []
    raised = None
    try:
        for _ in range(K_BOUNDARIES):
            state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
            rec = {
                "joint_q": state_0.joint_q.numpy().copy(),
                "joint_qd": state_0.joint_qd.numpy().copy(),
                "body_q": state_0.body_q.numpy().copy(),
                "body_qd": state_0.body_qd.numpy().copy(),
                "ideal_dt": solver._ideal_dt.numpy().copy(),
                "dt": solver._dt.numpy().copy(),
                "dt_ceiling": solver._dt_ceiling.numpy().copy(),
                "consec_rej": solver._consec_rej.numpy().copy(),
                "sim_time": solver._sim_time.numpy().copy(),
                "next_time": solver._next_time.numpy().copy(),
                "substeps_frame": solver.substeps.numpy().copy(),
                "diverged": solver.diverged.numpy().copy(),
                "fail_counts": solver.solve_failure_worlds.numpy().copy(),
            }
            records.append(rec)
    except RuntimeError as exc:
        raised = str(exc)
    return records, solver, raised, coords, dofs, init_state


def _world_bytes(arr: np.ndarray, w: int, per_world: int) -> bytes:
    a = np.ascontiguousarray(arr)
    if a.shape[0] == N_WORLDS:
        return a[w : w + 1].tobytes()
    return a[w * per_world : (w + 1) * per_world].tobytes()


def main() -> int:
    dev = wp.get_device()
    is_cuda = bool(dev.is_cuda)
    print(f"device: {dev} (cuda={is_cuda})")

    # Arm A: containment DEFAULT (env unset), forced failure.
    rec_a, sol_a, raised_a, coords, dofs, init_a = run_arm("containment", FAIL_CAP, containment=None)
    # Arm A repeat: oracle guard.
    rec_a2, _sol_a2, raised_a2, _, _, _ = run_arm("containment-repeat", FAIL_CAP, containment=None)
    # Arm B: control -- identical scene, cap large enough that every world converges.
    rec_b, sol_b, raised_b, _, _, _ = run_arm("control", HEALTHY_CAP, containment=None)

    model_rows = {
        "joint_q": coords,
        "joint_qd": dofs,
        "body_q": 1,
        "body_qd": 1,
    }

    # ---- Tier 1: vacuity guards ------------------------------------------
    a_fail_counts = rec_a[-1]["fail_counts"] if rec_a else np.zeros(N_WORLDS)
    first_latch = next((k for k, r in enumerate(rec_a) if bool(r["diverged"][FAIL_WORLD])), None)
    guards = {
        "containment arm defaulted ON (env unset)": sol_a.containment,
        "containment arm completed with no raise": raised_a is None and len(rec_a) == K_BOUNDARIES,
        "containment arm contained failures (events > 0)": sol_a.solve_failure_events > 0,
        "failing world named by the per-world counter": int(a_fail_counts[FAIL_WORLD]) > 0,
        "no healthy world ever counted a failure": all(
            int(a_fail_counts[w]) == 0 for w in range(N_WORLDS) if w != FAIL_WORLD
        ),
        "failing world latched diverged at some boundary": first_latch is not None,
        "control arm completed with no raise": raised_b is None and len(rec_b) == K_BOUNDARIES,
        # Zero control-arm failure events is the strict default. The healthy-row
        # isolation oracle only compares the contact-free worlds, whose rows
        # carry zero failures in BOTH arms (the impact world -- the forced
        # FAIL_WORLD -- is excluded from the comparison), so the oracle
        # tolerates RECOVERED control-arm events on the impact world. The
        # explicit opt-in exists for solve configurations (e.g. the fp32
        # stack) whose shipping iteration cap organically caps out on this
        # scene's first impact and recovers by shrink-retry; a LATCH in the
        # control arm still fails the next guard unconditionally.
        "control arm converged everywhere (zero failure events)": (
            sol_b.solve_failure_events == 0
            or (
                os.environ.get("SAP_CONTAINMENT_ALLOW_RECOVERED_CONTROL") == "1"
                and all(int(r["diverged"].sum()) == 0 for r in rec_b)
            )
        ),
        "control arm latched no world": all(int(r["diverged"].sum()) == 0 for r in rec_b),
        "containment boundaries counted (engagement)": sol_a._containment_boundaries == K_BOUNDARIES,
    }
    if is_cuda:
        guards["whole-march conditional tier engaged in containment arm"] = sol_a._conditional_launches > 0
    bad = [g for g, ok in guards.items() if not ok]
    for g, ok in guards.items():
        print(f"guard: {g}: {'ok' if ok else 'NOT MET'}")
    if bad:
        print("VACUOUS or CONTRACT-DEAD: the scene failed to force/contain the failure as designed.")
        if raised_a is not None:
            print(f"  containment arm raised: {raised_a}")
        return 3

    # ---- Tier 2: oracle validity (containment vs repeat, all worlds) -----
    if raised_a2 is not None or len(rec_a2) != K_BOUNDARIES:
        print("ORACLE-DEGRADED: containment repeat raised or truncated.")
        return 2
    for k, (ra, rb) in enumerate(zip(rec_a, rec_a2, strict=True)):
        for field in STATE_FIELDS + PER_WORLD_FIELDS + ORACLE_ONLY_FIELDS + ("diverged", "fail_counts"):
            if not np.array_equal(
                np.ascontiguousarray(ra[field]).view(np.uint8),
                np.ascontiguousarray(rb[field]).view(np.uint8),
            ):
                print(f"ORACLE-DEGRADED: containment arm not run-to-run bitwise at boundary {k}, field {field}.")
                return 2

    failed = False

    # ---- Tier 3b: never-consumed / latch invariants ----------------------
    # The first latch boundary committed NOTHING for the failing world: its
    # state equals the previous boundary's end state (the initial post-fk
    # state when the failure fires on boundary 0), byte for byte.
    pre_rec = init_a if first_latch == 0 else rec_a[first_latch - 1]
    for field in STATE_FIELDS:
        pw = model_rows[field]
        pre = _world_bytes(pre_rec[field], FAIL_WORLD, pw)
        at = _world_bytes(rec_a[first_latch][field], FAIL_WORLD, pw)
        if pre != at:
            print(
                f"FAIL[b]: failing world committed state during its failing boundary "
                f"{first_latch} (field {field}) -- unconverged result consumed"
            )
            failed = True
    for k in range(first_latch, K_BOUNDARIES):
        r = rec_a[k]
        if not bool(r["diverged"][FAIL_WORLD]):
            print(f"FAIL[b]: diverged latch dropped at boundary {k} while the stall persists")
            failed = True
            break
        # The latch ASSIGNS sim_time = next_time, so byte equality is the
        # invariant (not a float tolerance).
        if np.float32(r["sim_time"][FAIL_WORLD]).tobytes() != np.float32(r["next_time"][FAIL_WORLD]).tobytes():
            print(f"FAIL[b]: latched world's clock not frozen to its boundary at boundary {k}")
            failed = True
            break
        for field in STATE_FIELDS:
            pw = model_rows[field]
            if _world_bytes(rec_a[first_latch][field], FAIL_WORLD, pw) != _world_bytes(r[field], FAIL_WORLD, pw):
                print(f"FAIL[b]: failing world's frozen state changed at boundary {k} (field {field})")
                failed = True
                break
        else:
            continue
        break
    if not failed:
        print(
            f"PASS[b]: failing world {FAIL_WORLD} latched at boundary {first_latch}, state "
            f"bitwise-frozen and clock pinned through boundary {K_BOUNDARIES - 1}"
        )

    # ---- Tier 3c: exact cross-world isolation ----------------------------
    iso_failed = False
    for k, (ra, rb) in enumerate(zip(rec_a, rec_b, strict=True)):
        for field in STATE_FIELDS + PER_WORLD_FIELDS:
            pw = model_rows.get(field, 1)
            for w in range(N_WORLDS):
                if w == FAIL_WORLD:
                    continue
                if _world_bytes(ra[field], w, pw) != _world_bytes(rb[field], w, pw):
                    print(
                        f"FAIL[c]: healthy world {w} diverged from control at boundary {k}, "
                        f"field {field} -- cross-world isolation broken"
                    )
                    iso_failed = True
                    break
            if iso_failed:
                break
        if iso_failed:
            break
    if not iso_failed:
        print(
            f"PASS[c]: all {N_WORLDS - 1} healthy worlds bitwise identical to control over "
            f"{K_BOUNDARIES} boundaries ({len(STATE_FIELDS + PER_WORLD_FIELDS)} fields)"
        )
    failed |= iso_failed

    # ---- Tier 3d: strict mode still raises -------------------------------
    rec_d, sol_d, raised_d, _, _, _ = run_arm("strict", FAIL_CAP, containment="0")
    if sol_d.containment:
        print("FAIL[d]: NEWTON_SAP_CONTAINMENT=0 did not reach construction")
        failed = True
    elif raised_d is None:
        print("FAIL[d]: strict mode did not raise on the forced failure")
        failed = True
    elif "failed to converge" not in raised_d:
        print(f"FAIL[d]: strict mode raised the wrong error: {raised_d}")
        failed = True
    else:
        print(f"PASS[d]: strict mode raised at boundary {len(rec_d)} ({raised_d.splitlines()[0]})")

    print(
        f"containment events={sol_a.solve_failure_events} "
        f"control events={sol_b.solve_failure_events} (recovered-control allowed: "
        f"{os.environ.get('SAP_CONTAINMENT_ALLOW_RECOVERED_CONTROL') == '1'}) "
        f"fail_world_counts={a_fail_counts.tolist()} first_latch_boundary={first_latch}"
    )
    if failed:
        return 1
    print(
        f"SAP-CONTAINMENT: PASS (preset={os.environ.get('SAP_CONTAINMENT_PRESET', 'drake')}, "
        f"solve_precision={getattr(sol_a, 'solve_precision', 'fp64')})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
