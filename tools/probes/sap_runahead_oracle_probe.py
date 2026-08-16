"""Mixed-time correctness oracle for the SAP run-ahead single march
(NEWTON_SAP_RUNAHEAD, default OFF).

What the run-ahead march changes and what it must NOT change:
    Under run-ahead, a world reaching its boundary target inside the action
    window keeps marching (its boundary bookkeeping applied in place by a
    device crossing kernel; its contact set refreshed at ITS boundary-entry
    state by a crossing-batched masked collide + per-env adopt), so
    mid-window batch state holds worlds at MIXED boundary times. The claim
    under test is that this is a SCHEDULING change of the batch, not a
    physics change of any world: each world's committed trajectory must be
    what that world alone would produce, and the window edges (the
    action-cadence states the environment consumes) must stay
    batch-synchronized.

Oracle argument (why this is not a tautology or a snapshot):
    Two independent references, both recomputed every invocation:
    (a) PER-WORLD ISOLATION: for every world w, a SINGLE-WORLD run of the
        identical scene (same initial condition, same window structure, flag
        ON) is the world's own trajectory with no other world in the batch.
        Under NEWTON_SAP_DETERMINISTIC=1 every cross-world ordering effect
        (contact buffer order, per-env slot ranks, reduction orders) is
        canonicalized per world, so the batch run's world-w rows must be
        BITWISE equal to the solo run's at every window edge -- any coupling
        introduced by the run-ahead scheduler (masked collide, adopt store,
        crossing kernel) would break byte equality.
    (b) SEMANTIC PRESERVATION vs THE CERTIFIED MARCH: the flag-OFF run of
        the identical scenario is the certified per-boundary semantics. Per
        world, run-ahead preserves every controller and contact rule; the
        ONE construction-level deviation is float32 clock arithmetic (OFF
        rebases the clock every boundary, ON once per window, so
        boundary-landing slivers may round differently by ~ULP(window)).
        Window-edge states are therefore compared bitwise first; on any
        mismatch the probe reports the divergence magnitude and asserts a
        sliver-scale bound (BOUND_* below) -- a semantic error (wrong
        contact set, wrong bookkeeping, cross-world leak) shows up orders of
        magnitude above it, a pure rounding reschedule below it.
    A batch-vs-batch repeat guards oracle validity (bitwise judgments are
    void if the platform is not run-to-run deterministic), and engagement
    guards make vacuous passes impossible: the crossing counter must equal
    the exact structural count N_WORLDS * (WINDOW-1) * n_windows (every
    world crosses each interior boundary exactly once), at least one
    mid-window call must show a world genuinely AHEAD of the call target,
    and the OFF arm must show zero crossings and zero adopt batches (leak
    guard).

Assertion tiers:
    1. Vacuity/engagement guards (exit 3) -- see above.
    2. Oracle validity (exit 2): ON-batch vs ON-batch repeat bitwise.
    3. Isolation contract (exit 1): batch world-w rows == solo world-w rows,
       bitwise, every window edge, every recorded field.
    4. Semantic preservation (exit 1): ON vs OFF window-edge committed state
       within the sliver bound (bitwise result reported either way).

Residual risk (what a clean pass does NOT establish):
    * One sphere-on-plane contact per world; mesh manifolds and
      multi-contact scenes revalidate through the task-level gates.
    * The scene has no actuators, so "same actions" degenerates to zero
      control; control-constancy across the window is a caller contract the
      probe cannot test.
    * Window alignment with the caller's action cadence is a configuration
      contract (NEWTON_SAP_RUNAHEAD_WINDOW == decimation); this probe drives
      the solver directly and is aligned by construction.
    * det-unset mode is not judged here (its contact packing order is
      legally run-varying); production det-unset runs inherit the same
      per-world arithmetic with a different (legal) packing draw.

Run (single line, from the newton-adaptive repo root; GPU exercises the
graph + whole-march conditional tiers):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_runahead_oracle_probe.py
"""

from __future__ import annotations

import os
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("SAP_WARP_PATH", str(_REPO.parent / "sap_warp")))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

# Probe-local compatibility shim (see sap_flag_equivalence_probe.py): the
# sap_warp checkout reads pre-1.5 target names; alias them to the 1.5 names.
from newton._src.sim.control import Control as _Control  # noqa: E402
from newton._src.sim.model import Model as _Model  # noqa: E402

_Model.joint_target_pos = property(lambda self: self.joint_target_q)
_Model.joint_target_vel = property(lambda self: self.joint_target_qd)
_Control.joint_target_pos = property(lambda self: self.joint_target_q)
_Control.joint_target_vel = property(lambda self: self.joint_target_qd)

N_WORLDS = 8
WINDOW = 4
N_WINDOWS = 2
K_BOUNDARIES = WINDOW * N_WINDOWS
DT_OUTER = 2.0e-3
SEED = 20260816
TOL = 1e-5
R = 0.05
KE = 1.0e8

# ON-vs-OFF sliver bound: the only per-world construction-level deviation is
# float32 clock rounding of boundary-landing slivers (<= a few ULP of the
# window span per boundary). A sliver reschedules ~1e-9 s of integration at
# bounded velocity (|v| < 5 m/s here), giving per-boundary state deltas in
# the 1e-8 class; 1e-6 leaves two orders of headroom below any semantic
# error (a wrong contact set or bookkeeping shifts states at tol scale,
# >= 1e-5, or worse).
BOUND_Q = 1e-6
BOUND_QD = 1e-4

# Every solver flag the probe pins per arm (unset = solver default).
FLAGS = (
    "NEWTON_SAP_RUNAHEAD",
    "NEWTON_SAP_RUNAHEAD_WINDOW",
    "NEWTON_SAP_RUNAHEAD_PHASE",
    "NEWTON_SAP_DETERMINISTIC",
    "NEWTON_SAP_ATTEMPT_CONSISTENT_R",
    "NEWTON_ADAPTIVE_MARCH_LOG",
)


def build_model(n_worlds: int):
    t = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, n_worlds)
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def make_initial_conditions():
    """Per-world drop heights and impact speeds: every world impacts inside
    boundary 0, at per-world severity, so per-world dt spread (and hence
    genuine run-ahead) is generated by construction."""
    rng = np.random.default_rng(SEED)
    gap = rng.uniform(0.001, 0.003, N_WORLDS)
    vz = rng.uniform(-4.0, -2.0, N_WORLDS)
    return (R + gap).astype(np.float64), vz.astype(np.float64)


def _apply_env(env: dict[str, str]):
    for f in FLAGS:
        os.environ.pop(f, None)
    os.environ.update(env)


def run_arm(name: str, runahead: bool, world_subset: list[int], z0, vz):
    """March K_BOUNDARIES; record per-call clocks and window-edge state.

    ``world_subset`` selects which worlds exist in this arm's model (the
    solo arms build single-world models with the same per-world ICs)."""
    env = {
        # Rejection-generating law: under the attempt-consistent default this
        # scene accepts every attempt at the cap (uniform dt, no spread, no
        # run-ahead) -- vacuous for every guard below.
        "NEWTON_SAP_ATTEMPT_CONSISTENT_R": "0",
        # Canonical orders everywhere: the bitwise judgments require it.
        "NEWTON_SAP_DETERMINISTIC": "1",
        "NEWTON_SAP_RUNAHEAD_WINDOW": str(WINDOW),
    }
    if runahead:
        env["NEWTON_SAP_RUNAHEAD"] = "1"
    _apply_env(env)

    nw = len(world_subset)
    model = build_model(nw)
    coords = model.joint_coord_count // nw
    dofs = model.joint_dof_count // nw
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    q = state_0.joint_q.numpy()
    qd = state_0.joint_qd.numpy()
    for i, w in enumerate(world_subset):
        q[i * coords + 2] = z0[w]
        qd[i * dofs + 2] = vz[w]
    state_0.joint_q.assign(q)
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    state_1.assign(state_0)
    solver = newton.solvers.SolverSAPAdaptive(
        model,
        mode="adaptive",
        tol=TOL,
        dt_inner_init=DT_OUTER,
        dt_inner_min=1e-7,
        dt_inner_max=DT_OUTER,
        max_substeps=64,
    )
    assert solver.runahead == runahead, f"[{name}] runahead switch did not reach construction"

    calls = []
    edges = []
    for k in range(K_BOUNDARIES):
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        calls.append(
            {
                "sim_time": solver._sim_time.numpy().copy(),
                "next_time": solver._next_time.numpy().copy(),
            }
        )
        if (k + 1) % WINDOW == 0:
            edges.append(
                {
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
                }
            )
    crossings, adopts = solver.runahead_engagement()
    finite = all(np.isfinite(e["joint_q"]).all() and np.isfinite(e["joint_qd"]).all() for e in edges)
    assert finite, f"[{name}] non-finite committed state"
    return {
        "name": name,
        "n_worlds": nw,
        "coords": coords,
        "dofs": dofs,
        "calls": calls,
        "edges": edges,
        "crossings": crossings,
        "adopts": adopts,
    }


def rows(arr: np.ndarray, per: int, idx: int) -> np.ndarray:
    return arr[idx * per : (idx + 1) * per]


def world_fields(rec: dict, edge: dict, idx: int) -> dict[str, np.ndarray]:
    """Extract world ``idx``'s rows of one window-edge record."""
    # body arrays: bodies per world = body rows / n_worlds
    bq = edge["body_q"]
    n_b = bq.shape[0] // rec["n_worlds"]
    # "dt" is EXCLUDED: at a window edge it is attempt-transient dead state
    # (the boundary clamp zeroes a parked world's dt on every subsequent
    # march iteration, and the batch march runs more trailing iterations
    # than the solo march while other worlds finish; the next window-open
    # reseeds dt from ideal_dt, which IS compared). Every field below is
    # live carried state.
    return {
        "joint_q": rows(edge["joint_q"], rec["coords"], idx),
        "joint_qd": rows(edge["joint_qd"], rec["dofs"], idx),
        "body_q": bq[idx * n_b : (idx + 1) * n_b],
        "body_qd": edge["body_qd"][idx * n_b : (idx + 1) * n_b],
        "ideal_dt": edge["ideal_dt"][idx : idx + 1],
        "dt_ceiling": edge["dt_ceiling"][idx : idx + 1],
        "consec_rej": edge["consec_rej"][idx : idx + 1],
        "sim_time": edge["sim_time"][idx : idx + 1],
        "next_time": edge["next_time"][idx : idx + 1],
        "substeps_frame": edge["substeps_frame"][idx : idx + 1],
        "diverged": edge["diverged"][idx : idx + 1],
    }


def bitwise_equal(a: dict, b: dict) -> tuple[bool, str]:
    for f, va in a.items():
        if va.tobytes() != b[f].tobytes():
            return False, f
    return True, ""


def main() -> int:
    z0, vz = make_initial_conditions()
    all_worlds = list(range(N_WORLDS))

    on = run_arm("on-batch", True, all_worlds, z0, vz)
    on_rep = run_arm("on-batch-repeat", True, all_worlds, z0, vz)
    off = run_arm("off-batch", False, all_worlds, z0, vz)

    # ---- tier 1: engagement / vacuity ----
    expected_crossings = N_WORLDS * (WINDOW - 1) * N_WINDOWS
    if on["crossings"] != expected_crossings:
        print(
            f"VACUOUS: on-batch crossings {on['crossings']} != structural "
            f"{expected_crossings} (every world must cross each interior boundary exactly once)"
        )
        return 3
    if on["adopts"] <= 0:
        print("VACUOUS: on-batch consumed no collide+adopt batches")
        return 3
    # Genuine run-ahead: at some mid-window call, some world sits well past
    # the call target.
    targets = np.cumsum(np.full(K_BOUNDARIES, np.float32(DT_OUTER), dtype=np.float32)).astype(np.float32)
    max_ahead = 0.0
    for k, c in enumerate(on["calls"]):
        pos = k % WINDOW
        if pos == WINDOW - 1:
            continue
        t_call = float(targets[pos])  # window-local chain target
        max_ahead = max(max_ahead, float(c["sim_time"].max()) - t_call)
    if max_ahead < 0.25 * DT_OUTER:
        print(f"VACUOUS: no world ever ran ahead (max lead {max_ahead:.3e} s < {0.25 * DT_OUTER:.3e})")
        return 3
    if off["crossings"] != 0 or off["adopts"] != 0:
        print(f"LEAK: flag-OFF arm advanced run-ahead counters ({off['crossings']}, {off['adopts']})")
        return 3
    print(
        f"engagement: crossings={on['crossings']} (== structural), adopts={on['adopts']}, "
        f"max mid-window lead {max_ahead * 1e3:.3f} ms"
    )

    # ---- tier 2: oracle validity (determinism-in-mode) ----
    for e, (ea, eb) in enumerate(zip(on["edges"], on_rep["edges"], strict=True)):
        for f in ea:
            if ea[f].tobytes() != eb[f].tobytes():
                print(f"ORACLE-DEGRADED: on-batch repeat differs at window edge {e}, field {f}")
                return 2
    for k, (ca, cb) in enumerate(zip(on["calls"], on_rep["calls"], strict=True)):
        for f in ca:
            if ca[f].tobytes() != cb[f].tobytes():
                print(f"ORACLE-DEGRADED: on-batch repeat differs at call {k}, field {f} (mid-window clocks)")
                return 2
    print("oracle valid: ON repeat bitwise (window edges + mid-window clocks)")

    # ---- tier 3: per-world isolation (batch vs solo, bitwise) ----
    fail = False
    for w in all_worlds:
        solo = run_arm(f"on-solo-w{w}", True, [w], z0, vz)
        if solo["crossings"] != (WINDOW - 1) * N_WINDOWS:
            print(f"VACUOUS: solo world {w} crossings {solo['crossings']} != {(WINDOW - 1) * N_WINDOWS}")
            return 3
        for e in range(len(on["edges"])):
            bf = world_fields(on, on["edges"][e], w)
            sf = world_fields(solo, solo["edges"][e], 0)
            ok, field = bitwise_equal(bf, sf)
            if not ok:
                d = np.abs(bf[field].astype(np.float64) - sf[field].astype(np.float64)).max()
                print(f"FAIL isolation: world {w} window edge {e} field {field} differs (max |d| {d:.3e})")
                fail = True
    if fail:
        return 1
    print(f"isolation: batch world rows == solo runs, bitwise, {len(on['edges'])} window edges x {N_WORLDS} worlds")

    # ---- tier 4: semantic preservation vs the certified OFF march ----
    all_bitwise = True
    worst_q = 0.0
    worst_qd = 0.0
    for e in range(len(on["edges"])):
        for f in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a = on["edges"][e][f]
            b = off["edges"][e][f]
            if a.tobytes() != b.tobytes():
                all_bitwise = False
                d = np.abs(a.astype(np.float64).reshape(-1) - b.astype(np.float64).reshape(-1)).max()
                if f in ("joint_q", "body_q"):
                    worst_q = max(worst_q, d)
                else:
                    worst_qd = max(worst_qd, d)
    if all_bitwise:
        print("ON-vs-OFF: window-edge committed state BITWISE identical")
    else:
        print(
            f"ON-vs-OFF: not bitwise (expected class: float32 clock-rebase rounding of landing slivers); "
            f"max |dq| {worst_q:.3e} (bound {BOUND_Q:.1e}), max |dqd| {worst_qd:.3e} (bound {BOUND_QD:.1e})"
        )
        if worst_q > BOUND_Q or worst_qd > BOUND_QD:
            print("FAIL: ON-vs-OFF deviation exceeds the sliver bound -- semantic difference, not rounding")
            return 1

    print("SAP-RUNAHEAD-ORACLE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
