"""Flag-equivalence probe for the adaptive march's batched-scheduling flags.

Oracle argument (why this is not a tautology or a snapshot):
    NEWTON_ADAPTIVE_TWIN_EVAL and NEWTON_ADAPTIVE_TAIL_COMPACT are SCHEDULING
    changes, not numerical ones: the same physics kernels must run on the same
    per-world inputs in a different batching. The correct output of a flag-on
    run is therefore exactly computable by an independent route -- the flag-off
    run of the identical scenario -- executed in the SAME process, same build,
    same device, same seed. The probe asserts bitwise equality of everything
    the march commits or carries across boundaries. No golden files, no
    hand-picked tolerances: the reference is recomputed every invocation, and a
    flags-off-vs-flags-off repeat guards the oracle itself (if the platform is
    nondeterministic, feature equality cannot be judged bitwise and the probe
    says so instead of failing spuriously).

Assertion tiers:
    1. Vacuity guards (exit 3): the scene must actually exercise the machinery
       -- injected mjw contact rows present AND force-capable (dist <
       includemargin) in the REFERENCE run (measured post-injection via
       mjw_data.nacon, not at the pipeline-candidate stage, which can report
       rows the conversion never injects), a certified rejection (boundary 0
       needing >= 2 iterations -- dt is seeded at the cap there, so a second
       attempt can only follow a rejection; later boundaries cannot certify
       one because accepted steps also rewrite ideal_dt), the per-world dt
       diverging across worlds, twin runs showing the 2N-world data layout,
       and compaction runs ending some boundary with fewer active worlds than
       N (the ragged tail actually engaged -- otherwise the compacted and full
       paths run identical work and the compaction arms prove nothing). A
       probe that cannot fail is not a test.
    2. Oracle validity (exit 2, ORACLE-DEGRADED): reference vs reference-repeat
       must match bitwise; a mismatch means the environment itself reorders
       floating-point work run-to-run and bitwise feature judgments are void.
    3. Feature equivalence (exit 1): each flag configuration must match the
       reference bitwise on every recorded field. Compaction-only is STRICT:
       it changes no mjw launch and, by construction, no fp operation order
       (every compacted kernel writes world-private rows), so any mismatch is
       a real bug, full stop. For twin-bearing configurations a mismatch is
       printed as the first divergent (boundary, field, world, element) with
       hex bit patterns and ulp distance -- triage data, not a tolerance:
       whether ulp-level drift is acceptable is decided with the owner, not
       here.

Recorded per boundary (after step returns -- host reads are legal there):
    committed joint_q / joint_qd / body_q, the iteration count, cumulative
    substeps, and the controller-state arrays that carry the trajectory across
    boundaries (_ideal_dt, _dt, _dt_ceiling, _consec_rej, _sim_time,
    _next_time, diverged). Jointly these pin the accepted-step sequence:
    per-attempt dt records would require solver-side trace buffers (a
    preallocated device ring buffer written inside the captured march and read
    out afterwards -- capture-legal, but instrumentation the solver does not
    have); the probe records only post-march state.

Excluded by design: _last_error, _accepted_error, _accepted, _commit --
    attempt-transient telemetry that never feeds dynamics, the controller
    memory, or committed state. Under tail compaction the landed-world rows of
    _last_error/_accepted_error are documented byte-deltas: the compacted error
    kernel skips landed worlds (stale last-active value) where the full path
    writes fresh dt=0-eval garbage; the controller's dt <= 0 branch returns
    before either value can reach a decision.

Residual risk (what a clean pass does NOT establish):
    * Twin mode changes every mjw launch geometry (nworld = 2N), so on GPU the
      atomic-arrival order in contact-cid assignment and per-world efc row
      assembly can reorder floating-point reductions inside the constraint
      solve, producing ulp drift that may flip accept/reject near tolerance.
      A CPU pass (sequential launches) does not cover this; a GPU pass covers
      only the tested arch. If twin fails bitwise while the repeat run passes,
      the drift is real and the fallback criterion needs the owner's sign-off.
    * The Drake controller quantizes its output: the hysteresis deadband maps
      any new_step inside [k_Low*dt, k_High*dt] to dt, and the MIN_SHRINK /
      MAX_GROW clamps saturate outside it. An error-estimate corruption whose
      wrong per-world values land in the same deadband/clamp cell as the
      correct values on every attempt (e.g. a world permutation among worlds
      with near-equal errors) leaves ideal_dt, dt, and the accept decisions
      bit-identical and is masked. err ~= 0 corruptions (wrong gather offset,
      comparing identical ranges) move ideal_dt out of its cell and are
      caught through the recorded controller state.
    * The benign scene never reaches the dt floor, the NaN guard, the
      divergence sentinel, the ceiling knee, or max_substeps truncation; those
      paths stay unexercised.
    * The scene has no actuators: nu == 0 and na == 0, so ctrl mirroring and
      the act snapshot/seed path in twin mode are untested here.
    * Equality of the recorded fields does not check twin-row contents
      (expected garbage by construction) nor telemetry excluded above.
    * Compaction builds its index lists with atomics; the equality claim rests
      on every consumer writing world-private rows only. A CPU pass runs the
      list build sequentially, so GPU atomic-arrival patterns in the build
      itself (harmless by that argument, but unexercised here) wait for the
      GPU run. The compacted snapshot-freeze contract also depends on
      _restore_uncommitted_rows_kernel staying full-dim; if a later change
      compacts the restore, landed rows stay dt=0-poisoned and THIS probe is
      the tripwire (committed joint_q diverges).
    * Compaction under the conditional-march tier
      (NEWTON_MJ_ADAPTIVE_CONDITIONAL=1) and under max_substeps truncation is
      unexercised.

Run (single line, from the repo root; GPU by default, CPU via
CUDA_VISIBLE_DEVICES=""):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/march_equivalence_probe.py

Expected runtime: well under 2 minutes on a warm kernel cache (module
compilation dominates a cold run).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import warp as wp

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

N_WORLDS = 8
DT_OUTER = 1.0 / 60.0
K_BOUNDARIES = 5
SEED = 20260810
TOL = 1e-3
FORCE_SCALE = 0.5  # [N] / [N*m]; sized so ~0.2 kg boxes see O(g) accelerations

# The flags the march must be invariant to, singly and combined. The repeat
# run guards the oracle; the feature arms follow in strictness order
# (compaction changes no mjw launch, so it is judged strictly first).
ALL_FLAGS = ("NEWTON_ADAPTIVE_TWIN_EVAL", "NEWTON_ADAPTIVE_TAIL_COMPACT")
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("reference", {}),
    ("reference-repeat", {}),
    ("compact", {"NEWTON_ADAPTIVE_TAIL_COMPACT": "1"}),
    ("twin", {"NEWTON_ADAPTIVE_TWIN_EVAL": "1"}),
    (
        "twin+compact",
        {"NEWTON_ADAPTIVE_TWIN_EVAL": "1", "NEWTON_ADAPTIVE_TAIL_COMPACT": "1"},
    ),
]


def build_model():
    """One sub-world: static slab + a box resting on it (persistent contact)
    + a box dropped ~1 cm above the slab (impact mid-run), replicated into
    N_WORLDS worlds over a ground plane."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=100.0, kf=0.0, mu=0.5, margin=0.005)
    # Static slab: top face at z = 0.10.
    t.add_shape_box(-1, xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.05)), hx=0.2, hy=0.2, hz=0.05, cfg=cfg)
    # Resting box: seated on the slab from t = 0.
    rest = t.add_body(xform=wp.transform(p=wp.vec3(-0.08, 0.0, 0.13)))
    t.add_shape_box(rest, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
    # Falling box: bottom 1 cm above the slab, lands within the K boundaries.
    fall = t.add_body(xform=wp.transform(p=wp.vec3(0.08, 0.0, 0.14)))
    t.add_shape_box(fall, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane()
    return b.finalize()


def make_force_schedule(body_count: int) -> np.ndarray:
    """Fixed per-(boundary, body) wrenches, identical for every configuration.

    Drawn once from a seeded generator BEFORE any run so every configuration
    consumes the identical sequence; per-world variation (bodies of different
    worlds get different draws) is what spreads the per-world dt apart and
    exercises the ragged tail."""
    rng = np.random.default_rng(SEED)
    return rng.normal(0.0, FORCE_SCALE, size=(K_BOUNDARIES, body_count, 6)).astype(np.float32)


def run_config(name: str, env: dict[str, str], forces: np.ndarray | None):
    """Build a fresh model + solver + pipeline under the given flags and march
    K_BOUNDARIES boundaries, recording the committed per-boundary state."""
    for flag in ALL_FLAGS:
        os.environ.pop(flag, None)
    os.environ.update(env)

    model = build_model()
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=TOL,
        dt_inner_init=DT_OUTER,
        dt_inner_min=1e-6,
        dt_inner_max=DT_OUTER,
        nconmax=32,
        njmax=128,
        use_newton_contacts=True,
    )
    twin_on = env.get("NEWTON_ADAPTIVE_TWIN_EVAL") == "1"
    compact_on = env.get("NEWTON_ADAPTIVE_TAIL_COMPACT") == "1"
    if twin_on:
        # Feature-armed vacuity guard: if the flag silently stopped reaching
        # construction (e.g. someone moves the read to import time), this run
        # would silently equal the reference and the probe would prove nothing.
        assert solver.mjw_data.nworld == 2 * N_WORLDS, solver.mjw_data.nworld
        assert len(solver.mjw_model.opt.timestep) == 2 * N_WORLDS
    if compact_on:
        # Same tripwire for the compaction flag at construction time; the
        # iteration-body counterpart (the list build actually running) is
        # checked per boundary below via _active_counts.
        assert solver._tail_compact, "NEWTON_ADAPTIVE_TAIL_COMPACT did not reach construction"

    # deterministic=True sorts contact emission: without it the baseline march is
    # not run-to-run bitwise reproducible (atomic-ordered contact rows feed the
    # constraint assembly), and the reference-repeat oracle check fails before
    # any feature comparison can be judged.
    pipeline = newton.CollisionPipeline(
        model, rigid_contact_max=solver.get_max_contact_count(), deterministic=True
    )
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    if forces is None:
        forces = make_force_schedule(state_0.body_f.shape[0])

    device = model.device
    records = []
    injected_seen = 0
    active_injected_seen = 0
    final_active: list[int] | None = [] if compact_on else None
    for k in range(K_BOUNDARIES):
        pipeline.collide(state_0, contacts)
        wrench = wp.array(forces[k], dtype=wp.spatial_vector, device=device)

        def apply_forces(state, wrench=wrench):
            wp.copy(state.body_f, wrench)

        state_0, state_1 = solver.step(state_0, state_1, control, contacts, DT_OUTER, apply_forces=apply_forces)
        # Contact vacuity data, measured where it matters: nacon counts the
        # rows _convert_contacts_to_mjwarp actually injected (pipeline
        # candidates can be dropped wholesale by a conversion-side regression
        # and prove nothing), and a row is force-capable only when
        # dist < includemargin (mjw assembles constraint rows under that
        # test; margin-only candidates never produce force). Post-march host
        # reads are legal. Twin runs count duplicated twin rows too; the
        # guard consumes the reference run only.
        nacon = int(solver.mjw_data.nacon.numpy()[0])
        injected_seen = max(injected_seen, nacon)
        if nacon > 0:
            dist = solver.mjw_data.contact.dist.numpy()[:nacon]
            margin = solver.mjw_data.contact.includemargin.numpy()[:nacon]
            active_injected_seen = max(active_injected_seen, int((dist < margin).sum()))
        if twin_on:
            # A truncated twin duplication (primary injected rows exceeding
            # naconmax // 2) makes twins integrate with missing contacts;
            # tier 3 would then report an unexplained bitwise divergence and
            # the twin-geometry note would misdirect triage toward fp
            # reordering. Fail loudly with the real cause instead.
            assert not solver.twin_contact_overflow, (
                f"twin contact capacity exceeded at boundary {k}: primary injected rows > naconmax // 2 -- raise nconmax"
            )
        records.append(
            {
                "joint_q": state_0.joint_q.numpy().copy(),
                "joint_qd": state_0.joint_qd.numpy().copy(),
                "body_q": state_0.body_q.numpy().copy(),
                "iterations": np.array([int(solver.iteration_count.numpy()[0])], dtype=np.int64),
                "cumulative_substeps": np.array([solver.cumulative_substeps()], dtype=np.int64),
                "ideal_dt": solver._ideal_dt.numpy().copy(),
                "dt": solver._dt.numpy().copy(),
                "dt_ceiling": solver._dt_ceiling.numpy().copy(),
                "consec_rej": solver._consec_rej.numpy().copy(),
                "sim_time": solver._sim_time.numpy().copy(),
                "next_time": solver._next_time.numpy().copy(),
                "diverged": solver.diverged.numpy().copy(),
            }
        )
        if compact_on:
            # Last-iteration compaction counters (persist after the march):
            # counts[0] = active worlds in the final iteration (>= 1: the last
            # landing worlds were still active), counts[1] = snapshot set
            # (superset of active). Zeroes here mean the list build never ran
            # -- the flag reached construction but not the iteration body.
            counts = solver._active_counts.numpy()
            assert counts[0] >= 1, f"compaction list build never ran (counts={counts.tolist()})"
            assert counts[1] >= counts[0], counts.tolist()
            final_active.append(int(counts[0]))
    return (
        forces,
        records,
        {
            "injected_seen": injected_seen,
            "active_injected_seen": active_injected_seen,
            "final_active": final_active,
        },
    )


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lexicographic-order distance between float32 bit patterns (0 == equal)."""
    ai = a.astype(np.float32).view(np.int32).astype(np.int64)
    bi = b.astype(np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**31)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**31)) - bi, bi)
    return np.abs(ai - bi)


def _row_bytes(arr: np.ndarray) -> np.ndarray:
    """Byte view of each leading-dim row: shape (rows, bytes_per_row)."""
    a = np.ascontiguousarray(arr)
    return a.view(np.uint8).reshape(a.shape[0], -1)


def first_divergence(ref, cand):
    """Return (boundary, field, row) of the first byte-differing row, or None."""
    for k, (r, c) in enumerate(zip(ref, cand, strict=True)):
        for field in r:
            ra, ca = r[field], c[field]
            if ra.shape != ca.shape or ra.dtype != ca.dtype:
                return k, field, -1
            rb, cb = _row_bytes(ra), _row_bytes(ca)
            if not np.array_equal(rb, cb):
                return k, field, int(np.flatnonzero((rb != cb).any(axis=1))[0])
    return None


def report_divergence(name, ref, cand, where, twin_geometry: bool):
    k, field, row = where
    print(
        f"FAIL[{name}]: first divergence at boundary {k}, field '{field}', row {row} (row=world for per-world fields)"
    )
    if row >= 0:
        ra = np.ascontiguousarray(ref[k][field])[row].ravel()
        ca = np.ascontiguousarray(cand[k][field])[row].ravel()
        diff = _row_bytes(ra.reshape(ra.size, -1) if ra.ndim == 1 else ra) != _row_bytes(
            ca.reshape(ca.size, -1) if ca.ndim == 1 else ca
        )
        j = int(np.flatnonzero(diff.any(axis=1))[0])
        rv, cv = ra[j], ca[j]
        print(f"  element {j}: ref={rv!r} cand={cv!r}")
        if ra.dtype == np.float32:
            rb = int(np.frombuffer(np.float32(rv).tobytes(), dtype=np.uint32)[0])
            cb = int(np.frombuffer(np.float32(cv).tobytes(), dtype=np.uint32)[0])
            u = ulp_distance(np.array([rv]), np.array([cv]))[0]
            print(f"  bits: ref=0x{rb:08x} cand=0x{cb:08x} ulp={u}")
    if twin_geometry:
        print(
            "  note: twin mode changes every mjw launch geometry (nworld=2N); on GPU,"
            " atomic-arrival order in contact-cid / efc assembly can reorder fp"
            " reductions, so ulp-level drift here is the known suspect (see docstring)."
        )
    else:
        print(
            "  note: this configuration changes no mjw launch and no fp operation"
            " order by construction -- this mismatch is a real bug, not drift."
        )


def main() -> int:
    forces, ref, ref_extras = run_config(*CONFIGS[0], forces=None)

    runs = {}
    extras = {}
    for name, env in CONFIGS[1:]:
        _, rec, ext = run_config(name, env, forces)
        runs[name] = rec
        extras[name] = ext

    # --- Tier 1: vacuity guards -------------------------------------------
    iters = np.array([r["iterations"][0] for r in ref])
    dt_spread = any(len(np.unique(r["dt"])) > 1 for r in ref)
    # The ragged tail must actually engage for the compaction arms: some
    # boundary must END with fewer active worlds than N (worlds landed at
    # different iterations), otherwise the compacted and full paths run
    # identical work and their equality proves nothing.
    tail_engaged = all(any(c < N_WORLDS for c in extras[name]["final_active"]) for name in ("compact", "twin+compact"))
    # Rejection certification must use boundary 0: the controller rewrites
    # ideal_dt on ACCEPTED steps too, so from boundary 1 onward a world can
    # carry ideal_dt < DT_OUTER and take multiple accepted sub-steps with zero
    # rejections -- (iters >= 2).any() proves nothing about the reject path.
    # At boundary 0 every world's dt is seeded at the cap (ideal_dt starts at
    # dt_inner_init = DT_OUTER and _apply_dt_cap clamps to effective_dt_max =
    # DT_OUTER), so an accepted first attempt lands the world in one
    # iteration and a second iteration can ONLY come from a rejection.
    guards = {
        "reference run injected mjw contact rows (nacon > 0)": ref_extras["injected_seen"] > 0,
        "reference run had force-capable contacts (dist < includemargin)": ref_extras["active_injected_seen"] > 0,
        "rejection exercised (boundary 0 needed >= 2 iterations)": bool(iters[0] >= 2),
        "per-world dt diverged across worlds": dt_spread,
        "compaction tail engaged (< N active at some boundary end)": tail_engaged,
    }
    bad = [g for g, ok in guards.items() if not ok]
    for g, ok in guards.items():
        print(f"guard: {g}: {'ok' if ok else 'NOT MET'}")
    if bad:
        print("VACUOUS: the scene failed to exercise the march; fix the scene, not the flags.")
        return 3

    # --- Tier 2: oracle validity ------------------------------------------
    where = first_divergence(ref, runs["reference-repeat"])
    if where is not None:
        report_divergence("reference-repeat", ref, runs["reference-repeat"], where, twin_geometry=False)
        print("ORACLE-DEGRADED: flags-off is not run-to-run deterministic on this device;")
        print("bitwise feature equality cannot be judged here.")
        return 2

    # --- Tier 3: feature equivalence --------------------------------------
    failed = False
    for name, env in CONFIGS[2:]:
        where = first_divergence(ref, runs[name])
        if where is None:
            print(f"PASS[{name}]: bitwise identical to reference over {K_BOUNDARIES} boundaries")
        else:
            report_divergence(name, ref, runs[name], where, twin_geometry="NEWTON_ADAPTIVE_TWIN_EVAL" in env)
            failed = True
    if failed:
        return 1
    print(f"MARCH-EQUIVALENCE: PASS (iterations per boundary: {iters.tolist()})")
    print(f"  compaction final-iteration active counts per boundary: {extras['compact']['final_active']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
