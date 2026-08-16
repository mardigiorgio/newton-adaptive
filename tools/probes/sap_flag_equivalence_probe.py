"""Flag-equivalence + functional-smoke probe for SolverSAPAdaptive's
behavior-preserving switches (mirrors tools/probes/march_equivalence_probe.py).

Oracle argument (why this is not a tautology or a snapshot):
    The switches under test are SCHEDULING or OBSERVER changes, not numerical
    ones:
      * NEWTON_SAP_ADAPTIVE_GRAPH captures the substep body into a CUDA graph
        and replays it -- the same kernel sequence on the same inputs, launched
        by the driver instead of Python;
      * NEWTON_SAP_ADAPTIVE_CONDITIONAL records the WHOLE march as one
        conditional while-node (wp.capture_while) whose body is that same
        substep launch stream plus one dim-1 loop-condition kernel writing a
        private word nothing else reads -- the per-iteration host flag poll
        becomes a device-side loop condition, and the trip count is pinned to
        the per-iteration tier's by the identical in-body status/cap kernels;
      * NEWTON_ADAPTIVE_MARCH_LOG adds one reduction kernel (_count_rejects,
        writing only its private counter) plus post-march host READS;
      * dt_histogram=True adds accumulation kernels that write only the
        histogram buffers;
      * NEWTON_ADAPTIVE_TAIL_COMPACT restricts the per-world SAP pipeline
        work to still-marching worlds (landed worlds' solves are skipped and
        their scratch rows go stale) -- a scheduling change because every
        skipped value is DEAD by construction: the accept-gated commit and
        the controller's DONE branch never read a landed world's solve
        outputs, and each active world's kernels keep their exact fp
        operation order. Collision and the contact scatter stay full-width
        in both arms.
      * NEWTON_SAP_LS_COMPACT restricts the per-line-search-trip trial
        impulse assembly to envs still line-searching (a device-built subset
        of the Newton-trip list, rebuilt at each ls_active mutation) -- a
        scheduling change because de-listed envs' trial rows are frozen
        (their writers are ls_active-masked) and the skipped consumers are
        pure per-env overwrites of those frozen terms, so skipping leaves
        the exact values a full-width relaunch would recompute.
      * NEWTON_SAP_MARCH_COMPACT records the eval core (three SAP solves +
        error metric) as a device-side conditional with a wide body (the
        unchanged launch stream) and a narrow body (the SAME kernels with
        every list-indexed launch's env axis sized to a small budget),
        routed per march iteration by the active-world count -- a scheduling
        change because a list-indexed kernel exits at the device-read list
        count, so grid slots beyond it never did work at any width; the
        subset chain (world_active -> stage2 -> newton -> LS) bounds every
        list by the active count, the branch predicate bounds that by the
        budget, and in-branch capacity guards latch a poison word (the
        solver raises) on any violation. Mask-gated and full-width-by-design
        kernels (status, collision, contact scatter, controller/commit) are
        identical in both bodies. The deterministic active-list build
        becomes canonical-ascending under this flag; consumers write
        world-private rows, so list order is bitwise-irrelevant.
        Engagement is proven at three levels: both branch-execution
        counters advance (allocated unconditionally, so an OFF cell's
        (0, 0) is a real device read, not a restatement of the flag), the
        in-branch capacity guard stays clean, and the contact solve's
        emission-time site record must contain every converted launch
        family this scene drives -- with OFF cells showing an EMPTY record
        (no site ever emitted narrowed).
      * NEWTON_SAP_CONTAINMENT swaps the boundary-exit kernel for a variant
        that keeps a per-world solve failure OUT of the batch status word
        (containing it in the failing world's own reject/latch machinery)
        and adds one per-boundary memset of the sticky failure latch -- on a
        scene where every solve converges (this one; asserted via the
        solver's failure counters staying zero) neither kernel takes a
        failure branch, so the committed trajectories must be bitwise
        identical. The failure-path BEHAVIOR difference is real and is owned
        by tools/probes/sap_containment_probe.py, which forces failures.

    NEWTON_ADAPTIVE_CONTACT_REFRESH is NOT a scheduling switch: its two
    cadences (collide once per boundary vs at q_t and q_{t+h/2} every
    attempt) are different contact physics, so no bitwise relation across
    cadences exists or is asserted. Cells are grouped into cadence FAMILIES:
    the legacy cells pin "attempt" and are judged against "reference"; the
    boundary-cadence cells pin "1" and are judged against their own
    "boundary" reference (with a boundary-repeat oracle guard). Cadence
    engagement is proven by counting CollisionPipeline invocations
    host-side: 2 per attempt (eager) vs exactly 1 per boundary.

    NEWTON_SAP_ATTEMPT_CONSISTENT_R (default ON, "0" disables) is likewise
    NOT a scheduling switch: it pins the trial solves' near-rigid clamps to
    the attempt's dt, which is a different constitutive law for the
    step-doubling pair, so no bitwise relation across its states exists.
    Every scheduling-switch cell pins it "0" explicitly -- under the
    attempt-consistent law this scene accepts every attempt at the cap
    (no rejections, no per-world dt spread), which would leave the
    rejection/spread/compaction vacuity guards unmeetable. The ACR family
    cells leave the variable UNSET so the solver's default resolution is
    itself under test (a construction tripwire asserts it lands ON and
    that the constitutive dt buffer reaches the contact solve), with their
    own reference + repeat oracle and graph/conditional arms certifying
    that the attempt-consistent launch stream captures and replays
    bitwise. Law engagement is proven by divergence: the ACR family's
    committed march must differ from the ACR-off boundary family's, and
    every other flag separating those cells is individually certified
    bitwise-invisible, so any byte difference is the law's.

    NEWTON_SAP_FUSED_LS (default ON, "0" disables) is likewise NOT a
    scheduling switch: the fused armijo ladder evaluates trial costs in a
    different fp order (ray-advanced contact velocities, tile-parallel
    regularizer reduction) than the launch chain, so no bitwise relation
    across its states exists or is asserted. Every legacy cell pins "0"
    (preserving the launch-chain stream those cells certify, and arming
    the OFF-leak counter assert); the fused-LS family cells leave the
    variable UNSET so the default resolution is itself under test, with
    their own reference + repeat oracle and graph/conditional arms
    certifying that the single-kernel ladder stream captures and replays
    bitwise. Ladder engagement is proven by the device ladder-env counter
    (> 0 in family cells, == 0 in every pinned-off cell) -- not by
    divergence, which a scene without threshold-adjacent accept decisions
    legally never shows.

    NEWTON_SAP_FUSED_ALPHAMAX (default ON, "0" disables; effective only
    under the fused ladder) folds the alpha_max trial into the ladder
    kernel's rung 0 -- another fp evaluation-route change (ray-advanced
    trial cost, analytic ray derivative in place of the chain's
    dp/impulse dot products), so it gets the same treatment: every other
    cell pins "0", the fused-alphamax family cells leave it UNSET with
    their own reference + repeat oracle and graph/conditional arms, and
    engagement is the device alphamax-env counter.
    The correct output of a switch-on run is therefore exactly computable by an
    independent route: the switch-off run of the identical scenario, same
    process, same build, same device, same seed. The probe asserts bitwise
    equality of everything the march commits or carries across boundaries. No
    golden files, no hand-picked tolerances: the reference is recomputed every
    invocation, and a flags-off-vs-flags-off repeat guards the oracle itself
    (if the platform reorders floating-point work run-to-run, bitwise feature
    equality cannot be judged and the probe says so instead of failing
    spuriously).

    Deliberately NOT tested here: NEWTON_ADAPTIVE_RTOL and
    NEWTON_ADAPTIVE_CEILING_RELAX change controller behavior BY DESIGN (error
    metric / growth bound); on-vs-off equivalence is not their contract.
    CEILING_RELAX is additionally an import-time wp.constant, so it is uniform
    across every cell of this probe by construction. RTOL is left at its
    ambient default in every cell (uniform, hence never the varied factor).

Assertion tiers:
    1. Vacuity guards (exit 3): the scene must actually exercise the machinery
       -- live pipeline contacts in the reference run, a certified rejection
       (boundary 0 needing >= 2 iterations: dt is seeded at the cap there and
       the quantile stop is off, so a second iteration can only follow a
       rejection), per-world dt divergence, and per-arm engagement tripwires
       (graph cache populated with capture still enabled; march-log CSV rows
       written; histogram counts accumulated). A probe that cannot fail is not
       a test.
    2. Oracle validity (exit 2, ORACLE-DEGRADED): reference vs reference-repeat
       must match bitwise; a mismatch means the environment itself is not
       run-to-run deterministic and bitwise feature judgments are void.
    3. Feature equivalence (exit 1): each arm must match the reference bitwise
       on every recorded field. A mismatch prints the first divergent
       (boundary, field, row, element) with hex bit patterns and ulp distance
       -- triage data for the owner, not a tolerance.
    4. Functional smoke (exit 4): several hundred step_dt boundaries with the
       optimizations ON and OFF; every world must end finite in joint_q /
       joint_qd / body_q, and per-boundary wall time is reported for both.

Recorded per boundary (after step_dt returns -- host reads are legal there):
    committed joint_q / joint_qd / body_q / body_qd, the iteration count,
    cumulative substeps, and the controller-carry arrays (_ideal_dt, _dt,
    _dt_ceiling, _consec_rej, _sim_time, _next_time, diverged, _guard_hits).
    Jointly these pin the accepted-step sequence.

Excluded by design: _last_error / _accepted_error / _accepted / _limited --
    attempt-transient telemetry that never survives into the next boundary's
    dynamics, controller memory, or committed state.

Residual risk (what a clean pass does NOT establish):
    * The graph arm exercises capture-replay only for THIS scene's kernel set;
      scenes hitting other code paths (divergence sentinel, dt floor, quantile
      abandonment, max_substeps truncation) revalidate separately.
    * Contact machinery is exercised by a single sphere-on-plane pair per
      world; multi-contact manifolds and mesh collisions are untested here.
    * The scene has no actuators (control is empty), so the control path is
      untested.
    * Equality of the recorded fields does not check the excluded telemetry.
    * The manager-side single-boundary gate (mjwarp_manager.py) is outside
      this probe's scope; it drives the solver through IsaacLab, not here.

Run (single line, from the newton-adaptive repo root; GPU required for the
graph arm -- on CPU the probe drops it and says so):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_flag_equivalence_probe.py
"""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import time

# sap_warp is a sibling checkout of this repo; SAP_WARP_PATH overrides.
_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, os.environ.get("SAP_WARP_PATH", str(_REPO.parent / "sap_warp")))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

# PROBE-LOCAL COMPATIBILITY SHIM (no source edit): the sap_warp checkout reads
# the pre-1.5 names Model.joint_target_pos / joint_target_vel, which the
# working tree tombstones with RemovedAttribute; alias them to the 1.5 names.
# The model here builds with the legacy DOF-shaped target layout
# (newton.use_coord_layout_targets is False), which is the layout the old
# names carried, so the alias is shape-faithful for this scene.
from newton._src.sim.control import Control as _Control  # noqa: E402
from newton._src.sim.model import Model as _Model  # noqa: E402

_Model.joint_target_pos = property(lambda self: self.joint_target_q)
_Model.joint_target_vel = property(lambda self: self.joint_target_qd)
# Control must alias too: sap_warp reads control.joint_target_pos via getattr
# with a None fallback, and its None-fallback placeholder is solve-dtype
# (float64) where the PD-terms kernel wants the control dtype (float32) --
# the alias routes the real target arrays instead of the placeholder.
_Control.joint_target_pos = property(lambda self: self.joint_target_q)
_Control.joint_target_vel = property(lambda self: self.joint_target_qd)

N_WORLDS = 8
DT_OUTER = 2.0e-3
K_BOUNDARIES = 6
SMOKE_STEPS = 300
SMOKE_WARMUP = 30
SEED = 20260813
# Tight enough that the impact boundary's step-doubling position error MUST
# exceed it at the seeded 2 ms cap (forcing the certified rejection and the
# per-world dt spread the vacuity guards demand); loose enough that free
# flight still accepts at the cap.
TOL = 1e-5
R = 0.05
KE = 1.0e8

# Env switches the march must be invariant to. Reference cells PIN every
# switch off explicitly: NEWTON_SAP_ADAPTIVE_GRAPH and
# NEWTON_ADAPTIVE_TAIL_COMPACT are default-ON, and a reference that inherits
# a feature under test judges the feature against itself.
# NEWTON_ADAPTIVE_CONTACT_REFRESH is different in kind: its cadences are two
# DIFFERENT PHYSICS (boundary-frozen vs per-attempt contact sets), so cells
# are grouped into families by cadence and bitwise equality is only ever
# judged WITHIN a family -- the legacy cells pin "attempt" (the pre-hoist
# cadence, keeping them bitwise-valid against each other), and the boundary
# family pins "1" (the new default) with its own reference + repeat oracle.
ALL_FLAGS = (
    "NEWTON_SAP_ADAPTIVE_GRAPH",
    "NEWTON_SAP_ADAPTIVE_CONDITIONAL",
    "NEWTON_ADAPTIVE_MARCH_LOG",
    "NEWTON_ADAPTIVE_TAIL_COMPACT",
    "NEWTON_ADAPTIVE_CONTACT_REFRESH",
    "NEWTON_SAP_SOLVE_COMPACT",
    "NEWTON_SAP_LS_COMPACT",
    "NEWTON_SAP_DETERMINISTIC",
    "NEWTON_SAP_CONTAINMENT",
    "NEWTON_SAP_MARCH_COMPACT",
    "NEWTON_SAP_MARCH_COMPACT_WIDTH",
    "NEWTON_SAP_SHARED_ASSEMBLY",
    "NEWTON_SAP_ATTEMPT_CONSISTENT_R",
    "NEWTON_SAP_FUSED_LS",
    "NEWTON_SAP_FUSED_ALPHAMAX",
    "NEWTON_SAP_PACK_PERCONTACT",
    "NEWTON_SAP_FUSED_UPDATE",
    "NEWTON_SAP_NARROW_V3",
)
# Uniform-pinned across every cell (never the varied factor).
PINNED_OFF = ("NEWTON_SAP_SPREAD_LOG", "NEWTON_ADAPTIVE_DT_HIST")

_SCRATCH = tempfile.mkdtemp(prefix="sap_flag_probe_")

# Narrow-site families every march-compact arm of THIS scene must have
# emitted with a narrowed env grid: the per-solve prepare family emits on
# every narrow eval-core emission; the Newton/LS families emit whenever a
# narrow-routed iteration carries a constrained world (this scene's
# stragglers rest on the ground, so one always does). The armijo default
# line search drives the no-hessian trial family. The attempt-consistent
# scale sites (prep_scale_a_inv_pd/limit, scale_w_eff) are NOT listed:
# they emit only with the attempt-consistent law ON, and every
# march-compact cell here pins NEWTON_SAP_ATTEMPT_CONSISTENT_R=0 (under
# that law this scene accepts every attempt at the cap, leaving the
# compaction machinery nothing to engage).
EXPECTED_NARROW_SITES = frozenset(
    {
        "prep_copy_guess",
        "prep_a_diag",
        "prep_build_pd",
        "prep_build_limit",
        "prep_clear_pdof",
        "prep_mark_contact_pdof",
        "prep_mark_model_pdof",
        "proj_hessian",
        "hessian_total",
        "base_cost",
        "axpy_trial_init",
        "proj_gamma",
        "eval_pd",
        "eval_limit",
        "commit_ls",
        "proj_gamma_update",
        "model_terms_grad",
        "chol_factorize",
        "chol_solve",
    }
)

# The narrow-v3 routing's own list-indexed sites (contact_solve side): the
# fused update eval through the prepare list, the serial LS direction chain
# (direction data, state init, iteration accumulate) plus its base-cost call
# through the newton list. They may emit ONLY in narrow-v3 ON cells; any
# appearance in a pinned-off cell is a leak.
NARROW_V3_SITES = frozenset(
    {
        "fused_update",
        "ls_search_direction",
        "ls_init_backtracking",
        "ls_accumulate_iters",
    }
)

# Narrow-site families for march-compact cells running the PRODUCTION fused
# stack (fused LS ladder + folded alpha-max + fused update eval + per-contact
# pack): the chain-only trip sites (proj_hessian, axpy_trial_init, proj_gamma
# / eval / model_terms_grad, proj_gamma_update) never launch there, and
# base_cost narrows only through the v3 routing. Subset-asserted like the
# chain set: extra emitted tags are fine, missing ones are lost plumbing.
EXPECTED_NARROW_SITES_FUSED_STACK = frozenset(
    {
        "prep_copy_guess",
        "prep_a_diag",
        "prep_build_pd",
        "prep_build_limit",
        "prep_clear_pdof",
        "prep_mark_contact_pdof",
        "prep_mark_model_pdof",
        "hessian_total",
        "commit_ls",
        "chol_factorize",
        "chol_solve",
        "pack_j_solve",
    }
)
EXPECTED_NARROW_SITES_FUSED_STACK_V3 = EXPECTED_NARROW_SITES_FUSED_STACK | NARROW_V3_SITES | {"base_cost"}


def _cfg(
    name: str,
    graph: str,
    march_log: str | None,
    dt_hist: bool,
    compact: str = "0",
    refresh: str = "attempt",
    solve_compact: str = "0",
    ls_compact: str = "0",
    conditional: str = "0",
    containment: str = "0",
    march: str = "0",
    shared: str = "0",
    gemm: str = "0",
    acr: str | None = "0",
    fused: str | None = "0",
    alphamax: str | None = "0",
    pack: str = "0",
    fusedupdate: str | None = "0",
    narrowv3: str | None = "0",
):
    env = {
        "NEWTON_SAP_ADAPTIVE_GRAPH": graph,
        # Default-ON in the solver; every cell pins it so the whole-march
        # conditional tier is only ever the explicitly varied factor.
        "NEWTON_SAP_ADAPTIVE_CONDITIONAL": conditional,
        "NEWTON_ADAPTIVE_TAIL_COMPACT": compact,
        "NEWTON_ADAPTIVE_CONTACT_REFRESH": refresh,
        "NEWTON_SAP_SOLVE_COMPACT": solve_compact,
        "NEWTON_SAP_LS_COMPACT": ls_compact,
        # Default-ON in the solver; every cell pins it so the narrow-grid
        # tail body is only ever the explicitly varied factor.
        "NEWTON_SAP_MARCH_COMPACT": march,
        # Default-ON in the solver; reference cells pin the strict boundary
        # kernel so containment is only ever the explicitly varied factor.
        "NEWTON_SAP_CONTAINMENT": containment,
        # Default-ON in the solver; every cell pins it so half-1 assembly
        # reuse is only ever the explicitly varied factor.
        "NEWTON_SAP_SHARED_ASSEMBLY": shared,
        # Default-ON in the solver; every cell pins it so the live-k bounded
        # contact-Hessian GEMM pair is only ever the explicitly varied factor.
        "NEWTON_SAP_GEMM_RESHAPE": gemm,
        # Default-ON in the solver (effective only under the bounded GEMM
        # pair); every cell pins it so the per-contact read-once pack +
        # per-solve j_flat hoist is only ever the explicitly varied factor.
        # Bitwise class: ON keeps the per-element gj arithmetic and the
        # operand bytes the bounded GEMM reads identical, so pack arms are
        # judged bitwise against their family reference.
        "NEWTON_SAP_PACK_PERCONTACT": pack,
        # Pinned UNIFORMLY (never the varied factor): the deterministic and
        # legacy accumulation orders are different fp orders by design, so no
        # bitwise contract exists across them. All cells run the new
        # deterministic default; each family's reference is recomputed per
        # invocation, so this is an honest re-baseline, not a golden file.
        "NEWTON_SAP_DETERMINISTIC": "1",
    }
    # Attempt-consistent trial law (default ON in the solver) is DIFFERENT
    # PHYSICS for the step-doubling pair: scheduling-switch cells pin "0"
    # explicitly (the law whose rejections/spread this scene's guards
    # need); ACR-family cells pass None to leave the variable unset, so
    # the default resolution itself is the thing under test.
    if acr is not None:
        env["NEWTON_SAP_ATTEMPT_CONSISTENT_R"] = acr
    # Fused armijo line-search ladder (default ON in the solver) is a
    # DIFFERENT fp reduction/evaluation order for the trial costs, so no
    # bitwise contract exists across its states: every legacy cell pins
    # "0" (the launch-chain stream, byte-for-byte the pre-flag behavior),
    # and the fused-LS family cells pass None to leave the variable unset
    # so the default resolution is itself under test, with their own
    # reference + repeat oracle and graph/conditional arms.
    if fused is not None:
        env["NEWTON_SAP_FUSED_LS"] = fused
    # The alpha-max rung folded into the fused ladder (default ON in the
    # solver) is a DIFFERENT fp evaluation route for the alpha_max trial
    # cost/derivative, so no bitwise contract exists across its states:
    # every other cell pins "0" (the per-trip trial launch chain,
    # byte-for-byte the pre-flag behavior -- the fused-LS family keeps
    # certifying the exact stream it certified before the fold existed),
    # and the fused-alphamax family cells pass None to leave the variable
    # unset so the default resolution is itself under test, with their own
    # reference + repeat oracle and graph/conditional arms.
    if alphamax is not None:
        env["NEWTON_SAP_FUSED_ALPHAMAX"] = alphamax
    # The fused post-commit update evaluation (default ON in the solver) is
    # a DIFFERENT fp reduction route for the committed-point cost, gradient
    # totals and norms, so no bitwise contract exists across its states:
    # every other cell pins "0" (the launch-chain stream, byte-for-byte the
    # pre-flag behavior), and the fused-update family cells pass None to
    # leave the variable unset so the default resolution is itself under
    # test, with their own reference + repeat oracle and graph/conditional
    # arms.
    if fusedupdate is not None:
        env["NEWTON_SAP_FUSED_UPDATE"] = fusedupdate
    # The narrow-v3 routing (default ON in the solver) is BITWISE class: it
    # only changes which envs launch (list-indexed grids, world-gated
    # scatter rows) with per-env arithmetic untouched, so unlike the fused
    # families its ON arm is judged bitwise against a same-stack pinned-off
    # reference. Every legacy cell pins "0" (the fixed full-width launches,
    # byte-for-byte the pre-flag behavior); the narrow-v3 family cells pass
    # None so the default resolution is itself under test.
    if narrowv3 is not None:
        env["NEWTON_SAP_NARROW_V3"] = narrowv3
    if march_log is not None:
        env["NEWTON_ADAPTIVE_MARCH_LOG"] = march_log
    return (name, env, dt_hist)


CONFIGS = [
    _cfg("reference", "0", None, False),
    _cfg("reference-repeat", "0", None, False),
    _cfg("graph", "1", None, False),
    _cfg("march-log", "0", os.path.join(_SCRATCH, "march_log_arm.csv"), False),
    _cfg("dt-hist", "0", None, True),
    _cfg("tail-compact", "0", None, False, compact="1"),
    _cfg("tail-compact-graph", "1", None, False, compact="1"),
    _cfg(
        "all-on",
        "1",
        os.path.join(_SCRATCH, "march_log_allon.csv"),
        True,
        compact="1",
        solve_compact="1",
        ls_compact="1",
    ),
    _cfg("solve-compact", "0", None, False, solve_compact="1"),
    _cfg("solve-compact-graph", "1", None, False, solve_compact="1"),
    _cfg("ls-compact", "0", None, False, ls_compact="1"),
    _cfg("ls-solve-compact", "0", None, False, solve_compact="1", ls_compact="1"),
    _cfg("containment", "0", None, False, containment="1"),
    # ---- boundary-cadence family (the new default: refresh="1") ----
    _cfg("boundary", "0", None, False, compact="1", refresh="1"),
    _cfg("boundary-repeat", "0", None, False, compact="1", refresh="1"),
    _cfg("boundary-graph", "1", None, False, compact="1", refresh="1"),
    _cfg("boundary-nocompact", "0", None, False, compact="0", refresh="1"),
    _cfg("boundary-solve-compact", "0", None, False, compact="1", refresh="1", solve_compact="1"),
    _cfg("boundary-ls-compact", "0", None, False, compact="1", refresh="1", solve_compact="1", ls_compact="1"),
    # World-level march compaction (narrow-grid tail body): requires the full
    # compaction chain + boundary cadence; judged bitwise against "boundary"
    # (each other differing flag is itself certified bitwise by its own arm).
    _cfg(
        "march-compact",
        "0",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
    ),
    _cfg(
        "march-compact-graph",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
    ),
    # Full shipping tier: graph + whole-march conditional + march compaction
    # (the capture_if branches record inside the while-node body).
    _cfg(
        "march-compact-conditional",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        conditional="1",
        march="1",
    ),
    # Whole-march conditional tier: varies ONLY the conditional flag relative
    # to boundary-graph (itself already judged bitwise against boundary).
    _cfg("boundary-conditional", "1", None, False, compact="1", refresh="1", conditional="1"),
    # Containment under the full shipping stack (graph + conditional +
    # boundary cadence): the contained boundary kernel records into the
    # substep-body and whole-march graphs.
    _cfg(
        "boundary-containment",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        conditional="1",
        containment="1",
    ),
    # Half-1 shared assembly: the reuse must be bitwise-invisible in BOTH
    # contact cadences (the reused buffers are cadence-agnostic: same anchor
    # state and contact set within an attempt), and under the full shipping
    # tier where the skip changes the captured launch stream.
    _cfg("shared-assembly", "0", None, False, shared="1"),
    _cfg("boundary-shared", "0", None, False, compact="1", refresh="1", shared="1"),
    _cfg(
        "shared-full-stack",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        conditional="1",
        march="1",
        shared="1",
    ),
    # Live-k bounded contact-Hessian GEMM pair: truncation must be
    # bitwise-invisible against the full-padded walk in both cadences and
    # under the full shipping tier (the bounded kernels record a different
    # launch stream, so the graph tiers must key and replay it correctly).
    _cfg("gemm-reshape", "0", None, False, gemm="1"),
    _cfg("boundary-gemm-reshape", "0", None, False, compact="1", refresh="1", gemm="1"),
    _cfg(
        "gemm-full-stack",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        conditional="1",
        march="1",
        shared="1",
        gemm="1",
    ),
    # Per-contact read-once pack + per-solve j_flat hoist: must be
    # bitwise-invisible against the per-row bounded pack in both cadences
    # and under the full shipping tier (the pack kernels and the per-solve
    # j_flat build record a different launch stream, so the graph tiers
    # must key and replay it correctly).
    _cfg("pack-percontact", "0", None, False, gemm="1", pack="1"),
    _cfg("boundary-pack-percontact", "0", None, False, compact="1", refresh="1", gemm="1", pack="1"),
    _cfg(
        "pack-full-stack",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        conditional="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
    ),
    # ---- attempt-consistent-law family (the solver default: flag UNSET) ----
    # Boundary cadence (the shipping cadence) with shared assembly and the
    # bounded GEMM pair on uniformly (both certified bitwise-invisible by
    # their own arms above). Tail/march compaction stays off: under this
    # law every world accepts at the cap, so the partial-active and
    # narrow/wide-branch engagement guards have nothing to engage on this
    # scene. Varied factors within the family: graph capture and the
    # whole-march conditional tier, which must record and replay the
    # attempt-consistent scale kernels' launch stream bitwise.
    _cfg("acr", "0", None, False, refresh="1", shared="1", gemm="1", acr=None),
    _cfg("acr-repeat", "0", None, False, refresh="1", shared="1", gemm="1", acr=None),
    _cfg("acr-graph", "1", None, False, refresh="1", shared="1", gemm="1", acr=None),
    _cfg("acr-conditional", "1", None, False, refresh="1", conditional="1", shared="1", gemm="1", acr=None),
    # ---- fused-LS family (the solver default: flag UNSET) ----
    # The production stack shape (boundary cadence, shared assembly, bounded
    # GEMM, attempt-consistent law -- each certified by its own arms above)
    # with the fused armijo ladder at its unset default. The fused walk is a
    # different fp evaluation order for the ladder's trial costs, so the
    # family carries its own reference + repeat oracle; varied factors
    # within it are graph capture and the whole-march conditional tier,
    # which must record and replay the single-kernel ladder stream bitwise.
    # Engagement is the device ladder-env counter (asserted > 0 here and
    # == 0 in every pinned-off cell).
    _cfg("fusedls", "0", None, False, refresh="1", shared="1", gemm="1", acr=None, fused=None),
    _cfg("fusedls-repeat", "0", None, False, refresh="1", shared="1", gemm="1", acr=None, fused=None),
    _cfg("fusedls-graph", "1", None, False, refresh="1", shared="1", gemm="1", acr=None, fused=None),
    _cfg(
        "fusedls-conditional",
        "1",
        None,
        False,
        refresh="1",
        conditional="1",
        shared="1",
        gemm="1",
        acr=None,
        fused=None,
    ),
    # ---- fused-alphamax family (the solver default: flag UNSET) ----
    # The production stack shape with BOTH ladder flags at their unset
    # defaults: the folded rung 0 evaluates the alpha_max trial
    # cost/derivative along the ray in-kernel, a different fp route from
    # the launch-chain trial, so the family carries its own reference +
    # repeat oracle; varied factors within it are graph capture and the
    # whole-march conditional tier, which must record and replay the
    # folded single-kernel stream bitwise. Engagement is the device
    # alphamax-env counter (asserted > 0 here and == 0 in every
    # pinned-off cell).
    _cfg("fusedam", "0", None, False, refresh="1", shared="1", gemm="1", pack="1", acr=None, fused=None, alphamax=None),
    _cfg(
        "fusedam-repeat",
        "0",
        None,
        False,
        refresh="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
    ),
    _cfg(
        "fusedam-graph",
        "1",
        None,
        False,
        refresh="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
    ),
    _cfg(
        "fusedam-conditional",
        "1",
        None,
        False,
        refresh="1",
        conditional="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
    ),
    # ---- fused-update family (the solver default: flag UNSET) ----
    # The production stack shape with the ladder flags, pack and the fused
    # update evaluation at their unset defaults: the committed-point
    # eval + norm/convergence update runs as one kernel whose per-env
    # reductions are a different fp order from the launch chain's, and the
    # trip-opening hessian projection is deleted (its G comes from the
    # fused eval), so the family carries its own reference + repeat
    # oracle; varied factors within it are graph capture and the
    # whole-march conditional tier, which must record and replay the
    # fused single-kernel stream bitwise. Engagement is the device
    # fused-update-env counter (asserted > 0 here and == 0 in every
    # pinned-off cell).
    _cfg(
        "fusedup",
        "0",
        None,
        False,
        refresh="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
        fusedupdate=None,
    ),
    _cfg(
        "fusedup-repeat",
        "0",
        None,
        False,
        refresh="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
        fusedupdate=None,
    ),
    _cfg(
        "fusedup-graph",
        "1",
        None,
        False,
        refresh="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
        fusedupdate=None,
    ),
    _cfg(
        "fusedup-conditional",
        "1",
        None,
        False,
        refresh="1",
        conditional="1",
        shared="1",
        gemm="1",
        pack="1",
        acr=None,
        fused=None,
        alphamax=None,
        fusedupdate=None,
    ),
    # ---- narrow-v3 family (the solver default: flag UNSET) ----
    # The full march-compaction stack (tail + solve + LS compaction, narrow
    # grid) with the production fused kernels at their unset defaults, under
    # the rejection-generating law (acr="0": under the attempt-consistent
    # law this scene accepts every attempt and the compaction machinery has
    # nothing to engage). The v3 routing is BITWISE class -- list-indexed
    # launches with unchanged per-env arithmetic; only de-listed envs' dead-
    # state rewrites are skipped -- so the ON cells are judged bitwise
    # against the same-stack pinned-off reference. Varied factors within the
    # family: the v3 default itself, graph capture, and the whole-march
    # conditional tier (both graph keys carry the new flag). Engagement is
    # the narrow-site tags (contact_solve: NARROW_V3_SITES; contact_jacobian:
    # the world-gated scatter record), asserted present here and absent in
    # every pinned-off cell.
    _cfg(
        "narrowv3-ref",
        "0",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
        acr="0",
        fused=None,
        alphamax=None,
        fusedupdate=None,
        narrowv3="0",
    ),
    _cfg(
        "narrowv3-ref-repeat",
        "0",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
        acr="0",
        fused=None,
        alphamax=None,
        fusedupdate=None,
        narrowv3="0",
    ),
    _cfg(
        "narrowv3",
        "0",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
        acr="0",
        fused=None,
        alphamax=None,
        fusedupdate=None,
        narrowv3=None,
    ),
    _cfg(
        "narrowv3-graph",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
        acr="0",
        fused=None,
        alphamax=None,
        fusedupdate=None,
        narrowv3=None,
    ),
    _cfg(
        "narrowv3-conditional",
        "1",
        None,
        False,
        compact="1",
        refresh="1",
        solve_compact="1",
        ls_compact="1",
        conditional="1",
        march="1",
        shared="1",
        gemm="1",
        pack="1",
        acr="0",
        fused=None,
        alphamax=None,
        fusedupdate=None,
        narrowv3=None,
    ),
]

# Arms judged bitwise against "boundary" (scheduling-only changes within the
# new cadence); everything else in CONFIGS[2:] is judged against "reference".
BOUNDARY_FAMILY = (
    "boundary-repeat",
    "boundary-graph",
    "boundary-nocompact",
    "boundary-solve-compact",
    "boundary-ls-compact",
    "march-compact",
    "march-compact-graph",
    "march-compact-conditional",
    "boundary-conditional",
    "boundary-shared",
    "shared-full-stack",
    "boundary-gemm-reshape",
    "gemm-full-stack",
    "boundary-pack-percontact",
    "pack-full-stack",
    "boundary-containment",
)

# Arms judged bitwise against "acr" (scheduling-only changes under the
# attempt-consistent trial law).
ACR_FAMILY = (
    "acr-repeat",
    "acr-graph",
    "acr-conditional",
)

# Arms judged bitwise against "fusedls" (scheduling-only changes under the
# fused armijo ladder's trial-cost evaluation order).
FUSEDLS_FAMILY = (
    "fusedls-repeat",
    "fusedls-graph",
    "fusedls-conditional",
)

# Arms judged bitwise against "fusedam" (scheduling-only changes under the
# folded alpha-max rung's trial evaluation route).
FUSEDAM_FAMILY = (
    "fusedam-repeat",
    "fusedam-graph",
    "fusedam-conditional",
)

# Arms judged bitwise against "fusedup" (scheduling-only changes under the
# fused update evaluation's reduction route).
FUSEDUP_FAMILY = (
    "fusedup-repeat",
    "fusedup-graph",
    "fusedup-conditional",
)

# Arms judged bitwise against "narrowv3-ref" (the narrow-v3 routing is
# bitwise class: list-indexed launches, unchanged per-env arithmetic).
NARROWV3_FAMILY = (
    "narrowv3-ref-repeat",
    "narrowv3",
    "narrowv3-graph",
    "narrowv3-conditional",
)


def build_model():
    """One sub-world: a free sphere over the ground plane, replicated into
    N_WORLDS worlds. Contact stiffness is authored on BOTH the sphere and the
    ground: SAP combines pair stiffness in series, so an unauthored ground at
    the default ke would cap the pair far below the authored sphere."""
    t = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def make_initial_conditions():
    """Per-world drop heights and impact speeds, drawn once from a fixed seed.

    Sized so EVERY world's sphere reaches the ground inside boundary 0 (gap /
    |vz| < DT_OUTER): the impact error then forces the certified boundary-0
    rejection, and per-world variation spreads the controller's dt apart."""
    rng = np.random.default_rng(SEED)
    gap = rng.uniform(0.001, 0.003, N_WORLDS)  # bottom-to-plane, worst 3 mm
    vz = rng.uniform(-4.0, -2.0, N_WORLDS)  # worst 3e-3/2 = 1.5 ms < DT_OUTER
    return (R + gap).astype(np.float64), vz.astype(np.float64)


def fresh(z0: np.ndarray, vz: np.ndarray, dt_hist: bool):
    model = build_model()
    coords = model.joint_coord_count // N_WORLDS
    dofs = model.joint_dof_count // N_WORLDS
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
        tol=TOL,
        dt_inner_init=DT_OUTER,
        dt_inner_min=1e-7,
        dt_inner_max=DT_OUTER,
        max_substeps=64,
        dt_histogram=dt_hist,
    )
    return model, state_0, state_1, control, solver


def _apply_env(env: dict[str, str]):
    for flag in ALL_FLAGS + PINNED_OFF:
        os.environ.pop(flag, None)
    os.environ.update(env)


def run_config(name: str, env: dict[str, str], dt_hist: bool, z0, vz):
    _apply_env(env)
    model, s0, s1, control, solver = fresh(z0, vz, dt_hist)

    # Construction tripwires: the switch must actually reach the solver.
    graph_requested = env.get("NEWTON_SAP_ADAPTIVE_GRAPH") == "1"
    is_cuda = bool(wp.get_device(model.device).is_cuda)
    if graph_requested:
        assert solver._graph_enabled, f"[{name}] graph switch did not reach construction"
    else:
        assert not solver._graph_enabled, f"[{name}] reference cell inherited graph capture"
    if dt_hist:
        assert solver._dt_hist is not None, f"[{name}] dt_histogram did not allocate"
    if env.get("NEWTON_ADAPTIVE_MARCH_LOG"):
        assert solver._march_log_path == env["NEWTON_ADAPTIVE_MARCH_LOG"], f"[{name}] march log path did not land"
    else:
        assert solver._march_log_path is None, f"[{name}] cell inherited a march log"
    compact_requested = env.get("NEWTON_ADAPTIVE_TAIL_COMPACT") == "1"
    assert solver._tail_compact == compact_requested, f"[{name}] tail-compact switch did not reach construction"
    per_attempt_requested = env.get("NEWTON_ADAPTIVE_CONTACT_REFRESH") == "attempt"
    assert solver._contact_refresh_per_attempt == per_attempt_requested, (
        f"[{name}] contact-refresh cadence switch did not reach construction"
    )
    solve_compact_requested = env.get("NEWTON_SAP_SOLVE_COMPACT") == "1"
    assert solver._sap.contact_solve._solve_compact == solve_compact_requested, (
        f"[{name}] solve-compact switch did not reach construction"
    )
    ls_compact_requested = env.get("NEWTON_SAP_LS_COMPACT") == "1"
    assert solver._sap.contact_solve._ls_compact == ls_compact_requested, (
        f"[{name}] ls-compact switch did not reach construction"
    )
    conditional_requested = env.get("NEWTON_SAP_ADAPTIVE_CONDITIONAL") == "1"
    if conditional_requested:
        assert solver._conditional_enabled, f"[{name}] conditional-march switch did not reach construction"
    else:
        assert not solver._conditional_enabled, f"[{name}] cell inherited the conditional-march tier"
    containment_requested = env.get("NEWTON_SAP_CONTAINMENT") == "1"
    assert solver.containment == containment_requested, f"[{name}] containment switch did not reach construction"
    shared_requested = env.get("NEWTON_SAP_SHARED_ASSEMBLY") == "1"
    assert solver._shared_assembly == shared_requested, f"[{name}] shared-assembly switch did not reach construction"
    gemm_requested = env.get("NEWTON_SAP_GEMM_RESHAPE") == "1"
    assert solver._sap.contact_solve._gemm_reshape == gemm_requested, (
        f"[{name}] gemm-reshape switch did not reach construction"
    )
    # Attempt-consistent law: absent or "1" resolves ON (the solver's
    # default-ON convention); "0" disables. The constitutive-dt buffer is
    # the mechanism, so its presence/absence is the construction proof --
    # an OFF cell with the buffer set is a leak, an ON cell without it
    # never wired the law.
    acr_expected = env.get("NEWTON_SAP_ATTEMPT_CONSISTENT_R") != "0"
    assert solver._attempt_consistent_r == acr_expected, (
        f"[{name}] attempt-consistent-law switch did not resolve "
        f"(env {env.get('NEWTON_SAP_ATTEMPT_CONSISTENT_R')!r} -> expected {acr_expected})"
    )
    if acr_expected:
        assert solver._sap.contact_solve._constitutive_dt is not None, (
            f"[{name}] attempt-consistent law ON but no constitutive dt reached the contact solve"
        )
    else:
        assert solver._sap.contact_solve._constitutive_dt is None, (
            f"[{name}] constitutive dt reached the contact solve in an ACR-off cell (leak)"
        )
    # Fused armijo ladder: absent or "1" resolves ON (default-ON
    # convention); "0" disables. Construction proof is the resolved switch
    # itself; engagement is the device ladder-env counter, asserted after
    # the march below.
    fused_expected = env.get("NEWTON_SAP_FUSED_LS") != "0"
    assert solver._sap.contact_solve._fused_ls == fused_expected, (
        f"[{name}] fused-LS switch did not resolve "
        f"(env {env.get('NEWTON_SAP_FUSED_LS')!r} -> expected {fused_expected})"
    )
    # Folded alpha-max rung: absent or "1" resolves ON, but only under the
    # fused ladder (the fold is a property of that kernel). Construction
    # proof is the resolved switch; engagement is the device alphamax-env
    # counter, asserted after the march below.
    alphamax_expected = fused_expected and env.get("NEWTON_SAP_FUSED_ALPHAMAX") != "0"
    assert solver._sap.contact_solve._fused_alphamax == alphamax_expected, (
        f"[{name}] fused-alphamax switch did not resolve "
        f"(env {env.get('NEWTON_SAP_FUSED_ALPHAMAX')!r} -> expected {alphamax_expected})"
    )
    # Fused update evaluation: absent or "1" resolves ON (default-ON
    # convention); "0" disables. Construction proof is the resolved switch;
    # engagement is the device fused-update-env counter, asserted after the
    # march below.
    fusedupdate_expected = env.get("NEWTON_SAP_FUSED_UPDATE") != "0"
    assert solver._sap.contact_solve._fused_update == fusedupdate_expected, (
        f"[{name}] fused-update switch did not resolve "
        f"(env {env.get('NEWTON_SAP_FUSED_UPDATE')!r} -> expected {fusedupdate_expected})"
    )
    # Narrow-v3 routing: absent or "1" resolves ON (default-ON convention);
    # "0" disables. Construction proof is the resolved switch on BOTH
    # carriers (contact solve and contact jacobian); engagement is the
    # narrow-site tags, asserted after the march below.
    narrowv3_expected = env.get("NEWTON_SAP_NARROW_V3") != "0"
    assert solver._sap.contact_solve._narrow_v3 == narrowv3_expected, (
        f"[{name}] narrow-v3 switch did not resolve on the contact solve "
        f"(env {env.get('NEWTON_SAP_NARROW_V3')!r} -> expected {narrowv3_expected})"
    )
    assert solver._sap.contact_jacobian._narrow_v3 == narrowv3_expected, (
        f"[{name}] narrow-v3 switch did not resolve on the contact jacobian "
        f"(env {env.get('NEWTON_SAP_NARROW_V3')!r} -> expected {narrowv3_expected})"
    )
    march_requested = env.get("NEWTON_SAP_MARCH_COMPACT") == "1"
    march_expected = (
        march_requested
        and compact_requested
        and solve_compact_requested
        and ls_compact_requested
        and not per_attempt_requested
        and N_WORLDS >= 2
    )
    assert solver.march_compact == march_expected, f"[{name}] march-compact switch did not reach construction"
    if march_expected:
        # The narrow grid must be genuinely narrower than the batch, else the
        # wide branch is unreachable and the arm is vacuous by construction.
        assert solver.march_compact_width < N_WORLDS, (
            f"[{name}] march-compact width {solver.march_compact_width} is not narrower than the world count {N_WORLDS}"
        )

    records = []
    ncon_seen = 0
    partial_active_seen = False
    solve_hetero_seen = False
    for _ in range(K_BOUNDARIES):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, control)
        # Post-march host reads are legal here (never inside the inner loop).
        ncon_seen = max(ncon_seen, int(solver._contacts.rigid_contact_count.numpy()[0]))
        # Heterogeneous Newton convergence in the final attempt's solve:
        # distinct per-env iteration counts with at least one env needing a
        # trip prove that some trip ran with 0 < active envs < N -- the
        # inner-solve compaction's engagement condition (an all-envs-converge-
        # together solve is vacuous for that flag).
        _it = solver._sap.contact_solve.newton_iterations_env.numpy()
        solve_hetero_seen |= len(np.unique(_it)) >= 2 and int(_it.max()) >= 1
        if compact_requested:
            # Post-march, _world_active holds the FINAL iteration's active
            # set: 0 < sum < N proves some worlds landed strictly earlier
            # than the last straggler, i.e. compaction actually restricted
            # work (a scene where all worlds land together is vacuous here).
            wa_sum = int(solver._world_active.numpy().sum())
            partial_active_seen |= 0 < wa_sum < N_WORLDS
        records.append(
            {
                "joint_q": s0.joint_q.numpy().copy(),
                "joint_qd": s0.joint_qd.numpy().copy(),
                "body_q": s0.body_q.numpy().copy(),
                "body_qd": s0.body_qd.numpy().copy(),
                "iterations": np.array([int(solver.iteration_count.numpy()[0])], dtype=np.int64),
                "cumulative_substeps": np.array([solver.cumulative_substeps()], dtype=np.int64),
                "substeps_frame": solver.substeps.numpy().copy(),
                "ideal_dt": solver._ideal_dt.numpy().copy(),
                "dt": solver._dt.numpy().copy(),
                "dt_ceiling": solver._dt_ceiling.numpy().copy(),
                "consec_rej": solver._consec_rej.numpy().copy(),
                "sim_time": solver._sim_time.numpy().copy(),
                "next_time": solver._next_time.numpy().copy(),
                "diverged": solver.diverged.numpy().copy(),
                "guard_hits": solver._guard_hits.numpy().copy(),
            }
        )

    # Engagement tripwires: the switch must have actually run, not fallen back.
    if graph_requested and is_cuda:
        assert solver._graph_enabled, f"[{name}] graph capture fell back to eager mid-run"
        assert len(solver._graph_cache) > 0, f"[{name}] graph cache never populated"
    if conditional_requested and is_cuda:
        assert solver._conditional_enabled, f"[{name}] conditional-march tier downgraded mid-run"
        assert len(solver._conditional_graph_cache) > 0, f"[{name}] conditional graph cache never populated"
        assert solver._conditional_launches > 0, (
            f"[{name}] whole-march conditional graph never replayed -- the tier silently never executed"
        )
    else:
        assert solver._conditional_launches == 0, f"[{name}] conditional-march replay leaked into an OFF cell"
    if containment_requested:
        # Engagement: every boundary must have marched through the contained
        # boundary kernel (the counter increments once per contained boundary).
        assert solver._containment_boundaries == K_BOUNDARIES, (
            f"[{name}] contained boundary path never engaged "
            f"(counter {solver._containment_boundaries} != {K_BOUNDARIES})"
        )
        # On this all-converging scene the failure counters must stay zero:
        # containment may never manufacture a failure where every solve
        # converged (and a nonzero count would make the bitwise arm vacuous).
        assert solver.solve_failure_events == 0, f"[{name}] containment counted a failure on a converging scene"
        assert int(solver.solve_failure_worlds.numpy().sum()) == 0, (
            f"[{name}] per-world failure counter advanced on a converging scene"
        )
    else:
        assert solver._containment_boundaries == 0, f"[{name}] containment engaged in an OFF cell"
    if compact_requested:
        assert partial_active_seen, (
            f"[{name}] tail compaction never engaged a partial active set: every world landed "
            "on the same final iteration in every boundary -- vacuous for this flag; fix the scene."
        )
    if dt_hist:
        assert int(solver._dt_hist.numpy().sum()) > 0, f"[{name}] dt histogram never accumulated"
    if solver._march_log_path is not None:
        if solver._march_log_file is not None:
            solver._march_log_file.close()
        with open(solver._march_log_path) as f:
            n_rows = sum(1 for _ in f)
        assert n_rows >= 1 + K_BOUNDARIES, f"[{name}] march log has {n_rows} lines, expected header + {K_BOUNDARIES}"

    # March-compact engagement: BOTH branch counters must advance in an ON
    # arm -- narrow_execs > 0 proves compacted iterations really executed
    # with fewer-than-full worlds (the cond only routes narrow when the
    # active count fits the sub-batch budget), wide_execs > 0 proves the
    # cond genuinely routed (a scene where every iteration fits the narrow
    # budget never exercises the wide body and is vacuous). The capacity
    # poison must stay clear, and OFF cells must never advance either
    # counter (leak guard). Host reads are post-march only.
    mc_narrow, mc_wide = solver.march_compact_execs()
    narrow_sites = solver._sap.contact_solve.narrow_sites_emitted
    jac_narrow_sites = solver._sap.contact_jacobian.narrow_sites_emitted
    # Narrow-v3 engagement / leak guards. Contact-solve side: the v3 site
    # tags record only under a narrowed grid, so presence is asserted in the
    # march-narrow branch check below; absence everywhere the flag is off is
    # asserted here. Jacobian side: the world-gated scatter records whenever
    # a restricted (non-identity) mask reaches compute(), which tail
    # compaction guarantees on this scene.
    if narrowv3_expected:
        if compact_requested:
            assert "scatter_world_gate" in jac_narrow_sites, (
                f"[{name}] narrow-v3 ON with a live world mask but the world-gated "
                "scatter never recorded -- the gate does not reach the scatter."
            )
    else:
        leaked = narrow_sites & NARROW_V3_SITES
        assert not leaked, (
            f"[{name}] narrow-v3 site(s) {sorted(leaked)} emitted with the switch off -- the v3 routing leaked."
        )
        assert not jac_narrow_sites, (
            f"[{name}] world-gated scatter recorded with the switch off "
            f"({sorted(jac_narrow_sites)}) -- the v3 routing leaked."
        )
    if march_expected:
        assert mc_narrow > 0, (
            f"[{name}] march-compact narrow branch never executed -- no iteration "
            "ran with a sub-budget active set; the scene is vacuous for this flag."
        )
        assert mc_wide > 0, (
            f"[{name}] march-compact wide branch never executed -- the cond never "
            "routed a full-width iteration; the scene is vacuous for this flag."
        )
        assert int(solver._sap.contact_solve._env_grid_poison.numpy()[0]) == 0, (
            f"[{name}] march-compact capacity poison latched -- a device env list outgrew the narrow grid budget."
        )
        # Site-level engagement: the narrow body must actually contain every
        # converted list-indexed launch family this scene drives; a missing
        # tag means a converted site silently never emitted narrowed (dead
        # branch or lost plumbing), not that the feature is off. The chain
        # cells (fused update pinned off) drive the legacy trip sites; the
        # production fused stack replaces them, with the v3 routing adding
        # its own set on top.
        if fusedupdate_expected:
            expected_sites = (
                EXPECTED_NARROW_SITES_FUSED_STACK_V3 if narrowv3_expected else EXPECTED_NARROW_SITES_FUSED_STACK
            )
        else:
            expected_sites = EXPECTED_NARROW_SITES
        missing = expected_sites - narrow_sites
        assert not missing, (
            f"[{name}] march-compact narrow body never emitted converted "
            f"site(s) {sorted(missing)} -- the narrowing does not reach them."
        )
    else:
        assert (mc_narrow, mc_wide) == (0, 0), (
            f"[{name}] march-compact branch counters advanced in an OFF cell -- the narrow-grid path leaked."
        )
        assert not narrow_sites, (
            f"[{name}] list-indexed sites emitted with a narrowed grid in an "
            f"OFF cell ({sorted(narrow_sites)}) -- the narrow-grid path leaked."
        )

    # LS-interior compaction engagement: the device counter accumulates each
    # rebuilt LS-trip list's size, so a nonzero total proves the compacted
    # rebuild+consume path actually executed with live envs (a zero total
    # means the armijo loop never ran, the flag never reached the wiring, or
    # every built list was empty -- vacuous either way). Host read is
    # post-march only.
    ls_trip_env_total = int(solver._sap.contact_solve._ls_compact_trip_env_total.numpy()[0])
    if ls_compact_requested and not alphamax_expected:
        assert ls_trip_env_total > 0, (
            f"[{name}] LS-interior compaction never executed a compacted trip "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        # With the folded alpha-max rung the whole per-trip trial launch
        # chain (LS-list rebuilds included) is deleted, so the compacted
        # LS-trip machinery legitimately never runs: a zero counter is the
        # expected state there, and any nonzero count -- like one in an
        # ls-compact-off cell -- is a leak.
        assert ls_trip_env_total == 0, (
            f"[{name}] LS-compaction counter advanced where the compacted "
            "trip chain must not run -- the compacted path leaked."
        )

    # Shared-assembly engagement: the counter records once per half-1 solve
    # emission inside the captured body, so replays count; a zero total in an
    # ON cell means the reuse path never executed (vacuous), a nonzero total
    # in an OFF cell means it leaked. Host read is post-march only.
    sa_execs = solver.shared_assembly_execs()
    if shared_requested:
        assert sa_execs > 0, (
            f"[{name}] shared-assembly reuse never executed (engagement "
            "counter is zero) -- the flag is untested by this run."
        )
    else:
        assert sa_execs == 0, (
            f"[{name}] shared-assembly counter advanced with the switch off -- the reuse path leaked into an OFF cell."
        )

    # GEMM live-k truncation engagement: the pack increments once per skipped
    # k-tile row-block, so a nonzero total proves the bounded pair actually
    # ran with live truncation (a zero total in an ON cell means every env
    # filled its full padded k range or the bounded path never launched --
    # vacuous either way); any count in an OFF cell means the bounded
    # kernels leaked into the legacy stream. Host read is post-march only.
    gemm_skips = solver._sap.contact_solve.gemm_reshape_skips()
    if gemm_requested:
        assert gemm_skips > 0, (
            f"[{name}] gemm-reshape truncation never skipped a k-tile "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        assert gemm_skips == 0, (
            f"[{name}] gemm-reshape skip counter advanced with the switch off "
            "-- the bounded path leaked into an OFF cell."
        )

    # Fused-LS engagement: the counter accumulates once per env that enters
    # the in-kernel armijo ladder (an env still line-searching after the
    # alpha-max derivative accept), so a nonzero total proves the fused walk
    # actually executed ladder trips (a zero total in an ON cell means the
    # scene never backtracked past alpha_max -- vacuous for this flag); any
    # count in an OFF cell means the fused kernel leaked into the legacy
    # launch-chain stream. Host read is post-march only.
    fused_ladder_envs = solver._sap.contact_solve.fused_ls_ladder_envs()
    if fused_expected:
        assert fused_ladder_envs > 0, (
            f"[{name}] fused armijo ladder never walked an env "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        assert fused_ladder_envs == 0, (
            f"[{name}] fused-ladder counter advanced with the switch off -- the fused path leaked into an OFF cell."
        )

    # Folded alpha-max engagement: the counter accumulates once per env
    # whose alpha_max trial ran through the in-kernel rung 0, so a nonzero
    # total proves the folded evaluation actually replaced the trial launch
    # chain (a zero total in an ON cell means the line search never ran --
    # vacuous); any count in a pinned-off cell means the folded kernel
    # leaked into the certified launch-chain stream. Host read is
    # post-march only.
    # Per-contact pack: absent or "1" resolves ON, but only under the
    # bounded GEMM pair (the pack is that pair's operand builder).
    # Construction proof is the resolved switch; engagement is the device
    # env-launch counter, asserted after the march below.
    pack_expected = gemm_requested and env.get("NEWTON_SAP_PACK_PERCONTACT") != "0"
    assert solver._sap.contact_solve._pack_percontact == pack_expected, (
        f"[{name}] pack-percontact switch did not resolve "
        f"(env {env.get('NEWTON_SAP_PACK_PERCONTACT')!r} -> expected {pack_expected})"
    )

    fused_alphamax_envs = solver._sap.contact_solve.fused_alphamax_envs()
    if alphamax_expected:
        assert fused_alphamax_envs > 0, (
            f"[{name}] folded alpha-max rung never evaluated an env "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        assert fused_alphamax_envs == 0, (
            f"[{name}] alphamax counter advanced with the switch off -- the folded path leaked into an OFF cell."
        )

    # Per-contact pack engagement: the counter accumulates once per env per
    # gj-pack launch, so a nonzero total proves the read-once kernel really
    # built the GEMM operands (a zero total in an ON cell means the bounded
    # hessian assembly never ran -- vacuous); any count in an OFF cell means
    # the per-contact kernel leaked into the certified per-row stream. Host
    # read is post-march only.
    pack_execs = solver._sap.contact_solve.pack_percontact_execs()
    if pack_expected:
        assert pack_execs > 0, (
            f"[{name}] per-contact pack never launched an env "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        assert pack_execs == 0, (
            f"[{name}] pack-percontact counter advanced with the switch off "
            "-- the per-contact path leaked into an OFF cell."
        )

    # Fused-update engagement: the counter accumulates once per live listed
    # env per update evaluation, so a nonzero total proves the fused kernel
    # actually ran the committed-point eval (a zero total in an ON cell
    # means no Newton solve ever ran -- vacuous); any count in a pinned-off
    # cell means the fused kernel leaked into the certified launch-chain
    # stream. Host read is post-march only.
    fused_update_envs = solver._sap.contact_solve.fused_update_envs()
    if fusedupdate_expected:
        assert fused_update_envs > 0, (
            f"[{name}] fused update evaluation never ran an env "
            "(engagement counter is zero) -- the flag is untested by this run."
        )
    else:
        assert fused_update_envs == 0, (
            f"[{name}] fused-update counter advanced with the switch off -- the fused path leaked into an OFF cell."
        )

    # Host-side pipeline-invocation count. Exact for the boundary cadence in
    # every mode (the collide runs outside the captured body); exact for the
    # per-attempt cadence only on eager (graph=0) arms -- graph replays do
    # not re-enter Python, so cadence tripwires must read eager cells.
    return records, {
        "ncon_seen": ncon_seen,
        "collide_calls": int(solver._collide_calls),
        "solve_hetero_seen": bool(solve_hetero_seen),
        "ls_trip_env_total": ls_trip_env_total,
        "conditional_launches": int(solver._conditional_launches),
        "mc_execs": (mc_narrow, mc_wide),
        "narrow_sites": sorted(narrow_sites),
        "jac_narrow_sites": sorted(jac_narrow_sites),
        "sa_execs": sa_execs,
        "fused_ladder_envs": fused_ladder_envs,
        "fused_alphamax_envs": fused_alphamax_envs,
        "pack_execs": pack_execs,
        "fused_update_envs": fused_update_envs,
    }


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Lexicographic-order distance between float32 bit patterns (0 == equal)."""
    ai = a.astype(np.float32).view(np.int32).astype(np.int64)
    bi = b.astype(np.float32).view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**31)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**31)) - bi, bi)
    return np.abs(ai - bi)


def _row_bytes(arr: np.ndarray) -> np.ndarray:
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


def report_divergence(name, ref, cand, where):
    k, field, row = where
    print(f"FAIL[{name}]: first divergence at boundary {k}, field '{field}', row {row}")
    if row >= 0:
        ra = np.ascontiguousarray(ref[k][field])[row].ravel()
        ca = np.ascontiguousarray(cand[k][field])[row].ravel()
        diff = _row_bytes(ra.reshape(ra.size, -1)) != _row_bytes(ca.reshape(ca.size, -1))
        j = int(np.flatnonzero(diff.any(axis=1))[0])
        rv, cv = ra[j], ca[j]
        print(f"  element {j}: ref={rv!r} cand={cv!r}")
        if ra.dtype == np.float32:
            rb = int(np.frombuffer(np.float32(rv).tobytes(), dtype=np.uint32)[0])
            cb = int(np.frombuffer(np.float32(cv).tobytes(), dtype=np.uint32)[0])
            u = ulp_distance(np.array([rv]), np.array([cv]))[0]
            print(f"  bits: ref=0x{rb:08x} cand=0x{cb:08x} ulp={u}")


def run_smoke(name: str, env: dict[str, str], z0, vz) -> dict:
    """SMOKE_STEPS boundaries via step_dt only; finiteness asserted at the end
    and at sparse checkpoints (post-step host reads, never in an inner loop);
    wall time measured after SMOKE_WARMUP boundaries with device sync fences."""
    _apply_env(env)
    model, s0, s1, control, solver = fresh(z0, vz, dt_hist=False)
    # The ON smoke leaves the attempt-consistent law at its unset default
    # (must resolve ON); the OFF smoke pins "0" (must resolve OFF).
    acr_expected = env.get("NEWTON_SAP_ATTEMPT_CONSISTENT_R") != "0"
    assert solver._attempt_consistent_r == acr_expected, f"[{name}] attempt-consistent-law switch did not resolve"
    is_cuda = bool(wp.get_device(model.device).is_cuda)
    for _ in range(SMOKE_WARMUP):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, control)
    if is_cuda:
        wp.synchronize()
    t0 = time.perf_counter()
    timed = SMOKE_STEPS - SMOKE_WARMUP
    for i in range(timed):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, control)
        if (i + 1) % 60 == 0:
            q = s0.joint_q.numpy()
            assert np.all(np.isfinite(q)), f"[{name}] non-finite joint_q at timed step {i + 1}"
    if is_cuda:
        wp.synchronize()
    t1 = time.perf_counter()
    finite = {
        "joint_q": bool(np.all(np.isfinite(s0.joint_q.numpy()))),
        "joint_qd": bool(np.all(np.isfinite(s0.joint_qd.numpy()))),
        "body_q": bool(np.all(np.isfinite(s0.body_q.numpy()))),
        "body_qd": bool(np.all(np.isfinite(s0.body_qd.numpy()))),
    }
    diverged = int(solver.diverged.numpy().sum())
    return {
        "name": name,
        "ms_per_step": (t1 - t0) * 1e3 / timed,
        "steps_timed": timed,
        "finite": finite,
        "diverged_worlds": diverged,
        "cumulative_substeps": solver.cumulative_substeps(),
    }


def main() -> int:
    dev = wp.get_device()
    is_cuda = bool(dev.is_cuda)
    print(f"device: {dev} (cuda={is_cuda})")
    configs = list(CONFIGS)
    if not is_cuda:
        configs = [c for c in configs if c[1].get("NEWTON_SAP_ADAPTIVE_GRAPH") != "1"]
        print("note: CPU device -- graph arms dropped (capture is CUDA-only)")

    z0, vz = make_initial_conditions()

    runs = {}
    extras = {}
    for name, env, dt_hist in configs:
        rec, ext = run_config(name, env, dt_hist, z0, vz)
        runs[name] = rec
        extras[name] = ext
    ref = runs["reference"]

    # --- Tier 1: vacuity guards -------------------------------------------
    iters = np.array([r["iterations"][0] for r in ref])
    dt_spread = any(len(np.unique(r["dt"])) > 1 for r in ref)
    # Distinct per-world accepted-substep counts certify per-world-DISTINCT
    # landing iterations: worlds that need different numbers of accepted steps
    # cannot all land on the same iteration, so the tail-compact arm's active
    # set genuinely shrinks mid-boundary (its own engagement tripwire checks
    # the partial set directly).
    landing_spread = any(len(np.unique(r["substeps_frame"])) > 1 for r in ref)
    guards = {
        "reference run produced pipeline contacts (rigid_contact_count > 0)": extras["reference"]["ncon_seen"] > 0,
        "rejection exercised (boundary 0 needed >= 2 iterations)": bool(iters[0] >= 2),
        "per-world dt diverged across worlds": dt_spread,
        "per-world accepted-substep counts diverged (distinct landing iterations)": landing_spread,
        "no world diverged in reference": all(int(r["diverged"].sum()) == 0 for r in ref),
    }
    # Boundary-cadence family guards: the new default is DIFFERENT PHYSICS, so
    # it needs its own non-vacuity certificate (contacts, rejections, spread)
    # plus proof the cadence actually changed (pipeline-invocation counts:
    # per-attempt collides twice per attempt, boundary once per boundary).
    bref = runs["boundary"]
    b_iters = np.array([r["iterations"][0] for r in bref])
    guards.update(
        {
            "boundary arm produced pipeline contacts": extras["boundary"]["ncon_seen"] > 0,
            "boundary arm exercised a rejection (some boundary >= 2 iterations)": bool((b_iters >= 2).any()),
            "boundary arm per-world dt diverged": any(len(np.unique(r["dt"])) > 1 for r in bref),
            "boundary arm accepted-substep counts diverged": any(len(np.unique(r["substeps_frame"])) > 1 for r in bref),
            "boundary arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in bref),
            "boundary cadence engaged (1 collide per boundary)": extras["boundary"]["collide_calls"] == K_BOUNDARIES,
            "per-attempt cadence counted (2 collides per attempt, eager)": extras["reference"]["collide_calls"]
            == 2 * int(iters.sum()),
            "collision count dropped under boundary cadence": extras["boundary"]["collide_calls"]
            < extras["reference"]["collide_calls"],
        }
    )
    if "boundary-graph" in runs:
        # Exact even under graph replay: the boundary collide runs outside the
        # captured body.
        guards["boundary-graph cadence engaged (1 collide per boundary)"] = (
            extras["boundary-graph"]["collide_calls"] == K_BOUNDARIES
        )
    # Inner-solve compaction non-vacuity: heterogeneous per-env Newton
    # convergence (some trip with a partial active set) must occur in the
    # compaction arm itself.
    guards["inner-solve compaction engaged (heterogeneous Newton convergence)"] = extras["solve-compact"][
        "solve_hetero_seen"
    ]
    # LS-interior compaction non-vacuity: the ON arm's device counter must
    # show compacted LS-trip lists were built nonempty and consumed (the
    # per-arm assert already enforces this; the guard surfaces it in the
    # tier-1 report and fails the probe if the arm is ever skipped).
    guards["LS-interior compaction engaged (nonempty compacted LS-trip lists)"] = (
        extras.get("ls-compact", {}).get("ls_trip_env_total", 0) > 0
    )
    # Whole-march conditional non-vacuity: the ON arm must have replayed the
    # conditional graph on the post-warmup boundaries (K_BOUNDARIES minus the
    # per-iteration warmup tier), else the tier silently never executed.
    if "boundary-conditional" in runs:
        guards["whole-march conditional graph engaged (post-warmup boundaries replayed)"] = (
            extras["boundary-conditional"]["conditional_launches"] > 0
        )
    # March-compact non-vacuity: both branch bodies must have executed in the
    # ON arm (the per-arm asserts already hard-fail; this surfaces the counts
    # in the tier-1 report).
    if "march-compact" in extras:
        _mc = extras["march-compact"]["mc_execs"]
        guards["march compaction engaged (narrow and wide branches both executed)"] = _mc[0] > 0 and _mc[1] > 0
    # Attempt-consistent-law family: its own contact/divergence vacuity
    # certificate, plus law engagement BY DIVERGENCE -- every flag
    # separating the "acr" cell from the "boundary" cell (tail compaction,
    # shared assembly, bounded GEMM) is individually certified
    # bitwise-invisible above, so a bitwise-identical march would mean the
    # law itself never changed a trial and the family is vacuous.
    aref = runs["acr"]
    guards.update(
        {
            "acr arm produced pipeline contacts": extras["acr"]["ncon_seen"] > 0,
            "acr arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in aref),
            "attempt-consistent law engaged (acr family march differs from boundary family)": first_divergence(
                runs["boundary"], aref
            )
            is not None,
        }
    )
    # Fused-LS family: its own contact/divergence vacuity certificate plus
    # ladder engagement via the device counter (the per-arm assert already
    # hard-fails; the guard surfaces it in the tier-1 report). No divergence
    # guard against the acr family: the fused walk changes only the trial
    # costs' fp evaluation order, so a bitwise-identical march is a legal
    # outcome on a scene whose accept decisions never sit within an ulp of a
    # threshold -- the counter, not divergence, is the engagement proof.
    fref = runs["fusedls"]
    guards.update(
        {
            "fusedls arm produced pipeline contacts": extras["fusedls"]["ncon_seen"] > 0,
            "fusedls arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in fref),
            "fused armijo ladder engaged (device ladder-env counter > 0)": extras["fusedls"]["fused_ladder_envs"] > 0,
        }
    )
    # Fused-alphamax family: same shape as the fused-LS family -- its own
    # contact/divergence vacuity certificate plus rung-0 engagement via the
    # device counter (divergence from the fusedls family is NOT required:
    # the fold changes only the alpha_max trial's fp evaluation route, so a
    # bitwise-identical march is a legal outcome; the counter is the
    # engagement proof).
    amref = runs["fusedam"]
    guards.update(
        {
            "fusedam arm produced pipeline contacts": extras["fusedam"]["ncon_seen"] > 0,
            "fusedam arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in amref),
            "folded alpha-max rung engaged (device alphamax-env counter > 0)": extras["fusedam"]["fused_alphamax_envs"]
            > 0,
        }
    )
    # Fused-update family: same shape as the ladder families -- its own
    # contact/divergence vacuity certificate plus engagement via the device
    # counter (divergence from the fusedam family is NOT required: the
    # fusion changes only the committed-point totals' fp reduction route,
    # so a bitwise-identical march is a legal outcome; the counter is the
    # engagement proof).
    upref = runs["fusedup"]
    guards.update(
        {
            "fusedup arm produced pipeline contacts": extras["fusedup"]["ncon_seen"] > 0,
            "fusedup arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in upref),
            "fused update eval engaged (device fused-update-env counter > 0)": extras["fusedup"]["fused_update_envs"]
            > 0,
        }
    )
    # Narrow-v3 family: its own contact/divergence vacuity certificate plus
    # engagement via the emission-time site records (the per-arm asserts
    # already hard-fail on missing sites in the march-narrow branch; the
    # guards surface both carriers in the tier-1 report). Bitwise identity
    # to the pinned-off reference is the family's tier-3 judgment -- the v3
    # routing changes no arithmetic, so divergence is never a legal outcome.
    v3ref = runs["narrowv3"]
    guards.update(
        {
            "narrowv3 arm produced pipeline contacts": extras["narrowv3"]["ncon_seen"] > 0,
            "narrowv3 arm: no world diverged": all(int(r["diverged"].sum()) == 0 for r in v3ref),
            "narrow-v3 solve sites engaged (all v3 tags emitted narrowed)": NARROW_V3_SITES
            <= set(extras["narrowv3"]["narrow_sites"]),
            "narrow-v3 scatter gate engaged (world-gated scatter recorded)": "scatter_world_gate"
            in extras["narrowv3"]["jac_narrow_sites"],
        }
    )

    bad = [g for g, ok in guards.items() if not ok]
    for g, ok in guards.items():
        print(f"guard: {g}: {'ok' if ok else 'NOT MET'}")
    if bad:
        print("VACUOUS: the scene failed to exercise the march; fix the scene, not the flags.")
        return 3

    # --- Tier 2: oracle validity (one oracle per family) ------------------
    for oracle, repeat in (
        ("reference", "reference-repeat"),
        ("boundary", "boundary-repeat"),
        ("acr", "acr-repeat"),
        ("fusedls", "fusedls-repeat"),
        ("fusedam", "fusedam-repeat"),
        ("fusedup", "fusedup-repeat"),
        ("narrowv3-ref", "narrowv3-ref-repeat"),
    ):
        where = first_divergence(runs[oracle], runs[repeat])
        if where is not None:
            report_divergence(repeat, runs[oracle], runs[repeat], where)
            print(f"ORACLE-DEGRADED: {oracle} is not run-to-run deterministic on this device;")
            print("bitwise feature equality cannot be judged here.")
            return 2

    # --- Tier 3: feature equivalence (within each family) -----------------
    failed = False
    for name, _env_cell, _ in configs:
        if name in (
            "reference",
            "reference-repeat",
            "boundary",
            "boundary-repeat",
            "acr",
            "acr-repeat",
            "fusedls",
            "fusedls-repeat",
            "fusedam",
            "fusedam-repeat",
            "fusedup",
            "fusedup-repeat",
            "narrowv3-ref",
            "narrowv3-ref-repeat",
        ):
            continue
        if name in BOUNDARY_FAMILY:
            family = "boundary"
        elif name in ACR_FAMILY:
            family = "acr"
        elif name in FUSEDLS_FAMILY:
            family = "fusedls"
        elif name in FUSEDAM_FAMILY:
            family = "fusedam"
        elif name in FUSEDUP_FAMILY:
            family = "fusedup"
        elif name in NARROWV3_FAMILY:
            family = "narrowv3-ref"
        else:
            family = "reference"
        where = first_divergence(runs[family], runs[name])
        if where is None:
            print(f"PASS[{name}]: bitwise identical to {family} over {K_BOUNDARIES} boundaries")
        else:
            report_divergence(name, runs[family], runs[name], where)
            failed = True
    print(f"equivalence iterations per boundary (reference): {iters.tolist()}")
    print(f"equivalence iterations per boundary (boundary):  {b_iters.tolist()}")
    print(
        f"collide calls over {K_BOUNDARIES} boundaries: reference={extras['reference']['collide_calls']} "
        f"boundary={extras['boundary']['collide_calls']}"
    )

    # --- Tier 4: functional smoke -----------------------------------------
    smoke_on = run_smoke(
        "smoke-ON (graph,compact,boundary)" if is_cuda else "smoke-ON (cpu,compact,boundary)",
        {
            "NEWTON_SAP_ADAPTIVE_GRAPH": "1" if is_cuda else "0",
            "NEWTON_SAP_ADAPTIVE_CONDITIONAL": "1" if is_cuda else "0",
            "NEWTON_ADAPTIVE_TAIL_COMPACT": "1",
            "NEWTON_ADAPTIVE_CONTACT_REFRESH": "1",
            "NEWTON_SAP_SOLVE_COMPACT": "1",
            "NEWTON_SAP_LS_COMPACT": "1",
            "NEWTON_SAP_DETERMINISTIC": "1",
            "NEWTON_SAP_CONTAINMENT": "1",
            "NEWTON_SAP_MARCH_COMPACT": "1",
        },
        z0,
        vz,
    )
    smoke_off = run_smoke(
        "smoke-OFF (eager,nocompact,attempt)",
        {
            "NEWTON_SAP_ADAPTIVE_GRAPH": "0",
            "NEWTON_SAP_ADAPTIVE_CONDITIONAL": "0",
            "NEWTON_ADAPTIVE_TAIL_COMPACT": "0",
            "NEWTON_ADAPTIVE_CONTACT_REFRESH": "attempt",
            "NEWTON_SAP_SOLVE_COMPACT": "0",
            "NEWTON_SAP_LS_COMPACT": "0",
            "NEWTON_SAP_DETERMINISTIC": "1",
            "NEWTON_SAP_CONTAINMENT": "0",
            "NEWTON_SAP_MARCH_COMPACT": "0",
            "NEWTON_SAP_ATTEMPT_CONSISTENT_R": "0",
            "NEWTON_SAP_FUSED_LS": "0",
            "NEWTON_SAP_FUSED_ALPHAMAX": "0",
            "NEWTON_SAP_PACK_PERCONTACT": "0",
        },
        z0,
        vz,
    )
    smoke_fail = False
    print(f"{'config':28s} {'ms/step':>9s} {'steps':>6s} {'cum_substeps':>12s} {'diverged':>8s} finite")
    for s in (smoke_on, smoke_off):
        all_fin = all(s["finite"].values())
        smoke_fail |= not all_fin or s["diverged_worlds"] > 0
        print(
            f"{s['name']:28s} {s['ms_per_step']:9.3f} {s['steps_timed']:6d} "
            f"{s['cumulative_substeps']:12d} {s['diverged_worlds']:8d} "
            f"{'all' if all_fin else str(s['finite'])}"
        )

    if failed:
        return 1
    if smoke_fail:
        print("SMOKE: FAIL (non-finite state or diverged worlds)")
        return 4
    print("SAP-FLAG-EQUIVALENCE + SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
