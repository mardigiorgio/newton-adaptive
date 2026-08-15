# SAP-adaptive wall-clock ledger

Operational state for the continuous wall-time loop. Epistemics: every number
here carries provenance (log path or run dir); entries without provenance are
folklore — re-measure before building on them. The loop updates this file
every pass; Marco redirects the loop by editing it.

## Objective

Make a 4000-iteration training feasible. Primary metric: projected 4k-iter
wall from measured plateau curves at 1024 and 4096 envs. Reference points:
MuJoCo-adaptive plateau ~5.1 s/iter @1024 (4k ≈ 5.7 h). SAP-adaptive
pre-campaign plateau ~78 s/iter @1024 (4k ≈ 87 h). dt healthy band: Marco
accepts any demand profile with dt ≥ 1e-4 across worlds (measured equilibrium
1.2–1.5e-3 — criterion met with margin; demand axis is NOT the fight).

## Measured decomposition (2026-08-15, wf_55df0381-9e4 compare phase)

wall_SAP/wall_MJC = 15.3x = 1.56x (substep demand) x 9.8x (per-slab cost).
SAP slab (3 implicit fp64 solves, 1024-wide) = 15.0 ms; MJC slab = 1.5 ms.
Demand growth is shared physics (MJC substeps also grow x3.5 with training
violence); metric parity bitwise-refuted as a cause; plateau demand = genuine
truncation error at tol=1e-3. ALL wall relief must come from slab price.
Provenance: journal of wf_55df0381-9e4, agents a07793410/a6e00add.

## Landed and certified (all committed or pending commit)

- Converged-env compaction 3.7x; tail compact 2.6-5.8%; cadence hoist 7.9%;
  conditional capture ~2%; march compact v1 10.7-11.7% @1024 / 21% @4096.
- Narrow-grid v2 (2026-08-15): +8.2% over v1 late-window; it7 50.75 s vs
  55.61 v1 vs 61.12 OFF. All bitwise, 7/7 gates. Agent a06d1420.
- Attempt-consistent R (NEWTON_SAP_ATTEMPT_CONSISTENT_R, DEFAULT OFF,
  Marco's call to enable): -9 to -11% substeps in ramp regime, -14.9% late
  wall; penetration gate PASS; advantage ~0 at violent plateau. 9/9 gates.
  Agent aae1bab4.
- fp32 solve opt-in (NEWTON_SAP_SOLVE_PRECISION=fp32, default fp64
  verified): wired + certified (deterministic-in-fp32, containment,
  penetration identical to fp64) but MEASURED 16-29x SLOWER on training:
  per-substep 2-3.2x cheaper (4.6-7.5 vs 15.0 ms) yet substep demand
  explodes 30-50x — the fp32 Richardson-error floor demands us-scale dt in
  violent regimes (batch max inner dt 5.3e-8..7e-6 vs fp64 6.4-8.25e-3;
  healthy-band gate FAILED: 80324/95825 attempted-dt samples < 1e-4,
  dt_run_min 8.5e-9 — disqualified by Marco's dt>=1e-4 criterion alone).
  Gentle regimes indistinguishable from fp64. VERDICT: pure fp32 dead as
  the lever; the mechanism CONFIRMS the mixed-precision split — the
  error/residual path must stay fp64, the factorization/GEMM is where the
  2-3x lives. Agent ab10bc2d, wf_55df0381-9e4 journal; fp32 floor probe
  says K=16*eps32 target floor, cap-independent.
- Determinism default FLIPPED OFF (2026-08-15, Marco's order): all four
  resolution sites read env "0"-default, "1" opts in; probes pin "1"
  explicitly. Trainings pocket ~7.9%. Gate chain: detflip_gates.log.
- Blocked-Cholesky narrowing (2026-08-15, loop pass 1): listed twins of
  the masked factorize/solve pair launch at the env-grid budget; sites
  chol_factorize/chol_solve added to the narrow tripwire. Gates 6/6:
  construct, flag-equiv (new sites asserted), march-equiv, determinism,
  containment, speed A/B. Measured (1024x8, det-unset production
  config): late-window per-substep speedup 1.106 (10.6%), rising with
  tail depth (it7: 1.233); det-ON pair proves exact substep bitwise
  match (series 306..2019 equal). Raw late wall ratio 1.514 is
  trajectory-confounded (det-OFF arms diverge; OFF arm ran ~20% more
  substeps) — cite the per-substep number, not the wall ratio.
  Provenance: scratchpad chol_gates.log, chol_ab_*.{log,telemetry},
  sap_warp commit (see git log).
- Shared full/half1 assembly (2026-08-15, loop pass 3,
  NEWTON_SAP_SHARED_ASSEMBLY default ON, "0" disables): half-1 solves
  reuse the full solve's dt-independent assembly (rigid ID, tau, mass
  matrix + factorization, body/contact Jacobians, Delassus weights);
  only dt fill, v_star assembly and the contact solve re-run. Bitwise
  proof is TOTAL: det=1 ON/OFF 1024x8 training telemetry files are
  byte-identical (substep series 306..14757 equal). Gates 6/6: construct,
  flag-equiv (3 new arms shared-assembly / boundary-shared /
  shared-full-stack, engagement counter >0 ON / ==0 OFF), march-equiv
  fingerprint [6,25,20,24,19], determinism, containment, speed A/B.
  SPEED VERDICT: neutral within noise -- late-window (it5-7) per-substep
  ratio OFF/ON 0.979, whole-run per-substep ~1% apart, raw wall ON -6.0%
  (188.0 vs 200.0 s) but trajectory-confounded (OFF ran 7.6% more
  substeps). The duplicated assembly is a SMALL slab fraction at this
  scene/scale: the slab is contact-solve-dominated (dense packs, GEMM,
  LS trips), assembly kernels are small-grid. Kept default ON as pure
  work-deletion with zero measured cost and larger win potential on
  assembly-heavy scenes. Provenance: scratchpad sharedasm_g12.log,
  sharedasm_g3456.log, sharedasm_ab_{det1,prod}_{on,off}.{log,telemetry};
  sap_warp commit b1e48a3.
- Snapshot commits: newton-adaptive march-counter-log 9c9dc934, sap_warp
  main 79e43bd, IsaacLab develop 82c0679d88.

## Closed: mixed-precision line search (2026-08-15 loop pass 2 — MEASURED
## DEAD END, reverted; full diff preserved at scratchpad
## mixed_ls_attempt.patch, logs mx_*/mxg_*)

Attempted per the refinement plan; three decisive findings:
(1) THE PLAN'S PREMISE WAS STALE: the production preset (approx32) ALREADY
runs the blocked-Cholesky factorize/solve, free-motion, and weights in
fp32 (solver_sap.py preset table; contact_linear_solve_precision=fp32) —
the fp64->fp32 "refinement target" is pre-satisfied WITHOUT refinement and
converges to 1e-8 (production gates). The remaining fp64 cost is the
contact_solve_precision region: projection/gamma evals, gradient/Hessian
assembly, cost reductions, LINE-SEARCH TRIAL EVALS (the 37.6-trips hot
path — what this attempt made f32).
(2) SLOP LAW (kept knowledge): the LS accept slop must scale with the
dtype of the quantities COMPARED, not the mode — an f32-scaled slop on the
f64 alpha-max derivative accept admits ascent steps and cycles Newton
(measured: construct-mixed 1170 substeps + 59 contained failures at the
wrong slop vs 225 + 0, EXACTLY matching fp64, once reverted — dump
mx_fail_dump.json shows alpha pinned 1.25, ls_iterations 0, cost rising,
grad frozen 6.47e-3 at caps 30 AND 60).
(3) THE STRUCTURAL KILL: f32 evaluation of the FULL trial cost is
information-blind in mid-convergence on PD/multi-contact rigs — the body
must decide where |true dcost| < eps32*|cost_total|, and rounding a
dominant total (pd or many-contact) buries the decision; the f64
alpha-max accept covers only the endgame. Splitting pd/limit slots back
to f64 (contact-only f32) still failed the full Trossen rig within 25 s
of marching (contained failures in the determinism worker,
sap_det_probe_0zugulgg/run1.log). No cost-comparison-based LS can run
below the eval dtype's cancellation floor: f32 LS trial evaluation on
this rig class is dead by mechanism, not by tuning. Gradient/Hessian
assembly must stay f64 (certificate); Hessian-only-f32 adds a kernel
without removing one. VERDICT: no viable mixed-precision seam remains in
the LS/eval chain; per-slab fp64 cost is structural under the 1e-8
contract. Wall relief must come from work-DELETION (shared assembly,
overlap, narrowing) not precision.

## Backlog (ranked; teardown of contact_solve internals is AUTHORIZED)

1. Stream overlap of full and half1 solves (data-independent; half2 depends
   on half1). Hides one solve's latency where kernels underfill the GPU.
   Canonical-per-solve reductions keep det mode compatible. NOTE from the
   shared-assembly measurement (landed, neutral): the slab is
   contact-solve-dominated, so the overlap prize is the two Newton solves
   themselves; with half-1 assembly now skipped, its solve starts sooner.
2. LS-interior compaction isolated A/B at 1024 production scale: the ONE
   stack feature never isolated at scale; micro-scene measured it 1.6-1.8%
   SLOWER. If negative at scale too: default it OFF (one env var, free wall).
3. Fused attempt pipeline / dense-path tile reshape (the (envs,32,32) pack,
   GEMM tiles): structural surgery, authorized; measure kernel-level first.
4. Remaining un-narrowed env-axis launches outside contact_solve.py
   (full-width consumers listed in agent a06d1420 notes).
5. Housekeeping: run pre-commit across the accumulated commits and
   re-gate (hooks were deferred to keep certified bytes exact); fix the
   TAIL_COMPACT =="1" exact-match footgun; make the march-compact
   OFF-cell leak guard non-vacuous (review finding).

## Rails (non-negotiable)

optimality_rel_tol 1.0e-8 fp64 (pinned; fp32 path uses its derived
analogue per the fp32 agent's contract); dt_inner_min 1e-12; inner cap 30;
tol 1e-3; no force-accepts/data abandonment; Drake controller constants
untouched; comments carry logic+constraints only; bitwise changes get
default-ON flag + bitwise arm + engagement tripwire; physics-visible
changes need invariant gates + penetration check when contact law/R
touched; twins stay twins (march_equivalence PASS); one GPU process at a
time; no monitors; commits local only (GitHub auth broken — no push
without Marco).

## Escalations to Marco (decisions only he makes)

- Enable NEWTON_SAP_ATTEMPT_CONSISTENT_R for trainings? (measured -11%
  ramp wall, ~0 at plateau, penetration clean)
- fp32 default flip (opt-in exists; awaiting his read of the A/B)
- Phantom body fix: follower_left_ee_gripper_link active=false overlay
  (one line in stationary_ai_task.usda, mirrors right twin) — kernel-width
  savings 7/22 coords per world + closes latent-risk surface
- Any push to GitHub (auth broken; needs gh auth login first)
