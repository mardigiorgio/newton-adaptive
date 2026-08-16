# SAP-adaptive wall-clock ledger

Operational state for the continuous wall-time loop. Epistemics: every number
here carries provenance (log path or run dir); entries without provenance are
folklore — re-measure before building on them. The loop updates this file
every pass; Marco redirects the loop by editing it.

## Objective

Make a 4000-iteration training feasible. Primary metric: projected 4k-iter
wall from measured plateau curves at 1024 and 4096 envs. Reference points
(pass-9 re-measure, 2026-08-15 late): MuJoCo-adaptive plateau ~5.1 s/iter
@1024 (4k ≈ 5.7 h); SAP-adaptive CURRENT plateau 40.8 s/iter @1024
det-unset (4k ≈ 45.3 h); historical pre-campaign plateau was ~78 (dated
2026-08-15 morning, det ON — kept for scale). dt healthy band: Marco
accepts any demand profile with dt ≥ 1e-4 across worlds (measured
equilibrium 1.2–1.5e-3 — criterion met with margin; demand axis is NOT
the fight).

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
- LS-interior compaction isolated A/B (2026-08-15, loop pass 5): at 1024x8
  production scale, LS-compact ALONE (march-compact pinned off both arms)
  is a consistent +0.8% wall win (OFF/ON 1.0077 whole-run, 1.0084 late,
  positive sign at all 8 iterations; run-repeat noise today ~0.1-0.3%) —
  the micro-scene 1.6-1.8% slowdown inverts at scale. Disabling it in the
  production config costs 14.1% whole-run / 19.6% late because
  march-compact hard-requires it (solver_sap_adaptive.py:1464: enable
  condition includes contact_solve._ls_compact) and normalizes itself off.
  DEFAULT STAYS ON; no code changed. Bitwise tripwires: all four arms'
  telemetry byte-identical (det=1), cross-pair trajectory match confirmed.
  Provenance: scratchpad p5_lsc_{on1,off1,iso_on,iso_off}.{log,telemetry}.
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

## Closed: stream overlap of full and half1 solves (2026-08-15 loop pass 4
## — BLOCKED STRUCTURAL, zero edits made; probe overlap_capture_probe.py)

The item's premise ("full and half1 are data-independent") is FALSE in the
implementation, on two independent code-level grounds:
(1) WARM-START CHAIN: half1's guess is (v_t + v_full)/2 — it CONSUMES the
full solve's converged v_flat (solver_sap_adaptive.py substep-evals body:
full solve -> wp.copy(_vfull, contact_solve.v_flat) -> average ->
half1(guess=_vhalf1); half2 then consumes half1's state AND v_full). The
three solves are serial by Drake-CENIC warm-start design; overlapping
full/half1 requires changing half1's guess (e.g. v_t), which changes
Newton paths and accepted states — physics-visible, not bitwise, a
warm-start-semantics design change (Marco-level escalation, likely costs
Newton iterations — the chain exists because it is good).
(2) SHARED MUTABLE WORKSPACE: one SapContactSolve instance
(solver_sap.py:932) serves all three solves — v_flat, converged_env,
newton/LS lists, cost accumulators, dense/Hessian workspace are written
by each solve in sequence. Overlap requires duplicating the solve
workspace (the largest memory consumer) per concurrent solve; the
4096-env production point already sits at 81% of the 32.6 GB card.
CAPTURE MECHANICS (settled POSITIVE, durable knowledge): Warp 1.16 /
CUDA 12.9 / sm_120 DOES support two-stream event-ordered fork/join under
graph capture AND inside a wp.capture_while body, replaying exactly
(overlap_capture_probe.py, exit 0, modes A and B both PASS with exact
integer results). Future overlap ideas with genuinely independent work
(cross-attempt, cross-boundary, a two-instance design if memory allows)
are mechanically feasible.
Tree state: zero repo edits made this pass (git status clean both repos;
construct probe re-run PASS as restoration proof).

## Kernel-level slab profile (2026-08-15, loop pass 6 — nsys, MEASURED)

Method: nsys 2024.6.2 cuda-only traces of a scripted engaged-contact rig
(scratchpad pass6_profile_rig.py; Trossen task, 512 envs, production
defaults, det unset). Three traces: press+swing graph-mode (497 slabs,
1.04/boundary — moderate regime), press+flail graph-mode (3112 slabs,
6.5/boundary — money regime), press+flail EAGER (NEWTON_SAP_ADAPTIVE_GRAPH=0,
1553 slabs) for per-kernel attribution since this nsys traces captured
graphs at graph granularity (interior kernels invisible; the
--cuda-graph-trace=node attempt broke capture detection under CUPTI and
crashed with CUDA 700 — profiler artifact, rig runs clean without it;
manager also warned it fell back to eager for ITS capture in that config).
Shares are GPU-kernel-duration shares from the eager flail trace (launch
gaps excluded; relative shares defensible, absolutes are per-kernel GPU ms).

| group | ms/slab | % of GPU kernel time | notes |
|---|---|---|---|
| contact-Hessian GEMM tile | 0.575 | 51.3 | ONE kernel; 0.547 ms/launch avg |
| Hessian GEMM input pack | 0.185 | 16.5 | pair with above = 68% |
| collision pipeline (per-boundary refresh) | 0.132 | 11.8 | ~1.05 ms/boundary flat; mesh_triangle_contacts_to_reducer 94.4 ms + BVH (cuBQL) 64 ms |
| solve-misc/masks/lists | 0.060 | 5.3 | 16726 tiny launches |
| free-motion/assembly | 0.032 | 2.9 | post shared-assembly |
| projection/gamma evals | 0.030 | 2.6 | (envs,128) grids |
| LS trial chain | 0.009 | 0.8 | already compacted |
| blocked Cholesky | 0.007 | 0.6 | already fp32+narrowed |
| gradient/derivative | 0.006 | 0.6 | |
| error metric + controller | 0.001 | 0.1 | |

VERDICT — reshape-worthy, precisely targeted: the contact-Hessian GEMM
tile kernel + its input pack are 68% of engaged-regime GPU time; every
previously optimized group is now noise. Rough fp64 roofline for the
per-env tile GEMM suggests the kernel sits far from peak — reshape /
batching / tensor-core candidates all live here. NOTE the precision angle:
this GEMM is in the fp64 region, but the Hessian only shapes the Newton
DIRECTION (inexact-Newton tolerates approximate H; the 1e-8 certificate is
measured on the fp64 gradient, and the mixed-precision dead-end was about
LS COST COMPARISONS, not H) — an fp32/tf32 Hessian GEMM is therefore a
live, UNTESTED hypothesis, physics-visible (direction changes ->
trajectories), so opt-in flag + invariant gates + Newton-iteration-count
watch + Marco escalation. Surprises: (1) collision refresh is a constant
~1.05 ms/boundary tax that DOMINATES gentle regimes (44% of visible GPU
time in the press+swing trace) — at 4096 envs it scales and is pure
per-boundary overhead; (2) at 512 envs the eager GPU-busy content of a
slab is only ~1.1 ms vs ~15 ms/slab training telemetry at 1024 — the
wall-vs-GPU-kernel reconciliation at training scale is an OPEN question
(bigger slabs at 1024 + host/graph overhead split unmeasured).
Provenance: scratchpad pass6_prof{,_flail,_eager}.nsys-rep,
pass6_{eager,flail,stats}_cuda_gpu_kern_sum.csv, pass6_run*.log,
pass6_profile_rig.py.

WALL-VS-GPU SPLIT AT TRAINING SCALE (2026-08-15, loop pass 7 — closes the
pass-6 open question; measurement only, repos untouched): on the real rig
(1024 envs, 8 iters, seed 42, production flags, det unset) the host-shim +
nsys split shows the wall is GPU WORK, full stop. (a) Solver region
(SolverSAPAdaptive.step) = 98.4% of collection wall whole-run, 99.1% in
the late window — manager/task/rsl_rl python is ~1-2%. (b) Inside the
solver region at iters 5-7: union GPU-busy = 99.9% of solver wall
(94.6 of 94.8 s); ~98% of GPU time is the whole-march conditional graph
replays (one cudaGraphLaunch per boundary, 769 total, 0.10 s API cost);
eager kernels (collision refresh + manager + torch) ~0.5 s/iter; memops
negligible. Host-thread time sits inside cudaMemcpyAsync readbacks
(~48 small D2H per boundary) fully OVERLAPPED with GPU execution — a
waiting mechanism, not addable cost. ms-wall/slab 12.6-15.5 (late ~13.9)
reconciles the pass-6 cross-rig gap: training slabs are genuinely
~14 ms of GPU work at 1024 engaged (2x envs + deeper tails than the
512-env probe rig). VERDICT: GPU-dominated — GEMM attack confirmed #1;
no gap-closure/launch-overhead lever exists (0.1 s in 95 s). Caveat:
graph-granularity "busy" can hide intra-graph dependency bubbles; the
pass-6 roofline note (tile GEMM far from peak) still stands — the work
itself is the target, not the scheduling. Perturbation honest: nsys run
walls within ~3% of the clean shim run. Provenance: scratchpad
pass7_walls4.{log,telemetry} + pass7_walls4_split.json.1860438 (clean
walls), pass7_nsys.log + pass7_nsys_split.json.1861102 +
pass7_train.{nsys-rep,sqlite} (split), pass7_shim/sitecustomize.py
(method; per-PID output — a late-exiting helper process otherwise
clobbers the sim process's JSON, root-caused this pass).

GEMM LIVE-K TRUNCATION LANDED (2026-08-15, loop pass 8 — the campaign's
largest single win): the contact-Hessian pack wrote full padded
(384x24)x2 fp64 operands and the GEMM walked all 12 k-tiles regardless
of live contacts (mean live tiles 4.79 of 12 on the engaged
distribution). NEWTON_SAP_GEMM_RESHAPE (default ON, "0" = legacy
kernels byte-untouched): bounded pack skips k-tiles past the env's live
rows, bounded GEMM stops its ascending k-walk there — truncated tiles
hold only pack-written zeros, surviving tiles accumulate in the same
order, so ON/OFF are identical bytes (microbench bitwise-equal AND the
det=1 1024x8 training pair's telemetry files byte-identical). Gates
6/6: construct (225 fingerprint), flag-equivalence all arms + 3 new
(gemm-reshape / boundary-gemm-reshape / gemm-full-stack, engagement
skip-counter + OFF-leak asserts), march-equivalence ([6,25,20,24,19]
exact), determinism, containment, speed. MEASURED: microbench pair
x1.96 (16x16 tile variant x1.84 — rejected, inferior + reallocates);
det=1 A/B whole-run x1.611, late-window x1.547; production (det unset)
vs pass-7 baseline same seed/flags: it0 7.21 vs 14.09 (-48.8%), it3
10.22 vs 18.37 (-44.4%), it5 14.18 vs 25.46 (-44.3%), it7 22.54 vs
37.53 (-39.9%). Provenance: pass8_gemm_microbench.py,
p8_g2_flag_equiv.log, p8_g345.log, p8_{det1_on,det1_off,prod_on}.{log,
telemetry}, p8_speed_chain.log.

## Plateau re-measure + feasibility (2026-08-15 late, loop pass 9 —
## MEASURED; the table Marco decides from)

Runs (production defaults = det unset unless stated; seed 42; provenance
scratchpad p9_*.{log,telemetry,gpumem,stamps}):
(1) 1024x25 det-unset: walls 7.21 -> 41.87; PLATEAU (iters 19-24) mean
40.78 s/iter forming ~iter 19 (residual slope ~+0.3/iter at window end);
substeps plateau ~4972/iter; ms/substep FLAT 8.21 (was 15.6 pre-GEMM era,
47.7 at 4096 pre-GEMM). Sanity: physics_diverged fired 0, containment
warnings 0, capacity warnings 0. GPU peak 20,398 MiB (see OOM finding).
(2) 4096x10 det-unset: FAILED — CUDA OOM at construction (single
2,304,000,000-byte alloc). ROOT-CAUSED BY FLAG BISECT (1-iter arms, all
20.4 GB: ctrl/gemm0/shas0/march0/cond0 identical; det1 arm = 9,390 MiB
bit-exact baseline): 2.304e9 = 192e6 authored max_triangle_pairs x 12 B.
Deterministic mode's CONTACT_ID_BITS=25 clamp silently capped the pool at
33.5M in EVERY historical run; the det-OFF default flip removed the clamp
and the full authored 192M pool family now allocates (+11 GB @1024, OOM
@4096). The task's 192M was authored blind under the always-det era; live
demand measured 8.2% of 33.5M ~ 2.8M pairs. Fix is a task-config
right-size (Marco escalation) — NOT a solver defect.
(3) 4096x10 det=1 (the honest available 4096 point): walls 28.29, 39.02,
39.96, 43.30, 46.04, 57.22, 71.77, 83.57, 93.10, 107.89 — still rising at
window end; substeps BIT-IDENTICAL to the pre-GEMM det-ON baseline series
(1282..5596), so the comparison is pure speedup: vs 55.83..210.84 =
x1.95 at it9; ms/substep 47.7 -> 19.3-20.6 late. Peak 26,570 MiB.
(4) 1024x10 det=1: substeps BIT-IDENTICAL to the pre-campaign baseline
(1045..3782) — CAMPAIGN TOTAL AT MATCHED WORK: x1.67-1.89 per iteration
(x1.74 late); walls 8.51..34.01 vs 16.12..59.01. Fresh det tax on the new
stack: ms/substep 8.9-9.0 (det=1) vs 8.21 (det-unset) ~ +8.5%.

FEASIBILITY TABLE (4000 iterations; plateau x 4000; assumptions stated):
- MuJoCo-adaptive @1024: 5.1 s/iter -> 5.7 h (ledger provenance).
- SAP pre-campaign @1024 det-ON: 78 -> 86.7 h (historical).
- SAP NOW @1024 det-unset: 40.78 measured plateau -> 45.3 h.
- SAP NOW @1024 det=1: ~44.4 est (40.78 x 1.088 tax) -> ~49 h.
- SAP NOW @4096 det=1: plateau PROJECTED ~174 s/iter (substep saturation
  multiplier 1.554 from the fresh 1024 curve applied to it9 subs 5596,
  x ~20 ms/sub) -> ~193 h. Window ends still rising; projection.
- SAP NOW @4096 det-unset: blocked by the OOM until the tri-pair cap is
  right-sized; then est ~159 s/iter (174 x 0.915) -> ~177 h.
2000-iter @1024 det-unset ~ 22.6 h. 4096-at-4k remains infeasible on this
card regardless of the OOM fix; 1024 is the trainable scale point.

## Backlog (ranked; teardown of contact_solve internals is AUTHORIZED)

1. Collision-refresh cost (~1.05 ms/boundary constant at 512; share at
   the 1024 engaged plateau is small (~0.25% of a 425 ms boundary) but it
   DOMINATES gentle regimes (44% of GPU in press+swing) and scales with
   envs: profile mesh_triangle_contacts_to_reducer at 1024/4096; check
   whether the BVH rebuild cadence (cuBQL every refresh) can key on
   motion bounds (bitwise-visible only in contact ORDER? verify) —
   measure first. Re-ranked #1 by default of the GEMM landing; its
   engaged-regime ceiling is small — treat as a bounded win.
2. fp32/tf32 Hessian GEMM (physics-visible, Marco-gated escalation):
   inexact-Newton direction; gradient + certificate stay fp64; opt-in
   flag, full invariant gates + iteration-count watch. Re-estimate from
   a fresh engaged profile first — the fp64 pair halved, so the prize
   shrank; the remaining GEMM share at the plateau is the number to get.
3. Remaining un-narrowed env-axis launches outside contact_solve.py
   (full-width consumers listed in agent a06d1420 notes).
4. Housekeeping: run pre-commit across the accumulated commits and
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

- TRIANGLE-PAIR CAP RIGHT-SIZE (new, unblocks 4096 det-unset + frees
  ~11 GB @1024): the task cfg authors max_triangle_pairs=192M, sized
  blind in the always-det era when the CONTACT_ID_BITS clamp silently
  capped it at 33.5M; live demand ~2.8M. Authoring ~8-16M (3-6x margin)
  restores the historical footprint under det-off and un-OOMs 4096.
  One task-config line; task changes are Marco's.
- Enable NEWTON_SAP_ATTEMPT_CONSISTENT_R for trainings? (measured -11%
  ramp wall, ~0 at plateau, penetration clean)
- fp32 default flip (opt-in exists; awaiting his read of the A/B)
- Phantom body fix: follower_left_ee_gripper_link active=false overlay
  (one line in stationary_ai_task.usda, mirrors right twin) — kernel-width
  savings 7/22 coords per world + closes latent-risk surface
- Any push to GitHub (auth broken; needs gh auth login first)
