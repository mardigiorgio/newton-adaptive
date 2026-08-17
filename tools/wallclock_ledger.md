# SAP-adaptive wall-clock ledger

Operational state for the continuous wall-time loop. Epistemics: every number
here carries provenance (log path or run dir); entries without provenance are
folklore — re-measure before building on them. The loop updates this file
every pass; Marco redirects the loop by editing it.

## Marco's grant (2026-08-15 late, renewed 2026-08-16): ALL SOLVER CHANGES
## AUTHORIZED; GOAL IS NOW "AS LOW AS PHYSICALLY POSSIBLE"

"go ahead with all permissions to make any and all changes to the solver.
40s is unacceptable. bellow 40s is not the goal the goal to be as low as
physically possible. 10s would be ideal." Scope: solver + physics-layer
plumbing (sap_warp, newton-adaptive solvers, mjwarp_manager). Still
excluded: task/scene files (tri-pair cap fixed manager-side instead), tol
1e-3 and the step-doubling estimator (comparison semantics), optimality
1e-8, dt floor 1e-12 (rails). Physics-visible solver changes may now land
DEFAULT ON after full invariant gates (OFF escape hatch retained).
10 s/iter @1024 = 1.5x below the last MEASURED plateau 14.93 (pass-18
re-measure on the pass-17 stack; was 1.9x vs the pass-16 18.98, 3.5x
vs the pass-14 35.35, 4.0x vs the pass-9 40.78). Pass-17 LANDED the
alpha-max fold + per-contact pack; pass-18 landed the update-eval
fusion (recommendation C, entry below). The remaining 1.5x is
work-count territory: readback chain / cross-boundary overlap /
prep+free-motion consolidation (pass-18 map), or estimator-structure
(rail).

## Objective

Make a 4000-iteration training feasible. Primary metric: projected 4k-iter
wall from measured plateau curves at 1024 and 4096 envs. Reference points:
MuJoCo-adaptive plateau ~5.1 s/iter @1024 (4k ≈ 5.7 h); SAP-adaptive
last MEASURED plateau 14.93 s/iter @1024 det-unset (pass-18 re-measure
on the pass-17 fold+pack stack, 4k ≈ 16.6 h); prior points kept for
scale: 18.98 (pass 16, fused-LS stack, 2026-08-16), 35.35 (pass 14,
ACR-default pre-fused-LS, 2026-08-16), 40.78 (pass 9,
pre-ACR-default, 2026-08-15 late), ~78 (pre-campaign, det ON,
2026-08-15 morning). dt healthy band: Marco
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

FEASIBILITY TABLE (4000 iterations; plateau x 4000; assumptions stated;
"SAP NOW" refreshed by the pass-18 re-measure on the fold+pack stack):
- MuJoCo-adaptive @1024: 5.1 s/iter -> 5.7 h (ledger provenance).
- SAP pre-campaign @1024 det-ON: 78 -> 86.7 h (historical).
- SAP NOW @1024 det-unset: 14.93 measured plateau (pass 18, pass-17
  fold+pack stack) -> 16.6 h; 2000-iter ~ 8.3 h.
- SAP pass-16 @1024 det-unset (fused-LS stack, kept dated for scale):
  18.98 -> 21.1 h; 2000-iter ~ 10.5 h.
- SAP pass-14 @1024 det-unset (2026-08-16, ACR-default pre-fused-LS,
  kept dated for scale): 35.35 -> 39.3 h; 2000-iter ~ 19.6 h.
- SAP pass-9 @1024 det-unset (2026-08-15 late, pre-ACR-default, kept
  dated for scale): 40.78 -> 45.3 h; 2000-iter ~ 22.6 h.
- SAP NOW @1024 det=1: ~16.2 est (14.93 x 1.088 pass-9 det tax,
  unremeasured since pass 9) -> ~18.0 h.
- SAP @4096 det=1 (projection on projection, LOW confidence: the
  pass-9 ~174 s/iter projected plateau scaled by the measured 1024
  price ratio p18/p9 = 0.354): ~62 s/iter -> ~68 h; det-unset blocked
  by the tri-pair OOM, then est ~57 -> ~63 h. 4096-at-4k remains
  infeasible on this card regardless of the OOM fix; 1024 is the
  trainable scale point.

POST-GEMM KERNEL RE-PROFILE (2026-08-15, loop pass 10 — measurement only,
repos untouched; re-ranks the remaining backlog and closes two items):
(A) Eager flail rig at 512 (pass-6 method rerun on the current stack,
3086 slabs): the solver slab is now FLAT — no group above ~21%. Solver-
only shares: LS trial chain 21.3%, Hessian pack+GEMM 20.0% (was 68%;
pack is now ~65% of the pair), free-motion/assembly 15.4%, projection/
gamma 11.3%, small assembly/misc ~25% spread thin, Cholesky 2.6%,
gradient/cost 2.5%, controller/lists 1.9%, error metric 0.3%. Known
skew: eager-at-512 understated the pair pre-GEMM by ~1.4x vs production,
so its true plateau share is plausibly ~25-30%. Grouping note: kernel-
name keyword 'ls_' substring-matches Warp's '__locals__' mangling —
grouping scripts must use strict patterns (bug caught and fixed this
pass; pass-6 CSVs were grouped by hand and are unaffected).
(B) Production anchor at 1024 (3-iteration real training under nsys,
graph granularity; walls 7.24/7.92/8.67 s match pass-9 within ~2%):
graph (march) executions 20.01 s = ~96% of collection; eager kernels
1.43 s of which collision-group 1.21 s (85%). COLLISION REFRESH AT
PRODUCTION: 4.21 ms/boundary at 1024 (mesh_triangle 2.61 ms + BVH +
narrow-phase reducers), instance-normalized over 288 boundaries ->
0.404 s per 96-boundary iteration = 0.99% of the 40.78 s plateau
iteration (5.5% of an early 7.4 s iteration). VERDICT: collision-refresh
attack CLOSED — full elimination would buy <1% at the plateau where the
money is; the old '44% of gentle regimes' was probe-scene arithmetic
with almost no slabs, not a training cost. ms/slab cross-check: 20.01 s
/ 3063 telemetry slabs = 6.5 ms/slab early-1024, consistent with
pass-9's 8.21 ms/substep at the deeper-tail plateau.
(C) tf32/fp32-Hessian bound (honest ceiling): pair share ~25-30% of
plateau slab x pair-level speedup bound ~2.6x (pack is memory-bound,
~2x from f32 operand traffic; truncated GEMM tile ~35% of pair, up to
~6x compute-side) -> MAX ~15-18% of slab, realistic ~10-14% plateau
wall. Physics-visible (inexact-Newton direction), full invariant-gate
burden, Marco-gated. Escalation updated with this number; implement
only on his word.
CONCLUSION: no dominant kernel target remains — percent-scale bitwise
grinding is near exhaustion; remaining levers are Marco-gated (tf32
~10-14%) or hygiene. Provenance: pass10_prof_eager.nsys-rep +
pass10_eager_cuda_gpu_kern_sum.csv (grouped totals in this entry),
pass10_prof_train.{nsys-rep,sqlite} + pass10_train_cuda_gpu_kern_sum.csv,
pass10_run_{eager,train}.log, pass10_{eager,train}.telemetry.

## Housekeeping pass (2026-08-15/16, loop pass 11 — COMPLETE)

All three items closed, full 7-gate sweep green on the final bytes
(construct; flag-equivalence all arms; march-equivalence; determinism;
containment; err_tol 0/2880 viol, 0 floor visits, dt_run_min 2.87e-3,
0 samples below 1e-4; rest smoke 0 early terms). (1) Deferred
pre-commit applied: newton-adaptive hooks green (ruff/format/typos/
warp-syntax; 24 lint findings fixed minimally — renames, explicit
subprocess check, dict literal, noqa for deliberate path-setup/lazy
imports; unused BLE001 noqa stripped by ruff); IsaacLab hooks green on
campaign code files (formatting only; scene assets deliberately
excluded from hook runs). (2) TAIL_COMPACT footgun fixed in BOTH twins
(!= "0" convention; exact-match silently disabled the feature for
values like "true"). (3) OFF-cell leak guards audited: march-compact,
shared-assembly, GEMM-reshape, and LS-compaction guards ALL already
read unconditionally allocated device counters (the pass-3/4-era
vacuity was fixed during narrow-grid-finish; siblings copied the
correct pattern) — no change needed. Provenance: p11_gates.log +
p11_gates_retry.log + p11_{err_tol,rest}.json (note: scratchpad
probes require CHECK_OUT env — the first G6/G7 attempt failed on the
missing variable, not physics). Commits: newton-adaptive 718ebf7a,
IsaacLab b98f247a13.

## Closed: fp32-Hessian GEMM (2026-08-15/16, loop pass 12 — MEASURED
## SPEED-NEUTRAL, reverted; full diff preserved at scratchpad
## pass12_hessian_fp32_{sapwarp,newton}.patch, logs p12_*)

Implemented under the grant (NEWTON_SAP_HESSIAN_PRECISION, fp32 pack+GEMM
of the contact-Hessian pair, gradient/certificate fp64, cache-keyed in
both graph keys). Gates G2-G8 ALL GREEN on the implementation: construct,
flag-equivalence, march-equivalence, determinism, containment, err_tol
0/2880 viol (dt_run_min 2.9e-3), rest smoke, penetration phi0 IDENTICAL
TO THE LAST DIGIT vs fp64 in every phase. The decisive 1024x16 det-unset
training A/B (agent af90f473, killed by Marco just before writeup; arms
complete in scratchpad): raw walls hfp32 356.9 s vs hfp64 398.0 s (-10.3%)
BUT cumulative substeps 45,027 vs 50,208 (-10.3%) — the wall gap is
trajectory luck. HONEST METRIC: whole-run 7.917 vs 7.920 ms/substep
(0.0%); late-6-iter window 7.875 vs 8.030 (-1.9%, within trajectory-mix
noise; <=  ~-2.5% under generous learner-overhead subtraction). The
pass-10 bound (~10-14% plateau) is REFUTED: post-live-k-truncation the
pack+GEMM pair is latency/memory-latency-bound at 1024-scale tile sizes,
not FLOP- or bandwidth-bound — operand dtype does not move it. VERDICT:
speed-neutral + physics-visible fails the >5% default-ON requirement;
reverted to certified HEAD (verified clean both repos). DURABLE
KNOWLEDGE: (a) the ENTIRE cheaper-math axis on the Hessian pair is now
dead at the trainable scale (fp64->fp32 operand halving bought ~0);
remaining levers on the pair are launch-count/layout/fusion class, not
arithmetic; (b) the kill also invalidates any future tf32/tensor-core
estimate built on the pass-10 FLOP roofline. Side finding: scratchpad
probe_newton_profile.py is STALE vs the current solver (wraps
_substep_body assuming graphs off; its .numpy() syncs now fire inside
capture -> CUDA 906) — do not reuse it without setting
NEWTON_SAP_ADAPTIVE_GRAPH=0 in the run env. Provenance: p12_chain.log
(G2-G7 exit 0), p12_g6b_err_tol.json, p12_g7b_rest.json,
p12_g8_phi0_{hfp32,hfp64}.json, p12_ab16_hfp32.{log,telemetry},
p12_ab16b_hfp64.{log,telemetry}, p12_np_hfp32.log (probe crash).

## Landed: ACR default ON (2026-08-16, loop pass 13 — closes backlog
## item 1; commit 45095218, sap_warp untouched — the flag is
## solver-side only)

NEWTON_SAP_ATTEMPT_CONSISTENT_R is now default ON ("0" disables,
!= "0" convention) and joins BOTH graph cache keys (the constitutive
scale kernels record inside the captured solves, so the flag selects a
different launch stream). Gates 8/8 on the final bytes:
(1) construct PASS, 225 fingerprint intact — the probe pins ACR=0 for
its dt-spread guard march (under the attempt-consistent law the sphere
scene accepts EVERY attempt at the cap: 18 vs 81 substeps over 6
boundaries, no rejections, no dt spread — measured p13_scene_probe.py)
and adds a default-resolution arm (unset env must construct ON, wire
the constitutive dt, march finite).
(2) flag-equivalence PASS all arms: scheduling cells now pin ACR=0
explicitly; new ACR family (variable UNSET) with its own
reference/repeat oracle + graph/conditional arms proves the
attempt-consistent launch stream captures and replays bitwise; law
engagement proven by divergence from the ACR-off boundary family.
(3) march-equivalence [6,25,20,24,19] exact — twins stay twins.
(4) determinism certificate PASS (954 substeps both workers, ACR-ON
stack). (5) containment PASS (35 contained events, healthy worlds
bitwise clean). (6) err_tol: 0/2880 violations, 0 floor visits,
dt_run_min 1.87e-3, 0 samples < 1e-4. (7) rest smoke: 0 early
terminations. (8) penetration phi0 flip-OFF vs flip-ON: deepest
-5.396e-5 vs -5.584e-5 m, median P5 -2.755e-5 vs -2.756e-5, identical
pattern across rest/press/swing — no regression (1.9 um on a ~55 um
equilibrium depth).
DECISIVE A/B (1024x8, seed 42, production det-unset): cumulative
substeps ON/OFF 0.902 whole-run, 0.880 late-3-iter window; ms/substep
(collection wall) 7.002 vs 7.845 whole-run (0.893), 6.990 vs 8.075
late (0.866) — the demand lever also cuts rejected-attempt work per
accepted substep, so the per-substep price IMPROVES; raw wall 90.14 vs
111.84 s (0.806, trajectory-confounded — cite substeps and
per-substep). The prior opt-in A/B measured advantage ~0 at the
violent plateau, so book this as a ramp-regime win until a fresh
25-iter plateau run re-measures the plateau point.
Provenance: scratchpad p13_g1_construct.log, p13_g2_flag_equiv.log,
p13_g3_march_equiv.log, p13_g4_determinism.log,
p13_g5_containment.log, p13_g6_err_tol.{json,log},
p13_g7_rest.{json,log}, p13_g8_phi0_{off,on}.{json,log},
p13_ab_{on,off}.{log,telemetry,stamps,gpumem}, p13_ab_compare.py,
p13_scene_probe.py, p13_gates.sh, p13_ab_run.sh.

PASS-13 DISCOVERY — Newton-iteration budget per solve (measurement
only, no solver edits; backlog 2(a) measured): the contact solve
resets newton_iterations_env / ls_iterations_total at every solve
entry, so a probe-side wrapper around SolverSAPAdaptive.substep
accumulates them per solve class (full/half1/half2 by warm-start
guess identity) with device-side kernels gated by world_active; host
reads post-march only; eager required (GRAPH=0 CONDITIONAL=0
MARCH_COMPACT=0 — a Python wrapper records once under capture and
would misattribute replays). Scripted press+flail rig, 512 envs,
production law (ACR ON), 3064 slabs total.
FLAIL (money regime, 2944 slabs): Newton iters/solve full 2.441,
half1 1.608, half2 1.519 — the warm-start chain saves 33-36% of the
full solve's iteration count in each half, so the Richardson pair
costs 1.31x the full solve, not 2x (3-solve Newton-budget shares:
full 43.3%, half1 29.1%, half2 27.6%). LS trips per Newton iteration
2.29 / 2.67 / 2.97 — the halves' surviving iterations carry MORE
trips each; the trip chain does not shrink with the warm start.
PRESS (gentle, 120 slabs): halves floor-quantized at exactly 1.0
Newton iteration and exactly 2.0 LS trips per iteration (full 1.17)
— in gentle regimes slab price is per-solve FIXED work (assembly,
base cost, projection setup), not iteration count, consistent with
the pass-10 flat kernel profile.
READING FOR THE 10s GOAL: total demand is ~5.24 Newton iterations
per attempt vs 2.27 for the full solve alone (estimator marginal cost
~2.3x), but the halves already sit near the 1-iteration floor — no
large warm-start-waste pocket exists INSIDE the 3-solve scheme.
Factor-scale relief must come from per-solve fixed work (launch/
fusion consolidation, backlog 2(b)) or fewer solves (estimator
semantics — EXCLUDED rail). Caveats: 512-env eager rig, demand
counts not wall; the 1024 plateau regime mix is unmeasured.
Provenance: p13_newton_budget.{json,log}, p13_newton_budget_probe.py.

## Plateau re-measure on the ACR-default stack (2026-08-16, loop pass 14
## — MEASUREMENT ONLY, zero solver edits; supersedes the pass-9 headline)

Rig: exact pass-9 replica (p14_run.sh = p9_run.sh byte-for-byte except
the file prefix): 1024 envs x 25 iters, seed 42, production flags (env
clean of all NEWTON_SAP_* overrides, det unset), W&B/video off, stack =
certified HEADs newton-adaptive 44d6c49f / sap_warp 27dcada / IsaacLab
b98f247a13.
- PLATEAU (iters 19-24) mean 35.35 s/iter vs pass-9 40.78 = -13.3%.
  Walls it0 6.92 -> it24 36.20; window flat (34.12..36.20, no residual
  slope).
- Substeps/iter (window) 4878 vs 4972 = -1.9% — plateau DEMAND is
  unchanged by the ACR default; the prior "advantage ~0 at violent
  plateau" claim was demand-only and stands.
- ms/substep (window) 7.25 vs 8.20 = -11.6% — the win at plateau is
  per-substep PRICE (fewer rejected-attempt evals per accepted substep),
  matching the pass-13 late-window A/B (-13.4%).
- Whole-run: 654.6 s wall, cumulative substeps 91,689 vs p9 83,889
  (+9.3%; det-unset trajectories diverge — cite the window numbers, not
  whole-run walls).
- Sanity: physics_diverged 0, containment/capacity warnings 0, dt
  equilibrium 1.1-1.6e-3 (healthy band), GPU peak 20,430 MiB (p9
  20,398 — footprint unchanged).
Provenance: p14_run.sh, p14_1024x25.{log,telemetry,gpumem,stamps},
p14_plateau_analyze.py (re-derives the p9 numbers 40.78/4972/8.20 from
the p9 raw files with the same arithmetic — no method skew).

## PASS-14 DISCOVERY — fusion-target map of the per-solve kernel chain
## (2026-08-16, measurement only; the structural lever for the 10 s goal)

METHOD FIX FIRST (invalidates part of pass-6/10): both historical eager
nsys traces stopped collecting at ~64.8K kernel records (64,793 and
64,794 — a record cap), i.e. they captured the first ~6-7 s of each run
= startup/warmup, NOT the flail regime; their per-slab denominators and
the pass-10 "flat, no group >21%" ranking are window artifacts. Pass 14
scopes collection with --capture-range=cudaProfilerApi around a
saturated window: scripted rig (p14_profile_rig.py, pass-6 pattern) at
512 envs, GRAPH=0 eager, det unset, warmup+press+40 flail steps
UNPROFILED, then 20 profiled flail steps = 730 slabs (9.1
slabs/boundary), window read at the profiler-range edges.

THE SLAB IS A LAUNCH POPULATION, NOT A KERNEL LIST: 3,777 launches and
13.7 GPU-ms per slab; kernels under 50 us are 99.6% of launches and
84.3% of GPU time (<100 us: 87.2%). Per pass-7 the training wall is
GPU-busy, so these kernel DURATIONS (dominated by per-launch fixed
cost at tiny grids) are the cost structure itself; graphs already
erase the launch gaps.

BATCH-MAX TRIP STRUCTURE (the multiplier pass-13's per-env means hid):
the launch stream pays the batch-MAX iteration counts, not the mean —
21.7 Newton trips/slab (vs ~5.5 per-env mean across the 3 solves) and
194.4 LS trial evals/slab = ~9.0 ladder trials per trip (vs 2.3-3.0
per-env mean); converged envs ride every launch masked.

Ranked groups (ms/slab, % of 13.7): LS-chain bookkeeping 2.75 (20.0),
proj_gamma evals 2.64 (19.2), gemm_pack 1.48 (10.8), contact-impulse
J^T-gamma 1.37 (10.0), pd+limit evals 0.93 (6.8), lists/masks/counters
0.89 (6.5 — 1,061 launches/slab at ~0.8 us each), free-motion 0.64,
base_cost 0.62, grad/impulse-accum 0.61, prep/data-motion 0.40,
gemm_tile 0.38, collision 0.31, Cholesky 0.28, proj_hessian 0.22,
hessian_total+pack_dense+unpack 0.12, ACR scales 0.03.

CHAIN STRUCTURE (in-trace sequence = source map, contact_solve.py): per
Newton trip: proj_hessian -> gemm_pack -> gemm_tile -> hessian_total ->
pack_dense -> chol_factorize -> chol_solve -> unpack ->
search_direction (SERIAL per-env kernel, 27-31 us at grid 2!) ->
base_cost -> init_backtracking -> list rebuild; then PER LS TRIAL
(~14 launches, ~43 GPU-us): scale_alpha -> axpy -> base_cost ->
proj_gamma (12-14 us) -> pd -> limit -> impulse (6-8 us) -> acc_pd ->
acc_limit -> replace_trial_cost (serial, 4.8-5.7 us) -> update/accept
-> 3-launch list rebuild; then accumulate_ls -> commit; then the
update eval: proj_gamma -> impulse -> model_terms_grad -> norm_update
-> rebuild. Adjacent stages communicate ONLY through (envs,...) global
arrays (v_trial, trial gamma/vc/cost, grad/impulse, j_flat/gj_flat,
hess_contact/hessian, chol_a); each LS trial re-reads the live contact
Jacobian twice (~2 x 29.6 KB/env at ~51 live contacts, fp64).

TOTAL LS TRIAL MACHINERY = 7.9-8.7 ms/slab = 58-63% of slab GPU time
(ls_chain + trial shares of proj_gamma/impulse/pd/limit/base_cost/
impulse-accums + LS-trip list rebuilds).

PASS-15 CANDIDATE 1 (primary): FUSED ARMIJO BACKTRACKING LINE SEARCH on
the in-repo monotone_decay pattern (_unit_line_search_base_coeffs +
_unit_line_search_contact_delta_velocity + _unit_line_search_
fused_parallel already run an ENTIRE ladder in 3 launches — existence
proof in contact_solve.py): precompute J*dv once per Newton trip, walk
the alpha ladder in-kernel with the armijo accept rule and Drake-tight
tolerances unchanged. Deletes ~11 launches and the double J re-read per
trial; consolidation ceiling: replacing 7.9-8.7 ms/slab with 21.7 x
(dvc precompute ~14 us + coeffs ~3 us + fused ladder, bounded 50-200
us) = 1.4-4.7 ms/slab -> net 4-7 ms/slab = 30-50% of slab GPU at this
regime/scale (assumptions: 512-env flail window, batch-max trips carry
over, fused ladder cost bounded by the monotone kernel's class). fp
reduction order changes -> physics-visible class: default-ON flag,
full invariant gates, penetration check, fresh A/B; the pass-2 slop
law (accept slop scales with the COMPARED dtype) applies as-is.
PASS-15 CANDIDATE 2 (secondary, ~5%): direction-chain consolidation —
(a) search_direction serial->tiled (0.68 -> ~0.06 ms/slab); (b) GEMM
epilogue absorbs hessian_total + pack_dense (deletes the hess_contact
and hessian round-trips, 4 x 4.6 KB/env-trip; -0.08 ms/slab); (c)
replace_trial_cost folds into candidate 1. (a) reorders a dot product
-> flagged + gated; (b) can preserve accumulation order -> bitwise arm.
NOT candidates now: pack->GEMM full fusion (pack 1.48 ms/slab is real
bandwidth work; re-reading J per output tile at 9 tiles bounds
unfavorably); lists/masks attack alone (~0.45 of its 0.89 ms/slab is
LS-trip rebuilds that candidate 1 deletes as a byproduct).
Caveats named: single regime/scale window (512 eager, saturated
flail); 1024 plateau regime mix unmeasured at kernel granularity;
batch-max trip counts are regime-dependent; no speedup is promised
beyond the stated bounds.
Provenance: p14_prof_flail.{nsys-rep,sqlite},
p14_flail_cuda_gpu_kern_sum.csv, p14_nsys_run.log, p14_stats.log,
p14_profile_rig.py, p14_group_kernels.py (strict grouping — the naive
'ls_' keyword substring-matches Warp's '__locals__' mangling),
p14_tiny_kernels.py.

## Landed: fused armijo line search (2026-08-16, loop pass 15 —
## implements pass-14 candidate 1; sap_warp f49b20b, newton-adaptive
## 9757f69e; the campaign's second-largest single win)

NEWTON_SAP_FUSED_LS (default ON, "0" restores the launch chain
byte-for-byte): after the alpha-max derivative accept — which keeps its
exact launch-chain kernels — the remaining backtracking trips run inside
ONE tiled kernel (128-thread block per env) that walks each env's alpha
ladder in-place and exits at its own accept point. J*dv is computed once
per Newton trip inside the same kernel, per thread over its own strided
contact slots (reader == writer per element, no barrier); trials advance
the contact term as vc0 + alpha*dvc and reduce the regularizer
tile-parallel; the momentum term reuses the chain's stored quadratic
line coefficients verbatim. Accept rule, ladder values, slop source,
trial cap, converged-env masking and output fields are the chain's
exactly; what differs is fp evaluation/reduction order of trial costs —
physics-visible class, flagged, keyed into BOTH graph cache keys.

MID-PASS REGRESSION ROOT-CAUSED (durable design law): the first cut
precomputed J*dv with the monotone path's capacity-wide tiled kernel
(one 128-thread block per (env, contact) SLOT — 1024x128 blocks/trip at
production). Measured: that single launch = 10.81 ms/slab = 59% of the
512-eager flail window's GPU (488 us/launch, 22.15 launches/slab), the
fused ladder itself only 1.08 ms/slab; decisive A/B regressed ON/OFF
per-substep to 1.581 whole-run. LAW: a per-Newton-trip helper may not
launch capacity-wide tile grids; per-contact work at trip cadence must
be live-bounded and barrier-free (fold it into the consumer kernel).
The monotone-variant dvc kernel remains in-repo and is fine at ITS
cadence but is disqualified as an armijo-trip primitive. Provenance:
p15_ab_{on,off}.{log,telemetry}, p15_prof_flail.nsys-rep +
p15_flail_cuda_gpu_kern_sum.csv + p15_nsys_run.log.

POST-FIX PROFILE (p14 rig pattern, 512 envs eager, det unset, scoped
cudaProfilerApi window, 673 slabs): slab GPU 7.81 ms/slab vs 13.7 on
the pass-14 chain baseline (-43%, inside the predicted 30-50% band;
cross-run comparison, same rig/protocol). Ladder kernel 1.12 ms/slab
(51.6 us/launch — folding J*dv in cost +3 us). Trial-machinery groups
collapsed vs pass-14: proj_gamma 2.64 -> 0.56, ls_chain 2.75 -> 1.10
(alpha-max trial machinery remains), pd+limit 0.93 -> 0.11, base_cost
0.62 -> 0.13, lists/masks 0.89 -> 0.27 ms/slab. Provenance:
p15b_prof_flail.nsys-rep + p15b_flail_cuda_gpu_kern_sum.csv +
p15b_nsys_run.log (grouping: p14_group_kernels.py).

GATES 8/8 ON FINAL BYTES (chain p15b_progress.txt, all exit 0):
(1) construct PASS, 225 substeps (p15_g1b_construct.log — post-fix
bytes; first-cut construct also passed as p15_g1_construct.log).
(2) flag-equivalence PASS all 30 arms + new fused-LS family
(fusedls/-repeat/-graph/-conditional, variable UNSET so the default
resolution is under test, own repeat oracle; graph + conditional replay
the single-kernel stream bitwise); every legacy cell pins "0"; device
ladder-env counter asserted > 0 in family cells and == 0 in every
pinned-off cell (p15b_g2_flag_equiv.log; pre-fix run p15_g2_flag_equiv
+ p15_g2b also passed).
(3) march-equivalence [6,25,20,24,19] exact (p15b_g3_march_equiv.log).
(4) determinism certificate PASS, 954 substeps both workers — equal to
the pass-13 ACR-ON stack count (p15b_g4_determinism.log).
(5) containment PASS, 35 contained events (p15b_g5_containment.log).
(6) err_tol 0/2880 violations, 0 floor visits, dt_run_min 1.82e-3, 0
samples < 1e-4 (p15b_g6_err_tol.json).
(7) rest smoke 0 early terminations (p15b_g7_rest.json).
(8) penetration phi0 fused-OFF vs fused-ON IDENTICAL TO THE LAST DIGIT
in every phase (deepest -5.584e-5, median P5 -2.756e-5, rest/press/
swing; p15b_g8_phi0_{off,on}.json).

DECISIVE A/B (1024x8, seed 42, production det-unset, final bytes;
p15_ab_{on2,off2}.{log,telemetry,stamps,gpumem}, p15_ab_compare2.py):
ms/substep (collection wall) ON 4.358 vs OFF 6.866 whole-run (0.635,
-36.5%), late-3-window 4.018 vs 7.020 (0.572, -42.8%) — the >5%
default-ON bar cleared by 7x. Raw walls ON 53.75 vs OFF 106.34 s
whole-run coll (0.505), late 25.93 vs 60.49 (0.429) — trajectory-
confounded, cite per-substep. Cumulative substeps ON 12,334 vs OFF
15,488 (0.796; late window 0.749): demand did NOT inflate — the
det-unset trajectories diverged with the ON arm running fewer substeps
this seed; demand-neutrality at matched trajectories is certified by
the det=1 rig (954 == 954) and by G8's identical penetration, not by
this pair. physics_diverged 0 all iterations, no containment/capacity/
overflow warnings either arm. Projected plateau: pass-14's 35.35
s/iter was 7.25 ms/substep x ~4878 substeps; at the late-window 4.02
ms/substep the same demand prices at ~19.6 s/iter (projection, not a
measurement — a 25-iter plateau re-measure is the pass-16 opener).

Scratchpad-provenance caveat: the p13_* gate artifacts in the
scratchpad carry mtimes ~00:08 2026-08-16 — a prior (killed) agent
re-ran p13_gates.sh after the pass-13 entry was written. Treat the
ledger's pass-13 numbers, not the current p13_* files, as pass-13
evidence.

CLOSED IN THE SAME PASS: candidate 2(a) tiled search-direction
(NEWTON_SAP_TILED_DIRECTION) — MEASURED SPEED-NEUTRAL AT PRODUCTION,
reverted per the pass-12 rule (physics-visible + <5% fails default-ON).
Implemented and fully gated on final bytes (construct 225;
flag-equivalence all arms with the flag joined to the fused-LS family +
its own engagement/OFF-leak counters; march-equivalence
[6,25,20,24,19]; determinism 954/954; containment 35; err_tol 0/2880,
0 floor; rest 0 early terms; phi0 OFF vs ON identical to the last
digit — chain p15c_progress.txt all exit 0). Decisive A/B (1024x8 seed
42 det-unset, fused-LS ON both arms, p15_ab_{ton,toff}.*,
p15_ab_compare3.py): ms/substep ON/OFF 0.990 whole-run / 0.986 late-3
— ~1%, inside trajectory-mix noise; raw wall 0.875 is confounded (OFF
ran 13% more substeps). The pass-14 0.68 -> 0.06 ms/slab projection is
REFUTED as a production lever: the 512-eager flail window overweights
the serial kernel; on the post-fusion ~4 ms/substep production slab the
whole direction chain is ~1%. Candidate 2(b) (GEMM epilogue absorbing
hessian_total + pack_dense) is foreclosed by the same arithmetic:
measured 0.086 ms/slab on the p15b trace = ~1% class. Full diffs
preserved at p15_tiled_direction_{sapwarp,newton}.patch; restoration
proof p15_g1c_restore.log (construct PASS on the reverted trees).

## Plateau re-measure on the fused-LS stack (2026-08-16, loop pass 16
## — MEASUREMENT ONLY, zero solver edits; supersedes the pass-14
## headline)

Rig: exact pass-14 replica (p16_run.sh = p14_run.sh byte-for-byte
except the file prefix; diff-under-rename verified): 1024 envs x 25
iters, seed 42, production flags (env clean of all NEWTON_SAP_*
overrides, det unset), stack = certified HEADs newton-adaptive
0a976d22 / sap_warp f49b20b / IsaacLab b98f247a13.
- PLATEAU (iters 19-24) mean 18.98 s/iter vs pass-14 35.35 = -46.3%
  (vs pass-9 40.78 = -53.5%). Walls it0 5.61 -> it24 19.74; window
  17.98..19.74 with a mild residual slope (~+0.3/iter), same shape as
  the p9/p14 windows.
- Substeps/iter (window) 5050 vs pass-14 4878 = +3.5% — plateau
  DEMAND is unchanged by the fused ladder within trajectory noise;
  the pass-15 demand-neutrality certificates (det=1 954==954, phi0
  identical) stand.
- ms/substep (window) 3.76 vs pass-14 7.25 = -48.1% (ratio 0.519; vs
  pass-9 8.20 = 0.458) — the pass-15 A/B's late-window -42.8%
  CONFIRMED at the 25-iter plateau; the pass-15 projection (~19.6
  s/iter) was conservative by 0.6 s.
- Whole-run: 344.3 s wall, cumulative substeps 89,115 vs p14 91,689
  (-2.8%; det-unset trajectories diverge — cite the window numbers).
- Sanity: physics_diverged 0, containment/capacity/overflow warnings
  0, dt equilibrium 1.0-1.6e-3 (healthy band), GPU peak 20,428 MiB
  (p14 20,430 — footprint unchanged).
- Campaign total at the plateau: 78 (pre-campaign) -> 18.98 = 4.1x;
  price 8.20 (pass 9) -> 3.76 ms/substep in seven passes.
Provenance: p16_run.sh, p16_1024x25.{log,telemetry,gpumem,stamps},
p14_plateau_analyze.py (same arithmetic both runs — no method skew).

## PASS-16 DISCOVERY — post-fusion next-lever map (2026-08-16,
## measurement only; mined from the EXISTING p15b trace, no fresh
## profile — 673 slabs, 512 eager, scoped cudaProfilerApi window)

Method: p16_group_kernels.py = the p14 strict grouping extended with a
fused_ladder group + the post-fusion names the p14 set left ungrouped
(ACR a_inv scales, fp32 articulation pack/unpack, participating-dofs,
march/controller singles); UNGROUPED now 0.1%. Same script re-run on
the p14 CSV (730 slabs) so deltas carry no method skew. Slab GPU 7.81
ms (p14 13.7); trips/slab 21.76 (p14 21.68 — trip structure
untouched, as designed).

Ranked groups (ms/slab, % of 7.81): gemm_pack 1.58 (20.2),
fused_ladder 1.12 (14.4), ls_chain residual 1.10 (14.1),
free_motion_assembly 0.66 (8.4), proj_gamma 0.56 (7.1), solve_prep
0.39 (5.0), gemm_tile 0.37 (4.7), collision 0.33 (4.2, closed),
impulse_assembly 0.31 (4.0), cholesky 0.28 (3.6), lists/masks 0.27
(3.4), proj_hessian 0.23 (2.9), grad_update 0.17 (2.2), base_cost
0.13 (1.6), pd_limit 0.11 (1.4). Deltas vs p14 confirm the fusion
did what it claimed: proj_gamma 2.64->0.56, ls_chain 2.75->1.10,
impulse 1.37->0.31, pd_limit 0.93->0.11, base_cost 0.62->0.13,
lists 0.89->0.27, grad_update 0.61->0.17; gemm_pack 1.48->1.58
(cross-run noise, launches/slab equal).

MECHANISMS (launch-count attribution at 21.76 trips/slab):
(1) ALPHA-MAX TRIAL MACHINERY = 1.10 ms/slab = 14.1% CROSS-GROUP: the
alpha_max derivative-accept trial still runs the full launch chain
once per Newton trip — axpy 0.030 + trial base_cost 0.064 + trial
proj_gamma 0.261 + pd+limit 0.108 + trial impulse 0.146 +
acc_pd/limit 0.055 + replace_trial_cost 0.146 + line_derivative
0.097 + accept 0.038 + init_backtracking 0.039 + LS list rebuilds
~0.113 (contact_solve.py:6494-6604 in _run_sap_backtracking). Its
trial_* array round-trips exist ONLY to feed the chain's derivative
kernel (:2713: dp.(v_trial-v_star) - dv.impulse).
(2) ls_chain residual decomposes as search_direction 0.69 (the serial
kernel — production-REFUTED as a lever, pass-15c measured ~1%; the
512-eager window overweights it) + the (1) members in-group 0.41.
(3) gemm_pack is a per-trip layout transform that REWRITES THE
TRIP-INVARIANT OPERAND EVERY TRIP: j_flat is a pure repack of J
(fixed within a solve; :1633-1689) yet is rebuilt all 21.76
trips/slab, and each jac element is read ~10/3x redundantly across a
contact's three rows (per (c,d): 1 j-read + 9 gj-reads of 3 distinct
values). Pack runs 72.7 us/launch vs the GEMM's 16.8 — the pair's
cost is the pack's scattered fp64 reads, not the multiply.
(4) UPDATE-EVAL CHAIN = 0.85 ms/slab at ~1.14/trip cadence: update
proj_gamma 0.297 + update impulse 0.166 + assemble_model_terms_and_
grad 0.181 + norm_terms_and_update 0.119 + rebuild share ~0.09; PLUS
proj_hessian 0.226 re-projects the SAME committed point at the next
trip's start — a whole projection pass duplicated per trip.
(5) Fixed per-solve work now visible: eval_rigid_id 0.219 at 2/slab
(shared-assembly cadence holds: full+half2 only), solve_prep 0.39,
cholesky 0.28, gemm_tile 0.37 — near-floor, no single target.

MUJOCO GAP DECOMPOSITION REFRESH: production price is now 3.76
ms/substep vs the MJC-adaptive 1.5 ms/slab reference = 2.5x residual
(was 9.8x). Both solvers march the SAME 3-solve step-doubling
estimator, so the 3-solve structure explains NONE of the residual —
it is per-solve implementation price. Pass-13's Newton-budget
measurement (5.24 iters/attempt vs 2.27 full-only = estimator
marginal 2.3x) says a hypothetical single-solve SAP would price at
~1.6 ms/substep ~ the MJC reference — i.e. the estimator rail is
where the LAST 2.3x lives, and everything short of it is percent
levers: the map above totals ~35% of slab in honest ceilings.

PASS-17 RECOMMENDATION (two levers, one physics-visible + one
bitwise):
A. FOLD THE ALPHA-MAX RUNG INTO THE FUSED LADDER (primary, ~11-13%
of flail slab): make the ladder's rung 0 evaluate alpha_max with the
derivative accept rule in-kernel, deleting the (1) launch block
(~11-12 launches/trip + trial_* round-trips). Existence proof: the
ladder kernel ALREADY takes every constitutive input the trial eval
uses (:3132-3186 — phi0/w_eff/mu/k/tau_d/pd/limit arrays) and
evaluates full trial costs per rung; the derivative it needs is
analytic along the ray: (dell_a0 + a*d2ell_a) - sum gamma(vc0 +
a*dvc).dvc - pd/limit terms, from data already in-kernel. Accept
rule to reproduce EXACTLY (:3380): accept iff dl/da < 0 or dl/da <
(rel_slop/10)*max(1, 0.5(|ell|+|ell0|)); the ell_slop write moves
in-kernel (fp64 compared quantities — pass-2 slop law satisfied by
construction). Chain derivative (:2713) vs analytic derivative
differ in trailing fp digits -> physics-visible class, same as
pass-15: default-ON flag ("0" = current alpha-max chain), BOTH graph
keys, full 8-gate chain + phi0 + decisive A/B. Ceiling: delete 1.10,
add ~0.1-0.2 in-ladder -> net ~0.9-1.0 ms/slab.
B. GEMM-PACK SHARED-TILE REWRITE + J-SIDE HOIST (bitwise, ~10-14%):
(i) stage jac tiles once in shared memory so each element is read
once not 10/3x, keeping the per-element gj arithmetic order verbatim
-> identical operand bytes = bitwise arm + engagement counter;
(ii) hoist the j_flat half to per-solve cadence (J fixed within a
solve — assert-probe this invariant FIRST; count_live fixed per
attempt). Ceiling: pack 1.58 -> ~0.6-0.8.
C (measure-first, next in line, ~8%): fuse the update-eval chain into
one tiled kernel that also emits G, absorbing proj_hessian's
duplicate projection ((4) above; ~1.07 ms/slab of chain, in-kernel
replacement ~0.3-0.4).
Combined A+B honest ceiling ~20-25% of slab -> production ~2.9-3.0
ms/substep -> plateau ~14.6-15.2 s/iter; with C ~13.5. The 10 s goal
needs all three PLUS one further ~20% find (candidates: backlog 2c
readback chain, 2d cross-boundary overlap, prep/free-motion
consolidation) — or it is estimator-structure territory (rail).
Caveats: 512-eager saturated-flail window; trip-cadence shares are
regime-dependent (search_direction already proved the window can
overweight a kernel 8x vs production); no speedup promised beyond
the stated bounds; production translation history: pass-15 predicted
30-50%, landed 36.5-42.8%.
Provenance: p16_group_kernels.py, p16_p15b_grouped.txt,
p16_p14_regrouped.txt (same-method p14 re-run), p15b_prof_flail.
{nsys-rep,sqlite} + p15b_flail_cuda_gpu_kern_sum.csv (trace mined,
not re-profiled), source anchors sap_warp sim/contact_solve.py at
f49b20b.

## Landed: alpha-max fold + per-contact pack (2026-08-16, loop pass 17
## — implements pass-16 recommendations A and B; sap_warp a79539a (A)
## + 1ff0ea0 (B), newton-adaptive e5154ee0 (A) + 2574c070 (B))

A. NEWTON_SAP_FUSED_ALPHAMAX (default ON, effective only under the
fused ladder; "0" restores the per-trip alpha-max trial launch chain
byte-for-byte): rung 0 of the fused ladder kernel evaluates the
alpha_max trial cost and its ray derivative in-kernel and applies the
chain's accept rule verbatim — accept iff dl/da < 0 or dl/da <
(rel_slop/10)*max(1, 0.5(|ell|+|ell0|)), ell_slop write moved
in-kernel, every compared quantity in the solve dtype (pass-2 slop law
by construction) — deleting the whole per-Newton-trip trial chain
(LS-list rebuilds, axpy, trial eval, cost replace, serial derivative,
accept: the pass-16 mechanism-(1) block, ~11-12 launches/trip) and its
trial_* array round-trips. The cost advances along the ray as
vc0 + a*dvc; the derivative is the analytic (dell_a0 + a*d2ell_a)
- sum gamma(vc).dvc - sum dv_i*(pd_gamma_i + limit_grad_i), via a new
gamma+cost projection helper sharing the cost variant's arithmetic
expression-for-expression. Trailing-fp-digit differences from the
chain -> physics-visible class: flagged, keyed into BOTH graph cache
keys, own flag-equivalence family (fusedam/-repeat/-graph/
-conditional, variable UNSET, own oracle), device engagement counter
with OFF-leak asserts. The rejected-at-cap-1 corner keeps the env
LS-active, matching the chain's no-ladder-budget state.

B. NEWTON_SAP_PACK_PERCONTACT (default ON, effective only under the
bounded GEMM pair; "0" restores the per-row bounded pack
byte-for-byte): (i) the pack's work item becomes one (contact, dof)
pair that loads the contact's three Jacobian rows and its G block
ONCE for all three gj rows — register staging rather than the
recommendation's shared-memory tiles (same read-once outcome, no
cross-thread sync, no tile-load OOB surface); the per-element gj
expression is kept verbatim -> identical operand bytes for everything
the bounded GEMM reads = bitwise class, judged bitwise in three probe
arms (pack-percontact vs reference, boundary-pack-percontact,
pack-full-stack) with engagement counter + OFF-leak asserts; the
bounded pack's skip-counter accounting is preserved exactly (ON/OFF
totals equal on the dev rig, 7194 == 7194). (ii) J-side hoist: the
trip-invariant j_flat half moves to ONE launch per solve over the
world-active list; every trip's pack writes only the G-dependent gj
half. Precondition MEASURED FIRST per the pass-16 instruction
(p17_j_invariant_probe.py + p17_j_invariant.log): J, j_flat and the
live contact count byte-stable across trips within every solve — 101
trip-pairs over 38 multi-trip solves on a live friction march, zero
mismatches.

GATES 8/8 ON FINAL BYTES (chain p17_progress.txt, all exit 0):
(1) construct PASS (p17_g1a_construct.log after A,
p17_g1b_construct.log on final bytes).
(2) flag-equivalence PASS all 35 arms, 27 tier-1 guards ok: fusedam
family (BOTH ladder flags at unset defaults, production stack shape,
own repeat oracle; graph + conditional replay the folded stream
bitwise) and the three pack arms bitwise against their family
references; every engagement/OFF-leak counter assert armed
(p17_g2_flag_equiv.log).
(3) march-equivalence [6,25,20,24,19] exact (p17_g3_march_equiv.log).
(4) determinism certificate PASS, 954 substeps both workers — equal
to the pass-13/15 stack count (p17_g4_determinism.log).
(5) containment PASS, 35 contained events (p17_g5_containment.log).
(6) err_tol 0/2880 violations, 0 floor visits, dt_run_min 1.59e-3, 0
samples < 1e-4 (p17_g6_err_tol.json).
(7) rest smoke 0 early terminations (p17_g7_rest.json).
(8) penetration phi0 alphamax-OFF vs default-ON IDENTICAL TO THE LAST
DIGIT in every phase (deepest -5.584e-5, median P5 -2.756e-5;
p17_g8_phi0_{off,on}.json).

DECISIVE A/B (1024x8, seed 42, production det-unset, final bytes;
both-ON default vs both pinned OFF; p17_ab_{on,off}.{log,telemetry,
stamps,gpumem}, p17_ab_compare.py): ms/substep (collection wall) ON
3.119 vs OFF 4.229 whole-run (0.737, -26.3%), late-3-window 2.849 vs
3.882 (0.734, -26.6%) — the >5% default-ON bar cleared by 5x, and
beyond the pass-16 A+B ceiling (~20-25% of flail-slab GPU): the fold
also deletes launch/stream overhead and LS-trip list rebuilds the
GPU-time ceiling did not price. Raw walls ON 44.38 vs OFF 59.98 s
coll whole-run (0.740), late 22.44 vs 31.14 (0.721). Cumulative
substeps ON 14,230 vs OFF 14,182 (1.003; late 0.982) — demand flat
this seed; matched-trajectory demand-neutrality is certified by the
det=1 rig (954 == 954) and G8's identical penetration.
physics_diverged 0 all iterations, 0 containment/capacity/overflow
warnings either arm; GPU peak 20,428 MiB both arms (footprint
unchanged). Projected plateau: pass-16's 18.98 s/iter at the late
price ratio 0.734 -> ~13.9 s/iter (projection, not a measurement — a
25-iter plateau re-measure is the pass-18 opener).

Provenance: p17_j_invariant_probe.py, p17_j_invariant.log,
p17_alphamax_smoke.py, p17_pack_smoke.py + p17_pack_{on,off}.npz,
p17_full_chain.sh, p17_ab_run.sh, p17_ab_compare.py, gate artifacts
named above; source anchors sap_warp sim/contact_solve.py +
sim/sap_helpers.py at 1ff0ea0.

## Plateau re-measure on the fold+pack stack (2026-08-16, loop pass 18
## — MEASUREMENT ONLY on the pre-C bytes; supersedes the pass-16
## headline)

Rig: exact pass-16 replica (p18_run.sh = p16_run.sh byte-for-byte
except the file prefix; diff-under-rename verified): 1024 envs x 25
iters, seed 42, production flags (env clean of all NEWTON_SAP_*
overrides, det unset), stack = certified HEADs newton-adaptive
3839a7fd / sap_warp 1ff0ea0 / IsaacLab b98f247a13 (the pass-17
landing, BEFORE this pass's candidate-C commit).
- PLATEAU (iters 19-24) mean 14.93 s/iter vs pass-16 18.98 = -21.3%
  (vs pass-14 35.35 = -57.8%, pass-9 40.78 = -63.4%). Walls it0 4.63
  -> it24 15.02; window 14.26..15.74, flat. The pass-17 A/B's
  late-window ratio 0.734 realized as 0.787 at the 25-iter plateau
  (the 1024x8 late window overweights the win vs the plateau mix).
- Substeps/iter (window) 5146 vs pass-16 5050 = +1.9% — plateau
  DEMAND unchanged within trajectory noise; the matched-trajectory
  demand-neutrality certificates (det=1 954==954, phi0 identical)
  stand.
- ms/substep (window) 2.90 vs pass-16 3.76 = -22.8% (ratio 0.772; vs
  pass-9 8.20 = 0.354).
- Whole-run: 262.9 s wall sum, cumulative substeps 90,480 vs p16
  89,115 (+1.5%; det-unset trajectories diverge — cite the window).
- Sanity: physics_diverged 0, containment/capacity/overflow warnings
  0, late-window inner-dt band 5.4e-4..1.28e-3 (dt >= 1e-4 criterion
  met with 5.4x margin; band sits a notch below p16's 1.0-1.6e-3 —
  same class, healthy), GPU peak 20,428 MiB (unchanged).
- Campaign total at the plateau: 78 (pre-campaign) -> 14.93 = 5.2x;
  price 8.20 (pass 9) -> 2.90 ms/substep in nine passes.
Provenance: p18_run.sh, p18_1024x25.{log,telemetry,gpumem,stamps},
p14_plateau_analyze.py (same arithmetic all runs — no method skew).

## Landed: fused update evaluation (2026-08-16, loop pass 18 —
## implements pass-16 recommendation C; sap_warp 3bff5c1,
## newton-adaptive 00c59d4d)

NEWTON_SAP_FUSED_UPDATE (default ON, "0" restores the launch chain
byte-for-byte): the per-Newton-trip committed-point evaluation —
contact gamma/cost projection, J^T*gamma impulse assembly, model
terms + gradient, norm/convergence update — runs as ONE tiled kernel
per env that ALSO emits the projected G block (with y/rt/rn/mode)
from the same in-kernel vc, so the next trip's hessian projection of
the same committed point is redundant and the trip opener skips it
(pass-16 mechanism (4): chain ~0.85 ms/slab + proj_hessian 0.226
duplicated per trip on the p15b trace). Per-element arithmetic
(projection law, pd/limit terms, gradient formula, norm summands) is
the chain kernels' expression-for-expression; what differs is the
REDUCTION ROUTE (serial ascending impulse/A-row dots per dof,
fixed-schedule tile sums for cost and norms, replacing per-(env,dof)
tile reductions and scattered cost atomics / the canonical serial
sum) — totals and hence convergence decisions can differ in trailing
fp digits: physics-visible class, flagged, keyed into BOTH graph
cache keys, own flag-equivalence family, device engagement counter
with OFF-leak asserts. Launch shape is the norm-update kernel's own
trip-cadence idiom (one tile per env, full env axis — the kernel
writes the convergence mask for every env); per-contact/per-dof work
is live-count-bounded in-kernel (trip-cadence law respected; no
capacity-wide tile grid). Env categories reproduce the chain's
runtime-reachable behavior exactly, including the cost-memset +
frozen-array re-decision for Newton-cap-hit envs. The G consumed by
the per-contact pack keeps its array/layout unchanged (the pack
reads contact_g exactly as before — no pack-path change); the dev
smoke's 6-boundary march is BITWISE identical ON vs OFF on the
sphere rig (every Newton direction consumed identical G bytes).

GATES 8/8 ON FINAL BYTES (chain p18_progress.txt, all exit 0):
(1) construct PASS, 225 substeps (p18_g1a_construct.log).
(2) flag-equivalence PASS all 39 arms + new fused-update family
(fusedup/-repeat/-graph/-conditional, variable UNSET, own oracle;
graph + conditional replay the fused single-kernel stream bitwise);
every legacy cell pins "0"; engagement counter asserted > 0 in
family cells and == 0 in every pinned-off cell
(p18_g2_flag_equiv.log; re-run green on the post-format probe bytes
as p18_g2b_flag_equiv.log).
(3) march-equivalence [6,25,20,24,19] exact (p18_g3_march_equiv.log).
(4) determinism certificate PASS, 954 substeps both workers — equal
to the pass-13/15/17 stack count (p18_g4_determinism.log).
(5) containment PASS; 10 contained events this stack (the event
count is trajectory-dependent under a physics-visible flag; the
engagement and healthy-world-bitwise guards are the certificate;
p18_g5_containment.log).
(6) err_tol 0/2880 violations, 0 floor visits, dt_run_min 1.65e-3,
0 samples < 1e-4 (p18_g6_err_tol.json).
(7) rest smoke 0 early terminations (p18_g7_rest.json).
(8) penetration phi0 fusedup-OFF vs default-ON IDENTICAL TO THE LAST
DIGIT in every phase (deepest -5.584e-5, median P5 -2.756e-5;
p18_g8_phi0_{off,on}.json).

DECISIVE A/B (1024x8, seed 42, production det-unset, final bytes;
p18_ab_{on,off}.{log,telemetry,stamps,gpumem}, p18_ab_compare.py):
ms/substep (collection wall) ON 2.948 vs OFF 3.258 whole-run (0.905,
-9.5%), late-3-window 2.727 vs 3.007 (0.907, -9.3%) — the >5%
default-ON bar cleared; the pass-16 ~8% honest ceiling slightly
beaten (the fusion also deletes launch overhead the GPU-time ceiling
did not price). Raw walls ON 38.95 vs OFF 42.37 s (0.919,
trajectory-confounded — cite per-substep). Cumulative substeps ON
13,093 vs OFF 12,899 (1.015; late 1.036) — demand flat this seed;
matched-trajectory demand-neutrality certified by det=1 (954==954)
and G8. physics_diverged 0, no containment/capacity/overflow either
arm; GPU peak 20,428 MiB both (footprint unchanged). Projected
plateau: 14.93 x 0.907 -> ~13.5 s/iter — exactly the pass-16 "with C
~13.5" projection (25-iter re-measure on the post-C bytes is the
pass-19 opener).

Provenance: p18_fusedup_smoke.py + p18_smoke_{on,off}.npy,
p18_full_chain.sh, p18_ab_run.sh, p18_ab_compare.py, gate artifacts
named above; source anchors sap_warp sim/contact_solve.py at 3bff5c1.

## PASS-18 DISCOVERY — post-C ranked map (2026-08-16; fresh scoped
## profile on the final bytes: p14 rig, 512 eager, det unset,
## cudaProfilerApi window, 766 slabs, p18_group_kernels.py = the p16
## strict grouping + fused_update/percontact-pack patterns)

Slab GPU 5.48 ms/slab (p15b chain baseline 7.81; same rig/protocol,
cross-run). Ranked (ms/slab, % of 5.48): fused_ladder 1.245 (22.7),
ls_chain 0.775 (14.1), fused_update 0.700 (12.8), gemm_pack 0.559
(10.2), free_motion_assembly 0.465 (8.5), solve_prep 0.386 (7.0),
gemm_tile 0.362 (6.6), collision 0.303 (5.5, closed), cholesky 0.275
(5.0), lists/masks 0.139 (2.5), base_cost 0.064, pack_dense 0.045,
hessian_total 0.039, acr_scale 0.033, unpack 0.032. Tiny kernels
(<50 us) still 65.3% of GPU time. Deltas confirm the landings:
gemm_pack 1.58->0.56 (pass-17 B), proj_gamma/impulse/pd_limit/
grad_update/proj_hessian -> 0 (folded into fused_update 0.70, which
replaced ~1.07 of chain), fused_ladder 1.12->1.24 (absorbed the
alpha-max rung), lists 0.27->0.14.

HONEST CEILINGS, top of the map:
- fused_ladder (22.7%): real per-rung cost evaluation; cheaper math
  is dead (pass-2 slop law, pass-12 fp32 kill), fewer rungs = accept
  semantics (rail). Ceiling ~0 within rails.
- ls_chain (14.1%): dominated by the serial search_direction kernel
  the 512-eager window overweights ~8x (pass-15c measured ~1% at
  production, REFUTED as a lever); honest remainder ~2-3%.
- fused_update (12.8%): replaced ~1.07 ms/slab of chain; the dof
  phase idles most of the 128-thread tile on ~20-dof rows — in-kernel
  occupancy shave ~0.2-0.3 ms/slab = few-% production class.
- PREP/FREE-MOTION CONSOLIDATION = the top open target:
  free_motion 0.465 + solve_prep 0.386 + lists 0.139 ~ 0.99 ms/slab
  = 18.1% of window, spread over ~270 launches/slab at ~4-5 us each
  (launch-fixed-cost dominated, the same class C just attacked at
  trip cadence, here at solve/boundary cadence). C-style fusion
  ceiling ~half -> ~9% window, plausibly ~5% production.

PASS-16 "one further ~20% find" CANDIDATES vs the fresh trace:
(2c) per-boundary D2H readback chain: kernel sums cannot see host
waits, and pass-7 already measured the readbacks fully overlapped
(0.10 s API in 95 s) — LOW promise, deprioritize.
(2d) cross-boundary overlap: an occupancy lever, invisible to kernel
sums; needs a MEASUREMENT of plateau-tail active-world occupancy
(march telemetry, no profile) before any design; memory duplication
concern from pass 4 stands. UNPRICED.
(prep) consolidation above: the only percent-priced open lever.

MUJOCO GAP REFRESH: production price 2.90 ms/substep (pass-18
plateau, pre-C) vs the MJC-adaptive 1.5 ms/slab reference = 1.93x
residual (was 2.5x at pass 16); with C's 0.907 -> ~2.63 projected =
1.75x. Pass-13's estimator arithmetic (marginal 2.3x) now prices a
hypothetical single-solve SAP at ~1.15-1.26 ms/substep — BELOW the
MJC reference: the step-doubling estimator rail is now the DOMINANT
residual. Arithmetic to goal: 10 s/iter at plateau demand 5146 needs
1.94 ms/substep; C (0.907) + the priced percent levers (~10-15%)
project ~2.2-2.4 ms/substep ~ 11.5-12.5 s/iter. The last ~15-20% is
overlap (unpriced) or estimator territory (rail — Marco's call).

PASS-19 RECOMMENDATION: (1) 25-iter plateau re-measure on the post-C
bytes (standing opener; projection says ~13.5). (2) Implement the
prep/free-motion per-solve consolidation (fusion class, bitwise
where accumulation order is preserved, flagged where not; ~9% window
ceiling). (3) MEASURE plateau-tail occupancy from march telemetry to
price cross-boundary overlap honestly before writing any code.
(4) Escalate to Marco: the estimator rail is now where the last
factor lives — percent levers alone project ~11.5-12.5 s/iter, not
10. Caveats: 512-eager saturated-flail window shares are
regime-dependent (search_direction precedent: 8x overweight); no
speedup promised beyond stated bounds.
Provenance: p18_nsys_run.sh, p18_prof_flail.{nsys-rep,sqlite},
p18_flail_cuda_gpu_kern_sum.csv, p18_grouped.txt,
p18_group_kernels.py, p18_nsys_run.log (PROFILE_WINDOW wall_s=13.304
slabs=766).

## Plateau re-measure on the post-C stack (2026-08-16, loop pass 19 —
## MEASUREMENT ONLY, zero solver edits; supersedes the pass-18 headline)

Rig: exact pass-18 replica (p19_run.sh = p18_run.sh byte-for-byte except
the file prefix; diff-under-rename verified): 1024 envs x 25 iters, seed
42, production flags (env clean of NEWTON_SAP_* overrides, det unset),
stack = certified HEADs newton-adaptive e57d36f1 / sap_warp 3bff5c1 /
IsaacLab b98f247a13 (the pass-18 candidate-C landing).
- PLATEAU (iters 19-24) mean 14.21 s/iter vs pass-18 pre-C 14.93 = -4.8%
  (vs pass-16 18.98 = -25.1%, pass-9 40.78 = -65.2%). Walls it0 4.25 ->
  it24 15.16; window 12.59..15.24. The pass-18 A/B's late ratio 0.907
  realized as 0.952 at the plateau — the same A/B-overweights-the-win
  pattern as pass-17 (0.734 -> 0.787); the tail-occupancy entry below
  now explains the mechanism: the plateau mix is ~88% low-occupancy
  straggler slabs where the saturated-window fusion wins are smaller.
- Substeps/iter (window) 5170 vs 5146 = +0.5% — demand flat; the
  matched-trajectory certificates (det=1 954==954, phi0 identical)
  stand.
- ms/substep (window) 2.75 vs 2.90 = -5.3% (ratio 0.947; vs pass-9
  8.20 = 0.335).
- Whole-run: 250.2 s wall sum, cumulative substeps 89,694 vs p18
  90,480 (-0.9%; det-unset trajectories diverge — cite the window).
- Sanity: physics_diverged 0, containment/capacity/overflow warnings 0,
  late-window inner-dt max-band 8.95e-4..1.56e-3 (same class as p18's
  5.4e-4..1.28e-3, healthy), GPU peak 20,428 MiB (unchanged).
- Campaign total at the plateau: 78 (pre-campaign) -> 14.21 = 5.5x;
  price 8.20 (pass 9) -> 2.75 ms/substep in ten passes.
- GRANT ARITHMETIC REFRESH: 10 s/iter at plateau demand 5170 needs
  1.93 ms/substep = a further 0.70x. MJC-adaptive reference 1.5 ->
  residual 1.83x. Pass-13 estimator arithmetic (marginal 2.3x) prices a
  single-solve SAP at ~1.20 ms/substep — still below the MJC reference.
  With the prep/lists lever MEASURED DEAD (entry below), the remaining
  priced percent levers (fused_update dof-phase shave, few-% class)
  project ~2.6-2.7 ms/substep ~ 13.4-14.0 s/iter. The last ~1.4x is
  the estimator rail (escalated) or cross-boundary overlap — whose
  measured ceiling bracket now STRADDLES the 10 s requirement (entry
  below).
Provenance: p19_run.sh, p19_1024x25.{log,telemetry,gpumem,stamps},
p14_plateau_analyze.py (same arithmetic all runs — no method skew).

## Closed: prep/free-motion + env-list launch consolidation (2026-08-16,
## loop pass 19 — MEASURED SPEED-NEUTRAL AT PRODUCTION, reverted; full
## diffs preserved at scratchpad p19_fused_prep_lists_{sapwarp,newton}
## .patch; restoration proof p19_g1r_restore.log)

Implemented the pass-18 recommendation (2) as two default-ON BITWISE
flags, fully gated, then reverted on the decisive A/B per the <2% rule
(pass-12/15c precedent):
- NEWTON_SAP_FUSED_PREP: ONE elementwise (env, dof) launch replaced the
  per-solve prepare chain (velocity-input copy, A-diag extraction,
  attempt-consistent a_inv scales, PD/limit builders, participating-dof
  clear + model mark; the contact mark kept its contact-parallel
  launch) — cross-stage values moved through registers with every
  expression's operand bytes preserved (reader == writer per element
  throughout the chain, no reductions): bitwise class, ~7 launches per
  solve deleted.
- NEWTON_SAP_FUSED_LISTS: every env-list rebuild (reset + atomic build
  [+ accumulate] [+ capacity guard]) became ONE tiled
  tile_scan_exclusive kernel emitting the list in canonical ascending
  order (a legal refinement — consumers are order-insensitive by the
  list contract; count/poison/total values exact; dev property test
  p19_tile_rebuild_dev.py green over n_envs 1..4096). ~57 launches/slab
  deleted; ~78 of the map's ~266 tiny launches/slab total.
GATES 8/8 ON THE IMPLEMENTED BYTES (chain p19_progress.txt, all exit
0): construct 225 (p19_g1_construct.log); flag-equivalence 41 arms + 4
new bitwise arms (prep-fused, boundary-prep-fused, lists-fused,
preplists-full-stack) bitwise-identical to their family references, 30
tier-1 guards ok, engagement + OFF-leak counters asserted both ways,
march-compact expected narrow-site set made fused-prep-aware
(p19_g2_flag_equiv.log); march-equivalence [6,25,20,24,19] exact
(p19_g3); determinism 954==954 (p19_g4); containment 35 events
(p19_g5); err_tol 0/2880, 0 floor, dt_run_min 2.13e-3 (p19_g6); rest 0
early terminations (p19_g7); phi0 OFF-vs-ON byte-identical JSON
(deepest -5.584e-5, median P5 -2.756e-5; p19_g8); dev-rig committed
march BITWISE ON == OFF (p19_prep_smoke.py, p19_smoke_{on,off}.npy).
DECISIVE A/B (1024x8 seed 42 det unset; p19_ab_{on,off}.*,
p19_ab_compare.py): ms/substep ON/OFF 1.010 whole-run / 1.004 late-3 —
SPEED-NEUTRAL (per-iteration ratios straddle 1.0; physics_diverged 0,
no warnings, GPU peak 20,428 MiB both arms). REVERTED; both repos
clean at e57d36f1 / 3bff5c1 (restore construct PASS, 225).

DURABLE LAW (the pass's real finding): under the shipping whole-march
conditional graph, deleting LAUNCH COUNT buys nothing — graph replay
amortizes launch fixed cost, so the pass-18 map's "launch-fixed-cost
dominated, ~4-5 us each" pricing of the prep/free-motion/lists groups
is REFUTED at production. A fusion pays only when it deletes GPU WORK:
candidate C removed ~0.37 GPU-ms/slab of chain and priced -9.3% at the
A/B; prep+lists' fusable GPU time is ~0.1-0.15 ms/slab — the sub-1%
the A/B showed. Read the ranked map by deletable GPU-time, never by
launches/slab; the "tiny kernels 65.3% of GPU time" statistic stays
live only as a GPU-time sum. The remaining honest items in these
groups are the two real kernels (eval_rigid_id 0.218,
scatter_sap_contacts 0.159 ms/slab at 2/slab — per-solve real work,
no identified redundancy).

## Tail occupancy + cross-boundary overlap pricing (2026-08-16, loop
## pass 19 — MEASUREMENT ONLY; march-audit import hook, zero repo
## edits; prices backlog 2(d) honestly)

Rig: the p19 rig with the march-audit sitecustomize hook (pure
post-march observer; audited-run walls excluded from plateau claims —
host reads cost ~4-5%/iter). Two full 25-iter runs: p19_occ_1024x25
(histograms) and p19_pers_1024x25 (+ per-boundary top-16 straggler
identities). Late window = boundaries 1824-2400 (iters 19-24), ~9,915
march iterations, rejects 0 (accepts-based active counts exact).
- ACTIVE-WORLD DISTRIBUTION per march iteration: mean 12.8% of 1024,
  p10 0.0%, p50 0.6% (~6 worlds), p90 98.8% — bimodal: ~2 near-full
  iterations per boundary (the 2-accept bulk), then a straggler tail.
  88.4% of iterations run below 25% active (all of those below 6.25%
  = the march-compact narrow predicate; narrow share 82.5%).
  REPLICATED across both runs (mean 0.1282 / 0.1268, identical p90 and
  <25% share).
- GPU-TIME AT LOW OCCUPANCY: the <25%-active iterations carry between
  33% (cost proportional to active fraction, floored at the narrow
  budget) and 88% (uniform cost) of late-window GPU time — the models
  bracket the truth (list-indexed kernels are live-bounded;
  full-width-by-design kernels are not). Work-weighted mean active
  fraction 0.90: the WORK is high-occupancy, the SLABS are not. NOTE:
  every profile map so far (p14/p15b/p18) scoped the SATURATED
  512-eager flail window — the regime that is 88% of plateau slabs has
  never been profiled.
- OVERLAP CEILING (backlog 2d): one policy action spans decimation=4
  boundaries (120 Hz outer step), so a world may legally run ahead
  across boundaries only inside its 4-boundary action window.
  Slab-count ceiling from per-world window sums: identity-blind
  full-rotation bound 70.2%; identity-aware bound (top-16 accepts
  tails) 45.8% of late-window slabs mergeable. Straggler persistence:
  top-set overlap 0.374 at lag 1, 0.188 lag 2, 0.074 lag 4, 0.020 lag
  8 — stragglers ROTATE within an action window (transient contact
  events, not a fixed slow world), so the persistent-straggler
  objection is refuted by measurement. Wall-time value of the
  mergeable slabs: 16.7% (proportional cost) to 45.8% (uniform) of the
  late window — i.e. plateau ~14.21 -> ~11.8..7.7 s/iter at ceiling.
  THE BRACKET STRADDLES THE 10 s GOAL.
- VERDICT: overlap survives as the ONLY factor-scale lever inside the
  grant rails — RECOMMEND a pass-20 design measurement, not a build.
  Named blockers to resolve first: (a) the plateau-window profile that
  converts the [17%, 46%] bracket into a number (what does a 6-active
  slab actually cost?); (b) collision cadence — the boundary collide
  is batch-wide at one boundary state, so worlds at different
  boundary indices need per-group collision (structural); (c) the
  IsaacLab per-physics-step host pipeline between boundaries (actuator
  /event updates) must be shown state-decoupled within the action
  window; (d) prefer a per-world boundary-index SINGLE march (no
  second workspace — sidesteps the pass-4 memory concern) over
  two-instance overlap.
Provenance: p19_occ_run.sh, p19_occ_1024x25.{log,telemetry,audit,
stamps}, p19_pers_run.sh + p19_audit_hook/ (top-16 extension),
p19_pers_1024x25.{log,telemetry,audit,stamps}, p19_occ_analyze.py,
p19_pers_analyze.py; march_audit_hook/ (the straggler-era hook,
unmodified).

PASS-20 RECOMMENDATION: (1) PROFILE THE PLATEAU WINDOW — nsys scoped
to late iterations of the 1024 rig (not the 512-eager flail window),
slabs grouped by active-count regime: prices the straggler slab's
fixed cost, converts the overlap bracket into a number, and re-ranks
the percent levers for the regime that is 88% of plateau slabs. (2) On
that evidence, either open the overlap design (blockers (b)-(d) above)
or kill it with a number. (3) fused_update dof-phase occupancy shave
(~few-% class) only if the plateau map confirms fused_update is
material in the straggler mix. (4) The estimator escalation to Marco
stands and is sharpened: percent levers project ~13.4-14.0 s/iter;
the two routes to 10 s are overlap (bracket straddles it) and the
estimator rail (~1.20 ms/substep single-solve arithmetic).

## PASS-20 — straggler-regime profile + overlap point estimate + design
## (2026-08-16, MEASUREMENT + DESIGN ONLY, zero solver edits; profiles
## the 88%-of-plateau-slabs regime no map had ever scoped)

RIG (method choice per the pass-19 open question): profile whole REAL
plateau boundaries and split kernels by march-iteration index — NOT
profiler-gating on the narrow predicate, which is impossible without
duplicating the clamp+build on host (the attempt width is computed
INSIDE _substep_body, after _clamp_dt_to_boundary) and would fragment
the capture range hundreds of times. One rsl_rl training run, exact
p19_occ replica (1024x25, seed 42, production flags, det unset) run
EAGER (NEWTON_SAP_ADAPTIVE_GRAPH=0, kernel attribution; manager wraps
no outer capture for SAP-adaptive, so the tier-3 per-iteration loop
runs) under nsys `-t cuda,nvtx` with a cudaProfilerApi window scoped
to boundaries [1928,1967] (40 boundaries fully inside train iter 20).
The p20 import hook (p19 audit hook + profiler gating + NVTX) wraps
every _substep_body call in an NVTX range "sb<k>" and reads
_active_counts[0] post-body (the exact post-clamp dt>0 attempt width
the narrow/wide branch predicate consumed); boundary calls wrapped in
"bnd<b>". Offline: kernels -> runtime rows by correlationId -> NVTX by
per-thread bisect (698 slabs, 489,815 kernel rows, 0 unmatched).
Audited+eager+profiled walls are perturbed and never cited; only
kernel sums, the itermap and the audit are consumed. Run clean:
exit 0, physics_diverged 0, full 2400-boundary audit.

REGIME REPLICATION: window widths p50=12, narrow (a<=64) 80.8% of
slabs, deep (a<=16) 55.7%, mean 13.8% — the p19 occupancy structure
reproduced at the profiled window (82.5% narrow / mean 12.8%).
Slabs/boundary 17.45 in-window (audit late window 17.47). Demand
cross-check: late-window slabs 10,061/6 iters ~ 5,030 evals/iter vs
p19's 5,170 (det-unset trajectory, -3%). GPU-bound cross-check: eager
GPU-kernel sum 143.5 ms/boundary (131.7 slab + 11.5 boundary-cadence
+ 0.3 outside) vs graph-mode plateau wall 148 ms/boundary = 0.97 —
kernel-time fractions transfer to wall.

(a) THE STRAGGLER MAP — cost by active-width bucket (ms/slab GPU):
  full  a>=922:  80 slabs (11.5%)  19.53   29.7% of window GPU
  high 257-921:  10 slabs ( 1.4%)  16.07    3.1%
  mid   65-256:  44 slabs ( 6.3%)  13.24   11.1%
  narrow 17-64: 175 slabs (25.1%)   7.08   23.5%
  deep    1-16: 389 slabs (55.7%)   4.43   32.7%
Group ranking, DEEP slab (4.43 ms): fused_ladder 0.97 (21.9%),
fused_update 0.78 (17.5), ls_chain 0.63 (14.3), solve_prep 0.60
(13.6), free_motion 0.39 (8.8), gemm_tile 0.33 (7.5), cholesky 0.24
(5.4), gemm_pack 0.15 (3.4). FULL slab (19.53 ms): fused_ladder 4.62
(23.7), gemm_pack 4.50 (23.1), gemm_tile 2.43 (12.4), fused_update
2.25 (11.5), ls_chain 1.84 (9.4), free_motion 1.63 (8.3). WHOLE-WINDOW
plateau-mix map (7.55 ms/slab — the map that replaces the saturated
p18 ranking for lever pricing): fused_ladder 24.5%, fused_update
15.2, gemm_pack 12.5, ls_chain 12.1, gemm_tile 9.4, solve_prep 8.2,
free_motion 7.6, cholesky 4.4. (p18's 512-saturated map is a
different scale and regime — kept for its deltas, not for ranking.)
Boundary-cadence work: 11.54 ms/boundary, 98.9% collision
(mesh_triangle_contacts_to_reducer alone 9.06 ms/pass). Outside-bnd
GPU work: 0.28 ms/boundary (the whole env-side between-boundary
pipeline — empirically negligible).

(b) THE ANSWER: a ~4-active slab costs 4.43 ms = 0.227 of a full
slab. Piecewise fit: narrow branch c(a) = 3.91 + 0.0965*a ms (R2
0.65), wide c(a) = 12.58 + 0.0069*a (R2 0.91). The narrow-branch
FLOOR (3.9 ms) is the straggler regime's cost, and crossing the
mc_width=64 branch line triples the slab price (7.1 -> 13.2 ms).

(c) FULL-WIDTH-BY-DESIGN in the straggler regime: 46.8% of a deep
slab (2.07/4.43 ms), 44.9% over all narrow slabs = 25.2% OF WINDOW
GPU. Classification: per-kernel deep-vs-full per-instance time ratio
plus grid behavior. The FWBD population: fused_update (the WHOLE
kernel: grid fixed, t-ratio 0.70 — it never got the narrow env_grid),
the ls_chain serial direction/init components (0.60 of deep slab,
t-ratio ~0.74-0.97 — the pass-15c "~1% at production" refutation was
REGIME-DEPENDENT: in the straggler mix it is ~13% of the slab), the
per-attempt contact scatter _scatter_sap_contacts_to_env_direct*
(t-ratio 1.07: it walks ALL 1024 worlds' frozen contacts regardless
of world_active — 0.37 ms of every deep slab), base_cost, and the
prep copy/list fleet. This 25.2% is deletable GPU WORK (not launch
count — pass-19 law respected): route fused_update and the serial
LS-direction chain through the narrow env_grid, world-gate the
scatter's contact rows. It is the NEW TOP LEVER, semantics-free.

(d) OVERLAP BRACKET COLLAPSED: actual late-window cost model from the
reconstructed per-boundary widths (survival of the accepts hist,
validated in-window: mean |err| 11 worlds, bias 1.085 — see
CORRECTION below) under the measured c(a). Ideal merged single-march
schedule per 4-boundary action window, join across boundaries
bracketed and ANCHORED TO MEASUREMENT: rank-persistence Monte Carlo
with P = the measured lag-1 straggler-set overlap 0.374 (p19
rotation), validated: its per-window ideal-slab mean 39.3 vs the
top-16 identity arithmetic's 37.7 on this same audit (independent
P=0 gives 25.5, comonotone P=1 gives 55.9 — both rejected by the
identity check). RESULT:
  value = 0.197 of late-window GPU cost (bracket ends: 0.345
  independent, ~0.00 comonotone — the wide-branch price of merged
  tails kills the comonotone case);
  slab count 10,061 -> 5,661 (identity arithmetic: 5,434 = 0.4599,
  replicating p19's 0.458);
  PROJECTED PLATEAU AT OVERLAP CEILING: 14.21 -> ~11.4 s/iter gross;
  ~11.5 with masked catch-up collides (~12 extra masked passes/window
  at ~0.3-0.5 ms each), ~12.6 s/iter worst-case if catch-up collides
  stay full-batch (432 unmasked passes = 8.1% of cost).
The pass-19 bracket [16.7%, 45.8%] collapses to 19.7%, near its
proportional end — because a straggler slab costs 0.227 of full, not
1.0. INTERPLAY: landing the (c) FWBD narrowing first shrinks the
narrow floor c0 (~3.9 -> ~2.5 ms at 70% realization), repricing
overlap to roughly 12-15% of the post-narrowing window — still above
the 10% kill line, but second in order.

CORRECTION to pass-19: the audit's "rejects 0" was a NULL READ —
NEWTON_ADAPTIVE_MARCH_LOG was unset in the occ/pers runs, so
_count_rejects never launched and _reject_count_buf held its
allocated zeros. Rejects/floor-latch at the plateau are real:
measured attempt widths exceed accepts-survival widths by 8.5%
(p20 in-window). The p19 slab-count ceiling stands (per-boundary
iters were measured directly, not derived from accepts); the
accepts-based active-fraction stats are ~8% understated.

VERDICT: overlap SURVIVES the kill line (19.7% > ~10%) — the pass-21
IMPLEMENTATION DESIGN follows, but the recommended ORDER is FWBD
narrowing first (bigger, semantics-free, and it must land first so
overlap is priced against the stack it would ship on).

DESIGN — cross-boundary overlap as a RUN-AHEAD SINGLE MARCH (no
second workspace; blockers (b)(c)(d) resolved):
- SHAPE: one march per integrate call, as today, but a world reaching
  its boundary target inside the action window does not park: a new
  device crossing kernel (_advance_boundary) bumps its per-world
  next_time by dt_outer (capped at the action-window end), applies
  the per-world boundary bookkeeping in-place (_seed_dt's clamp of
  ideal_dt into dt/dt_half, _debt_guard's carry bound, per-world
  substeps_frame rollover), and sets a per-world "crossed" flag. The
  existing per-world clocks carry the whole design: sim_time,
  next_time, dt, dt_half, ideal_dt, dt_ceiling, consec_rej are
  already per-world arrays; _clamp_dt_to_boundary and _adapt_dt work
  UNCHANGED (each world lands exactly on each T_j via the clamp —
  landing-sliver and boundary-commit semantics preserved per world).
  Call-return predicate: mark_unfinished_contained compares sim_time
  against a per-call scalar target T_{j+1} (device 1-int written at
  integrate entry, graph-replay safe) instead of per-world next_time
  — run-ahead worlds count as finished for the call; the LAST call of
  the window returns only when every world sits at the window end, so
  the env-visible action-edge state stays batch-synchronized.
- COLLISION (blocker b): all boundary collides become crossing-
  batched conditional nodes inside the march body: wp.capture_if(any
  crossed since last collide) { masked collide + ADOPT }. Masking =
  one new world-mask input to compute_shape_aabbs (newton-adaptive
  newton/_src/sim/collide.py): non-crossing worlds' shapes emit
  sentinel AABBs, so SAP broadphase yields no pairs and every
  downstream pair/candidate-parallel kernel (incl. the 9.06 ms
  mesh_triangle reducer) scales with the crossing subset; pipeline
  call structure otherwise untouched. Contact-set persistence moves
  from the global Contacts buffer (today re-read by EVERY attempt) to
  a per-env SET store: split _scatter_sap_contacts_to_env_direct*
  (sap_warp sim/contact_jacobian.py, 4 variants) into ADOPT (global
  buffer -> per-env topology/material rows, runs once per crossing
  batch for crossed worlds only) and ANCHOR (per-env SET + body_q ->
  phi0/jac/R_WC, per attempt, list-indexed by world_active — which
  also deletes the scatter's FWBD cost from (c), 0.37 ms/deep-slab).
  Each world's contact set stays anchored at ITS boundary-entry
  states — per-world contact cadence and anchoring are SEMANTICALLY
  IDENTICAL to today; what changes is buffer packing order (det-off:
  same class as today's atomic arrival order; det=1: canonical ranks
  are per-world state functions, computed per crossing batch —
  det_slots_external becomes per-batch). mjwarp_manager: NO change
  (boundary call signature and cadence unchanged).
- HOST PIPELINE (blocker c): measured 0.28 ms/boundary GPU — the
  env-side per-physics-step work (apply_action, write_data_to_sim,
  scene.update, sensor FK/force accumulation, actuator state update)
  is element-wise per world and control is CONSTANT across the
  window (process_action once per action; apply_action rewrites the
  same targets — run-ahead under current control is exact). Two
  task-level invariants must hold and be asserted at construction:
  (1) no consumer of sub-action-cadence state — in this task the
  contact-sensor reward terms read latest-value force_matrix_w at
  action cadence and no history/air-time terms exist; (2) no
  per-physics-step control variation. Mid-window scene.update/sensor
  reads DO see run-ahead worlds at mixed times — dead reads in this
  task, but a BATCH-VISIBLE semantic change of stepping: flagged for
  Marco's consent in the escalation section (it is NOT the
  estimator/comparison-semantics rail: per-world dt control, the
  3-solve step-doubling estimator, accept/reject, tol and optimality
  are bit-for-bit untouched per world).
- TWINS/GATES: per-boundary iteration counts change by construction,
  so the march-equivalence gate g3 and the audit/telemetry
  (_iteration_count_buf, _substeps_frame, _log_march_boundary)
  redefine to per-window totals + per-world accept sequences; twin
  worlds still march in lockstep (identical per-world state =>
  identical crossings), so the twins rule survives at per-world
  granularity. Containment/status word semantics unchanged (slot 1
  sticky across the window's calls).
- CONDITIONAL GRAPH: the whole-march while-node body gains the
  capture_if collide/adopt node and the crossing kernel; the body
  stays a fixed conditional stream, keyed per dt_outer as today.
  max_substeps cap stays per call.

GRANT ARITHMETIC REFRESH: 10 s/iter needs 0.70x of the 2.75
ms/substep plateau price. Measured ceilings now in hand: FWBD
narrowing 25.2% of window (realizable share TBD by implementation,
est. 15-20%) THEN overlap ~12-15% post-repricing => combined
~0.65-0.75x — THE 10 s GOAL IS ARITHMETICALLY REACHABLE INSIDE THE
RAILS FOR THE FIRST TIME, without touching the estimator. The
estimator escalation stands as the route BEYOND ~10 s (single-solve
arithmetic ~1.20 ms/substep), no longer the only route to it.

PASS-21 RECOMMENDATION: (1) implement the FWBD narrowing set from
(c) — fused_update through the narrow env_grid, serial LS-direction
chain list-indexed, world-gated scatter rows (or pull the ADOPT/
ANCHOR split forward from the overlap design — it deletes the same
scatter cost and pre-builds the overlap foundation), default-ON
bitwise-classed flags, full gate chain, decisive production A/B;
(2) re-price overlap on the post-narrowing bytes with this pass's
c(a) method (the p20 splitter is committed tooling for that);
(3) put the mid-window-visibility consent question to Marco alongside
the standing estimator escalation. No speedup promised beyond stated
bounds; FWBD realizable share is the pass-21 measurement.
Provenance: p20_prof_run.sh, p20_prof_hook/ (import hook: audit +
profiler gating + NVTX, refuses to arm per-iteration wrapping unless
eager), p20_prof_1024x25.{log,telemetry,audit,itermap,stamps},
p20_prof_plateau.{nsys-rep,sqlite}, p20_split_kernels.py ->
p20_split.txt (the maps, FWBD, c(a) fits), p20_overlap_value.py ->
p20_overlap.txt (joins, validation, point estimate).

## PASS-21 — FWBD narrowing LANDED default-ON (2026-08-16; sap_warp
## aac9694 + newton-adaptive 52005367; kernel-time win real and large,
## wall realization limited by a dispatch-bound deep tail — the finding
## that reorders the roadmap)

IMPLEMENTATION (NEWTON_SAP_NARROW_V3, default ON, "0" restores the
full-width launches byte-for-byte; bitwise class — no arithmetic
changed, only which envs launch; both graph cache keys carry the two
carrier flags):
1. fused_update: list-indexed through the PREPARE (world-active) list
   at the env-grid budget. The prep list covers every env whose mask or
   state the kernel can change (newton-live, just-converged, cap-hit
   all belong to still-marching worlds; in-kernel category branches are
   unchanged for them); worlds outside the list entered the solve
   pre-converged from the full-width entry init, so only their
   dead-state rewrites (zero cost/norms, newton_active re-zero) are
   skipped — convergence masks stay correct for every env (proven:
   G8 phi0 ON==OFF byte-identical; G2 family bitwise).
2. Serial LS-direction chain: search_direction + init_backtracking +
   accumulate_ls_iterations newton-list-indexed, base_cost call routed
   through the same list. Init and accumulate narrow TOGETHER (a
   de-listed env's stale ls_accepted/ls_iterations must never be
   re-read into the total); the init narrow is gated on ladder budget
   > 1, where every ladder exit leaves ls_active 0 (the budget-1
   kept-active corner needs the full-width clear). Exact-root LS
   variant left unrouted (identity lists).
3. Per-attempt contact scatter (4 variants) + per-attempt det rank
   walk: world-gated early-outs after the env is computed; per-row
   arithmetic and per-world slot order unchanged for active worlds;
   the count reset honors the gate (inactive worlds keep their counts
   — the frozen buffer would reproduce them). Boundary-cadence
   external-slot walk stays ungated (every world marches a fresh
   boundary).
4. Copy fleet NOT narrowed, by measurement: p20/p21 t-ratios 0.97-1.15
   (deep vs full per-instance) = width-independent = launch-bound;
   pass-19 law says no pay, and the p21 split confirms (remaining
   deep-slab FWBD 0.277 ms = copies 0.107 + list machinery 0.090 +
   eval_rigid_tau 0.036 — the list machinery IS the compaction
   infrastructure and the entry init is full-width by contract).
Tripwires: emission-time narrow-site tags on both carriers
(fused_update / ls_search_direction / ls_init_backtracking /
ls_accumulate_iters; jacobian scatter_world_gate), env-grid capacity
guard bounds every new list-indexed launch, OFF-leak asserts in every
pinned-off probe cell.

GATES (all green on final bytes, p21_ artifacts): G1 construct; G2
flag-equivalence 40 PASS cells incl. the NEW narrowv3 family
(eager/graph/conditional judged BITWISE against a same-stack pinned-off
reference — production fused stack + march compaction + acr=0) with
engagement + leak guards; probe fix folded in: the ls-compact
engagement assert now expects a ZERO counter under the folded alpha-max
rung (the fold deletes the whole trial launch chain, LS-list rebuilds
included — the march+fused-stack combination had never been probed);
G3 march-equivalence PASS, iterations [6,25,20,24,19] UNCHANGED; G4
determinism PASS (954==954); G5 containment PASS; G6 err_tol 0 viol /
2880, max ratio 0.745, floor 0, dt_run_min 1.68e-3; G7 rest ok; G8
phi0 ON-vs-OFF byte-identical.

DECISIVE A/B (1024x8 seed 42, det unset, production): per-substep
(coll) whole-run ON 2.774 vs OFF 2.922 = -5.1%; late-3 2.614 vs 2.719
= -3.9%. Substeps ON/OFF 1.173 (det-unset trajectory divergence;
per-substep is the metric). Ops: first ON launch segfaulted in USD
parse during scene build (pre-solver, no SAP frames on stack); rerun
clean — startup flake, not the flag.

REGIME SPLIT (p21_prof_run.sh = exact p20 rig replica, same [1928,
1967] window, production flags eager; OFF baseline = the committed p20
profile): deep slab 4.4293 -> 3.7557 ms (-15.2%); narrow floor c0
3.907 -> 3.148 (-19.4%); wide c0 12.576 -> 11.435 (-9.1%); window GPU
at p20's FIXED slab mix 7.550 -> 6.859 ms/slab = -9.2%. Deep-slab FWBD
aggregate 2.074 (46.8%) -> 0.277 ms (7.4%); the classifier confirms
every targeted kernel now narrowed/live (fused_update grid 0.06,
ls _c/_i/_a 0.25, scatter t-ratio 1.07 -> 0.17 at 0.368 -> 0.052
ms/slab, base_cost 0.06). Per-site deep-slab deltas: scatter -0.32,
fused_update -0.09, ls_chain -0.04, base_cost -0.003. REALIZATION:
of p20's 2.07 ms/deep-slab "deletable" FWBD, ~0.67 was deletable
width-work; the rest is live-env SERIAL latency (one-thread-per-env
dof^2 fp64 chains) plus per-launch floors — pass-19's law, now
measured at kernel level.

PLATEAU RE-MEASURE (p21_run.sh replica, det unset): iters 19-24 mean
12.93 s/iter (p19 14.21) — but window demand 4641 substeps/iter vs
p19's 5170 (-10.2%, det-unset trajectory divergence), so the wall is
NOT clean lever evidence; ms/substep 2.79 vs 2.75 (+1.5%,
mix-confounded the other way: fewer, wider slabs). Whole-run 243.9 s,
cum substeps 89,271 (p19 89,694). CONTROLLED VERDICT — det=1
matched-trajectory 25-iter pair (p21_det_run.sh): demand IDENTICAL
(cum 88,935 == 88,935; window 4995 == 4995 — the bitwise class
certificate holding at production scale), plateau wall ON 14.71 vs
OFF 14.83 s/iter = -0.8%; ms/substep 2.94 vs 2.97 = -0.9%.

THE FINDING: the deep-straggler tail is DISPATCH-BOUND under graph
replay, not kernel-throughput-bound. Deleting 15-19% of the tail's
kernel time moved its wall < 1% at matched trajectory; kernel share of
plateau wall dropped from p20's 0.97 to ~0.89-0.90 on the v3 bytes
(134.7 ms/bnd wall vs ~120 ms/bnd eager kernel sum, cross-trajectory
approximate). The p20 window-GPU ceiling (25.2% FWBD) was a
KERNEL-TIME ceiling; its wall-realizable share at the plateau is the
~1-5% class (A/B -5.1%/-3.9% in the wide-heavier early window, det
pair -0.8% at the straggler-heavy plateau). DECISION: kept default ON
per the stated criterion (regime split material: deep -15.2%, c0
-19.4%, fixed-mix window -9.2%, gates green; A/B supporting) — the
kernel-time deletion is real, costs nothing, and every future lever is
priced against a cleaner stack; but the plateau headline stays
demand-honest: the lever's wall value at the plateau is ~1%, and
12.93 s/iter is a trajectory artifact, not the lever.

GRANT ARITHMETIC REFRESH + OVERLAP RE-PRICE: 10 s/iter at p19 demand
needs 1.93 ms/substep; production price after this pass ~2.75-2.79
(mix-dependent), det=1 2.94. REVISION of p20: "FWBD narrowing 15-20%
realizable" was kernel-time arithmetic — its WALL share at the
plateau measured ~1%; the 10 s goal is NOT reachable by work-price
levers inside the tail. What the tail is priced in is SLABS
(dispatch floors ~3.1-3.7 ms each). Cross-boundary overlap deletes
whole slabs — floor included — so its wall value is AT LEAST its
p20 kernel value (19.7% of late-window GPU; slab count 10,061 ->
5,661 at ceiling) and, in the dispatch-bound tail, likely above it:
overlap is now decisively the TOP lever, and slab-count/dispatch
reduction (not kernel-time) is the only currency that pays there.
Post-v3 repricing note: c0 fell only 19% (not the ~36% p20 assumed at
70% realization), so the p20 INTERPLAY estimate "overlap reprices to
12-15% post-narrowing" was pessimistic — the straggler share of
window cost barely moved (32.7% -> 30.5% fixed-mix): overlap stays
~18-20% of the (9% cheaper in kernel terms) window. Route to 10 s:
overlap (~11.4-12.6 s/iter projected, p20 design + consent pending)
THEN the estimator escalation (single-solve ~1.20 ms/substep) or an
equivalent slab-count lever; work-price levers inside the tail are
closed by this pass's measurement.

PASS-22 RECOMMENDATION: (1) implement the run-ahead single-march
overlap per the p20 design against these bytes — it is the only
priced lever whose currency (slab deletion) matches the measured
bottleneck (dispatch floors); put the mid-window-visibility consent
question to Marco first (escalation stands); the ADOPT/ANCHOR scatter
split it wanted is already half-realized by the world gate (scatter
0.052 ms/deep-slab). (2) Micro only if piggybacking: eval_rigid_tau
narrow (0.036, t 0.76) + fused_ladder list-bound (dead-tile share
~0.1 ms/deep-slab) — ~2% kernel class, wall value doubtful per this
pass. (3) If overlap is deferred: measure the dispatch floor directly
(slab wall vs kernel span inside one replay) to size a
launch-count-reduction lever honestly before building one.
Provenance: p21_full_chain.sh, p21_g1..g8 logs/JSONs,
p21_ab_run.sh -> p21_ab_{on,off}.*, p21_ab_compare.py, p21_prof_run.sh
-> p21_prof_plateau.{nsys-rep,sqlite} + p21_prof_1024x25.*,
p20_split_kernels.py -> p21_split.txt, p21_run.sh -> p21_plateau.*,
p21_det_run.sh -> p21_det_{on,off}.*, p14_plateau_analyze.py.

## PASS-22 — RUN-AHEAD single march LANDED, DEFAULT OFF (2026-08-16;
## implements the pass-20 design; newton-adaptive 45e07db2+7ea7f34e+
## e41cc070+8f3ef7e3, sap_warp 2a119d2; the default flip is Marco's
## one-line decision — consent question sharpened below with the
## measured value)

IMPLEMENTATION (NEWTON_SAP_RUNAHEAD, "1" enables; OFF = the per-boundary
march byte-for-byte — G2's 42 bitwise cells and G3's unchanged iteration
vector pin it):
1. THE MARCH (solver_sap_adaptive.py): a world reaching its boundary
   target inside the action window does not park — the device crossing
   kernel _ra_advance_boundary bumps its next_time by one float32
   dt_outer add (capped at the window end), applies its boundary
   bookkeeping in place (_seed_dt's clamp of ideal_dt into dt/dt_half,
   per-world substeps_frame rollover) and flags it crossed.
   _clamp_dt_to_boundary and _adapt_dt are BIT-UNTOUCHED (the
   estimator/accept-reject/per-world-dt rail held); every world lands
   exactly on each T_j via the existing clamp (landing-sliver and
   boundary-commit semantics preserved per world). Call-return
   predicate: mark_unfinished_{contained,with_status}_target (additive
   kernels in adaptive_boundary.py — the MuJoCo twin's kernels are
   byte-untouched; run-ahead is SAP-ONLY) compares sim_time against a
   per-call device scalar written at integrate entry (graph-replay
   safe); the LAST call of the window returns only when every world
   sits at the window end — action-edge state stays batch-synchronized.
   Window bookkeeping: NEWTON_SAP_RUNAHEAD_WINDOW (default 4; MUST
   equal the env decimation) + NEWTON_SAP_RUNAHEAD_PHASE (call-stream
   offset). Host-side per-window float32 add-chain targets are
   bit-identical to the device next_time chain, so window-end parking
   and the reset-resync signature (next_time == 0, exact) compare
   exactly. Clocks rebase once per WINDOW (Fix B's per-boundary rebase
   becomes per-window inside the window — the one construction-level
   fp deviation from OFF, see the oracle). A mid-window manager reset
   is re-seated at the current call start and flagged crossed (fresh
   contact set at its post-reset state). A floor-latch world does NOT
   run ahead: the crossing kernel parks it at the window end, latch
   visible for the post-call read (G5b). The debt guard is judged
   against the call target (_debt_guard_target; per-world carry bound
   verbatim); max_substeps caps per call as before.
2. COLLISION: boundary collides become ONE crossing-batched conditional
   node at the top of the march body — wp.capture_if(any crossed)
   { masked collide + ADOPT + disarm }. Masking = new
   compute_shape_aabbs_masked (collide(world_mask=...)): non-crossing
   worlds' shapes emit INVERTED sentinel AABBs (fail the interval
   overlap test against every partner), so broadphase yields no pairs
   for them and pair-parallel downstream work scales with the crossing
   subset; global-world shapes (ground plane) always participate. The
   window-open call runs the full-batch collide+adopt eagerly (every
   world crosses into its first boundary there) — today's cadence.
   CAPTURE-SAFETY (blockers the pass-20 design did not anticipate; all
   fixed at the root): a conditional CUDA-graph body may not contain
   allocation nodes, and the pipeline allocated per pass —
   wp.utils.array_scan (native temp alloc per call) at FIVE per-collide
   sites (broadphase sweep-range cumsum, mesh-mesh + mesh-plane
   block-offset scans x2) replaced by an exact chunked int32 scan
   (byte-identical integer output; new geometry/capture_safe_scan
   module), and per-launch zero-size placeholders cached. The
   deterministic contact sorter's native temp alloc is bypassed:
   in-march collides skip the global sort (collide(sort_contacts=
   False)) and the adopt derives the SAME canonical per-env slot order
   from the per-contact sort keys directly (two-pass key-ranked walk;
   the global sort is a stable sort by those keys, tiebreak = buffer
   index = stable arrival order). radix/segmented sorts are
   capture-safe as-is (bisected).
3. CONTACT SPLIT (sap_warp contact_jacobian.py): the per-attempt
   scatter splits into ADOPT (global buffer -> per-env SET store, once
   per crossing batch, crossed worlds only; raw dtypes) and ANCHOR
   (per-env set + body_q -> phi0/jac/R_WC per attempt, world-gated).
   ANCHOR per-row arithmetic is copied VERBATIM from the direct
   scatter (f32-pose and f64-pose variants; f64 contact buffers
   refused explicitly) — G8 phi0 ON==OFF to the digit. Per-world
   contact cadence and anchoring are semantically identical to the
   per-boundary march; under det-off only buffer packing order changes
   (today's atomic arrival-order class). capture_local_snapshots
   refused (its full-width fills would erase non-crossing rows).
4. KEYS/TRIPWIRES/TWINS: NEWTON_SAP_RUNAHEAD in BOTH graph cache key
   tuples; engagement counters (crossings, consumed adopt batches)
   allocated unconditionally so OFF reads are real observations;
   emission-time adopt/anchor site tags; OFF-leak asserts in every
   pinned-off G2 cell. mjwarp_manager: NO changes (the design's claim
   held — call signature/cadence unchanged; the manager's per-call
   diverged-mask reset is absorbed by the resync kernel). Telemetry
   drift under ON (only): dt_histogram unfinished_worlds and the
   march-log resid/n_debt columns count run-ahead worlds mid-boundary
   at call exits (an engagement measure, not under-advance).

WINDOW ALIGNMENT (measured, probe_runahead_alignment.py ->
p22_alignment.json): env construction and reset consume 0 boundary
calls; every env.step consumes exactly decimation=4; required phase 0;
verdict ALIGNED. The W==decimation contract is configuration — the
default flip should keep the alignment probe in the launch checklist.

GATES (all green on final bytes, p22_ artifacts): G1 construct OFF+ON;
G2 flag-equivalence 48 cells — every legacy cell pins runahead "0" and
re-passes BITWISE (OFF path byte-preserving), the new runahead family
carries its own repeat oracle (determinism-in-mode; a run-ahead march
is a different LEGAL schedule, so no cross-mode bitwise contract
exists) with graph + whole-march-conditional arms bitwise and
engagement/leak green; G3 march-equivalence PASS, iterations
[6,25,20,24,19] UNCHANGED (the cross-build OFF certificate, covering
the chunked-scan swap); G4 determinism OFF + ON PASS; G5 containment
OFF + ON PASS (ON isolation judged at window edges — mid-window batch
records are mixed-time by design; healthy worlds bitwise vs control at
every edge, latch visible same call, frozen state held); G6 err_tol ON
0 violations / 2880, max ratio 0.986, floor 0, dt_run_min 1.36e-3;
G7 rest ON ok; G8 phi0 ON-vs-OFF identical TO THE DIGIT at
rest/press/swing.

MIXED-TIME ORACLE (sap_runahead_oracle_probe.py, committed — the
decisive semantic gate; 8 worlds x 2 windows, det=1, per-world impact
spread): (a) ISOLATION: every world's batch rows == its SINGLE-WORLD
solo run, BITWISE, at every window edge (state + controller carries) —
the run-ahead scheduler couples no worlds; (b) ON-vs-OFF window-edge
committed state: positions BITWISE identical, velocities max |dqd| =
3.7e-9 — the predicted float32 per-window-vs-per-boundary clock-rebase
sliver class, five orders below tol; (c) engagement exact (crossings ==
the N*(W-1)*windows structural count; max mid-window lead 2.0 ms = a
full boundary); (d) ON repeat bitwise incl. mid-window clocks.
Post-march dt is excluded from (a) as attempt-transient dead state
(trailing no-op clamps zero it a schedule-dependent number of times;
the carried controller step is ideal_dt, which IS compared).

DECISIVE A/B (1024x8 seed 42, det unset, production flags, both arms on
final bytes, p22_ab_*): trajectories track through iter 6 (per-iter
slab counts reproduce each arm's independent replica exactly), then
det-unset chaos splits iter 7. MATCHED WINDOW (iters 0-6, the
wide/flail regime): slabs ON 3051 vs OFF 3249 (0.939 — little to merge
while marches are wide), but wall/slab ON 11.62 vs OFF 8.86 ms
(+31%): the CATCH-UP COLLIDE FIRE cost — the conditional node fires
once per march iteration carrying >= 1 crossing, so the wide regime
pays many masked passes per window against OFF's 4 boundary passes.
Whole-run ms/substep ON/OFF 1.054; ON's (heavier-draw) iter-7 window
ran 1493 slabs at 1.86 ms/substep vs OFF-late 2.5-3.7 — merged dense
marches are cheap per unit work. Both arms exit 0, physics_diverged 0,
no contained failures, no capture downgrades.

25-ITER PLATEAU (same-bytes pair, det unset, p22_plateau_{off,on}):
OFF 15.09 s/iter at 5533 evals/iter (2.73 ms/substep), whole run
249.5 s; ON 12.54 s/iter at 4608 evals/iter (2.72 ms/substep), whole
run 241.1 s. Plateau wall ON/OFF = 0.831 with the eval(=slab) axis at
0.833 and per-substep FLAT (0.997): at the plateau the merged schedule
costs the same per slab (narrow crossing batches make catch-up
collides cheap) and the wall drops with the slab count — exactly the
slab-deletion shape pass 20 priced, and the measured 12.54 lands
inside p20's projected 11.4-12.6 s/iter band (scaled to this pair's
heavier OFF draw: 15.09 x (1-0.197) ~ 12.1 + masked catch-up ~ 12.3-
12.6). HONEST LIMIT: det unset means per-world accepted DEMAND is not
observable in this telemetry (evals = 3 x batch march iterations), so
the -16.7% slab axis is (deletion) confounded with (trajectory draw);
the p21 det-pair trick is unavailable because ON is legally not
bitwise vs OFF. The demand instrument (log cum accepted steps next to
cumulative_substeps — a one-line telemetry addition, in-grant) is the
pass-23 decisive measurement. Whole-run net: -3.4% (the wide-regime
collide-fire cost eats most of the non-plateau win).

VERDICT: landed DEFAULT OFF as directed. The lever's currency (slab
deletion at a flat per-slab price) is measured live at the plateau;
its early-window catch-up collide cost is real (+31%/slab in the
matched wide window) and is the first thing pass 23 should shave
(bounded crossing-batch throttle: hold crossed worlds parked <= k
iterations so fires batch, trading a little merge value for far fewer
masked passes — semantics-preserving because held worlds simply wait
at their boundary, exactly the OFF behavior, before crossing late).

PASS-23 RECOMMENDATION: (1) put the sharpened consent question to
Marco (below) — the flip is one line and every semantic gate is green;
(2) add the accepted-demand counter to the manager telemetry and
re-run the plateau pair for the unconfounded deletion number;
(3) collide-fire accounting (the _ra_adopts device counter already
counts consumed batches — log it) and, if fires/window is large in the
wide regime, the bounded crossing-batch throttle above; (4) micro:
none — work-price levers stay closed per pass 21.
Provenance: p22_final_chain.sh, p22_g1..g8 logs/JSONs, p22_oracle.log,
sap_runahead_oracle_probe.py (committed), probe_runahead_alignment.py
-> p22_alignment.json, p22_ab_run.sh -> p22_ab_{on,off}.*,
p22_ab_compare.py, p22_run.sh -> p22_plateau_{off,on}.*,
p14_plateau_analyze.py, p22_alloc_trace*.py / p22_condif_bisect*.py
(the capture-safety forensics), p22_g5_repro.py.

## PASS-23 — demand instrument + crossing-batch throttle LANDED; the
## pass-22 plateau claim DECONFOUNDED and revised to ~0 (2026-08-16;
## sap_warp 2a119d2 untouched; default STAYS OFF — see the consent-relay
## note in the escalations section)

INSTRUMENT (demand counter; host-read-only, no kernel change): the
per-world ACCEPTED substep counter _cum_accepted (maintained by _adapt_dt
since the solver landed, never exposed) now reads out as
cumulative_accepted_steps() beside cumulative_substeps() (= slabs x 3 --
the batch-iteration axis, schedule-DEPENDENT); the manager telemetry line
appends cumulative_accepted= ra_cross= ra_fires= (getattr-guarded; the
MuJoCo twin is untouched), and the march CSV gains cum_acc, ra_cross,
ra_fires columns. Accepted demand is the schedule-invariant work axis;
ms/accepted-substep is the demand-normalized price. WHAT THE COUNTER
IMMEDIATELY EXPOSED: same-config det-unset draws differ 2-10% in demand
between arms (measured pairs: 1.024-1.104), and det=1 does NOT equalize
demand across the ON/OFF pair (the 4e-9 clock-sliver class seeds chaos;
p22 predicted the p21 det-pair trick would not transfer -- confirmed and
measured: det=1 pair demand ratio 1.0697). Demand EQUALITY is therefore
unprovable for ON-vs-OFF; demand OBSERVABILITY replaces it, and every
wall claim below is demand-normalized.

DECONFOUNDED PLATEAU (Task-1 deliverable; REPLACES the pass-22
"-16.9% (confounded)"): the run-ahead plateau value at matched demand is
-4% TO 0 PER ACCEPTED SUBSTEP:
  det=1 pair (shipped ON defaults, 1024x25 seed 42): plateau(19-24)
    ms/acc 0.959, whole-run 0.973, late-3 0.941, at demand ON/OFF
    1.088/1.070/1.082;
  det-unset pairs: unthrottled ON plateau 0.980 (demand 1.100), throttled
    default 1.009 (demand 1.104);
  25-iter det-unset WHOLE-RUN ms/acc scatters 0.947-1.056 across arms --
    single det-unset pairs cannot resolve effects below ~+-6% at this
    scale (that scatter IS the pass-22 headline's error bar).
The p22 slab axis 0.833 was predominantly demand draw. VALIDITY
CERTIFICATE: the det=1 OFF arm reproduced p21's det-pair trajectory
exactly (cum evals 88,935 == 88,935) -- the OFF stream is bit-preserved
at production scale on these bytes. det=1 x adopt-ranking interaction,
certified and priced: the canonical two-pass key-ranked adopt runs per
FIRE under det=1, amplifying fire cost (det=1 ON per-slab 1.126 vs its
OFF; det-unset ON 1.01-1.13); G4b certifies repeat-bitwise in-mode.

THROTTLE (crossing-batch; inside ON mode only, no new default-ON
surface): NEWTON_SAP_RUNAHEAD_BATCH (count bound; >=1 absolute, (0,1)
fraction of worlds) + NEWTON_SAP_RUNAHEAD_BATCH_AGE (max-hold march
iterations), defaults 0.5 / 2 (measured below). A world reaching its
boundary HOLDS parked there (no attempts -- the per-boundary march's
parked-world state) until COUNT (pending >= bound), AGE (a non-empty
pending set held `age` iterations -- the counter tracks consecutive
iterations with pending work, so the FIRST lander bounds every holder's
delay), or LIVENESS (nothing else can march) opens the gate. Crossing is
only ever DELAYED, never skipped or reordered; the masked collide still
fires at exactly each world's boundary-entry state (held worlds are
parked AT the boundary it reads -- the anchoring invariant is
structural). The diverged latch-park is ungated (containment visibility
unchanged). _debt_guard_target gains the held-world conjunct (sim_time <
next_time): a parked world completed its boundary and carries no debt;
no-op on unthrottled streams. Cost: two tiny device kernels per march
iteration; both knobs ride both graph cache keys.

FIRE DISTRIBUTION (p23_fire_b1.marchcsv, 1024x8 det unset, unthrottled):
3.19 fires/call wide / 4.55 late (max 14) vs OFF's one collide per call;
crossings arrive as one large window-sync batch plus small straggler
fires (worlds/fire mean 158 wide, 86 late) -- the small fires are the
tax.

TUNING (1024x8 det unset seed 42 vs shared OFF arm; W0-6 = matched wide
window; fires at final telemetry frame):
  arm            W0-6 wall  W0-6 ms/slab  W0-6 ms/acc  late-3 ms/acc  fires
  unthrottled    1.011      1.291         0.977        1.028          2532
  count 32       1.029      1.189         0.995        0.948          1968
  count 128      0.951      1.156         0.927        0.833          1776
  count 512      0.940      1.061         0.923        0.784           950
  count 1024     0.938      1.041         0.939        1.000           562
  0.5 + age 6    1.040      1.142         1.014        0.977          1001
  0.5 + age 2    0.903      1.160         0.886        0.756          1317
COUNT-ONLY FAILS AT THE PLATEAU (25-iter, count 512, pre-age bytes):
plateau slabs 1.264 / ms/acc 1.060 vs OFF -- the liveness rule alone
turns the sub-bound trailing set into a GLOBAL BARRIER (half the fleet
serializes behind the deepest straggler). The right edge (count 1024 =
lockstep generations) confirms the mechanism: late slabs return to OFF's
count (1.010), ms/acc to 1.000, demand tracking OFF to 1e-4. The AGE
rule deletes the barrier (max hold 2 thin iterations) while halving
fires: (0.5, 2) dominates the wide regime (-10% wall vs OFF where
unthrottled was +1%; tax +29% -> +16%/slab; merge preserved, slabs
0.778) and holds the plateau (det-unset 1.009; det=1 0.959). HOLD
LATENCY, NOT FIRE COUNT, IS THE DOMINANT THROTTLE COST. DEFAULTS KEPT:
0.5 / 2.

GATES (all green on final bytes, p23_ artifacts): G1 construct OFF+ON;
G2 flag-equivalence PASS (legacy cells pin runahead 0, bitwise; the
runahead family repeat/graph/conditional arms bitwise at the shipped
throttle defaults; both throttle env vars added to the probe's cleared
set); G3 march-equivalence PASS, iterations [6,25,20,24,19] UNCHANGED;
G4 determinism OFF+ON PASS; G5 containment OFF+ON PASS; G6 err_tol ON
0 violations / 2880, max ratio 0.995, floor 0, dt_run_min 1.64e-3;
G7 rest ON ok; G8 phi0 ON-vs-OFF identical TO THE DIGIT. MIXED-TIME
ORACLE (extended, committed): NEW tier 5 throttle invariance --
window-edge records BITWISE across gate rules {(1,inf),(3,inf),(8,inf),
(8,2)} ("dt" excluded as the already-documented attempt-transient dead
state; ideal_dt compared), crossings at the structural count in every
arm, adopts 16 -> 11/6/8 with an engagement guard (all-equal = vacuous);
tiers 1-4 byte-identical to the pass-22 run (crossings=48, adopts=16,
lead 2.000 ms, |dqd| 3.7e-9).

OPS INCIDENT (recorded because the one-GPU-process rail depends on it):
one 25-iter launch died in the known pre-solver USD-parse startup abort
(p21 class; no SAP frames on stack); the measurement wrapper's exit code
was swallowed by its trailing echo, so the && chain continued and
overlapped a second training process on the GPU -- three runs were
VOIDED (2x wall pollution), deleted, and re-run clean after the wrapper
gained an explicit exit. Every kept run's gpumem trace peaks at the
single-process 20.5 GB (det pairs 9.5 GB); cleanliness is certified per
run, not assumed.

VERDICT: the demand instrument did its job -- it cost nothing, and it
deconfounded the campaign's biggest outstanding number: run-ahead is NOT
a plateau lever (~0 at matched demand); its real, reproducible value is
the wide/flail regime (-10% matched-window wall with the throttle;
det=1 whole-run -2.7%), i.e. early training phases. Landed default OFF
as directed; the flip decision now rests on materially different
(weaker) value than pass 22 advertised.

GRANT ARITHMETIC REFRESH: OFF plateau this draw 12.88 s/iter (0.0609
ms/acc at 211k accepted/iter; p22's 15.09 and p19's 14.21 bracket the
demand-draw spread -- plateau walls move +-8% between same-config runs,
demand-driven). ON at the plateau buys -4% to 0: the 10 s goal is NOT
advanced by the flip (10 s needs ~-22% from the current OFF plateau).
The estimator escalation (single-solve ~1.20 vs current ~2.75
ms/substep arithmetic) remains the only priced lever with >20% headroom;
work-price levers in the tail stay closed (pass 21); overlap is now
measured ~plateau-neutral (this pass). 4k/2k projection: ON is
demand-neutral at the plateau, so 4096 feasibility is unchanged by the
flip (the un-OOM lever remains the triangle-pair cap task-cfg line --
Marco's); ON's early-regime -10% shortens the flail phase at any scale.

PASS-24 RECOMMENDATION: (1) put the RE-SHARPENED consent question
(escalations below) to Marco -- the honest pitch is now "faster early
training, plateau-neutral, semantics certified", not a plateau win;
(2) the ESTIMATOR escalation stands as the only route to 10 s -- nothing
else in-rails is both unmeasured and factor-scale; (3) in-rails live:
backlog item 1 (un-narrowed env-axis launches, mechanical bitwise,
few-percent class); (4) MEASURE (nsys, before building anything) whether
the masked collide's cost scales with the crossing subset or carries a
full-width broadphase floor -- the residual +16%/slab wide-regime tax is
the only overlap cost left, and its scaling law decides whether a
narrower collide is worth building; (5) if Marco flips: ride the
launch-checklist alignment probe + a task-level no-sub-action-cadence-
reader assert.
Provenance: p23_run.sh, p23_sweep.sh, p23_suite25.sh, p23_compare.py,
p23_fire_b1.*, p23_ab_{off,b1,b32,b128,b512,b1024,k5a6,k5a2}.*,
p23_plateau_{off,on,b1,k5a2}.*, p23_det_{off,on}.*, p23_final_chain.sh,
p23_g1..g8 logs/JSONs, p23_oracle.log, p23_oracle_smoke*.log.

## Backlog (ranked for the 10 s goal; teardown of contact_solve
## internals is AUTHORIZED)

1. Remaining un-narrowed env-axis launches outside contact_solve.py
   (full-width consumers listed in agent a06d1420 notes) — few-percent
   class, mechanical, bitwise.
2. STRUCTURAL DISCOVERY (the only route to 10 s): the slab is FLAT
   (no group >21%), GPU-busy 99.9%, dtype-insensitive — so the next
   factor-scale lever is work-count, not work-price. Candidates to
   measure, all in-grant: (a) Newton-iteration/LS-trip budget per
   solve: MEASURED pass 13 (discovery subsection above) — halves run
   at 1.5-1.6 iters/solve vs full 2.4, no warm-start-waste pocket;
   remaining open sub-question is the 1024 plateau mix; (b) launch/
   fusion consolidation: MEASURED pass 14 (discovery subsection above)
   — tiny kernels are 84.3% of flail-slab GPU time, LS trial machinery
   58-63%; pass-15 LANDED candidate 1 (fused armijo LS, entry above:
   -36.5%/-42.8% per-substep); candidate 2 (direction-chain
   consolidation) is CLOSED same pass — tiled search_direction measured
   ~1% at production and was reverted, GEMM-epilogue absorption
   foreclosed at ~1% class (closure paragraph in the pass-15 entry).
   Pass-16 DONE: plateau 18.98 s/iter / 3.76 ms/substep measured;
   next-lever map + pass-17 recommendation (ladder alpha-max fold,
   bitwise pack rewrite, update-eval fusion) in the pass-16 entry.
   Pass-17 DONE: recommendations A+B LANDED default-ON
   (-26.3%/-26.6% per-substep at 1024x8). Pass-18 DONE: plateau
   14.93 s/iter / 2.90 ms/substep measured on the pre-C bytes;
   recommendation C (fused update eval) LANDED default-ON
   (-9.3%/-9.5% per-substep, entry above, projected ~13.5 s/iter);
   fresh post-C map + pass-19 recommendation in the pass-18
   discovery entry. Pass-19 DONE: post-C plateau 14.21 s/iter / 2.75
   ms/substep measured; prep/free-motion + env-list consolidation
   MEASURED SPEED-NEUTRAL and reverted (launch-count law — closure
   entry above); plateau-tail occupancy measured and the overlap
   ceiling priced (entry above). Pass-20 DONE: the straggler regime
   profiled at production scale (entry above) — the two live levers,
   in order:
   (e) FWBD NARROWING: DONE pass 21 (entry above) — LANDED default-ON
   (NEWTON_SAP_NARROW_V3, bitwise class, full gate chain green).
   Kernel-time win large (deep slab -15.2%, narrow floor c0 -19.4%,
   fixed-mix window GPU -9.2%) but the deep tail measured
   DISPATCH-BOUND under graph replay: wall value ~1% at the plateau
   (det=1 matched pair), -3.9..-5.1% per-substep at the wide-heavier
   8-iter window. Work-price levers inside the tail are CLOSED by
   that measurement;
   (c) per-boundary D2H readback chain: DEPRIORITIZED (pass-7
   overlap evidence, pass-18 note); (d) cross-boundary overlap of
   independent worlds' marches: BUILT pass 22 (entry above) — the
   run-ahead single march LANDED DEFAULT OFF (NEWTON_SAP_RUNAHEAD),
   certified both modes (isolation bitwise vs solo runs; ON-vs-OFF
   action-edge positions bitwise). Measured ON: plateau 12.54 vs OFF
   15.09 s/iter (-16.9%, inside the p20 projected band) at FLAT
   per-slab price, slab axis -16.7% (det-unset demand-confounded —
   the unconfounded split is the pass-23 measurement); wide-regime
   catch-up collide fires cost +31%/slab in the matched early window
   (whole-run net -3.4%). Default flip = Marco's consent (escalation
   below); pass-23: demand counter, fire accounting, bounded
   crossing-batch throttle. Pass-23 DONE: all three landed (entry
   above) -- the deconfounded plateau value is -4%..0 at matched demand
   (the -16.9% was demand draw), the throttle (0.5/2) shaves the
   wide-regime tax +29% -> +16%/slab at -10% matched-window wall, and
   overlap is measured ~PLATEAU-NEUTRAL: the slab-deletion route to
   10 s at the plateau is CLOSED by measurement -- the estimator
   escalation is the remaining factor-scale lever.
3. Collision-refresh attack: CLOSED (pass 10, <1%). fp32-Hessian:
   CLOSED (pass 12, neutral). Mixed-precision LS: CLOSED (pass 2).
   Pure fp32 solve: CLOSED. Full/half1 overlap: BLOCKED (pass 4).

## EXPERIMENTAL VALIDITY AUDIT (pass 25) — 2026-08-16; MEASUREMENT +
## SOURCE AUDIT ONLY, ZERO CODE EDITS. Marco's directive: "make sure
## nothing that could compromise the results leaks in ... make sure
## something like [the near-rigid approximation] wasn't removed."

Stack audited: newton-adaptive 3c7e74f1 (march-counter-log), sap_warp
2a119d2 (main), IsaacLab 135480c7dc (develop) — all three verified at
the certified HEADs, worktrees clean, GPU idle (438 MiB, 0 compute
apps) before and after. Campaign ranges: newton-adaptive
9c9dc934..HEAD (38 commits), sap_warp 79e43bd..2a119d2 (11), IsaacLab
82c0679d88..HEAD (3). Every headline number below was re-measured this
session on the current bytes; ledger and commit-message prose was
treated as folklore and re-derived.

### AXIS A — NEAR-RIGID APPROXIMATION: **DRIFTED** (not removed)

WHERE IT LIVES. The SAP regularization is built in sap_warp, not in
newton-adaptive (newton/_src/solvers/sap/__init__.py:14 puts
SAP_WARP_PATH on sys.path). Canonical site
sap_warp/sim/sap_helpers.py:2395-2416, replicated verbatim at
contact_solve.py:947 / :1180 / :1278 / :2760 and the f32 twins:

    R_t     = sigma * W
    R_n     = max( beta^2/(4 pi^2) * W ,  1 / (h * k * (h + tau)) )
              \___ near-rigid clamp ___/  \____ compliant ____/
    vhat_n  = -phi0 / (h + tau)

WAS IT REMOVED OR ALTERED? No. Machine-diff of every contact-law
function body across both campaign ranges: the R arithmetic, the
friction cone, the projection ladder and the PD/limit clamps are
expression-for-expression IDENTICAL to the pre-campaign snapshots. The
only edits are launch-domain remapping (`env, c = wp.tid()` ->
list-indexed `env_idx/env_n`), which cannot perturb an fp value because
every write is env-private. Module constants unchanged across both
repos: beta 1.0, sigma 1e-3, fallback_stiffness 1e10, _SAP_PD_BETA 0.1,
_SAP_LIMIT_BETA 0.1, _SAP_LIMIT_STIFFNESS 1e12, _CONTACT_SOFT_NORM_TOL
1e-7. Sole added constant campaign-wide:
_CONTACT_HESSIAN_PACK_TILE_C = 32 (a GEMM tile width). The step-doubling
estimator and dt controller in newton-adaptive are byte-identical
(_step_error, _error_and_commit, _adapt_dt 163 lines, _seed_dt,
_clamp_dt_to_boundary, _debt_guard).

EFFECTIVE RUNTIME VALUES, dumped from a live production env (NOT from
source) — p25_nearrigid.json, 8 envs, task IsaacContrib-Lift-Spatula-
Trossen-v0, env clean of NEWTON_SAP_* overrides:
  tol 1e-3 | optimality_rel_tol 1e-8 | dt_inner_min 1e-12 |
  max_substeps 256 | line_search armijo_decay | preset approx32 |
  solve_precision fp64 | contact_solve_precision fp64 |
  contact_beta 1.0 | contact_sigma 1e-3 | attempt_consistent_r TRUE |
  contact_k 1250.0 N/m (single value, all contacts) |
  contact_tau_d 0.02 s (per-pair = 2 x the 0.01 cfg fallback) |
  w_eff median 14.917 | max_rigid_contact 16384 (= 2048/world).

THE DRIFT — ACR (NEWTON_SAP_ATTEMPT_CONSISTENT_R, flipped to DEFAULT ON
in pass 13, commit 45095218). The commit is a ONE-LINE default change
(`get(...,"0") == "1"` -> `get(...) != "0"`); the mechanism predates the
campaign. What it does: set_constitutive_dt(D) pre-scales W by
    s = D(D+tau) / (h(h+tau))
before R is built (contact_solve.py:4882-4909), where D = the ATTEMPT dt
and h = that solve's own dt. Consequences, derived from source and
confirmed against the measured constants:
  - rn_hard (near-rigid clamp) scaled by s.
  - rt = sigma*W scaled by s UNCONDITIONALLY — there is no max() to
    absorb it, so it applies at 100% of contacts.
  - rn_soft is NOT scaled (it reads h directly).
  - Full solve: h == D, so s == 1 exactly, bitwise no-op.
  - Half solves: h = D/2, so s = 4(D+tau)/(D+2tau). At the MEASURED
    tau 0.02 and dt 1.3e-3..3.2e-3 this is s = 2.06..2.15.
THE SOURCE COMMENT IS WRONG. solver_sap_adaptive.py:1494 asserts
"committed-step laws unchanged" because the full solve transforms at
s=1. But :1602 sets `self._commit_src = self._scratch_full if
self._mode == "fixed" else self._scratch_double`, and _scratch_double is
produced by half1+half2 (:3217-3220). In adaptive mode the full solve is
the DISCARDED trial; the committed state comes from the halves, which
are exactly the solves scaled by s ~ 2.1. So ACR DOES change the
committed constitutive law, and it changes it in the SOFTER direction.
Note this is adaptive-only by construction: in "fixed" mode
_do_doubling is False and s == 1, so ACR is inert there.

HOW MUCH DOES IT ACTUALLY MOVE THE PHYSICS? Measured, not argued
(p25_phi0_{on,off}.json, this session, current bytes):
  ACR OFF: deepest phi0 -5.396e-5 m, median boundary P5 -2.755e-5
  ACR ON : deepest phi0 -5.584e-5 m, median boundary P5 -2.756e-5
i.e. +3.48% on the deepest sample, +0.036% on P5, identical across
rest/press/swing. The effect is small BECAUSE the scene is mostly not
in the near-rigid branch (Axis B) — s scales rn_hard, which only reaches
rn for ~11% of contacts. RESIDUAL RISK, NAMED: the rt (tangential /
friction) scaling by s ~ 2.1 is unconditional and was NOT measured by
any gate in this campaign, including this pass. phi0 is a normal-
direction statistic and is blind to it. For a MUG-LIFT task, whose
success depends on friction holding the object, a ~2x softer tangential
regularization in every committed step is the single most consequential
unmeasured quantity in the stack. RECOMMENDED (not done — ACR changes
what the estimator measures, which is comparison semantics = MARCO'S):
a slip/grasp-retention A/B on ACR ON vs OFF.

### AXIS B — THE CENIC MECHANISM: **NOT PRESENT IN THIS CONFIGURATION**

The prior session's note ("adaptive fixes penetration via dt^-2
contact-stiffness coupling (SAP near-rigid), not the error metric; our
dt-independent solimp forecloses it by design") was treated as folklore
and re-derived from the current code plus a live probe. Verdict: the
CONCLUSION is right, the stated MECHANISM is wrong, and the real
situation is now measured rather than asserted.

DERIVATION FROM THE CODE. With gamma = R_n^-1 (vhat_n - v_n) and
f = gamma/h, the penetration-proportional force is
f = x / (R_n * h * (h+tau)), so k_eff = 1 / (R_n h (h+tau)).
  - COMPLIANT branch (R_n = rn_soft = 1/(h k (h+tau))):
    k_eff = k exactly. dt-INDEPENDENT. d ln k_eff / d ln h = 0.
  - NEAR-RIGID branch (R_n = rn_hard = beta^2/(4pi^2) W, no h in it):
    k_eff = 4pi^2 / (beta^2 W h (h+tau)),
    d ln k_eff / d ln h = -(1 + h/(h+tau))
    = -2 only when tau << h (or tau ~ h); = -1 when tau >> h.

WHICH BRANCH IS LIVE — MEASURED, p25_nearrigid.json, 154,479 contact
samples over rest/press/swing/flail:
  rn_hard / rn_soft, median            0.1115  (soft wins by ~9x)
  fraction in the near-rigid branch    11.09% overall
                                       14.6% rest / 14.5% press /
                                       14.7% swing / **0.25% flail**
  d ln k_eff / d ln dt, mean           -0.1435 overall
                                       **-0.0031 in flail**
                                       (min -1.294, max 0.0)
So the production scene runs COMPLIANT, not near-rigid, and in the
violent flail regime — the regime that dominates training — it is
compliant at 99.75% of contacts with a dt-exponent of essentially ZERO.

WHY. The near-rigid clamp is only reached when the authored stiffness
exceeds k_eff. sap_warp's own fallback is 1e10 (which WOULD be near-
rigid), but the IsaacLab model authors a real stiffness: NewtonShapeCfg
default ke = 2500 N/m per shape, combined in series by
_sap_combine_stiffness -> contact_k = 1250 N/m, which is what the live
probe reads. A 1250 N/m contact is soft enough that rn_soft dominates.
The exponent is additionally halved even where near-rigid DOES win,
because tau_d = 0.02 s >> dt ~ 1.3e-3 s, putting that minority at
dt^-1, never dt^-2.

WHAT THIS MEANS FOR THE KILLER EXPERIMENT. The paper's mechanism —
shrinking dt automatically stiffens contact as dt^-2 and thereby kills
penetration — is NOT the mechanism operating here. Any adaptive
advantage this stack demonstrates is TRUNCATION-ERROR CONTROL ALONE
(the step-doubling estimator holding local error <= tol), which is a
different and weaker claim than reproducing the PI's result. Two
independent settings would have to change to enter the paper's regime:
raise the authored contact stiffness by ~6-7 orders of magnitude
(task/asset config = Marco's), AND drive tau_d << dt (task config =
Marco's). NOTHING WAS CHANGED. This is a characterization, not a
regression: the campaign did not cause it, and the pre-campaign
snapshot had the same authored constants.

    [CORRECTION, pass 31, in place — the "~6-7 orders of magnitude" in
    the paragraph above is WRONG and is the ORIGIN of the error pass 30
    caught downstream. ORIGINAL CLAIM: entering the near-rigid regime
    needs authored contact stiffness raised ~6-7 orders of magnitude.
    THE ERROR: the distance was reasoned from the ratio of the authored
    k to sap_warp's 1e10 fallback, not measured against the branch
    boundary, which sits where rn_hard = rn_soft, i.e. at
    k_cross = 1/(h (h+tau) rn_hard) — a function of the SUBSTEP, not of
    the fallback. CORRECTED VALUE (pass 30's swept census,
    p30_reg_fixed_s2_seed42.json): the clamp takes the majority of
    contacts at k ~ 2.5e4 at the production substep, i.e. **1.3 decades**
    above the authored 1250, and 100% by k = 1e6. The paragraph's
    CONCLUSION — that this stack runs compliant and that an advantage
    here is truncation-error control, not the paper's dt^-2 coupling —
    is unaffected and confirmed. Only the distance was overstated, by
    four to five decades. Not re-measured in pass 31; corrected against
    pass 30's measurement.]

### AXIS C — FIXED-vs-ADAPTIVE COMPARISON FAIRNESS: **COMPROMISED**

C(i) OPTIMIZATION INHERITANCE — FAIR. All six named optimizations live
in SHARED sap_warp contact-solve code reached by any caller of
SapContactSolve.solve(), gated only by env vars that default ON: fused
armijo ladder (_run_sap_backtracking, contact_solve.py:7826), alpha-max
fold (:7889), per-contact pack + live-k GEMM truncation
(_assemble_contact_hessian_from_terms, :6736), fused update-eval
(_solver_update_active, :6134), narrow-v3 env lists (solve()), narrow-v3
scatter world gate (contact_jacobian.py:2669). A fixed-step SAP arm
would inherit every one. Adaptive-only (API opt-in, no in-repo caller):
run-ahead adopt/anchor, ACR constitutive-dt pinning, env-grid narrowing
budget, shared-assembly reuse, per-world dt arrays. So a wall-clock
comparison is NOT rigged by the kernel work — the timing claims survive
on this axis. Classification of the six: GEMM truncation, per-contact
pack, narrow-v3 and blocked-Cholesky narrowing are bit-identical; fused
ladder, alpha-max fold and fused update-eval are ALGEBRAICALLY exact but
NOT bitwise (they change fp reduction order and the route to an equal
expression), so convergence decisions can flip on ties. Combined with
the determinism default flip (ON->OFF at 4 sites: solver_sap_adaptive.py
:1621, contact_solve.py:5507, contact_jacobian.py:1941,
free_motion.py:1663), NO post-campaign run is bitwise reproducible
against a pre-campaign run, even with ACR forced off. That is an
accepted, already-priced campaign cost, not a new finding.

C(ii) IDENTICAL TASK/SCENE/SEEDS — the config source is fair, the
capacity is NOT. Both arms read the same cfg fields for the contact law
(contact_preset_variant, contact_tau_d, line_search_variant,
max_iterations), so the constitutive parameters are identical by
construction. Two asymmetries, both measured:
  1. CONTACT CAPACITY, 16x. mjwarp_manager.py:313-326 gives the
     ADAPTIVE arm max(128, min(2048, rigid_contact_max // world_count))
     plus a triangle-pair budget; :355-358 gives the FIXED arm
     solver_cfg.sap_max_rigid_contact = 128 verbatim, with no scene
     sizing and no triangle-pair budget. Measured live at 8 envs:
     adaptive jac max_contacts 16384 (2048/world) vs fixed 1024
     (128/world). MEASURED LIVE CONTACT DEMAND PEAKS AT 133 CONTACTS IN
     ONE WORLD during flail (p25_nearrigid.json) — already ABOVE the
     fixed arm's 128 budget, and the manager comment at :309-312 states
     overflow "drops mesh contacts silently". A fixed arm run this way
     would shed contacts in exactly the violent regime the comparison is
     about, and would "fail" for a reason that is not timestepping.
  2. ENV-VAR SURFACE. NEWTON_SAP_PRESET / NEWTON_SAP_LINE_SEARCH /
     NEWTON_SAP_SOLVE_PRECISION are read ONLY on the adaptive branch
     (:345-350). The fixed branch reads the cfg alone. Any of those set
     in the shell silently changes one arm's contact law and not the
     other's.

C(iii) CONTAINMENT — a real asymmetry, measured at zero occurrence.
Containment is adaptive-only and structurally cannot exist on the fixed
arm: _diverged_pending is allocated only under `if cls._adaptive`
(mjwarp_manager.py:431-437), and get_diverged_env_mask returns None
when not adaptive (:664-666). Therefore the `physics_diverged`
termination term is a PERMANENT NO-OP on any fixed arm. Splitting the
rescue into its two parts: the dt SHRINK-RETRY is intrinsic to
adaptivity and IS the thesis — legitimate, keep it. The rest (per-world
failure detection, floor-latch, bitwise state freeze, world isolation,
and the termination that excises the world from the training
distribution) is separable engineering that a fair fixed-step baseline
could also be given. Does it flatter adaptive in practice? Re-measured
this session: sap_containment_probe PASS on current bytes (35 contained
events, failing world 2 latched at boundary 0, all 5 healthy worlds
bitwise identical to control over 30 boundaries, strict mode raises).
But at PRODUCTION scale the term fires ZERO times — grep of the
production run logs returns physics_diverged: 0.0000, and the p14
1024x25 run recorded 0. So the asymmetry is structural and real but has
had no measured effect on the training distribution to date. It must be
either given to the fixed arm or characterized in the paper text; it
must not be left implicit.

C(iv) DEFAULTS — confirmed. Determinism default OFF (production runs
det-unset); run-ahead default OFF and confirmed inert in the plateau
artifacts (p23 telemetry ra_cross=0 ra_fires=0 on the OFF arm). Nothing
default-ON changes batch-visible semantics: the mixed-time oracle's
window-edge records are bitwise across throttle gate rules, and
run-ahead — the only default that would change batch-visible mid-window
state — is OFF.

C(v) **THE FINDING THAT OUTRANKS THE REST: THE FIXED-STEP SAP ARM DOES
NOT RUN, AND NO SAP COMPARISON HAS EVER BEEN RUN.**
  (a) MEASURED: building the task with the fixed arm
      (solver_cfg.adaptive = False, NEWTON_SAP=1) constructs SolverSAP
      successfully and then dies on the FIRST env.step with
      `TypeError: SolverSAP.update_contacts() takes 2 positional
      arguments but 3 were given` (p25_fixedarm.json /
      p25_fixedarm.log). The call site is
      newton_manager.py:2376 `cls._solver.update_contacts(eval_contacts,
      cls._state_0)`; sap_warp/sim/solver_sap.py:1553 defines
      `update_contacts(self, contacts)` and its body is
      `raise NotImplementedError("SolverSAP does not expose
      contact-force writeback yet.")`. So even with the signature fixed
      it would raise. The task registers contact sensors
      (trossen_spatula_lift_env_cfg.py:214/223), so _report_contacts is
      True and this path is unavoidable.
  (b) NOT A CAMPAIGN REGRESSION. The call site dates to IsaacLab
      77a17abf91 (2026-03-02, "Adds newton engine (#4761)") and the
      SolverSAP stub to sap_warp 431adf2 (2026-06-11, initial commit).
      Both far predate the campaign; nothing this campaign did caused
      it.
  (c) NO SAP RUN IS RECORDED. 416/416 dumped params/env.yaml across all
      354 run directories read `backend: mujoco` and `sap_adaptive:
      false`. The comparison script itself —
      trossen_spatula_lift/run_comparison.sh, headed "THE COMPARISON:
      fixed-step vs adaptive" — launches `--solver mujoco` vs
      `--solver mujoco-adaptive`. The killer experiment as actually
      executed is MuJoCo-Warp fixed (2 substeps) vs MuJoCo-Warp
      adaptive (1 substep). It is NOT a SAP experiment.
  (d) THE CAMPAIGN'S OWN SAP RUNS ARE REAL — do not over-read (c). The
      p13..p23 plateau/A-B runs force the SAP backend with
      `NEWTON_SAP=1 NEWTON_SAP_ADAPTIVE=1` env vars while passing
      `--solver mujoco-adaptive`; _resolve_solver_mode
      (mjwarp_manager.py:233-241) applies the env override AFTER
      params/env.yaml is dumped, so those runs are SolverSAPAdaptive
      despite dumping `backend: mujoco`. Verified independently: the p23
      plateau telemetry carries cumulative_accepted / ra_cross /
      ra_fires, which only SolverSAPAdaptive emits. The yaml is
      misleading for every SAP run in this campaign; the env vars are
      the only record.
  (e) THE DOCUMENTED SAP-ADAPTIVE LAUNCH PATH IS BROKEN.
      physics_presets.py:26 maps `sap-adaptive` to
      {backend: sap, adaptive: False, sap_adaptive: True}, but
      _validate_solver_substeps (trossen_spatula_lift_env_cfg.py:474)
      tests `not solver_cfg.adaptive and num_substeps < 2` — it reads
      the MuJoCo adaptivity latch, not sap_adaptive. So
      `--solver sap-adaptive physics=newton_mjwarp_adaptive` raises
      ValueError. The campaign's env-var workaround exists because of
      this.
NET: the comparison the experiment rests on has never been run on SAP,
and cannot be until (a) and (e) are fixed and the C(ii)-1 capacity
asymmetry is closed. All three are IsaacLab/task-side = MARCO'S. This
does not invalidate any wall-clock number in this ledger — those are
SAP-adaptive self-comparisons and stand — but the SUCCESS/FAILURE claim
has no SAP evidence behind it today.

### AXIS D — INVARIANT DRIFT ACROSS THE CAMPAIGN: **INTACT**

Every invariant re-measured this session on the current bytes and
compared to the EARLIEST equivalent artifact. Provenance for the
pre-campaign comparison point: scratchpad phi0_{off,on}.json, written
2026-08-15 12:07, i.e. BEFORE the snapshot commit 9c9dc934 (13:23) —
a genuine pre-campaign baseline, not a reconstruction.

  PENETRATION (phi0 rig, 8 envs, rest/press/swing, all phases):
    pre-campaign ACR-OFF 12:07 : deepest -5.396e-5, P5 -2.755e-5
    p25 ACR-OFF (this session) : deepest -5.396e-5, P5 -2.755e-5
    pre-campaign ACR-ON  12:07 : deepest -5.584e-5, P5 -2.756e-5
    p25 ACR-ON  (this session) : deepest -5.584e-5, P5 -2.756e-5
  IDENTICAL TO THE DIGIT across 20 hours, 38 newton-adaptive commits
  and 11 sap_warp commits. The full 28-artifact series (phi0_* through
  p23_g8_*) shows exactly two values, selected by ACR and nothing else.
  DRIFT: ZERO.

  ACCEPTED-STEP ERROR / TOL (p25_err_tol.json, 2880 samples):
    violations 0/2880; max accepted err/tol 0.7135;
    per-phase max ratio rest 0.0179 / press 0.6137 / swing 0.7135 —
    IDENTICAL to every recorded pass from cert_g5 (2026-08-14 17:30)
    through p23. Only "flail" varies across draws (0.292 this pass;
    0.50-0.99 historically), which is the expected det-unset chaotic
    phase, not drift.
    Rails confirmed live: tol 1e-3, dt_min 1e-12, inner_rel_tol 1e-8,
    solve fp64, contact_solve fp64.

  HEALTHY dt BAND: dt_run_min 3.166e-3; samples below 1e-4 = 0; floor
  visits 0; saturation depth 0.0; capped boundaries 0; unfinished
  worlds 0. Marco's dt >= 1e-4 criterion met with >30x margin.

  INNER NEWTON CONVERGENCE TO 1e-8 — the certificate, not the claim.
  The target is enforced, not advisory: solver_sap_adaptive.py:3346
  raises "SolverSAPAdaptive inner SAP solve failed to converge to
  optimality_rel_tol=1.000e-08", and an unconverged attempt is forced
  to the divergence sentinel so it can never be committed. The
  containment probe re-demonstrated the live raise this session
  (PASS[d], strict mode, boundary 0, message quotes 1.000e-08). Every
  probe run this pass completed without raising, so committed steps
  genuinely converged. This is a real certificate because the failure
  path is proven live in the same session.

  CONTACT-SET INTEGRITY: contact counts censused live — max 54/world in
  rest/press/swing, 133/world in flail; adaptive buffer 2048/world, so
  headroom 15x and no truncation on the adaptive arm. march-equivalence
  fingerprint [6,25,20,24,19] unchanged since pass 13.

### AXIS E — THE 7-10 s QUESTION, AND THE STANDING RED LINE

WHERE THE PLATEAU ACTUALLY IS. Recomputed this pass directly from the
run logs (not from ledger prose), plateau = mean of iterations 19-24 at
1024 envs:
    p23_plateau_off  12.88 s/iter  (min 12.02, max 13.37)
    p19_1024x25      14.21 s/iter
    p22 (recorded)   15.09 s/iter
    p14_1024x25      35.35 s/iter
Same-config draws move +-8% because demand is draw-dependent, so the
honest statement is: THE CURRENT DEFAULT STACK PLATEAUS AT 12.9-15.1
s/iter @1024, best estimate ~13-14. Demand-normalized, run-ahead ON
buys -4% to 0 at the plateau (pass 23), so the default-OFF number is
the operative one.

WHAT REMAINS TO 7-10 s. From 12.88 (best draw): 10 s needs -22%, 7 s
needs -46%. From 15.09 (worst draw): -34% and -54%. The measured
inventory of remaining levers:
  - Backlog item 1 (un-narrowed env-axis launches): few-percent,
    bitwise, in-rails. Real but nowhere near sufficient.
  - Overlap / run-ahead: measured ~plateau-neutral (pass 23). CLOSED as
    a plateau lever.
  - Work-price inside the tail: CLOSED by measurement (pass 21, the
    deep tail is dispatch-bound under graph replay).
  - Estimator structure (3-solve step doubling -> single solve): the
    ONLY priced lever with >20% headroom (~1.20 vs ~2.75 ms/substep).
CONCLUSION: 7-10 s/iter is NOT REACHABLE IN-RAILS. Every route with
enough headroom runs through the step-doubling estimator.

THE RED LINE (standing rule for every future pass). The step-doubling
estimator, tol 1e-3, optimality_rel_tol 1e-8, dt_inner_min 1e-12, the
contact law (R, k, tau_d, beta, sigma) and the fixed-vs-adaptive
comparison semantics are the PHYSICS BEING DEMONSTRATED, not
optimization surface. A wall-clock win purchased from any of them is
not a win — it deletes the result it was meant to enable. If a future
pass finds that only an estimator/tolerance/contact-law change reaches
the target, the correct output is an escalation to Marco, not a
landing. The experiment's integrity outranks the wall-clock goal, and
7-10 s is a goal, not a rail. No pass may treat the red-line list as a
lever without Marco's explicit, in-channel consent.

### FINDINGS, CLASSIFIED (loudest first)

F1 COMPROMISED / MARCO'S — the fixed-step SAP baseline does not run
   (measured TypeError -> NotImplementedError on step 1) and no SAP
   comparison has ever been executed (416/416 runs backend mujoco;
   run_comparison.sh is --solver mujoco vs mujoco-adaptive). The
   "adaptive trains / fixed fails" claim has no SAP evidence. Pre-
   existing, not campaign-caused. IsaacLab + sap_warp side.
F2 COMPROMISED / MARCO'S — 16x contact-capacity asymmetry between arms
   (128 vs 2048 per world) with measured live demand of 133/world
   already exceeding the fixed arm's budget, and silent contact
   dropping on overflow. Would make a fixed arm fail for a non-physics
   reason. mjwarp_manager.py:313-326 vs :355-358.
F3 DRIFTED / MARCO'S — ACR default-ON scales the near-rigid clamp AND
   the tangential regularization by s ~ 2.1 in the COMMITTED half-
   solves (the in-source "committed-step laws unchanged" comment is
   false; _commit_src is _scratch_double). Measured normal-penetration
   impact is small (+3.5% deepest, +0.04% P5); the tangential half is
   UNMEASURED and matters most for a grasping task. Adaptive-only by
   construction (s == 1 in fixed mode).
F4 CHARACTERIZATION — the stack is NOT in the near-rigid regime the
   CENIC mechanism needs (89% compliant overall, 99.75% in flail;
   dt-exponent -0.003 in flail). Any adaptive advantage demonstrated
   here is truncation-error control, not the paper's dt^-2 stiffness
   coupling. Config-level, pre-campaign, unchanged by the campaign.
F5 STRUCTURAL ASYMMETRY — containment / physics_diverged is adaptive-
   only and permanently dead on any fixed arm; measured firing rate at
   production is ZERO, so no measured distributional effect yet. Must
   be given to the fixed arm or stated in the paper text.
F6 HYGIENE — params/env.yaml records backend: mujoco for every SAP run
   in this campaign because the NEWTON_SAP env override is applied
   after the dump. Every SAP result's provenance lives only in the
   launch scripts. Recommend recording resolved solver identity in the
   run artifacts.
F7 HYGIENE — sap_warp is joined to newton-adaptive by SAP_WARP_PATH,
   not a submodule or pin. Any sap_warp commit silently changes the
   physics under an unchanged newton-adaptive HEAD. For an experiment
   whose validity is the contact law, that coupling should be pinned.
CLEAN — Axis D. No campaign optimization altered any physical
   invariant. Penetration, err/tol, dt band, convergence certificate,
   contact-set integrity and containment all re-measured this session
   and identical to the earliest available baselines.

### RESIDUAL RISK — what could NOT be established, and why

R1 The TANGENTIAL consequence of the ACR s-scaling. rt = sigma*W is
   scaled at 100% of contacts and no gate in this campaign measures a
   friction-direction observable; phi0 is normal-only. Unquantified.
R2 The magnitude of the fp divergence from the three approximate
   fusions (fused ladder, alpha-max fold, fused update-eval). They are
   algebraically exact but change reduction order, so convergence
   decisions can flip on ties. Bounded only by the flag-equivalence
   gates' pass/fail, never by a distributional measurement.
R3 Whether a fixed-step SAP arm, once runnable, reproduces the
   contact set the adaptive arm sees. The arms use DIFFERENT contact
   sources (fixed consumes the manager CollisionPipeline,
   _needs_collision_pipeline True; adaptive owns its own pipeline).
   Unmeasurable until F1 is fixed.
R4 Contact-sensor liveness. Both arms report identically zero force on
   pad_handle_contact and arm_body_contact across 280 scripted steps
   (p25_sensors_{sap,mjc}.json), and SolverSAPAdaptive.update_contacts
   is a documented no-op. This is SYMMETRIC across arms so it is not a
   fairness finding, but I could not distinguish "sensors are dead"
   from "my scripted actions never satisfied the prim-pair filters".
   The active reward terms do not consume them.
R5 narrow-v3's "dead state is never read" contract is enforced by
   convention, not by code. Not exhaustively verified.
R6 No pre-campaign SAP TRAINING baseline exists at all, so the effect
   of the campaign on learning outcomes (as opposed to on physical
   invariants, which is measured and null) is UNQUANTIFIED. Stated as
   unquantified rather than assumed benign.

Provenance (all this pass, p25_ prefix, no p13-p24 artifact
overwritten): p25_nearrigid_probe.py + p25_nearrigid.{json,log};
p25_fixedarm_probe.py + p25_fixedarm.{json,log};
p25_sensor_liveness_probe.py + p25_sensors_{sap,mjc}.{json,log};
p25_err_tol.{json,log}; p25_phi0_{on,off}.{json,log};
p25_containment.log. Pre-campaign baseline: phi0_{off,on}.json
(2026-08-15 12:07). Plateau recomputation: p23_plateau_off.log,
p19_1024x25.log, p14_1024x25.log.

## PASS 26 — ACR TANGENTIAL VERDICT + FIXED-STEP SAP ARM ENABLED
## 2026-08-16. Closes audit R1 (measurement) and F1/F2/F5-adjacent
## (code, in-grant only). Task/scene side untouched.

Stack: newton-adaptive 6e9cb932 (march-counter-log), sap_warp 2a119d2
(main), IsaacLab 135480c7dc (develop), all three clean at the certified
HEADs and GPU idle (438 MiB, 0 compute apps) before the first launch.
Every number below was measured this session on the bytes in the tree;
the pass-25 audit's own wording was treated as folklore and re-derived
from source before anything was built on it.

### PART 1 — WHAT ACR DOES TO FRICTION. Answer: it doubles the
### tangential regularization exactly as suspected, and that changes
### CREEP but NOT HOLDING CAPACITY.

MECHANISM, re-derived from source (not from the audit paragraph).
`_scale_w_eff_attempt_consistent` (contact_solve.py:4881-4908, launched
at :9198 before prepare, result rebound onto a shallow copy of the
jacobian result at :9213) multiplies the contact Delassus argument W by
s = D(D+tau)/(h(h+tau)). Downstream, in sap_helpers.py:2395-2416 and
its three replicas:
    rn_hard = beta^2/(4 pi^2) * W      -> scaled by s
    rn_soft = 1/(h k (h + tau))        -> reads h, NOT scaled
    rn      = max(rn_hard, rn_soft)
    rt      = sigma * W                -> scaled by s, unconditionally
Because rn_soft(h) already equals s*rn_soft(D) identically, rn as a
whole equals s*rn(D) and the NORMAL law is genuinely pinned to the
attempt dt. The tangential law is not pinned to anything: rt/rn changes
by s wherever the compliant branch wins, which is where 89% of contacts
sit. That ratio is not cosmetic — the projection's soft cone is
    mu_tilde = mu*sqrt(rt/rn),   mu_hat = mu*rt/rn
(sap_helpers.py:2436-2437) and the stick test is `yr <= mu*y.z`, i.e.
|v_t| <= mu_hat*(vhat_n - v_n). mu_hat IS the stick/slide threshold in
velocity space, so ACR rescales the stick region itself.

MEASURED, p26_acr_{on,off}.json, 8 envs, rest/press/swing/flail,
150,647 (ON) / 151,817 (OFF) live contact samples. s is measured per
contact as rt/(sigma * W_unscaled) — the scale actually applied, not
the formula re-evaluated:
  s, ACR OFF : exactly 1.000000 at all 151,817 samples.
  s, ACR ON  : median 2.344828, mean 2.2956, range [2.00509, 2.344828].
  Every ON sample lies strictly inside (2,4) — the interval implied by
  h = D/2 with tau > 0, which is an independent bound the code could
  have violated and did not, 150,647/150,647.
  THE AUDIT UNDERSTATED IT. The production accepted dt in
  rest/press/swing is the FULL 1/120 s boundary (dt_attempt median
  8.33333e-3; dt_solve median 4.16667e-3 = exactly D/2), giving
  s = 4(D+tau)/(D+2tau) = 2.3448276 at the measured tau 0.02 — the
  measured median to 7 significant figures. s falls toward 2.0 only
  where flail shrinks dt (min 2.00509 <-> D ~ 1.02e-4 s).
  R_n : median 7.944827 in BOTH arms, ratio 1.000000. The compliant
        branch is untouched by ACR, confirming the audit's normal-
        direction result from the R arrays themselves rather than phi0.
  R_t : 0.0149171 (OFF) vs 0.0349781 (ON), ratio 2.3448. The OFF median
        is sigma*W = 1e-3 * 14.9171.
  mu_hat : 0.00179386 (OFF) vs 0.00420628 (ON), ratio 2.3448. Measured
        mu is 1.0, so the regularized stick region is 0.18% (OFF) /
        0.42% (ON) of the Coulomb cone: the regularization is nowhere
        near binding in this scene, which is why doubling it is cheap.
  CONTACT REGIME CENSUS: across rest/press/swing the stiction/sliding
  split is IDENTICAL between arms — 2760/240, 4129/371, 5520/480 —
  i.e. not one of 13,500 forceful contact samples changes regime. Only
  flail differs (sliding fraction 0.33115 ON vs 0.33082 OFF), and there
  the two trajectories have already diverged, so that is chaos, not
  signal. Quasi-static slip speed |v_t| is 3.45e-7 (ON) vs 3.24e-7
  (OFF) m/s at rest — five orders below anything the task can see.

CONTROLLED LOAD RIG, p26_slip_{on,off}.json. Gravity tilted by theta at
constant 1 g magnitude, arm commanded still, mug on the tabletop,
per-env drift accumulated with the accumulator frozen at termination so
an auto-reset teleport cannot enter the measurement. Measured mu 1.0
(friction angle 45 deg). The design is a separation, not a level: below
the friction angle a RIGID Coulomb contact cannot slide at all, so all
observed drift is regularization creep and R_t governs it; above the
friction angle the object slides against mu*N, which R_t does not enter,
so the arms must agree there.
  theta = 0    : drift 0 in both arms.
  theta = 20   (well below 45): ON 7.114e-5 m / 2 s = 3.557e-5 m/s;
                OFF 1.050e-5 m / 2 s = 5.251e-6 m/s; ratio 6.77, and
                ON > OFF in 8/8 envs. The no-friction bound at this
                angle is 6.71 m, so both arms sit ~1e5 below it:
                FRICTION HOLDS IN BOTH; the difference is pure creep.
  theta = 55/70 (above 45): ratios 0.944 / 1.010, ON > OFF in 3/8 and
                4/8 envs — a coin flip. THE COULOMB LIMIT IS UNCHANGED.
  theta = 35   is NOT interpretable and is not used: the mug topples
                rather than slides (both arms drift ~0.3 m, envs hit
                object_off_table). Toppling is a geometry threshold,
                not a friction one.
  THE RIG'S FIRST BUILD WAS VOID AND ITS OWN TRIPWIRE CAUGHT IT:
  SapFreeMotion snapshots gravity into its own f64 array at
  construction (free_motion.py:1890, _make_model_array_f64 copies
  through numpy), so Model.set_gravity never reaches the SAP kernels and
  drift was 0 at every angle including 60 deg. The live array must be
  written directly. Any future pass changing gravity under SAP must do
  the same or it will measure nothing and not notice.

VERDICT (this is the deliverable). ACR default-ON IS materially
changing the committed tangential law — 2.34x on R_t and on the soft-
cone parameter mu_hat, at 100% of contacts, on every committed step —
and it DOES change grasp-relevant behaviour, but only in one of the two
channels that matter:
  * HOLDING CAPACITY: UNCHANGED. Above the friction angle the arms
    slide identically (0.94-1.01, 3/8 and 4/8 envs). R_t does not enter
    the Coulomb limit, and R_n is unchanged in the compliant branch
    where 89% of contacts live. A grasp that holds with ACR OFF holds
    with ACR ON. The mug-lift success/failure signal is NOT compromised.
  * SPURIOUS CREEP: ~6.8x LARGER. Under a sustained sub-Coulomb
    tangential load a held object oozes at 36 um/s (ON) vs 5.3 um/s
    (OFF) at a 0.34 g tangential load. Over a 5 s episode that is
    ~180 um vs ~26 um. The task's lift threshold is 8 cm and its
    tightest reward std is 5 cm, so the difference is ~3 orders of
    magnitude below the decision scale.
  * IN THE PRODUCTION PHASES IT IS INVISIBLE: identical stick/slide
    census over 13,500 forceful samples, |v_t| ~3e-7 m/s.
So: "is what we are running right now compromised?" — NO for grasp
success/failure and for every metric this campaign reports; YES for
sub-millimetre positional fidelity of a held object. ACR's DEFAULT WAS
NOT CHANGED and must not be changed on this evidence alone: it alters
what the estimator measures, which is comparison semantics = Marco's.

RESIDUAL RISK, NAMED:
 R1a The creep ratio (6.77) is NOT the R_t ratio (2.34). The creep
     channel is super-linear in R_t here and this pass did not isolate
     why; candidates are the mode-mix shift (sliding fraction 0.1028 ON
     vs 0.1183 OFF at 20 deg) and creep feeding back into the contact
     geometry. DIRECTION and ORDER OF MAGNITUDE are measured; the
     exponent is not.
 R1b The rig loads the MUG/TABLE pair, not the PAD/MUG pair. s is
     applied identically at every contact so the mechanism transfers,
     but W and mu at the pads differ, so absolute pad creep is not
     measured. A scripted grasp was not attempted; pass 25 could not
     make the pads report contact either.
 R1c One draw per arm, 8 envs, 2 s. The 8/8 vs 3/8 split is the
     evidence; there is no confidence interval behind it.
 R1d Only the compliant branch was loaded. In the near-rigid 11% both
     rt and rn scale by s, so rt/rn is ACR-invariant there — predicted
     null, not measured.

### PART 2 — THE FIXED-STEP SAP ARM NOW RUNS, AND IS FAIRER

2a WRITEBACK. The brief's premise was false and this was verified by
reading both bodies, not inferred: SolverSAPAdaptive.update_contacts
(solver_sap_adaptive.py:2182-2185) is a documented NO-OP that ignores
both arguments; SolverSAP.update_contacts raised NotImplementedError
behind a 1-argument signature the frontend calls with 2. There is no
shared sap_warp force-assembly helper either. "Mirror the twin exactly"
therefore resolves to a signature-matched no-op, and that is what
landed: sap_warp solver_sap.py `update_contacts(self, contacts,
state=None) -> None`. Both arms now report the SAME quantity by the
SAME arithmetic. MEASURED on final bytes, 360 scripted steps, both
arms: peak |force| 0.0 and 0 non-zero steps on pad_handle_contact and
arm_body_contact. Nothing in this task consumes those sensors (rewards
are reach/lift/goal/fine/action_rate/joint_vel; terminations are
time_out/dropping/off_table/speeding/robot_abnormal/physics_diverged),
so the dead sensors are symmetric and currently inert — they become a
live fairness defect the moment a contact-derived term is added.

2b CAPACITY. mjwarp_manager now derives the per-world SAP contact
budget ONCE, above the arm split, so both arms get
max(cfg, min(2048, rigid_contact_max // world_count)). MEASURED at 8
envs on final bytes: per-world budget 2048 on BOTH arms (fixed was
128). Live demand peaked at 144 contacts in one world on the fixed arm
during flail — ABOVE the old 128 budget, so the old fixed arm would
have silently shed contacts in exactly the violent regime the
comparison is about. Truncated contacts 0 on both arms; headroom 14.2x
(fixed) / 16.5x (adaptive). TRIANGLE PAIRS NEEDED NO CHANGE: the fixed
arm owns no pipeline, it consumes the manager's CollisionPipeline built
from the task's authored NewtonCollisionPipelineCfg (rigid_contact_max
8e6, max_triangle_pairs 192e6) at newton_manager.py:1919, so its
triangle-pair budget was already scene-sized.

2c ENV-VAR SURFACE. NEWTON_SAP_PRESET / NEWTON_SAP_LINE_SEARCH /
NEWTON_SAP_SOLVE_PRECISION are now resolved once above the split and
reach both constructors. SolverSAP takes four precision knobs where
SolverSAPAdaptive takes one string, so the expansion is applied
identically (fp64 passes NO overrides, preserving the default
construction path exactly). MEASURED, both arms, final bytes:
preset approx32, line_search armijo_decay, contact_solve_precision
fp64, beta 1.0, sigma 1e-3, tau_d 0.02, max_iterations 30 — identical.

2d IT RUNS. Invocation derived from _resolve_solver_mode, not from the
docs: NEWTON_SAP=1 with NEWTON_SAP_ADAPTIVE unset and the task's own
preset (backend -> sap via the env override; adaptive stays False
because sap_adaptive is False), num_substeps left at the task default 2
because the task's substep latch forbids 1 on a non-adaptive solver.
  * Probe: SolverSAP constructed, 360 scripted steps across
    rest/press/swing/flail, 0 non-finite steps, rewards finite, contact
    sensors reported. (p26_arm_fixed.json)
  * Training: 3 iterations at 64 envs, seed 42, through the normal
    `isaaclab.sh train` entrypoint. Completed without raising; reward,
    curriculum and all six termination counters logged.
    (p26_train_fixed.log) SOLVER IDENTITY PROVEN FROM THE RUN ITSELF,
    not assumed: params/env.yaml still records backend mujoco (the
    known F6 hygiene defect), but _supports_cuda_graph_capture
    (mjwarp_manager.py:697-713) returns False for a NON-adaptive solver
    ONLY when cls._sap is True, and the fixed run's log carries that
    eager-execution warning under --solver mujoco. So the arm was SAP.
  * CHARACTERIZATION ONLY — 8 envs, same script, same machine,
    sequential, ONE DRAW. THIS IS NOT THE KILLER COMPARISON AND MUST
    NOT BE QUOTED AS ONE:
      fixed     2 substeps/boundary          30.39 ms/step
      adaptive  ~1.26 substeps/world-boundary 26.11 ms/step
      (adaptive _cum_accepted 14517 over 360 x 4 x 8 = 11520
       world-boundaries)
      phi0 rest/press/swing: fixed -5.396e-5, adaptive -5.584e-5
      phi0 flail:            fixed -1.824e-2, adaptive -8.919e-3
    The fixed arm's quasi-static phi0 is IDENTICAL to the ACR-OFF
    value, because the fixed arm has s == 1 by construction (no
    doubling). That is a third independent confirmation of the ACR
    mechanism, from a solver that never runs the ACR code path.

### ADAPTIVE ARM: BYTE-UNCHANGED. The campaign record is intact.

By construction: the SolverSAPAdaptive constructor receives
argument-for-argument identical values — the hoisted expressions are
character-identical to the ones they replaced and read the same inputs.
sap_warp's only changed method is SolverSAP.update_contacts, which is
never reached on the adaptive path (the manager calls SolverSAPAdaptive's
own no-op). No kernel, no constant, no launch, no tolerance changed.
By measurement: gate results below.

### NEW FINDING F8 — THE TWO ARMS DO NOT SOLVE TO THE SAME TOLERANCE

Measured live off both solver objects on final bytes
(p26_arm_{fixed,adaptive}.json):
    fixed    optimality_rel_tol 1e-06   cost_abs_tol 1e-30  cost_rel_tol 1e-15
    adaptive optimality_rel_tol 1e-08   cost_abs_tol 0.0    cost_rel_tol 0.0
mjwarp_manager passes neither tolerance to SolverSAP, so the fixed arm
takes SolverSAP's own ctor defaults, while SolverSAPAdaptive pins 1e-8
and disables the cost early-exit (solver_sap_adaptive.py:1369-1401).
The fixed arm therefore accepts a contact-solve residual 100x looser
and may stop on a cost plateau where the adaptive arm structurally
cannot. This is the same class of defect as F2 and it survives this
pass's fixes. NOT TOUCHED: optimality_rel_tol is named in the pass-25
VALIDITY RED LINE as physics-being-demonstrated, so changing it — even
on an arm with no campaign record — is Marco's call, not a loop's. The
fix is one kwarg triple at the SolverSAP construction site.

### GATES (full chain, final bytes, adaptive arm)

All eight PASS on the final bytes, run strictly sequentially with the
chain aborting on any non-zero exit. The whole chain took 100 s wall
because the Warp kernel cache was warm (every module logs "(cached)");
that is not a skipped chain — each log carries its own PASS lines.
 G1 construct  PASS (SAP-NEWTON15-CONSTRUCT). Also re-confirms ACR
    defaults ON with the constitutive dt wired into the contact solve.
 G2 flag-equivalence + smoke  PASS. Every bitwise arm — acr, fusedls,
    fusedam, fusedup, narrowv3, runahead, each in graph and conditional
    form — bitwise identical to its reference over 6 boundaries.
    Equivalence iterations per boundary reference [11,4,4,3,3,2] ==
    boundary [11,4,4,3,3,2].
 G3 march-equivalence  PASS, fingerprint [6, 25, 20, 24, 19] —
    UNCHANGED since pass 13, which was the stop condition. compact,
    conditional and compact+conditional all bitwise identical over 5
    boundaries; all five guards (mjw rows injected, force-capable
    contacts, rejection exercised, per-world dt diverged, compaction
    tail engaged) reported ok.
 G4 determinism  PASS, bitwise over 20 steps at 256 envs, seed 7, det=1
    on cum / fail_step / joint_q / joint_qd / body_q / body_qd.
 G5 containment  PASS[b] failing world 2 latched at boundary 0 with
    state bitwise-frozen and clock pinned through boundary 29;
    PASS[c] all 5 healthy worlds bitwise identical to control over 30
    boundaries across 10 fields; PASS[d] strict mode raised at boundary
    0 quoting optimality_rel_tol=1.000e-08.
 G6 err_tol  0 violations / 2880 samples; max accepted err/tol 0.7135;
    per-phase rest 0.0179 / press 0.6137 / swing 0.7135 — identical to
    every recorded pass from cert_g5 onward. floor samples 0, capped
    boundaries 0, unfinished worlds 0, dt samples below 1e-4 = 0,
    dt_run_min 1.4649e-3 (>14x Marco's 1e-4 criterion). Rails confirmed
    live: tol 1e-3, dt_min 1e-12, inner_rel_tol 1e-8, solve fp64,
    contact_solve fp64.
 G7 rest smoke  PASS, z in [0.0198, 0.0210], 0 early terminations.
 G8 phi0  deepest -5.584e-5 and median boundary P5 -2.756e-5 in every
    phase, with narrow-v3 ON and OFF identical to each other. THIS IS
    THE BYTE-UNCHANGED PROOF: those are the pass-25 and pre-campaign
    ACR-ON values to the digit (-5.584e-5 / -2.756e-5). Penetration,
    the march fingerprint, determinism and err/tol are all exactly
    where the campaign record left them.

### WHAT STILL NEEDS MARCO

 M1 F8: align the fixed arm's inner-solve tolerances with the adaptive
    arm's (optimality_rel_tol 1e-8, cost tolerances 0.0). Red-line item;
    one kwarg triple. Until then the fixed arm is 100x looser and any
    "fixed fails" result is partly attributable to that, not to
    timestepping.
 M2 A REAL contact-force writeback. The recipe is known
    (Contacts.force[g] = R_WC[env,slot] @ gamma[env,slot] / dt for each
    global row with contact_input_env[g] >= 0) but it is NOT a
    mechanical mirror: the fixed arm's Contacts comes from the manager
    pipeline with valid geometry, while the adaptive arm's manager
    Contacts has no geometry writer at all
    (_needs_collision_pipeline False), so the arms would need different
    kernels to report the same number; and on the adaptive arm the
    committed impulse is not identifiable from last_contact_solve_result
    (that is whichever solve ran last, possibly a discarded trial).
    Making it identifiable is new plumbing in the estimator's
    committed-attempt bookkeeping = red line.
 M3 ACR default: unchanged, and this pass recommends leaving it. The
    measured cost is sub-millimetre creep on a task whose decision
    scale is centimetres; the measured benefit is a committed
    constitutive law consistent with the attempt dt. If the paper's
    claim ever depends on held-object positional fidelity, revisit.
 M4 The task-side items from pass 25 are untouched and still block a
    real comparison: the sap-adaptive validation latch
    (trossen_spatula_lift_env_cfg.py:474), containment/physics_diverged
    being adaptive-only (F5), and F6 provenance in params/env.yaml.

Provenance (all p26_ prefix, no p13-p25 artifact overwritten):
p26_acr_friction_probe.py + p26_acr_{on,off}.{json,log};
p26_slip_probe.py + p26_slip_{on,off}.{json,log};
p26_fixedarm_probe.py + p26_arm_{fixed,adaptive}.{json,log};
p26_train_{fixed,adaptive}.log; p26_acr_run.sh, p26_acr_run2.sh,
p26_run3.sh, p26_run4.sh, p26_gates.sh, p26_acr_chain.log;
p26_g{1..8}_*.{json,log}, p26_gate_progress.txt.

## PASS 27 — THE COMPARISON-FAIRNESS MATRIX + CONTACT-SENSOR VERDICT
## 2026-08-16. Sweeps systematically for the F8 class ("the two arms differ
## for a reason that is not timestepping"), which pass 26 found by accident.
## MEASUREMENT + SOURCE AUDIT ONLY. ZERO CODE EDITS in all three repos.

Stack: newton-adaptive c5502d33 (march-counter-log), sap_warp 345c9fe
(main), IsaacLab 13e049ead1 (develop) — all three verified at the
certified HEADs, worktrees clean, GPU idle (438 MiB, 0 compute apps)
before the first launch and after the last. Every value below was dumped
from LIVE CONSTRUCTED objects at 8 envs on task IsaacContrib-Lift-
Spatula-Trossen-v0; the pass-25/26 ledger prose was treated as folklore
and re-derived from source before anything was built on it.

METHOD (why this pass can claim completeness where pass 26 could not).
Named reads only cover what the author already suspects, so the probe
HARVESTS GENERICALLY: it walks the solver/manager/env object graph and
records every reachable scalar attribute and every warp-array shape, then
diffs the arms row by row. 630 rows on the fixed arm, 803 on the
adaptive, 312 on MuJoCo. An asymmetry nobody thought to name still shows
up as a differing row. Three arms were built — fixed SAP, adaptive SAP,
and the MuJoCo backend every recorded training run actually used —
sequentially, one GPU process at a time.

### THE HEADLINE: THE SAP SOLVE STACKS ARE ALREADY SYMMETRIC

Comparing like for like — the fixed arm's SolverSAP against the adaptive
arm's INNER SolverSAP, which is the same object in the same role:

  SolverSAP vs inner SolverSAP    3 differing rows out of ~60
  contact_solve                   4 differing rows out of 197
  contact_jacobian                0 differing rows out of 67
  free_motion                     0 differing rows out of 75

The 3 solver rows are exactly F8 (below). The 4 contact_solve rows are
exactly the ACR attempt-consistent-R buffers (_constitutive_dt,
_w_eff_att, _a_inv_pd_att, _a_inv_limit_att), allocated on the adaptive
arm and None on the fixed arm — which is the estimator semantics, not an
engineering defect. EVERYTHING ELSE IN THE CONTACT LAW AND THE SOLVE
STACK IS IDENTICAL, measured off the objects: max_iterations 30,
optimality_abs_tol 1e-14, line_search_max_iterations 40, armijo_c, rho,
line_search_variant armijo_decay, contact_beta 1.0, contact_sigma 1e-3,
contact_tau_d, fallback_stiffness, fallback_mu, diag_shift,
preset approx32 and all four precision knobs, contact_weight_mode,
contact_point_mode, position_integration, use_f64_boundary_pose,
static_substep False (8/6), graph_conditional True, per-world contact
budget 2048. Pass 26's capacity and env-var fixes hold on final bytes.

### THE MATRIX

Classification: (i) LEGITIMATE = the difference IS the thesis;
(ii) MUST-FIX-IN-GRANT = engineering asymmetry, no timestepping content;
(iii) MARCO'S = task/scene/tolerance/estimator/comparison semantics;
(iv) BENIGN = differs, PROVED it cannot affect behaviour;
(v) UNKNOWN = could not resolve, reason named.

| # | ROW | FIXED | ADAPTIVE | CLASS |
|---|-----|-------|----------|-------|
| 1 | optimality_rel_tol | 1e-6 | 1e-8 | iii (F8, red line) |
| 2 | cost_abs_tol | 1e-30 | 0.0 | iii (F8, red line) |
| 3 | cost_rel_tol | 1e-15 | 0.0 | iii (F8, red line) |
| 4 | optimality_abs_tol | 1e-14 | 1e-14 | same |
| 5 | max_iterations (Newton cap) | 30 | 30 | same |
| 6 | line_search cap / variant | 40 / armijo_decay | 40 / armijo_decay | same |
| 7 | armijo_c, rho, ls_rel_slop | identical | identical | same |
| 8 | static_substep (+ its 2 caps) | False, 8, 6 | False, 8, 6 | same |
| 9 | R inputs: beta, sigma, tau_d, k | 1.0, 1e-3, 0.02, 1250 | same | same |
| 10 | cone / soft-norm tol / mu | shared sap_helpers | shared | same |
| 11 | ACR constitutive-dt buffers | absent | allocated, s in (2,4) | i |
| 12 | preset + 4 precision knobs | approx32 / fp64 | approx32 / fp64 | same |
| 13 | per-world SAP contact budget | 2048 | 2048 | same |
| 14 | contact_jacobian (67 rows) | — | — | same |
| 15 | free_motion (75 rows) | — | — | same |
| 16 | contact SOURCE | manager CollisionPipeline | solver-owned pipeline | iv (row 17-19) |
| 17 | broad_phase | explicit (2104 allow-list) | sap (3720 cap, 2104 deny) | iv, PROVED |
| 18 | pipeline rigid_contact_max | 8,000,000 pooled | 16,384 pooled | iv, PROVED |
| 19 | max_triangle_pairs | 192e6 | 192e6 | same |
| 20 | truncated contacts / headroom | 0 / >=21x | 0 / >=11x | same |
| 21 | num_substeps per boundary | 2 | 1 (+adaptive march) | i |
| 22 | solver_dt | 4.16667e-3 | 8.33333e-3 | i |
| 23 | ADVANCE PER DECIMATION TICK | 8.33333e-3 | 8.33333e-3 | iv, PROVED |
| 24 | collision cadence | 1/tick (collision_decimation 0) | 1/boundary at entry | iv (row 23) |
| 25 | per-substep Jacobian re-anchor | yes | yes | same |
| 26 | warm start | 1 GLOBAL flag, shape (1,) | per-world + CENIC v_t/v_half/v_full | ii |
| 27 | reset granularity | global reset_runtime_state() | per-world masked reset | ii |
| 28 | reset host sync per step | local_mask.numpy().any() | none | ii |
| 29 | convergence certificate | last_converged, 0 consumers | raises RuntimeError | ii |
| 30 | containment: dt shrink-retry | absent | present | i |
| 31 | containment: latch/freeze/isolate | absent | present | ii (=F5) |
| 32 | physics_diverged termination | PERMANENT NO-OP | live | iii (=F5) |
| 33 | shared sap_warp env flags (8) | all reach it | all reach it | same |
| 34 | NEWTON_SAP_PRESET/LS/PRECISION | reaches it | reaches it | same |
| 35 | NEWTON_SAP_DETERMINISTIC | solve only, NOT pipeline | solve AND pipeline | ii |
| 36 | adaptive-only env flags (19) | n/a | n/a | i |
| 37 | manager CUDA-graph capture | False | False | same |
| 38 | solver-internal graph | none | graph + conditional | ii (wall only) |
| 39 | MDP terms (all 6 managers) | identical | identical | same |
| 40 | contact-sensor force channel | 0.0 | 0.0 | same (see below) |

ROW 17-18 ARE PROVED BENIGN BY MEASUREMENT, NOT BY ARGUMENT. The two
arms run DIFFERENT broadphase algorithms over differently-built pair
structures, which is not equivalent by construction, so the fixed arm was
run BOTH ways (probe-local cfg override, no repo edit) and compared:
  deepest phi0, explicit vs sap, IDENTICAL TO 6 SIGNIFICANT FIGURES in
  every quasi-static phase (rest -5.39571e-5, press -5.39552e-5,
  swing -5.39552e-5 on both);
  max contacts in one world 54 vs 54 in all three phases;
  total contact samples within 0.09% (25597/25574, 38428/38401,
  51245/51235) — a handful of marginal pairs at the collision margin,
  not a structural difference.
Row 18 is benign because neither cap binds: truncated contacts 0 on both
arms, and both arms are throttled by the SAME 2048/world SOLVER budget
downstream of whatever the pipeline produced. RESIDUAL RISK: equivalence
was established in the quasi-static phases only; flail trajectories have
already diverged so that phase cannot test it.

ROW 23 IS THE ONE THAT MATTERED MOST AND IT PASSES. num_substeps and
solver_dt differ (that is the experiment), but their PRODUCT — the wall
of simulated time each arm advances per decimation tick — is
8.33333333e-3 s on BOTH arms, read live off the manager. The arms
integrate the same interval; only the subdivision differs. Had this row
disagreed, every wall-clock and success number in the campaign would
have been comparing different amounts of physics.

### WHAT MARCO MUST DECIDE (iii rows), with the exact change

 M1 F8 — rows 1-3. UNCHANGED from pass 26 and still the loudest. The
    fixed arm accepts a contact-solve residual 100x looser and may stop
    on a cost plateau where the adaptive arm structurally cannot. One
    kwarg triple at the SolverSAP construction site
    (mjwarp_manager.py:382-390): `optimality_rel_tol=1e-8,
    cost_abs_tol=0.0, cost_rel_tol=0.0`. Red-line item (optimality_rel_tol
    is named in the pass-25 VALIDITY RED LINE), so a loop may not land it
    even on an arm with no campaign record. Until it is aligned, any
    "fixed fails" result is partly attributable to tolerance, not to
    timestepping.
 M2 Row 32 — `physics_diverged` is an ACTIVE termination term that is a
    PERMANENT NO-OP on the fixed SAP arm AND on the MuJoCo arm, confirmed
    live this pass (get_diverged_env_mask() returns None on both;
    mdp.py:174 then returns all-False). One arm can excise a broken world
    from the training distribution and the other cannot. Task-side (F5).
    Either give the fixed arm an equivalent or state it in the paper.
 M3 Row 35 — NEWTON_SAP_DETERMINISTIC reaches the adaptive arm's
    collision pipeline (canonical contact sort) but NOT the fixed arm's,
    because the manager pipeline reads the cfg field instead. With det=1
    the adaptive arm is order-canonical and the fixed arm is not, so a
    determinism comparison between arms is apples-to-oranges. One line:
    `NewtonCollisionPipelineCfg(..., deterministic=True)` in the task
    cfg, or an env-var propagation in mjwarp_manager gated on the fixed
    SAP arm. Left alone because the cfg field is shared with the MuJoCo
    backend, whose established trajectories it would move.

### WHY THIS PASS LANDED NOTHING (the (ii) rows, and why each is blocked)

Rows 26-29, 31, 35, 38 are real engineering asymmetries with no
timestepping content. NONE was landed, and the reason is structural
rather than a judgement call:
  * Rows 26/27 (warm start, reset granularity). The fixed arm's
    warm-start gate is a SINGLE GLOBAL flag —
    `_contact_solve_v_guess_active` has shape (1,), confirmed in the live
    dump — cleared wholesale by reset_runtime_state() whenever ANY world
    resets, where the adaptive arm resets per-world under a mask. Making
    it per-world means new state in sap_warp's contact solve, which the
    adaptive arm's inner SolverSAP also runs; it cannot be shown
    byte-inert for the adaptive arm, so it is not a fairness edit but a
    feature. INCIDENCE IS (v) UNKNOWN: measured 0 drops over 390 steps
    including a flail phase, but this rig did not independently confirm
    that env resets reached the manager mask, so that 0 says nothing
    about the training regime. Measuring the training-regime rate is the
    first thing the next pass should do.
  * Row 29 (convergence certificate). SolverSAP records `last_converged`
    and NOTHING in any of the three repos reads it — grep is exhaustive,
    2 hits, both writes. So an unconverged fixed-arm solve is committed
    silently and invisibly, where the adaptive arm raises quoting
    optimality_rel_tol=1.000e-08. Fix is a counter in mjwarp_manager
    gated on the fixed SAP arm; deferred because it is observability that
    belongs with the first real fixed-arm comparison run, not churn now.
  * Rows 17/18/31/38 would all have to be equalized by changing the
    ADAPTIVE arm, which the byte-unchanged gate forbids — and 17/18 turned
    out BENIGN on measurement anyway, so landing them would have been
    churn against a proof of no effect.
NET: after pass 26's fixes, there is no in-grant fairness edit left that
is both landable and non-speculative. The remaining asymmetries are
red-lined, task-side, or blocked by the adaptive arm's byte-unchanged
contract.

### CONTACT-SENSOR VERDICT (task 2)

CONSUMERS: NONE. Exhaustive source sweep of every MDP surface —
observations (5 terms, one `policy` group), rewards (6), terminations
(6), curriculum (2), events (2), commands (1), actions (2), the task's
own mdp.py, the RSL-RL cfg, recorders and wrappers. Not one term reads
`pad_handle_contact` or `arm_body_contact`. The two names appear exactly
twice in the whole tree: their definition sites
(trossen_spatula_lift_env_cfg.py:214, :223). The task's mdp.py DOES
contain contact readers — `_sensor_force_mag` (:51-63, reads
force_matrix_w only) and its eight callers `handle_contact`,
`body_contact`, `pad_contact_bipolar`, `handle_grasped`,
`handle_ee_distance`, `lift_progress_by_handle`,
`object_lifted_by_handle`, `object_goal_distance_by_handle` — and every
one of them is DEAD CODE, referenced by nothing. Note zero-weight reward
terms would NOT be pruned (reward_manager.py:145-148 keeps them and only
short-circuits evaluation), so "no consumer" is not hiding behind a
weight of 0; there simply is no term.

THE CHANNEL, HOWEVER, IS NOT SYMMETRIC ACROSS BACKENDS — and this pass
resolves the pass-25 R4 ambiguity ("sensors dead" vs "my scripted actions
never satisfied the filters") by reading the SOURCE array globally
instead of the sensor's filtered output. `SensorContact.update` reads
`contacts.force`; that array was read unfiltered, every row, on all three
arms over rest/press/flail:
    MuJoCo    contacts.force absmax 0.0671 (rest), 0.0671 (press),
              PEAK 8.962 over flail  -> THE WRITEBACK IS LIVE
    SAP fixed contacts.force absmax 0.0 everywhere, flail peak 0.0
    SAP adapt contacts.force absmax 0.0 everywhere, flail peak 0.0
So SolverMuJoCo.update_contacts genuinely fills the frontend force array
even under use_mujoco_contacts=False, while BOTH SAP arms' update_contacts
are the documented no-ops and leave it identically zero. The two SAP arms
agree with each other exactly, so the SAP-vs-SAP comparison is unaffected.

VERDICT. SAP training and MuJoCo training ARE the same task today, but
only because the channel is unused: no MDP term reads it, so a live
force array and a zero force array produce identical observations,
rewards, terminations, curricula and events. This is a coincidence of the
current task definition, not a property of the stack. THE MOMENT ANY
CONTACT-DERIVED TERM IS ADDED — and eight such functions are already
written and sitting in this task's mdp.py — SAP and MuJoCo become
DIFFERENT TASKS, with SAP's term reading a constant zero. That is the
finding: not a live defect, a loaded one.

RESIDUAL RISK, NAMED. A live sensor READING was not demonstrated on any
arm: `newton_sensor[...].total_force` was 0.0 on all three arms in all
phases, including MuJoCo where the underlying array is live. Sensing
objects and counterparts both resolved correctly (shapes [16]/[16,3] for
the 2 carriage bodies x 3 handle shapes, [48]/[48,9] for the 6 arm links
x 9 wall/base shapes), so the filters are wired; the scripted phases
simply never brought those specific pairs into contact. The verdict above
does not depend on that gap — it rests on the exhaustive zero-consumer
sweep and on the global force-array measurement, neither of which needs a
filtered reading.

### CHARACTERIZATION (one draw, 8 envs, NOT the killer comparison)

    arm        ms/step   rest/press/swing phi0   flail phi0
    MuJoCo      9.617    n/a (no SAP jacobian)   n/a
    SAP adapt  27.102    -5.5838e-5              -1.4816e-2
    SAP fixed  31.078    -5.3957e-5              -7.1775e-3
Adaptive cumulative_accepted_steps 14542 over 360x4x8 = 11520
world-boundaries (~1.26/boundary) vs the fixed arm's 2/boundary. The
fixed arm's quasi-static phi0 again equals the ACR-OFF value exactly
(s == 1 by construction with no doubling), the third independent
confirmation of the ACR mechanism from a solver that never runs it.
Peak per-world contact demand 97-102 (fixed) vs 173 (adaptive) in flail
— confounded by chaos, not a capacity finding; the quasi-static phases
agree at 54/54.

### ADAPTIVE ARM: BYTE-UNCHANGED, TRIVIALLY

Zero bytes changed in newton-adaptive, sap_warp or IsaacLab this pass
(`git status --porcelain` empty in all three at pass end; the only write
is this ledger entry). The 8-gate chain is therefore not applicable —
there are no "final bytes" distinct from the certified HEADs. The
campaign invariant was nonetheless re-measured incidentally by the
fairness probe on the unmodified stack: adaptive deepest phi0
-5.5838e-5 in rest, press AND swing, i.e. the -5.584e-5 held since
pre-campaign, unchanged.

### FINDINGS ADDED

F9  The contact-force channel is LIVE on MuJoCo and IDENTICALLY ZERO on
    both SAP arms (measured on contacts.force globally, not on a filtered
    sensor output). Currently inert because no MDP term consumes it, and
    symmetric across the two SAP arms. Becomes a live cross-backend
    defect the instant a contact-derived term is added — and eight such
    functions already exist unreferenced in the task's mdp.py.
F10 The fixed arm has NO convergence certificate: `last_converged` is
    written twice and read nowhere in any of the three repos, so an
    unconverged fixed-arm contact solve is committed silently, where the
    adaptive arm raises. Compounds F8.
F11 The fixed arm's contact-solve warm start is a SINGLE GLOBAL flag
    (shape (1,)) dropped wholesale on any world's reset, against the
    adaptive arm's per-world masked reset. Structural; firing rate in the
    training regime UNMEASURED.
CLEAN — the SAP solve stacks themselves. 3 tolerance rows and 4 ACR
    buffers is the entire difference across ~400 compared attributes in
    SolverSAP, contact_solve, contact_jacobian and free_motion. Contact
    law, caps, preset, precision and capacity are identical by
    measurement. Broadphase and pipeline-capacity asymmetries are proved
    benign. Both arms advance exactly the same simulated time per tick.

Provenance (all p27_ prefix, no p13-p26 artifact overwritten):
p27_fairness_probe.py + p27_arm_{fixed,adaptive,mujoco}.{json,log};
p27_sensor_probe.py + p27_sens_{fixed,adaptive,mujoco}.{json,log};
p27_asym_probe.py + p27_asym_{explicit,sap,warmstart}.{json,log};
p27_diff.py.

## PASS 28 — F11 DISMISSED WITH NUMBERS, F10 CHARACTERIZED, THE PASS-24
## CLOSERS PRICED, AND THE CAMPAIGN HANDOFF
## 2026-08-16. MEASUREMENT + SOURCE AUDIT ONLY. ZERO CODE EDITS in all
## three repos. Last queued in-grant pass.

Stack: newton-adaptive 71b96718 (march-counter-log), sap_warp 345c9fe
(main), IsaacLab 13e049ead1 (develop) — all three verified at the
certified HEADs, worktrees clean and GPU idle (438 MiB, 0 compute apps)
before the first launch and after the last; every run serial, one GPU
process at a time (gpumem sampled between runs in
p28_chain{1,2,3}.log). Passes 25-27's prose was treated as folklore and
re-derived from source before anything was built on it.

### TASK 1 — F11 WARM START: **IT FIRES ON 100% OF TRAINING STEPS, AND
### IT COSTS NOTHING MEASURABLE.** Dismissed with numbers.

THE MECHANISM, re-derived from source. `_contact_solve_v_guess_active`
is allocated `wp.zeros(1, dtype=int)` (solver_sap.py:947) and dumped
live as shape [1] — on BOTH arms, because the adaptive arm's inner
solver is the same SolverSAP object. `reset_runtime_state()`
(:1023-1035) zeroes it wholesale. The fixed-arm manager branch
(mjwarp_manager.py:651-655) is
`if cls._sap and not cls._adaptive: if local_mask.numpy().any():
cls._solver.reset_runtime_state()`. The sole consumer is
`_copy_solve_velocity_inputs_flat_batched_with_guess_flag`
(contact_solve.py:378-407): `use_v_guess[0] != 0` selects the previous
solve's terminal velocity `v_guess` over the boundary velocity `v0`;
the flag is re-armed to 1 after every solve (:9350). WHY THE SAME
GLOBAL FLAG IS HARMLESS ON THE ADAPTIVE ARM: `_set_solver_guess`
(solver_sap_adaptive.py:2213-2228) writes it explicitly before EVERY
inner solve (:2274), so that arm never depends on the persisted value.
The asymmetry pass 27 described is real and is exactly this — but its
matrix row 26 states the mechanism wrongly: the adaptive arm's flag is not
per-world either, it is the SAME shape-(1,) array, and what makes the
arm immune is per-attempt re-arming, not per-world granularity.

DOES IT FIRE? Pass 27 measured 0 firings over 390 steps but could not
confirm resets reached the manager mask, so that 0 was inconclusive.
Settled here in a REAL training run — fixed SAP arm, 1024 envs, seed
42, 40 iterations through the normal `train` entrypoint, observer-only
counters injected by a scratchpad copy of the entrypoint
(p28_train_instr.py + p28_observer.py; no repo file touched).
  SOLVER IDENTITY PROVEN FROM THE RUN: the observer's wrapper on
  `SolverSAP.reset_runtime_state` was entered 962 times, which only a
  live SolverSAP can do; the log also carries the SAP eager-execution
  warning.
  manager physics steps 3840 = 960 env.steps x decimation 4
  delegate invocations 8641 = 9.00 per env.step
  invocations with >=1 flagged world 961; manager steps with a fire 960
  **FIRING RATE = 960/960 env.steps = 1.000**
  worlds flagged 9549 total = 9.94 per env.step out of 1024
  reset_runtime_state calls 962 = once per env.step
  So ~1014 of 1024 worlds lose a valid warm start on EVERY training
  step because ~10 others reset.
RESETS PROVABLY REACH THE MASK — the check pass 27 lacked. rsl_rl's own
mean episode length over the last 10 iterations is 97.8 steps, so the
expected resets per step are 1024/97.8 = 10.5, against the measured
mask popcount of 9.94. Two unrelated bookkeeping paths agree to 6%.
The termination mix that produces it: object_off_table 0.253,
robot_abnormal 0.366, time_out 0.384, object_speeding 0.002,
object_dropping 0.000, physics_diverged 0.000.
A SCRIPTED RIG CANNOT SEE THIS. The same 1024 envs driven by an
untrained-Gaussian action stream produced ZERO early terminations in
200 steps, so all 1024 envs timed out together at step 149 and the
firing rate was 1/200 (p28_rate_fixed_prod.json). Early terminations
are what desynchronize resets; without a learning policy in the loop
the measurement understates the rate by ~150x. That is why the real
training run was required, and it is the correction to pass 27's 0.

WHAT DOES IT COST? Measured three ways, because a two-run A/B is
confounded (a firing step is also a just-reset step, whose physics is
trivially easy).
 (1) PAIRED, within one trajectory (p28_cost_alt_1024.json). The flag
     is force-cleared before every ODD env.step, so cold and warm
     first-solves interleave under near-identical conditions; only
     solve #1 of an env.step can be cold, since the flag re-arms at the
     end of every solve. 1024 envs, 200 steps:
       solve-0 Newton iterations summed over all worlds
         cold 1441.09 (n=100)  vs  warm 1441.02 (n=100)
       = 0.07 iterations out of 1441, i.e. 0.005%, i.e. 7e-5
         iterations per world.
       ms/step cold 77.22 vs warm 79.89 (cold nominally FASTER).
 (2) PAIRED at 64 envs over the four scripted phases including flail
     (p28_cost_alt_64.json): solve-0 iterations cold 84.51 vs warm
     85.50 (cold LOWER); ms/step 37.07 vs 37.31.
 (3) WHOLE-RUN BOUNDS, same seed and action stream, 1024 envs:
       always-cold  79.04 ms/step, solve-0 iters 1441.17
       never-cold   79.32 ms/step, solve-0 iters 1449.46
     The all-cold arm was 0.4% FASTER than the all-warm arm.
Every number is at or below the draw noise, and the sign is not even
consistent.

WHY THE NULL IS PHYSICAL, not a broken rig. The fixed arm's contact
solve takes ~1.41 Newton iterations per world on solve 0 (1441/1024,
with all 1024 worlds carrying contacts) and ~1.20 on later substeps.
There is almost nothing for a warm start to save, and the cold seed v0
(the boundary joint velocity) is already close to the warm seed v_flat
(the previous solve's terminal velocity) in this task's regime.

VERDICT. F11 has the highest possible incidence and no measurable
consequence. It is NOT a fairness defect worth fixing, and no change
was landed. RESIDUAL RISK, NAMED: the cost was measured in a regime
where the solve is easy (~1.4 iterations/world); a scene that drove the
solve toward its 30-iteration cap could make the warm start matter, and
that regime was not produced by any rig in this campaign.

THE FIX, IF IT IS EVER NEEDED (recipe only, NOT landed):
  1. sap_warp/sim/solver_sap.py:947 — `wp.zeros(1, ...)` ->
     `wp.zeros(max(int(getattr(model, "world_count", 1)), 1), ...)`.
  2. sap_warp/sim/contact_solve.py:404 — `use_v_guess[0]` ->
     `use_v_guess[env]`.
  3. sap_warp/sim/contact_solve.py:9350-9356 — the re-arm launch is
     `dim=1` into `_set_scalar_i32`, whose body is `value[0] = v`; it
     needs `dim=num_envs` and a per-index write (new 1-line kernel).
  4. newton-adaptive solver_sap_adaptive.py:2213-2228 — both
     `_set_scalar_i32` launches in `_set_solver_guess` go `dim=1` ->
     `dim=world_count`, per-index write.
  5. IsaacLab mjwarp_manager.py:651-655 — replace the host-gated
     `reset_runtime_state()` with a masked device launch clearing only
     the flagged worlds' flags.
  BYTE-INERTNESS FOR THE ADAPTIVE ARM is provable PROVIDED step 4 fills
  every world with the same value: the flag's only use is the branch at
  contact_solve.py:404, so an env-uniform array gives
  `use_v_guess[env] == use_v_guess[0]` for every env, the same branch,
  the same write, and every downstream fp operation identical.
  RESIDUAL RISK: that argument is source-level. The proof would be the
  8-gate chain (march fingerprint [6,25,20,24,19], phi0 -5.584e-5 /
  -2.756e-5) on the final bytes, which was NOT run because the change
  was not made.

ONE MORE ROW CLOSED IN PASSING — matrix row 28, the fixed arm's
host-synced reset gate. `local_mask.numpy().any()` forces a full device
sync on every delegate invocation, measured at 8-9 invocations per
env.step. Priced probe-locally at 1024 envs, 200 steps, three variants
of the same branch (p28_gate_*.json):
    production branch          78.30 ms/step
    no host gate (uncondit.)   77.57 ms/step
    branch removed entirely    78.40 ms/step
Spread 0.8 ms = 1%, and the variant doing the LEAST work was the
slowest — pure draw noise. The gate costs nothing measurable. Note the
fixed arm runs EAGER (no manager CUDA graph), so its device queue is
short and a host sync is cheap; on a graph-captured arm the same gate
would not be free.

### TASK 2 — F10, THE MISSING CONVERGENCE CERTIFICATE: the guard is
### absent AND the flag is vacuous, but the event it would catch did
### not occur in 1,182,720 observed env-solves.

SOURCE, re-derived. `SolverSAP.last_converged` is written at
solver_sap.py:1031 and :1535 and read nowhere in any of the three
repos — pass 27's grep re-confirmed. WORSE THAN UNREAD, AND THIS IS NEW:
the value is VACUOUS. `SapContactSolve.solve` returns
`self._make_result(self.last_iterations, self.last_line_search_iterations,
True)` — the converged field is the literal `True` on the
graph-conditional path (contact_solve.py:9358) and on the static path
(:8501), with `last_iterations` set to -1/0 alongside. So
`last_converged` is a constant True by construction and would carry no
information even if something read it. MEASURED: True in 100% of 4080
observed solves, `last_converged_values` = [True] only.
THE REAL DECISION lives per-env in `contact_solve.converged_env`
(allocated :5379, decided in the fused update/eval kernel :3029-3057):
`converged_env[env] = 1` on `opt_reached or cost_reached`; on
`iteration >= max_iterations` the kernel clears `newton_active[env]`,
sets `newton_max_reached[0] = 1` and LEAVES `converged_env[env] = 0` —
the env exits the loop unconverged and its velocity is integrated and
committed. THE ADAPTIVE ARM CONSUMES EXACTLY THIS ARRAY:
`_accumulate_solve_convergence` (solver_sap_adaptive.py:2277-2282)
folds `contact_solve.converged_env` into `_solve_ok`, which drives the
divergence sentinel and the raise at :3346. So the asymmetry is not
"one arm has a nicer error message": the same per-env array is read on
one arm and dropped on the other.

(a) DOES THE FIXED ARM PRODUCE UNCONVERGED SOLVES? Observer read
`converged_env`, `optimality_reached_env`, `cost_reached_env`,
`newton_iterations_env`, `newton_max_reached` and the residual norms
after EVERY solver substep.
    64 envs, 390 steps, rest/press/swing/flail:
      3120 solves, 199,680 env-solves with contacts,
      0 unconverged, 0 newton_max_reached, 0 cost-plateau exits.
    1024 envs, 120 steps, Gaussian action stream:
      960 solves, 983,040 env-solves with contacts,
      0 unconverged, 0 newton_max_reached, 0 cost-plateau exits.
    TOTAL 1,182,720 env-solves. RATE 0.
  The cost-plateau count matters on its own: with the fixed arm's
  cost_abs_tol 1e-30 / cost_rel_tol 1e-15 the solve MAY stop on a cost
  plateau where the adaptive arm structurally cannot (F8 rows 2-3), and
  it never did — 0 exits with cost_reached=1 and optimality_reached=0.

(b) WHAT WOULD IT COMMIT? The per-env optimality residual was
reconstructed with the kernel's OWN expression,
`opt_tol = optimality_abs_tol + optimality_rel_tol * max(|p|, |Jc|)`,
from the live `grad_norm2`/`p_norm2`/`jc_norm2` arrays and the arm's
live tolerances (1e-14 / 1e-6) — a ratio against the tolerance the arm
itself claims, not an invented one.
    worst ratio over the 64-env run 0.99844 (p99 0.8017);
      per phase rest 0.9601 / press 0.9802 / swing 0.2120 /
      flail 0.99844
    worst ratio over the 1024-env run 0.99378 (p99 0.9214)
Every observed solve terminated strictly inside its own tolerance, so
there is no committed residual to characterize; 0 non-finite steps in
both runs.

(c) FAIRNESS CONSEQUENCE, plainly. The guard is missing and the flag
that pretends to be it is a constant. In the regimes tested the event
it would catch did not occur, so F10 is not distorting any number this
campaign reports. What remains is an asymmetry of GUARANTEE, not of
behaviour: the adaptive arm CANNOT commit an unconverged solve (it
raises, re-proven live in the pass-26 G5 chain quoting
optimality_rel_tol=1.000e-08) while the fixed arm has no mechanism that
would notice. F10 is entangled with F8: part of the reason the fixed
arm never hit its cap is that it was asked for 100x less.
RESIDUAL RISK, NAMED: the matched adaptive-arm number does not exist
and cannot be taken from the host — the adaptive arm captures its inner
substep body into a CUDA graph, and a host read from inside it aborts
capture (Warp CUDA error 906) and segfaults the process, measured this
pass. The adaptive contrast therefore rests on source plus the G5 live
raise. And the hardest regimes tested are the scripted flail phase and
a 40-iteration policy; a longer-trained policy could be harsher.

### TASK 3(a) — THE RUN-AHEAD MASKED COLLIDE **DOES** CARRY A
### FULL-WIDTH FLOOR, AND IT IS NOT THE BROADPHASE.

The crossing node calls
`pipeline.collide(state, contacts, world_mask=self._ra_crossed,
sort_contacts=False)` (solver_sap_adaptive.py:2459). The pipeline
docstring asserts the pass's cost "scales with the unmasked subset";
that prose was not taken as evidence. The call was priced directly on
the live production scene (adaptive arm, 1024 worlds, 34,818 shapes,
BroadPhaseSAP) at a sweep of crossing counts, with Warp per-kernel CUDA
timing so the floor is NAMED (p28_collide_floor.json, 20 reps/point):

    crossing worlds k :  0     1     2     4     8    16    32
    kernel ms         : 0.725 1.000 0.982 0.981 0.981 0.987 1.016
    crossing worlds k :  64   128   256   512  1024  unmasked
    kernel ms         : 1.100 1.286 1.679 2.643 4.739  4.740

FLAT from k=1 to k=16, first moves at k=32, and k=1024 equals the
unmasked pass to 3 decimals — so the mask machinery itself is free and
the narrowing is real ABOVE ~32 crossing worlds and absent below it.
FLOOR = 0.98-1.00 ms = **20.7% of a full collide**, invariant in k.

THE FLOOR'S KERNELS, named from the timing rather than guessed — and
the source-level prediction was WRONG. The full-width AABB kernel
`compute_shape_aabbs_masked` (dim = shape_count = 34,818) costs 0.0090
ms and `_sap_broadphase_kernel` costs 0.013 ms; neither is the floor.
The floor is the contact-stream scan machinery, whose size is the
buffer capacity and not the participating set:
    _cs_scan_chunk_offsets            0.236 ms (k-invariant)
    _cs_scan_chunk_emit_exclusive     0.127
    _cs_scan_chunk_emit_inclusive     0.123
    _cs_scan_chunk_reduce             0.104
    _scan_chunk_* trio                0.048
    narrow_phase_kernel_gjk_mpr       0.098-0.103 (also k-invariant)
    + ~0.09 of small fixed kernels
    = ~0.83 ms of the 1.00 ms k=1 fire
What DOES scale: mesh_triangle_contacts_to_reducer (0.052 -> 2.670 ms,
51x), narrow_phase_find_mesh_triangle_overlaps, reduce_buffered_
contacts, mesh_plane_contacts_reduce, export_reduced_contacts.

HONEST NARROWING CEILING. At the run-ahead design point (a handful of
crossing worlds per fire) ~83% of the fire's kernel time is floor and
no world-mask narrowing can remove it. Removing it means making the
contact-stream scan and the gjk_mpr primitive pass proportional to the
participating PAIR count — a compacted pair list, not a mask — in
upstream Newton geometry (newton/_src/sim/collide.py and the
narrow-phase scan), outside the SAP solver. That is a rewrite, not a
narrowing, and it is not in this campaign's scope. The measured saving
that the mask DOES deliver, 4.74 -> 1.00 ms per fire (79%), is
consistent with pass 23's plateau verdict of -4%..0.

### TASK 3(b) — REMAINING ENV-AXIS LAUNCHES OUTSIDE contact_solve.py:
### CLOSED WITHOUT A PATCH, ON A MEASUREMENT RATHER THAN ON PASS 19'S.

The backlog item was not inherited. One adaptive march at 1024 envs
with the solver-internal graph disabled (so per-kernel timing is
attributable), grouped by kernel, 131 distinct kernels, 166.13 ms of
march kernel time per env.step (p28_march_kernels.json). The eight
env-axis kernels launched from contact_jacobian.py were named from
their launch sites; only two register at all:
    _assemble_dynamics_matrix_multi_env_sap   0.248 ms
    _zero_env_contact_counts_gated            0.119 ms
    TOTAL 0.367 ms = **0.221% of march kernel time**
Deleting 100% of that class's work buys 0.22%, an order of magnitude
below the 2% demand-normalized bar; applying the measured march
compaction (11-21% of worlds inactive) leaves 0.02-0.05%. There is no
candidate. No patch was written, so there is none to preserve.
RESIDUAL RISK: graph-off changes absolute kernel time, not the share
this closure rests on. free_motion.py, solver_sap.py and sap_helpers.py
carry NO env-axis launches at all (grep of `dim=...num_envs`), so
contact_jacobian.py was the whole remaining surface.

### A NUMBER THIS PASS PRODUCED THAT NOBODY ASKED FOR, AND ITS CAVEATS

The pass-27 matrix left the fixed arm runnable but never run at
production scale. This pass ran it: fixed SAP arm, 1024 envs, seed 42,
40 iterations. Iteration time rises 1.80 -> 3.75 s as the policy learns
to make contact; **iterations 19-24 mean 3.518 s/iter** — the SAME
window definition used for the adaptive plateau, which p23_plateau_off
put at 12.88 s/iter at the same env count and task.
DO NOT QUOTE 3.66x AS A RESULT. It is one draw against a
differently-seeded historical draw, and at least three things push the
same way: the fixed arm runs 2 solves per boundary against the adaptive
arm's ~3.8 (3 solves x ~1.26 accepted steps/boundary), the fixed arm
solves to a 100x looser tolerance (F8, unfixed), and the two runs'
contact demand is not matched (this run's mean episode length is 97.8;
the p23 run's is not in hand). What the number IS: the first evidence
that a matched fixed-vs-adaptive SAP comparison at production scale is
now cheap to run — ~4 minutes per 40-iteration arm — and the strongest
argument yet for closing F8 first, since the tolerance gap sits
directly on the axis being compared.

### ADAPTIVE ARM: BYTE-UNCHANGED, TRIVIALLY

Zero bytes changed in newton-adaptive, sap_warp or IsaacLab this pass
(`git status --porcelain` empty in all three at pass end; the only
write is this ledger entry). The 8-gate chain is not applicable —
there are no final bytes distinct from the certified HEADs. Every
probe-local behaviour change (forced warm-start drops, delegate
variants, broadphase sweeps) lived in the probe process and touched no
repo file.

### FINDINGS SETTLED

F11 CLOSED, NO ACTION. Fires on 100% of real training steps (960/960
    env.steps at 1024 envs, ~10 worlds resetting per step), and costs
    0.005% of Newton iterations and nothing detectable in wall. High
    incidence, null consequence. Pass 27's "0 firings" was an artifact
    of a scripted rig with no early terminations.
F10 CHARACTERIZED, MARCO'S. No certificate, and `last_converged` is a
    constant True rather than a dropped signal. 0 unconverged and 0
    cost-plateau exits in 1,182,720 env-solves across rest/press/swing/
    flail and a 1024-env Gaussian stream; worst optimality residual
    0.998 of the arm's own tolerance. An asymmetry of guarantee, not of
    measured behaviour. Two recipes below.
F12 NEW, HYGIENE — the fixed arm's `local_mask.numpy().any()` reset
    gate forces 8-9 full device syncs per env.step. Priced at 1% of
    ms/step, i.e. inside noise, ONLY because that arm runs eager. Worth
    knowing before anyone graph-captures the fixed arm.
CLEAN — the masked collide's mask machinery (free at full width), and
    the env-axis launch class outside contact_solve.py (0.22% of march
    kernel time, no candidate).

Provenance (all p28_ prefix, no p13-p27 artifact overwritten):
p28_fixedarm_probe.py + p28_{rate_fixed_prod,rate_fixed_keep,
cert_fixed_phases,cert_fixed_1024,smoke}.{json,log};
p28_warmstart_cost.py + p28_cost_{alt_1024,alt_64,always_1024,
never_1024}.{json,log} + p28_gate_{prod_clean,nogate,noop}.{json,log}
+ p28_rate_adaptive.{json,log}; p28_observer.py + p28_train_instr.py +
p28_train_fixed.log + p28_train_obs_fixed.json; p28_collide_floor.py +
p28_collide_floor.{json,log}; p28_march_kernels.py +
p28_march_kernels.{json,log}; p28_chain{1,2,3}.sh +
p28_chain{1,2,3}.log (gpumem samples between every run).


## PASS 29 — THE FAIRNESS FIXES LAND. Tolerance, certificate, containment,
## determinism and the two stragglers, all measured on the arms that run.
## 2026-08-16. FIRST CODE-LANDING PASS SINCE 26. Marco's in-channel "You have
## all perms keep going", against a message naming M1/F8, M2, M3 and F10,
## lifted the hold on those items and their class. Governing rule applied to
## every judgement call: ONLY MAKE THE ADAPTIVE ADVANTAGE HARDER TO CLAIM,
## NEVER EASIER. Every change below strengthens the FIXED arm or deletes a
## confound. The adaptive arm's physics is byte-unchanged and gated.
## Commits (LOCAL ONLY, nothing pushed): IsaacLab 27cf9c1ec2, 4e8f763acb,
## 6fdd695cda, 62c165b5f0; sap_warp afd5dc6; newton-adaptive: this entry.

### TASK 1 — M1/F8, THE TOLERANCE ASYMMETRY: CLOSED BY LIFTING THE FIXED
### ARM TO THE RAIL. Cost measured, not asserted.

SITE, re-derived rather than taken from pass 27's line numbers: the fixed
arm's SolverSAP is constructed in mjwarp_manager `_create_solver`, which
passed no tolerances at all, so the arm took SolverSAP's ctor defaults.
CONFIRMED LIVE BEFORE THE EDIT (p29_tol_fixed_pre.json, read off the
constructed object): optimality_rel_tol 1e-06, optimality_abs_tol 1e-14,
cost_abs_tol 1e-30, cost_rel_tol 1e-15, max_iterations 30.

THE EDIT: `optimality_rel_tol=1e-8, cost_abs_tol=0.0, cost_rel_tol=0.0`,
plus the fp32 analogue resolved by the SAME rule the adaptive arm uses
(max(1e-8, _FP32_OPTIMALITY_K * eps_fp32)) so the match survives a
precision change. The adaptive arm's 1e-8 was NOT touched; the rail is
satisfied by lifting the fixed arm to it.

BOTH ARMS DUMPED LIVE ON FINAL BYTES (p29_arm_{fixed,adaptive}.json), and
they are identical row for row:
    optimality_rel_tol 1e-08   optimality_abs_tol 1e-14
    cost_abs_tol 0.0           cost_rel_tol 0.0
    max_iterations 30          line_search armijo_decay / 40
    tau_d 0.02  beta 1.0  sigma 1e-3  preset approx32  solve fp64

WHAT IT COSTS THE FIXED ARM. Priced as an A/B INSIDE one process on one
build (the tolerances are re-read off the solver at every step and
forwarded as kernel launch arguments, and the manager captures no CUDA
graph for SAP, so mutating them between blocks genuinely changes what the
solve is asked for -- confirmed by the iteration counts moving). Blocks
alternate loose/tight/loose/tight so clock drift cannot be mistaken for
the effect. 64 envs, scripted rest/press/swing/flail, 360 steps/block:

                     ms/step (mean of 2)   inner Newton iterations/env-solve
  BEFORE the edit    1e-6: 34.74           1.1201
                     1e-8: 37.07  (+6.7%)  1.6202  (+44.6%)
  AFTER  the edit    1e-6: 34.05           1.1213
                     1e-8: 36.49  (+7.2%)  1.6793  (+49.8%)
  repeat spread      loose 4.5% / 7.1%, tight 2.1% / 0.6%

So the honest baseline is: the fixed arm now does ~50% more inner Newton
iterations per env-solve and pays ~7% wall for it. The wall number sits at
the edge of the repeat band; the iteration number does not and is the
mechanistic one. UNCHANGED by the tightening: 0 unconverged env-solves and
0 cost-plateau exits in 23,040 env-solves per block at BOTH tolerances --
the fixed arm was never using the slack it had.

### TASK 2 + 3 — F10 AND M2: THE FIXED ARM NOW CERTIFIES ITS OWN SOLVE AND
### IS CONTAINED. Implemented together because they are one mechanism.

WHY A READER WAS NOT ENOUGH, re-derived: `SolverSAP.last_converged` is
assigned from `solve_result.converged`, and the contact solve builds that
result with the literal `True` -- a constant, not a dropped signal. The
real per-env verdict is `contact_solve.converged_env`, which the adaptive
arm folds into `_solve_ok` (`_accumulate_solve_convergence`, the test
`converged_env[i] == 0`) and turns into a rejection. THE FIXED ARM NOW
READS THE SAME ARRAY WITH THE SAME TEST, so the two arms certify the same
quantity by the same arithmetic.

THE RESPONSE, and the split. Containment is the default, mirroring the
adaptive solver's own `NEWTON_SAP_CONTAINMENT` gate:
  GIVEN to the fixed arm (the separable half):
    * per-world failure detection off converged_env;
    * a per-world failure latch;
    * bitwise state freeze -- the latched world is excised from every
      later substep via SolverSAP's own `world_active` mask;
    * world isolation;
    * the pending mask `physics_diverged` consumes, so the episode ENDS
      instead of training on a solve that never converged;
    * per-world release on env reset;
    * strict converge-or-throw under NEWTON_SAP_CONTAINMENT=0, quoting the
      tolerance, exactly as the adaptive arm does.
  WITHHELD deliberately (and this is the thesis, not an oversight):
    * THE dt SHRINK-RETRY. Choosing a smaller step on rejection IS
      adaptivity. Handing it to the fixed arm would delete the result the
      comparison exists to measure. The fixed arm therefore latches on its
      FIRST failure where the adaptive arm latches only after the shrink
      ladder reaches the dt floor -- which is the correct mirror for an arm
      whose dt set has exactly one element, and is stated here because it
      is a real asymmetry in the fixed arm's disfavour.

PROOFS, all on final bytes, 8 envs, det=1, 40 steps (p29_cont_*.json/npz):
 (a) VACUITY GUARD. Control (shipping cap): 0 latches, 0 terminations, 0
     unconverged samples. A certificate that always fires proves nothing.
 (b) GENUINE FAILURE, NOTHING FORGED. Inner Newton cap set to 0 after
     construction, so no contacting env can reach its optimality test at
     any dt: all 8 worlds latched from step 0 and the mask carried them on
     every one of 40 steps.
 (c) THE TERMINATION FIRES, AND IT IS THE RIGHT ONE. Same forced failure
     with terminations live: 320 done flags over 40 steps x 8 envs, and
     per-term attribution is physics_diverged 320, time_out 0,
     object_dropping 0, object_off_table 0, object_speeding 0,
     robot_abnormal 0. The run completed with no raise (containment).
 (d) STRICT MODE RAISES. NEWTON_SAP_CONTAINMENT=0 on the same scene:
     RuntimeError "SolverSAP inner SAP solve failed to converge to
     optimality_rel_tol=1.000e-08." at the first step.
 (e) FREEZE + ISOLATION, BITWISE. The per-env verdict for ONE world is
     forced to 0 at ONE step immediately before the certificate reads it
     (the CAUSE is forged, the response is the real kernel). Exactly that
     world latched, from that step, for 30 steps. Then, against a control
     run of the identical scene in a separate process:
       ORACLE  control vs its repeat bitwise identical over 40 steps on
               joint_q, joint_qd and the manipulated object's root state;
       FREEZE  the latched world byte-frozen at its step-10 state through
               step 39 on all three fields;
       ISOLATE all 7 healthy worlds bitwise identical to the control run.
 (f) AT SCALE, IT IS SILENT: 0 worlds flagged in the 1024-env
     characterization runs and 0 in the 3-iteration training runs.

RESIDUAL, NAMED AND MEASURED: the fixed arm steps IN PLACE, so the failing
substep's own result is committed once before the verdict exists; the
freeze starts at the step containing the failure, not before it. Measured
size of that single committed substep, latched world vs control at the
failing step: joint_q 6.26e-2, joint_qd 2.16e-1, object root 1.48e-6 --
against a NORMAL step of that world of 7.20e-2, 2.71e-1, 1.19e-6. So it is
one substep's worth of motion, not an excursion. Closing it needs a
per-substep state snapshot on the fixed arm, which costs the fixed arm
wall time and would therefore flatter adaptive; NOT DONE, and left as
Marco's (D4b below).

CAVEAT THAT MUST NOT BE LOST: the fixed arm now always receives a
`world_active` mask where it previously received None, and it now solves to
a different tolerance, so ITS trajectories are not bitwise comparable to any
pre-pass-29 fixed-arm run. That is fine -- the fixed arm has no campaign
record to preserve, which is exactly why the grant allowed changing it --
but no fixed-arm number from passes 26-28 may be diffed against a number
from here on. The ADAPTIVE arm's record is the one that is protected, and it
is intact.

COST OF THE WHOLE CONTAINMENT PATH: nothing measurable. At MATCHED
tolerance (1e-6 both sides) the fixed arm is 34.05 ms/step after the edits
against 34.74 before -- 2% FASTER, inside a 4.5-7.1% repeat band, and the
sign is plausible because the same commit removed a per-step device sync
(F12 below). The certificate is one kernel launch per substep with no host
read; strict mode's host read is opt-in.

### TASK 4 — M3, DETERMINISM PROPAGATION: CLOSED WITHOUT TOUCHING THE
### SHARED CFG'S MEANING FOR MUJOCO.

NEWTON_SAP_DETERMINISTIC reached SolverSAPAdaptive's own pipeline and not
the manager pipeline the fixed arm consumes. It is now propagated in
`_apply_fixed_sap_pipeline_overrides`, applied to a COPY of the cfg and
gated on `cls._sap and not cls._adaptive`, so the MuJoCo backend's
trajectories are untouched (pass 27's blocker).

READ OFF THE NARROW PHASE EACH ARM RUNS (p29_scale_det*.json), not off the
patch:
    fixed, det=1     pipeline deterministic True,  reducer deterministic True
    fixed, det unset pipeline deterministic False, reducer False
    adaptive, det=1  pipeline deterministic True,  reducer True
And the consequence is the thing that matters: the fixed arm is now
BITWISE REPRODUCIBLE ACROSS PROCESSES (the oracle in (e) above, 40 steps x
8 worlds x 3 fields). That reproducibility is what made the freeze and
isolation verdicts judgeable at all.

### TASK 5(a) — THE DOCUMENTED LAUNCH PATH: FIXED AND EXERCISED.

`_validate_solver_substeps` tested `solver_cfg.adaptive` (the MuJoCo
latch) and therefore rejected the SAP step-doubling solver on the 1-substep
boundary it is entitled to own. Now tests `adaptive or sap_adaptive`. One
task-file edit, nothing else in that file touched.

PROVEN BY RUNNING THE DOCUMENTED INVOCATION, no env-var workaround
anywhere (p29_train_sap_adaptive.log):
  ./isaaclab.sh train --rl_library rsl_rl --task IsaacContrib-Lift-Spatula-
  Trossen-v0 --seed 42 --num_envs 64 --max_iterations 3
  --solver sap-adaptive physics=newton_mjwarp_adaptive
resolves `{'backend': 'sap', 'adaptive': False, 'sap_adaptive': True}` and
completes 3 learning iterations. The fixed arm's documented path
(`--solver sap`, task default 2 substeps) likewise completes 3 iterations.
physics_diverged 0.0000 on both.
SIDE EFFECT WORTH RECORDING (F6): because these runs need no env override,
their params/env.yaml records `backend: sap` and the right `sap_adaptive`
-- the provenance hole closes for anything launched the documented way. It
does NOT close for env-var launches, and the dump still records AUTHORED
rather than RESOLVED pipeline values (192,000,000 pairs, deterministic
false, whatever the manager then resolves).

### TASK 5(b) — THE TRIANGLE-PAIR CAP: SCENE-SIZED MANAGER-SIDE.
### 12.5 GB FREED AT 1024, AND 4096 NOW CONSTRUCTS.

RULE (mjwarp_manager, one helper used by both arms):
    max(1M, min(authored, 16384 * world_count)), clamped to 1 << 25 when
    deterministic (the reducer REJECTS a larger capacity outright).
16384/world is the per-world budget; the authored value remains a ceiling,
so a task can still ask for less.

MEASURED DEMAND (live `triangle_pairs_count`, not modelled): peak 3,687
pairs/world at 64 envs during scripted flail; 1,598-1,638/world at 1024 and
4096 under press. The budget is 4.4x the flail peak and ~10.2x the press
peak. Truncated contacts 0 on both arms in every probe this pass.

    [CORRECTION, pass 31, in place. ORIGINAL CLAIMS: 16,384/world is
    "4.4x the flail peak and ~10.2x the press peak", "truncated contacts
    0 on both arms in every probe", and (section header) "4096 NOW
    CONSTRUCTS". THE ERROR: every demand number above comes from a
    SCRIPTED action stream. A learned policy holds the mug against the
    gripper in all 1024 worlds simultaneously and reaches mesh proximity
    no scripted rig in this campaign reached. CORRECTED VALUES, measured
    pass 31 under the pass-30 trained policies at 1024 envs with the
    pool oversized so the trajectory is not itself degraded
    (p31_demand_{fixed,adapt}_m299.json, 4800 collides each): peak
    demand 39,977 pairs/world (fixed arm, model_299) and 36,679/world
    (adaptive arm, model_299) -- 10.8x and 9.9x the scripted flail peak,
    and 2.44x / 2.24x the budget this section landed. The "truncated
    contacts 0" claim is true OF THE PROBES and false of training: pass
    30 measured both arms overflowing from iteration ~95-110. The budget
    is raised to 65,536/world in pass 31 and 4096 no longer constructs
    at it; the 12.47 GB freed at 1024 and the 4096-OOM-at-authored
    measurement are unaffected and stand.]

MEASURED FOOTPRINT (device memory in use after build, same scene, the ONLY
difference being the sizing rule -- the uncapped arm monkeypatches the
helper back to the authored value in the probe, so this is a difference
between two builds, not an allocation formula):
    1024 adaptive, capped   16,777,216 pairs    7.350 GB
    1024 adaptive, uncapped 192,000,000 pairs  19.818 GB   -> 12.47 GB FREED
    1024 fixed,   capped    16,777,216 pairs    7.725 GB
    4096 adaptive, capped   67,108,864 pairs   25.32 GB, constructs and steps
    4096 adaptive, uncapped RuntimeError: Failed to allocate 2304000000
                            bytes on device 'cuda:0' -- and 2.304e9 = 192e6
                            x 12 B, i.e. the OOM IS the authored pair cap.

THE ADAPTIVE ARM'S OWN PIPELINE TAKES THIS RULE TOO, so it is a change to
the campaign arm and had to be proven byte-neutral rather than argued: the
phi0 rig runs det=1 at 8 worlds, where the capacity moves from 33,554,432
(the old deterministic clamp) to 1,000,000, and phi0 is IDENTICAL TO THE
DIGIT (-5.584e-5 deepest, -2.756e-5 P5, every phase). Gate results below.

### TASK 5(c) — F12: THE FIXED ARM'S PER-STEP HOST SYNC IS GONE.

The reset hook read the per-world mask on the host every reset boundary
(`local_mask.numpy().any()`) to decide whether to clear a warm start that
is ONE global flag. Replaced by `SolverSAP.reset_runtime_state_masked`, a
one-kernel device-side test with identical semantics (sap_warp afd5dc6).
The host-side diagnostic counters are deliberately not cleared: the step
path never reads them, and a staggered per-env reset has no business
zeroing a global clock. Effect is inside the repeat band; it is hygiene,
and it makes the fixed arm no slower, which is the only direction allowed.

### GATES (full 8-gate chain, adaptive arm, on the final bytes, strictly
### sequential with abort-on-failure). ALL PASS.

 G1 construct  PASS (SAP-NEWTON15-CONSTRUCT), ACR defaults ON with the
    constitutive dt wired into the contact solve.
 G2 flag-equivalence  PASS. Every bitwise arm -- acr, fusedls, fusedam,
    fusedup, narrowv3, runahead, graph and conditional forms -- bitwise
    identical to its reference over 6 boundaries. Equivalence iterations
    per boundary reference [11,4,4,3,3,2] == boundary [11,4,4,3,3,2].
 G3 march-equivalence  PASS, fingerprint [6, 25, 20, 24, 19] -- UNCHANGED
    since pass 13. compact / conditional / compact+conditional all bitwise.
 G4 determinism  PASS, bitwise over 20 steps at 256 envs, seed 7, det=1,
    on cum / fail_step / joint_q / joint_qd / body_q / body_qd.
 G5 containment  PASS[b] world 2 latched at boundary 0, state frozen and
    clock pinned through boundary 29; PASS[c] all 5 healthy worlds bitwise
    identical to control over 30 boundaries x 10 fields; PASS[d] strict
    raised quoting optimality_rel_tol=1.000e-08.
 G6 err_tol  0 violations / 2880 samples; max accepted err/tol 0.7135;
    per-phase rest 0.0179 / press 0.6137 / swing 0.7135 -- identical to
    every recorded pass since cert_g5. floor samples 0, capped boundaries
    0, unfinished worlds 0, dt samples below 1e-4 = 0, dt_run_min
    2.5276e-3 (25x Marco's criterion). Rails live: tol 1e-3, dt_min 1e-12,
    inner_rel_tol 1e-8, solve fp64, contact_solve fp64.
 G7 rest smoke  PASS, z in [0.0198, 0.0210], 0 early terminations.
 G8 phi0  deepest -5.584e-5 and median boundary P5 -2.756e-5 in EVERY
    phase, narrow-v3 ON and OFF identical. These are the pass-25 and
    pre-campaign ACR-ON values to the digit.
ADAPTIVE ARM: BYTE-UNCHANGED. Its solver construction is
argument-for-argument what it was except for the triangle-pair capacity,
and the penetration, march-fingerprint, determinism and err/tol gates all
land exactly where the campaign record left them.

### CHARACTERIZATION AT MATCHED SCALE — NOT THE KILLER EXPERIMENT

1024 envs, 120 timed steps after 20 warmup, seed 42, the SAME uniform
random action stream driving both arms, sequential, one GPU process at a
time, two repeats each (p29_char_*.json). Both arms now solve to
optimality_rel_tol 1e-8; both report 0 worlds flagged diverged and finite
state at the end.

                         ms per env step   accepted world-substeps
                                           per world-boundary
  FIXED  (2 substeps)    50.47 / 50.49     2.0000
  ADAPTIVE (march)       79.12 / 79.31     7.7417

                         us per ACCEPTED world-substep
  FIXED                  24.65 / 24.65
  ADAPTIVE                9.98 / 10.01

Demand-normalized, the adaptive arm buys an accepted world-substep for
0.405x what the fixed arm pays (2.47x cheaper per unit of integration
work), and spends that advantage plus more on 3.87x the substeps, netting
1.57x the wall per env step in this regime. Repeats agree to 0.05%.

    [CORRECTION, pass 30, restated in place by pass 31 so no reader
    meets these numbers without the retraction. ORIGINAL CLAIMS: the
    adaptive column "7.7417 accepted world-substeps per world-boundary"
    against the fixed arm's 2.0000; "3.87x the substeps"; "0.405x the
    cost per accepted substep (2.47x cheaper per unit of integration
    work)"; and the fixed arm's "24.65 us per ACCEPTED world-substep".
    THE ERROR: the two columns are in different units. `num_substeps`
    is substeps per PHYSICS BOUNDARY, while `cumulative_accepted_steps()`
    accumulates over a whole ENV STEP, and one env step contains
    `decimation` = 4 boundaries — so the adaptive column was 4x too
    large and the fixed per-substep price 4x too high. CORRECTED VALUES
    (pass 30 COUNTED the boundaries through the manager's own solver
    entry point, p30_demand_norm_probe.py / p30_char_*.json, same scale
    and conditions): accepted substeps per world-boundary FIXED 2.0000,
    ADAPTIVE 1.9354; us per accepted world-substep FIXED 6.153/6.170,
    ADAPTIVE 9.972/9.987. So the two arms do essentially the SAME
    integration work per boundary (adaptive 3.2% FEWER accepted
    substeps, at a 3.3% LARGER mean substep) and the adaptive arm costs
    **1.62x MORE** per accepted world-substep, not 2.47x less. The 1.57x
    wall ratio is almost entirely per-substep PRICE, not extra work.
    The raw ms/env-step column above reproduced to within 0.2% and
    STANDS; only the accepted-substep column and the two ratios derived
    from it are withdrawn.
    SCOPE OF THE ERROR, swept by pass 31: it reaches ONLY
    fixed-vs-adaptive normalizations. Every WITHIN-adaptive-arm
    demand-normalized number in this ledger — pass 13's flip A/B
    ("ms/substep 7.002 vs 7.845"), pass 14's ACR plateau ("ms/substep
    7.25 vs 8.20"), pass 23's run-ahead plateau ("-4% to 0 per accepted
    substep") and pass 28's "2% demand-normalized bar" — divides one
    `cumulative_accepted_steps()` by another, so the decimation factor
    cancels exactly and those numbers are UNAFFECTED. Checked, not
    assumed.]
WHAT THIS IS NOT: a uniform-random action stream is a violent regime that
inflates adaptive demand; the fixed arm's 2 substeps are cheap precisely
because nothing bounds their error, which is the question the killer
experiment exists to answer and this probe does not. It is a speed and
demand characterization of two arms that now solve the same problem to the
same tolerance -- nothing more.

### WHAT THIS PASS DID NOT DO, ON PURPOSE

  * No dt shrink-retry for the fixed arm (see Task 3).
  * No change to contact stiffness or tau_d. Entering the near-rigid CENIC
    regime is a research-design choice off a measured sweep, not a defect
    fix; D12 stands untouched.
  * No estimator, tol, dt_inner_min, cap, contact-law or ACR change.
  * Run-ahead default still OFF; twins still twins; nothing pushed.
  * D7, the matched fixed-vs-adaptive TRAINING comparison, was NOT run.
    It is now unblocked on every axis this pass could reach, and it is
    Marco's experiment to launch.

Provenance (all p29_ prefix, no p13-p28 artifact overwritten):
p29_tol_probe.py + p29_tol_fixed_{pre,post}.{json,log};
p29_arm_probe.py + p29_arm_{fixed,adaptive}.{json,log};
p29_containment_probe.py + p29_containment_chain{,2}.sh +
p29_cont_{control,control2,inject,forcemask,force,strict}.{json,log,npz} +
p29_cont_compare.py; p29_scale_probe.py + p29_scale_chain.sh +
p29_scale_{det1_fixed,det0_fixed,det1_adapt,a1024_capped,a1024_uncapped,
a4096_capped,a4096_uncapped,f1024_capped}.{json,log};
p29_char_probe.py + p29_char_chain.sh + p29_char_{fixed,adaptive}{1,2}.{json,log};
p29_train_chain.sh + p29_train_sap_{adaptive,fixed}.log;
p29_gates.sh + p29_g{1..8}_*.{json,log} + p29_gate_progress.txt.

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

VALIDITY RED LINE (added pass 25, standing): the step-doubling
estimator, tol 1e-3, optimality_rel_tol 1e-8, dt_inner_min 1e-12, the
contact law (R construction, contact_k, tau_d, beta, sigma) and the
fixed-vs-adaptive comparison semantics are the PHYSICS BEING
DEMONSTRATED, not optimization surface. No wall-clock target — 10 s,
7 s, or lower — authorizes touching them. A pass that concludes the
target is reachable only through one of them must ESCALATE to Marco
and land nothing. The experiment's integrity outranks the wall-clock
goal.

## Escalations to Marco (decisions only he makes)

- ESTIMATOR STRUCTURE (pass 18, reframed pass 20): the step-doubling
  3-solve estimator (excluded rail, comparison semantics) remains the
  route BEYOND ~10 s — single-solve arithmetic ~1.20 ms/substep vs
  the current 2.75. Pass-20 arithmetic makes ~10 s reachable WITHOUT
  it (FWBD narrowing + overlap); whether estimator-semantics changes
  are on the table is still his call; nothing has been touched.
- OVERLAP MID-WINDOW VISIBILITY (pass 20; built pass 22; DECONFOUNDED
  pass 23 — the value is REVISED DOWN; the flip stays one line:
  NEWTON_SAP_RUNAHEAD=1 with window=decimation=4, phase 0, throttle
  defaults 0.5/2). What Marco is consenting to, measured: mid-window
  scene.update/sensor reads see run-ahead worlds at mixed boundary
  times inside one action window; action-edge states stay
  batch-synchronized and per-world physics is bit-preserved
  (oracle: batch==solo bitwise; ON-vs-OFF edge positions bitwise,
  velocities at the 4e-9 clock-sliver class; phi0 anchoring identical
  to the digit; containment latches and isolates; window edges
  bitwise-invariant to the throttle's gate rules). These reads are
  dead in this task (contact-sensor rewards are latest-value at
  action cadence, no history terms) — that invariant is task-level
  and should ride the flip as a launch-checklist assert. VALUE,
  demand-normalized (pass 23): plateau -4%..0 per accepted substep
  (the pass-22 "-16.9%" was demand draw); wide/flail regime -10%
  matched-window wall with the throttle; det=1 whole-run -2.7%. The
  honest pitch is faster EARLY training at certified semantics, not a
  plateau lever. CONSENT-RELAY NOTE (2026-08-16): a coordinator
  message relayed Marco's reply "amazing you have all perms" to a
  report that the flip awaited his OK. Recorded verbatim as directed;
  NOT treated as consent to flip: it is agent-relayed rather than
  Marco's own message in this channel, it reads as acknowledging the
  standing solver-change grant rather than answering the mid-window-
  visibility question, and it predates the deconfounded numbers
  above, which materially weaken the case the pass-22 question
  advertised. The flip waits on Marco against THIS pass's numbers.
- TRIANGLE-PAIR CAP RIGHT-SIZE -- CLOSED PASS 29, manager-side (the
  task line is now cosmetic; see D10). Original statement:
  (unblocks 4096 det-unset + frees
  ~11 GB @1024): the task cfg authors max_triangle_pairs=192M, sized
  blind in the always-det era when the CONTACT_ID_BITS clamp silently
  capped it at 33.5M; live demand ~2.8M. Authoring ~8-16M (3-6x margin)
  restores the historical footprint under det-off and un-OOMs 4096.
  One task-config line; task changes are Marco's.
- FIXED-ARM INNER TOLERANCES -- CLOSED PASS 29 by lifting the fixed arm
  to the 1e-8 rail; cost measured (see D1). Original statement:
  (F8, raised pass 26, re-confirmed as the
  loudest row of the pass-27 matrix): the fixed arm solves to
  optimality_rel_tol 1e-6 with cost tols 1e-30/1e-15 where the adaptive
  arm pins 1e-8 with the cost early-exit disabled. One kwarg triple at
  mjwarp_manager.py:382-390. Red-line item, so no loop may land it.
- physics_diverged -- CLOSED PASS 29 FOR THE FIXED SAP ARM (certificate +
  containment); STILL OPEN ON BOTH MUJOCO ARMS' FIXED SIDE, see D2b.
  Original statement: A NO-OP ON THE FIXED ARM AND ON MUJOCO (pass-27
  live confirmation of F5): an active termination term that can excise a
  broken world on one arm only. Give the fixed arm an equivalent or state
  it in the paper text.
- CONTACT-DERIVED MDP TERMS ARE NOW A TRAP (pass 27, F9): contacts.force
  is live on MuJoCo (flail peak 8.96) and identically zero on both SAP
  arms. Nothing reads it today, so the backends are equivalent; the eight
  unreferenced contact functions already sitting in the task's mdp.py
  would each silently read a constant zero under SAP. Any decision to add
  one needs the writeback (M2 of pass 26) first.
- Phantom body fix: follower_left_ee_gripper_link active=false overlay
  (one line in stationary_ai_task.usda, mirrors right twin) — kernel-width
  savings 7/22 coords per world + closes latent-risk surface
- THE FOUR-ENGINE COMPARISON'S TASK-SIDE GATE (pass 35, B1-B4): the PhysX
  preset already exists (trossen_spatula_lift_env_cfg.py:172-178) and has
  never run once on this machine. Two task edits unblock it — make the two
  Newton-pinned contact sensors and the Newton visualizer preset fields so
  they are absent under the PhysX alternative — plus a material/contact-offset
  authoring pass so the PhysX arm does not silently run on the engine default
  friction (mu 0.5) against the Newton arms' 1.0. Until then the campaign runs
  p35_threeway_screen.yaml, which needs no edit anywhere.
- "AND RUN IRL" IS NOT TRUE OF THIS SCENE (pass 35, M1-M8, in that priority
  order): zero domain randomization of any kind, `enable_corruption=True` with
  every `noise=None` so it is a decoy, object pose observed as simulator ground
  truth with no perception path, mu 1.0 undifferentiated everywhere, and a
  contact/drive stiffness pair (1250 / 1000 N/m) that puts the finger 44% of
  the remaining commanded stroke inside the mug. M1 (randomization) and M2
  (perception) are prerequisites for transfer, not improvements. None of them
  blocks the four-engine comparison, which needs the MDP identical across arms
  rather than realistic.
- Any push to GitHub (auth broken; needs gh auth login first)

## DECISION SUMMARY — EVERYTHING STILL WAITING ON MARCO
## Rewritten 2026-08-16 at the end of pass 29, which landed the fairness
## fixes his "you have all perms keep going" released. Ordered by how much
## each can still change the paper's claim. Items closed by pass 29 are kept
## with their outcome so the record does not lose them. Each item: what it
## is, the exact change, where it lives, what it costs or buys — measured
## where measured, and explicitly marked UNMEASURED where not.

D7  THE MATCHED FIXED-vs-ADAPTIVE COMPARISON. **NOW THE TOP ITEM: every
    blocker this campaign could reach is gone, and it has still never been
    run.** Both SAP arms now construct, step and train from the documented
    CLI; both solve to optimality_rel_tol 1e-8 with the cost exit disabled;
    both carry the same contact law, contact capacity, collision-pipeline
    sizing, determinism resolution and convergence guarantee; and 4096 envs
    now constructs.
    CHANGE: run both arms back to back, same seed, same iteration count,
    W&B + video per the standing rule.
    WHERE: the existing entrypoint --
      ./isaaclab.sh train --rl_library rsl_rl --task
      IsaacContrib-Lift-Spatula-Trossen-v0 --solver sap  ... and
      --solver sap-adaptive physics=newton_mjwarp_adaptive
    COST: pass-29 characterization at 1024 envs measured 50.5 ms/env-step
    fixed and 79.2 adaptive on a scripted stream, so a 40-iteration arm is
    minutes, not hours.
    WHAT IT IS NOT YET: nothing in this campaign has compared LEARNING
    outcomes on SAP. This is the experiment the paper rests on.

D2b physics_diverged IS STILL A NO-OP ON THE MUJOCO ARMS' FIXED SIDE.
    Pass 29 gave the fixed SAP arm the whole containment half it could have
    (detection, latch, freeze, isolation, termination) and the MuJoCo fixed
    arm none of it, because it has no verified per-world convergence verdict
    to read. The historical killer experiment as actually executed was
    MuJoCo fixed vs MuJoCo adaptive, so on THAT pairing the asymmetry pass
    25 flagged as F5 is still fully present.
    CHANGE: find MuJoCo-Warp's per-world solver status equivalent and wire
    it the same way, or state the asymmetry in the paper text.
    WHERE: mjwarp_manager `_build_solver` / `_step_solver`, MuJoCo branch.
    COSTS/BUYS: UNMEASURED. Firing rate on the SAP arms is zero in every
    regime tested, so this is a guarantee gap, not an observed bias.

D4b THE FIXED ARM COMMITS ONE SUBSTEP BEFORE IT CAN KNOW THE VERDICT.
    The convergence certificate is consumed after the substep that failed
    and the fixed arm steps in place, so the failing substep's result lands
    once and the freeze starts from there. MEASURED size of that single
    committed substep (latched world vs control at the failing step):
    joint_q 6.26e-2, joint_qd 2.16e-1, object root 1.48e-6 — against a
    normal step of that world of 7.20e-2, 2.71e-1, 1.19e-6.
    CHANGE: snapshot per-world state before each fixed-arm substep and
    restore the failing worlds.
    COSTS/BUYS: buys exact parity with the adaptive arm's never-commit
    contract. COSTS the fixed arm wall time on every step to protect
    against an event measured at rate zero — i.e. it makes the baseline
    slower, which flatters adaptive. Deliberately not done for that reason.

D4c THE FIXED ARM LATCHES ON ITS FIRST FAILURE; THE ADAPTIVE ARM LATCHES
    ONLY AT THE dt FLOOR. This is the withheld shrink-retry, and it is the
    thesis: an arm with one dt has no ladder to climb down. Recorded so the
    paper states it rather than implying the containment is symmetric.

D5  REAL CONTACT-FORCE WRITEBACK, AND THE CONTACT-DERIVED-TERM TRAP
    (pass-26 M2 + F9). UNCHANGED by pass 29.
    WHAT: both SAP arms' `update_contacts` are documented no-ops, so
    `contacts.force` is identically zero on SAP and live on MuJoCo
    (measured flail peak 8.96). Nothing reads it today, so the backends are
    equivalent by coincidence — and eight contact-reading functions already
    sit unreferenced in the task's mdp.py.
    CHANGE: `Contacts.force[g] = R_WC[env,slot] @ gamma[env,slot] / dt`.
    WHERE: sap_warp solver_sap.py + newton-adaptive solver_sap_adaptive.py;
    the arms need DIFFERENT kernels (different Contacts providers).
    COSTS/BUYS: on the adaptive arm the committed impulse is not
    identifiable from `last_contact_solve_result`, so making it so is new
    plumbing in the estimator's committed-attempt bookkeeping = RED LINE.

D8  ACR DEFAULT (F3). LEAVE IT ON — recorded, not open. UNCHANGED by 29.
    Holding capacity unchanged; spurious creep 6.8x larger (36 vs 5.3 um/s
    at 0.34 g, ~180 vs ~26 um over a 5 s episode, against a 5-8 cm decision
    scale); normal penetration +3.5% deepest, +0.04% P5.
    REVISIT ONLY IF the claim depends on sub-millimetre positional fidelity.

D9  RUN-AHEAD DEFAULT FLIP. UNCHANGED by 29, still OFF, still Marco's.
    Demand-normalized value (pass 23): plateau -4%..0 per accepted substep;
    wide/flail -10% matched-window wall; det=1 whole-run -2.7%. Pass 28 adds
    why the plateau value is small: at few crossing worlds ~83% of a masked
    collide is un-narrowable floor. The honest pitch is faster EARLY
    training at certified semantics, not a plateau lever. The relayed
    "amazing you have all perms" was recorded but NOT treated as consent to
    this specific flip.

D10 TRIANGLE-PAIR CAP. **CLOSED BY PASS 29, MANAGER-SIDE.** Rule:
    max(1M, min(authored, 16384 * world_count)), clamped to 1<<25 when
    deterministic. MEASURED: 12.47 GB freed at 1024 envs (19.818 -> 7.350
    GB); 4096 envs now constructs and steps at 25.32 GB where the authored
    192M cap died with "Failed to allocate 2304000000 bytes"; live demand
    peaks at 3,687 pairs/world (flail, 64 envs) against the 16,384/world
    budget; truncated contacts 0 on both arms.
    RESIDUAL, MARCO'S: 16,384/world is OUR constant, so it is unvalidated
    by definition. It is 4.4x the worst demand this campaign has measured,
    but no trained policy has been run at 4096, and pair-cap overflow drops
    mesh contacts SILENTLY. If 4096 ever runs long, watch it.
    The task cfg still AUTHORS 192,000,000; the manager caps it. Editing the
    task line is still Marco's and is now cosmetic.
    [SUPERSEDED, pass 31. The residual above fired, and not at 4096: the
    constant is 2.44x TOO SMALL at 1024 under a trained policy (measured
    peak 39,977 pairs/world, fixed arm), and pass 30's D7 runs spent
    their back two thirds on silently truncated contact sets. Raised to
    65,536/world in pass 31 against that measurement. CONSEQUENCE, stated
    rather than hidden: at 65,536/world the 4096 build resolves to the
    task's authored 192M and 4096 no longer constructs on a 32 GB device
    -- see the pass-31 entry for the measured memory ladder and for why
    an honest 4096 is out of reach at any rule on this hardware.]

D11 PIN sap_warp TO newton-adaptive (F7). UNCHANGED, and MORE pressing:
    pass 29 changed sap_warp again (afd5dc6). sap_warp is joined by
    SAP_WARP_PATH on sys.path, so any sap_warp commit silently changes the
    physics under an unchanged newton-adaptive HEAD.
    CHANGE: submodule, or record the resolved commit hash in run artifacts.

D12 THE CENIC REGIME IS NOT THE ONE BEING RUN (F4). CHARACTERIZATION,
    UNCHANGED, and explicitly NOT touched by pass 29 on instruction.
    The scene runs COMPLIANT: 11.09% of contacts in the near-rigid branch
    overall, 0.25% in flail, d ln k_eff / d ln dt = -0.0031 in flail. Any
    adaptive advantage this stack demonstrates is truncation-error control
    alone. Entering the paper's regime needs authored contact stiffness up
    ~6-7 orders AND tau_d << dt — a claim-scoping decision off a measured
    sweep, not a defect fix.
    [CORRECTION, pass 31, in place. ORIGINAL CLAIM: "authored contact
    stiffness up ~6-7 orders". THE ERROR: inherited from pass 25's
    AXIS-B paragraph, which reasoned the distance from sap_warp's 1e10
    fallback instead of measuring the branch boundary. CORRECTED VALUE
    (pass 30 sweep): **~1.3 decades** — the clamp takes the majority of
    contacts at k ~ 2.5e4 at the production substep and 100% by 1e6.
    This makes D12 MORE urgent, not less: the regime is one line of
    asset authoring away, not out of reach. The rest of D12 — that the
    production scene runs compliant — stands as measured.]

D13 PHANTOM BODY. One line in stationary_ai_task.usda:
    `follower_left_ee_gripper_link active=false`, mirroring the right twin.
    Buys 7 of 22 coords per world in kernel width. Asset change = Marco's.

D14 GITHUB PUSH. Every commit in all three repos is LOCAL. GitHub auth on
    this machine is broken (invalid gh token, no SSH keys, verified
    2026-08-08 and re-confirmed unchanged). Nothing can be pushed until
    `gh auth login`. Pass 29 adds IsaacLab 27cf9c1ec2, 4e8f763acb,
    6fdd695cda, 62c165b5f0 and sap_warp afd5dc6 to the unpushed set.

D15 RUN-ARTIFACT PROVENANCE (F6 residue). Launching by the documented CLI
    now dumps the right `backend` and `sap_adaptive` (pass 29 verified),
    but env-var launches still dump `backend: mujoco`, and the dump always
    records AUTHORED rather than RESOLVED values — params/env.yaml says
    192,000,000 triangle pairs and deterministic false whatever the manager
    then builds. CHANGE: dump the resolved solver + pipeline identity after
    `_resolve_solver_mode` and the pipeline overrides run.

CLOSED BY PASS 29, NO ACTION NEEDED (kept for the record):
  D1  fixed-arm inner tolerances — both arms now resolve
      optimality_rel_tol 1e-8 / cost 0.0 / 0.0 / cap 30, dumped live off
      both solver objects. Cost to the fixed arm: +49.8% inner Newton
      iterations, +7.2% wall at 64 envs; 0 unconverged solves and 0
      cost-plateau exits at either tolerance in 23,040 env-solves.
  D3  determinism propagation — NEWTON_SAP_DETERMINISTIC now resolves
      identically on both SAP arms' pipelines (read off the narrow phase),
      gated on the fixed SAP arm so the MuJoCo backend's meaning is
      untouched. The fixed arm is now bitwise reproducible across
      processes.
  D4a convergence certificate — the fixed arm reads the same per-env array
      the adaptive arm folds into its solve-ok state, latches, freezes,
      isolates and terminates; strict mode raises quoting
      optimality_rel_tol=1.000e-08. Proven under forced failure.
  D6  the documented sap-adaptive launch path — runs, 3 training
      iterations, no env-var workaround.
CLOSED EARLIER, NO ACTION NEEDED: F11 (warm-start asymmetry); matrix row 28
  (the fixed arm's host-synced reset gate — now deleted outright, pass 29);
  the env-axis launch class outside contact_solve.py.

## WHERE THE CAMPAIGN ENDED — an honest statement

It ended at a plateau of 12.9-15.1 s/iter at 1024 envs on the adaptive
arm (best estimate ~13-14; same-config draws move +-8% because demand
is draw-dependent), down from 35.35 s/iter at pass 14 — a real ~2.6x,
earned by kernel fusion, per-contact packing, GEMM truncation, env-list
narrowing and FWBD narrowing, every one of them re-measured on current
bytes and every one of them landed with an 8-gate chain. What is PROVEN:
no campaign optimization altered any physical invariant — penetration
(deepest phi0 -5.584e-5, boundary P5 -2.756e-5) is identical to the
digit to a genuine pre-campaign baseline 20 hours and 40+ commits
earlier; accepted err/tol has 0 violations in 2880 samples with the same
per-phase maxima recorded since cert_g5; the dt band never approaches
its floor; the march fingerprint [6,25,20,24,19] has not moved since
pass 13; the inner solve's 1e-8 convergence is enforced by a raise that
has been demonstrated live in the same session as the claim. What is NOT
proven, and must not be implied: the 7-10 s/iter target is NOT reachable
in-rails — every remaining lever with enough headroom runs through the
step-doubling estimator, which is the physics being demonstrated; the
killer experiment itself has never been run on SAP, because until pass
26 the fixed arm did not start and the two arms are still not solving to
the same tolerance; the stack is not in the near-rigid regime the CENIC
mechanism needs, so what an adaptive advantage here would demonstrate is
truncation-error control, not the paper's dt^-2 stiffness coupling; and
there is no pre-campaign SAP training baseline at all, so the campaign's
effect on LEARNING outcomes — as opposed to on physical invariants,
which is measured and null — is unquantified. The wall-clock work is
finished. What remains is not optimization: it is D1, then D7.

ADDENDUM, pass 29 (2026-08-16). D1 is done: both SAP arms now solve the
contact problem to optimality_rel_tol 1e-8 with the cost early exit
disabled, and the fixed arm pays ~50% more inner Newton iterations for it.
With that, and with the fixed arm's convergence certificate, containment,
determinism resolution, collision-pipeline sizing and documented launch
path all landed and gated this pass, WHAT REMAINS IS D7 ALONE — the
fixed-vs-adaptive TRAINING comparison, which has still never been run on
SAP. Everything the campaign can do to make that comparison honest has now
been done; whether it is worth running in a regime that is not the paper's
near-rigid one (D12) is the question that outranks it.

## PASS 30 — THE NEAR-RIGID REGIME SWEEP, AND THE FIRST FIXED-vs-ADAPTIVE
## TRAINING COMPARISON EVER RUN ON SAP (D7).
## 2026-08-16. MEASUREMENT ONLY: ZERO code edits in all three repos; the
## stiffness/tau sweep is driven entirely through runtime overrides of the
## material arrays the contact jacobian already reads, so no task, scene or
## contact-law byte moved. Stack verified clean at the certified HEADs before
## and after: newton-adaptive 34c94740 (march-counter-log), sap_warp afd5dc6
## (main), IsaacLab 62c165b5f0 (develop); GPU idle (438 MiB, 0 compute apps)
## at start; one GPU process at a time throughout (p30_*_progress.txt carry
## the interleaved nvidia-smi traces).
## Every constant this entry quotes from earlier passes was RE-MEASURED here;
## two of them turned out to be wrong and are corrected below.

### WHAT WAS RE-DERIVED, NOT INHERITED (p30_parity_{fixed,adapt}.json,
### read off the constructed solver objects under the production config)

    authored per-shape ke      2500.0 (all 274 shapes)  -> contact_k 1250 N/m
    authored per-shape tau_d   0.01   (all 274 shapes)  -> contact_tau 0.02 s
    sap_warp fallback_stiffness 1e10, fallback_tau_d 0.01
    beta 1.0   sigma 1e-3   preset approx32   contact solve fp64
    optimality_rel_tol 1e-8   optimality_abs_tol 1e-14
    cost_abs_tol 0.0   cost_rel_tol 0.0   max_iterations 30   line search 40
    contact buffer 2048/world on BOTH arms; live peak 54-134/world
    physics boundary 1/120 s; fixed arm 2 substeps of 4.1667 ms;
    adaptive arm marches the same 8.3333 ms boundary.

The live pair values k=1250 / tau=0.02 confirm pass 25's numbers
independently. The P5 penetration this pass measures at the authored law,
-2.7556e-5 m, agrees with the campaign's recorded gate value (-2.756e-5) to
every digit the ledger records -- on a DIFFERENT rig (32 worlds with a flail
phase, against the gate's 8 worlds without one), which is the cross-check
that the sweep's instrument reads the same quantity the gates read.

GPU EXCLUSIVITY, certified rather than asserted: every chain writes
`nvidia-smi --query-compute-apps` at each run's start and exit, and every one
of those samples in every p30_*_progress.txt is EMPTY -- zero compute apps on
the device at every boundary, so no two runs in this pass overlapped.

### TASK 1 — THE NEAR-RIGID REGIME SWEEP. VERDICT: **NO CROSSOVER EXISTS.**

METHOD, and why it is a legitimate override rather than a law change. The SAP
contact jacobian resolves per-shape stiffness and dissipation into
`contact_shape_ke` / `contact_shape_tau` (plus scalar fallbacks) at
construction and passes them to every contact kernel as launch arguments. The
sweep rewrites those arrays in place between rungs. The kernel combines the
two shapes' stiffness in series and SUMS their tau, so a target pair value
(k, tau) is driven by writing (2k, tau/2) per shape -- and the result is
VERIFIED LIVE at every rung by reading back `contact_env_k_wp` /
`contact_env_tau_wp`, which report exactly the requested pair value. R's
form, the projection ladder, beta, sigma, the estimator, tol, dt_inner_min
and the iteration cap were not touched.

GRID: k in {1250, 5e3, 1e4, 2.5e4, 5e4, 1e5, 1e6, 1e8, 1e10} at the authored
tau; tau in {0.02, 4e-3, 8e-4, 1.6e-4, 3.2e-5, 0} at the authored k and at
k = 1e6 / 1e10. ARMS: the adaptive march, and the fixed arm at 1, 2, 4 and 8
substeps per boundary. 32 worlds, 270 scripted steps (rest / press into the
table / swing / flail), one action stream replayed identically at every rung
and on every arm, seeds 42 and 7, NEWTON_SAP_DETERMINISTIC=1.

#### (a) WHERE THE NEAR-RIGID BRANCH ACTUALLY STARTS — pass 25's "~6-7
#### orders of magnitude" is WRONG BY FOUR TO FIVE DECADES.

Measured fraction of live contacts landing in the clamp, production fixed arm
(h = 4.1667 ms), tau = 0.02 (p30_reg_fixed_s2_seed42.json):

      k [N/m]     1250    5e3    1e4   2.5e4    5e4    1e5   >=1e6
      frac NR    0.000  0.436  0.458   0.497  0.899  1.000   1.000
      rn_hard/rn_soft (median)
                 0.050  0.199  0.399   0.997  1.994  3.989   >=39.9

The branch flips at k ~ 2.5e4 -- **1.3 decades above the authored 1250**, not
six. The crossover is not a constant: it is k_cross = 1/(h (h+tau) rn_hard),
so it moves with the substep size, measured at 1.1e4 (h=8.33 ms), 2.5e4
(4.17 ms), ~1e5 (2.08 ms) and ~4e5 (1.04 ms).

The census checks itself twice. (i) The branch fraction passes 0.5 (0.497) at
exactly the rung where the MEDIAN rn_hard/rn_soft passes 1 (0.997) -- two
statistics from different reductions of the same live arrays agreeing on
where the boundary is. (ii) On the ADAPTIVE arm at the AUTHORED law this
pass measures a near-rigid fraction of 0.1221 (seed 42) / 0.1187 (seed 7)
against pass 25's independently measured 0.1109 on the same arm with a
different phase mix -- so the classifier reproduces the campaign's own
production-regime number, which is what licenses using it 7 decades away.

#### (b) THE CLAMP IS A CEILING, AND THAT IS WHY NOTHING BREAKS.

Above the crossover the trajectory STOPS RESPONDING to stiffness. k = 5e3
through 1e10 -- 6.3 decades, up to sap_warp's own 1e10 fallback -- return
statistics identical to every digit recorded (7 significant figures) on the
production fixed arm:

      deepest phi0  -3.535455e-03 m   at every k from 5e3 to 1e10
      P5 phi0       -1.176450e-05 m   at every k from 5e3 to 1e10

(The contacts that DO change branch over that range are the ones carrying no
impulse: a separated contact's normal force projects to zero whatever R_n is,
so its regularization cannot reach the dynamics.)

The reason is structural, and it is the finding that decides this task:
R_n = max(rn_hard, rn_soft) with rn_hard = beta^2/(4 pi^2) W carries no k at
all, so once the clamp wins, k_eff = 4 pi^2 / (beta^2 W h (h+tau)) is a
function of the SUBSTEP and the Delassus diagonal alone. The clamp is a
CEILING on effective stiffness, and it is a low one.

HOW LOW, measured rather than modelled, from the P5 penetration of the
production fixed arm. The arm is position-controlled and the phase script is
identical across rungs, so to the extent the load at the penetration-setting
contacts is unchanged the penetration ratio is the inverse k_eff ratio -- and
the cross-check two paragraphs down is what says that assumption holds:

    authored k=1250, tau=0.02, compliant      P5 -2.7556e-05 m
    k -> 1e10 (fully clamped), tau=0.02       P5 -1.17645e-05 m   2.342x
    k -> 1e10 (fully clamped), tau=0          P5 -2.0303e-06 m   13.573x

So the full 6.9 decades from the authored 1250 to sap_warp's 1e10 fallback
buy a factor 2.34 in effective stiffness at the penetration-setting contacts,
and adding tau -> 0 takes the total to 13.57. That total is PREDICTED,
without fitting anything, by
k_eff ∝ 1/(h(h+tau)) on the clamped branch: the tau factor alone is
(h+tau)/h = 0.0241667/0.0041667 = 5.800, and 2.342 x 5.800 = 13.585 against
the measured 13.573 -- 0.09%. Two independently measured ratios and one
structural relation agree, which is the check that the clamped-branch k_eff
law is the one actually operating.

**There is no authored contact stiffness, at any magnitude, that makes this
scene's contact stiff enough to break a fixed step: the ceiling sits ~2.3x
above the authored value at the production tau, and ~13.6x with tau driven to
zero.** The only knob that would raise the ceiling is beta, a Drake constant
on the validity red line.

#### (c) THE PAPER'S EXPONENT IS REACHABLE, AND THE SWEEP REACHES IT.

d ln k_eff / d ln dt, measured per contact from the live (k, tau, W, dt),
production fixed arm at k >= 1e6 (100% of contacts in the clamp):

      tau [s]     0.02    4e-3   8e-4   1.6e-4   3.2e-5     0
      exponent  -1.172  -1.510 -1.839   -1.963   -1.992  -2.000

(the 3.2e-5 column is measured at k=1e10; the rest at k=1e6. They are the
same rung physically -- k=1e6 and k=1e10 at tau=1.6e-4 both return -1.963 to
four digits, which is (b)'s saturation showing up again.)

So the CENIC regime is not unreachable -- it is two knobs away, and it needs
BOTH. The tau axis alone moves the WRONG WAY: at the authored k = 1250,
lowering tau from 0.02 to 0 drops rn_hard/rn_soft from 0.050 to 0.0086 and
leaves 0.0% of contacts in the clamp, because rn_soft = 1/(h k (h+tau)) grows
as tau falls. High k is the necessary condition; low tau is what takes the
exponent from -1 to -2 once the clamp is already winning.

#### (d) THE EXPONENT FORMULA, VALIDATED AGAINST A STATE OBSERVABLE.

(c) is read off R's own structure, so on its own it is arithmetic, not
evidence. The independent test: run the SAME rung at different fixed substep
sizes and measure how the penetration the solver actually leaves behind
scales with h. Compliant branch predicts 0 (k_eff = k, dt-free); near-rigid
predicts 1 + h/(h+tau). Measured slope of ln|P5 phi0| vs ln h, fixed arm,
seed 42 (p30_sweep_analysis.py over p30_reg_fixed_s{2,4,8}_seed42.json):

    rung                     h 4.167->2.083 ms     h 2.083->1.042 ms
                            measured  analytic    measured  analytic
    k=1e6  tau=4e-3            1.423     1.424       1.270     1.269
    k=1e6  tau=0.02            1.131     1.128       1.076     1.069
    k=1e8  tau=0.02            1.131     1.128       1.076     1.069
    k=1e10 tau=0.02            1.131     1.128       1.076     1.069
    k=1250 any tau (compliant) 0.000     0.000       0.000     0.000

Agreement is better than 1% at every rung where the penetration signal sits
above trajectory noise, in both branches, across three values of tau. The
test never recomputes R; it constrains an output.

THE tau -> 0 CORNER, MEASURED PROPERLY. The step-to-step slopes above go bad
below tau ~ 8e-4 because the near-rigid penetration there falls to 1e-7 m and
the 5th-percentile contact stops penetrating at all (one cell returns
+7.45e-8 m, i.e. separated). The fix is a longer lever and a quasi-static
window: the settled tail of the PRESS phase, where the commanded pose is
constant and the penetration is a load/k_eff equilibrium, read across the
FULL 4x substep range rather than adjacent 2x steps. k = 1e10 (fully clamped
at every h), p30_qs_*.json:

   tau [s]  P5 at h=4.167ms  2.083ms   1.042ms   slope h4x  analytic  err
   0.02        -1.1765e-05  -5.372e-06 -2.548e-06   1.103     1.094   +0.8%
   4e-3        -3.9777e-06  -1.483e-06 -6.147e-07   1.347     1.342   +0.4%
   8e-4        -2.4196e-06  +7.450e-08 -2.271e-07   1.707     1.723   -0.9%
   1.6e-4      -2.1080e-06  -3.562e-06 -1.993e-07   1.701     1.929  -11.8%
   0           -2.0303e-06  -3.496e-06 -1.730e-07   1.776     2.000  -11.2%

So the measured penetration exponent DOES climb with falling tau, from 1.10
to 1.78, and tracks 1 + h/(h+tau) to within 1% over the first three rungs --
but it SATURATES near 1.7-1.8 and does not reach 2. **The paper's dt^-2
coupling is confirmed as a trend and NOT confirmed at its limit.** The two
short rungs are also where the h=2.083 ms cell is non-monotonic (deeper than
the coarser step), so the shortfall may be that cell rather than the law; at
0.17 um of penetration this is at the edge of what the rig resolves. Named,
not explained away.

#### (e) THE FAILURE CENSUS. NOTHING BREAKS THAT THE TASK WOULD ACTUALLY RUN.

physics_diverged counts, 32 worlds x 270 steps = 8640 world-steps per rung.
On the fixed SAP arm this term has EXACTLY ONE SOURCE, re-verified in source
this session (every writer of `_diverged_pending` in mjwarp_manager is either
gated on `cls._adaptive` or is `_latch_sap_solve_failure`): the manager latch
off `contact_solve.converged_env`. So each fixed-arm count is a named
mechanism -- "the inner Newton loop did not reach optimality_rel_tol 1e-8
within max_iterations 30" -- not a symptom.

THE TWO ARMS DO NOT LATCH AT THE SAME THRESHOLD, and the comparison is only
readable with that in front of it. The fixed arm latches on its FIRST
non-convergence because it has no other move. The adaptive arm rejects that
attempt, shrinks dt and retries, and only latches if the ladder reaches
dt_inner_min. That asymmetry is pass-29's D4c and it is deliberate -- the
shrink-retry IS adaptivity, and handing it to the fixed arm would delete the
result. It also means an adaptive zero is not by itself evidence that the
adaptive arm met an easier problem; it is evidence that it never ran out of
ladder.

    arm / substep h    rungs   rungs with   worst rung                non-
                       swept   any event    (k, tau) : events        finite
    fixed 1  8.333 ms    40         0       --                          0
    fixed 2  4.167 ms    40         1       (1e6, 4e-3) : 2             0
    fixed 4  2.083 ms    23         4       (>=1e6, 1.6e-4) : 21        0
    fixed 8  1.042 ms    29        10       (1250, 1.6e-4) : 620        0
    adaptive marched     46         0       --                          0

Every arm is finite at every rung; no contact buffer ever saturated (peak
134/world against 2048); object_off_table and object_speeding fire once each
in the whole sweep, both on the 8-substep arm at its worst rung.

The production fixed arm is NOT literally at zero: 2 events in 8640
world-steps at one rung (k=1e6, tau=4e-3) on seed 42, and 0 at that same rung
on seed 7 -- a rare-event floor, not a regime. Set against the 8-substep
arm's 620 at its worst rung, and against the adaptive arm's 0 in 46 rungs,
the shape of the result is unambiguous.

THE ONLY REAL FAILURES ARE ON A FIXED ARM MADE **FINER** THAN THE TASK
AUTHORS IT. They concentrate at one rung -- k = 1250, tau = 1.6e-4, 8
substeps -- and they reproduce across seeds (620 events seed 42, 588 seed 7),
with object ejection to 0.15 m / 0.37 m penetration and joint speeds of
13.3 rad/s against a normal 7.5. The same rung at 4, 2 and 1 substeps, and
the adaptive arm, are all clean. And the adaptive arm's own accepted substep
is COARSER than the failing one -- 6.4 ms mean in this sweep rig, 4.31 ms in
the 1024-env characterization, against the 1.04 ms that fails -- so the one
place a fixed arm fails and adaptive holds is a place adaptive holds by
taking BIGGER steps. That is the opposite of the mechanism the claim needs.

The failure count is also non-monotonic in tau at fixed h (620 at
tau=1.6e-4 against 27 at tau=0 and 2 at tau=8e-4), so it is not a simple
"less dissipation is worse" law and MUST NOT be reported as one.

THE DIAGNOSIS, because "the fixed arm failed" is not a finding until the
mechanism and the causal order are known (p30_diag_probe.py traces EVERY
substep, not every env step, at the worst rung and at the same rung on the
production arm):

                                   s8 (h=1.042ms)   s2 (h=4.167ms), same rung
    substeps traced                     8640            2160
    substeps with an unconverged env      87               0
    of those, at the 30-iteration cap     87               0
    inner Newton iterations, mean/max  2.70 / 30       1.51 / 17
    first failure                    trace 112, env step 3    never
    deepest phi0 BEFORE first failure  -2.93e-04 m     -3.50e-04 m (whole run)
    deepest phi0 overall               -2.27e-01 m     -3.50e-04 m
    physics_diverged terminations          616               0

Three things are settled by that table.
  1. THE MECHANISM IS THE ITERATION CAP, exactly. Every one of the 87
     unconverged substeps is also at max_iterations=30 -- none exited on cost
     or anything else. The cap is a rail and is identical on both arms.
  2. THE CAUSAL ORDER IS SOLVE-FIRST. At the moment of the first failure the
     deepest penetration anywhere in the scene is 0.29 mm -- SHALLOWER than
     the 0.35 mm the production arm reaches at the same rung without ever
     failing. The 227 mm ejection is what happens AFTER the world latches,
     not what caused it. This is not "penetration defeated the solver".
  3. IT IS A GENUINE TIMESTEPPING EFFECT, IN THE UNEXPECTED DIRECTION. The
     same contact law, the same scene and the same 30-iteration budget
     converge in at most 17 iterations at h = 4.167 ms and blow through 30 at
     h = 1.042 ms. Shrinking h with tau held fixed raises the stabilization
     target vhat_n = -phi0/(h+tau) by 3.6x here and stiffens the per-substep
     convex problem with it. SMALLER STEPS MADE THE SOLVE HARDER, which is
     the reverse of the intuition the adaptive claim rests on.

#### (f) THE ADAPTIVE ARM PAYS NOTHING TO HOLD THE NEAR-RIGID LINE.

Accepted per-world substeps per boundary, adaptive arm, across the whole
grid: 1.298 to 1.334 -- flat to +-1.4% across seven decades of stiffness and
six values of tau, on both seeds. The per-world dt sampled at every boundary
end never fell below 2.55 ms, nine orders above the 1e-12 floor (that sample
is the controller's NEXT-step choice, not the march minimum, so it bounds the
floor question and nothing finer). There is no substep bill for entering the
paper's regime, because the clamp means the effective stiffness never
actually diverges (see (b)).

#### (g) A TASK-FILE CLAIM, RE-MEASURED AND FALSE FOR SAP.

`_validate_solver_substeps` refuses num_substeps < 2 with: "dt 0.01 sinks the
resting blade into the tabletop and goes non-finite on first grasp". The
message names the MJWarp solver and mj dt; it has never been measured on SAP.
Suspended at RUNTIME in the probe (no task file touched), the fixed SAP arm
at ONE uniform 8.333 ms step -- the strictly matched baseline for an adaptive
arm handed that same boundary -- ran all 17 rungs on both seeds with 0
physics_diverged, 0 robot_abnormal, 0 non-finite, and a P5 penetration of
-2.7563e-5 m against the 2-substep arm's -2.7556e-5, i.e. dt-independent
exactly as the compliant branch predicts. Whether the guard is right for the
MuJoCo solver it was written about was NOT re-measured here and is not
challenged; for SAP it is demonstrably false.

### TASK 1 VERDICT

There is NO stiffness at which the production fixed arm breaks while the
adaptive arm holds. The reachable range was exhausted: 6.9 decades of
stiffness up to sap_warp's own 1e10 fallback, tau from the authored 0.02 down
to exactly 0, crossed, on four fixed substep sizes and the adaptive march,
two seeds. The measured d ln k_eff / d ln dt does reach the paper's -2.000,
and the formula behind it is validated against penetration to <1% -- but
arriving there costs neither arm anything, because the clamp caps the
effective stiffness of the penetration-setting contacts at 2.34x the authored
value at the production tau, and 13.6x with tau driven to zero, no matter
what the asset authors. A NEGATIVE RESULT, and it is the real one: on this
scene, at this mass scale, with beta = 1, the CENIC mechanism has no room to
bite.

### TASK 2 — THE MATCHED TRAINING COMPARISON (D7), RUN AT LAST.

PRE-FLIGHT, ON THE EXACT TRAINING CONFIG (determinism unset, production
contact law, containment default), both arms dumped live off the constructed
objects before any GPU hour was spent -- p30_parity_{fixed,adapt}.json:

  IDENTICAL, row for row: optimality_rel_tol 1e-8 | optimality_abs_tol 1e-14
  | cost_abs_tol 0.0 | cost_rel_tol 0.0 | max_iterations 30 | line search
  armijo_decay/40 | preset approx32 | contact solve fp64 | beta 1.0 | sigma
  1e-3 | live contact_k 1250 | live contact_tau 0.02 | contact buffer
  2048/world | jacobian deterministic False | containment non-strict |
  divergence mask allocated | shape count 274 | authored ke 2500 | authored
  tau 0.01 | physics boundary 1/120 s | peak live contacts 54 on both.

  DIFFERENT, and every one of them intended:
    * substep: 2 x 4.1667 ms vs one marched 8.3333 ms boundary -- THE
      INDEPENDENT VARIABLE;
    * `tol` 1e-3 exists only on the adaptive arm -- the fixed arm estimates
      no error, which is what "fixed" means;
    * `_sap_world_active` allocated only on the fixed arm -- the two arms
      implement containment differently (manager mask vs the solver's own
      floor latch), same guarantee;
    * `attempt_consistent_r` True only on the adaptive arm -- structurally
      inert on a solver with no half solves. Known, ON by campaign default
      (D8), and named as residual risk P1.
  No unintended asymmetry was found, so the runs went ahead.

LAUNCH: the documented CLI on both arms, no env-var workaround.
  ./isaaclab.sh train --rl_library rsl_rl --task
  IsaacContrib-Lift-Spatula-Trossen-v0 --seed 42 --num_envs 1024
  --max_iterations 300 --logger wandb --log_project_name rubato-trossen
  --video --video_length 200 --video_interval 2400 --viz newton
  --solver sap                                    (fixed)
  --solver sap-adaptive physics=newton_mjwarp_adaptive   (adaptive)
Both runs' params/env.yaml record the resolved identity correctly
(`backend: sap` with `sap_adaptive` false/true and num_substeps 2/1), so
these two runs have real provenance -- the first SAP training runs in the
campaign that do.

HORIZON AND WHY: 300 iterations x 1024 envs x 24 steps = 7.37M env steps per
arm, ~49k episodes at 5 s each. Chosen because it is (i) long enough that the
task's own learning signal is unambiguous -- the fixed arm's mean reward
moves 5.7 -> 86.5 and its episode length 115 -> 144 inside it, so a real
difference in learning would have room to show; (ii) short enough to run BOTH
arms back to back in one pass and watch the videos, which a 4000-iteration
pair (~18 h) is not. It is NOT long enough to claim anything about final
policy quality, and nothing below does.

RUN INCIDENT, RECORDED SO THE ARTIFACTS MAKE SENSE: the adaptive arm's first
attempt was killed at iteration ~123 when the agent harness reaped the
background task holding its process group. Nothing physical went wrong -- the
log ends mid-block with the GPU released. It was restarted FROM SCRATCH
(same seed, same command, run_name suffixed `-r2`) under `setsid` so the next
reap could not reach it. The truncated log is kept as
p30_train_adaptive_attempt1_killed.log -- and it turned into the most
important control in this task.

#### THE ACCIDENTAL CONTROL: A SAME-CONFIG REPLICATE, AND WHAT IT COSTS THE
#### COMPARISON.

The killed attempt and the restart are the SAME ARM, SAME SEED, SAME COMMAND,
two processes. Determinism is OFF at production (the campaign default), so
the contact reduction order is nondeterministic and PPO compounds it. Their
common 11 iterations (p30_repro_check.py):

    iter          0     1     2     3     4     5     6     7     8     9    10
    adaptive #1  -0.0  -0.0  -0.0  -0.0  -0.0  0.01  0.05  0.22  0.85  1.49  2.19
    adaptive #2  -0.0  -0.0  -0.0   0.0  0.05  0.14  0.15  0.46  0.99  3.51  2.98
    FIXED arm    -0.0  -0.0  -0.0   0.0  0.10  0.03  0.16  0.78  1.57  2.61  3.45
    (Mean reward; episode length agrees to 2 dp through iteration 7 and then
     splits by up to 3.3 steps.)

The two ADAPTIVE replicates differ by 2.4x at iteration 9 -- and the FIXED
arm's value there (2.61) sits BETWEEN them (1.49 and 3.51). **The
between-arm difference is inside the within-arm, same-seed replicate
spread.** That is measured, not assumed, and it sets the resolution of
everything in this section: on this stack a single-seed pair of learning
curves cannot separate the two timesteppers, and no reward gap below the
replicate spread may be reported as an effect. Any future claim of a
learning difference needs seed replicates per arm, not one run each.

#### THE OTHER THING THE RUNS FOUND: **PASS 29's TRIANGLE-PAIR BUDGET IS
#### UNDER-SIZED FOR A TRAINED POLICY, AND BOTH ARMS OVERFLOW IT.**

Pass 29 sized the collision pipeline's triangle-pair pool at
max(1M, min(authored, 16384 * world_count)) and named 16,384/world as OUR
constant, therefore unvalidated -- justified against a measured worst case of
3,687 pairs/world (scripted flail, 64 envs) and warned that overflow drops
mesh contacts. Under a LEARNING policy at 1024 envs it is not enough
(p30_overflow_census.py):

    run                      first overflow   warnings   peak pairs/world
    FIXED    300 iterations     iter 110        17,564     29,464  (1.80x)
    ADAPTIVE 300 iterations     iter  95        19,462     31,492  (1.92x)
    ADAPTIVE attempt 1, killed  never (120 it)        0          -

Three things follow, and all three matter.
  1. THE CONSTANT IS WRONG FOR THIS WORKLOAD. Measured peak demand under a
     trained policy is 29,464 pairs/world -- 8x the 3,687/world that the
     budget was justified against, and 1.8x the budget itself.
  2. NO CAMPAIGN PROBE COULD HAVE CAUGHT IT. Every probe in this pass ran at
     32 or 64 envs, where the rule's max(1M, ...) FLOOR hands each world
     31,250 pairs -- nearly twice the per-world budget the 1024-env training
     run gets. The overflow is a large-world-count-only failure, and it
     appears only once a policy has learned to drive the scene.
  3. IT BOUNDS WHAT TASK 2 CAN CLAIM. Overflow drops mesh contacts, so from
     iteration 95 (adaptive) / 110 (fixed) onward the two arms are running
     DEGRADED contact sets. **The clean, like-for-like window is iterations
     0-94.** Everything past it is reported below but is not a controlled
     physics comparison, and no verdict here rests on it. Mitigating: over
     the FULL run the two arms overflow to a comparable degree (17,564
     against 19,462 warnings; 1.80x against 1.92x peak), so this is a shared
     ceiling both arms hit rather than a one-sided handicap -- but "similar
     amounts of silently dropped contacts" is not "the same contacts".
And the killed replicate sharpens it further: same arm, same seed, one run
overflows from iteration 95 and the other never does through 120. Overflow
incidence is itself draw-dependent, because it depends on what the policy
happens to learn to do to the mug.

#### RESULTS. BOTH ARMS COMPLETED 300 ITERATIONS.

Wall: FIXED 28.3 min (12:07:26-12:35:43), ADAPTIVE 103.8 min
(13:14:52-14:58:39) -- 3.67x. Curves are 5-band means over the 300
iterations, parsed from the rsl_rl blocks (p30_train_parse.py,
p30_train_series.json). Band 2 ends at 94 because that is where the
triangle-pair pool starts overflowing; bands 3-5 are reported but are not a
controlled physics comparison.

                          0-24     25-94   95-149  150-224  225-299
                                  <-CLEAN->  <---- pool overflowing ---->
  Mean reward     FIXED     5.67    19.76    60.37    79.51    86.54
                  ADAPT     5.51    21.62    60.97    77.52    86.03
  Episode length  FIXED   115.21   104.09   133.56   142.34   143.89
                  ADAPT   115.92   105.08   130.04   139.22   142.72
  time_out        FIXED   0.7024   0.4254   0.7240   0.8489   0.8927
                  ADAPT   0.7028   0.4216   0.6430   0.7779   0.8533
  object_off_tbl  FIXED   0.0874   0.3099   0.1876   0.1012   0.0754
                  ADAPT   0.0762   0.2802   0.2225   0.1568   0.1101
  robot_abnormal  FIXED   0.0863   0.2636   0.0871   0.0450   0.0275
                  ADAPT   0.0990   0.3001   0.1353   0.0633   0.0344
  object_speeding FIXED   0.0000   0.0015   0.0013   0.0041   0.0046
                  ADAPT   0.0003   0.0007   0.0024   0.0046   0.0040
  object_dropping BOTH ARMS EXACTLY 0.0000 IN EVERY BAND
  physics_diverg  FIXED   0.00185  0.00244  0.00242  0.00199  0.00167
                  ADAPT   0.00000  0.00000  0.00000  0.00000  0.00000
  lifting reward  FIXED    0.819    2.854    7.767    9.844   10.422
                  ADAPT    0.742    3.111    8.031    9.796   10.674
  reaching reward FIXED   0.0166   0.0353   0.1172   0.1823   0.2221
                  ADAPT   0.0163   0.0405   0.1154   0.1680   0.1993
  ori error       FIXED    1.747    1.784    1.777    1.738    1.679
                  ADAPT    1.733    1.731    1.627    1.518    1.449
  s / iteration   FIXED     3.48     4.29     6.18     6.45     6.20
                  ADAPT    10.48    16.81    22.14    23.63    23.71

LEARNING: NO SEPARATION. In the clean window the two arms are on top of each
other -- reward 5.67/19.76 against 5.51/21.62, episode length within 1%,
time_out within 1%. At the end of the run they are 86.54 against 86.03, a
0.6% difference, against a measured same-config replicate spread of 2.4x at
iteration 9. Both arms learn the same thing at the same rate. **The correct
report is "no separation observed at this horizon", and the replicate control
says a single-seed pair could not have shown one smaller than very large.**

COST: the adaptive arm is 3.67x the wall for that identical learning, and the
gap WIDENS with contact activity (3.01x in band 1 to 3.82x in band 5) because
both arms slow as the policy starts driving the mug and the adaptive arm
slows harder (2.26x over the run against the fixed arm's 1.78x).

THE ONE QUALITATIVE DIFFERENCE: the fixed arm terminates 0.17-0.24% of
episodes on physics_diverged in every band; the adaptive arm terminates
exactly zero, in all 300 iterations. On the fixed arm that number is a
mechanism -- the inner Newton solve did not reach optimality_rel_tol 1e-8
within 30 iterations, and the arm has no move but to latch.

Two readings compete for it, and they make OPPOSITE predictions:
  (a) adaptivity rescues solves the fixed step cannot converge -- the thesis.
      Predicts the fixed arm's rate FALLS if you give it a smaller step.
  (b) the arms latch at different thresholds (fixed on its FIRST
      non-convergence, adaptive only at the dt floor), so zero-vs-nonzero is
      an artefact of the threshold. Predicts the rate does not care about
      step size.
**IT WAS TESTED.** Same task, same seed, same launch path, 1024 envs, 40
iterations, only num_substeps moved (p30_substep_chain.sh):

    fixed substeps   h [ms]   physics_diverged   max     iters w/ any
       2             4.167       0.00245        0.0068      33/40
       4             2.083       0.01290        0.0241      37/40
       8             1.042       0.03819        0.0673      36/40

The rate RISES 15.6x as the step shrinks 4x, monotonically. Reading (a) is
REFUTED: the fixed arm's divergences are not a step-too-large problem, and
"take a smaller step" makes them an order of magnitude worse. That matches
the sweep's own diagnosis -- shrinking h with tau fixed raises the
stabilization target -phi0/(h+tau) and stiffens the per-substep convex
problem -- and it means the adaptive arm's zero cannot be attributed to it
choosing smaller steps, because its mean accepted substep (4.31 ms) is
LARGER than the fixed arm's 4.167 ms. What is left is reading (b), the
latch threshold, plus the fact that coarser steps are easier here. So the
zero should NOT be reported as adaptivity rescuing the physics.

And in any case: at 0.2% of terminations it did not move the learning curve,
which is the quantity the claim is about. (The 40-iteration s2 control also
reproduces the main run's rate -- 0.00245 against 0.00185-0.00244 across the
300-iteration run's bands -- so the probe and the run agree.)

TERMINATION MIX: both arms run the same trajectory through the same phases --
a violent exploration hump around iterations 25-94 (object_off_table 0.31/
0.28, robot_abnormal 0.26/0.30) that decays as the policy learns
(0.075/0.110 and 0.028/0.034 by the end), with time_out rising to 0.85-0.89
as episodes survive. object_dropping is EXACTLY zero on both arms in all 300
iterations. The adaptive arm knocks the mug off the table somewhat more in
the second half (0.110 vs 0.075) and leaves it better oriented (orientation
error 1.449 vs 1.679); both differences sit in the same unresolved band as
the reward gap.

WHAT THE VIDEOS ACTUALLY SHOW (filmstrips cropped onto the nearest world;
the 1024-env render puts the robot at ~40 px, so the raw clips are not
judgeable and the crops are the evidence):
  * iteration 0, BOTH ARMS: indistinguishable. The arm waves above and behind
    the table through the whole clip; the mug sits untouched. Same initial
    policy, same scene.
  * iteration 100, BOTH ARMS (and the killed adaptive replicate, three runs):
    the same learned strategy, and it is NOT a clean pick-up. The gripper
    comes down onto the mug, TIPS IT OVER, and hoists it -- the mug is
    horizontal or on its rim in most frames where it is off the table. All
    three runs discovered the same tipping strategy.
  * iteration 200, BOTH ARMS: more of the same, and still indistinguishable
    between arms -- the mug spends most of the clip lying on its side, gets
    picked up horizontally and hoisted. Neither arm has learned to right it
    or to grasp it upright.
  * NOTHING PATHOLOGICAL IS VISIBLE ON EITHER ARM: no jitter, no explosion,
    no object sinking through the table, no frozen worlds. The rising reward
    is real behaviour, and the behaviour is a tip-and-hoist, not a grasp.
    That is a TASK/REWARD observation, identical on both arms, not a
    timestepping one -- but it is what a reward of 86 actually looks like,
    and it should not be read as a solved lift.

FINAL-POLICY PLAYBACK AND THE CROSS EVALUATION. The in-training recorder's
last clip lands at iteration 200, so each arm's model_299 was replayed under
one identical protocol (32 envs, seed 42, 300-step clip) -- and each policy
was ALSO replayed on the OTHER arm's physics. If the two timesteppers
produced materially different dynamics, a policy trained under one should
degrade under the other. All four cells:

    policy \ physics        FIXED                    ADAPTIVE
    FIXED       acquires the mug, holds it   acquires, holds it aloft
                aloft for the rest of clip   for the rest of the clip
    ADAPTIVE    acquires, holds it aloft     acquires fastest of the four,
                for the rest of the clip     holds it for the whole clip

NO CELL DEGRADES. Both policies transfer to the other arm's physics with no
visible loss of competence, which is direct behavioural evidence that the two
timesteppers produce dynamics this task's policies cannot tell apart. (This
is qualitative: `play` emits no reward summary, so there is no number behind
the four cells, only the rendered behaviour.)

### TASK 2 VERDICT

**NO SEPARATION OBSERVED AT THIS HORIZON.** Over 300 iterations at 1024
envs, the two arms' reward, episode length and termination mix are on top of
each other -- 86.54 against 86.03 in the final band, 0.6% apart, against a
measured same-config replicate spread of 2.4x. Both learn the same
tip-and-hoist strategy at the same rate, the videos of all four
policy x physics cells are behaviourally indistinguishable, and each arm's
policy transfers to the other arm's physics without degrading. The adaptive
arm costs 3.67x the wall for that identical outcome.

The one difference that survives is the fixed arm's 0.17-0.24%
physics_diverged rate against the adaptive arm's exact zero -- and the
substep probe REFUTES the reading that adaptivity is rescuing those solves,
because giving the fixed arm smaller steps multiplies the rate by 15.6x
rather than removing it.

THIS IS A CHARACTERIZATION OF A BOUNDED RUN, NOT A PAPER CLAIM. It is one
seed per arm at a horizon neither arm has converged at, on a stack whose
same-seed replicate spread is larger than every difference reported, with
both arms' contact sets silently degraded past iteration ~100 by a
triangle-pair pool neither of them fits in. What it does establish, and what
the campaign did not have before today, is that the comparison RUNS, that it
runs matched, and that at this horizon it shows nothing.

### RESIDUAL RISK — WHAT THIS PASS COULD NOT ESTABLISH

P1 **ACR IS AT FULL STRENGTH IN THE REGIME THIS PASS SWEPT, AND WAS NOT
   SEPARATED.** `attempt_consistent_r` is ON for the adaptive arm and
   structurally inert on the fixed arm (s == 1 with no half solves) -- read
   live off both arms this pass. It scales W, hence rn_hard AND the
   tangential rt, by s ~ 2.1 in the adaptive arm's COMMITTED half solves.
   Pass 25 priced that at the production law and found it small (+3.5%
   deepest phi0) precisely BECAUSE only ~11% of contacts were in the clamped
   branch. In this pass's near-rigid rungs 100% are, so the same mechanism
   softens 100% of the adaptive arm's committed normal law there. Every
   adaptive-vs-fixed penetration number inside the near-rigid regime is
   therefore ACR-confounded, and this pass did not run the ACR-off arm to
   separate it. This does NOT touch the task-1 verdict, which rests on
   failure counts and on within-arm h-scaling.

P2 THE TWO ARMS STILL DRAW CONTACTS FROM DIFFERENT SOURCES (pass-25 R3).
   The fixed arm consumes the manager CollisionPipeline, the adaptive arm
   owns its own. Both were verified this pass to carry the same capacity
   (2048/world), the same triangle-pair budget and the same determinism
   resolution, and they report the same live contact count at rest (54 vs
   54) -- but the SETS were not proven identical. Between-arm penetration
   comparisons inherit that.

P3 TRAJECTORY DIVERGENCE BOUNDS WHAT A BETWEEN-ARM NUMBER MEANS. After 270
   scripted steps two arms with different dt are in different scene states.
   The trustworthy axis is WITHIN an arm across rungs (identical seed and
   action stream); between-arm differences are only load-bearing where they
   are qualitative (0 vs 620 divergences), not where they are a factor of
   two in a percentile.

P4 THE tau -> 0 EXPONENT ENDPOINT. See (d): -2.000 is validated by the
   formula's <1% agreement at tau >= 4 ms, not by a direct penetration
   measurement at tau = 0, where the signal falls below trajectory noise.

P5 **THE SCRIPTED RIG UNDERSTATES TRAINING, TWICE OVER, AND THIS IS THE
   PASS'S BIGGEST METHODOLOGICAL LESSON.** The sweep's rest/press/swing/flail
   stream produced ZERO physics_diverged on the production fixed arm at the
   authored law -- the training run fires that same term at ~0.24% of
   terminations on the same arm and the same law. The same rig showed a peak
   triangle-pair demand comfortably inside its budget -- the training run
   exceeds the budget by 1.8x. A learning policy reaches states no scripted
   stream in this campaign has reached, and BOTH of the failure modes this
   pass cares about are invisible below it. Every "no failure observed"
   result obtained on a scripted rig, in this pass and in passes 25-29, is a
   statement about the rig.

P5b PENETRATION UNDER THE LEARNED POLICIES WAS NOT INSTRUMENTED. The training
   runs report terminations, not phi0. The penetration comparison between the
   arms in this entry is the matched-scripted-stream one at the authored law
   (fixed deepest -4.316e-3 / P5 -2.7556e-5; adaptive -4.218e-3 /
   -2.7563e-5). Closing this needs a checkpoint-loading probe that dumps
   phi0 under each arm's own trained policy; not done.

P6 ONE MACHINE, TWO SEEDS. Penetration statistics repeat to 0.00-0.93%
   between seeds; RARE divergence events do not (2 vs 0 at one rung, 27 vs 7
   at another). No rare-event rate in this entry should be read to better
   than an order of magnitude.

P7 sap_warp is still joined by SAP_WARP_PATH rather than pinned (D11), so
   every number here is valid only against sap_warp afd5dc6.

### CORRECTION TO THE PASS-29 RECORD — THE DEMAND NORMALIZATION

Pass 29 reported the adaptive arm doing "7.7417 accepted world-substeps per
world-boundary against the fixed arm's 2.0000", concluded "3.87x the
substeps" and "0.405x the cost per accepted substep (2.47x cheaper per unit
of integration work)". THAT COMPARISON MIXES UNITS. `num_substeps` is
substeps per PHYSICS BOUNDARY; `cumulative_accepted_steps()` accumulates over
a whole ENV STEP, and one env step contains `decimation` = 4 boundaries.

MEASURED rather than reasoned (p30_demand_norm_probe.py wraps the manager's
own solver entry point and COUNTS the calls; p30_norm_*.json, 64 envs):

    fixed     240 solver calls / 30 env steps = 8.0 per env step
              = 4.0 boundaries x 2 substeps;  mean substep dt 4.1667 ms
    adaptive  120 solver calls / 30 env steps = 4.0 boundaries per env step
              accepted 1.95 per world-boundary; mean substep dt 4.2735 ms
    INVARIANT THAT PROVES THE UNITS MATCH: both arms advance exactly
    0.03333333 s of simulated time per env step.

Re-run at the pass-29 scale and conditions (1024 envs, 120 timed steps after
20 warmup, seed 42, same uniform-random stream, 2 repeats each,
p30_char_*.json):

                       ms/env step   accepted substeps    us per accepted
                                     per world-BOUNDARY   world-substep
      FIXED (2 sub)    50.41 / 50.55       2.0000           6.153 / 6.170
      ADAPTIVE (march) 79.06 / 79.17       1.9354           9.972 / 9.987

So the corrected statement is the opposite of the recorded one: the two arms
do essentially the SAME integration work per boundary (adaptive 3.2% FEWER
accepted substeps, at a 3.3% LARGER mean substep), and demand-normalized the
adaptive arm costs **1.62x MORE** per accepted world-substep, not 2.47x less.
The 1.57x wall ratio is almost entirely per-substep price, not extra work.
Repeats agree to 0.3%. Pass 29's raw wall and ms/step numbers are reproduced
here to within 0.2% and stand; only the normalization and the two ratios
derived from it are withdrawn.

### CORRECTION TO THE PASS-25 RECORD — HOW FAR THE REGIME ACTUALLY IS

Pass 25 (F4/D12) stated that entering the near-rigid regime needs "authored
contact stiffness up ~6-7 orders". Measured this pass: the clamp takes over
at k ~ 2.5e4 at the production substep, i.e. **1.3 decades**, and by k = 1e6
it is 100% of contacts. The conclusion pass 25 drew -- that the production
config runs compliant -- is confirmed and is not affected; only the distance
to the boundary was overstated, which matters because it made the regime look
out of reach when it is one line of asset authoring away.

### WHAT MARCO MUST DECIDE NEXT

M-A **DOES THE PAPER STILL WANT THE NEAR-RIGID CLAIM?** The regime is
    reachable (one NewtonShapeCfg `ke` and one `sap_contact_tau_d`), the
    exponent does hit -2.000, and BOTH arms are stable there with SHALLOWER
    penetration than production. So the honest framing is not "adaptive
    rescues near-rigid contact" -- there is nothing to rescue at beta = 1 --
    but "error-controlled dt at matched cost", which is a smaller claim.
    Decide which claim the paper makes before any more GPU time is spent.

M-B **beta IS THE ONLY KNOB THAT MOVES THE CEILING, AND IT IS ON THE RED
    LINE.** rn_hard = beta^2/(4 pi^2) W. Lowering beta raises the maximum
    effective stiffness quadratically and is the only way this scene can be
    made genuinely near-rigid. It is a Drake constant and the validity red
    line forbids a loop from touching it. If the paper needs the stiff
    regime, this is the decision, and it is his alone.

M-C **THE TASK'S num_substeps >= 2 GUARD IS WRONG FOR SAP** (section (g)).
    It blocks the strictly matched fixed baseline on the strength of a claim
    measured on the MuJoCo solver. Either scope the guard to the MuJoCo
    backend or delete it for SAP; task file = his.

M-D **ACR IN THE NEAR-RIGID REGIME** (residual P1). If any near-rigid
    experiment goes ahead, the ACR ON/OFF A/B has to run with it: at the
    production law ACR touched ~11% of contacts, in the swept regime it
    touches 100%. D8's "leave it on" was decided against the 11% number.

M-E The pass-29 demand-normalized numbers in the ledger were withdrawn and
    replaced above. If any of them reached a draft or a slide, they need
    correcting there too.

M-F **SINGLE-SEED TRAINING COMPARISONS ARE BELOW THE NOISE FLOOR ON THIS
    STACK** (the accidental control). Two same-seed adaptive runs differ by
    2.4x in mean reward at iteration 9. Any learning claim needs replicates
    per arm -- budget for at least 3 seeds x 2 arms, or run with
    NEWTON_SAP_DETERMINISTIC=1 and accept its cost, before anything about
    learning outcomes goes in a paper.
    [PARTLY ANSWERED, PARTLY WITHDRAWN by pass 31. The 3-seeds-per-arm
    half was done -- see the pass-31 D7 entry for the difference measured
    against the replicate spread. The DETERMINISM half is NOT AVAILABLE
    at 1024 envs: the deterministic contact-id budget clamps the
    triangle-pair pool to 1 << 25 = 32,768/world, which pass 31 measured
    to be BELOW this scene's trained-policy demand of 40,138/world, so a
    deterministic run at 1024 truncates contacts. Determinism and honest
    capacity now exclude each other at this world count.]

M-G **RAISE THE TRIANGLE-PAIR BUDGET BEFORE THE NEXT TRAINING RUN.** Measured
    peak demand under a trained policy at 1024 envs is 29,464 pairs/world
    against the 16,384/world budget pass 29 landed. This is a one-line change
    to `_sap_triangle_pair_budget` and it costs memory: at 1024 envs the pool
    is 12 B/pair, so 32,768/world would take the pool from 7.35 GB-era sizing
    to roughly double it -- against 32 GB of device memory and a measured
    7.35/7.73 GB footprint, that fits at 1024 and would need re-checking at
    4096, where pass 29 measured 25.32 GB already. NOT CHANGED THIS PASS: it
    is a landed pass-29 rule, changing it mid-experiment would have
    invalidated the runs in flight, and the right number should come off a
    demand measurement at the scale that will actually be used.
    [CLOSED by pass 31. Demand was measured at 1024 under the trained
    checkpoints with the pool oversized first: peak 40,138 pairs/world
    (fixed arm) and 36,655/world (adaptive). The constant is raised to
    65,536/world -- 1.63x the worst measured peak, +3.22 GB at 1024, and
    verified to zero truncation on both arms and across all six re-run
    training logs. The estimate quoted above (32,768/world "fits at
    1024") was itself too small: it is BELOW the measured peak. 4096 no
    longer constructs, and pass 31 measured that 4096 could never have
    held the real demand on this device anyway.]

### PROVENANCE (all p30_ prefix, no p13-p29 artifact overwritten)

  probes    p30_regime_probe.py (the sweep; the runtime material override and
            its live read-back verification), p30_demand_norm_probe.py (wraps
            the manager's solver entry point to COUNT boundaries),
            p30_char_probe.py (speed + demand at 1024 with the measured
            normalization), p30_diag_probe.py (per-substep causal order at
            the failing rung), p30_sweep_analysis.py, p30_train_parse.py,
            p30_repro_check.py, p30_filmstrip.sh
  chains    p30_regime_chain.sh, p30_corner_chain.sh, p30_s1_chain.sh,
            p30_norm_chain.sh, p30_char_chain.sh, p30_qs_chain.sh,
            p30_train_chain.sh + p30_train_adapt2.sh, p30_play_chain.sh,
            p30_substep_chain.sh, p30_post_driver.sh
  sweep     p30_reg_{fixed_s2,fixed_s4,fixed_s8,adapt}_seed{42,7}.{json,log}
            p30_cor_* (the CENIC corner), p30_s1_* (matched boundary),
            p30_qs_* (quasi-static), p30_sweep_tables.txt
  fairness  p30_parity_{fixed,adapt}.{json,log}
  demand    p30_norm_{fixed,adapt}.json, p30_char_{fixed,adaptive}{1,2}.json
  training  p30_train_{fixed,adaptive}.log,
            p30_train_adaptive_attempt1_killed.log, p30_train_series.json,
            p30_play_*.log, p30_substep_s{2,4,8}.log
  substep   p30_substep_chain.sh + p30_substep_s{2,4,8}.log (the step-size
            test of the fixed arm's divergence rate)
  overflow  p30_overflow_census.py
  video     p30_vid_*.png filmstrips (iterations 0/100/200 on both arms and
            on the killed replicate; all four final-policy playback cells),
            p30_play_{fixed_on_fixed,adaptive_on_adaptive}.mp4 and
            p30_play_cross_{fp_on_a,ap_on_f}.mp4
  progress  p30_*_progress.txt (each carries the interleaved
            nvidia-smi compute-app samples that certify GPU exclusivity)
  W&B       project rubato-trossen, runs p30-sap-fixed-1024x300 and
            p30-sap-adaptive-1024x300-r2

## PASS 33 — THE TRAINING HARNESS LANDS, AND THE SCALE QUESTION IS ANSWERED
## FROM EXISTING EVIDENCE
## 2026-08-16. SOFTWARE + ANALYSIS ONLY: ZERO GPU PROCESSES STARTED. The
## 4000-iteration main run was in flight (or waiting) throughout; nvidia-smi
## was polled read-only at every step and never showed a process this pass
## started. newton-adaptive, sap_warp and IsaacLab are byte-untouched; the only
## code written is new tooling in a FOURTH repo, IsaacLabRubato, which now
## carries campaign state (noted here so the record does not lose it).
## Marco's request: "a robust training lib similar to the sweep", plus "some
## scenes require 8k 4k 2k envs and wall time just isnt there yet", refined to
## "the default env for like allegro pose is 10k itters 8k evns which would
## take weeks" and "150 isnt enough to learn fucking shit".

### DELIVERABLE 1 — THE HARNESS: tools/rubato_sweep in IsaacLabRubato

WHERE AND WHY. Reusable Python goes in `IsaacLabRubato/tools/` beside
`wandb_done.py` and `dump_env_spec.py`; a campaign owns a directory under
`experiments/` holding its configs, a thin driver and gitignored artifacts.
That is the convention `experiments/rubato-ppo-sweep/sweep.sh` (267 lines, the
"sweep" Marco is referring to) and `experiments/rubato-ppo-quantile/` already
establish. So:

    IsaacLabRubato/tools/sweep.py                  one-command entry point
    IsaacLabRubato/tools/rubato_sweep/*.py         the library
    IsaacLabRubato/tools/rubato_sweep/configs/     experiment configs
    IsaacLabRubato/tools/rubato_sweep/tests/       CPU-only tests, 20/20 pass
    IsaacLabRubato/experiments/trossen-sap-scale/  this campaign's driver

ONE COMMAND RUNS AN EXPERIMENT:

    cd ~/Documents/code/IsaacLabRubato
    .venv/bin/python tools/sweep.py plan tools/rubato_sweep/configs/trossen_sap_d7.yaml
    .venv/bin/python tools/sweep.py run  tools/rubato_sweep/configs/trossen_sap_d7.yaml
    .venv/bin/python tools/sweep.py analyze tools/rubato_sweep/configs/trossen_sap_d7.yaml

`plan` touches no GPU and prints the cell order, what is already complete, and
both unit conversions resolved. `run` takes the flock, preflights, and executes.

WHAT EACH GUARD REPLACES (every one is a failure this campaign paid for):
  * video cadence expressed in ITERATIONS, converted internally to the env-step
    count `--video_interval` actually counts (x24 on this task). Verified in
    source this pass: `video_recorder.py:87-112` increments per `env.step()`,
    with no decimation and no num_steps_per_env factor anywhere.
  * skip-if-complete on `Training time: N seconds` (the true completion print;
    `train_rsl_rl.py:239-247`) with the last-iteration marker as fallback.
    Ctrl-C exits 0 and prints neither, so exit code is never trusted.
  * arms are the INNERMOST axis; a kill lands on a complete matched set. Tested.
  * journal.jsonl records start, exit and an nvidia-smi compute-app census at
    every boundary, plus the short HEAD of all FOUR repos (sap_warp is joined by
    sys.path, so physics can move with no commit in the other three).
  * a run that exits non-zero before reaching iteration 0 ABORTS the sweep.
  * exclusivity is enforced by census+poll+flock, not operator discipline.
  * packing (concurrent runs) is available but REFUSED unless the config says
    `timing_sensitive: false`; packed runs are tagged `exclusive: false` so a
    packed wall can never later be quoted as a timing.
  * per-run JSON: iteration series, reward/termination series, walls, GPU peak,
    telemetry-derived accepted substeps, demand normalization, and the
    `Triangle pair buffer overflowed` events with the FIRST offending iteration
    (pass 30 spent two thirds of a comparison inside one without noticing).
  * demand normalization returns BOTH denominators, named, and asserts their
    ratio is exactly `decimation`. Mixing them is the pass-29 error.
  * the aggregator REFUSES to report a between-arm difference from fewer than
    two replicates per arm; it returns UNRESOLVED and says why.

VALIDATED THIS PASS, ON REAL DATA, WITHOUT A GPU. The six pass-31 training logs
were ingested through the harness parser and run through the aggregator
(p33_ingest_p31.py; output p33_p31_ingest/summary.json):

    metric                    fixed (n=3)            adaptive (n=3)      verdict
    s/iter (last 50)   8.067 [7.442, 8.510]  18.207 [16.410, 19.170]  SEPARATED
    samples/s               3056 [2888,3302]       1357 [1282,1498]   SEPARATED
    Mean reward (last 50)  102.5 [87.8, 110.0]      57.9 [30.7, 86.8] NOT SEPARATED

The reward row is the harness doing its job: the adaptive arm's WITHIN-arm
reward range is 97% of its own mean at 150 iterations, so the 44-point
between-arm gap is inside the noise. This is Marco's "150 isnt enough to learn
fucking shit", measured. 150-iteration cells are throughput and failure-mode
instruments only.

STATUS. All CPU paths are tested (20/20). `preflight_probe.py` and the GPU
launch path were written under a no-GPU rail and have NOT been executed; their
solver access paths are lifted verbatim from the campaign's proven parity probe
(p30_regime_probe.py:142-170). First run is a shakedown.

INCIDENTAL FINDING, and it is the harness's own thesis. While this pass ran,
`train_main.sh`'s GPU-clear waiter sat spinning on an EMPTY device: the idiom
`n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . ||
echo 0)` yields the string "0\n0" when the device is empty, `[ "$n" -eq 0 ]`
then errors with "integer expression expected", and the loop never breaks. The
run self-recovers when the 60-minute for-loop expires, so it starts ~29 minutes
late; nothing was touched. It is the pass-23 swallowed-exit-code class exactly.
gpu.py parses the census structurally instead, and there is now a regression
test for an empty device, a populated one, and an unreadable one.

### DELIVERABLE 2a — MEMORY. THE MODEL, FITTED TO MEASUREMENT.

Four pass-31 construct probes (p31_con_*.json, adaptive arm,
`device_used_gb_after_build`) determine three constants:

    M(N, T) [GiB] = 1.359 + 4.917 MiB/env * N + 68.7 B/pair * min(192e6, T*N)

  N = envs, T = triangle pairs per world. The 68.7 B/pair is pool (12 B vec3i)
  + reducer (40 B) + hashtable (68 B on next_pow2(max(0.25P, 1024)) entries),
  independently confirmed from the array declarations in
  contact_reduction_global.py:851-881 and hashtable.py:224-225.

  MEASURED   1024 @ 16,384/world  ->  7.350 GiB   (fit point)
  MEASURED   1024 @ 65,536/world  -> 10.568 GiB   (fit point)
  MEASURED   4096 @ 16,384/world  -> 25.320 GiB   (fit point)
  MEASURED   4096 @ 32,768/world  -> 29.633 GiB   model 29.612, err -0.07% (HELD OUT)
  MEASURED   4096 @ 40,960/world  -> OOM (2,013,265,920 B alloc)  model 31.758
  MEASURED   4096 @ 65,536/world  -> OOM (2,304,000,000 B alloc)  model 33.307

The card is 31.84 GiB. The largest measured SUCCESS is 29.63 GiB and the
smallest measured FAILURE models at 31.76 GiB, so the practical ceiling with
fragmentation is ~30 GiB, not 31.84. Plan against 30.

THE PER-ENV SLOPE IS THE BINDING TERM, AND IT IS A CAPACITY, NOT A SCENE
PROPERTY. 4.917 MiB/env is dominated by `72*C*D + 612*C` with C =
`max_rigid_contact` per world = 2048 and D the per-env dof count
(contact_solve.py:5271-5640, contact_jacobian.py:2449-2515; the source formula
reproduces the fitted slope). C=2048 is set by the clamp
`min(2048, 8e6 // nworld)`, not by demand: measured peak contact demand is
54-144 per world. That is a ~14x margin.

Consequences, with the pool at zero so the floor is visible:

    C=2048/world:  4096 envs -> 21.03 GiB floor   8192 envs -> 40.70 GiB floor
    C=1024/world:  4096 envs -> 11.19 GiB floor   8192 envs -> 21.03 GiB floor
    C= 512/world:  4096 envs ->  6.28 GiB floor   8192 envs -> 11.19 GiB floor

**8192 envs is blocked by the per-env floor ALONE at the shipped contact budget
(40.70 GiB > 30 GiB usable) before a single triangle pair is allocated.**

HONEST CAPACITY IS THE CONSTRAINT THAT DECIDES 4096. Trained-policy triangle-
pair demand measured at 1024 is 40,138/world (fixed arm) / 36,655 (adaptive),
which is why pass 31 set the budget to 65,536/world. Anything below ~40,138
truncates mesh contacts SILENTLY.

    4096 @ 65,536/world, C=2048  -> 33.31 GiB  BLOCKED (matches measured OOM)
    4096 @ 32,768/world, C=2048  -> 29.61 GiB  fits, but 32,768 < 40,138:
                                                DISHONEST, would truncate
    4096 @ 65,536/world, C=1024  -> 23.47 GiB  FITS, effective 46,875/world
                                                (pool hits the authored 192M cap)
    2048 @ 65,536/world, C=2048  -> 19.78 GiB  FITS as shipped, honest
    8192 @ any T,        C=512   -> pool caps at 192M = 23,438/world:
                                                DISHONEST; honest 8192 needs
                                                ~328.8M pairs = 21.0 GiB of pool
                                                plus an 11.19 GiB floor = 32.2 GiB
                                                -> BLOCKED under every sizing rule

So pass 31's "an honest 4096 is out of reach at any rule on this hardware" needs
one correction and one confirmation: it is out of reach HOLDING C AT 2048, and
**cutting C to 1024 (still 7x the measured peak of 144) is projected to bring an
honest 4096 down to 23.5 GiB.** The 8192 half of the claim stands and is now
quantified. CAUTION, stated loudly: sizing a capacity from measured demand is
exactly what pass 29 did with the triangle-pair pool, and a trained policy then
blew past it by 2.44x. The C reduction must be confirmed by measuring contact
demand under a TRAINED policy at 4096 before it is trusted.

### DELIVERABLE 2b — WALL. TWO ESTIMATES THAT DISAGREE, BOTH REPORTED.

MEASURED BASE (pass 31, current stack, 1024 envs, last-50 mean of 150-iteration
runs, n=3 seeds): fixed 8.067 s/iter (3056 samples/s), adaptive 18.207 s/iter
(1357 samples/s). Note both arms moved since pass 30 (fixed 6.17 -> 8.07,
adaptive 23.79 -> 18.21): the honest triangle-pair budget stopped the fixed arm
silently dropping contacts, which cost it wall.

The env-count exponent alpha in T(N) = T0 (N/N0)^alpha has two estimates:
  CONSERVATIVE, MEASURED: pass-9, det=1, adaptive, 1024 vs 4096 at matched
    iteration index -> wall ratio 3.17-3.32 for 4x envs, alpha 0.83-0.87,
    decomposing as marches x1.40 (the adaptive demand tail: batch-wide accepted
    substeps track the SLOWEST world, and the max over 4x more worlds is bigger)
    and ms/march x2.2. On a ~30 h stale stack.
  OPTIMISTIC, PROJECTED: an occupancy model built on the current stack's p19/
    p20/p21 profile bytes, self-calibrated to the measured 1024 window within
    1.2% -> 2.06-2.25x for 4x envs, alpha 0.52-0.58. Its basis is measured:
    88.4% of plateau slabs run below 25% active worlds, 52.4% of a straggler
    slab's kernel time is in launches whose grids can absorb 4x more blocks
    inside the GPU's resident capacity, and march depth grows only ~21% per 4x
    (logarithmic; 64->1024 subsample of p19_occ_1024x25.audit).
The FIXED arm has no demand tail (accepted substeps per boundary constant at
2.0000, p30_norm_fixed.json), so it should sit at or below the low end of both.

FEASIBILITY TABLE — Trossen mug/spatula scene, SAP arms, this 32.6 GB card.
Wall is given as conservative..optimistic. Memory verdicts assume honest
capacity (T >= 40,138/world) unless stated.

 envs  arm       memory                       s/iter       samples/s   4000 it
 1024  fixed     10.57 GiB MEASURED-fits       8.07 MEAS      3046 MEAS   9.0 h MEAS
 1024  adaptive  10.57 GiB MEASURED-fits      18.21 MEAS      1350 MEAS  20.2 h MEAS
 2048  fixed     19.78 GiB PROJECTED-fits  11.6-12.0 PROJ 4104-4249 PROJ 12.9-13.3 h PROJ
 2048  adaptive  19.78 GiB PROJECTED-fits  26.7-32.8 PROJ 1498-1844 PROJ 29.6-36.5 h PROJ
 4096  fixed     BLOCKED at C=2048 (33.3 GiB, OOM MEASURED);
                 23.5 GiB PROJECTED-fits at C=1024   16.6-17.8 PROJ 5529-5926 PROJ 18.4-19.8 h PROJ
 4096  adaptive  same as fixed row                   39.0-59.2 PROJ 1662-2519 PROJ 43.4-65.7 h PROJ
 8192  fixed     BLOCKED-BY-MEMORY (40.7 GiB floor at C=2048; 32.2 GiB honest
                 even at C=512, vs ~30 GiB usable)   23.8-26.4 PROJ 7450-8266 PROJ 26.4-29.3 h PROJ
 8192  adaptive  BLOCKED-BY-MEMORY (same)            57.1-107  PROJ 1844-3441 PROJ 63.5-119 h PROJ

Throughput is the row that matters: samples/s RISES with env count on both arms
under both estimates, so s/iter alone makes width look worse than it is.

FIXED SAMPLE BUDGET (98.3M env steps = 1024 x 4000), the metric that decides
whether width pays. Break-even is alpha = 1.
    fixed    1024 -> 8.96 h | 2048 -> 6.4-6.7 h | 4096 -> 4.6-4.9 h
    adaptive 1024 -> 20.2 h | 2048 -> 14.8-18.2 h | 4096 -> 10.8-16.4 h
Width pays 1.35-1.9x for the fixed arm at 4096 and 1.2-1.9x for the adaptive
arm. The spread between the two estimates is larger than the effect at 2048,
which is precisely why the confirmation protocol leads with this measurement.

### DELIVERABLE 2c — THE ALLEGRO CLASS. MARCO'S "WEEKS" IS SOLVER-BOUND, NOT
### SCALE-BOUND, AND THE BENCHMARK HAS ALREADY BEEN RUN ON THIS CARD.

`Isaac-Reorient-Cube-Allegro-Direct` ships num_envs=8192
(allegro_hand_direct_env_cfg.py:55) with two agent cfgs: allegro_cube
(num_steps_per_env 24, 5000 iters) and allegro_hand (16, 10000). MEASURED on
THIS RTX 5090 (wandb-metadata confirms the device), in
`IsaacLabRubato/experiments/rubato-ppo-sweep/joblogs/`, July 2026, at the FULL
stock config -- 8192 envs, 10000 iterations, 1,310,720,000 env steps:

    solver            s/iter   samples/s   total wall   outcome
    mujoco (fixed)     1.033     149,145    172 min = 2.9 h   reward -7 -> 2876
    mujoco-adaptive   14.854      10,197   2184-2510 min = 36-42 h   -4 -> 2584

Both COMPLETED, twice for the fixed arm, and both LEARNED. `Isaac-Lift-
KukaAllegro` at 4096 envs x 32 steps: mujoco 1.580 s/iter, 123,634 samples/s
(12,495 of 15,000 iterations before an operator kill = 6.6 h projected full).

So the benchmark-scale default is 2.9 GPU-hours, not weeks, under the MuJoCo
backend. "Weeks" is what the SAP-adaptive arm would cost: projecting the mug
scene's 1024-env measurement forward gives 158-296 h (6.6-12.3 days) for
8192 x 10000 -- and that is moot, because it is memory-blocked first.

WOULD SAP EVEN RUN ON IT? UNKNOWN, and probably not usefully. The reorient cfg
declares a plain `PhysicsCfg()` with no SAP contact material authored; its
collision meshes carry no `physics:approximation`, so they are convex-hulled at
import (ALLEGRO_HAND_CFG's collision_props line is commented out), meaning the
pooled triangle-pair term that dominates the mug scene's 7.35/10.57 GiB has NO
counterpart there under any reachable solver. Its per-env SAP cost would be the
`72*C*D` floor scaled by dof count (Allegro ~22 dof vs Trossen ~14, roughly
1.36x -> ~6.7 MiB/env at C=2048 -> ~54 GiB at 8192: hopeless at the shipped
capacity, ~15 GiB at C=512). EXTRAPOLATION ASSUMPTIONS, stated: contact-capable
shape pairs 155/env (Allegro, resting on an unverified "one collider per link")
vs 262/env (Trossen, read from the decoded USD); no measurement of Allegro
substep demand under SAP exists at all, and in-hand manipulation is a
persistent-multi-contact regime where the adaptive arm's demand could be far
worse than the mug's. Treat the Allegro-under-SAP row as UNKNOWN, not as a
number.

 ALLEGRO REORIENT     memory                  s/iter    samples/s   5000 it  10000 it
 8192 mujoco fixed    MEASURED-fits            1.033 M    149,145 M   1.4 h M   2.9 h M
 8192 mujoco-adaptive MEASURED-fits           14.854 M     10,197 M  20.6 h M  41.2 h M
 2048/4096/8192 SAP   UNKNOWN; blocked at the shipped contact capacity by the
                      same per-env floor as the mug scene, and no SAP contact
                      law is authored for this asset

### DELIVERABLE 2d — RANKED LEVERS FOR SCALE (distinct from value at 1024)

 1. RAISING THE ENV COUNT ITSELF. Per-env GPU cost 0.52-0.73x at 4096
    (PROJECTED). Gated by memory, and the gate is the CONTACT CAPACITY C, not
    the triangle-pair pool. Cutting C 2048 -> 1024 is the single change that
    unblocks an honest 4096.
 2. MARCH COMPACTION (landed, default ON). Deletes (15/16)N dead env-slots per
    list-indexed launch: the saving is linear in N against an N-independent
    floor, so it grows with width. MEASURED 10.7-11.7% at 1024 by clean flag
    A/B. The "21% at 4096" is NOT an A/B -- it is a cross-run difference
    between two logs on a dead, contact-truncated stack; keep the direction,
    drop the digit. NOTE mc_width = max(64, N//16), so it is 64 at 1024 and 256
    at 4096; any constant tuned at 1024 does not port.
 3. COLLISION / BOUNDARY CADENCE. 7.8% of GPU at 1024 -> 13.8-15.1% at 4096
    (PROJECTED), the biggest relative promotion in the list, because it is the
    only cost that is BOTH linear in N and partly SERIAL: pass 28 measured the
    masked collide floor at 0.98-1.00 ms/fire invariant in crossing count, and
    its dominant term `_cs_scan_chunk_offsets` is a dim=1 loop over the pair
    buffer capacity (collide.py:557-616). This is the new top optimization
    target at scale.
 4. RESIDUAL FULL-WIDTH NARROWING. A deep straggler slab still dispatches
    118,604 N-proportional blocks, led by the fused armijo ladder at grid = N
    (18,696 blocks/deep-slab). Worth ~3.5% of window GPU at 1024 and ~7% at
    4096: it roughly doubles.
 5. RUN-AHEAD (NEWTON_SAP_RUNAHEAD, default OFF). **DOWN-ranked at scale, and
    this reverses the prior expectation.** Its benefit is flat-to-slightly-
    better with N (straggler cost share only falls 47.5% -> 42.7-46.2% at 4096)
    while its cost side -- the masked collide floor and the serial chunk scan --
    is linear in N and partly serial. Measured plateau value at 1024 is already
    -4% to 0. PROJECTED WORSE at 4096. It should stay OFF, and the reason is
    now mechanical rather than cautionary.
 6. FIXING THE SERIAL PER-ENV CHAINS (e.g. compute_search_direction: grid 1
    block at 4 active worlds against a 1020-block saturating grid, 13.6% of a
    straggler slab). A real lever at 1024 and an ANTI-lever at 4096 -- this is
    precisely the cost that a wider batch amortizes for free.
 7. THE ESTIMATOR (3-solve -> 1-solve, ~2.75 -> ~1.20 ms/substep arithmetic).
    Roughly N-invariant as a fraction, so it is not a scale lever, but it is the
    largest single multiplier still on the table. Comparison-semantics rail;
    Marco's call. Priced in the portfolio arithmetic below so the decision can
    be made on economics as well as physics.

CORRECTION TO A PRIOR PREMISE: the wall is NOT dispatch-bound in the sense of
"launch overhead waiting to be amortized". Kernel sum vs wall is 0.89-0.97 at
1024, i.e. 3-11% non-kernel, and pass 19's direct experiment (deleting ~78 of
~266 tiny launches per slab) moved ms/substep by 1.010 -- zero. What amortizes
at scale is GPU UNDER-OCCUPANCY, which is large. Two artifacts remain
unreconciled: pass 19 says launch-count deletion buys nothing, pass 21 says the
tail is dispatch-bound; no graph-mode profile exists to settle it (both p20 and
p21 sqlite traces were captured eager).

### DELIVERABLE 2e — PORTFOLIO ARITHMETIC. THE UNIT MARCO PLANS IN.

Horizons that resolve a LEARNING claim are 2000-4000 iterations. 150-300
iteration cells resolve failure modes, timing, capacity and stability, and
nothing about sample efficiency or final performance -- pass 31's replicates
show the reward difference at 150 iterations is inside the within-arm spread,
and pass 30's videos at 300 showed both arms doing a crude tip-and-hoist.

MEASURED per-run costs at 1024 envs on this card:
    MuJoCo fixed      3.13-3.52 s/iter (p28_train_fixed.log)      4000 it: 3.5-3.9 h
    SAP fixed         8.07 s/iter (p31, n=3)                      4000 it: 9.0 h
    MuJoCo adaptive   5.13-5.21 s/iter (mjc_1024x25.log, 25 it)   4000 it: 5.7 h
    SAP adaptive      18.21 s/iter (p31, n=3)                     4000 it: 20.2 h
    SAP 2x2 cross-eval of checkpoints  88 s/cell (p31_eval_*)     4 cells: ~6 min
    SAP play/video    41-70 s/cell, cost tracks the PHYSICS not the checkpoint

THE HEADLINE NUMBER: a 2-arm x 3-seed x 4000-iteration SAP comparison costs
3 x (9.0 + 20.2) = 87.6 h = **3.65 days**. At 2000 iterations, 1.8 days. That is
a handful of replicated learning comparisons per WEEK on one card, not per day.

THE THROUGHPUT STACK against that 87.6 h baseline:
  (a) WIDTH. At 2048 (fits today, honest) with a matched sample budget:
      3 x (6.4-6.7 + 14.8-18.2) = 63.6-74.6 h -> 1.17-1.38x. At 4096 (needs
      C=1024): 3 x (4.6-4.9 + 10.8-16.4) = 46.2-63.9 h -> 1.37-1.90x. PROJECTED.
  (b) FIXED ARM ONLY, where a cell does not need the adaptive arm: 3 x 9.0 =
      26.9 h, 3.25x cheaper than the pair. MEASURED cost; it is a different
      experiment, not a discount.
  (c) PACKING two runs at 1024: 87.6 / 1.6-1.8 = 48.7-54.8 h -> 1.6-1.8x.
      PROJECTED, unmeasured; the protocol to measure it is queued.
  (d) THE ESTIMATOR at 1.8-2.3x on the adaptive arm only: adaptive 20.2 -> 8.8-
      11.2 h, pair total 3 x (9.0 + 10.0) = 57 h -> 1.54x. HYPOTHETICAL.

  WIDTH AND PACKING COMPETE FOR THE SAME MEMORY AND CANNOT BE STACKED. Two
  1024-env runs are 21.1 GiB and fit; two 2048-env runs are 39.6 GiB at C=2048
  and do not. So the stack is (a) OR (c), then optionally (d):
      today                              87.6 h = 3.65 d
      + packing at 1024                  48.7-54.8 h = ~2.1 d
      + width at 4096 instead            46.2-63.9 h = ~2.2 d
      + estimator on top of either       ~26-36 h = ~1.2 d
  A replicated 4000-iteration learning comparison is a 1-day job only with the
  estimator. Without it, plan on 2 days at best and 3.65 days as shipped.

RECOMMENDED SHAPE — A STAIRCASE. Screening cells are cheap and answer the
questions that resolve fast; confirmation cells are expensive and few.
  SCREENING, 1024 envs x 40-150 iterations, both arms, 2-3 seeds:
      one 40-iteration pair = 148 s fixed + ~510 s adaptive = ~11 min MEASURED.
      One 150-iteration pair = 13.7 + 36-42 min = ~52 min MEASURED.
      A 24-cell screening block at 40 iterations = ~4.4 h; at 150 iterations,
      ~10 h. Packed, roughly 2.5-6 h.
      CAN SUPPORT: divergence rate, penetration, contact-capacity overflow,
      stability, s/iter, samples/s, demand, memory.
      CANNOT SUPPORT: sample efficiency, final performance, "trains better".
  CONFIRMATION, 2-4 cells at 2000-4000 iterations with >= 3 seeds per arm:
      1.8-3.65 days each as shipped.
  A WEEK ON THIS CARD = one 24-cell screening block plus one confirmation, or
  two confirmations and no screening. That is the real planning envelope.

### DELIVERABLE 2f — DESIGNS THAT FIT, PRICED

(a) THE SMALLEST CONFIGURATION WHERE FIXED FAILS AND ADAPTIVE HOLDS.
    The cheapest QUALITATIVE difference on record costs about one minute of
    GPU: 32 envs x 270 steps at k=1250, tau=1.6e-4, fixed at 8 substeps vs
    adaptive -> 620 divergence events vs 0 (p30_cor_*_seed42, reproducible at
    588 on seed 7). BUT it lives at tau 125x below authored and a step 4x FINER
    than authored, and there the adaptive arm "wins" by taking BIGGER steps
    (4.31 ms mean accepted vs the 1.04 ms that fails). That is the opposite of
    the mechanism the paper's claim needs, and the divergences are inner-solver
    non-convergence at the iteration cap, not penetration.
    THE VERSION THAT DOES SUPPORT A CLAIM, and it is also cheap: under a
    LEARNING policy at the AUTHORED law and the PRODUCTION step, the fixed arm
    diverges continuously (0.245% of terminations, 264 of 300 iterations
    non-zero) while the adaptive arm is 0.000000 in all 300
    (p30_train_{fixed,adaptive}.log). A 40-iteration 1024-env pair reproduces
    it for ~11 min MEASURED. Priced: 3 seeds x 2 arms x 40 iterations = ~33 min.
    SUPPORTS: "the fixed arm at this scene's authored law fails at a measurable
    rate and the adaptive arm does not". DOES NOT SUPPORT: any statement about
    learning outcome, near-rigid contact, or the CENIC dt-coupling mechanism.
    NOTE the clamp caveat: dt-dependent contact stiffness exists only on the
    `rn_hard` branch (verified at 8 sites in sap_warp), and the production fixed
    arm has 0.0000 of its contacts there while the adaptive arm has 0.12 --
    the branch boundary runs THROUGH the arm comparison. Engaging it is a
    task-config edit (shape ke 2500 -> 5e4..2e5 at three preset sites,
    trossen_spatula_lift_env_cfg.py:148/156/167), i.e. Marco's file.
(b) TRAIN FAST, VALIDATE ACCURATE. Train under MuJoCo-fixed (3.5-3.9 h per
    4000-iteration run at 1024, MEASURED), then evaluate the checkpoint under
    SAP. The evaluator exists and is numeric: p31_eval_probe.py, 88 s/cell
    MEASURED, so a 2x2 cross costs ~6 min. 3 seeds x 2 training arms + full
    cross-eval = ~24 h.
    SUPPORTS: dynamics-equivalence and transfer claims, and a cheap policy
    supply for physics experiments. DOES NOT SUPPORT: any claim that the
    accurate solver trains better, since it never trains.
    CAVEAT FOUND THIS PASS: `play` has no `--solver` (train-only flag; a p30
    chain died on this) -- use Hydra path overrides, as p30_play_chain.sh does.
(c) ACCURATE-SOLVER FINE-TUNE. `--resume/--load_run/--checkpoint` are
    orthogonal to `--solver` in source: the solver is latched into env_cfg
    before the checkpoint is resolved, and NOTHING compares checkpoint
    provenance with physics -- the run manifest does not even record physics.
    So fine-tuning a MuJoCo policy under SAP works today, unguarded. Price:
    3.9 h (MuJoCo 4000) + 500 SAP-adaptive iterations at 18.21 s = 2.5 h, i.e.
    ~6.4 h per seed vs 20.2 h for SAP from scratch: 3.2x cheaper per seed.
    SUPPORTS: "the accurate solver refines a policy the fast one cannot
    complete". DOES NOT SUPPORT: sample-efficiency comparison (the arms no
    longer share an initialization).
    RISK TO NAME: permissive by omission. Nothing warns on a solver change at
    resume and nothing records which physics produced a checkpoint. The harness
    records all four repo HEADs per run, which closes half of this.
(d) THE LARGEST ENV COUNT THAT FITS, RUN LONGER. Today that is 2048 at the
    shipped honest capacity (19.78 GiB, PROJECTED-fits): a 2-arm x 3-seed
    comparison at a 1024x4000-matched sample budget costs 63.6-74.6 h vs 87.6 h.
    After a C=1024 right-size and its confirmation, 4096 at 46.2-63.9 h.
    SUPPORTS: everything the 1024 design supports, with a larger batch and
    better throughput. DOES NOT SUPPORT: comparison with any 1024 result --
    changing env count changes PPO's effective batch size, which is why
    sweep-4070 keeps its results in a separate W&B project.

### QUEUED, UNRUN. Both are one command each through the new harness.

 1. `experiments/trossen-sap-scale/sweep.sh scale run`
    (configs/p33_scale_confirm.yaml). 1024/2048/4096 x {fixed, adaptive} x
    {42, 7} x 150 iterations, small-to-large, arms interleaved. Measures the
    env-count exponent per arm -- the largest measurable unknown in this
    analysis, whose two estimates differ by more than the effect they predict at
    2048 -- plus memory and honest-capacity behaviour at each width. Estimated
    ~7-11 h; the 4096 cells are expected to abort on the pool allocation, which
    the startup-abort guard turns into a clean stop with the byte count in the
    log. Run the 1024/2048 half first if wall is tight.
 2. `experiments/trossen-sap-scale/sweep.sh pack plan|run`
    (configs/p33_packing_probe.yaml). The packed-throughput multiplier: run the
    same 4 cells exclusively, then packed at 2 slots, and divide the totals.
    ~2-3 h. It is the cheapest lever on the list to verify and needs no capacity
    change.
 3. NOT YET WRITTEN AS A CONFIG, because it needs a probe rather than a
    training sweep: measure trained-policy CONTACT demand (not triangle-pair
    demand) at 4096 with the pool oversized, to justify or refute C=1024.
    p31_construct_probe.py already reports the capacity fields; it needs a
    contact-demand counter added. Do this BEFORE trusting any 4096 projection.

### PROVENANCE (all p33_ prefix; no p13-p32 artifact overwritten)

  harness   IsaacLabRubato/tools/sweep.py, tools/rubato_sweep/{config,gpu,
            runner,parse,analyze,preflight,cli}.py, preflight_probe.py,
            README.md, configs/{trossen_sap_d7,p33_scale_confirm,
            p33_packing_probe}.yaml, tests/test_sweep.py (20/20 pass),
            experiments/trossen-sap-scale/{sweep.sh,.gitignore}
  analysis  p33_scale_model.py (the memory fit + both wall scenarios; re-runnable,
            CPU-only), p33_ingest_p31.py (ingests the six pass-31 logs through the
            harness and runs the aggregator), p33_p31_ingest/*.json
  measured  p31_con_*.json (memory ladder), p31_train_*.log (1024 base),
            IsaacLabRubato/experiments/rubato-ppo-sweep/joblogs/reorient-cube-
            allegro-direct-mujoco{,-adaptive}-s42-r1.log and
            lift-kukaallegro-mujoco-s42-r1.log (the benchmark-scale points),
            summary.tsv rows for the same
  GPU       zero processes started this pass; nvidia-smi polled read-only only

### OPEN, AND EXPLICITLY NOT CLOSED BY THIS PASS

  * No 2048-env run has ever existed in any repo or scratchpad. Every 2048
    number here is interpolation.
  * No 4096 measurement postdates pass 9. Every later 4096 figure is a
    projection or a construction probe.
  * The C=1024 contact-capacity reduction is DERIVED from a source formula that
    reproduces the measured slope, not measured. Its demand margin rests on a
    peak of 144/world whose provenance under a trained policy at 4096 is not
    established.
  * The harness's GPU paths are unexecuted.
  * Allegro-under-SAP is UNKNOWN, not blocked-with-a-number.

## PASS 34 — THE ARTIFACT INSTRUMENT AND THE MATCHED-N PROTOCOL
## 2026-08-16. SOFTWARE + PROTOCOL ONLY: ZERO GPU PROCESSES STARTED. The
## 4000-iteration adaptive main run (PID 2271848, main-sap-adaptive-1024x4000-s42,
## W&B 8g1xi178) was live throughout at ~15.5-16.6 s/iter; nvidia-smi was polled
## read-only at every step and never showed a process this pass started (the only
## compute app was 2271848 at 11,129 MiB, start to finish). newton-adaptive,
## sap_warp and IsaacLab are byte-untouched. All code lands in IsaacLabRubato.
##
## Marco's plan, verbatim: "we need to show that fixed creates artifact the policy
## learns to exploit so after this run is finished or lets say whenever i see a
## reward plateue we can kill the run and same # itters as when the reward plat
## happened and then we see if fixed exploits an artifact".
##
## This pass builds the instrument that turns "is it exploiting an artifact" into
## a measurement, and the one-command protocol that fires the moment N is known.

### STACK, RE-DERIVED (the brief's certified list needed one correction)

    newton-adaptive 80d13a9a   sap_warp afd5dc6   IsaacLabRubato 93966d7
    IsaacLab        404eabb261 -- ONE COMMIT AHEAD of the brief's "certified"
                    62c165b5f0, which is its parent: 404eabb261 "Size the SAP
                    triangle-pair pool for a trained policy" is the pass-31
                    192M-pool / 65,536-per-world budget. All four trees clean.

### THE CENTRAL DESIGN DECISION, AND WHY IT IS NOT JUST THE 2x2

Cross-play transfer asymmetry is the claim's strongest form, but ON ITS OWN it
is ambiguous. A fixed-trained policy can collapse under adaptive physics for
ordinary distribution-shift reasons and never have exploited anything. So the
verdict requires THREE conditions, and the module names which ones held:

  (i)   ASYMMETRY. retention(fixed) < retention(adaptive), where retention is the
        PAIRED per-seed ratio of a policy's return under the other physics to its
        return under its own. Paired, not a ratio of means: an unpaired ratio
        hides exactly the seed variance this campaign keeps being burned by.
  (ii)  MECHANISM. Under ONE physics (fixed), the fixed-trained policy earns a
        larger share of its reward at invalid configurations than the adaptive-
        trained policy does. Holding physics constant is what separates "this
        POLICY exploits" from "this PHYSICS penetrates".
  (iii) LOCALITY. The fixed-trained policy's invalid-configuration rate collapses
        when the physics changes — the states it was farming stop existing.

(i) alone is reported as ASYMMETRIC TRANSFER WITHOUT AN IDENTIFIED ARTIFACT,
which is a real and different finding. (ii) without (i) is a physics difference
the policy never monetized. Only the conjunction is the paper's claim.

### WHAT THE TASK PAYS FOR — this is what fixes the target

Read off trossen_spatula_lift_env_cfg.py:343-354 and mdp.py:353-390, live this
pass. Of the 37 units of reward weight, 36 are gated on ONE predicate:

    object root WORLD z > 0.08 m      (LIFT_HEIGHT; the mug's rest z is ~0.021,
                                       the table slab top is z = 0.02)

    lifting_object            w=15   object_is_lifted(z > 0.08)          DIRECT
    object_goal_tracking      w=16   multiplied by (z > 0.08)            GATED
    object_goal_tracking_fine w= 5   multiplied by (z > 0.08)            GATED
    reaching_object           w= 1   1 - tanh(|obj - TCP| / 0.1)         ungated
    action_rate / joint_vel   -1e-4 each, ramping to -5e-1 at 10k steps

So "exploit" has an unambiguous operational meaning on this task: get the
object's root above 0.08 m by means the solver permits and reality does not.

### DELIVERABLE 1 — THE SIGNATURES, AND WHERE EACH THRESHOLD COMES FROM

A HARD CONSTRAINT FOUND BEFORE DESIGNING ANYTHING, and it removes a whole family
of signatures: pass 27 measured `contacts.force` as IDENTICALLY ZERO on BOTH SAP
arms (SolverSAP.update_contacts and its adaptive twin are documented no-ops;
only SolverMuJoCo fills the array). Every force-based test — force-balance
residual, contact-force implausibility, grasp-force plausibility — would
therefore silently read 0.0 on both arms and "pass". None are used. Every
signature below is built from GEOMETRY and from the object's own reported
motion, both of which are live under SAP.

The geometry source is the SAP contact jacobian's own per-(env, slot) arrays,
read live after each env step:
    contact_env_phi0_wp    signed gap at the solve anchor, float64, METRES,
                           NEGATIVE = penetration (the array pass 32's
                           differentially verified reduction reads; that
                           reduction is batch-global, so this pass adds a
                           PER-ENV one, which is what a reward correlation needs)
    contact_env_body0/1_wp global body index per contact slot, -1 = dead
    contact_env_count_wp   live slot count per env
    _set_margin0/1         per-slot collision margins, ADDED BACK so "overlap"
                           means shared volume and not "inside the margin band"
                           (NewtonShapeCfg default margin is 0.0, but it is read
                           rather than assumed)

  A1 INTERPENETRATION LIFT — the canonical exploit.
     MEASURED: deepest gripper<->object overlap at every step where the reward
     gate is open; reported as P95, max, the value AT THE ACQUISITION STEP (the
     False->True edge of the gate, where a lift is bought), and the same
     restricted to the top-decile-return episodes.
     THRESHOLD 1, relative: P95 overlap / the SAME PHYSICS' measured resting
     overlap of the mug on the table. That reference is the depth this compliant
     law needs to carry exactly one body weight, so it is a unit, not a guess.
     Flag at 3x.
     THRESHOLD 2, absolute and unanswerable by compliance: overlap deeper than
     the object's THINNEST COLLISION WALL, measured from the collision geometry's
     bounding extents. Past that the finger is through the part.
     RESIDUAL RISK, named: the resting reference carries 1x weight; a real grasp
     also carries the pads' squeeze preload, which this reference does not bound.
     The relative number is therefore suggestive alone and becomes evidence only
     next to the matched cell. The wall thickness comes from a convex
     decomposition, so a thin decomposed piece can UNDER-report the true wall.

  A2 LEVITATION — held up by nothing.
     MEASURED: share of reward-bearing steps at which the object's contact set is
     EMPTY (no jacobian row involves it, i.e. nothing within the collision margin)
     and its vertical acceleration is not -g.
     THRESHOLD: 10x the free-fall acceleration residual |a_z + g| measured in the
     SAME physics with the object dropped untouched. Integration error is
     subtracted rather than assumed away. Reference rate: 0.
     RESIDUAL RISK: an empty contact set is the collision pipeline's claim, not
     ground truth. VOID under triangle-pair overflow (see below).

  A3 ENERGY FROM THE INTEGRATOR.
     MEASURED: share of reward-bearing steps where 0.5 m|v|^2 + m g z ROSE during
     contact-free flight. Note this form is exactly conserved for a torque-free
     rigid body regardless of rotation, so rotational energy is correctly absent
     rather than missing.
     THRESHOLD: 10x the free-fall energy drift per control step, same physics.
     Reference rate: 0.

  A4 EJECTION.
     MEASURED: share of reward-bearing steps where |v_object| exceeds
     |v_TCP| + |w_object| * r. A body carried by a gripper cannot outrun it.
     THRESHOLD: 10x the measured noise floor of that same comparison, taken on
     the resting segment where both are at rest. Reference rate: 0.
     RESIDUAL RISK: a policy that legitimately tosses and re-catches the mug
     trips this. Video review is the control, which is why every cell renders one.

  A5 REWARD-WHILE-INVALID — the headline.
     exploit_fraction  = share of reward-bearing steps flagged by ANY of A1-A4.
     reward_from_invalid = share of POSITIVE income collected at those steps —
     how much of the score is bought with physics that is not real.

  A6 All of the above recomputed on the top-decile-return episodes, because a
     policy that clips through the mug while FAILING is not exploiting anything
     and pooling all episodes hides the difference.

THREE HONESTY PROPERTIES BUILT INTO THE INSTRUMENT:
  * A signature with no baseline reports UNCALIBRATED and is EXCLUDED from the
    verdict. It never falls back to a constant.
  * exploit_fraction is the union of the LIVE signatures, so it is a LOWER bound
    on invalidity. A zero means "none of these four", never "the physics was
    valid". An exploit living entirely in the friction cone is invisible to all
    four at once; that is stated in the module, not hidden.
  * If the triangle-pair buffer overflowed, EVERY signature reports VOID: a
    truncated contact set makes an empty contact set no evidence of no contact
    and a shallow overlap no evidence of no overlap.

SAMPLE HYGIENE, and it is not cosmetic: the env resets a finished world INSIDE
step(), so on a `done` step the object pose read afterwards is the NEW episode's
while the contact set is the OLD episode's and the reward belongs to the old one.
The step after a reset is equally unusable for anything built on a time
difference. Both are excluded from every mask and from the reward-gate
population — 2 of every ~150 steps, and the exclusion can only REMOVE flags.

VERIFICATION (Hephaestus). 61 CPU-only tests pass; none asserts a recomputed
value or a frozen output. The suite is built on analytic trajectories whose
correct verdict follows from physics rather than from the code:
  * exact ballistic flight (z = z0 - gt^2/2, v = -gt, so the finite difference of
    v is exact) must be flagged by NOTHING;
  * a hover at constant z with an EMPTY contact set must be flagged everywhere;
  * THE DISCRIMINATING PAIR: the identical hover, and the identical energy
    injection, WITH a contact present must be flagged by nothing — an instrument
    that ignored the contact gate would satisfy every structural check and flag
    every legitimate grasp in the campaign;
  * large POSITIVE gaps must never read as overlap (kills the sign error);
  * metamorphic: doubling the resting reference halves the ratio; deeper overlap
    never lowers the flag count; the exploit fraction is invariant to permuting
    and to duplicating environments and to a constant reward offset;
  * differential: episode segmentation is checked against an independent
    per-env accumulator loop over 5 seeded random cases, atol 1e-9.
The residual-risk list is written into the test file: the largest uncaught class
is the probe reading the WRONG ARRAY or mislabelling which body is the object,
which no trace-level test can see. Two mitigations are in the probe — the object
body index is resolved by label AND by stride and disagreement is recorded, and
the baseline's resting overlap and free-fall residual are physically
interpretable numbers that would be absurd if the array were wrong.

A BUG THIS PASS FOUND IN ITS OWN FIRST DRAFT, worth recording because it is
inherited from the shipped aggregator: `separated = |delta| > widest within-arm
range` DIVIDES BY ZERO VARIANCE. Two replicates that happen to agree closely give
a range near zero, and then ANY between-arm difference "separates" — a 0.02%
retention difference certified as a collapse in the first test run. The verdict
now requires the delta to clear BOTH the measured spread AND a declared
effect-size floor: 0.05 on retention (five points of retained score; the smallest
difference this campaign has ever resolved on anything) and 0.01 on exploit
fraction (one paying step in a hundred). These are declared judgements, labelled
as such, in one place.

### DELIVERABLE 2 — THE MATCHED-N PROTOCOL, ONE COMMAND

    cd ~/Documents/code/IsaacLabRubato/experiments/trossen-sap-scale
    nohup bash matched_n.sh <N> 42,7 7 run >> matched_n.out 2>&1 &

The default action is `plan`, which touches NO GPU, precisely so a fat-fingered
invocation cannot disturb a live run. `matched_n.sh <N>` alone plans and prices.

It runs, in order, aborting the whole protocol at the first failure:
  1. PARITY PREFLIGHT on BOTH arms, live, before a single GPU hour is spent:
     tolerances, capacities, contact law, determinism, containment, dumped off
     the CONSTRUCTED objects and diffed. Any difference outside the intended
     (solver class, substep count, adaptive tol) aborts.
  2. FIXED ARM at exactly N iterations, one run per seed, at 1024 envs with the
     adaptive run's seed, config and a 50-iteration video cadence.
  3. ADAPTIVE FILL for seeds the long run does not already cover.
  4. THE 2x2 CROSS-PLAY: per-physics baselines first, then every policy under
     every physics — 256 envs, 600 steps, eval seed 1234 held FIXED across all
     cells so the comparison is paired — writing a per-step trace, a mean
     episodic return and a video per cell.
  5. THE ARTIFACT ANALYSIS, printing the 2x2, the signatures, the verdict and
     the video-review guide.

N IS SNAPPED to the nearest k*50 + 1 at or below the requested value, and the
checkpoint tolerance is then ZERO. THE REASON MATTERS: rsl_rl saves every
save_interval = 50 iterations AND once at the end, so a run stopped at N ends on
model_{N-1}.pt while the long adaptive run passing through N only carries
model_{k*50}.pt. Those indices coincide exactly when N-1 is a multiple of 50.
Without the snap the two arms' policies differ by up to 49 iterations of
training, which is a mismatch in the one variable the whole experiment holds
fixed. Snapping costs at most 49 iterations of a horizon in the thousands.

A CONSEQUENCE WORTH STATING PLAINLY: the adaptive arm does NOT have to be killed.
Its policy at iteration N is already a checkpoint of the run in flight
(model_100.pt existed at iteration ~131 this pass, confirming the 50-iteration
cadence). Killing the run is a separate decision about whether the remaining
~3900 iterations are worth 17 more hours; it is not a prerequisite for this
experiment. If Marco wants the plateau policy AND the full run, he can have both.

WALL, as a function of N (MEASURED s/iter: fixed 8.067, adaptive 18.207, pass 31
n=3 at 1024 envs, exclusive device; cell cost PROJECTED from pass 31's 88 s
numeric cross-eval plus rendering):

    total_h(N, S, F) = [S*N*8.067 + F*N*18.207 + (4S + 2)*120] / 3600
      S = training seeds per arm, F = adaptive seeds NOT already covered

              S=1,F=0        S=2,F=1        S=3,F=2
    N =  501     1.3 h          5.1 h          8.9 h
    N = 1001     2.4 h          9.9 h         17.3 h
    N = 1151     2.8 h         11.3 h         19.8 h
    N = 2001     4.7 h         19.4 h         34.2 h
    N = 4001     9.2 h         38.5 h         67.8 h

    `.venv/bin/python tools/sweep.py cost <N> --seeds <S>` prints this for any N.

### DELIVERABLE 3 — REPLICATION, PRICED, AND WHAT A NULL MEANS

HOW MANY SEEDS THE CLAIM NEEDS: TWO PER ARM, MINIMUM, AND THE PROTOCOL'S DEFAULT
IS TWO. The aggregator returns UNRESOLVED below that and says why, and the
verdict function does the same regardless of how extreme the numbers are — there
is a test that asserts exactly this on a planted 40x asymmetry at n=1. The reason
is pass 30's own accidental control: a same-seed adaptive restart diverged 2.4x
in mean reward by iteration 9. One seed on this stack is not evidence.

THE PRICE OF HONESTY, at N ~ 1151: n=1 costs 2.8 h and CANNOT produce a verdict.
n=2 costs 11.3 h and can. The 8.5 h difference is almost entirely the second
ADAPTIVE seed (5.8 h), because the first one is free from the run in flight.
n=3 costs 19.8 h. TWO is the recommendation: three seeds doubles the bill to
tighten a spread the two-seed run will already have measured, and the campaign's
week-long planning envelope (pass 33) does not have room for it before the
question is answered once.

WHAT A NULL LOOKS LIKE, AND WHAT IT MEANS. The instrument reports four distinct
outcomes and three of them are not the headline:

  NULL — no condition separated. Both policies transfer and neither shows
    elevated invalid-configuration rates. MEANING: at this horizon and this
    contact law, fixed-step integration produced no exploitable artifact that
    the policy found. That is a real, publishable finding about THIS scene: it
    says the fixed arm's cost is wall-time and divergence rate, not a corrupted
    objective. It does NOT generalize to stiffer contact — the campaign has
    already measured that the production fixed arm has 0.0000 of its contacts on
    the dt-dependent `rn_hard` branch where the CENIC mechanism lives, so a null
    here is a null about a COMPLIANT scene.

  ASYMMETRIC TRANSFER WITHOUT AN IDENTIFIED ARTIFACT — the fixed policy does
    transfer worse, but the signatures do not separate. MEANING: the two
    timesteppers produce different dynamics and the fixed policy is brittle to
    the difference. Real, weaker, and honest.

  NO TRANSFER ASYMMETRY — signatures separate, transfer does not. MEANING: the
    fixed arm penetrates more, and the policy is not being paid for it.

  ARTIFACT EXPLOITATION SUPPORTED — all three conditions.

  UNRESOLVED — fewer than two replicates anywhere in the matrix.

WHY THE PASS-30 NULL DOES NOT PRE-EMPT THIS. Pass 30 ran the cross-play at 300
iterations and both policies transferred without degrading. At 300 iterations
neither had learned much — pass 30's own videos show a crude tip-and-hoist on
both arms — so there was nothing for an exploit to be made of. That is a null at
a horizon too short to matter, not a refutation. This protocol runs at the
horizon where the adaptive arm's reward has stopped improving, the earliest point
at which "the policy had time to find the exploit" is true.

### DELIVERABLE 4 — VIDEO

Every cross-play cell renders its own clip, to its OWN directory with its OWN
filename prefix. That is a fix by construction for a pass-30 failure: `play`
writes into the CHECKPOINT's run directory, so the two cross cells overwrote the
two same-arm ones and pass 30 had to re-run them. This probe sets
VideoRecorderCfg(output_dir=..., output_filename_prefix=<cell>) explicitly, so a
collision is impossible. The clip is the SAME rollout the numbers come from, not
a separate matched run, so a number can be checked against the frame it came
from. `sweep.py artifact` prints the review guide with the report; in short:

  fixed on FIXED     the cell the exploit should live in — a finger visibly
                     INSIDE the mug rather than pinching it, the mug rising with
                     no visible squeeze, buzzing while "held", or a lift that
                     starts the instant the gripper arrives rather than after the
                     jaws close. High exploit_fraction over a clip showing an
                     ordinary clean grasp means the signatures are measuring
                     something else; say so instead of quoting them.
  fixed on ADAPTIVE  the collapse — same approach and closure, then the mug stays
                     on the table or squirts out. If instead the ARM behaves
                     differently from the start, the transfer failure is upstream
                     of contact and the artifact reading is not supported.
  adaptive on ADAPTIVE  the control: what a policy at this horizon looks like.
  adaptive on FIXED  the mirror. If this collapses too, the asymmetry is not
                     asymmetric and the 2x2 is reporting a dynamics difference.

### PROVENANCE (all p34_ prefix; no p13-p33 artifact overwritten)

  code      IsaacLabRubato tools/rubato_sweep/{artifact,artifact_probe,
            crossplay}.py, cli.py (crossplay/artifact/cost subcommands),
            tests/test_artifact.py (61 tests, all pass), README.md,
            experiments/trossen-sap-scale/matched_n.sh, .gitignore
  smoke     p34_pipeline_smoke.py — writes the exact files the probe would write
            and runs the real aggregator over them; the planted exploit signature
            is read as ARTIFACT EXPLOITATION SUPPORTED and the planted symmetric
            case as NULL. Output in p34_smoke/ and p34_smoke_null/.
  GPU       zero processes started; nvidia-smi polled read-only throughout and
            never showed anything but the live training run.

### OPEN, AND EXPLICITLY NOT CLOSED BY THIS PASS

  * N IS NOT KNOWN. The plateau detector was still "warming" at iteration 131 of
    4000 (it needs 200 iterations of history). Everything above is parameterized
    on N and nothing presumes its value.
  * artifact_probe.py HAS NEVER RUN. Its solver access path is lifted from
    p32_pen_core.py and p31_eval_probe.py, but the per-env reduction, the body
    resolution, the wall-thickness read and the baseline measurements are new
    code executed zero times. Read baseline_fixed.json BEFORE trusting anything
    downstream: a resting overlap of 0, or a free-fall residual anywhere near g,
    means the probe is reading the wrong thing.
  * THE WALL-THICKNESS BASELINE MAY NOT RESOLVE. It reads mesh vertices off
    model.shape_source, which is defensive but unproven on this asset. If it
    returns None the absolute penetration signature reports UNCALIBRATED and the
    relative one carries A1 alone.
  * NO FORCE CHANNEL EXISTS UNDER SAP, so grasp plausibility is inferred from
    geometry and motion only. If a force-based signature is ever wanted, SAP's
    per-contact impulses live in contact_solve.contact_gamma (vec3 per (env,
    slot), tangential x/y and normal z) and are reachable — that is a probe
    someone could write, not something this pass measured.
  * THE VIDEO COST AT 256 ENVS IS UNMEASURED. Pass 30 rendered at 32. If it is
    too heavy, lower num_envs for the WHOLE cross-play rather than rendering a
    separate differently-sized rollout: the video has to be the run the numbers
    came from.

## PASS 35 — THE FOUR-ENGINE COMPARISON: DESIGNED, PRICED, AND MADE RUNNABLE;
## THE "RUNS IRL" REQUIREMENT AUDITED AND FAILED
## 2026-08-16. SOFTWARE + SOURCE AUDIT ONLY: ZERO GPU PROCESSES STARTED. The
## 4000-iteration main run (main-sap-adaptive-1024x4000-s42, W&B 8g1xi178) was in
## flight throughout; nvidia-smi was polled read-only and never showed a process
## this pass started. newton-adaptive, sap_warp and IsaacLab are BYTE-UNTOUCHED;
## every line written this pass is in IsaacLabRubato's harness, alongside pass
## 34's in-flight cross-play instrument, with no file shared between them.
## Marco's request: "a physx, mujoco fixed, sap fixed, sap adaptive comparison on
## a example that will train and run IRL will be very informative".
##
## Every number this entry inherits from an earlier pass was re-derived from the
## artifact, not from the prose. Two inherited numbers moved; both are corrected
## below and both make the current record WORSE, not better.

### TASK 1 — PhysX VIABILITY. VERDICT: **THE PRESET ALREADY EXISTS AND IS
### DOCUMENTED AS UNUSABLE; FOUR HARD BLOCKERS STAND BETWEEN IT AND A RUN.**

WHAT IS ALREADY THERE, quoted from source (trossen_spatula_lift_env_cfg.py:
128-178). The task DOES carry a physics `PresetCfg` with four alternatives:
`default` and `newton_mjwarp` (NewtonCfg, num_substeps=2), `newton_mjwarp_
adaptive` (num_substeps=1), and

    physx: PhysxCfg = PhysxCfg(
        bounce_threshold_velocity=0.01,
        friction_correlation_distance=0.00625,
        gpu_max_rigid_patch_count=2**20,
        gpu_total_aggregate_pairs_capacity=2**23,
        gpu_found_lost_aggregate_pairs_capacity=2**26,
    )

selected by the hydra token `physics=physx`. The class docstring states its own
status in as many words: "PhysX remains reachable via ``physics=physx`` as a
debugging escape hatch only."

IT HAS NEVER RUN. Not once, in any repo on this machine, at any time. Evidence,
three independent lines: `grep -li physx` over the 174 training logs in
`experiments/rubato-ppo-sweep/joblogs/` (~218 MB) returns ZERO files; every
config on disk in all four sweep drivers hard-codes `physics=newton_mjwarp`;
and the `--solver` flag — the only backend selector the harness emitted before
this pass — is Newton-only by construction (`PHYSICS_SOLVER_CHOICES` =
{mujoco, mujoco-adaptive, sap, sap-adaptive}, physics_presets.py:22-27), with
`train_rsl_rl.py:52-67` raising if the resolved physics has no `MJWarpSolverCfg`.
So there is no PhysX per-iteration cost anywhere in the campaign, and there is
no PhysX behaviour to reason from.

DEVIATION FROM THE MAINTAINED PATTERN. Every core task that supports both
backends types the field as `isaacsim_physx: PhysxCfg` plus
`physx: PhysxAutoCfg(isaacsim_physx=..., ovphysx=...)` — core/lift:483-510,
core/reach:39-41, core/cabinet, core/cartpole, core/velocity, core/handover,
core/reorient. This task types the field `PhysxCfg` directly, so
`physics=isaacsim_physx` raises `Unknown preset` (hydra.py:307) and the OvPhysX
path is unreachable. Renaming to the maintained triple is bookkeeping, but it is
Marco's file.

THE FOUR HARD BLOCKERS, ordered by how early they fire:

 B1 THE CONTACT SENSORS ARE PINNED TO NEWTON. Both are
    `isaaclab_newton.sensors.ContactSensorCfg` (env_cfg:214-229), whose own
    docstring says "Use this class directly only to force the Newton
    implementation regardless of the active backend". Its `_initialize_impl`
    calls `NewtonManager.add_contact_sensor(...)` unconditionally and wraps any
    exception in `RuntimeError` (contact_sensor.py:289-304). Under PhysX the
    Newton manager is not the active backend. VERIFIED IN SOURCE this pass;
    that it raises rather than degrades is an INFERENCE from those lines.
    FIX (Marco's, one of two): (a) delete both sensors under the PhysX preset —
    legitimate, because NOTHING reads them (below); or (b) switch both to
    `isaaclab.sensors.ContactSensorCfg`, the backend-dispatching class, and
    accept that `filter_shape_prim_expr` is silently ignored under PhysX
    (contact_sensor_cfg.py:98,107). (a) is correct today; (b) is required the
    moment a contact term is added.
 B2 `--solver` BECOMES ILLEGAL. Every launch script in the task directory
    passes it. FIX (in-grant, LANDED THIS PASS): `Arm.solver` is now optional in
    the harness; a PhysX arm declares `solver: null` and names its engine with
    `extra_args: ["physics=physx"]` alone. A config with `solver: null` and no
    `extra_args` is now refused, because that combination silently runs the
    task's DEFAULT preset under another arm's name.
 B3 `--viz newton` + PhysX. The Newton visualizer calls `NewtonManager.
    get_model()/get_state()/get_contacts()` (newton_visualizer.py:616-628,
    735-747) and `_validate_runtime` (sim_launcher.py:407-444) has NO guard for
    this combination. FIX (in-grant, LANDED): `viz` is per-arm in the p35
    configs, not global; the PhysX arm carries no `--viz`. The task's own
    `sim.default_visualizer_cfg = NewtonVisualizerCfg(...)` (env_cfg:499-503) is
    only an eye/lookat hint and is harmless, but it does hardcode a Newton class
    into `__post_init__` — Marco's, cosmetic.
 B4 THE RUN LEAVES THE KITLESS STACK. A `PhysxCfg` makes `has_kit_physics` true
    (sim_launcher.py:379-395, 522-524), so the PhysX arm boots Isaac Sim where
    all three Newton arms run kitless. Nothing in the harness breaks, but the
    startup cost, the memory floor and the renderer all change, and none of them
    has ever been measured here. This is why the PhysX arm's s/iter is UNKNOWN
    rather than bracketed from the Newton arms.

THE SILENT DIFFERENCES, which are worse than the blockers because they do not
fail. Each is a field the task already carries that is INERT under Newton and
LIVE under PhysX, or vice versa:

  * The mug's entire contact recipe evaporates. `mug_inomata_white.usd` carries
    `mjc:solimp = (0.9, 0.999, 0.002, 0.5, 2.0)` and `mjc:priority = 1` on all
    12 collision prims, authored deliberately (convert_mug.py:111-118: dmax
    0.999 "collapsing the pinch embed", priority 1 to "WIN pair mixing"). Those
    are MuJoCo-namespace custom attributes, consumed only at
    solver_mujoco.py:930-952. Under PhysX they are inert custom attrs; under
    BOTH SAP ARMS they are ALSO inert. There is no `UsdPhysics.MaterialAPI`
    bound anywhere in the mug or the rig. So the object's contact law differs
    across all three engine families and the difference is not in the task cfg.
  * `RigidBodyPropertiesCfg(solver_position_iteration_count=16,
    solver_velocity_iteration_count=1, max_depenetration_velocity=0.5,
    max_linear_velocity=1000, max_angular_velocity=1000)` on the object
    (env_cfg:202-209) and `max_depenetration_velocity=5.0` on the robot
    (assets.py:58-73) come alive under PhysX and are dead under Newton — the
    task's own comment at :196-198 says so. `max_linear_velocity=1000` clamps
    BEFORE the `object_speeding` termination's 20 m/s threshold can trip, so a
    termination the campaign treats as a shared experiment metric
    (env_cfg:385-390 calls it exactly that) changes meaning on the PhysX arm.
  * `physics_diverged` is a permanent no-op on PhysX, as it is on both MuJoCo
    arms (mdp.py:167-174 returns all-False when the mask is None). After pass
    29 the two SAP arms CAN excise a diverged world from the training
    distribution and the other two CANNOT. This is the largest MDP asymmetry in
    the four-way and it is not new — it is D2b, still open.
  * `_validate_solver_substeps` returns early when the physics has no
    `solver_cfg` (env_cfg:475-477), so the PhysX arm has no stability check of
    any kind and inherits `sim.dt = 1/120` unvalidated.
  * `enabled_self_collisions=False` (assets.py:64-70) is justified in its own
    comment by MuJoCo convex-hull overlap in the grasp channel. That premise
    does not hold under PhysX, where the rig meshes carry
    `physics:approximation="none"`, so the setting becomes unmotivated.

THE RECIPE, concretely, for whoever lands it (all four items are Marco's):

    trossen_spatula_lift_env_cfg.py:172-178  rename physx -> isaacsim_physx and
        add `physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)`,
        matching core/lift:485-510.
    trossen_spatula_lift_env_cfg.py:214-229  make the two contact sensors a
        preset field: present under the Newton alternatives, ABSENT under
        isaacsim_physx. Nothing reads them, so nothing regresses.
    trossen_spatula_lift_env_cfg.py:499-503  make default_visualizer_cfg a
        preset field too, or drop it for the PhysX alternative.
    assets.py + the mug USD  author a `RigidBodyMaterialCfg` and a
        `CollisionPropertiesCfg(contact_offset=..., rest_offset=...)` for the
        mug, the finger pads and the table guard, as core/lift:43-46 does.
        WITHOUT THIS the PhysX arm runs on the engine default material
        (mu 0.5, restitution 0) while the Newton arms run on mu 1.0 — a
        2x friction difference on the axis the task's success depends on.

The harness is ready for the arm today; the task is not. Until B1 and B3 land,
run `p35_threeway_screen.yaml`, which is the same design minus PhysX and runs on
today's stack with no edit to any of the three repos.

### TASK 2 — THE COMPARISON, AND WHAT IT MAY AND MAY NOT CLAIM

THE CONFOUND STATEMENT, first, because everything else is downstream of it.

**AN ENGINE COMPARISON IS NOT A TIMESTEP COMPARISON.** The four arms differ in
their CONSTITUTIVE LAW, not in a discretization of one shared law:

    PhysX         impulse/TGS with contact_offset/rest_offset compliance and a
                  solver-iteration budget; on this asset, the ENGINE DEFAULT
                  material, because none is authored (mu 0.5).
    MuJoCo-Warp   solref/solimp. solref comes from ke/kd by
                  timeconst = 2/kd, dampratio = (kd/2)*sqrt(1/ke)
                  (newton_manager_cfg.py:84-87) = (0.02, 1.0) at the authored
                  defaults, i.e. exactly MuJoCo's stock compliance; solimp is
                  overridden on the mug alone to (0.9, 0.999, 0.002, 0.5, 2.0).
    SAP           R_n = max(beta^2/4pi^2 * W, 1/(h k (h+tau))) with authored
                  pair k = 1250 N/m and tau = 0.02 s. IGNORES the mug's
                  mjc:solimp entirely.

Three different laws, on the same USD. Therefore:

  * Any statement of the form "engine X is better/more accurate than engine Y"
    is UNAVAILABLE from this design. There is no ground truth and no shared law.
  * The ONLY pair that isolates the timestep policy is SAP-fixed vs
    SAP-adaptive — same contact law, same tolerances (equalized pass 29), same
    capacities, same advance per boundary (8.33333e-3 s on both, measured pass
    27). And even that pair is NOT clean: `attempt_consistent_r` is ON for the
    adaptive arm and structurally inert on the fixed arm, and it scales W —
    hence rn_hard AND the tangential rt — by s ~ 2.1 in every COMMITTED half
    solve. For a task whose success is a friction-held grasp, a 2x softer
    tangential regularization on one arm only is a confound, and it has never
    been separated. THE FIX IS ONE ENV VAR ON AN EVALUATION-ONLY ARM
    (`NEWTON_SAP_ATTEMPT_CONSISTENT_R=0`), and it is COMPARISON SEMANTICS =
    MARCO'S. It is not in the p35 configs. Naming it is the honest move; adding
    it silently would not be.

WHAT IS EQUALIZED, and the preflight now PROVES it rather than asserting it
(`preflight.by_family: true` with an explicit `contract_keys` list; a
cross-engine sweep with an empty contract is refused by the config validator):

    task id and registered MDP        one gym id, one env cfg
    every MDP term                    name, function, weight and mode, per
                                      manager, for all six managers
    action space                      7 dims: 6 arm joint-position + 1 binary
                                      gripper; scale 0.25, clip (-6, 6)
    observation space                 33 dims, one `policy` group, actor and
                                      critic identical
    control contract                  sim.dt 1/120, decimation 4 -> 30 Hz
                                      control, 5 s = 150-step episodes
    batch                             1024 envs x 24 steps per iteration
    policy architecture + PPO cfg     one rsl_rl entry point
    seeds                             42, 7, 13 on every arm
    iteration budget                  identical per arm
    asset                             one USD per body

WHAT CANNOT BE EQUALIZED, listed so the paper says it before a reviewer does:
contact law (above); solver tolerances (SAP's optimality_rel_tol 1e-8 vs
MuJoCo's iteration/ls budget 100/15 vs PhysX's position/velocity iteration
counts 16/1 — these are not commensurable quantities); substep structure (SAP
fixed 2 per boundary, SAP adaptive a march, MuJoCo 2, PhysX its own); collision
pipeline and geometry approximation; the contact-force writeback (LIVE on
MuJoCo, IDENTICALLY ZERO on both SAP arms, unmeasured on PhysX — pass 27 F9);
and `physics_diverged`, which is live on the two SAP arms and a permanent no-op
on the other two.

**THEREFORE THE CLAIMS THIS DESIGN IS ALLOWED TO MAKE:**

 C1 COST. Wall to a fixed sample budget, per engine, on one task. Fair, because
    the budget and the batch are equalized. An ENGINEERING claim, and the
    strongest one available from the training stage.
 C2 LEARNABILITY. Whether each engine reaches the task's success predicate at
    that budget. Bounded, 0-1, and far better behaved than mean reward.
 C3 TRANSFER. The 4x4 cross-play matrix. This is where the claim lives.
 C4 ADAPTIVITY. SAP-fixed vs SAP-adaptive alone, ACR-confounded and labelled so.
 C5 SELF-CONSISTENCY. The refinement column (below) — the one accuracy axis this
    design legitimately has.

AND THE ONE IT MAY NOT: that any engine is more accurate than any other.

**WHY THE CLAIM GOES ON THE MATRIX AND NOT ON THE TRAINING CURVES.** Measured,
this task, this card:

    instrument                          spread                    provenance
    training-curve reward, 3 seeds      within-arm RANGE is 97%   p33_p31_ingest/
      at 150 iterations                 of the arm's own mean     summary.json
    one evaluation cell                 SEM 0.28-0.34 on a mean   p31_eval_*.json
      (256 envs x 600 steps,            of 112-137, i.e. 0.25%
       1024 completed episodes)

The evaluation instrument is roughly two orders of magnitude more sensitive than
the training instrument, and Task 4 prices it at about 2-4% of the training
bill. A design that puts its claim on the learning curves is spending 96% of its
GPU hours on its weakest measurement. Note precisely what the pooling does and
does not remove: it removes EPISODE noise, not SEED noise. The seed level offset
is cancelled instead by pairing — pass 34's `retention()` takes the per-replicate
ratio R(i,j)/R(i,i) rather than a ratio of means, and that is the statistic to
report.

**THE MATRIX, AND ITS REUSE OF PASS 34.** 4 policies x 4 physics x 3 training
seeds, evaluation seed fixed at 1234 so every cell is paired. Nothing new was
written for the statistics: pass 34's `crossplay_verdict(returns, exploit,
fixed=..., adaptive=...)` is already parameterized on two arm names, so the 4x4
is six of its 2x2s, and its three-condition structure is adopted verbatim —

    asymmetry  policy i retains less off its own physics than j does in the
               mirror direction;
    mechanism  under ONE physics, i earns more of its reward at invalid
               configurations than j does;
    locality   i's invalid-configuration fraction collapses when the physics
               changes.

Only the conjunction is an artifact-exploitation claim. Asymmetry alone is
brittleness or distribution shift and must be reported as such — that is pass
34's wording and it is right.

**THE REFINEMENT COLUMN — the accuracy axis that survives the confound.** Cross-
engine "which is right" is unanswerable here. Within an engine it is not: replay
a policy under the SAME contact law at a 4x finer fixed step and ask whether its
score survives. A score that collapses was bought from that engine's
discretization error, and no cross-engine ground truth is needed to say so.
Implemented as two extra physics columns (`sap_fixed_ref`, `mujoco_fixed_ref`,
num_substeps 2 -> 8, `trainable=False` so they can never be mistaken for a fifth
policy). Two deliberate choices:
  * the SAP-ADAPTIVE policy's reference is `sap_fixed_ref`, NOT a tightened
    estimator tolerance. `tol` is a campaign rail and is the physics being
    demonstrated; it is not an evaluation knob.
  * PhysX has NO reference arm. Its refinement schedule (substeps? iteration
    counts? both?) has never been run here, and inventing one would be a guess
    wearing a measurement's clothes.
The statistic is pass-34's `retention()` again, with `own` = the production arm
and `other` = its reference. No new estimator.

**WHAT THE 4x4 CANNOT DO ON A PhysX COLUMN.** Signature A1 (interpenetration)
reads the solver's own contact set; that channel exists under Newton and has no
PhysX equivalent wired up. PhysX cells must set `Trace.pen_channel_live=False`
so A1 reports UNCALIBRATED. A2 (levitation), A3 (energy gain) and A4 (ejection)
are purely kinematic and DO port. Three of four signatures survive the four-way.
Reporting a silent zero instead would be the worst available outcome.

### TASK 3 — "RUNS IRL". VERDICT: **NOT FIT, AND THE DECISIVE DEFECT IS NOT A
### BUG BUT A DELIBERATE, DOCUMENTED CHOICE.**

THE TASK HAS ZERO DOMAIN RANDOMIZATION. Quoted in full — this is the entire
`EventCfg` (env_cfg:413-428): `reset_all` (reset_scene_to_default) and
`reset_object_position` with `pose_range` = {x: (0,0), y: (0,0), z: (0,0)},
`velocity_range` = {}. Both deterministic; the second is a functional no-op on
top of the first. No prestartup, no startup, no interval terms. Nothing is
randomized: not friction, not mass, not inertia, not COM, not actuator gains,
not armature, not joint friction, not object pose, not object scale, not robot
init pose, not gravity, not external disturbance, not observation noise. The
comment says why — "the real-world protocol is a tape-measured FIXED placement
... Prove pickup first" — and that is a defensible research order. It is also
the end of the sim-to-real conversation until it is reversed.

AND `enable_corruption = True` IS A DECOY. It is set (env_cfg:326) while every
`ObsTerm` leaves `noise=None`, whose documented behaviour is "no noise is
added" (manager_term_cfg.py:174-175). The PLAY variant then disables (:534)
something that was never on. A reader of this config concludes observation noise
is configured. It is not. VERIFIED IN SOURCE this pass.

THE POLICY'S THIRD INPUT DOES NOT EXIST ON HARDWARE. `object_position =
ObsTerm(func=mdp.object_position_in_robot_root_frame)` is `root_pos_w` — the
simulator's exact mug pose, machine precision, 30 Hz, zero noise, zero latency,
zero dropout, zero occlusion. There is no camera sensor in the task (the rig USD
has three D405s; none is instantiated), no pose estimator, no teacher/student
split — `obs_groups = {"actor": ["policy"], "critic": ["policy"]}`. Deploying
this policy requires a perception system with ~zero error, which is exactly what
does not exist. The older Rubato cube task at least put object pose in a
separate `privileged` group; this task collapsed that split.

THE CONTACT LAW IS NOT A GRASP. This is the sharpest finding and it is DERIVED
from two numbers both read off source this pass, with no geometry assumed:

    authored per-shape ke        2500 N/m   (NewtonShapeCfg default, VERIFIED;
                                             no material is bound anywhere)
    series-combined pair k       1250 N/m   (live probe, passes 25/27/30)
    gripper carriage drive k     1000 N/m   (ImplicitActuatorCfg, prismatic)

Static equilibrium of a commanded close: the pad and the object share the
remaining commanded stroke d0 in inverse proportion to their stiffnesses, so the
embed is

    delta / d0 = k_drive / (k_drive + k_contact) = 1000 / 2250 = **0.444**

**THE FINGER EMBEDS 44% OF THE REMAINING COMMANDED STROKE INTO THE MUG**,
whatever the geometry. And the only reason it is not worse is that the gripper
drive was softened 218x below the vendor's own USD authoring: the shipped rig
authors carriage `stiffness = 217687` N/m, against which the same contact law
gives delta/d0 = 217687/218937 = **0.994** — the pad passes essentially all the
way through. Two sim-only softenings are cancelling each other out.

IS 1250 N/m PLAUSIBLE FOR A REAL GRIPPER ON A REAL MUG? No, by 0.6 to 2.7
decades. DERIVED, assumptions stated: a compliant pad in series with an
effectively rigid ceramic/plastic wall gives k ~ E*A/t with pad modulus
E = 1-10 MPa, patch A = 0.5-2 cm^2, thickness t = 3-10 mm, i.e.
k = 5e3 .. 6.7e5 N/m. The authored 1250 sits below the bottom of that range.
The assumptions are the pad's modulus, patch area and thickness; ALL THREE ARE
REPLACEABLE BY A TEN-MINUTE BENCH MEASUREMENT — press the real gripper pad onto
a kitchen scale through a dial indicator and read force against displacement.
That single measurement converts the largest sim-to-real unknown in this task
into a number, and it needs no GPU and no simulator.

AND HERE IS THE CONNECTION THAT MUST NOT BE OVERSOLD. Pass 30 measured this
scene's near-rigid branch boundary: the clamp takes the majority of contacts at
k ~ 2.5e4 and 100% by k = 1e6 (p30_reg_fixed_s2_seed42.json). The plausible real
pad range 5e3..6.7e5 STRADDLES that boundary. So authoring a physically
defensible stiffness is the same edit that engages the branch where the CENIC
dt^-2 coupling exists (env_cfg:148/156/167, shape ke). **THIS DOES NOT RESURRECT
THE ADAPTIVE ADVANTAGE.** Pass 30 already swept 6.9 decades of stiffness and tau
from 0.02 down to exactly 0, on four fixed substep sizes and the adaptive march,
two seeds, and found NO stiffness at which the fixed arm breaks while the
adaptive arm holds — because the clamp caps the effective stiffness of the
penetration-setting contacts at 2.34x the authored value at the production tau.
That negative stands and this pass does not touch it. What the stiffness edit
buys is a scene whose grasp is not a 44%-embed squeeze. Those are two different
claims and conflating them would be the easiest mistake available here.

THE REST OF THE AUDIT, briefly, each verified in source:

  * FRICTION IS mu = 1.0 EVERYWHERE, undifferentiated: pad-on-glaze,
    mug-on-tabletop, arm-on-anything (NewtonShapeCfg default, no material
    bound). Real pad-on-glazed-ceramic is ~0.4-0.8 and mug-on-wood ~0.3-0.5.
    A grasp that only holds at mu 1.0 has nowhere to go on hardware.
  * ARM GAINS DISAGREE WITH THEMSELVES BY 3.4x. `assets.py:94-96` says the
    gains are "verbatim from Trossen's official MuJoCo model" and sets
    stiffness 200 for joints 0-2; the same vendor's USD authors 664/735/738 for
    those joints. The softer source is silently selected and the discrepancy is
    undocumented. A policy trained against kp=200 will not behave the same on a
    driver running the shipped gains.
  * THE 400 N GRIPPER EFFORT CAP IS DEAD. At k=1000 N/m a fully-commanded
    carriage (0.044 m) can generate at most 44 N, so `effort_limit_sim=400.0`
    can never bind. It reads as a safety limit and is not one.
  * `enabled_self_collisions=False` (assets.py:70). Trained trajectories may be
    self-colliding or rail-colliding on the real rig. This is a hardware SAFETY
    issue, not a fidelity one.
  * NO PER-STEP ACTION RATE BOUND IN THE MDP. Actions are ABSOLUTE joint
    positions, target = q_home + 0.25 * clip(a, -6, 6), i.e. +-1.5 rad from home
    re-commanded from scratch every 33 ms. The only rate bound is Newton's
    actuator-side clamp (`NewtonCfg.enforce_velocity_limit=True`, default,
    clamping the commanded target's change to velocity_limit*dt). THAT CLAMP
    HAS NO EQUIVALENT IN THE REAL DRIVER. Deploying this policy requires
    re-imposing |dq_target| <= v_limit/30 per tick in the deployment loop, or
    the first action step commands a 3 rad swing at the hardware.
  * THE HAND-OFF DOCUMENT IS WRONG. `REAL_SETUP.md` describes the SPATULA (the
    task loads the mug), quotes 33.0 cm forward (the config is 0.450 m), and
    promises "+-10 cm lateral / +-7.5 cm forward jitter" that does not exist.
    The in-code comment at env_cfg:85 is 12 cm stale against its own constant.
    Whoever tape-measures the real setup will place the mug in the wrong spot.
  * THE PHANTOM BODY IS CONFIRMED AND THE FIX IS ONE LINE.
    `/stationary_ai/follower_left_ee_gripper_link`: PhysicsRigidBodyAPI +
    PhysicsMassAPI, mass 1e-4 kg, diagonalInertia (0,0,0), centerOfMass
    (-inf,-inf,-inf), NO children (no collider, no visual), and NO joint in the
    stage's 19-joint inventory references it. It free-falls for the whole run;
    `reset_scene_to_default` never touches it because it is not a registered
    asset. It cannot corrupt observations (not in `robot.body_names`), and the
    -inf COM is filtered rather than propagated (import_usd.py:736-738,
    2574-2576). The RIGHT twin is already disabled — `stationary_ai_task.usda`
    lines 723-727, verified verbatim this pass:
        over "follower_right_ee_gripper_link" ( active = false ) { }
    The left one has no such block. Adding the mirrored four lines before the
    closing brace at :728 is the entire fix. Hygiene, 7 of 22 coords per world,
    zero transfer impact. Marco's file.
  * THE CONTACT SENSORS AND EIGHT MDP FUNCTIONS ARE DEAD, re-confirmed. Two
    `NewtonContactSensorCfg` are constructed and stepped every tick; not one
    MDP term reads them. `pad_handle_contact` additionally targets
    `follower_left_carriage_.*` rather than the finger pads, so it would likely
    be mistargeted if it were ever wired up. Seven further non-contact MDP
    functions are also unreferenced.
  * WHAT IS TRANSFER-FRIENDLY AND SHOULD BE KEPT: the 30 Hz control rate; the
    absolute-joint-position + binary-gripper interface; the vendor joint
    position limits; action scale 0.25 (halved deliberately after a slow-mo
    review); and the fact that the policy is CONTACT-BLIND, which matches a
    sensorless real gripper exactly.

THE PRIORITIZED FIX LIST, split by who owns it.

  IN-GRANT (solver/manager/harness — this pass or the next):
   G1 LANDED. Multi-engine harness support: optional `--solver`, family-aware
      preflight with an explicit equalized-axes contract, a backend-agnostic
      contract probe, and one arm table shared by training, preflight and
      evaluation. Details below.
   G2 The 3-line change to pass 34's `artifact_probe.py` so the cross-play probe
      resolves its physics through `physics_arm.apply_to` instead of writing
      `solver_cfg.backend = "sap"` by hand. NOT MADE THIS PASS: that file is
      being written by pass 34 right now and two passes editing one file is how
      work gets lost. The exact replacement is in the crossplay config header.
   G3 `Trace.pen_channel_live=False` on PhysX cells so A1 reports UNCALIBRATED
      rather than a silent zero. Belongs with G2.
   G4 A convergence/step-count certificate for the MuJoCo and PhysX arms
      equivalent to the one the fixed SAP arm got in pass 29, so all four arms
      report whether their solve converged. Currently only the SAP arms do.

  MARCO'S (task / scene / asset — recipes only, nothing implemented):
   M1 **DOMAIN RANDOMIZATION.** The single decisive item. Minimum set for a
      transfer attempt, in priority order, with the event mode each needs:
        friction        `randomize_rigid_body_material` on mug, finger pads and
                        table, startup or reset. mu_s in [0.4, 1.0], mu_d in
                        [0.3, 0.9], restitution [0.0, 0.1]. FIRST, because mu
                        is currently 1.0 everywhere and the grasp depends on it.
        object mass     `randomize_rigid_body_mass`, reset, +-30% around
                        0.0181 kg (a real mug is 0.2-0.4 kg — the authored mass
                        should be checked against the physical object before
                        randomizing around it).
        object pose     `reset_root_state_uniform`, reset. The ranges
                        REAL_SETUP.md already promises: +-10 cm lateral,
                        +-7.5 cm forward, and yaw. Non-negotiable for a
                        tape-measured placement with human error.
        actuator gains  `randomize_actuator_gains`, startup or reset, kp/kd
                        +-30%. This is what covers the 3.4x vendor-source
                        disagreement instead of having to resolve it.
        joint friction/ `randomize_joint_parameters`, startup.
        armature
        observation     noise on `joint_pos`, `joint_vel` and above all
        noise           `object_position` — the last one is the perception
                        system's error budget and must be sized from the
                        estimator that will actually be used, not guessed.
                        `enable_corruption=True` is already set, so this is
                        purely a matter of giving the terms a `noise=`.
        latency         action and observation delay. Not expressible as a
                        stock event term; needs a buffer in the action/obs
                        pipeline. Flagged, not specified.
      Per the events skill: friction/mass/gain terms are backend-specific and
      must be checked against `envs/mdp/events.py` for Newton vs PhysX support
      before being written into a `PresetCfg`, and every one needs a
      small-num_envs repeated-reset smoke test per backend before training.
      Also per that skill: keep a SEPARATE deterministic nominal evaluation, or
      the cross-play matrix stops being comparable to the pre-DR results.
   M2 **PERCEPTION.** Either instantiate one of the three D405 cameras and split
      the MDP into privileged (teacher) and observable (student) groups, or
      commit to an external pose estimator and randomize its error and latency
      into `object_position`. Without one of these the policy is not deployable
      at any level of DR.
   M3 **CONTACT STIFFNESS AND FRICTION AUTHORING.** Bench-measure the pad, then
      author `ke` at env_cfg:148/156/167 and a real material. Note this moves
      the scene into or across the near-rigid branch, which is a physics change
      the campaign has already characterized (pass 30) and which must not be
      made mid-campaign without re-running the baseline.
   M4 GRIPPER DRIVE. Reconcile 1000 N/m against the vendor's 217687 N/m, or
      document why the sim gripper is a spring. Paired with M3, because the
      44% embed is the product of the two.
   M5 SELF-COLLISION. Turn `enabled_self_collisions` back on and solve the
      convex-hull overlap properly, or accept that trained trajectories are
      unvalidated for self-collision on hardware.
   M6 The phantom body one-liner (D13, still open since pass 28).
   M7 Delete the two dead contact sensors and the fifteen unreferenced MDP
      functions, or wire them — but wiring one requires the SAP writeback first
      (F9), or SAP and MuJoCo stop being the same task.
   M8 Fix `REAL_SETUP.md` and the stale 33.0 cm comment before anyone
      tape-measures anything.

READINESS VERDICT: the task is a good ENGINE-COMPARISON vehicle today and is not
a sim-to-real vehicle at all. M1 and M2 are prerequisites, not improvements. The
four-way comparison does not need them — it needs the MDP to be identical across
arms, which it is — so the comparison should not wait on them. But the sentence
"and run IRL" is not true of this scene today and no amount of solver work makes
it true.

### TASK 4 — THE PRICE

MEASURED s/iter at 1024 envs on this card, this scene. Every row RE-DERIVED
THIS PASS from the artifact rather than quoted from the ledger:

    arm            s/iter                     window              artifact
    mujoco_fixed   3.61 (last-10)             40-iteration run,   p28_train_fixed
                   3.57 (last-20)             STILL RISING        .log
    sap_fixed      8.067 [7.442, 8.510]       last-50 of 150, n=3 p33_p31_ingest
    sap_adaptive   18.208 [16.410, 19.170]    last-50 of 150, n=3 p33_p31_ingest
    physx          NONE. Never run.           --                  --

**CORRECTION TO THE PASS-33 RECORD.** Pass 33 quoted MuJoCo-fixed as
"3.13-3.52 s/iter". Re-derived from the same file this pass: the whole-run mean
is 3.132 and the last-10 mean is 3.614, with the series climbing monotonically
1.80 -> 3.75 across its 40 iterations and not yet flat. So (a) the quoted upper
bound was low, and (b) the number is not window-matched to the SAP rows, which
are last-50 tails of runs 3.75x longer. Treat 3.61 as a LOWER BOUND on the
steady state. This is not a small bookkeeping point: iteration time on this task
RISES with training because the policy finds more contact-rich states — the live
4000-iteration run reads 15.19 s/iter over its first 106 iterations and 18.64
over its last 10 — so EVERY short-run s/iter in this campaign is a lower bound
on the 4000-iteration mean. The screening stage exists partly to replace all
four of these with one window-matched measurement.

PHYSX IS UNKNOWN, NOT BRACKETED. Bracketing it from the Newton arms would
require assuming the two stacks are comparable, and the PhysX arm boots Kit
while the others run kitless. The screening stage measures it. Everything below
is quoted twice: once with PhysX omitted (a hard floor) and once with PhysX
assumed equal to MuJoCo (the optimistic case the screening will confirm or
refute).

THE FULL DESIGN, 1024 envs, 3 seeds, 4 arms, 4000 iterations
(= 98.3M env steps per run), computed by the harness itself
(`sweep.py plan ... --cost-per-iter`):

    arm            3 x 4000 iterations         hours
    mujoco_fixed   43,320 s                    12.0
    sap_fixed      96,804 s                    26.9
    sap_adaptive   218,496 s                   60.7
    physx          UNKNOWN (12.0 at MuJoCo's rate; 33.3 at 10 s/iter)
    ------------------------------------------------------------------
    THREE ARMS ONLY                            99.6 h  = 4.2 days
    FOUR ARMS, PhysX at MuJoCo's rate         111.6 h  = 4.7 days
    FOUR ARMS, PhysX 3x slower                132.9 h  = 5.5 days

    + cross-play evaluation (72 cells: 4 policies x 6 physics x 3 seeds,
      including the two refinement columns, plus 6 baselines and video)
                                              ~4 h PROJECTED
      Only the two SAP production columns are measured (88 s and 126-139 s per
      cell, p31_eval_*); MuJoCo, PhysX and the 4x-substep reference columns are
      scaled from their training s/iter and are PROJECTIONS.

    GRAND TOTAL, four arms at 4000            ~116-137 h = 4.8-5.7 days

THE STAGED VERSION, and every stage is a human gate:

    STAGE A  p35_threeway_screen.yaml   3 arms x 3 seeds x 200 iters
             5.0 h MEASURED-BASED. Runs on today's stack with no edit anywhere.
             BUYS: a window-matched n=3 s/iter for all three Newton arms (which
             does not exist today), the failure-mode census, memory, capacity
             overflow, and proof the 4-way harness path works end to end.
             CANNOT BUY: any reward statement. At 150 iterations the within-arm
             reward range is 97% of the mean; 200 does not fix that.
    STAGE A' p35_fourway_screen.yaml    adds PhysX, after Marco's B1/B3 fixes
             5.6-7.1 h. Its real product is the first PhysX number ever taken
             on this machine.
    STAGE B  p35_fourway_full.yaml      3 seeds at the chosen horizon
             4000 -> 111.6 h | 2000 -> 55.8 h | 1500 -> 41.9 h
             CHOOSE THE HORIZON FROM DATA, NOT FROM A ROUND NUMBER. The live
             main run finishes in ~16 h and hands over a 4000-iteration reward
             curve for free. Pick the smallest horizon at which its slope is
             inside the seed-spread band. If that is 1500, Stage B costs 38% of
             the 4000 figure and the claim is unchanged.
    STAGE C  p35_fourway_crossplay.yaml  ~4 h
             The matrix, the refinement column and the verdict. 3-4% of the
             bill, carrying the claim.

    CHEAPEST DEFENSIBLE PATH  A + B(1500) + C  = ~51 h = 2.1 days
    FULL PATH                 A' + B(4000) + C = ~121 h = 5.1 days

WHAT EACH HORIZON CAN AND CANNOT SHOW, stated once so nobody has to infer it:

    200 iterations   CAN: s/iter, samples/s, memory, contact and triangle-pair
                     overflow with the first offending iteration, divergence and
                     speeding termination rates, NaN, whether an arm runs.
                     CANNOT: sample efficiency, final performance, "trains
                     better", ANY reward comparison.
    1500-4000        CAN: whether each engine reaches the success predicate;
                     the checkpoints the matrix needs; wall to a fixed budget.
                     CANNOT: a between-arm reward difference from the training
                     curves alone — the within-arm seed spread has never been
                     measured at this horizon on this task, and at every horizon
                     where it HAS been measured it exceeded every between-arm
                     difference reported. Report the endpoint as success rate
                     and put the comparison on the matrix.

RESERVATIONS ON THE PRICE, named: all four training rows assume ONE GPU PROCESS
AT A TIME, which is the campaign rail. Packing two 1024-env runs is projected at
1.6-1.8x (pass 33, UNMEASURED) but would void every timing in the sweep, and
the p35 configs declare `timing_sensitive: true`, so the harness refuses it. The
evaluation total is a projection everywhere except the two SAP columns. And none
of this can start until the live 4000-iteration run finishes.

### SOFTWARE LANDED (IsaacLabRubato only; the other three repos are untouched)

    tools/rubato_sweep/physics_arm.py     NEW. One arm table, read by training,
        preflight and evaluation, so an evaluation cannot silently resolve a
        different engine than the training it is judging. `train_args()` emits
        the preset and, only for Newton arms, the `--solver` latch;
        `apply_to()` is its off-hydra twin and enforces the ordering that
        `apply_physics_preset` requires (it reloads the raw registry config, so
        anything written before it is discarded). Carries the two reference arms
        for the refinement column, marked `trainable=False`. Families are
        CONTACT-LAW families: `sap` and `mujoco` are separate even though both
        run through the Newton manager.
    tools/rubato_sweep/contract_probe.py  NEW. Backend-agnostic preflight probe.
        Dumps the equalized axes — every MDP term with its function, weight and
        mode; action and observation dimensions; sim.dt, decimation, the derived
        control rate, episode length; the scene inventory, joints, bodies and
        object mass — and records per-engine solver detail best-effort WITHOUT
        comparing it. A missing engine block is `null` with its reason, never a
        default. UNEXECUTED: written under this pass's no-GPU rail.
    tools/rubato_sweep/config.py          `Arm.solver` is now optional and
        `Arm.family` exists; an arm that names no engine at all is refused; a
        multi-family sweep must set `preflight.by_family` and must supply a
        non-empty `preflight.contract_keys`.
    tools/rubato_sweep/preflight.py       `compare_by_family()`: strict diff
        WITHIN each contact-law family (the D7 fairness check, unweakened),
        contract-only ACROSS families. A contract key that is absent on one arm
        is a violation, not agreement.
    tools/rubato_sweep/configs/p35_*.yaml four configs (threeway_screen,
        fourway_screen, fourway_full, fourway_crossplay).
    experiments/trossen-fourway/          the campaign driver and .gitignore.
    tools/rubato_sweep/tests/test_fourway.py  15 new CPU-only tests; the suite
        is 76/76 green including pass 33's and pass 34's.

NOT DONE, DELIBERATELY: `artifact_probe.py`, `artifact.py`, `crossplay.py` and
`cli.py` were being written by pass 34 while this pass ran and were not touched.
The 3-line change `artifact_probe.py` needs is specified in the crossplay
config's header rather than applied.

### RESIDUAL RISK — what this pass could not establish

 R1 EVERY GPU PATH ADDED HERE IS UNEXECUTED. `contract_probe.py` has never run.
    Its Newton access paths are copied from the proven parity probe; its PhysX
    block was written from source and has never touched a GPU. First launch is
    a shakedown, and the PhysX arm is expected to fail before the probe reports
    (blocker B1).
 R2 THE PhysX BLOCKERS ARE SOURCE INFERENCES. B1 and B3 are read off the code
    paths, not observed. They are high confidence and they are not measurements.
 R3 THE 44% EMBED IS A STATIC EQUILIBRIUM. It ignores solver damping, the
    contact's own dissipation term, and the drive's damping. It is an order-of-
    magnitude statement about the ratio of two authored stiffnesses, not a
    prediction of a trajectory. The direction is not in doubt; the digit is.
 R4 THE REAL-PAD STIFFNESS RANGE RESTS ON THREE ESTIMATED QUANTITIES (pad
    modulus, patch area, thickness) and on the assumption that the mug wall is
    rigid in series. The bench measurement in M3 replaces all four.
 R5 NO HORIZON HAS EVER BEEN SHOWN TO SEPARATE REWARD ON THIS TASK. 150 cannot
    (measured), 300 cannot (measured). 4000 is untested for seed spread — the
    live run is one seed. If Stage B's three seeds turn out to spread as widely
    at 4000 as they do at 150, the training-curve endpoint is unusable at ANY
    affordable horizon and the matrix is not merely the better instrument but
    the only one.
 R6 THE ACR CONFOUND ON THE ONE CLEAN PAIR IS UNSEPARATED and is not in these
    configs, because separating it is comparison semantics and therefore
    Marco's. Every SAP-fixed-vs-SAP-adaptive number the four-way produces
    inherits it.
 R7 THE MuJoCo ARM'S OBJECT CONTACT LAW IS NOT THE SAP ARMS'. The mug's
    `mjc:solimp` (dmax 0.999, priority 1) is consumed by MuJoCo and ignored by
    both SAP arms and by PhysX. This is an ASSET-level difference invisible to
    the task cfg and therefore invisible to the preflight contract, which
    compares the task, not the USD. Named here because it will not be caught.
 R8 THE EVALUATION COST IS A PROJECTION on four of its six physics columns.

### PROVENANCE (all p35_ prefix; no p13-p34 artifact overwritten)

  design/software  IsaacLabRubato tools/rubato_sweep/{physics_arm,
                   contract_probe}.py, config.py + preflight.py edits,
                   configs/p35_{threeway_screen,fourway_screen,fourway_full,
                   fourway_crossplay}.yaml, tests/test_fourway.py,
                   experiments/trossen-fourway/{fourway.sh,.gitignore}
  re-derived here  p28_train_fixed.log and mjc_1024x25.log (MuJoCo s/iter,
                   window by window); train_main_adaptive.log (the live run's
                   rising iteration time); p31_eval_*.json (the 2x2 returns and
                   their SEMs); p33_p31_ingest/summary.json (the seed spreads)
  verified in      trossen_spatula_lift_env_cfg.py:128-178 (the physx preset),
  source           :214-229 (the sensors), :309-330 (the observations),
                   :413-428 (the whole EventCfg), :484-496 (the timing);
                   newton_manager_cfg.py:67-102 (ke/kd/mu defaults and the
                   solref conversion); contact_sensor_cfg.py:16-26 and
                   contact_sensor.py:289-304 (the Newton pin);
                   physics_presets.py:22-27 and train_rsl_rl.py:52-67
                   (--solver is Newton-only); stationary_ai_task.usda:723-727
                   (the right twin's active=false)
  GPU              zero processes started this pass; nvidia-smi polled
                   read-only only, and it showed one process throughout: the
                   live 4000-iteration adaptive run.

## PASS 36 — PROVENANCE AUDIT: WHAT THE CENIC PAPER SAYS, WHAT WE BUILT,
## AND WHERE THE TWO PART COMPANY
## 2026-08-16. SOURCE + LITERATURE ONLY: ZERO GPU PROCESSES STARTED. The
## 4000-iteration main run (PID 2271848, 11129 MiB) was in flight throughout;
## nvidia-smi was polled read-only and never showed a process this pass started.
## NO CODE WAS CHANGED. The only files written are
## tools/sap_cenic_provenance_audit.md and this entry.
## Marco's request: "we did a lot of work on the solver i need you to determine
## all changes made to the SAP solver, categorized by in line with CENIC paper
## and our own changes."

### DELIVERABLE — tools/sap_cenic_provenance_audit.md

  A methods-section-grade document. Paper specification quoted from the source
  LaTeX; every code claim carried to a file:line; every divergence quantified in
  closed form. Counts: 17 FAITHFUL, 11 DIVERGENT, 9 physics-neutral additions,
  12 physics-VISIBLE additions (6 of them default ON in the live run), 8 named
  paper features not implemented. 17 sap_warp commits and 25 newton-adaptive
  commits audited; ZERO upstream commits touch either SAP path.

### THE PROVENANCE ANSWER, WHICH CHANGES WHAT "OUR CHANGE" MEANS

  sap_warp is a FORK of github.com/sap-sim/sap_warp (AIVC Lab UCLA + TRI,
  Apache-2.0), and the entire upstream repo at the fork point is FOUR commits
  dated 2026-06-11, three of them documentation. The whole SAP solver -- convex
  formulation, regularization, projection, line searches, Cholesky, collision,
  loader -- arrives in one upstream commit 431adf2 (f1shel). Everything after
  c0c861c is ours: 17 commits, 5810+/441-, all mardigiorgio.
  newton-adaptive's SolverSAPAdaptive is 100% ours: of 492 commits since the
  c336b7ae fork, 152 are ours and ~340 are merged upstream Newton, but the set
  of non-mardigiorgio commits touching newton/_src/solvers/sap/ or
  adaptive_boundary.py is EMPTY.
  NOTE for the paper: newton/_src/geometry/broad_phase_sap.py is Sweep-And-Prune,
  not Semi-Analytic Primal. Do not cite it as SAP-solver work.

### THE TWO LOAD-BEARING ITEMS, BOTH RE-DERIVED FROM SCRATCH THIS PASS

  (1) THE 4*PI^2 QUESTION IS SETTLED, AND OUR CODE IS RIGHT.
      CENIC's printed Eq. (18), k = (1/(4 pi^2 beta^2)) m/dt^2, is INTERNALLY
      INCONSISTENT. It contradicts (a) its own stated oscillator period beta*dt,
      which forces k = 4 pi^2 m/(beta^2 dt^2); (b) its own tau = (beta/pi) dt,
      which is the critical-damping value ONLY for that k; and (c) the source it
      cites, arXiv:2110.10107, which states verbatim "k = 4 pi^2 m/(beta^2
      dt^2)" and "R_n = beta^2/(4 pi^2) w" and its Eq. (29)
      R_n = max(beta^2/(4pi^2)||W_ii||_rms, 1/(dt k (dt+tau_d))) -- which is
      EXACTLY our code. The printed equation carries an inverted factor. Our
      rn_hard = beta^2/(4 pi^2) w_i is FAITHFUL, not divergent.
      Verified at all EIGHT contact-R sites (contact_solve.py 948/1180/1278/2760,
      sap_helpers.py 2400/2475/2587/2660). FIVE are upstream 431adf2; THREE are
      ours, added inside fused kernels by 3bff5c1 and a79539a -- and all three
      are TEXTUALLY IDENTICAL to their upstream counterparts, same op order, same
      dtype, so the fusions preserve R bitwise. That is checkable structure, not
      a claim.
      Had we used the printed form: R larger by 16 pi^4/(1+beta/pi) = 1182x at
      beta=1, so the crossover would sit at ~21 N/m (NOT the ~98 N/m an earlier
      pass quoted -- that number is superseded), i.e. 100% of contacts clamped,
      contradicting the measured 11% near-rigid fraction.

  (2) THE dt^-2 MECHANISM CANNOT OCCUR AT OUR AUTHORED TAU, AND THE PAPER NEVER
      CLAIMED IT FOR CONTACT ANYWAY.
      First, scope: "near-rigid" appears FOUR times in the CENIC paper and NEVER
      in the contact section. Sec. IV-A says verbatim "For point contact,
      stiffness is a user supplied parameter." The dt^-2 sentence in the Sec. V
      intro is explicitly scoped "for limit and holonomic constraints". So the
      dt^-2 CONTACT mechanism this campaign has been chasing is an SAP property,
      not a CENIC contact claim. Any text saying "adaptive fixes penetration via
      CENIC's dt^-2 contact-stiffness coupling" is wrong on the paper's own terms.
      Second, magnitude: on the near-rigid branch R_n is dt-INDEPENDENT, so
          k_eff = 1/(h(h+tau) R_n) = 4 pi^2/(beta^2 w h (h+tau))
          d ln k_eff / d ln h = -(1 + h/(h+tau))
      tau is a fixed MATERIAL value here: sap_contact_tau_d = 0.01 s per shape
      (mjwarp_manager_cfg.py:231), doubled per pair by upstream solver_sap.py:956
      (tau(shape0)+tau(shape1)), = 0.02 s effective, ~10x the step. That gives
          -1.005 at h=0.1ms   -1.048 at 1ms   -1.094 at the 2.083ms production
          step   -1.172 at the 4.15ms mean accepted step   -1.286 at 8ms
      against -1.759 (beta=1) / -2 (beta<<pi) for the published law.
      THE MEASURED -1.172 IS REPRODUCED TO FOUR FIGURES BY THIS CLOSED FORM at
      h = 4.15 ms. Measurement and derivation corroborate; the mechanism is fully
      explained by the fixed tau. Resting penetration correspondingly falls
      LINEARLY not quadratically: 5.2 um at 1ms vs the paper's 0.33 um (16x),
      11.4 um at 2.083ms vs 1.4 um (8x).
      ATTRIBUTION: the CODE is upstream SAP Eq. (29), which does keep material
      tau_d. The 0.01 s is OUR config. Not a code divergence; a parameter
      divergence -- and it is the one that decides whether this implementation
      can show the paper's mechanism at all. It cannot.

### WHAT ELSE DIVERGES (all quantified in the document)

  * FORMULATION. CENIC builds on the LAGGED member of ICF (normal impulse
    lagged, Hunt & Crossley dissipation). We run SAP (both components implicit,
    Kelvin-Voigt tau_d, analytic cone projection). A MODELLING difference that
    does not vanish as dt -> 0. Inherited from upstream.
  * ACR. Freezes the contact constitutive law at the attempt dt, s =
    2(D+tau)/(D/2+tau) = 2.05..2.35 over our step range (2.345 at D = dt_outer;
    the flat "2.34" on record is that endpoint, superseded by the formula).
    Two framings, do not conflate: vs the law at D, rt/rn is INVARIANT (the point
    of ACR); vs no-ACR at h, the SOFT branch -- ~89% of contacts -- gets rt/rn
    larger by s, i.e. friction ~2.1-2.35x softer, and normal compliance likewise.
    DEFAULT ON since 45095218, i.e. ON in the live run.
  * ERROR NORM. S = identity (solver_sap_adaptive.py:1590-1596), so eps_acc is
    NOT "digits of accuracy" as the paper intends but a mixed-unit threshold:
    1 mm translation AND 0.057 deg per revolute joint AND 0.115 deg of free-body
    rotation, simultaneously. Plus NEWTON_ADAPTIVE_RTOL=2e-6 is LIVE by default,
    making the test |d| <= atol + rtol|q| (effect <1% at this scene's |q|).
  * GEOMETRY CADENCE. Paper: 2 collision queries per accepted step. Ours: ONE
    per boundary; the contact SET is frozen for the boundary and only
    gap/points/Jacobian are re-anchored. The error estimator is structurally
    blind to contact-set changes.
  * LINE SEARCH. Paper specifies EXACT line search (Newton-Raphson + bisection);
    we default to Armijo backtracking (440e58a flipped it from upstream's
    monotone_decay). We IMPLEMENTED CENIC VI-D's cubic-Hermite seed (bd0c129)
    and then DO NOT USE IT -- it lives only in the non-default exact_root path.
  * VI-B. Commit 5a84f078 implemented CENIC Eq. (34) exactly
    (optimality_rel_tol = max(1e-3*tol, 1e-8)); commit 9c9dc934 REMOVED it for a
    hard 1e-8. Eq. (33)'s Theta criterion was never implemented, and the cost
    early-exit is explicitly zeroed. So VI-B is gone, deliberately.
  * VI-C Hessian reuse: not implemented on the SAP path at all. Largest untapped
    speedup we have.
  * CONVERGENCE-AS-REJECTION. The paper's headline claim #1 is that convexity
    ELIMINATES discarded iterations. We cap inner Newton at 30 and map a cap-out
    to err = 1e9, which forces a reject. That contradicts the claim, for a reason
    the paper does not have: a throughput cap.
  * THREE EXITS ALGORITHM 1 DOES NOT HAVE: floor acceptance (accept regardless of
    e at dt_min); floor latch (state FROZEN, clock FORCE-ADVANCED to the
    boundary); debt guard (carried debt capped at one dt_outer, remainder
    DROPPED). Rates for all three are UNVERIFIED and belong in any accuracy table.
  * CEILING MEMORY (9c9dc934): a one-sided cross-boundary hysteresis on top of
    the paper's deadband; a rejected world needs 2 accepted steps to recover its
    step. It shapes the accepted-dt distribution, which is exactly what a
    work-precision plot reports. Not in the paper.
  * k_Init: paper says 0.1*dt_max; ours effectively 1.0 (seeded ideal_dt clamps
    to dt_outer).

### THE REPORTABLE CONFIGURATION (read from /proc, not from a config file)

  The live run was launched with NO NEWTON_* env overrides, so every env flag is
  at source default. BUT the platform overrides three ctor defaults:
  contact_preset_variant "drake" -> "approx32" (f32 Jacobians + f32 contact
  linear solve, where the paper's reference is fp64 C++), max_substeps 16 -> 256,
  and it pins the fixed arm's inner tolerances to the adaptive arm's on purpose.
  Default ON and physics-visible in that run: fused update-eval, fused armijo
  ladder, folded alpha-max, ACR, armijo_decay, containment. Default OFF:
  determinism, run-ahead, static_substep, fp32 solve stack.

### PROVENANCE (all p36_ prefix; no p13-p35 artifact overwritten)

  paper     scratchpad p36_cenic.html/.txt (arxiv.org/html/2511.08771v1),
            p36_sap.html/.txt (arxiv.org/html/2110.10107v2). Equations taken
            from the LaTeXML alttext, i.e. the authors' own LaTeX.
  document  tools/sap_cenic_provenance_audit.md
  code read sap_warp @ afd5dc6, newton-adaptive @ 80d13a9a, both clean trees
  GPU       zero processes started this pass; nvidia-smi polled read-only only
  NOTE      no probe script was committed this pass (rails: audit only, no code).
            Every number in the document is either quoted, read at a file:line,
            or reproducible in closed form from a formula printed beside it.

### OPEN, AND EXPLICITLY NOT CLOSED BY THIS PASS

  * No C-class bitwise claim was re-measured. They rest on earlier
    flag-equivalence and oracle probe runs.
  * Floor-acceptance rate, floor-latch rate and _debt_guard fire count in the
    reportable run: UNKNOWN. Any accuracy claim needs them.
  * Share of rejections from inner-solve cap-out vs from error: UNKNOWN.
  * Whether sap_warp's optimality test applies the paper's D = diag(M)^-1/2
    scaling to the gradient norm: UNVERIFIED.
  * Whether upstream sap_warp has advanced past c0c861c since the last fetch.
  * TWO ONE-LINE RISKS FOUND, NEITHER FIXED (audit only, needs Marco's call):
    SolverSAPAdaptive.__init__ ends in **kwargs that is accepted and DISCARDED
    (solver_sap_adaptive.py:1275) -- a misspelled sweep parameter silently does
    nothing, so an intended ablation can silently not happen; and
    NEWTON_SAP_ATTEMPT_CONSISTENT_R is read with no default argument, so it is
    ON for any value except the literal "0". Three different parse conventions
    coexist in the NEWTON_SAP_* family.

## PASS 38 — TRUE CONTACT PARAMETERS FOR THE SAP PATH, DERIVED FROM THE
## LBM/DRAKE ASSET PROPERTIES
## 2026-08-16. SOURCE + GEOMETRY + CLOSED FORM ONLY: ZERO GPU PROCESSES
## STARTED. nvidia-smi was polled read-only at the start and the end and never
## showed a process this pass started: at the start the queued comparison
## (PID 2315962, 11509 MiB) held the card; by the end it had finished and
## pass 37's p37_divergence_probe.py (PID 2333768) had taken it. Neither is
## ours -- every script this pass ran is pxr/numpy on the CPU.
## NO TASK, SCENE, ASSET OR SOLVER FILE WAS CHANGED. Files
## written: tools/sap_contact_parameter_derivation.md,
## tools/probes/sap_contact_parameter_derivation.py, and this entry.
## Marco's request: derive the TRUE contact parameters for the SAP path from
## TRI's validated LBM asset properties, so the scene stops running engine
## placeholders.

### THE COMBINATION RULES, READ FROM SOURCE (sap_warp/sim/sap_helpers.py)

  _sap_combine_stiffness(k0,k1) = k0*k1/(k0+k1)        SERIES        (:199)
  _sap_combine_mu(mu0,mu1)      = 2*mu0*mu1/(mu0+mu1)  HARMONIC MEAN (:163)
  _contact_tau_pair(t0,t1)      = t0 + t1              SUM           (:233)

  The campaign note was right in kind, wrong by a factor of 2 in the place that
  matters: stiffness HALVES for two equal shapes, friction does NOT. All three
  rules are identical to Drake's own (discrete_update_manager.cc:940 for the
  series gradient, contact_properties.cc:153 for the tau sum), verified against
  Drake master 29a5d2e6.

### THE HEADLINE RESULT: THE AUTHORED ke IS INERT

  At steady state SAP realizes k_eff = min(k_pair, k_cross) with
  k_cross = 1/(rn_hard*h*(h+tau)), rn_hard = beta^2/(4pi^2)*w_eff. The derived
  pair stiffnesses (4.0e7 base-table, 3.9e6 handle-pad) exceed k_cross by 3-4
  orders of magnitude and are CLAMPED. The authored ke only stops being clamped
  below h ~ 5-20 us; the smallest inner dt this campaign has ever observed is
  2.24 ms. Authoring the true ke therefore changes NOTHING on its own. The
  entire achievable gain in realized contact stiffness is ~10x and it comes
  from shortening tau_d.

### THE DERIVED VALUES

  per shape   /Mug/collisions_base        ke 4.2e7   mu 0.2
              /Mug/collisions_wall_[0-7]  ke 4.6e7   mu 0.2
              /Mug/collisions_handle_[0-2] ke 3.9e6  mu 0.2
              TableGuard, carriages, grippers, links, ground  ke 1e9  mu 0.2
  global      sap_contact_tau_d 0.01 -> 6.6e-4  (pair 20 ms -> 1.33 ms)
  drive       left_gripper stiffness 1000 -> 200 N/m

  Stiffness by Drake's own rule k = A_e * g (discrete_update_manager.cc:1025),
  g = E/H from the convex pressure field (make_convex_field.h), A and H MEASURED
  facet by facet on assets/usd/mug_inomata_white.usd. Calibrated against the two
  LBM assets whose compliant tet meshes exist: H = 3.648 mm (spatula) and
  4.248 mm (plate) by Drake's max-interior-distance definition.

  tau by three independent routes -- H&C damping-term match tau = d*x0 solved
  self-consistently (0.11 ms), critical damping at the realized stiffness
  (2.43 ms), and Castro/CENIC (beta/pi)*h (1.33 ms). Range [0.11, 2.43] ms, all
  clustered at ~1 ms. Marco asked whether the derived value lands near CENIC's
  prescription: IT DOES, from two directions that never reference CENIC.

  Drake sanctions NO H&C -> tau_d conversion (multibody_plant.h:806 -- kSap uses
  relaxation_time and IGNORES hunt_crossley_dissipation). Castro 2023's own
  hand-matched pairs give d/tau_d of 5e5, 1e5, 5e4 -- not a constant. The
  mapping above is OURS, matched at a stated operating point.

### CROSS-CHECKS THAT THE CHAIN IS RIGHT

  * rn_hard/rn_soft at w=14.917, k=1250, tau=0.02, h=1/120 computes to 0.11152
    against the pass-25 dump's MEASURED 0.1115.
  * The mug's own contacts compute to w_eff 94-180 /kg against a near-rigid
    crossover of 133.8 /kg at h=1/120 -- so the measured 11% near-rigid fraction
    IS the mug.
  * Pass-35's measured 44% finger embed reproduces from first principles as
    k_d/(k_d+k_c) = 1000/2250 = 0.444, which independently proves today's pinch
    sits on the SOFT branch.
  * Vendor drive stiffness RE-READ from stationary_ai.usd: 217687 N/m, damping
    10884, maxForce 400. The task's 1000 is a 217.7x softening, confirming the
    inherited "218x".

### THE FOUR CONSEQUENCES MARCO ASKED FOR

  (a) BRANCH. Every derived pair lands far on the near-rigid branch. Today only
      11% do. R_n on that branch is dt-INDEPENDENT, so pushing everything there
      changes what the adaptive-vs-fixed comparison is measuring.
  (b) PENETRATION. Resting mug 2.8 um -- ceramic-sane. Under grasp 0.95 mm at
      the current drive stiffness -- NOT sane, and not fixable by ke.
  (c) DRIVE STIFFNESS. Should go DOWN, to ~200 N/m, not up: 2% embed, ~2 N grip,
      4.4x margin over the 0.444 N needed at mu 0.2. The vendor's 217687 is
      UNUSABLE at this substep -- it would need k_c >= 4.1e6, i.e. h <= 230 us.
  (d) RISKS. Authoring ke through the SHARED material channel BREAKS the MuJoCo
      arm: newton_manager_cfg.py:84 converts (ke,kd) to solref as
      dampratio = (kd/2)sqrt(1/ke), which at ke=4e7 with kd=100 collapses from
      1.0 to 0.0079. ke must be applied SAP-ONLY. Recommended order: mu first
      (real, 5x, safe on both arms), then tau_d (SAP-only), then k_drive, then
      ke last.

### FRICTION IS THE ONE THAT ACTUALLY BITES

  Pair mu 1.0 vs the validated 0.2 is exactly 5x. Pinch force to hold the mug
  goes 0.089 N -> 0.444 N. And the mug tips rather than slides when mu > r/z =
  0.726: at 1.0 it ALWAYS tips, at 0.2 it ALWAYS slides. That is the "sticky"
  signature on the video, in one line of statics.

### PROVENANCE (all p38_ prefix; no p13-p37 artifact overwritten)

  documents  tools/sap_contact_parameter_derivation.md
  probe      tools/probes/sap_contact_parameter_derivation.py (committed,
             re-runnable, closed form, no Warp/USD/GPU import)
  scratchpad p38_geometry.py, p38_geometry2.py (hull facet areas),
             p38_depth.py (LBM tet-mesh foundation depths),
             p38_drive.py (vendor drive attrs + per-facet H)
  code read  sap_warp/sim/{sap_helpers,contact_solve,contact_jacobian,
             solver_sap,loader/scene,resources/collision_model}.py;
             newton-adaptive/newton/_src/{utils/import_usd,usd/schemas,
             geometry/contact_reduction}.py; IsaacLab
             {newton_manager_cfg,mjwarp_manager,mjwarp_manager_cfg}.py and
             the trossen_spatula_lift task + assets
  Drake      master 29a5d2e6, read via web; every claim carries a file:line
  GPU        zero processes started this pass; nvidia-smi polled read-only only

### OPEN, AND EXPLICITLY NOT CLOSED BY THIS PASS

  * N, the contacts emitted per shape pair, is UNMEASURED (needs GPU). It scales
    the authored ke by up to 8x. Immaterial while clamped.
  * The mug's compliant tet mesh named in its SDF (mug_inomata_white_low_
    16faces.vtk) is ABSENT from the LBM bank, so the mug's H is inferred from
    the task's convex hulls, not from TRI's own volume mesh.
  * E = 1e8 Pa is TRI's compliant SURROGATE: ~700x below real ceramic and 10x
    above Drake's own default. "Validated" = TRI shipped it, not that it is the
    material's modulus.
  * Whether NewtonShapeCfg.gap = 0.01 shifts phi0 was checked by code path
    (phi0 uses shape_margin = 0.0, not gap) and NOT by probe. If that reading is
    wrong, every contact activates 1 cm early and the whole of section 3 moves.
  * No validated mu exists for the gripper pads or the table. 0.2 is chosen to
    reproduce the object's value under the harmonic mean, not measured.
  * sap_warp's own hydroelastic path is INERT: entry_k_eff is allocated but
    never written, and collision/pipeline.py:318 says the merge is in progress.
    So this pass pushes a hydroelastic number into a point-contact channel by
    hand -- exactly what Drake documents as a user aid and never applies itself.
  * NOTHING HERE HAS BEEN RUN. The first real evidence is a rest probe.
