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
- TRIANGLE-PAIR CAP RIGHT-SIZE (new, unblocks 4096 det-unset + frees
  ~11 GB @1024): the task cfg authors max_triangle_pairs=192M, sized
  blind in the always-det era when the CONTACT_ID_BITS clamp silently
  capped it at 33.5M; live demand ~2.8M. Authoring ~8-16M (3-6x margin)
  restores the historical footprint under det-off and un-OOMs 4096.
  One task-config line; task changes are Marco's.
- Phantom body fix: follower_left_ee_gripper_link active=false overlay
  (one line in stationary_ai_task.usda, mirrors right twin) — kernel-width
  savings 7/22 coords per world + closes latent-risk surface
- Any push to GitHub (auth broken; needs gh auth login first)
