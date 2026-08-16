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
10 s/iter @1024 = 1.9x below the last MEASURED plateau 18.98 (pass-16
re-measure on the fused-LS stack; was 3.5x vs the pass-14 35.35, 4.0x
vs the pass-9 40.78). Pass-17 LANDED two of the three identified
levers (alpha-max fold + per-contact pack, entry below): per-substep
-26.3% vs that stack at 1024x8 -> projected ~13.9 s/iter (25-iter
plateau re-measure pending); 10 s still needs the update-eval fusion
(pass-16 recommendation C) plus one further factor-scale find.

## Objective

Make a 4000-iteration training feasible. Primary metric: projected 4k-iter
wall from measured plateau curves at 1024 and 4096 envs. Reference points:
MuJoCo-adaptive plateau ~5.1 s/iter @1024 (4k ≈ 5.7 h); SAP-adaptive
last MEASURED plateau 18.98 s/iter @1024 det-unset (pass-16 re-measure
on the fused-LS stack, 4k ≈ 21.1 h; the pass-17 landing prices
-26.3%/substep below that stack at 1024x8 — projected ~13.9 s/iter,
plateau re-measure pending); prior points kept for scale: 35.35
(pass 14, ACR-default pre-fused-LS, 2026-08-16), 40.78 (pass 9,
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
"SAP NOW" refreshed by the pass-16 re-measure on the fused-LS stack):
- MuJoCo-adaptive @1024: 5.1 s/iter -> 5.7 h (ledger provenance).
- SAP pre-campaign @1024 det-ON: 78 -> 86.7 h (historical).
- SAP NOW @1024 det-unset: 18.98 measured plateau (pass 16, fused-LS
  stack) -> 21.1 h; 2000-iter ~ 10.5 h.
- SAP pass-14 @1024 det-unset (2026-08-16, ACR-default pre-fused-LS,
  kept dated for scale): 35.35 -> 39.3 h; 2000-iter ~ 19.6 h.
- SAP pass-9 @1024 det-unset (2026-08-15 late, pre-ACR-default, kept
  dated for scale): 40.78 -> 45.3 h; 2000-iter ~ 22.6 h.
- SAP NOW @1024 det=1: ~20.7 est (18.98 x 1.088 pass-9 det tax,
  unremeasured since pass 9) -> ~22.9 h.
- SAP @4096 det=1 (projection on projection, LOW confidence: the
  pass-9 ~174 s/iter projected plateau scaled by the measured 1024
  price ratio p16/p9 = 0.465): ~81 s/iter -> ~90 h; det-unset blocked
  by the tri-pair OOM, then est ~74 -> ~82 h. 4096-at-4k remains
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
   Pass-17 DONE: recommendations A+B LANDED default-ON (entry above:
   -26.3%/-26.6% per-substep at 1024x8, projected ~13.9 s/iter);
   next in line: 25-iter plateau re-measure, then recommendation C
   (update-eval fusion, ~8% ceiling in the pass-16 entry);
   (c) per-boundary D2H readback chain
   (~48/boundary, overlapped today but serializing the march's
   conditional structure?); (d) cross-boundary overlap of independent
   worlds' marches (capture mechanics proven feasible pass 4).
   Each candidate gets a MEASUREMENT first, no code.
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
