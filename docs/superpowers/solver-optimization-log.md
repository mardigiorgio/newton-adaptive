# Solver optimization log (autonomous loop)

Scene: `contact_objects` (9 spheres + 9 boxes/world, randomized seed=7).
Solver: `SolverMuJoCoAdaptive` (compaction + CUDA graphs baseline).
Goal: beat fixed-1ms by as much as possible at high N, while keeping the
accuracy guarantee (per-world per-step step-doubling error <= tol=1e-3, 100%).

## START
- Timestamp: Fri Jun  5 03:22:22 PM CDT 2026 (epoch 1780690942)
- Stop target: ~3h later, epoch >= 1780701742 (~6:22 PM CDT)
- Baseline commit: 40622e5 (clean tree confirmed at start)
- GPU: RTX 4070 Ti Super

## Baseline numbers (from handoff, to be reproduced in iteration 0)
- plain adaptive ~2829 ms/outer; +compaction ~700; +graphs ~245 (N=1024) => ~11.5x over plain.
- vs fixed-1ms: 0.33x@256, 0.95x@512, 1.08x@1024, 1.29x@2048 (>1 = adaptive faster).
- Accuracy honesty check PASSED: worst-world per-step error 100% <= tol(1e-3).

## Levers (highest ceiling first)
1. Kill the 3x step-doubling tax (cheaper error estimator / embedded / Richardson).
2. Reduce K (controller over-conservatism: deadband, safety, rejected iters).
3. Capture the whole iteration body (sort active worlds contiguous, one graph).
4. Profile which mjwarp kernels dominate the tier step; target them.
5. Higher-order integrator for fewer steps at equal accuracy.

## Gates each iteration
- Accuracy: `uv run python -m scripts.bench.benchmarks.tol_trace --n 512 --steps 160`
  -> adaptive worst-world error MUST stay 100% <= tol. Else REVERT.
- Speed: `uv run python -m scripts.bench.episode --ns 512 1024 2048`
  -> adaptive episode-median ms vs fixed-1ms, no NaN. Slower than prev best or NaN => REVERT.
- REVERT = restore changed file(s) from /tmp/solveropt_bak/ (never git checkout/stash).

---

## Iteration log

### Iter 0 -- Reproduce baseline (no code change)
Measurement infra added (not gated, never reverted): `scripts/bench/episode.py`
(episode median/mean per outer step, adaptive vs fixed-1ms, K + error + NaN).
Speed gate `episode --ns 512 1024 2048 --steps 120 --seed 7`:

| N    | adapt_med ms | adapt_mean ms | fixed_med ms | ratio_med | K_med | K_max | max_err  |
|------|--------------|---------------|--------------|-----------|-------|-------|----------|
| 512  | 161.92       | 173.64        | 167.20       | 1.033     | 36    | 56    | 9.15e-4  |
| 1024 | 228.48       | 246.33        | 281.59       | 1.232     | 38    | 65    | 7.65e-4  |
| 2048 | 327.65       | 382.53        | 550.56       | 1.680     | 40    | 65    | 8.99e-4  |

Accuracy gate `tol_trace --n 512 --steps 160`: adaptive worst-world err
[1.60e-07, 8.92e-04], **100% <= tol**. PASS.

**REFERENCE (best adapt_med to beat): 512=161.92, 1024=228.48, 2048=327.65 ms.**
Note: my median-based ratios (1.03/1.23/1.68) run hotter than the handoff's
quoted (0.95/1.08/1.29); methodology differs (episode median vs scaling-bench
steady median). Internal consistency is what the gate needs. Status: KEPT (baseline).
Observation: contact_objects does NOT settle -- K_med stays ~36-40 across the
whole 120-step episode (dense contact throughout). So K reduction is the lever
with the most steady-state leverage.

Diagnostic (N=256, /tmp/diag_reject.py, post-warmup, 30 outer steps):
K_med=39, **rejection fraction = 23%**, avg active worlds/iter = 116/256.
~1/4 of all 3-eval step attempts are wasted on rejected worlds. The controller
parks dt at the tol boundary (safety 0.9 + deadband lower edge 0.9 => an
accepted step at e~tol computes new_step~0.9*step, which the deadband HOLDS),
so noise pushes ~a quarter of steps over tol. Reducing rejections (without
shrinking dt) is the target.

### Iter 1 -- Lever 2: safety_factor 0.9 -> 0.8 (REVERTED)
File: scripts/adaptive/controller.py. Hypothesis: more conservative target dt
builds margin below tol, cutting rejections.
- Accuracy gate: 100% <= tol (max 9.69e-4). PASS.
- Speed gate (adaptive med ms, vs ref 162/228/328):

| N    | adapt_med | vs ref   | K_med | NaN  |
|------|-----------|----------|-------|------|
| 512  | 210.90    | +30% (!) | 36    | True |
| 1024 | 230.02    | +0.7%    | 39    | -    |
| 2048 | 342.35    | +4.5%    | 41    | -    |

REVERTED: slower at every N, K rose (smaller dt = more steps, swamping the
rejection savings), and a world diverged (NaN at 512). Lesson: blanket dt
shrink is the wrong direction; need to cut rejections while KEEPING dt size.
Points to a smarter controller (PI damping or asymmetric/proximity-aware
deadband), not a global safety change.

### Iter 2 -- Lever 2: growth_cap 5.0 -> 2.0 (REVERTED)
File: scripts/adaptive/controller.py. Hypothesis: 23% rejections are
growth-overshoot (dt quintuples when contact lightens, overshoots tol when it
returns); damping growth prevents that while keeping steady dt.
- Accuracy gate: 100% <= tol (max 8.91e-4). PASS.
- Speed (adaptive med ms vs ref 162/228/328): 512=161.4 (=), 1024=335.99 (+47%,
  NaN), 2048=344.24 (+5%, K_med 42 up). REVERTED.
- Lesson: capping growth HURTS -- worlds separating from interpenetrating spawn
  (free-flight transition) and worlds recovering from forced shrinks need fast
  dt growth; capping it adds many small steps. So rejections are NOT mainly
  growth-overshoot either.

### METHODOLOGY FIX (after iters 1-2 thrashed on chaos noise)
Ran the UNCHANGED baseline 3x: seed7/1024 = 228, 216 (no NaN); seed11/1024 =
292 **with NaN=True**. So divergence/NaN is seed- and FP-chaos-driven (the
randomized ICs spawn objects interpenetrating -> violent separation -> some
worlds blow up), NOT caused by my edits. Wall-time median swings +-30% from a
single stuck/diverged world.
BUT **K_median is stable at 37** across ALL of these runs. New gate:
- PRIMARY metric = K_median at seed 7 (low-noise; = physics work = the target).
  A real win must DROP K_median (baseline ~37-40).
- Wall-time median is confirmatory only; tolerate +-5%.
- seed 7 is NaN-free at baseline, so a seed-7 NaN = real regression.
- Fast pre-filter for rejection-cutting ideas: /tmp/diag_reject.py rejection %.
ABANDONING Lever 2 (controller): 2 reverts + analysis shows both dt-shrink and
growth-damping raise K; rejections look intrinsic to the stiff fluctuating
contact at the operating dt. Moving to profiling (Lever 4) to find the dominant
tier-step cost, then a structural change.

### PROFILE (Lever 4) -- where does the tier step time go?
/tmp/profile_phases.py (N=512, dense contact, full step ~7.0 ms):
collision 15%, fwd_position 19%, constraint solve dominant (~80% incl. accel),
integrate 1%. Collision-sharing ceiling is tiny.
**solver_niter MEAN = 1.0** (max 5): the qacc warmstart (already carried across
the 3 substeps via mjw_data persistence -- verified _update_mjc_data only writes
qpos/qvel) is so effective the Newton solver converges in ONE iteration. So
loosening opt.tolerance (1e-6) cannot help -- already at the iteration floor.
solver=NEWTON(2), integrator=IMPLICITFAST(3), cone=PYRAMIDAL, tol 1e-6.

### Iter 3 -- Lever 3/compaction: tier ratio 1.5 -> 1.3 (REVERTED)
File: solver_mujoco_adaptive.py _init_compaction. Hypothesis: finer tiers cut
padding waste (avg 116/256 active picks the 170-tier, ~32% padding).
- Accuracy: 100% <= tol (max 9.85e-4). PASS.
- Speed seed7: 1024=217 (=noise), **2048=373 (+14%, NaN)**. REVERTED.
- Cause: ~14 tiers vs ~9 at N=2048 ~doubles tier GPU memory -> occupancy/memory
  pressure; padding savings too small to offset.

### MICROBENCH (deterministic, removes chaos) -- scripts/bench/tier_microbench.py
Times the captured tier graph replay vs tier size on a FIXED dense-contact
state. N=1024: 89w=0.86ms ... 1024w=2.18ms. N=2048: 79w=0.82ms ... 2048w=3.52ms.
**Linear fit: tier step ~= 0.73-0.78 ms FIXED + ~1.4 us/active-world.**
The fixed ~0.75 ms (world-count-independent) is ~150 MuJoCo kernels x GPU
grid-launch latency that CUDA graphs do NOT remove (it is execution latency, not
CPU launch). This fixed cost is paid **3*K times per outer step** -> at N=1024,
0.75ms * 3 * 38 ~= 85 ms of irreducible floor (out of 228 ms, ~37%);
at N=2048 ~94 ms (of 328). 

### KEY COST MODEL
  outer_step_time ~= 3 * K * (0.75 ms fixed + 1.4 us * n_active)
- "3" = step-doubling evals: sacred (a 2-eval estimator would redefine the
  guarded error metric -> forbidden).
- "K" ~= 37-40, set by the stiffest world; near-Pareto-optimal (both dt-shrink
  and growth-damp raise it; 23% rejections look intrinsic to fluctuating
  contact stiffness at the operating dt).
- "0.75 ms fixed" = MuJoCo-Warp per-step kernel-latency floor (upstream-bound).
- compaction already minimizes n_active and the floor dominates small tiers, so
  granularity is near-optimal (finer tiers regressed).
The baseline is close to the floor this architecture allows. Remaining probes:
integrator order (Lever 5) and a predictive PI controller (Lever 2, last try),
both gated on the low-noise K_median + the fast rejection diagnostic.

### Iter 4 -- Lever 5: integrator IMPLICITFAST(3) -> IMPLICIT(2) (REVERTED)
File: solver_mujoco_adaptive.py. mujoco_warp raises
`NotImplementedError: integrator 2 not implemented`. Only EULER/RK4/IMPLICITFAST
exist; IMPLICITFAST is already the best for stiff contact (EULER is lower order,
RK4 is explicit -> unstable). Higher-order integrator (Lever 5) is unavailable.
REVERTED. Dead end.

### Iter 5 -- Lever 2 finale: predictive PI controller (REVERTED)
Files: controller.py (+beta), kernels.py (_calc_adjusted_step gains a
dopri5 predictive factor (err_prev/tol)**beta and integral exponent
0.5-0.75*beta), base.py (+_err_prev array, updated on accept only).
Direction validated on fast diagnostic (N=256):
  beta=0.04 -> rejection 23.0%->21.1%, K_med 39->38 (marginal);
  beta=0.15 -> rejection ->14.5% but K_med ->46 (over-damped: slower dt growth
  trades rejections for MORE small steps).
Full gate at beta=0.04 (seed 7):
- Accuracy: 100% <= tol (max 9.70e-4). PASS.
- **K_median UNCHANGED: 1024=38, 2048=40 (identical to baseline).** Wall-time
  within noise (1024=212, 2048=347 w/ chaos NaN). The diagnostic's 39->38 was
  noise. REVERTED.
- Conclusion: a proper predictive PI does not reduce K for this steady-state-
  dominated scene -- the predictive term ~=1 when error fluctuates around tol
  (the common case here), so it only helps transitions, which are rare. The
  controller is near-Pareto-optimal: cutting rejections costs step count.

### Iter 6 -- Newton linesearch cap ls_iterations 50 -> 10  (CLAIMED win, then
###          REFUTED in iter 7 as a measurement artifact -> REVERTED)
File: solver_mujoco_adaptive.py __init__ (set before graph capture). Attacks the
0.75 ms fixed per-substep cost (paid 3*K times). solver_niter~=1 (warm-started),
and mjwarp captures the Newton linesearch as a FIXED loop of ls_iterations, so
50->10 drops 40 wasted linesearch steps per substep.
Evidence (deterministic micro-bench is primary; episode wall-time is chaos-noisy
even run-to-run for the same seed):
- **Micro-bench (deterministic, NO chaos)** tier step:
  N=1024 89w 0.86->0.64 ms (-25%), large tiers neutral; fixed ~0.73->~0.50 ms.
  N=2048 79w 0.82->0.69 (-16%), 179w 0.91->0.78 (-14%), large tiers neutral.
  Never slower beyond noise.
- **Accuracy** tol_trace N=512: worst-world err [2.1e-7, 9.68e-4], 100% <= tol.
- **Matched multi-seed N=2048 steps=100** (ls=10 vs baseline, SAME seeds):
  s7 455 vs 557 (both NaN), s13 568 vs 576 (both NaN), s23 383 vs 387 (clean).
  Identical divergence -> ls=10 adds NO divergence; faster/equal every seed.
- **Regression** use_mujoco_contacts=False (non-native, compaction off): 20
  outer steps clean, no NaN, err 5.5e-4.
- Episode seed7 steps120: 512=149-158 (no NaN, consistently < baseline 162;
  ratio vs fixed 1.12 vs baseline ~1.03). 1024/2048 swing with random NaNs the
  SAME way baseline does -- not used for the decision.
Initially KEPT on this evidence. **But see iter 7: REFUTED and REVERTED.**

### Iter 7 -- push ls lower, and the CORRECTION that killed iter 6
Tested ls=5: accuracy 100% <= tol, but micro-bench vs ls=10 was MIXED and the
same-session ls=50 micro-bench (0.67..1.93 ms) came back ~20% off the EARLIER
ls=50 run (0.86..2.18). That 20% same-code swing exposed the flaw: the tier
micro-bench's 15-step warmup reaches a CHAOS-DEPENDENT contact state (denser ->
more Cholesky compute), so it is NOT deterministic. The apparent ls=10 "win" was
that confound, not ls.
**Clean isolation** (/tmp/deterministic_ls.py): time mjw.step on ONE FIXED dense
state, reloading the identical qpos each rep, varying ONLY ls_iterations:
  ls=50 -> 8.609 ms,  ls=10 -> 8.582 ms (+0.3%),  ls=5 -> 8.592 ms (+0.2%).
Repeatability +-0.1%. **ls_iterations has ~no effect** -- the Newton linesearch
EARLY-EXITS (data-dependent), it is NOT a fixed captured loop. iter 6 was a false
positive. **REVERTED ls=10 to the original ls=50 baseline.** The matched
multi-seed "win" was likewise chaos (ls=10 runs happened to diverge less).
Lesson: every speedup claim needs (a) a fixed-state deterministic measurement and
(b) a physics-correctness check -- the self-referential step-doubling error and
chaos-noisy episodes are not enough.

### Iter 8 -- disableflags for unused features (probe only, not applied)
Fixed-state probe (/tmp/probe_disableflags.py): disabling
ACTUATION|LIMIT|EQUALITY|FRICTIONLOSS|SENSOR (all unused by contact_objects'
free rigid bodies) -> 8.305 -> 8.240 ms, **+0.8%** (real, repeatable +-0.1%, but
negligible -- MuJoCo already skips zero-count features). Not worth a code change.

### Iter 9 -- collision-sharing prototype (probe only; ARTIFACT, not applied)
full(h) and half1(h/2) start from the IDENTICAL state_cur, so MuJoCo collides
twice on the same geometry. Prototype (/tmp/probe_collision_share.py): collide
once at state_cur (shared by full+half1) via run_collision_detection toggling ->
showed +20% on one iteration. **But the correctness check is fatal:** the shared
path gives max|dqpos|=7.2e-3 (> tol) and max|dqvel|=4.95 -- DIFFERENT physics.
run_collision_detection=False skips the per-dt constraint rebuild too, so the
"+20%" is wrong-and-cheaper work, not a real speedup. A correct geometry-only
share (rebuild make_constraint per dt) is not reachable via the mjw API and has a
~5% real ceiling (one 1.05 ms collision). Not pursued. Another would-be artifact
caught by the correctness gate.

---

## FINAL SUMMARY  (stop: ~80 min in; levers exhausted with a reliable method)

### Result
**No accuracy-safe speedup over the committed baseline was found.** All 5 code
edits were REVERTED; the tree ends at the baseline commit (40622e5) plus three
NEW measurement-infra files (kept, never gated):
  - scripts/bench/fixed_state_step.py -- **the reliable speed tool**: time one
    mjw.step on ONE FIXED state (+-0.1%), with a built-in physics-correctness
    check. Use THIS to evaluate per-step changes, not the two below.
  - scripts/bench/episode.py        -- episode median/mean, adaptive vs fixed-1ms
    (chaos-noisy +-30%; use only for big effects / K_median, not <10% deltas)
  - scripts/bench/tier_microbench.py -- tier graph-replay vs size (CONFOUNDED by
    chaotic warmup state; kept for reference, prefer fixed_state_step.py)
  - docs/superpowers/solver-optimization-log.md -- this log
The baseline `SolverMuJoCoAdaptive` (compaction + CUDA graphs) is **near-optimal**
for contact_objects on this GPU/mjwarp build. Accuracy stayed 100% <= tol at
every step and divergence behavior was never made worse.

### Reproduced baseline (episode median, seed 7, steps 120)
  N=512  162 ms (ratio vs fixed-1ms 1.03),  N=1024 228 ms (1.23),
  N=2048 328 ms (1.68).  Accuracy: worst-world step-doubling err 100% <= tol(1e-3).

### Cost model (the wall)
  outer_step ~= 3 * K * (~0.5-0.75 ms fixed + ~1.4 us * n_active)   [graph replay]
  fixed-state eager full step ~= 8.3 ms, dominated by the Newton **Cholesky**
  factorization over the contact set (`update_gradient_cholesky_blocked`);
  collision ~15%, integrate ~1%; solver_niter ~= 1 (qacc warmstart carried across
  the 3 substeps, so the iterative solver is already at its floor).
- "3" (step-doubling evals) is SACRED: a 2-eval estimator would redefine the
  guarded error metric.
- "K" (~37-40) is set by the stiffest world and the controller is near-Pareto-
  optimal here (safety-down, growth-cap, and a proper predictive PI all raise K
  or leave it unchanged; ~23% rejections are intrinsic to fluctuating contact).
- the per-step floor is MuJoCo-Warp kernel-bound (Cholesky + ~150 kernels'
  grid-launch latency that graphs do not remove).

### Levers tested and why each fails (all reverted/not-applied)
  1. controller safety 0.9->0.8 (iter1): slower, K up, NaN. dt-shrink adds steps.
  2. controller growth_cap 5->2 (iter2): slower, K up. blocks needed fast growth.
  3. compaction tier ratio 1.5->1.3 (iter3): 2048 +14% (tier memory ~2x -> occupancy).
  4. integrator IMPLICIT (iter4): NotImplementedError in mjwarp; only EULER/RK4/
     IMPLICITFAST exist and IMPLICITFAST is already best for stiff contact.
  5. predictive PI controller (iter5): K_median unchanged; helps only transitions,
     scene is steady-state-dominated.
  6/7. ls_iterations 50->10/5: ~no effect (linesearch early-exits); the apparent
     win was a chaotic-warmup artifact (fixed-state test: +0.3%).
  8. disableflags (unused features): +0.8%, negligible.
  9. collision-sharing: physics artifact (wrong-and-cheaper); ~5% real ceiling,
     not reachable via mjw API.

### Methodology (the most reusable output)
- Episode wall-time is UNRELIABLE for <~10% effects: chaotic per-world divergence
  (interpenetrating randomized spawn) gives +-30% swings and intermittent NaN,
  non-deterministic even run-to-run for the SAME seed. Baseline NaNs as often as
  any variant (matched-seed proof).
- The tier micro-bench is ALSO confounded (its warmup reaches a chaos-dependent
  contact state) -- a same-code 20% swing exposed this.
- Reliable speed signal = time mjw.step on ONE FIXED state, reload identical qpos
  each rep, eager, interleaved (+-0.1% repeatable). Plus K_median (low-noise) for
  K-changing experiments.
- EVERY speedup claim needs a physics-correctness check (compare qpos/qvel to
  baseline on a fixed state). The self-referential step-doubling error passing
  100% <= tol does NOT prove correctness -- two artifacts (ls=10, collision-share)
  passed it while being wrong/no-ops.

### What to try next (all higher-effort / upstream)
1. A provably-CONSERVATIVE 2-eval error estimator (must bound the step-doubling
   estimate from above) to drop 3->2 MuJoCo steps/iter (~33% ceiling). Hardest,
   biggest. Requires theory, not tuning.
2. Correct collision-geometry sharing across full+half1: split mjw collision from
   make_constraint so contacts are computed once but constraints rebuilt per dt.
   ~5% ceiling; needs mjwarp-internals work + must pass the fixed-state physics
   check. Add a verified `episode --solvers adaptive,fixed` regression first.
3. Upstream: fuse MuJoCo-Warp's per-step kernels to cut the ~0.5-0.75 ms grid-
   launch floor (helps the small-tier K-tail most).
4. Reduce divergence (a sleep/clamp for blown-up worlds) so episode wall-time
   becomes a usable optimization signal -- would unblock measuring small wins.
   (Note: project memory already flags a needed v=0 rest/sleep mechanism.)

</content>
