# CENIC sim-to-real GATED study — ledger

> **!!! CONCLUSION OVERTURNED -- THE PI WAS RIGHT (2026-06-16) !!!** The earlier "integration
> doesn't matter / KILL" conclusion below was WRONG: it rested on a broken metric AND a real bug in
> the integration. Two fixes, both validated:
> 1. **Eval metric was broken (closed-loop masking).** Phase 0b's only metric was a stiff Kp=150
>    PD+policy velocity-tracking error -- the controller regulates away the integration deviation
>    before the metric reads it, so the integrator falsely ranked last (only obs_noise, which
>    corrupts the loop INPUT, escaped). It measured the controller, not the physics.
> 2. **Independently: the impl gives the expected work-precision advantage.** On a clean OPEN-LOOP
>    unit test (single stiff sphere impact, no policy, no chaos, euler, `scripts/rl/v1_single_drop.py`)
>    CENIC reaches 1.3mm penetration accuracy in ~228 MuJoCo steps where fixed-step needs ~800 -- a
>    **~3.5x compute win, Pareto-dominant**; dt collapses at the impact. Figure
>    `results/plots/v1_single_drop.png`. So the adaptive advantage is real and the implementation is
>    sound; the Phase 0b null was the metric, full stop.
>
> CORRECTION (do not overclaim): I earlier blamed a third thing -- the code's error-scaling S had
> drifted from §V-E (added `diag(M)^{-1/2}` + clip[1,10] + normalize, none in the paper) -- and
> claimed it "stopped CENIC refining at impacts." A DIRECT ABLATION (`v1_norm_ablation.py`,
> articulated foot-strike, deviated S vs identity) REFUTES that: deviated S min_dt=0.49ms/480 steps
> vs identity 2.97ms/324 steps -- the deviated S refines MORE, same penetration. So the S deviation
> is NOT the cause of the null and did not break refinement. It is a genuine §V-E spec-deviation
> worth aligning (we set S=identity per [[project_s_removed_identity]]), and an honest question for
> the author, but it is housekeeping, not the bug. A velocity-term I also tried is excluded by §V-E
> and was removed. NET: the only load-bearing finding is closed-loop masking + the real work-precision
> win. Everything below the line is SUPERSEDED; do not cite it.

> **STUDY COMPLETE -- final answer to "will adaptive stepping close humanoid sim-to-real?"**
> NO, not on its own, and we now know exactly why and where it does help. Two findings:
> (1) **Transfer: dead.** In a controlled error-budget test on a trained walker, the integrator
> was the LEAST important of eight factors (moved tracking error <1%, even slightly improved it);
> sensor noise mattered ~7x more, motor gain ~3x more. Those dominant terms only grow with DOF,
> so humanoids are worse, not better, for this bet.
> (2) **Data fidelity: narrowly scoped.** Adaptive stepping shows NO accuracy-per-compute or
> penetration advantage in the soft, compliant contacts that dominate legged locomotion (verified
> on an ANYmal foot-strike AND an 18-object pile; MuJoCo soft contact is dt-forgiving and doesn't
> tunnel). Its real win is in NON-CONVEX SDF geometric tunneling (the poster's fork-through-a-rack),
> where coarse dt skips through thin geometry. Also learned: trajectory-error-vs-a-gold is invalid
> under multi-contact chaos.
> **Recommendation for the meeting:** don't pursue adaptive stepping to close humanoid sim-to-real;
> position it as a data-generation tool for stiff / non-convex / tunneling-contact scenes
> (manipulation, assembly), and invest the sim-to-real effort in actuator + sensing modeling, which
> the data says dominate. Single number to lead with: the integrator ranked 8th of 8 transfer
> factors, ~7x below sensor noise.

Status legend: [ ] todo · [~] in progress · [x] done/verified · [KILL]/[GO] gate verdict.

---

## Phase 0a — paired-eval RNG + single-term presets  [x] VERIFIED
**Gate:** prerequisite — make the transfer gap isolate the backend.

Code: `DRConfig.single(term)`; dedicated `ic_gen`/`cmd_gen` torch generators seeded
identically across backends and decoupled from the DR stream; `EnvConfig.eval_seed`;
`eval_transfer --seed` + frozen-command horizon + single-term ref tokens
(`ref_friction/ref_mass/ref_kp/ref_kd/ref_push/ref_obsnoise`); `analysis.py --rank`;
`check_phase0a.py`. Training path (eval_seed=None) unchanged.

Validation (real output, `uv run --extra rl -m scripts.rl.check_phase0a`):
```
[PASS] ic_qpos   cenic==fixed: max|d|=0.000e+00
[PASS] command   cenic==fixed: max|d|=0.000e+00
[PASS] ic_qpos   id==ref_friction: max|d|=0.000e+00   (pairing survives DR draws)
[PASS] command   id==ref_friction: max|d|=0.000e+00
[PASS] friction  id!=ref_friction: max|d|=1.740e-01   (friction actually perturbed)
[PASS] mass      id==ref_friction: max|d|=0.000e+00   (only friction changed)
```
=> the gap now isolates the backend. Bit-exact pairing confirmed.

## Phase 0b — error-budget ranking (frozen policy)  [KILL] V2 transfer claim dead
**Gate:** is the integrator-only gap (id vs ref_tol AND ref_dt) statistically nonzero AND
within ~5x of a dominant physical term? **ANSWER: NO.**
Policy `runs/cenic_dr-off_s1/model_1000.pt` (4096 envs, 1000 iters, ep-len maxed, lvte 0.15).
Eval: 9 backends x 4 paired seeds x 64 worlds, horizon 500. Error-budget ranking
(`results/plots/p0b_error_budget.png`):

LVTE gap vs id (m/s), ranked by degradation (4 seeds pooled):
```
ref_tol     integrator  -0.0008  [-0.0011, -0.0005]   <- integrator: NEGATIVE (no penalty)
ref_dt      integrator  -0.0006  [-0.0011, -0.0001]   <- integrator: NEGATIVE
ref_friction physical   -0.0002  [-0.0003, -0.0000]
ref_push    physical    -0.0000  [-0.0002, +0.0002]
ref_mass    physical    +0.0003  [+0.0000, +0.0005]
ref_kd      physical    +0.0005  [-0.0001, +0.0012]
ref_kp      physical    +0.0015  [+0.0006, +0.0024]   <- largest physical
ref_obsnoise sensing    +0.0062  [+0.0055, +0.0069]   <- WORST channel (~7.7x |integrator|)
```
AVTE agrees (sharper): integrator -0.0044/-0.0023 (negative), obs_noise +0.0102 worst.
**Verdict:** the integrator is the LEAST important transfer channel; making it more accurate
yields NO gap reduction (gap <= 0). Dominant channels are sensing (obs noise) and motor gain
(kp) -- exactly the literature's dominant terms. KILL the "adaptive stepping closes sim-to-real"
claim; do NOT port to humanoid for transfer (its dominant terms are even larger).
SCOPE CAVEAT: flat-ground walking = mild contact; the integrator's value should be largest in
stiff contact, which is the V1 data-fidelity question, pursued next.

## Phase 1/2/3 (V2 transfer) — NOT PURSUED (gated out by Phase 0b KILL)
DR-on interaction, matched-compute, ref_actuator construct-validity, and the H1/G1 port are
shelved: with no integrator-induced transfer gap on the quadruped, there is nothing for them to
rescue. Re-open only if V1 shows the integrator matters in stiff contact AND a transfer signal
reappears there.

## V1 — data-fidelity track (NOW THE ACTIVE THESIS)  [~] IN PROGRESS
**Gate:** does CENIC produce measurably more accurate, lower-artifact trajectory data per unit
compute than fixed-step, against a high-fidelity (ref_dt/ref_tol) gold rollout -- especially in
contact-rich / foot-strike regimes where Phase 0b's mild-contact caveat says it should?

V1-A (DONE, verified): cumulative substep counter wired into SolverMuJoCoCENIC. Non-resetting
`_cum_iters` buffer incremented each boundary-loop iteration (includes rejected attempts);
public `cumulative_substeps()` (= iterations*3 MuJoCo opt-steps), `cumulative_iterations`,
`reset_compute_counter()`. Verified: 10 step_dt -> 30 iters -> 90 substeps (=30*3), reset works.
Fixed backend compute is analytic (control_steps * control_dt/fixed_dt).

V1-B (DONE, INCONCLUSIVE on ANYmal soft contact): work-precision on the ANYmal drop/foot-strike
(`scripts/rl/v1_work_precision.py`). Result does NOT show a clean adaptive advantage:
- trajectory error vs a fixed gold: fixed dt=1ms (600 sub, 8e-4) beats every CENIC point; CENIC
  error non-monotonic in tol (2.1e-3 -> 4.0e-3 -> 1.8e-3).
- peak penetration: fixed 8ms -12.2mm, fixed 2ms -7.5mm, fixed 0.5ms -6.7mm (monotone);
  CENIC 1e-2 -9.9mm, CENIC 1e-3 -12.7mm (NON-monotone, worse than fixed 2ms).
Mechanistic read: CENIC's error controller is a position-only inf-norm on joint_q, NOT contact
penetration; ANYmal foot-ground contact is soft (MuJoCo solref) and gentle (~mm position error
at the strike), so the controller sits near tolerance and does not refine dt at impact. Plus the
outer-boundary sampling aliases the sub-step penetration peak. So this scenario is the WRONG
regime to show V1, not evidence CENIC is bad. (Do not over-claim a null; it is inconclusive.)
CONSISTENT with Phase 0b: locomotion is soft-contact-dominated -> integration barely matters.

V1-C (DONE, `scripts/rl/v1_stiff_contact.py`): contact_objects scene (18 objects dropped). STILL
no clean win, with two diagnosed reasons:
- trajectory error is CHAOS-CONFOUNDED: CENIC err RISES with compute (0.52->0.74) while fixed err
  FALLS (0.54->0.10), both vs a fixed gold -> in a chaotic pile, fixed tracks the fixed-gold's
  chaotic path, CENIC follows a different one. Cross-method trajectory matching is invalid under
  chaos (the chaos-skeptic's warning, now observed).
- penetration is SOFT-CONTACT COMPLIANCE, not tunneling: the gold itself penetrates -22.8 mm and
  all backends sit at ~20-30 mm (MuJoCo ke=1e4 resting overlap), dt-independent. No artifact to fix.
ROOT CAUSE: MuJoCo soft contacts (ANYmal feet AND contact_objects) are dt-forgiving and do not
tunnel, so CENIC's refinement has nothing to rescue. The DEMONSTRATED win ([[project_fork_figure]])
is NON-CONVEX SDF geometric tunneling (fork through a rack) -- Newton's SDF collision, a harder
contact path where coarse dt skips through thin geometry.

V1-D (NOT PURSUED -- study concluded): the clean tunneling demo would need non-convex SDF
geometric tunneling, but the current `scripts/scenes/dish_rack.py` uses a SOLID-BOX rack
approximation (its own docstring: wire-level contact is "a follow-up"), so the fork-through-rack
tunneling is NOT reproducible in the current code. That win already lives in the project poster
([[project_fork_figure]], verified vs dt->0). Reproducing it cleanly is an asset-heavy follow-up,
not a blocker for the conclusion. STUDY CONCLUDED here.

---

## FINAL SYNTHESIS
Question: will adaptive step integration close sim-to-real for large-DOF humanoids? **Answer: no,
not on its own.** Evidence, all on the real harness:
- Phase 0a: fixed a measurement confound (paired eval RNG; verified bit-exact). Without it every
  transfer number was contaminated.
- Phase 0b (the decisive gate): error-budget ranking of a trained walker -- integrator is the
  LEAST important of 8 transfer factors (gap negative, ~7x below sensor noise, ~3x below motor
  gain). KILL the transfer claim. Dominant terms scale with DOF -> humanoids worse, not better.
- V1 (data fidelity): no adaptive advantage in soft compliant contact (ANYmal foot-strike;
  18-object pile) -- MuJoCo soft contact is dt-forgiving; trajectory error is chaos-confounded;
  "penetration" is resting compliance, not tunneling. The genuine win is non-convex SDF tunneling
  (existing poster), a data-generation niche, not a sim-to-real lever.
RECOMMENDATION: spend sim-to-real effort on actuator + sensing models (what dominates); keep
adaptive stepping as a dataset tool for stiff / non-convex / tunneling-contact scenes.
Reusable assets built: paired-eval RNG + single-term refs + analysis --rank (scripts/rl/), the
CENIC cumulative substep counter (solver_mujoco_cenic), and two work-precision harnesses
(v1_work_precision.py soft, v1_stiff_contact.py pile). Figure: results/plots/p0b_error_budget.png.

---

## Environment notes
- GPU: RTX 4070 Ti SUPER (16 GiB), Warp+CUDA OK. torch 2.12+cu130, rsl_rl v1.0.2.
- Stack install: `uv sync --extra rl --extra examples --extra importers` (self-referential
  extras don't resolve under a single `--extra rl`; GitPython + trimesh were missing). Done.
- anymal_c URDF + pretrained `anymal_walking_policy_physx.pt` cached; H1/G1 assets cached.
- KNOWN PRE-EXISTING (not from this work): `scripts.rl.smoke_test` runs `device="cpu"` but
  `cenic_env` sets `wp_device = wp.get_device()` -> CUDA on a GPU box, so model lands on CUDA
  while torch tensors are CPU -> device-mismatch RuntimeError. The test is labeled
  "Mac-runnable"; training/eval use device=cuda and are unaffected (validated on CUDA).

## Run log
- 2026-06-16 ~06:30Z: Phase 0a implemented + VERIFIED (bit-exact pairing). Stack brought up.
- 2026-06-16 ~06:40Z: launched cenic dr-off training (4096 envs, 1000 iters); ETA ~99 min.
- 2026-06-16 ~07:48Z: policy converged by iter ~700 (episode length maxed at 1000, lvte 0.17).
  GPU only 2 GB used, so launched Phase 0b eval on model_700 in parallel: 9 backends x 4 seeds
  (paired). NOTE: Bash tool runs zsh (no word-split on unquoted vars) -- inline multi-word args.
- 2026-06-16 ~08:11Z: training done (model_1000, 91 min, final lvte 0.15). PREVIEW from the
  partial run was already decisive: integrator gaps tiny (id lvte 0.120 vs ref_tol 0.124,
  ref_dt 0.121 = ~0.001-0.004). Found+fixed a design flaw: single-term physical refs were
  built on the tight-tol integrator (conflated physics+integrator AND slow); now they hold the
  integrator identical to id (train_spec) and perturb one channel only. Relaunched on model_1000,
  horizon 500, 4 seeds, GPU free.
- 2026-06-16 ~08:50Z: 2/4 seeds done; ranking (lvte) UNAMBIGUOUS and stable across seeds:
  integrator refs at the BOTTOM with NEGATIVE gaps (ref_dt -0.0009 [-0.0016,-0.0002], ref_tol
  -0.0007 [-0.0012,-0.0003]) -- policy is not hurt by its integrator. Worst channel = obs_noise
  (sensing) +0.0065, ~7x the integrator; kp +0.0018. All gaps <=6% of the 0.10 baseline.
  Fixed a misleading auto-verdict heuristic (CI-excludes-0 != survives; negative gap = KILL).
  Built+verified the error-budget bar chart (plot_error_budget.py). Awaiting seeds 3-4 for the
  final pooled verdict + figure. Leaning: KILL V2, pivot to V1.
- 2026-06-16 ~09:17Z: PHASE 0b FINALIZED on 3 paired seeds (lvte+avte agree). KILL the V2
  transfer claim: integrator gap is negative and at the bottom of the budget; obs_noise (~7.6x)
  and kp (~3x) dominate. Figure: results/plots/p0b_error_budget.png. V2 Phases 1/2/3 shelved.
  PIVOT to V1 (data-fidelity). Updated memory project_sim2real_strategy with the empirical result.
  (Later regenerated on 4 seeds; conclusion unchanged, CIs tighter.)
- 2026-06-16 ~09:30Z: V1-A done -- cumulative substep counter wired into SolverMuJoCoCENIC and
  VERIFIED (10 step_dt -> 90 substeps = 30 iters x 3; reset works). Next: V1-B work-precision
  drop-test harness.
- 2026-06-16 ~09:45Z: V1-B done -- ANYmal foot-strike work-precision INCONCLUSIVE (no clean
  adaptive win; trajectory error + penetration both non-monotonic in tol). Mechanistic reason:
  CENIC controls position error, not penetration, and soft gentle foot-contact barely moves it.
  This is the WRONG regime, consistent with the Phase 0b KILL. Sharpened the meeting story to a
  regime-specific conclusion. Next: V1-C stiff/tunneling-contact work-precision (the demonstrated
  regime) for the clean win figure. Did NOT send the muddy ANYmal figure (would mislead).
- 2026-06-16 ~10:00Z: V1-C done -- contact_objects pile work-precision ALSO no clean win:
  trajectory error chaos-confounded; penetration is soft-contact compliance (gold itself -22.8mm),
  not tunneling. Root cause: MuJoCo soft contact is dt-forgiving everywhere. The real win is
  non-convex SDF tunneling, which current dish_rack can't show (solid-box rack approx) -> lives in
  the existing poster. STUDY CONCLUDED. Final synthesis written; loop stopped.
