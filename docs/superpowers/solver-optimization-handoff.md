# Handoff: push SolverMuJoCoAdaptive faster on dense contact

You are a fresh Claude Code instance continuing an optimization effort. Read this whole file,
then reproduce the current numbers before changing anything. Use the systematic-debugging skill
(root cause before fixes). Every speedup MUST keep the accuracy guarantee: per-world per-step
step-doubling error <= tol. Do not cherry-pick easier scenes. The user commits all git changes
themselves (do not commit). No em dashes; prioritize readability.

## Goal
Make `SolverMuJoCoAdaptive` as fast as possible on the CONTACT-HEAVY scene `contact_objects`
(9 spheres + 9 boxes per world, randomized), to beat the fixed-1ms baseline by as much as possible
at high world counts N. For a poster, deadline ~1 week.

## Repo / running
- /home/marcodigiorgio/Documents/CODE/newton-cenic (Newton fork, MuJoCo Warp backend)
- Run with `uv run`. Set `WARP_CACHE_ROOT=/tmp/claude/warp-cache-$$`.

## Already implemented (works; in newton/_src/solvers/mujoco/solver_mujoco_adaptive.py)
1. World compaction: each inner iteration step only active worlds via a smaller `mjw_data` tier
   (reuse full mjw_model; worlds are identical replicas). Gather active qpos/qvel/qacc_warmstart/dt
   into tier, step, scatter back. Geometric tiers (ratio 1.5) N->64. Needs native MuJoCo contacts
   (use_mujoco_contacts=True) so the tier recomputes contacts (no contact re-batch). Auto-on when
   use_mujoco_contacts=True and N>=256. Active set from CompactionMixin
   (scripts/adaptive/compaction_mixin.py): self._wrapper._active_indices / _n_active_buf, ASCENDING.
2. CUDA-graph capture of every step (per tier + full): _capture_step_graph / _run_step. Collapses
   MuJoCo Warp's dozens of per-step kernel launches into one replay. Key low-active-count win.

Read these to understand the current state:
- newton/_src/solvers/mujoco/solver_mujoco_adaptive.py  (compaction + graphs + _step_fn dispatch)
- scripts/adaptive/base.py  (AdaptiveWrapper: the K-iteration step-doubling loop, step_dt)
- scripts/adaptive/compaction_mixin.py  (active-world tracking)
- scripts/adaptive/controller.py + scripts/adaptive/kernels.py::_calc_adjusted_step  (dt controller)

## Current results (contact_objects, episode avg, RTX 4070 Ti Super)
- plain adaptive ~2829 ms/outer; +compaction ~700; +graphs ~245 (N=1024) => ~11.5x over plain.
- vs fixed-1ms: 0.33x@256, 0.95x@512, 1.08x@1024, 1.29x@2048 (>1 = adaptive faster). Win grows with N.
- Accuracy honesty check PASSED with compaction: worst-world per-step error 100% <= tol(1e-3).

## Current bottleneck (profiled)
Remaining time is genuine contact-solve compute for the stiff worlds: K (inner iters, ~24-39 in
dense contact) x 3 substeps (step-doubling: full + 2 halves) x tier step. Full-N state sync is
negligible (0.033 ms). Cutting MuJoCo solver iterations gave only ~4% and caused NaNs (dead end).

## Untried levers (highest ceiling first)
1. Kill the 3x step-doubling tax: 3 evals/step exist only to estimate error. Investigate a cheaper
   error estimator, or an embedded/Richardson scheme so the full eval isn't wasted and/or
   higher-order accuracy permits larger dt (fewer steps). Biggest potential win.
2. Reduce K: worst world drives K for all worlds in an outer step. Check controller for wasted/
   rejected iterations or over-conservatism (deadband [0.9,1.2], safety 0.9, kp=0 unused).
3. Capture the whole iteration body (gather/scatter are still eager, variable dim = n_active).
   Make active worlds contiguous (sort once per outer step) so scatter is unnecessary and the whole
   body can be one graph.
4. Profile which mjwarp kernels dominate the tier step (constraint solver vs collision) and target.
5. Higher-order integrator for fewer steps at equal accuracy.

## Measurement gotchas (important)
- Always measure error with speed: solver.last_error.numpy().max() must stay <= tol.
- contact_objects never fully settles; use EPISODE AVERAGE over ~100-120 outer steps from the drop,
  not a single warmup step (K spikes to ~39 during the violent drop -> ~2500 ms; steady ~245 ms).
- Worlds overlap at the origin (viewer offsets are visual only); physics worlds are independent.
- Chaos: trajectory comparison across configs is invalid after ~20-40 steps. Use error<=tol, energy,
  or penetration as accuracy metrics, not trajectory diff.
- scaling JSON medians in scripts/bench/results/ are in SECONDS.
- DT_OUTER=0.02. Compaction threshold N>=256.
- Build: `from scripts.scenes import contact_objects as co; m=co.build_model_randomized(N, seed=7)`;
  compaction solver:
  `newton.solvers.SolverMuJoCoAdaptive(m, tol=1e-3, dt_init=co.DT_OUTER, dt_min=1e-6,
   dt_max=co.DT_OUTER, use_mujoco_contacts=True, nconmax=co._NCON, njmax=co._NJM)`.
- Accuracy figure / honesty check:
  `uv run python -m scripts.bench.benchmarks.tol_trace --n 512 --steps 160`
  (adaptive worst-world error must be 100% <= tol).
- Regression: confirm the use_mujoco_contacts=False path (other scenes) still steps (compaction off,
  _run_step falls back to eager).

Start: reproduce the numbers, then attack lever 1 (step-doubling) and 2 (K). Keep error <= tol always.
```

## AUTONOMOUS LOOP MODE (read if launched via /loop)
You are running unattended for ~3 hours. Protect deliverables above all.

Setup (first iteration only):
- The user has committed the current good state as your BASELINE (the compaction + graph work).
  Confirm with `git status` that the tree is clean at start. If it is NOT clean, STOP and log a
  warning instead of proceeding (you need a clean baseline to revert against safely).
- Create a log file docs/superpowers/solver-optimization-log.md and record the START timestamp
  (`date`), the baseline numbers, and the lever list.

Every iteration:
1. Pick the next thing to try (continue the current lever; if exhausted, next lever from the list).
2. Before editing, back up the exact file(s) you will change: `mkdir -p /tmp/solveropt_bak &&
   cp <file> /tmp/solveropt_bak/`. Then make ONE focused change (smallest viable). Code changes only
   in: newton/_src/solvers/mujoco/ , scripts/adaptive/ , scripts/bench/ .
3. Gate it on BOTH:
   a. Accuracy: `uv run python -m scripts.bench.benchmarks.tol_trace --n 512 --steps 160`
      -> adaptive worst-world error MUST stay 100% <= tol. If not, REVERT.
   b. Speed: episode benchmark on contact_objects at N=512,1024,2048 vs fixed-1ms, no NaN.
      If slower than the previous best or any NaN, REVERT.
4. REVERT = restore the changed file(s) from /tmp/solveropt_bak/ (do NOT `git checkout --` or
   `git stash` -- that would also wipe earlier KEPT experiments, which are uncommitted). KEEP =
   leave the edit in place; it accumulates as an uncommitted change for the user to review later.
5. Append to the log: what was tried, speed table, error, KEPT or REVERTED, and why.

HARD rails (never violate):
- NEVER edit poster/ , scripts/figures/ , or any *.png. Those are finished deliverables.
- NEVER git commit, push, stash, reset, or `git checkout --` (the user commits; those would lose
  uncommitted kept work). Revert ONLY via the /tmp/solveropt_bak/ file backups.
- NEVER weaken or remove the accuracy check to claim a speedup. error <= tol is sacred.
- If you hit 3 reverts in a row on one lever, abandon it and move to the next lever; log it.
- Stop condition: if >3 hours have elapsed since START (check `date` vs the logged start), write a
  FINAL SUMMARY to the log (best config, total speedup, what to try next) and end the loop
  (do not schedule another iteration).

If anything is ambiguous or risky, prefer to log the question and skip rather than guess.
