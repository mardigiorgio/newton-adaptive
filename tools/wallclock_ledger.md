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
- Snapshot commits: newton-adaptive march-counter-log 9c9dc934, sap_warp
  main 79e43bd, IsaacLab develop 82c0679d88.

## Backlog (ranked; teardown of contact_solve internals is AUTHORIZED)

1. PENDING IMMEDIATE: determinism default flip to OFF (Marco's order,
   2026-08-15) — mjwarp itself is non-canonical (157 atomic_add sites, no
   det mode; their forward_test.py:461 sorts around it), so det-ON holds SAP
   to a standard the baseline doesn't meet. Flip default, keep opt-in for
   probes (they pin det=1 explicitly already). Frees ~7.9% for trainings.
   BLOCKED until fp32 agent stops editing solver files.
2. Blocked-Cholesky narrowing (sap_warp/sim/blocked_cholesky.py): the
   largest single remaining env-axis consumer after narrow-grid v2 (was out
   of that task's file scope). Same list-indexed pattern, bitwise.
3. Mixed-precision iterative refinement: fp32 factorization/GEMM + fp64
   residual + refinement loop — targets fp64-class accuracy at fp32 rate;
   candidate to keep the 1e-8 optimality contract honestly if pure fp32
   cannot. Flagship hands-dirty item.
4. Shared assembly between full and half1 solves: same anchor state q_t =>
   byte-identical contact set/Jacobians/Delassus; only R(dt) and vhat
   differ. Compute once, read twice — bitwise by construction. Est. ~10-15%
   if assembly is ~1/3 of slab.
5. Stream overlap of full and half1 solves (data-independent; half2 depends
   on half1). Hides one solve's latency where kernels underfill the GPU.
   Canonical-per-solve reductions keep det mode compatible.
6. LS-interior compaction isolated A/B at 1024 production scale: the ONE
   stack feature never isolated at scale; micro-scene measured it 1.6-1.8%
   SLOWER. If negative at scale too: default it OFF (one env var, free wall).
7. Fused attempt pipeline / dense-path tile reshape (the (envs,32,32) pack,
   GEMM tiles): structural surgery, authorized; measure kernel-level first.
8. Remaining un-narrowed env-axis launches outside contact_solve.py
   (full-width consumers listed in agent a06d1420 notes).

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
