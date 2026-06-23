# Trossen RL + Adaptive Solver — Detailed Roadmap

**Living doc — update the ☐/◐/☑ marks in place across sessions.** Last updated: 2026-06-21.

> **Banner (2026-06-21):** the container→native-Ubuntu transition is **done** (binary Isaac Sim +
> `run_native.sh` + `paths.py`; no podman/container). The still-hold **jitter fix is the actuator-gain
> change** in `stationary_ai.py` (`left_arm` 80/4), **committed but UNVALIDATED** — no native retrain has
> run against it yet.

This is the authoritative, self-contained phase/deliverable tracker. For pinned USD/Isaac values see
[`IMPL_GROUND_TRUTH.md`](IMPL_GROUND_TRUTH.md).

## North star
High-fidelity, contact-rich manipulation **sim data + RL**, produced with **adaptive + convex integration
(ICF/SAP) for RL manipulation, inside Isaac Lab**. Today that means the **adaptive solver** (adaptive
timestepping over MuJoCo-Warp); the goal is **true CENIC = adaptive + convex ICF/SAP contact**, not yet
built. Trossen Stationary AI now → **humanoid + LEAP hand** later (the ideal showcase: many stiff
fingertip contacts where fixed-step tunnels).

Two threads that converge — **A** is the vehicle, **B** is the thesis that rides in it:
- **Thread A — Trossen teacher/student RL testbed** (the manipulation task + policies).
- **Thread B — adaptive-solver integration into Isaac Lab** (swap fixed-step for adaptive dt; convex
  ICF/SAP contact is the follow-on toward true CENIC).

---

## Big Picture — Target Architecture (Newton → Sim → Lab)

One pipeline where **your adaptive integrator is the physics, top to bottom** — Trossen now,
humanoid + LEAP hand later:

```
  Isaac Lab    RL managers · teacher/student · GR00T / Mimic data-amp
     |         (manager-based env; NewtonMJWarpManager selects the solver)
  Isaac Sim    USD scene · RTX render · LIVE VIEWER · sensors
     |         (Newton is the physics backend here -- this is the "Sim" integration hop)
  Newton       ModelBuilder/Model/State + SolverMuJoCoAdaptive  (adaptive dt via step_dt)
               ^ YOUR integrator lives here -- already done
```

**The adaptive solver plugs in at exactly two seams** (inside the Newton backend / `NewtonMJWarpManager`):
`_build_solver` → construct `SolverMuJoCoAdaptive`; `_run_solver_substeps` → call `step_dt(outer_dt)`.
Everything above the outer-dt boundary (sensors, render, RL, the live viewer) is untouched — it just
sees a sim clock ticking at the outer rate.

**Principles that keep it industry-grade (not merge-hell):**
- **Solver, not fork.** Keep the adaptive solver a clean *registered Newton solver* that tracks upstream
  Newton, so it stays version-compatible with whatever Newton Isaac pins. "Use the adaptive solver"
  becomes a *selection*, not an engine rebase. (Same discipline will carry the future convex ICF/SAP
  contact — true CENIC — when it lands.)
- **Config-selectable end-state.** `solver = "mujoco_adaptive"` in the env/sim cfg builds it. (A literal
  Isaac Sim GUI dropdown is optional product polish, not required for the science.)

**Productive workflow ("use it like a tool"):**
- **Newton = the fast hypothesis loop** — config → run → log-log plot, seconds, no Kit/Isaac launch.
  Prove/disprove the integrator here (Phase B0). This is your Photoshop for the *science*.
- **Isaac Sim/Lab = scale + LIVE VIEWER + data-gen/GR00T** — where *proven* results go to run at scale,
  be watched live, and generate datasets. The live viewer is a first-class deliverable (Thread A).

---

**Status legend:** ☐ not started · ◐ in progress · ☑ done · ⊘ blocked

---

## Where we left off (latest)
- **▶ CURRENTLY WORKING ON (2026-06-21): validate the shipped jitter fix on a native retrain.**
  Decision (user): finish + polish single-arm now (Franka-faithful, unblocks the adaptive thesis); bimanual
  is a real follow-on, NOT a config bump -- see the deferred note below. Two open items gate the relaunch:
  1. **Spawn = the whole REACHABLE footprint, true-random.** "Whole table" is physically impossible for one
     arm bolted at y=+0.46 (the far half is unreachable -> unsolvable episodes poison domain randomization).
     The spawn is ALREADY continuous uniform random (the 9-cube grid was only a visualization, not how it
     samples). Current code spawns the cube at init `[0,0.13,0.05]` with reset y delta `(-0.075,0.075)` and
     commands the goal over pos_x `(-0.12,0.12)` / pos_y `(-0.10,0.05)` / pos_z `(0.08,0.25)` -- the
     measured graspable footprint in front of the arm. (FK footprint sampling lived in the now-archived
     `archive/reach_map.py` -> `archive/plot_reach_map.py`, run with `enabled_self_collisions=False` for
     pure kinematics; kept for re-measuring if the geometry changes.)
  2. **Jitter (still-hold) fix -- SHIPPED as an actuator-gain change, committed but UNVALIDATED.** The fix
     is in `trossen_cube/assets/stationary_ai.py`: `left_arm` `stiffness=80`, `damping=4` (Isaac Lab
     manipulation reference gains), replacing the ~500×-too-stiff USD-baked gains (~40000/340), whose
     ~30-100 Hz PD bandwidth the 50 Hz control loop could not follow -> the buzz. This is NOT a reward
     change: the reward block is still the stock Franka 4-term reward (only `minimal_height` 0.09). The
     reward-space levers (`action_rate`/`joint_vel` curriculum, entropy anneal) survive only as RESERVE
     remedies, wired as `train_teacher.py --joint_vel_weight/--action_rate_weight/--entropy_coef`, if the
     gain fix alone does not settle the hold. The superseded reward-lever guide (which prescribed a
     `stillness_reward.py` that was never created) is archived with a banner at
     [`archive/JITTER_FIX_GUIDE.md`](archive/JITTER_FIX_GUIDE.md). **No native retrain has run against the
     gain fix yet** -- that retrain is the validation.
  Relaunch to validate: `run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 2048
  --max_iterations 1500` on the no-rails scene.
- **⏭ DEFERRED MILESTONE -- bimanual HANDOFF task.** Actuate both arms (14 DOF) for full-table reach (left
  owns +y half, right owns -y half). Substantially harder RL: ~2x action space + which-arm credit
  assignment + arm-arm collision, and NO Franka reference reward (Franka is single-arm) -> needs its own
  bimanual reward design (min-dist over both grippers / handoff) + a spec. Do AFTER single-arm + adaptive
  are solid. (memory `project_trossen_teacher_student` M4 stretch.)
- **Faithful-to-Franka audit (2026-06-18):** reward is 100% the stock `LiftEnvCfg` 4-term reward (only
  `minimal_height` 0.09 on the lift + both goal-tracking terms, a table-height re-zero from the reference's
  0.04, NOT a shape change). Custom `grasp_rewards.py` is **NOT on disk** (was unused dead code, deleted;
  it can be recovered from git history if grasp-shaping is re-layered, but do not treat it as a live file).
  Forced deviations are hardware/geometry only (Trossen robot, +y workspace, cube scale 0.9, EE offset,
  rails removed) plus `std_type="log"` (crash-safe) and the privileged obs split (teacher/student design;
  the actor still sees the object).
- **Code-review fixes (2026-06-18):** corrected the stale PPO docstring (it described the OLD blind-actor
  asymmetric design -- the exact trap that would re-break the fix; it's a SYMMETRIC privileged teacher,
  both actor+critic see `privileged`). `train_teacher.py` default `num_envs` 4096→**2048** (validated;
  4096 unverified, contact-drop is silent). Added `--entropy_coef` override for the anneal continuation.
- **Precision levers, ranked by faithfulness-to-reference** (placement is precision-bottlenecked, not
  under-trained): **(1) entropy anneal** -- the cheapest, most faithful; `--entropy_coef 0.001` +
  `--resume_from` for a fine-tune continuation (rsl-rl has no built-in schedule). **(2) gamma 0.98→0.99**
  -- DIVERGES; 0.98 gives only ~1 s credit horizon (control dt 0.02 s) vs a 5 s episode, so goal-tracking
  is over-discounted during reach/lift; run as a controlled A/B vs (1), not a default. **(3) obs_norm on
  the CRITIC only** -- heterogeneous obs scales (rad/m/command); low-risk on the critic (better value →
  better placement credit); leave the actor off (it interacts with action std). Test in isolation.
- **⏸ (history) NO-RAILS RESTART (full retrain from scratch).** `model_3498` was still
  jittery AND was exploiting the rig's perimeter rail as a crutch (jamming the cube against `frame_link`),
  and the workspace let the goal/cube creep back under the arm. Three fixes: **(1) rails removed** --
  `stationary_ai_norails.usda` (`make_norails_usd.py`) sublayers the rig USD and deactivates `frame_link`
  collision + hides its visual; env robot spawn now points at it (arms + tabletop untouched, original USD
  intact). **(2) workspace re-centred out-in-front + reachable** -- cube init `[0,0.13,0.05]` with reset y
  delta `(-0.075,0.075)`; goal commanded over pos_x `(-0.12,0.12)` / pos_y `(-0.10,0.05)` / pos_z
  `(0.08,0.25)` (the region the rails run already grasped across).
  NOTE per user: **height is fine, the only constraint is reachability** (don't throw the goal past the
  arm's reach or back under the base) -- do NOT lower the z ceiling. **(3) fresh from scratch** so the
  action-smoothness curriculum ramps from step 0 (the resume had reset + under-penalised it). Old run
  archived → `<LOG_ROOT>/stationary_ai_lift_teacher.rails_v1` (`model_0..3498` preserved as fallback).
  Boot verified: iter 0 start, `object_dropping=0` (cube doesn't tunnel without rails). Log under
  `<ARTIFACT_ROOT>` (`full_train_norails.log`). On completion: render rollout (rails gone? jitter calmer?),
  re-plot. **If still jittery → entropy anneal** (`entropy_coef` 0.006→~0.001), the reserve lever.
- **✅ DONE (2026-06-18): precision extension run (`model_3498`).** Resumed `model_1499` → iter **3499**
  (+2000, 2048 envs, `<ARTIFACT_ROOT>/full_train_long.log`). Numbering continued so `model_0..1499` are
  preserved. **Result: small real gain, then plateau — precision is bottlenecked, not under-trained.**
  Before→after (last-100 mean): `lifting` 11.99→12.62, coarse `goal_tracking` 9.45→10.08,
  `fine_grained` 0.54→0.62 (peaked 0.76 @~3270 then drifted down — still only ~12-15% of its 5.0 ceiling),
  `reaching` 0.55 flat. Compare render `<ARTIFACT_ROOT>/compare_1499_vs_3498.mp4` (`--ckpts 1499 3498`): new
  policy LOOKS much calmer at the hold, but CONFOUNDED — the two episodes drew different random goals
  (old=high/aloft = harder to hold still, new=low/near-table = easier). The averaged reward numbers are the
  fair signal. **`model_3498` banked as the teacher** (slightly better than 1499; grasp preserved as fallback).
  **Next precision lever is NOT more iterations** — it's the entropy anneal (`entropy_coef` 0.006→~0.001 or a
  decay, optionally a stronger `action_rate`), a PPO-hyperparameter change faithful to the reference reward,
  which targets the precise-still-hold + chatter (same root cause). Plot tool: `plot_reward_curves.py`
  (now takes multiple logs); render filter: `render_timelapse.py --ckpts <iters>`.
- **✅ TEACHER GRASPS + LIFTS (A3 done, 2026-06-18).** Final 2048-env/1500-iter run: `lifting` 0→~12/15,
  `goal_tracking` →~9.5/16, mean reward ~100, 0 Xid. Rollout video (`<ARTIFACT_ROOT>/teacher_grasp_h264.mp4`)
  visually confirms the reach→close→lift. The STOCK reference reward worked once **four stacked bugs**
  were cleared (blind actor · free-lift `minimal_height` · 7cm EE offset · cube < closed gripper) — see A3.
- **Code is a minimal documented diff from the reference Franka lift.** Speculative additions we piled on
  while chasing the symptom (grasp term, DR, obs-norm) were reverted; `grasp_rewards.py` is NOT on disk
  (deleted dead code, recoverable from git history) and `diag_grasp_geom.py` now lives in `archive/` — both
  for when we re-layer grasp/DR *deliberately*. Diff = Trossen robot + +y geometry +
  `minimal_height=0.09` + GPU buffers + teacher/student obs split + `std_type="log"`.
- **Next — pick one:** (A8) re-add DR now that the baseline grasps · (A4) vision student / depth-CNN distill ·
  or **(Thread B) pivot to the adaptive thesis** (the actual research goal) using this proven Trossen scene.
- **Lessons banked (the day's real cost):** (1) watch the *rollout*, the reward number lies; (2) when a
  policy fails like a reward/exploration problem, FIRST check what the actor can observe — a blind actor
  mimics every other failure; (3) don't diverge from a proven reference without a reason.
- Adaptive integration parked. The plan (the Thread B deliverable list) is captured inline in Thread B
  below: B0 standalone Newton adaptive-vs-fixed work-precision demo (go/no-go) → B1 stand up the Isaac Lab
  3.0 Beta Newton backend → B2 reconcile the `newton-cenic` fork with that pinned Newton and swap
  `SolverMuJoCoAdaptive` in → B3 a Level-B substep driver subclassing `NewtonMJWarpManager` → B4 in-Isaac
  work-precision validation on a stiff scene → B5 humanoid + LEAP-hand dexterous scene with GR00T/Mimic
  data-amp on adaptive data.

---

## Thread A — Trossen Stationary AI teacher/student RL testbed

Privileged-state **PPO teacher → depth-CNN student (distillation)** that makes the LEFT arm of the
bimanual Stationary AI rig pick up a cube, in Isaac Lab (2.3.2 + PhysX), running **natively on Ubuntu**
against a binary Isaac Sim install.

- ☑ **A0 — Native Isaac Lab bring-up.** Binary Isaac Sim install under `~/Documents/code/` (beside the
  repo + IsaacLab clone). Scripts run through `run_native.sh`, which calls
  `${ISAACLAB:-~/Documents/code/IsaacLab}/isaaclab.sh -p`. One-time: symlink the binary Isaac Sim as
  `IsaacLab/_isaac_sim`, then `isaaclab.sh --install`; editable-install this package with
  `isaaclab.sh -p -m pip install -e scripts/rl/trossen`. Filesystem roots come from `trossen_cube/paths.py`
  (default data root `~/Documents/code/isaac-data`, per-root env overrides). The old podman container path is
  archived in `legacy/CONTAINER.md`.
- ☑ **A1 — Greenfield Stationary AI articulation (16 DOF).** `trossen_cube/assets/stationary_ai.py`.
  Left+right arms `follower_{l,r}_joint_[0-5]`; each gripper actuates only its LEFT carriage (right
  carriage is a verified `physxMimicJoint`, gearing -1.0). The `14 != 16 actuators` warning is benign.
- ☑ **A2 — Teacher manager env.** `trossen_cube/tasks/cube_lift/cube_lift_env_cfg.py`. Obs groups
  `policy`[39] (joint_pos/vel + last action) + `privileged`[10] (object pose + command); action 7
  (6 arm + 1 gripper). **Geometry corrected 2026-06-17** (see Infra log below): cube in the left
  arm's +y workspace, ee_frame at the grasp TCP, rig's own tabletop, GPU buffers raised.
- ☑ **A3 — PPO teacher. GRASPS + LIFTS (2026-06-18).** `train_teacher.py` + `agents/rsl_rl_ppo_cfg.py`.
  Final 2048-env / 1500-iter run: `lifting_object` 0→~12/15, `object_goal_tracking` →~9.5/16, mean
  reward ~100, 31 checkpoints, **0 Xid**. Rollout render (`teacher_grasp_h264.mp4`, model_0→1499)
  VISUALLY CONFIRMS the grasp — the left arm reaches, closes on the cube, lifts it. Grasp-and-lift is
  solid; goal-*placement* is partial (~9.5/16, carries toward target but doesn't always nail the pose).
  **No grasp term, no DR needed** — the STOCK reference 4-term reward worked once **four stacked bugs**
  were cleared: (1) **blind actor** — `object_position` was in the critic-only `privileged` group, so
  the actor never saw the cube → fixed via `obs_groups` actor `["policy","privileged"]` (the single
  biggest fix; `reaching` 0.23→0.62); (2) **free-lift reward** — `minimal_height` 0.04 < the cube's
  ~0.048 rest height on the raised rig table → specification gaming → fixed to **0.09** on lift + both
  goal-tracking terms; (3) **EE offset 7cm off** — reach TCP was at `ee_gripper_link` (0.1561), ~7cm
  PAST the fingers → fixed to the finger midpoint **0.087**; (4) **cube too small** — DexCube scale 0.8
  (4.8cm) < closed gripper gap 4.83cm → fingers closed past it → fixed to scale **0.9** (~5.4cm).
  Held-back refinement if placement precision matters: a **contact-gated grasp term** (robosuite-style) —
  reward the grasp only when both fingers register contact with the cube (gate on left+right fingertip
  contact sensors), which shapes a firm two-finger pinch instead of crediting a one-sided nudge or a
  hover; add it deliberately on top of the proven stock reward, not as a debugging shotgun.
- ☐ **A4 — Vision student (CNN distillation).** NOT STARTED. Add a `cam_high` depth `TiledCamera` +
  `images` obs group (NHWC→permute CHW), `RslRlCNNModelCfg`. obs_groups
  `{"student":["policy","images"], "teacher":["policy","privileged"]}`. Plan: distill the privileged-state
  PPO teacher (which sees the cube pose via `privileged`) into a student that sees only proprioception +
  a depth image, training the student to match the teacher's actions so the deployable policy never needs
  ground-truth object state.
- ☐ **A5 — Eval.** `eval.py` predicate success (grasped ∧ lift-height ∧ at-rest); teacher vs student.
- ☑ **A6 — Timelapse.** `render_timelapse.py` works (camera reframed to +y). Rendered 24 checkpoints
  2026-06-17 — which is how we *saw* the A3 reward bug (arm never grasps). Reward curves hid it; the
  render exposed it. Lesson banked: watch the rollout, not just the metrics.
- ☐ **A7 — Live viewer (REQUIRED).** Watch policies/scenes *live*, not via file renders. Now that we run
  natively on Ubuntu, the simplest path is a **native Kit window** (run a script non-`--headless` through
  `run_native.sh`). WebRTC livestream (Isaac Sim streams the viewport → a browser) stays an option for
  headless/remote viewing. First-class per the Big Picture workflow.
- ☐ **A8 — Domain randomization (generality / sim-to-real).** *Wiring PROVEN then REVERTED 2026-06-17
  to align the baseline with the reference (re-add deliberately AFTER the bare reference reward grasps).
  When re-adding: friction + mass + gripper-finger friction as startup events (validated, 256-env smoke
  rc=0); cube stays a cube; scale/shape later; then privileged-obs of the DR params.*
  Randomize the manipuland so the policy generalizes — and so 1500 iters actually *learn* something
  instead of converging on a do-nothing optimum. **friction + mass first** (reset events `mdp.randomize_rigid_body_material` on the cube +
  gripper fingers, `randomize_rigid_body_mass` — easy, biggest payoff for grasping), then **scale/size**
  (per-env startup), then **shape** (multi-asset spawner) for a truly general grasper. **Hard prereq:
  the cube-centric reward (A3 fix)** — "lifted" must be relative to the cube's rest/table height, else
  every randomized size re-triggers the absolute-height bug. Put the randomized params (friction/mass/
  size) in the TEACHER's privileged obs so the teacher conditions on them and the student learns to infer them.

**Known caveat to re-check before trusting grasp quality:** the ee_frame TCP offset
`EE_TCP_OFFSET=(0.087,0,0)` (link_6 local x, the finger midpoint) was derived from the USD; confirm
visually in the `ee_closeup` render / by whether the trained arm actually closes around the cube.

---

## Thread B — adaptive-solver integration into Isaac Lab

Integrates the **adaptive solver** (`SolverMuJoCoAdaptive` — adaptive timestepping over MuJoCo-Warp) that
exists today. Convex ICF/SAP contact — the rest of **true CENIC** (adaptive + convex contact) — is the
follow-on after this lands, not part of B0–B5.

Full deliverable list (B0–B5), captured inline here so this tracker stays self-contained:

- ☐ **B0 — Preliminary adaptive-vs-fixed demo** (standalone Newton, fast evidence): import a Trossen
  arm USD via `ModelBuilder.add_usd`, build a stiff SDF insertion scene, work-precision sweep
  (reuse `scripts/rl/adaptive_expts/v1_work_precision.py`, the B0 harness). **Go/no-go for the Isaac spend.**
- ☐ **B1 — Isaac Lab Newton backend** stood up (Isaac Lab **3.0 Beta** / `isaaclab_newton` — NOT in
  the installed 2.3.2; this is a real migration, do it in an isolated env/branch).
- ☐ **B2 — adaptive solver into the backend** (reconcile the `newton-cenic` fork with 3.0 Beta's pinned
  Newton; swap `SolverMuJoCoAdaptive` in).
- ☐ **B3 — Level B substep driver** (subclass `NewtonMJWarpManager`: `_build_solver`→`SolverMuJoCoAdaptive`,
  `_run_solver_substeps`→`step_dt`, `use_cuda_graph=False`; confirm `opt.timestep` isn't clobbered).
- ☐ **B4 — In-Isaac validation** (work-precision adaptive-vs-fixed on a stiff Isaac scene = M0 in target).
- ☐ **B5 — Future:** humanoid + LEAP hand dexterous scene; GR00T/Mimic data-amp on adaptive data.

---

## Infrastructure & hard-won facts (don't re-derive these)

- **NVIDIA driver Xid 56 (legacy Fedora note):** on the old Fedora host the open kernel module 580.159.03
  threw Xid 56 display-engine crashes under load on the RTX 4070 Ti SUPER; the fix was to force the
  closed/proprietary module via RPM Fusion. Fedora/RPM-Fusion-specific — see `legacy/CONTAINER.md` for the
  full procedure; not relevant on the native Ubuntu host.
- **Env geometry (current):** the env was a WXAI single-arm copy with the wrong workspace. Fixed: cube
  init `[0,0.13,0.05]` (left arm +y, on the rig tabletop) with reset y delta `(-0.075,0.075)`; goal command
  ranges pos_x `(-0.12,0.12)` / pos_y `(-0.10,0.05)` / pos_z `(0.08,0.25)`; dropped the foreign
  SeattleLabTable (`self.scene.table=None`), ground at z=0. ee_frame: `follower_left_ee_gripper_link` is
  NOT an articulation body → use `EE_LINK=follower_left_link_6` + `EE_TCP_OFFSET=(0.087,0,0)` (the finger
  midpoint; the earlier `0.1561` = `ee_gripper_link` put the TCP ~7cm past the fingers).
- **GPU contact buffers (2026-06-17):** at 2048 envs the bimanual rig overflows the default GPU
  collision buffers → contacts dropped → **cube tunnels through the table → ~88% object_dropping**.
  Fixed in the env cfg: `gpu_max_rigid_patch_count=2**20`, `gpu_total_aggregate_pairs_capacity=2**23`,
  `gpu_found_lost_aggregate_pairs_capacity=2**26`. (Default patch count 5*2**15=163840 was too small.)
- **Code gotchas (load-bearing, NOT container-specific):** launch `AppLauncher` before importing
  `isaaclab.*` / `trossen_cube`; call `handle_deprecated_rsl_rl_cfg` after loading the rsl_rl cfg;
  `torch.no_grad` (not `inference_mode`) for multi-checkpoint rollouts; detect success via marker files
  (Kit eats stdout). (Legacy container note: Fedora ffmpeg lacked libx264, so mp4s had to be re-encoded
  in-container — moot on the native Ubuntu host where libx264 is available.)
- **PPO stability (2026-06-17):** the corrected (harder, sparser) reward crashed rsl-rl at iter 119 with
  `RuntimeError: normal expects all elements of std >= 0.0` — a loss spike pushed the **raw `std_type="scalar"`**
  action-noise std negative. Fix in `agents/rsl_rl_ppo_cfg.py`: `std_type="log"` (std = exp(clamp(log_std)),
  always > 0) + `obs_normalization=True` on actor & critic. The broken (idle) reward never hit this; harder
  rewards do.
- **Privileged-teacher obs (2026-06-17) -- THE root cause of the no-grasp stall:** the teacher's
  ACTOR must observe the object. We had `obs_groups={"actor":["policy"],"critic":["policy","privileged"]}`
  with `object_position` only in the `privileged` group -> the actor was BLIND to the randomized cube,
  could only reach the average spawn (~10 cm off), and could never grasp. (The reference Isaac Lab lift
  task puts `object_position` directly in its single policy obs group.) Fix: `obs_groups` actor =
  `["policy","privileged"]`. Asymmetric actor-critic with a blind actor is for a *deployable*
  proprioception-only policy, NOT a privileged teacher. This masqueraded as a reward/geometry/grasp bug.

---

## Open risks / unknowns
- **A:** TCP offset accuracy (grasp quality); student camera offset + CNN channels/kernels unverified.
- **B:** the `newton-cenic` fork ↔ Isaac-Lab-3.0-Beta Newton version skew (B2); whether the Lab
  manager re-broadcasts/clobbers the adaptive solver's per-world `opt.timestep` between substeps (B3); 3.0 Beta
  migration risk to the working 2.3.2 Trossen setup (do it isolated).

## Command reference

All scripts run natively through `run_native.sh` (calls `isaaclab.sh -p`); outputs default to the roots in
`trossen_cube/paths.py` (`<ARTIFACT_ROOT>` for renders, `<LOG_ROOT>` for checkpoints) when `--out`/`--out_dir`
are omitted.

```bash
# one-time: editable-install the package into Isaac's bundled python
~/Documents/code/IsaacLab/isaaclab.sh -p -m pip install -e scripts/rl/trossen

# train teacher (corrected env): checkpoints every 50, writes train_done.json
scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py \
  --headless --num_envs 2048 --max_iterations 1500
# scene-inspection render (geometry pre-flight, light load): 4 labeled PNGs (-> <ARTIFACT_ROOT>/scene_check)
scripts/rl/trossen/run_native.sh scripts/rl/trossen/inspect_scene.py \
  --headless --enable_cameras
# training timelapse (after checkpoints exist) -> <ARTIFACT_ROOT>/teacher_timelapse.mp4
scripts/rl/trossen/run_native.sh scripts/rl/trossen/render_timelapse.py \
  --headless --enable_cameras --steps 120
```
