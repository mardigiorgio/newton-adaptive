# Trossen RL + CENIC Adaptive — Detailed Roadmap

**Living doc — update the ☐/◐/☑ marks in place across sessions.** Last updated: 2026-06-18.

This is the authoritative phase/deliverable tracker. For the *current-state snapshot* see
[`HANDOFF.md`](HANDOFF.md); for pinned USD/Isaac values see [`IMPL_GROUND_TRUTH.md`](IMPL_GROUND_TRUTH.md);
for the fresh-session resume prompt see [`RESUME_PROMPT.md`](RESUME_PROMPT.md).

## North star
High-fidelity, contact-rich manipulation **sim data + RL**, produced with **CENIC adaptive
time-stepping inside Isaac Lab**. Trossen Stationary AI now → **humanoid + LEAP hand** later
(the ideal adaptive showcase: many stiff fingertip contacts where fixed-step tunnels).

Two threads that converge — **A** is the vehicle, **B** is the thesis that rides in it:
- **Thread A — Trossen teacher/student RL testbed** (the manipulation task + policies).
- **Thread B — CENIC adaptive integration into Isaac Lab** (swap fixed-step for adaptive dt).

---

## Big Picture — Target Architecture (Newton → Sim → Lab)

One pipeline where **your adaptive integrator is the physics, top to bottom** — Trossen now,
humanoid + LEAP hand later:

```
  Isaac Lab    RL managers · teacher/student · GR00T / Mimic data-amp
     |         (manager-based env; NewtonMJWarpManager selects the solver)
  Isaac Sim    USD scene · RTX render · LIVE VIEWER · sensors
     |         (Newton is the physics backend here -- this is the "Sim" integration hop)
  Newton       ModelBuilder/Model/State + SolverMuJoCoCENIC  (adaptive dt via step_dt)
               ^ YOUR integrator lives here -- already done
```

**CENIC plugs in at exactly two seams** (inside the Newton backend / `NewtonMJWarpManager`):
`_build_solver` → construct `SolverMuJoCoCENIC`; `_run_solver_substeps` → call `step_dt(outer_dt)`.
Everything above the outer-dt boundary (sensors, render, RL, the live viewer) is untouched — it just
sees a sim clock ticking at the outer rate.

**Principles that keep it industry-grade (not merge-hell):**
- **Solver, not fork.** Keep CENIC a clean *registered Newton solver* that tracks upstream Newton, so
  it stays version-compatible with whatever Newton Isaac pins. "Use CENIC" becomes a *selection*, not
  an engine rebase.
- **Config-selectable end-state.** `solver = "mujoco_cenic"` in the env/sim cfg builds it. (A literal
  Isaac Sim GUI dropdown is optional product polish, not required for the science.)

**Productive workflow ("use it like a tool"):**
- **Newton = the fast hypothesis loop** — config → run → log-log plot, seconds, no Kit/container.
  Prove/disprove the integrator here (Phase B0). This is your Photoshop for the *science*.
- **Isaac Sim/Lab = scale + LIVE VIEWER + data-gen/GR00T** — where *proven* results go to run at scale,
  be watched live, and generate datasets. The live viewer is a first-class deliverable (Thread A).

---

**Status legend:** ☐ not started · ◐ in progress · ☑ done · ⊘ blocked

---

## Where we left off (latest)
- **▶ CURRENTLY WORKING ON (2026-06-18): refine the SINGLE arm; bimanual handoff is a later milestone.**
  Decision (user): finish + polish single-arm now (Franka-faithful, unblocks the adaptive thesis); bimanual
  is a real follow-on, NOT a config bump -- see the deferred note below. Two open items gate the relaunch:
  1. **Spawn = the whole REACHABLE footprint, true-random.** "Whole table" is physically impossible for one
     arm bolted at y=+0.46 (the far half is unreachable -> unsolvable episodes poison domain randomization).
     The spawn is ALREADY continuous uniform random (the 9-cube grid was only a visualization, not how it
     samples). Measuring the real graspable footprint by FK sampling: `reach_map.py` -> `plot_reach_map.py`.
     First pass was garbage (self-collisions ON made wild configs explode, TCPs flew past the table edges);
     rerunning with `enabled_self_collisions=False` (pure kinematics). Set cube + goal spawn to that
     measured footprint, then relaunch.
  2. **Jitter (still-hold) fix** -- full implementation guide: [`JITTER_FIX_GUIDE.md`](JITTER_FIX_GUIDE.md)
     (self-contained, for a fresh chat). Diagnosis reordered the levers: the **PRIMARY** cause is the
     reward is flat in stillness-space (motion penalties ~3 orders too weak vs the +31 task reward -> no
     gradient toward a static pose), so the fix REQUIRES a reward change. Lever 1 = strengthen
     `action_rate`/`joint_vel` (1A weight bump) or add an on-goal-gated stillness term (1B); Lever 2 =
     actuator damping; entropy anneal is only a SECONDARY finishing pass (no still optimum to sharpen
     toward until Lever 1 exists). Fold the chosen fix into the relaunch config.
  Relaunch once 1+2 settle: `train_teacher.py --num_envs 2048 --max_iterations 1500` on the no-rails scene.
- **⏭ DEFERRED MILESTONE -- bimanual HANDOFF task.** Actuate both arms (14 DOF) for full-table reach (left
  owns +y half, right owns -y half). Substantially harder RL: ~2x action space + which-arm credit
  assignment + arm-arm collision, and NO Franka reference reward (Franka is single-arm) -> needs its own
  bimanual reward design (min-dist over both grippers / handoff) + a spec. Do AFTER single-arm + adaptive
  are solid. (memory `project_trossen_teacher_student` M4 stretch.)
- **Faithful-to-Franka audit (2026-06-18):** reward is 100% the stock `LiftEnvCfg` 4-term reward (only
  `minimal_height` 0.04→0.09, a table-height re-zero, NOT a shape change). Custom `grasp_rewards.py`
  **deleted** (was unused dead code; recoverable from git). Forced deviations are hardware/geometry only
  (Trossen robot, +y workspace, cube scale 0.9, EE offset, rails removed) plus `std_type="log"` (crash-safe)
  and the privileged obs split (teacher/student design; the actor still sees the object).
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
  intact). **(2) workspace re-centred out-in-front + reachable** -- cube spawn & goal both
  x[-0.12,0.12] y[0.15,0.30] (inside the region the rails run already grasped across), goal z[0.08,0.25].
  NOTE per user: **height is fine, the only constraint is reachability** (don't throw the goal past the
  arm's reach or back under the base) -- do NOT lower the z ceiling. **(3) fresh from scratch** so the
  action-smoothness curriculum ramps from step 0 (the resume had reset + under-penalised it). Old run
  archived → `logs/trossen/stationary_ai_lift_teacher.rails_v1` (`model_0..3498` preserved as fallback).
  Boot verified: iter 0 start, `object_dropping=0` (cube doesn't tunnel without rails). Log
  `~/isaac-rl/full_train_norails.log`. On completion: render rollout (rails gone? jitter calmer?),
  re-plot. **If still jittery → entropy anneal** (`entropy_coef` 0.006→~0.001), the reserve lever.
- **✅ DONE (2026-06-18): precision extension run (`model_3498`).** Resumed `model_1499` → iter **3499**
  (+2000, 2048 envs, `~/isaac-rl/full_train_long.log`). Numbering continued so `model_0..1499` are
  preserved. **Result: small real gain, then plateau — precision is bottlenecked, not under-trained.**
  Before→after (last-100 mean): `lifting` 11.99→12.62, coarse `goal_tracking` 9.45→10.08,
  `fine_grained` 0.54→0.62 (peaked 0.76 @~3270 then drifted down — still only ~12-15% of its 5.0 ceiling),
  `reaching` 0.55 flat. Compare render `~/isaac-rl/compare_1499_vs_3498.mp4` (`--ckpts 1499 3498`): new
  policy LOOKS much calmer at the hold, but CONFOUNDED — the two episodes drew different random goals
  (old=high/aloft = harder to hold still, new=low/near-table = easier). The averaged reward numbers are the
  fair signal. **`model_3498` banked as the teacher** (slightly better than 1499; grasp preserved as fallback).
  **Next precision lever is NOT more iterations** — it's the entropy anneal (`entropy_coef` 0.006→~0.001 or a
  decay, optionally a stronger `action_rate`), a PPO-hyperparameter change faithful to the reference reward,
  which targets the precise-still-hold + chatter (same root cause). Plot tool: `plot_reward_curves.py`
  (now takes multiple logs); render filter: `render_timelapse.py --ckpts <iters>`.
- **✅ TEACHER GRASPS + LIFTS (A3 done, 2026-06-18).** Final 2048-env/1500-iter run: `lifting` 0→~12/15,
  `goal_tracking` →~9.5/16, mean reward ~100, 0 Xid. Rollout video (`~/isaac-rl/teacher_grasp_h264.mp4`)
  visually confirms the reach→close→lift. The STOCK reference reward worked once **four stacked bugs**
  were cleared (blind actor · free-lift `minimal_height` · 7cm EE offset · cube < closed gripper) — see A3.
- **Code is a minimal documented diff from the reference Franka lift.** Speculative additions we piled on
  while chasing the symptom (grasp term, DR, obs-norm) were reverted; `grasp_rewards.py` + `diag_grasp_geom.py`
  stay on disk for when we re-layer grasp/DR *deliberately*. Diff = Trossen robot + +y geometry +
  `minimal_height=0.09` + GPU buffers + teacher/student obs split + `std_type="log"`.
- **Next — pick one:** (A8) re-add DR now that the baseline grasps · (A4) vision student / depth-CNN distill ·
  or **(Thread B) pivot to the adaptive thesis** (the actual research goal) using this proven Trossen scene.
- **Lessons banked (the day's real cost):** (1) watch the *rollout*, the reward number lies; (2) when a
  policy fails like a reward/exploration problem, FIRST check what the actor can observe — a blind actor
  mimics every other failure; (3) don't diverge from a proven reference without a reason.
- Adaptive integration parked: see
  [`docs/superpowers/specs/2026-06-17-cenic-isaac-adaptive-roadmap.md`](../../../docs/superpowers/specs/2026-06-17-cenic-isaac-adaptive-roadmap.md).

---

## Thread A — Trossen Stationary AI teacher/student RL testbed

Privileged-state **PPO teacher → depth-CNN student (distillation)** that makes the LEFT arm of the
bimanual Stationary AI rig pick up a cube, in Isaac Lab (2.3.2 + PhysX) inside the `isaaclab`
podman container.

- ☑ **A0 — Isaac Lab container.** Ubuntu-22.04 container `isaaclab`, image `localhost/isaac-rl:base`.
  Run flags mandatory: `--device nvidia.com/gpu=all --security-opt=label=disable`. (memory
  `reference_isaaclab_container`.)
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
  Held-back refinement if placement precision matters: a contact-gated grasp term (robosuite-style —
  see `docs/superpowers/reports/2026-06-17-cube-pickup-reward-function-research.md`).
- ☐ **A4 — Vision student (CNN distillation).** NOT STARTED. Add a `cam_high` depth `TiledCamera` +
  `images` obs group (NHWC→permute CHW), `RslRlCNNModelCfg`. obs_groups
  `{"student":["policy","images"], "teacher":["policy","privileged"]}`. (Plan:
  `docs/superpowers/plans/2026-06-16-trossen-cube-pickup-teacher-student.md`.)
- ☐ **A5 — Eval.** `eval.py` predicate success (grasped ∧ lift-height ∧ at-rest); teacher vs student.
- ☑ **A6 — Timelapse.** `render_timelapse.py` works (camera reframed to +y). Rendered 24 checkpoints
  2026-06-17 — which is how we *saw* the A3 reward bug (arm never grasps). Reward curves hid it; the
  render exposed it. Lesson banked: watch the rollout, not just the metrics.
- ☐ **A7 — Live viewer (REQUIRED).** Watch policies/scenes *live*, not via file renders. On
  Fedora+container: **(1) WebRTC livestream** (Isaac Sim streams viewport → host browser; preferred,
  no X11/Wayland fighting) or **(2) X11 passthrough** (mount X socket + `xhost +local:` + run
  non-`--headless`). Native window on Windows/Ubuntu. First-class per the Big Picture workflow.
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
`EE_TCP_OFFSET=(0.1561,0,0)` (link_6 local x) was derived from the USD; confirm visually in the
`ee_closeup` render / by whether the trained arm actually closes around the cube.

---

## Thread B — CENIC adaptive integration into Isaac Lab

Full deliverable list in
[`docs/superpowers/specs/2026-06-17-cenic-isaac-adaptive-roadmap.md`](../../../docs/superpowers/specs/2026-06-17-cenic-isaac-adaptive-roadmap.md).
Summary:

- ☐ **B0 — Preliminary adaptive-vs-fixed demo** (standalone Newton, fast evidence): import a Trossen
  arm USD via `ModelBuilder.add_usd`, build a stiff SDF insertion scene, work-precision sweep
  (reuse `scripts/rl/expts/v1_work_precision.py`). **Go/no-go for the Isaac spend.**
- ☐ **B1 — Isaac Lab Newton backend** stood up (Isaac Lab **3.0 Beta** / `isaaclab_newton` — NOT in
  the installed 2.3.2; this is a real migration, do it in an isolated env/branch).
- ☐ **B2 — CENIC into the backend** (reconcile the `newton-cenic` fork with 3.0 Beta's pinned Newton;
  swap `SolverMuJoCoCENIC` in).
- ☐ **B3 — Level B substep driver** (subclass `NewtonMJWarpManager`: `_build_solver`→CENIC,
  `_run_solver_substeps`→`step_dt`, `use_cuda_graph=False`; confirm `opt.timestep` isn't clobbered).
- ☐ **B4 — In-Isaac validation** (work-precision adaptive-vs-fixed on a stiff Isaac scene = M0 in target).
- ☐ **B5 — Future:** humanoid + LEAP hand dexterous scene; GR00T/Mimic data-amp on adaptive data.

---

## Infrastructure & hard-won facts (don't re-derive these)

- **NVIDIA driver (2026-06-17):** the open kernel module 580.159.03 throws **Xid 56 display-engine
  crashes under load** on the RTX 4070 Ti SUPER (known regression). Fixed by forcing the
  **closed/proprietary** module at the same 580.159.03 via RPM Fusion:
  `echo '%_without_kmod_nvidia_detect 1' > /etc/rpm/macros.nvidia-kmod`, swap to `akmod-nvidia`,
  remove the prebuilt open kmod, `akmods --force`, reboot. Verify closed via
  `/proc/driver/nvidia/version` (no "Open") and `modinfo nvidia | grep license` → `NVIDIA`.
- **Env geometry (2026-06-17):** the env was a WXAI single-arm copy with the wrong workspace. Fixed:
  cube `[0,0.25,0.05]` (left arm +y, on the rig tabletop), command ranges x(-0.1,0.1)/y(0.15,0.35)/
  z(0.08,0.25), reset range tightened to the +y band, dropped the foreign SeattleLabTable
  (`self.scene.table=None`), ground at z=0. ee_frame: `follower_left_ee_gripper_link` is NOT an
  articulation body → use `EE_LINK=follower_left_link_6` + `EE_TCP_OFFSET=(0.1561,0,0)`.
- **GPU contact buffers (2026-06-17):** at 2048 envs the bimanual rig overflows the default GPU
  collision buffers → contacts dropped → **cube tunnels through the table → ~88% object_dropping**.
  Fixed in the env cfg: `gpu_max_rigid_patch_count=2**20`, `gpu_total_aggregate_pairs_capacity=2**23`,
  `gpu_found_lost_aggregate_pairs_capacity=2**26`. (Default patch count 5*2**15=163840 was too small.)
- **Code gotchas (container + native):** launch `AppLauncher` before importing `isaaclab.*`; call
  `handle_deprecated_rsl_rl_cfg` after loading the rsl_rl cfg; `torch.no_grad` (not `inference_mode`)
  for multi-checkpoint rollouts; detect success via marker files (Kit eats stdout). Fedora ffmpeg
  lacks libx264 → re-encode mp4s in-container; view videos OFF this machine.
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
  manager re-broadcasts/clobbers CENIC's per-world `opt.timestep` between substeps (B3); 3.0 Beta
  migration risk to the working 2.3.2 Trossen setup (do it isolated).

## Command reference
```bash
# train teacher (corrected env): checkpoints every 50, writes train_done.json
podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python scripts/rl/trossen/train_teacher.py \
  --headless --num_envs 2048 --max_iterations 1500"
# scene-inspection render (geometry pre-flight, light load): 4 labeled PNGs
podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python scripts/rl/trossen/inspect_scene.py \
  --headless --enable_cameras --out_dir /isaac/scene_check"
# training timelapse (after checkpoints exist) -> re-encode H.264 -> view off-device
podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python scripts/rl/trossen/render_timelapse.py \
  --headless --enable_cameras --steps 120 --out /isaac/teacher_timelapse.mp4"
podman exec isaaclab bash -lc "ffmpeg -y -i /isaac/teacher_timelapse.mp4 -c:v libx264 \
  -pix_fmt yuv420p -movflags +faststart /isaac/teacher_timelapse_h264.mp4"
```
