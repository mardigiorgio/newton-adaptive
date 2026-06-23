# Still-Hold (Jitter) Reward Fix — Implementation & Exploration Guide

> **⚠ SUPERSEDED (2026-06-21).** The shipped jitter fix is an **actuator-gain correction**
> (`trossen_cube/assets/stationary_ai.py`: `left_arm` stiffness=80 / damping=4, replacing the
> ~500×-too-stiff USD-baked gains), **not** the reward-space levers below. The `stillness_reward.py`
> prescribed here was never created, and the reward block is still the stock Franka 4-term reward.
> The reward levers (`action_rate`/`joint_vel` curriculum, entropy anneal — still wired as
> `train_teacher.py --joint_vel_weight/--action_rate_weight/--entropy_coef`) remain documented here as
> **RESERVE** remedies if the gain fix alone does not settle the hold after the (pending) native
> retrain. Companion paths in this doc reference the old container layout (`/isaac`, podman); see the
> native layout in `../README.md` and `trossen_cube/paths.py`. Archived as history.

**Self-contained handoff for a fresh chat / context window.** Goal: make the trained teacher hold the
cube **still** once it's on-goal, instead of the shaky limit-cycle it currently settles into. You can
execute this guide cold — it pins every file, value, command, and gotcha you need.

Date written: 2026-06-18. Owner workstream: Trossen Stationary AI single-arm cube lift (Thread A).
Companion docs (read if you need more): [`ROADMAP.md`](ROADMAP.md) (full tracker),
[`IMPL_GROUND_TRUTH.md`](IMPL_GROUND_TRUTH.md) (pinned USD/Isaac values).

---

## 0. One-line problem statement

The rendered rollout uses the **deterministic mean action** (`get_inference_policy`), so the shake is in
the policy's mean output, not an exploration-noise artifact. The teacher grasps, lifts, and carries the
cube to the goal, then **jitters in place** instead of holding still.

---

## 1. The diagnosis (the WHY — do not skip)

Ranked by contribution:

1. **PRIMARY — the reward is flat in "stillness space."** Once the cube is on-goal the policy already
   banks ~31 reward (`object_goal_tracking` +16, `lifting_object` +15). The only terms that penalize
   motion (`action_rate_l2`, `joint_vel_l2`) top out near **−1e-2** even after their curriculum — three
   orders of magnitude smaller. So a perfectly still hold and a visibly shaky hold score within rounding
   of each other. **PPO has no gradient toward a static pose**; it converged to exactly what the reward
   describes ("cube near goal"). This is why more iterations don't help and why `fine_grained` plateaued
   then drifted — there was nothing to climb.

2. **SECONDARY — the exploration floor.** `entropy_coef=0.006` is fixed for the whole run, so the mean
   μ(s) was fit under a policy that always explores → a rough function of the obs. Annealing entropy
   sharpens μ, **but only toward whatever optimum exists** — and per (1) there is no still optimum to
   sharpen toward. So entropy annealing alone does nothing; it's a finishing pass *after* (1).

3. **AMPLIFIERS that keep residual motion alive:** the velocity + last-action feedback loop with
   *relative* joint-position actions (settles into a deterministic limit cycle); the binary gripper
   toggling open/close near its threshold; possibly underdamped USD-baked PD gains.

**Consequence you must accept:** fixing this **requires changing the reward**. The flatness is not a bug
we introduced — it *is* the stock Franka lift reward (Franka holds are shaky too). This is a *motivated*
divergence from the reference (diagnosed cause), unlike the speculative additions that were reverted
earlier. Keep it deliberate and documented.

---

## 2. Current reward (what you're modifying)

Inherited unchanged from Isaac Lab `LiftEnvCfg` (the Franka reference); the env only overrides
`minimal_height`. Source of truth: `isaaclab_tasks .../manipulation/lift/lift_env_cfg.py` (`RewardsCfg`
+ `CurriculumCfg`), and our overrides in
[`trossen_cube/tasks/cube_lift/cube_lift_env_cfg.py`](trossen_cube/tasks/cube_lift/cube_lift_env_cfg.py).

| term | func | weight | params |
|---|---|---|---|
| `reaching_object` | `object_ee_distance` | 1.0 | std 0.1 |
| `lifting_object` | `object_is_lifted` | 15.0 | minimal_height **0.09** |
| `object_goal_tracking` | `object_goal_distance` | 16.0 | std 0.3, minimal_height 0.09 |
| `object_goal_tracking_fine_grained` | `object_goal_distance` | 5.0 | std 0.05, minimal_height 0.09 |
| `action_rate` | `action_rate_l2` | −1e-4 → **−1e-1** (curriculum @ num_steps=10000) | — |
| `joint_vel` | `joint_vel_l2` | −1e-4 → **−1e-1** (curriculum @ num_steps=10000) | asset=robot |

Timing: `decimation=2`, `sim.dt=0.01` → **control dt 0.02 s**, episode 5 s (≈250 control steps).
`gamma=0.98` → ~1 s credit horizon. `entropy_coef=0.006`. `num_envs` validated at **2048**.

---

## 3. The fix — three levers, in causal order

### Lever 1 (PRIMARY): give the reward a stillness gradient

Two variants. **Try 1A first** (most faithful, smallest change); escalate to 1B only if 1A makes the arm
sluggish / under-reach.

#### 1A — strengthen the existing motion penalties (recalibrate, keep Franka structure)

The penalties already exist; they're just ~10-100x too weak to compete with the +31 task reward. Bump the
**terminal curriculum weight** so the pressure only bites the converged hold (early reach learning sits
behind the curriculum and is untouched). In `StationaryAiCubeLiftEnvCfg.__post_init__` (after
`super().__post_init__()`):

```python
# Jitter fix 1A: the stock action_rate/joint_vel penalties top out ~-1e-2 vs a +31 task reward, so a
# still hold and a shaky hold score the same -> no gradient toward stillness. Strengthen the curriculum
# TERMINAL weight (it ramps at num_steps=10000; early reach learning is unaffected). Start 10x, tune.
self.curriculum.action_rate.params["weight"] = -1.0   # was -1e-1
self.curriculum.joint_vel.params["weight"] = -1.0     # was -1e-1
```

- **Verify the curriculum term names first** (`action_rate`, `joint_vel`) by grepping the reference
  `CurriculumCfg` in `lift_env_cfg.py`. If named differently, adjust.
- **Tune:** −1.0 (10x) is the starting probe. If still shaky, go −2.0/−5.0. If the arm gets sluggish or
  reach regresses (`Episode_Reward/reaching_object` drops, slower grasps), back off → escalate to 1B.
- Optional: also raise the *base* weights (`self.rewards.action_rate.weight`, `self.rewards.joint_vel.weight`)
  so some pressure exists before the curriculum kicks in — usually unnecessary.

#### 1B — on-goal-gated stillness term (surgical; use if 1A trades away reach speed)

A penalty that is **zero during reach/lift/carry** and only pays for low joint velocity **once the cube is
lifted and near the goal** — so it can't slow the approach, only the hold. New mdp function + reward term.

Create `trossen_cube/tasks/cube_lift/stillness_reward.py`:

```python
"""On-goal-gated stillness reward: pay for a quiet arm only once the cube is lifted and near the goal,
so reach/lift/carry (where motion is needed) pay nothing. Fixes the 'reward is flat in stillness space'
cause of the still-hold jitter -- see JITTER_FIX_GUIDE.md."""
from __future__ import annotations
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms


def still_when_on_goal(
    env: ManagerBasedRLEnv,
    on_goal_radius: float,        # [m] how close to the goal counts as "on goal"
    minimal_height: float,        # cube must be lifted above this
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["follower_left_joint_[0-5]"]),
) -> torch.Tensor:
    """Reward in [0, 1]: 1 - tanh(sum joint_vel^2), gated by (lifted AND cube within radius of goal)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    cmd = env.command_manager.get_command(command_name)           # desired pose in robot-root frame
    des_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], cmd[:, :3])
    dist = torch.norm(des_w - obj.data.root_pos_w[:, :3], dim=1)
    on_goal = (obj.data.root_pos_w[:, 2] > minimal_height) & (dist < on_goal_radius)
    vel = torch.sum(torch.square(robot.data.joint_vel[:, arm_cfg.joint_ids]), dim=1)
    return on_goal.float() * (1.0 - torch.tanh(vel))
```

Wire it in `__post_init__` (import `from trossen_cube.tasks.cube_lift import stillness_reward` and
`from isaaclab.managers import RewardTermCfg as RewTerm`):

```python
# Jitter fix 1B: a still optimum the policy can actually climb, gated to the hold phase only.
self.rewards.hold_still = RewTerm(
    func=stillness_reward.still_when_on_goal,
    weight=5.0,   # tune; comparable to fine_grained's 5.0 so stillness is worth pursuing
    params={"on_goal_radius": 0.05, "minimal_height": 0.09, "command_name": "object_pose"},
)
```

- **Tune** `weight` (start 5.0) and `on_goal_radius` (start 0.05 m = the fine_grained tolerance).
- This is a new custom term (less faithful than 1A) but it's the principled fix; it directly creates the
  static-pose optimum the diagnosis says is missing, without taxing the reach.

### Lever 2 (AMPLIFIER): damp the limit cycle physically (no reward change)

The velocity/last-action feedback loop sustains residual motion. Adding joint **damping** suppresses it
and is reward-faithful. The arm actuators currently inherit USD-baked PD (`stiffness=None, damping=None`
in [`trossen_cube/assets/stationary_ai.py`](trossen_cube/assets/stationary_ai.py)).

1. **First dump the baked gains** (so you raise damping from a known base, and keep stiffness):
   add a print to a probe (or extend `reach_probe.py`): `robot.data.joint_stiffness`,
   `robot.data.joint_damping` for the `follower_left_joint_[0-5]` ids.
2. Override in `__post_init__` (localizes the change to this task):
   ```python
   # Jitter fix 2: raise arm damping to kill the velocity-feedback limit cycle (keep baked stiffness).
   self.scene.robot.actuators["left_arm"].stiffness = <baked_stiffness>   # from the dump
   self.scene.robot.actuators["left_arm"].damping   = <baked_damping * 1.5..3.0>   # tune up
   ```
   Too much damping → laggy/slow arm; tune from the rollout.
3. **Gripper latch (optional):** the binary gripper toggles near its threshold. Simplest mitigation is to
   keep it commanded closed once the cube is grasped; lower priority than 1+2.

### Lever 3 (SECONDARY): entropy anneal — now it has a target

After Lever 1 creates a still optimum, sharpen μ toward it with a **continuation run** (the flag is
already wired in `train_teacher.py`):

```bash
scripts/rl/trossen/docker/run.sh scripts/rl/trossen/train_teacher.py --headless \
  --num_envs 2048 --max_iterations 800 --entropy_coef 0.001 \
  --resume_from /isaac/logs/trossen/stationary_ai_lift_teacher/model_<best>.pt
```

On its own this does nothing (no still optimum); only run it after Lever 1 is in.

---

## 4. How to run + verify each experiment

**Run a fresh training (Levers 1/2 are env-cfg changes → fresh run):**
```bash
# ARCHIVE the previous run first so you don't clobber it:
mv ~/isaac-rl/logs/trossen/stationary_ai_lift_teacher \
   ~/isaac-rl/logs/trossen/stationary_ai_lift_teacher.<label>
# then:
scripts/rl/trossen/docker/run.sh scripts/rl/trossen/train_teacher.py --headless \
  --num_envs 2048 --max_iterations 1500 > ~/isaac-rl/full_train_<label>.log 2>&1
```
Run it backgrounded/harness-tracked; ~2-4 h. Watch boot for `Learning iteration 0/1500` and
`object_dropping ≈ 0` (physics sane).

**Objective stillness metric (no eyeballing needed):** the penalty magnitudes ARE the metric. A stiller
policy makes `Episode_Reward/joint_vel` and `Episode_Reward/action_rate` **closer to 0**. Plot them:
```bash
uv run --with matplotlib scripts/rl/trossen/plot_reward_curves.py \
  ~/isaac-rl/full_train_<label>.log -o ~/isaac-rl/curves_<label>.png
```
Compare `joint_vel`/`action_rate` (less negative = stiller) and `fine_grained` (should climb if the hold
tightens) against the baseline log.

**Visual confirm (the shake is in the mean, so the render shows the truth):**
```bash
scripts/rl/trossen/docker/run.sh scripts/rl/trossen/render_timelapse.py --headless --enable_cameras \
  --ckpts <iter> --steps 200 --out /isaac/hold_<label>.mp4 \
  --log_dir /isaac/logs/trossen/stationary_ai_lift_teacher.<label>
```
Then extract a few **consecutive late frames** (hold phase) and compare them — near-identical consecutive
frames = still; differing = chatter. (See how `cmp_frames` was built earlier with ffmpeg `select=eq(n,..)`.)

**Fair A/B caveat:** the goal target is randomized per episode, so two rollouts draw different (easy/hard)
targets. For a clean chatter comparison, render both policies on the **same fixed command** (set a fixed
`object_pose` target, or seed identically) — otherwise an easy low target looks "stiller" for free.

---

## 5. Suggested experiment order

1. **Baseline**: confirm the current no-rails policy's `joint_vel`/`action_rate` curve + a hold clip (the
   reference to beat). Current shaky checkpoint to compare against: `…stationary_ai_lift_teacher.rails_v1/model_3498.pt`.
2. **Exp A = Lever 1A** (−1.0 weights). Train, plot, watch the hold. Did `joint_vel` get less negative
   without `reaching` collapsing?
3. If A is sluggish/under-reaching → **Exp B = Lever 1B** (gated term, revert 1A).
4. **Exp C = best of A/B + Lever 2** (damping). Train, compare.
5. **Exp D = Lever 3** entropy-anneal continuation from the best of C.
6. Keep one variable per run; log what you changed in `ROADMAP.md`.

---

## 6. Pinned facts / gotchas (so you don't relearn them)

- **Container:** Isaac Lab runs only in the `isaaclab` podman container. Always launch via
  `scripts/rl/trossen/docker/run.sh <script> <args>` (it `podman exec`s into it). Non-python: `podman exec
  isaaclab bash -lc "<cmd>"`. Host paths `~/isaac-rl/...` map to `/isaac/...` inside.
- **Env count:** 2048 is validated; 4096 is unverified (contact-drop is silent → shows as `object_dropping`).
- **GPU buffers** in the env cfg are sized for 1-2k envs (`gpu_max_rigid_patch_count=2**20`, etc.). Don't lower.
- **Reward = stock Franka lift** + only `minimal_height` 0.04→0.09 (cube rests at ~0.048 on the raised
  tabletop, so 0.04 = free lift = reward gaming). Keep that override.
- **Scene = no-rails** (`stationary_ai_norails.usda`, generated by `make_norails_usd.py`): the rig's
  `frame_link` rail/gantry collision is deactivated + hidden so the policy can't jam the cube against it.
- **Spawn workspace** is being finalized to the arm's reachable footprint (`reach_map.py` /
  `plot_reach_map.py`; the FK map needs clipping to the table extent). It's continuous uniform random
  already — independent of the jitter work, so don't block on it.
- **Render uses the deterministic mean** (`get_inference_policy`) — so a render IS a faithful stillness
  test; don't dismiss jitter as sampling noise.
- **PPO is a SYMMETRIC privileged teacher**: actor AND critic both see `["policy","privileged"]`. Do NOT
  "simplify" the actor to `["policy"]` — that blinds it to the cube (the single biggest past bug).
- **gamma=0.98** (~1 s horizon) and **obs_normalization=False** are logged as *separate* controlled A/Bs
  in `ROADMAP.md` — don't conflate them with the jitter work; change one variable at a time.

---

## 7. File map

| file | role |
|---|---|
| `trossen_cube/tasks/cube_lift/cube_lift_env_cfg.py` | env: rewards/curriculum overrides, scene, workspace, robot |
| `trossen_cube/tasks/cube_lift/agents/rsl_rl_ppo_cfg.py` | PPO cfg: entropy_coef, gamma, obs_groups, obs_norm |
| `trossen_cube/assets/stationary_ai.py` | articulation cfg: arm actuators (damping lives here) |
| `train_teacher.py` | training entrypoint (`--num_envs`, `--max_iterations`, `--resume_from`, `--entropy_coef`) |
| `render_timelapse.py` | rollout render (`--ckpts <iters>` to pick checkpoints) |
| `plot_reward_curves.py` | per-term reward curves from one or more logs |
| `docker/run.sh` | container launcher wrapper |
| `ROADMAP.md` / `IMPL_GROUND_TRUTH.md` | tracker / pinned values |

---

## 8. Done = ?

A rollout where, once the cube is on-goal, consecutive frames are near-identical (no visible shake), with
`Episode_Reward/joint_vel` and `action_rate` measurably closer to 0 than the baseline and `reaching`/
`lifting`/`goal_tracking` not regressed. That's a still, faithful hold.
```
