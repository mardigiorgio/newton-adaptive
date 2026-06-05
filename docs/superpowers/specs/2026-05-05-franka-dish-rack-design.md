# Franka Dropping Mug, Fork, Spatula Into Dish Rack

**Status:** approved (brainstorm)
**Date:** 2026-05-05

## Goal

Add a CENIC scene where a Franka FR3 arm executes a scripted joint trajectory and releases three held kitchen items (mug, fork, spatula) into the LBM dish drying rack. The arm + held bodies illustrate adaptive stepping under arm-dynamics + impact + multi-body settling against a non-convex (SDF) rack.

This is the Franka scene that was discussed and cut from the 2026-04-28 new-scenes batch; revisited now that the pattern from `anymal_clutter` (visual-mesh + primitive-collision split) and `dish_rack` (SDF rack collision) are proven.

## Non-goals

- Real grasping physics (friction-pinch). Held objects are kinematically pinned to the gripper until release; this is a deliberate simplification that avoids the friction-tuning fragility seen in `falling_gripper`.
- Inverse kinematics, motion planning, or learned policy. Trajectory is a fixed keyframe sequence.
- Realistic gripper dynamics (compliance, force feedback). The Franka is essentially a kinematic puppet for its arm joints during the trajectory; finger joints use position targets so they visibly open at the right moment.
- Per-world distinct trajectories. All worlds run the same keyframes; only initial held-object yaw / final-pose offset jitters per world.
- Benchmark integration. The scene exposes the standard API surface so `scripts/bench/` can pick it up later, but this change does not modify any benchmark file.

## Architecture

Mirrors the standard scene pattern (`scripts/scenes/<name>.py` for pure data, `scripts/demos/<name>.py` for viewer):

- **`scripts/scenes/franka_dish_rack.py`** — torch-free scene module exposing:
  - `build_template() -> ModelBuilder`
  - `build_model(n_worlds) -> Model`
  - `build_model_randomized(n_worlds, seed=42) -> Model`
  - `make_solver(model, tol=..., dt_mode="per_world") -> SolverMuJoCoCENIC`
  - `make_fixed_solver(model) -> SolverMuJoCo`
  - Module constants: `DT_OUTER`, `TOL`, `LOG_EVERY`, plus the trajectory keyframe table and release timing constants so the demo can read them without duplicating values.
  - Public helper: `update_held_objects(model, state, sim_time, world_count)` which performs the kinematic pin + release logic. The demo calls this once per outer step; the scene module owns the timing and pose-following logic.
- **`scripts/demos/franka_dish_rack.py`** — thin viewer using the canonical `solver.step_dt` loop. Calls the scene's `update_held_objects` and joint-target update on each outer step, then steps the solver and renders. CLI: `--num-worlds`, `--num-steps`, `--headless`, `--tol`.
- **No new abstractions across scenes.** Even though the kinematic-pin trick may be reusable later, factor it out only if a second scene needs it.

## Geometry

All in world coordinates, Z-up.

### Dish rack (static, body=-1)
Reuse the LBM rack from `scripts/scenes/dish_rack.py` verbatim — base + wireframe + utensil_holder, SDF collision, attached at world origin. Copy the three `_RACK_*_GLTF` paths and the wireframe / utensil-holder weld transforms; do not refactor. The rack's top is at z ≈ 0.124 m and its xy footprint is roughly ±0.15 × ±0.22 m.

### Franka FR3 with hand (static base, articulated)
- URDF: `newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"`
- Base xform: `wp.transform(wp.vec3(0.55, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0,0,1), math.pi))` — base sits 55 cm to +x of the rack and faces back toward it (so the natural arm reach swings the EE over the rack).
- `floating=False`, `enable_self_collisions=False`, `collapse_fixed_joints=True`, `ignore_inertial_definitions=False`.
- Joints (verified from URDF): 7 arm revolute joints `fr3_joint1..7` plus 2 prismatic finger joints `fr3_finger_joint1`, `fr3_finger_joint2`. Total 9 coords / 9 dofs.
- Per-joint actuator gains: `target_ke=600, target_kd=30` for arm joints; `target_ke=400, target_kd=20` for finger joints. Gains chosen to track keyframe targets crisply without overshoot at the relatively low control rate (`DT_OUTER=2 ms`).

### Held objects (free 6-DOF bodies)
Each starts kinematically pinned at a fixed offset from the EE wrist (`fr3_link8`) until release.

| Object  | Asset                                                        | Visual         | Collision                        | Density      |
|---------|--------------------------------------------------------------|----------------|----------------------------------|--------------|
| Mug     | `assets/lbm/mugs/.../mug_inomata_..._mesh_collision.gltf`    | LBM glTF mesh  | LBM mesh **convex hull** (16 vert) | LBM-provided |
| Fork    | `assets/lbm/forks/.../cambridge_jubilee_..._fork.gltf`       | LBM glTF mesh  | LBM mesh **convex hull** (16 vert) | LBM-provided |
| Spatula | `assets/033_spatula_berkeley_meshes/033_spatula/tsdf/textured.obj` | YCB textured mesh | **Box primitive** matching mesh bbox | 80 kg/m^3 |

The spatula uses the visual-mesh + primitive-collision split (same trick as `anymal_clutter`) because it is essentially a flat slab — a primitive box collision is both faster and more accurate than a convex hull of a thin scanned mesh.

The mug and fork keep mesh collision because (a) they are already wired up that way in `dish_rack` and that is known to work, (b) their convex hulls capped at 16 vertices are reasonable approximations of cup and cutlery silhouettes, (c) they need to nest into the wire cage and primitive proxies would be too coarse for the rack interior. SDF collision on the held bodies is unnecessary; `add_shape_mesh` defaults to convex-hull collision and that is the right choice here.

Mug / fork mass + inertia: parsed from the LBM `.sdf` files (same loader as `dish_rack`). Spatula: ShapeConfig.density drives mass + inertia consistently.

## Holding & release

While `sim_time < RELEASE_TIME`:
- For each held body, set its joint_q (free-joint 7-tuple) explicitly each outer step to:
  - position: `EE_pose * pin_offset_pos[i]`
  - orientation: `EE_pose.quat * pin_offset_quat[i]`
- Set its joint_qd (free-joint spatial velocity 6-tuple) to zero. Held bodies are not dynamically integrated; the solver's contact resolution will not affect them because their pose is overwritten before each step.

After `sim_time >= RELEASE_TIME`:
- Stop overriding pose / velocity. Each body becomes a normal free-falling object.
- Velocity at release: implicit from the previous-step pin update. Because the kinematic update sets joint_qd to zero each step, the released velocity is approximately zero. This is acceptable: the trajectory ends with the EE held still over the rack for ~200 ms before release, so a near-zero release velocity is physically correct anyway.

`RELEASE_TIME` is chosen so it fires ~200 ms after the gripper begins opening (gripper open command at t=1.5s, release at t=1.7s). This 200 ms lag prevents the bodies from clipping the still-closing fingers as they part.

Per-object pin offsets are positioned along the gripper's local +y axis (or whichever axis matches finger separation in the URDF — verify at implementation time) so they sit between the fingers, axis-aligned with the gripper:
- Mug: centered, axis-aligned, -2 cm below tool-center-point so the rim points up.
- Fork: centered, long axis aligned with gripper +z, slightly to one side.
- Spatula: centered, long axis aligned with gripper +y, slightly to other side. Held flat.

The exact pin offsets are tuned at implementation time using a single-step FK probe — the spec only fixes the strategy.

## Trajectory

Joint-space keyframes for the 7 arm joints + 2 finger joints. Linear interpolation between keyframes with a smoothstep time mapping (cubic ease-in/ease-out) so the arm doesn't snap.

Sketch of keyframe schedule (refined at implementation time using IK probes; values below are the design target):

| t [s] | Phase                | Arm pose                                                | Finger pos | Notes                                |
|-------|----------------------|---------------------------------------------------------|------------|--------------------------------------|
| 0.0   | Carrying             | EE positioned 30 cm above rack center, gripper down     | 0.0 (closed) | Held objects pinned in gripper     |
| 1.5   | Above-rack-still     | EE at same position (small inward joint adjustments OK) | 0.0 (closed) | Begin gripper opening at this frame |
| 1.7   | Releasing            | EE position unchanged                                   | 0.04 (open) | RELEASE_TIME — pin override stops   |
| 4.0   | Hold                 | EE position unchanged                                   | 0.04 (open) | Objects fall + settle in rack       |

The arm joint values for each keyframe are determined by a one-shot probe script that runs IK against a target EE pose (over the rack center, gripper down) and prints the resulting joint angles. The implementation plan includes this probe as its first task; the printed angles are then hardcoded as a NumPy table in the scene module. No runtime IK.

## Solver

```python
SolverMuJoCoCENIC(
    model,
    tol=1e-3,           # configurable via --tol
    dt_inner_init=0.005,
    dt_inner_min=1e-6,
    dt_inner_max=0.01,
    dt_mode="per_world",
    nconmax=2048,       # rack SDF + 3 mesh objects + Franka self/external
    njmax=8192,
    cone="elliptic",
    iterations=100,
    impratio=10.0,
    ccd_iterations=100,
)
```

`DT_OUTER = 0.002` (2 ms) so trajectory updates are fine-grained enough for the keyframe sweep.

## Per-world randomization

`build_model_randomized` jitters per world:
- Held-object yaw inside the gripper: ±15° random rotation per object about the gripper's pin axis. Causes objects to enter the rack at slightly different angles.
- Final-pose lateral offset: ±5 cm random offset on the EE keyframe at t=1.5s. Causes per-world divergence in landing position so the worlds don't all stack identically.

Same per-world seed semantics as `dish_rack` and `anymal_clutter`.

## Demo loop (`scripts/demos/franka_dish_rack.py`)

```python
while viewer.is_running():
    # Update Franka joint targets for this outer step (interpolated from keyframes).
    update_franka_targets(control, sim_time, ...)

    # Pin held objects to gripper (or release them, depending on sim_time).
    update_held_objects(model, state_0, sim_time, world_count)

    # Physics: one DT_OUTER worth of CENIC inner steps.
    state_0, state_1 = solver.step_dt(
        DT_OUTER, state_0, state_1, control,
        apply_forces=viewer.apply_forces,
    )
    sim_time += DT_OUTER

    if step % LOG_EVERY == 0:
        _print_status(solver, step)

    viewer.begin_frame(sim_time)
    viewer.log_state(state_0)
    viewer.end_frame()
```

CLI flags identical to `scripts/demos/anymal_clutter.py`:
- `--num-worlds N` (default 1)
- `--num-steps N` (default 0 = run forever)
- `--headless`
- `--tol FLOAT` (default unspecified — uses scene's `TOL`)

Status logging: same `_print_status` shape as the other demos (collapses for N>4, full per-world table for N≤4).

## Testing & validation

- **Smoke test N=1**: `uv run python -m scripts.demos.franka_dish_rack --num-worlds 1 --num-steps 2000 --headless` (4 s sim time at DT_OUTER=2 ms). Expect: completes without exception, no NaN, all three objects end up inside the rack footprint (z < 0.13 m, |x| < 0.16 m, |y| < 0.23 m).
- **Smoke test N=4 randomized**: `--num-worlds 4 --num-steps 2000 --headless`. Expect: completes; per-world dt diverges; objects land in the rack with slight per-world variation.
- **Visual sanity check (interactive)**: user runs without `--headless`, watches one full trajectory; the arm should swing smoothly, fingers open at t=1.5s, objects drop straight down (~20 cm) into the rack, settle within ~1 s.

No new unit tests; consistent with how the other CENIC scenes (`contact_objects`, `dish_rack`, `falling_*`, `anymal_clutter`) ship.

## File inventory

New files:
- `scripts/scenes/franka_dish_rack.py`
- `scripts/demos/franka_dish_rack.py`
- (Spec) `docs/superpowers/specs/2026-05-05-franka-dish-rack-design.md`
- (Plan) `docs/superpowers/plans/2026-05-05-franka-dish-rack.md` (created by writing-plans)

No files modified.

## Risks

1. **Pin offsets misaligned with gripper frame** — gripper's local +y / +z axis may not match what I assume above; a single FK probe at the start of the implementation plan settles this in 5 minutes. If the URDF EE frame is different than expected, only the pin-offset table changes.
2. **Released bodies clip the rack on landing** — if the trajectory keyframe places the EE too far from rack center, objects miss. Iteration on the keyframe table during smoke testing.
3. **Mug / fork convex hulls don't fit the rack interior cleanly** — known-good in `dish_rack`, but the held bodies will arrive at a slightly different angle and starting pose; might need to pre-rotate the pin to match the natural settled orientation. Tunable.
4. **Franka self-collision disabled** — could let the held objects clip the gripper fingers visually during the close phase. If visible, narrow the held-object pin offsets (move them further from the fingertip).
