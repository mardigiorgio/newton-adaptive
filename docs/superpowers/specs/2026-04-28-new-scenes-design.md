# New CENIC Scenes: Falling Cylinder, Falling Gripper, ANYmal Clutter

**Status:** approved (brainstorm)
**Date:** 2026-04-28

## Goal

Add three new scenes to the CENIC scene library to broaden the dynamics regimes covered by demos and benchmarks:

1. **Falling cylinder** — minimal free-body cylinder dropped onto the ground. Tests edge/rolling contact for a curved primitive (geometrically harder for CENIC than a sphere).
2. **Falling gripper** — parallel-jaw gripper carriage on a vertical prismatic rail, fingers closed and pinching a held box, all falling under gravity. Tests frictional grasping under acceleration plus a 1-DOF carriage constraint plus impact at the bottom.
3. **ANYmal walking through cube clutter** — ANYmal C with the existing pretrained walking policy, walking forward through a per-world random scatter of light cube primitives. Stress test for high-DOF articulation interacting with many small contacts.

A Franka scene was discussed and explicitly cut from this batch.

## Non-goals

- Multi-scene benchmark sweeps. The existing `scripts/bench/` runner currently only consumes `contact_objects`; extending it to compare across scenes is a separate design.
- New shared abstractions for the three scenes. Each scene is short enough that copy-tune is clearer than premature factoring.
- Modifying or replacing `scripts/control/cenic_step_anymal_walk.py`. The new ANYmal scene+demo replaces its role going forward, but we leave the legacy script in place.
- Block-on-rail as a standalone scene. The rail constraint lives only inside the falling-gripper scene.

## Architecture

Mirrors the existing `scripts/scenes/contact_objects.py` and `scripts/scenes/dish_rack.py` pattern:

- **Scene modules** under `scripts/scenes/<name>.py`: pure data, no CLI, no viewer. Each exposes:
  - `build_template() -> ModelBuilder`
  - `build_model(n_worlds) -> Model`
  - `build_model_randomized(n_worlds, seed=42) -> Model`
  - `make_solver(model, tol=..., dt_mode="per_world") -> SolverMuJoCoCENIC`
  - `make_fixed_solver(model) -> SolverMuJoCo`
  - Module constants: `DT_OUTER`, `TOL`, `LOG_EVERY`.
- **Demo modules** under `scripts/demos/<name>.py`: thin viewer wrappers using the canonical `solver.step_dt(...)` pattern from `CLAUDE.md`. Same CLI as `scripts/demos/contact_objects.py`: `--num-worlds`, `--num-steps`, `--headless`, `--fixed-dt`.
- **Benchmark integration**: scenes expose the standard API surface so `scripts/bench/` can consume them later, but this change does not modify any benchmark file.

## Scene specs

### `scripts/scenes/falling_cylinder.py`

- 1 dynamic body per world: `add_shape_cylinder(radius=0.05, half_height=0.10)`.
- Initial pose: z = 0.5 m, small fixed tilt (~15° about x) so it lands on its rim, not on a circular face.
- Materials: `ke=1e4, kd=200, mu=0.4, margin=5e-3`.
- `build_model_randomized`: per-world xy ∈ [-0.2, 0.2] m, z ∈ [0.4, 0.8] m, fully random orientation (Shoemake quaternion sampler, same helper signature as in `contact_objects.py`).
- Solver: `dt_inner_init=0.01, dt_inner_min=1e-6, dt_inner_max=0.01, tol=1e-3, nconmax=64, njmax=128`.

### `scripts/scenes/falling_gripper.py`

Geometry, all primitives:

- **Rail (static, body=-1)**: tall thin box `hx=0.05, hy=0.05, hz=0.6`, centered at `(0, -0.10, 0.6)` so it sits next to the carriage's vertical axis. Visual + reference geometry only — `collision_group` set so it does not collide with the carriage (matches the image, where the rail is a guide rather than a collider).
- **Carriage** ("wrist") body: small box `hx=0.04, hy=0.04, hz=0.04`, attached to world by a **vertical prismatic joint** with axis +z at xy = `(0, 0)`, limits [0, 1.0] m, starting at q = 0.8 m.
- **Finger bodies (×2)**: thin boxes `hx=0.005, hy=0.04, hz=0.05`, each attached to the carriage by a **prismatic joint** along ±x, mirrored. Joint targets driven by `target_ke`/`target_kd` to a fixed inward position so fingers maintain a constant pinch on the held box.
- **Held box** (free 6-DOF body): `hx=0.02, hy=0.02, hz=0.04`, initial position centered between fingers, held purely by friction + pinch normal force (no kinematic attachment).

Tuning notes:

- Solver: `dt_inner_init=0.005, dt_inner_min=1e-7, dt_inner_max=0.01, tol=1e-3, nconmax=128, njmax=256`. Tighter `dt_min` than the cylinder scene because pinch contact + free body interaction at the moment of rail-stop impact will require finer adaptation.
- Pinch friction: `mu=0.8` between fingers and held box (rubber-like). `mu=0.3` between any rail-vs-carriage incidental contact.
- `viewer.apply_forces` works on the carriage out of the box because it's an articulated body; no extra wiring.

`build_model_randomized`: per-world starting carriage q ∈ [0.6, 0.95] m, small per-world random tilt of the held box about its z axis (rotation only, position fixed between the fingers) so each world has a slightly different friction-cone configuration.

### `scripts/scenes/anymal_clutter.py`

Robot setup mirrors `scripts/control/cenic_step_anymal_walk.py`:

- URDF: `newton.utils.download_asset("anybotics_anymal_c") / "urdf" / "anymal.urdf"`.
- Base spawn: `xform=wp.transform(wp.vec3(0, 0, 0.62), wp.quat_from_axis_angle(wp.vec3(0,0,1), pi/2))`, floating, no self-collisions, fixed joints collapsed.
- Defaults: `default_joint_cfg.armature=0.06, limit_ke=1e3, limit_kd=10`. `default_shape_cfg.ke=5e4, kd=5e2, kf=1e3, mu=0.75`.
- Sphere foot fix: scales `r -> 2r` for `GeoType.SPHERE` shapes (same as the existing script).
- Initial joint pose: same `initial_q` dict (12 leg joints) as the existing script.
- Per-joint actuator gains: `target_ke=150, target_kd=5`.

Clutter:

- Per-world random scatter of **30 cube primitives**, side length uniform [5, 15] cm, mass 50 g, position uniformly sampled over a 3 m × 3 m floor patch in front of the robot's spawn (note: the spawn applies a 90° rotation about +z, so body-frame forward points in world +y; the scatter region is therefore `y ∈ [0.3, 3.3], x ∈ [-1.5, 1.5]`), z = side/2 (resting on ground).
- Random yaw orientation per cube.
- Materials: `ke=1e4, kd=200, mu=0.5`.
- Per-world RNG seeded by world index for reproducibility.

Solver:

- `tol=1e-3, dt_init=0.005, dt_min=2e-7, dt_max=0.02, njmax=200, nconmax=400, solver="newton", ls_parallel=False, ls_iterations=50`.
- `njmax`/`nconmax` bumped from the legacy script's 50/100 to absorb extra contacts from clutter.

API split:

- The scene module is **torch-free**: it builds the model and exposes solver factories, but the policy and observation pipeline live in the demo file. Benchmarks can therefore import the scene without pulling torch.
- The policy/obs/act helpers (`compute_obs_for_world`, `quat_rotate_inverse`, joint-index remap tables) move from `cenic_step_anymal_walk.py` into `scripts/demos/anymal_clutter.py`. The legacy script is left untouched.

`build_model_randomized` ≡ `build_model` for ANYmal because the clutter scatter is already per-world randomized inside `build_template`.

## Demo specs

All four demos (cylinder, gripper, anymal_clutter — three new files) use the canonical CLAUDE.md loop:

```python
while viewer.is_running():
    state_0, state_1 = solver.step_dt(
        DT_OUTER, state_0, state_1, control,
        apply_forces=viewer.apply_forces,
    )
    # demo-specific control update (only ANYmal uses this)
    t += DT_OUTER
    viewer.render(state_0, t)
```

CLI flags identical to `scripts/demos/contact_objects.py`:

- `--num-worlds N` (default 1)
- `--num-steps N` (0 = run until closed)
- `--headless`
- `--fixed-dt DT` (use `make_fixed_solver` with this inner dt instead of CENIC)

ANYmal demo additionally:

- Loads `policy = torch.jit.load(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt", map_location=torch_device)`.
- Computes per-world observation, runs the policy, writes `joint_target_pos` once per `DT_OUTER` boundary (zero-order hold inside `step_dt`).
- Forward command vector hard-coded to `[1.0, 0.0, 0.0]`.

Status logging in each demo: reuse the `_print_status` helper shape from `scripts/demos/contact_objects.py` (collapses to a summary grid for N>4, full per-world table for N≤4).

## Testing & validation

- **Smoke test each demo at N=1**: `uv run python -m scripts.demos.<name> --num-steps 200`. Expect: no crashes, sim_time advances monotonically, cylinder lands and stops, gripper carriage hits the rail floor with the box still pinched, ANYmal walks forward roughly through the clutter.
- **Smoke test each scene at N=4 randomized**: `--num-worlds 4 --num-steps 200`. Expect: per-world dt diverges (expected from non-associative GPU reductions), no rejected-everywhere warnings.
- **Headless data-collection check**: `--num-worlds 16 --headless --num-steps 500`. Expect: completes without OOM, FPS print at end.
- **Fixed-step parity check**: `--fixed-dt 0.002`. Expect: scene runs to completion under `SolverMuJoCo`; visual behavior qualitatively matches CENIC.

No new unit tests; these are scenes, not library code, and the existing `contact_objects` and `dish_rack` scenes don't have unit tests either.

## File inventory

New files:

- `scripts/scenes/falling_cylinder.py`
- `scripts/scenes/falling_gripper.py`
- `scripts/scenes/anymal_clutter.py`
- `scripts/demos/falling_cylinder.py`
- `scripts/demos/falling_gripper.py`
- `scripts/demos/anymal_clutter.py`

No files modified. Legacy `scripts/control/cenic_step_anymal_walk.py` left as-is.

## Implementation order

Ship-as-you-go, simple first:

1. `falling_cylinder` (scene + demo) — ~30 + ~80 LOC.
2. `falling_gripper` (scene + demo) — ~80 + ~80 LOC.
3. `anymal_clutter` (scene + demo) — ~120 + ~250 LOC.

Each step is verifiable on its own with the smoke tests above.
