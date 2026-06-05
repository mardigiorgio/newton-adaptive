# Franka Farm — 100-arm adaptive-dt hero figure

**Date:** 2026-06-01
**Status:** Approved (design)

## Goal

Produce one headless-rendered PNG for the poster: a 10x10 grid of Franka FR3
arms running the existing cube-stacking pick-and-place task in parallel, with
each arm/world tinted by its live per-world inner timestep `dt`
(red = small step / stiff contact, green = large step / free motion). Angled
hero camera, front rows sharp.

The figure visualizes the paper's core contribution: per-world adaptive
stepping across many parallel worlds. A frozen frame shows a spread of `dt`
across the grid because worlds desync (randomized cubes, independent FSM
advance, the paper's own trajectory-divergence effect).

## Non-goals

- Not a video/animation (poster is static). The coloring helper is written so
  a future per-frame video path is trivial, but that is out of scope here.
- Not a new physics task. We reuse the upstream cube-stacking task verbatim.
- Not a benchmark or accuracy measurement. This is figure tooling.

## Source of reuse

`newton/examples/ik/example_ik_cube_stacking.py` already provides:

- Multi-world Franka FR3 + table + randomized cubes, `--world-count` scalable.
- Grid layout via `viewer.set_world_offsets(wp.vec3(1.5, 1.5, 0.0))`. The grid
  is `ceil(sqrt(N))` per side (see `newton/_src/utils/__init__.py::compute_world_offsets`),
  so `N=100` gives a 10x10 grid spanning ~13.5 m at 1.5 m spacing.
- Per-world IK (`ik.IKSolver`, `n_problems=world_count`) + a 9-phase pick/place
  FSM (APPROACH -> ... -> HOME), randomized cube pose/orientation/density/color.
- `test_final()` success check.

It runs on the fixed-step `SolverMuJoCo`. We swap in `SolverMuJoCoAdaptive`.

## Verified facts (already checked against the codebase)

- `SolverMuJoCoAdaptive` subclasses `SolverMuJoCo`, so it inherits
  `register_custom_attributes` including `mujoco:gravcomp` /
  `mujoco:jnt_actgravcomp` (the IK tolerances depend on gravity compensation).
- `solver.dt` -> `wp.array[world_count]`, float32, on device (current per-world
  inner dt).
- `model.shape_world` -> per-shape world index (used to map shapes to worlds).
- `viewer.update_shape_colors({shape_idx: (r,g,b)})` recolors shapes
  (`newton/_src/viewer/viewer.py:1428`).
- `ViewerGL(headless=True)` renders to an offscreen FBO;
  `ViewerGL.get_frame(target_image=None) -> wp.array(h, w, 3) uint8`
  (`newton/_src/viewer/viewer_gl.py:1039`).
- `viewer.set_camera(pos: wp.vec3, pitch: float, yaw: float)`.
- Pillow (`PIL`) is available -> PNG write with no new dependency.

## Architecture

All new code lives under `scripts/`. No edits to vendored Newton examples.

### `scripts/demos/franka_farm.py`

`class FrankaFarm(Example)` subclasses the upstream `Example`, reusing the
entire scene build / IK / FSM, overriding three seams:

1. `capture_sim(self)` -> set `self.graph = None`. The adaptive `solver.step`
   does a host-side boundary-flag sync per inner iteration and cannot be
   CUDA-graph captured. IK graph capture (`capture_ik`) is left intact.

2. `simulate(self)` -> a single canonical CENIC step per frame:
   `self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)`.
   Updates `state_0` in place (no buffer swap). FSM/IK already run once per
   frame in `set_joint_targets()` -> matches the required per-DT-boundary
   control update.

3. In `__init__`, call `super().__init__()` then replace `self.solver` with:
   ```python
   newton.solvers.SolverMuJoCoAdaptive(
       self.model,
       tol=TOL,                 # default 1e-3
       dt_init=0.005,
       dt_min=1e-6,
       dt_max=self.frame_dt,    # ~0.0167 s; inner step never exceeds outer
       use_mujoco_contacts=True,
       nconmax=1000, njmax=2000,  # CLI-tunable; matches the upstream example
       cone="elliptic",
       impratio=1000.0,
   )
   ```
   The base ctor briefly builds the fixed solver; we discard it. Minor waste,
   chosen over editing vendored Newton to keep a clean code boundary.
   `dt_min < dt_init <= dt_max` per CLAUDE.md.

`render(self)` is overridden to call `_apply_dt_colors()` (when enabled) then
`begin_frame(sim_time)` / `log_state(state_0)` / `end_frame()`. Contacts: use
`self.solver.contacts` if logging contacts; for the still we may skip contact
rendering (decided during impl by what looks best).

### `_apply_dt_colors(self)` helper

- Read `self.solver.dt.numpy()` once (outside any inner loop — same call-site
  class as `_print_status`; respects the no-sync-in-hot-path rule).
- Map each world's dt to RGB via a red->green ramp on `log(dt)` normalized over
  `[dt_min, dt_max]` (clamped). Small dt -> red, large dt -> green.
- Build the shape->world map once (cached) from `model.shape_world.numpy()`.
- `viewer.update_shape_colors({shape: rgb})` for every dynamic shape (arm +
  cubes) in each world; ground stays neutral.

### Capture flow (in `main()`)

1. Build `ViewerGL(headless=True, width, height)`; `FrankaFarm(viewer, world_count=100, args)`.
2. Set the angled hero camera (aimed at grid center ~ (6.75, 6.75, 0) for 10x10
   @1.5 m). Camera fully CLI-tunable.
3. Step `--frames` frames (default ~120, ~2 s sim) so arms are mid pick/lift/place
   across varied phases.
4. `_apply_dt_colors()`, one `render()`, then `get_frame().numpy()` ->
   `PIL.Image.fromarray(...)` -> save to `--out`.

### CLI arguments

- `--world-count` (default 100)
- `--frames` (default 120)
- `--out` (default `franka_farm.png`)
- `--width` / `--height` (default 2560 x 1440)
- `--cam-pos` / `--cam-pitch` / `--cam-yaw` (angled hero defaults)
- `--tol` (default 1e-3), `--nconmax` (default 1000), `--njmax` (default 2000)
- `--no-color` (disable dt coloring, plain render — debugging / comparison)

## Coloring scope decision

Color **all** per-world dynamic shapes (arm links + gripper + cubes) by dt.
This makes each grid tile read as a single dt "heat" cell — strongest visual.
The cubes' original red/green/blue identity is sacrificed; acceptable for this
figure. `--no-color` restores the stock colored render.

## Testing

Smoke test under `scripts/tests/` (e.g. extend `test_scenes.py` or a new
`test_franka_farm.py`):

- Run `FrankaFarm` with `world_count=4`, headless, ~5 frames.
- Assert the PNG is written, has the expected dimensions, and is not a single
  flat color (verifies render + coloring actually executed).

Kept light because this is figure tooling, not library code.

## Risks / watch items

- `nconmax` / `njmax` sizing at N=100 — start modest, expose as flags, bump if
  contacts overflow.
- Franka URDF collision meshes under `use_mujoco_contacts=True` route through
  mjwarp's convex collision (standard SolverMuJoCo path).
- Slower than the fixed example (no sim graph + per-frame host syncs), but we
  only need ~2 s of sim for a still.
- Camera framing for an angled hero shot may need a couple of iterations; that
  is why all camera params are CLI args (re-grab without code edits).

## Out of scope / follow-ups

- Video/GIF loop of the farm (the coloring helper already supports per-frame
  use).
- A matching "fixed-step penetration vs adaptive" close-up (separate figure).
