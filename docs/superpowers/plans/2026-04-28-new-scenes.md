# New CENIC Scenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three new scenes to the CENIC scene library — `falling_cylinder`, `falling_gripper`, `anymal_clutter` — each with a paired interactive demo following the canonical `step_dt` loop pattern.

**Architecture:** Each scene is a pure-data module under `scripts/scenes/<name>.py` exposing the standard scene API (`build_template`, `build_model`, `build_model_randomized`, `make_solver`, `make_fixed_solver`). Each is paired with a thin viewer demo under `scripts/demos/<name>.py`. The ANYmal scene module stays torch-free; the policy lives in the demo. No new abstractions or shared helpers — copy-tune from existing patterns (`contact_objects.py`, `dish_rack.py`).

**Tech Stack:** Python 3.10+, warp, newton, numpy, torch (ANYmal demo only), matplotlib (none here)

**Spec:** `docs/superpowers/specs/2026-04-28-new-scenes-design.md`

**Convention:** No `git add` / `git commit` steps in tasks — the user commits all changes themselves at phase boundaries.

---

## File Inventory

New files (6):

- `scripts/scenes/falling_cylinder.py`
- `scripts/scenes/falling_gripper.py`
- `scripts/scenes/anymal_clutter.py`
- `scripts/demos/falling_cylinder.py`
- `scripts/demos/falling_gripper.py`
- `scripts/demos/anymal_clutter.py`

No files modified.

---

## Phase 1: Falling cylinder

### Task 1: Create `scripts/scenes/falling_cylinder.py`

**Files:**
- Create: `scripts/scenes/falling_cylinder.py`

- [ ] **Step 1: Write the scene module**

Create `scripts/scenes/falling_cylinder.py` with this exact content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Falling cylinder scene: a single cylinder dropped onto the ground.

Tests CENIC adaptive stepping during edge-rolling contact for a curved
primitive (geometrically harder than a sphere because the contact point
sweeps along an edge as the cylinder rolls).

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01  # 100 Hz control / render cadence [s]
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

CYL_RADIUS = 0.05
CYL_HALF_HEIGHT = 0.10
DROP_Z_DEFAULT = 0.5
TILT_RAD = math.radians(15.0)


def _tilt_quat(angle_rad: float) -> wp.quat:
    """Rotation about +x by angle_rad so the cylinder lands on its rim."""
    return wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle_rad)


def build_template() -> newton.ModelBuilder:
    """Single-world template: one tilted cylinder dropped above the ground."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.4, margin=5e-3)
    body = template.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, DROP_Z_DEFAULT), q=_tilt_quat(TILT_RAD)),
    )
    template.add_shape_cylinder(
        body, radius=CYL_RADIUS, half_height=CYL_HALF_HEIGHT, cfg=cfg,
    )
    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    return builder.finalize()


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    """Shoemake's uniform random quaternion."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (s1 * math.sin(a1), s1 * math.cos(a1), s2 * math.sin(a2), s2 * math.cos(a2))


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds with per-world random xy, z, and orientation. Seeded per world."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    body_q_np = model.body_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        for b in range(bodies_per_world):
            x = rng.uniform(-0.2, 0.2)
            y = rng.uniform(-0.2, 0.2)
            z = rng.uniform(0.4, 0.8)
            qx, qy, qz, qw = _random_unit_quaternion(rng)

            base = w * coords_per_world + b * 7
            joint_q_np[base + 0] = x
            joint_q_np[base + 1] = y
            joint_q_np[base + 2] = z
            joint_q_np[base + 3] = qx
            joint_q_np[base + 4] = qy
            joint_q_np[base + 5] = qz
            joint_q_np[base + 6] = qw

            body_idx = w * bodies_per_world + b
            body_q_np[body_idx] = (x, y, z, qx, qy, qz, qw)

    model.joint_q.assign(joint_q_np)
    model.body_q.assign(body_q_np)
    return model


def make_solver(
    model: newton.Model,
    tol: float = TOL,
    dt_mode: str = "per_world",
) -> newton.solvers.SolverMuJoCoCENIC:
    """CENIC solver with falling-cylinder defaults."""
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode=dt_mode,
        nconmax=64,
        njmax=128,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=64, njmax=128,
    )
```

- [ ] **Step 2: Verify the scene builds at the Python level**

Run:
```bash
uv run python -c "from scripts.scenes.falling_cylinder import build_model, make_solver; m = build_model(1); s = make_solver(m); print('OK', m.body_count, 'bodies')"
```

Expected output (last line): `OK 1 bodies`

If you get an `ImportError` for `add_shape_cylinder`, confirm the symbol via:
```bash
uv run python -c "import newton; b = newton.ModelBuilder(); print(hasattr(b, 'add_shape_cylinder'))"
```
Expected: `True`

If false, switch the call in step 1 to `add_shape_capsule` with the same `radius` and `half_height` (capsule is the closest available primitive in older Newton revisions).

- [ ] **Step 3: Verify randomization produces distinct worlds**

Run:
```bash
uv run python -c "
from scripts.scenes.falling_cylinder import build_model_randomized
m = build_model_randomized(4)
import numpy as np
q = m.joint_q.numpy().reshape(4, 7)
print('positions z:', q[:, 2])
print('all distinct:', len(set(q[:, 2].tolist())) == 4)
"
```
Expected: 4 different z values in [0.4, 0.8], `all distinct: True`.

---

### Task 2: Create `scripts/demos/falling_cylinder.py`

**Files:**
- Create: `scripts/demos/falling_cylinder.py`

- [ ] **Step 1: Write the demo module**

Create `scripts/demos/falling_cylinder.py` with this exact content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive falling-cylinder demo using the CENIC adaptive solver.

Drops a tilted cylinder (or N randomized cylinders, one per world) onto
a ground plane. Supports CENIC adaptive stepping and fixed-dt stepping
for comparison.

Usage::

    uv run python -m scripts.demos.falling_cylinder [--num-worlds N] [--headless] [--fixed-dt DT]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.falling_cylinder import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_fixed_solver,
    make_solver,
)

_grid_lines = 0


def _print_status(solver, step):
    global _grid_lines

    n = solver.model.world_count

    if n > 4:
        s = solver.get_status_summary()
        lines = [
            f"  step {step}  tol={solver._tol:.1e}  worlds={n}",
            f"  sim_time  [{s['sim_time_min']:.4f}, {s['sim_time_max']:.4f}] s",
            f"  dt        [{s['dt_min']:.6f}, {s['dt_max']:.6f}] s",
            f"  err_max   {s['error_max']:.3e}",
            f"  accepted  {s['accept_count']}/{n}",
        ]
    else:
        sim_times = solver.sim_time.numpy()
        dts = solver.dt.numpy()
        errors = solver.last_error.numpy()
        accepted = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'L2 error':>{col}}{'status':>{col}}"
        lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
        for i in range(len(sim_times)):
            lines.append(
                f"{'world ' + str(i):>{col}}"
                f"{sim_times[i]:>{col}.4f}"
                f"{dts[i]:>{col}.6f}"
                f"{errors[i]:>{col}.3e}"
                f"{'ok' if accepted[i] else 'REJECT':>{col}}"
            )
        lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=0, help="0 = run until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-dt", type=float, default=None,
        help="Use fixed-step SolverMuJoCo with this dt instead of CENIC",
    )
    args = parser.parse_args()

    model = build_model_randomized(args.num_worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    use_fixed = args.fixed_dt is not None

    if use_fixed:
        solver = make_fixed_solver(model)
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step demo: {args.num_worlds} world(s)  solver=SolverMuJoCo  "
            f"dt={args.fixed_dt:.4e}  substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model)
        print(
            f"CENIC cylinder demo: {args.num_worlds} world(s)  solver=SolverMuJoCoCENIC  "
            f"tol={solver._tol:.1e}  dt_inner_init={solver._dt.numpy()[0]:.4f}  "
            f"dt_inner_max={solver._dt_max:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(1.5, -1.5, 0.8), pitch=-20.0, yaw=135.0)

    contacts = newton.Contacts(
        rigid_contact_max=solver.mjw_data.naconmax if not use_fixed else 64,
        soft_contact_max=0,
        requested_attributes={"force"},
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        if use_fixed:
            for _ in range(n_inner):
                state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                state_0, state_1 = state_1, state_0
        else:
            state_0, state_1 = solver.step_dt(
                DT_OUTER, state_0, state_1, control,
                apply_forces=viewer.apply_forces,
            )
        t += DT_OUTER
        step += 1

        if not use_fixed and step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        if viewer.show_contacts:
            solver.update_contacts(contacts, state_0)
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test, headless, N=1**

Run:
```bash
uv run python -m scripts.demos.falling_cylinder --num-worlds 1 --num-steps 200 --headless
```

Expected:
- No exceptions.
- Final line of output looks like `200 steps  2.000 s sim  X.XX s wall  YYY.Y fps`.
- The intermediate status lines (every 250 steps — none expected at 200 steps) may or may not print.

- [ ] **Step 3: Smoke test at N=4 randomized, headless**

Run:
```bash
uv run python -m scripts.demos.falling_cylinder --num-worlds 4 --num-steps 200 --headless
```

Expected:
- Completes without errors.
- A status grid prints with 4 distinct sim_time / dt values per world.
- No "REJECT" rows on every world for every status print (occasional rejects are fine; all-rejects across multiple prints means tol is too tight or initial conditions are too aggressive).

- [ ] **Step 4: Visual sanity check (interactive)**

Run:
```bash
uv run python -m scripts.demos.falling_cylinder --num-worlds 1
```

Expected: a viewer window opens, a tilted cylinder falls and lands on its rim, then settles or rolls. Close the window to exit.

If the cylinder phases through the ground, increase `margin` in `cfg` from `5e-3` to `1e-2` in `scripts/scenes/falling_cylinder.py`.

If the cylinder explodes (flies off-screen), reduce `ke` in `cfg` from `1e4` to `5e3`.

---

## Phase 2: Falling gripper

### Task 3: Create `scripts/scenes/falling_gripper.py`

**Files:**
- Create: `scripts/scenes/falling_gripper.py`

- [ ] **Step 1: Write the scene module**

Create `scripts/scenes/falling_gripper.py` with this exact content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Falling gripper scene: parallel-jaw gripper on a vertical rail, holding a box.

Geometry (all primitives):
  * Rail (static, body=-1): tall thin box, visual + reference geometry only.
    Filtered out of carriage collision so the rail does not push the carriage.
  * Carriage ("wrist"): small box on a vertical prismatic joint to world.
  * Two finger boxes: thin boxes attached to the carriage by mirrored
    prismatic joints along ±x. Joint targets driven by target_ke / target_kd
    so the fingers maintain a constant pinch on the held box.
  * Held box (free 6-DOF body): clamped between the fingers by friction +
    pinch normal force, no kinematic attachment.

Tests frictional grasping under acceleration (gravity), prismatic joint
dynamics, and impact at the rail bottom.

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01
TOL = 1e-3
DT_INNER_MIN = 1e-7
LOG_EVERY = 250

# --- Geometry [m] -----------------------------------------------------------

RAIL_HX = 0.05
RAIL_HY = 0.05
RAIL_HZ = 0.6
RAIL_CENTER = wp.vec3(0.0, -0.10, RAIL_HZ)  # base on ground

CARRIAGE_HALF = 0.04  # cube body
CARRIAGE_Q_INIT = 0.8
CARRIAGE_Q_MIN = 0.0
CARRIAGE_Q_MAX = 1.0

FINGER_HX = 0.005
FINGER_HY = 0.04
FINGER_HZ = 0.05
# Outer face of carriage at ±CARRIAGE_HALF; finger inner face begins inset by
# this amount so the finger surface contacts the held box.
FINGER_X_OPEN = CARRIAGE_HALF + 0.030  # joint q at rest (open position)
FINGER_X_CLOSED = CARRIAGE_HALF + 0.020  # target (squeezes against held box)

HELD_HX = 0.02
HELD_HY = 0.02
HELD_HZ = 0.04

# Pinch controller gains: stiff enough to hold against gravity and impact,
# soft enough not to explode the held box. dampratio kd/(2 sqrt(ke)) ~ 1.0.
FINGER_TARGET_KE = 5e3
FINGER_TARGET_KD = 1.4e2

# Collision groups: rail (-2) is filtered from everything; carriage / fingers
# / held box share group +1 so they collide with each other and the ground.
GROUP_RAIL = -2
GROUP_DYN = 1


def build_template() -> newton.ModelBuilder:
    """Single-world template: rail + carriage + 2 fingers + held box."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    cfg_rail = newton.ModelBuilder.ShapeConfig(
        ke=1e4, kd=200, mu=0.0, margin=5e-3, collision_group=GROUP_RAIL,
    )
    cfg_carriage = newton.ModelBuilder.ShapeConfig(
        ke=1e4, kd=200, mu=0.3, margin=5e-3, collision_group=GROUP_DYN,
    )
    cfg_finger = newton.ModelBuilder.ShapeConfig(
        ke=1e4, kd=200, mu=0.8, margin=5e-3, collision_group=GROUP_DYN,
    )
    cfg_held = newton.ModelBuilder.ShapeConfig(
        ke=1e4, kd=200, mu=0.8, margin=5e-3, collision_group=GROUP_DYN,
    )

    # Rail: visual / reference only.
    template.add_shape_box(
        body=-1,
        xform=wp.transform(p=RAIL_CENTER, q=wp.quat_identity()),
        hx=RAIL_HX, hy=RAIL_HY, hz=RAIL_HZ, cfg=cfg_rail,
    )

    # Carriage on a prismatic joint to world (axis +z).
    carriage = template.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, CARRIAGE_Q_INIT), q=wp.quat_identity()),
    )
    template.add_shape_box(
        carriage,
        hx=CARRIAGE_HALF, hy=CARRIAGE_HALF, hz=CARRIAGE_HALF,
        cfg=cfg_carriage,
    )
    template.add_joint_prismatic(
        parent=-1, child=carriage,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(0.0, 0.0, 1.0),
        limit_lower=CARRIAGE_Q_MIN, limit_upper=CARRIAGE_Q_MAX,
    )

    # Fingers on prismatic joints to the carriage (axis ±x).
    for sign in (+1.0, -1.0):
        finger = template.add_body(
            xform=wp.transform(
                p=wp.vec3(sign * FINGER_X_OPEN, 0.0, CARRIAGE_Q_INIT),
                q=wp.quat_identity(),
            ),
        )
        template.add_shape_box(
            finger, hx=FINGER_HX, hy=FINGER_HY, hz=FINGER_HZ, cfg=cfg_finger,
        )
        template.add_joint_prismatic(
            parent=carriage, child=finger,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(sign, 0.0, 0.0),
            limit_lower=0.0, limit_upper=FINGER_X_OPEN + 0.05,
            target=FINGER_X_CLOSED,
            target_ke=FINGER_TARGET_KE,
            target_kd=FINGER_TARGET_KD,
            mode=newton.JointMode.TARGET_POSITION,
        )

    # Held box: free body initially centered between the fingers.
    held = template.add_body(
        xform=wp.transform(p=wp.vec3(0.0, 0.0, CARRIAGE_Q_INIT), q=wp.quat_identity()),
    )
    template.add_shape_box(
        held, hx=HELD_HX, hy=HELD_HY, hz=HELD_HZ, cfg=cfg_held,
    )

    return template


def build_model(n_worlds: int) -> newton.Model:
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    return builder.finalize()


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds. Randomizes carriage starting q and held-box yaw per world."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds

    # Joint layout per world (matches build_template):
    #   [0]      carriage prismatic q
    #   [1]      finger +x prismatic q
    #   [2]      finger -x prismatic q
    #   [3..9]   held box free-joint (3 pos + 4 quat)
    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        base = w * coords_per_world

        carriage_q = rng.uniform(0.6, 0.95)
        joint_q_np[base + 0] = carriage_q

        # Held box xyz follow carriage; yaw is randomized.
        yaw = rng.uniform(-math.pi, math.pi)
        cz = math.cos(yaw / 2.0)
        sz = math.sin(yaw / 2.0)
        joint_q_np[base + 3] = 0.0
        joint_q_np[base + 4] = 0.0
        joint_q_np[base + 5] = carriage_q
        joint_q_np[base + 6] = 0.0
        joint_q_np[base + 7] = 0.0
        joint_q_np[base + 8] = sz
        joint_q_np[base + 9] = cz

    model.joint_q.assign(joint_q_np)

    # body_q follows from forward kinematics; rebuild it via newton.eval_fk.
    state = model.state()
    state.joint_q.assign(joint_q_np)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    model.body_q.assign(state.body_q.numpy())

    return model


def make_solver(
    model: newton.Model,
    tol: float = TOL,
    dt_mode: str = "per_world",
) -> newton.solvers.SolverMuJoCoCENIC:
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=0.005,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode=dt_mode,
        nconmax=128,
        njmax=256,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=128, njmax=256,
    )
```

- [ ] **Step 2: Verify the scene builds**

Run:
```bash
uv run python -c "
from scripts.scenes.falling_gripper import build_model, make_solver
m = build_model(1)
s = make_solver(m)
print('OK', m.body_count, 'bodies', m.joint_coord_count, 'joint coords')
"
```
Expected: `OK 4 bodies 10 joint coords` (carriage + 2 fingers + held box; 1+1+1+7 = 10 coords).

If `add_joint_prismatic` complains about the `target`/`target_ke`/`target_kd` kwargs, switch to passing them via `JointDofConfig`:
```python
dof_cfg = newton.ModelBuilder.JointDofConfig(
    target=FINGER_X_CLOSED, target_ke=FINGER_TARGET_KE, target_kd=FINGER_TARGET_KD,
    limit_lower=0.0, limit_upper=FINGER_X_OPEN + 0.05,
    mode=newton.JointMode.TARGET_POSITION,
)
template.add_joint_prismatic(parent=..., child=..., axis=..., dof_cfg=dof_cfg)
```

If the symbol `newton.JointMode.TARGET_POSITION` does not exist, search Newton's joint module for the equivalent enum value:
```bash
uv run python -c "import newton; print([x for x in dir(newton) if 'Mode' in x or 'Joint' in x])"
```
Use whatever enum corresponds to "PD position controller" (likely `newton.JointMode.TARGET_POSITION` or `newton.JointMode.POSITION`).

- [ ] **Step 3: Verify state initializes without nan**

Run:
```bash
uv run python -c "
from scripts.scenes.falling_gripper import build_model_randomized
import numpy as np
m = build_model_randomized(2)
q = m.joint_q.numpy()
print('joint_q:', q)
print('any nan:', np.any(np.isnan(q)))
"
```
Expected: 20 finite floats (10 per world × 2 worlds), `any nan: False`.

---

### Task 4: Create `scripts/demos/falling_gripper.py`

**Files:**
- Create: `scripts/demos/falling_gripper.py`

- [ ] **Step 1: Write the demo**

Create `scripts/demos/falling_gripper.py` with this exact content (this is identical in structure to `scripts/demos/falling_cylinder.py` — only the imports and camera angle change):

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive falling-gripper demo using the CENIC adaptive solver.

A parallel-jaw gripper carriage on a vertical rail, fingers closed on a
held box, falls under gravity and impacts the rail bottom / ground.

Usage::

    uv run python -m scripts.demos.falling_gripper [--num-worlds N] [--headless] [--fixed-dt DT]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.falling_gripper import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_fixed_solver,
    make_solver,
)

_grid_lines = 0


def _print_status(solver, step):
    global _grid_lines

    n = solver.model.world_count

    if n > 4:
        s = solver.get_status_summary()
        lines = [
            f"  step {step}  tol={solver._tol:.1e}  worlds={n}",
            f"  sim_time  [{s['sim_time_min']:.4f}, {s['sim_time_max']:.4f}] s",
            f"  dt        [{s['dt_min']:.6f}, {s['dt_max']:.6f}] s",
            f"  err_max   {s['error_max']:.3e}",
            f"  accepted  {s['accept_count']}/{n}",
        ]
    else:
        sim_times = solver.sim_time.numpy()
        dts = solver.dt.numpy()
        errors = solver.last_error.numpy()
        accepted = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'L2 error':>{col}}{'status':>{col}}"
        lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
        for i in range(len(sim_times)):
            lines.append(
                f"{'world ' + str(i):>{col}}"
                f"{sim_times[i]:>{col}.4f}"
                f"{dts[i]:>{col}.6f}"
                f"{errors[i]:>{col}.3e}"
                f"{'ok' if accepted[i] else 'REJECT':>{col}}"
            )
        lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-dt", type=float, default=None)
    args = parser.parse_args()

    model = build_model_randomized(args.num_worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    use_fixed = args.fixed_dt is not None

    if use_fixed:
        solver = make_fixed_solver(model)
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step gripper demo: {args.num_worlds} world(s)  dt={args.fixed_dt:.4e}  "
            f"substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model)
        print(
            f"CENIC gripper demo: {args.num_worlds} world(s)  tol={solver._tol:.1e}  "
            f"dt_inner_init={solver._dt.numpy()[0]:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(1.2, -1.2, 0.7), pitch=-15.0, yaw=135.0)

    contacts = newton.Contacts(
        rigid_contact_max=solver.mjw_data.naconmax if not use_fixed else 128,
        soft_contact_max=0,
        requested_attributes={"force"},
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        if use_fixed:
            for _ in range(n_inner):
                state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                state_0, state_1 = state_1, state_0
        else:
            state_0, state_1 = solver.step_dt(
                DT_OUTER, state_0, state_1, control,
                apply_forces=viewer.apply_forces,
            )
        t += DT_OUTER
        step += 1

        if not use_fixed and step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        if viewer.show_contacts:
            solver.update_contacts(contacts, state_0)
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test, headless, N=1**

Run:
```bash
uv run python -m scripts.demos.falling_gripper --num-worlds 1 --num-steps 200 --headless
```

Expected:
- No exceptions, final FPS line prints.

- [ ] **Step 3: Tune the pinch if the held box slips**

Run interactive:
```bash
uv run python -m scripts.demos.falling_gripper --num-worlds 1
```

Watch what happens to the held box during the fall.

- If the held box slips out of the fingers and falls separately:
  - First try increasing `mu` on `cfg_finger` and `cfg_held` from `0.8` to `1.2`.
  - If still slipping, raise `FINGER_TARGET_KE` from `5e3` to `1e4` (and `FINGER_TARGET_KD` from `1.4e2` to `2.0e2` to keep dampratio ~1).
- If the held box explodes outward (fingers overshoot the closed target and squeeze too hard):
  - Reduce `FINGER_TARGET_KE` to `2e3` and `FINGER_TARGET_KD` to `9e1`.
  - Increase `FINGER_X_CLOSED` from `CARRIAGE_HALF + 0.020` to `CARRIAGE_HALF + 0.022` (less overlap with held box).

Re-run the smoke test (step 2) after any retune to make sure it still completes.

- [ ] **Step 4: Smoke test at N=4 randomized**

Run:
```bash
uv run python -m scripts.demos.falling_gripper --num-worlds 4 --num-steps 200 --headless
```

Expected: completes; per-world dt diverges slightly; held box stays pinched in the majority of worlds (per-world divergence is expected).

---

## Phase 3: ANYmal walking through clutter

### Task 5: Create `scripts/scenes/anymal_clutter.py`

**Files:**
- Create: `scripts/scenes/anymal_clutter.py`

- [ ] **Step 1: Write the scene module (torch-free)**

Create `scripts/scenes/anymal_clutter.py` with this exact content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""ANYmal C walking through random cube clutter.

Reproduces the robot setup from scripts/control/cenic_step_anymal_walk.py
(URDF, joint config, foot-sphere fix, initial pose, actuator gains) and
adds a per-world random scatter of light cube primitives that the robot
walks through.

The scene module is intentionally torch-free; the walking policy and
observation pipeline live in scripts/demos/anymal_clutter.py so this
module can be imported by benchmarks without pulling torch.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.solvers
import newton.utils
from newton import GeoType

DT_OUTER = 0.002  # 2 ms — policy / control cadence [s]
TOL = 1e-3
LOG_EVERY = 5

# --- Clutter parameters -----------------------------------------------------

CLUTTER_COUNT = 30
CLUTTER_SIZE_MIN = 0.05
CLUTTER_SIZE_MAX = 0.15
CLUTTER_MASS = 0.05  # 50 g per cube

# Robot spawns at origin with a 90° z rotation; body-frame +x (forward) maps
# to world +y. Scatter the clutter in front of the robot in world coords.
CLUTTER_Y_MIN = 0.3
CLUTTER_Y_MAX = 3.3
CLUTTER_X_MIN = -1.5
CLUTTER_X_MAX = 1.5

# --- Robot defaults ---------------------------------------------------------

INITIAL_Q = {
    "RH_HAA": 0.0, "RH_HFE": -0.4, "RH_KFE": 0.8,
    "LH_HAA": 0.0, "LH_HFE": -0.4, "LH_KFE": 0.8,
    "RF_HAA": 0.0, "RF_HFE": 0.4, "RF_KFE": -0.8,
    "LF_HAA": 0.0, "LF_HFE": 0.4, "LF_KFE": -0.8,
}


def _add_clutter(builder: newton.ModelBuilder, rng) -> None:
    """Scatter CLUTTER_COUNT cubes over the floor patch in front of the robot."""
    cfg = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.5, margin=5e-3)
    for _ in range(CLUTTER_COUNT):
        side = rng.uniform(CLUTTER_SIZE_MIN, CLUTTER_SIZE_MAX)
        half = side / 2.0
        x = rng.uniform(CLUTTER_X_MIN, CLUTTER_X_MAX)
        y = rng.uniform(CLUTTER_Y_MIN, CLUTTER_Y_MAX)
        z = half
        yaw = rng.uniform(-math.pi, math.pi)
        cz = math.cos(yaw / 2.0)
        sz = math.sin(yaw / 2.0)

        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat(0.0, 0.0, sz, cz)),
            mass=CLUTTER_MASS,
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=cfg)


def build_template(seed: int = 42) -> newton.ModelBuilder:
    """ANYmal C robot + per-world cube scatter (this template is one world)."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    template.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1,
    )
    template.default_shape_cfg.ke = 5.0e4
    template.default_shape_cfg.kd = 5.0e2
    template.default_shape_cfg.kf = 1.0e3
    template.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    template.add_urdf(
        str(asset_path / "urdf" / "anymal.urdf"),
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.62),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi * 0.5),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )

    # Foot spheres in this URDF have a tiny radius; scale by 2x so the robot
    # stands on its feet rather than its calves.
    for i in range(len(template.shape_type)):
        if template.shape_type[i] == GeoType.SPHERE:
            r = template.shape_scale[i][0]
            template.shape_scale[i] = (r * 2.0, 0.0, 0.0)

    # Set initial joint pose (offset by 6 because of the floating-base 6 DOFs
    # at the head of joint_q).
    for name, value in INITIAL_Q.items():
        idx = next(
            (i for i, lbl in enumerate(template.joint_label) if lbl.endswith(f"/{name}")),
            None,
        )
        if idx is None:
            raise ValueError(f"Joint '{name}' not found in ANYmal URDF")
        template.joint_q[idx + 6] = value

    # PD gains for joint targets.
    for i in range(len(template.joint_target_ke)):
        template.joint_target_ke[i] = 150
        template.joint_target_kd[i] = 5

    # Clutter (per-template — each replicated world will share the same scatter
    # pattern unless build_model_randomized regenerates it; see below).
    rng = np.random.default_rng(seed)
    _add_clutter(template, rng)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane. All worlds share the same clutter."""
    template = build_template(seed=42)
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()
    return builder.finalize()


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds with per-world distinct clutter scatter.

    Builds each world from a freshly seeded template so every world has a
    different cube layout, then concatenates via a top-level builder.
    """
    builder = newton.ModelBuilder()
    for w in range(n_worlds):
        per_world = build_template(seed=seed + w)
        builder.replicate(per_world, 1)
    builder.add_ground_plane()
    return builder.finalize()


def make_solver(
    model: newton.Model,
    tol: float = TOL,
    dt_mode: str = "per_world",
) -> newton.solvers.SolverMuJoCoCENIC:
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=0.005,
        dt_inner_min=2e-7,
        dt_inner_max=0.02,
        dt_mode=dt_mode,
        njmax=200,
        nconmax=400,
        solver="newton",
        ls_parallel=False,
        ls_iterations=50,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=True,
        njmax=200,
        nconmax=400,
        solver="newton",
        ls_parallel=False,
        ls_iterations=50,
    )
```

- [ ] **Step 2: Verify the scene builds (no torch import)**

Run:
```bash
uv run python -c "
from scripts.scenes.anymal_clutter import build_model, make_solver
m = build_model(1)
s = make_solver(m)
print('OK', m.body_count, 'bodies', m.joint_coord_count, 'coords')
"
```
Expected: `OK <some number> bodies <some number> coords`. The body count includes the robot's links (typically 13 after `collapse_fixed_joints`) plus 30 cubes ≈ 43 bodies for N=1.

- [ ] **Step 3: Verify scene module is torch-free**

Run:
```bash
uv run python -c "
import sys
import scripts.scenes.anymal_clutter  # noqa
print('torch loaded:', 'torch' in sys.modules)
"
```
Expected: `torch loaded: False`. If True, find the offending import and remove it.

- [ ] **Step 4: Verify per-world clutter is distinct in randomized model**

Run:
```bash
uv run python -c "
from scripts.scenes.anymal_clutter import build_model_randomized
m = build_model_randomized(2)
import numpy as np
q = m.joint_q.numpy()
coords_per_world = m.joint_coord_count // 2
# Take the first cube's xy in each world (after the robot's joints).
# The exact offset depends on URDF parsing, so just confirm the two halves
# of joint_q are not identical.
print('worlds identical:', np.allclose(q[:coords_per_world], q[coords_per_world:]))
"
```
Expected: `worlds identical: False`.

---

### Task 6: Create `scripts/demos/anymal_clutter.py`

**Files:**
- Create: `scripts/demos/anymal_clutter.py`

- [ ] **Step 1: Write the demo (loads torch policy, uses step_dt)**

Create `scripts/demos/anymal_clutter.py` with this exact content. The policy / observation helpers are copied from `scripts/control/cenic_step_anymal_walk.py` and adapted to use the canonical `step_dt` loop pattern:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive ANYmal-walking-through-clutter demo using the CENIC adaptive solver.

Runs the pretrained ANYmal walking policy with a forward command of 1.0 m/s
through a per-world random scatter of light cube primitives.

Usage::

    uv run python -m scripts.demos.anymal_clutter [--num-worlds N] [--headless] [--num-steps N]
"""

import argparse
import sys
import time

import numpy as np
import torch
import warp as wp

import newton
import newton.solvers
import newton.utils

from scripts.scenes.anymal_clutter import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_solver,
)

# Joint index remapping between lab convention and MuJoCo convention.
LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

_grid_lines = 0


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def compute_obs_for_world(
    actions_w, state, joint_pos_initial, torch_device, lab_indices, gravity_vec,
    command, q_offset, qd_offset, coords_per_world, dofs_per_world,
):
    q = q_offset
    qd = qd_offset
    root_quat = torch.tensor(state.joint_q[q + 3:q + 7], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel = torch.tensor(state.joint_qd[qd:qd + 3], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel = torch.tensor(state.joint_qd[qd + 3:qd + 6], device=torch_device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.tensor(
        state.joint_q[q + 7:q + coords_per_world],
        device=torch_device, dtype=torch.float32,
    ).unsqueeze(0)
    joint_vel = torch.tensor(
        state.joint_qd[qd + 6:qd + dofs_per_world],
        device=torch_device, dtype=torch.float32,
    ).unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat, root_lin_vel)
    ang_vel_b = quat_rotate_inverse(root_quat, root_ang_vel)
    grav = quat_rotate_inverse(root_quat, gravity_vec)
    joint_pos_rel = torch.index_select(joint_pos - joint_pos_initial, 1, lab_indices)
    joint_vel_rel = torch.index_select(joint_vel, 1, lab_indices)
    return torch.cat([vel_b, ang_vel_b, grav, command, joint_pos_rel, joint_vel_rel, actions_w], dim=1)


def _print_status(solver, step):
    global _grid_lines
    sim_times = solver.sim_time.numpy()
    dts = solver.dt.numpy()
    errors = solver.last_error.numpy()
    accepted = solver.accepted.numpy()
    n = len(sim_times)

    col = 16
    bar = "+" + ("-" * col + "+") * 5
    hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'RMS error':>{col}}{'status':>{col}}"
    lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
    for i in range(min(n, 8)):
        lines.append(
            f"{'world ' + str(i):>{col}}{sim_times[i]:>{col}.4f}{dts[i]:>{col}.6f}"
            f"{errors[i]:>{col}.3e}{'ok' if accepted[i] else 'REJECT':>{col}}"
        )
    lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    device = wp.get_device()
    torch_device = wp.device_to_torch(device)

    model = build_model_randomized(args.num_worlds)
    coords_per_world = model.joint_coord_count // args.num_worlds
    dofs_per_world = model.joint_dof_count // args.num_worlds

    solver = make_solver(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    policy = torch.jit.load(
        str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt"),
        map_location=torch_device,
    )

    joint_pos_initial = torch.tensor(
        state_0.joint_q[7:coords_per_world],
        device=torch_device, dtype=torch.float32,
    ).unsqueeze(0)

    actions = torch.zeros(args.num_worlds, 12, device=torch_device, dtype=torch.float32)
    lab_indices = torch.tensor(LAB_TO_MUJOCO, device=torch_device)
    mujoco_indices = torch.tensor(MUJOCO_TO_LAB, device=torch_device)
    gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
    command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
    command[0, 0] = 1.0

    all_targets = torch.zeros(
        args.num_worlds * dofs_per_world, device=torch_device, dtype=torch.float32,
    )

    print(
        f"CENIC ANYmal-clutter demo: {args.num_worlds} world(s)  tol={solver._tol:.1e}  "
        f"coords/world={coords_per_world}  dofs/world={dofs_per_world}",
        flush=True,
    )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(2.5, -2.0, 1.4), pitch=-15.0, yaw=120.0)

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        # Policy update on the DT_OUTER boundary (zero-order hold).
        with torch.no_grad():
            for w in range(args.num_worlds):
                q_offset = w * coords_per_world
                qd_offset = w * dofs_per_world
                obs_w = compute_obs_for_world(
                    actions[w:w + 1], state_0, joint_pos_initial, torch_device,
                    lab_indices, gravity_vec, command, q_offset, qd_offset,
                    coords_per_world, dofs_per_world,
                )
                act_w = policy(obs_w)
                actions[w] = act_w[0]
                rearranged = torch.gather(act_w, 1, mujoco_indices.unsqueeze(0))
                targets = joint_pos_initial + 0.5 * rearranged
                all_targets[w * dofs_per_world:(w + 1) * dofs_per_world] = torch.cat([
                    torch.zeros(6, device=torch_device, dtype=torch.float32),
                    targets.squeeze(0),
                ])
            wp.copy(control.joint_target_pos, wp.from_torch(all_targets, dtype=wp.float32))

        # Physics: one DT_OUTER worth of CENIC inner steps.
        state_0, state_1 = solver.step_dt(
            DT_OUTER, state_0, state_1, control,
            apply_forces=viewer.apply_forces,
        )
        t += DT_OUTER
        step += 1

        if step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test, headless, N=1, short**

Run:
```bash
uv run python -m scripts.demos.anymal_clutter --num-worlds 1 --num-steps 100 --headless
```

Expected:
- The policy file downloads or is found in the cached `anybotics_anymal_c` asset.
- 100 steps complete (≈0.2 s sim time at `DT_OUTER=2 ms`) without exceptions.
- The status grid prints with a finite `RMS error` and most worlds `ok`.

If torch is not installed, run:
```bash
uv run --extra torch-cu12 python -m scripts.demos.anymal_clutter --num-worlds 1 --num-steps 100 --headless
```

- [ ] **Step 3: Smoke test at N=3, headless, longer**

Run:
```bash
uv run python -m scripts.demos.anymal_clutter --num-worlds 3 --num-steps 1000 --headless
```

Expected:
- 1000 steps complete (≈2 s sim time).
- The robot has visibly moved forward — verify by sampling base position before and after:
```bash
uv run python -c "
from scripts.scenes.anymal_clutter import build_model_randomized
m = build_model_randomized(1)
print('base xyz at start:', m.joint_q.numpy()[:3])
"
```
Initial base z should be ~0.62. After 1000 steps the demo's status grid should show a positive RMS error trace and the world's robot should still be standing (no nan).

- [ ] **Step 4: Visual sanity check**

Run:
```bash
uv run python -m scripts.demos.anymal_clutter --num-worlds 1
```

Expected: viewer opens; the robot walks forward (in world +y) and bumps into / pushes through the scattered light cubes. Close the window to exit.

If `nconmax`/`njmax` are exceeded (look for warnings about contact buffer overflow), bump them in `scripts/scenes/anymal_clutter.py:make_solver` from `nconmax=400` to `nconmax=800`, and `njmax=200` to `njmax=400`.

If the robot collapses / never walks, verify the policy file path printed at startup matches what's in the asset cache:
```bash
uv run python -c "
import newton.utils
print(newton.utils.download_asset('anybotics_anymal_c') / 'rl_policies' / 'anymal_walking_policy_physx.pt')
"
```

---

## Phase 4: Cross-scene verification

### Task 7: Confirm all three scenes coexist and the existing scenes still work

- [ ] **Step 1: Run the existing contact_objects demo to confirm no regression**

Run:
```bash
uv run python -m scripts.demos.contact_objects --num-worlds 1 --num-steps 100 --headless
```

Expected: completes without errors.

- [ ] **Step 2: Run all three new demos back-to-back**

Run:
```bash
uv run python -m scripts.demos.falling_cylinder --num-worlds 1 --num-steps 100 --headless
uv run python -m scripts.demos.falling_gripper  --num-worlds 1 --num-steps 100 --headless
uv run python -m scripts.demos.anymal_clutter   --num-worlds 1 --num-steps 100 --headless
```

Expected: all three complete successfully, FPS line prints for each.

- [ ] **Step 3: Run pre-commit hooks across the new files**

Run:
```bash
uvx pre-commit run -a
```

Expected: all hooks pass on the new files. If any reformats files, review the diff before staging.

- [ ] **Step 4: Hand off**

Stop here and notify the user that the implementation is complete and ready for them to commit.

---

## Self-Review Checklist

- **Spec coverage:**
  - Falling cylinder (scene + demo): Tasks 1, 2 ✓
  - Falling gripper (scene + demo): Tasks 3, 4 ✓
  - ANYmal clutter (scene + demo): Tasks 5, 6 ✓
  - Cross-scene smoke + lint: Task 7 ✓
  - Spec note "ANYmal scene module is torch-free": Task 5 step 3 explicitly verifies ✓
  - Spec note "rail visual-only, collision filtered": collision_group `GROUP_RAIL=-2` distinct from `GROUP_DYN=1` in Task 3 ✓
  - Spec note "ANYmal forward direction is world +y": clutter scatter region uses `y ∈ [0.3, 3.3], x ∈ [-1.5, 1.5]` in Task 5 ✓
- **Type / signature consistency:** All scene modules expose the same five-symbol API (`build_template`, `build_model`, `build_model_randomized`, `make_solver`, `make_fixed_solver`) ✓
- **No placeholders:** every step has concrete code or commands ✓
- **No commits:** none of the tasks contain `git add` or `git commit` per user convention ✓
