# Franka Dish-Rack Drop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CENIC scene where a Franka FR3 arm executes a scripted joint-space trajectory and releases a mug, fork, and spatula into the LBM dish drying rack.

**Architecture:** Scene module (torch-free, standard 5-function API) loads the LBM rack, the Franka FR3 URDF, three held bodies (LBM mug + fork with mesh collision; YCB spatula with visual mesh + box-primitive collision), and exposes `update_held_objects` (kinematic pin-then-release) and `update_franka_targets` (smoothstep keyframe interpolation) helpers. Demo wraps the scene in the canonical `solver.step_dt` loop with the standard CLI flags.

**Tech Stack:** Python 3.10+, warp, newton, numpy, trimesh

**Spec:** `docs/superpowers/specs/2026-05-05-franka-dish-rack-design.md`

**Convention:** No `git add` / `git commit` steps in tasks — the user commits all changes themselves at phase boundaries. (Per `MEMORY.md`.)

---

## File inventory

New files:
- `scripts/scenes/franka_dish_rack.py`
- `scripts/demos/franka_dish_rack.py`
- (Plan) this file
- (Spec) `docs/superpowers/specs/2026-05-05-franka-dish-rack-design.md` (already written)

No files modified.

---

## Phase 1: Probe Franka geometry

### Task 1: Probe EE pose and resolve body / joint indices

**Files:** No code committed. This task runs a one-shot probe and harvests numeric constants for Task 2.

- [ ] **Step 1: Run the probe script**

The probe loads the Franka URDF at world origin in a "carrying" arm pose, runs FK, and prints (a) the EE world pose so we can pick the base xform, (b) the per-world body / joint indices we will need at runtime, (c) the URDF joint labels for sanity. Run this exact command:

```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -c '
import math
import warp as wp
import numpy as np
import newton, newton.solvers, newton.utils

# Candidate "carrying" arm pose -- EE roughly in front of and above the base.
ARM_Q = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]
FINGER_Q = [0.0, 0.0]  # closed

b = newton.ModelBuilder()
newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(b)

asset = newton.utils.download_asset("franka_emika_panda")
b.add_urdf(
    str(asset / "urdf" / "fr3_franka_hand.urdf"),
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    floating=False,
    enable_self_collisions=False,
    collapse_fixed_joints=True,
    ignore_inertial_definitions=False,
)
for i in range(7):
    b.joint_q[i] = ARM_Q[i]
for i in range(2):
    b.joint_q[7 + i] = FINGER_Q[i]

m = b.finalize()
state = m.state()
newton.eval_fk(m, state.joint_q, state.joint_qd, state)

print("=== Joint labels ===")
for i, lbl in enumerate(m.joint_key):
    print(f"  joint[{i}]: {lbl}")
print()
print("=== Body labels (world coords) ===")
bq = state.body_q.numpy()
for i, lbl in enumerate(m.body_key):
    print(f"  body[{i}] {lbl:>30}: pos=({bq[i,0]:+.4f}, {bq[i,1]:+.4f}, {bq[i,2]:+.4f})")
print()
ee_idx = len(m.body_key) - 1  # last collapsed body should be the hand/finger end
print(f"=== Last body (assumed EE) idx={ee_idx} ===")
print(f"  pos: {bq[ee_idx, :3]}")
print(f"  quat: {bq[ee_idx, 3:7]}")
print()
print(f"=== Per-world counts (single-world template) ===")
print(f"  body_count = {m.body_count}")
print(f"  joint_count = {len(m.joint_key)}")
print(f"  joint_coord_count = {m.joint_coord_count}")
print(f"  joint_dof_count = {m.joint_dof_count}")
'
```

Expected: prints joint labels (9 of them: `fr3_joint1..7` + 2 finger joints), body labels (8-9 collapsed bodies), the EE position with the carrying pose, and per-world counts.

- [ ] **Step 2: Harvest the constants for Task 2**

From the probe output, **write down these values** (you will paste them into Task 2):

1. The **EE world position** with the candidate ARM_Q pose — call this `(ee_x, ee_y, ee_z)`. Expected to be roughly in front of the base, e.g. `(0.4, 0.0, 0.5)` ish.
2. The **base xform** that will put the EE directly above the rack center (rack is at world origin, top at z ≈ 0.124 m, we want EE at world `(0, 0, 0.4)` — i.e. 0.28 m above rack top, gripper pointing down).

   Compute: `BASE_X = -(ee_x - 0.0)`, `BASE_Y = -(ee_y - 0.0)`, `BASE_Z = -(ee_z - 0.4)`. So `BASE_XFORM = wp.transform(wp.vec3(BASE_X, BASE_Y, BASE_Z), wp.quat_identity())`. (No rotation — we just translate the base so that the same arm pose puts the EE over the rack.)
3. The **EE quaternion** at the carrying pose. Verify it has the gripper pointing down (z-axis of the gripper roughly = world -z). If not, adjust `ARM_Q[6]` (joint7) by ±π/4 and re-run the probe.
4. The **EE body index** (last item in the body_key list).
5. The **per-world Franka body count, joint coord count, and joint dof count.**

Record these values; they go into Task 2 verbatim.

- [ ] **Step 3: Sanity-check the carrying pose**

If the EE quaternion does not have its z-axis pointing down (e.g., its rotation matrix's third column is not approximately `[0, 0, -1]`), iterate on `ARM_Q` (typically `joint5` controls wrist orientation around the y axis, `joint7` controls roll around the z axis) until it does. Re-run Step 1.

If the EE position can't reach `(0, 0, 0.4)` even after translating the base by 0.5 m or more (i.e. the arm is fully extended and still falls short), reduce the rack-EE offset from 0.4 m down to 0.3 m or even 0.25 m. The exact height is not critical; what matters is that there's some clearance for the objects to fall.

---

## Phase 2: Scene module

### Task 2: Create `scripts/scenes/franka_dish_rack.py`

**Files:**
- Create: `scripts/scenes/franka_dish_rack.py`

This is one big self-contained file (~350 LOC). The full code is below — fill in the constants from Task 1 in the marked spots.

- [ ] **Step 1: Write the file**

Create `scripts/scenes/franka_dish_rack.py` with this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Franka FR3 drops mug, fork, spatula into the LBM dish drying rack.

The arm executes a scripted joint-space trajectory (smoothstep-interpolated
keyframes). Held objects are kinematically pinned to the gripper EE pose
until ``RELEASE_TIME``; after that they are free-falling rigid bodies that
settle into the rack via SDF-ish (mesh) and primitive contact with the
existing LBM rack assets.

Geometry choices:
  * Rack          -- LBM glTF meshes with SDF collision (reused from
    scripts/scenes/dish_rack.py).
  * Franka FR3    -- URDF from newton.utils.download_asset("franka_emika_panda").
  * Mug           -- LBM glTF mesh, convex-hull collision (16-vert cap).
  * Fork          -- LBM glTF mesh, convex-hull collision (16-vert cap).
  * Spatula       -- YCB textured glTF as visual-only; box primitive matching
    the bbox handles collision (the spatula is a thin slab; a 16-vert hull
    of it is much rougher than the primitive box).

Public API mirrors scripts/scenes/dish_rack.py and
scripts/scenes/anymal_clutter.py:
  build_template, build_model, build_model_randomized,
  make_solver, make_fixed_solver,
  update_franka_targets, update_held_objects.

The module is torch-free.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers
import newton.utils

# --- Timing / solver defaults ---------------------------------------------

DT_OUTER = 0.002   # 2 ms control / render cadence [s]
TOL = 1e-3
LOG_EVERY = 50     # status print cadence (in outer steps)

# --- Asset paths -----------------------------------------------------------

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_LBM_ROOT = _ASSETS / "lbm"

_RACK_BASE_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_base.gltf"
_RACK_WIREFRAME_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_wireframe.gltf"
_RACK_UTENSIL_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_utensil_holder.gltf"

_LBM_MUG_DIR = _LBM_ROOT / "mugs"
_LBM_MUG_GLTF = (
    _LBM_MUG_DIR / "assets"
    / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.gltf"
)
_LBM_MUG_SDF = (
    _LBM_MUG_DIR
    / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.sdf"
)

_LBM_FORK_DIR = _LBM_ROOT / "forks"
_LBM_FORK_GLTF = (
    _LBM_FORK_DIR / "assets"
    / "cambridge_jubilee_stainless_plastic_black_fork.gltf"
)
_LBM_FORK_SDF = (
    _LBM_FORK_DIR
    / "cambridge_jubilee_stainless_plastic_black_fork.sdf"
)

_YCB_SPATULA_OBJ = (
    _ASSETS / "033_spatula_berkeley_meshes" / "033_spatula" / "tsdf" / "textured.obj"
)

# --- Welded transforms inside the LBM rack (copied from dish_rack.py) ------

_WIREFRAME_OFFSET_P = wp.vec3(-0.00527, -0.01075, 0.009861)
_WIREFRAME_OFFSET_PITCH_RAD = 0.013858
_UTENSIL_OFFSET_P = wp.vec3(-0.1518, 0.0735, 0.11335)
_UTENSIL_OFFSET_PITCH_RAD = math.radians(-4.5)

# --- Franka base placement (FILL FROM TASK 1) ------------------------------
# Compute BASE_X/Y/Z so the carrying-pose EE lands at world (0, 0, 0.4) above
# the rack center. Values come from the Task 1 probe output.
_FRANKA_BASE_X = 0.55      # <-- TASK1: replace with -ee_x from probe (typical ~0.4-0.6)
_FRANKA_BASE_Y = 0.0       # <-- TASK1: -ee_y
_FRANKA_BASE_Z = -0.05     # <-- TASK1: -(ee_z - 0.4) (negative if EE was above 0.4)
_FRANKA_BASE_XFORM = wp.transform(
    wp.vec3(_FRANKA_BASE_X, _FRANKA_BASE_Y, _FRANKA_BASE_Z),
    wp.quat_identity(),
)

# Per-world body / joint counts (FILL FROM TASK 1) --------------------------
# After collapse_fixed_joints, the FR3+hand URDF expands to:
#   bodies_per_world = ?  (rack adds 0; Franka collapsed bodies; 3 held)
#   ee_body_idx_per_world = bodies_per_world - 4  (hand body, before the 3 held)
# Update these from the probe output.
_FRANKA_BODY_COUNT = 9     # <-- TASK1: from m.body_count
_FRANKA_JOINT_COUNT = 9    # 7 arm + 2 finger
_HELD_COUNT = 3
_BODIES_PER_WORLD = _FRANKA_BODY_COUNT + _HELD_COUNT
_EE_BODY_IDX_PER_WORLD = _FRANKA_BODY_COUNT - 1  # last Franka body == hand

# Per-world joint coords:
#   _FRANKA_JOINT_COUNT (revolute/prismatic, 1 coord each) + 3 held (free, 7 coords each)
_COORDS_PER_WORLD = _FRANKA_JOINT_COUNT + _HELD_COUNT * 7
# Per-world joint dofs:
#   _FRANKA_JOINT_COUNT (1 dof each) + 3 held (free, 6 dofs each)
_DOFS_PER_WORLD = _FRANKA_JOINT_COUNT + _HELD_COUNT * 6

# --- Trajectory keyframes --------------------------------------------------
# Columns: t [s], joint1..joint7 [rad], finger1, finger2 [m].
# Rows must be sorted by time. Smoothstep interpolation is applied between
# consecutive keyframes. The carrying pose is intentionally identical at
# t=0 and t=1.5 so the arm holds still until release; finger position
# changes between t=1.5 and t=1.7 to open the gripper.
_ARM_CARRYING = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]  # match probe ARM_Q
_GRIPPER_CLOSED = [0.0, 0.0]
_GRIPPER_OPEN = [0.04, 0.04]

KEYFRAMES = np.array(
    [
        [0.0, *_ARM_CARRYING, *_GRIPPER_CLOSED],
        [1.5, *_ARM_CARRYING, *_GRIPPER_CLOSED],
        [1.7, *_ARM_CARRYING, *_GRIPPER_OPEN],
        [4.0, *_ARM_CARRYING, *_GRIPPER_OPEN],
    ],
    dtype=np.float64,
)
RELEASE_TIME = 1.7

# --- Pin offsets (held body position relative to EE frame) -----------------
# Mug centered & below TCP; fork and spatula offset slightly to either side
# so they don't overlap. Tuned at smoke-test time if visual clipping appears.
_PIN_OFFSETS_POS = [
    np.array([0.0, 0.0, -0.06], dtype=np.float64),   # mug
    np.array([0.0, 0.04, -0.07], dtype=np.float64),  # fork
    np.array([0.0, -0.04, -0.07], dtype=np.float64), # spatula
]
_PIN_OFFSETS_QUAT = [
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # mug -- identity
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # fork
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # spatula
]

# --- LBM helpers (verbatim copies from scripts/scenes/dish_rack.py) --------

_MESH_CACHE: dict[Path, newton.Mesh] = {}
_HULL_BUDGET = 16


def _gltf_y_up_to_z_up(verts: np.ndarray) -> np.ndarray:
    return np.stack([verts[:, 0], -verts[:, 2], verts[:, 1]], axis=1)


def _parse_lbm_inertial(sdf_path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    root = ET.parse(str(sdf_path)).getroot()
    inertial = root.find(".//inertial")
    if inertial is None:
        raise ValueError(f"No <inertial> in {sdf_path}")
    pose = np.fromstring(inertial.findtext("pose", "0 0 0 0 0 0"), sep=" ")
    com = pose[:3].astype(np.float64)
    mass = float(inertial.findtext("mass", "0"))
    it = inertial.find("inertia")
    ixx = float(it.findtext("ixx", "0"))
    ixy = float(it.findtext("ixy", "0"))
    ixz = float(it.findtext("ixz", "0"))
    iyy = float(it.findtext("iyy", "0"))
    iyz = float(it.findtext("iyz", "0"))
    izz = float(it.findtext("izz", "0"))
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return mass, com, inertia


def _load_lbm_mesh(
    gltf_path: Path,
    sdf_path: Path | None = None,
    *,
    target_mass: float | None = None,
) -> newton.Mesh:
    if gltf_path in _MESH_CACHE and sdf_path is None and target_mass is None:
        return _MESH_CACHE[gltf_path]

    import trimesh

    if not gltf_path.exists():
        raise FileNotFoundError(f"LBM glTF not found: {gltf_path}")

    raw = trimesh.load(str(gltf_path), process=False, force="mesh")
    verts = _gltf_y_up_to_z_up(np.asarray(raw.vertices, dtype=np.float64)).astype(np.float32)
    faces = np.asarray(raw.faces, dtype=np.int32).flatten()

    uvs = None
    texture = None
    if hasattr(raw.visual, "uv") and raw.visual.uv is not None:
        uvs = np.asarray(raw.visual.uv, dtype=np.float32)
    if uvs is not None and hasattr(raw.visual, "material"):
        embedded = getattr(raw.visual.material, "baseColorTexture", None)
        if embedded is not None:
            texture = np.asarray(embedded)

    mesh = newton.Mesh(
        verts, faces, uvs=uvs, texture=texture,
        compute_inertia=False, maxhullvert=_HULL_BUDGET,
    )

    if sdf_path is not None:
        lbm_mass, lbm_com, lbm_inertia = _parse_lbm_inertial(sdf_path)
        if target_mass is not None and lbm_mass > 0.0:
            lbm_inertia = lbm_inertia * (target_mass / lbm_mass)
            lbm_mass = target_mass
        mesh.mass = lbm_mass
        mesh.com = wp.vec3(*lbm_com)
        mesh.inertia = wp.mat33(lbm_inertia)

    if sdf_path is None and target_mass is None:
        _MESH_CACHE[gltf_path] = mesh
    return mesh


def _load_ycb_mesh(obj_path: Path) -> newton.Mesh:
    if obj_path in _MESH_CACHE:
        return _MESH_CACHE[obj_path]
    import trimesh
    if not obj_path.exists():
        raise FileNotFoundError(f"YCB OBJ not found: {obj_path}")
    raw = trimesh.load(str(obj_path), process=False, force="mesh")
    verts = np.asarray(raw.vertices, dtype=np.float32)
    faces = np.asarray(raw.faces, dtype=np.int32).flatten()
    uvs = None
    texture = None
    if hasattr(raw.visual, "uv") and raw.visual.uv is not None:
        uvs = np.asarray(raw.visual.uv, dtype=np.float32)
    if uvs is not None and hasattr(raw.visual, "material"):
        embedded = getattr(raw.visual.material, "baseColorTexture", None)
        if embedded is not None:
            texture = np.asarray(embedded)
    mesh = newton.Mesh(
        verts, faces, uvs=uvs, texture=texture,
        compute_inertia=False, maxhullvert=_HULL_BUDGET,
    )
    _MESH_CACHE[obj_path] = mesh
    return mesh


def _add_drainer(builder: newton.ModelBuilder, cfg, voxel: float) -> None:
    """Static LBM dish rack -- copy from scripts/scenes/dish_rack.py."""
    q_wf = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), _WIREFRAME_OFFSET_PITCH_RAD)
    wf_xform = wp.transform(p=_WIREFRAME_OFFSET_P, q=q_wf)
    q_ut_local = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), _UTENSIL_OFFSET_PITCH_RAD)
    ut_local_xform = wp.transform(p=_UTENSIL_OFFSET_P, q=q_ut_local)
    ut_xform = wp.transform_multiply(wf_xform, ut_local_xform)

    base_mesh = _load_lbm_mesh(_RACK_BASE_GLTF)
    wf_mesh = _load_lbm_mesh(_RACK_WIREFRAME_GLTF)
    ut_mesh = _load_lbm_mesh(_RACK_UTENSIL_GLTF)
    for m in (base_mesh, wf_mesh, ut_mesh):
        if m.sdf is None:
            m.build_sdf(target_voxel_size=voxel)

    unit_scale = wp.vec3(1.0, 1.0, 1.0)
    builder.add_shape_mesh(
        body=-1, xform=wp.transform_identity(),
        mesh=base_mesh, cfg=cfg, scale=unit_scale,
    )
    builder.add_shape_mesh(
        body=-1, xform=wf_xform,
        mesh=wf_mesh, cfg=cfg, scale=unit_scale,
    )
    builder.add_shape_mesh(
        body=-1, xform=ut_xform,
        mesh=ut_mesh, cfg=cfg, scale=unit_scale,
    )


# --- Held-object spawn helpers --------------------------------------------

def _add_held_objects(builder: newton.ModelBuilder) -> None:
    """Add 3 free-body held objects: mug, fork, spatula."""
    cfg_held_mesh = newton.ModelBuilder.ShapeConfig(
        ke=4e4, kd=400.0, mu=0.4, margin=5e-3, collision_group=2,
    )

    # Mug
    mug_body = builder.add_body(xform=wp.transform_identity())
    mug_mesh = _load_lbm_mesh(_LBM_MUG_GLTF, _LBM_MUG_SDF)
    builder.add_shape_mesh(mug_body, mesh=mug_mesh, cfg=cfg_held_mesh)

    # Fork
    fork_body = builder.add_body(xform=wp.transform_identity())
    fork_mesh = _load_lbm_mesh(_LBM_FORK_GLTF, _LBM_FORK_SDF)
    builder.add_shape_mesh(fork_body, mesh=fork_mesh, cfg=cfg_held_mesh)

    # Spatula: visual mesh + box collision (visual+primitive split).
    spat_body = builder.add_body(xform=wp.transform_identity())
    spat_mesh = _load_ycb_mesh(_YCB_SPATULA_OBJ)
    cfg_spat_visual = newton.ModelBuilder.ShapeConfig(
        ke=4e4, kd=400.0, mu=0.4, margin=5e-3, collision_group=2,
        has_shape_collision=False, has_particle_collision=False,
    )
    builder.add_shape_mesh(spat_body, mesh=spat_mesh, cfg=cfg_spat_visual)
    bbox_lo = spat_mesh.vertices.min(axis=0)
    bbox_hi = spat_mesh.vertices.max(axis=0)
    centre = (bbox_hi + bbox_lo) * 0.5
    size = bbox_hi - bbox_lo
    cfg_spat_col = newton.ModelBuilder.ShapeConfig(
        ke=4e4, kd=400.0, mu=0.4, margin=5e-3, collision_group=2,
        density=80.0, is_visible=False,
    )
    builder.add_shape_box(
        spat_body,
        xform=wp.transform(
            p=wp.vec3(float(centre[0]), float(centre[1]), float(centre[2])),
            q=wp.quat_identity(),
        ),
        hx=float(size[0] * 0.5),
        hy=float(size[1] * 0.5),
        hz=float(size[2] * 0.5),
        cfg=cfg_spat_col,
    )


# --- Public scene API ------------------------------------------------------

def build_template() -> newton.ModelBuilder:
    """Single-world template: rack + Franka + 3 held bodies."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    template.default_shape_cfg.ke = 4e4
    template.default_shape_cfg.kd = 400.0
    template.default_shape_cfg.kf = 1e3
    template.default_shape_cfg.mu = 0.6

    # Rack (static).
    cfg_rack = newton.ModelBuilder.ShapeConfig(
        ke=4e4, kd=400.0, mu=0.3, margin=5e-3,
        collision_group=-1, is_hydroelastic=False,
    )
    _add_drainer(template, cfg_rack, voxel=0.001)

    # Franka FR3.
    asset_path = newton.utils.download_asset("franka_emika_panda")
    template.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=_FRANKA_BASE_XFORM,
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    # Initial Franka joint q = first keyframe (carrying pose, gripper closed).
    arm_q = KEYFRAMES[0, 1:8]
    finger_q = KEYFRAMES[0, 8:10]
    for i in range(7):
        template.joint_q[i] = float(arm_q[i])
    for i in range(2):
        template.joint_q[7 + i] = float(finger_q[i])

    # Per-joint actuator gains: arm vs finger.
    for i in range(7):
        template.joint_target_ke[i] = 600.0
        template.joint_target_kd[i] = 30.0
    for i in (7, 8):
        template.joint_target_ke[i] = 400.0
        template.joint_target_kd[i] = 20.0

    # Held bodies (free, 7 coords each, total 21 coords appended).
    _add_held_objects(template)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    ground_cfg = newton.ModelBuilder.ShapeConfig(collision_group=-1)
    builder.add_ground_plane(cfg=ground_cfg)
    return builder.finalize()


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N worlds with per-world held-object yaw jitter (held bodies start at
    identity + jitter; the kinematic pin overrides them on the first outer
    step anyway)."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        base = w * _COORDS_PER_WORLD
        # Held body free joints start at offset _FRANKA_JOINT_COUNT (=9)
        # and occupy the next _HELD_COUNT * 7 coords.
        held_start = base + _FRANKA_JOINT_COUNT
        for i in range(_HELD_COUNT):
            yaw = rng.uniform(-math.radians(15.0), math.radians(15.0))
            cz = math.cos(yaw / 2.0)
            sz = math.sin(yaw / 2.0)
            # Position will be overwritten by update_held_objects on step 1;
            # leave at template default.
            qx_idx = held_start + i * 7 + 3
            joint_q_np[qx_idx + 0] = 0.0
            joint_q_np[qx_idx + 1] = 0.0
            joint_q_np[qx_idx + 2] = sz
            joint_q_np[qx_idx + 3] = cz
    model.joint_q.assign(joint_q_np)
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
        dt_inner_min=1e-6,
        dt_inner_max=0.01,
        dt_mode=dt_mode,
        nconmax=2048,
        njmax=8192,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=True,
        nconmax=2048,
        njmax=8192,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
        solver="newton",
    )


# --- Runtime helpers (called from the demo loop) ---------------------------

def _interpolate_keyframes(t: float) -> np.ndarray:
    """Smoothstep-interpolated joint targets at time t (returns 9 floats)."""
    if t <= KEYFRAMES[0, 0]:
        return KEYFRAMES[0, 1:].copy()
    if t >= KEYFRAMES[-1, 0]:
        return KEYFRAMES[-1, 1:].copy()
    for i in range(len(KEYFRAMES) - 1):
        t0 = KEYFRAMES[i, 0]
        t1 = KEYFRAMES[i + 1, 0]
        if t0 <= t < t1:
            alpha = (t - t0) / (t1 - t0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            return (1.0 - alpha) * KEYFRAMES[i, 1:] + alpha * KEYFRAMES[i + 1, 1:]
    return KEYFRAMES[-1, 1:].copy()


def update_franka_targets(model, control, sim_time: float) -> None:
    """Set joint_target_pos for all worlds' Franka joints from KEYFRAMES."""
    target = _interpolate_keyframes(sim_time)
    n_worlds = model.world_count
    targets_np = control.joint_target_pos.numpy()
    for w in range(n_worlds):
        base = w * _DOFS_PER_WORLD
        for i in range(_FRANKA_JOINT_COUNT):
            targets_np[base + i] = float(target[i])
    control.joint_target_pos.assign(targets_np)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions stored as (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by quaternion (x, y, z, w)."""
    qx, qy, qz, qw = q
    t = 2.0 * np.cross(np.array([qx, qy, qz]), v)
    return v + qw * t + np.cross(np.array([qx, qy, qz]), t)


def update_held_objects(model, state, sim_time: float) -> None:
    """Pin each held body to the EE pose (with offset) -- or release if t >= RELEASE_TIME.

    Called once per outer step BEFORE solver.step_dt(). Held bodies have their
    joint_q (free 7-tuple) and joint_qd (free 6-tuple) overwritten while
    pinned; after RELEASE_TIME this function is a no-op and they fall freely.
    """
    if sim_time >= RELEASE_TIME:
        return

    n_worlds = model.world_count
    body_q_np = state.body_q.numpy()
    joint_q_np = state.joint_q.numpy()
    joint_qd_np = state.joint_qd.numpy()

    for w in range(n_worlds):
        ee_idx = w * _BODIES_PER_WORLD + _EE_BODY_IDX_PER_WORLD
        ee_pos = body_q_np[ee_idx, :3].astype(np.float64)
        ee_quat = body_q_np[ee_idx, 3:7].astype(np.float64)  # (x, y, z, w)

        held_start_coord = w * _COORDS_PER_WORLD + _FRANKA_JOINT_COUNT
        held_start_dof = w * _DOFS_PER_WORLD + _FRANKA_JOINT_COUNT
        held_body_start = w * _BODIES_PER_WORLD + _FRANKA_BODY_COUNT

        for i in range(_HELD_COUNT):
            offset_pos = _PIN_OFFSETS_POS[i]
            offset_quat = _PIN_OFFSETS_QUAT[i]

            world_pos = ee_pos + _quat_rotate(ee_quat, offset_pos)
            world_quat = _quat_mul(ee_quat, offset_quat)

            c = held_start_coord + i * 7
            joint_q_np[c + 0] = world_pos[0]
            joint_q_np[c + 1] = world_pos[1]
            joint_q_np[c + 2] = world_pos[2]
            joint_q_np[c + 3] = world_quat[0]
            joint_q_np[c + 4] = world_quat[1]
            joint_q_np[c + 5] = world_quat[2]
            joint_q_np[c + 6] = world_quat[3]

            d = held_start_dof + i * 6
            for k in range(6):
                joint_qd_np[d + k] = 0.0

            # Also update body_q so the viewer sees the correct pose without
            # waiting for FK.
            b = held_body_start + i
            body_q_np[b, 0] = world_pos[0]
            body_q_np[b, 1] = world_pos[1]
            body_q_np[b, 2] = world_pos[2]
            body_q_np[b, 3] = world_quat[0]
            body_q_np[b, 4] = world_quat[1]
            body_q_np[b, 5] = world_quat[2]
            body_q_np[b, 6] = world_quat[3]

    state.joint_q.assign(joint_q_np)
    state.joint_qd.assign(joint_qd_np)
    state.body_q.assign(body_q_np)
```

- [ ] **Step 2: Verify the scene builds**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -c "
from scripts.scenes.franka_dish_rack import build_model, make_solver, _BODIES_PER_WORLD, _COORDS_PER_WORLD, _DOFS_PER_WORLD
m = build_model(1)
print('OK')
print(f'  body_count={m.body_count} (expected {_BODIES_PER_WORLD})')
print(f'  joint_coord_count={m.joint_coord_count} (expected {_COORDS_PER_WORLD})')
print(f'  joint_dof_count={m.joint_dof_count} (expected {_DOFS_PER_WORLD})')
s = make_solver(m)
print('  solver built OK')
"
```

Expected: `OK` line + counts that match the constants. If counts don't match, the `_FRANKA_BODY_COUNT` / `_COORDS_PER_WORLD` / `_DOFS_PER_WORLD` constants need adjustment based on the actual probe values from Task 1.

- [ ] **Step 3: Verify the EE body index is correct**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -c "
import newton
from scripts.scenes.franka_dish_rack import build_model, _EE_BODY_IDX_PER_WORLD
m = build_model(1)
state = m.state()
newton.eval_fk(m, state.joint_q, state.joint_qd, state)
ee_pos = state.body_q.numpy()[_EE_BODY_IDX_PER_WORLD, :3]
print(f'EE world position with carrying pose: {ee_pos}')
print(f'(Should be approximately (0, 0, 0.4) above rack center)')
"
```

Expected: EE pos ~`(0, 0, 0.4)` ± a few cm. If far from that, adjust the `_FRANKA_BASE_*` constants to compensate.

- [ ] **Step 4: Verify update_held_objects places bodies near the EE on call**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -c "
import newton
from scripts.scenes.franka_dish_rack import (
    build_model, update_held_objects,
    _EE_BODY_IDX_PER_WORLD, _FRANKA_BODY_COUNT, _BODIES_PER_WORLD,
)
m = build_model(1)
state = m.state()
newton.eval_fk(m, state.joint_q, state.joint_qd, state)
update_held_objects(m, state, sim_time=0.0)
bq = state.body_q.numpy()
ee = bq[_EE_BODY_IDX_PER_WORLD, :3]
print(f'EE pos:        {ee}')
for i, name in enumerate(['mug', 'fork', 'spatula']):
    held = bq[_FRANKA_BODY_COUNT + i, :3]
    print(f'  {name:>7}: {held}  (offset from EE: {held - ee})')
print('Expected: each held body within ~10 cm of EE')
"
```

Expected: each held body within 10 cm of the EE. If they're at the world origin instead, the body indexing is off — re-check `_FRANKA_BODY_COUNT` and `_EE_BODY_IDX_PER_WORLD`.

---

## Phase 3: Demo module

### Task 3: Create `scripts/demos/franka_dish_rack.py`

**Files:**
- Create: `scripts/demos/franka_dish_rack.py`

- [ ] **Step 1: Write the demo file**

Create `scripts/demos/franka_dish_rack.py` with this content:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive Franka FR3 drops mug, fork, spatula into the LBM dish rack.

Uses the canonical CENIC step_dt loop with two per-step helpers:
  update_franka_targets: writes joint_target_pos from the keyframe table.
  update_held_objects:   kinematically pins held bodies to the EE until
                         RELEASE_TIME, then lets them fall freely.

Usage::

    uv run python -m scripts.demos.franka_dish_rack [--num-worlds N] [--headless] [--tol FLOAT]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers

from scripts.scenes.franka_dish_rack import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_fixed_solver,
    make_solver,
    update_franka_targets,
    update_held_objects,
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
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'Linf error':>{col}}{'status':>{col}}"
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
        "--tol", type=float, default=None,
        help="CENIC tolerance override (default uses scene's TOL).",
    )
    parser.add_argument(
        "--fixed-dt", type=float, default=None,
        help="Use fixed-step SolverMuJoCo with this dt instead of CENIC.",
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
            f"Fixed-step Franka demo: {args.num_worlds} world(s)  dt={args.fixed_dt:.4e}  "
            f"substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model, tol=args.tol) if args.tol is not None else make_solver(model)
        print(
            f"CENIC Franka demo: {args.num_worlds} world(s)  tol={solver._tol:.1e}  "
            f"dt_inner_init={solver._dt.numpy()[0]:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.9, -0.9, 0.55), pitch=-15.0, yaw=135.0)

    if use_fixed:
        contacts = newton.Contacts(
            rigid_contact_max=2048, soft_contact_max=0,
            requested_attributes={"force"},
        )
    else:
        contacts = solver.contacts

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        # Per-step trajectory + held-object updates.
        update_franka_targets(model, control, t)
        update_held_objects(model, state_0, t)

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

- [ ] **Step 2: Smoke test, headless, N=1, 200 steps (covers carry + release window)**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -m scripts.demos.franka_dish_rack --num-worlds 1 --num-steps 200 --headless
```

Expected: completes without exception. 200 steps × DT_OUTER (2 ms) = 0.4 s sim time, well before RELEASE_TIME=1.7s, so we're verifying the carry phase works.

If this fails with "EE body index out of bounds" or similar, re-check the constants in `franka_dish_rack.py`.

If it fails with "Number of Newton contacts exceeded MJWarp limit", bump `nconmax` in `make_solver` from 2048 to 4096.

- [ ] **Step 3: Smoke test through release, headless, N=1, 1000 steps**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -m scripts.demos.franka_dish_rack --num-worlds 1 --num-steps 1000 --headless
```

Expected: completes (2 s sim, just past release at 1.7s). Verify final body positions:

```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -c "
import warp as wp
import newton
from scripts.scenes.franka_dish_rack import (
    build_model_randomized, make_solver, update_franka_targets,
    update_held_objects, DT_OUTER, _FRANKA_BODY_COUNT,
)
m = build_model_randomized(1)
solver = make_solver(m, tol=1e-2)  # bumped for speed
s0 = m.state(); s1 = m.state(); ctrl = m.control()
import newton as nt
nt.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
import time
t = 0.0
t0 = time.time()
for step in range(1500):
    update_franka_targets(m, ctrl, t)
    update_held_objects(m, s0, t)
    s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    t += DT_OUTER
print(f'wall={time.time()-t0:.1f} s, sim={t:.2f} s')
bq = s0.body_q.numpy()
for i, name in enumerate(['mug', 'fork', 'spatula']):
    p = bq[_FRANKA_BODY_COUNT + i, :3]
    in_rack = abs(p[0]) < 0.16 and abs(p[1]) < 0.23 and 0.0 <= p[2] < 0.20
    print(f'  {name:>7}: pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})  in_rack={in_rack}')
"
```

Expected: all three objects have `z < 0.20 m` (settled), and `|x| < 0.16, |y| < 0.23` (within rack footprint). If `in_rack=False` for any object, adjust the carrying-pose EE position downward (closer to rack) so they don't bounce out.

---

## Phase 4: Smoke + iteration

### Task 4: N=4 randomized smoke + interactive sanity

- [ ] **Step 1: N=4 randomized**

Run:
```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -m scripts.demos.franka_dish_rack --num-worlds 4 --num-steps 1500 --headless --tol 1e-2
```

Expected: completes; status grid shows 4 worlds with diverging dt; objects land in slightly different poses per world.

- [ ] **Step 2: Hand off to user for visual sanity check**

Notify the user that the implementation is complete and they should run:

```bash
uv run --project /home/marcodigiorgio/Documents/CODE/newton-cenic python -m scripts.demos.franka_dish_rack
```

Watch for:
- Arm holds carrying pose for 1.5 s with all 3 objects pinned in the gripper.
- At t=1.5 s gripper opens.
- At t=1.7 s objects detach and fall ~30 cm into the rack.
- Objects settle inside the rack within ~1 s.

If any of those don't happen cleanly, common knobs:
- **Objects clip the fingers as they release** → widen `_PIN_OFFSETS_POS` (move them further from gripper centerline).
- **Objects miss the rack** → re-tune `_FRANKA_BASE_*` so the carrying-pose EE is more directly above the rack center, OR drop the carrying-pose EE z target from 0.4 m to 0.3 m so they have less time to drift.
- **Arm overshoots / oscillates** → reduce `joint_target_ke` from 600 to 400 (and matching kd from 30 to 20).

---

## Self-Review Checklist

- **Spec coverage:**
  - Scene module with 5-function API + `update_franka_targets` + `update_held_objects` helpers: Task 2 ✓
  - LBM rack reused verbatim: Task 2 (`_add_drainer` copy) ✓
  - Franka FR3 URDF loaded with proper base xform from probe: Tasks 1+2 ✓
  - Mug + fork mesh collision (16-vert hull): Task 2 (`_add_held_objects`) ✓
  - Spatula visual mesh + box primitive collision: Task 2 (`_add_held_objects` spatula branch) ✓
  - Kinematic pin-then-release of held objects: Task 2 (`update_held_objects`) ✓
  - Smoothstep keyframe interpolation for arm targets: Task 2 (`_interpolate_keyframes` + `update_franka_targets`) ✓
  - Per-world held-object yaw jitter randomization: Task 2 (`build_model_randomized`) ✓
  - Demo with standard `--num-worlds`, `--num-steps`, `--headless`, `--tol` CLI: Task 3 ✓
  - Smoke tests N=1 + N=4: Tasks 3+4 ✓
- **Placeholder scan:** All Task 2 / Task 3 code blocks are complete. Task 1 has explicit TODO markers for the harvested probe values, by design (the operator runs the probe and substitutes the real numbers — these are not "TBD" abandonments, they are explicit harvest points with formulas). ✓
- **Type / signature consistency:** `update_franka_targets(model, control, sim_time)` and `update_held_objects(model, state, sim_time)` signatures match between Task 2 (definition) and Task 3 (caller). ✓
- **No commits:** No `git add` / `git commit` steps. ✓
