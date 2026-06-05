# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Franka FR3 drops mug, fork, spatula into the LBM dish drying rack.

The arm executes a scripted joint-space trajectory (smoothstep-interpolated
keyframes). Held objects are kinematically pinned to the gripper EE pose
until ``RELEASE_TIME``; after that they are free-falling rigid bodies that
settle into the rack via mesh / primitive contact with the LBM rack.

Geometry choices:
  * Rack          -- LBM glTF meshes with SDF collision (reused from
    scripts/scenes/dish_rack.py).
  * Franka FR3    -- URDF from newton.utils.download_asset("franka_emika_panda").
  * Mug           -- LBM glTF mesh, convex-hull collision (16-vert cap).
  * Fork          -- LBM glTF mesh, convex-hull collision (16-vert cap).
  * Spatula       -- YCB textured glTF as visual-only; box primitive matching
    the bbox handles collision.

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

DT_OUTER = 0.002
TOL = 1e-3
LOG_EVERY = 50

# --- Asset paths -----------------------------------------------------------

_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_LBM_ROOT = _ASSETS / "lbm"

_RACK_BASE_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_base.gltf"
_RACK_WIREFRAME_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_wireframe.gltf"
_RACK_UTENSIL_GLTF = _LBM_ROOT / "drying_racks" / "assets" / "sweet_home_dish_drying_rack_utensil_holder.gltf"

_LBM_MUG_DIR = _LBM_ROOT / "mugs"
_LBM_MUG_GLTF = _LBM_MUG_DIR / "assets" / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.gltf"
_LBM_MUG_SDF = _LBM_MUG_DIR / "mug_inomata_ceramic_dense_patterned_yellow_mesh_collision.sdf"

_LBM_FORK_DIR = _LBM_ROOT / "forks"
_LBM_FORK_GLTF = _LBM_FORK_DIR / "assets" / "cambridge_jubilee_stainless_plastic_black_fork.gltf"
_LBM_FORK_SDF = _LBM_FORK_DIR / "cambridge_jubilee_stainless_plastic_black_fork.sdf"

_YCB_SPATULA_OBJ = _ASSETS / "033_spatula_berkeley_meshes" / "033_spatula" / "tsdf" / "textured.obj"

# --- Welded transforms inside the LBM rack (copied from dish_rack.py) ------

_WIREFRAME_OFFSET_P = wp.vec3(-0.00527, -0.01075, 0.009861)
_WIREFRAME_OFFSET_PITCH_RAD = 0.013858
_UTENSIL_OFFSET_P = wp.vec3(-0.1518, 0.0735, 0.11335)
_UTENSIL_OFFSET_PITCH_RAD = math.radians(-4.5)

# --- Franka base placement (from Task 1 probe) -----------------------------
# With ARM_Q = [0, -0.3, 0, -2.2, 0, 1.9, 0.785] and base at origin, wrist
# (link7) lands at world (+0.46, 0, +0.61). Base at (-0.46, 0, 0) puts wrist
# over the rack center x=0, at z=0.61 -- 49 cm above the rack top (z=0.124).
_FRANKA_BASE_XFORM = wp.transform(
    wp.vec3(-0.46, 0.0, 0.0),
    wp.quat_identity(),
)

# Per-world body / joint counts (from Task 1 probe) -------------------------
# After collapse_fixed_joints: 7 arm links (link1..7) + 2 fingers = 9 bodies.
# Then we append 3 free held bodies. EE = link7 = body index 6.
_FRANKA_BODY_COUNT = 9
_FRANKA_JOINT_COUNT = 9  # 7 arm + 2 finger (each 1 coord / 1 dof)
_HELD_COUNT = 1  # mug only
_BODIES_PER_WORLD = _FRANKA_BODY_COUNT + _HELD_COUNT
_EE_BODY_IDX_PER_WORLD = 6  # link7 = wrist (NOT a finger)

# Per-world joint coords / dofs.
_COORDS_PER_WORLD = _FRANKA_JOINT_COUNT + _HELD_COUNT * 7
_DOFS_PER_WORLD = _FRANKA_JOINT_COUNT + _HELD_COUNT * 6

# --- Trajectory keyframes --------------------------------------------------
# Columns: t [s], joint1..joint7 [rad], finger1, finger2 [m].
# Carry pose held at t=0 and t=1.5 (arm still). Fingers open between t=1.5
# and t=1.7.  Release happens at t=1.7 (pin override stops).
_ARM_CARRYING = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.0]  # joint7=0 keeps wrist axis-aligned
_GRIPPER_CLOSED = [0.1, 0.1]    # fingers touching (max physical close)
_GRIPPER_OPEN = [0.04, 0.04]    # 4 cm per finger -- at joint limit (fully open)

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

# --- Pin offsets (held body pose relative to EE/link7 frame) ---------------
# Wrist orientation in world has gripper +z mapped to world -z (180 deg flip
# about y).  Mug centered ~13 cm below the wrist (gripper throat).
_PIN_OFFSETS_POS = [
    np.array([0.0, 0.0, 0.18], dtype=np.float64),  # mug centered, ~18 cm below wrist (gripper around mug body, not the rim)
]
_PIN_OFFSETS_QUAT = [
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),  # identity
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
        verts,
        faces,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
        maxhullvert=_HULL_BUDGET,
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
        verts,
        faces,
        uvs=uvs,
        texture=texture,
        compute_inertia=False,
        maxhullvert=_HULL_BUDGET,
    )
    _MESH_CACHE[obj_path] = mesh
    return mesh


def _add_drainer(builder: newton.ModelBuilder, cfg, voxel: float) -> None:
    """Static LBM dish rack (copy of scripts/scenes/dish_rack._add_drainer)."""
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
        body=-1,
        xform=wp.transform_identity(),
        mesh=base_mesh,
        cfg=cfg,
        scale=unit_scale,
    )
    builder.add_shape_mesh(
        body=-1,
        xform=wf_xform,
        mesh=wf_mesh,
        cfg=cfg,
        scale=unit_scale,
    )
    builder.add_shape_mesh(
        body=-1,
        xform=ut_xform,
        mesh=ut_mesh,
        cfg=cfg,
        scale=unit_scale,
    )


def _add_held_objects(builder: newton.ModelBuilder) -> None:
    """1 free-body held object: mug.

    Visual: the LBM textured glTF mesh (no collision).
    Collision: a cylinder primitive sized to the mesh bbox -- smooth contact
    manifold, no convex-hull discontinuities. Same trick as anymal_clutter.

    Collision_group=10 keeps it off the Franka (URDF default group); it still
    collides with the rack and ground (group=-1, collide with everything).
    """
    mug_body = builder.add_body(xform=wp.transform_identity())
    mug_mesh = _load_lbm_mesh(_LBM_MUG_GLTF, _LBM_MUG_SDF)

    # Visual only -- no contact participation.
    cfg_mug_visual = newton.ModelBuilder.ShapeConfig(
        ke=4e4,
        kd=400.0,
        mu=0.4,
        margin=5e-3,
        collision_group=10,
        has_shape_collision=False,
        has_particle_collision=False,
    )
    builder.add_shape_mesh(mug_body, mesh=mug_mesh, cfg=cfg_mug_visual)

    # Cylinder collision primitive matching the mesh bbox.
    bbox_lo = mug_mesh.vertices.min(axis=0)
    bbox_hi = mug_mesh.vertices.max(axis=0)
    centre = (bbox_hi + bbox_lo) * 0.5
    radius = float(max(bbox_hi[0] - bbox_lo[0], bbox_hi[1] - bbox_lo[1]) * 0.5)
    half_h = float((bbox_hi[2] - bbox_lo[2]) * 0.5)

    cfg_mug_col = newton.ModelBuilder.ShapeConfig(
        ke=4e4,
        kd=400.0,
        mu=0.4,
        margin=5e-3,
        collision_group=10,
        density=300.0,  # ceramic-ish density so mass+inertia are reasonable
        is_visible=False,
    )
    builder.add_shape_cylinder(
        mug_body,
        xform=wp.transform(
            p=wp.vec3(float(centre[0]), float(centre[1]), float(centre[2])),
            q=wp.quat_identity(),
        ),
        radius=radius,
        half_height=half_h,
        cfg=cfg_mug_col,
    )


# --- Public scene API ------------------------------------------------------


def build_template() -> newton.ModelBuilder:
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)

    template.default_shape_cfg.ke = 4e4
    template.default_shape_cfg.kd = 400.0
    template.default_shape_cfg.kf = 1e3
    template.default_shape_cfg.mu = 0.6

    cfg_rack = newton.ModelBuilder.ShapeConfig(
        ke=4e4,
        kd=400.0,
        mu=0.3,
        margin=5e-3,
        collision_group=-1,
        is_hydroelastic=False,
    )
    _add_drainer(template, cfg_rack, voxel=0.001)

    asset_path = newton.utils.download_asset("franka_emika_panda")
    template.add_urdf(
        str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
        xform=_FRANKA_BASE_XFORM,
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )
    arm_q = KEYFRAMES[0, 1:8]
    finger_q = KEYFRAMES[0, 8:10]
    for i in range(7):
        template.joint_q[i] = float(arm_q[i])
    for i in range(2):
        template.joint_q[7 + i] = float(finger_q[i])
    for i in range(7):
        template.joint_target_ke[i] = 600.0
        template.joint_target_kd[i] = 30.0
    for i in (7, 8):
        template.joint_target_ke[i] = 400.0
        template.joint_target_kd[i] = 20.0

    _add_held_objects(template)

    return template


def build_model(n_worlds: int) -> newton.Model:
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    ground_cfg = newton.ModelBuilder.ShapeConfig(collision_group=-1)
    builder.add_ground_plane(cfg=ground_cfg)
    return builder.finalize()


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """Per-world held-object yaw jitter; pose is overwritten by the kinematic
    pin on the first outer step anyway."""
    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        base = w * _COORDS_PER_WORLD
        held_start = base + _FRANKA_JOINT_COUNT
        for i in range(_HELD_COUNT):
            yaw = rng.uniform(-math.radians(15.0), math.radians(15.0))
            cz = math.cos(yaw / 2.0)
            sz = math.sin(yaw / 2.0)
            qx_idx = held_start + i * 7 + 3
            joint_q_np[qx_idx + 0] = 0.0
            joint_q_np[qx_idx + 1] = 0.0
            joint_q_np[qx_idx + 2] = sz
            joint_q_np[qx_idx + 3] = cz
    model.joint_q.assign(joint_q_np)
    return model


def make_solver(
    model: newton.Model,
    tol: float | None = None,
) -> newton.solvers.SolverMuJoCoAdaptive:
    return newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=TOL if tol is None else tol,
        dt_init=0.005,
        dt_min=1e-6,
        dt_max=0.01,
        nconmax=8192,
        njmax=16384,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model,
        separate_worlds=True,
        nconmax=8192,
        njmax=16384,
        cone="elliptic",
        iterations=100,
        impratio=10.0,
        ccd_iterations=100,
        solver="newton",
    )


# --- Runtime helpers (called from the demo loop) ---------------------------


def _interpolate_keyframes(t: float) -> np.ndarray:
    if t <= KEYFRAMES[0, 0]:
        return KEYFRAMES[0, 1:].copy()
    if t >= KEYFRAMES[-1, 0]:
        return KEYFRAMES[-1, 1:].copy()
    for i in range(len(KEYFRAMES) - 1):
        t0 = KEYFRAMES[i, 0]
        t1 = KEYFRAMES[i + 1, 0]
        if t0 <= t < t1:
            alpha = (t - t0) / (t1 - t0)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            return (1.0 - alpha) * KEYFRAMES[i, 1:] + alpha * KEYFRAMES[i + 1, 1:]
    return KEYFRAMES[-1, 1:].copy()


def update_franka_targets(model, control, sim_time: float) -> None:
    target = _interpolate_keyframes(sim_time)
    n_worlds = model.world_count
    targets_np = control.joint_target_pos.numpy()
    for w in range(n_worlds):
        base = w * _DOFS_PER_WORLD
        for i in range(_FRANKA_JOINT_COUNT):
            targets_np[base + i] = float(target[i])
    control.joint_target_pos.assign(targets_np)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    qx, qy, qz, qw = q
    qvec = np.array([qx, qy, qz])
    t = 2.0 * np.cross(qvec, v)
    return v + qw * t + np.cross(qvec, t)


def update_held_objects(model, state, sim_time: float) -> None:
    if sim_time >= RELEASE_TIME:
        return

    n_worlds = model.world_count
    body_q_np = state.body_q.numpy()
    joint_q_np = state.joint_q.numpy()
    joint_qd_np = state.joint_qd.numpy()

    for w in range(n_worlds):
        ee_idx = w * _BODIES_PER_WORLD + _EE_BODY_IDX_PER_WORLD
        ee_pos = body_q_np[ee_idx, :3].astype(np.float64)
        ee_quat = body_q_np[ee_idx, 3:7].astype(np.float64)

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
