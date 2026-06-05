# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""ANYmal C walking through YCB-object clutter.

Reproduces the robot setup from scripts/control/cenic_step_anymal_walk.py
(URDF, joint config, foot-sphere fix, initial pose, actuator gains) and
adds a tight cluster of YCB benchmark objects (mini soccer ball, cracker
box, wood block) directly in the robot's walking path.

The scene module is intentionally torch-free; the walking policy and
observation pipeline live in scripts/demos/anymal_clutter.py so this
module can be imported by benchmarks without pulling torch.
"""

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers
import newton.utils
from newton import GeoType

DT_OUTER = 0.002  # 2 ms -- policy / control cadence [s]
TOL = 1e-3
LOG_EVERY = 5

# --- YCB clutter ------------------------------------------------------------
#
# Robot spawns at origin with a 90 deg z rotation; body-frame +x (forward)
# maps to world +y. Cluster the YCB objects ~1 m in front of the robot so
# they are right in its walking path.

_ASSETS = Path(__file__).resolve().parents[1] / "assets"

# YCB meshes have their LOCAL origin at the bottom of the bounding box, so
# placing a body at world z=0 puts the object sitting on the ground; no
# per-object z offset is needed.
#
# Density [kg/m^3] is chosen per object so volume * density gives a sensible
# mass and a CONSISTENT inertia tensor; passing mass= to add_body would
# override mass without rescaling inertia, producing extreme accelerations
# and visual flicker on contact.
#
# (mesh subpath, label, density [kg/m^3], xy offset from cluster centre [m],
#  yaw [rad])
_YCB_OBJECTS = [
    (
        "053_mini_soccer_ball_berkeley_meshes/053_mini_soccer_ball/tsdf",
        "soccer_ball",
        100.0,           # ~140 g for the ~1.4 L ball -- realistic mini ball
        (-0.30, 0.00),
        0.0,
    ),
    (
        "003_cracker_box_berkeley_meshes/003_cracker_box/tsdf",
        "cracker_box",
        120.0,           # cardboard + crackers -> ~430 g for ~3.6 L
        (0.20, 0.20),
        math.radians(20),
    ),
    (
        "036_wood_block_berkeley_meshes/036_wood_block/tsdf",
        "wood_block",
        80.0,            # lighter than real wood so the robot can punt it
        (0.20, -0.20),
        math.radians(-15),
    ),
]

# Cluster centre xy in world coords (in front of robot).
CLUTTER_CENTER = wp.vec3(0.0, 1.2, 0.0)

# Per-shape friction / contact stiffness for the YCB objects.
_CLUTTER_KE = 1e4
_CLUTTER_KD = 200.0
_CLUTTER_MU = 0.6
_CLUTTER_MARGIN = 5e-3

# --- Robot initial joint pose -----------------------------------------------

INITIAL_Q = {
    "RH_HAA": 0.0,
    "RH_HFE": -0.4,
    "RH_KFE": 0.8,
    "LH_HAA": 0.0,
    "LH_HFE": -0.4,
    "LH_KFE": 0.8,
    "RF_HAA": 0.0,
    "RF_HFE": 0.4,
    "RF_KFE": -0.8,
    "LF_HAA": 0.0,
    "LF_HFE": 0.4,
    "LF_KFE": -0.8,
}


_MESH_CACHE: dict[Path, newton.Mesh] = {}


def _load_ycb_mesh(mesh_dir: Path) -> newton.Mesh:
    """Load YCB textured.obj as a :class:`newton.Mesh`.

    YCB processed meshes are Z-up and metres -- no axis swap or rescale needed.
    The TSDF reconstruction is closed and watertight, suitable for both visual
    and (via convex hull) collision.
    """
    obj_path = mesh_dir / "textured.obj"
    if obj_path in _MESH_CACHE:
        return _MESH_CACHE[obj_path]

    import trimesh

    if not obj_path.exists():
        raise FileNotFoundError(f"YCB mesh not found: {obj_path}")

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

    # Cap convex-hull vertex count for collision -- YCB TSDF meshes have
    # thousands of triangles; without this cap MuJoCo Warp generates more
    # constraints than nconmax/njmax can accommodate. 16 keeps each shape
    # cheap while still preserving recognizable silhouettes.
    # compute_inertia=False -- ShapeConfig.density on the body drives both
    # mass and inertia from the actual mesh volume, consistently.
    mesh = newton.Mesh(
        verts, faces, uvs=uvs, texture=texture,
        compute_inertia=False, maxhullvert=16,
    )
    _MESH_CACHE[obj_path] = mesh
    return mesh


def _add_clutter(builder: newton.ModelBuilder, rng) -> None:
    """Cluster YCB objects directly in the robot's walking path.

    rng controls per-world yaw jitter so each replicated world looks slightly
    different without changing the cluster identity.
    """
    # Visual-mesh config: textured YCB mesh, NO collision (collision is
    # handled by a primitive shape on the same body for fast/clean contact).
    cfg_visual = newton.ModelBuilder.ShapeConfig(
        ke=_CLUTTER_KE, kd=_CLUTTER_KD, mu=_CLUTTER_MU,
        margin=_CLUTTER_MARGIN,
        has_shape_collision=False, has_particle_collision=False,
    )

    for mesh_subpath, label, density, rel_xy, base_yaw in _YCB_OBJECTS:
        mesh_dir = _ASSETS / mesh_subpath
        mesh = _load_ycb_mesh(mesh_dir)

        # Add a small per-world yaw jitter so worlds diverge gently.
        yaw = base_yaw + rng.uniform(-math.radians(15), math.radians(15))
        cz = math.cos(yaw / 2.0)
        sz = math.sin(yaw / 2.0)
        pos = wp.vec3(
            CLUTTER_CENTER[0] + rel_xy[0],
            CLUTTER_CENTER[1] + rel_xy[1],
            0.0,  # YCB mesh local origin is at object bottom -- z=0 sits on ground
        )
        body = builder.add_body(
            xform=wp.transform(p=pos, q=wp.quat(0.0, 0.0, sz, cz)),
        )

        # Visual: full textured mesh.
        builder.add_shape_mesh(body, mesh=mesh, cfg=cfg_visual)

        # Collision: primitive whose shape and dimensions match the mesh
        # bounding box.  Origin offset because the mesh's local origin sits
        # at the bottom of the bounding box rather than its centre.
        bbox_lo, bbox_hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
        size = bbox_hi - bbox_lo
        centre = (bbox_hi + bbox_lo) * 0.5
        cfg_col = newton.ModelBuilder.ShapeConfig(
            ke=_CLUTTER_KE, kd=_CLUTTER_KD, mu=_CLUTTER_MU,
            margin=_CLUTTER_MARGIN, density=density, is_visible=False,
        )
        col_xform = wp.transform(
            p=wp.vec3(float(centre[0]), float(centre[1]), float(centre[2])),
            q=wp.quat_identity(),
        )
        if label == "soccer_ball":
            radius = float(min(size) * 0.5)  # ball is roughly spherical
            builder.add_shape_sphere(body, xform=col_xform, radius=radius, cfg=cfg_col)
        else:
            builder.add_shape_box(
                body, xform=col_xform,
                hx=float(size[0] * 0.5), hy=float(size[1] * 0.5), hz=float(size[2] * 0.5),
                cfg=cfg_col,
            )


def build_template(seed: int = 42) -> newton.ModelBuilder:
    """ANYmal C robot + per-world cube scatter (this template is one world)."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)

    template.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06,
        limit_ke=1.0e3,
        limit_kd=1.0e1,
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

    # Clutter scattered in front of the robot. Per-template -- each replicated
    # world will share the same scatter pattern from build_model; build_model_randomized
    # below regenerates per-world for distinct layouts.
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


# mjwarp multiplies nconmax/njmax by nworld internally — pass per-world values.
_NCON = 200
_NJM = 600


def make_solver(
    model: newton.Model,
    tol: float = TOL,
) -> newton.solvers.SolverMuJoCoAdaptive:
    return newton.solvers.SolverMuJoCoAdaptive(
        model, tol=tol, dt_init=0.005, dt_min=2e-7,
        dt_max=0.02, nconmax=_NCON, njmax=_NJM,
        solver="newton", ls_parallel=False, ls_iterations=50,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=_NCON, njmax=_NJM,
        solver="newton", ls_parallel=False, ls_iterations=50,
    )


from scripts.scenes import _solvers as _s  # noqa: E402

# Only mujoco-family for anymal — articulated robot + meshes break XPBD/Featherstone.
SOLVER_FACTORIES: dict = {
    "mujoco_adaptive_1e-3": _s.mujoco_adaptive_factory(
        tol=1e-3, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
        dt_inner_min=2e-7, dt_inner_max=0.02,
    ),
    "mujoco_adaptive_1e-2": _s.mujoco_adaptive_factory(
        tol=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
        dt_inner_min=2e-7, dt_inner_max=0.02,
    ),
    "mujoco_fixed_1ms": _s.mujoco_fixed_factory(
        dt=1e-3, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
    "mujoco_fixed_10ms": _s.mujoco_fixed_factory(
        dt=1e-2, nconmax=_NCON, njmax=_NJM, dt_outer=DT_OUTER,
    ),
}
