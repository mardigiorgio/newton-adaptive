# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact objects scene: N_SPHERES_BUILD spheres + N_BOXES_BUILD tilted boxes
per world (default 3 + 3 = 6, on a shared 3x3 grid).

Shared scene definition used by demos and benchmarks. No main(), no CLI,
no viewer logic.
"""

import math

import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.02  # 50 Hz control / render cadence [s] -- gives adaptive room above the fixed_10ms baseline
TOL = 1e-3
DT_INNER_MIN = 1e-6
LOG_EVERY = 250

SPHERE_RADIUS = 0.050
BOX_HALF = 0.050
GRID_STEP = 0.200
GRID_OFFSETS = [-GRID_STEP, 0.0, GRID_STEP]
Z_SPHERES = 1.00
Z_BOXES = 1.25

# Contact material. Stiffness sets the contact timescale: softer contact (lower
# OBJ_KE) has a longer timescale, so the adaptive controller can take larger
# stable steps where the fixed-dt baseline cannot -- this is the knob that opens
# the adaptive-vs-fixed savings on this scene. Trade-off: too soft visibly
# interpenetrates. Tune against penetration (see scripts/bench/buffer_autosize
# and the difficulty sweep). [N/m], [N*s/m], dimensionless.
OBJ_KE = 1e5
OBJ_KD = 500.0
OBJ_MU = 0.3

# Object counts. Fewer / less-crowded objects settle to rest instead of jostling
# forever, which is what lets the adaptive controller grow its step (and beat the
# fixed-dt baseline). Capped at the 9-slot (3x3) grid. Spheres are added before
# boxes, so N_SPHERES_BUILD also marks the sphere/box split for rejection-sampling.
N_SPHERES_BUILD = 3
N_BOXES_BUILD = 3


def build_template() -> newton.ModelBuilder:
    """Single-world template: ``N_SPHERES_BUILD`` spheres + ``N_BOXES_BUILD``
    tilted boxes (default 3 + 3), taken from the front of a shared 3x3 grid."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)

    cfg_obj = newton.ModelBuilder.ShapeConfig(ke=OBJ_KE, kd=OBJ_KD, mu=OBJ_MU, margin=0.005)

    _grid = [(ox, oy) for ox in GRID_OFFSETS for oy in GRID_OFFSETS]
    for (ox, oy) in _grid[:N_SPHERES_BUILD]:
        b = template.add_body(
            xform=wp.transform(p=wp.vec3(ox, oy, Z_SPHERES), q=wp.quat_identity()),
        )
        template.add_shape_sphere(b, radius=SPHERE_RADIUS, cfg=cfg_obj)

    _box_angles = [
        (15, 0, 0),
        (-20, 10, 0),
        (35, 0, 15),
        (0, 25, -10),
        (49, 0, 0),
        (-30, 20, 5),
        (10, -35, 0),
        (0, 15, 40),
        (-15, 0, -25),
    ]
    for (ox, oy), (ax, ay, az) in zip(
        _grid[:N_BOXES_BUILD],
        _box_angles[:N_BOXES_BUILD],
    ):
        rx, ry, rz = math.radians(ax), math.radians(ay), math.radians(az)
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        q = wp.quat(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
        b = template.add_body(xform=wp.transform(p=wp.vec3(ox, oy, Z_BOXES), q=q))
        template.add_shape_box(b, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_obj)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane + invisible walls."""
    template = build_template()
    builder = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()

    cfg_wall = newton.ModelBuilder.ShapeConfig(ke=OBJ_KE, kd=OBJ_KD, mu=OBJ_MU, margin=0.005, is_visible=False)
    half_inner = 0.350
    wt = 0.025
    wh = 0.750
    for px, py, hx, hy in [
        (-(half_inner + wt), 0.0, wt, half_inner + wt),
        (half_inner + wt, 0.0, wt, half_inner + wt),
        (0.0, -(half_inner + wt), half_inner + wt, wt),
        (0.0, half_inner + wt, half_inner + wt, wt),
    ]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(px, py, wh), q=wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=wh,
            cfg=cfg_wall,
        )
    # Required by SolverVBD; harmless for other solvers.
    builder.color()
    return builder.finalize()


def build_model_perturbed(n_worlds: int, epsilon: float = 1e-4) -> newton.Model:
    """N replicated worlds with deterministic per-world z-perturbation.

    Each world's bodies get z-offset = world_index * epsilon [m].
    World 0 is unperturbed (identical to build_model output).
    """
    model = build_model(n_worlds)

    # Perturb joint_q on CPU, then write back.
    joint_q_np = model.joint_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            z_idx = w * coords_per_world + b * 7 + 2  # z-component of position
            joint_q_np[z_idx] += offset

    model.joint_q.assign(joint_q_np)

    # Also perturb body_q (used by renderer / solver sync).
    body_q_np = model.body_q.numpy()
    for w in range(n_worlds):
        offset = w * epsilon
        for b in range(bodies_per_world):
            body_idx = w * bodies_per_world + b
            body_q_np[body_idx][2] += offset  # z component of position in transform

    model.body_q.assign(body_q_np)

    return model


def _random_unit_quaternion(rng) -> tuple[float, float, float, float]:
    """Sample a uniform random rotation quaternion (Shoemake's method)."""
    import numpy as np

    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    s1 = math.sqrt(1.0 - u1)
    s2 = math.sqrt(u1)
    a1 = 2.0 * math.pi * u2
    a2 = 2.0 * math.pi * u3
    return (s1 * math.sin(a1), s1 * math.cos(a1), s2 * math.sin(a2), s2 * math.cos(a2))


#: Clearance beyond touching when spawning objects [m]. Keeps the initial state
#: free of contact so the box GJK never starts in a deep-overlap degeneracy.
SPAWN_GAP = 0.01


def build_model_randomized(n_worlds: int, seed: int = 42) -> newton.Model:
    """N replicated worlds with randomized, NON-OVERLAPPING object positions.

    Each world gets deterministically randomized xyz positions and orientations
    for all bodies (N_SPHERES_BUILD spheres + N_BOXES_BUILD boxes), placed by rejection sampling so no
    two objects interpenetrate at spawn. Bounding-sphere radii are used for the
    overlap test (boxes use their corner distance ``half*sqrt(3)``), so boxes
    never interpenetrate regardless of orientation. This avoids the deep-overlap
    initial conditions that put MuJoCo's convex (GJK) collision into degenerate,
    non-converging configurations during the drop (the ``opt.ccd_iterations``
    warnings) and the unphysical stuck-overlap contacts they produce.

    Args:
        n_worlds: Number of parallel worlds.
        seed: Base RNG seed. World w uses seed + w.
    """
    import numpy as np

    model = build_model(n_worlds)

    joint_q_np = model.joint_q.numpy()
    body_q_np = model.body_q.numpy()
    coords_per_world = model.joint_coord_count // n_worlds
    bodies_per_world = model.body_count // n_worlds

    # Bounds: stay inside the walled enclosure with margin.
    xy_lo, xy_hi = -0.25, 0.25
    z_lo, z_hi = 0.15, 1.50

    sphere_bound = SPHERE_RADIUS
    box_bound = BOX_HALF * np.sqrt(3.0)  # corner distance: box bounding sphere

    def _bound(body_idx: int) -> float:
        return sphere_bound if body_idx < N_SPHERES_BUILD else box_bound

    max_tries = 400
    overflow = 0
    for w in range(n_worlds):
        rng = np.random.default_rng(seed + w)
        placed: list[tuple[float, float, float, float]] = []  # (x, y, z, bound)

        for b in range(bodies_per_world):
            rb = _bound(b)
            x = y = z = 0.0
            for _ in range(max_tries):
                x = rng.uniform(xy_lo, xy_hi)
                y = rng.uniform(xy_lo, xy_hi)
                z = rng.uniform(z_lo, z_hi)
                clear = True
                for (px, py, pz, pr) in placed:
                    min_d = rb + pr + SPAWN_GAP
                    if (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2 < min_d * min_d:
                        clear = False
                        break
                if clear:
                    break
            else:
                overflow += 1  # could not find a clear spot (scene too dense)
            placed.append((x, y, z, rb))
            qx, qy, qz, qw = _random_unit_quaternion(rng)

            # joint_q: 7 floats per body [px, py, pz, qx, qy, qz, qw]
            base = w * coords_per_world + b * 7
            joint_q_np[base + 0] = x
            joint_q_np[base + 1] = y
            joint_q_np[base + 2] = z
            joint_q_np[base + 3] = qx
            joint_q_np[base + 4] = qy
            joint_q_np[base + 5] = qz
            joint_q_np[base + 6] = qw

            # body_q: transform [px, py, pz, qx, qy, qz, qw]
            body_idx = w * bodies_per_world + b
            body_q_np[body_idx] = (x, y, z, qx, qy, qz, qw)

    if overflow:
        import warnings
        warnings.warn(
            f"build_model_randomized: {overflow} object placements could not find "
            f"a non-overlapping spot in {max_tries} tries; scene may be too dense.",
            stacklevel=2,
        )

    model.joint_q.assign(joint_q_np)
    model.body_q.assign(body_q_np)

    return model


def make_solver(
    model: newton.Model,
    tol: float = TOL,
) -> newton.solvers.SolverMuJoCoAdaptive:
    """CENIC solver with canonical contact-demo parameters.

    Uses mjwarp's native contact pipeline (use_mujoco_contacts=True) since this
    scene is all primitives. Avoids the Newton SAP pipeline's O(N*nconmax)
    allocation that overflows int32 above N=2048.

    Args:
        model: The model to simulate.
        tol: Inf-norm error tolerance on joint_q per world.
    """
    return newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=tol,
        dt_init=DT_OUTER,
        dt_min=DT_INNER_MIN,
        dt_max=DT_OUTER,
        nconmax=128,
        njmax=640,
        use_mujoco_contacts=True,
    )


def make_fixed_solver(model: newton.Model) -> newton.solvers.SolverMuJoCo:
    """Fixed-step SolverMuJoCo with matching contact parameters."""
    return newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=128, njmax=640,
    )


# --- Multi-solver factories for cross-solver benchmarks ----------------------
# Legacy fallback budgets (mjwarp multiplies per-world by nworld). These are
# ~3-8x oversized for this scene's actual peak (measured: ~30-37 contacts/world,
# worst-world nefc ~200-260); they remain only as the fallback for kinds absent
# from BUFFER_TABLE. njmax=600 OOMs the scaling sweep before N=2^14.
_NCON = 200
_NJM = 600

# --- Finely-tuned per-(solver, N) MuJoCo buffer sizes ------------------------
# njmax (the per-world constraint cap) dominates GPU memory: it sizes efc_J ~
# njmax*nv*nworld, the largest mjw_data array. The stock njmax=600 exceeds 16 GB
# by N~2^14 and makes the whole scaling sweep OOM; right-sizing to the measured
# worst-world nefc (+margin) lets every solver reach 2^14 on a 16 GB GPU.
# Values measured by ``scripts/bench/buffer_autosize.py`` (seed-robust peak over
# a 50-step episode x 1.5 margin). (nconmax, njmax) per (kind, N). HAND-TUNE
# FREELY: raise njmax if mjwarp warns "nefc exceeded" / drops constraints; lower
# it to save memory. nconmax is the cheap shared-contact-pool multiplier.
# Measured under the 6-object (3+3) non-overlapping scene by buffer_autosize
# (seed-robust 50-step peak x1.5): ~14 contacts/world, worst-world nefc ~72.
# Tiny and ~N-independent, so 2^14 fits trivially for every solver including the
# adaptive compaction tiers. Re-run buffer_autosize if you change the object
# count or stiffness (OBJ_KE / N_*_BUILD).
BUFFER_TABLE: dict[str, dict[int, tuple[int, int]]] = {
    "mujoco_fixed_1ms":     {256: (24, 112), 2048: (24, 112), 8192: (24, 112), 16384: (24, 128)},
    "mujoco_fixed_10ms":    {256: (24, 112), 2048: (24, 112), 8192: (24, 112), 16384: (24, 112)},
    "mujoco_adaptive_1e-3": {256: (24, 112), 2048: (24, 112), 8192: (24, 112), 16384: (24, 128)},
    "mujoco_adaptive_1e-2": {256: (24, 112), 2048: (24, 112), 8192: (24, 112), 16384: (24, 128)},
}


def buffer_sizes(kind: str, n: int) -> tuple[int, int]:
    """Minimal-safe ``(nconmax, njmax)`` for a solver ``kind`` at ``n`` worlds.

    Looks up :data:`BUFFER_TABLE`; for ``n`` between measured points it
    interpolates ``njmax`` linearly in ``log2(n)`` (``nconmax`` is ~N-independent)
    and clamps outside the measured range. Unknown kinds fall back to the legacy
    ``(_NCON, _NJM)``.
    """
    import math
    table = BUFFER_TABLE.get(kind)
    if not table:
        return _NCON, _NJM
    if n in table:
        return table[n]
    ns = sorted(table)
    if n <= ns[0]:
        return table[ns[0]]
    if n >= ns[-1]:
        return table[ns[-1]]
    lo = max(k for k in ns if k <= n)
    hi = min(k for k in ns if k >= n)
    (c_lo, j_lo), (c_hi, j_hi) = table[lo], table[hi]
    t = (math.log2(n) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    njm = int(math.ceil((j_lo + t * (j_hi - j_lo)) / 16) * 16)
    return max(c_lo, c_hi), njm


def _buf_ncon(kind: str):
    return lambda n: buffer_sizes(kind, n)[0]


def _buf_njm(kind: str):
    return lambda n: buffer_sizes(kind, n)[1]


from scripts.scenes import _solvers as _s  # noqa: E402
from scripts.adaptive import factories as _af  # noqa: E402

SOLVER_FACTORIES: dict = {
    "mujoco_adaptive_1e-3": _s.mujoco_adaptive_factory(
        tol=1e-3, nconmax=_buf_ncon("mujoco_adaptive_1e-3"), njmax=_buf_njm("mujoco_adaptive_1e-3"),
        dt_outer=DT_OUTER, use_mujoco_contacts=True,
    ),
    "mujoco_adaptive_1e-2": _s.mujoco_adaptive_factory(
        tol=1e-2, nconmax=_buf_ncon("mujoco_adaptive_1e-2"), njmax=_buf_njm("mujoco_adaptive_1e-2"),
        dt_outer=DT_OUTER, use_mujoco_contacts=True,
    ),
    "mujoco_fixed_1ms": _s.mujoco_fixed_factory(
        dt=1e-3, nconmax=_buf_ncon("mujoco_fixed_1ms"), njmax=_buf_njm("mujoco_fixed_1ms"),
        dt_outer=DT_OUTER,
    ),
    "mujoco_fixed_10ms": _s.mujoco_fixed_factory(
        dt=1e-2, nconmax=_buf_ncon("mujoco_fixed_10ms"), njmax=_buf_njm("mujoco_fixed_10ms"),
        dt_outer=DT_OUTER,
    ),
    # Featherstone uses spring-damper contacts that go NaN on the bench's
    # randomized ICs (objects spawn slightly interpenetrating). Stable on the
    # deterministic build_model() but not build_model_randomized().
    # "featherstone_1ms": _s.featherstone_factory(dt=1e-3, dt_outer=DT_OUTER),
    "semi_implicit_1ms": _s.semi_implicit_factory(dt=1e-3, dt_outer=DT_OUTER),
    "xpbd_1ms": _s.xpbd_factory(dt=1e-3, dt_outer=DT_OUTER),
    "vbd_1ms": _s.vbd_factory(dt=1e-3, dt_outer=DT_OUTER),
    "xpbd_adaptive_1e-3": _af.adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
    "semi_adaptive_1e-3": _af.adaptive_semi_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
}
