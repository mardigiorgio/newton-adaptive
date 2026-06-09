# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

Same 3 spheres + 3 boxes as the benchmark, but dropped into a small box so they
settle into a heap in mutual contact, the visual the poster needs.
"""
import argparse, math
import numpy as np, warp as wp
from PIL import Image
import newton, newton.solvers
from scripts.scenes.contact_objects import (
    SPHERE_RADIUS, BOX_HALF, OBJ_KE, OBJ_KD, OBJ_MU, DT_OUTER, DT_INNER_MIN, TOL)


def quat_xyz(ax, ay, az):
    rx, ry, rz = math.radians(ax), math.radians(ay), math.radians(az)
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    return wp.quat(sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
                   cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settle", type=int, default=160)
    ap.add_argument("--half", type=float, default=0.150, help="container inner half-extent [m]")
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.50, -0.56, 0.40])
    ap.add_argument("--cam-pitch", type=float, default=-27.0)
    ap.add_argument("--cam-yaw", type=float, default=126.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--out", default="/tmp/pile_hr.png")
    args = ap.parse_args()

    wp.init()
    # Objects go in a template (replicated into world 0); statics on the main builder.
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(template)
    cfg = newton.ModelBuilder.ShapeConfig(ke=OBJ_KE, kd=OBJ_KD, mu=OBJ_MU, margin=0.005)
    # tight cluster, staggered heights so they drop onto each other and pile
    placements = [
        ("b", (0.00, 0.00, 0.07), (10, 5, 0)),
        ("s", (0.10, 0.03, 0.18), None),
        ("b", (-0.08, 0.06, 0.22), (20, 0, 15)),
        ("s", (0.02, -0.09, 0.33), None),
        ("b", (0.06, 0.07, 0.45), (-15, 10, 0)),
        ("s", (-0.05, -0.03, 0.56), None),
    ]
    for kind, (px, py, pz), ang in placements:
        q = wp.quat_identity() if ang is None else quat_xyz(*ang)
        body = template.add_body(xform=wp.transform(p=wp.vec3(px, py, pz), q=q))
        if kind == "s":
            template.add_shape_sphere(body, radius=SPHERE_RADIUS, cfg=cfg)
        else:
            template.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)

    b = newton.ModelBuilder()
    b.replicate(template, 1)
    b.add_ground_plane()
    cfg_wall = newton.ModelBuilder.ShapeConfig(ke=OBJ_KE, kd=OBJ_KD, mu=OBJ_MU, margin=0.005, is_visible=False)
    hi, wt, wh = args.half, 0.02, 0.5
    for px, py, hx, hy in [(-(hi + wt), 0, wt, hi + wt), (hi + wt, 0, wt, hi + wt),
                           (0, -(hi + wt), hi + wt, wt), (0, hi + wt, hi + wt, wt)]:
        b.add_shape_box(body=-1, xform=wp.transform(p=wp.vec3(px, py, wh), q=wp.quat_identity()),
                        hx=hx, hy=hy, hz=wh, cfg=cfg_wall)
    b.color()
    model = b.finalize()

    s0, s1, ctrl = model.state(), model.state(), model.control()
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model, tol=TOL, dt_init=DT_OUTER, dt_min=DT_INNER_MIN, dt_max=DT_OUTER,
        nconmax=128, njmax=640, use_mujoco_contacts=True)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(*args.cam_pos), pitch=args.cam_pitch, yaw=args.cam_yaw)
    t = 0.0
    for _ in range(args.settle):
        solver.step(s0, s1, ctrl, None, DT_OUTER)
        t += DT_OUTER
    wp.synchronize()
    viewer.begin_frame(t)
    viewer.log_state(s0)
    viewer.end_frame()
    Image.fromarray(viewer.get_frame().numpy(), "RGB").save(args.out)
    print("saved", args.out, "obj z:", np.round(s0.body_q.numpy()[:, 2], 3))


if __name__ == "__main__":
    main()
