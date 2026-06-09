# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""High-res render of the contact_objects benchmark scene for the poster.

Settles 3 spheres + 3 boxes on the ground, then captures one frame via a headless
ViewerGL offscreen FBO. Camera params are CLI args for tuning. Crop to the objects
afterwards (e.g. with PIL) for the poster's benchmark-scene figure.

Example::

    uv run python -m scripts.demos.clutter_figure --width 3200 --height 2000 \\
        --out poster/poster_finalv2/CLUTTER_hr.png
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from PIL import Image

import newton
import newton.solvers

from scripts.scenes.contact_objects import DT_OUTER, build_model_randomized, make_solver


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--settle", type=int, default=28, help="outer steps; ~12-30 is the contact phase (dt plunges), 90+ is at rest")
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.5, -0.6, 0.42])
    ap.add_argument("--cam-pitch", type=float, default=-30.0)
    ap.add_argument("--cam-yaw", type=float, default=125.0)
    ap.add_argument("--width", type=int, default=3200)
    ap.add_argument("--height", type=int, default=2000)
    ap.add_argument("--out", default="/tmp/clutter_hr.png")
    args = ap.parse_args()

    wp.init()
    model = build_model_randomized(1, seed=args.seed)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    solver = make_solver(model)
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
    frame = viewer.get_frame().numpy()
    Image.fromarray(frame, mode="RGB").save(args.out)
    print("saved", args.out, frame.shape, "obj z:", np.round(s0.body_q.numpy()[:, 2], 3))


if __name__ == "__main__":
    main()
