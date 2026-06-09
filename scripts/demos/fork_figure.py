# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Headless film-strip renderer for the fork-on-rack poster figures.

Drops a single fork onto the dish rack and captures PNGs at chosen frames, for
either solver. Because the fall is deterministic, fixed (``mujoco``) and adaptive
(``cenic``) reach the rack at the same sim time -- render both at the same frame
and the contrast is fixed-penetrates vs adaptive-rests. Prints the fork z-height
at each captured frame so the impact frame is obvious. Requires a display
(pyglet/GLX); renders to an offscreen FBO.

Example::

    uv run python -m scripts.demos.fork_figure --solver mujoco --seed 42 \\
        --frames 12 16 20 24 28 32 --out-prefix /tmp/fork_fixed
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp
from PIL import Image

import newton
import newton.solvers

from scripts.scenes.dish_rack import (
    DRAINER_TOP_Z,
    DT_OUTER,
    FIXED_DT_INNER,
    build_model_randomized,
    make_pipeline,
    make_solver,
    make_solver_fixed,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", choices=("cenic", "mujoco"), required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frames", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28, 32, 40])
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.85, -0.95, 0.70])
    ap.add_argument("--cam-pitch", type=float, default=-25.0)
    ap.add_argument("--cam-yaw", type=float, default=135.0)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1200)
    args = ap.parse_args()

    model = build_model_randomized(1, seed=args.seed)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    solver = make_solver(model) if args.solver == "cenic" else make_solver_fixed(model)

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(*args.cam_pos), pitch=args.cam_pitch, yaw=args.cam_yaw)

    pipeline, contacts = make_pipeline(model, solver)
    if args.solver == "cenic":
        contacts = solver.contacts

    nsub = int(round(DT_OUTER / FIXED_DT_INNER))
    cap = set(args.frames)
    t = 0.0
    print(f"solver={args.solver}  rack_top_z={DRAINER_TOP_Z:.3f} m", flush=True)
    for step in range(1, max(args.frames) + 1):
        if args.solver == "cenic":
            solver.step(s0, s1, ctrl, None, DT_OUTER)
        else:
            for _ in range(nsub):
                s0.clear_forces()
                pipeline.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, FIXED_DT_INNER)
                s0, s1 = s1, s0
        t += DT_OUTER
        viewer.begin_frame(t)
        viewer.log_state(s0)
        viewer.log_contacts(contacts, s0)
        viewer.end_frame()
        if step in cap:
            frame = viewer.get_frame().numpy()
            Image.fromarray(frame, mode="RGB").save(f"{args.out_prefix}_f{step:03d}.png")
            fork_z = float(s0.body_q.numpy()[0][2])
            print(f"  frame {step:3d}: fork_z={fork_z:+.4f} m  saved {args.out_prefix}_f{step:03d}.png", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
