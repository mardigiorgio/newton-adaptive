# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the Part-1 test scenes (the paper's Fig. 6 idea): each scene at
its initial state and after settling under ICF error control (eps 1e-3),
drawn from the simulation state -- spheres, oriented cubes, the bin, the
ball. Writes scripts/bench/results/figures/scenes.{png,pdf}.

    uv run python scripts/bench/part1_scenes_figure.py
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from scripts.bench.four_arms import build_model, make_arm  # noqa: E402
from scripts.scenes.cenic_scenes import (  # noqa: E402
    BALL_R,
    BIN_HALF,
    BIN_WALL_H,
    CLUTTER_CUBE_HALF,
    CLUTTER_SPHERE_R,
    DT_OUTER,
    SCENES,
)

FIG = os.path.join(os.path.dirname(__file__), "results", "figures")


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _draw_box(ax, center, half, rot, color, alpha=0.9):
    c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    v = (rot @ c.T).T + center
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    ax.add_collection3d(Poly3DCollection([v[f] for f in faces], facecolors=color, edgecolors="k", linewidths=0.3, alpha=alpha))


def _draw_sphere(ax, center, r, color):
    u, v = np.mgrid[0 : 2 * np.pi : 14j, 0 : np.pi : 8j]
    ax.plot_surface(center[0] + r * np.cos(u) * np.sin(v), center[1] + r * np.sin(u) * np.sin(v),
                    center[2] + r * np.cos(v), color=color, linewidth=0, antialiased=False, shade=True)


def _draw_bin(ax):
    hw, h = BIN_HALF, BIN_WALL_H
    for cx, cy, hx, hy in ((-hw, 0, 0.005, hw), (hw, 0, 0.005, hw), (0, -hw, hw, 0.005), (0, hw, hw, 0.005)):
        _draw_box(ax, np.array([cx, cy, h / 2]), np.array([hx, hy, h / 2]), np.eye(3), "#9ecae1", alpha=0.25)
    xx, yy = np.meshgrid([-hw, hw], [-hw, hw])
    ax.plot_surface(xx, yy, np.zeros_like(xx), color="#deebf7", alpha=0.6, linewidth=0)


def _draw_state(ax, model, state, scene):
    bq = state.body_q.numpy().reshape(-1, 7)
    from newton._src.geometry.types import GeoType
    st, sb, sc = model.shape_type.numpy(), model.shape_body.numpy(), model.shape_scale.numpy()
    if scene != "ball":
        _draw_bin(ax)
    for i, b in enumerate(sb):
        if b < 0:
            continue
        p, q = bq[b, :3], bq[b, 3:]
        if GeoType(int(st[i])) == GeoType.SPHERE:
            _draw_sphere(ax, p, sc[i][0], "#e6550d" if scene != "ball" else "#31a354")
        else:
            _draw_box(ax, p, sc[i], _quat_to_mat(q), "#3182bd")
    if scene == "ball":
        xx, yy = np.meshgrid([-0.3, 0.3], [-0.3, 0.3])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color="#deebf7", alpha=0.6, linewidth=0)
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0, 1.2)
        ax.set_box_aspect((1, 1, 2))
    else:
        L = BIN_HALF + 0.05
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, 0.5)
        ax.set_box_aspect((1, 1, 1.25))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)


def main() -> None:
    fig = plt.figure(figsize=(9.6, 5.6), constrained_layout=True)
    specs = [("soft-clutter", "(a) Soft clutter"), ("hard-clutter", "(b) Hard clutter"), ("ball", "(c) Bouncing ball")]
    for col, (scene, title) in enumerate(specs):
        model = build_model(1, scene=scene)
        arm = make_arm(model, "icf-adaptive", scene=scene, tol=1e-3, max_substeps=4096)
        s0, s1, ctrl = model.state(), model.state(), model.control()
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        _draw_state(ax, model, s0, scene)
        ax.set_title(title + "\nt = 0", fontsize=7.5)
        t_end = 0.45 if scene == "ball" else 1.5
        for _ in range(int(round(t_end / DT_OUTER))):
            s0, s1 = arm.boundary(s0, s1, ctrl)
        ax = fig.add_subplot(2, 3, col + 4, projection="3d")
        _draw_state(ax, model, s0, scene)
        ax.set_title(f"t = {t_end:g} s (ICF error control, ε = 10⁻³)", fontsize=7.5)
    fig.suptitle("Part-1 test scenes (CENIC Sec. VII, Figs. 6 and 8).  (a) 20 spheres, k = 10³ N/m, v_s = 1 cm/s.  "
                 "(b) 10 spheres + 10 cubes, k = 10⁵ N/m, v_s = 0.1 mm/s.  (c) 0.1 kg ball, k = 10³ N/m, zero dissipation, 1 m drop.\n"
                 "Assumed (not stated in the paper): r = h = 2.5 cm, water density, 30 cm bin, ball r = 5 cm.", fontsize=7.5)
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"scenes.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote figures/scenes.png/.pdf")


if __name__ == "__main__":
    main()
