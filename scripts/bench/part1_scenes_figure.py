# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the Part-1 test scenes : each scene at
its initial state and after settling under ICF error control (eps 1e-3),
drawn from the simulation state -- spheres, oriented cubes, the bin, the
ball. Writes scripts/bench/results/figures/scenes.{png,pdf}.

    uv run python scripts/bench/part1_scenes_figure.py
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.colors
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


class _Scene:
    """Every face of every object goes into ONE depth-sorted collection:
    matplotlib's 3D painter draws separate collections in call order, which
    is what made bodies show through each other and through the walls."""

    def __init__(self):
        self.polys, self.colors = [], []

    def box(self, center, half, rot, color, alpha=1.0, tiles=1):
        # each face tiled so large translucent walls sort against small bodies
        for axis in range(3):
            for sign in (-1, 1):
                u, v = [i for i in range(3) if i != axis]
                for i in range(tiles):
                    for j in range(tiles):
                        corners = []
                        for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                            c = np.zeros(3)
                            c[axis] = sign * half[axis]
                            c[u] = -half[u] + 2 * half[u] * (i + du) / tiles
                            c[v] = -half[v] + 2 * half[v] * (j + dv) / tiles
                            corners.append(c)
                        self.polys.append((rot @ np.array(corners).T).T + center)
                        self.colors.append((*matplotlib.colors.to_rgb(color), alpha))

    def sphere(self, center, r, color, nu=16, nv=10):
        u = np.linspace(0, 2 * np.pi, nu + 1)
        v = np.linspace(0, np.pi, nv + 1)
        light = np.array([0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
        for i in range(nu):
            for j in range(nv):
                quad = []
                for uu, vv in ((u[i], v[j]), (u[i + 1], v[j]), (u[i + 1], v[j + 1]), (u[i], v[j + 1])):
                    quad.append(center + r * np.array([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)]))
                n = (quad[0] + quad[2]) / 2 - center
                shade = 0.55 + 0.45 * max(0.0, float(n @ light) / max(np.linalg.norm(n), 1e-9))
                rgb = np.array(matplotlib.colors.to_rgb(color)) * shade
                self.polys.append(np.array(quad))
                self.colors.append((*rgb, 1.0))

    def draw(self, ax):
        coll = Poly3DCollection(self.polys, facecolors=self.colors, edgecolors=(0, 0, 0, 0.15), linewidths=0.15, zsort="average")
        ax.add_collection3d(coll)


def _draw_state(ax, model, state, scene):
    from newton._src.geometry.types import GeoType

    bq = state.body_q.numpy().reshape(-1, 7)
    st, sb, sc = model.shape_type.numpy(), model.shape_body.numpy(), model.shape_scale.numpy()
    S = _Scene()
    if scene != "ball":
        hw, h = BIN_HALF, BIN_WALL_H
        S.box(np.array([0, 0, -0.004]), np.array([hw + 0.02, hw + 0.02, 0.004]), np.eye(3), "#c6dbef", 1.0, tiles=4)
        for cx, cy, hx, hy in ((-hw - 0.005, 0, 0.005, hw + 0.01), (hw + 0.005, 0, 0.005, hw + 0.01),
                               (0, -hw - 0.005, hw + 0.01, 0.005), (0, hw + 0.005, hw + 0.01, 0.005)):
            S.box(np.array([cx, cy, h / 2]), np.array([hx, hy, h / 2]), np.eye(3), "#9ecae1", 0.22, tiles=4)
    else:
        S.box(np.array([0, 0, -0.004]), np.array([0.3, 0.3, 0.004]), np.eye(3), "#c6dbef", 1.0, tiles=4)
    for i, b in enumerate(sb):
        if b < 0:
            continue
        p, q = bq[b, :3], bq[b, 3:]
        if GeoType(int(st[i])) == GeoType.SPHERE:
            S.sphere(p, sc[i][0], "#e6550d" if scene != "ball" else "#31a354")
        else:
            S.box(p, sc[i], _quat_to_mat(q), "#3182bd")
    S.draw(ax)
    if scene == "ball":
        ax.set_xlim(-0.3, 0.3); ax.set_ylim(-0.3, 0.3); ax.set_zlim(0, 1.2)
        ax.set_box_aspect((1, 1, 2))
    else:
        L = BIN_HALF + 0.05
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0, 0.5)
        ax.set_box_aspect((1, 1, 1.25))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-58)


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
    # No baked-in caption: the LaTeX caption in PART1.md carries the description.
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"scenes.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote figures/scenes.png/.pdf")


if __name__ == "__main__":
    main()
