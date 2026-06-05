# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Franka Farm -- 100-arm adaptive-dt hero figure.

Renders a grid of Franka FR3 arms running the upstream cube-stacking
pick-and-place task in parallel, each world tinted by its live per-world inner
timestep ``dt`` (red = small step / stiff contact, green = large step / free
motion). Writes a single headless PNG for the poster.

The task / IK / FSM are reused verbatim from
``newton.examples.ik.example_ik_cube_stacking``; this module only swaps the
fixed-step solver for :class:`~newton.solvers.SolverMuJoCoAdaptive`, adds the
dt coloring, and captures an offscreen frame.

Usage::

    uv run python -m scripts.demos.franka_farm \
        --world-count 100 --frames 120 --out franka_farm.png
"""

from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.solvers
from newton.examples.ik.example_ik_cube_stacking import Example

# --- Solver defaults -------------------------------------------------------

TOL = 1e-3
DT_INIT = 0.005
DT_MIN = 1e-6
WORLD_SPACING = 1.5  # metres between worlds in the grid (matches upstream)


class FrankaFarm(Example):
    """Cube-stacking farm on the adaptive solver, tinted by per-world dt."""

    def __init__(self, viewer, *, world_count=100, args=None, color=True,
                 nconmax=1000, njmax=2000, tol=TOL):
        self._adaptive_pending = SimpleNamespace(
            nconmax=nconmax, njmax=njmax, tol=tol
        )
        # headless=False: keep the GL viewer we pass in. The base class swaps
        # to ViewerNull only when headless=True, and ViewerNull cannot render.
        super().__init__(viewer, world_count=world_count, headless=False, args=args)

        # Replace the fixed solver built by the base ctor with the adaptive one.
        # dt_min < dt_init <= dt_max; dt_max = frame_dt so the inner step never
        # overshoots the outer control/render boundary.
        self.solver = newton.solvers.SolverMuJoCoAdaptive(
            self.model,
            tol=self._adaptive_pending.tol,
            dt_init=DT_INIT,
            dt_min=DT_MIN,
            dt_max=self.frame_dt,
            use_mujoco_contacts=True,
            nconmax=self._adaptive_pending.nconmax,
            njmax=self._adaptive_pending.njmax,
            cone="elliptic",
            impratio=1000.0,
        )

        self.color = color
        self._shape_world = None  # lazy shape -> world map

    # --- Overridden seams --------------------------------------------------

    def capture_sim(self):
        """Adaptive step does a host-side boundary sync per iteration and
        cannot be CUDA-graph captured."""
        self.graph = None

    def capture_ik(self):
        """Skip IK graph capture. Newton's tiled IK solver fails above ~256
        worlds; capturing it there raises a fatal CUDA 901, while running it
        eagerly only logs non-fatal launch errors and still renders. Eager IK
        is required for the high-N hero figure."""
        self.graph_ik = None

    def simulate(self):
        """One canonical CENIC outer step per frame. ``solver.step`` owns the
        inner adaptive loop and updates ``state_0`` in place (no swap)."""
        self.solver.step(self.state_0, self.state_1, self.control, None, self.frame_dt)

    def render(self):
        if self.color:
            self._apply_dt_colors()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    # --- dt coloring -------------------------------------------------------

    # Neutral colour for the shared ground plane (no world). Per-world shapes
    # (arm, base/table, cubes) all take their world's dt colour.
    NEUTRAL = (0.22, 0.22, 0.25)

    def _shape_world_map(self) -> np.ndarray:
        if self._shape_world is None:
            self._shape_world = self.model.shape_world.numpy()
        return self._shape_world

    def _apply_dt_colors(self):
        """Tint each world (arm + base + cubes) by its current dt; only the
        shared ground stays neutral.

        One ``.numpy()`` sync per call -- this runs once per rendered frame
        (outside the inner physics loop), same call-site class as a status
        print, so it does not violate the hot-path no-sync rule.
        """
        dt = self.solver.dt.numpy()  # [world_count], float32
        shape_world = self._shape_world_map()  # [shape_count]

        # Auto-range the colormap over the actual per-world dt spread (log
        # scale). Contact regime sets the absolute band (e.g. ~1e-4..1e-3 for
        # cube stacking), so a fixed band would saturate to one colour; ranging
        # over min..max guarantees the slowest world reads red and the fastest
        # green regardless of regime. p2..p98 clamps outliers.
        finite = dt[np.isfinite(dt) & (dt > 0.0)]
        if finite.size == 0:
            return
        lo = math.log(float(np.percentile(finite, 2)))
        hi = math.log(float(np.percentile(finite, 98)))
        span = hi - lo if hi > lo else 1.0

        colors: dict[int, tuple[float, float, float]] = {}
        for shape_idx in range(shape_world.shape[0]):
            world = int(shape_world[shape_idx])
            # Only the shared ground (no world) stays neutral.
            if world < 0 or world >= dt.shape[0]:
                colors[shape_idx] = self.NEUTRAL
                continue
            d = float(dt[world])
            t = (math.log(max(d, 1e-12)) - lo) / span
            t = min(1.0, max(0.0, t))
            # red (small dt / stiff) -> green (large dt / free)
            colors[shape_idx] = (1.0 - t, t, 0.15)

        self.viewer.update_shape_colors(colors)


# --- Capture driver --------------------------------------------------------


# Hero-camera framing (look-at the grid centre, distance auto-fit so the grid
# fills the frame). ELEVATION is the look-down angle in degrees; DIST_MULT
# scales the camera distance with the grid extent (smaller = tighter / more
# fill). These are the values _default_camera uses.
ELEVATION_DEG = 40.0
DIST_MULT = 0.18
TARGET_Z = 0.35  # aim a little above the ground so arms sit mid-frame [m]


def _grid_extent(world_count: int, spacing: float) -> float:
    """Full XY span of the world grid. The viewer centres worlds on the origin,
    so the grid runs from -extent/2 to +extent/2 in both X and Y."""
    side = int(math.ceil(world_count ** 0.5))
    return (side - 1) * spacing


def _grid_center(world_count: int, spacing: float) -> wp.vec3:
    # Viewer world offsets are symmetric about the origin (e.g. -5.25..+5.25),
    # so the grid centre is the origin, not a corner.
    return wp.vec3(0.0, 0.0, TARGET_Z)


def look_at_camera(world_count, spacing, *, elevation_deg, dist_mult, azimuth_deg=225.0):
    """Place a camera that *looks at* the grid centre from a given elevation /
    distance, returning ``(pos, pitch, yaw)`` for ``viewer.set_camera``.

    Aiming at the centre (the origin) keeps the grid centred, and distance scaled
    to the grid span makes it fill the frame regardless of world count. For the
    Z-up camera, ``pitch = -elevation`` and the look-at yaw points back toward
    the grid.
    """
    extent = _grid_extent(world_count, spacing)
    c = _grid_center(world_count, spacing)
    dist = dist_mult * extent

    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    # Offset direction from centre to camera (unit), then place the camera.
    ox, oy, oz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    pos = wp.vec3(float(c[0]) + dist * ox, float(c[1]) + dist * oy, float(c[2]) + dist * oz)

    # Look direction = centre - pos = -offset; derive pitch/yaw (Z-up).
    pitch = math.degrees(math.asin(-oz))
    yaw = math.degrees(math.atan2(-oy, -ox))
    return pos, pitch, yaw


def _default_camera(world_count: int, spacing: float):
    """Hero 3/4 view: look at the grid centre, distance auto-fit so the robot
    grid fills the frame (surrounding ground cropped out)."""
    return look_at_camera(
        world_count, spacing, elevation_deg=ELEVATION_DEG, dist_mult=DIST_MULT
    )


def _print_dt_stats(dt: np.ndarray):
    pct = np.percentile(dt, [0, 25, 50, 75, 100])
    print(
        "dt stats [s]: min={:.2e} p25={:.2e} med={:.2e} p75={:.2e} max={:.2e}".format(*pct),
        flush=True,
    )


def save_png(viewer, out_path: str):
    frame = viewer.get_frame()  # wp.array(h, w, 3) uint8
    img = frame.numpy()
    from PIL import Image

    Image.fromarray(img, mode="RGB").save(out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=100)
    parser.add_argument("--cube-count", type=int, default=3)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=str, default="franka_farm.png")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--elevation", type=float, default=ELEVATION_DEG,
                        help="Camera look-down angle in degrees (look-at hero).")
    parser.add_argument("--dist-mult", type=float, default=DIST_MULT,
                        help="Camera distance as a fraction of grid span "
                             "(smaller = tighter fill).")
    parser.add_argument("--cam-pos", type=float, nargs=3, default=None,
                        help="Override camera position x y z (else look-at auto).")
    parser.add_argument("--cam-pitch", type=float, default=None)
    parser.add_argument("--cam-yaw", type=float, default=None)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--nconmax", type=int, default=1000)
    parser.add_argument("--njmax", type=int, default=2000)
    parser.add_argument("--no-color", action="store_true",
                        help="Disable dt coloring (plain stock render).")
    parser.add_argument("--use-mujoco-contacts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    wp.init()

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)

    example = FrankaFarm(
        viewer,
        world_count=args.world_count,
        args=args,
        color=not args.no_color,
        nconmax=args.nconmax,
        njmax=args.njmax,
        tol=args.tol,
    )

    viewer.set_world_offsets(wp.vec3(WORLD_SPACING, WORLD_SPACING, 0.0))

    pos, pitch, yaw = look_at_camera(
        args.world_count, WORLD_SPACING,
        elevation_deg=args.elevation, dist_mult=args.dist_mult,
    )
    if args.cam_pos is not None:
        pos = wp.vec3(*args.cam_pos)
    if args.cam_pitch is not None:
        pitch = args.cam_pitch
    if args.cam_yaw is not None:
        yaw = args.cam_yaw
    viewer.set_camera(pos, pitch, yaw)

    for _ in range(args.frames):
        example.step()
        example.render()

    _print_dt_stats(example.solver.dt.numpy())
    save_png(viewer, args.out)
    print(f"Wrote {args.out}  ({args.width}x{args.height}, {args.world_count} worlds, "
          f"{args.frames} frames)", flush=True)


if __name__ == "__main__":
    main()
