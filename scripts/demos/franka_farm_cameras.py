# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fast camera-angle previewer for the Franka Farm hero figure.

Simulates the farm ONCE, then renders every candidate camera angle from that
same state and writes a single labelled contact sheet. Each tile is captioned
with the exact ``--cam-pos / --cam-pitch / --cam-yaw`` flags to reproduce it in
the full-res render, so you can pick an angle without re-simulating per try.

Framing is relative to the grid extent, so a small ``--world-count`` (fast)
previews the same composition you will get at the final high N.

Usage::

    uv run python -m scripts.demos.franka_farm_cameras \
        --world-count 64 --frames 30 --out /tmp/farm_cams.png
"""

from __future__ import annotations

import argparse

import warp as wp

import newton
from scripts.demos.franka_farm import WORLD_SPACING, FrankaFarm, look_at_camera

# Candidate cameras: (label, elevation deg, distance x extent). All look AT the
# grid centre; distance auto-fits so the robot grid fills the frame. Smaller
# distance = tighter fill; larger elevation = more top-down. Edit freely.
CANDIDATES = [
    ("e40-d0.85", 40.0, 0.85),
    ("e48-d0.80", 48.0, 0.80),
    ("e55-d0.75", 55.0, 0.75),
    ("e35-d0.90", 35.0, 0.90),
]


def _montage(tiles, labels, cols=3):
    """Assemble RGB tiles (list of HxWx3 uint8) into a labelled grid PNG."""
    from PIL import Image, ImageDraw

    rows = (len(tiles) + cols - 1) // cols
    th, tw = tiles[0].shape[:2]
    sheet = Image.new("RGB", (cols * tw, rows * th), (10, 10, 12))
    draw = ImageDraw.Draw(sheet)
    for i, (tile, label) in enumerate(zip(tiles, labels, strict=True)):
        r, cc = divmod(i, cols)
        x, y = cc * tw, r * th
        sheet.paste(Image.fromarray(tile, mode="RGB"), (x, y))
        # text shadow + text for legibility on any background
        draw.text((x + 7, y + 6), label, fill=(0, 0, 0))
        draw.text((x + 6, y + 5), label, fill=(255, 255, 0))
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-count", type=int, default=64,
                        help="Small N renders fast; framing matches high N.")
    parser.add_argument("--cube-count", type=int, default=3)
    parser.add_argument("--frames", type=int, default=30,
                        help="Sim frames before capturing (arms into the task).")
    parser.add_argument("--tile-width", type=int, default=720)
    parser.add_argument("--tile-height", type=int, default=405)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out", type=str, default="franka_cams.png")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--use-mujoco-contacts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    wp.init()

    viewer = newton.viewer.ViewerGL(
        width=args.tile_width, height=args.tile_height, headless=True
    )
    example = FrankaFarm(
        viewer,
        world_count=args.world_count,
        args=args,
        color=not args.no_color,
    )

    # Simulate ONCE.
    for _ in range(args.frames):
        example.step()
    if not args.no_color:
        example._apply_dt_colors()

    tiles, labels = [], []
    for name, elevation, dist in CANDIDATES:
        pos, pitch, yaw = look_at_camera(
            args.world_count, WORLD_SPACING, elevation_deg=elevation, dist_mult=dist
        )
        viewer.set_camera(pos, pitch, yaw)
        viewer.begin_frame(example.sim_time)
        viewer.log_state(example.state_0)
        viewer.end_frame()
        tiles.append(viewer.get_frame().numpy().copy())
        flags = (f"--cam-pos {pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                 f"--cam-pitch {pitch:.1f} --cam-yaw {yaw:.1f}")
        labels.append(f"{name}\n{flags}")
        print(f"[{name}]  {flags}", flush=True)

    sheet = _montage(tiles, labels, cols=args.cols)
    sheet.save(args.out)
    print(f"\nWrote contact sheet {args.out}  "
          f"({len(CANDIDATES)} angles, {args.world_count} worlds)", flush=True)
    print("Pick an angle, then run the full render with its --cam-* flags.",
          flush=True)


if __name__ == "__main__":
    main()
