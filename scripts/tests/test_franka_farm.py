# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the Franka Farm hero-figure renderer.

Renders a tiny grid headless and verifies the PNG is written, has the right
dimensions, and is not a single flat colour (i.e. the scene + dt coloring + GL
capture actually ran). Runs the demo as a subprocess so the GL context lives in
its own process, matching real usage.

Skipped automatically when no CUDA GPU or no display is available, since the GL
viewer cannot create an offscreen context in those environments.

Run with:

    uv run -m pytest scripts/tests/test_franka_farm.py -v
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import warp as wp

_HAS_DISPLAY = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_cuda() -> bool:
    try:
        wp.init()
        return any(d.is_cuda for d in wp.get_devices())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_HAS_DISPLAY and _has_cuda()),
    reason="Franka Farm render needs a CUDA GPU and a display for the GL viewer",
)


def test_franka_farm_renders_png(tmp_path):
    out = tmp_path / "farm.png"
    width, height = 320, 200

    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.demos.franka_farm",
            "--world-count", "2",
            "--cube-count", "1",
            "--frames", "2",
            "--width", str(width),
            "--height", str(height),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    assert proc.returncode == 0, f"render failed:\n{proc.stdout}\n{proc.stderr}"
    assert out.exists(), "no PNG written"

    from PIL import Image

    img = np.asarray(Image.open(out))
    assert img.shape == (height, width, 3), f"unexpected image shape {img.shape}"
    # Not a single flat colour -> the scene actually rendered.
    assert img.std() > 1.0, "rendered image is flat (likely empty/black frame)"
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 50, "too few distinct colours"
