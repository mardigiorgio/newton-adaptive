# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run-to-run determinism of the four arms: two identical arms on the same
model from the same initial state must produce bit-identical states, else a
restarted-window comparison (part1_consistency.py) measures the solver's
own noise on a chaotic scene rather than its step error. Reports the max
position difference after 1 s on each scene and, for hard clutter, its
growth window by window.

    uv run python scripts/bench/probe_determinism.py
"""

from __future__ import annotations

import sys

import numpy as np

from scripts.bench.four_arms import _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive, build_model
from scripts.scenes.cenic_scenes import SCENES

BOUNDARY_S = 0.01


def _arm(model, scene, backend, kind, knob):
    icf, solref = SCENES[scene].icf, SCENES[scene].mujoco_solref
    if backend == "icf":
        return _make_icf(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, icf) if kind == "fixed" else _make_icf_adaptive(model, knob, BOUNDARY_S, icf, 4096)
    return _make_mujoco(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, solref) if kind == "fixed" else _make_mujoco_adaptive(model, knob, BOUNDARY_S, 4096, solref)


def run_pair(scene, backend, kind, knob, n=8, seconds=1.0):
    model = build_model(n, scene=scene)
    a1, a2 = _arm(model, scene, backend, kind, knob), _arm(model, scene, backend, kind, knob)
    s = [model.state(), model.state()], [model.state(), model.state()]
    c = model.control()
    diffs = []
    for b in range(int(round(seconds / BOUNDARY_S))):
        s[0][0], s[0][1] = a1.boundary(s[0][0], s[0][1], c)
        s[1][0], s[1][1] = a2.boundary(s[1][0], s[1][1], c)
        if (b + 1) % 10 == 0:
            x1 = s[0][0].body_q.numpy().reshape(-1, 7)[:, :3]
            x2 = s[1][0].body_q.numpy().reshape(-1, 7)[:, :3]
            diffs.append(float(np.abs(x1 - x2).max()))
    return diffs


def main() -> int:
    ok = True
    for scene in ("ball", "soft-clutter", "hard-clutter"):
        for backend, kind, knob in (("icf", "fixed", 1e-3), ("mujoco", "fixed", 1e-3), ("icf", "adaptive", 1e-3), ("mujoco", "adaptive", 1e-3)):
            d = run_pair(scene, backend, kind, knob)
            arm = backend if kind == "fixed" else f"{backend}-adaptive"
            trail = " ".join(f"{v:.1e}" for v in d)
            print(f"{scene:13s} {arm:16s} max |dx| after 1 s: {d[-1]:.2e} m   (every 0.1 s: {trail})", flush=True)
            ok &= d[-1] == 0.0
    print("DETERMINISTIC" if ok else "NOT bit-reproducible")
    return 0


if __name__ == "__main__":
    sys.exit(main())
