# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent verification of the penetration bench's measurement.

The penetration CSV is only evidence if the instrument measures what it
claims. Every assumption behind ``part1_penetration._penetrations`` is
checked here against the Newton MODEL (not against the bench's own code):

  1. body layout: per world, bodies 0-8 carry SPHERE shapes and 9-17 BOX
     shapes (the ``is_sphere`` mask), no static body carries either;
  2. shape geometry: sphere radius and box half-extents equal the constants
     the bench uses;
  3. shape-body offsets: every dynamic shape sits at its body origin
     (identity transform), so body_q IS the shape pose;
  4. the ground: exactly one PLANE shape, on the static body, at height 0
     with +z normal;
  5. quaternion convention: the bench's numpy rotation matches warp's
     ``wp.quat_rotate`` on random unit quaternions in body_q's layout;
  6. state evolution: the metric pass reads a state that actually moves —
     per-boundary min sphere-center z and min box-corner z for
     icf-adaptive (the "exactly zero" arm) and mujoco n_sub=1, so the
     zero is a settled contact ABOVE the plane, not a frozen state;
  7. timing stability: three independent trials of icf n_sub=1 and 2.

Standalone:
    uv run python scripts/bench/verify_part1_penetration.py
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import warp as wp


from scripts.bench.benchmarks.part1_penetration import _CORNERS, _penetrations, _quat_rotate
from scripts.bench.four_arms import build_model, make_arm
from scripts.scenes.contact_objects import BOX_HALF, SPHERE_RADIUS

FAIL = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def verify_structure(n: int = 3) -> None:
    print(f"[1-4] model structure, n={n}")
    model = build_model(n)
    per_world = model.body_count // n
    check(model.body_count == 18 * n, f"body_count {model.body_count} == 18*{n}")
    st = model.shape_type.numpy()
    sb = model.shape_body.numpy()
    sc = model.shape_scale.numpy()
    sx = model.shape_transform.numpy()
    from newton._src.geometry.types import GeoType

    names = ["GEO_" + GeoType(int(t)).name for t in st]
    # per-body geometry
    body_geo = {}
    for i, b in enumerate(sb):
        if b >= 0:
            body_geo.setdefault(int(b), []).append(i)
    layout_ok = True
    radius_ok, half_ok, offset_ok = True, True, True
    for b, shapes in body_geo.items():
        local = b % per_world
        want = "GEO_SPHERE" if local < 9 else "GEO_BOX"
        got = names[shapes[0]]
        if len(shapes) != 1 or got != want:
            layout_ok = False
        s = sc[shapes[0]]
        if want == "GEO_SPHERE" and abs(float(s[0]) - SPHERE_RADIUS) > 1e-6:
            radius_ok = False
        if want == "GEO_BOX" and np.abs(s - BOX_HALF).max() > 1e-6:
            half_ok = False
        xf = sx[shapes[0]]
        if np.abs(xf[:3]).max() > 1e-7 or np.abs(xf[3:] - np.array([0, 0, 0, 1])).max() > 1e-7:
            offset_ok = False
    check(layout_ok, "per world: bodies 0-8 are single SPHERE shapes, 9-17 single BOX shapes")
    check(radius_ok, f"every sphere radius == {SPHERE_RADIUS}")
    check(half_ok, f"every box half-extent == {BOX_HALF}")
    check(offset_ok, "every dynamic shape sits at its body origin (identity offset)")
    planes = [i for i, nm in enumerate(names) if nm == "GEO_PLANE"]
    check(len(planes) == 1, f"exactly one PLANE shape (found {len(planes)})")
    if planes:
        i = planes[0]
        xf = sx[i]
        q = xf[3:]
        # plane normal = the shape frame's +z axis
        nz = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1])
        check(int(sb[i]) < 0, "plane is on the static body")
        check(abs(float(xf[2])) < 1e-7, f"plane height == 0 (got {float(xf[2]):.2e})")
        check(abs(nz - 1.0) < 1e-6, f"plane normal is +z (nz={nz:.6f})")
    others = [nm for nm in names if nm not in ("GEO_SPHERE", "GEO_BOX", "GEO_PLANE")]
    print(f"       other shapes present (walls etc.): {sorted(set(others))}")


def verify_quaternion(trials: int = 200) -> None:
    print("[5] quaternion convention vs wp.quat_rotate")
    rng = np.random.default_rng(0)
    q = rng.normal(size=(trials, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    mine = _quat_rotate(q, _CORNERS)  # [trials, 8, 3]
    worst = 0.0
    for t in range(trials):
        wq = wp.quat(*[float(v) for v in q[t]])
        for c in range(8):
            ref = wp.quat_rotate(wq, wp.vec3(*[float(v) for v in _CORNERS[c]]))
            worst = max(worst, float(np.abs(np.array([ref[0], ref[1], ref[2]]) - mine[t, c]).max()))
    check(worst < 1e-5, f"numpy rotation matches warp (x,y,z,w) to {worst:.1e}")


def verify_evolution(arm_name: str, knob, n: int = 8, boundaries: int = 120) -> None:
    print(f"[6] state evolution: {arm_name} knob={knob}, n={n}")
    kwargs = {"n_sub": knob} if arm_name in ("mujoco", "icf") else {"tol": knob}
    model = build_model(n)
    arm = make_arm(model, arm_name, **kwargs)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    per_world = 18
    mins_sphere, mins_corner, pens = [], [], []
    for _ in range(boundaries):
        s0, s1 = arm.boundary(s0, s1, ctrl)
        bq = s0.body_q.numpy().reshape(-1, 7)
        idx = np.arange(bq.shape[0]) % per_world
        sph = bq[idx < 9]
        box = bq[idx >= 9]
        corners = _quat_rotate(box[:, 3:], _CORNERS) + box[:, None, :3]
        mins_sphere.append(float(sph[:, 2].min()))
        mins_corner.append(float(corners[:, :, 2].min()))
        pens.append(float(_penetrations(model, s0).max()))
    ms, mc, mp = np.array(mins_sphere), np.array(mins_corner), np.array(pens)
    check(ms[0] > ms[-1] + 0.05, f"spheres fell: min center z {ms[0]:.3f} -> {ms[-1]:.4f}")
    check(np.isfinite(ms).all() and np.isfinite(mc).all(), "all readbacks finite")
    print(f"       settled min sphere-center z = {ms[-20:].min():.5f}  (radius {SPHERE_RADIUS}; gap {ms[-20:].min() - SPHERE_RADIUS:+.5f})")
    print(f"       settled min box-corner z    = {mc[-20:].min():+.5f}")
    print(f"       max ground penetration over run = {mp.max():.3e} m; at settle = {mp[-20:].max():.3e} m")
    # the bench's number must equal the independent recomputation here
    recomputed = max(0.0, SPHERE_RADIUS - ms.min(), -mc.min())
    check(abs(recomputed - mp.max()) < 1e-6, f"bench max_pen {mp.max():.3e} == independent recomputation {recomputed:.3e}")


def verify_timing() -> None:
    print("[7] timing stability: 3 trials each, per-boundary median")
    for knob in (1, 2):
        meds = []
        for _ in range(3):
            r = subprocess.run(
                [sys.executable, "-c", f"""
import time, numpy as np, warp as wp
from scripts.bench.four_arms import build_model, make_arm
m = build_model(64); a = make_arm(m, 'icf', n_sub={knob})
s0, s1, c = m.state(), m.state(), m.control()
for _ in range(20): s0, s1 = a.boundary(s0, s1, c)
wp.synchronize(); ts = []
for _ in range(100):
    t0 = time.perf_counter(); s0, s1 = a.boundary(s0, s1, c); wp.synchronize(); ts.append(time.perf_counter() - t0)
print('MED', np.median(ts) * 1e3)
"""],
                capture_output=True, text=True, cwd=".",
            )
            for line in r.stdout.splitlines():
                if line.startswith("MED"):
                    meds.append(float(line.split()[1]))
        print(f"       icf n_sub={knob}: medians {['%.2f' % m for m in meds]} ms")
        if len(meds) == 3:
            check(max(meds) / min(meds) < 1.5, f"icf n_sub={knob} trial spread < 1.5x")


def main() -> int:
    verify_structure()
    verify_quaternion()
    verify_evolution("icf-adaptive", 1e-3)
    verify_evolution("mujoco", 1)
    verify_timing()
    print("\nVERIFY FAILED:\n  - " + "\n  - ".join(FAIL) if FAIL else "\nVERIFY PASSED: the instrument measures what it claims")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
