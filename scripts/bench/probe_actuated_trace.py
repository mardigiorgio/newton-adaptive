# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trajectory forensics for the actuated push (part1_actuated.py): per
boundary, tip and box positions, box vertical velocity and tip penetration,
so a summary number in the sweep (an 18 mm tip penetration, a box vertical
velocity RMS of 0.1 m/s) can be traced to what the bodies did.

Also runs the same scene with the push frozen (x target held at 0) to
separate a sliding-contact artifact from a resting-contact one.

    uv run python scripts/bench/probe_actuated_trace.py
"""

from __future__ import annotations

import numpy as np

from scripts.bench.benchmarks.part1_actuated import BOUNDARY_S, _arm
from scripts.scenes.actuated_press import BOX_HALF, TIP_R, X0, ActuatedSpec, build, program


def trace(backend, kind, knob, kp, k=1e5, speed=0.3, freeze_push=False):
    spec = ActuatedSpec(k=k, kp=kp, slide_speed=speed)
    model = build(1, spec)
    arm = _arm(model, spec, backend, kind, knob)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    tq = ctrl.joint_target_q.numpy()
    nb = int(round(spec.horizon_s / BOUNDARY_S))
    rows = []
    for b in range(nb):
        t = b * BOUNDARY_S
        x_t, z_t = program(t, spec)
        if freeze_push:
            x_t = 0.0
        tq[-2], tq[-1] = x_t, z_t
        ctrl.joint_target_q.assign(tq)
        s0, s1 = arm.boundary(s0, s1, ctrl)
        bq = s0.body_q.numpy().reshape(-1, 7)
        bqd = s0.body_qd.numpy().reshape(-1, 6)
        xb, zb, xt, zt = bq[0, 0], bq[0, 2], bq[2, 0], bq[2, 2]
        beside = abs(zt - zb) < BOX_HALF
        pen = max(0.0, (xt + TIP_R) - (xb - BOX_HALF)) if beside else 0.0
        rows.append((t, xb, zb, xt, zt, bqd[0, 2], bqd[0, 4], pen, x_t))
    return np.array(rows)


def summarize(label, r):
    t, xb, zb, xt, zt, vzb, wyb, pen, x_t = r.T
    push = (t >= 0.7) & (t <= 0.7 + 0.3 / 0.3 + 0.2)
    i = int(np.argmax(pen))
    print(f"{label}")
    print(f"   box z: min {zb.min()*1e3:.3f} mm max {zb.max()*1e3:.3f} mm (rest {BOX_HALF*1e3:.1f});  box vz RMS push {np.sqrt(np.mean(vzb[push]**2)):.4f} / settle {np.sqrt(np.mean(vzb[t > t.max()-0.8]**2)):.4f} m/s;  pitch rate RMS push {np.sqrt(np.mean(wyb[push]**2)):.3f} rad/s")
    print(f"   tip z: min {zt.min()*1e3:.1f} mm max {zt.max()*1e3:.1f} mm (box mid {BOX_HALF*1e3:.0f});  box x end {xb[-1]:.3f} m;  tip pen max {pen[i]*1e3:.2f} mm at t={t[i]:.2f}s (xt={xt[i]:.4f} xb={xb[i]:.4f} zt={zt[i]:.4f} zb={zb[i]:.4f} target x={X0 + x_t[i]:.4f})")
    big = np.where(pen > 1e-3)[0]
    if len(big):
        print(f"   tip pen > 1 mm at {len(big)} boundaries, t in [{t[big[0]]:.2f}, {t[big[-1]]:.2f}] s")


if __name__ == "__main__":
    for backend in ("mujoco", "icf"):
        for kp in (1e2, 1e4):
            for dt in (1e-2, 1e-3):
                summarize(f"== {backend} fixed dt={dt*1e3:g} ms Kp={kp:g}", trace(backend, "fixed", dt, kp))
            summarize(f"== {backend} fixed dt=1 ms Kp={kp:g} PUSH FROZEN (box at rest, tip beside it)", trace(backend, "fixed", 1e-3, kp, freeze_push=True))
