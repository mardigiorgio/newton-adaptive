"""Intuition figure: penetration depth over time for one stiff sphere impact.
Coarse fixed-step gets the physics qualitatively wrong (overshoots / ejects), while
CENIC (paper-faithful, S=identity) tracks the dt->0 reference by refining dt at impact.

    uv run --extra rl --extra examples --extra importers -m scripts.rl.v1_penetration_trace
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.solvers

from scripts.bench.plotting import save_fig

T = 0.12
Z0, VZ, R, KE = 0.052, -4.0, 0.05, 1.0e4
INTEG = "euler"


def _build():
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.005)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, Z0)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, 1)
    b.add_ground_plane()
    return b.finalize()


def trace_cenic(tol, dt_rec):
    m = _build()
    s = newton.solvers.SolverMuJoCoCENIC(m, tol=tol, dt_inner_init=dt_rec, dt_inner_min=1e-7,
                                         dt_inner_max=dt_rec, nconmax=64, njmax=128, integrator=INTEG)
    s0, s1 = m.state(), m.state(); wp.to_torch(s0.joint_qd)[2] = VZ; c = m.control()
    ts, pen = [], []
    for i in range(round(T / dt_rec)):
        s0, s1 = s.step_dt(dt_rec, s0, s1, c)
        ts.append((i + 1) * dt_rec * 1e3); pen.append((R - float(s0.joint_q.numpy()[2])) * 1e3)
    return np.array(ts), np.array(pen)


def trace_fixed(fixed_dt, dt_rec):
    m = _build()
    s = newton.solvers.SolverMuJoCo(m, separate_worlds=True, nconmax=64, njmax=128, integrator=INTEG)
    s0, s1 = m.state(), m.state(); wp.to_torch(s0.joint_qd)[2] = VZ; c = m.control()
    n = max(1, round(dt_rec / fixed_dt)); ts, pen = [], []
    for i in range(round(T / dt_rec)):
        for _ in range(n):
            s0.clear_forces(); s.step(s0, s1, c, None, fixed_dt); s0, s1 = s1, s0
        ts.append((i + 1) * dt_rec * 1e3); pen.append((R - float(s0.joint_q.numpy()[2])) * 1e3)
    return np.array(ts), np.array(pen)


def main():
    tg, pg = trace_fixed(2e-5, 0.002)
    tc, pc = trace_cenic(1e-3, 0.002)
    tff, pff = trace_fixed(1e-3, 0.002)
    tco, pco = trace_fixed(1e-2, 0.01)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(tg, pg, "-", color="k", lw=2.5, label="reference (dt -> 0)")
    ax.plot(tc, pc, "o-", color="tab:blue", lw=1.8, ms=4, label="CENIC adaptive (tol 1e-3)")
    ax.plot(tff, pff, "--", color="tab:green", lw=1.4, label="fixed 1 ms")
    ax.plot(tco, pco, "s-", color="tab:orange", lw=1.8, ms=7, label="fixed 10 ms (coarse)")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("time after impact [ms]")
    ax.set_ylabel("penetration depth into ground [mm]  (negative = bounced off above surface)")
    ax.set_title("Same stiff impact, four steppers: coarse fixed-step ejects the ball;\nCENIC refines dt and tracks the reference")
    ax.grid(True, alpha=0.3)
    ax.legend()
    os.makedirs("results/plots", exist_ok=True)
    save_fig(fig, "results/plots/v1_penetration_trace.png")
    print(f"gold max pen {pg.max():.2f}mm | CENIC {pc.max():.2f}mm | fixed1ms {pff.max():.2f}mm | fixed10ms {pco.max():.2f}mm")
    print("saved results/plots/v1_penetration_trace.png")


if __name__ == "__main__":
    main()
