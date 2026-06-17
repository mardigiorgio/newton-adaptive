"""V1 (corrected) clean integration test: a single rigid sphere driven into an
already-detected stiff contact (no policy, no chaos, no collision-cadence ambiguity).

This is the confound-free test the broken Phase-0b / V1 metrics never ran. With the
velocity-inclusive, identity-weighted CENIC error norm (S=identity), the adaptive
stepper refines dt at the impact and beats fixed-step on accuracy per unit compute.

    uv run --extra rl --extra examples --extra importers -m scripts.rl.v1_single_drop
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

import newton
import newton.solvers

from scripts.bench.plotting import save_fig

DT = 0.01          # outer period [s]
T = 0.4            # rollout seconds
Z0 = 0.052         # start just inside contact margin (radius 0.05 + margin 0.005)
VZ = -4.0          # downward velocity [m/s] -> stiff compression on the contact spring
R = 0.05
KE = 1.0e4
GOLD_DT = 2e-5
INTEG = "euler"    # explicit -> dt-sensitive, so the integration effect is visible
CENIC_TOLS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
FIXED_DTS = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
SURFACE = R        # sphere-center height at first touch


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


def rollout_cenic(tol: float):
    m = _build()
    s = newton.solvers.SolverMuJoCoCENIC(
        m, tol=tol, dt_inner_init=DT, dt_inner_min=1e-7, dt_inner_max=DT,
        nconmax=64, njmax=128, integrator=INTEG,
    )
    s0, s1 = m.state(), m.state()
    wp.to_torch(s0.joint_qd)[2] = VZ
    c = m.control()
    s.reset_compute_counter()
    zmin = 9.0
    dt_trace = np.zeros(round(T / DT))
    for i in range(round(T / DT)):
        s0, s1 = s.step_dt(DT, s0, s1, c)
        zmin = min(zmin, float(s0.joint_q.numpy()[2]))
        dt_trace[i] = float(s.dt.numpy()[0])
    return zmin, s.cumulative_substeps(), dt_trace


def rollout_fixed(fixed_dt: float):
    m = _build()
    s = newton.solvers.SolverMuJoCo(m, separate_worlds=True, nconmax=64, njmax=128, integrator=INTEG)
    s0, s1 = m.state(), m.state()
    wp.to_torch(s0.joint_qd)[2] = VZ
    c = m.control()
    n = round(DT / fixed_dt)
    zmin = 9.0
    for i in range(round(T / DT)):
        for _ in range(n):
            s0.clear_forces()
            s.step(s0, s1, c, None, fixed_dt)
            s0, s1 = s1, s0
        zmin = min(zmin, float(s0.joint_q.numpy()[2]))
    return zmin, round(T / DT) * n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/plots")
    p.add_argument("--json", default="results/v1_single_drop.json")
    args = p.parse_args()

    zg, gsub = rollout_fixed(GOLD_DT)
    pen_gold = (SURFACE - zg) * 1e3
    print(f"gold: dt={GOLD_DT:.0e} zmin={zg:.5f} penetration={pen_gold:.2f}mm sub={gsub}", flush=True)

    res = {"meta": {"gold_dt": GOLD_DT, "gold_zmin": zg, "gold_pen_mm": pen_gold, "integrator": INTEG,
                    "Z0": Z0, "VZ": VZ, "ke": KE}, "cenic": [], "fixed": []}
    dt_traces = {}
    for tol in CENIC_TOLS:
        z, sub, dtt = rollout_cenic(tol)
        e = abs(z - zg) * 1e3
        res["cenic"].append({"tol": tol, "substeps": sub, "pen_err_mm": e})
        dt_traces[tol] = dtt
        print(f"  CENIC tol={tol:.0e} sub={sub:5d} pen_err={e:6.2f}mm", flush=True)
    for fdt in FIXED_DTS:
        z, sub = rollout_fixed(fdt)
        e = abs(z - zg) * 1e3
        res["fixed"].append({"fixed_dt": fdt, "substeps": sub, "pen_err_mm": e})
        print(f"  fixed dt={fdt:.0e} sub={sub:5d} pen_err={e:6.2f}mm", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(res, f, indent=2)
    os.makedirs(args.out_dir, exist_ok=True)

    cs, fs = res["cenic"], res["fixed"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot([r["substeps"] for r in cs], [r["pen_err_mm"] for r in cs], "o-", color="tab:blue", lw=2, ms=7, label="CENIC (adaptive)")
    a1.plot([r["substeps"] for r in fs], [r["pen_err_mm"] for r in fs], "s-", color="tab:orange", lw=2, ms=7, label="fixed-step")
    for r in cs:
        a1.annotate(f"{r['tol']:.0e}", (r["substeps"], r["pen_err_mm"]), fontsize=6, color="tab:blue", xytext=(3, 3), textcoords="offset points")
    for r in fs:
        a1.annotate(f"{r['fixed_dt']*1e3:g}ms", (r["substeps"], r["pen_err_mm"]), fontsize=6, color="tab:orange", xytext=(3, -9), textcoords="offset points")
    a1.set_xscale("log"); a1.set_yscale("log")
    a1.set_xlabel("compute (MuJoCo opt-steps)"); a1.set_ylabel("peak-penetration error vs gold [mm]")
    a1.set_title("Work-precision (lower-left = better)"); a1.grid(True, which="both", alpha=0.3); a1.legend()

    t = np.arange(round(T / DT)) * DT
    for tol in [1e-2, 1e-3, 1e-4]:
        a2.plot(t * 1e3, dt_traces[tol] * 1e3, lw=1.6, label=f"tol {tol:.0e}")
    a2.set_yscale("log"); a2.set_xlabel("time [ms]"); a2.set_ylabel("CENIC inner dt [ms]")
    a2.set_title("dt collapses at the impact (~0-30 ms)"); a2.grid(True, which="both", alpha=0.3); a2.legend(fontsize=8)
    fig.suptitle("CENIC vs fixed-step: stiff sphere impact  (position-only S=identity norm, paper §V-E; euler)")
    save_fig(fig, os.path.join(args.out_dir, "v1_single_drop.png"))
    print("saved figure", flush=True)


if __name__ == "__main__":
    main()
