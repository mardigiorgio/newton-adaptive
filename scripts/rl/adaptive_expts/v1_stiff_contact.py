"""V1-C work-precision in the STIFF / tunneling-contact regime (the regime where
the adaptive solver's data-fidelity advantage is demonstrated): 9 spheres + 9 tilted boxes dropped
~1 m into a walled box (scripts/scenes/contact_objects). At coarse fixed dt an object
moves ~its own size per step -> deep penetration / tunneling; adaptive stepping refines
dt at impact and stays accurate.

Compares the adaptive solver (tol sweep) vs fixed-step (dt sweep incl. coarse 10 ms) against a fine
fixed gold, measuring trajectory error AND peak penetration vs compute, plus the
dt-collapse-at-impact trace.

    uv run --extra rl --extra examples --extra importers -m scripts.rl.adaptive_expts.v1_stiff_contact
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

from scripts.bench.plotting import save_fig
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_fixed_solver, make_solver

T = 1.0  # rollout seconds (fall ~1 m, impact, settle)
GOLD_DT = 1e-4
ADAPTIVE_TOLS = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
FIXED_DTS = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]  # 1e-2 == DT_OUTER (the coarse 10 ms case)


def _pen(solver) -> float:
    return float(solver.mjw_data.contact.dist.numpy().min())


def rollout_adaptive(tol: float):
    model = build_model(1)
    solver = make_solver(model, tol=tol)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    solver.reset_compute_counter()
    n = round(T / DT_OUTER)
    traj = np.zeros((n, model.joint_coord_count), dtype=np.float64)
    worst_pen = 0.0
    dt_trace = np.zeros(n)
    for i in range(n):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        traj[i] = wp.to_torch(s0.joint_q).detach().cpu().numpy()
        worst_pen = min(worst_pen, _pen(solver))
        dt_trace[i] = float(solver.dt.numpy()[0])
    return traj, solver.cumulative_substeps(), worst_pen, dt_trace


def rollout_fixed(fixed_dt: float):
    model = build_model(1)
    solver = make_fixed_solver(model)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    n_outer = round(T / DT_OUTER)
    nsub = round(DT_OUTER / fixed_dt)
    traj = np.zeros((n_outer, model.joint_coord_count), dtype=np.float64)
    worst_pen = 0.0
    for i in range(n_outer):
        for _ in range(nsub):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, fixed_dt)
            s0, s1 = s1, s0
        traj[i] = wp.to_torch(s0.joint_q).detach().cpu().numpy()
        worst_pen = min(worst_pen, _pen(solver))
    return traj, n_outer * nsub, worst_pen


def err(traj, gold):
    return float(np.mean(np.max(np.abs(traj - gold), axis=1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="results/plots")
    p.add_argument("--json", default="results/v1_stiff_contact.json")
    args = p.parse_args()

    print(f"gold: fixed dt={GOLD_DT:.0e}  DT_OUTER={DT_OUTER}  T={T}", flush=True)
    gold, gold_sub, gold_pen = rollout_fixed(GOLD_DT)
    print(f"  gold substeps={gold_sub}  gold peak_pen={gold_pen * 1e3:.3f} mm", flush=True)

    res = {
        "meta": {
            "T": T,
            "DT_OUTER": DT_OUTER,
            "gold_dt": GOLD_DT,
            "gold_substeps": gold_sub,
            "gold_pen_mm": gold_pen * 1e3,
        },
        "adaptive": [],
        "fixed": [],
    }
    dt_traces = {}
    for tol in ADAPTIVE_TOLS:
        traj, sub, pen, dtt = rollout_adaptive(tol)
        res["adaptive"].append({"tol": tol, "substeps": sub, "error": err(traj, gold), "pen_mm": pen * 1e3})
        dt_traces[tol] = dtt
        print(f"  adaptive tol={tol:.0e}  sub={sub:6d}  err={err(traj, gold):.3e}  pen={pen * 1e3:7.2f} mm", flush=True)
    for fdt in FIXED_DTS:
        traj, sub, pen = rollout_fixed(fdt)
        res["fixed"].append({"fixed_dt": fdt, "substeps": sub, "error": err(traj, gold), "pen_mm": pen * 1e3})
        print(f"  fixed dt={fdt:.0e}  sub={sub:6d}  err={err(traj, gold):.3e}  pen={pen * 1e3:7.2f} mm", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(res, f, indent=2)
    os.makedirs(args.out_dir, exist_ok=True)

    cs, fs = res["adaptive"], res["fixed"]
    # Fig 1: error + penetration vs compute (two panels).
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    a1.plot(
        [r["substeps"] for r in cs], [r["error"] for r in cs], "o-", color="tab:blue", lw=1.8, ms=6, label="adaptive"
    )
    a1.plot(
        [r["substeps"] for r in fs], [r["error"] for r in fs], "s-", color="tab:orange", lw=1.8, ms=6, label="fixed"
    )
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("compute (MuJoCo opt-steps)")
    a1.set_ylabel("trajectory error vs gold (inf-norm joint_q)")
    a1.set_title("Accuracy vs compute")
    a1.grid(True, which="both", alpha=0.3)
    a1.legend()
    a2.plot(
        [r["substeps"] for r in cs],
        [abs(r["pen_mm"]) for r in cs],
        "o-",
        color="tab:blue",
        lw=1.8,
        ms=6,
        label="adaptive",
    )
    a2.plot(
        [r["substeps"] for r in fs],
        [abs(r["pen_mm"]) for r in fs],
        "s-",
        color="tab:orange",
        lw=1.8,
        ms=6,
        label="fixed",
    )
    a2.axhline(abs(res["meta"]["gold_pen_mm"]), color="k", ls="--", lw=0.8, label="gold")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("compute (MuJoCo opt-steps)")
    a2.set_ylabel("peak penetration |dist| [mm]")
    a2.set_title("Penetration vs compute")
    a2.grid(True, which="both", alpha=0.3)
    a2.legend()
    fig.suptitle("V1 stiff-contact work-precision: 18 objects dropped into a box (lower-left = better)")
    save_fig(fig, os.path.join(args.out_dir, "v1_stiff_contact.png"))

    # Fig 2: dt-collapse trace at impact.
    fig2, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(round(T / DT_OUTER)) * DT_OUTER
    for tol in [1e-2, 1e-3, 3e-4]:
        ax.plot(t, dt_traces[tol] * 1e3, lw=1.4, label=f"tol {tol:.0e}")
    ax.set_yscale("log")
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("adaptive inner dt [ms]")
    ax.set_title("dt collapses at impact (objects hit ~0.45 s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    save_fig(fig2, os.path.join(args.out_dir, "v1_stiff_dt_trace.png"))
    print("saved figures", flush=True)


if __name__ == "__main__":
    main()
