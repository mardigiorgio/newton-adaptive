"""V1-B work-precision: accuracy vs compute for CENIC (adaptive) vs fixed-step on a
deterministic ANYmal foot-strike, against a fine fixed gold rollout.

Scenario: a single ANYmal, standing PD targets, base raised ~0.08 m, dropped so the
feet strike the ground (a stiff contact transient -- the regime where Phase 0b's caveat
says error-controlled stepping should help). For each backend we roll out the same
T seconds at the same outer period DT, record joint_q at every DT boundary, and compare
to the gold rollout. Compute is the total MuJoCo opt-step count (CENIC: the solver's
cumulative_substeps incl. rejected attempts; fixed: analytic).

    uv run --extra rl --extra examples --extra importers -m scripts.rl.v1_work_precision
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: TID253
import warp as wp

import newton

from scripts.bench.plotting import save_fig

from ..anymal.anymal import build_anymal_model
from ..anymal.backends import Backend, BackendSpec

DT = 0.008  # outer period [s]; every fixed_dt below divides it
T = 0.6  # rollout seconds
DROP_H = 0.08  # base raise above standing [m] -> foot-strike on drop
GOLD_DT = 2.5e-4  # gold fixed timestep (32 substeps per DT)
CENIC_TOLS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
FIXED_DTS = [8e-3, 4e-3, 2e-3, 1e-3]


def _standing_control(model, meta):
    control = model.control()
    tgt = getattr(control, "joint_target_q", None)
    if tgt is None:
        tgt = getattr(control, "joint_target_pos", None)
    tv = wp.to_torch(tgt)
    k = len(meta.default_joint_q)
    tv[-k:] = torch.tensor(meta.default_joint_q, device=tv.device, dtype=tv.dtype)
    return control


def rollout(spec: BackendSpec, t_sec: float, dt: float, drop_h: float):
    """Return (traj [n_outer, coords], total MuJoCo opt-steps)."""
    model, meta = build_anymal_model(1)
    backend = Backend(model, spec)
    s0, s1 = model.state(), model.state()

    # Drop IC: default standing pose, base raised, zero velocity.
    jq = wp.to_torch(s0.joint_q)
    jq[2] = jq[2] + drop_h
    s0.joint_qd.zero_()
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)

    control = _standing_control(model, meta)
    if spec.is_adaptive:
        backend.solver.reset_compute_counter()

    n_outer = round(t_sec / dt)
    coords = model.joint_coord_count
    traj = np.zeros((n_outer, coords), dtype=np.float64)
    for i in range(n_outer):
        s0, s1 = backend.advance(dt, s0, s1, control)
        traj[i] = wp.to_torch(s0.joint_q).detach().cpu().numpy()

    if spec.is_adaptive:
        substeps = backend.solver.cumulative_substeps()
    else:
        substeps = n_outer * round(dt / spec.fixed_dt)
    return traj, int(substeps)


def traj_error(traj: np.ndarray, gold: np.ndarray) -> float:
    """Mean over DT boundaries of the inf-norm joint_q deviation from gold."""
    return float(np.mean(np.max(np.abs(traj - gold), axis=1)))


def main():
    p = argparse.ArgumentParser(description="V1 work-precision: ANYmal foot-strike.")
    p.add_argument("--out", default="results/plots/v1_work_precision.png")
    p.add_argument("--json", default="results/v1_work_precision.json")
    args = p.parse_args()

    print(f"gold: fixed dt={GOLD_DT:.0e}  DT={DT}  T={T}  drop={DROP_H}", flush=True)
    gold, gold_sub = rollout(BackendSpec(kind="fixed", fixed_dt=GOLD_DT), T, DT, DROP_H)
    print(f"  gold substeps={gold_sub}", flush=True)

    res = {"meta": {"DT": DT, "T": T, "drop_h": DROP_H, "gold_dt": GOLD_DT, "gold_substeps": gold_sub},
           "cenic": [], "fixed": []}
    for tol in CENIC_TOLS:
        traj, sub = rollout(BackendSpec(kind="cenic", tol=tol), T, DT, DROP_H)
        err = traj_error(traj, gold)
        res["cenic"].append({"tol": tol, "substeps": sub, "error": err})
        print(f"  CENIC tol={tol:.0e}  substeps={sub:6d}  err={err:.3e}", flush=True)
    for fdt in FIXED_DTS:
        traj, sub = rollout(BackendSpec(kind="fixed", fixed_dt=fdt), T, DT, DROP_H)
        err = traj_error(traj, gold)
        res["fixed"].append({"fixed_dt": fdt, "substeps": sub, "error": err})
        print(f"  fixed dt={fdt:.0e}  substeps={sub:6d}  err={err:.3e}", flush=True)

    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(res, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    cs, fs = res["cenic"], res["fixed"]
    ax.plot([r["substeps"] for r in cs], [r["error"] for r in cs], "o-",
            color="tab:blue", lw=1.8, ms=6, label="CENIC (adaptive)")
    ax.plot([r["substeps"] for r in fs], [r["error"] for r in fs], "s-",
            color="tab:orange", lw=1.8, ms=6, label="fixed-step")
    for r in cs:
        ax.annotate(f"tol {r['tol']:.0e}", (r["substeps"], r["error"]), fontsize=6,
                    color="tab:blue", xytext=(3, 3), textcoords="offset points")
    for r in fs:
        ax.annotate(f"{r['fixed_dt']*1e3:.0f}ms", (r["substeps"], r["error"]), fontsize=6,
                    color="tab:orange", xytext=(3, -8), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compute  (total MuJoCo opt-steps over the rollout)")
    ax.set_ylabel("trajectory error vs gold  (mean inf-norm joint_q)")
    ax.set_title("V1 work-precision: ANYmal foot-strike (drop + standing PD)\nlower-left = more accurate per unit compute")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save_fig(fig, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
