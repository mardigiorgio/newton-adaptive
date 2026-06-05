# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-step error vs tolerance: the adaptive controller working.

Figure 4 of the poster. Plots the per-step integration error (the weighted
inf-norm step-doubling estimate -- the exact quantity the tolerance bounds) over
simulation time, for two solvers on the same contact-heavy scene:

  * Per-world adaptive (tol = eps_acc): the controller shrinks dt to hold the
    error AT or BELOW the tolerance line.
  * Global fixed 10 ms: no error control -- the same estimator, measured on a
    forced 10 ms step (10 ms vs two 5 ms half-steps), sits well ABOVE the
    tolerance and does nothing about it.

Both errors come from the IDENTICAL estimator, so the comparison is apples to
apples. The fixed-10ms error is obtained by clamping the adaptive wrapper to a
fixed 10 ms inner step (dt_min = dt_max = 10 ms); it still reports the step-
doubling error every step but cannot adapt -- exactly a fixed 10 ms stepper that
also tells us its error.

Example::

    uv run python -m scripts.bench.benchmarks.tol_trace --n 64 --steps 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.bench.plotting import save_fig
from scripts.scenes import _solvers as _s
from scripts.scenes import contact_objects as co

TOL = 1e-3
FIXED_DT = 0.01  # 10 ms
# Measure at a 10 ms outer cadence so the fixed-10ms solver takes exactly ONE
# clean inner step per boundary (K=1). With dt_outer=2*dt_inner, floating-point
# 0.01+0.01 != 0.02 intermittently spawns a 3rd no-op inner step whose ~0 error
# contaminates last_error for ALL lockstep worlds -> a spurious error floor.
DT_OUTER = FIXED_DT

# Style keys reuse the bench palette family.
_ADAPT_COLOR = "#1f5fb4"
_FIXED_COLOR = "#e07b1a"


def _series(builder, model_n: int, steps: int, n: int) -> np.ndarray:
    """Run `steps` outer steps; return last_error[steps, n]."""
    model = co.build_model_randomized(model_n)
    solver, step_fn = builder(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    errs = []
    for i in range(steps):
        s0, s1 = step_fn(model, s0, s1, ctrl)
        wp.synchronize()
        errs.append(solver.last_error.numpy().copy())
        if i % 20 == 0 or i < 3:
            e = errs[-1]
            print(f"    step={i:3d}  err med={np.median(e):.2e}  max={e.max():.2e}", flush=True)
    return np.asarray(errs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--tol", type=float, default=TOL)
    p.add_argument("--out-dir", default="scripts/bench/results/tol_trace")
    args = p.parse_args()

    wp.init()

    adaptive = _s.mujoco_adaptive_factory(
        tol=args.tol, nconmax=co._NCON, njmax=co._NJM, dt_outer=DT_OUTER,
        dt_inner_min=1e-6, dt_inner_max=DT_OUTER, use_mujoco_contacts=True,
    )
    # Fixed 10 ms: clamp inner dt to 10 ms so it cannot adapt, but still report
    # the step-doubling error each step under the identical estimator.
    fixed10 = _s.mujoco_adaptive_factory(
        tol=args.tol, nconmax=co._NCON, njmax=co._NJM, dt_outer=DT_OUTER,
        dt_inner_min=FIXED_DT, dt_inner_max=FIXED_DT, use_mujoco_contacts=True,
    )

    print(f"=== adaptive (tol={args.tol:.0e}) ===", flush=True)
    e_ad = _series(adaptive, args.n, args.steps, args.n)
    print("=== fixed 10 ms ===", flush=True)
    e_fx = _series(fixed10, args.n, args.steps, args.n)

    t = np.arange(1, e_ad.shape[0] + 1) * DT_OUTER

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _smooth(y, k=7):
        if len(y) < k:
            return y
        pad = k // 2
        yp = np.pad(y, pad, mode="edge")
        return np.convolve(yp, np.ones(k) / k, mode="valid")

    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13,
                         "legend.fontsize": 12.5})
    fig, ax = plt.subplots(figsize=(8.0, 5.6))

    m = min(len(t), len(e_ad), len(e_fx))
    tt = t[:m]
    # Verdict shading: below tolerance = accuracy met (green), above = not (red).
    ax.axhspan(1e-8, args.tol, color="#2e7d32", alpha=0.07, zorder=0)
    ax.axhspan(args.tol, 1e1, color="#c62828", alpha=0.07, zorder=0)

    # One clean line per solver: the worst-world per-step error (the quantity the
    # controller bounds), lightly smoothed. Worst-world keeps the claim exact --
    # adaptive's worst world stays <= tol; fixed-10ms's worst world stays > tol
    # (even its best free-fall steps don't dip below).
    for e, color, label in ((e_ad, _ADAPT_COLOR, f"per-world adaptive (tol = {args.tol:.0e})"),
                            (e_fx, _FIXED_COLOR, "global fixed 10 ms")):
        ax.plot(tt, _smooth(np.max(e[:m], axis=1)), color=color, lw=2.8, label=label)
    ax.axhline(args.tol, ls="--", color="k", lw=1.6,
               label=f"tolerance $\\epsilon_{{acc}}$ = {args.tol:.0e}")

    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel(r"Worst-world per-step error  $\max_i |\Delta q_i|$")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2.0)
    ax.set_xlim(tt[0], tt[-1])
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contact_objects_N{args.n}_steps{args.steps}_tol{args.tol:.0e}"
    save_fig(fig, out_dir / f"{stem}.png")
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"n": args.n, "steps": args.steps, "tol": args.tol,
                   "t": t.tolist(),
                   "adaptive_max": np.max(e_ad, axis=1).tolist(),
                   "fixed10_max": np.max(e_fx, axis=1).tolist()}, f, indent=2)
    ad_max, fx_max = np.max(e_ad, axis=1), np.max(e_fx, axis=1)
    print(f"\nadaptive worst-world err range: [{ad_max.min():.2e}, {ad_max.max():.2e}]  "
          f"(tol={args.tol:.0e}; below tol: {(ad_max <= args.tol).mean()*100:.0f}% of steps)")
    print(f"fixed10  worst-world err range: [{fx_max.min():.2e}, {fx_max.max():.2e}]  "
          f"(above tol: {(fx_max > args.tol).mean()*100:.0f}% of steps)")
    print(f"Plot: {out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
