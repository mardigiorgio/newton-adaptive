# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SAP-solver penetration statistics: same probe as v1_penetration_stats, SAP backend.

Runs the identical chaos-immune protocol (256 independent single-impact worlds,
bit-identical initial conditions via the same rng seed, impact-window horizon,
per-world fine-fixed gold) against SolverSAPAdaptive in fixed and adaptive modes.
Because the initial conditions match v1_penetration_stats exactly, the ABSOLUTE
peak-penetration columns are directly comparable across contact models: MuJoCo's
constraint compliance floor (gold median 8-20 mm depending on solref) vs SAP's
near-rigid convex contact.

Compute accounting: cumulative_substeps() returns accepted*3 unconditionally
(written for step-doubling); in fixed mode each accepted step is ONE opt-solve,
so evals = accepted for fixed and accepted*3 for adaptive.

    uv run --extra rl --extra examples --extra importers -m scripts.rl.adaptive_expts.v1_penetration_stats_sap
"""

from __future__ import annotations

import json
import os

import matplotlib  # noqa: TID253

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: TID253
import numpy as np
import warp as wp

import newton
import newton.solvers
from scripts.bench.plotting import save_fig

N_WORLDS = int(os.environ.get("PEN_WORLDS", "256"))
T = float(os.environ.get("PEN_T", "0.10"))
DT_REC = 0.002
GOLD_DT = float(os.environ.get("PEN_GOLD_DT", "5e-5"))
R = 0.05
KE = float(os.environ.get("PEN_KE", "1e8"))  # authored contact stiffness (SAP maps shape ke)
TAU_D = float(os.environ.get("PEN_TAU_D", "0.01"))  # per-shape dissipation fallback [s]
TOLS = (1e-2, 1e-3, 3e-4)
FIXED_DTS = (2e-3, 1e-3, 5e-4)


def _build():
    t = newton.ModelBuilder()
    # margin=0: in Newton's native pipeline (which SAP consumes), ShapeConfig.margin
    # is an OUTWARD SURFACE OFFSET: bodies rest margin-above-surface by design,
    # so a pair of 5 mm margins holds the bodies ~10 mm apart. Detection reach
    # comes from the
    # separate `gap` (builder.rigid_gap default 0.1 m). NOTE: mjw ZEROES authored
    # margins instead -- the two backends define margin incompatibly.
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=0.0, kf=0.0, mu=0.0, margin=0.0)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    # Author the ground's material too: SAP combines pair stiffness in SERIES
    # (k0*k1/(k0+k1)), so an unauthored ground at Newton's default ke=2.5e3 would
    # dominate any sphere stiffness and cap the pair at ~2.5e3.
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def _init_conditions():
    rng = np.random.default_rng(0)  # SAME seed/distributions as v1_penetration_stats
    z0 = rng.uniform(0.055, 0.075, N_WORLDS)
    vz = rng.uniform(-5.0, -2.0, N_WORLDS)
    return z0, vz


def _set_state(model, state, z0, vz):
    q = wp.to_torch(state.joint_q)
    qd = wp.to_torch(state.joint_qd)
    coords = model.joint_coord_count // N_WORLDS
    dofs = model.joint_dof_count // N_WORLDS
    for w in range(N_WORLDS):
        q[w * coords + 2] = float(z0[w])
        qd[w * dofs + 2] = float(vz[w])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _z_per_world(model, state) -> np.ndarray:
    coords = model.joint_coord_count // N_WORLDS
    return state.joint_q.numpy().reshape(N_WORLDS, coords)[:, 2]


def rollout(mode, dt_or_tol, z0, vz):
    m = _build()
    if mode == "fixed":
        dt = float(dt_or_tol)
        solver = newton.solvers.SolverSAPAdaptive(
            m,
            mode="fixed",
            dt_inner_init=dt,
            dt_inner_min=dt,
            dt_inner_max=dt,
            max_substeps=max(4, int(round(DT_REC / dt)) + 2),
            contact_tau_d=TAU_D,
        )
    else:
        solver = newton.solvers.SolverSAPAdaptive(
            m,
            mode="adaptive",
            tol=float(dt_or_tol),
            dt_inner_init=DT_REC,
            dt_inner_min=1e-7,
            dt_inner_max=DT_REC,
            max_substeps=64,
            contact_tau_d=TAU_D,
        )
    s0, s1 = m.state(), m.state()
    _set_state(m, s0, z0, vz)
    s1.assign(s0)
    c = m.control()
    peak = np.zeros(N_WORLDS)
    for _ in range(round(T / DT_REC)):
        s0, s1 = solver.step_dt(DT_REC, s0, s1, c)
        peak = np.maximum(peak, R - _z_per_world(m, s0))
    accepted = solver.cumulative_substeps() // 3
    evals = accepted if mode == "fixed" else accepted * 3
    return peak, evals


def stats(peak, gold_peak):
    err = np.abs(peak - gold_peak) * 1e3
    return {
        "median_err_mm": float(np.median(err)),
        "p90_err_mm": float(np.percentile(err, 90)),
        "max_err_mm": float(err.max()),
        "bias_mm": float(np.mean(peak - gold_peak) * 1e3),
        "median_pen_mm": float(np.median(peak) * 1e3),
        "max_pen_mm": float(peak.max() * 1e3),
    }


def main():
    print(f"SAP penetration probe: ke={KE:.0e} tau_d(per-shape)={TAU_D} n={N_WORLDS} T={T}", flush=True)
    z0, vz = _init_conditions()
    gold_peak, gold_evals = rollout("fixed", GOLD_DT, z0, vz)
    print(
        f"gold: fixed dt={GOLD_DT:.0e} evals={gold_evals}  peak pen over worlds: "
        f"median={np.median(gold_peak) * 1e3:.3f}mm  max={gold_peak.max() * 1e3:.3f}mm",
        flush=True,
    )

    res = {
        "meta": {
            "n_worlds": N_WORLDS,
            "T": T,
            "dt_rec": DT_REC,
            "gold_dt": GOLD_DT,
            "ke": KE,
            "tau_d": TAU_D,
            "gold_median_pen_mm": float(np.median(gold_peak) * 1e3),
            "gold_max_pen_mm": float(gold_peak.max() * 1e3),
        },
        "adaptive": [],
        "fixed": [],
    }

    for tol in TOLS:
        peak, evals = rollout("adaptive", tol, z0, vz)
        st = stats(peak, gold_peak)
        res["adaptive"].append({"tol": tol, "evals": evals, **st})
        print(
            f"  adaptive tol={tol:.0e}  evals={evals:6d}  median_err={st['median_err_mm']:6.3f}mm  "
            f"bias={st['bias_mm']:+6.3f}mm  median_pen={st['median_pen_mm']:6.3f}mm",
            flush=True,
        )

    for fdt in FIXED_DTS:
        peak, evals = rollout("fixed", fdt, z0, vz)
        st = stats(peak, gold_peak)
        res["fixed"].append({"dt": fdt, "evals": evals, **st})
        print(
            f"  fixed dt={fdt:.0e}     evals={evals:6d}  median_err={st['median_err_mm']:6.3f}mm  "
            f"bias={st['bias_mm']:+6.3f}mm  median_pen={st['median_pen_mm']:6.3f}mm",
            flush=True,
        )

    os.makedirs("results", exist_ok=True)
    with open(f"results/v1_penetration_stats_sap_ke{KE:.0e}.json", "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for series, marker, color, label in (
        (res["adaptive"], "o", "tab:blue", "SAP adaptive (tol sweep)"),
        (res["fixed"], "s", "tab:orange", "SAP fixed (dt sweep)"),
    ):
        subs = [r["evals"] for r in series]
        med = [max(r["median_err_mm"], 1e-4) for r in series]
        ax.plot(subs, med, marker + "-", color=color, lw=1.8, ms=6, label=f"{label} — median err")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SAP opt-solves (compute)")
    ax.set_ylabel("per-world peak-penetration error vs own gold [mm]")
    ax.set_title(f"SAP penetration accuracy vs compute — {N_WORLDS} single impacts (ke={KE:.0e})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, f"results/plots/v1_penetration_stats_sap_ke{KE:.0e}.png")
    print("saved results json + plot", flush=True)


if __name__ == "__main__":
    main()
