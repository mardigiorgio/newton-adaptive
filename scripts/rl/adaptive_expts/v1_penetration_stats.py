# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Chaos-immune penetration comparison: N independent single-impact worlds.

The multi-body peak-penetration metric (v1_stiff_contact) is chaos-contaminated:
after trajectories diverge, a rollout's peak |penetration| reflects which valid
trajectory was sampled, not contact-resolution quality. This experiment removes the
confound: N worlds each contain ONE sphere dropped onto the ground with randomized
per-world height/velocity, simulated only through the impact window (before
nondeterministic divergence is measurable), and each world is scored against ITS OWN
fine-fixed gold. The reported metric is the distribution over worlds of the
per-world peak-penetration error |pen_method(w) - pen_gold(w)|, plus the signed bias
(negative = shallower than truth, e.g. spurious ejection; positive = deeper, e.g.
overshoot/tunneling), for an adaptive tolerance sweep vs a fixed-dt sweep, against
compute (MuJoCo evals).

    uv run --extra rl --extra examples --extra importers -m scripts.rl.adaptive_expts.v1_penetration_stats
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

N_WORLDS = 256
T = 0.10  # impact window [s]; per-world impact occurs within the first ~30 ms
DT_REC = 0.002  # control/sampling period [s]
GOLD_DT = float(os.environ.get("PEN_GOLD_DT", "2e-5"))
R = 0.05
KE = float(os.environ.get("PEN_KE", "1e4"))  # contact stiffness; sweep via env for regime studies
KD = float(os.environ.get("PEN_KD", "0"))  # contact damping. NOTE: convert_solref only maps
# (ke, kd) -> MuJoCo solref when BOTH are > 0; with kd=0 the contact runs at MuJoCo DEFAULT
# compliance (solref 0.02/1) and ke is inert. timeconst=2/kd, dampratio=(kd/2)*sqrt(1/ke).
INTEG = "euler"
TOLS = (1e-2, 3e-3, 1e-3, 3e-4)
FIXED_DTS = (5e-3, 2e-3, 1e-3, 5e-4)


def _build():
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=KE, kd=KD, kf=0.0, mu=0.0, margin=0.005)
    bd = t.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.07)))
    t.add_shape_sphere(bd, radius=R, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    # Author the ground's material too: with equal geom priority mjw AVERAGES the
    # two geoms' solref 50/50, so an unauthored ground (default ke=2.5e3, kd=0 ->
    # default solref 0.02/1) dilutes any authored sphere stiffness to ~half-default
    # (this capped the original stiffness sweep at ~10 ms pair timeconst).
    b.add_ground_plane(cfg=cfg)
    return b.finalize()


def _init_conditions():
    rng = np.random.default_rng(0)
    z0 = rng.uniform(0.055, 0.075, N_WORLDS)  # 5..25 mm clearance above ground
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


def _z_per_world(model, state) -> np.ndarray:
    coords = model.joint_coord_count // N_WORLDS
    return state.joint_q.numpy().reshape(N_WORLDS, coords)[:, 2]


def rollout_adaptive(tol, z0, vz):
    m = _build()
    s = newton.solvers.SolverMuJoCoAdaptive(
        m,
        tol=tol,
        dt_inner_init=DT_REC,
        dt_inner_min=1e-7,
        dt_inner_max=DT_REC,
        nconmax=64,
        njmax=128,
        integrator=INTEG,
    )
    s0, s1 = m.state(), m.state()
    _set_state(m, s0, z0, vz)
    c = m.control()
    peak = np.zeros(N_WORLDS)
    for _ in range(round(T / DT_REC)):
        s0, s1 = s.step_dt(DT_REC, s0, s1, c)
        peak = np.maximum(peak, R - _z_per_world(m, s0))
    return peak, s.cumulative_substeps()


def rollout_fixed(fixed_dt, z0, vz):
    m = _build()
    s = newton.solvers.SolverMuJoCo(m, separate_worlds=True, nconmax=64, njmax=128, integrator=INTEG)
    s0, s1 = m.state(), m.state()
    _set_state(m, s0, z0, vz)
    c = m.control()
    n = max(1, round(DT_REC / fixed_dt))
    peak = np.zeros(N_WORLDS)
    for _ in range(round(T / DT_REC)):
        for _k in range(n):
            s0.clear_forces()
            s.step(s0, s1, c, None, fixed_dt)
            s0, s1 = s1, s0
        peak = np.maximum(peak, R - _z_per_world(m, s0))
    return peak, n * round(T / DT_REC)


def stats(peak, gold_peak):
    err = np.abs(peak - gold_peak) * 1e3  # mm
    bias = float(np.mean(peak - gold_peak) * 1e3)
    return {
        "median_err_mm": float(np.median(err)),
        "p90_err_mm": float(np.percentile(err, 90)),
        "max_err_mm": float(err.max()),
        "bias_mm": bias,
    }


def main():
    if KE > 0.0 and KD > 0.0:
        print(
            f"effective contact solref: timeconst={2.0 / KD:.2e}s dampratio={KD / 2.0 * (1.0 / KE) ** 0.5:.2f}",
            flush=True,
        )
    else:
        print("effective contact solref: MUJOCO DEFAULT (0.02, 1) -- ke/kd not mapped (need both > 0)", flush=True)
    z0, vz = _init_conditions()
    gold_peak, gold_sub = rollout_fixed(GOLD_DT, z0, vz)
    print(
        f"gold: dt={GOLD_DT:.0e} sub={gold_sub}  peak pen over worlds: "
        f"median={np.median(gold_peak) * 1e3:.2f}mm  max={gold_peak.max() * 1e3:.2f}mm",
        flush=True,
    )

    res = {
        "meta": {
            "n_worlds": N_WORLDS,
            "T": T,
            "dt_rec": DT_REC,
            "gold_dt": GOLD_DT,
            "ke": KE,
            "gold_median_pen_mm": float(np.median(gold_peak) * 1e3),
        },
        "adaptive": [],
        "fixed": [],
    }

    for tol in TOLS:
        peak, sub = rollout_adaptive(tol, z0, vz)
        st = stats(peak, gold_peak)
        res["adaptive"].append({"tol": tol, "substeps": sub, **st})
        print(
            f"  adaptive tol={tol:.0e}  sub={sub:6d}  median_err={st['median_err_mm']:6.3f}mm  "
            f"p90={st['p90_err_mm']:6.3f}mm  bias={st['bias_mm']:+6.3f}mm",
            flush=True,
        )

    for fdt in FIXED_DTS:
        peak, sub = rollout_fixed(fdt, z0, vz)
        st = stats(peak, gold_peak)
        res["fixed"].append({"dt": fdt, "substeps": sub, **st})
        print(
            f"  fixed dt={fdt:.0e}     sub={sub:6d}  median_err={st['median_err_mm']:6.3f}mm  "
            f"p90={st['p90_err_mm']:6.3f}mm  bias={st['bias_mm']:+6.3f}mm",
            flush=True,
        )

    os.makedirs("results", exist_ok=True)
    with open(f"results/v1_penetration_stats_ke{KE:.0e}_kd{KD:.0e}.json", "w") as f:
        json.dump(res, f, indent=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for series, marker, color, label in (
        (res["adaptive"], "o", "tab:blue", "adaptive (tol sweep)"),
        (res["fixed"], "s", "tab:orange", "fixed (dt sweep)"),
    ):
        subs = [r["substeps"] for r in series]
        med = [r["median_err_mm"] for r in series]
        p90 = [r["p90_err_mm"] for r in series]
        ax.plot(subs, med, marker + "-", color=color, lw=1.8, ms=6, label=f"{label} — median")
        ax.plot(subs, p90, marker + "--", color=color, lw=1.0, ms=4, alpha=0.6, label=f"{label} — p90")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MuJoCo evals (compute)")
    ax.set_ylabel("per-world peak-penetration error vs own gold [mm]")
    ax.set_title(
        f"Penetration accuracy vs compute — {N_WORLDS} independent single impacts (ke={KE:.0e})\n"
        "(chaos-immune: per-world gold, impact-window horizon)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    save_fig(fig, f"results/plots/v1_penetration_stats_ke{KE:.0e}_kd{KD:.0e}.png")
    print("saved results/v1_penetration_stats.json + results/plots/v1_penetration_stats.png", flush=True)


if __name__ == "__main__":
    main()
