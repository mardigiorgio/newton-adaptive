"""Smoking-gun ablation for a CENIC author: the controller's inner dt over a stiff
articulated foot-strike, under our DEVIATED error scaling S (= diag(M)^{-1/2}, normalized,
clipped to [1,10]) vs the paper-faithful S = identity (Sec V-E).

With the deviated S the light leg DoFs dominate the inf-norm and the heavy base/contact
coordinates are suppressed, so e stays below eps_acc through the impact and dt never
refines -- effectively a fixed coarse step at exactly the contact event. S = identity
restores refinement.

    uv run --extra rl --extra examples --extra importers -m scripts.rl.v1_norm_ablation
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch  # noqa: TID253
import warp as wp

import newton
import newton.solvers

from scripts.bench.plotting import save_fig
from scripts.rl.anymal import build_anymal_model

DT = 0.01
N_STEPS = 45
DROP_H = 0.05
VZ = -1.5
TOL = 1e-3


def deviated_S(solver, world_count):
    """Reconstruct the old non-paper scaling: S = diag(M)^{-1/2} per joint (max invweight
    over the joint's DoFs), normalized so the heaviest coord = 1, clipped to [1, 10]."""
    jnt_qposadr = solver.mjw_model.jnt_qposadr.numpy()
    jnt_dofadr = solver.mjw_model.jnt_dofadr.numpy()
    invweight = solver.mjw_model.dof_invweight0.numpy()
    coords, dofs = solver._coords_per_world, solver._dofs_per_world
    njnt = len(jnt_qposadr)
    qw = np.ones((world_count, coords), dtype=np.float32)
    for w in range(world_count):
        dof_w = np.clip(invweight[w], 1e-30, None)
        for j in range(njnt):
            q_s = int(jnt_qposadr[j]); q_e = int(jnt_qposadr[j + 1]) if j + 1 < njnt else coords
            qd_s = int(jnt_dofadr[j]); qd_e = int(jnt_dofadr[j + 1]) if j + 1 < njnt else dofs
            qw[w, q_s:q_e] = np.sqrt(float(dof_w[qd_s:qd_e].max()))
        qw[w] /= qw[w].min()
    return np.clip(qw, 1.0, 10.0)


def run(scaling: str):
    m, meta = build_anymal_model(1)
    s = newton.solvers.SolverMuJoCoCENIC(m, tol=TOL, dt_inner_init=0.005, dt_inner_min=1e-6,
                                         dt_inner_max=DT, nconmax=100, njmax=210, integrator="euler")
    if scaling == "deviated":
        s._state_scale = wp.array(deviated_S(s, 1), dtype=wp.float32, device=m.device)
    s0, s1 = m.state(), m.state()
    jq = wp.to_torch(s0.joint_q); jq[2] = jq[2] + DROP_H
    s0.joint_qd.zero_(); wp.to_torch(s0.joint_qd)[2] = VZ
    newton.eval_fk(m, s0.joint_q, s0.joint_qd, s0)
    c = m.control()
    tgt = getattr(c, "joint_target_q", None) or getattr(c, "joint_target_pos", None)
    tv = wp.to_torch(tgt); tv[-12:] = torch.tensor(meta.default_joint_q, device=tv.device, dtype=tv.dtype)
    s.reset_compute_counter()
    ts, dts, zs, pen = [], [], [], 0.0
    for i in range(N_STEPS):
        s0, s1 = s.step_dt(DT, s0, s1, c)
        ts.append(i * DT * 1e3); dts.append(float(s.dt.numpy()[0])); zs.append(float(wp.to_torch(s0.joint_q)[2]))
        pen = min(pen, float(s.mjw_data.contact.dist.numpy().min()))
    return np.array(ts), np.array(dts), np.array(zs), pen, s.cumulative_substeps()


def main():
    td, dd, zd, pend, subd = run("deviated")
    ti, di, zi, peni, subi = run("identity")
    impact = int(zi.argmin())

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(td, dd * 1e3, "s-", color="tab:red", lw=2, ms=5,
            label=f"deviated S = diag(M)$^{{-1/2}}$, clip[1,10]  (our bug) — never refines, {subd} steps")
    ax.plot(ti, di * 1e3, "o-", color="tab:blue", lw=2, ms=5,
            label=f"S = identity (paper §V-E) — refines at impact, {subi} steps")
    ax.axvline(impact * DT * 1e3, color="0.5", ls=":", lw=1.2)
    ax.text(impact * DT * 1e3 + 1, ax.get_ylim()[0] * 1.3 if False else 0.012, "foot-strike",
            rotation=90, va="bottom", fontsize=8, color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("CENIC inner dt [ms]")
    ax.set_title("Our error-scaling regression vs §V-E: same articulated foot-strike\n"
                 "deviated S keeps dt pinned through the impact; S=identity collapses it")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    os.makedirs("results/plots", exist_ok=True)
    save_fig(fig, "results/plots/v1_norm_ablation.png")
    print(f"deviated: min_dt={dd.min()*1e3:.3f}ms pen={pend*1e3:.2f}mm sub={subd}")
    print(f"identity: min_dt={di.min()*1e3:.3f}ms pen={peni*1e3:.2f}mm sub={subi}")
    print("saved results/plots/v1_norm_ablation.png")


if __name__ == "__main__":
    main()
