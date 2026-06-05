# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Energy-drift trace: per-world adaptive vs global fixed-dt.

Figure 4 of the poster. Measures the **relative drift of total mechanical
energy** (E = KE + PE) away from a fine-dt reference trajectory, over
simulation time, on a contact-heavy scene. Per-world adaptive stepping keeps
energy close to the reference; a global fixed-10ms step drifts much further
(it cannot resolve the stiff contact at 10 ms).

Why total energy, not raw KE: kinetic energy is dominated by the gross physical
motion (objects fall, then settle), which is identical for every solver, so it
hides the numerical error. Energy *drift* -- the spurious change in total
mechanical energy relative to an accurate reference -- is the quantity adaptive
stepping controls and a too-large fixed step violates. It is a scalar magnitude
vs a reference, so it is not confounded by trajectory chaos.

Definitions per world, summed over its free bodies:

    E       = sum_b [ 0.5 m_b |v_b|^2 + 0.5 w_b . (R I_b R^T) w_b ]   (KE)
              - sum_b m_b (g . p_b)                                    (PE)
    drift   = | E_solver(t) - E_ref(t) | / | E_ref(t=0) |

Example::

    uv run python -m scripts.bench.benchmarks.energy_trace \\
        --scene contact_objects \\
        --kinds mujoco_adaptive_1e-3 mujoco_fixed_10ms \\
        --ref-kind mujoco_fixed_1ms --n 64 --steps 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from scripts.bench.plotting import STYLES, save_fig
from scripts.scenes import _registry as _scene_registry

_BIG = 1.0e12  # clip for diverged (NaN/inf) worlds so plots stay finite


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Quaternions [N,4] (x,y,z,w) -> rotation matrices [N,3,3]."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def total_energy_per_world(model, state, n: int, g: np.ndarray) -> np.ndarray:
    """Total mechanical energy (KE + gravitational PE) per world [J], shape [n]."""
    body_q = state.body_q.numpy().astype(np.float64)        # [B,7] (p, quat)
    body_qd = state.body_qd.numpy().astype(np.float64)       # [B,6] (v, w)
    mass = model.body_mass.numpy().astype(np.float64)        # [B]
    inertia = model.body_inertia.numpy().astype(np.float64)  # [B,3,3] (body frame)

    v = body_qd[:, 0:3]
    w = body_qd[:, 3:6]
    p = body_q[:, 0:3]
    R = _quat_to_R(body_q[:, 3:7])
    I_world = R @ inertia @ np.transpose(R, (0, 2, 1))
    ke = 0.5 * mass * np.einsum("bi,bi->b", v, v) + 0.5 * np.einsum("bi,bij,bj->b", w, I_world, w)
    pe = -mass * (p @ g)  # PE = -m (g . p); g points down so this rises with height
    e = ke + pe
    e = np.where(np.isfinite(e), e, _BIG)
    bpw = e.shape[0] // n
    return e.reshape(n, bpw).sum(axis=1)


def energy_series(scene_entry, kind: str, n: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Run `steps` outer steps; return (sim_time[steps], E[steps, n])."""
    builder = scene_entry.solver_factories[kind]
    model = scene_entry.build_model_randomized(n)
    solver, step_fn = builder(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    import newton
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    if model.gravity is not None:
        g = model.gravity.numpy().astype(np.float64)
        g = g[0] if g.ndim > 1 else g  # per-world array -> single vector (uniform)
    else:
        g = np.array([0.0, 0.0, -9.81])
    dt_outer = getattr(scene_entry, "dt_outer", 0.02)

    times, Es = [], []
    t = 0.0
    for i in range(steps):
        try:
            s0, s1 = step_fn(model, s0, s1, ctrl)
        except Exception as e:
            print(f"  {kind}: step {i} raised {type(e).__name__}: {e}")
            break
        wp.synchronize()
        t += dt_outer
        times.append(t)
        Es.append(total_energy_per_world(model, s0, n, g))
        if i % 20 == 0 or i < 3:
            print(f"    {kind} step={i:3d} t={t:.3f}s  E med={np.median(Es[-1]):.3f}", flush=True)
    return np.asarray(times), np.asarray(Es)


def plot_drift(times, drift: dict[str, np.ndarray], scene: str, n: int,
               ref_kind: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for kind, d in drift.items():
        style = STYLES.get(kind)
        color = style.color if style else None
        label = style.label if style else kind
        ax.plot(times, np.median(d, axis=1), color=color, lw=2.0, label=label)
        ax.fill_between(times, np.percentile(d, 25, axis=1), np.percentile(d, 75, axis=1),
                        color=color, alpha=0.18)
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Relative energy drift  |E - E_ref| / |E_ref(0)|\n(median, IQR band over worlds)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title(f"Energy drift vs {ref_kind} reference  (scene={scene}, N={n})", fontsize=11)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    save_fig(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="contact_objects")
    p.add_argument("--kinds", nargs="+", default=["mujoco_adaptive_1e-3", "mujoco_fixed_10ms"])
    p.add_argument("--ref-kind", default="mujoco_fixed_1ms",
                   help="Fine-dt solver used as the energy ground truth.")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--out-dir", default="scripts/bench/results/energy_trace")
    args = p.parse_args()

    scene = _scene_registry.get(args.scene)
    available = scene.solver_kinds()
    for k in [args.ref_kind, *args.kinds]:
        if k not in available:
            print(f"Unknown kind {k!r}. Available: {available}")
            return 1

    print(f"=== reference: {args.ref_kind} ===", flush=True)
    _, E_ref = energy_series(scene, args.ref_kind, args.n, args.steps)
    E0 = np.abs(E_ref[0]).mean()

    drift: dict[str, np.ndarray] = {}
    times = None
    for k in args.kinds:
        print(f"=== {k} ===", flush=True)
        t, E = energy_series(scene, k, args.n, args.steps)
        m = min(len(E), len(E_ref))
        drift[k] = np.abs(E[:m] - E_ref[:m]) / E0
        times = t[:m]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scene}_N{args.n}_steps{args.steps}_drift"
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump({"scene": args.scene, "n": args.n, "steps": args.steps,
                   "ref_kind": args.ref_kind, "times": times.tolist(),
                   "drift": {k: {"median": np.median(d, axis=1).tolist(),
                                 "p25": np.percentile(d, 25, axis=1).tolist(),
                                 "p75": np.percentile(d, 75, axis=1).tolist()}
                             for k, d in drift.items()}}, f, indent=2)
    plot_path = out_dir / f"{stem}.png"
    plot_drift(times, drift, args.scene, args.n, args.ref_kind, plot_path)
    for k, d in drift.items():
        print(f"  {k}: mean drift={np.median(d, axis=1).mean():.3e}  final={np.median(d[-1]):.3e}")
    print(f"\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
