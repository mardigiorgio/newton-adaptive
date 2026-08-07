# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""dt-over-time trace for the adaptive solver on an in-hand Allegro manipulation scene.

Answers the question a floor-occupancy histogram cannot: does the controller's timestep
sit below a "bad" threshold for a PROLONGED stretch, or only spike there? A histogram
aggregates the whole run, so a sustained collapse and a one-frame transient look the same.

Scene: Wonik Allegro hand holding a cube (``allegro_left_hand_with_cube.usda``), fingers
driven by the same sinusoidal trajectory as ``newton.examples.robot_allegro_hand``. That
is the same contact regime as the IsaacLab reorient task, runnable without IsaacLab.

Per boundary it records, with one host read per boundary (a diagnostic, not a hot path):
  * iterations used -> effective mean dt = dt_outer / iterations
  * the controller's carried ``ideal_dt`` across worlds (min / median / max)

Reported: the dt trace, the fraction of time below the threshold, and -- the headline --
the LONGEST CONTIGUOUS stretch below it.

Usage:
    uv run python scripts/rl/adaptive_expts/dt_trace_allegro.py --worlds 8 --frames 200
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode, ModelFlags

wp.init()


@wp.kernel
def _move_hand(
    joint_q_start: wp.array[wp.int32],
    joint_limit_lower: wp.array[wp.float32],
    joint_limit_upper: wp.array[wp.float32],
    sim_time: wp.array[wp.float32],
    sim_dt: float,
    hand_rotation: wp.quat,
    joint_target_q: wp.array[wp.float32],
    joint_parent_xform: wp.array[wp.transform],
):
    """Sinusoidal finger trajectory + slow root rotation (mirrors the stock example)."""
    world_id = wp.tid()
    root_joint_id = world_id * 22
    t = sim_time[world_id]
    root_dof_start = joint_q_start[root_joint_id]

    for i in range(20):
        di = root_dof_start + i
        target = wp.sin(t + float(i * 6) * 0.1) * 0.08 + 0.3
        joint_target_q[di] = wp.clamp(target, joint_limit_lower[di], joint_limit_upper[di])

    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(t) * 0.1)
    root_xform = joint_parent_xform[root_joint_id]
    joint_parent_xform[root_joint_id] = wp.transform(root_xform.p, q * hand_rotation)
    sim_time[world_id] += sim_dt


def build_scene(world_count: int):
    """Allegro hand + cube, replicated. Returns (model, hand_rotation)."""
    newton.use_coord_layout_targets = True
    hand_rotation = wp.normalize(wp.quat(0.21643, 0.706218, -0.648166, 0.185191))

    hand = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(hand)
    hand.default_shape_cfg.ke = 1.0e3
    hand.default_shape_cfg.kd = 1.0e2
    hand.default_shape_cfg.margin = 0.005
    hand.default_shape_cfg.gap = 0.015

    asset_path = newton.utils.download_asset("wonik_allegro")
    hand.add_usd(
        str(asset_path / "usd" / "allegro_left_hand_with_cube.usda"),
        xform=wp.transform(wp.vec3(0, 0, 0.5), wp.quat_identity()),
        enable_self_collisions=False,
        ignore_paths=[".*Dummy", ".*CollisionPlane"],
        hide_collision_shapes=True,
    )

    # Drive gains on the hand only; the trailing 6 dofs are the free-floating cube.
    for i in range(hand.joint_dof_count - 6):
        hand.joint_target_ke[i] = 150
        hand.joint_target_kd[i] = 5
        hand.joint_q[i] = 0.3
        hand.joint_target_q[i] = 0.3
        if hand.joint_label[i][-2:] == "_0":
            hand.joint_q[i] = 0.6
            hand.joint_target_q[i] = 0.6
        hand.joint_target_mode[i] = int(JointTargetMode.POSITION)
        if hand.joint_type[i] == newton.JointType.REVOLUTE:
            hand.joint_armature[i] = 1e-2

    q = np.array(hand.joint_q)
    q[-7:-4] += np.array([0.0, 0.0, 0.05])
    q[-4:] = wp.quat_rpy(0.3, 0.5, 0.1)
    hand.joint_q = q.tolist()

    builder = newton.ModelBuilder()
    builder.replicate(hand, world_count)
    builder.default_shape_cfg.ke = 1.0e3
    builder.default_shape_cfg.kd = 1.0e2
    builder.add_ground_plane()
    return builder.finalize(), hand_rotation


def longest_run_below(values: np.ndarray, dt_per_sample: float, threshold: float) -> float:
    """Longest contiguous span [s] over which ``values`` stays below ``threshold``."""
    below = values < threshold
    best = run = 0
    for b in below:
        run = run + 1 if b else 0
        best = max(best, run)
    return best * dt_per_sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--frames", type=int, default=200, help="control ticks (boundaries) to run")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--threshold", type=float, default=1e-4, help="'bad dt' threshold [s]")
    p.add_argument("--dt-min", type=float, default=1e-6)
    p.add_argument("--dt-init", type=float, default=1e-2)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--max-substeps", type=int, default=256)
    p.add_argument(
        "--dt-outer",
        type=float,
        default=1.0 / 120.0,
        help="boundary period [s]; default 1/120 matches the IsaacLab Allegro tasks",
    )
    p.add_argument("--out", type=str, default="dt_trace_allegro")
    args = p.parse_args()

    dt_outer = args.dt_outer

    model, hand_rotation = build_scene(args.worlds)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        solver="newton",
        integrator="implicitfast",
        njmax=200,
        nconmax=300,
        impratio=20.0,
        cone="elliptic",
        iterations=100,
        ls_iterations=50,
        tol=args.tol,
        dt_inner_init=args.dt_init,
        dt_inner_min=args.dt_min,
        max_substeps=args.max_substeps,
        dt_histogram=True,
    )

    s0, s1 = model.state(), model.state()
    control = model.control()
    world_time = wp.zeros(args.worlds, dtype=wp.float32)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model.state())

    def drive():
        wp.launch(
            _move_hand,
            dim=args.worlds,
            inputs=[
                model.joint_q_start,
                model.joint_limit_lower,
                model.joint_limit_upper,
                world_time,
                dt_outer,
                hand_rotation,
            ],
            outputs=[control.joint_target_q, model.joint_X_p],
        )
        solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)

    for _ in range(args.warmup):
        drive()
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)

    solver.reset_dt_histogram()
    solver.reset_compute_counter()

    iters, ideal_lo, ideal_med, ideal_hi = [], [], [], []
    for _ in range(args.frames):
        drive()
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)
        # One host read per boundary. Diagnostic only -- never do this in a training loop.
        iters.append(int(solver.iteration_count.numpy()[0]))
        d = solver._ideal_dt.numpy()
        ideal_lo.append(float(d.min()))
        ideal_med.append(float(np.median(d)))
        ideal_hi.append(float(d.max()))

    iters = np.asarray(iters, dtype=np.float64)
    eff_dt = dt_outer / np.maximum(iters, 1.0)
    ideal_lo = np.asarray(ideal_lo)
    ideal_med = np.asarray(ideal_med)
    ideal_hi = np.asarray(ideal_hi)
    t = np.arange(args.frames) * dt_outer

    stats = solver.dt_histogram_stats()
    edges = solver.dt_histogram_edges
    counts = solver.dt_histogram.numpy()
    # Bins 1..k-1 lie below the threshold; edges[i-1] is bin i's lower edge.
    below_bins = int(np.searchsorted(edges, args.threshold, side="left")) + 1
    frac_below_hist = counts[1:below_bins].sum() / max(counts.sum(), 1)

    summary = {
        "scene": "wonik_allegro hand + cube (in-hand manipulation)",
        "worlds": args.worlds,
        "frames": args.frames,
        "dt_outer": dt_outer,
        "threshold": args.threshold,
        "dt_min_cfg": args.dt_min,
        "tol": args.tol,
        "eff_dt_min": float(eff_dt.min()),
        "eff_dt_median": float(np.median(eff_dt)),
        "eff_dt_max": float(eff_dt.max()),
        "iters_per_boundary_max": float(iters.max()),
        "frac_time_below_threshold_effdt": float((eff_dt < args.threshold).mean()),
        "longest_run_below_threshold_s": longest_run_below(eff_dt, dt_outer, args.threshold),
        "frac_steps_below_threshold_histogram": float(frac_below_hist),
        "floor_pct": 100.0 * stats["floor_fraction"],
        "capped_boundaries": stats["capped_boundaries"],
        "unfinished_worlds": stats["unfinished_worlds"],
        "boundaries": stats["boundaries"],
        "total_samples": stats["total_samples"],
    }

    print(json.dumps(summary, indent=2))
    np.savez(
        f"{args.out}.npz",
        t=t,
        eff_dt=eff_dt,
        ideal_lo=ideal_lo,
        ideal_med=ideal_med,
        ideal_hi=ideal_hi,
        iters=iters,
        counts=counts,
        edges=edges,
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    ax[0].fill_between(t, ideal_lo, ideal_hi, alpha=0.25, label="ideal_dt spread (min-max over worlds)")
    ax[0].plot(t, ideal_med, lw=1.0, label="ideal_dt median")
    ax[0].plot(t, eff_dt, lw=1.4, label="effective dt = dt_outer / iterations")
    ax[0].axhline(args.threshold, ls="--", c="crimson", lw=1.5, label=f"threshold {args.threshold:g} s")
    ax[0].axhline(args.dt_min, ls=":", c="k", lw=1.0, label=f"dt_min floor {args.dt_min:g} s")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("inner timestep [s]")
    ax[0].legend(loc="lower left", fontsize=8, ncol=2)
    ax[0].set_title(
        f"Adaptive dt over time — Allegro hand + cube, {args.worlds} worlds, tol={args.tol:g}\n"
        f"longest contiguous stretch below {args.threshold:g} s: "
        f"{summary['longest_run_below_threshold_s'] * 1e3:.1f} ms "
        f"({100 * summary['frac_time_below_threshold_effdt']:.1f}% of run)"
    )
    ax[0].grid(alpha=0.3, which="both")

    ax[1].plot(t, iters, lw=1.0, c="tab:purple")
    ax[1].axhline(args.max_substeps, ls="--", c="crimson", lw=1.0, label=f"max_substeps {args.max_substeps}")
    ax[1].set_ylabel("iterations\nper boundary")
    ax[1].set_xlabel("simulation time [s]")
    ax[1].legend(loc="upper left", fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=130)
    print(f"wrote {args.out}.png and {args.out}.npz")


if __name__ == "__main__":
    main()
