# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Is the quantile boundary stop sound across scenes, or just tuned to one?

Every number behind ``landed_fraction`` came from a single Allegro in-hand scene. This
runs the same comparison over several genuinely different contact regimes -- quadruped
ground impacts, a free-falling articulated body, an under-actuated control problem, and
in-hand manipulation -- and checks three things per scene:

  correctness  every world lands EXACTLY on its boundary (forced completion never leaves
               a world at the wrong simulation time). This is the property that must hold
               everywhere; a violation is a bug, not a tuning issue.
  contract     the force-completed fraction stays at or under ``1 - landed_fraction``.
  benefit      loop length, straggler waste and wall clock versus ``landed_fraction=1.0``.
               This one is EXPECTED to vary: a scene whose per-world attempt counts are
               tightly clustered has no tail to cut, and the stop should be ~neutral there.

Usage:
    uv run python scripts/rl/adaptive_expts/quantile_soundness.py --worlds 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import JointTargetMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dt_trace_allegro import build_scene as _build_allegro

MJ = {
    "solver": "newton",
    "integrator": "implicitfast",
    "impratio": 20.0,
    "cone": "elliptic",
    "iterations": 100,
    "ls_iterations": 50,
}


def _pd(builder, ke=150.0, kd=5.0, skip_tail=0):
    """Position-drive every actuated dof (``skip_tail`` trailing free-body dofs excluded)."""
    for i in range(builder.joint_dof_count - skip_tail):
        builder.joint_target_ke[i] = ke
        builder.joint_target_kd[i] = kd
        builder.joint_target_mode[i] = int(JointTargetMode.POSITION)


def scene_anymal(worlds):
    """Quadruped dropped onto a ground plane: repeated foot impacts, stiff normal contact."""
    art = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(art)
    art.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    art.default_shape_cfg.ke = 2.0e3
    art.default_shape_cfg.kd = 1.0e2
    art.default_shape_cfg.kf = 1.0e3
    art.default_shape_cfg.mu = 0.75
    art.add_usd(
        str(newton.utils.download_asset("anybotics_anymal_d") / "usd" / "anymal_d.usda"),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        hide_collision_shapes=True,
    )
    art.joint_q[:3] = [0.0, 0.0, 0.68]
    if len(art.joint_q) > 6:
        art.joint_q[3:7] = [0.0, 0.0, 0.0, 1.0]
    _pd(art)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(worlds):
        b.add_world(art)
    b.default_shape_cfg.ke = 1.0e3
    b.default_shape_cfg.kd = 1.0e2
    b.add_ground_plane()
    return b.finalize(), {"nconmax": 45, "njmax": 100}


def _bundled(asset, worlds, height=None, pd=True):
    art = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverMuJoCo.register_custom_attributes(art)
    art.add_usd(newton.examples.get_asset(asset), enable_self_collisions=False, collapse_fixed_joints=True)
    if height is not None and len(art.joint_q) >= 3:
        art.joint_q[2] = height
    if pd:
        _pd(art)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for _ in range(worlds):
        b.add_world(art)
    b.add_ground_plane()
    return b.finalize(), {"nconmax": 100, "njmax": 200}


def scene_ant(worlds):
    """Free-falling articulated body settling on the ground: many simultaneous contacts."""
    return _bundled("ant.usda", worlds, height=0.9)


def scene_humanoid(worlds):
    """High-DOF articulation collapsing onto the ground: the hardest contact set here."""
    return _bundled("humanoid.usda", worlds, height=1.4)


def scene_cartpole(worlds):
    """Control case: essentially contact-free, so per-world attempts should be uniform
    and the quantile stop should have no tail to cut."""
    return _bundled("cartpole.usda", worlds, pd=False)


def scene_allegro(worlds):
    """In-hand manipulation -- the regime every earlier measurement came from."""
    model, _ = _build_allegro(worlds)
    return model, {"nconmax": 300, "njmax": 200}


SCENES = {
    "anymal_d": scene_anymal,
    "ant": scene_ant,
    "humanoid": scene_humanoid,
    "cartpole": scene_cartpole,
    "allegro_hand": scene_allegro,
}


def run(scene_fn, worlds, frac, dt_outer, frames, warmup, tol):
    model, extra = scene_fn(worlds)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        **MJ,
        **extra,
        tol=tol,
        dt_inner_init=1e-2,
        dt_inner_min=1e-6,
        max_substeps=256,
        landed_fraction=frac,
        dt_histogram=True,
    )
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    for _ in range(warmup):
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)
    wp.synchronize_device()
    solver.reset_dt_histogram()
    solver.reset_compute_counter()

    t0 = time.perf_counter()
    for _ in range(frames):
        s0, s1 = solver.step_dt(dt_outer, s0, s1, control)
    wp.synchronize_device()
    wall = time.perf_counter() - t0

    st = solver.dt_histogram_stats()
    behind = float((solver.sim_time.numpy() - solver._next_time.numpy()).min())
    loop = int(solver.cumulative_iterations.numpy()[0]) / frames
    per_world = st["total_samples"] / max(st["boundaries"] * worlds, 1)
    rec = {
        "ms_per_tick": 1e3 * wall / frames,
        "loop": loop,
        "per_world": per_world,
        "waste": loop / max(per_world, 1e-9),
        "forced_pct": 100.0 * st["unfinished_worlds"] / max(st["boundaries"] * worlds, 1),
        "worst_world_behind_s": behind,
        "finite": bool(np.all(np.isfinite(s0.joint_q.numpy()))),
    }
    del solver, model, s0, s1, control
    wp.synchronize_device()
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--tol", type=float, default=1e-3)
    p.add_argument("--dt-outer", type=float, default=1.0 / 120.0)
    p.add_argument("--fraction", type=float, default=0.95)
    p.add_argument("--scenes", type=str, default=",".join(SCENES))
    args = p.parse_args()

    print(f"{args.worlds} worlds, tol={args.tol:g}, dt_outer={args.dt_outer:.4g}, landed_fraction={args.fraction}\n")
    hdr = f"{'scene':<14}{'frac':>6}{'ms/tick':>10}{'loop':>8}{'per-world':>11}{'waste':>8}{'forced%':>9}{'behind[s]':>12}"
    print(hdr)
    print("-" * len(hdr))

    out, failures = {}, []
    for raw in args.scenes.split(","):
        name = raw.strip()
        if name not in SCENES:
            continue
        out[name] = {}
        for frac in (1.0, args.fraction):
            try:
                r = run(SCENES[name], args.worlds, frac, args.dt_outer, args.frames, args.warmup, args.tol)
            except Exception as exc:  # a scene failing to build must not hide the others
                print(f"{name:<14}{frac:>6.2f}  BUILD/RUN FAILED: {type(exc).__name__}: {str(exc)[:60]}")
                failures.append(f"{name}@{frac}: {type(exc).__name__}")
                continue
            out[name][frac] = r
            print(
                f"{name:<14}{frac:>6.2f}{r['ms_per_tick']:>10.2f}{r['loop']:>8.2f}{r['per_world']:>11.2f}"
                f"{r['waste']:>8.2f}{r['forced_pct']:>8.2f}%{r['worst_world_behind_s']:>12.2e}"
            )
            # correctness gates -- these must hold in EVERY scene
            if r["worst_world_behind_s"] < -1e-6:
                failures.append(f"{name}@{frac}: world left {-r['worst_world_behind_s']:.2e}s short of boundary")
            if not r["finite"]:
                failures.append(f"{name}@{frac}: non-finite joint_q")
            if frac < 1.0 and r["forced_pct"] > 100.0 * (1.0 - frac) + 0.5:
                failures.append(f"{name}@{frac}: forced {r['forced_pct']:.2f}% > budget {100 * (1 - frac):.1f}%")

    print()
    for name, byfrac in out.items():
        if 1.0 in byfrac and args.fraction in byfrac:
            a, b = byfrac[1.0], byfrac[args.fraction]
            print(
                f"  {name:<14} speedup {a['ms_per_tick'] / max(b['ms_per_tick'], 1e-9):>5.2f}x   "
                f"loop {a['loop']:>6.2f} -> {b['loop']:<6.2f}  waste {a['waste']:>5.2f} -> {b['waste']:.2f}"
            )

    print()
    if failures:
        print("SOUNDNESS FAILURES:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("SOUND: every world landed on its boundary in every scene, state finite, forced fraction within budget.")
    with open("quantile_soundness.json", "w") as fh:
        json.dump({"args": vars(args), "results": out, "failures": failures}, fh, indent=2)


if __name__ == "__main__":
    main()
