# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive falling-cylinder demo using the CENIC adaptive solver.

Drops a tilted cylinder (or N randomized cylinders, one per world) onto
a ground plane. Supports CENIC adaptive stepping and fixed-dt stepping
for comparison.

Usage::

    uv run python -m scripts.demos.falling_cylinder [--num-worlds N] [--headless] [--fixed-dt DT]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.falling_cylinder import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_fixed_solver,
    make_solver,
)

_grid_lines = 0


def _print_status(solver, step):
    global _grid_lines

    n = solver.model.world_count

    if n > 4:
        s = solver.get_status_summary()
        lines = [
            f"  step {step}  tol={solver._tol:.1e}  worlds={n}",
            f"  sim_time  [{s['sim_time_min']:.4f}, {s['sim_time_max']:.4f}] s",
            f"  dt        [{s['dt_min']:.6f}, {s['dt_max']:.6f}] s",
            f"  err_max   {s['error_max']:.3e}",
            f"  accepted  {s['accept_count']}/{n}",
        ]
    else:
        sim_times = solver.sim_time.numpy()
        dts = solver.dt.numpy()
        errors = solver.last_error.numpy()
        accepted = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'inf-norm err':>{col}}{'status':>{col}}"
        lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
        for i in range(len(sim_times)):
            lines.append(
                f"{'world ' + str(i):>{col}}"
                f"{sim_times[i]:>{col}.4f}"
                f"{dts[i]:>{col}.6f}"
                f"{errors[i]:>{col}.3e}"
                f"{'ok' if accepted[i] else 'REJECT':>{col}}"
            )
        lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=0, help="0 = run until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-dt",
        type=float,
        default=None,
        help="Use fixed-step SolverMuJoCo with this dt instead of CENIC",
    )
    parser.add_argument(
        "--slow-mo", type=float, default=1.0,
        help="Wall-clock slow-motion factor. 1.0 = realtime. 10.0 = each outer step takes "
        "10x DT_OUTER seconds of wall time.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model = build_model_randomized(args.num_worlds, seed=args.seed)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    use_fixed = args.fixed_dt is not None

    if use_fixed:
        solver = make_fixed_solver(model)
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step demo: {args.num_worlds} world(s)  solver=SolverMuJoCo  "
            f"dt={args.fixed_dt:.4e}  substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model)
        print(
            f"CENIC cylinder demo: {args.num_worlds} world(s)  solver=SolverMuJoCoAdaptive  "
            f"tol={solver._tol:.1e}  dt_init={solver._dt.numpy()[0]:.4f}  "
            f"dt_max={solver._dt_max:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(1.5, -1.5, 0.8), pitch=-20.0, yaw=135.0)

    # Allocate our own Newton-format contacts buffer; for mjwarp-pipeline solvers
    # (use_mujoco_contacts=True), solver.contacts is None and update_contacts pulls
    # contacts out of mjw_data on demand.
    contacts = newton.Contacts(
        rigid_contact_max=solver.mjw_data.naconmax, soft_contact_max=0,
        requested_attributes={"force"},
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        step_start = time.perf_counter()

        if not viewer.is_paused():
            if use_fixed:
                for _ in range(n_inner):
                    state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                    state_0, state_1 = state_1, state_0
            else:
                if viewer.apply_forces is not None:
                    viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, None, DT_OUTER)
            t += DT_OUTER
            step += 1

            if not use_fixed and step % LOG_EVERY == 0:
                _print_status(solver, step)

            if args.num_steps > 0 and step >= args.num_steps:
                break

        if viewer.show_contacts:
            solver.update_contacts(contacts, state_0)
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

        if args.slow_mo > 1.0 and not viewer.is_paused():
            target = DT_OUTER * args.slow_mo
            elapsed = time.perf_counter() - step_start
            if elapsed < target:
                time.sleep(target - elapsed)

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
