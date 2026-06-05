# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive contact-objects demo using the CENIC adaptive solver.

Runs a scene of 9 spheres and 9 tilted boxes falling onto a ground plane,
contained by invisible walls. Supports both CENIC adaptive stepping and
fixed-dt stepping for comparison.

Usage::

    uv run python -m scripts.demos.contact_objects [--num-worlds N] [--headless] [--fixed-dt DT]
"""

import argparse
import math
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.contact_objects import (
    DT_INNER_MIN,
    DT_OUTER,
    LOG_EVERY,
    TOL,
    build_model,
    build_model_randomized,
    make_solver,
)

_grid_lines = 0

# HUD state shared between the main loop (updates) and the imgui callback (reads).
# Updated every HUD_EVERY outer steps so we get one extra GPU sync per ~150 ms
# instead of every frame — invisible alongside the viewer's existing per-frame sync.
HUD_EVERY = 15
_hud = {
    "sim_time": 0.0,
    "mean_dt_ms": 0.0,
    "dt_min_ms": 0.0,
    "dt_max_ms": 0.0,
    "error_max": 0.0,
    "iters_last": 0,
    "step": 0,
}


def _update_hud(solver, step: int) -> None:
    """Read aggregated CENIC stats off the GPU. One reduction + a few small copies."""
    s = solver.get_status_summary()
    mean_dt = float(solver.dt.numpy().mean())  # N float32s — negligible
    k = int(solver.iteration_count.numpy()[0])  # 1 int32 — negligible
    _hud["sim_time"] = s["sim_time_max"]
    _hud["mean_dt_ms"] = mean_dt * 1e3
    _hud["dt_min_ms"] = s["dt_min"] * 1e3
    _hud["dt_max_ms"] = s["dt_max"] * 1e3
    _hud["error_max"] = s["error_max"]
    _hud["iters_last"] = k
    _hud["step"] = step


def _hud_callback(imgui) -> None:
    imgui.separator()
    imgui.text("Solver live stats")
    imgui.text(f"Sim time:  {_hud['sim_time']:7.3f} s")
    imgui.text(f"Mean dt:   {_hud['mean_dt_ms']:7.3f} ms")
    imgui.text(f"dt range:  [{_hud['dt_min_ms']:.3f}, {_hud['dt_max_ms']:.3f}] ms")
    imgui.text(f"Max error: {_hud['error_max']:.2e}")
    imgui.text(f"Substeps/outer: {_hud['iters_last']}")
    imgui.text(f"Step:      {_hud['step']}")


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
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'L2 error':>{col}}{'status':>{col}}"
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
    parser.add_argument("--num-worlds", type=int, default=1, help="parallel worlds")
    parser.add_argument("--num-steps", type=int, default=0, help="0 = run until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-dt", type=float, default=None, help="Use fixed-step SolverMuJoCo with this dt instead of CENIC"
    )
    parser.add_argument(
        "--xpbd-dt", type=float, default=None,
        help="Use fixed-step SolverXPBD with this dt (mutually exclusive with --fixed-dt and adaptive)",
    )
    parser.add_argument(
        "--tol", type=float, default=TOL,
        help=f"CENIC error tolerance (default {TOL:.0e}). Larger = looser = faster (fewer substeps).",
    )
    parser.add_argument(
        "--randomized", action="store_true",
        help="Per-world randomized initial poses (recommended for many-world demos).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for --randomized initial conditions.",
    )
    parser.add_argument(
        "--grid-camera", action="store_true",
        help="Pull camera back proportional to N so the whole world grid fits in frame.",
    )
    parser.add_argument(
        "--camera-scale", type=float, default=None,
        help="Override the grid-camera scale factor (higher = farther back). Default auto-scales with N.",
    )
    parser.add_argument(
        "--render-worlds", type=int, default=None,
        help="Cap how many worlds the viewer renders (physics still steps all --num-worlds). "
        "Drop this to recover FPS at high N — the bottleneck is the viewer, not the solver.",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Viewer window width (smaller = faster rasterization).",
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Viewer window height.",
    )
    parser.add_argument(
        "--start-delay", type=float, default=0.0,
        help="Seconds to render the initial state before starting the simulation (gives time to start recording).",
    )
    parser.add_argument(
        "--nconmax-per-world", type=int, default=120,
        help="Max contact slots per world (mjwarp multiplies by nworld internally). "
        "Random ICs at N=1 with the Newton SAP pipeline peak around 70-80 candidates; 120 gives margin.",
    )
    parser.add_argument(
        "--njmax-per-world", type=int, default=480,
        help="Max constraint slots per world. ~4x nconmax fits pyramidal friction expansion.",
    )
    parser.add_argument(
        "--slow-mo", type=float, default=1.0,
        help="Wall-clock slow-motion factor. 1.0 = realtime. 10.0 = each outer step takes "
        "10x DT_OUTER seconds of wall time (sim physics unchanged). Useful for capturing artifacts.",
    )
    args = parser.parse_args()

    if args.randomized:
        model = build_model_randomized(args.num_worlds, seed=args.seed)
    else:
        model = build_model(args.num_worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    if args.fixed_dt is not None and args.xpbd_dt is not None:
        parser.error("--fixed-dt and --xpbd-dt are mutually exclusive")
    use_fixed = args.fixed_dt is not None
    use_xpbd = args.xpbd_dt is not None
    is_adaptive = not (use_fixed or use_xpbd)

    # mjwarp's nconmax/njmax are PER WORLD; it multiplies by nworld internally.
    nconmax = args.nconmax_per_world
    njmax = args.njmax_per_world

    if use_xpbd:
        solver = newton.solvers.SolverXPBD(model)
        n_inner = round(DT_OUTER / args.xpbd_dt)
        print(
            f"XPBD demo: {args.num_worlds} world(s)  solver=SolverXPBD  "
            f"dt={args.xpbd_dt:.4e}  substeps/outer={n_inner}",
            flush=True,
        )
    elif use_fixed:
        solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True, nconmax=nconmax, njmax=njmax)
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step demo: {args.num_worlds} world(s)  solver=SolverMuJoCo  "
            f"dt={args.fixed_dt:.4e}  substeps/outer={n_inner}  "
            f"nconmax={nconmax}  njmax={njmax}",
            flush=True,
        )
    else:
        solver = newton.solvers.SolverMuJoCoAdaptive(
            model,
            tol=args.tol,
            dt_init=DT_OUTER,
            dt_min=DT_INNER_MIN,
            dt_max=DT_OUTER,
            nconmax=nconmax,
            njmax=njmax,
            use_mujoco_contacts=True,
        )
        print(
            f"CENIC contact demo: {args.num_worlds} world(s)  solver=SolverMuJoCoAdaptive  "
            f"tol={solver._tol:.1e}  dt_init={solver._dt.numpy()[0]:.4f}  "
            f"dt_max={solver._dt_max:.4f}  nconmax={nconmax}  njmax={njmax}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless, width=args.width, height=args.height)
    viewer.set_model(model, max_worlds=args.render_worlds)
    viewer.set_world_offsets((0.9, 0.9, 0.0))

    if is_adaptive:
        viewer.register_ui_callback(_hud_callback, position="stats")

    if args.grid_camera:
        # Scale the default framing by the grid radius so all worlds stay in view.
        side = math.ceil(math.sqrt(max(1, args.num_worlds)))
        scale = args.camera_scale if args.camera_scale is not None else max(1.0, side * 0.45)
        viewer.set_camera(
            pos=wp.vec3(1.97 * scale, -2.07 * scale, 1.07 * scale),
            pitch=-22.5,
            yaw=136.3,
        )
    else:
        viewer.set_camera(
            pos=wp.vec3(1.97, -2.07, 1.07),
            pitch=-22.5,
            yaw=136.3,
        )

    if use_xpbd:
        # XPBD uses Newton's own contact pipeline (not mjwarp).
        contacts = model.contacts()
    else:
        contacts = newton.Contacts(
            rigid_contact_max=solver.mjw_data.naconmax,
            soft_contact_max=0,
            requested_attributes={"force"},
        )

    step = 0
    t = 0.0

    if args.start_delay > 0.0:
        print(f"Start delay: rendering initial state for {args.start_delay:.1f} s...", flush=True)
        delay_start = time.perf_counter()
        while viewer.is_running() and (time.perf_counter() - delay_start) < args.start_delay:
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()

    t_start = time.perf_counter()

    while viewer.is_running():
        step_start = time.perf_counter()

        if not viewer.is_paused():
            if use_xpbd:
                for _ in range(n_inner):
                    model.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, args.xpbd_dt)
                    state_0, state_1 = state_1, state_0
            elif use_fixed:
                for _ in range(n_inner):
                    state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                    state_0, state_1 = state_1, state_0
            else:
                if viewer.apply_forces is not None:
                    viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, None, DT_OUTER)
            t += DT_OUTER
            step += 1

            if is_adaptive and step % HUD_EVERY == 0:
                _update_hud(solver, step)

            if is_adaptive and step % LOG_EVERY == 0:
                _print_status(solver, step)

            if args.num_steps > 0 and step >= args.num_steps:
                break

        if viewer.show_contacts and not use_xpbd:
            # update_contacts pulls mjw_data contacts into the Newton buffer.
            # XPBD already populated contacts via model.collide above.
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
    print(
        f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps",
        flush=True,
    )


if __name__ == "__main__":
    main()
