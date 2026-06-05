# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive ANYmal-walking-through-clutter demo using the CENIC adaptive solver.

Runs the pretrained ANYmal walking policy with a forward command of 1.0 m/s
through a per-world random scatter of light cube primitives.

Usage::

    uv run python -m scripts.demos.anymal_clutter [--num-worlds N] [--headless] [--num-steps N]
"""

import argparse
import sys
import time

import torch  # noqa: TID253 (demo script -- top-level torch needed for @torch.jit.script)
import warp as wp

import newton
import newton.solvers
import newton.utils
from scripts.scenes.anymal_clutter import (
    DT_OUTER,
    LOG_EVERY,
    build_model_randomized,
    make_solver,
)

# ANYmal has 1 floating-base joint (7 coords / 6 dofs) + 12 revolute joints.
# Clutter cubes are appended after the robot in the builder, so the robot
# occupies the first ROBOT_COORDS / ROBOT_DOFS entries of each world's slice.
_ROBOT_JOINTS = 13  # 1 floating base + 12 revolute
_ROBOT_COORDS = 7 + 12  # 19
_ROBOT_DOFS = 6 + 12  # 18

# Joint index remapping between lab convention and MuJoCo convention.
LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]

_grid_lines = 0


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def compute_obs_for_world(
    actions_w,
    state,
    joint_pos_initial,
    torch_device,
    lab_indices,
    gravity_vec,
    command,
    q_offset,
    qd_offset,
):
    """Build the 48-dim policy observation for a single world.

    Only the robot joints (first _ROBOT_COORDS coords / _ROBOT_DOFS dofs) are
    used; clutter body states are not part of the observation.
    """
    q = q_offset
    qd = qd_offset
    root_quat = torch.tensor(state.joint_q[q + 3 : q + 7], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel = torch.tensor(state.joint_qd[qd : qd + 3], device=torch_device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel = torch.tensor(state.joint_qd[qd + 3 : qd + 6], device=torch_device, dtype=torch.float32).unsqueeze(0)
    # Robot revolute joints: q offsets 7..18 (12 joints), qd offsets 6..17.
    joint_pos = torch.tensor(
        state.joint_q[q + 7 : q + _ROBOT_COORDS],
        device=torch_device,
        dtype=torch.float32,
    ).unsqueeze(0)
    joint_vel = torch.tensor(
        state.joint_qd[qd + 6 : qd + _ROBOT_DOFS],
        device=torch_device,
        dtype=torch.float32,
    ).unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat, root_lin_vel)
    ang_vel_b = quat_rotate_inverse(root_quat, root_ang_vel)
    grav = quat_rotate_inverse(root_quat, gravity_vec)
    joint_pos_rel = torch.index_select(joint_pos - joint_pos_initial, 1, lab_indices)
    joint_vel_rel = torch.index_select(joint_vel, 1, lab_indices)
    return torch.cat([vel_b, ang_vel_b, grav, command, joint_pos_rel, joint_vel_rel, actions_w], dim=1)


def _print_status(solver, step):
    global _grid_lines
    sim_times = solver.sim_time.numpy()
    dts = solver.dt.numpy()
    errors = solver.last_error.numpy()
    accepted = solver.accepted.numpy()
    n = len(sim_times)

    col = 16
    bar = "+" + ("-" * col + "+") * 5
    hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'Linf error':>{col}}{'status':>{col}}"
    lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
    for i in range(min(n, 8)):
        lines.append(
            f"{'world ' + str(i):>{col}}{sim_times[i]:>{col}.4f}{dts[i]:>{col}.6f}"
            f"{errors[i]:>{col}.3e}{'ok' if accepted[i] else 'REJECT':>{col}}"
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
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--tol", type=float, default=None,
        help="CENIC adaptive-step tolerance (default 1e-3). Higher = faster, less accurate. Try 1e-2 for ~5-10x speedup.",
    )
    args = parser.parse_args()

    device = wp.get_device()
    torch_device = wp.device_to_torch(device)

    model = build_model_randomized(args.num_worlds)
    coords_per_world = model.joint_coord_count // args.num_worlds
    dofs_per_world = model.joint_dof_count // args.num_worlds

    solver = make_solver(model, tol=args.tol) if args.tol is not None else make_solver(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    policy = torch.jit.load(
        str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt"),
        map_location=torch_device,
    )

    # Initial pose for the 12 robot revolute joints only (not clutter bodies).
    joint_pos_initial = torch.tensor(
        state_0.joint_q[7:_ROBOT_COORDS],
        device=torch_device,
        dtype=torch.float32,
    ).unsqueeze(0)

    actions = torch.zeros(args.num_worlds, 12, device=torch_device, dtype=torch.float32)
    lab_indices = torch.tensor(LAB_TO_MUJOCO, device=torch_device)
    mujoco_indices = torch.tensor(MUJOCO_TO_LAB, device=torch_device)
    gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32)
    command = torch.zeros((1, 3), device=torch_device, dtype=torch.float32)
    command[0, 0] = 1.0

    all_targets = torch.zeros(
        args.num_worlds * dofs_per_world,
        device=torch_device,
        dtype=torch.float32,
    )

    print(
        f"CENIC ANYmal-clutter demo: {args.num_worlds} world(s)  tol={solver._tol:.1e}  "
        f"coords/world={coords_per_world}  dofs/world={dofs_per_world}",
        flush=True,
    )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(2.5, -2.0, 1.4), pitch=-15.0, yaw=120.0)

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        # Policy update on the DT_OUTER boundary (zero-order hold inside solver.step).
        with torch.no_grad():
            for w in range(args.num_worlds):
                q_offset = w * coords_per_world
                qd_offset = w * dofs_per_world
                obs_w = compute_obs_for_world(
                    actions[w : w + 1],
                    state_0,
                    joint_pos_initial,
                    torch_device,
                    lab_indices,
                    gravity_vec,
                    command,
                    q_offset,
                    qd_offset,
                )
                act_w = policy(obs_w)
                actions[w] = act_w[0]
                rearranged = torch.gather(act_w, 1, mujoco_indices.unsqueeze(0))
                targets = joint_pos_initial + 0.5 * rearranged
                # Full target vector: 6 float-base zeros + 12 joint targets +
                # zeros for clutter DOFs (clutter bodies are free, no PD target).
                n_clutter_dofs = dofs_per_world - _ROBOT_DOFS
                all_targets[w * dofs_per_world : (w + 1) * dofs_per_world] = torch.cat(
                    [
                        torch.zeros(6, device=torch_device, dtype=torch.float32),
                        targets.squeeze(0),
                        torch.zeros(n_clutter_dofs, device=torch_device, dtype=torch.float32),
                    ]
                )
            wp.copy(control.joint_target_pos, wp.from_torch(all_targets, dtype=wp.float32))

        # Physics: one DT_OUTER worth of CENIC inner steps.
        if viewer.apply_forces is not None:
            viewer.apply_forces(state_0)
        solver.step(state_0, state_1, control, None, DT_OUTER)
        t += DT_OUTER
        step += 1

        if step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps", flush=True)


if __name__ == "__main__":
    main()
