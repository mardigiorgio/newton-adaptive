"""Name the mechanism behind unbounded MuJoCo-adaptive step cost on the Trossen scene.

The wall-clock symptom is that cost per control step grows without bound once
the arm engages the mug. Wall clock cannot distinguish the candidate causes, so
this probe records, per control step, the quantities that separate them:

  substeps           inner opt-steps the adaptive controller spent
  solver_niter       MuJoCo constraint-solver iterations per world. Saturation
                     at the configured ``iterations`` cap means the INNER solve
                     did not converge -- the step-doubling error estimate is
                     then differencing two unconverged states, and the
                     controller subdivides against noise rather than against
                     integration error.
  nefc / ncon        constraint rows and contacts, i.e. how much of the growth
                     is simply more contact to solve
  floor_fraction     share of inner steps clamped at ``dt_inner_min``
  saturation_depth   smallest dt the controller ASKED for while clamped
  unfinished_worlds  world-boundaries that ended short of the control interval
  diverged           per-world divergence latch

Reading: substeps rising while solver_niter sits at the cap is a convergence
failure, not an accuracy demand. Substeps rising with solver_niter well under
the cap and floor_fraction near zero is the controller honestly buying accuracy
against a stiffening contact. Substeps rising with floor_fraction climbing means
the controller has hit the floor and can no longer meet tolerance at all.

This probe measures; it prescribes nothing, and a single run characterizes one
scene, one policy regime (untrained, scripted descent), and one contact config.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv NEWTON_ADAPTIVE_DT_HIST=1 $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/adaptive_cost_forensics.py
"""

from __future__ import annotations

import argparse
import os
import sys

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"
N_ENVS = int(os.environ.get("N_ENVS", "256"))
STEPS = int(os.environ.get("STEPS", "160"))


def main() -> int:
    os.environ.setdefault("NEWTON_ADAPTIVE_DT_HIST", "1")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    torch.manual_seed(7)

    env_cfg = parse_env_cfg(TASK, num_envs=N_ENVS)
    # Force the adaptive arm and turn on the floor histogram regardless of preset.
    env_cfg.sim.physics.solver_cfg.adaptive = True
    env_cfg.sim.physics.solver_cfg.adaptive_dt_histogram = True
    solver_iter_cap = int(env_cfg.sim.physics.solver_cfg.iterations)

    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped
    env.reset()

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager

    solver = NewtonManager._solver
    for hook in ("cumulative_substeps", "dt_histogram_stats", "diverged"):
        if not hasattr(solver, hook):
            print(f"FAIL: solver {type(solver).__name__} has no '{hook}'; wrong solver on this path")
            return 1

    data = solver.mjw_data
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)

    print(f"solver={type(solver).__name__}  envs={N_ENVS}  iterations_cap={solver_iter_cap}")
    print(f"{'step':>5} {'substeps':>9} {'niter_max':>10} {'niter_mean':>11} {'atcap%':>7} {'nefc':>7} {'ncon':>6} {'div':>4}")
    print("-" * 68)

    # The regime under investigation is a LOADED GRASP, which random actions
    # never reach: they leave the arm in free space, where the controller has
    # nothing to subdivide against. Rather than depend on a trained policy, put
    # the scene in that regime directly -- park the mug between the fingers at
    # the arm's home pose and command the gripper closed. Zero action holds the
    # arm at its default joint targets, so the only thing driving the system is
    # the pads squeezing the mug.
    ee = u.scene["ee_frame"]
    obj = u.scene["object"]

    def park_mug() -> None:
        tcp = ee.data.target_pos_w[:, 0, :]
        pose = obj.data.root_state_w[:, :7].clone()
        pose[:, :3] = tcp
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros((u.num_envs, 6), device=u.device))

    # REGIME selects what the probe puts the solver through:
    #   "grasp"  parked mug, fingers squeezing, arm held -- sustained contact,
    #            no impact. Isolates "is loaded contact expensive to solve?".
    #   "rl"     random actions with episode auto-reset, i.e. the regime the
    #            trainer actually runs: impacts, flight, resets, many worlds in
    #            unlike states. Isolates "does the trainer's motion cost?".
    regime = os.environ.get("REGIME", "grasp")
    if regime == "grasp":
        park_mug()

    # A probe that does not reach the regime measures nothing about it, so
    # state the geometry it actually set up before reporting any cost.
    _tcp = ee.data.target_pos_w[:, 0, :]
    _sep = torch.linalg.norm(obj.data.root_pos_w - _tcp, dim=-1)
    print(f"setup: mug-TCP separation min/mean/max = "
          f"{_sep.min():.4f} / {_sep.mean():.4f} / {_sep.max():.4f} m")

    rows = []
    with torch.inference_mode():
        for step in range(STEPS):
            if regime == "grasp":
                act.zero_()
                act[:, -1] = -1.0  # squeeze
            else:
                act.uniform_(-1.0, 1.0)

            solver.reset_compute_counter()
            env.step(act)
            raw_substeps = solver.cumulative_substeps
            substeps = int(raw_substeps() if callable(raw_substeps) else raw_substeps)

            niter = data.solver_niter.numpy().reshape(-1)
            nefc = data.nefc.numpy().reshape(-1)
            # Contacts come from Newton's pipeline on this path
            # (use_mujoco_contacts=False), so MuJoCo's own ncon is not the
            # contact measure -- the injected count is.
            contacts = NewtonManager._contacts
            ncon = -1
            if contacts is not None:
                for attr in ("rigid_contact_count", "contact_count", "count"):
                    buf = getattr(contacts, attr, None)
                    if buf is not None:
                        ncon = int(np.max(np.atleast_1d(buf.numpy() if hasattr(buf, "numpy") else buf)))
                        break
            at_cap = 100.0 * float(np.mean(niter >= solver_iter_cap))
            div = int(np.count_nonzero(solver.diverged.numpy()))
            rows.append((step, substeps, int(niter.max()), float(niter.mean()), at_cap, int(nefc.max()), ncon, div))

            if step % 10 == 0 or step == STEPS - 1:
                s = rows[-1]
                print(f"{s[0]:>5} {s[1]:>9} {s[2]:>10} {s[3]:>11.1f} {s[4]:>7.1f} {s[5]:>7} {s[6]:>6} {s[7]:>4}")

    stats = solver.dt_histogram_stats()
    print("\ndt controller:")
    for k in ("total_samples", "floor_samples", "floor_fraction", "saturation_depth",
              "boundaries", "capped_boundaries", "unfinished_worlds"):
        print(f"  {k:<20} {stats[k]}")

    early = rows[: max(1, len(rows) // 4)]
    late = rows[-max(1, len(rows) // 4):]

    def avg(sub, i):
        return sum(r[i] for r in sub) / len(sub)

    print("\nearly quarter vs late quarter:")
    print(f"  substeps    {avg(early,1):>10.1f} -> {avg(late,1):>10.1f}   ({avg(late,1)/max(avg(early,1),1e-9):.1f}x)")
    print(f"  niter_mean  {avg(early,3):>10.1f} -> {avg(late,3):>10.1f}")
    print(f"  at-cap %    {avg(early,4):>10.1f} -> {avg(late,4):>10.1f}")
    print(f"  nefc_max    {avg(early,5):>10.1f} -> {avg(late,5):>10.1f}")
    print(f"  ncon_max    {avg(early,6):>10.1f} -> {avg(late,6):>10.1f}")

    # The discriminator, stated as a claim about THIS run only.
    cost_growth = avg(late, 1) / max(avg(early, 1), 1e-9)
    contact_growth = avg(late, 5) / max(avg(early, 5), 1e-9)
    print("\nreading:")
    if avg(late, 4) > 5.0:
        print(f"  inner solve saturates the {solver_iter_cap}-iteration cap on {avg(late,4):.1f}% of worlds:")
        print("  the step-doubling estimate is differencing UNCONVERGED states.")
    elif stats["floor_fraction"] > 0.01:
        print(f"  controller is clamped at dt_inner_min for {100*stats['floor_fraction']:.2f}% of inner steps;")
        print(f"  it asked for {stats['saturation_depth']:.3e} s and could not go there.")
    elif cost_growth > 2.0 * contact_growth:
        print(f"  cost grew {cost_growth:.1f}x while contact grew only {contact_growth:.1f}x, with the")
        print("  inner solve converging and the floor untouched: the controller is")
        print("  subdividing for accuracy, not failing.")
    else:
        print(f"  cost grew {cost_growth:.1f}x, tracking contact growth {contact_growth:.1f}x.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
