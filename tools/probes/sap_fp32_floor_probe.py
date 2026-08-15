"""fp32 residual-floor probe: where does the SAP optimality residual stagnate in float32?

Purpose: the adaptive stack's inner convex solve converges to a fixed
``optimality_rel_tol``.  The convergence norms (``||grad||`` vs
``max(||p||, ||jc||)``) are evaluated IN the solve dtype, so cancellation in the
gradient (two O(norm)-sized impulse terms subtracting to ~0 at convergence)
floors the achievable relative residual near the dtype's epsilon times an
accumulation factor.  A target below that floor can never be met: every solve
caps out, contained rejection shrinks dt to the floor, and every world latches
diverged.  The fp32-selected configuration therefore couples its target to
``max(1e-8, K * eps_fp32)``; THIS probe measures the floor that justifies K.

Method (oracle argument -- why this is a measurement, not a tautology):
    The probe drives the real task scene under the PRODUCTION (default, fp64
    contact solve) adaptive solver, so the sampled states/contacts are exactly
    the population production solves.  At sampled boundaries it hands the
    committed state + live contact set + control to two INDEPENDENT fixed-step
    ``SolverSAP`` twins built on the same ``SapModel``:

      * an fp32 twin (all four precision knobs float32, approx32 preset modes),
      * an fp64 control twin (approx32 preset verbatim = the production inner
        solve's precisions),

    both given an UNREACHABLE target (rel_tol=1e-12, abs_tol=0) so they run to
    their iteration cap and expose the stagnated residual, at two caps (30,
    120) to separate stagnation (floor) from slow convergence, and at two dt
    arms (each world's controller ideal_dt, and half of it -- the two solve
    scales of a step-doubling attempt, which set the near-rigid regularization
    through R ~ 1/(dt*k*(dt+tau))).  The fp64 control arm is the probe's own
    vacuity guard: its floor must sit far below 1e-8, proving the method can
    see floors below the production target (a method that reported ~1e-6 for
    fp64 would be measuring its own noise).

    Twins only READ shared objects (SapModel immutable; SapContacts/SapControl
    read-only in the pipeline; state deep-copied per sample), so the production
    march is unperturbed.

Reported: per (dtype, cap, dt-arm) the distribution of
``sqrt(grad2)/max(sqrt(p2), sqrt(jc2))`` over engaged (contact-bearing,
uncapped-unconverged) envs, plus the implied K = floor/eps_fp32.  Verdict JSON
to $CHECK_OUT.

Run (single line, from the IsaacLab root, GPU required):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv NEWTON_SAP=1 NEWTON_SAP_ADAPTIVE=1 NEWTON_SAP_DETERMINISTIC=1 CHECK_OUT=/tmp/fp32_floor.json ./isaaclab.sh -p $HOME/Documents/code/newton-adaptive/tools/probes/sap_fp32_floor_probe.py --headless
"""

import argparse
import json
import os
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"
PHASES = (
    ("rest", 40, lambda t, act: act.zero_()),
    ("press", 60, lambda t, act: (act.zero_(), act[:, 1:3].fill_(4.0), act[:, 6:].fill_(-1.0))),
    ("swing", 60, lambda t, act: (act.zero_(), act[:, 0:3].fill_(6.0 * (1 if (t // 15) % 2 else -1)), act[:, 6:].fill_(-1.0))),
)
SAMPLE_EVERY = 10
CAPS = (30, 120)
# Below both dtypes' possible floors (fp64 eps ~2.2e-16): forces every twin
# solve to cap out AT its stagnation level instead of converging early, so the
# recorded residual is the floor itself for fp64 and fp32 alike.
UNREACHABLE_REL_TOL = 1.0e-16
EPS32 = float(np.finfo(np.float32).eps)

res: dict = {}

try:
    env_cfg = parse_env_cfg(TASK, num_envs=8)
    env_cfg.sim.physics.num_substeps = 1
    env_cfg.sim.physics.solver_cfg.adaptive = True
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager  # noqa: E402
    from newton.solvers import SolverSAPAdaptive  # noqa: E402
    from sim.solver_sap import SolverSAP  # noqa: E402

    solver = NewtonManager._solver
    assert isinstance(solver, SolverSAPAdaptive), type(solver).__name__
    res["production_solver"] = type(solver).__name__
    res["production_solve_precision"] = str(getattr(solver, "_solve_precision", "fp64(pre-option)"))
    res["production_optimality_rel_tol"] = float(solver._optimality_rel_tol)
    res["production_contact_solve_precision"] = str(solver._sap.contact_solve.solve_precision)

    env.reset()
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)

    twins: dict[str, SolverSAP] = {}
    states: dict[str, tuple] = {}

    def build_twins():
        base_kwargs = dict(
            max_rigid_contact=int(solver._sap.max_rigid_contact),
            max_iterations=CAPS[0],
            optimality_abs_tol=0.0,
            optimality_rel_tol=UNREACHABLE_REL_TOL,
            cost_abs_tol=0.0,
            cost_rel_tol=0.0,
            contact_tau_d=0.01,
            contact_preset_variant="approx32",
            line_search_variant=str(solver._sap.line_search_variant),
        )
        twins["fp32"] = SolverSAP(
            solver._sap_model,
            **base_kwargs,
            free_motion_solve_precision="fp32",
            contact_solve_precision="fp32",
            contact_linear_solve_precision="fp32",
            sap_contact_weight_precision="fp32",
        )
        twins["fp64"] = SolverSAP(solver._sap_model, **base_kwargs)
        # Parity guards: the twins must pose the same problem the production
        # inner solve poses (same dissipation, same per-world capacity).
        assert float(twins["fp32"].contact_solve.contact_tau_d) == float(solver._sap.contact_solve.contact_tau_d)
        assert int(twins["fp32"].max_rigid_contact) == int(solver._sap.max_rigid_contact)
        assert str(twins["fp64"].contact_solve.solve_precision) == str(solver._sap.contact_solve.solve_precision)
        for k in twins:
            states[k] = (solver._sap_model.state(), solver._sap_model.state())

    samples: list[dict] = []
    converged_counts: dict[str, int] = {}

    def measure(phase: str, t: int):
        if not twins:
            build_twins()
        ideal = solver._ideal_dt.numpy().astype(np.float64)
        base = np.clip(ideal, 1.0e-5, None)
        contacts = solver._sap_contacts
        control = solver._sap_control
        n = int(solver._world_count)
        for arm, dt_vals in (("full", base), ("half", 0.5 * base)):
            dt_arr = wp.array(dt_vals, dtype=wp.float64, device=solver.model.device)
            for cap in CAPS:
                for name, twin in twins.items():
                    s_in, s_out = states[name]
                    s_in.assign(solver._state_cur)
                    twin.max_iterations = int(cap)
                    twin.reset_runtime_state()
                    twin.step(s_in, s_out, control, contacts, dt_arr)
                    cs = twin.contact_solve
                    conv = cs.converged_env.numpy()
                    g2 = cs.grad_norm2.numpy().astype(np.float64)
                    p2 = cs.p_norm2.numpy().astype(np.float64)
                    jc2 = cs.jc_norm2.numpy().astype(np.float64)
                    iters = cs.newton_iterations_env.numpy()
                    for e in range(n):
                        scale = max(np.sqrt(max(p2[e], 0.0)), np.sqrt(max(jc2[e], 0.0)))
                        engaged = jc2[e] > 0.0
                        if conv[e] == 1:
                            converged_counts[name] = converged_counts.get(name, 0) + 1
                            continue
                        if scale <= 0.0:
                            continue
                        samples.append(
                            {
                                "phase": phase,
                                "t": t,
                                "dtype": name,
                                "cap": cap,
                                "arm": arm,
                                "env": e,
                                "dt": float(dt_vals[e]),
                                "rel_res": float(np.sqrt(max(g2[e], 0.0)) / scale),
                                "engaged": bool(engaged),
                                "iters": int(iters[e]),
                            }
                        )

    for pname, steps, fn in PHASES:
        env.reset()
        for t in range(steps):
            fn(t, act)
            env.step(act)
            if (t + 1) % SAMPLE_EVERY == 0:
                measure(pname, t)

    res["n_samples"] = len(samples)
    res["converged_at_1e-16"] = converged_counts
    summary: dict[str, dict] = {}
    for name in ("fp32", "fp64"):
        for cap in CAPS:
            for arm in ("full", "half"):
                vals = [s["rel_res"] for s in samples if s["dtype"] == name and s["cap"] == cap and s["arm"] == arm and s["engaged"]]
                if not vals:
                    continue
                v = np.array(vals)
                summary[f"{name}_cap{cap}_{arm}"] = {
                    "n": int(v.size),
                    "p50": float(np.percentile(v, 50)),
                    "p90": float(np.percentile(v, 90)),
                    "p99": float(np.percentile(v, 99)),
                    "max": float(v.max()),
                }
    res["floor_summary"] = summary

    fp32_all = [s["rel_res"] for s in samples if s["dtype"] == "fp32" and s["cap"] == max(CAPS) and s["engaged"]]
    fp64_all = [s["rel_res"] for s in samples if s["dtype"] == "fp64" and s["cap"] == max(CAPS) and s["engaged"]]
    if fp32_all and fp64_all:
        f32max = float(np.max(fp32_all))
        f64max = float(np.max(fp64_all))
        res["fp32_floor_max"] = f32max
        res["fp32_floor_max_in_eps32"] = f32max / EPS32
        res["fp64_floor_max"] = f64max
        # Vacuity guard: the method must resolve floors below the production
        # target on the production dtype, else it measures its own noise.
        res["fp64_control_below_1e-8"] = bool(f64max < 1.0e-8)
        # Stagnation guard: cap-120 floor within 2x of cap-30 floor means the
        # residual has stopped descending (a floor, not slow convergence).
        f32_c30 = [s["rel_res"] for s in samples if s["dtype"] == "fp32" and s["cap"] == min(CAPS) and s["engaged"]]
        if f32_c30:
            res["fp32_stagnation_ratio_cap30_over_cap120"] = float(np.max(f32_c30)) / f32max if f32max > 0 else None
        for margin in (4, 8):
            k_raw = margin * f32max / EPS32
            k_pow2 = 1
            while k_pow2 < k_raw:
                k_pow2 *= 2
            res[f"K_margin{margin}_pow2"] = k_pow2
            res[f"target_margin{margin}"] = k_pow2 * EPS32
    res["samples_tail"] = samples[-8:]
    res["ok"] = bool(fp32_all) and bool(fp64_all) and res.get("fp64_control_below_1e-8", False)
    env.close()
except Exception as e:
    res.update({"ok": False, "err": repr(e), "tb": traceback.format_exc()[-1500:]})

with open(os.environ["CHECK_OUT"], "w") as f:
    json.dump(res, f, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "samples_tail"}, indent=1))
os._exit(0 if res.get("ok") else 1)
