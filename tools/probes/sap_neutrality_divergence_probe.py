"""Pass-37: trajectory-divergence probe for the SAP-adaptive optimization stack.

WHAT IS UNDER TEST
    The SAP-adaptive optimization stack splits into two classes that must not
    be conflated:

      * EIGHT SCHEDULING optimizations (env-list compaction, blocked-Cholesky
        narrowing, live-k GEMM truncation, per-contact packing, shared
        assembly, narrow-v3, tail and march compaction) which should be
        bitwise-neutral; and
      * THREE KERNEL FUSIONS (the fused update evaluation, the fused armijo
        ladder and the folded alpha-max rung) which change floating-point
        REDUCTION ORDER.  These are algebraically exact but cannot be proved
        bitwise, and this probe does not claim they are: it BOUNDS them.

    The bound is only meaningful against scales measured on the same rig in
    the same session, so the probe measures three:

      (a) TOL          -- the method's accepted local-error budget
                          (adaptive_tol = 1e-3).  NOTE the category
                          difference, which the report states rather than
                          hides: tol bounds the LOCAL error of one accepted
                          step, whereas the divergence measured here is a
                          trajectory difference ACCUMULATED over the horizon.
                          The commensurable per-step comparison is the
                          ENSEMBLE of accepted errors, reported separately.
      (b) NONDET       -- the run-to-run spread of the SHIPPED configuration
                          (NEWTON_SAP_DETERMINISTIC at its "0" default), i.e.
                          the irreproducibility already present in every
                          reported run.  This is the scale that matters.
      (c) SEED         -- the seed-to-seed spread of the action sequence.

    And it runs the AGGREGATE case: every optimization flag forced to its
    legacy/OFF state versus the shipped defaults, same seed, det=1 -- plus
    the fusions-only arm, whose difference from the aggregate arm isolates
    the eight scheduling flags exactly.

ANTI-VACUITY: ENGAGEMENT COUNTERS
    "ON and OFF are bitwise identical" is worthless if the OFF arm silently
    ran the same code.  Every arm therefore reports the solver's own device
    engagement counters (fused-ladder envs, alpha-max envs, fused-update
    envs, per-contact pack execs, GEMM-truncation skips, narrowed launch
    sites, shared-assembly execs).  A neutrality claim from this probe is
    only admissible when the ON arm's counters are large and the OFF arm's
    are exactly zero.

ORACLE ARGUMENT (why this is not a tautology or a snapshot)
    Nothing here asserts a computed physical value.  Every arm is compared
    against another arm of the SAME scenario -- same task, same seed, same
    pre-generated action sequence, same build, same device, same process
    recipe -- so the reference is recomputed on every invocation and there
    are no golden files.  The quantities reported are DIFFERENCES between
    independently-produced trajectories, and the thresholds they are judged
    against are measured in the same invocation (b, c) or read from the
    solver's own configured tolerance (a).  A claim of the form "the fusions
    perturb the trajectory by less than X" is therefore falsifiable by this
    probe and is not a restatement of the implementation.

    The det=1 REPEAT arm is the oracle guard: if two identical det=1 runs are
    not bitwise identical, the platform is not reproducible and no bitwise
    judgement in this probe is valid.  The probe says so instead of failing
    spuriously.

METRIC
    Per control step, over the committed generalized coordinates joint_q:
        Linf   = max over all (env, coord) of |q_arm - q_ref|
        RMS    = sqrt(mean over all (env, coord) of (q_arm - q_ref)^2)
    joint_q is the same coordinate vector the solver's accuracy metric uses
    (position-only, unit-scaled, S = I).  That makes Linf commensurable with
    tol in UNITS but not in KIND -- tol bounds one step's local error, Linf
    here is an accumulated trajectory difference.  Read the Linf/tol column as
    a scale marker, never as a tolerance violation.  body_q translation (the
    world-frame body positions, metres) is reported alongside as a
    physical-units cross-check.

WHAT A PASS DOES NOT ESTABLISH (residual risk, stated up front)
    * One task, one scene, one action distribution, one env count, one
      device.  Divergence growth is regime-dependent; a different contact
      regime can grow faster.
    * The comparison is only valid while both arms have the same reset
      history.  The probe records per-step reset masks and reports the first
      step at which they differ; numbers past that point mix trajectory
      divergence with re-randomized episodes and are flagged, not hidden.
    * Bounding divergence bounds nothing about which arm is MORE ACCURATE.
      Neither arm is a reference solution.  The claim available is
      "indistinguishable at the scale of the irreproducibility the shipped
      configuration already has", never "equally correct".
    * Bitwise equality on one horizon does not generalize to another.  At 256
      worlds x 60 steps every arm here was bitwise identical; at 512 worlds x
      150 steps the fusion arms separate at step 13.  Any bitwise claim must
      carry the horizon it was measured on.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_neutrality_divergence_probe.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

N_ENVS = int(os.environ.get("P37_N_ENVS", "256"))
STEPS = int(os.environ.get("P37_STEPS", "60"))
ACT_SEED_A = int(os.environ.get("P37_ACT_SEED_A", "1337"))
ACT_SEED_B = int(os.environ.get("P37_ACT_SEED_B", "2024"))
TORCH_SEED = 7
TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"

# The solver's configured accuracy budget in the reportable run
# (adaptive_tol; audit Sec. 2.4).  Read here as a constant so the report
# names the number it compares against; the worker asserts the live solver
# actually carries it, so a config drift fails the probe rather than
# silently invalidating the comparison.
TOL = 1e-3

_BASE = {
    "NEWTON_SAP": "1",
    "NEWTON_SAP_ADAPTIVE": "1",
}

# Every optimization flag at its LEGACY / OFF state.  Graph capture and the
# whole-march conditional tier are deliberately LEFT AT THEIR DEFAULTS: they
# are pure launch-mechanism switches whose bitwise invariance is certified
# in the same session by tools/probes/sap_flag_equivalence_probe.py, and
# disabling them multiplies wall time without changing arithmetic.
_LEGACY = {
    # fp-reduction-order fusions (audit D1, D2, D3)
    "NEWTON_SAP_FUSED_UPDATE": "0",
    "NEWTON_SAP_FUSED_LS": "0",
    "NEWTON_SAP_FUSED_ALPHAMAX": "0",
    # scheduling / narrowing (audit C2, C3, C4, C5, C6, C7)
    "NEWTON_SAP_SOLVE_COMPACT": "0",
    "NEWTON_SAP_LS_COMPACT": "0",
    "NEWTON_SAP_GEMM_RESHAPE": "0",
    "NEWTON_SAP_PACK_PERCONTACT": "0",
    "NEWTON_SAP_NARROW_V3": "0",
    "NEWTON_SAP_SHARED_ASSEMBLY": "0",
    "NEWTON_SAP_MARCH_COMPACT": "0",
    "NEWTON_ADAPTIVE_TAIL_COMPACT": "0",
}

_FUSIONS_OFF = {k: v for k, v in _LEGACY.items() if k.startswith("NEWTON_SAP_FUSED")}


def _arm(det: str, extra: dict[str, str], act_seed: int) -> dict:
    env = dict(_BASE)
    env["NEWTON_SAP_DETERMINISTIC"] = det
    env.update(extra)
    env["P37_ACT_SEED"] = str(act_seed)
    return env


ARMS = {
    # --- det=1 family: bitwise judgements are legal inside it -------------
    "ref": _arm("1", {}, ACT_SEED_A),
    "ref-repeat": _arm("1", {}, ACT_SEED_A),
    "nofuse": _arm("1", _FUSIONS_OFF, ACT_SEED_A),
    "refmode": _arm("1", _LEGACY, ACT_SEED_A),
    "seedB": _arm("1", {}, ACT_SEED_B),
    # --- shipped production configuration (det at its "0" source default) -
    "prod-a": _arm("0", {}, ACT_SEED_A),
    "prod-b": _arm("0", {}, ACT_SEED_A),
}

# (candidate, reference, what the pair measures)
PAIRS = [
    ("ref-repeat", "ref", "ORACLE: det=1 reproducibility (must be bitwise 0)"),
    ("nofuse", "ref", "D1+D2+D3 fp-reduction-order fusions ON vs OFF"),
    ("refmode", "ref", "AGGREGATE: every optimization legacy/OFF vs shipped"),
    ("prod-b", "prod-a", "SCALE (b): run-to-run spread of the SHIPPED config"),
    ("seedB", "ref", "SCALE (c): seed-to-seed spread"),
]


# ---------------------------------------------------------------- worker
def worker(out_path: str) -> None:
    import argparse  # noqa: PLC0415

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym  # noqa: PLC0415
    import isaaclab_tasks  # noqa: F401, PLC0415
    import numpy as np  # noqa: PLC0415
    import torch
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

    torch.manual_seed(TORCH_SEED)

    env_cfg = parse_env_cfg(TASK, num_envs=N_ENVS)
    # PROBE-LOCAL termination override (the cfg OBJECT in this process; no
    # task or scene file is touched).  The task's early terminations are RL
    # devices, not physics: under the random-action regime this probe needs
    # they fire within a few control steps and restart episodes with fresh
    # randomization, which would confound a trajectory-divergence measurement
    # with re-randomized initial conditions.  Only `time_out` is kept (it
    # cannot fire inside the horizon run here: episode_length_s = 5.0 at a
    # 1/30 s control step = 150 steps).  Divergence-induced world failures
    # are still OBSERVED -- the solver's own per-world `diverged` latch is
    # recorded every step -- they simply do not restart the episode.
    for _term in ("object_dropping", "object_off_table", "object_speeding", "robot_abnormal", "physics_diverged"):
        if hasattr(env_cfg.terminations, _term):
            setattr(env_cfg.terminations, _term, None)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager  # noqa: PLC0415

    from newton.solvers import SolverSAPAdaptive  # noqa: PLC0415

    solver = NewtonManager._solver
    assert isinstance(solver, SolverSAPAdaptive), type(solver).__name__

    # Configuration tripwires: the numbers this probe compares against are
    # only meaningful if the live solver actually carries them.
    live_tol = float(solver._tol)
    assert abs(live_tol - TOL) < 1e-15, f"live adaptive tol {live_tol} != probe reference {TOL}"
    flag_state = {
        "fused_update": bool(solver._sap.contact_solve._fused_update),
        "fused_ls": bool(solver._sap.contact_solve._fused_ls),
        "fused_alphamax": bool(solver._sap.contact_solve._fused_alphamax),
        "gemm_reshape": bool(solver._sap.contact_solve._gemm_reshape),
        "pack_percontact": bool(solver._sap.contact_solve._pack_percontact),
        "solve_compact": bool(solver._sap.contact_solve._solve_compact),
        "ls_compact": bool(solver._sap.contact_solve._ls_compact),
        "narrow_v3": bool(solver._sap.contact_solve._narrow_v3),
        "deterministic": bool(solver._sap.contact_solve._deterministic),
        "shared_assembly": bool(solver._shared_assembly),
        "tail_compact": bool(solver._tail_compact),
        "march_compact": bool(solver._march_compact),
        "attempt_consistent_r": bool(solver._attempt_consistent_r),
        "containment": bool(solver.containment),
    }
    # Each flag must have RESOLVED to what this arm's environment asked for.
    for name, envvar in (
        ("fused_update", "NEWTON_SAP_FUSED_UPDATE"),
        ("fused_ls", "NEWTON_SAP_FUSED_LS"),
        ("fused_alphamax", "NEWTON_SAP_FUSED_ALPHAMAX"),
        ("gemm_reshape", "NEWTON_SAP_GEMM_RESHAPE"),
        ("solve_compact", "NEWTON_SAP_SOLVE_COMPACT"),
        ("ls_compact", "NEWTON_SAP_LS_COMPACT"),
        ("narrow_v3", "NEWTON_SAP_NARROW_V3"),
        ("shared_assembly", "NEWTON_SAP_SHARED_ASSEMBLY"),
        ("tail_compact", "NEWTON_ADAPTIVE_TAIL_COMPACT"),
    ):
        if os.environ.get(envvar) == "0":
            assert not flag_state[name], f"{envvar}=0 did not reach construction ({name} is ON)"
    assert flag_state["deterministic"] == (os.environ.get("NEWTON_SAP_DETERMINISTIC") == "1"), (
        "determinism switch did not resolve as requested"
    )

    env.reset()
    dim = u.action_manager.total_action_dim
    rng = np.random.default_rng(int(os.environ["P37_ACT_SEED"]))
    act_seq = rng.uniform(-2.0, 2.0, size=(STEPS, u.num_envs, dim)).astype(np.float32)

    state = NewtonManager._state_0
    frames: dict[str, list] = {
        "joint_q": [],
        "joint_qd": [],
        "body_q": [],
        "resets": [],
        "cum": [],
        "acc_err": [],
        "dt": [],
        "diverged": [],
    }
    fail_step = -1
    with torch.inference_mode():
        for step in range(STEPS):
            act = torch.as_tensor(act_seq[step], device=u.device)
            try:
                _, _, terminated, truncated, _ = env.step(act)
            except RuntimeError:
                fail_step = step
                break
            frames["joint_q"].append(state.joint_q.numpy().copy())
            frames["joint_qd"].append(state.joint_qd.numpy().copy())
            frames["body_q"].append(state.body_q.numpy().copy())
            frames["resets"].append((terminated | truncated).detach().to("cpu").numpy().astype(np.uint8).copy())
            frames["cum"].append(np.int64(solver.cumulative_substeps()))
            # The controller's own accepted local error and accepted step for
            # this boundary: the scale the method already tolerates every step.
            frames["acc_err"].append(solver._accepted_error.numpy().copy())
            # Per-world accepted substeps this boundary (the demand proxy).
            # NOTE: solver._dt is the CURRENT ATTEMPT's step and is zero for a
            # landed world, so it is useless as a post-step statistic.
            frames["dt"].append(solver.substeps.numpy().astype(np.int64).copy())
            frames["diverged"].append(solver.diverged.numpy().astype(np.uint8).copy())

    # ENGAGEMENT COUNTERS -- the anti-vacuity evidence for the whole run.
    # "ON and OFF are bitwise identical" means nothing unless the OFF arm
    # genuinely executed a different path. These device counters are advanced
    # by the optimized kernels themselves, so a nonzero count in the ON arm
    # and a zero count in the OFF arm is execution proof, not a restatement of
    # the flag. Host reads are post-run only.
    cs = solver._sap.contact_solve
    engagement = {
        "fused_ls_ladder_envs": int(cs.fused_ls_ladder_envs()),
        "fused_alphamax_envs": int(cs.fused_alphamax_envs()),
        "fused_update_envs": int(cs.fused_update_envs()),
        "pack_percontact_execs": int(cs.pack_percontact_execs()),
        "gemm_reshape_skips": int(cs.gemm_reshape_skips()),
        "narrow_sites_emitted": len(cs.narrow_sites_emitted),
        "shared_assembly_execs": int(solver._sa_execs.numpy()[0]),
    }
    print(f"ENGAGEMENT {json.dumps(engagement)}")

    np.savez(
        out_path,
        fail_step=np.array([fail_step], dtype=np.int64),
        flag_state=np.array([json.dumps(flag_state)]),
        engagement=np.array([json.dumps(engagement)]),
        **{k: (np.stack(v) if v else np.zeros((0,), dtype=np.float32)) for k, v in frames.items()},
    )
    env.close()
    os._exit(0)


# ---------------------------------------------------------------- driver
def _run_arm(name: str, arm_env: dict, scratch: str) -> str | None:
    out = os.path.join(scratch, f"p37_{name}.npz")
    if os.path.exists(out) and os.environ.get("P37_REUSE") == "1":
        print(f"arm {name}: reusing {out}")
        return out
    env = dict(os.environ)
    env.update(arm_env)
    env["P37_WORKER"] = "1"
    env["P37_OUT"] = out
    log = os.path.join(scratch, f"p37_{name}.log")
    census = os.path.join(scratch, "p37_gpu_census.txt")
    with open(census, "a") as cf:
        cf.write(f"--- before arm {name} ---\n")
        cf.write(
            subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        )
    with open(log, "w") as lf:
        r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, stdout=lf, stderr=lf, check=False)
    with open(census, "a") as cf:
        cf.write(f"--- after arm {name} (exit {r.returncode}) ---\n")
        cf.write(
            subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout
        )
    if r.returncode != 0 or not os.path.exists(out):
        print(f"FAIL: arm {name} exited {r.returncode}; log tail:")
        print(open(log).read()[-3000:])
        return None
    print(f"arm {name}: ok")
    return out


def main() -> int:
    if os.environ.get("P37_WORKER"):
        worker(os.environ["P37_OUT"])
        return 0

    import numpy as np  # noqa: PLC0415

    scratch = os.environ.get(
        "P37_SCRATCH",
        "/tmp/claude-1002/-home-mdigiorgio-Documents-code/fe8a844e-d1b0-4d64-833c-48934ee6d700/scratchpad",
    )
    os.makedirs(scratch, exist_ok=True)

    only = os.environ.get("P37_ONLY")
    arms = {k: v for k, v in ARMS.items() if (only is None or k in only.split(","))}

    paths = {}
    failed_arms = []
    for name, arm_env in arms.items():
        p = _run_arm(name, arm_env, scratch)
        if p is None:
            # An arm that will not run is itself a finding (a legacy path that
            # no longer executes at this HEAD is not a neutral optimization,
            # it is a deleted alternative). Record it and keep going so the
            # remaining comparisons still land.
            failed_arms.append(name)
            continue
        paths[name] = p
    if "ref" not in paths:
        print("FAIL: the reference arm did not run; no comparison is possible.")
        return 2
    if failed_arms:
        print(f"\nWARNING: arms that failed to run: {', '.join(failed_arms)}")

    data = {k: np.load(v, allow_pickle=False) for k, v in paths.items()}
    for k, d in data.items():
        fs = int(d["fail_step"][0])
        flags = json.loads(str(d["flag_state"][0]))
        on = sorted(n for n, s in flags.items() if s)
        print(f"\narm {k}: steps={d['joint_q'].shape[0]} fail_step={fs}")
        print(f"  flags ON: {', '.join(on)}")

    # The method's OWN accepted local error, measured on this rig -- the
    # sharpest form of reference scale (a): the controller accepts this much
    # position error on every accepted step, by design.
    if "ref" in data and data["ref"]["acc_err"].size:
        ae = data["ref"]["acc_err"].astype(np.float64).ravel()
        ae = ae[np.isfinite(ae)]
        dts = data["ref"]["dt"].astype(np.float64).ravel()
        print("\naccepted local error on the reference arm (per world, per boundary):")
        print(f"  median {np.median(ae):.3e}  p90 {np.quantile(ae, 0.9):.3e}  max {ae.max():.3e}  (tol = {TOL:g})")
        print(f"  accepted dt: median {np.median(dts):.3e} s, min {dts.min():.3e} s")

    results = {}
    print("\n" + "=" * 78)
    print("DIVERGENCE (joint_q Linf, metres/radians; the coordinate space of the")
    print(f"solver's own accuracy metric, S = I) -- reference budget tol = {TOL:g}")
    print("=" * 78)
    for cand, ref, what in PAIRS:
        if cand not in data or ref not in data:
            continue
        a, b = data[ref], data[cand]
        n = min(a["joint_q"].shape[0], b["joint_q"].shape[0])
        if n == 0:
            print(f"\n{cand} vs {ref}: NO STEPS RECORDED")
            continue
        # Reset-history validity: an env whose episode restarted is no longer
        # running the same scenario in the two arms, so it is dropped from the
        # comparison from that step on.  (Early terminations are disabled in
        # the worker, so this is a safety net, not the normal path.)
        ra, rb = a["resets"][:n].astype(bool), b["resets"][:n].astype(bool)
        nenv = ra.shape[1]
        alive = np.cumsum(ra | rb, axis=0) == 0
        diff_reset = np.flatnonzero((ra != rb).any(axis=1))
        first_reset_div = int(diff_reset[0]) if diff_reset.size else -1
        any_reset = int(ra.sum()), int(rb.sum())

        def _per_env(x, _n=n, _nenv=nenv):
            return x[:_n].astype(np.float64).reshape(_n, _nenv, -1)

        dq = _per_env(a["joint_q"]) - _per_env(b["joint_q"])
        bq_all = _per_env(a["body_q"]) - _per_env(b["body_q"])
        # body_q rows are 7-wide transforms per body; keep the translation.
        nb = bq_all.shape[2] // 7
        bq = bq_all.reshape(n, nenv, nb, 7)[:, :, :, :3].reshape(n, nenv, -1)

        m = alive[:, :, None]
        linf = np.where(m, np.abs(dq), 0.0).reshape(n, -1).max(axis=1)
        cnt = np.maximum(alive.sum(axis=1), 1) * dq.shape[2]
        rms = np.sqrt(np.where(m, dq, 0.0).reshape(n, -1).__pow__(2).sum(axis=1) / cnt)
        bq_lin = np.where(alive[:, :, None], np.abs(bq), 0.0).reshape(n, -1).max(axis=1)
        alive_n = alive.sum(axis=1)
        cum_a, cum_b = a["cum"][:n], b["cum"][:n]
        div_a = int(a["diverged"][:n].max(axis=0).sum()) if a["diverged"].size else 0
        div_b = int(b["diverged"][:n].max(axis=0).sum()) if b["diverged"].size else 0

        first_nonzero = np.flatnonzero(linf > 0)
        fz = int(first_nonzero[0]) if first_nonzero.size else -1
        results[cand] = {
            "vs": ref,
            "what": what,
            "steps": int(n),
            "bitwise_identical": bool(linf.max() == 0.0 and bq_lin.max() == 0.0),
            "first_divergent_step": fz,
            "linf_final": float(linf[-1]),
            "linf_max": float(linf.max()),
            "rms_final": float(rms[-1]),
            "body_linf_final": float(bq_lin[-1]),
            "linf_over_tol_final": float(linf[-1] / TOL),
            "resets": any_reset,
            "first_reset_mask_divergence": first_reset_div,
            "alive_envs_final": int(alive_n[-1]),
            "cum_substeps": [int(cum_a[-1]), int(cum_b[-1])],
            "diverged_worlds": [div_a, div_b],
            "linf_curve": [float(x) for x in linf],
            "alive_curve": [int(x) for x in alive_n],
        }
        print(f"\n{cand} vs {ref}  --  {what}")
        if results[cand]["bitwise_identical"]:
            print(f"  BITWISE IDENTICAL over all {n} steps ({int(alive_n[-1])} envs compared)")
        else:
            print(f"  first divergent step: {fz}")
            # Early steps matter more than the endpoint: in a contact-violent
            # regime every seed of perturbation grows to the same saturation
            # level, so what separates the arms is ONSET and EARLY GROWTH.
            marks = sorted({0, 1, 2, 3, 5, 10, n // 4, n // 2, 3 * n // 4, n - 1} & set(range(n)))
            print("     step | alive |   joint_q Linf |  joint_q RMS |  body_q Linf(m) | Linf/tol")
            for s in marks:
                print(
                    f"    {s:5d} | {int(alive_n[s]):5d} | {linf[s]:14.6e} | {rms[s]:12.6e} | "
                    f"{bq_lin[s]:15.6e} | {linf[s] / TOL:8.3e}"
                )
        print(f"  cumulative substeps: {ref}={int(cum_a[-1])} {cand}={int(cum_b[-1])}")
        print(f"  worlds that ever latched diverged: {ref}={div_a} {cand}={div_b}")
        print(
            f"  resets fired: {ref}={any_reset[0]} {cand}={any_reset[1]}; "
            f"first reset-mask divergence at step {first_reset_div}; "
            f"envs still compared at the last step: {int(alive_n[-1])}/{nenv}"
        )

    # ---- ensemble statistics -------------------------------------------
    # Pointwise trajectory divergence is the WRONG headline in a chaotic
    # contact system: any perturbation, down to one ulp, grows and saturates.
    # What a paper's accuracy claim actually rests on is the ENSEMBLE -- the
    # distribution of the controller's own accepted error and accepted step
    # over all (world, boundary) samples.  Two arms whose ensembles agree are
    # running the same physics even when no individual world's trajectory
    # matches.  These are reported alongside, not instead of, the pointwise
    # numbers.
    print("\n" + "=" * 78)
    print("ENSEMBLE STATISTICS (all worlds x all boundaries)")
    print("=" * 78)
    print(
        f"{'arm':12} {'acc_err med':>12} {'acc_err p90':>12} {'acc_err max':>12} "
        f"{'sub/bnd mean':>11} {'sub/bnd max':>11} {'substeps':>10}"
    )
    ens = {}
    for k, d in data.items():
        if not d["acc_err"].size:
            continue
        ae = d["acc_err"].astype(np.float64).ravel()
        ae = ae[np.isfinite(ae)]
        dt = d["dt"].astype(np.float64).ravel()
        ens[k] = {
            "acc_err_median": float(np.median(ae)),
            "acc_err_p90": float(np.quantile(ae, 0.9)),
            "acc_err_max": float(ae.max()),
            "substeps_per_boundary_mean": float(dt.mean()),
            "substeps_per_boundary_max": float(dt.max()),
            "cum_substeps": int(d["cum"][-1]),
            "engagement": json.loads(str(d["engagement"][0])) if "engagement" in d else None,
        }
        e = ens[k]
        print(
            f"{k:12} {e['acc_err_median']:12.4e} {e['acc_err_p90']:12.4e} {e['acc_err_max']:12.4e} "
            f"{e['substeps_per_boundary_mean']:11.4f} {e['substeps_per_boundary_max']:11.0f} {e['cum_substeps']:10d}"
        )
    if "ref" in ens:
        print("\nratio to the reference arm (1.000 = statistically identical ensemble):")
        for k, e in ens.items():
            if k == "ref":
                continue
            r = ens["ref"]
            print(
                f"  {k:12} acc_err med {e['acc_err_median'] / r['acc_err_median']:6.3f}  "
                f"p90 {e['acc_err_p90'] / r['acc_err_p90']:6.3f}  "
                f"sub/bnd {e['substeps_per_boundary_mean'] / max(r['substeps_per_boundary_mean'], 1e-30):6.3f}  "
                f"substeps {e['cum_substeps'] / max(r['cum_substeps'], 1):6.3f}"
            )

    # ---- verdict --------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    rc = 0
    oracle = results.get("ref-repeat")
    if oracle is None:
        print("  ORACLE NOT RUN -- bitwise judgements below are not validated.")
    elif not oracle["bitwise_identical"]:
        print("  ORACLE-DEGRADED: two identical det=1 runs are NOT bitwise identical")
        print(f"    (Linf {oracle['linf_final']:.3e} by step {oracle['steps'] - 1}).")
        print("    Every bitwise judgement in this probe is void; the bounded")
        print("    comparisons below still stand as measurements.")
        rc = 2
    else:
        print("  ORACLE OK: det=1 is run-to-run bitwise reproducible on this rig.")

    scale_b = results.get("prod-b", {}).get("linf_final")
    scale_c = results.get("seedB", {}).get("linf_final")
    print(f"\n  reference scales at step {STEPS - 1} (joint_q Linf):")
    print(f"    (a) method's accepted local-error budget tol          = {TOL:.3e}")
    print(
        f"    (b) run-to-run spread of the SHIPPED config (det=0)   = "
        f"{scale_b if scale_b is None else f'{scale_b:.3e}'}"
    )
    print(
        f"    (c) seed-to-seed spread                               = "
        f"{scale_c if scale_c is None else f'{scale_c:.3e}'}"
    )
    nondet_curve = results.get("prod-b", {}).get("linf_curve")
    for key, label in (("nofuse", "fusions D1+D2+D3"), ("refmode", "AGGREGATE all-legacy")):
        r = results.get(key)
        if r is None:
            continue
        v = r["linf_final"]
        print(f"\n  {label}: Linf = {v:.3e}")
        print(f"    vs (a) tol      : {v / TOL:.3e} x")
        if scale_b:
            print(f"    vs (b) nondet   : {v / scale_b:.3e} x")
        if scale_c:
            print(f"    vs (c) seed     : {v / scale_c:.3e} x")
        # Endpoint ratios are weak once divergence saturates. The curve-vs-
        # curve comparison is the discriminating one: if this arm's growth
        # curve never rises materially above the run-to-run nondeterminism
        # curve, the optimization introduces no perturbation the shipped
        # configuration does not already introduce by itself.
        if nondet_curve:
            m = min(len(nondet_curve), len(r["linf_curve"]))
            num = np.array(r["linf_curve"][:m], dtype=np.float64)
            den = np.array(nondet_curve[:m], dtype=np.float64)
            live = den > 0
            if live.any():
                ratio = num[live] / den[live]
                print(
                    f"    curve vs nondet : median {np.median(ratio):.3f} x, "
                    f"max {ratio.max():.3f} x over {int(live.sum())} steps"
                )
                r["curve_ratio_to_nondet_median"] = float(np.median(ratio))
                r["curve_ratio_to_nondet_max"] = float(ratio.max())

    with open(os.path.join(scratch, "p37_divergence_results.json"), "w") as f:
        json.dump(
            {
                "config": {"n_envs": N_ENVS, "steps": STEPS, "tol": TOL},
                "pairs": results,
                "ensemble": ens,
                "failed_arms": failed_arms,
            },
            f,
            indent=2,
        )
    print(f"\nresults -> {os.path.join(scratch, 'p37_divergence_results.json')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
