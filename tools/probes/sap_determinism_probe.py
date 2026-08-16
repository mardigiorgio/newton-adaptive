"""Run-to-run determinism certificate for the SAP-adaptive stack on the FULL rig.

Oracle argument: determinism's contract IS run-to-run bitwise identity, so the
oracle is a second, independent process running the identical scenario (same
seed, same build, same device). No golden files, no tolerances: the reference
is recomputed every invocation. A PASS certifies that same-seed reruns of the
Trossen task through SolverSAPAdaptive produce bitwise-identical committed
trajectories; it does NOT certify determinism across devices, builds, env
counts, or under capacity-cap overflow (an overflowing triangle-pair or
contact buffer drops entries by atomic arrival order, which no sort
canonicalizes -- caps must stay scene-sized, and the worker asserts zero
overflow warnings).

Structure: this file is both driver and worker. The driver spawns the worker
TWICE as separate processes (fresh CUDA contexts -- true run-to-run, not
replay) with the SAP-adaptive production defaults, then compares the recorded
per-control-step state (joint_q / joint_qd / body_q / body_qd, plus the
cumulative substep counter) byte-for-byte and reports the first divergent
(step, field, element) with bit patterns on failure.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/sap_determinism_probe.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

N_ENVS = int(os.environ.get("N_ENVS", "256"))
STEPS = int(os.environ.get("STEPS", "20"))
SEED = int(os.environ.get("PROBE_SEED", "7"))
TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"

WORKER_ENV = {
    "NEWTON_SAP": "1",
    "NEWTON_SAP_ADAPTIVE": "1",
    # Production defaults, pinned explicitly so the certificate names the
    # configuration it certifies.
    "NEWTON_SAP_ADAPTIVE_GRAPH": "1",
    "NEWTON_SAP_ADAPTIVE_CONDITIONAL": "1",
    "NEWTON_ADAPTIVE_TAIL_COMPACT": "1",
    "NEWTON_ADAPTIVE_CONTACT_REFRESH": "1",
    "NEWTON_SAP_SOLVE_COMPACT": "1",
    "NEWTON_SAP_LS_COMPACT": "1",
    "NEWTON_SAP_CONTAINMENT": "1",
    "NEWTON_SAP_DETERMINISTIC": os.environ.get("NEWTON_SAP_DETERMINISTIC", "1"),
    "NEWTON_SAP_LINE_SEARCH": os.environ.get("NEWTON_SAP_LINE_SEARCH", "armijo_decay"),
}


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

    torch.manual_seed(SEED)

    env_cfg = parse_env_cfg(TASK, num_envs=N_ENVS)
    env = gym.make(TASK, cfg=env_cfg)
    u = env.unwrapped

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager  # noqa: PLC0415

    from newton.solvers import SolverSAPAdaptive  # noqa: PLC0415

    solver = NewtonManager._solver
    assert isinstance(solver, SolverSAPAdaptive), type(solver).__name__

    env.reset()
    act = torch.zeros((u.num_envs, u.action_manager.total_action_dim), device=u.device)
    state = NewtonManager._state_0
    frames: dict[str, list[np.ndarray]] = {"joint_q": [], "joint_qd": [], "body_q": [], "body_qd": []}
    # A deterministic stack must reproduce FAILURES too: the rand-action
    # regime can hit the solver's converge-or-throw ceiling, and the
    # certificate then requires the throw at the IDENTICAL step with
    # identical prior frames (fail_step -1 = completed all steps).
    fail_step = -1
    with torch.inference_mode():
        for step in range(STEPS):
            act.uniform_(-2.0, 2.0)
            try:
                env.step(act)
            except RuntimeError:
                fail_step = step
                break
            # Post-step host reads (never inside the inner loop).
            frames["joint_q"].append(state.joint_q.numpy().copy())
            frames["joint_qd"].append(state.joint_qd.numpy().copy())
            frames["body_q"].append(state.body_q.numpy().copy())
            frames["body_qd"].append(state.body_qd.numpy().copy())
    cum = solver.cumulative_substeps()
    np.savez(
        out_path,
        cum=np.array([cum], dtype=np.int64),
        fail_step=np.array([fail_step], dtype=np.int64),
        **{k: (np.stack(v) if v else np.zeros((0,), dtype=np.float32)) for k, v in frames.items()},
    )
    env.close()
    os._exit(0)


def main() -> int:
    if os.environ.get("SAP_DET_PROBE_WORKER"):
        worker(os.environ["SAP_DET_PROBE_OUT"])
        return 0

    import numpy as np  # noqa: PLC0415

    scratch = tempfile.mkdtemp(prefix="sap_det_probe_")
    outs = []
    for run in (1, 2):
        out = os.path.join(scratch, f"run{run}.npz")
        env = dict(os.environ)
        env.update(WORKER_ENV)
        env["SAP_DET_PROBE_WORKER"] = "1"
        env["SAP_DET_PROBE_OUT"] = out
        log = os.path.join(scratch, f"run{run}.log")
        with open(log, "w") as lf:
            r = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env, stdout=lf, stderr=lf, check=False)
        if r.returncode != 0 or not os.path.exists(out):
            print(f"FAIL: worker run {run} exited {r.returncode}; log tail:")
            print(open(log).read()[-2000:])
            return 2
        with open(log) as lf:
            log_text = lf.read()
        if "overflow" in log_text.lower():
            print(f"FAIL: worker run {run} reported buffer overflow -- capacity-cap truncation is")
            print("arrival-order nondeterministic; size the caps before certifying.")
            return 3
        outs.append(out)
        print(f"worker run {run}: ok")

    a = np.load(outs[0])
    b = np.load(outs[1])
    failed = False
    fs_a, fs_b = int(a["fail_step"][0]), int(b["fail_step"][0])
    if fs_a >= 0 or fs_b >= 0:
        print(f"note: converge-or-throw fired (run1 step {fs_a}, run2 step {fs_b}); the")
        print("certificate then covers the identical failure point plus all prior frames.")
    for field in ("cum", "fail_step", "joint_q", "joint_qd", "body_q", "body_qd"):
        fa, fb = a[field], b[field]
        if fa.shape != fb.shape or fa.tobytes() != fb.tobytes():
            failed = True
            if fa.shape != fb.shape:
                print(f"FAIL[{field}]: shape {fa.shape} vs {fb.shape}")
                continue
            flat_a, flat_b = fa.reshape(fa.shape[0], -1), fb.reshape(fb.shape[0], -1)
            for step in range(flat_a.shape[0]):
                if flat_a[step].tobytes() != flat_b[step].tobytes():
                    j = int(np.flatnonzero(flat_a[step] != flat_b[step])[0])
                    va, vb = flat_a[step][j], flat_b[step][j]
                    print(f"FAIL[{field}]: first divergence at step {step}, element {j}: {va!r} vs {vb!r}")
                    if flat_a.dtype == np.float32:
                        ba = int(np.frombuffer(np.float32(va).tobytes(), np.uint32)[0])
                        bb = int(np.frombuffer(np.float32(vb).tobytes(), np.uint32)[0])
                        print(f"  bits: 0x{ba:08x} vs 0x{bb:08x}")
                    break
        else:
            print(f"PASS[{field}]: bitwise identical over {STEPS} steps")
    print(f"cumulative substeps: run1={int(a['cum'][0])} run2={int(b['cum'][0])}")
    if failed:
        print("SAP-DETERMINISM: FAIL")
        return 1
    print(
        f"SAP-DETERMINISM: PASS ({N_ENVS} envs, {STEPS} steps, seed {SEED}, det="
        f"{WORKER_ENV['NEWTON_SAP_DETERMINISTIC']}, ls={WORKER_ENV['NEWTON_SAP_LINE_SEARCH']}, "
        f"ls_compact={WORKER_ENV['NEWTON_SAP_LS_COMPACT']}, "
        f"conditional={WORKER_ENV['NEWTON_SAP_ADAPTIVE_CONDITIONAL']}, "
        f"containment={WORKER_ENV['NEWTON_SAP_CONTAINMENT']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
