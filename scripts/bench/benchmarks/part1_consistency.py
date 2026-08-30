# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Self-consistency convergence on clutter, the way Erez, Tassa & Todorov
(ICRA 2015) and Le Lidec et al. (T-RO 2024) measure it: each backend's
reference is itself at a very small fixed step; every test setting is run
over SHORT trajectory pieces re-initialised from the reference state, so a
chaotic pile cannot amplify differences; the deviation at the end of each
piece is averaged over pieces and worlds.

    e(setting) = mean over pieces of max over bodies ||x_test(t_end) - x_ref(t_end)||

Fixed arms sweep dt; error-controlled arms sweep eps_acc; both backends'
error-controlled arms are judged against their own backend's fixed reference
(same model, converged solution). Pieces are 0.1 s, advanced in 10 ms
boundaries so every setting aligns with the stored reference states.

    uv run python -m scripts.bench.benchmarks.part1_consistency --scene hard-clutter --n 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

import time

import numpy as np
import warp as wp

from scripts.bench.four_arms import _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive, build_model
from scripts.scenes.cenic_scenes import SCENES

PIECE_S = 0.1
BOUNDARY_S = 0.01
N_PIECES = 20
T_START = 0.2  # skip the initial free fall
REF_DT = 1e-4
# The last fixed dt IS the reference step: that row is the reference restarted
# against itself and measures the floor of the instrument (the solvers are
# not bit-reproducible on clutter, scripts/bench/probe_determinism.py), below
# which a deviation is noise, not step error.
FIXED_DTS = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 1e-4]
ACCURACIES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]


def _arm(model, scene, backend, kind, knob):
    icf = SCENES[scene].icf
    solref = SCENES[scene].mujoco_solref
    if backend == "icf":
        if kind == "fixed":
            return _make_icf(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, icf)
        return _make_icf_adaptive(model, knob, BOUNDARY_S, icf, 4096)
    if kind == "fixed":
        return _make_mujoco(model, int(round(BOUNDARY_S / knob)), BOUNDARY_S, solref)
    return _make_mujoco_adaptive(model, knob, BOUNDARY_S, 4096, solref)


def _snapshot(state):
    return {k: getattr(state, k).numpy().copy() for k in ("body_q", "body_qd", "joint_q", "joint_qd")}


def _restore(state, snap):
    for k, v in snap.items():
        getattr(state, k).assign(v)


def _reference(scene, backend, n):
    """States every PIECE_S from T_START, from the backend's own tiny-step run."""
    model = build_model(n, scene=scene)
    arm = _arm(model, scene, backend, "fixed", REF_DT)
    s0, s1, c = model.state(), model.state(), model.control()
    snaps = {}
    t = 0.0
    steps_total = int(round((T_START + N_PIECES * PIECE_S) / BOUNDARY_S))
    per_piece = int(round(PIECE_S / BOUNDARY_S))
    for b in range(steps_total + 1):
        t = b * BOUNDARY_S
        if b >= int(round(T_START / BOUNDARY_S)) and (b - int(round(T_START / BOUNDARY_S))) % per_piece == 0:
            snaps[b] = _snapshot(s0)
        if b < steps_total:
            s0, s1 = arm.boundary(s0, s1, c)
    return model, snaps


def _run(scene, backend, kind, knob, n):
    model, snaps = _reference(scene, backend, n)
    arm = _arm(model, scene, backend, kind, knob)
    per_piece = int(round(PIECE_S / BOUNDARY_S))
    starts = sorted(snaps)
    devs, walls = [], []
    s0, s1, c = model.state(), model.state(), model.control()
    _restore(s0, snaps[starts[0]])
    for _ in range(2):  # eager load + graph capture on the first buffers, untimed
        s0, s1 = arm.boundary(s0, s1, c)
    for i in range(len(starts) - 1):
        _restore(s0, snaps[starts[i]])
        wp.synchronize()
        t0 = time.perf_counter()
        for _ in range(per_piece):
            s0, s1 = arm.boundary(s0, s1, c)
        wp.synchronize()
        walls.append(time.perf_counter() - t0)
        x = s0.body_q.numpy().reshape(-1, 7)[:, :3]
        xr = snaps[starts[i + 1]]["body_q"].reshape(-1, 7)[:, :3]
        d = np.linalg.norm(x - xr, axis=1)
        devs.append(float(d.max()) if np.isfinite(d).all() else float("nan"))
    devs = np.array(devs)
    return {"dev_mean_m": float(np.nanmean(devs)), "dev_max_m": float(np.nanmax(devs)), "pieces": int(np.isfinite(devs).sum()),
            "wall_s_per_sim_s": float(np.median(walls) / PIECE_S)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", default="hard-clutter", choices=sorted(SCENES))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--single", nargs=3, metavar=("BACKEND", "KIND", "KNOB"), default=None)
    p.add_argument("--self-check", action="store_true", help="instrument validation, three properties per backend: "
                   "(1) every window runs; (2) the restart floor (the reference solver restarted from its own states -- nonzero "
                   "because solver-internal state such as warm starts is not restored, then amplified by the scene) is deterministic "
                   "across two runs; (3) the floor's dev_mean sits >= 10x below the coarsest knob's, i.e. the instrument resolves "
                   "the signal above its own noise. The floor VALUE is scene-dependent and is reported, not asserted.")
    args = p.parse_args()
    out = args.out or f"scripts/bench/results/part1_consistency_{args.scene}.csv"
    if args.self_check:
        ok = True
        for backend in ("icf", "mujoco"):
            f1 = _run(args.scene, backend, "fixed", REF_DT, args.n)
            f2 = _run(args.scene, backend, "fixed", REF_DT, args.n)
            coarse = _run(args.scene, backend, "fixed", FIXED_DTS[0], args.n)
            pieces_ok = f1["pieces"] == N_PIECES and coarse["pieces"] == N_PIECES
            det_ok = f1["dev_max_m"] == f2["dev_max_m"] and f1["dev_mean_m"] == f2["dev_mean_m"]
            res_ok = f1["dev_mean_m"] * 10.0 <= coarse["dev_mean_m"]
            good = pieces_ok and det_ok and res_ok
            ok &= good
            print(
                f"self-check {backend}: floor dev_mean {f1['dev_mean_m']:.2e} dev_max {f1['dev_max_m']:.2e} m, "
                f"coarse({FIXED_DTS[0]:g}) dev_mean {coarse['dev_mean_m']:.2e}, pieces {'ok' if pieces_ok else 'BAD'}, "
                f"deterministic {'ok' if det_ok else 'BAD'}, resolution {'ok' if res_ok else 'BAD'} -> {'PASS' if good else 'FAIL'}",
                flush=True,
            )
        return 0 if ok else 1
    if args.single is not None:
        backend, kind, knob = args.single
        print("ROW " + json.dumps(_run(args.scene, backend, kind, float(knob), args.n)), flush=True)
        return 0
    rows = []
    configs = [(b, "fixed", d) for b in ("icf", "mujoco") for d in FIXED_DTS] + [(b, "adaptive", e) for b in ("icf", "mujoco") for e in ACCURACIES]
    for backend, kind, knob in configs:
        r = subprocess.run([sys.executable, "-m", "scripts.bench.benchmarks.part1_consistency", "--scene", args.scene, "--n", str(args.n), "--single", backend, kind, str(knob)],
                           capture_output=True, text=True, timeout=7200)
        if "over the scannable budget" in r.stderr:
            print(f"CONTACT OVERFLOW {backend} {kind} {knob}", flush=True); continue
        got = None
        for line in r.stdout.splitlines():
            if line.startswith("ROW "):
                got = json.loads(line[4:])
        if got is None:
            print(f"FAIL {backend} {kind} {knob}: {r.stderr[-400:]}", flush=True); continue
        arm = backend if kind == "fixed" else f"{backend}-adaptive"
        row = {"scene": args.scene, "arm": arm, "dt_s": knob if kind == "fixed" else "", "accuracy": knob if kind == "adaptive" else "",
               "ref_dt_s": REF_DT, "piece_s": PIECE_S, "n_worlds": args.n, **got}
        rows.append(row); print(row, flush=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
