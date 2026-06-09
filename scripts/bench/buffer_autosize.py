# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Systematically measure minimal-safe MuJoCo buffer sizes per (solver, N).

The contact_objects buffers (nconmax=200, njmax=600 per world) are badly
oversized: at the measured peak only ~23 contacts/world and a worst-world
constraint count (nefc) of ~160-184. ``njmax`` is the memory-critical knob --
it sizes ``efc_J`` (~njmax*nv*nworld), the dominant array -- so njmax=600 blows
past 16 GB by N~16384 and makes the whole scaling sweep OOM well before 2^14.

This tool runs each solver's actual episode with generous-but-feasible probe
buffers, captures the true peak usage (every substep, including the violent
drop; for the adaptive solver it hooks the compaction tier step so the active,
densest worlds are measured), and reports the minimal-safe buffers:

  nconmax_needed = ceil(peak_total_contacts / N)   # naconmax = nconmax*N is a shared pool
  njmax_needed   = peak_worst_world_nefc           # njmax is a per-world cap
  recommended    = ceil(needed * margin)            # rounded up

These are starting points for manual tuning, wired into the benchmark via
``contact_objects.buffer_sizes()``.

Example::

    uv run python -m scripts.bench.buffer_autosize \\
      --kinds mujoco_fixed_1ms mujoco_adaptive_1e-3 --ns 256 1024 4096 \\
      --margin 1.4 --out scripts/bench/results/poster_2026-06-05/buffers.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.solvers

from scripts.scenes import contact_objects as co

# Probe buffers: generous (>~2x the observed peaks) yet small enough to fit even
# the adaptive solver's compaction tiers up to ~8192 worlds. The peak usage is
# N-independent in distribution (i.i.d. world replicas); only the extreme over N
# worlds and over the run length creeps up, so we probe with headroom and a long
# episode and apply a safety margin.
PROBE_NCON = 96
PROBE_NJM = 384
EPISODE_OUTER = 50  # worst-world nefc is an extreme value that creeps up over the
                    # run; measure over a run at least as long as the benchmark's.


@wp.kernel
def _accum_max_i32(src: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.atomic_max(out, 0, src[i])


@wp.kernel
def _accum_max_density(nacon: wp.array(dtype=wp.int32), nworld: int, out: wp.array(dtype=wp.int32)):
    # Per-slot contact occupancy = ceil(total contacts / world count) for THIS
    # data object. nconmax sizes the per-world slot pool (naconmax = nconmax*N),
    # so this is the quantity nconmax must cover. For the adaptive tiers nworld
    # is the tier size, which correctly captures the active (densest) worlds.
    wp.atomic_max(out, 0, (nacon[0] + nworld - 1) // nworld)


class _Peak:
    """GPU-side running max of per-slot contact occupancy and worst-world nefc,
    so a long episode costs only kernel launches (no per-substep host sync)."""

    def __init__(self, device):
        self.dev = device
        self.density = wp.zeros(1, dtype=wp.int32, device=device)  # max contacts/world-slot
        self.nefc = wp.zeros(1, dtype=wp.int32, device=device)     # max worst-world nefc

    def capture(self, d):
        wp.launch(_accum_max_density, dim=1, inputs=[d.nacon, int(d.nworld), self.density], device=self.dev)
        wp.launch(_accum_max_i32, dim=d.nefc.shape[0], inputs=[d.nefc, self.nefc], device=self.dev)

    def result(self):
        wp.synchronize()
        contacts_per_world = int(self.density.numpy()[0])
        nefc_worst = int(self.nefc.numpy()[0])
        return {
            "contacts_per_world": contacts_per_world,
            "nefc_worst": nefc_worst,
            "probe_ncon_hit": contacts_per_world >= PROBE_NCON,
            "probe_njm_hit": nefc_worst >= PROBE_NJM,
        }


def _measure_fixed(N, dt_sub, seed):
    model = co.build_model_randomized(N, seed=seed)
    solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True,
                                         nconmax=PROBE_NCON, njmax=PROBE_NJM)
    contacts = model.contacts()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    d = solver.mjw_data
    peak = _Peak(model.device)
    n_sub = max(1, round(co.DT_OUTER / dt_sub))
    for _ in range(EPISODE_OUTER):
        for _ in range(n_sub):
            solver.step(s0, s1, ctrl, contacts, dt_sub)
            s0, s1 = s1, s0
            peak.capture(d)
    out = peak.result()
    out["nan"] = bool(np.isnan(s0.joint_q.numpy()).any())
    return out


def _measure_adaptive(N, tol, seed):
    model = co.build_model_randomized(N, seed=seed)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model, tol=tol, dt_init=co.DT_OUTER, dt_min=1e-6, dt_max=co.DT_OUTER,
        use_mujoco_contacts=True, nconmax=PROBE_NCON, njmax=PROBE_NJM,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    peak = _Peak(model.device)
    # Hook the per-(sub)step graph replay so we sample the active tier (or full
    # data) right after each MuJoCo step, where contacts/constraints are live.
    orig = solver._run_step

    def hooked(data):
        orig(data)
        peak.capture(data)

    solver._run_step = hooked
    for _ in range(EPISODE_OUTER):
        solver.step(s0, s1, ctrl, None, co.DT_OUTER)
    out = peak.result()
    out["nan"] = bool(np.isnan(s0.joint_q.numpy()).any())
    return out


# Map scaling-bench kind names -> measurement spec.
_KIND_SPEC = {
    "mujoco_fixed_1ms": ("fixed", {"dt_sub": 1e-3}),
    "mujoco_fixed_10ms": ("fixed", {"dt_sub": 1e-2}),
    "mujoco_adaptive_1e-3": ("adaptive", {"tol": 1e-3}),
    "mujoco_adaptive_1e-2": ("adaptive", {"tol": 1e-2}),
}


def _round_up(x, step):
    return int(math.ceil(x / step) * step)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kinds", type=str, nargs="+", default=list(_KIND_SPEC))
    p.add_argument("--ns", type=int, nargs="+", default=[256, 1024, 4096])
    p.add_argument("--margin", type=float, default=1.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1],
                   help="Measure across these seeds and size to the worst (seed-robust).")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    wp.init()
    results: dict = {"margin": args.margin, "seeds": args.seeds, "probe": [PROBE_NCON, PROBE_NJM], "table": {}}
    hdr = (f"{'kind':>22} {'N':>6} {'cont/world':>10} {'nefc_worst':>10} "
           f"{'->nconmax':>9} {'->njmax':>8} {'sat?':>5} {'nan':>4}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for kind in args.kinds:
        family, kw = _KIND_SPEC[kind]
        results["table"][kind] = {}
        for n in args.ns:
            # Seed-robust: take the worst peak across all seeds.
            agg = {"contacts_per_world": 0, "nefc_worst": 0, "probe_ncon_hit": False,
                   "probe_njm_hit": False, "nan": False}
            failed = None
            for sd in args.seeds:
                try:
                    if family == "fixed":
                        peak = _measure_fixed(n, seed=sd, **kw)
                    else:
                        peak = _measure_adaptive(n, seed=sd, **kw)
                except Exception as e:  # OOM / build error at this (kind, N)
                    failed = f"{type(e).__name__}: {str(e)[:50]}"
                    continue
                agg["contacts_per_world"] = max(agg["contacts_per_world"], peak["contacts_per_world"])
                agg["nefc_worst"] = max(agg["nefc_worst"], peak["nefc_worst"])
                agg["probe_ncon_hit"] |= peak["probe_ncon_hit"]
                agg["probe_njm_hit"] |= peak["probe_njm_hit"]
                agg["nan"] |= peak["nan"]
            peak = agg
            if peak["nefc_worst"] == 0 and failed:  # all seeds failed to measure
                results["table"][kind][str(n)] = {"failed": failed}
                print(f"{kind:>22} {n:>6} {'FAIL':>10} ({failed})", flush=True)
                if args.out:
                    Path(args.out).write_text(json.dumps(results, indent=2))
                continue
            cont_pw = peak["contacts_per_world"]
            nconmax = max(16, _round_up(cont_pw * args.margin, 8))
            njmax = max(32, _round_up(peak["nefc_worst"] * args.margin, 16))
            sat = peak["probe_ncon_hit"] or peak["probe_njm_hit"]
            results["table"][kind][str(n)] = {
                "contacts_per_world": round(cont_pw, 2),
                "nefc_worst": peak["nefc_worst"],
                "nconmax": nconmax, "njmax": njmax,
                "saturated": sat, "nan": peak["nan"],
            }
            print(f"{kind:>22} {n:>6} {cont_pw:>10.2f} {peak['nefc_worst']:>10} "
                  f"{nconmax:>9} {njmax:>8} {str(sat):>5} {str(peak['nan']):>4}", flush=True)
            if args.out:
                Path(args.out).write_text(json.dumps(results, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nJSON -> {args.out}", flush=True)
    print("\nNote: 'sat?'=True means a probe buffer was hit -> raise PROBE_* and re-measure.", flush=True)


if __name__ == "__main__":
    main()
