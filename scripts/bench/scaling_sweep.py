# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Wide-N scaling sweep with per-(N, kind) subprocess isolation.

The stock runner isolates per N but runs all kinds in one process, so a heavy
kind that OOMs at very high N (the MuJoCo adaptive solver allocates ~14 compaction
tiers and exceeds 16 GB by N~16384) can disturb later kinds and waste time. This
driver runs every (kind, N) in its OWN fresh process: each combo gets the whole
GPU, failures are recorded and skipped, and every kind is plotted up to its own
feasible ceiling. Output goes to a caller-specified dir (never the poster/).

Example::

    uv run python -m scripts.bench.scaling_sweep \\
      --ns 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 \\
      --kinds mujoco_adaptive_1e-3 mujoco_fixed_1ms mujoco_adaptive_1e-2 mujoco_fixed_10ms \\
      --steps 15 --warmup 5 --out-dir scripts/bench/results/poster_2026-06-05
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.bench.infra import power_law_exponent
from scripts.bench.plotting import STYLES, save_fig


def _measure(kind: str, n: int, steps: int, warmup: int, scene: str, tmp: Path) -> tuple[float | None, dict, float]:
    """Run one (kind, N) in a fresh process. Returns (median_seconds|None, extra, wall)."""
    d = tmp / f"{kind}__n{n}"
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "scripts.bench.benchmarks.scaling",
        "--ns", str(n), "--kinds", kind, "--steps", str(steps),
        "--warmup", str(warmup), "--scene", scene, "--out-dir", str(d),
    ]
    t0 = time.perf_counter()
    timeout = min(7200, 300 + n * 3)
    try:
        subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, {"error": "timeout"}, time.perf_counter() - t0
    wall = time.perf_counter() - t0
    suffix = "" if scene == "contact_objects" else f"_{scene}"
    jf = d / f"scaling{suffix}.json"
    if not jf.exists():
        return None, {"error": "no json"}, wall
    data = json.loads(jf.read_text())
    md = data.get("modes", {}).get(kind, {})
    meds = md.get("medians", [])
    if not meds:
        fail = data.get("failures", {}).get(kind, [{}])
        return None, {"error": (fail[0].get("error") if fail else "no data")}, wall
    extra = {"p25": md["p25"][0], "p75": md["p75"][0],
             "k_mean": md["k_means"][0], "k_max": md["k_maxes"][0]}
    return meds[0], extra, wall


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=int, nargs="+", required=True)
    p.add_argument("--kinds", type=str, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--scene", type=str, default="contact_objects")
    p.add_argument("--out-dir", type=str, default="scripts/bench/results/scaling_sweep")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_per_combo"
    tmp.mkdir(exist_ok=True)
    ns = sorted(args.ns)

    results: dict = {"scene": args.scene, "steps": args.steps, "warmup": args.warmup,
                     "ns": ns, "kinds": args.kinds, "modes": {}}
    for kind in args.kinds:
        results["modes"][kind] = {"ns": [], "medians_s": [], "p25_s": [], "p75_s": [],
                                  "k_mean": [], "k_max": [], "failures": []}

    print(f"{'kind':>22} {'N':>7} {'median_ms':>10} {'K_mean':>7} {'wall_s':>7}", flush=True)
    print("-" * 60, flush=True)
    for kind in args.kinds:
        for n in ns:
            med, extra, wall = _measure(kind, n, args.steps, args.warmup, args.scene, tmp)
            m = results["modes"][kind]
            if med is None:
                m["failures"].append({"n": n, "error": extra.get("error")})
                print(f"{kind:>22} {n:>7} {'FAIL':>10}  ({extra.get('error','?')[:40]}) {wall:6.1f}s", flush=True)
                continue
            m["ns"].append(n)
            m["medians_s"].append(med)
            m["p25_s"].append(extra["p25"])
            m["p75_s"].append(extra["p75"])
            m["k_mean"].append(extra["k_mean"])
            m["k_max"].append(extra["k_max"])
            print(f"{kind:>22} {n:>7} {med*1e3:>10.2f} {extra['k_mean']:>7.1f} {wall:6.1f}s", flush=True)
            # Persist incrementally so a crash keeps everything so far.
            (out_dir / "scaling_sweep.json").write_text(json.dumps(results, indent=2))

    # Power-law exponents per kind.
    results["exponents"] = {
        k: power_law_exponent(m["ns"], m["medians_s"])
        for k, m in results["modes"].items() if len(m["medians_s"]) >= 2
    }
    (out_dir / "scaling_sweep.json").write_text(json.dumps(results, indent=2))
    _plot(results, out_dir)


def _plot(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for kind, m in results["modes"].items():
        if not m["medians_s"]:
            continue
        style = STYLES.get(kind)
        ns = m["ns"]
        ms = [v * 1e3 for v in m["medians_s"]]
        p25 = [v * 1e3 for v in m["p25_s"]]
        p75 = [v * 1e3 for v in m["p75_s"]]
        exp = results.get("exponents", {}).get(kind)
        label = (style.label if style else kind) + (f"  $N^{{{exp:.2f}}}$" if exp is not None else "")
        color = style.color if style else None
        marker = style.marker if style else "o"
        ls = style.linestyle if style else "-"
        ax.plot(ns, ms, color=color, marker=marker, ls=ls, lw=2, ms=5, label=label)
        ax.fill_between(ns, p25, p75, color=color, alpha=0.12)
    ax.set_xlabel("N worlds")
    ax.set_ylabel("Wall time per outer step [ms]")
    ax.set_title(f"Scaling: wall time vs N  (scene={results['scene']})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    save_fig(fig, out_dir / "scaling_sweep_wall_time.png")
    print(f"\nPlot -> {out_dir / 'scaling_sweep_wall_time.png'}", flush=True)


if __name__ == "__main__":
    main()
