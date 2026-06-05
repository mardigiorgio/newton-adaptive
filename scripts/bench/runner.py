# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark orchestrator.

Discovers benchmark modules, runs each in a subprocess, saves
version-keyed results to scripts/bench/results/<git-hash>/.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import warp as wp

BENCHMARKS_PKG = "scripts.bench.benchmarks"
RESULTS_ROOT = Path("scripts/bench/results")

# Benchmark modules in execution order.
BENCHMARK_NAMES = ["scaling", "accuracy"]


def _git_short_hash() -> str:
    """Return 7-char git hash of HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gpu_info() -> dict:
    """Collect GPU metadata."""
    info = {"device": "unknown", "warp": wp.__version__}
    try:
        dev = wp.get_device("cuda:0")
        info["device"] = dev.name
    except Exception:
        pass
    return info


def _run_benchmark_subprocess(
    bench_name: str,
    args: dict,
    out_dir: Path,
) -> tuple[float, dict | None]:
    """Run a benchmark in a subprocess. Returns (duration_s, data_dict).

    For ``scaling``, runs one subprocess per N value so Warp's CUDA mempool
    resets between Ns. Without per-N isolation, allocations from earlier Ns
    stay resident in the pool and trigger OOM far below the hardware ceiling.
    Partial JSON is written after each successful N, so a crash at any N
    preserves data for all smaller Ns.
    """
    if bench_name == "scaling" and "ns" in args:
        return _run_scaling_per_n(args, out_dir)

    # Other benchmarks (accuracy): one subprocess, all params at once.
    cmd = [sys.executable, "-m", f"{BENCHMARKS_PKG}.{bench_name}"]
    cmd.extend(["--out-dir", str(out_dir)])
    if "trials" in args and bench_name == "accuracy":
        cmd.extend(["--trials", str(args["trials"])])

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running benchmark: {bench_name}", flush=True)
    print(f"Command: {' '.join(cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)

    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False, capture_output=False, text=True, timeout=1800)
    duration = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})", flush=True)
        return duration, None

    json_path = out_dir / f"{bench_name}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return duration, data
    return duration, None


def _run_scaling_per_n(args: dict, out_dir: Path) -> tuple[float, dict | None]:
    """Run scaling.py once per N value for memory-pool isolation."""
    ns = sorted(args["ns"])
    scene = args.get("scene", "contact_objects")
    suffix = "" if scene == "contact_objects" else f"_{scene}"
    final_json = out_dir / f"scaling{suffix}.json"
    tmp_dir = out_dir / "scaling_per_n_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"Running benchmark: scaling (per-N subprocess isolation)", flush=True)
    print(f"N values: {ns}", flush=True)
    print(f"{'=' * 60}", flush=True)

    merged: dict | None = None
    failed_ns: list[int] = []
    t0 = time.perf_counter()

    for n in ns:
        per_n_dir = tmp_dir / f"n{n}"
        per_n_dir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, "-m", f"{BENCHMARKS_PKG}.scaling",
            "--out-dir", str(per_n_dir),
            "--scene", scene,
            "--ns", str(n),
        ]
        if "steps" in args:
            cmd.extend(["--steps", str(args["steps"])])
        if "warmup" in args:
            cmd.extend(["--warmup", str(args["warmup"])])

        print(f"\n  --- N={n} ---", flush=True)
        # Per-N timeout scales with N: small Ns finish in seconds, but N>=4096 with the
        # full 9-solver set at DT_OUTER=20ms can run 30-90+ min. Cap at 2h for safety.
        per_n_timeout = min(7200, 600 + n * 2)
        proc = subprocess.run(cmd, check=False, capture_output=False, text=True, timeout=per_n_timeout)
        if proc.returncode != 0:
            print(f"  N={n}: FAILED (exit {proc.returncode}) — continuing", flush=True)
            failed_ns.append(n)
            continue

        per_n_json = per_n_dir / f"scaling{suffix}.json"
        if not per_n_json.exists():
            print(f"  N={n}: no JSON output", flush=True)
            failed_ns.append(n)
            continue

        with open(per_n_json) as f:
            per_n_data = json.load(f)
        merged = _merge_scaling_n(merged, per_n_data)

        # Persist incrementally so a later crash still leaves us with this N.
        with open(final_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"  N={n}: ok -> appended to {final_json}", flush=True)

    duration = time.perf_counter() - t0

    if merged is None:
        print(f"\nAll Ns failed for scaling.", flush=True)
        return duration, None

    # Recompute power-law exponents per kind using each kind's own surviving Ns.
    from scripts.bench.infra import power_law_exponent
    merged["exponents"] = {
        mode: power_law_exponent(merged["kinds_ns"][mode], mode_data["medians"])
        for mode, mode_data in merged["modes"].items()
        if mode_data["medians"]
    }
    with open(final_json, "w") as f:
        json.dump(merged, f, indent=2)

    if failed_ns:
        print(f"\nNs that failed: {failed_ns}", flush=True)
        print(f"Ns kept:       {merged['ns']}", flush=True)

    return duration, merged


def _merge_scaling_n(merged: dict | None, per_n: dict) -> dict:
    """Append one per-N scaling result into the accumulator dict.

    Per-kind ``modes`` arrays may be empty when a kind failed at this N.
    We track per-kind ``ns_list`` so failures don't misalign other kinds.
    """
    if merged is None:
        merged = {
            "ns": [],  # union of all Ns that produced at least one mode's data
            "steps": per_n.get("steps"),
            "warmup": per_n.get("warmup"),
            "scene": per_n.get("scene"),
            "modes": {},
            "kinds_ns": {},  # per-kind list of Ns that succeeded
        }
    new_n = per_n["ns"][0]
    any_kind_ok = False
    for mode, mode_data in per_n["modes"].items():
        if mode not in merged["modes"]:
            merged["modes"][mode] = {k: [] for k in mode_data}
            merged["kinds_ns"][mode] = []
        # Skip kinds that failed at this N (empty arrays).
        if not mode_data.get("medians"):
            continue
        for key, vals in mode_data.items():
            merged["modes"][mode][key].append(vals[0])
        merged["kinds_ns"][mode].append(new_n)
        any_kind_ok = True
    if any_kind_ok:
        merged["ns"].append(new_n)
    return merged


def run(
    only: str | None = None,
    skip: list[str] | None = None,
    args: dict | None = None,
) -> Path:
    """Run benchmarks, save results. Returns output directory path."""
    if args is None:
        args = {}
    if skip is None:
        skip = []

    commit = _git_short_hash()
    out_dir = RESULTS_ROOT / commit
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine which benchmarks to run.
    to_run = BENCHMARK_NAMES
    if only:
        to_run = [only]
    to_run = [b for b in to_run if b not in skip]

    print(f"Benchmark run: commit={commit}  benchmarks={to_run}", flush=True)
    print(f"Output: {out_dir}", flush=True)

    # Run each benchmark.
    meta_benchmarks = {}
    for bench_name in to_run:
        duration, data = _run_benchmark_subprocess(bench_name, args, out_dir)
        status = "ok" if data is not None else "failed"
        meta_benchmarks[bench_name] = {
            "status": status,
            "duration_s": round(duration, 1),
        }

        # Generate plots from the saved JSON data. Scaling bench plots go
        # into a per-scene subdir so multi-scene output stays organized.
        if data is not None:
            try:
                mod = importlib.import_module(f"{BENCHMARKS_PKG}.{bench_name}")
                scene = data.get("scene")
                target_dir = plots_dir / scene if (bench_name == "scaling" and scene) else plots_dir
                mod.plot(data, target_dir)
            except Exception as e:
                print(f"  Plot generation failed: {e}", flush=True)

    # Write meta.json.
    meta = {
        "commit": commit,
        "branch": _git_branch(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_gpu_info(),
        "args": args,
        "benchmarks": meta_benchmarks,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved -> {out_dir / 'meta.json'}", flush=True)

    # Summary.
    print(f"\n{'=' * 60}")
    print(f"All benchmarks complete. Results in: {out_dir}")
    for name, info in meta_benchmarks.items():
        print(f"  {name}: {info['status']} ({info['duration_s']:.1f}s)")
    print(f"{'=' * 60}", flush=True)

    return out_dir


def list_benchmarks() -> None:
    """Print available benchmarks."""
    print("Available benchmarks:")
    for name in BENCHMARK_NAMES:
        try:
            mod = importlib.import_module(f"{BENCHMARKS_PKG}.{name}")
            doc = (mod.__doc__ or "").strip().split("\n")[0]
        except ImportError:
            doc = "(import failed)"
        print(f"  {name:15s}  {doc}")
