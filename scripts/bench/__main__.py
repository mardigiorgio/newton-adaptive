# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CENIC benchmark platform entry point.

Usage:
    uv run -m scripts.bench                           # run scaling + accuracy
    uv run -m scripts.bench --only scaling            # run one benchmark
    uv run -m scripts.bench --skip accuracy           # skip one
    uv run -m scripts.bench --list                    # list available benchmarks
    uv run -m scripts.bench --ns 1 4 16 64 256        # override N values
    uv run -m scripts.bench --steps 50 --warmup 20    # override timing params
"""

import argparse

from scripts.bench.runner import list_benchmarks, run


def main():
    parser = argparse.ArgumentParser(
        description="CENIC benchmark platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", type=str, default=None, help="Run only this benchmark")
    parser.add_argument("--skip", type=str, nargs="*", default=[], help="Skip these benchmarks")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit")

    # Benchmark parameter overrides.
    parser.add_argument("--ns", type=int, nargs="+", default=None, help="Override N values for scaling")
    parser.add_argument("--kinds", type=str, nargs="+", default=None,
                        help="Restrict scaling to these solver kinds (default: all).")
    parser.add_argument("--steps", type=int, default=None, help="Override timed steps for scaling")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup steps for scaling")
    parser.add_argument("--trials", type=int, default=None, help="Override trial count for accuracy")
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene name for scaling benchmark (default: contact_objects). "
        "See scripts.scenes._registry.bench_scenes() for options.",
    )

    args = parser.parse_args()

    if args.list:
        list_benchmarks()
        return

    bench_args = {}
    if args.ns is not None:
        bench_args["ns"] = sorted(args.ns)
    if args.kinds is not None:
        bench_args["kinds"] = args.kinds
    if args.steps is not None:
        bench_args["steps"] = args.steps
    if args.warmup is not None:
        bench_args["warmup"] = args.warmup
    if args.trials is not None:
        bench_args["trials"] = args.trials
    if args.scene is not None:
        bench_args["scene"] = args.scene

    run(only=args.only, skip=args.skip, args=bench_args)


if __name__ == "__main__":
    main()
