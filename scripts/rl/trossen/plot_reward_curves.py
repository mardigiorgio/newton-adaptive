"""Plot per-term reward curves from an rsl_rl training log (Isaac Lab console output).

Reads the ``Episode_Reward/<term>: <value>`` lines rsl_rl prints once per iteration and
plots each term against the iteration index. This is a time series (x = iteration), so per
the repo plotting convention it uses a linear x-axis.

    uv run --with matplotlib scripts/rl/trossen/plot_reward_curves.py \
        ~/isaac-rl/full_train.log -o ~/isaac-rl/reward_curves.png
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE = re.compile(r"Episode_Reward/(\S+?):\s*([-\d.]+)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="+", help="one or more training logs, concatenated in order")
    p.add_argument("-o", "--out", default="reward_curves.png")
    args = p.parse_args()

    series: dict[str, list[float]] = defaultdict(list)
    for path in args.log:
        with open(path) as f:
            for line in f:
                m = LINE.search(line)
                if m:
                    series[m.group(1)].append(float(m.group(2)))

    fig, ax = plt.subplots(figsize=(10, 6))
    for term, vals in sorted(series.items(), key=lambda kv: -max(kv[1])):
        ax.plot(range(1, len(vals) + 1), vals, label=f"{term} (max {max(vals):.2f})", lw=1.5)

    ax.set_xlabel("training iteration")
    ax.set_ylabel("mean episode reward (per term)")
    ax.set_title("Teacher reward terms over training -- settle vs. still-climbing")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  ({len(series)} terms, {max(len(v) for v in series.values())} iters)")


if __name__ == "__main__":
    main()
