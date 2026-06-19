"""Plot the arm's graspable table footprint from reach_map.py output (top-down, robot-root frame).

Shows the full table, every reachable grasp-TCP sample, the subset that can actually grasp on the
table (gripper down, near table height), and a robust rectangle over that graspable cloud -- the
proposed cube/goal spawn region (true-random uniform over the WHOLE reachable area).

    uv run --with matplotlib python scripts/rl/trossen/plot_reach_map.py \
        ~/isaac-rl/reach_map.json -o ~/isaac-rl/reach_map.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("json")
    p.add_argument("-o", "--out", default="reach_map.png")
    p.add_argument("--pct", type=float, default=4.0, help="percentile trim for the spawn box")
    args = p.parse_args()

    d = json.load(open(args.json))
    pts = np.array(d["pts"])  # [n, 4] = x,y,z,ax_z
    m = d["meta"]
    x, y, z, axz = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    gz0, gz1 = m["grasp_z"]
    grasp = (z > gz0) & (z < gz1) & (axz < m["down_thresh"])
    gx, gy = x[grasp], y[grasp]

    # robust spawn rectangle over the graspable cloud (trim outliers)
    bx = (np.percentile(gx, args.pct), np.percentile(gx, 100 - args.pct))
    by = (np.percentile(gy, args.pct), np.percentile(gy, 100 - args.pct))

    fig, ax = plt.subplots(figsize=(8, 9))
    tx, ty = m["table"]["x"], m["table"]["y"]
    ax.add_patch(mpatches.Rectangle((tx[0], ty[0]), tx[1] - tx[0], ty[1] - ty[0],
                                    fill=False, ec="black", lw=2, label="table"))
    ax.scatter(x, y, s=2, c="0.75", alpha=0.3, label="reachable (any height)")
    ax.scatter(gx, gy, s=4, c="tab:green", alpha=0.5, label="graspable on table")
    ax.add_patch(mpatches.Rectangle((bx[0], by[0]), bx[1] - bx[0], by[1] - by[0],
                                    fill=False, ec="tab:red", lw=2.5, label="proposed spawn box"))
    ax.scatter([-0.02], [m["base_y"]], marker="s", s=90, c="tab:blue", label="arm base", zorder=5)
    ax.scatter([m["rest_tcp"][0]], [m["rest_tcp"][1]], marker="*", s=160, c="orange",
               label="rest gripper", zorder=5)

    ax.set_xlabel("x  [m]  (robot-root)")
    ax.set_ylabel("y  [m]  (toward arm base ->)")
    ax.set_title(f"Left-arm graspable footprint ({m['n']} samples)\n"
                 f"proposed spawn  x[{bx[0]:.2f}, {bx[1]:.2f}]  y[{by[0]:.2f}, {by[1]:.2f}]")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")
    print(f"proposed spawn box: x[{bx[0]:.3f}, {bx[1]:.3f}]  y[{by[0]:.3f}, {by[1]:.3f}]  "
          f"(graspable samples: {grasp.sum()}/{len(pts)})")


if __name__ == "__main__":
    main()
