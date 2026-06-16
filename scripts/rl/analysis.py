"""Post-hoc analysis: aggregate eval results across seeds with IQM + bootstrap CIs.

Consumes the ``.npz`` files written by :mod:`scripts.rl.eval_transfer` (one per
training cell x seed) and produces the hypothesis-test comparison: CENIC vs fixed
transfer gaps with stratified-bootstrap 95% CIs (via ``rliable`` when available).

    uv run -m scripts.rl.analysis --glob 'results/*.npz' --metric lvte
"""

from __future__ import annotations

import argparse
import glob
import re

import numpy as np


def _load_cell(path: str) -> dict:
    """Parse a results npz into {backend: {metric: array}} plus the cell label."""
    data = np.load(path)
    cell = {}
    for key in data.files:
        backend, metric = key.split("__", 1)
        cell.setdefault(backend, {})[metric] = data[key]
    label = re.sub(r"\.npz$", "", path.rsplit("/", maxsplit=1)[-1])
    return label, cell


def iqm(x: np.ndarray) -> float:
    """Interquartile mean (robust central tendency)."""
    x = np.sort(x.ravel())
    lo, hi = int(0.25 * len(x)), int(0.75 * len(x))
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(x: np.ndarray, stat=iqm, n_boot: int = 5000, alpha: float = 0.05):
    """Percentile bootstrap CI of ``stat`` over the samples in ``x``."""
    rng = np.random.default_rng(0)
    x = x.ravel()
    boots = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def main():
    p = argparse.ArgumentParser(description="Aggregate transfer results across seeds.")
    p.add_argument("--glob", required=True, help="glob over results/*.npz")
    p.add_argument("--metric", default="lvte", choices=["lvte", "avte", "survival"])
    p.add_argument("--train-backend", default="id", help="eval backend treated as in-distribution")
    p.add_argument("--ref-backend", default="ref_tol", help="eval backend treated as the reference")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match {args.glob!r}")

    print(f"metric={args.metric}  id={args.train_backend}  ref={args.ref_backend}\n")
    for path in paths:
        label, cell = _load_cell(path)
        if args.train_backend not in cell or args.ref_backend not in cell:
            print(f"{label}: missing backends {list(cell)}")
            continue
        m_id = cell[args.train_backend][args.metric]
        m_ref = cell[args.ref_backend][args.metric]
        gap = m_ref - m_id  # transfer gap (higher = worse degradation for error metrics)
        lo, hi = bootstrap_ci(gap)
        print(
            f"{label:34s}  id_IQM={iqm(m_id):.3f}  ref_IQM={iqm(m_ref):.3f}  "
            f"gap_IQM={iqm(gap):+.3f}  95%CI=[{lo:+.3f}, {hi:+.3f}]"
        )

    print(
        "\nInterpretation: a CENIC-trained cell CONFIRMS H1b when its gap CI is "
        "below the fixed-trained cell's gap CI and the two CIs are separated."
    )


if __name__ == "__main__":
    main()
