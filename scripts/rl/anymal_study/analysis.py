"""Post-hoc analysis: aggregate eval results across seeds with IQM + bootstrap CIs.

Consumes the ``.npz`` files written by :mod:`scripts.rl.anymal_study.eval_transfer` (one per
training cell x seed) and produces the hypothesis-test comparison: adaptive vs fixed
transfer gaps with stratified-bootstrap 95% CIs (via ``rliable`` when available).

    uv run -m scripts.rl.anymal_study.analysis --glob 'results/*.npz' --metric lvte
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


# Channel taxonomy for the error-budget ranking: integration (what the adaptive
# solver controls) vs physical dynamics terms vs the sensing term. obs noise is NOT a physical-
# dynamics term and is ranked on its own axis.
_INTEGRATOR = {"ref_tol", "ref_dt"}
_SENSING = {"ref_obsnoise"}


def _channel(backend: str) -> str:
    if backend in _INTEGRATOR:
        return "integrator"
    if backend in _SENSING:
        return "sensing"
    return "physical"


def load_pooled(paths: list[str]) -> dict:
    """Pool {backend: {metric: concatenated-array}} across per-seed npz files."""
    pooled: dict[str, dict[str, list]] = {}
    for path in paths:
        data = np.load(path)
        for key in data.files:
            backend, metric = key.split("__", 1)
            pooled.setdefault(backend, {}).setdefault(metric, []).append(data[key])
    return {b: {m: np.concatenate(v) for m, v in md.items()} for b, md in pooled.items()}


def rank_terms(paths: list[str], metric: str, id_backend: str = "id") -> None:
    """Rank every reference's transfer gap vs ``id`` by gap-IQM (Phase 0b).

    With paired eval seeds, world w is identical across backends, so the per-world
    gap ``m_ref - m_id`` is a valid paired sample. Prints terms sorted ascending so
    the integrator's rank against the physical terms is read directly.
    """
    pooled = load_pooled(paths)
    if id_backend not in pooled:
        raise SystemExit(f"id backend {id_backend!r} not in results ({list(pooled)})")
    m_id = pooled[id_backend][metric]
    rows = []
    for backend, md in pooled.items():
        if backend == id_backend or metric not in md:
            continue
        n = min(len(m_id), len(md[metric]))
        gap = md[metric][:n] - m_id[:n]
        lo, hi = bootstrap_ci(gap)
        rows.append((iqm(gap), lo, hi, backend))
    rows.sort()

    print(f"Error-budget ranking  metric={metric}  id={id_backend}  (n_seeds*worlds pooled)\n")
    print(f"{'rank':>4}  {'backend':14s}  {'channel':10s}  {'gap_IQM':>9}  {'95% CI':>22}  excl0")
    integ_rank = []
    for i, (g, lo, hi, b) in enumerate(rows, 1):
        ch = _channel(b)
        excl0 = "yes" if (lo > 0 or hi < 0) else "no"
        if ch == "integrator":
            integ_rank.append((i, b, g, lo, hi))
        print(f"{i:>4}  {b:14s}  {ch:10s}  {g:>9.4f}  [{lo:>+8.4f}, {hi:>+8.4f}]  {excl0}")

    phys = [g for g, _, _, b in rows if _channel(b) == "physical"]
    med_phys = float(np.median(phys)) if phys else float("nan")
    worst = max((g for g, _, _, _ in rows), default=0.0)  # largest degradation (positive gap)
    print(f"\nmedian physical-term gap_IQM = {med_phys:.4f}   worst-channel gap_IQM = {worst:.4f}")
    # A transfer penalty is a POSITIVE gap (ref worse than id). The integrator is a
    # meaningful sim2real channel only if it degrades transfer (gap>0, CI excludes 0)
    # by an amount comparable to the worst channel. A gap<=0 (policy unaffected or
    # better under the high-fidelity integrator) or a gap dwarfed by another channel
    # is a KILL for the V2 transfer claim.
    for i, b, g, lo, hi in integ_rank:
        excl0 = lo > 0 or hi < 0
        if (not excl0) or g <= 0:
            verdict = "KILL (no integrator-induced degradation: gap<=0 or CI straddles 0)"
        elif worst > 0 and g < worst / 5.0:
            verdict = f"KILL (integrator degradation {g:+.4f} is >5x below worst channel {worst:+.4f})"
        else:
            verdict = "survives-this-gate"
        print(
            f"integrator {b}: degradation rank {i}/{len(rows)}, gap {g:+.4f}, CI excl0={'yes' if excl0 else 'no'} -> {verdict}"
        )


def main():
    p = argparse.ArgumentParser(description="Aggregate transfer results across seeds.")
    p.add_argument("--glob", required=True, help="glob over results/*.npz")
    p.add_argument("--metric", default="lvte", choices=["lvte", "avte", "survival"])
    p.add_argument("--train-backend", default="id", help="eval backend treated as in-distribution")
    p.add_argument("--ref-backend", default="ref_tol", help="eval backend treated as the reference")
    p.add_argument("--rank", action="store_true", help="error-budget ranking: all refs vs id, pooled across seeds")
    args = p.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"no files match {args.glob!r}")

    if args.rank:
        rank_terms(paths, args.metric, args.train_backend)
        return

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
        "\nInterpretation: an adaptive-trained cell CONFIRMS H1b when its gap CI is "
        "below the fixed-trained cell's gap CI and the two CIs are separated."
    )


if __name__ == "__main__":
    main()
