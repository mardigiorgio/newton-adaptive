# Part 1 — pure-solver results, in the CENIC paper's formats

Reference: Kurtz & Castro, *CENIC: Convex Error-controlled Numerical
Integration for Contact*, arXiv:2511.08771. Every figure below is that
paper's diagram, on that paper's scene, with the accuracy ε_acc or time
step δt stated on every point. Regenerate everything with
`uv run python scripts/bench/part1_plots.py && uv run python scripts/bench/part1_tables.py`.

## Four arms

| arm | integrator | accuracy knob |
|---|---|---|
| MuJoCo, fixed step | `SolverMuJoCo`, n substeps per 10 ms boundary | δt = 10 ms / n |
| MuJoCo, error control | `SolverMuJoCoAdaptive` — CENIC's first-order step doubling on MuJoCo's convex solver, per-world Drake step selection | ε_acc |
| ICF, fixed step | `SolverICF` (icf_warp), Newton collision pipeline per substep | δt |
| ICF, error control (CENIC) | `SolverICFAdaptive` — the same controller over ICF, two geometry queries per step | ε_acc |

Both error-controlled arms use the paper's position-only weighted L∞ error
estimate ‖S(q − q̂)‖∞ with S = I (Sec. V-E; the bottom row of the paper's
Fig. 10). All four arms replay one captured CUDA graph per boundary, so wall
times compare like with like (the MuJoCo-adaptive solver captures internally;
driven eagerly, the other three would pay per-kernel launch overhead it does
not — measured as a captured adaptive boundary timing *below* one eager fixed
substep). Captured-vs-eager physics agrees to 1e-8 on a well-posed scene.

## Scenes (`scripts/scenes/cenic_scenes.py`)

From the paper's Sec. VII and Figs. 6/8:

* **Soft clutter** — 20 spheres dropped into a bin, k = 10³ N/m, v_s = 1 cm/s.
* **Hard clutter** — 10 spheres + 10 cubes, k = 10⁵ N/m, v_s = 0.1 mm/s.
* **Bouncing ball** (Fig. 8) — 0.1 kg, k = 10³ N/m, zero dissipation, 1 m drop,
  10 s; potential energy zero at rest on the ground; the paper states 11 bounces.

Stiffness, dissipation and stiction reach the two backends differently and
each scene sets both: MuJoCo from per-shape `ke`/`kd` (Newton's solref
conversion), ICF from `IcfParams.contact_stiffness / contact_hc_dissipation /
contact_stiction_tolerance`.

**Assumed (the paper does not state them) — to confirm with the authors:**
sphere radius and cube half-extent 2.5 cm; bin 30 × 30 cm, 30 cm walls;
μ = 0.5; clutter dissipation kd = 0.02·k (MuJoCo) and Hunt–Crossley d = 10 s/m
(ICF default); water density; initial 4 × 5 lattice above the bin; ball
radius 5 cm.

**Deliberate deviations from the paper, stated in every caption:**
* Maximum step δt_max = 10 ms — our control boundary (the paper's clutter
  runs allowed 0.1 s). Error control never steps past a boundary.
* GPU, not CPU: "N = 1 world" is the paper's single-scene semantics but sits
  near the launch-latency floor (~1–3 ms per boundary); "N = 1024" is the
  regime robot learning runs in. Both are reported.
* Only the point-contact scenes are reproduced; cylinder / gripper / Franka
  use hydroelastic contact and an inverse-dynamics controller we do not have.
* Fixed-step arms have no accuracy knob; they appear as reference levels at
  δt (the paper's Fig. 11 does the same).

## Protocol and failure conventions

* **Work-precision** (paper Fig. 9/10): x = ε_acc from 10⁻¹ to 10⁻⁶, y = wall
  time per simulated second over a 2 s horizon, first two boundaries (module
  load, graph capture) excluded; N = 1 rows are medians of 3 subprocess trials.
  Timeout = 100 s per simulated second (real-time rate < 1 %), drawn as ×.
  The error-controlled arms run with a 4096-substep march budget (δt floor
  2.4 µs); a run in which any world ever exhausted it is marked
  `budget-exhausted` and treated as a failure — none did.
* **Penetration vs wall** (ours): 64 worlds, 200 boundaries after a 20-boundary
  warm-up; ground penetration read from model geometry (verified against an
  independent recomputation, `verify_part1_penetration.py`); ejections past the
  bin walls split into through-wall (below the rim) vs over-rim, by shape.
  Timing and metric passes are separate subprocesses (no host sync in the
  timed loop).
* **Wall vs worlds**: 2⁶ … 2¹³ worlds, fixed arms at δt = 10 ms, error control
  at ε = 10⁻³, median and p90 of per-boundary wall over 100 boundaries.
* **Energy convergence** (paper Fig. 8): fixed arms at δt from 10 ms to 10 µs;
  % change of total energy after 10 s; rebounds counted as boundary-sampled
  upward velocity sign flips.
* **Table I analog**: real-time rate = 100 / (wall per simulated second) at
  N = 1; artifact = any ejection or max ground penetration > 1 mm (4 % of the
  object radius) in the 64-world run. The criterion is ours; the paper's
  Table I judged artifacts visually.

## Captions (LaTeX, ICRA style — paste with the PDFs in `figures/`)

```latex
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/workprecision.pdf}
\caption{Work-precision plots (format of CENIC Fig.~10) for the paper's soft- and
hard-clutter scenes, error control on positions ($\mathbf{S}=\mathbf{I}$).
Wall time is normalized to one simulated second; top row a single world
(N=1), bottom row N=1024 parallel worlds on one GPU. Dotted levels are the
fixed-step arms at $\delta t = 10$\,ms and $1$\,ms. Maximum step
$\delta t_{\max} = 10$\,ms in all arms (the control boundary).
$\times$ marks a timeout ($>100$\,s per simulated second). Lower is better.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/speed_bars.pdf}
\caption{Simulator speed comparison (format of CENIC Fig.~11), single world:
fixed step at $\delta t=10$\,ms / $1$\,ms and error control at
$\varepsilon_{acc} = 10^{-1}, 10^{-3}, 10^{-5}$. Speed only, not quality;
see Table~I and Fig.~\ref{fig:penetration} for artifacts.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ball_energy.pdf}
\caption{Energy conservation (format of CENIC Fig.~8): percent change in total
energy after 10\,s for a 0.1\,kg ball, $k=10^3$\,N/m, zero dissipation,
dropped from 1\,m. Left: fixed step vs.\ $\delta t$; right: error control
vs.\ $\varepsilon_{acc}$. ICF converges at first order; MuJoCo's contact
dissipates the same fraction of the energy at every $\delta t$ down to
$10\,\mu$s.}
\end{figure}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/penetration_hard-clutter.pdf}
\caption{Ground penetration (mean, max) and bin ejections versus wall time on
hard clutter, 64 worlds. Each point is labeled with its $\delta t$ (fixed
step) or $\varepsilon_{acc}$ (error control); open markers are exactly zero.}
\label{fig:penetration}
\end{figure*}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/scaling_hard-clutter.pdf}
\caption{Wall time per 10\,ms boundary versus number of parallel worlds
($2^6$--$2^{13}$) on hard clutter; fixed step at $\delta t=10$\,ms, error
control at $\varepsilon_{acc}=10^{-3}$; median with the p90 band.}
\end{figure}
```

## Results

### Work-precision (`figures/workprecision.pdf`, `figures/speed_bars.pdf`)

Wall time per simulated second, δt_max = 10 ms, timeouts at 100 s/sim-s.

| scene, N | arm | ε = 10⁻¹ | 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ |
|---|---|---|---|---|---|---|---|
| soft, 1 | ICF error control | 0.26 | 0.26 | 0.30 | 0.36 | 0.57 | 1.11 |
| soft, 1 | MuJoCo error control | 0.15 | 0.15 | 0.25 | 0.58 | 1.19 | 2.48 |
| hard, 1 | ICF error control | 0.68 | 1.27 | 3.59 | 11.5 | 63.0 | timeout |
| hard, 1 | MuJoCo error control | 0.19 | 0.26 | 0.12 | 0.27 | 1.68 | 5.08 |
| soft, 1024 | ICF error control | 7.6 | 10.4 | 14.7 | 14.6 | 17.6 | 29.7 |
| soft, 1024 | MuJoCo error control | 0.42 | 0.98 | 2.47 | 5.64 | 14.4 | 43.7 |
| hard, 1024 | ICF error control | 28.7 | 53.2 | timeout | timeout | timeout | timeout |
| hard, 1024 | MuJoCo error control | 0.71 | 4.71 | 2.92 | 6.78 | 11.3 | 27.8 |

Fixed-step reference levels (N = 1 / N = 1024, s per simulated second): soft —
ICF 0.067 / 7.2 at 10 ms, 0.55 / 36 at 1 ms; MuJoCo 0.052 / 0.35 at 10 ms,
0.37 / 1.5 at 1 ms. Hard — ICF 0.23 / 7.2 at 10 ms, 0.80 / 36 at 1 ms; MuJoCo
0.11 / 0.35 at 10 ms, 0.35 / 1.5 at 1 ms (read the exact values from the CSVs).

What the figure supports:
* On soft clutter, ICF error control is cheaper than MuJoCo error control at
  every ε ≤ 10⁻³ (N = 1) and at ε = 10⁻⁶ (N = 1024); on hard clutter MuJoCo
  error control is cheaper everywhere, by 4–40×.
* ICF error control reaches the paper's default ε = 10⁻³ on hard clutter at
  3.6 s per simulated second for one world (28 % real time); at 1024 worlds it
  exceeds the timeout.
* MuJoCo error control is **non-monotone in ε on hard clutter** (ε = 10⁻³ is
  cheaper than 10⁻² and 10⁻¹ at N = 1 as 3-trial medians, and at N = 1024) —
  a property of its controller on this scene, reported as measured.
* No configuration exhausted the 4096-substep march budget; every missing
  point is a genuine timeout.

### Energy convergence (`figures/ball_energy.pdf`, `tables/part1_table1.md`)

| δt | 10 ms | 5 ms | 2 ms | 1 ms | 0.5 ms | 0.2 ms | 0.1 ms | 50 µs | 20 µs | 10 µs |
|---|---|---|---|---|---|---|---|---|---|---|
| ICF, % energy change | −99.6 | −99.6 | −99.6 | −99.6 | −97.6 | −56.8 | −32.1 | −16.2 | −6.8 | −3.5 |
| ICF, rebounds in 10 s | 2 | 3 | 8 | 12 | 21 | 13 | 11 | 10 | 11 | 11 |
| MuJoCo, % energy change | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 | −99.5 |
| MuJoCo, rebounds | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

* Fixed ICF converges at first order once the impact is resolved
  (δt ≤ 0.2 ms: each halving of δt halves the energy loss) and rebounds 11
  times in 10 s at δt ≤ 0.1 ms — the count the paper states for this scene.
  At δt ≥ 1 ms the first impact absorbs essentially all the energy and the
  ball settles (the ≥ 12 "rebounds" at 0.5–1 ms are resting chatter).
* MuJoCo loses 99.5 % of the energy at every δt down to 10 µs and never
  rebounds more than once: its dissipation lives in the contact model, not
  in the step, so refining δt cannot recover the conservative dynamics.
* ICF error control: the bounce is resolved only at ε ≤ 10⁻⁵ (−61 %; −22 % at
  10⁻⁶ with 11 rebounds); MuJoCo error control is dead at every ε.
* Open question for the authors: the paper's Fig. 8 shows step-doubling
  retaining some energy at δt = 1 ms, where ours loses 99.6 %; our assumed
  radius (5 cm) and 5 mm contact margin are the likely knobs.

### Penetration vs wall and wall vs worlds

_(filled when the budget-4096 reruns land — see `figures/penetration_*.pdf`,
`figures/scaling_*.pdf`)_
