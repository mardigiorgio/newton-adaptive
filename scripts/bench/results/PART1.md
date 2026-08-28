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

## Contact budgets (validity precondition)

A contact the collision pipeline generates but the solver cannot scan is
dropped silently, and every number downstream is physics of a different
scene. Measured peak demand per world: hard clutter ~500 pipeline contacts
(ICF) / ~380 active contacts (MuJoCo); soft clutter ~280 / ~256. The
budgets in `four_arms.py` (ICF 1024, MuJoCo nconmax 1024, njmax 4096) hold
≥ 2× that; `verify_contact_budgets.py` re-measures and fails otherwise, and
every bench marks a configuration whose subprocess reported dropped contacts
as `contact-overflow`, never as a data point.

## Protocol and failure conventions

* **Work-precision** (paper Fig. 9/10): x = ε_acc from 10⁻¹ to 10⁻⁶, y = wall
  time per simulated second over a 2 s horizon, first two boundaries (module
  load, graph capture) excluded; N = 1 rows are medians of 3 subprocess trials.
  Timeout = the paper's 100 s per simulated second **of one scene**: a batch of
  N worlds simulates N scenes, so the rule is wall / (N × simulated s) > 100 s
  (unchanged at N = 1); drawn as × on the threshold line. A separate practical
  wall budget (1 h per run) bounds the N = 1024 sweeps; a run killed by it is
  `budget`, drawn as +, and is not the paper's timeout.
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

> **Re-measurement in progress.** The clutter numbers below were taken with
> budgets of 256 contacts per world — below the measured demand on both
> scenes for both backends — and are superseded by the rerun under the
> budgets above. The ball scene (one contact) is unaffected.

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
| hard, 1024 | ICF error control | 28.7 | 53.2 | 124 | rerun | rerun | rerun |
| hard, 1024 | MuJoCo error control | 0.71 | 4.71 | 2.92 | 6.78 | 11.3 | 27.8 |

Fixed-step reference levels at δt = 10, 5, 2, 1 ms for N = 1 and 1024 are
generated into `tables/part1_table1.md` ("Fixed-step reference levels").

What the figure supports:
* On soft clutter, ICF error control is cheaper than MuJoCo error control at
  every ε ≤ 10⁻³ (N = 1) and at ε = 10⁻⁶ (N = 1024); on hard clutter MuJoCo
  error control is cheaper everywhere, by 4–40×.
* ICF error control reaches the paper's default ε = 10⁻³ on hard clutter at
  3.6 s per simulated second for one world (28 % real time); at 1024 worlds
  124 s per batch-second (0.12 s per scene-second — far inside the paper's
  timeout; the tighter ε points are being rerun under the per-scene rule).
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

### Penetration and ejections vs wall time (`figures/penetration_*.pdf`)

64 worlds, 200 boundaries; every point labeled; no budget exhaustion.

**Hard clutter** (k = 10⁵ N/m):

| arm | setting | mean pen. | max pen. | ejected | of which over the rim / through a wall |
|---|---|---|---|---|---|
| MuJoCo fixed | δt = 10 → 1 ms | 0.68 → 0.19 mm | 12.5 → 4.9 mm | 0 | — |
| MuJoCo error control | ε = 10⁻¹ → 10⁻⁴ | 0.18–0.24 mm | 3.8–7.9 mm | 0 | — |
| ICF fixed | δt = 10 / 5 / 2 / 1 ms | 2.9 µm / 1.1 µm / 4.6 nm / 0 | 8.4 / 3.9 / 0.58 / 0 mm | **9.6 % / 9.7 % / 6.0 % / 5.4 %** | all over the rim, all spheres (through-wall ≤ 0.16 %) |
| ICF error control | ε = 10⁻¹ / 10⁻² / 10⁻³ / 10⁻⁴ | 0.3 µm / 10 µm / 0 / 0 | 3.9 / 22.5 / 0 / 0 mm | 2.0 % / 0.16 % / 0 / 0 | over the rim, spheres |

* MuJoCo never ejects a body and never resolves contact below ~0.2 mm mean /
  ~5 mm max penetration, at any δt or ε — its compliance is in the model.
* Fixed ICF at δt ≥ 1 ms **launches 5–10 % of the spheres out of a 30 cm bin**:
  the contact period at k = 10⁵ N/m is ~2.5 ms, so a 10 ms first-order step
  does not resolve the impact and injects energy. The ejections are over the
  rim (through-wall ≤ 0.16 %) and spheres only — dynamics, not a collision
  failure.
* ICF error control removes the launches (2 % at ε = 0.1, 0.16 % at 10⁻², none
  at ≤ 10⁻³) and reaches exactly zero ground penetration at ε ≤ 10⁻³, at 25×
  the fixed-10 ms cost (300 ms vs 12 ms per boundary at 64 worlds) or 4.5× the
  fixed-1 ms cost.

**Soft clutter** (k = 10³ N/m) — read with the mass caveat below:

| arm | setting | mean pen. | max pen. | ejected |
|---|---|---|---|---|
| MuJoCo fixed | δt = 10 → 1 ms | 9.6 → 19 mm | **1.8–5.3 m** | 6–7 % |
| MuJoCo error control | ε = 10⁻¹ → 10⁻⁴ | 11 → 19 mm | **2.5–3.2 m** | 5–8 % |
| ICF fixed | δt = 10 / 5 / 2 / 1 ms | 1.7 mm / 0.19 mm / 25 µm / 16 µm | 110 / 24 / 6.0 / 5.1 mm | 0 |
| ICF error control | ε = 10⁻¹ / 10⁻² / 10⁻³ / 10⁻⁴ | 0.29 / 0.23 / 0.13 / 0.007 mm | 42 / 27 / 45 / 3.0 mm | 0 |

* With the assumed objects (r = 2.5 cm at water density, 65 g), an impact at
  ~2.8 m/s penetrates v·√(m/k) ≈ 22 mm — the sphere radius — so k = 10³ N/m
  is near the limit of what point contact can represent for these masses.
  The paper's own ball (0.1 kg, k = 10³, 1 m drop) penetrates ~4 cm of a 5 cm
  radius, so deep penetration is in the paper's regime; the object mass is
  the assumption to confirm with the authors.
* MuJoCo (fixed and error-controlled): bodies **pass through the floor and
  keep falling** (max "penetration" of metres = free fall after tunnelling)
  and 5–8 % are launched over the rim, at every δt and ε.
* ICF keeps every body in the bin at every setting; fixed 10 ms tunnels one
  body 11 cm; error control at ε = 10⁻⁴ holds max penetration to 3 mm.

### Wall vs worlds (`figures/scaling_*.pdf`)

_(3-trial rerun in progress; the single-run sweep's ICF error-control row on
hard clutter scattered with world count — zero exhaustion, chaotic impact
window — and is superseded.)_
