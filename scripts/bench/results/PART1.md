# Part 1 — pure-solver results

Test cases and diagrams follow Kurtz & Castro, *CENIC* (arXiv:2511.08771;
`\cite{cenic}` below); every point states its accuracy ε_acc or time step δt.
Regenerate everything with
`uv run python scripts/bench/part1_plots.py && uv run python scripts/bench/part1_tables.py`.

## Four arms

| arm | integrator | accuracy knob |
|---|---|---|
| MuJoCo, fixed step | `SolverMuJoCo`, n substeps per 10 ms boundary | δt = 10 ms / n |
| MuJoCo, error control | `SolverMuJoCoAdaptive` — CENIC's first-order step doubling on MuJoCo's convex solver, per-world Drake step selection | ε_acc |
| ICF, fixed step | `SolverICF` (icf_warp), Newton collision pipeline per substep | δt |
| ICF, error control (CENIC) | `SolverICFAdaptive` — the same controller over ICF, two geometry queries per step | ε_acc |

Both error-controlled arms use the position-only weighted L∞ error estimate
‖S(q − q̂)‖∞ with S = I of~\cite{cenic} (their Sec. V-E; the bottom row of
their Fig. 10). All four arms replay one captured CUDA graph per boundary, so wall
times compare like with like (the MuJoCo-adaptive solver captures internally;
driven eagerly, the other three would pay per-kernel launch overhead it does
not — measured as a captured adaptive boundary timing *below* one eager fixed
substep). Captured-vs-eager physics agrees to 1e-8 on a well-posed scene.

## Scenes (`scripts/scenes/cenic_scenes.py`)

Three test cases from~\cite{cenic}, rebuilt on Newton with point contact
(force only at penetration; no collision margin), the maximum step of their
Table III (0.1 s on clutter), and a single deterministic initial condition:

* **Soft clutter** — 20 spheres (r = 2.5 cm, water density) dropped into a
  30 × 30 × 30 cm bin; k = 10³ N/m, v_s = 1 cm/s, μ = 0.5.
* **Hard clutter** — 10 spheres and 10 cubes (h = 2.5 cm) into the same bin;
  k = 10⁵ N/m, v_s = 0.1 mm/s, μ = 0.5.
* **Bouncing ball** — 0.1 kg, r = 5 cm, k = 10³ N/m, zero dissipation,
  dropped from 1 m and simulated for 10 s; potential energy zero at rest on
  the ground (11 rebounds in~\cite{cenic}).

The clutter drop starts from a 4 × 5 lattice above the bin, alternate layers
staggered by half the column spacing, every body jittered by ±1.5 cm in xy
and ±5 mm in z and every cube tilted by a random rotation, under a fixed
seed (7). Dissipation: kd = 0.02·k for MuJoCo, Hunt–Crossley d = 10 s/m for
ICF (0 for the ball). Stiffness, dissipation and stiction reach the two
backends differently and each scene sets both: MuJoCo from per-shape
`ke`/`kd` (Newton's solref conversion), ICF from
`IcfParams.contact_stiffness / contact_hc_dissipation /
contact_stiction_tolerance`.

Object and bin sizes, μ, dissipation and the initial arrangement are not
specified in~\cite{cenic}; the values above are this work's definition.

**Deviations from~\cite{cenic}, stated in every caption:**
* Maximum step δt_max = 0.1 s on the clutter cases (their Table III) and
  10 ms on the ball; the first attempt is k_Init·δt_max = 0.1·δt_max.
  Error control never steps past a boundary.
* GPU, not CPU: N = 1 is the single-scene setting and sits near the
  launch-latency floor (~1–3 ms per boundary); N = 1024 is the regime robot
  learning runs in. Both are reported.
* Only the point-contact test cases are reproduced; cylinder, gripper and
  Franka use hydroelastic contact and an inverse-dynamics controller.
* Fixed-step arms have no accuracy knob; they appear as reference levels at
  δt.
* ICF's Newton convergence tolerance follows their Sec. VI-B: ε_tol =
  max(κ·ε_acc, 10⁻⁸) with κ = 10⁻³ under error control, 10⁻⁸ in fixed-step
  mode. (Measured effect on the march step count: < 10 %.)
* Step selection uses their constants: safety 0.9, dead band 0.9–1.2, growth
  cap 5, shrink floor 0.1, error order p = 2.
* Point contact, no collision margin, as in their model; a variant with a
  5 mm margin (contact activated before touching — what a collision margin
  does to ICF's law, `distance − margin`) is reported for the penetration
  and ejection metrics so the two can be compared.

## Contact budgets (validity precondition)

A contact the collision pipeline generates but the solver cannot scan is
dropped silently, and every number downstream is physics of a different
scene. Measured peak demand per world: hard clutter ~500 pipeline contacts
(ICF) / ~380 active contacts (MuJoCo); soft clutter ~280 / ~256. The
budgets in `four_arms.py` (ICF 2048, MuJoCo nconmax 1024, njmax 1024) hold
≥ 2× that; `verify_contact_budgets.py` re-measures and fails otherwise, and
every bench marks a configuration whose subprocess reported dropped contacts
as `contact-overflow`, never as a data point.

## Protocol and failure conventions

* **Work-precision** (paper Fig. 9/10): x = ε_acc from 10⁻¹ to 10⁻⁶, y = wall
  time per simulated second over a 2 s horizon, first two boundaries (module
  load, graph capture) excluded; N = 1 rows are medians of 3 subprocess trials.
  Timeout = 100 s per simulated second **of one scene** (the criterion
  of~\cite{cenic}): a batch of N worlds simulates N scenes, so the rule is
  wall / (N × simulated s) > 100 s (unchanged at N = 1); drawn as × on the
  threshold line. A separate practical wall budget (1 h per run) bounds the
  N = 1024 sweeps; a run killed by it is `budget`, drawn as +, and is not a
  timeout.
  The error-controlled arms run with a 4096-substep march budget (δt floor
  2.4 µs); a run in which any world ever exhausted it is marked
  `budget-exhausted` and treated as a failure — none did.
* **Penetration vs wall** (ours): 64 worlds, 200 boundaries after a 20-boundary
  warm-up; ground penetration read from model geometry and verified against
  an independent recomputation on the final scene and budgets
  (`verify_part1_penetration.py`: model structure, plane, rotation
  convention, live evolution, timing stability — all pass); ejections past
  the bin walls split into through-wall (below the rim) vs over-rim, by shape.
  Timing and metric passes are separate subprocesses (no host sync in the
  timed loop).
* **Wall vs worlds**: 2⁶ … 2¹³ worlds, fixed arms at δt = 10 ms, error control
  at ε = 10⁻³, median and p90 of per-boundary wall over 100 boundaries.
* **Energy convergence** (paper Fig. 8): fixed arms at δt from 10 ms to 10 µs;
  % change of total energy after 10 s; rebounds counted as boundary-sampled
  upward velocity sign flips.
* **Table I analog**: real-time rate = 100 / (wall per simulated second) at
  N = 1; artifact = any ejection, or max ground penetration above 10× the
  scene's single-object static penetration m·g/k — the compliance the
  contact model itself prescribes (6.5 µm on hard clutter at k = 10⁵ N/m,
  0.65 mm on soft clutter at k = 10³ N/m, 65 g objects) — in the 64-world
  run. A few static depths is the model; tens of them is the step. The
  criterion is ours; Table I of~\cite{cenic} judged artifacts visually.

## Captions (LaTeX, ICRA style — paste with the PDFs in `figures/`; `\cite{cenic}` = Kurtz & Castro, arXiv:2511.08771)

```latex
\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/scenes.pdf}
\caption{Test cases, from~\cite{cenic}: (a) soft clutter, 20 spheres dropped
into a bin, $k = 10^3$\,N/m, $v_s = 1$\,cm/s; (b) hard clutter, spheres and
cubes, $k = 10^5$\,N/m, $v_s = 0.1$\,mm/s; (c) a 0.1\,kg ball with zero
dissipation dropped from 1\,m. Top: initial condition; bottom: after
settling under error control.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/workprecision.pdf}
\caption{Work-precision plots for soft and hard clutter, error control on
positions ($\mathbf{S}=\mathbf{I}$), $\delta t_{\max} = 10$\,ms. Wall times
are normalized to one simulated second; top row a single scene, bottom row
1024 scenes in parallel on one GPU. Dotted levels are fixed-step
integration at $\delta t = 10$\,ms and $1$\,ms. Lower is better. A cross
marks a timeout (100\,s per simulated second).}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/speed_bars.pdf}
\caption{Simulator speed comparison for a single scene: fixed step at
$\delta t = 10$\,ms and $1$\,ms, error control at $\varepsilon_{acc} =
10^{-1}, 10^{-3}, 10^{-5}$. This test measures speed only, not quality.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ball_energy.pdf}
\caption{Energy conservation error (percent of energy lost after 10\,s) for
a 0.1\,kg bouncing ball with zero dissipation, $k = 10^3$\,N/m. Left: fixed
step versus $\delta t$; right: error control versus $\varepsilon_{acc}$.
ICF converges at first order; MuJoCo's contact dissipates the same energy
at every $\delta t$.}
\end{figure}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/artifacts.pdf}
\caption{Contact artifacts versus cost on soft and hard clutter, 64 scenes.
Top: maximum ground penetration relative to the contact model's own impact
depth $v\sqrt{m/k}$ -- above the line the time step, not the model, made
the depth; a ring marks a setting that ejected a body from the bin; a star
marks the cheapest artifact-free setting of each arm. Bottom: mean
penetration relative to the model's resting depth $mg/k$. Every point is
labeled with its $\delta t$ (fixed step) or $\varepsilon_{acc}$ (error
control).}
\label{fig:artifacts}
\end{figure*}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ball_workprecision.pdf}
\caption{Energy error versus cost for the bouncing ball: fixed step sweeps
$\delta t$, error control sweeps $\varepsilon_{acc}$, on the same axes.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/scaling_per_world_hard-clutter.pdf}
\caption{Wall time per world per step versus number of parallel scenes on
hard clutter: cost per world falls with batch size until the GPU saturates.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/penetration_hard-clutter.pdf}
\caption{Ground penetration versus wall time on hard clutter, 64 scenes.
Each point is labeled with its $\delta t$ (fixed step) or
$\varepsilon_{acc}$ (error control). ICF penetrates by the contact model's
static compliance ($m g / k$); MuJoCo by its constraint impedance, at every
resolution.}
\label{fig:penetration}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/realtime_trace_n64.pdf}
\caption{Real-time rate, solver steps per 10\,ms and cumulative wall time
along a 5\,s hard-clutter drop, 64 scenes. Fixed step pays the same at
every step; error control pays during the impacts and coasts at
$\delta t_{\max}$ once the pile settles, so at artifact-free quality it is
the cheapest way to simulate the horizon.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/scaling_hard-clutter.pdf}
\caption{Wall time per 10\,ms step versus number of parallel scenes
($2^6$--$2^{13}$) on hard clutter; fixed step at $\delta t = 10$\,ms, error
control at $\varepsilon_{acc} = 10^{-3}$; median of three runs, band the
spread.}
\end{figure}
```

## Results

All numbers are generated from the committed CSVs into
`tables/results_tables.md` (work-precision, fixed-step levels, penetration
and ejections, wall vs worlds, ball energy), `tables/part1_table1.md` (Table I
analog) and `tables/march_cost.md`; the prose below reads them. Every clutter
number comes from one rerun under the hyperparameters of \cite{cenic} (δt_max = 0.1 s,
k_Init = 0.1, the Newton-tolerance rule, point contact, the perturbed drop,
contact budgets ≥ 2× measured demand).

### Work-precision (`figures/workprecision.pdf`, `figures/speed_bars.pdf`)

_(prose from the final CSVs — pending the rerun)_

### Penetration and ejections (`figures/penetration_*.pdf`, `figures/penetration_*_margin5mm.pdf`)

_(prose from the final CSVs — pending the rerun; the 5 mm-margin variant is
reported alongside point contact)_

### Real-time rate along a drop (`figures/realtime_trace_n64.pdf`, `_n1.pdf`)

_(prose from the final CSVs — pending the rerun)_

### Wall vs worlds (`figures/scaling_*.pdf`)

_(prose from the final CSVs — pending the rerun)_

### Energy convergence (`figures/ball_energy.pdf`)

_(prose from the final CSVs — pending the rerun)_
