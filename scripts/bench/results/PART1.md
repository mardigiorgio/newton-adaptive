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
backends differently and each scene sets both: ICF from
`IcfParams.contact_stiffness / contact_hc_dissipation /
contact_stiction_tolerance`; MuJoCo from an explicit contact solref per
scene. MuJoCo's contact is a soft constraint whose stiffness scales as
1/(τ²ζ²) and whose time constant τ is clamped to ≥ 2δt; Newton's `ke`/`kd`
→ solref conversion sets ζ from kd (3.16 for the clutters' kd = 0.02 k; 1.0
for the ball that asked for kd = 0), which made MuJoCo ~100× softer than
the requested k and critically damped the ball — our configuration, not
MuJoCo. Each scene therefore carries `mujoco_solref`: τ = 2.4 ms at
k = 10⁵ N/m and 24 ms at k = 10³ N/m (calibrated so a resting sphere sinks
by the model's m·g/k, `tables/mujoco_stiffness_probe.md`), ζ = 1 on the
clutters and the smallest ζ MuJoCo runs stably with for the ball — MuJoCo
admits no zero damping ratio (ζ = 0 diverges), and with ζ = 0.05 the ball
rebounds 25 times yet still loses all its energy: a conservative contact
cannot be represented in MuJoCo's soft-constraint model, which is the
finding of the energy figure stated fairly. At δt > τ/2 MuJoCo's contact is
softer by its own design.

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

## Open: the actuated test case

The clutter cases have no actuator. The regime in which MuJoCo error control
took ~10× longer per iteration than ICF error control in training — a stiff
PD-driven arm on objects — is the "Franka with box" case of~\cite{cenic}
(high-gain controller → short time scales), which Part 1 does not yet
reproduce. That case, built on Newton with a PD-driven arm pushing a box, is
the figure that would show MuJoCo error control's step count and MuJoCo fixed
step's artifacts under actuation.

## Results


All numbers are generated from the committed CSVs into
`tables/results_tables.md` (work-precision, fixed-step levels, penetration
and ejections, wall vs worlds, ball energy), `tables/part1_table1.md` (Table I
analog) and `tables/march_cost.md`; the prose below reads them. Every clutter
number comes from one rerun under the hyperparameters of \cite{cenic} (δt_max = 0.1 s,
k_Init = 0.1, the Newton-tolerance rule, point contact, the perturbed drop,
contact budgets ≥ 2× measured demand).

### Contact artifacts vs cost (`figures/artifacts.pdf`; data in `tables/results_tables.md`)

Top row: max penetration relative to the contact model's own impact depth
v·√(m/k) — above 1 the step made the depth, not the model. Bottom row: mean
penetration relative to the resting depth m·g/k. With MuJoCo's contact
specified as the scene's k (calibrated solref), **both models converge to
the same resting compliance**: at δt = 1 ms fixed MuJoCo and fixed ICF both
sit at the model's 6.4 µm on hard clutter, and ICF error control does so from
ε = 10⁻² on.

* **Cheapest artifact-free setting, hard clutter:** MuJoCo fixed δt = 2 ms
  (0.57 s per simulated second, 64 scenes), MuJoCo error control ε = 10⁻³
  (1.9 s), ICF error control ε = 10⁻² (2.3 s), fixed ICF δt = 1 ms (4.1 s).
  On soft clutter every setting of every arm is artifact-free.
* **Fixed ICF at δt = 10 ms ejects 1.6 % of the bodies** — the large-step
  passthrough failure: at 2.8 m/s a body moves 2.8 cm per step, more than its
  2.5 cm radius, so with point contact the first contact it sees is already
  buried past its centre and the lagged spring launches it. MuJoCo's soft
  constraint at 10 ms is 3× too deep but does not launch. Neither happens at
  δt ≤ 5 ms, nor under error control at any ε.
* **Coarse steps are soft in both models, for different reasons:** MuJoCo
  clamps its contact time constant to ≥ 2δt (50× the resting depth at 10 ms,
  15× at 5 ms); first-order ICF under-resolves the impact (35× at 10 ms).
  Both recover the model as δt shrinks; error control recovers it as ε
  tightens.

### Work-precision (`figures/workprecision.pdf`, `figures/speed_bars.pdf`)

Wall time per simulated second vs requested accuracy, δt_max = 0.1 s;
N = 1 rows are 3-trial medians; no timeouts, no budget exhaustion.

* MuJoCo error control is cheaper than ICF error control at every ε on both
  scenes: 1.5–4× for a single hard-clutter scene (equal at ε = 10⁻⁴), 5–30×
  at 1024 scenes. Per march iteration ICF costs 2.7–6.9 ms against MuJoCo's
  1.6–2.1 ms (three convex solves vs one soft-constraint solve per attempt),
  and both take comparable step counts (`tables/march_cost.md`).
* Error control reaches any requested ε at a cost growing as ε^(-1/2); the
  fixed-step levels show what the same solver costs at 10 ms and 1 ms.

### Error control pays only when something happens (`figures/realtime_trace_n64.pdf`)

Real-time rate, solver steps per boundary and cumulative wall along a 5 s
hard-clutter drop, 64 scenes. Fixed step's rate is flat by construction;
error control at ε = 10⁻² starts at ~10 % real time during the impacts and
climbs past 100 % once the pile settles, taking ~10 steps per 100 ms where
fixed 1 ms takes 100. Over the 5 s, ICF error control at ε = 10⁻² costs
about half of fixed ICF at 1 ms.

### Energy convergence (`figures/ball_energy.pdf`, `figures/ball_workprecision.pdf`)

* Fixed ICF converges at first order once the impact is resolved (δt ≤
  0.2 ms: each halving of δt halves the loss; 3.5 % at 10 µs) and rebounds
  11 times in 10 s at δt ≤ 0.1 ms, the count \cite{cenic} states for this
  scene.
* MuJoCo with the lightest damping it runs with (ζ = 0.05; ζ = 0 diverges)
  **gains** energy at δt ≥ 5 ms (+500× — a fixed-step instability of the
  soft constraint) and loses all of it at δt ≤ 1 ms at every finer step:
  its soft-constraint contact cannot represent a conservative impact at any
  resolution. Its error control does not catch the instability either
  (+15–50× at ε ≥ 10⁻³): the position-only error norm does not see energy.
* ICF error control on positions likewise does not see the energy a soft
  impact loses: it resolves the bounce (11 rebounds) only at ε ≤ 10⁻⁵, where
  the 4096-substep march budget exhausts and the point is marked.

### Wall vs worlds (`figures/scaling_per_world_*.pdf`, `figures/scaling_*.pdf`)

Median of three independent runs per point (spread as the band), fixed step
at δt = 10 ms, error control at ε = 10⁻³, 2 s timed window; no exhaustion.

* Cost per world falls with batch size until the GPU saturates. At 2¹³
  worlds on hard clutter, per 100 ms step: MuJoCo fixed 11 µs per world,
  MuJoCo error control 270 µs, fixed ICF 620 µs, ICF error control 1.5 ms.
* Under point contact our ICF step costs 40–60× MuJoCo's per world at
  ≥ 2¹⁰ worlds. Not the Newton tolerance (10⁻⁵…10⁻⁸ changes wall by < 5 %,
  `tables/newton_tolerance_probe.md`): it is the cost of resolving stiff
  point contact to the model's compliance with a convex Newton solve, and a
  batch pays for its slowest world.
* Reproducibility: ICF's run-to-run spread is < 1 % at every N; MuJoCo error
  control's narrows with N.

### Energy convergence (`figures/ball_energy.pdf`, `figures/ball_workprecision.pdf`)

* Fixed ICF converges at first order once the impact is resolved (δt ≤
  0.2 ms: each halving of δt halves the loss; 3.5 % at 10 µs) and rebounds
  11 times in 10 s at δt ≤ 0.1 ms, the count \cite{cenic} states for this
  scene.
* MuJoCo with the lightest damping it runs with (ζ = 0.05; ζ = 0 diverges)
  **gains** energy at δt ≥ 5 ms (+500× — a fixed-step instability of the
  soft constraint) and loses all of it at δt ≤ 1 ms at every finer step:
  its soft-constraint contact cannot represent a conservative impact at any
  resolution. Its error control does not catch the instability either
  (+15–50× at ε ≥ 10⁻³): the position-only error norm does not see energy.
* ICF error control on positions likewise does not see the energy a soft
  impact loses: it resolves the bounce (11 rebounds) only at ε ≤ 10⁻⁵, where
  the 4096-substep march budget exhausts and the point is marked.

### Wall vs worlds (`figures/scaling_per_world_*.pdf`, `figures/scaling_*.pdf`)

Median of three independent runs per point (spread as the band), fixed step
at δt = 10 ms, error control at ε = 10⁻³, 2 s timed window; no exhaustion;
tables in `tables/results_tables.md`.

* Cost per world falls with batch size until the GPU saturates: on hard
  clutter at 2¹³ worlds MuJoCo fixed step costs 17 µs per world per 100 ms
  step, MuJoCo error control 370 µs, fixed ICF 610 µs, ICF error control
  1.5 ms; fixed ICF and ICF error control saturate from 2¹⁰ worlds.
* Under point contact and the hyperparameters of~\cite{cenic}, our ICF
  step costs ~40× MuJoCo's per world at ≥ 2¹⁰ worlds on hard clutter. This
  is not the Newton tolerance: sweeping it from 10⁻⁵ to 10⁻⁸ changes wall
  by < 5 % at 64, 1024 and 4096 worlds and penetration not at all
  (`tables/newton_tolerance_probe.md`). It is the cost of resolving stiff
  point contact to the model's compliance (the fidelity row of
  `figures/artifacts.pdf`) — a hardened impact needs many Newton iterations,
  and a batch pays for its slowest world.
* Reproducibility: ICF's run-to-run spread is < 1 % at every N; MuJoCo error
  control's is wide at small N and narrows with N.

### Energy convergence (`figures/ball_energy.pdf`, `figures/ball_workprecision.pdf`)

* Fixed ICF converges at first order once the impact is resolved (δt ≤
  0.2 ms: each halving of δt halves the loss; 3.5 % at 10 µs) and rebounds
  11 times in 10 s at δt ≤ 0.1 ms, the count~\cite{cenic} states for this
  scene. MuJoCo loses 99.5 % at every δt down to 10 µs and never rebounds
  more than once: its dissipation is in the contact model.
* Error control on positions does not see the energy a soft impact loses:
  both arms lose ~100 % at ε ≥ 10⁻⁴; ICF resolves the bounce (11 rebounds,
  −52 %) only at ε = 10⁻⁵, where the 4096-substep march budget is exhausted
  and the point is marked. A property of the position-only norm of
  Sec. V-E in~\cite{cenic} on a soft bounce, stated as such.

### Wall vs worlds (`figures/scaling_*.pdf`)

_(prose from the final CSVs — pending the rerun)_

### Energy convergence (`figures/ball_energy.pdf`)

_(prose from the final CSVs — pending the rerun)_
