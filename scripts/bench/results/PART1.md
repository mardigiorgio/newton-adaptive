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
MuJoCo. Each scene therefore carries `mujoco_solref`. The clutters use
the reference format (τ, ζ): τ = 2.4 ms at k = 10⁵ N/m (calibrated so a
resting sphere sinks by the model's m·g/k) and 24 ms at k = 10³ N/m (scaled
as 1/√k; the realized stiffness there is 1.37× the model — the reference
format's impedance is not exactly ∝ 1/τ² — and is reported as such), ζ = 1,
and MuJoCo's refsafe clamp τ ≥ 2δt, so at δt > τ/2 the contact is softer by
MuJoCo's own design. The reference format admits no zero damping ratio
(ζ = 0 diverges), so the ball — which asks for zero dissipation — uses
MuJoCo's direct format (−stiffness, −damping) = (−2.24·10³, 0), calibrated
to the same resting depth: an undamped soft constraint that conserves the
ball's energy to 0.2 % at δt ≤ 1 ms. The direct format has no clamp and is
stable only while ω·δt ≲ 2 (ω = √(k/m_eff)): at k = 10⁵ it launches bodies
for δt ≥ 5 ms, which is why the clutters keep the reference format
(`tables/mujoco_stiffness_probe.md`).

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
Energy is read at the last apex. ICF converges at first order; MuJoCo's
undamped direct-format constraint keeps the energy within 0.03\,\% at
$\delta t \le 1$\,ms and loses 7\,\% at 10\,ms.}
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


\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/stiffness_sweep.pdf}
\caption{Realized contact stiffness: resting penetration of one 65\,g sphere
over the model's $m g / k$ as the requested $k$ rises. ICF realizes the model
up to $10^7$\,N/m at $\delta t = 10$\,ms, 1\,ms and under error control at
$\varepsilon_{acc} = 10^{-3}$ alike; MuJoCo's soft constraint is clamped to
$\tau \ge 2\delta t$ and floors at $k \approx 10^3$\,N/m (10\,ms) and
$10^5$\,N/m (1\,ms), and its error control at $\varepsilon_{acc} = 10^{-3}$
does not recover the clamp.}
\label{fig:stiffness}
\end{figure}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/consistency.pdf}
\caption{Measured error versus cost on the clutter scenes: position
deviation from a $\delta t = 0.1$\,ms reference of the same model after
0.1\,s windows restarted from the reference ($\ell_\infty$ over bodies,
averaged over 20 windows), against wall time per simulated second (8
scenes). Every point states its $\delta t$ or
$\varepsilon_{acc}$.}
\label{fig:consistency}
\end{figure*}

\begin{figure*}[t]\centering
\includegraphics[width=\textwidth]{figures/actuated.pdf}
\caption{Actuated stiff contact: a PD gantry with gain $K_p$ ($K_d = 2\sqrt{K_p m}$)
pushes a 1\,kg box from the side at 300\,mm/s, $k = 10^5$\,N/m, $\mu = 0.5$,
targets held at the 100\,Hz boundary. Left to right: box lift during the
push (ICF's box never leaves its resting depth; drawn at the floor), box
pitch rate, fingertip penetration into the box face (MuJoCo's reading at
$K_p \le 10^3$ is the overlap with a lifted, pitched box), tip--box relative
velocity in the cruise; fixed step at $\delta t = 1$\,ms and error control
at $\varepsilon_{acc} = 10^{-3}$. MuJoCo is unstable at $K_p = 10^5$ for
$\delta t \ge 5$\,ms and at $10^6$ for $\delta t \ge 2$\,ms; ICF is stable
in every cell.}
\label{fig:actuated}
\end{figure*}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/actuated_chatter.pdf}
\caption{Actuated push at $K_p = 10^5$\,N/m: tip--box relative velocity in
the cruise against cost. With targets held at the 100\,Hz boundary each
3\,mm target step is a 300\,N kick on the 0.1\,kg fingertip; the chatter
it excites appears as the step is refined and a 10\,ms step integrates it
away entirely. Every point states its $\delta t$ or $\varepsilon_{acc}$.}
\label{fig:actuated_chatter}
\end{figure}



\begin{figure*}[!t]\centering
\includegraphics[width=\textwidth]{figures/story_step.pdf}
\caption{At the step a learner uses, fixed stepping makes artifacts and
error control removes them. (a) Resting penetration of one 65\,g sphere
over the model's $mg/k$ as the requested stiffness rises: ICF realizes the
model at $\delta t = 10$\,ms and under error control alike; MuJoCo's soft
constraint is clamped to $\tau \ge 2\delta t$ and floors at $k \approx
10^3$\,N/m at 10\,ms, which its position-only error control does not
recover. (b) Hard clutter, 64 scenes: maximum ground penetration over the
model's impact depth $v\sqrt{m/k}$ at the learner's step (10\,ms), under
error control at $\varepsilon_{acc} = 10^{-2}$, and at the cheapest
artifact-free fixed step of each solver; fixed ICF at 10\,ms passes through
the ground and ejects 1.6\,\% of the bodies. (c) A PD gantry pushing a
1\,kg box from the side ($k = 10^5$\,N/m, $\mu = 0.5$, targets held at
100\,Hz): MuJoCo's box lifts off the table by the millimetres shown and
its explicit joint gain diverges ($\times$) at $K_p \ge 10^5$\,N/m for
$\delta t \ge 5$\,ms; ICF is stable in every cell and never lifts the box.
(d) Tip--box relative velocity in the cruise at $K_p = 10^5$\,N/m: the
chatter a held target excites appears as the step is refined and is
integrated away entirely at 10\,ms.}
\label{fig:story_step}
\end{figure*}

\begin{figure*}[!t]\centering
\includegraphics[width=\textwidth]{figures/story_cost.pdf}
\caption{Being artifact-free costs fixed stepping every step and error
control only the impacts. (a) Hard clutter, 64 scenes: wall time per
simulated second at the learner's coarse setting (hatched; an artifact by
the criterion of Fig.~\ref{fig:story_step}b) and at the cheapest
artifact-free setting of each arm (solid) -- the matched-accuracy cost.
(b) Cumulative wall time along a 5\,s drop: fixed step pays the same at
every step; error control pays during the impacts (shaded) and coasts at
$\delta t_{\max}$ once the pile settles.}
\label{fig:story_cost}
\end{figure*}

\begin{figure*}[!t]\centering
\includegraphics[width=\textwidth]{figures/story_convergence.pdf}
\caption{Both solvers converge. (a) Bouncing ball with zero dissipation,
energy read at the last apex after 10\,s: ICF converges at first order;
MuJoCo's undamped direct-format constraint keeps the energy within
0.03\,\% at $\delta t \le 1$\,ms. (b) Soft clutter, measured error against
cost: position deviation from a $\delta t = 0.1$\,ms reference of the same
model after 0.1\,s windows restarted from the reference ($\ell_\infty$ over
bodies, mean over 20 windows, 8 scenes); hollow markers are the reference
restarted against itself, the instrument's floor. Both fixed arms converge
at first order; ICF error control gives about half the deviation of fixed
ICF at the same cost; MuJoCo error control lands on its own fixed-step
line.}
\label{fig:story_convergence}
\end{figure*}

## The actuated test case (`scripts/scenes/actuated_press.py`, `part1_actuated.py`)

The clutter cases have no actuator. The regime in which MuJoCo error control
took ~10× longer per iteration than ICF error control in training — a stiff
PD-driven arm on objects — is the "Franka with box" case of~\cite{cenic}
(high-gain controller → short time scales). Part 1 reproduces it with the
smallest scene that has the ingredients (`PART1_LITERATURE.md`, Theme E): a
PD-driven prismatic gantry (x, z) carrying a 0.1 kg fingertip (r = 1 cm)
pushes a 1 kg, 10 cm box across the table from the side on a trapezoidal
profile (300 mm/s, 0.1 s ramps, 0.3 m), k = 10⁵ N/m, μ = 0.5, the gain K_p
swept from 10² to 10⁶ N/m with K_d = 2√(K_p m). Per world, from the state
only: tip penetration into the box face against the quasi-static push depth
μ·m·g/k, box vertical velocity RMS during the push (chatter), tip–box
relative velocity RMS in the cruise (a steady push has none), tracking RMS,
final box displacement against the commanded 0.28 m, instability (non-finite
or |v| > 10 m/s), and wall time. Fixed δt ∈ {10, 5, 2, 1} ms and
ε ∈ {10⁻¹ … 10⁻⁴}, one world (the scene is deterministic).

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

Energy is read at the last apex of the 10 s run (the flight energy of the
final flight; a ball that never leaves the ground is read at the end), so
neither a mid-impact instant nor an earlier flight enters the number.

* Fixed ICF converges at first order once the impact is resolved (from
  δt = 0.1 ms each halving of δt halves the loss: −30 %, −16 %, −6.2 %,
  −3.2 % at 100, 50, 20, 10 µs) and rebounds 11 times in 10 s at δt ≤
  0.1 ms, the count~\cite{cenic} states for this scene. At δt ≥ 1 ms the
  ball comes to rest within the run.
* MuJoCo's undamped direct-format constraint keeps the ball's energy to
  within 0.03 % at every δt ≤ 1 ms (−0.02 % at 1 ms, −0.004 % at 0.5 ms,
  ≤ 0.002 % below) with 10 rebounds; at coarse steps it loses 7 % (10 and
  5 ms) and gains 0.8 % at 2 ms. On a conservative single impact the
  implicit soft constraint is the better integrator, and the figure says
  so.
* Error control on positions does not see the energy a soft impact loses:
  ICF resolves the bounce (11 rebounds) only at ε ≤ 10⁻⁵, where the
  4096-substep march budget exhausts and the point is marked. A property of
  the position-only norm of Sec. V-E in~\cite{cenic} on a soft bounce,
  stated as such.
* MuJoCo error control at ε ≥ 10⁻² stays at δt_max = 10 ms (+0.8 %, 9
  rebounds). At ε ≤ 10⁻³ it **gains** energy, about 5 % per impact (+57 %
  after ten bounces at ε = 10⁻³; +22 %, +13 %, +4.0 % at 10⁻⁴, 10⁻⁵,
  10⁻⁶), while fixed MuJoCo at the same steps conserves: the gain is in our
  adaptive wrapper's step changes through the undamped constraint, not in
  MuJoCo's step. Open defect of the `SolverMuJoCoAdaptive` arm, reported as
  measured.

### Wall vs worlds (`figures/scaling_per_world_*.pdf`, `figures/scaling_*.pdf`)

Median of three independent runs per point (spread as the band), fixed step
at δt = 10 ms, error control at ε = 10⁻³, 2 s timed window; no exhaustion;
tables in `tables/results_tables.md`.

* Cost per world falls with batch size until the GPU saturates. At 2¹³
  worlds on hard clutter, per 100 ms boundary: MuJoCo fixed 11 µs per
  world, MuJoCo error control 270 µs, fixed ICF 620 µs, ICF error control
  1.5 ms; fixed ICF and ICF error control saturate from 2¹⁰ worlds.
* Under point contact our ICF step costs ~55× MuJoCo's per world at 2¹³
  worlds (fixed step) and ~6× under error control. Not the Newton tolerance
  (10⁻⁵…10⁻⁸ changes wall by < 5 %, `tables/newton_tolerance_probe.md`): it
  is the cost of resolving stiff point contact to the model's compliance
  with a convex Newton solve, and a batch pays for its slowest world.
* Reproducibility: ICF's run-to-run spread is < 1 % at every N; MuJoCo error
  control's is wide at small N and narrows with N.

### Realized stiffness (`figures/stiffness_sweep.pdf`)

One 65 g sphere at rest, k requested from 10³ to 10⁸ N/m, penetration over
the model's m·g/k (1 = the model), the axis of Fig. 18 in the ICF paper.

* ICF realizes the requested stiffness up to 10⁷ N/m (ratio 0.99) at
  δt = 10 ms, at 1 ms and under ε = 10⁻³ alike — the curves coincide — and
  reads 0.87 at 10⁸, the float32 floor of the Newton solve. The stiffness
  is a property of the model, not of the step.
* MuJoCo realizes k only while τ(k) ≥ 2δt: up to ~10³ N/m at δt = 10 ms and
  10⁵ N/m at 1 ms; beyond, penetration floors at the clamp (57× the model
  at k = 10⁵ and 10 ms) and grows ∝ k. Error control at ε = 10⁻³ does not
  recover it (17× at k = 10⁵): the controller sizes steps by position error,
  the clamped contact is self-consistent at those steps, and the softening
  is invisible to the norm. At ε = 10⁻⁵ it recovers k = 10⁵ (1.2×) and then
  floors where fixed 1 ms does — tighter tolerance buys back stiffness only
  as far as the steps it drives the solver to.

### Determinism (`tables/determinism_probe.md`)

Two identical arms from the same state: the ball reproduces bit for bit on
every arm; on clutter neither contact solver does — GPU reduction order in
the contact solve differs run to run and the pile amplifies it, to
millimetres within 0.3 s on soft clutter for ICF (micrometres for MuJoCo)
and to centimetres within 0.5 s on hard clutter for both. Two training runs
with the same seed are therefore not the same run under clutter contact on
either backend, and any comparison against a reference trajectory has this
noise as its floor: the consistency bench's restart oracle passes exactly on
the ball for both backends and on soft clutter for MuJoCo (0.7 µm), and
fails by the solvers' own noise on soft clutter for ICF (0.19 mm) and on
hard clutter for both (8–9 mm).

### Momentum (`tables/momentum_probe.md`)

Two spheres collide head-on with gravity off: every arm conserves the pair's
linear momentum to ≤ 10⁻⁵ (ICF ≤ 10⁻⁷, exactly under error control) — the
per-world step controller injects none.

### Self-consistency (`figures/consistency.pdf`)

Position deviation from a δt = 0.1 ms reference of the same model after
0.1 s windows restarted from the reference — the ℓ∞ over bodies the error
controller itself uses, averaged over 20 windows of the 2 s drop, 8 scenes
— against wall time per simulated second, every point labeled. Each arm is
compared with its own model's reference, so the figure ranks convergence
and cost, not fidelity to the physical model (that is
`figures/stiffness_sweep.pdf` and the artifact rows). The δt = 0.1 ms row
(hollow marker, dotted line) is the reference restarted against itself:
the instrument's floor, the solvers' run-to-run noise amplified over the
window (`tables/determinism_probe.md`) — 11 µm (ICF) and 0.2 µm (MuJoCo)
on soft clutter, 0.62 mm and 0.28 mm on hard clutter. A deviation within a
few × of the floor is not a step-size measurement. Idle GPU, one run.

* Soft clutter (not chaotic, floor ≤ 11 µm): both fixed-step solvers
  converge at first order (ICF 3.7 mm at 10 ms → 0.12 mm at 0.5 ms, MuJoCo
  2.4 → 0.08 mm), and the requested ε bounds the measured deviation
  (0.39 mm at ε = 10⁻³, 41 µm at 10⁻⁵). ICF error control gives about 2×
  less deviation than fixed ICF at the same cost across the range
  (ε = 10⁻³: 0.39 mm at 0.45 s vs 0.73 mm at 0.58 s for fixed 2 ms;
  ε = 10⁻⁴: 0.14 mm at 1.4 s vs 0.29 mm at 1.0 s for fixed 1 ms; ε = 10⁻⁵:
  41 µm at 3.2 s vs 0.12 mm at 1.9 s for fixed 0.5 ms). MuJoCo error
  control lands on its own fixed-step line (ε = 10⁻⁴: 96 µm at 0.54 s,
  between fixed 1 ms at 0.42 s and 0.5 ms at 0.84 s) — a wash. Error
  control pays for the solver whose per-step cost is high and whose error
  falls fastest with the step, not for the cheap one.
* Hard clutter (chaotic, floor 0.3–0.6 mm): the coarse end is a step-size
  measurement — ICF 20 mm at 10 ms → 4.6 mm at 1 ms → 2.7 mm at 0.5 ms,
  MuJoCo 8.3 → 2.3 → 1.2 mm, about O(δt^0.7) for both, MuJoCo at 2–3×
  lower cost for the same δt; and the requested ε is not the measured
  deviation there (5.6 mm at ε = 10⁻³, amplified by the pile). The tight
  end sits within 2–4× of the floor (ICF ε = 10⁻⁵: 1.1 mm at 9.1 s; MuJoCo
  ε = 10⁻⁵: 0.9 mm at 6.3 s, no better than its ε = 10⁻⁴) and ranks
  nothing. In the middle, ICF error control at ε = 10⁻³ (5.6 mm, 2.6 s)
  matches fixed ICF between 1 and 2 ms at the same cost, and MuJoCo error
  control at ε = 10⁻³ (2.5 mm, 1.4 s) sits right of fixed 1 ms (2.3 mm,
  0.9 s): on an impact-dominated window neither controller beats its
  fixed-step line, since it has to take small steps everywhere; the saving
  is in the settled phase (`figures/realtime_trace_n64.pdf`).

### Actuated push (`figures/actuated.pdf`, `figures/actuated_chatter.pdf`; trace in `tables/actuated_trace.md`)

One world, k = 10⁵ N/m, μ = 0.5, 300 mm/s; targets held at the 100 Hz
boundary as a policy's would be. Data in `part1_actuated.csv` (80 cells:
5 gains × 2 backends × 4 fixed steps and 4 tolerances).

* **Stability.** ICF is stable in all 40 cells. MuJoCo blows up in 6:
  K_p = 10⁵ at δt = 10 and 5 ms, K_p = 10⁶ at 10, 5 and 2 ms and under
  error control at ε = 10⁻¹ — the joint gain is explicit in MuJoCo's step
  (√(K_p/m)·δt ≳ 2 diverges: 10⁵ needs δt ≤ 2 ms, 10⁶ δt ≤ 1 ms). Error
  control at ε = 10⁻¹ on K_p = 10⁵ survives but throws the box 1.6 m
  (lift 19 cm).
* **The pushed box.** ICF slides it flat at every gain, step and tolerance:
  lift ≤ 0 (it stays at its 25 µm resting depth, m·g/(4k) exactly), pitch
  rate ≤ 0.02 rad/s, vertical velocity ≤ 0.2 mm/s, displacement 0.280 m as
  commanded for K_p ≥ 10³ (0.235 m at K_p = 10², where the 100 N/m
  controller cannot exceed friction until it lags 5 cm). MuJoCo's box rocks
  and hops in every stable cell — lift 2–21 mm and pitch rate 0.7–3.4 rad/s
  at K_p ≤ 10⁴, 4–9 mm at 10⁵, 10–58 mm at 10⁶, with the tip climbing onto
  the box at K_p = 10² (δt = 5, 2 ms) — and error control does not remove
  it (ε = 10⁻⁴: 1.7–10 mm). A 10 cm cube pushed at mid-height with μ = 0.5
  slides flat (tipping needs the push above its top face); the hop is a
  sliding-contact behaviour of the soft constraint (`tables/actuated_trace.md`:
  with the push frozen neither backend moves the box).
* **Tip penetration.** ICF ≤ 0.25 mm at every cell against a quasi-static
  push depth of 49 µm (the maximum is the ramming impact, 0.3 m/s into a
  1 kg box). MuJoCo ≤ 0.6 mm where its box stays near flat (K_p ≥ 10⁴,
  δt ≤ 2 ms); at K_p ≤ 10³ the reading (up to 11 mm) is the horizontal
  overlap of the tip with a pitched, lifted box, not a compliance.
* **Resolved chatter under held targets.** At K_p ≥ 10⁵ each 3 mm target
  step is a ≥ 300 N kick on the 0.1 kg tip. Both solvers resolve the
  resulting tip–box chatter as the step shrinks — ICF 0 at 10 ms, 0.08 m/s
  RMS at 5 ms, 0.20 at 2 ms, 0.27 at 1 ms for K_p = 10⁵ (0.36 at 10⁶);
  MuJoCo 0.36 at 1 ms — and the 10 ms step integrates it away entirely:
  a policy trained at 10 ms never sees the contact its own gains excite.
  ICF error control at ε ≤ 10⁻³ sits at the 5 ms level (0.08) and resolves
  it only at ε = 10⁻⁴ (0.26): a position norm does not see the velocity
  chatter of a light tip.
* **Cost.** One world, 0.06–0.5 s per simulated second in every cell — the
  launch-latency floor; this scene does not reproduce the 2048-environment,
  7-DoF regime in which MuJoCo error control took ~10× longer per iteration
  than ICF error control in training, and makes no claim about it.
