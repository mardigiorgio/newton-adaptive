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
(force only at penetration; no collision margin) and a single deterministic
initial condition:

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
* Maximum step δt_max = 10 ms — the control boundary (their clutter runs
  allowed 0.1 s). Error control never steps past a boundary.
* GPU, not CPU: N = 1 is the single-scene setting and sits near the
  launch-latency floor (~1–3 ms per boundary); N = 1024 is the regime robot
  learning runs in. Both are reported.
* Only the point-contact test cases are reproduced; cylinder, gripper and
  Franka use hydroelastic contact and an inverse-dynamics controller.
* Fixed-step arms have no accuracy knob; they appear as reference levels at
  δt.
* ICF's Newton convergence tolerance is fixed at 10⁻⁵ (relative, on the
  scaled residual). The rule ε_tol = max(κ·ε_acc, 10⁻⁸), κ = 10⁻³, was tested
  on hard clutter at ε_acc = 10⁻²…10⁻⁵ and changes the march step count by
  < 10 % (`probe_march_cost.py`), so it is not adopted.

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
\includegraphics[width=\linewidth]{figures/workprecision.pdf}
\caption{Work-precision plots for the soft- and
hard-clutter test cases of~\cite{cenic}, error control on positions ($\mathbf{S}=\mathbf{I}$).
Wall time is normalized to one simulated second; top row a single world
(N=1), bottom row N=1024 parallel worlds on one GPU. Dotted levels are the
fixed-step arms at $\delta t = 10$\,ms and $1$\,ms. Maximum step
$\delta t_{\max} = 10$\,ms in all arms (the control boundary).
$\times$ marks a timeout ($>100$\,s per simulated second). Lower is better.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/speed_bars.pdf}
\caption{Simulator speed comparison, single world:
fixed step at $\delta t=10$\,ms / $1$\,ms and error control at
$\varepsilon_{acc} = 10^{-1}, 10^{-3}, 10^{-5}$. Speed only, not quality;
see Table~I and Fig.~\ref{fig:penetration} for artifacts.}
\end{figure}

\begin{figure}[t]\centering
\includegraphics[width=\linewidth]{figures/ball_energy.pdf}
\caption{Energy conservation: percent change in total
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

All clutter numbers below are from the rerun under the verified contact
budgets (ICF 2048, MuJoCo nconmax 1024, njmax 1024 — ≥ 2× measured demand); no
configuration timed out, exhausted its march budget, or dropped a contact.

### Work-precision (`figures/workprecision.pdf`, `figures/speed_bars.pdf`)

Wall time per simulated second, δt_max = 10 ms; N = 1 rows are 3-trial medians.

| scene, N | arm | ε = 10⁻¹ | 10⁻² | 10⁻³ | 10⁻⁴ | 10⁻⁵ | 10⁻⁶ |
|---|---|---|---|---|---|---|---|
| hard, 1 | ICF error control | 0.23 | 0.23 | 0.23 | 0.28 | 0.42 | 0.89 |
| hard, 1 | MuJoCo error control | 0.14 | 0.34 | 0.12 | 0.26 | 1.61 | 4.94 |
| soft, 1 | ICF error control | 0.13 | 0.13 | 0.16 | 0.18 | 0.29 | 0.61 |
| soft, 1 | MuJoCo error control | 0.10 | 0.10 | 0.11 | 0.19 | 0.40 | 1.06 |
| hard, 1024 | ICF error control | 5.2 | 5.2 | 5.3 | 7.2 | 11.9 | 21.4 |
| hard, 1024 | MuJoCo error control | 0.78 | 5.07 | 3.55 | 6.27 | 12.0 | 29.8 |
| soft, 1024 | ICF error control | 2.3 | 2.3 | 3.3 | 3.8 | 6.2 | 12.7 |
| soft, 1024 | MuJoCo error control | 0.22 | 0.22 | 0.24 | 0.43 | 0.98 | 2.68 |

Fixed-step levels at δt = 10 / 1 ms (s per simulated second): hard N=1 ICF
0.045 / 0.33, MuJoCo 0.11 / 0.34; hard N=1024 ICF 0.54 / 4.3, MuJoCo 0.33 /
1.6; soft N=1 ICF 0.041 / 0.31, MuJoCo 0.048 / 0.26; soft N=1024 ICF 0.85 /
5.0, MuJoCo 0.14 / 0.66 (full ladder in `tables/part1_table1.md`).

What the figure supports:
* Every error-controlled configuration runs faster than real time for a
  single world on both scenes, down to ε = 10⁻⁶ — no timeouts.
* Hard clutter: ICF error control is flat at 0.23 s/sim-s for ε ≥ 10⁻³
  (one accepted 10 ms step per boundary) and grows only to 0.89 s at 10⁻⁶;
  MuJoCo error control is cheaper at loose accuracy but 4–6× more expensive
  at ε ≤ 10⁻⁵ (1.6 s and 4.9 s). At N = 1024 the crossover sits at 10⁻⁵.
* Soft clutter: MuJoCo error control is cheaper at every ε for N = 1024
  (its per-step cost is lower); at N = 1 the two are within 2× everywhere.
* MuJoCo error control is non-monotone in ε on hard clutter (10⁻³ cheaper
  than 10⁻² at both N; 3-trial medians at N = 1) — a property of its
  controller on this scene, reported as measured.
* Why the crossover (`tables/march_cost.md`, `probe_march_cost.py`, N = 1):
  on hard clutter ICF error control takes 100 / 100 / 100 / 171 / 342 / 1027
  march iterations per simulated second at ε = 10⁻¹ … 10⁻⁶ (one accepted
  10 ms step per boundary down to 10⁻³) at 1.6–2.5 ms per iteration (three
  ICF solves + two geometry queries, kernel-launch-bound for one world);
  MuJoCo error control takes 100 / 118 / 127 / 239 / 1220 / 3812 at 1.1–1.5
  ms — cheaper per iteration, but 3.7× more iterations at 10⁻⁶ because its
  local error estimate on the stiff contact is larger. The paper's CPU
  implementation reports ~500 steps at 10⁻³ with δt_max = 0.1 s (Table III).
* Timing floor: fixed ICF at δt = 10 ms costs 0.38 ms per boundary at 64
  worlds (0.045 s per simulated second at N = 1). On an idle GPU three
  repeats agree to < 1.5× at that scale (`verify_part1_penetration.py`,
  passed on the final scene and budgets); one repeat taken while another
  job was starting read 1.9× higher — sub-millisecond wall numbers are only
  reproducible on an otherwise idle GPU, which is how every sweep here ran.

### Penetration and ejections vs wall time (`figures/penetration_*.pdf`)

64 worlds, 200 boundaries, model-read geometry (`verify_part1_penetration.py`).

| scene | arm | setting | mean penetration | max penetration | ejected |
|---|---|---|---|---|---|
| hard | MuJoCo fixed | δt = 10 → 1 ms | 0.69 → 0.19 mm | 13.6 → 4.9 mm | 0 |
| hard | MuJoCo error control | ε = 10⁻¹ → 10⁻⁴ | 0.18–0.26 mm | 3.8–7.5 mm | 0 |
| hard | ICF fixed | δt = 10 → 1 ms | **0** | **0** | 0 |
| hard | ICF error control | ε = 10⁻¹ → 10⁻⁴ | **0** | **0** | 0 |
| soft | MuJoCo fixed | δt = 10 → 1 ms | 1.1 → 0.89 mm | 34 → 23 mm | 0 |
| soft | MuJoCo error control | ε = 10⁻¹ → 10⁻⁴ | 0.88–0.89 mm | 22–24 mm | 0 |
| soft | ICF fixed | δt = 10 → 1 ms | 4–7 µm | 1.7–2.2 mm | 0 |
| soft | ICF error control | ε = 10⁻¹ → 10⁻⁴ | 5–6 µm | 1.9–2.2 mm | 0 |

* On hard clutter ICF resolves contact to exactly zero ground penetration at
  every δt and every ε — including fixed 10 ms steps — at 0.4–5 ms per
  boundary for 64 worlds; MuJoCo never gets below ~0.2 mm mean / ~5 mm max
  at any setting, because its compliance is in the model, not the step.
* On soft clutter (k = 10³ N/m, 65 g objects) MuJoCo's max penetration is
  about the object radius (22–34 mm; the pile's bottom layer) at every δt
  and ε; ICF stays at ~2 mm max and ~5 µm mean.
* No body left the bin in any configuration on either scene.
* The earlier "fixed ICF launches spheres over the rim" and "MuJoCo tunnels
  through the floor" observations were artifacts of the starved contact
  budgets and do not survive re-measurement.

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

Wall per 10 ms boundary [ms], median of per-run medians over 3 independent
runs (100 timed boundaries after 20 warm-up, t = 0.2–1.2 s — the impact
phase); the band in the figure is the spread across runs; no exhaustion.

**Hard clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| MuJoCo fixed, δt = 10 ms | 1.7 | 1.9 | 2.1 | 2.6 | 3.4 | 5.7 | 9.8 | 15.8 |
| MuJoCo error control, ε = 10⁻³ | 2.4 | 5.8 | 9.6 | 14.9 | 21.2 | 38.2 | 69 | 105 |
| ICF fixed, δt = 10 ms | 0.38 | 0.44 | 1.7 | 2 | 1.4 | 2.4 | 4.6 | 9 |
| ICF error control, ε = 10⁻³ | 5.4 | 8.3 | 13.6 | 25.2 | 48.1 | 94.5 | 188 | 373 |

**Soft clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| MuJoCo fixed, δt = 10 ms | 0.45 | 0.46 | 0.52 | 0.67 | 0.96 | 1.7 | 2.9 | 4.9 |
| MuJoCo error control, ε = 10⁻³ | 1.4 | 1.4 | 1.5 | 1.7 | 2.2 | 3.8 | 6.5 | 12 |
| ICF fixed, δt = 10 ms | 1.2 | 1.9 | 3.1 | 5.8 | 11 | 21.5 | 42.9 | 84.8 |
| ICF error control, ε = 10⁻³ | 3 | 4.5 | 7.4 | 13.4 | 24.7 | 48 | 94.9 | 187 |

* All four arms scale sub-linearly to 2¹³ worlds; with intact contacts ICF
  error control is monotone (hard: 5.4 → 373 ms; the non-monotone single-run
  sweep is superseded) and its run-to-run spread is < 1 % at every N.
* Hard clutter at 2¹³ worlds: ICF fixed 10 ms 9.0 ms per boundary, MuJoCo
  fixed 15.8 ms, ICF error control 373 ms,
  MuJoCo error control 105 ms.
* MuJoCo error control's run-to-run spread is wide at small N (e.g. 2.0–5.2 ms
  at 64 worlds) and narrows with N; the ICF arms are reproducible throughout.
* The MuJoCo arms at 2¹³ worlds run under njmax = 1024 (capacity; demand
  ~200 rows per world) — the 4096-row Jacobian allocation exhausted the 32 GB
  GPU at that world count.

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
