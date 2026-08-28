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

## Results

_(filled from the final CSVs — see `figures/` and `tables/`)_
