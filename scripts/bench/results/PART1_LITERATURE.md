# Literature review for the Part-1 experiment design

Purpose: ground every metric, scene and reporting convention of Part 1 in
prior work, and derive the refinements from it. Each theme lists the sources
read, what they measure, and the refinement adopted (or explicitly not
adopted, with the reason). Citations are primary where available.

## Theme A — Benchmarking physics engines on contact

**Sources.** Erez, Tassa & Todorov, "Simulation tools for model-based
robotics", ICRA 2015 (DOI 10.1109/ICRA.2015.7139807). Kang & Hwangbo,
SimBenchmark, 2018 (leggedrobotics.github.io/SimBenchmark; code, no paper).
Acosta, Yang & Posa, "Validating robotics simulators on real-world impacts",
RA-L 2022 (arXiv:2110.00541). Choi et al., PNAS 2021 (DOI
10.1073/pnas.1907856118). Castro, Permenter & Han, SAP, T-RO 2022
(arXiv:2110.10107). Le Lidec et al., "Contact models in robotics: a
comparative analysis", T-RO 2024 (arXiv:2304.06372). Howell et al., Dojo
(arXiv:2203.00806). ComFree-Sim (arXiv:2603.12185, 2026). GAUGE
(arXiv:2608.05948, 2026). Isaac Lab (arXiv:2511.04831).

**What they measure.**
* Erez 2015: *consistency violation* — an engine-specific reference at
  h = 1/64 ms; for each larger h the position deviation over short trajectory
  pieces re-initialised from the reference (so chaos does not amplify),
  averaged over 10 pieces; energy and momentum drift on conservative
  variants; grasp stability as the largest h that holds the object; speed as
  "× real time". Plots: x = speed (log), y = accuracy (log, inverted so
  top-right is ideal), one curve per engine parametrised by h; curves
  truncated where the engine goes unstable.
* SimBenchmark: real-time factor vs accuracy on rolling (analytic velocity
  error), 666-sphere drop (time-mean squared pairwise + ground penetration),
  elastic 666 and bouncing (energy error vs E₀), ANYmal momentum/energy;
  primitives only; notes MuJoCo cannot set restitution (only solref).
* Acosta 2022: 550 real cube tosses; error = normalised position + rotation
  error per trajectory, mean ± σ; parameters fitted per simulator; lowest
  vertex height vs time (negative = penetration) per simulator; "decreasing
  the timestep further did not improve prediction" at 1480 Hz.
* SAP 2022: clutter of spheres and boxes in a bin (k = 10¹² N/m, δt = 10 ms,
  40–100+ bodies), iterations and wall per step vs time, contact count,
  dimensionless momentum and complementarity errors vs solver tolerance,
  closed-form resting penetration in the near-rigid regime; spring–cylinder
  convergence with first/second-order reference slopes.
* Le Lidec 2024: primal/dual/complementarity residuals as physical accuracy;
  *self-consistency* ∑‖q_τ − q̄_τ‖Δt vs Δt against a reference at 10⁻⁵ s.
* Dojo: foot penetration vs time step (MuJoCo failure / −28 / −46 mm at
  0.1/0.01/0.001 s). ComFree-Sim: mean ± σ penetration over all contacts
  (MJWarp 1.7 ± 4.9 mm at 2 ms), throughput = single-env steps/s vs
  256–4096 envs. GAUGE: trajectory RMSE normalised by the real data's own
  repeatability; energy loss (E_start − E_end)/E_start, gains flagged as
  unphysical. Isaac Lab: FPS = env-steps / (sim + learn time), log vs envs.

**What has precedent in our current design.**
* Penetration mean/max in mm at the engine's δt (ComFree, Dojo, Acosta's
  vertex-height plots). Normalising by the model's own resting and impact
  depth: not plotted anywhere found; grounded by SAP's analytic φ(δt) and by
  Hunt–Crossley/solref statics — keep, and cite SAP for the idea of the
  model's own predicted depth.
* Energy after 10 s: SimBenchmark (conservative scenes), Erez (drift),
  GAUGE (fractional loss; gain = unphysical). Keep; mark gains explicitly.
* Wall per simulated second = inverse real-time factor (SimBenchmark, Erez);
  speed–accuracy curves parametrised by δt / ε with top-right ideal (Erez,
  SAP). Our artifact-vs-cost figure is that layout.
* Wall vs worlds: ComFree (per-env steps/s vs envs), Isaac Lab (FPS vs
  envs), SimBenchmark N×N, Erez N capsules. Keep per-world and aggregate.
* Ejection fraction: no precedent; nearest are Erez's "curve truncated where
  unstable" and grasp "largest stable h". Present it as a stability
  indicator, not an accuracy metric.

**Refinements adopted from Theme A.**
1. *Self-consistency convergence on clutter (Erez 2015 / Le Lidec 2024)*:
   per arm, a reference at a very small step (fixed arms) or very tight ε
   (error-controlled arms); short pieces (0.1 s) re-initialised from the
   reference state; deviation at the piece end averaged over pieces. This
   measures "converges as δt → 0" on the chaotic scene without chaos
   amplification — the PI-list item, done the way the field does it.
2. *Zero-gravity momentum check (SimBenchmark, Erez)*: two-body collision
   with gravity off; ‖p(t) − p₀‖ per arm — certifies the per-world adaptive
   controller injects no momentum.
3. *Report contact count* alongside clutter timings (SAP), and mean ± σ over
   independent runs (Acosta) — the scaling bench already does the latter.
4. *Energy figure*: show signed ΔE/E₀ with gains marked as unphysical
   (GAUGE), not only |ΔE|.
5. *Matched-accuracy speed comparison* (Erez, SAP): the "cheapest
   artifact-free setting" star is the matched-accuracy reading; keep it and
   cite the convention.

**Not adopted.** SimBenchmark's elastic tests (e = 1) — inapplicable to
compliant, dissipative models; Acosta's real-impact validation — no hardware
data for these scenes in Part 1 (the IRL section is Part 2's).
