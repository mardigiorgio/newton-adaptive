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

## Theme B — Compliant and convex contact formulations, and how they are validated

**Sources.** Todorov, ICRA 2014 + MuJoCo docs (computation, modeling, XML
reference, MJX/MJWarp). Castro, Permenter & Han, SAP, T-RO 2023
(arXiv:2110.10107). Castro, Han & Masterjohn, ICF, T-RO 2025
(arXiv:2312.03908). Kurtz & Castro, CENIC (arXiv:2511.08771). Masterjohn et
al., hydroelastic, RA-L 2022. Howell et al., Dojo (arXiv:2203.00806).
Anitescu 2006; Hunt & Crossley 1975 (full texts not retrieved).

**What the formulations are and what their authors validated.**
* MuJoCo: velocity-stepping strictly convex relaxation; per constraint
  a_c + d(b v + k r) = (1 − d) a_u with b = 2/(d_width·τ), k =
  d(r)/(d_width²·τ²·ζ²) — stiffness *per unit effective mass* ("smart
  spring-dampers that scale with inertia"); resting penetration r =
  a_u(1 − d)τ²ζ² (mass-independent); `refsafe` enforces τ ≥ 2δt; slip during
  stiction is by design; direct format (−stiffness, −damping) allows damping
  = 0. Todorov 2014 validates forward/inverse consistency and speed only and
  writes that systematic validation "remains to be done".
* SAP: linear spring + dissipation time scale, near-rigid regime ties
  stiffness to δt (k ∝ 1/δt²), predicted resting depth β²gδt²/(4π²);
  validated by spring–cylinder convergence order, clutter iterations/wall,
  momentum and complementarity residuals, stiction slip.
* ICF: f_n = k(−φ)₊(1 − d v_n)₊ with Hunt–Crossley d; proves the gliding
  offset μ(δt + τ_d)‖v_t‖ of SAP/MuJoCo-type models does not vanish as
  δt → 0 while the lagged scheme is consistent; validated by conveyor
  stick-slip, box convergence vs δt (reference at 0.2 ms), sliding-to-rolling
  sphere, clutter mean rest penetration vs stiffness over eight decades
  (Fig. 18), grasp hold times. No MuJoCo runs.
* CENIC: work-precision, contact-force smoothness, ball energy vs δt, step
  counts vs controller gain, dishrack Table I; MuJoCo comparison is speed
  only and at default parameters (Spot: 3.1 cm vs 6.4 mm penetration).
* Dojo: Atlas foot penetration vs rate (MuJoCo −28/−46 mm at 10/1 ms,
  parameters unstated); stability 20–500 Hz.

**Refinements adopted from Theme B.**
1. *MuJoCo configuration, stated with the equations.* Our per-scene solref
   is calibrated so a resting sphere sinks by m·g/k (Theme B gives the map
   τ = √(m_eff/k)/ζ up to MuJoCo's diagonal m_eff approximation and the
   impedance d(r); the calibration absorbs both). The document now states
   the representability limit k_max ≈ m_eff/(4δt²) from `refsafe` — the
   reason MuJoCo is soft at coarse δt — as MuJoCo's own design, cited.
2. *Undamped contact in MuJoCo* is expressible in the direct solref format
   (damping = 0); its energy behaviour is characterized nowhere, so it is
   measured here (`tables/mujoco_stiffness_probe.md`, undamped section)
   before any "MuJoCo cannot represent a conservative contact" sentence.
3. *Stiffness sweep (ICF Fig. 18 protocol)*: resting penetration relative to
   m·g/k vs requested k over 10³–10⁸ N/m at fixed δt and under error
   control, both backends — the stiff-contact regime the paper is about,
   measured the way its own contact model was validated.
4. *Cite, don't re-derive*: MuJoCo's rest penetration and slip-by-design
   (docs); the δt-capped stiffness (refsafe); the non-vanishing gliding of
   convex relaxations vs ICF's consistency (ICF); error control removing
   thin-object artifacts more cheaply than shrinking δt (CENIC Table I).
   What no source provides — a parameter-matched MuJoCo-vs-ICF accuracy
   comparison — is this section's contribution.

**Not adopted.** ICF's conveyor-belt gliding test (Part 1 has no sliding
scene; it belongs with the slide task in Part 2). Hydroelastic contact (not
available in either arm).

## Theme D — GPU simulators, RL throughput reporting, and policies exploiting simulator artifacts

**Sources.** Makoviychuk et al., Isaac Gym, NeurIPS D&B 2021
(arXiv:2108.10470). Rudin et al., CoRL 2021 (arXiv:2109.11978). Freeman et
al., Brax (arXiv:2106.13281). MJX docs; MuJoCo Playground (arXiv:2502.08844);
MuJoCo Warp nightly; Isaac Lab benchmark page and paper (arXiv:2511.04831);
ManiSkill3 (arXiv:2410.00425); Genesis and its critique (Tao; issue #181).
Lehman et al. 2018 (arXiv:1803.03453) incl. Cheney et al. 2013; Krakovna
2018; Baker et al. 2019; Narang et al., Factory, RSS 2022; Handa et al.,
DeXtreme 2022; Hwangbo 2019 / Lee 2020; Zhao 2020 survey; Du et al.,
Embedded IPC (arXiv:2409.16385); Tallec et al., ICML 2019; MPGOS.

**What they establish.**
* Throughput is reported as aggregate env-steps per second (control steps)
  on log axes vs number of environments, one named GPU, with the saturation
  knee (Isaac Gym ~8192, Rudin ~4000, MJX degrading with contact count);
  Isaac Lab separates step-only / +inference / +train tiers; ManiSkill3
  matches solver settings across simulators before timing; the Genesis
  episode shows what happens when substeps, self-collision and idle steps
  are not stated (a claimed 43M FPS became 0.29M under matched settings).
* Policies exploit integrator and time-step artifacts: Sims 1994 creatures
  harvesting Euler error; Cheney 2013 creatures shrinking to obtain a larger
  stability-heuristic δt and penetrating the ground for "free" energy (fix:
  contact damping and a corrected δt rule); Factory: "model-free RL agents
  reveal and exploit any inaccuracies or instabilities in the simulator";
  DeXtreme: real-to-sim replay produced interpenetrations; Embedded IPC:
  Isaac Sim penetration artifacts on the dish rack; CENIC Table I: fixed
  step needs 1 ms to remove dishrack artifacts.
* No batched robot simulator steps worlds adaptively; error control exists
  on CPU (Simbody, Drake, CENIC); per-thread adaptive RK on GPU exists only
  for independent ODEs (MPGOS). No controlled experiment varies only the
  stepping scheme with scene, reward and seeds fixed.

**Refinements adopted from Theme D.**
1. *Throughput figure conventions*: per-world cost and aggregate on log axes
   vs N with the knee named; δt, substeps, tolerance, self-collision and
   actuation stated in the caption; adaptive points labeled with ε and the
   mean inner steps per boundary (recorded in the benches).
2. *Both a wall-matched and an accuracy-matched fixed arm* in Part 2, as
   ManiSkill3 (matched settings) and CENIC (matched accuracy) do.
3. *The controller cannot be gamed*: error-driven with a hard δt_inner_min
   floor — Cheney 2013's adaptive-δt heuristic was optimized into an exploit;
   one sentence in the paper.
4. *Citations for the motivation*: Cheney/Lehman, Factory, DeXtreme,
   Embedded IPC, CENIC Table I; and the explicit statement of the gap the
   killer experiment fills.
