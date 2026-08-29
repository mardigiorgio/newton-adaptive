# Part 1 — every constant, where it comes from, and whether the comparison is symmetric (audit 2026-08-29 evening)

Source key: **P** = the CENIC paper (Kurtz & Castro), quoted; **MJ** = MuJoCo's own default (as shipped, via Newton's SolverMuJoCo); **ICF** = icf_warp default; **ours** = set by us and stated; **assumed** = not stated by the paper, our choice, to confirm with the authors. Code: `scripts/bench/four_arms.py`, `scripts/scenes/cenic_scenes.py`, `scripts/scenes/actuated_press.py`, the benches under `scripts/bench/benchmarks/`.

## 1. Error-controlled stepping (both EC arms — identical by construction)

| constant | value | source | note |
|---|---|---|---|
| error estimate | step doubling: full δt vs two δt/2, half-step result kept | P (Sec. V-A) | both arms; three solves per attempt |
| error norm | ‖q − q̂‖∞ on positions, S = I, absolute (rtol = 0) | P (Sec. V-E) | MuJoCo `NEWTON_ADAPTIVE_RTOL=0` (parity a24e1071); ICF `rtol=0.0` |
| accept | e ≤ ε_acc | P (Alg. 1) | both |
| step update | δt̂ = k_Safe·δt·(ε/e)^{1/2}; deadband (k_Low, k_High) = (0.9, 1.2); δt ← min(δt̂, k_MaxGrow·δt, δt_max) | P: k_Safe 0.9, k_Low 0.9, k_High 1.2, k_MaxGrow 5.0 | both; p = 2 (second-order estimate of a first-order scheme) |
| minimum shrink | 0.1·δt per rejection | ours (Drake's `CalcAdjustedStepSize`) | the paper's Alg. 1 has no explicit floor on the shrink factor; both arms |
| first attempt | k_Init·δt_max, k_Init = 0.1 | P | both (`K_INIT`) |
| δt_max | 0.1 s (clutter), 10 ms (ball, actuated) | P: 0.1 s on clutter (Table III); ours for the ball/actuated (= the control boundary) | stated in every caption |
| δt_min (floor) | 1e-6 s | ours (`DT_INNER_MIN`) | the paper has no floor; at ε ≥ 1e-6 the controller never approaches it (steps ≥ ~1e-4 s); a floor accept would be an unreported over-tolerance step — none observed (no divergence latches) but not counted explicitly (follow-up: count floor accepts) |
| march budget | 65536 attempts per boundary (work-precision); 4096 (penetration, scaling, consistency, trace, ball, actuated) | ours (safety cap) | a run that exhausts it is marked `budget-exhausted` and never plotted as a result |
| divergence | non-finite state → world latched diverged, clock snapped to the boundary | ours (parity) | both; counted in `exhausted_frac`/status |
| geometry cadence | narrow-phase at x^n and at x^{n+1/2} every attempt (two queries) | P (Sec. V-A) | MuJoCo arm via the attached Newton pipeline (3d672820); ICF native |
| inner solve tolerance under EC | ICF: ε_tol = max(κ ε_acc, 1e-8), κ = 1e-3 | P (eq. 34, κ = 0.001) | **asymmetry:** MuJoCo keeps its own solver tolerance 1e-8 in both modes (its criterion is MuJoCo's scaled-gradient test, not eq. 32/33); measured effect of ε_tol on the ICF march < 10 %, on wall < 5 % (`newton_tolerance_probe.md`) |

## 2. Fixed stepping (both fixed arms)

| constant | value | source |
|---|---|---|
| δt | 10, 5, 2, 1 ms (work-precision levels, penetration, actuated); 10 ms (scaling); 10 ms → 10 µs ladder (ball); 0.5 and 0.1 ms added (consistency) | P: 10 ms and 1 ms levels (Fig. 11); the rest ours |
| substeps per boundary | boundary / δt, collision per substep | ours |
| ICF Newton tolerance | 1e-8 | P (Sec. VI-B, fixed-step mode) |
| MuJoCo solver tolerance | 1e-8 (default) | MJ |

## 3. Solvers as instantiated

| item | MuJoCo arms | ICF arms | symmetric? |
|---|---|---|---|
| contact model | soft constraint: a_ref = −k·imp(r)·r − b·v, impedance solimp (0.9, 0.95, 1 mm, 0.5, 2) | compliant point contact f = k(−φ)₊(1 − d v_n)₊, lagged normal, regularized friction | different models by design (the comparison is between models at their own compliance) |
| stiffness | reference solref τ = 2.4 ms (k = 1e5), 31.8 ms (k = 1e3), ζ = 1; calibrated so a resting 65 g sphere sinks m g/k at δt = 1 ms; refsafe clamp τ ≥ 2δt on | k = 1e5 / 1e3 exactly (`contact_stiffness`) | MuJoCo's stiffness is per effective mass (heavier bodies stiffer at the same τ) and step-clamped — measured, stated (`stiffness_sweep`) |
| dissipation | ζ = 1 (critically damped reference format) | Hunt–Crossley d = 1 s/m (**assumed**; paper silent; sensitivity 10 / 1 / 0 in `hard_clutter_forensics.md`); ball d = 0 (P) | not matchable exactly; both stated |
| ball | direct solref (−2.24e3, 0): undamped, calibrated to m g/k | k = 1e3, d = 0 | both conservative models (P: zero dissipation) |
| friction | μ = 0.5 (**assumed**), pyramidal cone (Newton's MuJoCo default), condim 3, impratio 1 (MJ default) | μ = 0.5, regularized Coulomb, stiction tolerance v_s = 0.1 mm/s (hard) / 1 cm/s (soft) (P), σ = 1e-3 (ICF), μ_dynamic/μ_static = 1 (ICF) | **note:** the training scenes use elliptic cone + impratio 10; Part 1 runs MuJoCo's defaults — follow-up sensitivity check (elliptic cone) queued after the rerun |
| integrator | implicitfast (Newton's MuJoCo default): joint damping implicit, joint stiffness explicit | first-order implicit velocity, semi-explicit position; joint PD fully implicit (K = δt·kp + kd) | model difference; the reason for the actuated stability map; stated |
| inner solver | Newton, 100 iterations, line search 50, tolerance 1e-8, ls_tolerance 0.01 (MJ defaults) | Newton, 100 iterations, exact line search (≤ 100), tolerance per §1/§2, Jacobi shift 1e-6, float32 | comparable iteration budgets; ICF is float32 throughout (ICF), MuJoCo Warp float32 |
| collision detection | MuJoCo's own collider (`use_mujoco_contacts=True`), margin 0, gap 0.1 (candidates only; verified physics-identical with gap 0 and with nconmax 4096) | Newton `CollisionPipeline`, margin 0, shape_gap 0.1 (candidates only) | different narrowphases (MuJoCo box collider vs Newton SAT); the bin-wall fault of `hard_clutter_forensics.md` was in the Newton SAT path and is fixed by authoring |
| contact budgets | nconmax 1024, njmax 1024 per world | `max_rigid_contact` 2048 per world | ≥ 2× measured peak demand (`verify_contact_budgets.py`: hard ~340–360/world, soft ~280/world); benches fail on dropped contacts |
| CUDA graph | one captured graph per boundary (all four arms) | same | symmetric; captured vs eager physics agree to 1e-8 |
| host syncs | 4-byte flag per march iteration (EC); none (fixed) | same | symmetric |

## 4. Scenes (`cenic_scenes.py`, `actuated_press.py`)

| item | value | source |
|---|---|---|
| soft clutter | 20 spheres r = 2.5 cm, ρ = 1000 (65 g), k = 1e3, v_s = 1 cm/s | P (k, v_s, count); size/density **assumed** |
| hard clutter | 10 spheres + 10 cubes (half 2.5 cm), k = 1e5, v_s = 0.1 mm/s | P (k, v_s, mix); size **assumed** |
| bin | inner half-width 0.15 m, wall thickness 4 cm, walls span z ∈ [−0.3, 0.3] (rim 0.3) | **assumed**; walls through the floor since b333592c (the coincident-face SAT fault) |
| initial arrangement | 4 columns × 5 layers from z = 0.12 m, staggered, jitter ±1.5 cm xy / ±5 mm z, cubes randomly rotated, seed 7; bench seed 42 | ours (the paper does not state it) |
| contact margin | 0 (point contact) | ours; 5 mm variant reported separately (`_margin5mm`) |
| ball | 0.1 kg, r = 5 cm, k = 1e3, d = 0, 1 m drop, 10 s, energy at the last apex | P (mass, k, dissipation, height, duration); radius **assumed** |
| actuated push | gantry PD K_p ∈ {1e2…1e6}, K_d = 2√(K_p m), tip 0.1 kg r = 1 cm, box 1 kg 10 cm, k = 1e5, μ = 0.5, d = 1 s/m, 300 mm/s trapezoid (0.1 s ramps, 0.3 m), targets held at 100 Hz, gap 2 cm | ours (design from the literature review, Theme E) |
| actuated throughput | N ∈ {64, 256, 1024, 4096}; per world K_p log-uniform [1e4, 1e5], speed uniform [0.15, 0.30] m/s, start delay uniform [0, 0.3] s, seed 42; EC ε = 1e-3, fixed 1 ms | ours |
| gravity | 9.81 m/s² | — |

## 5. Bench protocol

| bench | settings |
|---|---|
| work-precision | ε ∈ {1e-1 … 1e-6}; horizon 2 s; first two boundaries excluded; N = 1 median of 3 subprocess trials, N = 1024 once; timeout 100 s per simulated second (P); 1 h practical budget; budget 65536 |
| penetration / artifacts | 64 scenes, 2 s from t = 0 (first impacts included since 133fcaf4), seed 42; analytic ground penetration from the state (`verify_part1_penetration.py`); artifact = ejection or max penetration > v√(m/k) (2.27 mm hard, 22.7 mm soft) |
| scaling | 2⁶ … 2¹³ worlds; fixed 10 ms and ε = 1e-3; 0.2 s warm-up + 2 s timed; median of 3 trials; budget 4096 |
| real-time trace | 64 (and 1) scenes, 5 s, per-boundary sync (rates carry the sync floor) |
| consistency | reference = the same solver at δt = 0.1 ms; from t = 0.2 s, 20 windows of 0.1 s restarted from the reference; ℓ∞ over bodies, mean over windows; 8 scenes; the 0.1 ms row = reference vs itself (floor); idle GPU |
| stiffness sweep | one 65 g sphere at rest 3 s; k ∈ {1e3 … 1e8}; MuJoCo τ(k) through the two calibrated anchors (τ ∝ k^−0.56) |
| momentum / determinism | zero-g head-on pair; two identical arms, 8 worlds, 1 s |
| march cost | one world, idle GPU, iterations and µs per iteration vs ε |
| actuated | one world, 80 cells; metrics in the bench docstring |

## 6. Known asymmetries and deviations (state in the paper)
1. Contact models differ by design; each arm runs at its own model's compliance, calibrated to the same resting depth on one body.
2. MuJoCo's dissipation (ζ = 1) and ICF's (d = 1 s/m) are different laws; the clutters' dissipation is our assumption (sensitivity recorded).
3. MuJoCo's inner tolerance is not relaxed under error control (no eq. 34 analogue); ICF's is.
4. Friction cones differ (pyramidal default vs regularized round cone); MuJoCo's default cone is used.
5. Joint PD: implicit in ICF, stiffness-explicit in MuJoCo's implicitfast — the actuated stability map is that difference.
6. Narrowphases differ (MuJoCo collider vs Newton pipeline); both point contact, margin 0.
7. A 1e-6 s step floor and a 0.1 minimum shrink exist in our controllers but not in the paper's Alg. 1; neither was reached in the reported runs (no latches); floor accepts are not counted separately (follow-up).
8. δt_max = 10 ms on the ball and the actuated scene (the paper used 0.1 s on clutter only).
