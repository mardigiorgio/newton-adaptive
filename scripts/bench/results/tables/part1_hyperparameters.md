# Part 1 — constants and comparisons

Sources: **P** CENIC paper · **MJ** MuJoCo default · **ICF** icf_warp default · **D** Drake · **O** ours · **A** assumed (paper silent).

| symbol | value | source |
|---|---|---|
| k_init | 0.1 | P |
| k_safe | 0.9 | P |
| k_low, k_high | 0.9, 1.2 | P |
| k_maxgrow | 5.0 | P |
| min shrink | 0.1 | D |
| order p | 2 | P |
| error norm | ‖q − q̂‖∞, absolute | P |
| δt_max | 0.1 s clutter · 10 ms ball, actuated | P · O |
| δt_min | 1 µs (hits: ≤ 2 / 6950 attempts, one cell; 0 elsewhere) | O |
| ε_tol, ICF | max(10⁻³ ε_acc, 10⁻⁸) | P |
| ε_tol, MuJoCo | 10⁻⁸ | MJ |
| march budget | 65536 work-precision · 4096 others | O |

*Table 1. Step-size controller, identical in both error-controlled configurations. Error estimate by step doubling (three solves, two geometry queries per attempt), accept when e ≤ ε_acc. The paper's Alg. 1 has no minimum shrink and no δt floor; measured floor occupancy at ε down to 10⁻⁶ is zero except 2 of 6950 attempts in one hard-clutter cell (`floor_occupancy.md`). ε_tol is the inner Newton tolerance under error control; MuJoCo keeps its own default. A world that exhausts the march budget is marked and never plotted.*

| δt | benches | source |
|---|---|---|
| 10, 5, 2, 1 ms | work-precision, penetration, actuated | P (10, 1 ms) · O |
| 10 ms | scaling, throughput (1 ms) | O |
| 10 ms → 10 µs | ball | O |
| 0.5, 0.1 ms | consistency (added) | O |
| inner tolerance | 10⁻⁸, both solvers | P · MJ |

*Table 2. Fixed stepping: substeps = boundary/δt, collision every substep.*

| parameter | MuJoCo | ICF | source |
|---|---|---|---|
| stiffness, hard | τ = 2.4 ms | k = 10⁵ N/m | O calib. · P |
| stiffness, soft | τ = 31.8 ms | k = 10³ N/m | O calib. · P |
| dissipation | ζ = 1 | d = 1 s/m | MJ · A |
| ball | solref (−2.24·10³, 0) | k = 10³, d = 0 | O calib. · P |
| friction μ | 0.5 | 0.5 | A |
| cone | pyramidal, impratio 1 | regularized, σ = 10⁻³ | MJ · ICF |
| stiction v_s | — | 0.1 mm/s hard · 1 cm/s soft | P |
| impedance | solimp (0.9, 0.95, 1 mm, 0.5, 2) | — | MJ |
| refsafe | on (τ ≥ 2 δt) | — | MJ |

*Table 3. Contact parameters per solver. The models differ by design; MuJoCo's τ is calibrated so a resting 65 g sphere sinks m g/k at δt = 1 ms, the same depth ICF gives with k exactly. MuJoCo's stiffness scales with effective mass and is clamped by refsafe (stiffness sweep); the ball is undamped in both. The clutters' dissipation is our assumption (sensitivity recorded). The training scenes use MuJoCo's elliptic cone with impratio 10; measured effect on hard-clutter penetration ≤ 15 % at matched knobs, no ejections either way (`mujoco_cone_probe.md`).*

| setting | MuJoCo | ICF |
|---|---|---|
| integrator | implicitfast | implicit velocity, semi-explicit position |
| inner solver | Newton, 100 it., line search 50, tol 10⁻⁸ | Newton, 100 it., exact line search ≤ 100 |
| precision | float32 | float32 |
| collision | MuJoCo collider, margin 0, gap 0.1 | Newton pipeline (SAT), margin 0, gap 0.1 |
| contact budget | nconmax 1024, njmax 1024 | 2048 per world |
| joint PD | stiffness explicit, damping implicit | fully implicit |
| CUDA graph | one per boundary | one per boundary |
| host sync | one flag per march iteration | one flag per march iteration |

*Table 4. Solver and collision settings. Budgets are ≥ 2× measured peak demand and benches fail on a dropped contact. The narrowphases differ; both are point contact at margin 0 (gap only widens the candidate set). Captured graphs agree with eager stepping to 10⁻⁸.*

| scene | bodies | contact | source |
|---|---|---|---|
| soft clutter | 20 spheres, r 2.5 cm, 65 g | k 10³, v_s 1 cm/s | P · A (size) |
| hard clutter | 10 spheres + 10 cubes, 2.5 cm | k 10⁵, v_s 0.1 mm/s | P · A (size) |
| bin | half-width 0.15 m, wall 4 cm, z ∈ [−0.3, 0.3] | — | A |
| ball | 0.1 kg, r 5 cm, 1 m drop, 10 s | k 10³, d 0 | P · A (r) |
| actuated push | tip 0.1 kg r 1 cm · box 1 kg, 10 cm · K_p 10²…10⁶, K_d = 2√(K_p m) · 0.3 m/s | k 10⁵, μ 0.5, d 1 | O |
| throughput | N 64…4096 · K_p ~ logU[10⁴, 10⁵] · v ~ U[0.15, 0.3] · delay ~ U[0, 0.3] s | same | O |

*Table 5. Scenes. Clutter starts as 4 columns × 5 layers from z = 0.12 m with ±1.5 cm / ±5 mm jitter, cubes randomly rotated, scene seed 7, bench seed 42, margin 0 (a 5 mm variant is reported separately). Gravity 9.81 m/s².*

| bench | worlds | horizon | repeats | knob |
|---|---|---|---|---|
| work-precision | 1 · 1024 | 2 s | 3 · 1 | ε 10⁻¹…10⁻⁶ · δt 10…1 ms |
| penetration | 64 | 2 s | 1 | same |
| scaling | 2⁶…2¹³ | 0.2 + 2 s | 3 | δt 10 ms · ε 10⁻³ |
| trace | 64 · 1 | 5 s | 1 | — |
| consistency | 8 | 20 × 0.1 s | 1 | reference δt 0.1 ms |
| stiffness | 1 | 3 s | 1 | k 10³…10⁸ |
| ball | 1 | 10 s | 1 | δt ladder · ε |
| actuated | 1 | 80 cells | 1 | δt · ε |
| throughput | 64…4096 | 2 s | 1 | ε 10⁻³ · δt 1 ms |

*Table 6. Bench protocol. Wall time is the per-boundary median with the first two boundaries excluded, captured graphs, idle GPU. Artifact = ejection or max penetration > v√(m/k) (2.27 mm hard, 22.7 mm soft). Consistency windows restart from the same solver's 0.1 ms reference. Timeout 100 s per simulated second (P).*

## Asymmetries to state in the paper

1. Different contact models by design, each at its own calibrated compliance.
2. Dissipation laws differ (ζ = 1 vs d = 1 s/m); d is assumed.
3. ICF relaxes its inner tolerance under error control (eq. 34); MuJoCo does not.
4. Friction cones differ (pyramidal vs regularized round); MuJoCo's cone choice moves penetration ≤ 15 % (probe).
5. Joint PD implicit in ICF, stiffness-explicit in MuJoCo's implicitfast.
6. Different narrowphases; both point contact, margin 0.
7. A 1 µs floor and a 0.1 minimum shrink exist in our controllers, not in Alg. 1; measured floor occupancy is zero to ε = 10⁻⁶ (2/6950 attempts in one cell).
8. δt_max = 10 ms on the ball and actuated scenes; the paper used 0.1 s on clutter only.
