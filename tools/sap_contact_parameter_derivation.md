# Deriving the SAP contact parameters from the LBM/Drake asset properties

Pass 38 of the SAP campaign. Derivation only — nothing here was run on the GPU,
and no task, scene, asset or solver file was changed.

Re-runnable numeric core: `tools/probes/sap_contact_parameter_derivation.py`.
Every number below comes out of that script or out of a source line cited inline.

---

## 0. The problem, re-measured

Three sites in `trossen_spatula_lift_env_cfg.py` construct `NewtonShapeCfg()`
bare (lines 148, 156, 167), so every shape in the scene takes Isaac Lab's
placeholders: `ke = 2.5e3 N/m`, `kd = 100 N·s/m`, `mu = 1.0`
(`isaaclab_newton/physics/newton_manager_cfg.py:78, 89, 97`). `tau` has no
per-shape channel at all and falls to `MJWarpSolverCfg.sap_contact_tau_d = 0.01`
(`mjwarp_manager_cfg.py:231`, applied at `mjwarp_manager.py:472,504`).

Combined per pair that is `k = 1250`, `tau_d = 0.02`, `mu = 1.0` — which is
exactly what the pass-25 dump recorded (`contact_k_unique: [1250.0]`,
`contact_tau_d_unique: [0.02]`). The validated LBM value for `mu` is **0.2**.

The assets' real physics is authored in `drake:proximity_properties` in the LBM
SDFs, a channel nothing in this stack reads:

| asset | mu_static/dynamic | hydroelastic_modulus | hunt_crossley_dissipation | type |
|---|---|---|---|---|
| mug (`mug_inomata_...sdf`) | 0.2 | 1e8 Pa | 40 s/m | compliant |
| spatula (`thimma_...sdf`) | 0.2 | 1e8 Pa | 40 s/m | compliant |
| plate (`ikea_dinera...sdf`) | 0.3 | 1e8 Pa | 40 s/m | compliant |
| fork | 0.15 | — | — | rigid |
| drying rack | 0.15 | 1e8 Pa | 40 s/m | compliant |

No asset in the bank sets `drake:point_contact_stiffness`, and none sets
`resolution_hint`.

---

## 1. Combination rules — read from source, not inherited

`sap_warp/sim/sap_helpers.py`:

```
_sap_combine_stiffness(k0,k1)          = k0*k1/(k0+k1)          # SERIES   (:199)
_sap_combine_mu(mu0,mu1)               = 2*mu0*mu1/(mu0+mu1)    # HARMONIC MEAN (:163)
_contact_tau_pair(t0,t1)               = t0 + t1                # SUM      (:233)
```

All three are consumed at `contact_jacobian.py:745-747`.

The campaign note "series / sums / harmonic" was **right in kind but the two
harmonic-looking rules differ by a factor of 2, and that factor is the whole
point**:

* stiffness is the **series** rule — two equal shapes give **half** the authored
  value (2500 & 2500 → 1250);
* friction is the **harmonic mean** — two equal shapes give **exactly** the
  authored value (0.2 & 0.2 → 0.2).

Authoring `mu` as if it halved, or `ke` as if it did not, is a 2x error each way.

Two further exactnesses worth having:

* `_sap_combine_stiffness` returns the *other* value when one side is non-finite
  (`sap_helpers.py:200-203`) — i.e. it implements the rigid-partner limit. But
  `_shape_stiffness_or_fallback` (`:191`) rejects a non-finite per-shape value
  first, so `inf` is **not authorable**; use a large finite stand-in.
* `contact_jacobian.py:727-733` places the contact point as
  `pC = k0/(k0+k1)·x0 + k1/(k0+k1)·x1`, so a stiff partner pulls the contact
  point onto its own witness point. That is the physically right behaviour for
  a rigid-compliant pair and it comes free.

**These are Drake's own rules.** Verified against Drake master
`29a5d2e6` (2026-08-15):

* stiffness/gradient: `discrete_update_manager.cc:940`
  `const T g = 1.0 / (1.0 / gM + 1.0 / gN);` — series, with the rigid side set
  to `+inf` (`contact_properties.cc:39-45`);
* relaxation time: `contact_properties.cc:153-162`
  `GetCombinedDissipationTimeConstant` — a plain **sum**.

So the SAP-warp rules are not an invention to be worked around; matching Drake
is the correct target.

---

## 2. Stiffness: hydroelastic modulus → point stiffness

**Drake's rule, used rather than invented** —
`multibody/plant/discrete_update_manager.cc:1025`, inside
`AppendDiscreteContactPairsForHydroelasticContact` (the function has moved out
of `compliant_contact_manager.cc`):

```cpp
const T& Ae = s.area(face);        // :896   contact-surface face area  [m^2]
const T g = 1.0/(1.0/gM + 1.0/gN); // :940   normal pressure gradient   [Pa/m]
const T k = Ae * g;                // :1025  point stiffness            [N/m]
```

Quadrature is first order, **one point per face** (comment at `:812`), so the
patch stiffness is the sum over faces and a single face contributes `A·g`.

The pressure field is `p = E·φ/H` (`hydroelastic_parameters_doxygen.h` eq. 1).
`H` is a **single global scalar per geometry**, not a local half-thickness:
for a general tet mesh it is the maximum distance-to-boundary over all vertices
(`make_mesh_field.h`); for a convex hull (`make_convex_field.h`) the field is
`E` at one interior vertex and 0 on the boundary, so the gradient into a facet
is `E / dist(interior vertex, facet plane)`.

Our mug collision prims are authored `convexHull` (`convert_mug.py:131`), so the
convex-field construction is the applicable one, and `H` is measurable per
facet. Measured on `assets/usd/mug_inomata_white.usd`:

| shape | contacting facet | A [mm²] | H [mm] | g = E/H [Pa/m] | k = A·g [N/m] |
|---|---|---|---|---|---|
| `collisions_base` | −Z (rests on table) | 2068 | 5.16 | 1.94e10 | **4.01e7** |
| `collisions_wall_[0-7]` | outer side | 1106 | 2.54 | 3.94e10 | **4.35e7** |
| `collisions_handle_[0-2]` | ±X (pad pinch) | 113 | 2.92 | 3.43e10 | **3.87e6** |

Calibration that the `H` construction is right: the two LBM assets whose
compliant **tet meshes are actually present** give, by Drake's max-interior-
distance definition, `H = 3.648 mm` (spatula `..._low.vtk`) and `H = 4.248 mm`
(plate `..._low_8faces.vtk`). The mug's own VTK
(`mug_inomata_white_low_16faces.vtk`) is **named in its SDF but absent from the
bank**, which is why the mug's `H` comes from the task hulls instead. The hull
values (2.5–5.2 mm) bracket cleanly against the two measured ones.

**Sensitivity.** `k` is exactly linear in `A` and in `1/H`. Doubling the patch
doubles `k`; halving it halves `k`. If the pipeline emits `N` contacts across
one facet, the faithful per-point value is `k/N` (Drake's per-face quadrature),
so the plausible band on any one number here is roughly `[k/8, k]`. **Section 3
shows this uncertainty does not reach the simulation.**

**Drake's own defaults, for scale.** `DefaultProximityProperties::point_stiffness
= 1e6 N/m` (`geometry/scene_graph_config.h:155`), backfilled onto every geometry.
The derived handle value 3.9e6 is within 4x of it. Drake's legacy heuristic
`k = 2·m·g/penetration_allowance` with the 1 mm default gives 355 N/m per
geometry for a 18.1 g mug. The derived numbers sit between Drake's two defaults,
nearer the modern one.

Also confirmed from source, and worth knowing: **Drake ignores
`hydroelastic_modulus` entirely under a point-contact model.** The parser files
it under `("hydroelastic", ...)` while the point path reads
`("material", "point_contact_stiffness")`
(`discrete_update_manager.cc:641-802`). The `k = E·A/H` formula appears in
Drake's docs only as a hand-estimation aid for users, never applied
programmatically. So the conversion above is a *documented* hand conversion,
not a code path anyone ships.

---

## 3. The authored `ke` is inert at every timestep this campaign runs

SAP's normal regularization (`contact_solve.py:946-965`, identical to Drake's
`sap_friction_cone_constraint.cc:64-76`):

```
rn_hard = beta^2/(4 pi^2) * w_eff        # beta = 1.0  -> 1/(4 pi^2) = 0.025330
rn_soft = 1/(h * k_pair * (h + tau_pair))
rn      = max(rn_hard, rn_soft)
vhat_n  = -phi0/(h + tau_pair)
```

At steady state (`v_n = 0`, penetration `x = -phi0`) the normal force
`f = gamma_n/h` collapses to `f = k_eff·x` with

```
k_eff = min(k_pair, k_cross),   k_cross = 1/(rn_hard · h · (h + tau_pair))
```

`k_cross` is therefore a **hard ceiling on the stiffness the solver can
realize**, no matter what is authored.

`w_eff` is the mean of the contact-frame Delassus diagonal
(`contact_jacobian.py:928`). Closed form for a contact at offset `r` from a free
body's COM:

```
w = 1/m + (1/3)[ (ry²+rz²)/I1 + (rx²+rz²)/I2 + (rx²+ry²)/I3 ]
```

For the mug (m = 18.1 g, I = 2.33/2.51/2.18e-5 kg·m², COM 45.6 mm up):

| pair | w_eff [1/kg] | × scene median (14.917) |
|---|---|---|
| mug base — table | 140.2 | 9.4 |
| mug wall — table | 94.5 | 6.3 |
| mug handle — pad | 180.0 | 12.1 |
| mug wall — pad | 104.4 | 7.0 |

The carriage side contributes `w ≤ 1/(armature + I_eq) = 9.9 /kg`; armature **is**
summed into the SAP mass matrix (`free_motion.py:1428`). Static bodies
contribute 0.

Cross-check that the whole chain is right: with `w = 14.917`, `k = 1250`,
`tau = 0.02`, `h = 1/120`, the formulas give `rn_hard/rn_soft = 0.11152` against
the pass-25 dump's measured `0.1115`. The crossover is at `w = 133.8 /kg` at
`h = 1/120` — and the mug's own contacts are 94–180, straddling it. That is
what the measured 11% near-rigid fraction *is*.

### (a) Which branch, with the derived values

`h = 1/240 s` (fixed arm, `num_substeps = 2`, `sim.dt = 1/120`):

| pair | derived k_pair | tau_pair | k_cross | **k_eff realized** | vs today's 1250 |
|---|---|---|---|---|---|
| base–table | 4.01e7 | 20 ms | 2796 | 2796 | 2.2x |
| base–table | 4.01e7 | 1.33 ms | 1.23e4 | 1.23e4 | 9.8x |
| base–table | 4.01e7 | 0 | 1.62e4 | 1.62e4 | 13.0x |
| handle–pad | 3.87e6 | 20 ms | 2178 | 2178 | 1.7x |
| handle–pad | 3.87e6 | 1.33 ms | 9581 | 9581 | 7.7x |
| handle–pad | 3.87e6 | 0 | 1.26e4 | 1.26e4 | 10.1x |

Every derived pair lands **far** on the near-rigid branch. The authored `ke`
would only stop being clamped below `h ≈ 5 µs` (base) / `h ≈ 20 µs` (handle).
The smallest inner dt this campaign has ever observed is 2.24 ms.

> **The single most important result of this pass: authoring the true `ke`
> changes nothing on its own. The realized contact stiffness is set by `w_eff`,
> `h` and `tau_d`. The entire achievable gain is ~10x, and it comes from
> shortening `tau_d`.**

This also disposes of the patch-area uncertainty from §2: a 4x error in `A`
moves `k_pair` by 4x and `k_eff` by 0%.

### (b) Steady-state penetration

Resting mug (0.1776 N over N = 4 contacts) on the derived values:
**2.8 µm** at the self-consistent `tau` (see §4), rising to ~16 µm at
`tau_pair = 20 ms`. Microns — physically sane for ceramic.

Under a **grasp** it is not. A 10 mm commanded finger overshoot against
`k_c = 9581` and the current drive `k_d = 1000` gives **0.95 mm** of embed at
~9 N grip. True ceramic at that load would deflect well under a micron. The
near-rigid ceiling, not the material, is what sets this — and the fix is `h` and
`k_d`, not `ke`.

### (c) Gripper drive stiffness

Vendor USD, re-read this pass:
`follower_left_left_carriage_joint drive:linear:physics:stiffness = 217687`,
`damping = 10884`, `maxForce = 400`, `physics:JointEquivalentInertia = 9.82e-4`.
The task authors 1000 / 50 (`assets.py:121-128`) — a **217.7x** softening,
confirming the campaign's "218x".

Finger equilibrium: `x = delta·k_d/(k_d + k_c)`, grip `f = k_c·x`.

| case | k_c | k_d | embed | x [mm] | grip [N] | margin at mu 0.2 |
|---|---|---|---|---|---|---|
| today | 1250 | 1000 | **44.4%** | 4.44 | 5.56 | 12.5x |
| derived, tau 20 ms | 2178 | 1000 | 31.5% | 3.15 | 6.85 | 15.4x |
| derived, tau 1.33 ms | 9581 | 1000 | 9.5% | 0.95 | 9.06 | 20.4x |
| **derived, tau 1.33 ms** | 9581 | **200** | **2.0%** | 0.20 | 1.96 | 4.4x |
| derived, tau 1.33 ms | 9581 | 100 | 1.0% | 0.10 | 0.99 | 2.2x |
| derived, tau 1.33 ms | 9581 | **217687** | **95.8%** | 9.58 | 91.8 | 206.7x |

The "today" row reproduces pass-35's measured 44% embed from first principles
(`1000/2250 = 0.444`), which independently confirms that today's pinch sits on
the *soft* branch — as the `w_eff` analysis predicts.

**Recommendation: `k_drive ≈ 200 N/m`, i.e. softer still, not stiffer.** That
gives 2% embed and ~2 N of grip, a 4.4x margin over the 0.444 N needed to hold
the mug at the validated `mu = 0.2`.

**The vendor stiffness cannot be restored.** For 217687 N/m to embed ≤5% you
need `k_c ≥ 4.1e6 N/m`, which requires `h ≤ 230 µs` (4.3 kHz) — 18x the current
substep. That conclusion is independent of anything authored on the asset.

Caveat on the drive side: at `k_d = 200 N/m` the joint's own friction (0.1 N,
`assets.py:126`) costs 0.5 mm of tracking error. That is tolerable but should be
looked at rather than assumed.

### (d) Risks of the change

1. **Authoring `ke` through the shared material channel breaks the MuJoCo arm.**
   `newton_manager_cfg.py:84-86` states `(ke, kd)` convert to MuJoCo `solref` as
   `timeconst = 2/kd`, `dampratio = (kd/2)·sqrt(1/ke)`. At `ke = 4e7` with the
   current `kd = 100` the dampratio drops from 1.0 to 0.0079 — a wildly
   underdamped contact. Holding dampratio at 1 requires `kd = 2·sqrt(ke) ≈ 12649`,
   giving `timeconst = 0.158 ms`, ~50x below MuJoCo's usual `≥ 2·dt` floor. **The
   derived `ke` must be applied SAP-only** (write `model.shape_material_ke` in the
   SAP branch, or gate it on `NEWTON_SAP=1`), never through `NewtonShapeCfg.ke`.
2. **Making every contact near-rigid changes what the campaign is measuring.**
   On the near-rigid branch `R_n = beta²/(4π²)·w` is **dt-independent** (audit
   §B1/§A). Pushing all contacts there means halving dt no longer halves the
   contact compliance — which is precisely the mechanism the adaptive-vs-fixed
   comparison exists to probe. Expect the adaptive arm's advantage to *shrink*
   for reasons that have nothing to do with timestepping.
3. **Solver cost.** `R_n` on the near-rigid branch is larger than `rn_soft` was,
   so the Hessian is *better* regularized — iteration counts should be flat or
   slightly lower. But `vhat_n = -phi0/(h+tau)` grows ~15x when `tau` goes
   20 ms → 1.33 ms, so per-step impulses and velocity jumps grow. Expect more
   line-search work, and on the adaptive arm more subdivision and longer wall
   time. This is a prediction, not a measurement.
4. **`mu` 1.0 → 0.2 invalidates every trained policy.** It is a re-baseline, not
   a tweak.

**Smoke-test order** (each in isolation, both arms where applicable):

1. `mu` alone → rest probe + scripted grasp. Cheapest, biggest real effect.
2. `tau_d` alone, SAP-only → compare iteration counts and the adaptive dt
   histogram against the pass-25 baseline.
3. `ke` last, SAP-only, and only after confirming the MuJoCo arm does not see it.
4. Before any of it: re-run the pass-25 near-rigid dump to get `N` (contacts per
   pair) and the `w_eff` distribution per pair, which this pass could only
   compute in closed form.

---

## 4. Dissipation: Hunt & Crossley 40 s/m → SAP `tau_d`

**There is no sanctioned conversion, and I checked.** Drake
(`multibody_plant.h:806-813`) states that exactly one of
`hunt_crossley_dissipation` and `relaxation_time` is used, chosen by the
discrete approximation: kSap uses `relaxation_time` and **ignores** Hunt &
Crossley; kTamsi/kSimilar/kLagged do the reverse. Neither Castro 2021 (SAP,
arXiv 2110.10107) nor Castro 2023 (ICF, arXiv 2312.03908) gives a formula;
Castro 2023's own hand-matched pairs have `d/tau_d` of 5e5, 1e5 and 5e4 — not a
constant. The one relation Drake does state,
`d = tau_d·k` (`sap_friction_cone_constraint.h:109-119`), relates `tau_d` to a
**linear Kelvin-Voigt coefficient in N·s/m**, a different quantity from Hunt &
Crossley's `d` in s/m. `tau_d = d_HC/k` is dimensionally wrong.

So what follows is **our** mapping, matched at a stated operating point, by
three independent routes.

**(A) Match the damping term.**

```
H&C:  f = k·x·(1 + d·ẋ)_+   ->  damping force  k·d·x·ẋ
SAP:  f = k·(x + tau·ẋ)_+   ->  damping force  k·tau·ẋ
equal at penetration x0  =>   tau = d · x0            [per shape; pairs sum]
```

The two models cannot be matched at all `x` — H&C's damping is proportional to
penetration and SAP's is not — so a one-point match is the most that exists.

| x0 | tau (per shape) | tau_pair |
|---|---|---|
| 1 µm | 0.04 ms | 0.08 ms |
| 10 µm | 0.40 ms | 0.80 ms |
| 100 µm | 4.0 ms | 8.0 ms |

*External check on the rule:* Castro 2023 hand-pairs `(k=1e7, d=500)` with
`tau_d = 1e-3`, i.e. `x0 = tau/d = 2 µm`; and `(d=10, tau_d=1e-4)`, i.e.
`x0 = 10 µm`. Both are physically sane operating penetrations. The rule
reproduces their hand-tuned pairs, so it is a real mapping rather than a
dimensional coincidence.

Solved self-consistently on the near-rigid branch (`tau = d·x` and
`x = f/(N·k_cross(tau))`) for the resting mug: **tau_pair = 0.11 ms, x = 2.8 µm**.

**(B) Match energy per impact.** `zeta = tau·omega_n/2` with
`omega_n = sqrt(k_eff/m)` (Castro 2021 §V-B). At the *realized* stiffness
1.23e4 N/m, critical damping is `tau_pair = 2.43 ms`. Restitution then:
`tau_pair = 1.33 ms → zeta 0.55, e = 0.13`; `2.42 ms → e ≈ 0`;
`20 ms → zeta 8.2, badly overdamped`. Target: H&C at `d = 40 s/m` is fully
plastic above `v = 1/d = 2.5 cm/s`, so `e ≈ 0` is what the asset asks for.

**(C) Castro/CENIC near-rigid prescription.** `tau = (beta/pi)·h = 1.33 ms` per
pair at `h = 1/240`.

**Verdict.** The three routes give `tau_pair ∈ [0.11, 2.43] ms` — agreement
within an order of magnitude, all clustered around 1 ms. The prompt asked
whether the derived value lands near CENIC's `(beta/pi)·dt`: **it does**, and
from two directions that do not reference CENIC at all.

**Recommend `tau_pair = 1.33 ms`, i.e. `sap_contact_tau_d = 6.6e-4 s` per shape.**
The current 0.01 (pair 20 ms) is **15x too long**.

Sensitivity: route (A) is linear in the assumed operating penetration — a 10x
error in `x0` is a 10x error in `tau`. Routes (B) and (C) do not depend on `x0`
at all, which is why they bound (A). For context, Drake's own default is
`relaxation_time = 0.1 s` per geometry (0.2 s combined,
`scene_graph_config.h:132`), so the current 0.02 is already 10x shorter than
Drake's default and the recommendation is 150x shorter. That is deliberate:
Drake's default is a safe-and-mushy value, not a derived one.

---

## 5. Friction

`mu = 0.2` needs no conversion, and under the harmonic mean two shapes both
carrying 0.2 give a pair `mu` of exactly 0.2. So the authoring is direct — but
it only works if **both** sides carry it. Authoring 0.2 on the mug alone against
today's 1.0 pads gives a pair `mu` of 0.333, not 0.2.

Where the current 5x error bites:

* **Grasp.** Holding the mug against gravity needs `mg/(2·mu)` of pinch: 0.089 N
  at `mu = 1.0`, **0.444 N at 0.2**. The policy has been trained needing 5x less
  squeeze than the validated physics demands.
* **Tip vs slide.** With base radius 33.1 mm and COM height 45.6 mm the mug tips
  rather than slides when `mu > r/z = 0.726`. At `mu = 1.0` it **always tips**;
  at 0.2 it **always slides**. This is the direct explanation for the "sticky"
  behaviour on video: nothing skates, everything either sticks or topples.
* **Sliding on the table.** 0.178 N of lateral force to move the mug today,
  0.036 N derived — a glancing arm touch that currently drags the mug will
  instead push it away.
* **Rotation in the pinch.** At `mu = 1.0` the mug cannot rotate about the grip;
  at 0.2 it will, under its own weight. The policy has never had to handle that.

There is **no validated source for the gripper-pad or table `mu`**. Authoring
0.2 on them is the choice that reproduces the object's validated pair value
exactly. Alternatives, for the record: elastomer pads (1.0) would give a pair
`mu` of 0.333; the LBM bank's own stainless value (0.15) would give 0.171.

---

## 6. The authoring recipe

### Per-shape values

| shape | `ke` [N/m] | `mu` | `tau` [s] |
|---|---|---|---|
| `/Mug/collisions_base` | 4.2e7 | 0.2 | global |
| `/Mug/collisions_wall_[0-7]` | 4.6e7 | 0.2 | global |
| `/Mug/collisions_handle_[0-2]` | 3.9e6 | 0.2 | global |
| `TableGuard` cuboid | 1e9 | 0.2 | global |
| `follower_left_carriage_{left,right}` | 1e9 | 0.2 | global |
| `follower_left_gripper_{left,right}` | 1e9 | 0.2 | global |
| `follower_left_link_[1-6]`, base, camera | 1e9 | 0.2 | global |
| `/World/GroundPlane` | 1e9 | 0.2 | global |

`ke` values already have the series rule inverted against the 1e9 partner (the
raw `A·g` targets are 4.01e7 / 4.35e7 / 3.87e6; the residual error against a
truly rigid partner is 4.4% on the stiffest piece). Divide by `N` if the
per-pair contact count is measured and you want per-point faithfulness — but see
§3, it does not reach the simulation at any dt this campaign runs.

### Resulting per-pair values

| pair | k_pair | **k_eff realized** | mu_pair | tau_pair |
|---|---|---|---|---|
| mug base — table | 4.01e7 | 1.23e4 (clamped) | 0.2 | 1.33 ms |
| mug wall — table | 4.35e7 | 1.83e4 (clamped) | 0.2 | 1.33 ms |
| mug handle — pad | 3.87e6 | 9.58e3 (clamped) | 0.2 | 1.33 ms |
| mug wall — pad | 4.35e7 | 1.65e4 (clamped) | 0.2 | 1.33 ms |
| robot — table | 5.0e8 | k_cross(w) | 0.2 | 1.33 ms |

### Channels

* **`mu`**: bind a `UsdPhysics.MaterialAPI` material per collision prim with
  `physics:dynamicFriction` / `physics:staticFriction = 0.2`
  (`import_usd.py:3749`). Pass-35 found no material bound anywhere in this
  scene, so this is new authoring, not an edit. Alternatively set
  `NewtonShapeCfg.mu = 0.2` for a uniform scene-wide value — which, given the
  table above is uniform, is the simpler and equivalent move.
* **`ke`**: `newton:contactStiffness` on the same bound material
  (`newton/_src/usd/schemas.py:291`). **SAP-only** — see risk (1) in §3(d).
* **`tau`**: **not per-shape authorable.** There is no `shape_material_tau` on
  the Newton `Model` (`collision_model.py:290` `getattr(..., None)`) and no
  Newton USD schema attribute for it. Set the global
  `MJWarpSolverCfg.sap_contact_tau_d = 6.6e-4` (per shape → 1.33 ms per pair).
* **`k_drive`**: `assets.py` `"left_gripper"` `stiffness: 1000.0 → 200.0`.

### What to actually apply, and in what order

Given §3, the honest recommendation is **not** to apply all of it at once:

1. **`mu = 0.2`** — real, 5x, and the most likely cause of the observed
   stickiness. Safe on both arms (0.2 is the validated value for both). Requires
   a re-baseline.
2. **`sap_contact_tau_d = 6.6e-4`** — SAP-only, no MuJoCo-arm effect, and the
   only knob that actually moves the realized contact stiffness (up to ~10x).
3. **`k_drive = 200`** — pairs with (2) to cut the finger embed from 44% to 2%.
4. **`ke`** — last, SAP-only, and mostly for correctness of record: it is inert
   at every timestep this campaign runs.

---

## 7. Residual risks

1. **`N`, the contacts emitted per shape pair, was not measured** — it needs the
   GPU. It scales the authored `ke` by up to 8x. Immaterial while clamped;
   material below `h ≈ 5–20 µs`.
2. **The `k = A·g` conversion is a hand conversion.** Drake never applies it
   programmatically (§2), and sap_warp's own hydroelastic path is inert —
   `entry_k_eff` is allocated but never written, and `collision/pipeline.py:318`
   says the merge is still in progress. So this pushes a hydroelastic number
   into a point-contact channel by hand, which is exactly what Drake documents
   as a user aid and nothing more.
3. **`E = 1e8 Pa` is TRI's compliant surrogate, not ceramic.** Real ceramic is
   ~7e10 Pa — 700x stiffer. It is also 10x above Drake's own default
   hydroelastic modulus (1e7). "Validated" here means TRI shipped it, not that
   it is the material's modulus. Every stiffness above inherits that.
4. **The mug's compliant tet mesh is missing from the bank.** Its `H` is derived
   from the task's convex hulls; only the spatula (3.65 mm) and plate (4.25 mm)
   `H` are measured from TRI's own meshes. They bracket the hull values, but
   they do not confirm them.
5. **`w_eff` for the pad hinges on armature entering `M`.** It does
   (`free_motion.py:1428`), giving 9.9 /kg; without it the vendor inertia alone
   gives 1018 /kg — a 103x swing that would move `k_cross` by ~6x. Verified in
   code, not measured.
6. **No validated `mu` exists for the pads or the table.** 0.2 is chosen to
   reproduce the object's value under the harmonic mean, not measured.
7. **The 10 mm commanded finger overshoot in §3(c) is an assumption.** The embed
   *fractions* are ratio-only and hold regardless; the absolute `x` and grip
   forces scale linearly with it.
8. **Whether `NewtonShapeCfg.gap = 0.01` shifts `phi0` was checked by code path,
   not by probe.** `phi0` uses `rigid_contact_margin0/1`
   (`contact_jacobian.py:740-744`), fed from `shape_margin` (0.0), not `gap`. If
   that reading is wrong, every contact activates 1 cm early and all of §3 moves.
9. **Nothing here has been run.** The whole document is closed form against
   source and geometry. The first real evidence will be the rest probe.
