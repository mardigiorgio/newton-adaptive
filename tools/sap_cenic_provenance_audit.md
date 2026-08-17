# SAP SOLVER PROVENANCE AUDIT — CENIC (arXiv:2511.08771) vs. THIS IMPLEMENTATION

Pass 36 of the SAP campaign. 2026-08-16. **Source + literature only: zero GPU
processes started** (`nvidia-smi` polled read-only throughout; the only compute
process on the card was the live 4000-iteration training run, PID 2271848).
**No code was changed this pass.** The only files written are this document and
the ledger entry.

Audience: the methods section of a paper. Everything below is either (a) quoted
from the published paper, (b) read out of source at a named file and line, or
(c) derived in closed form from (a) and (b). Claims that are none of those are
marked **UNVERIFIED**. Nothing here is taken from a commit message, a docstring,
a code comment, or an earlier ledger entry without re-deriving it.

---

## 0. HOW TO READ THIS

Three separate questions are constantly conflated in this codebase and must be
kept apart:

1. **What does the CENIC paper specify?** (Section 1.)
2. **Who wrote the code?** Upstream authors' code is *validated*; anything
   authored in our forks is *unvalidated* until measured. (Section 2.)
3. **Does the code do what the paper says?** (Sections 3-8.)

A change being "ours" does not make it a divergence, and a divergence is not
necessarily ours — the largest single divergence in this audit
(Section 4.1) is inherited from upstream `sap_warp`, which implements SAP
(Castro et al. 2021), a *different member of the same ICF family* than the one
CENIC builds on.

Evidence classes used throughout:

| Class | Meaning |
|---|---|
| **PAPER** | Quoted from arXiv:2511.08771 (CENIC) or arXiv:2110.10107 (SAP). |
| **SOURCE** | Read at a named file:line in this working tree at the stated HEAD. |
| **DERIVED** | Closed-form consequence of PAPER + SOURCE. Reproducible with pen and paper. |
| **MEASURED** | A number produced by a probe/run in this campaign. Cited, not re-run here. |
| **ASSERTED** | Someone wrote it down. Not evidence. Flagged wherever it is load-bearing. |

### Counts

| Category | Items | Ours | Upstream | Default ON in the reportable run |
|---|---|---|---|---|
| **A** — faithful to the paper | 17 | 12 | 5 | all |
| **B** — divergent, delta quantified | 11 | 9 | 2 (B1, and the code half of B2) | all |
| **C** — not in the paper, claimed physics-neutral | 9 | 9 | 0 | 8 of 9 (run-ahead OFF) |
| **D** — not in the paper, physics-visible | 12 | 12 | 0 | **6 of 12** (D1, D2, D3, D5, D6, D8) |
| **E** — infrastructure | 6 groups | 6 | 0 | n/a |
| **In the paper, not implemented** | 8 | — | — | — |

Commits audited: **17** in `sap_warp` (`c0c861c..afd5dc6`, all ours) and **25**
in `newton-adaptive` touching the SAP solver or the shared adaptive boundary
(all ours; zero upstream commits touch either).

---

## 1. WHAT THE PAPER SPECIFIES (the reference standard)

**Paper:** V. Kurtz and A. Castro, *CENIC: Convex Error-controlled Numerical
Integration for Contact*, arXiv:2511.08771. Retrieved 2026-08-16 from the
LaTeXML HTML rendering (`arxiv.org/html/2511.08771v1`); all equation numbers
below are the paper's own. Full text extracted to
`p36_cenic.txt` in the pass scratchpad.

The paper's own reference implementation is **C++ in Drake, single CPU thread**
("All implementations are in C++ using Drake"; "all results so far used a
single CPU thread"), with CPU/GPU parallelization named as *future work*. This
matters: every multi-world, batched-GPU concern in our implementation is
outside the paper's scope by the paper's own statement, not a deviation from it.

### 1.1 Formulation stack

CENIC = **error-controlled integration wrapped around ICF** (Irrotational
Contact Fields), a first-order symplectic IMEX scheme (Eq. 11-12) whose
velocity update is the unconstrained convex problem

    v^{n+1} = argmin_v  l(v; x^n, dt),
    l(v) = 1/2 v^T A v - r^T v + l_c(v),     A = M^n,  r = M^n v^n - dt k^n    (15-16)

ICF is a *family*. The paper states explicitly: **"CENIC builds on the Lagged
approach"** (Sec. IV-B), whose tangential impulse is

    gamma_t^{n+1} = -mu_d * v_t^{n+1} / sqrt(||v_t^{n+1}||^2 + v_s^2) * gamma_n^n     (14)

— tangential component implicit, **normal component lagged** at the previous
step. SAP is named as a *different* member of the same family.

### 1.2 Contact compliance (Sec. IV-A)

    gamma_n^{n+1} = dt * (f_e^n - dt * k_i * v_n)_+ * (1 - d_i * v_n)_+          (13)

with `f_e^n` the algebraic elastic force, `k_i` the contact stiffness and `d_i`
the **Hunt & Crossley** dissipation coefficient. The paper states: *"For point
contact, stiffness is a user supplied parameter."* There is **no dt-dependent
stiffness for point contact** anywhere in Sec. IV-A.

### 1.3 Near-rigid regularization (Sec. IV-D) — the load-bearing equation

The near-rigid law appears **only** in Sec. IV-D, *Limit and Holonomic
Constraints*. Verbatim (Eq. 18):

    k_i = (1 / (4 pi^2 beta^2)) * m_i / dt^2 ,     tau_i = (beta/pi) * dt      (18)

with `m_i = ||W_ii||^{-1}`, `W = G M G^T` the Delassus operator, and the stated
interpretation *"the i-th constraint behaves as a critically damped harmonic
oscillator with period beta*dt. A value beta = 0.1 is typical."*

The dt^-2 claim is likewise scoped to limits/holonomic constraints. Verbatim
(Sec. V, intro): *"Moreover, **for limit and holonomic constraints**, CENIC
regularizes the original DAE using ICF in a way similar to a Singular
Perturbation Problem ... as the user-specified accuracy tightens and time steps
shrink, ICF's near-rigid regularization, which scales as dt^-2 (18), yields
increasingly tight constraint enforcement."*

**The printed Eq. (18) is internally inconsistent, and the inconsistency is
decidable.** Three independent checks, all DERIVED:

1. *Against its own stated period.* A harmonic oscillator of mass `m` and
   stiffness `k` has period `T = 2 pi sqrt(m/k)`. Setting `T = beta*dt` gives
   `k = 4 pi^2 m / (beta^2 dt^2)` — the reciprocal of the printed form.
2. *Against its own tau.* Critical damping requires `tau = 2 sqrt(m/k)`. With
   `k = 4 pi^2 m/(beta^2 dt^2)` this is exactly `tau = (beta/pi) dt`, the
   paper's own second half of Eq. (18). With the printed `k`, critical damping
   would require `tau = 4 pi beta dt`, i.e. `4 pi^2 ~ 39.5x` larger than what
   the paper prints.
3. *Against the source it cites.* Eq. (18) cites [11] = Castro, Permenter, Han,
   *An Unconstrained Convex Formulation of Compliant Contact*,
   arXiv:2110.10107. That paper states verbatim: `tau_d = (beta/pi) dt`;
   *"we can estimate the value of stiffness from the frequency omega_n as
   k = 4 pi^2 m / (beta^2 dt^2)"*; and therefore

       R_n = (beta^2 / (4 pi^2)) * w ,   w = 1/m                        [SAP Eq. before (29)]
       R_n = max( (beta^2/(4 pi^2)) ||W_ii||_rms , 1 / (dt k (dt + tau_d)) )   [SAP Eq. (29)]

   with `||W_ii||_rms = ||W_ii||/3` and *"In all of our simulations, we use
   beta = 1.0"* and `sigma = 1e-3`.

**Verdict: the printed CENIC Eq. (18) carries an inverted `4 pi^2 beta^2`
factor.** The intended law is `k = 4 pi^2 m / (beta^2 dt^2)`, equivalently
`R_near-rigid = beta^2/(4 pi^2) * w`. Our implementation uses the latter and is
therefore **faithful** — see Section 3.1. The audit hypothesis that "one of the
two placements is wrong" is resolved in favour of the code, against the printed
equation.

### 1.4 Step doubling (Sec. V-A)

    x^{n+1/2} = ICF(x^n; dt/2)                                          (22)
    x^{n+1}   = ICF(x^{n+1/2}; dt/2)                                    (23)
    xhat^{n+1} = ICF(x^n; dt)                                           (24)

Three convex solves per step. *"Quantities to build the full-step problem are
the same as those for the first half-step, and can be reused. Thus **only two
geometry queries** are required, since both (22) and (24) use the same geometric
information at x^n, while (23) requires a separate query at x^{n+1/2}."*

First-order scheme, **second-order error estimate** — hence `p = 2` in the
step-size law.

A second-order trapezoid variant (Sec. V-B) also exists but the paper's own
conclusion is that *"instability in a second-order trapezoidal method severely
degraded performance compared to first-order step-doubling"*.

### 1.5 Error norm (Sec. V-E)

    e^{n+1} = || S (q^{n+1} - qhat^{n+1}) ||_inf

Weighted, **position-only**, `L^inf`. *"the diagonal scaling matrix S maps each
component to a dimensionless unit. This makes eps_acc ... an acceptable fraction
of unit error, or equivalently, the desired digits of accuracy in the solution
(0.1 roughly corresponding to 10%). S can be estimated from knowledge of
coordinate types or it can be specified by expert users."*

Velocities are deliberately excluded (impulsive velocity spikes at impact would
otherwise dominate).

### 1.6 Controller (Sec. V-F, Algorithm 1)

    accept iff  e^{n+1} <= eps_acc
    dthat = Deadband( k_Safe * dt * (eps_acc/e^{n+1})^{1/p} , dt )
    dt   <- min( dthat , k_MaxGrow * dt , dt_max )
    Deadband(dt_new, dt) = dt   if  k_Low*dt < dt_new < k_High*dt,  else dt_new

Constants, verbatim: **k_Init = 0.1, k_Safe = 0.9, k_Low = 0.9, k_High = 1.2,
k_MaxGrow = 5.0.** On reject, time is not advanced and the step is retried at
the new dt. There is **no dt floor** in Algorithm 1. The paper explicitly
sanctions extra guards: *"A mature implementation must also include other safety
checks for corner cases such as divide-by-zero or not-a-number (NAN) errors."*

Accuracy used throughout the paper's experiments: `eps_acc = 1e-3` (the value it
calls most effective; also `1e-1 ... 1e-5` in sweeps).

### 1.7 Inner convex solve (Sec. VI)

Newton with **exact linesearch** (`alpha_i` "computed with exact linesearch
[11]"), plus four named optimizations:

* **VI-A Warm starts.** Full step from `v^n`; first half-step from
  `(v^n + vhat^{n+1})/2`; second half-step from `vhat^{n+1}`.
* **VI-B Adaptive convergence criteria.** Two criteria: optimality
  `||D g|| <= eps_tol max(1, ||D r||)` (Eq. 32, `D = diag(M)^{-1/2}`) and a
  Newton-error criterion (Eq. 33) using `Theta_i = ||dv_{i+1}||/||dv_i||`,
  `eta_i = Theta_i/(1-Theta_i)`. **Fixed-step mode uses `eps_tol = 1e-8`;
  error-controlled mode uses `eps_tol = max(kappa * eps_acc, 1e-8)` with
  `kappa = 0.001`** (Eq. 34).
* **VI-C Hessian reuse.** Refactor only when
  `Theta^{N-i}/(1-Theta) ||dv_i|| >= eps_tol max(1,||D r||)` (Eq. 35), `N = 10`.
  Reuse spans iterations, solves and time steps.
* **VI-D Cubic linesearch initialization.** Exact linesearch = Newton-Raphson
  with bisection fallback; the initial guess is the minimizer of the Hermite
  cubic through `l(0), l'(0), l(alpha_max), l'(alpha_max)`, `alpha_max = 1`
  (up to 1.5).

Critically, the paper's headline claim #1 is: *"By building on the convexity of
ICF, CENIC **guarantees convergence for any dt, eliminating discarded
iterations**."* There is no iteration cap and no non-convergence branch in the
method as specified.

---

## 2. PROVENANCE — WHO WROTE WHAT

### 2.1 `sap_warp` is a fork, and the fork point is unusually early

    origin    https://github.com/mardigiorgio/sap_warp.git
    upstream  https://github.com/sap-sim/sap_warp.git   (remotes/upstream/main = c0c861c)

`sap_warp` is *not* the CENIC authors' release and *not* Drake. Per its own
README it is a Warp implementation of the SAP contact formulation
(arXiv:2110.10107) that "follows Drake's implementation as a reference",
maintained by the **AIVC Lab (UCLA)** and **TRI**; listed core contributors
include Joseph Masterjohn (Drake/TRI). Apache-2.0.

**The entire upstream repository at the fork point is four commits, all dated
2026-06-11:**

| Commit | Author | Subject |
|---|---|---|
| `431adf2` | f1shel | initial commit |
| `62da4a6` | yunuo | add doc building workflow |
| `6907f9c` | yunuo | [add] doc link |
| `c0c861c` | cffjiang | [fix] url in doc |

Three of those four are documentation. **The whole SAP solver — the convex
formulation, the regularization, the projection, the line searches, the
Cholesky, the collision stack, the loader — arrives in a single upstream commit
`431adf2` and is upstream code, not ours.** `upstream/main` has not moved since
(UNVERIFIED whether the upstream repo has advanced without being fetched; last
fetch state is what is recorded here).

Everything after `c0c861c` — **17 commits, `37663f5 .. afd5dc6`, all authored by
mardigiorgio** — is ours. 5810 insertions / 441 deletions across 8 files.

### 2.2 `newton-adaptive`: `SolverSAPAdaptive` is 100% ours

Fork point from newton-physics/newton is `c336b7ae`; HEAD `80d13a9a`. 492
commits since the fork, of which 152 are mardigiorgio's and ~340 are upstream
Newton authors' (Alain Denzler, Eric Heiden, Jan Carius, Han Xudong, Ruben
Grandia, ... — merged upstream work, **not ours**).

Of those 492, the ones touching `newton/_src/solvers/sap/` or
`newton/_src/solvers/adaptive_boundary.py` number **20 + 5, and every single one
is authored by mardigiorgio** (verified by filtering `%an`: the non-mardigiorgio
set is empty). `newton/_src/solvers/sap/solver_sap_adaptive.py` (3729 lines) has
no upstream ancestor.

### 2.3 Name collision to keep out of the paper

`newton/_src/geometry/broad_phase_sap.py` is upstream Newton's **Sweep-And-Prune
broad phase**. It has nothing to do with the Semi-Analytic Primal solver. Do not
cite it as SAP-solver work.

### 2.4 The reportable configuration

The live 4000-iteration run (`main-sap-adaptive-1024x4000-s42`, started
2026-08-16 19:51:21, i.e. at HEAD `80d13a9a`) was launched with **no
`NEWTON_*` environment overrides at all** (verified by reading
`/proc/2271848/environ`). Every environment flag in this document is therefore at
its **source default** in the run a paper would report.

Constructor arguments are a different matter: the platform layer
(`IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager.py:455-473`)
overrides three of `SolverSAPAdaptive`'s own defaults from
`MJWarpSolverCfg`. **The reportable configuration is therefore:**

| Parameter | Solver default | **Value in the run** | Source |
|---|---|---|---|
| `contact_preset_variant` | `"drake"` (fp64 throughout) | **`"approx32"`** — f32 Jacobians + f32 contact linear solve | `mjwarp_manager_cfg.py:206` |
| `max_substeps` | `16` | **`256`** | `mjwarp_manager_cfg.py` / `adaptive_max_substeps` |
| `max_iterations` (inner Newton) | `30` | `30` | `sap_solver_iterations` |
| `contact_tau_d` (per shape) | `0.01` | `0.01` -> **`0.02` per pair** | `sap_contact_tau_d` |
| `line_search_variant` | `"armijo_decay"` | `"armijo_decay"` | `sap_line_search` |
| `tol` (`eps_acc`) | `1e-3` | `1e-3` | `adaptive_tol` |
| `dt_inner_init` / `dt_inner_min` | `0.01` / `1e-12` | `0.01` / `1e-12` | cfg |
| `solve_precision` | `"fp64"` | `"fp64"` | cfg |

Note that the *fixed-step* SAP arm is constructed with the adaptive arm's
tolerances deliberately matched (`mjwarp_manager.py:500-515`) so that a
difference between the arms isolates timestepping — a fairness decision worth
stating in any comparison.

---

## 3. CATEGORY A — FAITHFUL TO THE PAPER

| # | Item | Code site | Commit / author | Evidence | Paper |
|---|---|---|---|---|---|
| A1 | Near-rigid clamp `R_n = max(beta^2/(4 pi^2) w_i, 1/(h k (h+tau)))` | `sap_warp/sim/sap_helpers.py:2399-2406` (+7 twins) | `431adf2` **upstream** | SOURCE + PAPER | CENIC (18) *as intended*; SAP (29) verbatim |
| A2 | `beta = 1.0`, `sigma = 1e-3` defaults | `sap_warp/sim/solver_sap.py:787-788` | `431adf2` **upstream** | SOURCE | SAP: "In all of our simulations, we use beta = 1.0"; "sigma = 1e-3" |
| A3 | Tangential regularization `R_t = sigma * w_i` | `sap_warp/sim/sap_helpers.py:2407` | `431adf2` **upstream** | SOURCE | SAP (30) |
| A4 | Near-rigid `beta = 0.1` for joint limits and PD constraints | `sap_helpers.py:2315` (`_SAP_PD_BETA`), `contact_solve.py:797` (`_SAP_LIMIT_BETA`) | `431adf2` **upstream** | SOURCE | CENIC IV-D: "A value beta = 0.1 is typical" — and IV-D *is* the limit/holonomic section |
| A5 | Effort-limited linear control law in the convex potential | `contact_solve.py:616,681` (`pd_limit` from `joint_effort_limit`) | `431adf2` **upstream** | SOURCE | CENIC IV-E, Eq. (19-20) |
| A6 | Step doubling: full step then two halves, halves committed | `solver_sap_adaptive.py:2334-2375` | `5d7c1c73` ours | SOURCE | CENIC (22-24) |
| A7 | Error metric `e = ||S (q_double - q_full)||_inf`, position-only | `solver_sap_adaptive.py:310-368` | `5d7c1c73` ours | SOURCE | CENIC V-E (the kernel docstring even cites it) |
| A8 | Step-size law `dt_new = 0.9 * dt * sqrt(tol/e)` | `solver_sap_adaptive.py:694, 719` | `5d7c1c73` ours | SOURCE | CENIC V-F with `p = 2`, `k_Safe = 0.9` |
| A9 | Symmetric deadband `[0.9, 1.2]`, growth cap `5.0` | `solver_sap_adaptive.py:110-114, 695-697, 722-724` | `5d7c1c73` ours | SOURCE | CENIC V-F constants **exactly**: k_Safe 0.9, k_Low 0.9, k_High 1.2, k_MaxGrow 5.0 |
| A10 | Accept iff `e <= tol`; reject holds state and retries | `solver_sap_adaptive.py:728-730, 707-717` | `5d7c1c73` ours | SOURCE + DERIVED (see 3.2) | CENIC Algorithm 1 |
| A11 | Warm starts: full from `v^n`, half-1 from `(v^n+vhat)/2`, half-2 from `vhat` | `solver_sap_adaptive.py:2334-2364` | `5d7c1c73` ours | SOURCE | CENIC VI-A, term for term |
| A12 | Shared assembly reuse for the full step and half-1 | `sap_warp` `b1e48a3`; driven from `solver_sap_adaptive.py:2349-2361` | ours | SOURCE | CENIC V-A: "quantities to build the full-step problem are the same as those for the first half-step, and can be reused" |
| A13 | NaN/inf guards mapping to divergence | `solver_sap_adaptive.py:365-366` | ours | SOURCE | CENIC V-F explicitly sanctions this |
| A14 | Cubic-Hermite linesearch initialization | `sap_warp/sim/contact_solve.py:4644-4662`, flag `SAP_CUBIC_INIT` default ON | `bd0c129` ours | SOURCE | CENIC VI-D — but see B6: it only runs in a non-default line search |
| A15 | `eps_tol = 1e-8` optimality target | `solver_sap_adaptive.py:1371-1374` | ours | SOURCE | CENIC VI-B **fixed-step** value; see B7 for the adaptive-mode mismatch |
| A16 | **No Richardson divisor** on the error estimate | `solver_sap_adaptive.py:351-358` | ours | SOURCE + PAPER | The paper's `e = ||S(q - qhat)||_inf` has no `1/(2^p - 1)` either. Both are conservative by ~3x relative to the true local error of the committed doubled state. Matching the paper here is the right call; it should be *stated*, since a reviewer familiar with step doubling will look for the divisor. |
| A17 | Rejected steps hold state by construction | `solver_sap_adaptive.py:762-798` | ours | SOURCE | Algorithm 1's "do not advance" — implemented as an accept-gated commit out of read-only scratch, so there is no rollback path to get wrong. |

### 3.1 A1 in detail — the `4 pi^2` question, settled

Verified independently at **all eight contact-`R` construction sites** (I read
each one; they are token-for-token identical modulo dtype):

    sim/contact_solve.py:948, 1180, 1278, 2760
    sim/sap_helpers.py  :2400, 2475, 2587, 2660

Each reads:

    beta_factor = beta*beta / (4.0 * PI * PI)
    rn_hard = beta_factor * wi
    rn_soft = 1.0 / (dt * k_c * (dt + tau_c))
    rn      = max(rn_hard, rn_soft)
    rt      = sigma * wi

This is **SAP Eq. (29) verbatim**. Of the eight sites, five are upstream
(`431adf2`) and three were added by us inside fused kernels: `contact_solve.py:2760`
(`3bff5c1`, fused update-eval) and `sap_helpers.py:2475, 2660` (`a79539a`, fused
alpha-max ladder). I diffed all three of ours against their upstream
counterparts: **the R-construction block is textually identical, in the same
operation order, at the same dtype** — so the fusions preserve `R` bitwise. That
is a checkable structural fact, not a claim.

Quantified consequence of the printed-Eq.-(18) alternative, had we implemented
it: `R` would be larger by `16 pi^4 / (1 + beta/pi) = 1182x` at `beta = 1`
(`16 pi^4 = 1558.5` ignoring the `(1+beta/pi)` factor), so the effective
stiffness ceiling would fall by the same factor. With the campaign's measured
near-rigid crossover at ~2.5e4 N/m, the printed form puts it at ~21 N/m — below
every authored stiffness in the scene, so **100%** of contacts would sit on the
clamp branch, contradicting the MEASURED 11% near-rigid fraction. (An earlier
session quoted ~98 N/m for this number; my derivation gives ~21 N/m. The
discrepancy does not affect the conclusion — both are far below any authored
stiffness — but the 98 N/m figure should not be reused without re-derivation.)

### 3.2 A10 in detail — the accept rule is faithful, and the extra clause is inert

`solver_sap_adaptive.py:728` reads

    acc = e <= tol or new_step >= step

which *looks* like it accepts over-tolerance steps whenever the controller still
wants to grow. **DERIVED: at the shipped constants it cannot.** Suppose
`e > tol`. Then `new_step = 0.9 h sqrt(tol/e) < 0.9 h`. The deadband fires only
if `new_step > k_Low h = 0.9 h`, so it cannot fire. The clamp to
`[0.1 h, 5 h]` can only raise `new_step` to `0.1 h`, still `< h`. Hence
`new_step >= step` is false and the disjunction reduces exactly to `e <= tol`.
The clause becomes live only if `k_Low` is lowered below `k_Safe = 0.9` — at
`k_Low = 0.8` it would admit steps up to `e = 1.27 tol`. **Do not change
`_DRAKE_HYSTERESIS_LOW` without re-deriving this.**

---

## 4. CATEGORY B — DIVERGENT FROM THE PAPER (deltas quantified)

### B1. The contact law is SAP, not CENIC's Lagged ICF — and the near-rigid clamp is applied to *contact*

**Origin: upstream `sap_warp` (`431adf2`). Not ours.**

| | CENIC (PAPER) | This implementation (SOURCE) |
|---|---|---|
| ICF family member | **Lagged** (Sec. IV-B, Eq. 14): tangential implicit, **normal impulse lagged** at `gamma_n^n` | **SAP**: both components implicit, analytic cone projection with `mu_tilde = mu sqrt(Rt/Rn)`, `mu_hat = mu Rt/Rn` |
| Normal dissipation | **Hunt & Crossley**, multiplicative `(1 - d_i v_n)_+` (Eq. 13) | **Kelvin-Voigt relaxation time** `tau_d`, additive inside the ramp: `gamma_n = (1/R_n)(vhat_n - v_n)_+`, `vhat_n = -phi_0/(h+tau_d)` |
| Point-contact stiffness | **user-supplied, dt-independent** (Sec. IV-A verbatim) | user-supplied `k`, **but clamped by the dt-coupled near-rigid branch** `R_n = max(beta^2/4pi^2 w, 1/(h k (h+tau)))` |
| Near-rigid dt^-2 scope | **limits and holonomic constraints only** (Sec. IV-D and the Sec. V intro sentence) | **contacts, limits and PD** all carry a near-rigid clamp |

Why this matters for the paper: the memory of this campaign records the CENIC
mechanism as "adaptive fixes penetration via dt^-2 contact-stiffness coupling".
Re-reading the paper, **`near-rigid` occurs four times and never in the contact
section**; the dt^-2 sentence is scoped, in the paper's own words, to *"limit and
holonomic constraints"*. So the dt^-2 contact mechanism this campaign has been
chasing is **an SAP property that CENIC inherits for constraints, not a CENIC
contact claim.** Any methods text should say so.

Expanding `R_n^soft` shows the two normal laws are close but not equal:

    SAP:   gamma_n = h ( f_e - h k v_n - tau k v_n )_+           (linearized Kelvin-Voigt)
    CENIC: gamma_n = h ( f_e - h k v_n )_+ (1 - d v_n)_+          (Hunt & Crossley, Eq. 13)

They agree to first order in `v_n` when `d = tau k / f_e`, i.e. only at a
particular force level. **This is a modelling difference, not a discretization
difference**, and it does not vanish as `dt -> 0`.

### B2. `tau_d` is a fixed material constant, not `(beta/pi) dt` — the dt^-2 mechanism is flattened to dt^-1.17

**The single most consequential item in this audit.**

* **SOURCE.** `tau_d` enters as a per-contact array `contact_tau_d[env, c]`
  filled from per-shape material values combined **pairwise as a sum**
  (`sap_warp/sim/solver_sap.py:956`, upstream `431adf2`:
  `contact_tau_d = shape_fallback_tau_d + shape_fallback_tau_d`). Our platform
  authors `sap_contact_tau_d = 0.01` s per shape
  (`IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/mjwarp_manager_cfg.py:231`;
  same default on `SolverSAPAdaptive`, `solver_sap_adaptive.py:1273`), so the
  **effective per-pair `tau_d` is 0.02 s** — independent of `dt`, and ~10x the
  production step.
* **PAPER.** CENIC (18) and SAP both set `tau_d = (beta/pi) dt`. SAP's
  experiments state *"The dissipation time scale is set to equal the time
  step"*.
* **DERIVED delta.** On the near-rigid branch `R_n` is dt-independent, so the
  effective contact stiffness is

      k_eff = 1 / (h (h + tau) R_n) = 4 pi^2 / (beta^2 w h (h + tau))

  whose dt exponent is

      d ln k_eff / d ln h  =  -(1 + h/(h + tau))

  | `h` | `tau = 0.02 s` (ours) | `tau = (beta/pi) h` (paper) |
  |---|---|---|
  | 0.1 ms | **-1.005** | -1.759 (h-independent) |
  | 1 ms | **-1.048** | -1.759 |
  | 2.083 ms (production step) | **-1.094** | -1.759 |
  | 4.15 ms (campaign mean accepted step) | **-1.172** | -1.759 |
  | 8 ms | **-1.286** | -1.759 |

  The MEASURED campaign value of **-1.172** is reproduced to four significant
  figures by this closed form at `h = 4.15 ms` — which is the campaign's mean
  accepted step. The measurement and the derivation corroborate each other; the
  mechanism is fully explained by the fixed `tau`.

  (The paper's `-1.759` rather than the naive `-2` is the exact value of
  `-(1 + 1/(1+1/pi))` at `beta = 1`; the paper's `dt^-2` is the `beta << pi`
  limit, `-2` exactly at `beta = 0.1`.)

* **Physical delta.** Resting penetration of a point mass,
  `phi = (beta^2/4 pi^2) g h (h + tau)`:

  | `h` | ours (`tau = 0.02`) | paper (`tau = beta h/pi`) | ratio |
  |---|---|---|---|
  | 1 ms | 5.2 um | 0.33 um | 16x |
  | 2.083 ms | 11.4 um | 1.4 um | 8x |
  | 10 ms | 74.6 um | 32.8 um | 2.3x |

  and ours shrinks **linearly** in `h` where the paper's shrinks quadratically.

* **Attribution.** The *code* is upstream SAP Eq. (29), which does keep the
  material `tau_d` in the soft branch. The *parameter choice* (`0.01` s per
  shape, fixed) is **ours**, in our platform config. So: not a code divergence,
  a configuration divergence — and it is the one that determines whether this
  implementation can exhibit the paper's dt^-2 mechanism at all. **It cannot,
  at the authored `tau`.**

### B3. Attempt-consistent R (ACR) — a change with no counterpart in the paper

**Ours.** Kernels `_scale_{w_eff,a_inv_pd,a_inv_limit}_attempt_consistent`,
`sap_warp/sim/contact_solve.py:4881-4988` (added `79e43bd`, list-indexed
`5a6cbf4`); gated from `newton-adaptive` and **flipped to default ON by
`45095218`, 2026-08-16 — i.e. ON in the live run.**

SOURCE (verbatim, `contact_solve.py:4902-4908`):

    w = w_eff_in[env, c]
    ...
    w = w * ((d_att * (d_att + tau)) / (h * (h + tau)))

DERIVED, writing `P(x) = x(x+tau)` and `s = P(D)/P(h)`:

* `rn_hard(scaled) = s * beta^2/(4pi^2) w = s * rn_hard(D)`
* `rn_soft(h) = 1/(k P(h)) = s * 1/(k P(D)) = s * rn_soft(D)` **identically**
* therefore `rn = max(...)` at `h` equals `s * rn(D)`: **same branch choice,
  value scaled by exactly `s`**
* `k_eff = 1/(P(h) * s * rn(D)) = 1/(P(D) rn(D))` = the value at the attempt dt

So ACR **freezes the entire contact constitutive law at the attempt step `D`**
for every sub-solve, making the step-doubling difference a pure integration
error rather than a mixture of truncation error and a dt-dependent change of the
contact model. `s == 1` exactly for the full solve (bitwise identity there).

**Magnitude, DERIVED.** For the half solves `h = D/2`, so

    s = 2 (D + tau) / (D/2 + tau)

| attempt step `D` | `s` |
|---|---|
| 1 ms | 2.049 |
| 2.083 ms (nominal) | 2.099 |
| 4.15 ms (mean accepted) | 2.188 |
| 8.333 ms (= `dt_outer`, the seeded first attempt) | **2.345** |

so `s` runs **2.05 - 2.35** over the campaign's step range; the "~2.34" on record
is the value at `D = dt_outer`. `s -> 2` exactly in the `tau >> D` limit and
`s -> 4` in the `tau << D` limit, so the fixed `tau` of B2 also pins `s` near 2.

**Two different comparisons, both true, easy to conflate — state which one you
mean:**

* *ACR at `h` versus the law evaluated at `D`*: `rt` and `rn` both carry the
  same factor `s`, so `rt/rn` — and hence `mu_hat = mu Rt/Rn`,
  `mu_tilde = mu sqrt(Rt/Rn)` — is **identical**. The friction cone geometry is
  invariant. That is precisely what makes ACR the right fix.
* *ACR at `h` versus no ACR at `h`* (the change the flag flip actually made):
  on the **near-rigid branch** `rt/rn = 4 pi^2 sigma / beta^2` is scale-free and
  nothing moves. On the **soft branch** — which is ~89% of contacts, given the
  MEASURED 11% near-rigid fraction — `rn_soft` is unscaled while `rt` gains `s`,
  so `rt/rn` is **`s ~ 2.1-2.35` times larger**: friction regularization is
  correspondingly softer/slippier, and normal compliance is `s` times softer.

The second comparison is the one that matters for "what changed on 2026-08-16",
and it is the origin of the "ACR softens friction ~2x" entry in this campaign's
notes — now derived rather than inherited.

**Why the paper has no counterpart:** CENIC's contact stiffness is
dt-independent (Sec. IV-A), so its `ICF(x; dt)` and `ICF(x; dt/2)` discretize the
*same* continuous contact model and Richardson extrapolation is valid without
any correction. Ours does not, on the near-rigid branch. ACR is a genuine repair
of a genuine defect — and it is **not in the paper, and changes trajectories.**
A reviewer will ask why it is needed; the honest answer is B1/B2.

### B4. The error norm's scaling matrix `S` is the identity

**Ours.** `solver_sap_adaptive.py:1590-1596`:

    # Accuracy-metric scaling S = identity (per PI directive); overwrite after
    # construction for expert per-coordinate scales.
    self._state_scale = wp.array(np.ones((wc, coords_per_world), np.float32), ...)

The paper's `S` exists specifically to *"map each component to a dimensionless
unit"* so that `eps_acc` reads as "digits of accuracy". With `S = I` the
`L^inf` is taken across **mixed units**: at `adaptive_tol = 1e-3` the budget is
simultaneously 1 mm of translation, 1e-3 rad = 0.057 deg per revolute joint, and
1e-3 of a free-body quaternion component (~0.115 deg). Those are *coincidentally*
sane, not *designed*. The paper's interpretation of `eps_acc` as a fraction of
unit error does not hold in our runs.

**And a second, live deviation in the same kernel.** `NEWTON_ADAPTIVE_RTOL`
defaults to `2e-6` (`solver_sap_adaptive.py:1543`), so
`rtol_over_atol = rtol/tol = 2e-3` and the per-coordinate difference is
normalized (`:355-357`):

    d <- d / (1 + 2e-3 * max(|q_double|, |q_full|))

turning the paper's fixed-tolerance test into the mixed test
`|d| <= atol + rtol |q|`. DERIVED magnitude: the tolerance is loosened by a
factor `(1 + 2e-3 |q|)` — 0.2% at `|q| = 1` (m or rad), 2% at `|q| = 10`, 20% at
`|q| = 100`. For this scene's coordinate magnitudes the effect is **well under
1%**, i.e. numerically negligible but formally not the paper's norm. Setting
`NEWTON_ADAPTIVE_RTOL=0` restores the paper's form on a bit-identical path.

### B5. Geometry cadence: one collision query per boundary, not two per step

**Ours.** `solver_sap_adaptive.py:1644-1663`,
`NEWTON_ADAPTIVE_CONTACT_REFRESH` default `"1"` = collide **once per boundary**
at the entry state; every attempt and both half-steps reuse that contact SET
(pairs, frames, materials), with gap/points/Jacobian re-derived per evaluation
state.

| | CENIC | ours |
|---|---|---|
| Collision queries per accepted step | **2** (at `x^n`, at `x^{n+1/2}`) | ~0.5 (one per outer boundary; at `dt_outer = 1/120` and a ~4 ms mean accepted step, roughly 2 attempts share one query) |

Consequences: contact **pairs** that appear inside a boundary are invisible until
the next boundary, and the error estimator is structurally blind to contact-set
changes — it only sees gap/Jacobian motion. This is a genuine accuracy
divergence and the first thing a reviewer should attack in a fast-contact
scenario.

### B6. The default line search is Armijo backtracking, not the paper's exact line search

**Ours (default change).** `sap_warp/sim/solver_sap.py:805`, commit `440e58a`
changed the default `line_search_variant` from upstream's `"monotone_decay"` to
`"armijo_decay"`; our platform pins the same
(`mjwarp_manager_cfg.py: sap_line_search = "armijo_decay"`). Flipping the
variant also flips three coupled tolerances (`cost_abs_tol 0 -> 1e-30`,
`cost_rel_tol 5e-3 -> 1e-15`, `cost_min_alpha 0 -> 0.5`).

The paper (VI, VI-D) specifies **exact linesearch** — Newton-Raphson on
`l'(alpha)` with bisection fallback — and SAP's convergence argument is built on
it. We ship inexact backtracking with `armijo_c = 1e-4`, `rho = 0.8`, at most 40
rungs (upstream defaults). Consequence: the paper's guaranteed-descent /
global-convergence argument does not transfer verbatim; convexity still gives
descent, but the iteration counts and the accepted `alpha` differ.

Sharpest form of this: **we implemented CENIC VI-D (`bd0c129`, cubic-Hermite
seed) and then do not use it**, because it lives only in the `exact_root`
variant which is not the default.

### B7. VI-B (adaptive convergence criteria) was implemented, then removed

Three distinct deviations, all in the inner-solve convergence test.

**(a) The adaptive tolerance rule was deleted.** Commit `5a84f078` (2026-06-27)
constructed the inner solver with

    optimality_rel_tol = max(1.0e-3 * float(tol), 1.0e-8)

which is **CENIC Eq. (34) exactly**, `kappa = 1e-3`, the paper's own value. At
`tol = 1e-3` that is `1e-6`. Commit `9c9dc934` (2026-08-15, campaign start)
replaced it with a hard `self._optimality_rel_tol = 1.0e-8`, decoupling it from
`tol`. So in error-controlled mode we now solve **100x tighter than the method
prescribes**, using the paper's *fixed-step* constant. Conservative for accuracy,
but it discards the VI-B speedup, and the deletion was deliberate rather than an
omission.

**(b) The second criterion (Eq. 33) does not exist.** There is no `Theta_i`,
no `eta_i`, no Newton-error estimate. Convergence is purely gradient-based.
Worse, the cost-based early exit that `sap_warp` would otherwise provide is
explicitly disabled: `SolverSAPAdaptive` passes `cost_abs_tol = cost_rel_tol =
0.0` (`solver_sap_adaptive.py:1391-1403`), so `|dcost| < 0` is never true.

**(c) The optimality test's reference scale differs from the paper's.**
`sap_warp/sim/contact_solve.py:3024` computes

    opt_tol = optimality_abs_tol + optimality_rel_tol * max(||p||, ||jc||)

i.e. `1e-14 + 1e-8 * max(||p||, ||jc||)`, whereas CENIC Eq. (32) is
`||D g|| <= eps_tol max(1, ||D r||)` with `D = diag(M)^{-1/2}`. Both are
"gradient small relative to a problem scale", but the scales are different
quantities. **Whether the `D` mass-scaling is applied to `grad_norm` in
`sap_warp` is UNVERIFIED in this pass** and should be checked before any claim
that our convergence criterion is the paper's.

### B8. Three exits that break the time/accuracy contract, none of them in Algorithm 1

Algorithm 1 has exactly two outcomes: accept and advance, or reject and retry.
Our controller has three additional exits, all ours, all physics-visible.

**(a) The `dt_min` floor with forced acceptance.**
`solver_sap_adaptive.py:674-705`: at `dt <= dt_min * 1.001` a step is **accepted
regardless of `e`** (unless the inner solve failed) because there is nothing left
to subdivide. DERIVED: with `dt_inner_min = 1e-12` and `dt_outer = 8.33e-3`,
reaching the floor by repeated `0.1x` shrinks takes ~10 consecutive rejections,
and `max_substeps = 256` allows it — but the floor is 10 orders below the working
step, so this exit is unlikely to be the binding one. Rate in the reportable run:
**UNVERIFIED.**

**(b) The floor-latch freezes state but advances the clock.** Same site,
`is_div` branch: `accept = False; sim_time[w] = next_time[w]; diverged[w] =
True`. The world's state is frozen and its clock is force-advanced to the
boundary — **simulated time passes with no dynamics**. This is a deliberate
containment contract (the consumer reads `solver.diverged` and is expected to
reset/terminate the world; in our task that is IsaacLab's `physics_diverged`
`DoneTerm`, `trossen_spatula_lift_env_cfg.py:402`), but it means a "completed"
boundary is not evidence that the boundary was integrated.

**(c) The debt guard silently drops time.** `_debt_guard`,
`solver_sap_adaptive.py:887-915`, run once per boundary: a world still short of
its boundary after `max_substeps` attempts has its **carried debt capped at
exactly one `dt_outer`** and its controller reset. Debt beyond one outer step is
discarded — simulated time is dropped. With `max_substeps = 256` in the
reportable run this should be far from binding, but *should* is not evidence:
`_debt_guard` fire counts are **UNVERIFIED** this pass and belong in any
accuracy table.

None of (a), (b) or (c) exists in CENIC, which simply keeps shrinking. All three
are consequences of running a fixed-cadence batched simulator where every world
must reach the same boundary in bounded work.

### B8b. The initial step is the full outer step, not `k_Init * dt_max`

Algorithm 1 starts at `dt = k_Init * dt_max` with **`k_Init = 0.1`**. Ours seeds
from the carried `ideal_dt` clamped to the per-boundary cap
(`_seed_dt`, `solver_sap_adaptive.py:258-281`; `eff_dt_max = min(dt_max,
dt_outer)`, `:3426`). At construction `ideal_dt = dt_inner_init = 0.01 s`, which
clamps to `dt_outer = 8.33e-3`. DERIVED: **the effective `k_Init` is 1.0, not
0.1** — the very first attempt of a fresh world is a full outer step. Minor, and
self-correcting after one rejection, but it is a stated constant of the method
and we do not use its value.

### B9. Non-convergence is a rejection path — contradicting the paper's claim #1

**Ours.** `_apply_solve_convergence_to_error`, `solver_sap_adaptive.py:588-595`:
an inner solve that reports `ok == 0` has its error overwritten with
`divergence_threshold`, which drives the controller's reject/shrink branch.

The paper asserts convexity **eliminates** discarded iterations. We need this
path because the inner solve is capped: our platform sets
`sap_solver_iterations = 30` (`mjwarp_manager_cfg.py:200-204`) against upstream's
`max_iterations = 100`, and the paper caps nothing (its `N = 10` in VI-C is
described as "simply a desired maximum" — *"the actual number of iterations can
be higher"*). A capped Newton loop can miss the tolerance for reasons that have
nothing to do with convexity, and we then reject the step. Reported honestly,
this is: *"we retain a non-convergence rejection path that the published method
does not require, because our inner iteration budget is capped for throughput."*

### B10. Not the second-order trapezoid method (Sec. V-B)

Only step doubling is implemented. This matches the paper's own recommendation
and is listed here only for completeness of the comparison.

---

## 5. CATEGORY C — NOT IN THE PAPER, CLAIMED PHYSICS-NEUTRAL

All of these are ours, all in `sap_warp`, all default **ON** unless noted. The
"proof class" column states what is *actually on record*, per CLAUDE.md's rule
that a comment is not evidence.

| # | Change | Commit | Site | Default | Proof class actually on record |
|---|---|---|---|---|---|
| C1 | Per-world `dt` threading (scalar -> `dt[env]`) | `37663f5` | ~10 kernels in `contact_solve.py`, `free_motion.py`, 4 integrators in `solver_sap.py` | n/a | **ASSERTED** (source comment): uniform `dt` fills `_dt_world` uniformly, so `dt[env]` reproduces `scalar(dt)`. Structurally credible, never measured. |
| C2 | Env-list compaction (`NEWTON_SAP_SOLVE_COMPACT`, `NEWTON_SAP_LS_COMPACT`) | `79e43bd`, `5a6cbf4` | ~25 kernels converted to `(env_idx, env_n)` list indexing | ON | Flag-equivalence probe `tools/probes/sap_flag_equivalence_probe.py` (scheduling-only oracle). Per-env work is env-private by construction. |
| C3 | Blocked-Cholesky narrowing (`*_masked_listed`) | `fe98f46` | `sap_warp/sim/blocked_cholesky.py` (+201 lines), used at `_solve_newton_direction` | ON when narrowed | Kernel bodies copied **verbatim** from the masked twins; env axis only. Structural. |
| C4 | Live-`k` contact-Hessian GEMM truncation (`NEWTON_SAP_GEMM_RESHAPE`) | `27dcada` | `contact_solve.py` bounded pack + GEMM factories | ON | DERIVED: truncated k-tiles contain only zero rows; surviving tiles accumulate in the same ascending `k` order. Engagement counter `gemm_reshape_skips()`. |
| C5 | Per-contact read-once GEMM pack + `j_flat` hoist (`NEWTON_SAP_PACK_PERCONTACT`) | `1ff0ea0` | `contact_solve.py`, `_CONTACT_HESSIAN_PACK_TILE_C = 32` | ON (AND `_gemm_reshape`) | Per-element `gj` expression preserved verbatim on the same operand loads. The `j_flat` hoist additionally rests on J-invariance within a solve — **ASSERTED**, probe-cited, not re-verified here. |
| C6 | Shared assembly reuse for half-1 | `b1e48a3` | `contact_jacobian.py`, `free_motion.py`, `solver_sap.py` | opt-in, driven ON by the adaptive solver | Skips launches that would rewrite byte-identical buffers. Correct **iff** the caller's contract (state/contacts/control/mask unchanged) holds — and the caller is ours. Also matches CENIC V-A (see A12). |
| C7 | Narrow-v3 list-indexed trip-cadence launches (two distinct flags share the name `NEWTON_SAP_NARROW_V3`) | `aac9694` | `contact_jacobian.py:1925`, `contact_solve.py:5502` | ON | Flag-equivalence probe family (`52005367` certified). |
| C8 | Run-ahead ADOPT/ANCHOR contact split | `2a119d2` (sap_warp), `e41cc070` (newton) | `contact_jacobian.py` +789 lines, 5 new kernels | **OFF** | Bitwise oracle probe `tools/probes/sap_runahead_oracle_probe.py`. Anchor arithmetic is a verbatim copy of the direct scatter's. |
| C9 | Host-sync-free masked runtime-state reset | `afd5dc6` | `solver_sap.py:1053` | n/a | Same outcome as the host-read path for the warm-start gate; benign same-value write race. |

**Honest caveat that belongs in the paper's supplementary, not buried here:**
none of the C-class bitwise claims were re-measured in this pass. This audit
verified the *structure* of C4 and the R-construction identity under C-class
fusions; the rest rests on the flag-equivalence and oracle probes listed, which
were run in earlier passes.

---

## 6. CATEGORY D — NOT IN THE PAPER, PHYSICS-VISIBLE

Ordered by how much each can move a trajectory. "In the run" = state in the live
4000-iteration run at HEAD `80d13a9a` with no env overrides.

| # | Change | Commit | Default / in the run | What it changes physically | Gate evidence |
|---|---|---|---|---|---|
| D1 | **Fused post-commit update eval** (`NEWTON_SAP_FUSED_UPDATE`) | `3bff5c1` | **ON** | One tiled kernel replaces the whole committed-point chain (projection + G block + `J^T gamma` + model terms + gradient + all three convergence norms). Reduction route changes from scattered `atomic_add` + per-`(env,dof)` tiles to serial ascending loops + fixed-schedule `wp.tile_sum`. **Totals, and hence convergence decisions, can differ in trailing fp digits.** | Flag-equivalence probe + graph-cache keying (`00c59d4d`). Not a bitwise claim. |
| D2 | **Fused armijo ladder** (`NEWTON_SAP_FUSED_LS`) | `f49b20b` | **ON** | Trial cost is evaluated along the ray `vc0 + alpha*dvc` instead of re-projecting `J` at `v + alpha*dv`; the regularizer reduction goes serial -> `wp.tile_sum`. Can change which rung is accepted. | `9757f69e` keyed the graph caches on the flag; flag-equivalence family. |
| D3 | **Alpha-max rung folded into the ladder** (`NEWTON_SAP_FUSED_ALPHAMAX`) | `a79539a` | **ON** (AND `_fused_ls`) | Rung 0's cost *and derivative* are computed in-kernel by the analytic ray form; the slop `scale = max(1, 0.5(|ell|+|current|))` moves in-kernel. Adds two full re-derivations of the contact projection (verified textually identical, Sec. 3.1). | `e5154ee0` keyed/gated. |
| D4 | **`monotone_decay` accept rule rewritten, unflagged** | `79e43bd` §5d | shipped, **no escape hatch** — but *not* the default variant | Accept changes from "first rung within tolerance" to "grid-minimizer bracket"; caller-side `decay 0.5 -> rho`, new `alpha0 = 1/rho`, `cost_relax_{a,r}` retied to `line_search_relative_slop/10`. | **None on record.** This is the least-covered physics-visible change in the range. Only bites if `monotone_decay` is selected, which our runs do not. |
| D5 | **Attempt-consistent R (ACR)** | `79e43bd` §5c, ON by `45095218` | **ON in the run** | See B3: freezes the contact constitutive law at the attempt dt; `s = 2.345` in the committed halves; normal contact 2.34x softer, soft-branch friction ratio 2.34x larger. | Derivation in B3; ACR arm in the campaign's probe set. `s == 1` bitwise for the full solve. |
| D6 | **Line-search default `monotone_decay -> armijo_decay`** | `440e58a` | **ON in the run** | Different line search + three coupled tolerances. See B6. | Rationale is a source comment (**ASSERTED**): monotone's 5e-3 cost early-exit can stop Newton on a plateau. Never measured against the exact line search. |
| D7 | **Cubic-Hermite seed** (`SAP_CUBIC_INIT`) | `bd0c129` | ON, but **dead** in our runs (exact_root only) | New seed formula for the exact-root iteration. | n/a — not on the default path. |
| D8 | **Per-world containment + `physics_diverged`** | ours, `newton-adaptive` | **ON** (`NEWTON_SAP_CONTAINMENT` default `"1"`) | A world whose inner solve fails is contained: its result is not committed, it does not kill the batch, and it does not perturb any other world. `NEWTON_SAP_CONTAINMENT=0` restores strict converge-or-throw. | Forced-failure probe `tools/probes/sap_containment_probe.py` (per-world isolation contract, bitwise non-interference). |
| D9 | **Determinism mode** (`NEWTON_SAP_DETERMINISTIC`) | `79e43bd` §5b, flipped OFF by `5a6cbf4` | **OFF in the run** | When ON: canonical contact-slot ranks instead of atomic arrival order; pull-based reverse tau sweep instead of atomic pushes; canonical cost reduction instead of fp atomics. All three change fp summation order, hence trajectories. Default was ON for exactly one commit. | Run-to-run bitwise certificate `tools/probes/sap_determinism_probe.py`. **Our reported runs are non-deterministic run-to-run.** |
| D10 | **fp32 solve-precision option** | ours | `sap_solve_precision = "fp64"` (default), preset `approx32` for Jacobians/contact linear solve | The shipped preset already uses f32 Jacobians and f32 contact linear solve; `"fp32"` additionally drops the whole solve stack and couples the optimality target to `max(1e-8, 16*eps_fp32)`. | `tools/probes/sap_fp32_floor_probe.py` (residual stagnation floor). |
| D11 | **`static_substep`** (fixed 8 Newton / 6 LS iterations) | `37663f5` | **OFF** | A different iteration budget commits a differently-converged velocity. | n/a — off. |
| D12 | **Run-ahead single march** | `2a119d2` + `e41cc070` | **OFF** | Worlds keep marching past their boundary within the action window. | Bitwise oracle probe; landed default OFF pending consent. |

---

### 6.1 Defaults that CHANGED over the history (as opposed to being introduced)

Every one of these silently redefines what "the adaptive solver" means between
two runs. All are ours; all are in
`newton/_src/solvers/sap/solver_sap_adaptive.py` unless noted.

| Parameter | Old -> New | Commit | Date | Physics? |
|---|---|---|---|---|
| `max_substeps` | `256` -> `16` | `5a84f078` | 06-27 | bounds achievable accuracy (platform re-raises it to 256) |
| `_MODE_CODES["adaptive"]` | `2` -> `1` | `6d7ecc69` | 06-27 | no (feedforward mode deleted) |
| `optimality_rel_tol` | `max(1e-3*tol, 1e-8)` = **CENIC Eq. (34)** -> hard `1e-8` | `9c9dc934` | 08-15 | **yes** — see B7(a) |
| **`dt_inner_min`** | **`1e-6` -> `1e-12`** | `9c9dc934` | 08-15 | **yes** — six decades of extra subdivision before the floor latch |
| `landed_fraction` / quantile boundary stop | present -> **removed** | `9c9dc934` | 08-15 | **yes** — a forced-completion branch that bypassed the error test is gone. Good riddance, but pre-`9c9dc934` results were produced with it. |
| `NEWTON_SAP_DETERMINISTIC` | ON -> **OFF** (and the parse sense inverted) | `050be7a9` | 08-15 | **yes** — fp summation order |
| `optimality_rel_tol` (fp32 arm) | `1e-8` -> `max(1e-8, 16 eps_f32)` ~ `1.9e-6` | `050be7a9` | 08-15 | fp64 path bit-identical |
| `NEWTON_ADAPTIVE_TAIL_COMPACT` | `== "1"` -> `!= "0"` | `718ebf7a` | 08-15 | no (unset behaviour unchanged; previously `"true"` silently disabled it) |
| **`NEWTON_SAP_ATTEMPT_CONSISTENT_R`** | **OFF -> ON** | `45095218` | 08-16 | **yes** — B3/D5. Note the flip also dropped the default argument to `os.environ.get`, so the flag is ON for any value except the literal `"0"`. |
| `line_search_variant` (sap_warp) | `"monotone_decay"` -> `"armijo_decay"` | `440e58a` | 07-03 | **yes** — B6/D6 |

**Never changed since introduction:** `tol = 1e-3`, `dt_inner_init = 0.01`,
`max_rigid_contact = 128`, `max_iterations = 30`, `contact_tau_d = 0.01`,
`contact_preset_variant = "drake"` (overridden by the platform to `approx32`),
`_divergence_threshold = 1e9`, the NaN sentinel `1e10`, all five Drake
constants, `_CEILING_MARGIN = 0.9`, `_CEILING_RELAX = 1.1`,
`_FP32_OPTIMALITY_K = 16.0`, `NEWTON_ADAPTIVE_RTOL = "2e-6"`.

### 6.2 Controller state the paper does not have: ceiling memory

`_CEILING_MARGIN = 0.9`, `_CEILING_RELAX = 1.1`
(`solver_sap_adaptive.py:115-122`), added in `9c9dc934`. A rejection at step `h`
records `dt_ceiling <- min(dt_ceiling, 0.9 h)`; every accepted step relaxes it
`dt_ceiling <- min(1.1 * dt_ceiling, dt_max)`; and the sized step is capped
`new_step <- min(new_step, dt_ceiling)`. This is **cross-boundary memory** — it
survives the boundary and is only reset by `reset()` or the debt guard.

CENIC's hysteresis device is the deadband alone. Ceiling memory is a second,
stronger, one-sided hysteresis: after a rejection a world cannot return to its
pre-rejection step for `ceil(log(1/0.9)/log(1.1)) = 2` accepted steps, and in a
sustained-contact regime it re-probes its knee only every ~2 accepted steps.
This changes the accepted-`dt` distribution — the very quantity a work-precision
plot reports — and it is ours, unvalidated, and not in the paper.

## 7. CATEGORY E — INFRASTRUCTURE (no physics)

* **Diagnostics/counters** (`sap_warp`): `factorization_count` (`37663f5`);
  `gemm_reshape_skips()`, `fused_ls_ladder_envs()`, `fused_alphamax_envs()`,
  `pack_percontact_execs()`, `fused_update_envs()`; the `_narrow_sites`
  emission tripwire and `narrow_sites_emitted` property (~30 tagged launch
  sites, `5a6cbf4`, `aac9694`).
* **Warnings and documentation** (`440e58a`, `046d40a`): zero-dissipation
  warning; the per-pair `tau_d` convention documented at its three trap sites.
  Zero code change in `046d40a`.
* **API/compat**: `SolverSAP.update_contacts` no-op writeback with the 2-arg
  signature (`345c9fe`) — this was a *build blocker*: with a contact sensor
  registered, the fixed-step SAP arm died on its first step, which is why the
  fixed-step SAP arm has no run history; Newton-1.5 target-layout remap
  (`SapTargetRemap`, `79e43bd` §5f); `zero_control` dtype fix (§5g).
* **Graph-cache keying** on every new flag (`9757f69e`, `e5154ee0`, `2574c070`,
  `00c59d4d`, `52005367`) — required so a flag flip cannot replay a stale
  captured graph.
* **Capacity/plumbing** (`newton-adaptive`, IsaacLab): `sap_max_rigid_contact`,
  contact presets, the quantile boundary stop (`ba3c6fbc`, `6709eb2d`,
  `09cbd8b1`, `bd65a146`), demand counter and crossing-batch throttle
  (`3baeb24a`).
* **Probes** (`newton-adaptive/tools/probes/`): `sap_containment_probe.py`,
  `sap_determinism_probe.py`, `sap_flag_equivalence_probe.py`,
  `sap_fp32_floor_probe.py`, `sap_newton15_construct_probe.py`,
  `sap_runahead_oracle_probe.py`.

### 7.1 Complete environment-flag roster (both repos), with defaults at HEAD

Ours, all of it. "P" = physics-visible, "S" = scheduling/perf (claimed neutral),
"T" = telemetry/forensics. Everything is at its default in the reportable run.

**`newton-adaptive/newton/_src/solvers/sap/solver_sap_adaptive.py`**

| Flag | Default | Class |
|---|---|---|
| `SAP_WARP_PATH` (`__init__.py:14`) | hardcoded absolute path | **P** (selects the entire inner solve) |
| `NEWTON_SAP_ATTEMPT_CONSISTENT_R` | **ON** (no default arg; ON unless literally `"0"`) | **P** |
| `NEWTON_ADAPTIVE_RTOL` | `2e-6` -> **ON** | **P** (error metric) |
| `NEWTON_ADAPTIVE_CEILING_RELAX` | `1.1` | **P** (dt trajectory) |
| `NEWTON_ADAPTIVE_CONTACT_REFRESH` | `1` = boundary cadence | **P** (B5) |
| `NEWTON_SAP_CONTAINMENT` | `1` = contained | **P** (failure semantics) |
| `NEWTON_SAP_SOLVE_PRECISION` | `fp64` | **P** |
| `NEWTON_SAP_DETERMINISTIC` | **`0` = OFF** | **P** (fp order) |
| `NEWTON_SAP_RUNAHEAD` | **`0` = OFF** | **P** (batch time semantics) |
| `NEWTON_SAP_RUNAHEAD_WINDOW` / `_PHASE` | `4` / `0` | **P** (window must equal decimation) |
| `NEWTON_SAP_RUNAHEAD_BATCH` / `_BATCH_AGE` | `0.5` (= `wc/2`) / `2` | S |
| `NEWTON_ADAPTIVE_TAIL_COMPACT` | ON | S |
| `NEWTON_SAP_MARCH_COMPACT` / `_WIDTH` | ON / `max(64, wc/16)` capped at `wc/2` | S |
| `NEWTON_SAP_SHARED_ASSEMBLY` | ON | S |
| `NEWTON_SAP_ADAPTIVE_GRAPH` | ON | S |
| `NEWTON_SAP_ADAPTIVE_CONDITIONAL` | ON | S |
| `NEWTON_SAP_SPREAD_LOG` / `_EVERY` | off / `10` | T |
| `NEWTON_ADAPTIVE_MARCH_LOG` / `_EVERY` | off / `48` | T |
| `NEWTON_SAP_FAILURE_DUMP` / `_REPLAY` / `_REPLAY_DT` | off | T (replay mutates `max_iterations`, but only on the raise path) |

**`sap_warp`** (defaults live outside the audited repo — this is the point of
reviewer challenge 12)

| Flag | Default | Class |
|---|---|---|
| `NEWTON_SAP_FUSED_UPDATE` | ON | **P** (D1) |
| `NEWTON_SAP_FUSED_LS` | ON | **P** (D2) |
| `NEWTON_SAP_FUSED_ALPHAMAX` | ON (AND `_fused_ls`) | **P** (D3) |
| `SAP_CUBIC_INIT` | ON (dead: `exact_root` only) | P-if-selected (D7) |
| `NEWTON_SAP_DETERMINISTIC` (3 sites) | **OFF** | **P** (D9) |
| `NEWTON_SAP_SOLVE_COMPACT` / `_LS_COMPACT` | ON / ON | S (C2) |
| `NEWTON_SAP_GEMM_RESHAPE` | ON | S (C4) |
| `NEWTON_SAP_PACK_PERCONTACT` | ON (AND `_gemm_reshape`) | S (C5) |
| `NEWTON_SAP_NARROW_V3` (2 distinct sites, same name) | ON / ON | S (C7) |

**Polarity trap worth a code comment we did not write this pass:** the
`NEWTON_SAP_*` performance flags parse as `!= "0"` (anything but `"0"` is ON)
while `NEWTON_SAP_DETERMINISTIC` parses as `== "1"` (only `"1"` is ON), and
`NEWTON_SAP_ATTEMPT_CONSISTENT_R` has **no default argument at all**. Three
parse conventions in one family.

---

## 8. IN THE PAPER, NOT IMPLEMENTED

| Paper item | Status | Consequence |
|---|---|---|
| **V-B second-order trapezoid** | Absent (grep: no `trapezoid`/`second_order` in `sap_warp/sim`) | None — the paper itself recommends against it. |
| **V-C external systems** (Rosenbrock linearization of `z`, `tau = -C v + d` from forward differences) | Absent | Stiff external controllers are integrated explicitly and force the step down instead of being absorbed implicitly. Our PD/effort-limit path (IV-E) exists, but the general external-system machinery does not. |
| **V-D static vs dynamic friction** (`mu(s)` lagged, `mu_s != mu_d`) | Absent — a single `mu` per contact, harmonically combined (`_sap_combine_mu`) | Cannot reproduce the paper's stick-slip results; a named "novel capability" of CENIC is unavailable. |
| **VI-B adaptive convergence criteria** (`eps_tol = max(kappa eps_acc, 1e-8)`, plus the Eq. 33 Newton-error criterion with `Theta`, `eta`) | Eq. (34) **was implemented** in `5a84f078` and **removed** in `9c9dc934`; Eq. (33) never implemented; cost early-exit explicitly zeroed | See B7. Costs performance, does not cost accuracy. |
| **VI-C Hessian reuse** (Eq. 35 with `N = 10`, reuse across solves and time steps) | Absent from the SAP path (`grep` for `reuse_hessian`/`hessian_reuse` in `sap_warp/sim` returns nothing; the existing `test_adaptive_hessian_reuse` covers the MuJoCo adaptive solver) | Every Newton trip refactorizes. This is one of the paper's four named optimizations and the single largest untapped speedup available to us. |
| **VI-D cubic init** | Implemented (`bd0c129`) but on a non-default line search | Effectively absent from our runs. See B6. |
| **Exact linesearch** (Newton-Raphson + bisection) | Implemented as `exact_root`, not the default | See B6. |
| **`S` mapping components to dimensionless units** | `S = I` | See B4. |
| **`k_Init = 0.1 dt_max` initial step** | Effective `k_Init = 1.0` | See B8b. Minor, self-correcting. |

---

## 9. PLAIN-LANGUAGE SUMMARY FOR A PAPER DRAFT

> **Implementation.** We implement the CENIC error-controlled integration scheme
> of Kurtz and Castro (arXiv:2511.08771) on GPU, over many parallel worlds,
> inside NVIDIA Newton. The convex per-step solve is provided by `sap_warp`, an
> open-source Warp implementation (AIVC Lab / TRI, Apache-2.0) of the SAP
> contact formulation of Castro, Permenter and Han (arXiv:2110.10107), which we
> use unmodified in its numerical core. Around it we wrote a batched
> step-doubling controller (`SolverSAPAdaptive`) that follows CENIC Algorithm 1
> per world: a full step of size `h` and two half steps of size `h/2` from the
> same state, a weighted position-only `L^inf` error estimate between the two
> results, acceptance when that estimate is within tolerance, and Drake's
> step-size law `h_new = 0.9 h sqrt(eps/e)` with the paper's deadband `[0.9,
> 1.2]` and growth cap `5.0`. The paper's warm-start scheme (VI-A) is followed
> exactly, and the full step and first half step share one assembly, as the
> paper prescribes.
>
> **Deviations from the published method.** Four are material and we state them
> plainly.
>
> *(i) Contact model.* CENIC builds on the Lagged member of the ICF family with
> Hunt & Crossley dissipation and a user-supplied, step-independent point-contact
> stiffness. We use SAP, whose normal law is a linearized Kelvin-Voigt model with
> a relaxation time `tau_d` and whose regularization is clamped from below by the
> near-rigid value `R_n = beta^2/(4 pi^2) ||W_ii||_rms`. The difference is a
> modelling difference, not a discretization difference: it does not vanish as
> the step shrinks.
>
> *(ii) The dt^-2 mechanism does not apply to our contacts.* CENIC's `dt^-2`
> stiffness scaling is stated for limit and holonomic constraints, and its
> derivation assumes the dissipation time scale tracks the step,
> `tau = (beta/pi) dt`. In our configuration `tau_d` is a fixed material
> constant (0.02 s effective per contact pair, roughly ten times the nominal
> step). The effective near-rigid stiffness is then
> `k_eff = 4 pi^2 / (beta^2 w h (h+tau))`, whose step exponent is
> `-(1 + h/(h+tau))`: `-1.09` at the production step and `-1.17` at the mean
> accepted step, against `-1.76` (`beta = 1`) or `-2` (`beta << pi`) for the
> published law. Steady-state penetration correspondingly falls linearly rather
> than quadratically in the step, and is 8-16x larger than the published estimate
> over our step range. **Any accuracy improvement we report from error control
> therefore cannot be attributed to the paper's stiffness-tightening mechanism.**
>
> *(iii) Attempt-consistent regularization.* Because our regularization depends
> on the step size on the near-rigid branch, the full step and the half steps
> would otherwise discretize different contact models, and their difference would
> not be a truncation error. We therefore evaluate the contact constitutive law
> at the attempt step for all three solves, which rescales the Delassus estimate
> by `s = D(D+tau)/(h(h+tau))` (`2.35` for the half steps at our settings) and is
> exactly the identity for the full step. This has no counterpart in the
> published method; it is required by our contact law, not by theirs.
>
> *(iv) Solver and geometry economy.* We cap the inner Newton iteration at 30 and
> treat a cap-out as a rejected step, whereas the published method relies on
> convexity to converge without discarded work; we use inexact Armijo
> backtracking rather than the published exact line search; we perform one
> collision query per outer boundary rather than two per accepted step, refreshing
> gaps, contact points and Jacobians but not the contact set within a boundary;
> we run the inner solve to a fixed `1e-8` rather than the paper's
> `max(1e-3 eps_acc, 1e-8)` and omit its second (Newton-error) convergence
> criterion; and we do not implement the paper's Hessian reuse. The first three
> trade accuracy for throughput; the last two trade throughput for accuracy. Our
> reported runs use mixed precision (float32 contact Jacobians and contact linear
> solve) where the published reference is double throughout.
>
> **Additions.** Beyond the published method we add: per-world failure
> containment (a world whose inner solve fails is not committed, is shrunk and
> retried, and does not perturb any other world in the batch — verified bitwise
> non-interfering); a step floor and a per-boundary work cap, below and beyond
> which a world's step is accepted or its remaining debt is bounded, so that every
> world reaches the shared action boundary in bounded work; a one-sided "ceiling
> memory" hysteresis on step growth in addition to the paper's deadband; and a set
> of GPU scheduling optimizations (launch-grid narrowing, live-row GEMM
> truncation, per-contact packing, kernel fusion). Three of the fusions change
> floating-point reduction order and are therefore not bitwise neutral; they are
> enumerated in the supplementary material. Runs are not deterministic
> run-to-run at default settings; a verified deterministic mode exists, at a cost.

---

## 10. WHAT A REVIEWER COULD CHALLENGE

Ranked by how much damage each does if unanswered.

1. **"You did not implement CENIC's contact model, so what exactly did you
   validate?"** We implemented CENIC's *integrator* over SAP's *contact model*.
   Every claim must be phrased as "CENIC-style error control applied to SAP",
   never as "CENIC". The paper's own text distinguishes SAP from the Lagged
   approach it builds on.
2. **"Your near-rigid law can't produce the paper's mechanism."** (B2.) Fixed
   `tau_d` halves the stiffness exponent. Pre-empt this: state the exponent, show
   the closed form, show that the measured `-1.172` matches it. Anything framed
   as "adaptive fixes penetration through dt^-2 stiffness coupling" is
   indefensible at the authored `tau_d`.
3. **"ACR changes the physics between your arms."** (B3, D5.) ACR is ON in the
   reportable run and softens the committed half-solves by 2.35x. If the fixed
   baseline does not carry the same constitutive treatment, the comparison is
   confounded. Either both arms freeze the law at the same `D`, or the delta
   must be reported.
4. **"Your error norm isn't the paper's."** (B4.) `S = I` means `eps_acc` is not
   "digits of accuracy"; it is a mixed-unit threshold. Report the per-coordinate
   budgets it implies.
5. **"One collision query per boundary is not two per step."** (B5.) Demands a
   passthrough / fast-contact control experiment, or an explicit scope limit.
6. **"You cap Newton at 30 and call the cap-out divergence."** (B9.) This
   directly contradicts the paper's claim that convexity eliminates discarded
   iterations. Report the rejection rate attributable to cap-out separately from
   the rate attributable to error.
7. **"Your controller can accept without meeting tolerance, and can drop
   time."** (B8.) Three exits: floor acceptance, floor latch (state frozen,
   clock force-advanced), and the debt guard (carried debt capped at one
   `dt_outer`, remainder discarded). All three rates are currently UNVERIFIED
   and belong in the accuracy table, not in a footnote.
7b. **"Your inner-solve tolerance is not the paper's, and you deleted the rule
   that was."** (B7.) The commit that removed CENIC Eq. (34) is on record
   (`9c9dc934`). Be ready to explain why.
7c. **"Ceiling memory is an undocumented second hysteresis."** (6.2.) It
   directly shapes the accepted-`dt` distribution that any work-precision plot
   reports, and has no counterpart in the paper.
8. **"Three of your 'neutral' optimizations change reduction order."** (D1-D3,
   all default ON.) They are not bitwise; say so, and give the flag-equivalence
   evidence rather than claiming bitwise identity.
9. **"Runs are not reproducible."** (D9.) Determinism is OFF by default. State
   it, and report seeds/variance accordingly.
10. **"An explicit `fallback_mu` can silently overwrite authored friction."**
    `sap_warp/sim/sap_helpers.py:48-58` (`_resolve_contact_shape_mu`): when
    `fallback_mu` is not `None` and all authored per-shape `mu` agree within
    `1e-7`, the entire per-shape array is replaced by the fallback. Upstream
    behaviour, but ours to disclose if we ever pass `fallback_mu`.
11. **"You compare against a single-threaded C++ reference."** The paper's
    numbers are single-CPU-thread Drake and its scenes are single-scene. Any
    throughput comparison must be scoped accordingly; the paper itself lists
    CPU/GPU parallelization as future work.
12. **"Half your solver isn't in the repository you published."** The entire
    inner convex solve is loaded off `sys.path` from an absolute default,
    `newton/_src/solvers/sap/__init__.py:14`:
    `_sap_root = os.environ.get("SAP_WARP_PATH", "/home/.../sap_warp")`. It is
    **not version-pinned**, not vendored, and its ~13 default-ON optimization
    flags live outside this repository's history. Any reproducibility statement
    must name the `sap_warp` commit (`afd5dc6` here).
13. **"Your solver silently swallows unknown arguments."**
    `SolverSAPAdaptive.__init__` ends in `**kwargs` that is accepted and
    discarded (`solver_sap_adaptive.py:1275`). A misspelled solver parameter in a
    sweep config produces no error and no warning — which means an intended
    ablation can silently not happen. This is the highest-value one-line fix
    available and it is *not* made in this pass (audit only).
14. **"Capacity overflow drops contacts silently."**
    `max_triangle_pairs = 1_000_000` is a **global** cap (not per world), and its
    own docstring says the default only fits small scenes; overflow drops mesh
    contacts by arrival order. Determinism is void whenever any cap saturates.
    Report the saturation counters with any result.
15. **"You changed the precision preset from the solver's own default."** The
    reportable run uses `approx32` (f32 Jacobians and f32 contact linear solve),
    not the `drake` fp64 preset the solver defaults to. The paper's reference is
    fp64 C++.

---

## 11. APPENDIX — COMMIT LEDGER

Every commit in scope, in order, with its bucket. Author is `mardigiorgio`
throughout unless the row says otherwise.

### 11.1 `sap_warp` (17 commits, `c0c861c..afd5dc6`)

| Commit | Date | Subject | Files (+/-) | Buckets |
|---|---|---|---|---|
| `431adf2` | 06-11 | initial commit — **upstream, f1shel** | whole repo | the SAP solver itself |
| `62da4a6`/`6907f9c`/`c0c861c` | 06-11 | docs — **upstream, yunuo/cffjiang** | docs | — |
| `37663f5` | 06-27 | SAP step-doubling support + factorization diagnostics | 3 files, 363/119 | **C** (scalar `dt` -> per-world `dt[env]`), **E** (`factorization_count`, `static_loop`), **D** (`static_substep`, OFF) |
| `bd0c129` | 06-27 | Cubic-Hermite guess for exact-root LS (CENIC VI-D) | 1 file, 30/1 | **A14** + **D7** (dead at our default) |
| `440e58a` | 07-03 | Warn on zero dissipation; default to `armijo_decay` | 1 file, 15/1 | **D6** (default flip + 3 tolerances), **E** (warning) |
| `046d40a` | 07-03 | Document per-pair `tau_d` at its three trap sites | 2 files, 12/2 | **E** (comments only) |
| `79e43bd` | 08-15 | Snapshot solve compaction + determinism state | 6 files, 1588/107 | **C** (env-list compaction), **D** (determinism ON, ACR kernels, `monotone_decay` rewrite = D4), **E** (Newton-1.5 remap, tripwires) |
| `5a6cbf4` | 08-15 | Complete narrow-grid conversion; add ACR scaling | 3 files, 392/65 | **C** (10 more kernels list-indexed), **D** (determinism default -> OFF), **E** (28 tripwire tags) |
| `fe98f46` | 08-15 | Narrow blocked-Cholesky launches to the env grid | 2 files, 221/2 | **C3** |
| `b1e48a3` | 08-15 | Shared-assembly reuse for half-1 solves | 3 files, 117/79 | **C6** + **A12** |
| `27dcada` | 08-15 | Truncate contact-Hessian GEMM pair at live rows | 1 file, 215/0 | **C4** |
| `f49b20b` | 08-16 | Fuse armijo backtracking ladder into one kernel | 1 file, 366/19 | **D2** |
| `a79539a` | 08-16 | Fold alpha-max rung into fused ladder | 2 files, 599/52 | **D3**; adds 2 of the 8 R sites (verified identical) |
| `1ff0ea0` | 08-16 | Rewrite GEMM pack per-contact; hoist `j_flat` | 1 file, 275/25 | **C5** |
| `3bff5c1` | 08-16 | Fuse post-commit update eval into one kernel | 1 file, 657/51 | **D1**; adds 1 of the 8 R sites (verified identical) |
| `aac9694` | 08-16 | Narrow trip-cadence kernels to live env lists | 2 files, 217/18 | **C7** |
| `2a119d2` | 08-16 | Run-ahead adopt/anchor contact split | 1 file, 789/1 | **C8** (opt-in, OFF) |
| `345c9fe` | 08-16 | `SolverSAP` no-op contact-sensor writeback | 1 file, 19/3 | **E** — build-blocker fix; this is why the fixed-step SAP arm had no run history |
| `afd5dc6` | 08-16 | Host-sync-free masked runtime-state reset | 1 file, 39/0 | **C9/E** |

### 11.2 `newton-adaptive` (20 commits in `newton/_src/solvers/sap/`, 5 more in `adaptive_boundary.py`)

| Commit | Date | Subject | Buckets |
|---|---|---|---|
| `5d7c1c73` | 06-26 | Add `SolverSAPAdaptive` (convex SAP-CENIC) + shared controller | **A6-A11, A13, A16, A17** — the CENIC implementation itself (even/global tiling at this point) |
| `5a84f078` | 06-27 | Refine SAP/MuJoCo adaptive solvers | true per-world `dt`; the Drake constants and the accept law authored here; **CENIC Eq. (34) implemented**; `max_substeps` 256->16 |
| `6d7ecc69` | 06-27 | Remove global dt, even tiling, feedforward | **E** |
| `cee43fc0` | 07-01 | Fix SAP NaN detection | **A13** — per-component NaN flag (`fmaxf` returns the non-NaN operand) |
| `98fb7c18` | 07-02 | Correct unmeasured claims in docs | **E** (zero executable lines) |
| `ba3c6fbc`, `6709eb2d` | 08-06 | Shared quantile boundary stop | **E**/**D** (its forced-completion bypassed the error test) |
| `09cbd8b1`, `bd65a146` | 08-07 | Adopt the quantile stop; skip forced completion when nothing abandoned | **D** |
| `9c9dc934` | 08-15 | **Campaign start.** Ceiling memory, mixed atol/rtol, velocity finiteness, sliver exemption, debt guard, dt histogram, containment, tail/march compaction, deterministic collision, forensics; **quantile stop removed** | **B4, B7a, B8, 6.2, D8, D9** — the single most consequential commit in the history (+1971/-149) |
| `050be7a9` | 08-15 | Narrow-grid v2, ACR and fp32 opt-ins; **determinism OFF** | **D9, D10** |
| `e36ab5f6` | 08-15 | Wire half-1 shared assembly | **C6** |
| `4fee3b12`, `9f78fab2` | 08-15 | Key GEMM truncation; assert Cholesky narrow sites | **E** |
| `718ebf7a` | 08-15 | Fix tail-compact flag read | **E** (parse-sense fix) |
| `45095218` | 08-16 | **Flip attempt-consistent R to default ON** | **B3 / D5** |
| `9757f69e`, `e5154ee0`, `2574c070`, `00c59d4d`, `52005367` | 08-16 | Key/gate the five `sap_warp` optimization flags in both graph caches | **E** (required: a flag flip must not replay a stale captured graph) |
| `e41cc070`, `8f3ef7e3` | 08-16 | Run-ahead single-march + its oracle probe | **D12** (OFF) |
| `3baeb24a` | 08-16 | Demand counter and crossing-batch throttle | **E** + a run-ahead debt-guard correctness fix |
| `b232b707` | 08-07 | Point the SAP tests at the sibling `sap_warp` checkout | **E** — and evidence for reviewer challenge 12 |

---

## 12. PROVENANCE OF THIS DOCUMENT

* Paper text: `arxiv.org/html/2511.08771v1` (CENIC) and
  `arxiv.org/html/2110.10107v2` (SAP), fetched 2026-08-16, extracted to
  `p36_cenic.txt` / `p36_sap.txt` in the pass-36 scratchpad. All equations
  quoted from the LaTeXML `alttext`, i.e. the authors' own LaTeX.
* Code: `sap_warp` @ `afd5dc6` and `newton-adaptive` @ `80d13a9a`, both clean
  working trees at the time of the audit.
* Every closed-form number in Sections 3.1, B2 and B3 is reproducible from the
  formulas printed alongside it; no new script was committed this pass, by
  design (the rails for this pass forbade code changes).
* **Not verified this pass (explicit list):**
  1. Any C-class bitwise claim — these rest on earlier flag-equivalence and
     oracle probe runs, not on this pass.
  2. The floor-acceptance rate, the floor-latch rate, and the `_debt_guard` fire
     count in the reportable run (B8 a/b/c).
  3. The share of rejections attributable to inner-solve cap-out rather than to
     error (B9).
  4. Whether `sap_warp`'s optimality test applies the paper's `D = diag(M)^{-1/2}`
     scaling to the gradient norm (B7 c).
  5. The MEASURED numbers quoted from earlier passes and re-used here as
     corroboration — the `-1.172` stiffness exponent, the 11% near-rigid
     fraction, and the ~2.5e4 N/m crossover. Each is cited as MEASURED and each
     is *predicted* by a closed form printed alongside it, which is why they are
     used; none was re-run this pass.
  6. Whether upstream `sap_warp` has advanced beyond `c0c861c` since the last
     fetch.
* **Corrected in this pass** (numbers previously on record that this audit does
  not reproduce): the ~98 N/m figure for the misplaced-`4 pi^2` crossover
  (derivation here gives ~21 N/m at `beta = 1`); and the flat "`s ~ 2.34`" for
  ACR (derivation here gives `s = 2(D+tau)/(D/2+tau)`, i.e. 2.05-2.35 over the
  campaign's step range, with 2.345 only at `D = dt_outer`).
