# SAP SOLVER PROVENANCE AUDIT — CENIC (arXiv:2511.08771) vs. THIS IMPLEMENTATION

Pass 36 of the SAP campaign. 2026-08-16. **Source + literature only: zero GPU
processes started** (`nvidia-smi` polled read-only throughout; the only compute
process on the card was the live 4000-iteration training run, PID 2271848).
**No code was changed this pass.** The only files written are this document and
the ledger entry.

**Amended by pass 37, 2026-08-16/17.** Pass 36 closed Section 5 with the
caveat that none of its physics-neutrality claims had been re-measured. Pass
37 measured them and **rewrote Section 5 end to end**: the "proof class
actually on record" column is gone, replaced by the test actually run, the
result, and the evidence class earned. Section 12.1 records the scope
corrections that verification turned up. Sections 1-4 and 6-11 are pass-36
text, unamended. The source verified by pass 37 is byte-identical to the
source pass 36 audited (`git diff --stat 80d13a9a..d16db463` touches only
`tools/*.md`).

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

**Rewritten by pass 37 (2026-08-16/17): the "proof class actually on record"
column has been replaced with measured evidence.** Pass 36 closed this
section with the correct caveat that none of the C-class bitwise claims had
been re-measured. They have been. Every statement below was produced on the
bytes at `newton-adaptive d16db463` / `sap_warp afd5dc6`; `git diff --stat
80d13a9a..HEAD` touches only `tools/*.md`, so **the source verified here is
byte-identical to the source Section 4 audits** — the audit and its
verification are not describing different trees.

All of these items are ours, all live in `sap_warp`, and all default **ON**
unless noted.

### 5.0 Evidence classes, and why they must not be conflated

| Class | Meaning | What it licenses |
|---|---|---|
| **BITWISE-VERIFIED** | A fresh run at this HEAD, under `NEWTON_SAP_DETERMINISTIC=1`, produced byte-identical committed trajectories AND byte-identical controller carry state with the switch ON and OFF, against a reference recomputed in the same invocation and guarded by a reference-vs-reference-repeat oracle. | "This optimization does not change the physics." |
| **SOURCE-EXHAUSTIVE** | No run-level A/B exists at this HEAD because the alternative path was **deleted, not retained**. The claim rests on a mechanical, exhaustive reading of the current bytes — a diff, not a judgement. | "This optimization cannot change the physics, by construction." The residual risk is compiler-level and is named. |
| **BOUNDED** | Not bitwise, and never will be: the change reorders floating-point reductions. The effect is measured on a real contact-violent trajectory against reference scales measured in the same session. | "This perturbs trajectories by no more than X", with X stated. |
| **PHYSICS-VISIBLE** | Not neutral, not claimed neutral. | Nothing. A design decision with stated semantics. |
| **UNVERIFIED** | Stated, not established. | Nothing. |
### 5.1 The verification table

Every C and D item, the test actually run this pass, the result, and the
evidence class earned. Chain and censuses: `p37_progress.txt`; per-gate logs
`p37_g1_mirror.log` .. `p37_g8_prodpair.log`.

| # | Change | Commit | Default | Test actually run this pass | Result | Class earned |
|---|---|---|---|---|---|---|
| C1 | Per-world `dt` threading (scalar -> `dt[env]`) | `37663f5` | n/a | (a) mirror-pair env-privacy probe, det=1, per-world `dt` spread enforced by a vacuity guard; (b) exhaustive classification of every `dt`-touching line in the landing diff | **PASS.** 8 mirror pairs x 8 boundaries x 10 recorded fields, bitwise identical, at three tolerances (1e-7/1e-8/1e-9) with all six vacuity guards armed (10-12 distinct per-world `dt` values, rejections exercised). Every `dt`-touching changed line is one of three purely-indexing forms. | **PASS.** 8 mirror pairs x 8 boundaries x 10 recorded fields, bitwise identical, at three tolerances (1e-7/1e-8/1e-9) with all six vacuity guards armed (10-12 distinct per-world `dt` values, rejections exercised). Every `dt`-touching changed line is one of three purely-indexing forms.C |
| C2 | Env-list compaction (`NEWTON_SAP_SOLVE_COMPACT`, `_LS_COMPACT`) | `79e43bd`, `5a6cbf4` | ON | flag-equivalence cells `solve-compact`, `solve-compact-graph`, `ls-compact`, `ls-solve-compact`, `boundary-solve-compact`, `boundary-ls-compact`, judged bitwise against a same-family reference; plus the mirror-pair probe | **PASS.** 6 flag-equivalence cells bitwise; and in the aggregate run the eight scheduling flags together are bitwise (see 5.5). | **PASS.** 6 flag-equivalence cells bitwise; and in the aggregate run the eight scheduling flags together are bitwise (see 5.5).C |
| C3 | Blocked-Cholesky narrowing | `fe98f46` | unconditional | (a) flag-equivalence `march-compact` family (the only runtime toggle of the launch width); (b) mechanical diff of both listed kernels against their retained non-listed twins | **PASS.** `march-compact`, `-graph`, `-conditional` bitwise against `boundary`, both branch bodies executed. Diff of both listed kernels vs their retained twins: **zero arithmetic lines differ**. | **PASS.** `march-compact`, `-graph`, `-conditional` bitwise against `boundary`, both branch bodies executed. Diff of both listed kernels vs their retained twins: **zero arithmetic lines differ**.C |
| C4 | Live-`k` GEMM truncation (`NEWTON_SAP_GEMM_RESHAPE`) | `27dcada` | ON | flag-equivalence cells `gemm-reshape`, `boundary-gemm-reshape`, `gemm-full-stack`; plus a closed-form identity of the pack's and GEMM's k-tile bounds on current bytes | **PASS.** 3 cells bitwise, engagement counter 5.5e8 tile-skips in the aggregate run. Pack bound and GEMM bound are the same expression, so no truncated tile is ever read. | **PASS.** 3 cells bitwise, engagement counter 5.5e8 tile-skips in the aggregate run. Pack bound and GEMM bound are the same expression, so no truncated tile is ever read.C |
| C5 | Per-contact pack + `j_flat` hoist (`NEWTON_SAP_PACK_PERCONTACT`) | `1ff0ea0` | ON | (a) flag-equivalence cells `pack-percontact`, `boundary-pack-percontact`, `pack-full-stack`; (b) the `j_flat` hoist's J-invariance-within-a-solve premise re-measured by byte-comparing `J`, `j_flat` and the live contact count across Newton trips inside every solve | **PASS.** 3 cells bitwise (2.94e6 pack execs engaged). J-invariance re-measured: **101 trip-pairs over 38 multi-trip solves, 152 trips, zero mismatches** in `J`, `j_flat` or the live contact count. | **PASS.** 3 cells bitwise (2.94e6 pack execs engaged). J-invariance re-measured: **101 trip-pairs over 38 multi-trip solves, 152 trips, zero mismatches** in `J`, `j_flat` or the live contact count.C |
| C6 | Shared assembly for half-1 (`NEWTON_SAP_SHARED_ASSEMBLY`) | `b1e48a3` | ON | flag-equivalence cells `shared-assembly`, `boundary-shared`, `shared-full-stack`; plus a both-ends check of the caller contract on current bytes | **PASS.** 3 cells bitwise, 2996 reuse executions in the aggregate run. Caller contract holds at both ends: the skipped region uses no `dt`, and half-1 receives the full solve's own state/contacts/control/mask objects. | **PASS.** 3 cells bitwise, 2996 reuse executions in the aggregate run. Caller contract holds at both ends: the skipped region uses no `dt`, and half-1 receives the full solve's own state/contacts/control/mask objects.C |
| C7 | Narrow-v3 (`NEWTON_SAP_NARROW_V3`, two sites sharing the name) | `aac9694` | ON | flag-equivalence `narrowv3` family (`narrowv3-ref`, `-ref-repeat`, `narrowv3`, `-graph`, `-conditional`) judged bitwise against a same-stack pinned-off reference | **PASS.** `narrowv3`, `-graph`, `-conditional` bitwise against a same-stack pinned-off reference; 20-27 narrowed launch sites emitted. | **PASS.** `narrowv3`, `-graph`, `-conditional` bitwise against a same-stack pinned-off reference; 20-27 narrowed launch sites emitted.C |
| C8 | Run-ahead ADOPT/ANCHOR split (`NEWTON_SAP_RUNAHEAD`) | `2a119d2`, `e41cc070` | **OFF** | dedicated bitwise oracle probe, plus the flag-equivalence `runahead` family | **PASS with a correction.** The oracle is bitwise for ON-repeat and for batch-vs-solo isolation (8 worlds x 2 window edges). **ON-vs-OFF is NOT bitwise**: positions agree exactly (max |dq| 0.000e+00) but velocities differ by max |dqd| 3.7e-09 (f32 clock-rebase rounding of landing slivers). | **PASS with a correction.** The oracle is bitwise for ON-repeat and for batch-vs-solo isolation (8 worlds x 2 window edges). **ON-vs-OFF is NOT bitwise**: positions agree exactly (max |dq| 0.000e+00) but velocities differ by max |dqd| 3.7e-09 (f32 clock-rebase rounding of landing slivers).C |
| C9 | Host-sync-free masked runtime-state reset | `afd5dc6` | n/a | exhaustive read of what the unmasked reset clears vs what the step path consumes, plus a call-site census | **PASS, and out of scope for the adaptive arm.** The only step-path-consumed field the unmasked reset clears is `_contact_solve_v_guess_active`, which the masked variant clears with an identical value write; the other ten are write-only host bookkeeping never read. Its sole call site is fixed-arm only. | **PASS, and out of scope for the adaptive arm.** The only step-path-consumed field the unmasked reset clears is `_contact_solve_v_guess_active`, which the masked variant clears with an identical value write; the other ten are write-only host bookkeeping never read. Its sole call site is fixed-arm only.C |
| D1 | Fused update eval (`NEWTON_SAP_FUSED_UPDATE`) | `3bff5c1` | **ON** | bounded: fusions-OFF vs shipped on the Trossen rig, det=1, against three reference scales measured in the same session | Aggregate run, 512 worlds x 150 steps: divergence onset step 13, max `joint_q` Linf **4.75e-2** over the horizon. Ensemble identical to the shipped arm (accepted-error median and p90 ratios 1.000, substeps ratio 1.001). | **BOUNDED** |
| D2 | Fused armijo ladder (`NEWTON_SAP_FUSED_LS`) | `f49b20b` | **ON** | same run as D1 (the three fusions are toggled together; they are not separable at the trajectory level) | as D1 (measured jointly) | **BOUNDED** |
| D3 | Folded alpha-max (`NEWTON_SAP_FUSED_ALPHAMAX`) | `a79539a` | **ON** | same run as D1 | as D1 (measured jointly) | **BOUNDED** |

**Why D1-D3 are tested together and not separately.** `_fused_alphamax` is
gated on `_fused_ls` (`contact_solve.py:5452-5453`), so the two cannot be
varied independently, and all three change the same class of quantity (a
reduction total feeding a convergence or accept decision). A per-flag
trajectory bound would therefore report three correlated numbers from one
mechanism. What the run does establish is the bound on **all three at once**,
which is the quantity a reader needs: it is the difference between the
shipped fast path and the fusion-free path.

**C8 correction.** Pass 36 recorded C8's evidence as a "bitwise oracle
probe". That is right about what the probe certifies bitwise — the ON-repeat
determinism and the batch-vs-solo world isolation — but wrong about ON-vs-OFF,
which the probe itself reports as *not* bitwise: committed positions agree
exactly, committed velocities differ by up to 3.7e-9 from f32 clock-rebase
rounding of landing slivers. C8 is a BOUNDED item, not a bitwise one. It is
default OFF, so nothing in the reportable configuration depends on this.

**A second isolation certificate worth quoting.** The same run-ahead probe
establishes, bitwise, that **a world's committed rows in a 8-world batch equal
its rows from a solo run** at every window edge. That is an independent
batch-vs-solo confirmation of the env-privacy premise the whole C-class rests
on, obtained on a different rig from the mirror-pair probe.

**Why the D1-D3 fusions cannot change the contact law, only the route to it.**
Three of the eight contact-`R` construction sites were added by these
fusions. A structural digest of all eight sites on current bytes pairs each
added site with an upstream twin of identical arithmetic structure:
`contact_solve.py:948` (upstream) with `:2760` (`3bff5c1`, D1), and
`sap_helpers.py:2400`/`:2587` (upstream, f64/f32) with `:2475`/`:2660`
(`a79539a`, D3). Likewise the fused ladder calls the *same* `sap_armijo_ok`
accept helper the launch chain calls (`contact_solve.py:3928-3947, 4261-4280`
vs `:4435-4436, 4495-4496, 4515, 4545`). The constitutive law, the accept
rule, the ladder values and the tolerances are the chain's; what differs is
the order in which floating-point contributions are summed.
### 5.2 Item detail — what was actually established, and how

**C1 — per-world `dt` threading.** The claim has two halves and they earn
different classes.

*Threading* (every world reads its own `dt`; nothing leaks across the env
axis) is measured by `tools/probes/sap_neutrality_mirror_probe.py`. The scene
is built as MIRROR PAIRS: world `i` and world `N-1-i` get byte-identical
initial conditions but occupy different env indices with different
neighbours. Physics is env-local, so the pair must agree bitwise. A
wrongly indexed `dt` read hands world `i` one neighbour's step and world `N-1-i`
a *different* neighbour's, so the mirror breaks; likewise any cross-env leak
in the list-indexed kernels. The probe's vacuity guards require live
contacts, a certified rejection, per-world `dt` spread WITHIN a boundary, and
distinct per-world substep counts — without `dt` spread the threading claim
is untested and the probe fails rather than passing vacuously.

*Uniform reduction* (a scalar `dt` reproduces the deleted scalar-kernel path)
has **no run-level A/B at this HEAD**: `37663f5` converted the kernels in
place; the scalar signatures were deleted, not retained. What is established
is mechanical and exhaustive. `git show 37663f5 -U0 -- sim/` classifies every
`dt`-touching changed line, and each is one of exactly three forms:
`dt: scalar` -> `dt: wp.array(dtype=scalar)` in a signature, `scalar(dt)` ->
`scalar(dt[env])`, or `dt` -> `dt[env]`. No arithmetic expression, operand
order or dtype changed, and `env` is the same index every other per-env array
in the same kernel uses (`pd_limit[env, i]`, `contact_tau_d[env, c]`). The
fill is value-preserving at the dtype the kernel reads:
`contact_solve._dt_world` is allocated at `self.solve_dtype`
(`contact_solve.py:5236`) and the scalar path is `fill_(float(dt))`
(`:9111`), so `scalar(dt[env])` is the same rounding of the same Python
double that `scalar(dt)` was — there is no intermediate cast to round twice.
`solver_sap._dt_world` and `free_motion._dt_world` are `float64`, matching
their kernels' `wp.float64` scalar parameters.

Residual risk, named: a `dt` broadcast identical for every world (every
kernel reading `dt[0]`) is mirror-symmetric and the probe would not catch it;
the source classification above is what excludes it.

**C2 — env-list compaction.** The audit described this as "~25 kernels
converted to `(env_idx, env_n)` list indexing", which understates how strong
the neutrality argument is. ON and OFF **run the same kernel binaries**: with
the switch off, the consumers read a *static identity list at full width*
(`_identity_env_idx` / `_full_env_n`, `contact_solve.py:5512-5531`); with it
on, they read the device-built active list. The only difference between the
arms is which envs are visited and in what order. The bitwise claim therefore
reduces exactly to env-privacy of the kernel bodies plus the deadness of
de-listed envs' skipped writes — which is what the mirror-pair probe and the
flag-equivalence arms test.

**C3 — blocked-Cholesky narrowing.** This item has **no escape hatch at
HEAD**: `factorize_masked_listed` / `solve_masked_listed` are called
unconditionally (`contact_solve.py:7633, 7644`); the only thing that varies
is `self._env_grid`, which march compaction narrows. The pre-`fe98f46` masked
twins are therefore *unreachable at runtime*, and no ON/OFF A/B of the
conversion itself exists. Two things are established instead. First, a
mechanical diff of each listed kernel against its retained non-listed twin
(both twins are still in `blocked_cholesky.py`) shows the complete set of
changed code lines is: the rename, two added list parameters
(`env_idx`, `env_n`), and a four-line prologue replacing
`env, tid = wp.tid()` with `i_env, tid = wp.tid(); if i_env >= env_n[0]:
return; env = env_idx[i_env]`. **Zero arithmetic lines differ.** Second, the
flag-equivalence probe's march-compact arms exercise the listed kernels at
two different grid widths and judge them bitwise — which is the available
test of "grid width does not change results".

**C4 — live-`k` GEMM truncation.** Established in closed form on the current
bytes, and the two bounds are the same expression. The bounded pack skips
k-tiles with `k_tile >= tiles_live`, `tiles_live = (count_live*3 + tile_k -
1)//tile_k` (`contact_solve.py:1659-1662`). The bounded GEMM walks
`k0 < rows_stop`, `rows_stop = tiles_live * tile_k` clamped to
`padded_contact_rows` (`:1729-1731`), i.e. exactly k-tiles
`0 .. tiles_live-1` — precisely the tiles the pack wrote. There is no stale
read. Against the full walk, the extra tiles the full GEMM visits hold only
pack-written zeros (the per-element guard `c < count` yields `0` for both
operands), and `acc + J^T·0 = acc` exactly in IEEE arithmetic, in the same
ascending accumulation order. Residual risk, named: the sign of a zero can
differ in principle; the flag-equivalence `gemm-reshape` arms are what
exclude it empirically.

**C5 — per-contact pack + `j_flat` hoist.** Two halves again.

*The pack* keeps the `gj` expression verbatim. The per-row form is
`g[r,0]*contact_jac[env,c,0,d] + g[r,1]*contact_jac[env,c,1,d] +
g[r,2]*contact_jac[env,c,2,d]` (`:1686-1690`); the per-contact form stages
the three loads in registers and evaluates
`g[r,0]*jv0 + g[r,1]*jv1 + g[r,2]*jv2` (`:1839-1841`) — same operands, same
values, same left-to-right association. Coverage matches too: for a live
contact `k0c+2 = 3c+2 <= 3*count_live - 1 < rows_live_stop`, so the
three-row write can never cross the live boundary, and padding dofs
(`d >= dof_per_env`) take the zero branch in both forms exactly as the
per-row guard does. Residual risk, named: register staging could in principle
let the compiler contract multiply-adds differently; that is what the
`pack-percontact` probe arms measure rather than assume.

*The `j_flat` hoist* rests on a precondition the audit correctly flagged as
ASSERTED: that `J` (and therefore `j_flat`, a pure guarded repack of `J`) and
the live contact count are fixed within a solve. That precondition is
re-measured this pass by `p17_j_invariant_probe.py`, which byte-compares the
device buffers across Newton trips inside each solve on a friction march with
mode switching, and fails as vacuous if no solve runs two or more trips.

**C6 — shared assembly for half-1.** This is the one C-class item whose
correctness is a *caller contract*, and the caller is ours, so the contract
is verified at both ends on current bytes. Callee: everything the reuse path
skips is dt-independent — `contact_jacobian.compute` forwards `dt` only to
`free_motion.compute` and then returns early at `if reuse_assembly:`
(`contact_jacobian.py:2635`), and `grep` over the entire skipped region finds
no use of `dt`. Caller: `solver_sap_adaptive.py:2352-2362` calls half-1 with
the *same objects* the full solve received — `self._state_cur`,
`self._sap_control`, `self._sap_contacts`, `world_active=wa` — the only
differing argument being `self._dt_half` and the warm-start guess, neither of
which is an assembly input. Only half-1 qualifies; half-2 anchors at the
midpoint state and keeps its own assembly.

**C9 — host-sync-free masked runtime-state reset.** Two findings, one of them
a scope correction the audit should carry.

*Equivalence.* The unmasked `reset_runtime_state` clears eleven fields
(`solver_sap.py:1038-1051`); the masked variant clears one,
`_contact_solve_v_guess_active`. That is sufficient because it is the **only
one of the eleven that anything in the step path consumes**: it is passed
into the solve at `:1532`, while `sim_time`, `frame_id`, `last_*` and
`_has_contact_solve_v_guess` are written at `:1498, 1568-1569, 1575,
1579-1580` and **never read** anywhere in `solver_sap.py` or
`contact_solve.py`. The race is benign by construction — every firing thread
writes the same value `0` to the same slot `0`.

*Scope correction.* `reset_runtime_state_masked` has exactly one call site,
`mjwarp_manager.py:910`, and it sits inside `if cls._sap and not
cls._adaptive`. **C9 is on the fixed-step SAP path only; it never executes in
the SAP-adaptive arm**, so it cannot affect any adaptive result, reportable
or otherwise.
### 5.3 What "OFF" actually is, per item

An ON/OFF bitwise test only means what the OFF arm is. These are not the same
kind of switch, and the audit previously did not distinguish them. Established
by reading each landing commit's diff and each flag's use site on current
bytes.

| # | What the OFF arm runs | Kernel identity across the arms |
|---|---|---|
| C2 | a **static identity list at full width** (`_identity_env_idx` / `_full_env_n`, `contact_solve.py:5512-5531`) | **SAME kernels**; only the visited env set and its order differ |
| C3 | nothing — the listed twins are called unconditionally (`:7633, 7644`); only `self._env_grid` varies, via march compaction | **SAME kernels**, two launch widths. The pre-`fe98f46` masked twins are retained in `blocked_cholesky.py` but unreachable at runtime |
| C4 | the legacy padded pack + full GEMM, **byte-untouched**: `27dcada` is `215 insertions / 0 deletions` — a pure addition of a second, bounded pair | different kernels; the legacy pair was not edited at all |
| C5 | the per-row bounded pack | different kernels, legacy retained and reached by a flag branch |
| C6 | the assembly launches are emitted rather than skipped | **SAME kernels**; one arm simply emits more launches |
| C7 | a **static identity list at full width** (`:6162`, `upd_idx, upd_n, upd_grid = self._identity_env_idx, self._full_env_n, self.num_envs`) | **SAME kernels**; the conversion was in place, exactly as C2 |
| C8 | the per-boundary contact stream (the default) | different path, opt-in, default OFF |
| C9 | nothing — there is no OFF, and the call site is fixed-arm only | n/a on the adaptive path |
| D1, D2, D3 | the retained launch chain, reached by a flag branch | different kernels, legacy retained |

Two consequences worth stating explicitly:

* For **C2 and C7 the ON/OFF bitwise claim is not a claim about two
  implementations of the same math** — it is a claim that a kernel's output
  for a given env does not depend on which *other* envs share its launch, nor
  on its position in the list. That is a strictly weaker and more checkable
  proposition, and it is what the mirror-pair probe attacks directly.
* For **C4 the OFF arm is upstream-equivalent code that the landing commit
  never touched**, which makes it the strongest OFF arm in the set.
### 5.4 The contribution table (speedups cited from the ledger, not re-measured)

Every speedup below is quoted from the wallclock ledger entry that landed the
item, at the pass and rig stated. **Pass 37 re-measured neutrality, not
speed**; no timing number here was produced this pass.

| # | Contribution | What it does | Measured speedup (ledger, pass cited) | Neutrality evidence class |
|---|---|---|---|---|
| C2 | Env-list compaction (`NEWTON_SAP_SOLVE_COMPACT`, `NEWTON_SAP_LS_COMPACT`) | ON and OFF run the SAME kernels; only the `(env_idx, env_n)` list differs — a device-built active list vs a static identity list at full width (`contact_solve.py:5512-5531`) | converged-env compaction **3.7x** (ledger "Landed and certified"); LS-compact alone **+0.8%** whole-run @1024x8, and disabling it costs **14.1% whole-run / 19.6% late** because march-compact hard-requires it (pass 5) | **BITWISE-VERIFIED** |
| C3 | Blocked-Cholesky narrowing | listed twins of the masked factorize/solve launch at the env-grid budget instead of the full batch | late-window per-substep **1.106 (10.6%)**, rising with tail depth to **1.233** at it7 (pass 1) | **BITWISE-VERIFIED** + SOURCE-EXHAUSTIVE |
| C4 | Live-`k` contact-Hessian GEMM truncation (`NEWTON_SAP_GEMM_RESHAPE`) | bounded pack skips k-tiles past the env's live contact rows; bounded GEMM stops its ascending k-walk at the same bound | **the campaign's largest single win**: microbench **x1.96**; det=1 A/B whole-run **x1.611**, late-window **x1.547**; production vs the pass-7 baseline it0 **-48.8%** … it7 **-39.9%** (pass 8) | **BITWISE-VERIFIED** |
| C5 | Per-contact read-once GEMM pack + `j_flat` hoist (`NEWTON_SAP_PACK_PERCONTACT`) | one `(contact, dof)` work item loads the contact's three Jacobian rows and its G block once for all three `gj` rows; the trip-invariant `j_flat` half moves to one launch per solve | landed jointly with C-D3: combined A/B ms/substep **3.119 vs 4.229 = -26.3%** whole-run, **-26.6%** late (pass 17) | **BITWISE-VERIFIED** |
| C6 | Shared assembly for half-1 (`NEWTON_SAP_SHARED_ASSEMBLY`) | the first half-step reuses the full step's dt-independent assembly (rigid ID, tau, mass matrix + factorization, body/contact Jacobians, Delassus weights) — CENIC V-A's own prescription | **speed-neutral within noise** (late-window per-substep OFF/ON 0.979); kept as pure work-deletion with larger potential on assembly-heavy scenes (pass 3) | **BITWISE-VERIFIED** |
| C7 | Narrow-v3 list-indexed trip-cadence launches (`NEWTON_SAP_NARROW_V3`) | fused-update, the serial LS-direction chain and the per-attempt contact scatter run through the world-active list at the env-grid budget | per-substep whole-run **-5.1%**, late-3 **-3.9%**; deep slab **4.4293 -> 3.7557 ms (-15.2%)** (pass 21) | **BITWISE-VERIFIED** |
| C8 | Run-ahead ADOPT/ANCHOR contact split (`NEWTON_SAP_RUNAHEAD`, **default OFF**) | worlds crossing a boundary inside the action window keep marching instead of parking | demand-normalized **-4% to 0** per accepted substep at the plateau (pass 23) — which is why it landed default OFF | **BOUNDED** (positions bitwise, |dqd| <= 3.7e-9); default OFF |
| C9 | Host-sync-free masked runtime-state reset | replaces a per-boundary host read of the reset mask (a full device sync) with a one-kernel device-side test | not separately A/B'd; deletes one full device synchronization per reset boundary | **SOURCE-EXHAUSTIVE**; inert on the adaptive path |
| D1 | Fused post-commit update eval (`NEWTON_SAP_FUSED_UPDATE`) | one tiled kernel replaces the whole committed-point chain (projection, G block, `J^T gamma`, model terms, gradient, all three convergence norms) | ms/substep ON **2.948 vs 3.258 = -9.5%** whole-run, **-9.3%** late-3 (pass 18) | **BOUNDED** (jointly 4.75e-2, ~27x below run-to-run nondeterminism) |
| D2 | Fused armijo ladder (`NEWTON_SAP_FUSED_LS`) | one tiled kernel walks each env's alpha ladder in place; trials advance along the ray `vc0 + alpha*dvc` | **the campaign's second-largest single win**: slab GPU **7.81 vs 13.7 ms/slab = -43%** (pass 15) | **BOUNDED** (as D1) |
| D3 | Alpha-max rung folded into the ladder (`NEWTON_SAP_FUSED_ALPHAMAX`) | rung 0's cost *and* derivative computed in-kernel by the analytic ray form, deleting the per-trip trial launch chain | jointly with C5: **-26.3%** whole-run, **-26.6%** late (pass 17) | **BOUNDED** (as D1) |

**Campaign aggregate (ledger, pass 31):** ~78 s/iter pre-campaign (det ON) ->
35.35 s/iter at pass 14 -> a plateau of **12.9-15.1 s/iter @1024 envs** (best
estimate 13-14). Same-config draws move +-8% because substep demand is
draw-dependent, so the plateau is a band, not a point.

### Two contributions that are NOT speed, and NOT neutral

These belong in the document as designed features with stated semantics. They
are physics-visible by nature and must not be folded into the neutral set.

| # | Contribution | What it is for | Why it is physics-visible |
|---|---|---|---|
| D5 | **Attempt-consistent regularization (ACR)** | On the near-rigid branch our regularization depends on the step size, so `ICF(x; h)` and `ICF(x; h/2)` would discretize *different contact models* and their difference would not be a truncation error. ACR evaluates the contact constitutive law at the ATTEMPT step for all three solves, which is what makes the Richardson pair a pure integration-error estimate. | It rescales the Delassus estimate by `s = D(D+tau)/(h(h+tau))` — exactly `1` for the full solve, `2.05-2.35` for the halves over our step range. Against no-ACR at `h`, the soft branch (~89% of contacts) gets `rt/rn` larger by `s`: friction regularization ~2.1-2.35x softer. See B3. |
| D8 | **Per-world failure containment** | A world whose inner solve fails is not committed, does not kill the batch, and does not perturb any other world; the consumer reads `solver.diverged` and terminates that world. This is what makes a batched implementation viable at all — the alternative is one world's failure aborting 1023 healthy ones. | It changes failure SEMANTICS: `NEWTON_SAP_CONTAINMENT=0` restores strict converge-or-throw. On a batch where every solve converges the two are bitwise identical; where one does not, they are different runs by design. |
### 5.5 The aggregate test, and the three reference scales

Per-flag equivalence is necessary but not sufficient: what a reader needs is
the difference between **the shipped fast path and a fully legacy path**, on a
real contact-violent trajectory long enough for divergence to develop. That
is the aggregate run.

**Rig.** `IsaacContrib-Lift-Spatula-Trossen-v0`, driven by a pre-generated
random action sequence in `[-2, 2]` — the flail regime. Two configurations
were run, and the difference between them is itself a result:
**256 worlds x 60 control steps** (2 s), where every arm came out bitwise
identical, and **512 worlds x 150 control steps** (5 s), which is where the
fusion arms separate. All headline numbers below are the 512 x 150 run; the
256 x 60 run is reported in the horizon caveat. The task's early terminations
are disabled in the probe process only (the cfg object, not the task file), so
no episode restarts confound the comparison; worlds that fail are still
observed through the solver's own `diverged` latch, and comparisons are
per-env masked from any reset onward. The metric is the `L^inf` and RMS
difference of the committed generalized coordinates `joint_q` — the same
coordinate vector the solver's accuracy metric uses, with `S = I`, so it
shares `tol`'s units (see the note on kind, below).

**Arms.** `ref` (shipped defaults, det=1); `ref-repeat` (identical — the
oracle); `nofuse` (D1+D2+D3 OFF); `refmode` (every optimization at its
legacy/OFF state: the three fusions plus `SOLVE_COMPACT`, `LS_COMPACT`,
`GEMM_RESHAPE`, `PACK_PERCONTACT`, `NARROW_V3`, `SHARED_ASSEMBLY`,
`MARCH_COMPACT`, `TAIL_COMPACT`); `seedB` (different action seed);
`prod-a`/`prod-b` (two runs of the SHIPPED configuration with
`NEWTON_SAP_DETERMINISTIC` at its `"0"` source default). Graph capture and the
whole-march conditional tier stay at their defaults in every arm: they are
launch-mechanism switches whose bitwise invariance is certified in the same
session by the flag-equivalence probe, and disabling them multiplies wall time
without changing arithmetic.

**The three reference scales, all measured in the same session:**

| | Scale | Value |
|---|---|---|
| (a) | The method's accepted local-error budget — `tol`, and the accepted error actually realized | `tol = 1e-3`; realized: **median 3.54e-5, p90 8.65e-5, max 9.997e-4** — zero violations, and *identical between the arms* |
| (b) | Run-to-run spread of the SHIPPED configuration (`NEWTON_SAP_DETERMINISTIC` at its `"0"` default) — the irreproducibility already present in every reported run | onset step 12; Linf **1.88e-1** at step 37, **1.30** at 75, **1.20** at 112 |
| (c) | Seed-to-seed spread | onset step 0; Linf **2.90** at step 37, **7.74** at 75, **1.23e1** at 112 |

**Result — and it separates the stack cleanly in two.**

| Comparison | What it isolates | Onset | Linf @37 | @75 | @112 |
|---|---|---|---|---|---|
| `ref-repeat` vs `ref` | ORACLE (det=1 reproducibility) | never | 0 | 0 | 0 |
| **`nofuse` vs `refmode`** | **the EIGHT scheduling optimizations, in aggregate** | **never** | **0** | **0** | **0** |
| `nofuse` vs `ref` | the three fp-reduction-order fusions | step 13 | 8.23e-3 | 4.46e-2 | 4.46e-2 |
| `refmode` vs `ref` | all eleven flags | step 13 | 8.23e-3 | 4.46e-2 | 4.46e-2 |
| `prod-b` vs `prod-a` | scale (b): the shipped config against itself | step 12 | 1.88e-1 | 1.30 | 1.20 |
| `seedB` vs `ref` | scale (c) | step 0 | 2.90 | 7.74 | 1.23e1 |

Three things follow, and the engagement counters make none of them vacuous
(shipped arm: 2.34e6 fused-ladder envs, 2.94e6 alpha-max envs, 4.88e6
fused-update envs, 2.94e6 pack execs, 5.51e8 GEMM tile-skips, 20 narrowed
sites, 2996 assembly reuses; `refmode`: **every one of those exactly zero**;
`nofuse`: the three fusion counters zero, the rest fully engaged):

1. **`nofuse` and `refmode` are bitwise identical to each other.** Turning off
   all eight scheduling optimizations on top of the fusions changes **not one
   byte** of the committed trajectory over 512 worlds x 150 control steps.
   The eight are bitwise-neutral in aggregate, not merely cell by cell on a
   sphere rig.
2. **All divergence from the shipped fast path is the three fusions**, and it
   is bounded at max Linf **4.75e-2** over the horizon.
3. **That bound is ~27x smaller than the shipped configuration's own
   run-to-run irreproducibility.** Curve against curve over the 137 steps
   where both are live: the fusion divergence is a **median 0.037x** of the
   non-determinism curve (p90 0.205x), and the ratio of the two maxima is
   **0.0365** (4.75e-2 vs 1.30). The fusions perturb trajectories by far less
   than the non-determinism already present in every reported run.

   **With one honest exception, which is transient and small in absolute
   terms.** The two perturbations seed one step apart (non-determinism first
   registers at step 12 at 4.0e-9; the fusions at step 13 at 6.0e-7), and for
   **12 of those 137 steps — the window from step 13 to about step 20 — the
   fusion divergence is LARGER than the non-determinism**, peaking at a ratio
   of 12708x at step 17. That ratio is a small-denominator artifact: the
   absolute magnitudes there are 2.1e-5 against 1.5e-8, i.e. the fusion
   divergence during its worst relative excursion is still **~50x below `tol`**
   (2.1e-5 rad = 0.0012 deg = 21 um). From step ~25 onward, once both have
   saturated, the fusion curve sits an order of magnitude or more below the
   non-determinism curve for the rest of the horizon. The claim to make is
   therefore the saturated-regime one, stated with its transient, not a blanket
   "always smaller".

**A horizon caveat that must travel with the bitwise claims.** At the smaller
256-world x 60-step configuration *every* arm — fusions included — was bitwise
identical. The separation at step 13 only appears at 512 worlds x 150 steps.
Bitwise equality is therefore a property of the horizon it was measured on,
and the fusion arms' equality at the shorter horizon must not be quoted as a
general result. The eight scheduling flags were bitwise at **both** horizons.

**On comparing any of this to `tol`.** The Linf/tol ratios above are scale
markers, not tolerance violations: `tol` bounds the local error of one
accepted step, whereas Linf here is a trajectory difference accumulated over
up to 150 control steps in a chaotic contact regime. The commensurable
per-step statement is the ensemble one, and it is unambiguous: the accepted
local-error distributions of `ref`, `nofuse` and `refmode` agree to every
printed digit (median 3.5387e-5 vs 3.5386e-5, p90 identical, substeps ratio
1.001), while `prod-a` — the same shipped code with determinism at its
default — differs from `ref` by 1.55x in substep demand.

**A caution about the headline number.** In a contact-violent regime any
perturbation, down to one ulp, grows and saturates; the endpoint `L^inf` of
two arms therefore says little on its own. Three things are reported instead:
the divergence ONSET, the early GROWTH curve against the run-to-run
nondeterminism curve, and the ENSEMBLE statistics (the distribution of the
controller's own accepted error and accepted step over all world-boundary
samples). Two arms whose ensembles agree are running the same physics even
when no individual world's trajectory matches — and it is the ensemble, not
the pointwise trajectory, that any accuracy or work-precision plot reports.

**Recommendation on a reference mode: NOT NEEDED for the optimizations, and
REQUIRED for reproducibility — but the two are different knobs, and the
evidence points at the second one.**

* For the **eight scheduling optimizations there is nothing to choose**: the
  fast path *is* the reference path, bitwise, in aggregate, at both horizons
  tested. Producing accuracy numbers with them off would buy nothing and cost
  the entire speedup.
* For the **three fusions**, a reference mode exists and is one environment
  variable each. But the case for using it is weak on this evidence: their
  effect is ~27x below the irreproducibility the shipped configuration already
  has, and the accepted-error ensembles are identical. Making accuracy claims
  on the fusion-free path while the same runs remain non-reproducible
  run-to-run would be precision theatre.
* **The knob that actually matters is `NEWTON_SAP_DETERMINISTIC`.** It is OFF
  by default, and it — not any optimization — is what makes reported runs
  irreproducible, by a margin of 1.2 rad/m over 5 seconds. Any result a
  reviewer might try to reproduce should be produced with `=1`.

So: **report accuracy and work-precision results with determinism ON and the
optimization stack at its shipped defaults.** State that the eight scheduling
optimizations are bitwise-verified, that the three fusions are bounded at
4.75e-2 against a 1.20 run-to-run scale, and keep `refmode` documented as an
available bitwise-exact fallback rather than the production path for results.
### 5.6 The complement — what is NOT covered by the neutrality claim

The neutrality claim's scope must be unambiguous, so the complement is
enumerated rather than implied. **Nothing below is claimed physics-neutral,
and no result in this section applies to any of it.** These are either design
decisions with stated semantics (the D-class items) or genuine divergences
from the published method (the B-class items, Section 4).

**Physics-visible and ON in the reportable run:**

| # | Item | Why it is not neutral |
|---|---|---|
| D5 | Attempt-consistent regularization (ACR) | Freezes the contact constitutive law at the attempt step. Against no-ACR at `h`, the soft branch — ~89% of contacts — gets `rt/rn` larger by `s = 2(D+tau)/(D/2+tau) = 2.05-2.35` over our step range. This is a **correctness contribution** (it is what makes the Richardson pair a pure truncation-error estimate when the regularization is dt-dependent), not an optimization, and it has no counterpart in the paper. See B3. |
| D6 | Line-search default `monotone_decay` -> `armijo_decay` | A different line search plus three coupled tolerances. The paper specifies exact line search. See B6. |
| D8 | Per-world failure containment | Changes failure *semantics*. On a batch where every solve converges it is bitwise identical to strict converge-or-throw; where one does not, the two are different runs by design. This is what makes a batched implementation viable at all. |
| D9 | Determinism mode **OFF** | The shipped default. Reported runs are not bitwise reproducible run-to-run; the size of that irreproducibility is measured in 5.4 and is itself one of the reference scales. |
| D10 | `approx32` contact preset | f32 Jacobians and f32 contact linear solve, against the solver's own `drake` fp64 default and the paper's fp64 C++ reference. |

**Physics-visible but OFF or unreachable in the reportable run:** D4
(`monotone_decay` accept-rule rewrite — unflagged, but not the default
variant), D7 (cubic-Hermite seed — implemented, dead on the default line
search), D11 (`static_substep`), D12 (run-ahead single march).

**Divergences from the published method (Section 4), none of them
neutrality questions:** B1 (SAP rather than CENIC's Lagged ICF — inherited
from upstream), B2 (fixed material `tau_d`, which flattens the stiffness
exponent from `-1.76` to `-1.17`), B4 (`S = I` and a live `rtol`), B5 (one
collision query per boundary rather than two per step), B7 (CENIC VI-B
removed), B8/B8b (three controller exits Algorithm 1 does not have; effective
`k_Init = 1.0`), B9 (inner-solve cap-out treated as a rejection), B10 (no
trapezoid variant).

**The one-sentence scope statement for the methods text:**

> The GPU scheduling optimizations — env-list compaction, launch-grid
> narrowing, blocked-Cholesky narrowing, live-row GEMM truncation,
> per-contact packing and assembly reuse — are verified bitwise-neutral, both
> individually and in aggregate: with all eight disabled, 512 parallel worlds
> integrated over 150 control steps of contact-rich motion reproduce the
> optimized trajectories byte for byte, while the solver's own engagement
> counters confirm the optimized kernels ran in one arm and not the other. The
> three kernel fusions are algebraically exact but reorder floating-point
> reductions; they are therefore bounded rather than bitwise, and the bound —
> a maximum coordinate deviation of 4.8e-2 over that horizon — is roughly 27
> times smaller than the run-to-run irreproducibility the same configuration
> already exhibits with its default non-deterministic reductions. The
> attempt-consistent regularization, the failure containment, the line-search
> default and the contact preset are physics-visible design decisions stated
> as such.

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
  formulas printed alongside it; no new script was committed by pass 36, by
  design (the rails for that pass forbade code changes).
* **Pass-37 provenance (Section 5).** Two re-runnable probes were committed:
  `tools/probes/sap_neutrality_mirror_probe.py` (env-privacy and per-world
  `dt` threading) and `tools/probes/sap_neutrality_divergence_probe.py`
  (aggregate divergence, engagement counters and the three reference scales).
  The pass also re-ran, unmodified, `sap_flag_equivalence_probe.py` (42
  bitwise cells, 30 vacuity guards), `sap_runahead_oracle_probe.py`, and the
  pass-17 J-invariance probe. Chain, gate order and an `nvidia-smi` census at
  every run boundary: `p37_progress.txt`; logs `p37_g1_mirror.log` ..
  `p37_g8_prodpair.log`; per-arm trajectories `p37_stress/p37_*.npz`. The GPU
  was held one process at a time and was verified idle before every run.
* **Not verified in the pass-36 audit (explicit list). Item 1 was closed by
  pass 37; see Section 5. The rest remain open.**
  1. ~~Any C-class bitwise claim~~ — **CLOSED by pass 37**, which re-measured
     them on the bytes at `d16db463`/`afd5dc6` and rewrote Section 5 with the
     evidence class each item earned. Anything pass 37 could not establish is
     marked UNVERIFIED there rather than assumed.
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

### 12.1 Scope corrections found by pass 37 while verifying Section 5

Three statements in the pass-36 text were too generous about scope. None
changes a conclusion; all three change what a claim covers.

1. **C9 never executes on the adaptive path.**
   `reset_runtime_state_masked` has exactly one call site,
   `mjwarp_manager.py:910`, inside `if cls._sap and not cls._adaptive`. It is
   a **fixed-step SAP** optimization. It cannot affect any SAP-adaptive
   result, reportable or otherwise, and it should not be listed among the
   optimizations the adaptive arm carries.
2. **C3 has no escape hatch at all.** The listed Cholesky twins are called
   unconditionally (`contact_solve.py:7633, 7644`); only the launch *width*
   varies, via march compaction. The pass-36 "Default: ON when narrowed" is
   therefore not a flag state — the pre-`fe98f46` masked twins are retained in
   `blocked_cholesky.py` but unreachable at runtime.
3. **C2 and C7 are not two implementations of the same math.** Both were
   converted *in place*: ON and OFF run the same kernel binaries, differing
   only in whether the consumers read the device-built active list or a static
   identity list at full width. Their bitwise claim is the strictly weaker and
   more checkable proposition that a kernel's output for an env does not
   depend on which other envs share its launch. See 5.3.
