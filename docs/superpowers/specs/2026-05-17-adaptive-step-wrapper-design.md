# Adaptive Step Wrapper Design

**Date:** 2026-05-17
**Author:** mardigiorgio (with Claude)
**Status:** Approved, ready for plan
**Audience:** PI demo on Monday; long-term Newton contribution

## 1. Problem

`SolverMuJoCoCENIC` is the only adaptive-step solver in this project, and it
is tightly coupled to MuJoCo Warp. We want to compare adaptive vs fixed
stepping across multiple Newton solvers (MuJoCo, XPBD, SemiImplicit) on the
same scene, ideally at `N = 2^15` worlds. CENIC's `step_dt` waits for the
slowest world at the outer boundary; per-iteration kernels run on all `N`
worlds even after most finish. K_global grows with `N` because P(some world
is in a hard contact moment) grows with `N`. The result is `O(K × N)` total
work where K_global scales as `N^0.2-0.3` on dense-contact scenes.

We will build a ground-up adaptive wrapper that:
1. Hot-swaps between solvers via a `step_fn` callable.
2. Minimizes PCIe traffic in the hot path (one 4-byte sync per iter).
3. Adds active-set masking (XPBD/Semi: kernel-level early-return) and
   wrapper-side compaction (MuJoCo: re-batch into smaller `mjw_data`) to
   address the `K × N` scaling problem.

## 2. Goals + non-goals

### Goals

- One wrapper class hierarchy used by all three adaptive variants.
- Functional parity with `SolverMuJoCoCENIC` for the MuJoCo path
  (correctness oracle: state matches CENIC to float32 tolerance).
- Per-world adaptive dt math preserved **for the MuJoCo adaptive variant**
  (no regression vs CENIC). XPBD and SemiImplicit adaptive variants run in
  effectively-global dt mode because those solvers only accept scalar dt
  (see Section 3.5). This is an accepted limitation of v1.
- Asymptotic cost approaching `O(N)` (not `O(K × N)`) for MuJoCo via
  compaction, for XPBD/Semi via active-set masking.
- Reach `N = 2^15` on `falling_cylinder` for the comparison plot.

### Non-goals

- Replacing `SolverMuJoCoCENIC` in v1 (it stays put, untouched).
- Active-set masking in `mujoco_warp` itself (that's weeks of third-party
  work; we use wrapper-side compaction instead).
- New error estimator beyond step doubling.
- New controller variants beyond CENIC's Drake PI.
- Particle / soft-body support (VBD lives in this codebase but isn't a
  rigid-contact solver in our scenes).

## 3. Architecture

### 3.1 File tree

```
scripts/adaptive/
├── __init__.py
├── base.py              # AdaptiveWrapper core class (~250 LoC)
├── kernels.py           # solver-agnostic Warp kernels (~150 LoC, lifted from CENIC)
├── controller.py        # Drake PI tuning constants (~50 LoC)
├── masking_mixin.py     # ActiveSetMaskingMixin (~100 LoC)
├── compaction_mixin.py  # CompactionMixin (~250 LoC)
├── factories.py         # adaptive_{mujoco,xpbd,semi}_factory (~100 LoC)
└── tests/
    ├── test_kernels.py
    ├── test_controller.py
    ├── test_masking.py
    ├── test_compaction.py
    └── test_integration.py

newton/_src/solvers/xpbd/             # MODIFIED: world_active arg added to per-world kernels
newton/_src/solvers/semi_implicit/    # MODIFIED: world_active arg added to per-world kernels
```

### 3.2 Class hierarchy

```
AdaptiveWrapper                  ← base: step doubling + controller + error + boundary
  │
  ├── AdaptiveMaskingWrapper     ← AdaptiveWrapper + ActiveSetMaskingMixin
  │     used by adaptive_xpbd_factory, adaptive_semi_factory
  │
  └── AdaptiveCompactionWrapper  ← AdaptiveWrapper + CompactionMixin
        used by adaptive_mujoco_factory
```

Two concrete classes, one per active-set strategy. No multi-mixin stacking
(strategies are alternatives, not complements).

### 3.3 Public API

The wrapper is invisible to the bench. The bench sees factories that return
`(solver, step_fn)` pairs, same as every other solver factory in
`scripts/scenes/_solvers.py`.

```python
from scripts.adaptive.factories import adaptive_mujoco_factory

builder = adaptive_mujoco_factory(
    tol=1e-3,
    dt_init=DT_OUTER,
    dt_min=1e-6,
    dt_max=DT_OUTER,
    dt_outer=DT_OUTER,
    nconmax=20,
    njmax=80,
    compaction_sizes=(1.0, 0.25, 0.0625, 0.015625),  # N, N/4, N/16, N/64
)
solver, step_fn = builder(model)
# step_fn(model, s0, s1, ctrl) -> (s0, s1)
```

### 3.4 Per-world dt asymmetry

- MuJoCo accepts per-world dt via `mjw_model.opt.timestep` (a `wp.array`).
  The MuJoCo step_fn shim copies the wrapper's dt array into it.
- XPBD and SemiImplicit accept scalar `dt: float` only. Their step_fn
  shims read `dt[0]` (with masking, all active worlds share the same dt
  because step_fn can only pass scalar — see Section 3.5).

The wrapper always builds a `wp.array(N)` dt. The shim translates.

### 3.5 Per-world dt + masking interaction

ActiveSetMaskingMixin (XPBD/Semi) cannot support true per-world dt because
those solvers take scalar dt. Two options:

1. **Effectively-global dt**: take `max(dt[active])`, pass scalar. All
   active worlds step that. Loses per-world adaptivity but kernel runs
   correctly.
2. **Per-world dt via dt_array support added to XPBD/Semi kernels**:
   would require deeper modification of those solvers. Out of scope for v1.

v1 uses option 1: XPBD/Semi adaptive runs in effectively-global dt mode.
MuJoCo retains full per-world dt via the `opt.timestep` hook.

## 4. Components

### 4.1 `base.py` — AdaptiveWrapper

Constructor args:

```python
AdaptiveWrapper(
    model: newton.Model,
    step_fn: Callable,  # step_fn(model, state_in, state_out, ctrl, contacts, dt_array)
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
    needs_collide: bool,  # call model.collide() before step_fn?
    contacts: newton.Contacts | None = None,
    stuck_policy: Literal["freeze", "raise"] = "freeze",
    max_iters: int = 500,
)
```

Owns:
- 4 scratch states (allocated once): `_state_saved`, `_state_full`,
  `_state_mid`, `_state_double`.
- Per-world arrays: `_dt`, `_ideal_dt`, `_dt_half`, `_sim_time`,
  `_next_time`, `_accepted`, `_last_error`, `_accepted_error`.
- 1-element host-sync arrays: `_boundary_flag`, `_iteration_count_buf`.
- `_q_weights` (per-coord weights for error norm, computed from
  `body_inv_mass`, same as CENIC).

Public methods:
- `step_dt(dt_outer, s0, s1, ctrl) -> tuple[State, State]`
- `iteration_count` (property → most recent K, on device)
- `dt` (property → per-world `_dt` array, on device)
- `status_summary() -> dict` (6 scalars: sim_time min/max, dt min/max,
  error_max, floor_accept_count)

### 4.2 `kernels.py` — Lifted Warp kernels

Lifted verbatim from `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`
(solver-agnostic — operate on plain State arrays):

- `_inf_norm_state_error_kernel(state_full.q, state_double.q, weights, coords_per_world) -> last_error[N]`
  - q-only L∞ norm, weighted by `weights`. Writes NaN if either state has NaN.
- `_apply_dt_cap(dt_ideal, dt_min, dt_max, dt_out, dt_half_out)`
- `_clamp_dt_to_boundary(dt, dt_half, sim_time, next_time)`
- `_calc_adjusted_step(last_error, dt, ideal_dt, accepted_out, tol, dt_min)`
  - Drake PI controller; sets per-world `accepted` bool and `ideal_dt` for next iter.
- `_select_float_kernel`, `_select_transform_kernel`, `_select_spatial_vector_kernel`
  - per-world mux: state_cur := accepted ? state_double : state_saved
- `_advance_sim_time(sim_time, dt, accepted, last_error, accepted_error)`
- `_boundary_check(sim_time, next_time, boundary_flag)`
- `_boundary_reset(boundary_flag)`
- `_iter_count_increment(count)`

No algorithm change; just relocation.

### 4.3 `controller.py` — Drake PI constants

`ControllerConfig` dataclass holding `KP`, `KI`, `safety_factor`,
`growth_cap`, `shrink_cap`. Default values copied from CENIC. Exists so
v2 can swap controller variants without touching kernels.

### 4.4 `masking_mixin.py` — ActiveSetMaskingMixin

Adds:
- `_world_active: wp.array(N, bool)` initialized `True` at start of `step_dt`.
- After `_advance_sim_time`: `_update_active_mask` kernel sets
  `_world_active[i] = (sim_time[i] < next_time[i])`.
- `_n_active: wp.array(1, int32)` updated via reduction over `_world_active`
  (1 PCIe sync per iter, 4 bytes).

step_fn signature for masked solvers takes `world_active` as additional arg:
```python
step_fn(model, state_in, state_out, ctrl, contacts, dt_array, world_active)
```

Modified XPBD / SemiImplicit kernels accept `world_active: wp.array(N, bool)`
and early-return if `world_active[world_idx] == False`. Approximately 10
kernels per solver need this change.

### 4.5 `compaction_mixin.py` — CompactionMixin

Wrapper-side compaction for MuJoCo (no third-party kernel modifications).

Constructor adds:
- `compaction_sizes: tuple[float, ...]` — fractions of N to pre-allocate
  mjw_data at. Default `(1.0, 0.25, 0.0625, 0.015625)` = `[N, N/4, N/16, N/64]`.
- OOM fallback chain: if (1) fails to allocate, try `(1.0, 0.25, 0.0625)`,
  then `(1.0, 0.25)`, then `(1.0,)` (compaction disabled).

Maintains:
- `_mjw_data_tiers: list[mjw_data]` pre-allocated at construction.
- `_current_tier_idx: int` (Python int, host-side).
- `_active_indices: wp.array(N, int32)` — permutation: compact slot → original world id.

Per-iter logic:
- `_active_count` reduction (1 PCIe sync, 4 bytes — replaces boundary_check's sync; we batch them in one round-trip).
- If `_active_count <= 0.5 × current_tier.size` and a smaller tier fits:
  - Build `_active_indices` via prefix-sum kernel.
  - `_compact_gather` kernel: read state from canonical layout, write to compact layout in smaller mjw_data.
  - `_current_tier_idx += 1`.
  - From now on, step_fn operates on smaller mjw_data.
- After `step_fn`: state in compact layout. Next iter operates compact.

End-of-step_dt:
- If `_current_tier_idx > 0`: `_compact_scatter` kernel copies compact state back to canonical N layout in `_state_cur`.

Custom kernels (~60 LoC each):
- `_active_indices_prefix_sum(world_active) -> active_indices`
- `_compact_gather(state_canonical, active_indices) -> state_compact`
- `_compact_scatter(state_compact, active_indices, state_canonical)`

### 4.6 `factories.py` — Concrete factory builders

```python
def adaptive_mujoco_factory(*, tol, dt_init, dt_min, dt_max, dt_outer,
                            nconmax, njmax, compaction_sizes=...): ...

def adaptive_xpbd_factory(*, tol, dt_init, dt_min, dt_max, dt_outer): ...

def adaptive_semi_factory(*, tol, dt_init, dt_min, dt_max, dt_outer): ...
```

Each builds the right concrete wrapper class + the right step_fn shim:

- **MuJoCo shim**: copies wrapper's per-world dt array into
  `mjw_model.opt.timestep`, then calls underlying `SolverMuJoCo.step` on
  the current compact mjw_data.
- **XPBD/Semi shim**: takes `dt_array.max()` (or `dt_array[0]`),
  passes scalar to `solver.step(...)`; also passes `world_active` mask.

### 4.7 Newton kernel modifications

`newton/_src/solvers/xpbd/`: ~10 kernels per solver get
`world_active: wp.array(N, bool)` arg + early-return guard:

```python
@wp.kernel
def some_per_world_kernel(..., world_active: wp.array(dtype=wp.bool)):
    tid = wp.tid()
    world_idx = tid // PER_WORLD_DIM  # solver-specific
    if not world_active[world_idx]:
        return
    # ... existing kernel body ...
```

Backward compatibility chosen: **add the `world_active` arg to every
modified kernel; existing Python callers of `SolverXPBD.step` / `SolverSemiImplicit.step`
get an all-True default mask constructed lazily inside the solver's Python
step method**. No kernel-signature variants. No silent behavior change for
existing callers.

Same pattern for `newton/_src/solvers/semi_implicit/`.

## 5. Data flow

### 5.1 One `step_dt(dt_outer, s0, s1, ctrl)` call

Setup:
1. `_ideal_dt` (warm-started from previous call) → `_dt`, clamp to `[dt_min, dt_max]`.
2. `s0 → _state_cur` (input copy).
3. `_next_time = _sim_time + dt_outer`.
4. `_iteration_count = 0`, `_boundary_flag = 1`.
5. Masking mixin: `_world_active[:] = True`.
6. Compaction mixin: `_active_count = N`, `_current_tier_idx = 0`.

Inner loop — repeat until `_boundary_flag == 0` or `_iteration_count >= max_iters`:

```
iter++

[Compaction mixin]
if _active_count <= 0.5 × current_tier.size:
    pick next-smaller tier that fits _active_count
    _active_indices_prefix_sum(_world_active) → _active_indices
    _compact_gather: state_cur → state_compact
    current_tier ← smaller mjw_data

_clamp_dt_to_boundary

_save_state: _state_cur → _state_saved

step_fn(state_cur, _state_full, ctrl, contacts, dt)
step_fn(state_cur, _state_mid,  ctrl, contacts, dt/2)
step_fn(state_mid, _state_double, ctrl, contacts, dt/2)

_inf_norm_state_error_kernel
_calc_adjusted_step (sets _accepted + new _ideal_dt)

_select_*_kernel: state_cur := accepted ? state_double : state_saved
_advance_sim_time

[Masking mixin]
_update_active_mask: world_active[i] = sim_time[i] < next_time[i]

[Compaction mixin]
_active_count := reduce_sum(_world_active) → host (1 × int32, batched with boundary_flag)

_apply_dt_cap, _ideal_dt := new_dt
_boundary_reset, _boundary_check
_boundary_flag.numpy() → host (or batched with _active_count above)
```

Exit:
7. Compaction mixin: if `_current_tier_idx > 0`, `_compact_scatter` state back to N layout.
8. `_state_cur → s0`; return `(s0, s1)`.

### 5.2 PCIe traffic per inner iter

- 1 × int32 boundary flag (always).
- 1 × int32 active count (Compaction only — batched in same round-trip when possible).
- **Max 8 bytes per iter.** Matches CENIC.

### 5.3 Cost model

| variant | per-iter work |
|---|---|
| Base | `3 × cost(solver.step on N)` + ~10 control kernels (~50 µs total) |
| + Masking | `3 × cost(solver.step on N, fast path for inactive worlds)` ≈ `3 × N_active/N × base` after warps converge |
| + Compaction | `3 × cost(solver.step on current_tier.size)` + occasional gather/scatter at tier transitions |

For MuJoCo with Compaction, total work for K iterations at decaying active count:
`Σ_k 3 × N_k × c` where `N_k` shrinks geometrically.
If active count halves per tier: `≈ 3 × 2N × c` total, vs CENIC's `3 × K × N × c`.

**Asymptotic target: O(N) instead of O(K × N).**

## 6. Error handling

### 6.1 NaN

- `_inf_norm_state_error_kernel` produces NaN in `_last_error` if state has NaN.
- `_calc_adjusted_step` treats NaN error as reject → world halves dt → eventually hits `dt_min`.
- At `dt_min` with NaN: world is **stuck**.

`stuck_policy` constructor arg:
- `"freeze"` (default): log world id + sim_time, set `_world_active[i] = False`. Other worlds finish. Return state has NaN for stuck worlds.
- `"raise"`: `WrapperConvergenceError(world_id, sim_time, dt_min)`.

### 6.2 dt-floor accept (non-NaN)

When dt = `dt_min` and error > tol but state is finite: accept anyway (matches CENIC). Counter `_floor_accept_count` exposed via `status_summary()` for caller introspection. No hard failure.

### 6.3 Max iteration cap

`max_iters` constructor param (default 500). If exceeded: raise `WrapperIterationLimitError`. Guards against infinite loops from controller pathology.

### 6.4 Compaction OOM at construction

Pre-allocation of mjw_data tiers can OOM. Fallback chain:
1. `(1.0, 0.25, 0.0625, 0.015625)`
2. `(1.0, 0.25, 0.0625)`
3. `(1.0, 0.25)`
4. `(1.0,)` — compaction disabled

Log which tier set succeeded.

### 6.5 step_fn exceptions

Propagate. Wrapper does not catch. Bench's per-N subprocess isolation handles process-level failure.

### 6.6 Active mask consistency invariant

Compaction's `_active_indices[0..n_active]` must always be a permutation of currently-active world ids. Debug-build assertion: `_world_active.sum() == n_active`. Production builds skip the check (one extra reduction per iter).

### 6.7 What we explicitly do NOT do

- No retry on step_fn exception.
- No automatic tol relaxation when stuck.
- No NaN repair / pre-NaN state rollback.

## 7. Testing strategy

### 7.1 Unit tests (`scripts/adaptive/tests/`)

- `test_kernels.py`: each lifted kernel exercised with hand-crafted inputs and known outputs.
- `test_controller.py`: synthetic error sequences, verify dt convergence to expected steady state.
- `test_masking.py`: mock step_fn records which worlds it sees; verify only active worlds visible after one finishes.
- `test_compaction.py`: mock state; mark half worlds inactive; verify `_compact_gather` produces correct compact layout, `_compact_scatter` restores.

### 7.2 Integration tests (`test_integration.py`)

- **Correctness vs CENIC oracle** (must-pass): `adaptive_mujoco_factory` wrapper vs `SolverMuJoCoCENIC` on `falling_cylinder` at N=4, 100 outer steps, `body_q` matches within 1e-5.
- **Stability**: each variant on `falling_cylinder` N=4, 200 outer steps. No NaN. `pos_max < 5m`.
- **Compaction correctness**: `adaptive_mujoco` with explicit `compaction_sizes=(1.0, 0.25)` vs same wrapper with `compaction_sizes=(1.0,)` (disabled). Output states match within 1e-5.

### 7.3 Performance bench (the wow figure)

Add the 3 adaptive factories to `scripts/scenes/falling_cylinder.py` `SOLVER_FACTORIES`. Run:

```
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 4 16 64 256 1024 4096 16384 32768 \
  --steps 50 --warmup 10
```

Acceptance criteria:
- **Must-pass**: All 3 adaptive variants complete `N ≥ 1024` without OOM or NaN.
- **Must-pass**: `adaptive_xpbd` and `adaptive_semi` produce monotonic finite curves at all completed N.
- **Target (not gate)**: `adaptive_mujoco` with compaction shows better-than-CENIC wall time at `N ≥ 4096`. If compaction overhead exceeds the active-set win on this scene, we accept that and document — the wrapper still ships with correct output and the gate above is met.

### 7.4 Long-run smoke (manual, not gated)

1000 outer steps on `falling_cylinder` at N=64 per variant. Diff final body_q distributions. Sanity walk before declaring v1 done.

## 8. Parallelization map (for multi-agent dev)

| sub-agent | files owned | depends on |
|---|---|---|
| **A — Core** | `base.py`, `kernels.py`, `controller.py` | nothing (lifts from CENIC) |
| **B — Masking** | `masking_mixin.py`, kernel edits in `newton/_src/solvers/xpbd/` + `semi_implicit/` | A's `base.py` interface |
| **C — Compaction** | `compaction_mixin.py` + gather/scatter kernels | A's `base.py` interface |
| **D — Factories + tests** | `factories.py`, all `tests/*` files | A, B, C interfaces (mock until ready) |
| **E — Bench wiring** | `scripts/scenes/falling_cylinder.py` SOLVER_FACTORIES, full sweep | D's factories ready |

Order: A first (serial), then B+C+D parallel, then E. Estimated 2 days wall-time with 4 parallel agents.

## 9. Open questions deferred to implementation

- Exact prefix-sum kernel pattern for `_active_indices` (Warp has primitives; pick one during implementation).
- Whether to batch `_active_count` and `_boundary_flag` into one host transfer (likely yes; verify in profiler).

## 10. Out-of-scope follow-ups

- v2: Refactor `SolverMuJoCoCENIC` to use this wrapper internally (eliminates duplication).
- v3: Active-set masking inside mujoco_warp (third-party fork or upstream PRs; weeks of work; the "real" research contribution).
- Particle / soft-body adaptive stepping (different solver class entirely).

---

End of design.
