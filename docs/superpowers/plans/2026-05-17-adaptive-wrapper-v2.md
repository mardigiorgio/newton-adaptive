# Adaptive Step Wrapper v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address the 5 items from v1's final code review: collapse per-iter PCIe syncs, hoist `dt_array.max()` out of XPBD/Semi inner loops, wire ControllerConfig into kernels, implement real mjw_data tier-swap compaction, refactor SolverMuJoCoCENIC to use the new wrapper.

**Architecture:** Three sequential phases, each independently shippable. Phase 1 fixes perf bugs (small, surgical). Phase 2 makes compaction real (substantial — multiple mjw_data tiers + gather/scatter for transform/spatial_vector types). Phase 3 unifies CENIC under the wrapper.

**Tech Stack:** Same as v1 — Newton 1.1.0.dev0, Warp 1.12.0rc2, mujoco_warp 3.5.0.2, Python 3.12, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-05-17-adaptive-step-wrapper-design.md` (architecture unchanged; this plan implements deferred items + fixes review issues).

**Commit policy:** Per user preference, this plan does NOT include `git commit` steps. Each task ends with verification only. The user batches commits at their own cadence.

**Prerequisite:** v1 is shipped and all 23 tests pass. This plan assumes v1's files exist.

---

## File map

### Phase 1 (Critical fixes) — modified files only

- `scripts/adaptive/base.py` — collapse two `.numpy()` calls into one packed read
- `scripts/adaptive/factories.py` — hoist `dt_array.max()` reduction; pass `ControllerConfig` to wrapper
- `scripts/adaptive/kernels.py` — convert `_calc_adjusted_step` to read controller constants from inputs (not compile-time `wp.constant`)
- `scripts/adaptive/controller.py` — minor: add docstring noting v2 wiring

### Phase 2 (Real compaction) — modified + new files

- `scripts/adaptive/compaction_mixin.py` — extend with transform + spatial_vector gather/scatter kernels; tier-list management; tier-swap logic
- `scripts/adaptive/factories.py` — `adaptive_mujoco_factory` pre-allocates `mjw_data` instances at configured tier sizes with OOM fallback
- `scripts/adaptive/base.py` — minor: expose `_state_cur` re-pointing for compaction

### Phase 3 (CENIC refactor) — modified files

- `newton/_src/solvers/mujoco/solver_mujoco_cenic.py` — refactor: delegate step-doubling to `AdaptiveCompactionWrapper` while preserving public API
- `newton/solvers.py` — no change (re-exports stay valid)
- `scripts/scenes/*.py` — verify each scene's CENIC factory still works unchanged

### Test files

- `scripts/adaptive/tests/test_perf.py` — NEW: regression test that step_dt loop body issues ≤ 1 PCIe sync per iter (instrumented via Warp profiler)
- `scripts/adaptive/tests/test_compaction.py` — extend with multi-tier test + transform/spatial_vector roundtrip
- `scripts/adaptive/tests/test_integration.py` — extend with v2 perf gate + CENIC-via-wrapper backward-compat test

---

## Phase 1: Critical perf fixes (Agent A, serial, ~half day)

### Task 1: Collapse two per-iter PCIe syncs into one

**Files:**
- Modify: `scripts/adaptive/base.py:285-294`

**Background.** Current loop reads `_iteration_count_buf.numpy()` AND `_boundary_flag.numpy()` per iteration — two independent CUDA syncs. We pack them into a single `wp.array(2, dtype=wp.int32)` and read both in one `.numpy()` call.

- [ ] **Step 1: Write failing perf-count test**

Append to `scripts/adaptive/tests/test_integration.py`:

```python
def test_step_dt_does_one_pcie_sync_per_iter(monkeypatch):
    """Each inner iter should do exactly 1 host transfer (the packed (iter, boundary) read).

    We monkey-patch wp.array.numpy to count calls and assert the count == iterations.
    """
    import warp as wp
    from scripts.adaptive.base import AdaptiveWrapper

    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()

    def _stepfn(model, sin, sout, ctrl, contacts, dt_array):
        pass

    w = AdaptiveWrapper(
        model=model, step_fn=_stepfn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # Count numpy() calls on the wrapper's host-sync packed buffer only.
    # Use the new _loop_status array (added in this task).
    real_numpy = wp.array.numpy
    counts = {"n": 0}
    def _counting_numpy(self, *a, **kw):
        if hasattr(w, "_loop_status") and self is w._loop_status:
            counts["n"] += 1
        return real_numpy(self, *a, **kw)
    monkeypatch.setattr(wp.array, "numpy", _counting_numpy)

    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()

    K_done = int(w.iteration_count.numpy()[0])
    # Allow 1 extra sync at the end for the public iteration_count read above.
    assert counts["n"] in (K_done, K_done + 1), (
        f"expected {K_done} or {K_done+1} numpy() on _loop_status; got {counts['n']}"
    )
```

- [ ] **Step 2: Run, verify fails on missing `_loop_status` attribute**

Run:
```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_step_dt_does_one_pcie_sync_per_iter -v
```
Expected: FAIL with `AttributeError: 'AdaptiveWrapper' object has no attribute '_loop_status'`.

- [ ] **Step 3: Add packed status array to `AdaptiveWrapper.__init__`**

In `scripts/adaptive/base.py`, locate `self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)` and `self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)` (around lines 95-96). Add right after them:

```python
        # Packed host-sync buffer: index 0 = iteration count, index 1 = boundary flag.
        # Read both in a single .numpy() call per iteration (v2 perf fix).
        self._loop_status = wp.zeros(2, dtype=wp.int32, device=device)
```

- [ ] **Step 4: Add packing kernel to `scripts/adaptive/kernels.py`**

Append to `scripts/adaptive/kernels.py`:

```python
@wp.kernel
def _pack_loop_status(
    iter_count: wp.array(dtype=wp.int32),
    boundary_flag: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
):
    out[0] = iter_count[0]
    out[1] = boundary_flag[0]
```

- [ ] **Step 5: Rewrite the boundary loop in `AdaptiveWrapper.step_dt`**

In `scripts/adaptive/base.py`, find the `while True:` loop (around line 286). Replace with:

```python
        # Boundary loop: one packed PCIe sync per iteration.
        while True:
            self._run_iteration_body(effective_dt_max)
            wp.launch(
                K._pack_loop_status, dim=1,
                inputs=[self._iteration_count_buf, self._boundary_flag, self._loop_status],
                device=device,
            )
            status = self._loop_status.numpy()
            if int(status[0]) >= self.max_iters:
                raise RuntimeError(
                    f"AdaptiveWrapper: max_iters={self.max_iters} exceeded "
                    f"in step_dt(dt_outer={dt_outer})"
                )
            if int(status[1]) == 0:
                break
```

- [ ] **Step 6: Run perf test + full suite**

Run:
```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: 24 passed (23 prior + 1 new perf test).

If `test_adaptive_mujoco_matches_cenic_oracle` regresses, the packing kernel is reading stale values (race with `_iter_count_increment` / `_boundary_check` launches). Add `wp.synchronize()` before the pack launch as a debug step, then redesign.

---

### Task 2: Hoist `dt_array.max()` reduction out of XPBD/Semi inner loop

**Files:**
- Modify: `scripts/adaptive/factories.py:174, 213`
- Add: max-reduction kernel to `scripts/adaptive/kernels.py`

**Background.** v1's step_fn shim for XPBD and SemiImplicit calls `float(dt_array.numpy().max())` per substep — 3 syncs per inner iter (one per substep call), transferring N floats each. Per CLAUDE.md hot-path rule this is forbidden. Fix: pre-compute `max(dt_array)` on device once per iteration in the wrapper, read it as a scalar with one sync, pass the scalar to the shim.

- [ ] **Step 1: Add a single-thread max-reduction kernel + scratch to `AdaptiveWrapper.__init__`**

Append to `scripts/adaptive/kernels.py`:

```python
@wp.kernel
def _scalar_max_dt(
    dt: wp.array(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
):
    """Single-thread max over per-world dt. For N <= ~4096 launch overhead
    dominates a parallel reduction; this is faster + zero atomics."""
    n = dt.shape[0]
    m = dt[0]
    for i in range(1, n):
        if dt[i] > m:
            m = dt[i]
    out[0] = m
```

Add to `scripts/adaptive/base.py` `__init__` (next to `_loop_status`):

```python
        # Pre-reduced scalar dt (for solvers that take scalar, not per-world).
        # Used by XPBD/Semi step_fn shims; MuJoCo ignores it (uses opt.timestep array).
        self._dt_scalar = wp.zeros(1, dtype=wp.float32, device=device)
```

- [ ] **Step 2: Update `_run_iteration_body` to populate `_dt_scalar` once per iter**

In `scripts/adaptive/base.py::_run_iteration_body`, add right before the first `step_fn` call (the `self.step_fn(model, self._state_cur, self._state_full, ...)` line):

```python
        # Reduce per-world dt to a scalar (one sync per iter — value used by shims).
        wp.launch(K._scalar_max_dt, dim=1,
                  inputs=[self._dt, self._dt_scalar], device=dev)
```

This launch costs ~5µs and is dwarfed by the substep cost.

- [ ] **Step 3: Update step_fn API to pass scalar-dt scratch through**

In `scripts/adaptive/base.py`, change the `StepFn` type alias (around line 22-26) to include scalar dt:

```python
StepFn = Callable[
    [newton.Model, newton.State, newton.State, newton.Control,
     newton.Contacts, wp.array, wp.array],
    None,
]
# Last two args: dt_array (per-world wp.array(N, float32)),
#                dt_scalar (1-elem wp.array(1, float32), pre-reduced max).
```

Update the 3 step_fn calls in `_run_iteration_body` to pass `self._dt_scalar` as a 7th arg:

```python
        self.step_fn(model, self._state_cur, self._state_full, None,
                     self.contacts, self._dt, self._dt_scalar)
        self.step_fn(model, self._state_cur, self._state_mid, None,
                     self.contacts, self._dt_half, self._dt_scalar)
        self.step_fn(model, self._state_mid, self._state_double, None,
                     self.contacts, self._dt_half, self._dt_scalar)
```

Note: `_dt_scalar` reflects `max(self._dt)` not `max(self._dt_half)`. For step doubling the scalar shims must accept this asymmetry: full step uses `dt_scalar`, half steps use `dt_scalar / 2.0` computed in the shim from the scalar (avoiding another reduction). Update the shim signature in Tasks 2.4-2.5 to handle.

- [ ] **Step 4: Rewrite XPBD shim in `scripts/adaptive/factories.py`**

In `scripts/adaptive/factories.py::adaptive_xpbd_factory`, replace the step_fn (around line 170-180) with:

```python
        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg,
                    dt_array, dt_scalar_buf):
            """XPBD shim: read pre-reduced scalar dt (1 sync per iter, not 3).

            dt_scalar_buf reflects max(dt) at this iteration. For half-step
            substeps, divide by 2 on host.
            """
            # Detect half-step by comparing array refs (dt_array vs wrapper._dt_half).
            # Cleaner: shim reads scalar, halves if dt_array is the half buffer.
            scalar_dt = float(dt_scalar_buf.numpy()[0])
            if dt_array is wrapper_ref[0]._dt_half:
                scalar_dt *= 0.5
            model_arg.collide(state_in, contacts)
            mask = wrapper_ref[0]._world_active if wrapper_ref else None
            if mask is not None:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)
            else:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt)
```

Note: `dt_scalar_buf.numpy()` is still a sync per substep (3 per iter — same as before in count, but transferring 4 bytes not 4N bytes). For a deeper fix we'd want the shim to cache the scalar across the 3 substeps; that's deferred. The key win here is the **4-bytes-vs-4N-bytes transfer size**, not the sync count.

Actually, even better: do the read ONCE per iter (not per substep). Refactor to expose a `_pre_iter_setup` hook that runs once per iter, reads dt_scalar into a Python float, and stores it on `self._scalar_dt_host`. Then the 3 step_fn calls read the Python float. Implement this in Step 5.

- [ ] **Step 5: Add `_pre_iter_setup` hook + Python-host scalar caching**

In `scripts/adaptive/base.py::_run_iteration_body`, add right after the `_scalar_max_dt` launch:

```python
        # Cache scalar dt on host once per iter (saves 2 syncs/iter for shims that need it).
        self._scalar_dt_host = float(self._dt_scalar.numpy()[0])
        self._scalar_dt_half_host = self._scalar_dt_host * 0.5
```

Then update XPBD/Semi shims to read from `wrapper_ref[0]._scalar_dt_host` / `_scalar_dt_half_host` instead of `dt_scalar_buf.numpy()`:

```python
        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg,
                    dt_array, dt_scalar_buf):
            wrapper = wrapper_ref[0]
            scalar_dt = (wrapper._scalar_dt_half_host
                         if dt_array is wrapper._dt_half
                         else wrapper._scalar_dt_host)
            model_arg.collide(state_in, contacts)
            mask = wrapper._world_active
            underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)
```

This reduces XPBD/Semi inner-loop transfers from `3 × N × 4 bytes` per iter to `1 × 4 bytes` per iter. At N=32768, that's 384KB → 4 bytes per iter.

- [ ] **Step 6: Update MuJoCo shim to use new signature (no behavior change)**

In `scripts/adaptive/factories.py::adaptive_mujoco_factory`, update step_fn signature to accept `dt_scalar_buf` (ignored for MuJoCo, which uses `opt.timestep` array):

```python
        def step_fn(model_arg, state_in, state_out, ctrl, contacts,
                    dt_array, dt_scalar_buf):  # NEW: dt_scalar_buf accepted but unused
            wp.copy(underlying.mjw_model.opt.timestep, dt_array)
            # ... rest unchanged
```

- [ ] **Step 7: Update Semi shim same as XPBD (Step 5 pattern)**

- [ ] **Step 8: Run full suite**

Run:
```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: 24 passed. If oracle test (`test_adaptive_mujoco_matches_cenic_oracle`) fails, the signature change broke MuJoCo factory — verify Step 6 was applied.

- [ ] **Step 9: Smoke bench to confirm perf improvement**

Run:
```bash
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 64 1024 --steps 20 --warmup 5 2>&1 | tail -15
```

Compare `xpbd_adaptive_1e-3` and `semi_adaptive_1e-3` wall times to v1's bench JSON at the same N. Expect:
- N=1024: noticeable reduction (sync overhead amortized over K=1 means small win on this scene; bigger win at higher K).

---

### Task 3: Wire `ControllerConfig` into `_calc_adjusted_step` kernel inputs

**Files:**
- Modify: `scripts/adaptive/kernels.py:59-110` — remove compile-time `_DRAKE_*` `wp.constant`s, accept them as kernel inputs
- Modify: `scripts/adaptive/base.py` — pass controller config to kernel launch
- Modify: `scripts/adaptive/controller.py` — add `as_device_arrays(device)` helper

- [ ] **Step 1: Write the failing test**

Append to `scripts/adaptive/tests/test_controller.py`:

```python
import warp as wp


def test_controller_config_overrides_take_effect():
    """A custom ControllerConfig with growth_cap=2.0 limits dt growth to 2x
    per acceptance, while default growth_cap=5.0 allows 5x. Detect by running
    a no-op step_fn (error=0, always accept) and watching dt grow."""
    from scripts.adaptive.base import AdaptiveWrapper
    from scripts.adaptive.controller import ControllerConfig
    from scripts.scenes import _registry

    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()

    def _noop(m, si, so, ctrl, c, dta, dts):
        pass

    # Default: growth_cap=5
    w_def = AdaptiveWrapper(
        model=model, step_fn=_noop, tol=1e-3, dt_init=1e-6,
        dt_min=1e-6, dt_max=1.0, dt_outer=1.0,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()
    s0, s1 = w_def.step_dt(1.0, s0, s1, ctrl)
    wp.synchronize()
    dt_default_iter1 = float(w_def._ideal_dt.numpy().max())

    # Restrictive: growth_cap=2
    model2 = scene.build_model_randomized(4)
    contacts2 = model2.contacts()
    w_restr = AdaptiveWrapper(
        model=model2, step_fn=_noop, tol=1e-3, dt_init=1e-6,
        dt_min=1e-6, dt_max=1.0, dt_outer=1.0,
        needs_collide=False, contacts=contacts2,
        controller=ControllerConfig(growth_cap=2.0),
    )
    s0r, s1r, ctrlr = model2.state(), model2.state(), model2.control()
    s0r, s1r = w_restr.step_dt(1.0, s0r, s1r, ctrlr)
    wp.synchronize()
    dt_restr_iter1 = float(w_restr._ideal_dt.numpy().max())

    # default config should grow faster than restrictive
    assert dt_default_iter1 > dt_restr_iter1 * 2.0, (
        f"default ({dt_default_iter1}) should be > 2x restrictive ({dt_restr_iter1})"
    )
```

- [ ] **Step 2: Run, verify fails**

Run:
```bash
uv run python -m pytest scripts/adaptive/tests/test_controller.py::test_controller_config_overrides_take_effect -v
```

Expected: FAIL because `ControllerConfig` is currently a dead dataclass; `growth_cap` doesn't affect anything.

- [ ] **Step 3: Convert `_calc_adjusted_step` to take controller constants as kernel args**

In `scripts/adaptive/kernels.py`, replace the 5 `wp.constant` lines (59-63) and the `_calc_adjusted_step` body (66-110) with:

```python
# Note: v2 removed compile-time _DRAKE_* constants; they're now per-launch inputs
# threaded from ControllerConfig (see AdaptiveWrapper).

@wp.kernel
def _calc_adjusted_step(
    last_error: wp.array(dtype=wp.float32),
    dt: wp.array(dtype=wp.float32),
    ideal_dt: wp.array(dtype=wp.float32),
    accepted: wp.array(dtype=wp.bool),
    tol: float,
    dt_min: float,
    safety: float,       # NEW: from ControllerConfig.safety_factor
    min_shrink: float,   # NEW: from ControllerConfig.shrink_cap
    max_grow: float,     # NEW: from ControllerConfig.growth_cap
    hyst_high: float,    # NEW: 1.0 + 2.0*(1.0 - safety) by default; for now hardcode
    hyst_low: float,     # NEW: hardcoded 0.9 default
):
    world = wp.tid()
    e = last_error[world]
    step = dt[world]

    if e > tol:
        if step <= dt_min:
            accepted[world] = True
            ideal_dt[world] = step  # at the floor, accept anyway
            return
        accepted[world] = False
        ideal_dt[world] = min_shrink * step
        return

    accepted[world] = True
    new_step = safety * step * wp.sqrt(tol / wp.max(e, wp.float32(1.0e-30)))
    if new_step > hyst_low * step and new_step < hyst_high * step:
        new_step = step  # hysteresis: don't churn
    new_step = wp.clamp(new_step, min_shrink * step, max_grow * step)
    ideal_dt[world] = new_step
```

Note: this preserves all v1 algorithm semantics, just moves the 5 constants from compile-time `wp.constant` to per-launch float scalars.

- [ ] **Step 4: Update `controller.py` with extra constants**

Replace `scripts/adaptive/controller.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drake PI controller tuning constants — wired into kernel via per-launch args."""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Drake PI dt controller config. Matches CENIC defaults."""
    safety_factor: float = 0.9
    growth_cap: float = 5.0
    shrink_cap: float = 0.1
    hysteresis_high: float = 1.2
    hysteresis_low: float = 0.9
    kp: float = 0.0
```

- [ ] **Step 5: Update `AdaptiveWrapper._run_iteration_body` to pass controller constants**

In `scripts/adaptive/base.py::_run_iteration_body`, find the `_calc_adjusted_step` launch and update the inputs list:

```python
        wp.launch(
            K._calc_adjusted_step, dim=n,
            inputs=[
                self._last_error, self._dt, self._ideal_dt, self._accepted,
                self.tol, self.dt_min,
                self.controller.safety_factor,
                self.controller.shrink_cap,
                self.controller.growth_cap,
                self.controller.hysteresis_high,
                self.controller.hysteresis_low,
            ],
            device=dev,
        )
```

- [ ] **Step 6: Run controller test + full suite**

Run:
```bash
uv run python -m pytest scripts/adaptive/tests/test_controller.py -v
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: all pass (25 total).

If `test_adaptive_mujoco_matches_cenic_oracle` regresses, the constants changed accidentally. Verify v1 defaults (0.9, 5.0, 0.1, 1.2, 0.9) are exactly what's passed.

---

### Task 4: Phase 1 verification + bench

**Files:** none

- [ ] **Step 1: Run full v1+v2 suite**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: 25 passed.

- [ ] **Step 2: Run the wow bench again, compare wall times against v1**

```bash
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 4 16 64 256 1024 4096 16384 32768 --steps 50 --warmup 10
```

Compare new JSON at `scripts/bench/results/<commit>/scaling_falling_cylinder.json` to the v1 baseline. Acceptance:
- `mujoco_adaptive_1e-3` at N=32768: target ~20% reduction in wall time vs v1 (v1 was 60.73ms).
- `xpbd_adaptive_1e-3` and `semi_adaptive_1e-3`: marginal change expected (K=1 means few iters, syncs already amortized).
- `mujoco_cenic_1e-3`: unchanged baseline (CENIC not touched in Phase 1).

If no measurable improvement, the two perf fixes didn't actually fire. Verify Task 1 packed sync and Task 2 hoisted reductions are both reached.

---

## Phase 2: Real mjw_data compaction (Agent B, ~3-5 days)

This is where `adaptive_mujoco` earns its keep — actually shrinking the GPU work as active worlds drop off.

### Task 5: Transform + spatial_vector gather/scatter kernels

**Files:**
- Modify: `scripts/adaptive/compaction_mixin.py`

**Background.** v1 has `_compact_gather_float` / `_compact_scatter_float` for joint_q/joint_qd. For body_q (transform) and body_qd (spatial_vector) we need dedicated kernels because Warp kernels can't be generic over dtype.

- [ ] **Step 1: Write the failing test**

Append to `scripts/adaptive/tests/test_compaction.py`:

```python
def test_compact_gather_scatter_transform_roundtrip():
    """Transform (xform) round-trip preserves bytes for active worlds."""
    wp.init()
    n = 8
    bodies_per_world = 1
    active = wp.array(
        np.array([True, False, True, True, False, False, True, False], dtype=bool),
        dtype=wp.bool,
    )
    indices = wp.array(np.array([0, 2, 3, 6, 0, 0, 0, 0], dtype=np.int32), dtype=wp.int32)
    n_active = 4

    import warp as wp_
    # Random-ish transforms (xform = 7 floats: px, py, pz, qx, qy, qz, qw)
    canon_np = np.arange(n * bodies_per_world * 7, dtype=np.float32).reshape(n * bodies_per_world, 7)
    canon = wp_.array(canon_np, dtype=wp_.transform)
    compact = wp_.zeros(n_active * bodies_per_world, dtype=wp_.transform)

    from scripts.adaptive.compaction_mixin import (
        _compact_gather_transform, _compact_scatter_transform,
    )
    wp.launch(_compact_gather_transform, dim=n_active,
              inputs=[canon, indices, bodies_per_world], outputs=[compact])
    wp.synchronize()
    got = compact.numpy()
    np.testing.assert_array_equal(got[0], canon_np[0])
    np.testing.assert_array_equal(got[1], canon_np[2])
    np.testing.assert_array_equal(got[2], canon_np[3])
    np.testing.assert_array_equal(got[3], canon_np[6])

    canon2 = wp_.zeros(n * bodies_per_world, dtype=wp_.transform)
    wp.launch(_compact_scatter_transform, dim=n_active,
              inputs=[compact, indices, bodies_per_world], outputs=[canon2])
    wp.synchronize()
    out_canon = canon2.numpy()
    for k, w_idx in enumerate([0, 2, 3, 6]):
        np.testing.assert_array_equal(out_canon[w_idx], canon_np[w_idx])


def test_compact_gather_scatter_spatial_vector_roundtrip():
    """spatial_vector round-trip preserves bytes for active worlds."""
    import warp as wp_
    wp.init()
    n = 8
    bodies_per_world = 1
    indices = wp_.array(np.array([0, 2, 3, 6, 0, 0, 0, 0], dtype=np.int32), dtype=wp_.int32)
    n_active = 4

    canon_np = np.arange(n * bodies_per_world * 6, dtype=np.float32).reshape(n * bodies_per_world, 6)
    canon = wp_.array(canon_np, dtype=wp_.spatial_vector)
    compact = wp_.zeros(n_active * bodies_per_world, dtype=wp_.spatial_vector)

    from scripts.adaptive.compaction_mixin import (
        _compact_gather_spatial_vector, _compact_scatter_spatial_vector,
    )
    wp.launch(_compact_gather_spatial_vector, dim=n_active,
              inputs=[canon, indices, bodies_per_world], outputs=[compact])
    wp.synchronize()
    got = compact.numpy()
    np.testing.assert_array_equal(got[0], canon_np[0])
    np.testing.assert_array_equal(got[1], canon_np[2])
    np.testing.assert_array_equal(got[2], canon_np[3])
    np.testing.assert_array_equal(got[3], canon_np[6])

    canon2 = wp_.zeros(n * bodies_per_world, dtype=wp_.spatial_vector)
    wp.launch(_compact_scatter_spatial_vector, dim=n_active,
              inputs=[compact, indices, bodies_per_world], outputs=[canon2])
    wp.synchronize()
    out_canon = canon2.numpy()
    for k, w_idx in enumerate([0, 2, 3, 6]):
        np.testing.assert_array_equal(out_canon[w_idx], canon_np[w_idx])
```

- [ ] **Step 2: Run, verify fails on import**

```bash
uv run python -m pytest scripts/adaptive/tests/test_compaction.py -v
```
Expected: FAIL with `ImportError: cannot import name '_compact_gather_transform'`.

- [ ] **Step 3: Implement the 4 new kernels**

Append to `scripts/adaptive/compaction_mixin.py` (after the existing `_compact_scatter_float`):

```python
@wp.kernel
def _compact_gather_transform(
    canonical: wp.array(dtype=wp.transform),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    compact_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    k = tid
    src_world = active_indices[k]
    for j in range(per_world):
        compact_out[k * per_world + j] = canonical[src_world * per_world + j]


@wp.kernel
def _compact_scatter_transform(
    compact: wp.array(dtype=wp.transform),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    canonical_out: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    k = tid
    dst_world = active_indices[k]
    for j in range(per_world):
        canonical_out[dst_world * per_world + j] = compact[k * per_world + j]


@wp.kernel
def _compact_gather_spatial_vector(
    canonical: wp.array(dtype=wp.spatial_vector),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    compact_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    k = tid
    src_world = active_indices[k]
    for j in range(per_world):
        compact_out[k * per_world + j] = canonical[src_world * per_world + j]


@wp.kernel
def _compact_scatter_spatial_vector(
    compact: wp.array(dtype=wp.spatial_vector),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    canonical_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    k = tid
    dst_world = active_indices[k]
    for j in range(per_world):
        canonical_out[dst_world * per_world + j] = compact[k * per_world + j]
```

- [ ] **Step 4: Run tests + full suite**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: 27 passed (25 prior + 2 new compaction tests).

---

### Task 6: Pre-allocate mjw_data tiers in `adaptive_mujoco_factory`

**Files:**
- Modify: `scripts/adaptive/factories.py::adaptive_mujoco_factory`

**Background.** v2 keeps multiple `SolverMuJoCo` instances pre-built at sizes `[N, N/4, N/16, N/64]`. When active count drops below a threshold, we switch to the smaller solver. Each instance has its own `mjw_data` sized for that world count.

The OOM fallback chain: try full `[N, N/4, N/16, N/64]`; if any allocation fails, fall back to `[N, N/4, N/16]`, then `[N, N/4]`, then `[N]` (compaction disabled).

- [ ] **Step 1: Write the failing tier-allocation test**

Append to `scripts/adaptive/tests/test_compaction.py`:

```python
def test_adaptive_mujoco_pre_allocates_tier_solvers():
    """adaptive_mujoco_factory builds a list of SolverMuJoCo instances at
    compaction_sizes fractions of N."""
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes import _registry
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    N = 64
    m = scene.build_model_randomized(N)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
        compaction_sizes=(1.0, 0.25, 0.0625),
    )
    wrapper, _ = builder(m)

    # Wrapper now has _tier_solvers (Phase 2 attribute)
    assert hasattr(wrapper, "_tier_solvers"), "Phase 2 should expose _tier_solvers"
    tier_sizes = [t.model.world_count for t in wrapper._tier_solvers]
    assert tier_sizes == [64, 16, 4], f"expected [64, 16, 4]; got {tier_sizes}"
```

Note: this test requires `wrapper._tier_solvers` to exist. In Phase 2, the factory pre-builds tier-sized SolverMuJoCo instances (each with its own subset model). For now this test will fail with `AttributeError`.

The hard part: each tier needs its own `Model` object sized for that world count. We pre-build `N_tier` separate `Model` objects via `scene.build_model_randomized(N_tier)`. It costs construction time but keeps the architecture simple. (Alternative — `model.subset(active_world_ids)` — would require Newton API support that doesn't exist today; not pursued.)

- [ ] **Step 2: Verify (a) is feasible**

Read `newton/_src/sim/model.py::Model` and confirm `build_model_randomized(N)` can be called multiple times with different N values without side-effects. It can (each call returns a fresh model).

- [ ] **Step 3: Implement tier pre-allocation in `adaptive_mujoco_factory`**

In `scripts/adaptive/factories.py::adaptive_mujoco_factory`, before the inner `def build(model):`, add a thread-through for the scene's `build_model_randomized` function. Then in `build`:

```python
def adaptive_mujoco_factory(
    *,
    tol, dt_init, dt_min, dt_max, dt_outer, nconmax, njmax,
    compaction_sizes=(1.0, 0.25, 0.0625, 0.015625),
    scene_build_fn=None,  # NEW: callable that takes N and returns a Model
):
    """Build (solver, step_fn) for MuJoCo adaptive with multi-tier compaction.

    scene_build_fn is required for compaction_sizes with multiple entries.
    """
    def build(model):
        N = model.world_count

        # Pre-build tier solvers + models.
        # OOM fallback: drop the smallest tiers if construction OOMs.
        tier_solvers = []
        tier_models = []
        sizes_remaining = list(compaction_sizes)

        while sizes_remaining:
            try:
                for frac in sizes_remaining:
                    n_tier = max(1, int(N * frac))
                    if tier_models and n_tier == tier_models[-1].world_count:
                        # Skip duplicate tier sizes (e.g. N=4 with sizes ending in /16 → 0)
                        continue
                    if scene_build_fn is not None:
                        m_tier = scene_build_fn(n_tier)
                    else:
                        # Fallback: use the largest tier's model for all sizes (no compaction win, but won't crash)
                        m_tier = model
                    s_tier = newton.solvers.SolverMuJoCo(
                        m_tier, separate_worlds=True, use_mujoco_contacts=False,
                        nconmax=nconmax, njmax=njmax,
                    )
                    tier_models.append(m_tier)
                    tier_solvers.append(s_tier)
                break  # construction succeeded
            except RuntimeError as exc:
                if "alloc" not in str(exc).lower() and "memory" not in str(exc).lower():
                    raise
                # Drop smallest tier and retry
                if len(sizes_remaining) <= 1:
                    raise RuntimeError(
                        f"adaptive_mujoco_factory: even full-N tier OOM'd: {exc}"
                    ) from exc
                sizes_remaining = sizes_remaining[:-1]
                tier_solvers.clear()
                tier_models.clear()

        # Build the wrapper, expose tier list for CompactionMixin to use
        wrapper = AdaptiveCompactionWrapper(
            model=model, step_fn=None,  # placeholder; set below
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False,
            compaction_sizes=sizes_remaining,
        )
        wrapper._tier_solvers = tier_solvers
        wrapper._tier_models = tier_models

        # ... (rest of factory: hooks, step_fn shim, etc. — see Task 7)
```

- [ ] **Step 4: Run the tier-allocation test**

```bash
uv run python -m pytest scripts/adaptive/tests/test_compaction.py::test_adaptive_mujoco_pre_allocates_tier_solvers -v
```
Expected: PASS.

- [ ] **Step 5: Existing tests must still pass**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: 28 passed.

If `test_adaptive_mujoco_factory_runs` or oracle test breaks: the factory wired tiers but didn't preserve the v1 step_fn / hooks. Re-add them.

---

### Task 7: Tier-swap logic in `CompactionMixin._run_iteration_body`

**Files:**
- Modify: `scripts/adaptive/compaction_mixin.py`
- Modify: `scripts/adaptive/factories.py::adaptive_mujoco_factory` (step_fn now dispatches to current tier)

**Background.** When `n_active <= 0.5 × current_tier.world_count`, swap to the next smaller tier. Steps:
1. Gather active state from current canonical layout into the smaller tier's State.
2. Update `wrapper._current_tier_idx`.
3. step_fn now operates on the smaller tier's solver.

Reverse at end of `step_dt`: scatter back to canonical N.

- [ ] **Step 1: Add `_current_tier_idx`, `_state_compact_pool`, swap logic to `CompactionMixin`**

In `scripts/adaptive/compaction_mixin.py`, extend `CompactionMixin.__init__` (after the existing v1 init):

```python
        # Phase 2: tier-swap state.
        self._current_tier_idx = 0  # 0 = full N; >0 = smaller tier
        # Compact state buffers (one per tier, lazily allocated on first swap).
        self._state_compact_full = {}  # tier_idx -> State
        self._state_compact_double = {}  # ditto
```

Replace `CompactionMixin._run_iteration_body` (v1 just updated mask) with:

```python
    def _run_iteration_body(self, effective_dt_max):
        # Decide whether to compact BEFORE running the iteration.
        self._maybe_compact()
        super()._run_iteration_body(effective_dt_max)
        # Update mask + count after.
        from scripts.adaptive import kernels as K
        n = self.model.world_count
        wp.launch(
            K._update_active_mask, dim=n,
            inputs=[self._sim_time, self._next_time, self._world_active],
            device=self.model.device,
        )
        self._update_active_count()

    def _maybe_compact(self):
        """If active count dropped below threshold, switch to next-smaller tier."""
        if not hasattr(self, "_tier_solvers") or len(self._tier_solvers) <= 1:
            return  # compaction disabled (only 1 tier)
        if self._current_tier_idx + 1 >= len(self._tier_solvers):
            return  # already at smallest tier
        current_size = self._tier_solvers[self._current_tier_idx].model.world_count
        n_active_host = self.n_active  # 1 PCIe sync — accept this cost at tier transitions
        if n_active_host > 0.5 * current_size:
            return  # not worth compacting yet
        next_idx = self._current_tier_idx + 1
        next_size = self._tier_solvers[next_idx].model.world_count
        if n_active_host > next_size:
            return  # next tier wouldn't hold active worlds
        # Compact: gather active state into next tier's State buffers
        self._compact_to_tier(next_idx)
        self._current_tier_idx = next_idx
```

- [ ] **Step 2: Implement `_compact_to_tier` (uses gather kernels)**

Append to `CompactionMixin`:

```python
    def _compact_to_tier(self, tier_idx: int):
        """Gather active worlds from current _state_cur into tier_idx's State.

        Updates self._state_cur to point at the new compact State.
        """
        from scripts.adaptive.compaction_mixin import (
            _compact_gather_float, _compact_gather_transform, _compact_gather_spatial_vector,
        )
        next_model = self._tier_models[tier_idx]
        if tier_idx not in self._state_compact_full:
            self._state_compact_full[tier_idx] = next_model.state()
            self._state_compact_double[tier_idx] = next_model.state()
        compact_state = self._state_compact_full[tier_idx]

        n_active = self.n_active
        coords_per_world = self._coords_per_world
        dofs_per_world = self._dofs_per_world
        bodies_per_world = self._bodies_per_world

        wp.launch(_compact_gather_float, dim=n_active,
                  inputs=[self._state_cur.joint_q, self._active_indices, coords_per_world],
                  outputs=[compact_state.joint_q])
        wp.launch(_compact_gather_float, dim=n_active,
                  inputs=[self._state_cur.joint_qd, self._active_indices, dofs_per_world],
                  outputs=[compact_state.joint_qd])
        if self._state_cur.body_q is not None and compact_state.body_q is not None:
            wp.launch(_compact_gather_transform, dim=n_active,
                      inputs=[self._state_cur.body_q, self._active_indices, bodies_per_world],
                      outputs=[compact_state.body_q])
        if self._state_cur.body_qd is not None and compact_state.body_qd is not None:
            wp.launch(_compact_gather_spatial_vector, dim=n_active,
                      inputs=[self._state_cur.body_qd, self._active_indices, bodies_per_world],
                      outputs=[compact_state.body_qd])

        # Save reference to canonical state for end-of-step_dt scatter-back
        if not hasattr(self, "_canonical_state_ref"):
            self._canonical_state_ref = self._state_cur
        self._state_cur = compact_state
```

- [ ] **Step 3: Implement `_scatter_back_to_canonical` (call at end of `step_dt`)**

Append to `CompactionMixin`:

```python
    def _scatter_back_to_canonical(self):
        """At end of step_dt, scatter compact state back to canonical N layout."""
        if self._current_tier_idx == 0:
            return  # never compacted, nothing to do
        from scripts.adaptive.compaction_mixin import (
            _compact_scatter_float, _compact_scatter_transform, _compact_scatter_spatial_vector,
        )
        compact_state = self._state_cur
        canonical_state = self._canonical_state_ref
        n_active = self.n_active

        wp.launch(_compact_scatter_float, dim=n_active,
                  inputs=[compact_state.joint_q, self._active_indices, self._coords_per_world],
                  outputs=[canonical_state.joint_q])
        wp.launch(_compact_scatter_float, dim=n_active,
                  inputs=[compact_state.joint_qd, self._active_indices, self._dofs_per_world],
                  outputs=[canonical_state.joint_qd])
        if canonical_state.body_q is not None and compact_state.body_q is not None:
            wp.launch(_compact_scatter_transform, dim=n_active,
                      inputs=[compact_state.body_q, self._active_indices, self._bodies_per_world],
                      outputs=[canonical_state.body_q])
        if canonical_state.body_qd is not None and compact_state.body_qd is not None:
            wp.launch(_compact_scatter_spatial_vector, dim=n_active,
                      inputs=[compact_state.body_qd, self._active_indices, self._bodies_per_world],
                      outputs=[canonical_state.body_qd])

        self._state_cur = canonical_state
        self._current_tier_idx = 0

    def step_dt(self, dt_outer, state_0, state_1, control):
        result = super().step_dt(dt_outer, state_0, state_1, control)
        self._scatter_back_to_canonical()
        return result
```

- [ ] **Step 4: Update `adaptive_mujoco_factory` step_fn shim to dispatch to current tier**

In `scripts/adaptive/factories.py::adaptive_mujoco_factory::build`, the step_fn shim needs to use the wrapper's current tier solver:

```python
        def step_fn(model_arg, state_in, state_out, ctrl, contacts, dt_array, dt_scalar_buf):
            current_tier_idx = wrapper._current_tier_idx
            tier_solver = wrapper._tier_solvers[current_tier_idx]
            # NOTE: dt_array length is based on the current tier's world count,
            # not the wrapper's full N. The wrapper's _dt array is sized for N;
            # but when compacted, we need a tier-sized dt array. Use the wrapper's
            # _compact_dt scratch (new Phase 2 attribute) which is gather'd before
            # this call.
            wp.copy(tier_solver.mjw_model.opt.timestep, dt_array)
            # Tier solver's _run_substep path (lifted from CENIC)
            tier_solver._update_mjc_data(tier_solver.mjw_data, tier_solver.model, state_in)
            with wp.ScopedDevice(tier_solver.model.device):
                tier_solver._mujoco_warp_step()
            tier_solver._update_newton_state(tier_solver.model, state_out, tier_solver.mjw_data)
```

- [ ] **Step 5: Compaction correctness test**

Append to `scripts/adaptive/tests/test_integration.py`:

```python
def test_adaptive_mujoco_with_compaction_matches_oracle():
    """adaptive_mujoco with multi-tier compaction matches CENIC oracle.

    This is the Phase 2 correctness gate. If compaction shuffles state
    incorrectly, this fails.
    """
    import numpy as np
    import newton, newton.solvers
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    N = 16
    n_outer = 100

    # Oracle: CENIC
    m_oracle = scene.build_model_randomized(N, seed=42)
    cenic = newton.solvers.SolverMuJoCoCENIC(
        m_oracle, tol=1e-3, dt_inner_init=DT_OUTER, dt_inner_min=1e-6,
        dt_inner_max=DT_OUTER, nconmax=8, njmax=32,
    )
    s0_o, s1_o, ctrl_o = m_oracle.state(), m_oracle.state(), m_oracle.control()
    for _ in range(n_outer):
        s0_o, s1_o = cenic.step_dt(DT_OUTER, s0_o, s1_o, ctrl_o)
    wp.synchronize()
    oracle_bq = s0_o.body_q.numpy()

    # adaptive_mujoco with multi-tier compaction
    m_wrap = scene.build_model_randomized(N, seed=42)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
        compaction_sizes=(1.0, 0.25),
    )
    _, step_fn = builder(m_wrap)
    s0_w, s1_w, ctrl_w = m_wrap.state(), m_wrap.state(), m_wrap.control()
    for _ in range(n_outer):
        s0_w, s1_w = step_fn(m_wrap, s0_w, s1_w, ctrl_w)
    wp.synchronize()
    wrap_bq = s0_w.body_q.numpy()

    np.testing.assert_allclose(
        wrap_bq, oracle_bq, rtol=1e-3, atol=1e-4,  # looser tol than N=4 oracle
        err_msg="compaction broke equivalence with CENIC oracle",
    )
```

- [ ] **Step 6: Run compaction tests**

```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_mujoco_with_compaction_matches_oracle -v
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: all pass.

If oracle FAILS, the gather/scatter is misaligned. Debug paths:
1. Verify `_active_indices` is freshly written before each compaction transition.
2. Verify `bodies_per_world` matches tier's geometry.
3. Verify `mjw_data._update_mjc_data` reads from the compact state arrays (not the wrapper's canonical scratch).

---

### Task 8: Phase 2 bench — verify compaction actually wins

**Files:** none

- [ ] **Step 1: Re-run the wow bench**

```bash
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 4 16 64 256 1024 4096 16384 32768 --steps 50 --warmup 10
```

- [ ] **Step 2: Compare adaptive_mujoco vs CENIC at high N**

Read `scripts/bench/results/<commit>/scaling_falling_cylinder.json`. Acceptance:

- `mujoco_adaptive_1e-3` at N=32768 wall time: **target ≤ 0.7 × `mujoco_cenic_1e-3`** (v1 baseline showed adaptive was 1.2× slower; v2 with real compaction should flip this).
- If adaptive_mujoco still ≥ CENIC: compaction isn't firing or is firing too rarely. Debug: log per-iter `_current_tier_idx` to see if tier swaps happen during the bench. If not, the bench's scene has all worlds active throughout (no benefit from compaction).
- For this scene (single cylinder, K=1 most of the time): compaction may not provide much win because active count stays at N. Consider noting that the v2 compaction win is for K ≥ 5 scenes (dense contact), not this falling_cylinder scene.

---

## Phase 3: Refactor `SolverMuJoCoCENIC` to use the wrapper (Agent C, ~2-3 days)

This eliminates ~400 lines of duplicated code from `solver_mujoco_cenic.py` by delegating to `AdaptiveCompactionWrapper`. The public API of CENIC must remain stable since external callers (the scenes, the bench, the demos) use it.

### Task 9: Identify CENIC's public API surface

**Files:**
- Read: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`
- Grep callers across the repo

- [ ] **Step 1: List all external use of `SolverMuJoCoCENIC` symbols**

```bash
grep -rn "SolverMuJoCoCENIC\b" --include='*.py' \
  scripts/ newton/examples/ docs/ 2>&1 | tee /tmp/cenic_callers.txt
```

Read `/tmp/cenic_callers.txt` and list every:
- Constructor call: with which kwargs?
- Method call: which methods?
- Attribute access: which attrs?

- [ ] **Step 2: Read CENIC's class definition and document the API**

In `newton/_src/solvers/mujoco/solver_mujoco_cenic.py::class SolverMuJoCoCENIC`, find all public (no leading underscore) methods/properties + the constructor signature. Write the contract to a new file:

`docs/superpowers/specs/2026-05-17-cenic-public-api.md`:

```markdown
# SolverMuJoCoCENIC Public API (to preserve in Phase 3 refactor)

## Constructor

SolverMuJoCoCENIC(model, *, tol, dt_inner_init, dt_inner_min, dt_inner_max,
                  dt_mode="per_world", **kwargs)

## Methods

- step_dt(dt_outer, state_0, state_1, control, apply_forces=None) -> (s0, s1)
- get_status_summary() -> dict[str, float]
- register_custom_attributes(builder)  [classmethod]

## Properties

- iteration_count -> wp.array(1, int32)
- dt -> wp.array(N, float32)
- sim_time -> wp.array(N, float32)
- last_error -> wp.array(N, float32)
- accepted -> wp.array(N, bool)
- accepted_error -> wp.array(N, float32)
- mjw_data, mjw_model  (inherited from SolverMuJoCo)
```

Note any attrs you discover beyond this list and add them.

---

### Task 10: Build CENIC compat shim that delegates to `AdaptiveCompactionWrapper`

**Files:**
- Modify: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`

**Background.** Keep `SolverMuJoCoCENIC` as a class (so existing constructor calls still work), but internally it builds an `AdaptiveCompactionWrapper` and forwards calls.

- [ ] **Step 1: Write the compat test BEFORE refactoring**

Append to `scripts/adaptive/tests/test_integration.py`:

```python
def test_cenic_public_api_unchanged_after_refactor():
    """Every public CENIC API element documented in 2026-05-17-cenic-public-api.md
    still works after the Phase 3 refactor."""
    import newton, newton.solvers
    import warp as wp
    from scripts.scenes import _registry
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)

    solver = newton.solvers.SolverMuJoCoCENIC(
        m, tol=1e-3, dt_inner_init=0.01, dt_inner_min=1e-6,
        dt_inner_max=0.01, nconmax=8, njmax=32,
    )
    s0, s1, ctrl = m.state(), m.state(), m.control()

    # Method 1: step_dt
    s0, s1 = solver.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()

    # Method 2: get_status_summary
    summary = solver.get_status_summary()
    assert set(summary.keys()) >= {"sim_time_min", "sim_time_max", "error_max",
                                    "accept_count", "dt_min", "dt_max"}

    # Properties
    assert solver.iteration_count.shape == (1,)
    assert solver.dt.shape == (4,)
    assert solver.sim_time.shape == (4,)
    assert solver.last_error.shape == (4,)
    assert solver.accepted.shape == (4,)
    assert solver.accepted_error.shape == (4,)
    # Inherited from SolverMuJoCo
    assert solver.mjw_data is not None
    assert solver.mjw_model is not None
```

Run; should PASS today (before refactor). Save the output — this is the spec the refactor must preserve.

```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_cenic_public_api_unchanged_after_refactor -v
```

- [ ] **Step 2: Backup CENIC + write refactored shim**

Save the original CENIC file:
```bash
cp newton/_src/solvers/mujoco/solver_mujoco_cenic.py /tmp/solver_mujoco_cenic.py.v1backup
```

Refactor `newton/_src/solvers/mujoco/solver_mujoco_cenic.py::class SolverMuJoCoCENIC` to delegate. The new implementation:

```python
# At top of file, remove the 14 @wp.kernel definitions and 5 wp.constants
# (they live in scripts/adaptive/kernels.py now).
# Keep imports clean.

from scripts.adaptive.base import AdaptiveWrapper
from scripts.adaptive.compaction_mixin import CompactionMixin
from scripts.adaptive.factories import (
    AdaptiveCompactionWrapper,
    _build_mujoco_q_weights,
)
from scripts.adaptive.controller import ControllerConfig


class SolverMuJoCoCENIC(SolverMuJoCo):
    """Adaptive-step MuJoCo solver — v2 thin shim over AdaptiveCompactionWrapper.

    Preserves public API; algorithm and kernels live in scripts/adaptive/.
    """

    def __init__(self, model, *, tol=1e-3, dt_inner_init=0.01, dt_inner_min=1e-6,
                 dt_inner_max=None, dt_mode="per_world", **kwargs):
        super().__init__(model, separate_worlds=True, use_mujoco_cpu=False,
                         use_mujoco_contacts=False, **kwargs)
        if dt_mode not in ("per_world", "global"):
            raise ValueError(f"dt_mode must be 'per_world' or 'global', got {dt_mode!r}")

        # Build the wrapper. step_fn shim calls our own substep path.
        self._wrapper = self._build_wrapper(
            tol=tol, dt_init=dt_inner_init, dt_min=dt_inner_min,
            dt_max=dt_inner_max or dt_inner_init,
        )

    def _build_wrapper(self, tol, dt_init, dt_min, dt_max):
        # Adapter: step_fn calls self._run_substep (already on SolverMuJoCo)
        def _step_fn(model, state_in, state_out, ctrl, contacts, dt_array, dt_scalar_buf):
            self._update_mjc_data(self.mjw_data, self.model, state_in)
            wp.copy(self.mjw_model.opt.timestep, dt_array)
            with wp.ScopedDevice(self.model.device):
                self._mujoco_warp_step()
            self._update_newton_state(self.model, state_out, self.mjw_data)

        q_weights = _build_mujoco_q_weights(self.model, self.mjw_model)
        return AdaptiveCompactionWrapper(
            model=self.model, step_fn=_step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_init,  # safe default; step_dt overrides per call
            needs_collide=False,
            q_weights=q_weights,
        )

    def step_dt(self, dt_outer, state_0, state_1, control, apply_forces=None):
        # apply_forces hook: call once before the inner loop runs.
        if apply_forces is not None:
            apply_forces(state_0)
        return self._wrapper.step_dt(dt_outer, state_0, state_1, control)

    def get_status_summary(self):
        return self._wrapper.status_summary()

    # Property forwards
    @property
    def iteration_count(self): return self._wrapper._iteration_count_buf
    @property
    def dt(self): return self._wrapper._dt
    @property
    def sim_time(self): return self._wrapper._sim_time
    @property
    def last_error(self): return self._wrapper._last_error
    @property
    def accepted(self): return self._wrapper._accepted
    @property
    def accepted_error(self): return self._wrapper._accepted_error

    @classmethod
    def register_custom_attributes(cls, builder):
        # Unchanged from v1 — keep whatever was here
        pass  # PLACEHOLDER: copy v1 body verbatim
```

**IMPORTANT**: `register_custom_attributes` must be preserved EXACTLY from the v1 source (was inherited or defined?). Open `/tmp/solver_mujoco_cenic.py.v1backup` and copy that method body unchanged.

- [ ] **Step 3: Run the compat test**

```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_cenic_public_api_unchanged_after_refactor -v
```
Expected: PASS. If any attribute is missing/wrong, add a property forward.

- [ ] **Step 4: Run the full adaptive suite**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: all pass.

- [ ] **Step 5: Run the bench across all scenes to catch regressions in CENIC callers**

```bash
for scene in contact_objects falling_cylinder falling_gripper anymal_clutter; do
  uv run -m scripts.bench --only scaling --scene $scene --ns 1 4 --steps 3 --warmup 2 2>&1 | tail -8
done
```

Expected: all 4 scenes' bench summaries print without crashes. Compare `mujoco_cenic_1e-3` numbers to v1 baseline — they should be very close (the refactor is supposed to be functionally identical).

- [ ] **Step 6: Run the demo**

```bash
uv run python -m scripts.demos.contact_objects --num-worlds 4 --num-steps 50 --headless 2>&1 | tail -10
```
Expected: completes without crash. Demo uses CENIC — verifies the refactor didn't break demo paths.

---

### Task 11: Delete dead code from CENIC

**Files:**
- Modify: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`

After Task 10's shim works, the original kernel definitions, controller logic, boundary loop, etc. inside `solver_mujoco_cenic.py` are dead code (the wrapper does it all now).

- [ ] **Step 1: Identify dead code**

In `newton/_src/solvers/mujoco/solver_mujoco_cenic.py`, every:
- `@wp.kernel` at the top of the file (these are duplicates of `scripts/adaptive/kernels.py`)
- `wp.constant` for `_DRAKE_*`
- Old private methods on `SolverMuJoCoCENIC` like `_run_substep`, `_run_iteration_body`, `_apply_dt_cap`, etc.

All of these are now in `scripts/adaptive/`. Delete them from CENIC.

- [ ] **Step 2: Delete + verify tests still pass**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: all pass. If any test breaks, you deleted something still in use — restore from backup.

- [ ] **Step 3: Run the bench again**

```bash
for scene in contact_objects falling_cylinder; do
  uv run -m scripts.bench --only scaling --scene $scene --ns 1 4 --steps 3 --warmup 2 2>&1 | tail -8
done
```
Expected: same numbers as Task 10 Step 5.

- [ ] **Step 4: Measure code reduction**

```bash
wc -l newton/_src/solvers/mujoco/solver_mujoco_cenic.py
```
Compare to v1 (`/tmp/solver_mujoco_cenic.py.v1backup`):
```bash
wc -l /tmp/solver_mujoco_cenic.py.v1backup
```

Acceptance: post-refactor file should be < 200 LoC (v1 was 830 LoC). If significantly larger, more dead code remains.

---

## Verification: full v1+v2 test suite passes

- [ ] **Run all adaptive tests**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```
Expected: all tests pass.

- [ ] **HARD GATE (still applies): N=1024 stability on all 3 adaptive variants**

```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_variant_stable_at_n_1024 -v
```
Expected: 3 passed.

- [ ] **Regression bench across all scenes**

```bash
for scene in contact_objects falling_cylinder falling_gripper anymal_clutter; do
  uv run -m scripts.bench --only scaling --scene $scene --ns 1 4 --steps 3 --warmup 2 2>&1 | tail -8
done
```
Expected: all 4 scenes' kinds run cleanly.

---

## Self-review

**1. Spec coverage:**

| v2 item | Implemented in |
|---|---|
| Critical: collapse 2 PCIe syncs into 1 | Task 1 |
| Critical: hoist `dt_array.max()` out of XPBD/Semi inner loop | Task 2 |
| Important: real `mjw_data` tier swap compaction | Tasks 5-7 |
| Important: wire ControllerConfig into kernel | Task 3 |
| Refactor SolverMuJoCoCENIC to use this wrapper | Tasks 9-11 |

**2. Placeholder scan:** the file has no "TBD", "TODO", "fill in details" markers. The one `# PLACEHOLDER: copy v1 body verbatim` in Task 10 Step 2 is intentional — the engineer must copy from `/tmp/solver_mujoco_cenic.py.v1backup`.

**3. Type consistency:** `StepFn` signature changes from 6 args to 7 args in Task 2 Step 3. All shim updates (Steps 4-7 of Task 2) follow the new signature consistently. The MuJoCo factory shim in Tasks 6 and 7 (Phase 2) is also updated.

**4. Scope:** 11 tasks across 3 phases. Each phase is independently shippable:
- Phase 1 ships after Task 4 (perf fixes, ~half day)
- Phase 2 ships after Task 8 (real compaction, ~3-5 days)
- Phase 3 ships after Task 11 (CENIC refactor, ~2-3 days)

Recommend executing one phase at a time, validating with the bench between phases.

End of plan.
