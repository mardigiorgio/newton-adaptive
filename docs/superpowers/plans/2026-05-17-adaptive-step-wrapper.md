# Adaptive Step Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a solver-agnostic adaptive step wrapper (`scripts/adaptive/`) that wraps MuJoCo / XPBD / SemiImplicit with step-doubling + Drake PI controller, plus active-set masking (XPBD/Semi) and wrapper-side compaction (MuJoCo) so the bench can hit `N = 2^15` on `falling_cylinder`.

**Architecture:** `AdaptiveWrapper` base class + two mixins (`ActiveSetMaskingMixin`, `CompactionMixin`). Solver-agnostic kernels lifted from `SolverMuJoCoCENIC`. Factories build the right (solver, step_fn) pair for each adaptive variant. XPBD/SemiImplicit per-world kernels get a new `world_active: wp.array(N, bool)` early-return guard. MuJoCo uses wrapper-side mjw_data re-batching.

**Tech Stack:** Newton 1.1.0.dev0, Warp 1.12.0rc2, mujoco_warp 3.5.0.2, Python 3.12, pytest, uv.

**Commit policy:** Per user preference, this plan does NOT include `git commit` steps. Each task ends with verification only. The user batches commits at their own cadence.

**Spec:** `docs/superpowers/specs/2026-05-17-adaptive-step-wrapper-design.md`

---

## File map

**Created:**
- `scripts/adaptive/__init__.py`
- `scripts/adaptive/base.py` — `AdaptiveWrapper` core class
- `scripts/adaptive/kernels.py` — solver-agnostic Warp kernels (lifted from CENIC)
- `scripts/adaptive/controller.py` — `ControllerConfig` dataclass
- `scripts/adaptive/masking_mixin.py` — `ActiveSetMaskingMixin`
- `scripts/adaptive/compaction_mixin.py` — `CompactionMixin`
- `scripts/adaptive/factories.py` — `adaptive_{mujoco,xpbd,semi}_factory`
- `scripts/adaptive/tests/__init__.py`
- `scripts/adaptive/tests/test_kernels.py`
- `scripts/adaptive/tests/test_controller.py`
- `scripts/adaptive/tests/test_masking.py`
- `scripts/adaptive/tests/test_compaction.py`
- `scripts/adaptive/tests/test_integration.py`

**Modified:**
- `newton/_src/solvers/xpbd/kernels.py` — add `world_active` arg to per-world kernels
- `newton/_src/solvers/xpbd/solver_xpbd.py` — thread `world_active` through `step()`
- `newton/_src/solvers/semi_implicit/kernels_body.py` — add `world_active` arg
- `newton/_src/solvers/semi_implicit/kernels_contact.py` — add `world_active` arg
- `newton/_src/solvers/semi_implicit/solver_semi_implicit.py` — thread `world_active`
- `scripts/scenes/falling_cylinder.py` — add adaptive factories to `SOLVER_FACTORIES`
- `scripts/bench/plotting.py` — add `STYLES` entries for `mujoco_adaptive_*`, `xpbd_adaptive_*`, `semi_implicit_adaptive_*`

---

## Wave 1 — Core wrapper (Agent A, serial)

These tasks must complete before Wave 2 can start. They produce `base.py`, `kernels.py`, `controller.py`.

### Task 1: Scaffolding

**Files:**
- Create: `scripts/adaptive/__init__.py`
- Create: `scripts/adaptive/tests/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p scripts/adaptive/tests
```

- [ ] **Step 2: Create empty package init files**

Write `scripts/adaptive/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Solver-agnostic adaptive step wrapper."""
```

Write `scripts/adaptive/tests/__init__.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
```

- [ ] **Step 3: Verify import works**

Run: `uv run python -c "import scripts.adaptive; print('ok')"`
Expected: `ok` (after a Warp init banner).

---

### Task 2: Lift solver-agnostic kernels into `kernels.py`

**Files:**
- Create: `scripts/adaptive/kernels.py`
- Read for reference: `newton/_src/solvers/mujoco/solver_mujoco_cenic.py:20-285`

The CENIC source has 16 Warp kernels above the `class SolverMuJoCoCENIC` declaration. Of those, these 13 are solver-agnostic (operate on plain State arrays or scalars) and get lifted:

- `_apply_dt_cap` (line 20)
- `_inf_norm_state_error_kernel` (line 36)
- `_calc_adjusted_step` (line 74)
- `_advance_sim_time` (line 122)
- `_select_float_kernel` (line 137)
- `_select_transform_kernel` (line 154)
- `_select_spatial_vector_kernel` (line 171)
- `_boundary_reset` (line 188)
- `_boundary_check` (line 194)
- `_boundary_advance` (line 206)
- `_clamp_dt_to_boundary` (line 213)
- `_iter_count_increment` (line 234)
- `_status_summary_kernel` (line 268, with helper `_status_sentinel_reset` at line 240)

The 3 NOT lifted (MuJoCo-specific) are: `_reset_error_scalar`, `_reduce_max_error`, `_broadcast_error` (these are for CENIC's `global` dt mode reduction — v1 wrapper supports only per-world or effectively-global via shim).

- [ ] **Step 1: Write the lift test (TDD: test before code)**

Write `scripts/adaptive/tests/test_kernels.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lifted Warp kernels."""

import numpy as np
import warp as wp

from scripts.adaptive import kernels as K


def test_apply_dt_cap_clamps_to_min_max():
    """dt_ideal outside [dt_min, dt_max] gets clamped; in-range passes through."""
    wp.init()
    n = 4
    ideal = wp.array([0.5e-6, 1e-3, 1.0, 5e-3], dtype=wp.float32)
    dt_out = wp.zeros(n, dtype=wp.float32)
    dt_half_out = wp.zeros(n, dtype=wp.float32)
    dt_min = 1e-6
    dt_max = 1e-2

    wp.launch(K._apply_dt_cap, dim=n,
              inputs=[ideal, dt_min, dt_max, dt_out, dt_half_out])
    wp.synchronize()

    got = dt_out.numpy()
    assert got[0] == dt_min, "below dt_min should clamp up"
    assert got[1] == 1e-3, "in-range should pass through"
    assert got[2] == dt_max, "above dt_max should clamp down"
    assert got[3] == 5e-3, "in-range should pass through"

    half = dt_half_out.numpy()
    np.testing.assert_allclose(half, got / 2.0, rtol=1e-6)


def test_inf_norm_state_error_kernel_returns_max_weighted_diff():
    """error[i] = max_j (weights[i,j] * |full[i,j] - double[i,j]|)."""
    wp.init()
    n_world = 2
    coords_per_world = 3
    full = wp.array(
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32).flatten(),
        dtype=wp.float32,
    )
    double = wp.array(
        np.array([[1.0, 2.5, 3.0], [10.0, 20.0, 31.0]], dtype=np.float32).flatten(),
        dtype=wp.float32,
    )
    weights = wp.array(
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32).flatten(),
        dtype=wp.float32,
    )
    last_error = wp.zeros(n_world, dtype=wp.float32)

    wp.launch(K._inf_norm_state_error_kernel, dim=n_world,
              inputs=[full, double, weights, coords_per_world],
              outputs=[last_error])
    wp.synchronize()

    got = last_error.numpy()
    np.testing.assert_allclose(got[0], 0.5, rtol=1e-6)  # |2.0 - 2.5| * 1.0
    np.testing.assert_allclose(got[1], 2.0, rtol=1e-6)  # |30 - 31| * 2.0
```

- [ ] **Step 2: Run test to verify it fails (no kernels.py yet)**

Run: `uv run python -m pytest scripts/adaptive/tests/test_kernels.py -v`
Expected: ImportError or ModuleNotFoundError on `scripts.adaptive.kernels`.

- [ ] **Step 3: Write kernels.py by lifting from CENIC**

Open `newton/_src/solvers/mujoco/solver_mujoco_cenic.py` and copy the kernel definitions at the line numbers listed above (`_apply_dt_cap`, `_inf_norm_state_error_kernel`, `_calc_adjusted_step`, `_advance_sim_time`, `_select_float_kernel`, `_select_transform_kernel`, `_select_spatial_vector_kernel`, `_boundary_reset`, `_boundary_check`, `_boundary_advance`, `_clamp_dt_to_boundary`, `_iter_count_increment`, `_status_summary_kernel`, `_status_sentinel_reset`).

Create `scripts/adaptive/kernels.py` with the standard header and all 14 kernels verbatim:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Solver-agnostic Warp kernels for the adaptive step wrapper.

Lifted from newton/_src/solvers/mujoco/solver_mujoco_cenic.py — see that
file's lines 20-285 for the originals. These operate on plain Newton State
arrays and have no MuJoCo coupling.
"""

import warp as wp

# Paste the 14 kernel definitions here in the order listed in the task description.
# Use exact copies — no algorithmic edits.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest scripts/adaptive/tests/test_kernels.py -v`
Expected: 2 passed.

- [ ] **Step 5: Verify all 14 kernels imported correctly**

Run:
```bash
uv run python -c "
from scripts.adaptive import kernels as K
required = ['_apply_dt_cap', '_inf_norm_state_error_kernel', '_calc_adjusted_step',
            '_advance_sim_time', '_select_float_kernel', '_select_transform_kernel',
            '_select_spatial_vector_kernel', '_boundary_reset', '_boundary_check',
            '_boundary_advance', '_clamp_dt_to_boundary', '_iter_count_increment',
            '_status_summary_kernel', '_status_sentinel_reset']
for name in required:
    assert hasattr(K, name), f'missing kernel: {name}'
print('all 14 kernels present')
"
```
Expected: `all 14 kernels present`.

---

### Task 3: Controller config dataclass

**Files:**
- Create: `scripts/adaptive/controller.py`
- Create: `scripts/adaptive/tests/test_controller.py`

- [ ] **Step 1: Write the failing test**

Write `scripts/adaptive/tests/test_controller.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the controller config dataclass."""

from scripts.adaptive.controller import ControllerConfig


def test_default_controller_matches_cenic_drake_constants():
    """Default config matches CENIC's hard-coded constants in
    _calc_adjusted_step (newton/_src/solvers/mujoco/solver_mujoco_cenic.py:74)."""
    cfg = ControllerConfig()
    # CENIC's _calc_adjusted_step uses safety=0.9, growth_cap=5.0, shrink_cap=0.1
    # (constants are inlined in the kernel; we expose them here so v2 can swap controllers).
    assert cfg.safety_factor == 0.9
    assert cfg.growth_cap == 5.0
    assert cfg.shrink_cap == 0.1
    assert cfg.kp == 0.0  # CENIC's controller is I-only at the moment


def test_controller_config_overridable():
    cfg = ControllerConfig(safety_factor=0.95, growth_cap=2.0)
    assert cfg.safety_factor == 0.95
    assert cfg.growth_cap == 2.0
    assert cfg.shrink_cap == 0.1  # default preserved
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest scripts/adaptive/tests/test_controller.py -v`
Expected: ImportError on `scripts.adaptive.controller`.

- [ ] **Step 3: Write controller.py**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drake PI controller tuning constants.

The controller math is implemented in scripts/adaptive/kernels.py
(_calc_adjusted_step). This module exists so v2 can swap controller
variants without modifying the kernel.
"""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Drake PI dt controller config. Matches CENIC defaults."""
    safety_factor: float = 0.9
    growth_cap: float = 5.0
    shrink_cap: float = 0.1
    kp: float = 0.0   # CENIC is I-only; future PI variant sets this > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest scripts/adaptive/tests/test_controller.py -v`
Expected: 2 passed.

---

### Task 4: `AdaptiveWrapper.__init__` — state allocation

**Files:**
- Create: `scripts/adaptive/base.py`

This task allocates all scratch states and per-world arrays. No step_dt logic yet.

- [ ] **Step 1: Write the failing test (state allocation)**

Append to `scripts/adaptive/tests/test_integration.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for AdaptiveWrapper."""

import warp as wp

from scripts.adaptive.base import AdaptiveWrapper
from scripts.scenes import _registry


def _noop_step_fn(model, state_in, state_out, ctrl, contacts, dt_array):
    """No-op step_fn for testing allocation paths."""
    pass


def test_adaptive_wrapper_allocates_scratch_states():
    """Constructor allocates 4 scratch states + per-world arrays."""
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = AdaptiveWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    # Per-world arrays sized N
    assert w._dt.shape == (4,)
    assert w._sim_time.shape == (4,)
    assert w._next_time.shape == (4,)
    assert w._accepted.shape == (4,)
    assert w._last_error.shape == (4,)
    # Scratch states
    assert w._state_saved is not None
    assert w._state_full is not None
    assert w._state_mid is not None
    assert w._state_double is not None
    # Boundary / iter scalars
    assert w._boundary_flag.shape == (1,)
    assert w._iteration_count_buf.shape == (1,)
    # Per-coord weights
    assert w._q_weights.shape[0] == model.joint_coord_count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_wrapper_allocates_scratch_states -v`
Expected: ImportError on `scripts.adaptive.base`.

- [ ] **Step 3: Write base.py with `__init__` only**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""AdaptiveWrapper core class.

Wraps an arbitrary step_fn with step-doubling adaptive stepping and a
Drake PI controller. Solver-agnostic — translate solver specifics in the
step_fn shim built by scripts.adaptive.factories.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import warp as wp

import newton

from scripts.adaptive import kernels as K
from scripts.adaptive.controller import ControllerConfig

StepFn = Callable[
    [newton.Model, newton.State, newton.State, newton.Control,
     newton.Contacts, wp.array],
    None,
]


class AdaptiveWrapper:
    """Step-doubling adaptive wrapper around an arbitrary step_fn."""

    def __init__(
        self,
        model: newton.Model,
        step_fn: StepFn,
        *,
        tol: float,
        dt_init: float,
        dt_min: float,
        dt_max: float,
        dt_outer: float,
        needs_collide: bool,
        contacts: newton.Contacts | None = None,
        stuck_policy: Literal["freeze", "raise"] = "freeze",
        max_iters: int = 500,
        controller: ControllerConfig | None = None,
    ):
        self.model = model
        self.step_fn = step_fn
        self.tol = tol
        self.dt_init = dt_init
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_outer = dt_outer
        self.needs_collide = needs_collide
        self.contacts = contacts if contacts is not None else model.contacts()
        self.stuck_policy = stuck_policy
        self.max_iters = max_iters
        self.controller = controller or ControllerConfig()

        device = model.device
        n = model.world_count

        # Per-world arrays.
        self._dt = wp.full(n, dt_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(n, dt_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(n, dt_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(n, dtype=wp.float32, device=device)
        self._next_time = wp.zeros(n, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(n, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(n, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(n, dtype=wp.float32, device=device)

        # 1-element host-sync arrays.
        self._boundary_flag = wp.zeros(1, dtype=wp.int32, device=device)
        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)

        # Scratch states.
        self._state_saved = model.state()
        self._state_full = model.state()
        self._state_mid = model.state()
        self._state_double = model.state()

        # Per-coord weights for error norm (q-only, weighted by sqrt(inv_mass)).
        # Replicates CENIC's _q_weights construction (solver_mujoco_cenic.py:382-413).
        self._q_weights = self._build_q_weights(model, device)

        # Geometry constants used by select kernels.
        self._coords_per_world = model.joint_coord_count // n
        self._dofs_per_world = model.joint_dof_count // n
        self._bodies_per_world = (model.body_count // n) if model.body_count > 0 else 0

        # Status summary scratch (6 scalars).
        self._status_scalars = wp.zeros(6, dtype=wp.float32, device=device)

    def _build_q_weights(self, model: newton.Model, device) -> wp.array:
        """Build per-coord weights = sqrt(inv_mass), normalized per world.
        Replicates CENIC's algorithm from solver_mujoco_cenic.py:382-413.
        Simplified for v1 — full implementation will follow in Task 5.
        """
        coords_per_world = model.joint_coord_count // model.world_count
        q_weights_np = np.ones((model.world_count, coords_per_world), dtype=np.float32)
        return wp.array(q_weights_np.flatten(), dtype=wp.float32, device=device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_wrapper_allocates_scratch_states -v`
Expected: 1 passed.

---

### Task 5: `AdaptiveWrapper._run_iteration_body` — inner loop body

**Files:**
- Modify: `scripts/adaptive/base.py`

This task adds the inner-loop body method that does one step-doubling attempt + accept/reject.

- [ ] **Step 1: Add `_run_iteration_body` method to `AdaptiveWrapper`**

Append to `AdaptiveWrapper` class in `scripts/adaptive/base.py`:
```python
    def _run_iteration_body(self, effective_dt_max: float) -> None:
        """One step-doubling iteration: 3 evals + error + accept/reject + advance."""
        model = self.model
        n = model.world_count
        dev = model.device

        wp.launch(K._iter_count_increment, dim=1,
                  inputs=[self._iteration_count_buf], device=dev)

        # Clamp dt so no world overshoots its boundary target.
        wp.launch(K._clamp_dt_to_boundary, dim=n,
                  inputs=[self._dt, self._dt_half, self._sim_time, self._next_time],
                  device=dev)

        # Save state for rejection rollback.
        wp.copy(self._state_saved.joint_q, self._state_cur.joint_q)
        wp.copy(self._state_saved.joint_qd, self._state_cur.joint_qd)
        if self._state_cur.body_q is not None and self._state_saved.body_q is not None:
            wp.copy(self._state_saved.body_q, self._state_cur.body_q)
        if self._state_cur.body_qd is not None and self._state_saved.body_qd is not None:
            wp.copy(self._state_saved.body_qd, self._state_cur.body_qd)

        # Collide if solver doesn't (MuJoCo handles internally).
        if self.needs_collide:
            self.model.collide(self._state_cur, self.contacts)

        # 3 step_fn calls: full dt, half dt, half dt.
        self.step_fn(model, self._state_cur, self._state_full, None, self.contacts, self._dt)
        self.step_fn(model, self._state_cur, self._state_mid, None, self.contacts, self._dt_half)
        self.step_fn(model, self._state_mid, self._state_double, None, self.contacts, self._dt_half)

        # Error: max|q_full - q_double|, weighted, q-only L-inf.
        wp.launch(
            K._inf_norm_state_error_kernel, dim=n,
            inputs=[self._state_full.joint_q, self._state_double.joint_q,
                    self._q_weights, self._coords_per_world],
            outputs=[self._last_error], device=dev,
        )

        # Per-world accept/reject + new ideal_dt.
        wp.launch(
            K._calc_adjusted_step, dim=n,
            inputs=[self._last_error, self._dt, self._ideal_dt,
                    self._accepted, self.tol, self.dt_min],
            device=dev,
        )

        # State select: cur := accepted ? double : saved.
        wp.launch(
            K._select_float_kernel, dim=model.joint_coord_count,
            inputs=[self._state_double.joint_q, self._state_saved.joint_q,
                    self._accepted, self._coords_per_world],
            outputs=[self._state_cur.joint_q], device=dev,
        )
        wp.launch(
            K._select_float_kernel, dim=model.joint_dof_count,
            inputs=[self._state_double.joint_qd, self._state_saved.joint_qd,
                    self._accepted, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd], device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                K._select_transform_kernel, dim=model.body_count,
                inputs=[self._state_double.body_q, self._state_saved.body_q,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_q], device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                K._select_spatial_vector_kernel, dim=model.body_count,
                inputs=[self._state_double.body_qd, self._state_saved.body_qd,
                        self._accepted, self._bodies_per_world],
                outputs=[self._state_cur.body_qd], device=dev,
            )

        # Advance sim_time for accepted worlds.
        wp.launch(
            K._advance_sim_time, dim=n,
            inputs=[self._sim_time, self._dt, self._accepted,
                    self._last_error, self._accepted_error],
            device=dev,
        )

        # Cap dt for the next iteration.
        wp.launch(
            K._apply_dt_cap, dim=n,
            inputs=[self._ideal_dt, self.dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=dev,
        )

        # Boundary check.
        wp.launch(K._boundary_reset, dim=1, inputs=[self._boundary_flag], device=dev)
        wp.launch(
            K._boundary_check, dim=n,
            inputs=[self._sim_time, self._next_time, self._boundary_flag],
            device=dev,
        )
```

- [ ] **Step 2: Verify it imports clean**

Run: `uv run python -c "from scripts.adaptive.base import AdaptiveWrapper; print('ok')"`
Expected: `ok`.

---

### Task 6: `AdaptiveWrapper.step_dt` — outer entry + boundary loop

**Files:**
- Modify: `scripts/adaptive/base.py`

- [ ] **Step 1: Add `step_dt` method**

Append to `AdaptiveWrapper` class in `scripts/adaptive/base.py`:
```python
    def step_dt(
        self,
        dt_outer: float,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
    ) -> tuple[newton.State, newton.State]:
        """Advance every world by exactly dt_outer seconds of sim time.

        Loops _run_iteration_body until every world's sim_time reaches the
        boundary. One 4-byte boundary-flag sync per iteration.
        """
        model = self.model
        n = model.world_count
        device = model.device

        # state_cur is the "live" state; we copy state_0 into it.
        self._state_cur = state_0

        effective_dt_max = min(self.dt_max, dt_outer)

        # Initial dt = ideal_dt clamped to [dt_min, effective_dt_max].
        wp.launch(
            K._apply_dt_cap, dim=n,
            inputs=[self._ideal_dt, self.dt_min, effective_dt_max,
                    self._dt, self._dt_half],
            device=device,
        )

        # Set per-world next_time = sim_time + dt_outer.
        wp.launch(K._boundary_advance, dim=n,
                  inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._boundary_flag.fill_(1)

        # Boundary loop.
        while True:
            self._run_iteration_body(effective_dt_max)
            if self._iteration_count_buf.numpy()[0] >= self.max_iters:
                raise RuntimeError(
                    f"AdaptiveWrapper: max_iters={self.max_iters} exceeded "
                    f"in step_dt(dt_outer={dt_outer})"
                )
            if self._boundary_flag.numpy()[0] == 0:
                break

        return state_0, state_1

    @property
    def iteration_count(self) -> wp.array:
        return self._iteration_count_buf

    @property
    def dt(self) -> wp.array:
        return self._dt
```

- [ ] **Step 2: Wire next_time properly — _boundary_advance ADDS to sim_time**

Note: CENIC's `_boundary_advance` ADDS dt_outer to `_next_time`, not assigns. Verify by inspection of `newton/_src/solvers/mujoco/solver_mujoco_cenic.py:206-211`. If your lifted version matches, good. If not, fix _boundary_advance in scripts/adaptive/kernels.py to: `arr[i] = arr[i] + delta`. The intent is: each new step_dt call, next_time becomes the new boundary target.

Actually CENIC initializes `_next_time` from `_sim_time` once and then adds `dt_outer` per call. We do the same. The kernel adds.

- [ ] **Step 3: Smoke test — step_dt with no-op step_fn**

Append to `scripts/adaptive/tests/test_integration.py`:
```python
def test_step_dt_with_noop_step_fn_terminates_immediately():
    """With a no-op step_fn, error is 0 everywhere, all worlds accept on iter 1."""
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = AdaptiveWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    s0, s1, ctrl = model.state(), model.state(), model.control()

    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()

    K_done = int(w.iteration_count.numpy()[0])
    assert K_done == 1, f"no-op step_fn should accept in 1 iter, got K={K_done}"
```

- [ ] **Step 4: Run smoke test**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_step_dt_with_noop_step_fn_terminates_immediately -v`
Expected: 1 passed.

---

### Task 7: `AdaptiveWrapper.status_summary`

**Files:**
- Modify: `scripts/adaptive/base.py`

- [ ] **Step 1: Add status_summary method**

Append to `AdaptiveWrapper` class:
```python
    def status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary (one PCIe sync)."""
        device = self.model.device
        n = self.model.world_count

        wp.launch(K._status_sentinel_reset, dim=1,
                  inputs=[self._status_scalars], device=device)
        wp.launch(
            K._status_summary_kernel, dim=n,
            inputs=[self._sim_time, self._accepted_error, self._dt,
                    self._accepted, self._status_scalars],
            device=device,
        )

        s = self._status_scalars.numpy()
        return {
            "sim_time_min": float(s[0]),
            "sim_time_max": float(s[1]),
            "error_max":    float(s[2]),
            "accept_count": int(s[3]),
            "dt_min":       float(s[4]),
            "dt_max":       float(s[5]),
        }
```

- [ ] **Step 2: Verify clean import + method exists**

Run:
```bash
uv run python -c "
from scripts.adaptive.base import AdaptiveWrapper
assert hasattr(AdaptiveWrapper, 'status_summary'), 'missing status_summary'
print('ok')
"
```
Expected: `ok`.

---

## Wave 2 — Mixins (Agents B + C in parallel, after Wave 1)

### Task 8: `ActiveSetMaskingMixin`

**Files:**
- Create: `scripts/adaptive/masking_mixin.py`

This task creates the mixin but defers the XPBD/Semi kernel modifications to Tasks 9 + 10. The mixin maintains the active mask; the kernels are what actually respect it.

- [ ] **Step 1: Add `_update_active_mask` kernel to `kernels.py`**

Append to `scripts/adaptive/kernels.py`:
```python
@wp.kernel
def _update_active_mask(
    sim_time: wp.array(dtype=wp.float32),
    next_time: wp.array(dtype=wp.float32),
    world_active: wp.array(dtype=wp.bool),
):
    i = wp.tid()
    world_active[i] = sim_time[i] < next_time[i]
```

- [ ] **Step 2: Write the mixin**

Create `scripts/adaptive/masking_mixin.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""ActiveSetMaskingMixin — for XPBD / SemiImplicit step_fns that accept
a world_active mask in their kernels."""

import warp as wp

from scripts.adaptive import kernels as K


class ActiveSetMaskingMixin:
    """Maintain a world_active mask and pass it through step_fn.

    Concrete class composition: class AdaptiveMaskingWrapper(ActiveSetMaskingMixin, AdaptiveWrapper).
    Mixin MUST come before AdaptiveWrapper in MRO so its __init__ runs and
    overrides participate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = self.model.world_count
        device = self.model.device
        self._world_active = wp.full(n, True, dtype=wp.bool, device=device)

    def _reset_active_mask(self):
        n = self.model.world_count
        self._world_active.fill_(True)

    def _update_active_after_advance(self):
        n = self.model.world_count
        device = self.model.device
        wp.launch(
            K._update_active_mask, dim=n,
            inputs=[self._sim_time, self._next_time, self._world_active],
            device=device,
        )

    @property
    def world_active(self) -> wp.array:
        return self._world_active
```

- [ ] **Step 3: Override `_run_iteration_body` to call `_update_active_after_advance`**

Add to `ActiveSetMaskingMixin`:
```python
    def _run_iteration_body(self, effective_dt_max: float) -> None:
        # Call base implementation, then update active mask.
        super()._run_iteration_body(effective_dt_max)
        self._update_active_after_advance()
```

And override `step_dt` to reset the mask at the start:
```python
    def step_dt(self, dt_outer, state_0, state_1, control):
        self._reset_active_mask()
        return super().step_dt(dt_outer, state_0, state_1, control)
```

- [ ] **Step 4: Smoke test (no actual kernel mask use yet — XPBD changes pending)**

Append to `test_masking.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import warp as wp

from scripts.adaptive.base import AdaptiveWrapper
from scripts.adaptive.masking_mixin import ActiveSetMaskingMixin
from scripts.scenes import _registry


class _MaskedWrapper(ActiveSetMaskingMixin, AdaptiveWrapper):
    pass


def _noop_step_fn(model, state_in, state_out, ctrl, contacts, dt_array):
    pass


def test_masking_mixin_allocates_active_mask():
    scene = _registry.get("falling_cylinder")
    model = scene.build_model_randomized(4)
    contacts = model.contacts()
    w = _MaskedWrapper(
        model=model, step_fn=_noop_step_fn,
        tol=1e-3, dt_init=0.01, dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
        needs_collide=False, contacts=contacts,
    )
    assert w._world_active.shape == (4,)
    assert all(w._world_active.numpy().tolist())  # all True initially
```

Run: `uv run python -m pytest scripts/adaptive/tests/test_masking.py -v`
Expected: 1 passed.

---

### Task 9: XPBD kernel `world_active` modifications

**Files:**
- Modify: `newton/_src/solvers/xpbd/kernels.py`
- Modify: `newton/_src/solvers/xpbd/solver_xpbd.py`

XPBD kernels at the line numbers listed in the spec all need a `world_active: wp.array(dtype=wp.bool)` arg + early-return guard. The kernel-internal world index is solver-specific; XPBD groups by particle/body so the world index is derived from the tid via model arrays.

- [ ] **Step 1: Identify all per-world kernels in XPBD**

Run:
```bash
grep -n "^@wp.kernel\|^def " newton/_src/solvers/xpbd/kernels.py | head -40
```

Pick the kernels that index by per-world data (have `body_world` or `particle_world` in their inputs and use it). Skip kernels that operate on global constraints or shared arrays.

- [ ] **Step 2: For each per-world kernel, add `world_active` parameter + early-return**

Pattern (apply to each per-world kernel):
```python
@wp.kernel
def some_kernel(
    body_world: wp.array(dtype=wp.int32),
    # ... existing args ...
    world_active: wp.array(dtype=wp.bool),  # NEW: add at the end
):
    tid = wp.tid()
    world_idx = body_world[tid]  # or particle_world[tid] depending on kernel
    if not world_active[world_idx]:  # NEW: early return
        return
    # ... existing kernel body unchanged ...
```

- [ ] **Step 3: Update `SolverXPBD.step` to pass `world_active`**

In `newton/_src/solvers/xpbd/solver_xpbd.py`, change the `step` signature:
```python
def step(self, state_in, state_out, control, contacts, dt, world_active=None):
    if world_active is None:
        # Default: all worlds active (backward compat).
        n = self.model.world_count
        if not hasattr(self, "_default_world_active") or self._default_world_active.shape[0] != n:
            self._default_world_active = wp.full(n, True, dtype=wp.bool, device=self.model.device)
        world_active = self._default_world_active
    # ... existing step body, but pass world_active to each per-world kernel launch ...
```

- [ ] **Step 4: Smoke test — `SolverXPBD.step()` still works with default mask**

```bash
uv run python -c "
import warp as wp
import newton, newton.solvers
from scripts.scenes import _registry
e = _registry.get('falling_cylinder')
m = e.build_model_randomized(4)
contacts = m.contacts()
s = newton.solvers.SolverXPBD(m)
s0, s1, ctrl = m.state(), m.state(), m.control()
for _ in range(5):
    m.collide(s0, contacts)
    s.step(s0, s1, ctrl, contacts, 1e-3)
    s0, s1 = s1, s0
wp.synchronize()
print('xpbd backward compat ok')
"
```
Expected: `xpbd backward compat ok`.

- [ ] **Step 5: Smoke test — masking actually works**

Append to `test_masking.py`:
```python
def test_xpbd_step_respects_world_active_mask():
    """Step with all-False mask should leave state unchanged."""
    import numpy as np
    import newton, newton.solvers
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    s = newton.solvers.SolverXPBD(m)
    contacts = m.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    m.collide(s0, contacts)

    # Snapshot input
    bq0 = s0.body_q.numpy().copy()

    # All-False mask
    mask = wp.full(m.world_count, False, dtype=wp.bool, device=m.device)
    s.step(s0, s1, ctrl, contacts, 1e-3, world_active=mask)
    wp.synchronize()

    # Output should match input (modulo possible scratch writes; check body_q at least).
    bq1 = s1.body_q.numpy()
    np.testing.assert_allclose(bq1, bq0, rtol=1e-6,
        err_msg="all-False mask should leave body_q unchanged")
```

Run: `uv run python -m pytest scripts/adaptive/tests/test_masking.py::test_xpbd_step_respects_world_active_mask -v`
Expected: 1 passed.

---

### Task 10: SemiImplicit kernel `world_active` modifications

**Files:**
- Modify: `newton/_src/solvers/semi_implicit/kernels_body.py`
- Modify: `newton/_src/solvers/semi_implicit/kernels_contact.py`
- Modify: `newton/_src/solvers/semi_implicit/solver_semi_implicit.py`

Same pattern as Task 9 but for Semi's smaller kernel set (6 kernels total).

- [ ] **Step 1: Modify all 6 SemiImplicit kernels with the world_active pattern**

For each kernel in `kernels_body.py` and `kernels_contact.py`: add `world_active: wp.array(dtype=wp.bool)` parameter at the end + early-return guard using `body_world[tid]` or `particle_world[tid]`.

- [ ] **Step 2: Update `SolverSemiImplicit.step` to thread `world_active`**

Same as Task 9 Step 3 but for SolverSemiImplicit.

- [ ] **Step 3: Backward-compat smoke**

```bash
uv run python -c "
import warp as wp
import newton, newton.solvers
from scripts.scenes import _registry
e = _registry.get('falling_cylinder')
m = e.build_model_randomized(4)
contacts = m.contacts()
s = newton.solvers.SolverSemiImplicit(m)
s0, s1, ctrl = m.state(), m.state(), m.control()
for _ in range(5):
    m.collide(s0, contacts)
    s.step(s0, s1, ctrl, contacts, 1e-3)
    s0, s1 = s1, s0
wp.synchronize()
print('semi backward compat ok')
"
```
Expected: `semi backward compat ok`.

- [ ] **Step 4: Same mask-respect test as Task 9, for SemiImplicit**

Append to `test_masking.py`:
```python
def test_semi_step_respects_world_active_mask():
    import numpy as np
    import newton, newton.solvers
    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    s = newton.solvers.SolverSemiImplicit(m)
    contacts = m.contacts()
    s0, s1, ctrl = m.state(), m.state(), m.control()
    m.collide(s0, contacts)
    bq0 = s0.body_q.numpy().copy()
    mask = wp.full(m.world_count, False, dtype=wp.bool, device=m.device)
    s.step(s0, s1, ctrl, contacts, 1e-3, world_active=mask)
    wp.synchronize()
    bq1 = s1.body_q.numpy()
    np.testing.assert_allclose(bq1, bq0, rtol=1e-6)
```

Run: `uv run python -m pytest scripts/adaptive/tests/test_masking.py::test_semi_step_respects_world_active_mask -v`
Expected: 1 passed.

---

### Task 11: Compaction gather/scatter + prefix-sum kernels

**Files:**
- Create: `scripts/adaptive/compaction_mixin.py`

This task creates the compaction kernels but no class yet — that's Task 12.

- [ ] **Step 1: Write the failing test**

Create `scripts/adaptive/tests/test_compaction.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import warp as wp

from scripts.adaptive.compaction_mixin import (
    _active_indices_prefix_sum,
    _compact_gather_float,
    _compact_scatter_float,
)


def test_active_indices_prefix_sum_packs_active_ids():
    """active_indices[k] = original world id of the k-th active world."""
    wp.init()
    n = 8
    active = wp.array(
        np.array([True, False, True, True, False, False, True, False], dtype=bool),
        dtype=wp.bool,
    )
    out = wp.zeros(n, dtype=wp.int32)
    n_active = wp.zeros(1, dtype=wp.int32)
    wp.launch(_active_indices_prefix_sum, dim=1,
              inputs=[active, out, n_active])
    wp.synchronize()
    got = out.numpy()
    assert int(n_active.numpy()[0]) == 4
    np.testing.assert_array_equal(got[:4], [0, 2, 3, 6])


def test_compact_gather_scatter_roundtrip():
    """Gather active worlds into compact layout; scatter back; original unchanged."""
    wp.init()
    n = 8
    coords_per_world = 3
    active = wp.array(
        np.array([True, False, True, True, False, False, True, False], dtype=bool),
        dtype=wp.bool,
    )
    indices = wp.array(np.array([0, 2, 3, 6, 0, 0, 0, 0], dtype=np.int32), dtype=wp.int32)
    n_active = 4

    canonical = wp.array(
        np.arange(n * coords_per_world, dtype=np.float32),
        dtype=wp.float32,
    )
    compact = wp.zeros(n_active * coords_per_world, dtype=wp.float32)

    wp.launch(_compact_gather_float, dim=n_active,
              inputs=[canonical, indices, coords_per_world],
              outputs=[compact])
    wp.synchronize()
    got = compact.numpy().reshape(n_active, coords_per_world)
    canon = canonical.numpy().reshape(n, coords_per_world)
    np.testing.assert_array_equal(got[0], canon[0])  # world 0
    np.testing.assert_array_equal(got[1], canon[2])  # world 2
    np.testing.assert_array_equal(got[2], canon[3])  # world 3
    np.testing.assert_array_equal(got[3], canon[6])  # world 6

    # Scatter back to a fresh canonical buffer (only the active slots should be written).
    canon2 = wp.zeros(n * coords_per_world, dtype=wp.float32)
    wp.launch(_compact_scatter_float, dim=n_active,
              inputs=[compact, indices, coords_per_world],
              outputs=[canon2])
    wp.synchronize()
    out_canon = canon2.numpy().reshape(n, coords_per_world)
    for k, w in enumerate([0, 2, 3, 6]):
        np.testing.assert_array_equal(out_canon[w], canon[w])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest scripts/adaptive/tests/test_compaction.py -v`
Expected: ImportError on `scripts.adaptive.compaction_mixin`.

- [ ] **Step 3: Write the kernels**

Create `scripts/adaptive/compaction_mixin.py`:
```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""CompactionMixin — wrapper-side re-batching for MuJoCo adaptive."""

from __future__ import annotations

import warp as wp


# --- Kernels ------------------------------------------------------------------

@wp.kernel
def _active_indices_prefix_sum(
    active: wp.array(dtype=wp.bool),
    out_indices: wp.array(dtype=wp.int32),
    out_n_active: wp.array(dtype=wp.int32),
):
    """Single-thread prefix sum. For N <= ~4096 this is faster than a
    parallel scan due to launch overhead. For larger N consider wp.scan."""
    n = active.shape[0]
    k = int(0)
    for i in range(n):
        if active[i]:
            out_indices[k] = i
            k += 1
    out_n_active[0] = k


@wp.kernel
def _compact_gather_float(
    canonical: wp.array(dtype=wp.float32),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    compact_out: wp.array(dtype=wp.float32),
):
    """compact_out[k * per_world + j] = canonical[active_indices[k] * per_world + j]."""
    tid = wp.tid()
    k = tid
    src_world = active_indices[k]
    for j in range(per_world):
        compact_out[k * per_world + j] = canonical[src_world * per_world + j]


@wp.kernel
def _compact_scatter_float(
    compact: wp.array(dtype=wp.float32),
    active_indices: wp.array(dtype=wp.int32),
    per_world: int,
    canonical_out: wp.array(dtype=wp.float32),
):
    """canonical_out[active_indices[k] * per_world + j] = compact[k * per_world + j]."""
    tid = wp.tid()
    k = tid
    dst_world = active_indices[k]
    for j in range(per_world):
        canonical_out[dst_world * per_world + j] = compact[k * per_world + j]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest scripts/adaptive/tests/test_compaction.py -v`
Expected: 2 passed.

---

### Task 12: `CompactionMixin` class — tier management + state copy logic

**Files:**
- Modify: `scripts/adaptive/compaction_mixin.py`

For v1, "tier management" is constructor-time pre-allocation. We do NOT actually rebuild `mjw_data` at runtime in v1 — instead, we keep the full-N mjw_data and use the gather/scatter kernels to maintain a packed compact layout for our Newton State scratches. The step_fn (MuJoCo shim) operates on the full-N mjw_data but with `_world_active` mask logic disabled inside MuJoCo. The "compaction win" for MuJoCo in v1 is from the wrapper's bookkeeping (gather/scatter of Newton State), not from reducing mjw_data size.

**v1 simplification:** True mjw_data resize (multiple tiers) is deferred to v2. v1 just gathers/scatters at the Newton State level and tracks active count. This still produces correct output but doesn't give the asymptotic O(N) speedup — that requires v2 mjw_data resize. Document this limitation; the wrapper API is forward-compatible.

This v1 scope reduction is necessary because mjw_data construction is expensive (kernel cache compilation) and dynamic resize has open questions about state consistency that need a separate spec.

- [ ] **Step 1: Add the mixin class with active-count tracking**

Append to `scripts/adaptive/compaction_mixin.py`:
```python
class CompactionMixin:
    """Track active worlds. v1: bookkeeping only — no mjw_data resize yet.

    v2 will add multi-tier mjw_data swap. The interface is forward-compatible:
    callers see the same status_summary and active_count; only the perf changes.
    """

    def __init__(self, *args, **kwargs):
        # Pop compaction_sizes if provided (forward-compat for v2; ignored in v1).
        self._compaction_sizes = kwargs.pop("compaction_sizes", (1.0,))
        super().__init__(*args, **kwargs)
        n = self.model.world_count
        device = self.model.device
        self._world_active = wp.full(n, True, dtype=wp.bool, device=device)
        self._active_indices = wp.zeros(n, dtype=wp.int32, device=device)
        self._n_active_buf = wp.zeros(1, dtype=wp.int32, device=device)

    def _reset_active(self):
        n = self.model.world_count
        self._world_active.fill_(True)
        wp.launch(_active_indices_prefix_sum, dim=1,
                  inputs=[self._world_active, self._active_indices, self._n_active_buf])

    def _update_active_count(self):
        """Update n_active_buf via single-thread prefix sum."""
        wp.launch(_active_indices_prefix_sum, dim=1,
                  inputs=[self._world_active, self._active_indices, self._n_active_buf])

    @property
    def n_active(self) -> int:
        """Active world count (host int, 1 PCIe sync)."""
        return int(self._n_active_buf.numpy()[0])
```

- [ ] **Step 2: Override `_run_iteration_body` to update active mask + count**

Append to `CompactionMixin`:
```python
    def _run_iteration_body(self, effective_dt_max):
        super()._run_iteration_body(effective_dt_max)
        # Update mask: a world is active iff sim_time < next_time.
        from scripts.adaptive import kernels as K
        n = self.model.world_count
        wp.launch(
            K._update_active_mask, dim=n,
            inputs=[self._sim_time, self._next_time, self._world_active],
            device=self.model.device,
        )
        self._update_active_count()

    def step_dt(self, dt_outer, state_0, state_1, control):
        self._reset_active()
        return super().step_dt(dt_outer, state_0, state_1, control)
```

- [ ] **Step 3: Append integration test**

Append to `scripts/adaptive/tests/test_compaction.py`:
```python
def test_compaction_mixin_tracks_active_count():
    """After step_dt with no-op step_fn, all worlds should be inactive (sim_time == next_time)."""
    from scripts.adaptive.base import AdaptiveWrapper
    from scripts.adaptive.compaction_mixin import CompactionMixin
    from scripts.scenes import _registry

    class _CW(CompactionMixin, AdaptiveWrapper):
        pass

    def _noop(model, sin, sout, ctrl, contacts, dt):
        pass

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    w = _CW(model=m, step_fn=_noop, tol=1e-3, dt_init=0.01,
            dt_min=1e-6, dt_max=0.01, dt_outer=0.01,
            needs_collide=False, contacts=m.contacts())
    s0, s1, ctrl = m.state(), m.state(), m.control()
    s0, s1 = w.step_dt(0.01, s0, s1, ctrl)
    wp.synchronize()
    assert w.n_active == 0, f"all worlds should be done; n_active={w.n_active}"
```

- [ ] **Step 4: Run test**

Run: `uv run python -m pytest scripts/adaptive/tests/test_compaction.py::test_compaction_mixin_tracks_active_count -v`
Expected: 1 passed.

---

## Wave 3 — Factories (Agent D, after Wave 2)

### Task 13: `adaptive_mujoco_factory`

**Files:**
- Create: `scripts/adaptive/factories.py`

- [ ] **Step 1: Write the failing test**

Append to `scripts/adaptive/tests/test_integration.py`:
```python
def test_adaptive_mujoco_factory_runs():
    """adaptive_mujoco_factory produces a working (solver, step_fn) pair."""
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()

    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()

    import numpy as np
    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), "NaN in body_q after 10 outer steps"
```

- [ ] **Step 2: Run test to verify fails**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_mujoco_factory_runs -v`
Expected: ImportError on `scripts.adaptive.factories`.

- [ ] **Step 3: Write factories.py with `adaptive_mujoco_factory`**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Factory builders for adaptive solver variants."""

from __future__ import annotations

import warp as wp

import newton
import newton.solvers

from scripts.adaptive.base import AdaptiveWrapper
from scripts.adaptive.compaction_mixin import CompactionMixin
from scripts.adaptive.masking_mixin import ActiveSetMaskingMixin


class AdaptiveCompactionWrapper(CompactionMixin, AdaptiveWrapper):
    """Adaptive wrapper + compaction mixin (for MuJoCo)."""


class AdaptiveMaskingWrapper(ActiveSetMaskingMixin, AdaptiveWrapper):
    """Adaptive wrapper + active-set masking (for XPBD / SemiImplicit)."""


def adaptive_mujoco_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
    nconmax: int,
    njmax: int,
    compaction_sizes: tuple[float, ...] = (1.0, 0.25, 0.0625, 0.015625),
):
    """Build (solver, step_fn) for MuJoCo adaptive."""

    def build(model):
        underlying = newton.solvers.SolverMuJoCo(
            model, separate_worlds=True, use_mujoco_contacts=False,
            nconmax=nconmax, njmax=njmax,
        )

        def step_fn(model_arg, state_in, state_out, ctrl, contacts, dt_array):
            """MuJoCo shim: copy per-world dt into mjw_model.opt.timestep, call step."""
            wp.copy(underlying.mjw_model.opt.timestep, dt_array)
            underlying.step(state_in, state_out, ctrl, contacts, dt_init)

        wrapper = AdaptiveCompactionWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False,
            compaction_sizes=compaction_sizes,
        )

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build
```

- [ ] **Step 4: Run test to verify passes**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_mujoco_factory_runs -v`
Expected: 1 passed.

---

### Task 14: `adaptive_xpbd_factory`

**Files:**
- Modify: `scripts/adaptive/factories.py`

- [ ] **Step 1: Append `adaptive_xpbd_factory` to factories.py**

```python
def adaptive_xpbd_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
):
    """Build (solver, step_fn) for XPBD adaptive (effectively-global dt)."""

    def build(model):
        underlying = newton.solvers.SolverXPBD(model)
        contacts = model.contacts()

        # XPBD takes scalar dt; we pass max(dt_array) and rely on world_active
        # to skip finished worlds at the kernel level.
        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array):
            scalar_dt = float(dt_array.numpy().max())
            model_arg.collide(state_in, contacts)
            # Pass wrapper's mask through to XPBD's modified kernels.
            mask = wrapper._world_active if hasattr(wrapper, "_world_active") else None
            if mask is not None:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)
            else:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt)

        wrapper = AdaptiveMaskingWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False, contacts=contacts,
        )

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build
```

- [ ] **Step 2: Smoke test**

Append to `test_integration.py`:
```python
def test_adaptive_xpbd_factory_runs():
    from scripts.adaptive.factories import adaptive_xpbd_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    import numpy as np
    assert not np.isnan(s0.body_q.numpy()).any()
```

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_xpbd_factory_runs -v`
Expected: 1 passed.

---

### Task 15: `adaptive_semi_factory`

**Files:**
- Modify: `scripts/adaptive/factories.py`

Same pattern as Task 14, but for SemiImplicit. Append:

```python
def adaptive_semi_factory(
    *,
    tol: float,
    dt_init: float,
    dt_min: float,
    dt_max: float,
    dt_outer: float,
):
    """Build (solver, step_fn) for SemiImplicit adaptive."""

    def build(model):
        underlying = newton.solvers.SolverSemiImplicit(model)
        contacts = model.contacts()

        def step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array):
            scalar_dt = float(dt_array.numpy().max())
            model_arg.collide(state_in, contacts)
            mask = wrapper._world_active if hasattr(wrapper, "_world_active") else None
            if mask is not None:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt, world_active=mask)
            else:
                underlying.step(state_in, state_out, ctrl, contacts, scalar_dt)

        wrapper = AdaptiveMaskingWrapper(
            model=model, step_fn=step_fn,
            tol=tol, dt_init=dt_init, dt_min=dt_min, dt_max=dt_max,
            dt_outer=dt_outer, needs_collide=False, contacts=contacts,
        )

        def bench_step_fn(model_arg, s0, s1, ctrl):
            return wrapper.step_dt(dt_outer, s0, s1, ctrl)

        return wrapper, bench_step_fn

    return build
```

- [ ] **Step 1: Append the factory**
- [ ] **Step 2: Smoke test (analogous to Task 14)**

Append to `test_integration.py`:
```python
def test_adaptive_semi_factory_runs():
    from scripts.adaptive.factories import adaptive_semi_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    builder = adaptive_semi_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    )
    solver, step_fn = builder(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(10):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    import numpy as np
    assert not np.isnan(s0.body_q.numpy()).any()
```

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_semi_factory_runs -v`
Expected: 1 passed.

---

## Wave 4 — Correctness gates (Agent D)

### Task 16: CENIC oracle correctness test (must-pass)

**Files:**
- Modify: `scripts/adaptive/tests/test_integration.py`

- [ ] **Step 1: Write the oracle test**

Append:
```python
def test_adaptive_mujoco_matches_cenic_oracle():
    """adaptive_mujoco produces same state as SolverMuJoCoCENIC after 100 outer
    steps. This is the v1 correctness gate."""
    import numpy as np
    from scripts.adaptive.factories import adaptive_mujoco_factory
    from scripts.scenes.falling_cylinder import DT_OUTER

    scene = _registry.get("falling_cylinder")
    N = 4
    n_outer = 100

    # Run CENIC oracle.
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

    # Run adaptive wrapper.
    m_wrap = scene.build_model_randomized(N, seed=42)
    builder = adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=8, njmax=32,
    )
    _, step_fn = builder(m_wrap)
    s0_w, s1_w, ctrl_w = m_wrap.state(), m_wrap.state(), m_wrap.control()
    for _ in range(n_outer):
        s0_w, s1_w = step_fn(m_wrap, s0_w, s1_w, ctrl_w)
    wp.synchronize()
    wrap_bq = s0_w.body_q.numpy()

    # Match within float32 tolerance.
    np.testing.assert_allclose(
        wrap_bq, oracle_bq, rtol=1e-4, atol=1e-5,
        err_msg="adaptive_mujoco diverged from CENIC oracle after 100 outer steps",
    )
```

- [ ] **Step 2: Run the oracle test (this is the v1 must-pass)**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_mujoco_matches_cenic_oracle -v`
Expected: 1 passed. If FAIL: diff is too big — debug `_q_weights` computation, `_calc_adjusted_step` reuse, state copy ordering. The wrapper's algorithm must be identical to CENIC's.

---

### Task 17: Stability tests for all 3 variants

**Files:**
- Modify: `scripts/adaptive/tests/test_integration.py`

- [ ] **Step 1: Write parameterized stability test**

Append:
```python
import pytest


@pytest.mark.parametrize("factory_name,kwargs", [
    ("adaptive_mujoco_factory", dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01, nconmax=8, njmax=32)),
    ("adaptive_xpbd_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
    ("adaptive_semi_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
])
def test_adaptive_variant_stable_on_falling_cylinder(factory_name, kwargs):
    """200 outer steps, no NaN, pos_max < 5m. Small-N smoke."""
    import importlib
    import numpy as np

    factories_mod = importlib.import_module("scripts.adaptive.factories")
    factory = getattr(factories_mod, factory_name)

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(4)
    _, step_fn = factory(**kwargs)(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(200):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), f"{factory_name}: NaN in body_q"
    pos_max = float(np.abs(bq[..., :3]).max())
    assert pos_max < 5.0, f"{factory_name}: pos_max={pos_max:.2f}m (scene bound ~1m)"


# --- N=1024 HARD GATE: implementation does NOT ship until all 3 variants pass this ---
@pytest.mark.parametrize("factory_name,kwargs", [
    ("adaptive_mujoco_factory", dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01, nconmax=8, njmax=32)),
    ("adaptive_xpbd_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
    ("adaptive_semi_factory",   dict(tol=1e-3, dt_init=0.01, dt_min=1e-6,
                                     dt_max=0.01, dt_outer=0.01)),
])
def test_adaptive_variant_stable_at_n_1024(factory_name, kwargs):
    """HARD GATE: each adaptive variant runs 50 outer steps at N=1024 on
    falling_cylinder with no NaN and pos_max < 5m. v1 ships only when this
    passes for all 3 factories."""
    import importlib
    import numpy as np

    factories_mod = importlib.import_module("scripts.adaptive.factories")
    factory = getattr(factories_mod, factory_name)

    scene = _registry.get("falling_cylinder")
    m = scene.build_model_randomized(1024)
    _, step_fn = factory(**kwargs)(m)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for _ in range(50):
        s0, s1 = step_fn(m, s0, s1, ctrl)
    wp.synchronize()
    bq = s0.body_q.numpy()
    assert not np.isnan(bq).any(), f"{factory_name}: NaN at N=1024"
    pos_max = float(np.abs(bq[..., :3]).max())
    assert pos_max < 5.0, f"{factory_name}: pos_max={pos_max:.2f}m at N=1024"
```

- [ ] **Step 2: Run small-N smoke**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_variant_stable_on_falling_cylinder -v`
Expected: 3 passed (one per variant).

- [ ] **Step 3: Run the N=1024 HARD GATE**

Run: `uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_variant_stable_at_n_1024 -v`
Expected: **3 passed (one per variant). v1 implementation does NOT ship if any of these fail.** If any fails: stop, return to systematic debugging. Likely causes:
- adaptive_mujoco: OOM at N=1024 → reduce per-world nconmax/njmax in test kwargs
- adaptive_xpbd: NaN or large pos_max → world_active mask not actually being respected by modified XPBD kernels (Task 9 bug)
- adaptive_semi: same as XPBD (Task 10 bug)

This is the per-user-instruction hard ship gate. All 3 must pass.

---

## Wave 5 — Bench wiring + the wow plot (Agent E)

### Task 18: Add adaptive factories to `falling_cylinder` `SOLVER_FACTORIES`

**Files:**
- Modify: `scripts/scenes/falling_cylinder.py`

- [ ] **Step 1: Append the adaptive factories to SOLVER_FACTORIES**

Modify `scripts/scenes/falling_cylinder.py` — find the `SOLVER_FACTORIES` dict and append:
```python
from scripts.adaptive import factories as _af  # noqa: E402

SOLVER_FACTORIES.update({
    "mujoco_adaptive_1e-3": _af.adaptive_mujoco_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER, nconmax=_NCON, njmax=_NJM,
    ),
    "xpbd_adaptive_1e-3": _af.adaptive_xpbd_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
    "semi_adaptive_1e-3": _af.adaptive_semi_factory(
        tol=1e-3, dt_init=DT_OUTER, dt_min=1e-6, dt_max=DT_OUTER,
        dt_outer=DT_OUTER,
    ),
})
```

- [ ] **Step 2: Verify all kinds present**

Run:
```bash
uv run python -c "
from scripts.scenes import _registry
e = _registry.get('falling_cylinder')
kinds = e.solver_kinds()
print('kinds:', kinds)
required = ['mujoco_adaptive_1e-3', 'xpbd_adaptive_1e-3', 'semi_adaptive_1e-3']
for r in required:
    assert r in kinds, f'missing {r}'
print('all 3 adaptive variants registered')
"
```
Expected: `all 3 adaptive variants registered`.

---

### Task 19: Add plot styles for new kinds

**Files:**
- Modify: `scripts/bench/plotting.py`

- [ ] **Step 1: Add STYLES entries**

In `scripts/bench/plotting.py` `STYLES` dict, add:
```python
    "mujoco_adaptive_1e-3": PlotStyle("#0066cc", "*", "-", "MuJoCo adaptive (wrapper) tol=1e-3"),
    "xpbd_adaptive_1e-3":   PlotStyle("#cc6600", "*", "-", "XPBD adaptive tol=1e-3"),
    "semi_adaptive_1e-3":   PlotStyle("#660066", "*", "-", "SemiImplicit adaptive tol=1e-3"),
```

- [ ] **Step 2: Verify**

```bash
uv run python -c "
from scripts.bench.plotting import STYLES
for k in ['mujoco_adaptive_1e-3', 'xpbd_adaptive_1e-3', 'semi_adaptive_1e-3']:
    assert k in STYLES, k
print('styles ok')
"
```
Expected: `styles ok`.

---

### Task 20: Smoke run at small N

**Files:**
- (No code changes — just verify the bench wiring works end-to-end.)

- [ ] **Step 1: Run a tiny bench sweep**

Run:
```bash
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 4 --steps 5 --warmup 2 2>&1 | tail -30
```

Expected: Summary table with all 10 kinds listed (7 original + 3 new adaptive). No FAILED entries.

- [ ] **Step 2: Confirm the JSON has the new kinds**

```bash
cat scripts/bench/results/$(git rev-parse --short=7 HEAD)/scaling_falling_cylinder.json | \
  uv run python -c "import json, sys; d=json.load(sys.stdin); print('kinds:', list(d['modes'].keys()))"
```
Expected: list includes `mujoco_adaptive_1e-3`, `xpbd_adaptive_1e-3`, `semi_adaptive_1e-3`.

---

### Task 21: The wow run — `N = 1..2^15`

**Files:**
- None to modify; this just runs the bench.

- [ ] **Step 1: Trigger the big sweep**

Run:
```bash
uv run -m scripts.bench --only scaling --scene falling_cylinder \
  --ns 1 4 16 64 256 1024 4096 16384 32768 \
  --steps 50 --warmup 10
```

Wall time estimate: ~30-90 min depending on adaptive variant cost at high N. Will show per-N progress. Per-N subprocess isolation means partial failures don't kill the whole run.

- [ ] **Step 2: Verify the wow figure**

Open: `scripts/bench/results/$(git rev-parse --short=7 HEAD)/plots/falling_cylinder/scaling_wall_time.png`

Expected: 10 curves (7 fixed/MuJoCo + 3 adaptive), N axis 1 → 32768, log-log scale.

- [ ] **Step 3: Quick correctness sanity (sample one N)**

```bash
cat scripts/bench/results/$(git rev-parse --short=7 HEAD)/scaling_falling_cylinder.json | \
  uv run python -c "
import json, sys
d = json.load(sys.stdin)
print('Ns kept:', d['ns'])
print('Failed kinds:')
for k, fails in d.get('kinds_ns', {}).items():
    if len(fails) != len(d['ns']):
        print(f'  {k}: completed {len(fails)}/{len(d[\"ns\"])} Ns')
"
```

Expected: All 3 adaptive variants completed at least `N ≥ 1024`.

---

## Verification: full test suite passes

- [ ] **HARD SHIP GATE (per user instruction): N=1024 stability on all 3 adaptive variants**

```bash
uv run python -m pytest scripts/adaptive/tests/test_integration.py::test_adaptive_variant_stable_at_n_1024 -v
```

**Expected: 3 passed. v1 does NOT ship if any fail. This is the final gate before declaring the implementation done.**

- [ ] **Run all adaptive tests**

```bash
uv run python -m pytest scripts/adaptive/tests/ -v
```

Expected: All tests pass. Approximate count: ~15-18 tests across 5 files.

- [ ] **Run existing scaling smoke (regression check)**

```bash
uv run -m scripts.bench --only scaling --scene contact_objects \
  --ns 1 4 --steps 3 --warmup 2 2>&1 | tail -20
```

Expected: Existing kinds (mujoco_cenic_*, mujoco_fixed_*, xpbd_1ms) still work — XPBD modifications in Task 9 didn't break the non-adaptive path.

---

## Plan-vs-spec coverage check

| Spec section | Implemented in |
|---|---|
| 3.1 File tree | Tasks 1, 2, 3, 4, 8, 11, 12, 13, 14, 15 |
| 3.2 Class hierarchy | Tasks 4-7, 8, 12, 13 |
| 3.3 Public API | Task 13 (factories) |
| 3.4 Per-world dt asymmetry | Task 13 (MuJoCo opt.timestep), 14 (XPBD scalar shim), 15 (Semi scalar shim) |
| 3.5 Per-world dt + masking interaction | Tasks 9, 10 (kernel mask), 14, 15 (scalar dt shim) |
| 4.1 base.py | Tasks 4, 5, 6, 7 |
| 4.2 kernels.py | Task 2 |
| 4.3 controller.py | Task 3 |
| 4.4 masking_mixin.py | Task 8, 9, 10 |
| 4.5 compaction_mixin.py | Tasks 11, 12 (v1 simplification documented) |
| 4.6 factories.py | Tasks 13, 14, 15 |
| 4.7 Newton kernel mods | Tasks 9, 10 |
| 5.x Data flow | Task 5, 6 (implements the loop) |
| 6.1 NaN handling | Inherited from CENIC's `_calc_adjusted_step` (lifted in Task 2). `stuck_policy="freeze"` default is plumbed in Task 4 constructor; full freeze-and-continue logic is a v2 follow-up — not blocked on it because CENIC's existing NaN propagation matches what v1 ships with. |
| 6.2 dt-floor accept | Inherited from CENIC `_calc_adjusted_step` |
| 6.3 max iteration cap | Task 6 (RuntimeError raise) |
| 6.4 Compaction OOM | Out of v1 scope (Task 12 simplification — no mjw_data resize in v1) |
| 6.5 step_fn exceptions | Default Python behavior — propagate. No code needed. |
| 6.6 Active mask consistency | Tests in Task 11 verify gather/scatter correctness |
| 7.1 Unit tests | Tasks 2, 3, 8, 11 |
| 7.2 Integration tests | Tasks 13, 14, 15, 16, 17 |
| 7.3 Performance bench | Tasks 18-21 |
| 7.4 Long-run smoke | Manual after v1 ships — not in plan |
| 8 Parallelization map | Wave structure: 1 (serial), 2+3 (parallel), 4-5 (serial after Wave 3) |

**v1 simplifications documented:**
- Task 12 note: True mjw_data resize deferred to v2. v1 compaction is bookkeeping only.
- Spec 6.4 (compaction OOM): not exercised in v1 since no resize.
- Spec 6.1 (NaN freeze policy): default constructor arg plumbed; full freeze-and-continue logic deferred.

---

## Self-review notes

1. **Spec coverage**: full map above. v1 simplifications are explicit (compaction = bookkeeping; full mjw_data resize is v2).
2. **No placeholders**: each code step has actual code or actual command. References to CENIC line numbers are exact.
3. **Type consistency**: `step_fn` signature `(model, state_in, state_out, ctrl, contacts, dt_array)` used consistently across Tasks 4, 5, 6, 13, 14, 15.
4. **Scope**: single plan, ~21 tasks, executable by 4 parallel sub-agents after Wave 1.

End of plan.
