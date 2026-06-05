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


# --- Mixin --------------------------------------------------------------------

class CompactionMixin:
    """Track active worlds. v1: bookkeeping only -- no mjw_data resize yet.

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
