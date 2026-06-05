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

    # Scatter back to a fresh canonical buffer.
    canon2 = wp.zeros(n * coords_per_world, dtype=wp.float32)
    wp.launch(_compact_scatter_float, dim=n_active,
              inputs=[compact, indices, coords_per_world],
              outputs=[canon2])
    wp.synchronize()
    out_canon = canon2.numpy().reshape(n, coords_per_world)
    for k, w in enumerate([0, 2, 3, 6]):
        np.testing.assert_array_equal(out_canon[w], canon[w])


def test_compaction_mixin_tracks_active_count():
    """After step_dt with no-op step_fn, all worlds should be inactive (sim_time == next_time)."""
    from scripts.adaptive.base import AdaptiveWrapper
    from scripts.adaptive.compaction_mixin import CompactionMixin
    from scripts.scenes import _registry

    class _CW(CompactionMixin, AdaptiveWrapper):
        pass

    def _noop(model, sin, sout, ctrl, contacts, dt_array, dt_scalar_buf):
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
