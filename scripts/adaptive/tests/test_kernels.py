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
    weights = wp.from_numpy(
        np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0]], dtype=np.float32),
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
