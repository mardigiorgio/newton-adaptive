# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drake PI controller tuning constants — wired into kernel via per-launch args (v2)."""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Drake PI dt controller config. Matches CENIC defaults from v1.

    All 5 fields are passed as per-launch float scalars to
    scripts/adaptive/kernels.py::_calc_adjusted_step. Override any of them
    via constructor kwargs to AdaptiveWrapper.
    """
    safety_factor: float = 0.9
    growth_cap: float = 5.0
    shrink_cap: float = 0.1
    hysteresis_high: float = 1.2
    hysteresis_low: float = 0.9
    kp: float = 0.0   # reserved for v2.5 PI variant; unused today
