# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GUI dropdown registration for the adaptive Newton solver mode.

Adds a "Newton -- Adaptive" row to the Isaac Sim viewport "Simulation" physics
backend dropdown by registering a second simulation backend with
``omni.physics.core`` whose Newton stage builds :class:`SolverMuJoCoAdaptive`.

See :func:`register_adaptive_dropdown` and ``register_adaptive_dropdown.py`` for
the registration entrypoint, and ``README.md`` for how it is loaded (Kit ext vs
post-launch run hook) and the documented Trossen asset-variant limitation.
"""

from .register_adaptive_dropdown import (
    ADAPTIVE_BACKEND_NAME,
    ADAPTIVE_SELECT_SETTING,
    AdaptiveDropdownRegistry,
    register_adaptive_dropdown,
    unregister_adaptive_dropdown,
)

__all__ = [
    "ADAPTIVE_BACKEND_NAME",
    "ADAPTIVE_SELECT_SETTING",
    "AdaptiveDropdownRegistry",
    "register_adaptive_dropdown",
    "unregister_adaptive_dropdown",
]
