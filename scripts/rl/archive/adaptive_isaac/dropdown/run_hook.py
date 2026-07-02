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

"""Standalone run hook to add the "Newton -- Adaptive" dropdown row.

Use this when launching Isaac Sim / Isaac Lab from a Python entrypoint (e.g. a
training script) rather than as a Kit extension. Call :func:`install` **after**
``SimulationApp`` / ``AppLauncher`` has started Kit and after the stock
``isaacsim.physics.newton`` extension is loaded -- only then do the impl modules
and ``omni.physics.core`` exist.

Example (after AppLauncher):

    from scripts.rl.adaptive_isaac.dropdown.run_hook import install
    install()  # adds the "Newton -- Adaptive" row to the viewport dropdown

This is GUI-only sugar; for headless training the env-var / config solver-mode
selector (see overlay ``_get_solver``) is the selection mechanism.
"""

from __future__ import annotations

from .register_adaptive_dropdown import (
    AdaptiveDropdownRegistry,
    register_adaptive_dropdown,
    unregister_adaptive_dropdown,
)


def install() -> AdaptiveDropdownRegistry | None:
    """Register the adaptive dropdown row (idempotent).

    Returns:
        The active registry, or ``None`` if disabled or registration failed.
    """
    return register_adaptive_dropdown()


def uninstall() -> None:
    """Remove the adaptive dropdown row if present."""
    unregister_adaptive_dropdown()


if __name__ == "__main__":
    # Convenience: only meaningful inside a running Kit process. Outside Kit the
    # omni.* imports fail and register() logs an error rather than crashing.
    install()
