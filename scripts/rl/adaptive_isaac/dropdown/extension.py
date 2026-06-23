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

"""Kit extension wrapper for the "Newton -- Adaptive" dropdown row.

Load order matters: this extension must start **after** ``isaacsim.physics.newton``
(the stock Newton backend) so the impl modules exist and the overlay is in place.
Declare that dependency in ``extension.toml`` (see ``README.md``).

When loaded as a Kit ext, ``on_startup`` registers the row; ``on_shutdown``
removes it. For non-Kit / standalone launches use the run hook in
``run_hook.py`` instead.
"""

from __future__ import annotations

import omni.ext

from .register_adaptive_dropdown import register_adaptive_dropdown, unregister_adaptive_dropdown


class AdaptiveDropdownExtension(omni.ext.IExt):
    """Registers the adaptive Newton backend dropdown row on extension startup."""

    def on_startup(self, ext_id: str) -> None:
        """Register the "Newton -- Adaptive" dropdown row.

        Args:
            ext_id: Extension identifier provided by the Kit extension manager.
        """
        register_adaptive_dropdown()

    def on_shutdown(self) -> None:
        """Remove the "Newton -- Adaptive" dropdown row."""
        unregister_adaptive_dropdown()
