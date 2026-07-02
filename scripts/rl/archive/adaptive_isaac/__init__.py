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

"""Isaac Sim / Isaac Lab integration glue for the adaptive Newton solver.

Sub-modules:

* :mod:`adaptive_isaac.selection` -- config-driven selector that bridges an
  Isaac-Lab-side "adaptive" choice into the env-var contract the wheel overlay's
  ``NewtonStage._get_solver`` reads (the only path that reaches the wheel which
  actually runs Trossen). Import-safe at config-build time (no Isaac Sim import).
* :mod:`adaptive_isaac.dropdown` -- GUI dropdown registration (later step).

The overlay file (``overlay/newton_stage.py``) and ``apply_overlay.sh`` install
the modified wheel extension; see ``README.md``.
"""

from . import selection

__all__ = ["selection"]
