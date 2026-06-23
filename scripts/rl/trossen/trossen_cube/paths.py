# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for the Trossen testbed's filesystem roots.

Native layout: everything lives under ``~/Documents/code/isaac-rl`` (beside the
``newton-cenic`` repo and the Isaac Sim install), overridable per-root by env var.
Defaults resolve with zero env exports.

Pure stdlib -- no ``isaaclab``/``pxr`` import. Importing it as ``trossen_cube.paths``
does still run the package ``__init__`` (which imports ``gymnasium`` and registers gym
ids), so host-only tools that lack ``gymnasium`` (e.g. :mod:`make_norails_usd`, run under a
``usd-core``-only interpreter) must replicate the two env lookups inline rather than import
this module. See the native bring-up in ``README.md``.

Env overrides:
    TROSSEN_DATA_ROOT       parent of all three roots (default ``~/Documents/code/isaac-rl``)
    TROSSEN_ASSET_ROOT      the ``trossen_ai_isaac`` clone (USD assets)
    TROSSEN_LOG_ROOT        training log/checkpoint root
    TROSSEN_ARTIFACT_ROOT   renders/plots/diagnostic dumps
    STATIONARY_AI_USD       absolute path to the rig USD (else derived from the asset root)
    STATIONARY_AI_NORAILS_USD  absolute path to the no-rails override USD
"""

from __future__ import annotations

import os

_DATA_ROOT = os.path.expanduser(os.environ.get("TROSSEN_DATA_ROOT", "~/Documents/code/isaac-rl"))

ASSET_ROOT = os.path.expanduser(os.environ.get("TROSSEN_ASSET_ROOT", os.path.join(_DATA_ROOT, "trossen_ai_isaac")))
LOG_ROOT = os.path.expanduser(os.environ.get("TROSSEN_LOG_ROOT", os.path.join(_DATA_ROOT, "logs", "trossen")))
ARTIFACT_ROOT = os.path.expanduser(os.environ.get("TROSSEN_ARTIFACT_ROOT", os.path.join(_DATA_ROOT, "artifacts")))

_ROBOT_DIR = os.path.join(ASSET_ROOT, "assets", "robots", "stationary_ai")
STATIONARY_AI_USD = os.environ.get("STATIONARY_AI_USD", os.path.join(_ROBOT_DIR, "stationary_ai.usd"))
STATIONARY_AI_NORAILS_USD = os.environ.get(
    "STATIONARY_AI_NORAILS_USD", os.path.join(_ROBOT_DIR, "stationary_ai_norails.usda")
)
