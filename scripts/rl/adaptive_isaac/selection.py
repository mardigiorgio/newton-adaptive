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

"""Config-driven selector for the adaptive Newton solver mode (Isaac Lab side).

WHY THIS MODULE EXISTS
----------------------
"Newton -- Adaptive" is a *solver mode* within the single Newton engine, not a
fourth physics engine. Selecting it would ideally be a field on Isaac Lab's
``NewtonCfg`` / ``MJWarpSolverCfg``. It is NOT, because of an architectural
split between two independent Newton code paths:

* **PATH A** -- Isaac Lab's *native* backend ``isaaclab_newton``. ``NewtonCfg``
  resolves here (``class_type = {DIR}.mjwarp_manager:NewtonMJWarpManager``;
  ``mjwarp_manager_cfg.py:31``). It builds ``SolverMuJoCo`` directly
  (``mjwarp_manager.py:_build_solver``) and steps it via a **5-positional**
  ``solver.step(state_0, state_1, control, contacts, substep_dt)``
  (``newton_manager.py:1351``). The adaptive solver does NOT implement that
  signature -- it advances a whole frame via ``step_dt(dt, s0, s1, control)``
  -- so it cannot run on PATH A unmodified.

* **PATH B** -- the legacy Isaac Sim wheel extension
  ``isaacsim.physics.newton`` (``NewtonStage``). This is the path that actually
  runs under the Isaac Lab launcher (verified: the adaptive overlay's load
  marker + ``step_dt`` routing took effect for Cartpole). PATH B builds a fresh
  ``NewtonConfig()`` with hardcoded defaults (``extension.py:118-120``) and
  reads **nothing** from Isaac Lab's ``NewtonCfg`` / ``MJWarpSolverCfg``. There
  is no config bridge from Isaac Lab into the wheel.

Because the wheel ignores Isaac Lab's cfg objects, the only bridge from an
Isaac-Lab-side *choice* down into the wheel's ``NewtonStage._get_solver`` (where
``SolverMuJoCoAdaptive`` is constructed) is a set of **process environment
variables** that the overlay's ``_get_solver`` reads. This module is the single
source of truth for that env-var contract and the helpers that set it.

USAGE
-----
Call exactly one of the selector functions BEFORE the simulation app starts
stepping (i.e. before ``SimulationContext`` builds the solver -- in practice,
before the env's first ``reset()``). The cleanest place is right after parsing
the env cfg in a training/eval entrypoint, or inside an env cfg
``__post_init__`` (selection only mutates ``os.environ``; it does not import
Isaac Sim, so it is safe at config-build time).

    from adaptive_isaac import selection

    # config-driven: read whatever the task/launch chose and bridge it
    selection.resolve(env_cfg)            # honours an `adaptive` flag if present
    # or be explicit:
    selection.select_adaptive(tol=1e-3, dt_mode="per_world")
    selection.select_stock_mjwarp()       # force stock MuJoCo-Warp (clears the flag)

The env-var fallback (set the vars yourself before launch) keeps working and is
the documented manual override::

    NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_DTMODE=per_world \
        ./run_native.sh train_teacher.py --headless --num_envs 64 --max_iterations 1

CONTRACT (env vars read by the overlay's ``_get_solver``)
---------------------------------------------------------
* ``NEWTON_ADAPTIVE``          -- ``"1"`` selects ``SolverMuJoCoAdaptive``; anything
                                  else (or unset) keeps stock ``SolverMuJoCo``.
* ``NEWTON_ADAPTIVE_TOL``      -- float, ``tol`` (default ``1e-3``).
* ``NEWTON_ADAPTIVE_DTMODE``   -- ``"per_world"`` (default) or ``"global"``.
* ``NEWTON_ADAPTIVE_DT_INIT``  -- float, ``dt_inner_init`` (default ``0.01``).
* ``NEWTON_ADAPTIVE_DT_MIN``   -- float, ``dt_inner_min`` (default ``1e-6``).

``dt_mode`` defaults to ``"per_world"`` so each env adapts its dt independently
-- per-env dt divergence is exactly the signal that proves the solver is
adapting (the LEAD's "dt actually varies" check). ``"global"`` forces one shared
dt across all worlds and is for the adaptive-vs-naive-batched comparison only.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Env-var contract -- single source of truth (the overlay reads these names).
# Keep these constants identical to what the overlay's _get_solver reads and to
# the dropdown's ADAPTIVE_SELECT_SETTING so all three layers agree on one name.
# ---------------------------------------------------------------------------
ENV_ADAPTIVE = "NEWTON_ADAPTIVE"
ENV_TOL = "NEWTON_ADAPTIVE_TOL"
ENV_DT_MODE = "NEWTON_ADAPTIVE_DTMODE"
ENV_DT_INIT = "NEWTON_ADAPTIVE_DT_INIT"
ENV_DT_MIN = "NEWTON_ADAPTIVE_DT_MIN"

# Defaults mirror SolverMuJoCoAdaptive.__init__
# (solver_mujoco_adaptive.py:319-351). dt_mode default is "per_world" so per-env
# dt divergence is observable.
DEFAULT_TOL = 1e-3
DEFAULT_DT_MODE = "per_world"
DEFAULT_DT_INIT = 0.01
DEFAULT_DT_MIN = 1e-6

_VALID_DT_MODES = ("per_world", "global")


def select_adaptive(
    *,
    tol: float = DEFAULT_TOL,
    dt_mode: str = DEFAULT_DT_MODE,
    dt_init: float = DEFAULT_DT_INIT,
    dt_min: float = DEFAULT_DT_MIN,
) -> None:
    """Select the adaptive Newton solver mode for the next run.

    Sets the process env vars the overlay's ``NewtonStage._get_solver`` reads so
    it builds :class:`SolverMuJoCoAdaptive` instead of stock ``SolverMuJoCo``.
    Must be called before the solver is built (before the env's first step).

    Args:
        tol: Inf-norm per-world error tolerance [m or rad] -> ``tol``.
        dt_mode: ``"per_world"`` (default; each env adapts independently) or
            ``"global"`` (one shared dt across worlds).
        dt_init: Initial inner timestep [s] -> ``dt_inner_init``. Must be
            ``<=`` the physics frame dt or the first inner step is clamped to
            the frame dt immediately (harmless but wastes the warm start).
        dt_min: Minimum inner timestep [s] -> ``dt_inner_min``. Must be ``<``
            ``dt_init``.

    Raises:
        ValueError: If ``dt_mode`` is not one of ``"per_world"`` / ``"global"``,
            or ``dt_min >= dt_init``.
    """
    if dt_mode not in _VALID_DT_MODES:
        raise ValueError(f"dt_mode must be one of {_VALID_DT_MODES}, got {dt_mode!r}")
    if dt_min >= dt_init:
        raise ValueError(f"dt_min ({dt_min}) must be strictly less than dt_init ({dt_init}).")

    os.environ[ENV_ADAPTIVE] = "1"
    os.environ[ENV_TOL] = repr(float(tol))
    os.environ[ENV_DT_MODE] = dt_mode
    os.environ[ENV_DT_INIT] = repr(float(dt_init))
    os.environ[ENV_DT_MIN] = repr(float(dt_min))


def select_stock_mjwarp() -> None:
    """Force stock MuJoCo-Warp (clear the adaptive selection).

    Removes the adaptive env vars so the overlay's ``_get_solver`` falls back to
    ``SolverMuJoCo``. Use to override a flag inherited from a base cfg.
    """
    for key in (ENV_ADAPTIVE, ENV_TOL, ENV_DT_MODE, ENV_DT_INIT, ENV_DT_MIN):
        os.environ.pop(key, None)


def is_adaptive_selected() -> bool:
    """Whether the adaptive solver mode is currently selected via the env var."""
    return os.environ.get(ENV_ADAPTIVE) == "1"


def resolve(env_cfg: object | None = None) -> bool:
    """Bridge a config-driven choice into the env-var contract the wheel reads.

    This is the *config-driven* entrypoint. It honours, in priority order:

    1. An explicit ``NEWTON_ADAPTIVE`` env var already set by the caller
       (the documented manual override / fallback) -- left untouched.
    2. An ``adaptive`` flag discovered on ``env_cfg`` (see :func:`_cfg_adaptive`),
       e.g. ``env_cfg.sim.physics.solver_cfg.adaptive`` or a top-level
       ``env_cfg.adaptive``. When truthy, calls :func:`select_adaptive` reading
       any sibling ``adaptive_tol`` / ``adaptive_dt_mode`` / ``adaptive_dt_init``
       / ``adaptive_dt_min`` fields; otherwise leaves the selection unset
       (stock MuJoCo-Warp).

    Because Isaac Lab's ``NewtonCfg`` is ignored by the wheel that actually runs
    Trossen, this function is the supported way to make a task's solver choice
    take effect: put the flag on the cfg, call ``resolve(env_cfg)`` once before
    launch, and the wheel sees it through the env var.

    Args:
        env_cfg: The resolved env cfg (or any object carrying the flag). ``None``
            falls back to the env var only.

    Returns:
        ``True`` if the adaptive mode is selected after resolution.
    """
    # (1) explicit env var wins -- manual override / fallback path.
    if os.environ.get(ENV_ADAPTIVE) == "1":
        # Backfill any unset tuning vars with documented defaults so a bare
        # `NEWTON_ADAPTIVE=1` still produces a fully-specified contract.
        os.environ.setdefault(ENV_TOL, repr(DEFAULT_TOL))
        os.environ.setdefault(ENV_DT_MODE, DEFAULT_DT_MODE)
        os.environ.setdefault(ENV_DT_INIT, repr(DEFAULT_DT_INIT))
        os.environ.setdefault(ENV_DT_MIN, repr(DEFAULT_DT_MIN))
        return True

    # (2) config-driven flag on the cfg.
    flag, params = _cfg_adaptive(env_cfg)
    if flag:
        select_adaptive(**params)
        return True

    return False


def _cfg_adaptive(env_cfg: object | None) -> tuple[bool, dict]:
    """Discover an ``adaptive`` flag (+ tuning) anywhere it is conventionally put.

    Looks, in order, at:

    * ``env_cfg.sim.physics.solver_cfg`` -- the natural home if a future
      ``MJWarpSolverCfg.adaptive`` field exists (PATH A parity).
    * ``env_cfg.sim.physics`` -- a flag hung directly on the ``NewtonCfg``.
    * ``env_cfg`` -- a top-level convenience flag.

    Tuning fields (``adaptive_tol`` / ``adaptive_dt_mode`` / ``adaptive_dt_init``
    / ``adaptive_dt_min``) are read from the same object that carried the flag.
    Missing tuning fields fall back to the module defaults.

    Returns:
        ``(selected, params)`` where ``params`` is the kwargs for
        :func:`select_adaptive`. ``selected`` is ``False`` when no flag is found.
    """
    if env_cfg is None:
        return False, {}

    candidates: list[object] = []
    sim = getattr(env_cfg, "sim", None)
    physics = getattr(sim, "physics", None) if sim is not None else None
    solver_cfg = getattr(physics, "solver_cfg", None) if physics is not None else None
    for obj in (solver_cfg, physics, env_cfg):
        if obj is not None:
            candidates.append(obj)

    for obj in candidates:
        if bool(getattr(obj, "adaptive", False)):
            return True, {
                "tol": float(getattr(obj, "adaptive_tol", DEFAULT_TOL)),
                "dt_mode": str(getattr(obj, "adaptive_dt_mode", DEFAULT_DT_MODE)),
                "dt_init": float(getattr(obj, "adaptive_dt_init", DEFAULT_DT_INIT)),
                "dt_min": float(getattr(obj, "adaptive_dt_min", DEFAULT_DT_MIN)),
            }

    return False, {}
