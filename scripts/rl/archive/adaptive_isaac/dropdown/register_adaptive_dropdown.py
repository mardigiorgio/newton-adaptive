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

"""Register a "Newton -- Adaptive" row in the Isaac Sim physics-backend dropdown.

The viewport "Simulation" menu (``omni.physics.ui`` ``SimulationConfigViewportMenu``)
is populated entirely from the ``omni.physics.core`` registry: one row per
``physics.register_simulation(simulation, name)`` call, labelled by ``name``
(``IPhysics.get_simulation_name``). It is **not** driven by
``register_simulator_variant`` -- that API is only a name->USD-variant mapping
used by the auto variant-switcher (see ``README.md``).

So a literal "Newton -- Adaptive" dropdown row is achieved by registering a
**second** Newton-family backend with the core under that display name, whose
``NewtonStage`` builds :class:`newton.solvers.SolverMuJoCoAdaptive` instead of the
stock ``SolverMuJoCo``. The dropdown auto-refreshes via the registry-event
subscription in ``SimulationConfigManager``.

Per-stage adaptive selection
-----------------------------
The overlay ``NewtonStage._get_solver`` chooses the adaptive solver. To make the
**new** stage adaptive while the stock "Newton" stage stays MJWarp, selection must
be per-stage, not a process-global env var. This module sets a carb setting
(:data:`ADAPTIVE_SELECT_SETTING`) to ``True`` only while constructing/initializing
the adaptive stage, and the overlay's ``_get_solver`` reads that setting (with the
``NEWTON_ADAPTIVE`` env var kept as the locked fallback). See ``README.md`` ->
"Required overlay hook" for the exact one-line check the overlay must contain.

Limitations
-----------
- This registers a real dropdown row and a working adaptive backend, but it does
  **not** drive USD-variant asset swapping: the Trossen assets define no
  ``Physics`` variantSet, so the auto variant-switcher is a no-op for them
  (documented in ``README.md``). Selecting the row swaps the *solver*, not asset
  variants -- which is exactly what "Newton -- Adaptive as a solver mode" wants.
- ``NewtonStage`` holds per-startup stage subscriptions; two coexisting stages are
  gated by the dropdown's "Toggle Exclusive" setting (on by default), so only one
  is active at a time. The LEAD should validate two-stage coexistence at runtime
  (see ``README.md`` -> "Verification plan").
"""

from __future__ import annotations

import os
from typing import Any

# Display name of the new dropdown row. Matches the "Newton -- Adaptive" label
# requested in the design. ``get_simulation_name`` returns this verbatim.
ADAPTIVE_BACKEND_NAME = "Newton -- Adaptive"

# Carb setting the overlay ``_get_solver`` reads to select the adaptive solver
# for *this* stage. Set True only around adaptive-stage init (see _adaptive_selection()).
ADAPTIVE_SELECT_SETTING = "/exts/isaacsim.physics.newton/adaptive_solver"

# Variant-switcher mapping registered so the auto-switcher does not error when the
# adaptive backend is activated. No-op for assets without a Physics variantSet
# (e.g. Trossen). "mujoco" matches the stock Newton mapping.
ADAPTIVE_VARIANT_NAME = "mujoco"


class _AdaptiveSelection:
    """Context manager that flips the adaptive carb setting while a stage builds.

    The overlay reads :data:`ADAPTIVE_SELECT_SETTING` inside ``_get_solver``; that
    runs during ``NewtonStage._initialize_newton_impl`` (lazily, on first play).
    Because solver construction is deferred to first ``step_sim``, a plain
    construct-time flip is not sufficient on its own -- the recommended overlay
    hook also stamps a per-stage attribute at construction (``README.md``). This
    context manager is the construction-time half and is safe to keep set, since
    the carb key namespaces the adaptive backend.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._prev: Any = None

    def __enter__(self) -> _AdaptiveSelection:
        import carb

        settings = carb.settings.get_settings()
        self._prev = settings.get(ADAPTIVE_SELECT_SETTING)
        settings.set(ADAPTIVE_SELECT_SETTING, self._enabled)
        return self

    def __exit__(self, *exc: Any) -> None:
        import carb

        carb.settings.get_settings().set(ADAPTIVE_SELECT_SETTING, self._prev)


class AdaptiveDropdownRegistry:
    """Registers/unregisters the "Newton -- Adaptive" physics backend + dropdown row.

    Wraps a dedicated :class:`NewtonStage` (configured to build the adaptive solver)
    and registers it with ``omni.physics.core`` under :data:`ADAPTIVE_BACKEND_NAME`,
    mirroring ``isaacsim.physics.newton.impl.register_simulation.NewtonSimulationRegistry``
    but with a parameterized backend name and adaptive-solver selection.
    """

    def __init__(self) -> None:
        self.simulation_id: int | None = None
        self.newton_stage: Any = None
        self.simulation: Any = None
        self.sim_fns: Any = None
        self.stage_update_fns: Any = None

    def _build_adaptive_stage(self) -> Any:
        """Construct a NewtonStage whose _get_solver yields the adaptive solver.

        Returns:
            A ``NewtonStage`` instance configured for the adaptive solver mode.

        Raises:
            RuntimeError: If the overlay is not installed (no adaptive routing) or
                the Newton extension impl modules cannot be imported.
        """
        # These live in the installed wheel (impl package). Import lazily so this
        # module is importable from a plain Python shell for inspection/tests.
        # Fail fast if the overlay is not applied: the stock wheel _get_solver has
        # no adaptive branch, so the "Newton -- Adaptive" row would silently build
        # the stock MJWarp solver. The overlay leaves a sentinel constant/marker.
        import inspect

        from isaacsim.physics.newton.impl.newton_config import NewtonConfig
        from isaacsim.physics.newton.impl.newton_stage import NewtonStage

        src = ""
        try:
            src = inspect.getsource(NewtonStage._get_solver)
        except (OSError, TypeError):
            pass
        if "ADAPTIVE OVERLAY" not in src:
            raise RuntimeError(
                "[adaptive-dropdown] overlay not detected in NewtonStage._get_solver -- "
                "run apply_overlay.sh before registering the 'Newton -- Adaptive' backend, "
                "otherwise the row would build the stock MJWarp solver."
            )

        cfg = NewtonConfig()
        # The adaptive solver owns a data-dependent substep count, incompatible
        # with a static CUDA graph (the overlay also enforces this once the solver
        # is built; we set it up front so the first-step path is correct too).
        cfg.use_cuda_graph = False

        # Build the stage with the adaptive carb flag set so any eager solver
        # construction picks adaptive. The overlay's per-stage attribute (set in
        # _get_solver when it sees this flag at construction) carries the choice
        # through the deferred first-step build. See README "Required overlay hook".
        with _AdaptiveSelection(enabled=True):
            stage = NewtonStage(cfg=cfg)
            # Stamp the stage so the overlay's _get_solver can prefer per-stage
            # state over the global carb flag if both are present.
            try:
                stage._adaptive_solver = True
            except Exception:
                pass
        return stage

    def register(self) -> int | None:
        """Register the adaptive backend and add its dropdown row.

        Returns:
            The simulation id assigned by the physics interface, or ``None`` on
            failure.
        """
        try:
            import carb

            # Reuse the wheel's wrapper classes verbatim: the adaptive stage routes
            # through the SAME simulate()/step_sim plumbing -- the only difference
            # is which solver _get_solver builds.
            from isaacsim.physics.newton.impl.simulation_functions import NewtonSimulationFunctions
            from isaacsim.physics.newton.impl.stage_update_functions import NewtonStageUpdateFunctions
            from omni.physics.core import Simulation, get_physics_interface, k_invalid_simulation_id

            self.newton_stage = self._build_adaptive_stage()

            self.sim_fns = NewtonSimulationFunctions(self.newton_stage)
            self.stage_update_fns = NewtonStageUpdateFunctions(self.newton_stage)
            self.newton_stage.simulation_functions = self.sim_fns

            self.simulation = Simulation()

            sf = self.simulation.simulation_fns
            sf.initialize = self.sim_fns.initialize
            sf.close = self.sim_fns.close
            sf.get_attached_stage = self.sim_fns.get_attached_stage
            sf.simulate = self.sim_fns.simulate
            sf.fetch_results = self.sim_fns.fetch_results
            sf.check_results = self.sim_fns.check_results
            sf.flush_changes = self.sim_fns.flush_changes
            sf.pause_change_tracking = self.sim_fns.pause_change_tracking
            sf.is_change_tracking_paused = self.sim_fns.is_change_tracking_paused
            sf.subscribe_physics_contact_report_events = self.sim_fns.subscribe_physics_contact_report_events
            sf.unsubscribe_physics_contact_report_events = self.sim_fns.unsubscribe_physics_contact_report_events
            sf.get_simulation_time_steps_per_second = self.sim_fns.get_simulation_time_steps_per_second
            sf.get_simulation_timestamp = self.sim_fns.get_simulation_timestamp
            sf.get_simulation_step_count = self.sim_fns.get_simulation_step_count
            sf.subscribe_physics_on_step_events = self.sim_fns.subscribe_physics_on_step_events
            sf.unsubscribe_physics_on_step_events = self.sim_fns.unsubscribe_physics_on_step_events
            sf.is_capable_of_simulating = self.sim_fns.is_capable_of_simulating

            su = self.simulation.stage_update_fns
            su.start_simulation = self.stage_update_fns.start_simulation
            su.on_attach = self.stage_update_fns.on_attach
            su.on_detach = self.stage_update_fns.on_detach
            su.on_update = self.stage_update_fns.on_update
            su.on_resume = self.stage_update_fns.on_resume
            su.on_pause = self.stage_update_fns.on_pause
            su.on_reset = self.stage_update_fns.on_reset
            su.force_load_physics_from_usd = self.stage_update_fns.force_load_physics_from_usd
            su.release_physics_objects = self.stage_update_fns.release_physics_objects
            su.handle_raycast = self.stage_update_fns.handle_raycast
            su.reset_simulation = self.stage_update_fns.reset_simulation

            physics = get_physics_interface()
            self.simulation_id = physics.register_simulation(self.simulation, ADAPTIVE_BACKEND_NAME)

            if self.simulation_id == k_invalid_simulation_id:
                carb.log_error("[adaptive-dropdown] register_simulation returned invalid id")
                self.simulation_id = None
                return None

            self.sim_fns.simulation_id = self.simulation_id

            self._register_variant_mapping()

            carb.log_warn(
                f"[adaptive-dropdown] Registered '{ADAPTIVE_BACKEND_NAME}' backend "
                f"(simulation id {self.simulation_id})."
            )
            return self.simulation_id

        except Exception as e:
            try:
                import carb

                carb.log_error(f"[adaptive-dropdown] Failed to register '{ADAPTIVE_BACKEND_NAME}': {e}")
            except Exception:
                print(f"[adaptive-dropdown] Failed to register '{ADAPTIVE_BACKEND_NAME}': {e}", flush=True)
            import traceback

            traceback.print_exc()
            return None

    def _register_variant_mapping(self) -> None:
        """Register a variant-switcher mapping so activation does not error.

        No-op for assets without a Physics variantSet (e.g. Trossen). Best-effort.
        """
        try:
            from omni.physics.isaacsimready import get_variant_switcher

            get_variant_switcher().register_simulator_variant(ADAPTIVE_BACKEND_NAME, ADAPTIVE_VARIANT_NAME)
        except Exception as e:
            try:
                import carb

                carb.log_warn(f"[adaptive-dropdown] variant mapping skipped: {e}")
            except Exception:
                pass

    def unregister(self) -> None:
        """Unregister the adaptive backend and remove its dropdown row."""
        if self.simulation_id is not None:
            try:
                import carb
                from omni.physics.core import get_physics_interface

                get_physics_interface().unregister_simulation(self.simulation_id)
                carb.log_warn(f"[adaptive-dropdown] Unregistered '{ADAPTIVE_BACKEND_NAME}'.")
            except Exception as e:
                print(f"[adaptive-dropdown] Failed to unregister: {e}", flush=True)
            finally:
                self.simulation_id = None
        try:
            from omni.physics.isaacsimready import get_variant_switcher

            get_variant_switcher().unregister_simulator_variant(ADAPTIVE_BACKEND_NAME)
        except Exception:
            pass
        if self.newton_stage is not None:
            try:
                self.newton_stage.init()
            except Exception:
                pass
            self.newton_stage = None


# Process-global registry so a run-hook caller and a Kit-extension caller share
# one instance and never double-register the row.
_REGISTRY: AdaptiveDropdownRegistry | None = None


def register_adaptive_dropdown() -> AdaptiveDropdownRegistry | None:
    """Register the "Newton -- Adaptive" dropdown row (idempotent).

    Safe to call from a Kit extension ``on_startup`` or as a post-launch run hook
    after ``SimulationApp`` exists. Honors ``ADAPTIVE_DROPDOWN_DISABLE=1`` to skip.

    Returns:
        The active :class:`AdaptiveDropdownRegistry`, or ``None`` if disabled or
        registration failed.
    """
    global _REGISTRY

    if os.environ.get("ADAPTIVE_DROPDOWN_DISABLE") == "1":
        return None

    if _REGISTRY is not None and _REGISTRY.simulation_id is not None:
        return _REGISTRY

    reg = AdaptiveDropdownRegistry()
    if reg.register() is None:
        return None
    _REGISTRY = reg
    return reg


def unregister_adaptive_dropdown() -> None:
    """Unregister the dropdown row if it was registered (idempotent)."""
    global _REGISTRY
    if _REGISTRY is not None:
        _REGISTRY.unregister()
        _REGISTRY = None
