# Copyright (c) 2026, Newton-Adaptive workstream. SPDX-License-Identifier: BSD-3-Clause
"""GUI toggle for the Newton adaptive integrator (Isaac Lab / Isaac Sim editor).

Open the Script Editor (Window > Script Editor), paste this whole file, press Run.
A floating "Newton Integrator" window appears with one checkbox:

    [x] Adaptive timestepping  (SolverMuJoCoAdaptive)

The checkbox writes the carb setting ``/isaaclab/newton/adaptive``. ``NewtonMJWarpManager.
_build_solver`` reads that setting when it builds the solver, so the toggle switches the
integrator between fixed-step ``SolverMuJoCo`` (off) and error-controlled step-doubling
``SolverMuJoCoAdaptive`` (on).

Because switching the integrator means *rebuilding* the solver, the change applies on the
next solver build -- press **Stop** then **Play** (or reload the scene/task) after toggling.
A live mid-sim swap is intentionally avoided (it would have to re-seat contacts and state on
the running model). Watch ``/tmp/newton_adaptive.log`` to confirm: with adaptive ON, contact
frames show ``inner_dt`` spread > 0 and many substeps; OFF, it is fixed.
"""

import carb
import omni.ui as ui

_SETTING = "/isaaclab/newton/adaptive"
_settings = carb.settings.get_settings()
if _settings.get(_SETTING) is None:
    _settings.set_bool(_SETTING, False)


def _on_changed(model):
    on = bool(model.get_value_as_bool())
    _settings.set_bool(_SETTING, on)
    mode = "SolverMuJoCoAdaptive (step-doubling)" if on else "SolverMuJoCo (fixed-step)"
    print(f"[Newton Integrator] adaptive = {on}  ->  {mode}.  Stop + Play to rebuild the solver.")


# Keep a module-level ref so the window/callback survive past the script's run scope.
_window = ui.Window("Newton Integrator", width=360, height=150)
with _window.frame:
    with ui.VStack(spacing=10, height=0):
        ui.Label("Newton solver — timestepping mode", style={"font_size": 16})
        with ui.HStack(height=26, spacing=8):
            _cb = ui.CheckBox(width=22)
            _cb.model.set_value(_settings.get_as_bool(_SETTING))
            _cb.model.add_value_changed_fn(_on_changed)
            ui.Label("Adaptive timestepping  (SolverMuJoCoAdaptive)")
        ui.Label(
            "ON = error-controlled step-doubling, subdivides dt at stiff contact.\n"
            "OFF = fixed-step MuJoCo-Warp.\n"
            "Applies on the next Stop → Play (the solver is rebuilt).",
            style={"font_size": 12, "color": 0xFF9AA0A6},
            word_wrap=True,
        )

print(f"[Newton Integrator] toggle ready; {_SETTING} = {_settings.get_as_bool(_SETTING)}")
