# "Newton -- Adaptive" physics-backend dropdown row

Adds a **`Newton -- Adaptive`** entry to the Isaac Sim viewport **Simulation**
physics-backend menu. Selecting it runs the Newton backend with
`SolverMuJoCoAdaptive` (error-controlled adaptive timestepping) instead of stock
MJWarp.

This is the **GUI** selector for the adaptive solver *mode*. The locked primary
selector for headless training is config/env-var driven in the overlay
`_get_solver` (`NEWTON_ADAPTIVE=1`); this dropdown is the interactive companion.

## How the dropdown actually works (verified against installed source)

The viewport menu is **not** built from `register_simulator_variant`. It is
populated one row per backend registered with `omni.physics.core`:

- `SimulationConfigManager.refresh()` lists `physics.get_simulation_ids()` and
  labels each via `physics.get_simulation_name(sim_id)`
  (`omni.physics.ui-110.1.11.../scripts/simulation_config.py:113-121`).
- Each row toggles `physics.activate_simulation(id)` / `deactivate_simulation(id)`
  (`simulation_config.py:49-72`).
- A backend (and thus a row) is created by
  `IPhysics.register_simulation(simulation, name) -> SimulationId`
  (`omni.physics-110.1.11.../omni/physics/core/bindings/_physics.pyi:666`). The
  row label is exactly the `name` argument.
- Stock Newton registers its row at
  `isaacsim.physics.newton/.../impl/register_simulation.py:121`
  (`register_simulation(self.simulation, "Newton")`).

So a real `Newton -- Adaptive` row = **a second backend registered with a distinct
name** whose `NewtonStage._get_solver` returns the adaptive solver. That is what
this module does: it mirrors `NewtonSimulationRegistry.register_newton` verbatim
(same `simulation_fns` / `stage_update_fns` wiring) but parameterizes the name and
forces adaptive selection for that stage. The dropdown auto-refreshes via the
registry-event subscription in `SimulationConfigManager`.

`register_simulator_variant` is also called (mapping `Newton -- Adaptive` ->
`mujoco`) only so the auto variant-switcher does not error on activation. It is
**inert for Trossen** (see Limitation below).

## Files

| File | Purpose |
| --- | --- |
| `register_adaptive_dropdown.py` | Core: `AdaptiveDropdownRegistry`, `register_adaptive_dropdown()`, `unregister_adaptive_dropdown()`. |
| `extension.py` | Kit `IExt` wrapper (`on_startup`/`on_shutdown`). |
| `extension.toml` | Kit manifest (declares load-after deps on the Newton backend). |
| `run_hook.py` | `install()` / `uninstall()` for non-Kit / standalone launches. |
| `__init__.py` | Package re-exports. |

## How it is loaded

Two paths; pick one.

### A. As a Kit extension (interactive GUI sessions)

Point Kit at the parent folder of this package and enable it. The ext must load
**after** `isaacsim.physics.newton` (dependency declared in `extension.toml`):

```bash
# layout-dependent; adjust the ext name to your packaging
--ext-folder ~/Documents/code/newton-adaptive/scripts/rl/adaptive_isaac \
--enable     adaptive_isaac.dropdown
```

> The VERIFIED FACTS note `--ext-folder` did **not** take precedence under
> IsaacLab's launcher *for the newton_stage overlay*. That was a file-override
> conflict (two copies of the same module). This dropdown registers a **new**
> backend rather than overriding a wheel file, so the conflict does not apply.
> If `--ext-folder` still does not load it under your launcher, use path B.

### B. As a post-launch run hook (standalone / training entrypoints)

After `AppLauncher`/`SimulationApp` has started Kit and the stock Newton
extension is loaded:

```python
from scripts.rl.adaptive_isaac.dropdown.run_hook import install
install()   # adds the "Newton -- Adaptive" row; idempotent
```

`ADAPTIVE_DROPDOWN_DISABLE=1` skips registration.

## Required overlay hook (one line in `overlay/newton_stage.py`)

For the **new** stage to be adaptive while the stock `Newton` stage stays MJWarp,
selection must be **per-stage**, not a process-global env var. This module sets a
per-stage signal in two redundant ways while building the adaptive stage:

1. a carb setting `/exts/isaacsim.physics.newton/adaptive_solver = True`
   (`ADAPTIVE_SELECT_SETTING`), and
2. a stage attribute `stage._adaptive_solver = True`.

The overlay `_get_solver` (currently gated only on `NEWTON_ADAPTIVE==1`) must also
honor these. Recommended change in `overlay/newton_stage.py` `_get_solver`
(`elif solver_type == "mujoco":` branch), keeping the env var as fallback:

```python
# [ADAPTIVE OVERLAY] per-stage selection for the "Newton -- Adaptive" dropdown row.
_adaptive = (
    getattr(cls_self, "_adaptive_solver", False)            # per-stage attr (preferred)
    or carb.settings.get_settings().get("/exts/isaacsim.physics.newton/adaptive_solver")  # carb flag
    or _ovl_os.environ.get("NEWTON_ADAPTIVE") == "1"        # locked env-var fallback
)
if _adaptive:
    ... SolverMuJoCoAdaptive(...)
```

Note `_get_solver` is currently a `@classmethod`, so it has no `self`. To read the
per-stage attribute (option 1) the overlay should either make `_get_solver` an
instance method or pass the stage in. If that refactor is undesired, the **carb
flag alone (option 2) is sufficient** when the dropdown registers the adaptive
backend before the stock one plays, since this module sets the flag at adaptive
stage construction. Simplest robust path: have `_get_solver` read the carb flag +
env var (no `self` needed). This module sets the carb flag at the right time.

This overlay edit is **not** part of this dropdown component's deliverable (it
belongs to the overlay component) but is the contract the dropdown depends on. The
registry fails fast with a clear error if the overlay's adaptive branch is absent
(it greps `NewtonStage._get_solver` source for the `ADAPTIVE OVERLAY` sentinel).

## Limitation: USD asset variants (Trossen)

A correctly registered `Newton -- Adaptive` -> variant mapping switches **nothing**
on Trossen: the Trossen assets (`isaac-data/.../stationary_ai*.usd[a]`) contain
**zero `variantSet`s** and no `Physics` variant set, so
`switch_variants_for_simulation` early-returns 0
(`omni.physics.isaacsimready-110.1.11.../scripts/variant_switcher.py:159,183-185`).

This is fine for the adaptive **solver mode**: selecting the row swaps the solver
(via the second registered backend), not asset variants. Full USD-variant-driven
asset swapping is **out of scope** and would require authoring `physx` / `newton`
/ `adaptive` variants into every Trossen USD.

**Follow-up (asset-variant work, deferred):** add a `Physics` `variantSet` to the
Trossen robot/scene USDs with at least a `mujoco`/adaptive variant selection, so
the auto variant-switcher (`register_simulator_variant` mapping) takes effect. Not
required for the adaptive-solver demonstration.

## Two-backend coexistence caveat

`NewtonStage` holds per-startup stage subscriptions (timeline events, USD change
tracking). Two coexisting Newton-family stages are gated by the dropdown's
**Toggle Exclusive** setting (on by default), so only one is active at a time.
Whether two `NewtonStage` instances coexist cleanly (no subscription/stage-attach
conflict) is **unverified in source** and must be confirmed at runtime.

## Verification plan (for the LEAD)

1. **Row appears.** Launch Isaac Sim GUI with the stock Newton ext + this module
   (path A or B). Open the viewport **Simulation** menu. Confirm three rows:
   `PhysX`, `Newton`, `Newton -- Adaptive`. (Equivalently, in a Kit Python
   console: `from omni.physics.core import get_physics_interface as g;
   [g().get_simulation_name(i) for i in g().get_simulation_ids()]` includes
   `"Newton -- Adaptive"`.)
2. **Activation works.** Select `Newton -- Adaptive`; confirm
   `is_simulation_active(id)` is True for that id and (with Toggle Exclusive on)
   False for `Newton`/`PhysX`.
3. **Adaptive solver is actually built.** With a stage loaded and playing, confirm
   the adaptive backend's `NewtonStage.solver` is a `SolverMuJoCoAdaptive`
   (`type(reg.newton_stage.solver).__name__`), and that the overlay routed through
   `step_dt` (it sets `cfg.use_cuda_graph = False`). Rely on the overlay's
   file-based marker + per-frame substep/dt log.
4. **dt varies.** Read the adaptive solver telemetry
   (`solver.sim_time` / `solver.dt` / `solver.iteration_count` /
   `solver.cumulative_substeps()`): a **non-constant** `dt` (and substep count
   varying with contact) confirms adaptation. A constant dt means the row fell
   back to stock MJWarp -- check the overlay hook above is in place.
5. **Overlay precondition.** If registration logs the fail-fast
   `overlay not detected in NewtonStage._get_solver` error, run
   `../apply_overlay.sh` first.

## Relationship to the locked config/env-var selector

This dropdown is **secondary** to the locked primary path (config-driven
`_get_solver` + `NEWTON_ADAPTIVE` env-var fallback). For headless Trossen training
there is no GUI; selection is via the env var / Trossen physics preset. The
dropdown is only relevant to interactive GUI sessions and is the clean answer to
"register a `Newton -- Adaptive` entry in the variant switcher" within the limits
of what the installed `omni.physics` stack supports.
