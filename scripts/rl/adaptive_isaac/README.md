# adaptive_isaac — the adaptive solver into the Isaac Lab Newton backend (Thread B, B1–B4)

**Status: NOT STARTED (scaffold).** Home for the *integration* glue — the Newton → Isaac Sim →
Isaac Lab hop the native install was set up for. No code here yet; B0 (standalone-Newton evidence,
`../adaptive_expts/`) gates it. The version facts below were confirmed for **Isaac Sim 6.0** (2026-06).

> **Scope.** This integrates the **adaptive solver** (`SolverMuJoCoAdaptive` — adaptive timestepping over
> MuJoCo-Warp) that exists today. **True CENIC** (adaptive + convex ICF/SAP contact) is a later goal, not
> what is wired here.

## The stack (verified for Isaac Sim 6.0)

- **Isaac Sim 6.0** is GA and supports the **Newton** physics backend — and *only* Newton's **MJWarp
  Solver** is currently supported in Isaac Sim. `SolverMuJoCoAdaptive` **is** a MuJoCo-Warp solver, so
  the adaptive solver is in the supported family: the integration is "select the adaptive MJWarp solver
  instead of the stock one," not a foreign engine.
- **Isaac Lab `develop` branch** pairs with Isaac Sim 6.0 (the 3.0-beta line). `./isaaclab.sh -i`
  installs Isaac Lab **+ Newton 1.0**. So the Newton backend **ships** — B1 is largely done-by-install,
  not a from-scratch migration.
- Python **3.12** (host has it), `torch==2.10.0` / cu128. The old binary-python3.11 path is obsolete.

Bring-up: see **[SETUP.md](SETUP.md)** — the verified, ordered, copy-pasteable environment (fork + Isaac
Lab worktrees, venv, the **mandatory** `isaacsim[all,extscache]==6.0.0.1` wheel, `./isaaclab.sh -i`, the
editable-fork override) plus the integration edit map and verification ladder. The wheel is mandatory, not
"viz-only": the `isaacsim.physics.newton` extension you patch ships **inside** it (it is not in the Isaac
Lab repo).

## Deliverables (see `../trossen/ROADMAP.md` Thread B)

- **B1 — Newton backend.** ✅ ships in Isaac Lab `develop` + Isaac Sim 6.0 (the `isaacsim.physics.newton`
  extension comes in the isaacsim wheel). Remaining: smoke a **stock** Newton MJWarp scene first — the
  `newton 1.2.0`→fork `1.4.0.dev0` runtime-interop check is the primary unknown.
- **B2 — adaptive solver into the backend.** No version reconcile needed: the fork = the pinned Newton
  commit `811968b` **+ 47 adaptive-solver commits, 0 behind**, swapped in editable by dist-name `newton`.
  Don't rebase.
- **B3 — make it adaptive (TWO edits, verified at the v6.0.0 GA tag).** The old "subclass a manager /
  `_build_solver`→`SolverMuJoCoAdaptive`" seam does **not** exist in 6.0 — `isaacsim.physics.newton`
  hard-loops a fixed
  `num_substeps` and hard-codes solver selection (no injection seam). So: **(A)** add `adaptive=True` to the
  **stock** `SolverMuJoCo` (the class Isaac builds); **(B)** patch the extension to call the solver once/frame
  with the full dt; **(C)** disable `use_cuda_graph`. Details + file:line targets in [SETUP.md](SETUP.md).
- **B4 — in-Isaac validation.** Adaptive-vs-fixed work-precision on a stiff Isaac scene.

## Selecting the adaptive solver mode (`selection.py`)

"Newton — Adaptive" is a **solver mode within the single Newton engine**, not a fourth
engine. Ideally it would be a field on Isaac Lab's `NewtonCfg` / `MJWarpSolverCfg`, but it
cannot be, because of a split between two Newton code paths:

- **PATH A — Isaac Lab native (`isaaclab_newton`).** `NewtonCfg` resolves here
  (`class_type = …mjwarp_manager:NewtonMJWarpManager`). It builds `SolverMuJoCo` directly
  and steps it 5-positional: `solver.step(s0, s1, control, contacts, substep_dt)`
  (`newton_manager.py:1351`). The adaptive solver advances a whole frame via
  `step_dt(dt, s0, s1, control)` and does **not** implement that signature → it cannot run
  on PATH A unmodified.
- **PATH B — the wheel extension (`isaacsim.physics.newton`, `NewtonStage`).** This is the
  path that actually runs Trossen/Cartpole under the Isaac Lab launcher (verified: the
  overlay's load marker + `step_dt` routing took effect). PATH B builds its own
  `NewtonConfig()` with hardcoded defaults and reads **nothing** from Isaac Lab's
  `NewtonCfg`. There is no config bridge from Isaac Lab into the wheel.

Because the wheel ignores Isaac Lab's cfg objects, the only bridge from an Isaac-Lab-side
*choice* into the wheel's `NewtonStage._get_solver` (where `SolverMuJoCoAdaptive` is
constructed) is a set of **process environment variables**. `selection.py` is the single
source of truth for that contract.

### Two independent switches are required

Running the adaptive solver needs **both**, set before the env's first `reset()`:

1. **Newton engine active** — `env.sim.physics` must be a `NewtonCfg` (not the default
   `PhysxCfg`). On the `train_teacher.py` entrypoint, `parse_env_cfg` → `resolve_presets`
   collapses any `PresetCfg` to its `.default` branch, so a preset alone always lands on
   PhysX; the Trossen cfg must assign a **direct** `NewtonCfg` under an env-var gate (this
   is the *trossen-on-newton* component's job, not `selection.py`).
2. **Adaptive solver mode** — `NEWTON_ADAPTIVE=1` (+ tuning), which `selection.py` sets and
   the overlay's `_get_solver` reads. This is what `selection.py` owns.

`selection.resolve(env_cfg)` (called in `train_teacher.py:main`, right after
`parse_env_cfg`) handles switch (2): it honours an explicit `NEWTON_ADAPTIVE=1` (manual
override / fallback), otherwise reads an `adaptive` flag off
`env_cfg.sim.physics(.solver_cfg)` / `env_cfg`. It only mutates `os.environ` — no Isaac Sim
import — so it is safe at config-build time.

### Env-var contract (read by the overlay's `_get_solver`)

| Var | Meaning | Default |
| --- | --- | --- |
| `NEWTON_ADAPTIVE`         | `"1"` → `SolverMuJoCoAdaptive`; else stock `SolverMuJoCo` | unset |
| `NEWTON_ADAPTIVE_TOL`     | `tol` — per-world inf-norm error tol [m or rad] | `1e-3` |
| `NEWTON_ADAPTIVE_DTMODE`  | `per_world` (each env adapts independently) or `global` | `per_world` |
| `NEWTON_ADAPTIVE_DT_INIT` | `dt_inner_init` — initial inner dt [s] | `0.01` |
| `NEWTON_ADAPTIVE_DT_MIN`  | `dt_inner_min` — floor inner dt [s] (must be `< dt_init`) | `1e-6` |

`dt_mode` defaults to **`per_world`** so each env adapts its dt independently — per-env dt
divergence is exactly the signal the LEAD uses to confirm the solver is adapting. (The old
overlay default `global` forces one shared dt and hides that divergence.) Coordinate this
name set with the overlay author's `_get_solver` and the dropdown's `ADAPTIVE_SELECT_SETTING`
so all three layers agree.

### Launch overrides

- **Adaptive on the Trossen teacher entrypoint** (env var is the bridge; the Trossen cfg's
  env-var gate also flips the engine to Newton):

  ```bash
  NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_DTMODE=per_world \
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py \
      --headless --num_envs 4 --max_iterations 1
  ```

- **Tune the controller** without code changes:

  ```bash
  NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_TOL=5e-4 NEWTON_ADAPTIVE_DT_INIT=0.005 \
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 4
  ```

- **Programmatic / from another entrypoint** (config-driven, no env var):

  ```python
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/rl
  from adaptive_isaac import selection
  selection.select_adaptive(tol=1e-3, dt_mode="per_world")   # or selection.resolve(env_cfg)
  ```

> The wheel must carry the overlay (`./apply_overlay.sh`) for any of this to take effect —
> a stock `newton_stage.py` ignores `NEWTON_ADAPTIVE` entirely. `dt_inner_init=0.01` equals
> the base lift task's `sim.dt` (100 Hz); if the Trossen frame dt is smaller the first inner
> step is clamped down to it (harmless, just wastes the warm start) — set
> `NEWTON_ADAPTIVE_DT_INIT` to the frame dt to avoid that.

## Open decisions

- **Binary vs pip Isaac Sim — resolved: pip is enough.** The `isaacsim==6.0.0.1` **wheel** is mandatory
  either way (it carries the `isaacsim.physics.newton` extension you patch — pure Python, editable in place
  inside the installed extscache tree, so no Isaac Sim *source build* / fork is needed). The binary install
  is only for the Omniverse RTX viewer.
- **Thread A coexistence.** The Trossen testbed is built on Isaac Lab **2.3.2** (manager API + PhysX).
  Running it on `develop` is a real port. Pin the chosen versions in `../trossen/IMPL_GROUND_TRUTH.md`.
