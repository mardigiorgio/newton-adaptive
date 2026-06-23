# Adaptive Newton backend — Isaac Lab integration (PROVEN, 2026-06)

`SolverMuJoCoAdaptive` (error-controlled step-doubling) runs as a **selectable solver mode** of Isaac Lab's
Newton backend and **visibly adapts** on contact-rich manipulation. End-to-end verified on Trossen cube-lift.

## The integration point — `isaaclab_newton`, NOT the wheel

Isaac Lab's `env.sim.physics=newton_mjwarp` is stepped by the **native** `isaaclab_newton.NewtonMJWarpManager`
— it builds the solver and steps it itself, explicitly bypassing the Isaac Sim wheel extension
`isaacsim.physics.newton` ("no isaacsim.physics.newton extension needed", `newton_manager.py:392`). So the
integration is **5 edits to that manager + its cfg** (editable Isaac Lab source = the "modified Isaac Lab"
the monorepo houses), captured in [`isaaclab_newton_adaptive.patch`](isaaclab_newton_adaptive.patch):

1. `MJWarpSolverCfg`: `adaptive` + `adaptive_tol`/`adaptive_dt_mode`/`adaptive_dt_init`/`adaptive_dt_min` fields.
2. `NewtonMJWarpManager._build_solver`: when `adaptive`, construct `SolverMuJoCoAdaptive` (popping the kwargs it
   forces — `use_mujoco_contacts`/`use_mujoco_cpu`/`separate_worlds`) instead of stock `SolverMuJoCo`.
3. `NewtonMJWarpManager._step_solver`: adaptive → `step_dt(substep_dt, s0, s1, control)` once per substep (the
   stock 5-positional `step()` is incompatible; `step_dt` owns the inner error-controlled loop + its contacts).
4. `_supports_cuda_graph_capture` → `False` when adaptive (the per-frame substep count is data-dependent).
5. `_log_solver_debug` → throttled **file** telemetry (`$NEWTON_ADAPTIVE_LOG`, default `/tmp/newton_adaptive.log`)
   — Kit swallows stdout, so the dt/substep proof must go to a file.

Re-apply on a fresh IsaacLab clone: `git -C ~/Documents/code/IsaacLab apply scripts/.../isaaclab_newton_adaptive.patch`
(in the monorepo these edits live on the IsaacLab fork branch).

## Selecting adaptive
`_build_solver` builds `SolverMuJoCoAdaptive` when **any** of these is set (checked in this order):
- **Config-driven** (the platform path): `NewtonCfg(solver_cfg=MJWarpSolverCfg(adaptive=True, adaptive_dt_init=0.005, …))`.
  The Trossen `newton_adaptive` preset (`cube_lift_env_cfg.py`) does exactly this.
- **Env toggle** (any task, no cfg edit): `NEWTON_ADAPTIVE=1`; tune with `NEWTON_ADAPTIVE_DT_INIT`,
  `NEWTON_ADAPTIVE_DTMODE` (`per_world`|`global`), `NEWTON_ADAPTIVE_TOL`, `NEWTON_ADAPTIVE_LOG_EVERY`.
- **GUI toggle** (carb setting `/isaaclab/newton/adaptive`): a checkbox in the editor. Flip it, then
  **Stop → Play** (the toggle applies at solver-build time — switching the integrator rebuilds the solver;
  a live mid-sim swap is intentionally avoided). Confirm via `/tmp/newton_adaptive.log` (spread > 0 +
  many substeps = on). Two ways to get the checkbox:
  - **Autorun (default):** the Kit extension `IsaacLab/source/newton_adaptive_ui/` is listed in
    `apps/isaaclab.python.kit` `[dependencies]` (`"newton_adaptive_ui" = {order = 1001}`), so a
    **Newton Integrator** window appears automatically in every Isaac Lab **GUI** session. It is *not* in
    the headless kits (`isaaclab.python.headless*.kit` define their own deps), so headless training never
    loads the UI. The rendering GUI variant inherits it via its `"isaaclab.python"` dependency.
  - **No-install fallback:** paste [`gui_toggle.py`](gui_toggle.py) into the Script Editor (Window >
    Script Editor > Run) — same window, same carb setting, for a stock IsaacLab clone without the extension.

## Proof — Trossen cube-lift (16 envs, `train_teacher.py --max_iterations 2`)

| solver run | inner-dt spread | substeps/frame | inner-dt min | result |
|---|---|---|---|---|
| Cartpole (trivial, no stiff contact) | `0.000` (flat) | ~3 (1 accepted × 3-eval doubling) | pinned to boundary | runs |
| **Trossen (contact-rich)** | **up to `5.3e-3`** | **~35** | **`1.3e-4`** (stiff grasp/table contact) | trains clean, finite rewards, exit 0 |

On Trossen the controller subdivides per-world: `dt` ranges `1e-4` (stiff contact) → `5e-3` (free motion),
~12× more substeps than the trivial cartpole. That is the thesis demonstrated — adaptive timestepping does real
work exactly where contact is stiff.

## Verify it yourself
```bash
# Trossen on the adaptive backend (per-world), telemetry to a fresh log:
rm -f /tmp/newton_adaptive.log
NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_LOG_EVERY=10 scripts/rl/trossen/run_native.sh \
  scripts/rl/trossen/train_teacher.py --headless --num_envs 16 --max_iterations 2
tail /tmp/newton_adaptive.log   # spread>0 and substeps/frame >> 3 == adapting
# Stock Newton (baseline) for comparison: NEWTON_MJWARP=1 instead.
```

## GUI control: a checkbox, not the engine dropdown
Adaptive is a **solver mode** of the Newton backend, not a separate physics engine, so the right GUI affordance
is a **toggle** (`gui_toggle.py`, above), not a new entry in the `omni.physics` engine dropdown. The dropdown
lists engines (PhysX / Newton); the checkbox flips the active Newton solver between fixed-step and adaptive.

`overlay/` (the wheel `isaacsim.physics.newton` patch) and `dropdown/` were authored against the Isaac Sim
**standalone** path and are **superseded for Isaac Lab** (Isaac Lab never steps the wheel — it builds the solver
in `isaaclab_newton`, which is what the checkbox + the three selection paths above drive). Kept only for the
standalone-Isaac-Sim GUI path.
