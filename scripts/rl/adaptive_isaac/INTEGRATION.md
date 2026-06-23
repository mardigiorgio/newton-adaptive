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
- **Config-driven** (the platform path): `NewtonCfg(solver_cfg=MJWarpSolverCfg(adaptive=True, adaptive_dt_init=0.005, …))`.
  The Trossen `newton_adaptive` preset (`cube_lift_env_cfg.py`) does exactly this.
- **Env toggle** (any task, no cfg edit): `NEWTON_ADAPTIVE=1` (read by `_build_solver`); tune with
  `NEWTON_ADAPTIVE_DT_INIT`, `NEWTON_ADAPTIVE_DTMODE` (`per_world`|`global`), `NEWTON_ADAPTIVE_TOL`,
  `NEWTON_ADAPTIVE_LOG_EVERY`.

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

## Status of the GUI dropdown / wheel overlay (superseded for Isaac Lab)
`overlay/` (the wheel `isaacsim.physics.newton` patch) and `dropdown/` were authored against the Isaac Sim
**standalone** path and are **superseded for Isaac Lab** by the `isaaclab_newton` edits above (Isaac Lab never
steps the wheel). They are kept for the standalone-Isaac-Sim GUI path. Remaining work for the "Newton — Adaptive"
**GUI dropdown**: re-point its selection to set `MJWarpSolverCfg.adaptive=True` (the cfg flag that actually
engages adaptive) and verify interactively in the editor — the headless/training path is complete and proven.
