# Isaac Sim Newton-backend overlay (adaptive)

Modified `isaacsim.physics.newton/.../impl/newton_stage.py`: makes the Newton backend run
`SolverMuJoCoAdaptive` (error-controlled step-doubling) as a **solver mode** of the single Newton
engine. `newton_stage.py.patch` = diff vs pristine Isaac Sim 6.0.0.1 (`newton_stage.pristine.bak`).

Edits:
- **`_get_solver` is config-driven.** Pops adaptive control fields (`adaptive`, `adaptive_tol`,
  `adaptive_dt_mode`, `adaptive_dt_init`, `adaptive_dt_min`) from the solver-config dict
  (defensively via `pop(..., None)`, so a stock unpatched `solver_config.py` still works). Adaptive
  is selected when `solver_cfg.adaptive` is truthy **or** env `NEWTON_ADAPTIVE=1` (locked fallback);
  config wins when present. `dt_mode` defaults to **`per_world`** so each env adapts its own dt — the
  per-env divergence is the observable "dt varies" signal. Fields the adaptive solver sets itself
  (`use_mujoco_contacts`/`use_mujoco_cpu`/`separate_worlds`) are popped before forwarding.
- **File-based instrumentation** (Kit swallows stdout): markers go to `$NEWTON_ADAPTIVE_LOG`
  (default `/tmp/newton_adaptive.log`). On adaptive construction, a confirmation line; in
  `simulate()`, every `NEWTON_ADAPTIVE_LOG_EVERY` frames (default 30) a line with per-world inner dt
  `min`/`max`/`spread`, `iteration_count`, and per-frame + cumulative substep counts (read from
  `solver.dt` / `solver.cumulative_substeps()` / `solver.iteration_count`). Reads are gated per-N
  **frames**, never per substep, to stay out of the hot path.
- **Routing kept:** one `step_dt(step_dt, s0, s1, control)` per frame for any `step_dt` solver
  (adaptive does not match Isaac's 5-positional `step(...)`); `use_cuda_graph=False` for it
  (data-dependent substep count can't be CUDA-graph captured).

Selection env vars: `NEWTON_ADAPTIVE`, `NEWTON_ADAPTIVE_TOL`, `NEWTON_ADAPTIVE_DTMODE`
(`per_world`|`global`), `NEWTON_ADAPTIVE_DTINIT`, `NEWTON_ADAPTIVE_DTMIN`. Telemetry env vars:
`NEWTON_ADAPTIVE_LOG`, `NEWTON_ADAPTIVE_LOG_EVERY`.

## Applying the overlay

Run `../apply_overlay.sh` (from this directory's parent, `adaptive_isaac/`). It resolves the installed
wheel file from the active venv (`$VIRTUAL_ENV`, else `~/Documents/code/IsaacLab/env_isaaclab`) and swaps
this `newton_stage.py` into it. The real, working apply path is a **hardlink-safe `rm` + `cp` into the
installed wheel**: uv installs `newton_stage.py` as a hardlink into its global cache, so the script `rm`s
the wheel file first (breaking the hardlink so the shared cache is never mutated) and then `cp`s the overlay
in. The Kit `--ext-folder` override did **not** take precedence under IsaacLab's launcher — the wheel copy
of `newton_stage.py` was always the one imported — so swapping the wheel file is the only path that takes
effect. On its first run the script captures the pristine wheel file to `newton_stage.pristine.bak` (seeded
from `/tmp/newton_stage.wheel.bak` if present), verifies the wheel now contains the `ADAPTIVE OVERLAY`
sentinel (fail-fast otherwise), and is idempotent and re-appliable after any `uv`/`pip` reinstall blows the
file away. Restore the pristine file with `../apply_overlay.sh --restore`.
