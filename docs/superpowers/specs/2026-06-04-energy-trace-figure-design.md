# Fig 4 — Energy-drift trace: per-world adaptive vs global fixed-10ms

**Date:** 2026-06-04
**Status:** Approved-to-build (time-boxed; user is on a deadline)
**Poster slot:** replaces the `WORK_PRECISION.png` panel (Figure 4).

## Goal

Show that **per-world adaptive stepping keeps every world's energy bounded**,
while a **single global fixed 10 ms step blows up on the stiff worlds** of a
contact-heavy scene. This is a direct visual argument for per-world adaptation:
one global dt must satisfy the worst world, so fixed-10ms is fine on easy worlds
but injects energy / explodes on stiff ones; adaptive holds them all.

## Why an energy metric (not trajectory error)

Contact-rich rigid-body dynamics are chaotic, so `‖q − q_ref‖` vs a fine-dt
reference diverges for *any* solver over time (established earlier in this
project). An energy-magnitude metric measures **solver quality, not trajectory
identity**, so it is not confounded by chaos.

## Metric

Per-world **total kinetic energy** `KE(t)` summed over the world's free bodies:

```
KE_world = sum_bodies [ 0.5 * m_b * |v_b|^2  +  0.5 * w_b . (I_b w_b) ]
```

- `m_b`: body mass (`model.body_mass`).
- `v_b`, `w_b`: linear / angular velocity from `state.body_qd`
  (spatial_vector `[wx,wy,wz, vx,vy,vz]`). Confirm frame convention during
  implementation; transform inertia to the velocity frame as needed.
- `I_b`: body inertia (`model.body_inertia`). If the rotational term is
  awkward to get right quickly, ship translational KE alone (still a clean,
  unambiguous instability signal) and note it in the caption.

Rationale: as objects settle, a good solver drives `KE -> 0`. An unstable
fixed-10ms step **injects energy** -> `KE` spikes and never settles. `KE` is
trivially defined, has a clean zero baseline at rest, and "going crazy" reads
as a clear upward spike. (We operationalize "energy drift" as residual kinetic
energy because total mechanical-energy drift is confounded by *physical*
contact dissipation in a settling scene.)

## Figure

- **Scene:** `contact_objects` (the contact-heavy scene, matches Fig 1).
- **N = 64** worlds (randomized ICs via `build_model_randomized(64)`).
- **Solvers:** `mujoco_adaptive_1e-3` (per-world adaptive) vs
  `mujoco_fixed_10ms` (global fixed). Both from the scene's `solver_factories`.
- **Axes:** x = simulation time [s] (linear — it is a time series, per
  CLAUDE.md plotting rule); y = total kinetic energy [J] (log).
- **Per solver:** median line over the 64 worlds + min/max band (shaded).
  The fixed-10ms band fans out / spikes (worst worlds explode); the adaptive
  band stays low and decays.
- **Duration:** enough sim time for the adaptive case to clearly settle
  (~1.5-2.0 s; tune to the scene). Sampled once per outer step.
- Style: reuse `scripts/bench/plotting.py` `STYLES` colors for the two kinds
  and `save_fig`.

## Implementation

New benchmark `scripts/bench/benchmarks/energy_trace.py`, mirroring the
structure of `error_trace.py`:

- `kinetic_energy_per_world(model, state) -> np.ndarray[N]` helper.
- `trace_kind(scene, kind, n, steps)`: build model + solver via the scene's
  factory, step `steps` outer steps, record per-world KE (max/median/min over
  worlds) and `sim_time` each step. Works for BOTH adaptive and fixed kinds
  (unlike `error_trace.py`, which needs `last_error`).
- `plot_traces(...)`: single panel, median + min/max band per kind, log y,
  linear x.
- CLI: `--scene contact_objects --kinds mujoco_adaptive_1e-3 mujoco_fixed_10ms
  --n 64 --steps <K> --out-dir ...`.
- Output: JSON + PNG under `scripts/bench/results/<commit>/`, plus a poster
  copy to `scripts/figures/energy_drift.png`.

## Risks / watch items

- **Fixed-10ms may NaN/explode** mid-run on the stiffest worlds. That is the
  point, but guard the KE computation against NaN/inf (clip to a large finite
  value for plotting; note any worlds that diverged).
- **Body velocity frame**: rotational KE needs `I` and `w` in consistent
  frames. If uncertain, ship translational KE and label accordingly.
- **Scene must actually settle** under adaptive within the sim window; if not,
  extend `--steps`.
- `contact_objects` solver factories use `use_mujoco_contacts=True` /
  per-world budgets already set in `scripts/scenes/contact_objects.py`; reuse
  them, do not re-tune.

## Out of scope

- Figs 1 (scaling, contact_objects), 2 & 3 (fork fixed vs adaptive stills) are
  tracked separately and produced in parallel.
- The original tolerance-proxy "work-precision" plot is dropped in favor of
  this energy-drift trace.
