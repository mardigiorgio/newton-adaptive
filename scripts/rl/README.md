# RL + adaptive-solver workstream

**North star:** one pipeline where an adaptive integrator — adaptive timestepping today, the goal being
**true CENIC = adaptive + convex integration (ICF/SAP)** for RL manipulation — is the physics, top to
bottom.

```
  Isaac Lab    RL managers · teacher/student · data-gen
     |         NewtonMJWarpManager selects the solver
  Isaac Sim    USD scene · RTX render · live viewer · sensors  ← native BINARY install (editable source)
     |         Newton is the physics backend  ← the integration hop (Thread B)
  Newton       Model/State + SolverMuJoCoAdaptive (adaptive dt)   ← lives in THIS repo (newton/)
```

The native Ubuntu + **binary, editable-source** Isaac Sim install exists for exactly this: to plug
`SolverMuJoCoAdaptive` into Isaac Sim's **Newton backend** and drive it from Isaac Lab. The adaptive
solver stays a *registered Newton solver* (a selection, not an engine fork) — the integration is just two
seams in `NewtonMJWarpManager`: `_build_solver` → construct `SolverMuJoCoAdaptive`, `_run_solver_substeps`
→ call `step_dt(outer_dt)`. Everything above the outer-dt boundary (sensors, render, RL, viewer) is
untouched.

> **Today vs. the goal.** What this repo ships is the **adaptive solver**: adaptive step-doubling
> timestepping over MuJoCo-Warp (`SolverMuJoCoAdaptive`). **True CENIC** — adaptive timestepping plus
> convex ICF/SAP contact, the PI's actual method — is the research goal and is **not yet built**.
> "Adaptive" below always means the current solver; "CENIC" is reserved for that future ICF+adaptive target.

## Layout (mapped to the pipeline)

- **Newton / adaptive solver** — [`newton/`](../../newton) (this repo). The adaptive solver already
  exists; the fast hypothesis loop (config → run → log-log plot, seconds, no Kit) lives here. (Convex
  ICF/SAP contact — the rest of true CENIC — is future work.)
- **`adaptive_expts/`** — standalone-Newton adaptive-vs-fixed work-precision evidence (no Isaac
  needed). `v1_work_precision.py` is the fast hypothesis loop.
- **`archive/adaptive_isaac/`** — SUPERSEDED scaffold for the Newton → Sim → Lab glue; the
  integration was implemented natively in the IsaacLab fork instead (see the archive README banner).
- **`anymal_study/`** — a **completed** reference study. Its written conclusions were deleted as
  unverifiable; the scripts remain. Re-measure before relying on anything it appeared to show.

(The Trossen teacher/student workstream was removed 2026-07-02: it predated and did not use the
icra2027 (then isaac-rubato) study setup.)

## Running

**Standalone Newton evidence** — no Isaac:
```bash
uv run --extra rl --extra examples --extra importers -m scripts.rl.adaptive_expts.v1_work_precision
```

**The Isaac integration** is live in the IsaacLab fork (`NewtonMJWarpManager`, `--solver` CLI flag,
`physics=newton_mjwarp` preset); the old `adaptive_isaac/` scaffold is archived.
