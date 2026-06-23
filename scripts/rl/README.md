# RL + adaptive-solver workstream

**North star:** one pipeline where an adaptive integrator — adaptive timestepping today, the goal being
**true CENIC = adaptive + convex integration (ICF/SAP)** for RL manipulation — is the physics, top to
bottom.

```
  Isaac Lab    RL managers · teacher/student · data-gen       Thread A scene runs here
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
- **Thread A — `trossen/`** — the Trossen Stationary-AI cube-lift teacher/student scene in Isaac Lab.
  The *vehicle*: a real contact-rich manipulation task to run the integrator through, plus the live
  viewer / data-gen surface. See [`trossen/README.md`](trossen/README.md) · [`trossen/ROADMAP.md`](trossen/ROADMAP.md).
- **Thread B — the adaptive-solver → Isaac integration (the reason for the native install):**
  - **`adaptive_expts/`** — **B0**: standalone-Newton adaptive-vs-fixed work-precision evidence (no Isaac
    needed). `v1_work_precision.py` is the go/no-go gate before the Isaac spend.
  - **`adaptive_isaac/`** — **B1–B4**: the Newton → Sim → Lab glue — stand up the Isaac Lab Newton backend,
    reconcile the `newton-cenic` fork with Isaac's pinned Newton, subclass `NewtonMJWarpManager` to
    select `SolverMuJoCoAdaptive`, validate in-Isaac. **This is what the editable binary install enables.**
    See [`adaptive_isaac/README.md`](adaptive_isaac/README.md).
- **`anymal_study/`** — a **completed** reference study. Its `STUDY_LOG.md` records the key negative
  result: the sim-to-real *transfer* framing is dead; the surviving value is **data fidelity for
  stiff / non-convex-SDF tunneling contact**. Read before re-pitching Thread B.

Full phase tracker (A0–A8, B0–B5) + the two-seam architecture: [`trossen/ROADMAP.md`](trossen/ROADMAP.md).

## Running

**Thread A (Isaac Lab scene)** — native binary Isaac Sim, no container:
```bash
scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 2048
```
Bring-up + path config: [`trossen/README.md`](trossen/README.md). Roots centralized in
`trossen/trossen_cube/paths.py` (default `~/Documents/code/isaac-data`, `TROSSEN_*` overrides).

**Thread B / B0 (standalone Newton evidence)** — no Isaac:
```bash
uv run --extra rl --extra examples --extra importers -m scripts.rl.adaptive_expts.v1_work_precision
```

**Thread B / B1–B4 (the integration)** — runs through the Isaac Lab Newton backend once it's stood up;
see [`adaptive_isaac/README.md`](adaptive_isaac/README.md) (gated on the installed Isaac Lab version exposing
the Newton backend).
