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

## Open decisions

- **Binary vs pip Isaac Sim — resolved: pip is enough.** The `isaacsim==6.0.0.1` **wheel** is mandatory
  either way (it carries the `isaacsim.physics.newton` extension you patch — pure Python, editable in place
  inside the installed extscache tree, so no Isaac Sim *source build* / fork is needed). The binary install
  is only for the Omniverse RTX viewer.
- **Thread A coexistence.** The Trossen testbed is built on Isaac Lab **2.3.2** (manager API + PhysX).
  Running it on `develop` is a real port. Pin the chosen versions in `../trossen/IMPL_GROUND_TRUTH.md`.
