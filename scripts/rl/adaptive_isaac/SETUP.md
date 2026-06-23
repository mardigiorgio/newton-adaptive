# Ideal environment for the adaptive-solver → Isaac Sim 6.0 integration

Verified against Isaac Sim **6.0.0.1** + Isaac Lab **`develop`** (v3.0.0 beta) source/docs and the local
fork, 2026-06-21. Copy-pasteable. Read the **Reframings** first — they correct earlier assumptions.

> **What's being integrated.** The **adaptive solver** (`SolverMuJoCoAdaptive` — adaptive timestepping over
> MuJoCo-Warp), which exists today. **True CENIC** (adaptive + convex ICF/SAP contact) is a later research
> goal and is not what this setup wires in.

## Reframings (what the source actually says)

- **No Newton version skew.** Isaac Lab `develop` pins Newton by **git SHA** `811968b`, not PyPI "1.0".
  Your fork (`newton-cenic` HEAD `183fa2e4`) = that exact commit **+ 47 adaptive-solver commits, 0 behind**
  (`git merge-base --is-ancestor 811968b HEAD` → true). Both report `newton 1.4.0.dev0`, both depend on
  `mujoco-warp~=3.8.0`. The editable swap works by dist-name `newton`. **Do not rebase the fork.**
- **The Isaac Sim wheel is MANDATORY.** The `isaacsim.physics.newton` extension you must patch ships inside
  the `isaacsim[all,extscache]` wheel, **not** in Isaac Lab, and bare `./isaaclab.sh -i` does **not** pull
  isaacsim. Install it first, pinned `==6.0.0.1` (`==6.0.0` resolves to an older `6.0.0.0`).
- **Two edits, not one** (confirmed at the `v6.0.0` GA tag). The extension drives a fixed substep loop and
  hard-codes solver selection — see the Integration edit map below.

## Prerequisites

- Ubuntu, **python3.12** (host has 3.12.3), **uv** on PATH. NVIDIA driver ≥ 580.95.05 (host 595.x ✓).
- **GPU caveat:** host is RTX 4070 Ti SUPER (16 GB). Isaac Sim 6.0 docs list RTX 4080 as *minimum*; 16 GB
  VRAM meets the floor but the card is one tier below. Risk, not blocker — prefer `--headless` + the Newton
  OpenGL viewer; keep `num_envs` modest.
- **RTX viewer / visualization:** the Step-3 `isaacsim[all,extscache]` wheel **is** the full Isaac Sim (Kit
  + RTX renderer) — it provides the Omniverse RTX viewer from the venv, so a **separate standalone binary
  download is redundant** (and mixing it with the pip venv causes "which Isaac Sim" confusion). Expect the
  RTX viewer to be heavy on the 4070 Ti SUPER; the lighter Newton OpenGL viewer is the day-to-day option.
- Disk: `isaacsim[all,extscache]` + extscache is **tens of GB** — ensure free space before Step 2.
- For any manual `uv pip`: export the indexes the installer sets automatically:
  `export UV_EXTRA_INDEX_URL=https://pypi.nvidia.com PIP_FIND_LINKS=https://py.mujoco.org/` and pass
  `--index-strategy unsafe-best-match`.

## Part 1 — Environment (ordered, copy-pasteable)

```bash
# 0. Isolate Thread B. Fork worktree on a new branch (Thread A keeps newton-cenic untouched).
git -C ~/Documents/code/newton-adaptive worktree add \
    ~/Documents/code/newton-adaptive-adaptive-int -b mardigiorgio/adaptive-isaac-integration 183fa2e4

# 1. Isaac Lab on develop. (~/Documents/code/IsaacLab is a fresh clone on main@2.3.2; nothing live uses
#    main, so switch in place. If you want to KEEP a 2.3.2 reference, use a worktree instead — see Isolation.)
git -C ~/Documents/code/IsaacLab fetch origin develop
git -C ~/Documents/code/IsaacLab checkout develop
test -f ~/Documents/code/IsaacLab/source/isaaclab_newton/pyproject.toml && echo "newton ext present ✓"

# 2. Venv (py3.12) at the IsaacLab root.
cd ~/Documents/code/IsaacLab
uv venv --python 3.12 --seed env_isaaclab
source env_isaaclab/bin/activate

# 3. Isaac Sim 6.0 wheel — MANDATORY (contains isaacsim.physics.newton), exact pin, BEFORE isaaclab.sh -i.
uv pip install "isaacsim[all,extscache]==6.0.0.1" \
    --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

# 4. Torch (install.py hard-pins these on x86_64; pre-seeding avoids churn).
uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# 5. Isaac Lab + the git-pinned Newton + isaaclab_newton.
./isaaclab.sh -i

# 6. THE OVERRIDE — install the fork worktree editable into the SAME venv, AFTER step 5 (order is
#    load-bearing: editable-after-install wins). Dist name "newton" matches the pin, so it swaps in place.
uv pip install -e "$HOME/Documents/code/newton-adaptive-adaptive-int[sim]"
```

Verify the env:
```bash
python -c "import newton, newton.solvers as s; print(newton.__file__); print(newton.__version__); \
print('adaptive', hasattr(s,'SolverMuJoCoAdaptive'), 'MuJoCo', hasattr(s,'SolverMuJoCo'))"
# EXPECT: __file__ under .../newton-cenic-adaptive-int ; 1.4.0.dev0 ; adaptive True MuJoCo True
python -c "import isaaclab, isaaclab_newton; print('isaaclab ok')"
```

**Persist the override** (so a future `./isaaclab.sh -i` doesn't re-clobber it): in
`~/Documents/code/IsaacLab/source/isaaclab_newton/pyproject.toml`, replace the
`newton[sim] @ git+https://github.com/newton-physics/newton.git@811968b...` line with
`newton[sim] @ file:///home/mdigiorgio/Documents/code/newton-adaptive-adaptive-int`. Commit on the local
`develop` branch only — **do not push**. (Or just re-run Step 6 after any reinstall.)

## Part 2 — Integration edit map (where the adaptive-solver work goes)

The extension (`isaacsim.physics.newton`, **pure Python** — editable in place in the installed extscache
tree, no source build) drives stepping like this at `v6.0.0`:
```
NewtonStage.simulate(dt)            # newton_stage.py:496
  for i in range(num_substeps):     #   :535  ← fixed loop, pre-divides dt
      self.solver.step(s0, s1, control, contacts, step_dt/num_substeps)   # :536
  # solver built in _get_solver (:461) — hard-codes 'mujoco'→SolverMuJoCo, no injection seam
  # cfg.use_cuda_graph default True → captures a STATIC substep graph
```
So the adaptive work is three coordinated edits:

- **(A) Fork — primary.** Add an opt-in `adaptive=True` mode to the **stock** `SolverMuJoCo`
  (`newton/_src/solvers/mujoco/solver_mujoco.py`, the class Isaac instantiates) so its existing
  `step(self, state_in, state_out, control, contacts, dt)` runs the step-doubling-to-boundary loop over the
  passed-in `dt`. Lift the reusable core from `solver_mujoco_adaptive.py` — `_run_iteration_body` (436),
  `_run_substep` (414), the Drake controller kernel `_calc_adjusted_step` (78). Keep `SolverMuJoCoAdaptive` +
  `step_dt` for standalone scripts; Isaac never calls them (its 5-arg `step` doesn't line up —
  `SolverMuJoCoAdaptive`'s `step` drops `dt`).
- **(B) Extension patch.** In `NewtonStage.simulate`, collapse the `for i in range(num_substeps)` loop to a
  single `solver.step(…, step_dt)` (or `num_substeps=1`) in adaptive mode, and make `_get_solver` pass the
  adaptive flags into `SolverMuJoCo`. Keep this as a tracked `.patch` + re-apply script — `./isaaclab.sh -i`
  or a wheel update overwrites the extscache tree.
- **(C)** Set `cfg.use_cuda_graph=False` in adaptive mode (the adaptive solver's data-dependent per-world
  substep count is incompatible with a static captured graph). Assert it at startup.

## Part 3 — Verification ladder (cheapest first)

1. **Override took** — the `python -c` checks above (no GPU).
2. **Fork's standalone adaptive path runs** — from the worktree: `uv run -m scripts.bench --only scaling
   --ns 1 4 16 --steps 50 --warmup 20` (exercises `SolverMuJoCoAdaptive.step_dt`).
3. **Runtime interop (PRIMARY UNKNOWN)** — before editing, smoke the *stock* Newton path to surface any
   `1.2.0→1.4.0.dev0` `step()/collide()/contacts` drift:
   `./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct --num_envs 128 --headless`
   (add `physics=newton_mjwarp` if required). Breakpoint in `_get_solver`/`simulate` to confirm the
   constructed solver + the per-frame call.
4. **Adaptive integration gate** — after (A)+(B)+(C): same task; confirm `solver.step` is called **once**
   per frame, substep counts vary, `dt` shrinks in dense contact / grows in free flight, env validity passes.

## Part 4 — Top risks / gotchas

- **Runtime API drift `1.2.0`→`1.4.0.dev0`** (the extension targets 1.2.0; fork is 1.4.0.dev0). Constructor
  compat is *verified*; `step/collide/contacts` shapes are **not** runtime-tested — this is the first thing
  to check (ladder step 3). Note `ls_parallel` already emits a DeprecationWarning at construction.
- **Extension patch is brittle** — overwritten by any reinstall. Keep it as a `.patch` + re-apply script.
- **Two fixed-cadence layers** — besides the inner `num_substeps` loop, `on_update` runs an outer
  `while sim_time < final: simulate(sim_dt)` at `physics_frequency` (USD-overridable). Decide whether
  `step_dt` absorbs the frame dt or the per-tick `sim_dt`; verify `sim_time` advances once per boundary.
- **GPU below documented minimum** — prefer headless + Newton OpenGL viz; modest `num_envs`.

## Isolation from Thread A

- **Fork:** worktree (`newton-cenic-adaptive-int`, branch `adaptive-isaac-integration`) — Thread A's
  `newton-cenic` checkout is never the live package. **Never** `uv pip install -e ~/Documents/code/newton-adaptive`
  (the Thread-A tree) into `env_isaaclab`.
- **Isaac Lab:** Step 1 switches `~/Documents/code/IsaacLab` to `develop` **in place**. Confirmed nothing
  live depends on its `main`@2.3.2 (Thread A historically used the container's IsaacLab, not this clone). If
  you want to preserve a 2.3.2 reference, use a worktree instead:
  `git -C ~/Documents/code/IsaacLab worktree add ../IsaacLab-develop develop` and run Part 1 there.
- **Venv:** `env_isaaclab` is exclusive to Thread B; don't activate it from a Thread-A shell.
