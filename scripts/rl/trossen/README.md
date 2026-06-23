# Trossen Stationary AI — cube pickup (teacher/student, Isaac Lab)

Privileged-state PPO teacher → depth-camera CNN student (distillation) that makes one
arm of the Trossen Stationary AI bimanual rig pick a cube off the table.

The design and plan for this workstream are summarized in `ROADMAP.md`.

See `../README.md` for the top-level Thread A / Thread B signpost.

## Environment

Runs natively on Ubuntu against a binary Isaac Sim install (under `~/Documents/code/`,
beside this repo and the IsaacLab clone) — no container. Scripts go through the
`run_native.sh` launcher, which calls `${ISAACLAB:-~/Documents/code/IsaacLab}/isaaclab.sh -p`.
Filesystem roots (data, assets, logs, artifacts) come from `trossen_cube/paths.py`
(default data root `~/Documents/code/isaac-data`, overridable per-root via env var).

```bash
# install this package into Isaac's bundled python (editable)
~/Documents/code/IsaacLab/isaaclab.sh -p -m pip install -e scripts/rl/trossen

# run any task script natively
scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless
```

One-time bring-up: symlink the binary Isaac Sim as `IsaacLab/_isaac_sim`, then run
`isaaclab.sh --install`. The deprecated container path is documented in `legacy/CONTAINER.md`.

Always launch `AppLauncher` before importing `isaaclab.*` / `trossen_cube`.
