# Trossen Stationary AI — cube pickup (teacher/student, Isaac Lab)

Privileged-state PPO teacher → depth-camera CNN student (distillation) that makes one
arm of the Trossen Stationary AI bimanual rig pick a cube off the table.

- Design: `docs/superpowers/specs/2026-06-16-trossen-cube-pickup-teacher-student-design.md`
- Plan: `docs/superpowers/plans/2026-06-16-trossen-cube-pickup-teacher-student.md`

## Environment

Isaac Sim does not run natively on this Fedora host; everything runs in the Ubuntu-22.04
podman container `isaaclab` (see `docker/Containerfile` and the `reference_isaaclab_container`
memory). The repo is mounted at `/repo` inside the container; the package venv is `/opt/venv`.

```bash
# install this package into the container venv (editable)
podman exec isaaclab bash -lc "/opt/venv/bin/pip install -e /repo/scripts/rl/trossen"

# run any task script in the container
scripts/rl/trossen/docker/run.sh trossen_cube/... --headless        # python in /repo
```

Always launch `AppLauncher` before importing `isaaclab.*` / `trossen_cube`.
