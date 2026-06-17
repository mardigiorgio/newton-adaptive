#!/usr/bin/env bash
# Run a Python script (with args) inside the persistent Isaac Lab container.
#
# The container `isaaclab` mounts this repo at /repo and the Isaac Lab / trossen
# clones at /isaac, with the venv at /opt/venv and GPU passthrough configured.
# Isaac Sim does not run natively on this Fedora host; always use this wrapper.
#
# Usage:
#   scripts/rl/trossen/docker/run.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 64
#   scripts/rl/trossen/docker/run.sh -c "print('hi')"          # passes through to python
#
# For non-python commands (pip, ls, ...), use the container directly:
#   podman exec isaaclab bash -lc "<cmd>"
set -euo pipefail

CONTAINER=isaaclab
if ! podman ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "error: container '$CONTAINER' is not running." >&2
  echo "start it with:" >&2
  echo "  podman start $CONTAINER   # if it exists" >&2
  echo "  # or recreate (see scripts/rl/trossen/docker/Containerfile + reference_isaaclab_container memory)" >&2
  exit 1
fi

# Paths are relative to the repo root (/repo inside the container).
exec podman exec "$CONTAINER" bash -lc "cd /repo && OMNI_KIT_ACCEPT_EULA=YES /opt/venv/bin/python $*"
