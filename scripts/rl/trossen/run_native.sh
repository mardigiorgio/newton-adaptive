#!/usr/bin/env bash
# Native launcher for Trossen testbed scripts -- replaces the old docker/run.sh (podman).
#
# Runs a Python script (with args) through the binary Isaac Sim interpreter that
# `isaaclab.sh -p` resolves (IsaacLab/_isaac_sim/python.sh, once the binary Isaac Sim
# install is symlinked into IsaacLab). Override the IsaacLab location with $ISAACLAB.
#
# Usage:
#   scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --headless --num_envs 2048
#   scripts/rl/trossen/run_native.sh scripts/rl/trossen/trossen_cube/tests/test_env_smoke.py
#
# For non-python commands (pip, etc.) call isaaclab.sh directly:
#   ~/Documents/code/IsaacLab/isaaclab.sh -p -m pip install -e scripts/rl/trossen
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ISAACLAB_DIR="${ISAACLAB:-$HOME/Documents/code/IsaacLab}"

if [ ! -x "$ISAACLAB_DIR/isaaclab.sh" ]; then
  echo "error: $ISAACLAB_DIR/isaaclab.sh not found." >&2
  echo "  set \$ISAACLAB to your IsaacLab clone, and symlink the binary Isaac Sim as" >&2
  echo "  \$ISAACLAB/_isaac_sim, then run: \$ISAACLAB/isaaclab.sh --install" >&2
  exit 1
fi

cd "$REPO"
exec env OMNI_KIT_ACCEPT_EULA=YES "$ISAACLAB_DIR/isaaclab.sh" -p "$@"
