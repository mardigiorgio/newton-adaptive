#!/usr/bin/env bash
# Apply the adaptive-Newton delta onto an Isaac Lab clone.
#
# The "modified Isaac Lab" = stock upstream IsaacLab @ the pinned commit + this small delta:
#   1. isaaclab_newton_adaptive.patch  -- SolverMuJoCoAdaptive integration in NewtonMJWarpManager
#      (mjwarp_manager.py + mjwarp_manager_cfg.py) and the GUI-toggle dependency line in
#      apps/isaaclab.python.kit.
#   2. newton_adaptive_ui/            -- the auto-loading GUI toggle Kit extension, copied into
#      <IsaacLab>/source/ (a sibling of isaaclab_newton, already on Kit's extension search path).
#
# Idempotent: skips the patch if already applied; re-copies the extension.
#
# Usage:
#   scripts/rl/adaptive_isaac/apply_isaaclab_delta.sh [ISAACLAB_DIR]
# Defaults ISAACLAB_DIR to $ISAACLAB or ~/Documents/code/IsaacLab.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="${1:-${ISAACLAB:-$HOME/Documents/code/IsaacLab}}"
PATCH="$HERE/isaaclab_newton_adaptive.patch"
EXT_SRC="$HERE/newton_adaptive_ui"

[ -d "$ISAACLAB_DIR/source" ] || { echo "error: $ISAACLAB_DIR is not an Isaac Lab checkout (no source/)" >&2; exit 1; }
[ -f "$PATCH" ] || { echo "error: missing $PATCH" >&2; exit 1; }

cd "$ISAACLAB_DIR"

echo "==> applying isaaclab_newton_adaptive.patch"
if git apply --reverse --check "$PATCH" >/dev/null 2>&1; then
  echo "    already applied (reverse-check passed) -- skipping."
elif git apply --check "$PATCH" >/dev/null 2>&1; then
  git apply "$PATCH"
  echo "    applied."
else
  echo "    WARNING: patch does not apply cleanly to this IsaacLab commit." >&2
  echo "    This delta was generated against IsaacLab develop @ 546551f5ba." >&2
  echo "    Re-pin to that commit, or 3-way merge:  git apply --3way \"$PATCH\"" >&2
  exit 1
fi

echo "==> installing newton_adaptive_ui extension into source/"
rm -rf "$ISAACLAB_DIR/source/newton_adaptive_ui"
mkdir -p "$ISAACLAB_DIR/source/newton_adaptive_ui"
cp -r "$EXT_SRC/config" "$EXT_SRC/newton_adaptive_ui" "$ISAACLAB_DIR/source/newton_adaptive_ui/"
echo "    source/newton_adaptive_ui/ in place."

echo "==> done. Adaptive backend + GUI toggle are wired into $ISAACLAB_DIR."
