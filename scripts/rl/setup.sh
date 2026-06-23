#!/usr/bin/env bash
# ============================================================================
# Adaptive-Isaac platform installer  (native Ubuntu, no container)
# ----------------------------------------------------------------------------
# Reconstructs the full working environment on a fresh machine:
#   Isaac Sim 6.0.0.1 (pip wheel) + Isaac Lab develop@PIN + the adaptive-Newton
#   fork (this repo) + the Isaac Lab adaptive delta + the Trossen cube-lift task.
#
# This repo (newton-cenic) is the monorepo. Isaac Lab is reconstructed from
# upstream at a pinned commit and the small adaptive delta is applied on top --
# we do NOT vendor Isaac Lab or Isaac Sim (31 GB; Isaac Sim is a wheel).
#
# Usage (from a fresh clone of this repo):
#   git clone https://github.com/mardigiorgio/newton-cenic.git ~/Documents/code/newton-adaptive
#   cd ~/Documents/code/newton-adaptive
#   git checkout mardigiorgio/trossen-cube-teacher-student
#   bash scripts/rl/setup.sh
#
# Override any path/pin via env vars (see DEFAULTS). Re-runnable (idempotent-ish):
#   skips clone/venv if present; re-applies the delta safely.
# ============================================================================
set -euo pipefail

# ---- DEFAULTS (override via environment) -----------------------------------
NEWTON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # this repo root
CODE_DIR="${CODE_DIR:-$(dirname "$NEWTON_DIR")}"                   # parent of this repo
ISAACLAB_DIR="${ISAACLAB_DIR:-$CODE_DIR/IsaacLab}"
DATA_DIR="${DATA_DIR:-$CODE_DIR/isaac-data}"

ISAACLAB_REPO="${ISAACLAB_REPO:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-546551f5ba8e8e4fbbcbf589b63c6f40b7cacb3f}"  # develop, verified
ISAACSIM_PIN="${ISAACSIM_PIN:-6.0.0.1}"      # ==6.0.0 resolves to an OLDER 6.0.0.0 -- pin .1
TORCH_PIN="${TORCH_PIN:-2.10.0}"
TORCHVISION_PIN="${TORCHVISION_PIN:-0.25.0}"
PYTHON_VER="${PYTHON_VER:-3.12}"
VENV="${VENV:-env_isaaclab}"

export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.nvidia.com}"
export PIP_FIND_LINKS="${PIP_FIND_LINKS:-https://py.mujoco.org/}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$*" >&2; exit 1; }

# ---- 0. preflight ----------------------------------------------------------
say "preflight"
command -v git >/dev/null || die "git not found"
command -v uv  >/dev/null || die "uv not found -- install: curl -LsSf https://astral.sh/uv/install.sh | sh"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  [ "${drv:-0}" -ge 580 ] 2>/dev/null || warn "driver < 580.95.05 -- Isaac Sim 6.0 / cu128 wants >= 580 (RTX 5090 needs a Blackwell-capable driver)."
else
  warn "nvidia-smi not found -- a CUDA GPU + driver >= 580 is required to actually run."
fi
avail_gb="$(df -BG --output=avail "$CODE_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
[ "${avail_gb:-0}" -ge 60 ] 2>/dev/null || warn "only ${avail_gb}G free under $CODE_DIR; the Isaac Sim wheel + extscache needs ~40-60 GB."
echo "newton fork : $NEWTON_DIR"
echo "isaac lab   : $ISAACLAB_DIR  (clone @ ${ISAACLAB_COMMIT:0:10})"
echo "data dir    : $DATA_DIR"

# ---- 1. Isaac Lab @ pinned commit ------------------------------------------
say "Isaac Lab clone @ pin"
if [ ! -d "$ISAACLAB_DIR/.git" ]; then
  git clone "$ISAACLAB_REPO" "$ISAACLAB_DIR"
fi
git -C "$ISAACLAB_DIR" fetch --depth 1 origin "$ISAACLAB_COMMIT" 2>/dev/null \
  || git -C "$ISAACLAB_DIR" fetch origin
git -C "$ISAACLAB_DIR" checkout --detach "$ISAACLAB_COMMIT"
test -f "$ISAACLAB_DIR/source/isaaclab_newton/pyproject.toml" || die "isaaclab_newton extension missing -- wrong commit?"

# ---- 2. venv (py3.12) ------------------------------------------------------
say "venv ($VENV, python $PYTHON_VER)"
cd "$ISAACLAB_DIR"
[ -d "$VENV" ] || uv venv --python "$PYTHON_VER" --seed "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python --version

# ---- 3. Isaac Sim wheel (MANDATORY, before isaaclab.sh -i) ------------------
say "Isaac Sim $ISAACSIM_PIN wheel (full Kit + RTX viewer; no separate binary needed)"
uv pip install "isaacsim[all,extscache]==${ISAACSIM_PIN}" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

# ---- 4. torch pin (cu128) --------------------------------------------------
say "torch ${TORCH_PIN}+cu128 / torchvision ${TORCHVISION_PIN}+cu128"
uv pip install -U "torch==${TORCH_PIN}" "torchvision==${TORCHVISION_PIN}" \
  --index-url https://download.pytorch.org/whl/cu128

# ---- 5. Isaac Lab editable + its git-pinned Newton + rsl-rl -----------------
say "isaaclab.sh -i  (Isaac Lab extensions editable, pinned Newton, rsl-rl)"
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -i

# ---- 6. OVERRIDE: swap the git-pinned Newton for THIS fork (order matters) --
say "override Newton with the adaptive fork (editable, dist-name 'newton' replaces the pin)"
uv pip install -e "${NEWTON_DIR}[sim]" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow

# ---- 7. apply the Isaac Lab adaptive delta (solver + GUI toggle) ------------
say "apply adaptive delta to Isaac Lab (patch + newton_adaptive_ui extension)"
ISAACLAB="$ISAACLAB_DIR" bash "$NEWTON_DIR/scripts/rl/adaptive_isaac/apply_isaaclab_delta.sh" "$ISAACLAB_DIR"

# ---- 8. Trossen task editable ----------------------------------------------
say "trossen_cube task (editable)"
uv pip install -e "$NEWTON_DIR/scripts/rl/trossen" --no-deps

# ---- 9. data dir + Trossen rig assets --------------------------------------
say "data dir + Trossen rig assets"
mkdir -p "$DATA_DIR/artifacts"
TROSSEN_ASSETS_REPO="${TROSSEN_ASSETS_REPO:-https://github.com/TrossenRobotics/trossen_ai_isaac.git}"
if [ ! -d "$DATA_DIR/trossen_ai_isaac/.git" ]; then
  git clone "$TROSSEN_ASSETS_REPO" "$DATA_DIR/trossen_ai_isaac"
else
  echo "    trossen_ai_isaac already present."
fi
RIG="$DATA_DIR/trossen_ai_isaac/assets/robots/stationary_ai/stationary_ai.usd"
test -f "$RIG" && echo "    rig USD: $RIG" || warn "rig USD not found at $RIG -- check the asset clone."
echo "    ($DATA_DIR is the data root; TROSSEN_* env vars / paths.py override it."
echo "     The optional 'no-rails' variant USD is user-generated -- see INSTALL.md if you need it.)"

# ---- 10. verify ------------------------------------------------------------
say "verify"
python - <<'PY'
import newton, newton.solvers as s, isaaclab, isaaclab_newton
print("newton     :", newton.__file__)
assert "newton-adaptive" in newton.__file__ or "newton-cenic" in newton.__file__, \
    "FORK NOT ACTIVE -- newton resolves to the pin, re-run step 6"
print("newton ver :", newton.__version__)
print("adaptive   :", hasattr(s, "SolverMuJoCoAdaptive"), "| fixed:", hasattr(s, "SolverMuJoCo"))
assert hasattr(s, "SolverMuJoCoAdaptive"), "SolverMuJoCoAdaptive missing from the fork"
print("isaaclab   : ok")
PY
test -f "$ISAACLAB_DIR/source/newton_adaptive_ui/config/extension.toml" && echo "GUI toggle  : installed"
git -C "$ISAACLAB_DIR" apply --reverse --check \
  "$NEWTON_DIR/scripts/rl/adaptive_isaac/isaaclab_newton_adaptive.patch" >/dev/null 2>&1 \
  && echo "delta       : applied"

cat <<EOF

============================================================================
  Install complete.

  Smoke test (adaptive backend, headless, ~1-2 min on a 5090):
    cd "$NEWTON_DIR"
    rm -f /tmp/newton_adaptive.log
    NEWTON_ADAPTIVE=1 NEWTON_ADAPTIVE_LOG_EVERY=10 \\
      scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py \\
      --headless --num_envs 16 --max_iterations 2
    tail /tmp/newton_adaptive.log     # spread>0 + substeps>>3 == adaptive is working

  GUI (interactive viewer + the auto-loading "Newton Integrator" toggle window):
    scripts/rl/trossen/run_native.sh scripts/rl/trossen/train_teacher.py --num_envs 16

  Full install + asset notes:  scripts/rl/INSTALL.md
============================================================================
EOF
