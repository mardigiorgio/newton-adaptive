#!/usr/bin/env bash
#
# apply_overlay.sh -- install the adaptive Newton-backend overlay into the
# installed Isaac Sim wheel (isaacsim.physics.newton), idempotently.
#
# Why a file swap (not --ext-folder): under IsaacLab's launcher the Kit
# --ext-folder override did NOT take precedence -- the wheel copy of
# newton_stage.py was always the one imported. The working apply path is to
# overwrite the file *inside the installed wheel*. uv installs that file as a
# hardlink into its global cache, so we `rm` it first (breaking the hardlink)
# and then `cp` the overlay in, which prevents corrupting the shared cache.
#
# Idempotent: re-running re-applies the overlay (e.g. after `pip/uv install`
# blows the file away). On first run it captures the pristine wheel file to
# overlay/newton_stage.pristine.bak so --restore can put it back.
#
# Usage:
#   ./apply_overlay.sh            # apply the overlay (default)
#   ./apply_overlay.sh --restore  # restore the pristine wheel file
#   ./apply_overlay.sh --help
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY_DIR="${SCRIPT_DIR}/overlay"
OVERLAY_SRC="${OVERLAY_DIR}/newton_stage.py"
PRISTINE_BAK="${OVERLAY_DIR}/newton_stage.pristine.bak"

# Seed backup left by the original manual swap-test (optional source of truth
# for the pristine file if it still exists).
WHEEL_SEED_BAK="/tmp/newton_stage.wheel.bak"

# Relative location of the target file inside any venv's site-packages.
REL_WHEEL_PATH="isaacsim/exts/isaacsim.physics.newton/isaacsim/physics/newton/impl/newton_stage.py"

# Sentinel that proves the overlay (not the pristine wheel file) is in place.
SENTINEL="ADAPTIVE OVERLAY"

# Fallback venv if the script is run outside an activated environment.
DEFAULT_VENV="${HOME}/Documents/code/IsaacLab/env_isaaclab"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[apply_overlay] $*"; }

# ---------------------------------------------------------------------------
# Resolve the active venv and the wheel file inside it
# ---------------------------------------------------------------------------
resolve_venv() {
    local venv=""
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        venv="${VIRTUAL_ENV}"
    elif [[ -d "${DEFAULT_VENV}" ]]; then
        venv="${DEFAULT_VENV}"
    else
        die "No venv found. Activate the IsaacLab venv or set VIRTUAL_ENV. Tried: \$VIRTUAL_ENV and ${DEFAULT_VENV}"
    fi
    [[ -d "${venv}" ]] || die "venv directory does not exist: ${venv}"
    echo "${venv}"
}

resolve_wheel_file() {
    local venv="$1"
    # site-packages lives under lib/python3.*/site-packages -- glob the minor version.
    local hit=""
    shopt -s nullglob
    for sp in "${venv}"/lib/python3.*/site-packages; do
        if [[ -f "${sp}/${REL_WHEEL_PATH}" ]]; then
            hit="${sp}/${REL_WHEEL_PATH}"
            break
        fi
    done
    shopt -u nullglob
    [[ -n "${hit}" ]] || die "Could not find ${REL_WHEEL_PATH} under ${venv}/lib/python3.*/site-packages (is isaacsim.physics.newton installed?)"
    echo "${hit}"
}

file_has_overlay() {
    grep -q "${SENTINEL}" "$1" 2>/dev/null
}

# Hardlink-safe overwrite: rm to break the uv-cache hardlink, then copy.
swap_in() {
    local src="$1" dst="$2"
    rm -f "${dst}"
    cp "${src}" "${dst}"
}

print_restore_help() {
    cat <<EOF

To restore the pristine wheel file:
    ${BASH_SOURCE[0]} --restore

Pristine backup is kept at:
    ${PRISTINE_BAK}
EOF
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
do_apply() {
    local venv wheel
    venv="$(resolve_venv)"
    wheel="$(resolve_wheel_file "${venv}")"
    info "venv:    ${venv}"
    info "target:  ${wheel}"
    info "overlay: ${OVERLAY_SRC}"

    [[ -f "${OVERLAY_SRC}" ]] || die "Overlay source missing: ${OVERLAY_SRC}"
    file_has_overlay "${OVERLAY_SRC}" || die "Overlay source does not contain the '${SENTINEL}' sentinel -- wrong/corrupt file: ${OVERLAY_SRC}"

    # First-run pristine capture. Source order: existing pristine bak (no-op),
    # the /tmp seed bak, else the current wheel file -- but only if the current
    # wheel file is itself pristine (does not already contain the overlay).
    if [[ ! -f "${PRISTINE_BAK}" ]]; then
        if [[ -f "${WHEEL_SEED_BAK}" ]] && ! file_has_overlay "${WHEEL_SEED_BAK}"; then
            cp "${WHEEL_SEED_BAK}" "${PRISTINE_BAK}"
            info "Captured pristine backup from ${WHEEL_SEED_BAK} -> ${PRISTINE_BAK}"
        elif ! file_has_overlay "${wheel}"; then
            cp "${wheel}" "${PRISTINE_BAK}"
            info "Captured pristine backup from current wheel file -> ${PRISTINE_BAK}"
        else
            die "No pristine source available: ${PRISTINE_BAK} missing, ${WHEEL_SEED_BAK} missing/overlaid, and the installed wheel file already contains the overlay. Reinstall isaacsim.physics.newton to recover a clean file, then re-run."
        fi
    else
        info "Pristine backup already present: ${PRISTINE_BAK}"
    fi

    swap_in "${OVERLAY_SRC}" "${wheel}"

    # Fail-fast verification: the wheel file must now contain the sentinel.
    file_has_overlay "${wheel}" || die "Verification failed: '${SENTINEL}' not found in ${wheel} after copy."

    info "OK -- overlay applied. The wheel file now contains the '${SENTINEL}' sentinel."
    print_restore_help
}

do_restore() {
    local venv wheel
    venv="$(resolve_venv)"
    wheel="$(resolve_wheel_file "${venv}")"
    info "venv:   ${venv}"
    info "target: ${wheel}"

    # Prefer the captured pristine backup; fall back to the /tmp seed.
    local src=""
    if [[ -f "${PRISTINE_BAK}" ]]; then
        src="${PRISTINE_BAK}"
    elif [[ -f "${WHEEL_SEED_BAK}" ]] && ! file_has_overlay "${WHEEL_SEED_BAK}"; then
        src="${WHEEL_SEED_BAK}"
    else
        die "No pristine source to restore from (${PRISTINE_BAK} and a clean ${WHEEL_SEED_BAK} both missing). Reinstall isaacsim.physics.newton to recover a clean file."
    fi
    file_has_overlay "${src}" && die "Refusing to restore: backup ${src} itself contains the overlay sentinel (not pristine)."

    swap_in "${src}" "${wheel}"

    file_has_overlay "${wheel}" && die "Verification failed: overlay sentinel still present in ${wheel} after restore."
    info "OK -- restored pristine wheel file from ${src}."
}

usage() {
    cat <<EOF
apply_overlay.sh -- install/restore the adaptive Newton-backend overlay.

Usage:
    apply_overlay.sh             Apply the overlay (default).
    apply_overlay.sh --restore   Restore the pristine wheel file.
    apply_overlay.sh --help      Show this help.

Resolves the wheel file from the active venv (\$VIRTUAL_ENV, else
${DEFAULT_VENV}). Idempotent and re-appliable after any reinstall.
EOF
}

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
case "${1:-}" in
    --restore) do_restore ;;
    -h|--help) usage ;;
    "")        do_apply ;;
    *)         usage; die "Unknown argument: $1" ;;
esac
