# Legacy podman container path (deprecated)

**Status: superseded by the native Ubuntu install (2026-06-21).** Kept as a fallback and as
the authoritative manifest of the exact stack to match.

The project originally ran Isaac Lab inside the `isaaclab` podman container because Isaac Sim
does not run natively on Fedora. We have since moved to a **native Ubuntu** install with a
**binary Isaac Sim** under `~/Documents/code/` and run scripts via
[`../run_native.sh`](../run_native.sh) (which calls `IsaacLab/isaaclab.sh -p`). The container is
no longer required.

## What `docker/` contained
- `docker/Containerfile` — image recipe. Use it as the **version manifest** for the native
  install: Isaac Sim 5.1, torch 2.7.0 / cu128, `OMNI_KIT_ACCEPT_EULA=YES`, `rsl-rl-lib` 5.x.
- `docker/run.sh` — `podman exec`-based launcher (the predecessor of `run_native.sh`). It mounted
  the repo at `/repo` and `~/isaac-rl` at `/isaac`, with the venv at `/opt/venv`. Those paths are
  the reason older docstrings referenced `/isaac` and `/repo`; they are gone in the native layout
  (paths now come from `trossen_cube/paths.py`, default root `~/Documents/code/isaac-data`).

## If you ever need the container again
Rebuild from `docker/Containerfile`, then inside it: `pip install -e /repo/scripts/rl/trossen`.
Run flags were mandatory: `--device nvidia.com/gpu=all --security-opt=label=disable`. After any
NVIDIA driver change, regenerate the CDI spec: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`.
