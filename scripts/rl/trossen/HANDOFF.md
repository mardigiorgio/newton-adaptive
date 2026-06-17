# Trossen Stationary AI Cube-Pickup — Session Handoff

**Date:** 2026-06-16 · **Branch:** `mardigiorgio/trossen-cube-teacher-student`
**Paste `RESUME_PROMPT.md` into a fresh session to pick this up.**

> **⏸ PAUSED — owner is focused on HARDWARE right now** (Fedora upgrade + NVIDIA
> driver fix). The RL workstream is intentionally on hold until the GPU is stable.
> Do **not** resume heavy GPU work (training/rendering) until the `libnvidia-eglcore`
> crash (see Blocker below) is confirmed fixed. Everything is committed + on disk and
> waits safely.

## TL;DR
Privileged-state **PPO teacher → depth-camera CNN student (distillation)** that makes one arm
of the Trossen **Stationary AI** bimanual rig pick a cube, built in **Isaac Lab running inside a
podman container** on this Fedora box. **Phases 0–3 are done and validated; the PPO teacher is
trained to iter 650/1500** (checkpoints on disk). Work is paused because the **NVIDIA-580 open
kernel driver crashes the whole machine under sustained GPU load** (`libnvidia-eglcore` hang) —
resume after the Fedora/driver upgrade.

## ⛔ THE BLOCKER (read first)
4 full system crashes this session, all the same: a hang in `libnvidia-eglcore.so.580.159.03`
(the **NVIDIA open kernel module**) under GPU load — both RTX rendering AND headless training
(the last crash fired as the resumed training spun up Isaac Sim, before it wrote a checkpoint).
It is NOT our code and NOT the mp4 file.

**Fix before resuming heavy GPU work:** switch nvidia-open → **proprietary `akmod-nvidia`**
(rpmfusion), or the Fedora upgrade you're doing now may resolve it.
**After the upgrade, if the driver version changed:** regenerate the container GPU spec —
`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` — then re-verify GPU passthrough.

## State persists across the Fedora upgrade
Everything lives in `~/` + rootless podman storage (`~/.local/share/containers`), which a Fedora
`dnf system-upgrade` preserves:
- Image `localhost/isaac-rl:base` (17.7 GB), container **`isaaclab`** (stopped).
- Checkpoints: `~/isaac-rl/logs/trossen/stationary_ai_lift_teacher/model_{0..650}.pt`.
- Isaac Lab + trossen clones: `~/isaac-rl/{IsaacLab,trossen_ai_isaac}`.
- Timelapse: `~/isaac-rl/teacher_timelapse_h264.mp4` (watchable H.264).

## Phase status
- **0 Isaac Lab container** ✅ (see `reference_isaaclab_container` memory + `docker/Containerfile`)
- **1 Greenfield Stationary AI articulation** ✅ validated (16 DOF) — `trossen_cube/assets/stationary_ai.py`
- **2 Teacher manager env** ✅ validated (obs groups `policy`[39] + `privileged`[10], action 7) — `trossen_cube/tasks/cube_lift/cube_lift_env_cfg.py`
- **3 PPO teacher** ✅ pipeline validated; ⏳ **training at 650/1500** — `train_teacher.py`, `trossen_cube/tasks/cube_lift/agents/rsl_rl_ppo_cfg.py`
- **4 Vision student (CNN distillation)** ◻️ NOT STARTED
- **5 Eval** ◻️ not written · **6 Timelapse** ✅ render script works (`render_timelapse.py`)

## Critical environment facts (also in `reference_isaaclab_container` memory)
- Isaac Sim does NOT run native on Fedora; everything runs in the container.
- Run flags are mandatory: **`--device nvidia.com/gpu=all --security-opt=label=disable`**.
- Mounts: repo → `/repo`, `~/isaac-rl` → `/isaac`. venv `/opt/venv`. **rsl-rl-lib 5.4.1** (upgraded
  from the pinned 5.0.1; needed for the `stochastic` model field).
- Any train/eval/render script MUST: launch `AppLauncher` before importing `isaaclab.*`; call
  `handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))` after loading
  the rsl_rl cfg; use `torch.no_grad()` (NOT `inference_mode`) for multi-checkpoint rollouts;
  detect success via a written marker file (Kit swallows stdout); raw-SimulationContext smoke
  tests need `os._exit()` (app.close hangs).
- Fedora's ffmpeg lacks `libx264` → re-encode mp4s **in the container** (`-c:v libx264`); the raw
  OpenCV `mp4v` output is unplayable on Drive/phones. **View videos OFF this machine.**

## Resume commands (after upgrade + driver fix)
```bash
podman start isaaclab
# verify GPU in container:
podman exec isaaclab bash -lc "nvidia-smi -L"   # via --security-opt=label=disable if recreating
# resume teacher 650 -> 1500 (checkpoints every 50; writes train_done.json):
podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python scripts/rl/trossen/train_teacher.py \
  --headless --num_envs 2048 --max_iterations 850 \
  --resume_from /isaac/logs/trossen/stationary_ai_lift_teacher/model_650.pt"
# render a timelapse, then re-encode to H.264 (container ffmpeg), then view off-device:
podman exec isaaclab bash -lc "cd /repo && /opt/venv/bin/python scripts/rl/trossen/render_timelapse.py \
  --headless --enable_cameras --steps 40 --every 2 --out /isaac/teacher_timelapse.mp4"
podman exec isaaclab bash -lc "ffmpeg -y -i /isaac/teacher_timelapse.mp4 -c:v libx264 \
  -pix_fmt yuv420p -movflags +faststart /isaac/teacher_timelapse_h264.mp4"
```
If the container was removed, recreate it (see `docker/Containerfile` + `reference_isaaclab_container`
memory) and re-`pip install -e /repo/scripts/rl/trossen`.

## Known caveats / TODO before trusting grasp success
- **ee_frame is at the WRIST** (`follower_left_link_6`), not the gripper tool-center — the real
  `*_ee_gripper_link` is a USD frame merged out of the articulation. Refine with an offset toward
  `follower_left_gripper_{left,right}` before relying on grasp accuracy. The teacher may reach but
  not cleanly grasp until this is fixed (`cube_lift_env_cfg.py`, `EE_LINK`).
- Gripper actuates only `follower_left_left_carriage_joint` (right carriage assumed USD mimic; the
  `14 != 16` actuator warning is expected/benign).
- Cube spawn pos `[0.3,0,0.055]` + command ranges copied from WXAI; tune to the left-arm reach.

## Next (Phase 4–6) — see the plan
`docs/superpowers/plans/2026-06-16-trossen-cube-pickup-teacher-student.md` and
`scripts/rl/trossen/IMPL_GROUND_TRUTH.md` (pinned Isaac Lab values + corrections).
- **Phase 4:** add a `cam_high` depth `TiledCamera` + `images` obs group (NHWC→permute CHW) +
  `RslRlCNNModelCfg` distillation. obs_groups `{"student":["policy","images"],"teacher":["policy","privileged"]}`.
- **Phase 5:** `eval.py` predicate success (grasped ∧ lift-height ∧ at-rest), teacher vs student.
- **Phase 6:** adapt `render_timelapse.py` for the student.
