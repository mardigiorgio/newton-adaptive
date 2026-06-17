# Resume prompt — paste this into a fresh Claude Code session

```
Resume the Trossen Stationary AI cube-pickup teacher/student RL workstream (Isaac Lab in a
podman container on Fedora). I just upgraded Fedora / fixed the NVIDIA driver that was crashing
the machine under GPU load.

START by reading: scripts/rl/trossen/HANDOFF.md (full state + commands), and recall the memories
project_trossen_teacher_student and reference_isaaclab_container. Plan/spec/ground-truth live in
docs/superpowers/ and scripts/rl/trossen/IMPL_GROUND_TRUTH.md.

Where we are: Phases 0-3 done and validated. The PPO teacher is trained to iter 650/1500;
checkpoints are at ~/isaac-rl/logs/trossen/stationary_ai_lift_teacher/model_*.pt. Everything runs
in the podman container `isaaclab` (image localhost/isaac-rl:base) with mandatory run flags
--device nvidia.com/gpu=all --security-opt=label=disable. rsl-rl-lib is 5.4.1.

Do, in order:
1. Restart the container: `podman start isaaclab`. If the driver version changed in the upgrade,
   regenerate the GPU spec first (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) and
   verify GPU passthrough with a CUDA container. If the container was removed, recreate it from
   scripts/rl/trossen/docker/Containerfile and `pip install -e /repo/scripts/rl/trossen`.
2. Do a quick GPU-stability sanity check (short headless run) before committing to a long one.
3. Resume teacher training 650 -> 1500:
   train_teacher.py --headless --num_envs 2048 --max_iterations 850
     --resume_from /isaac/logs/trossen/stationary_ai_lift_teacher/model_650.pt
4. When it finishes (train_done.json marker), render a teacher timelapse, re-encode to H.264 in
   the container (Fedora ffmpeg lacks libx264), and I'll view it OFF this machine.
5. Then start Phase 4: depth camera + CNN vision-student distillation per the plan.

Gotchas (all in HANDOFF.md): launch AppLauncher before importing isaaclab.*; call
handle_deprecated_rsl_rl_cfg after loading the rsl_rl cfg; use torch.no_grad (not inference_mode)
for multi-checkpoint rollouts; detect success via marker files (Kit eats stdout). Known TODO:
refine ee_frame from the wrist (follower_left_link_6) to the gripper tool-center before trusting
grasp success. Commit when I ask; no Co-Authored-By lines.
```
