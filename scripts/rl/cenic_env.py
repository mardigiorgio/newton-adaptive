"""CenicLocomotionEnv: rsl_rl-compatible vectorized env around a Newton backend.

One ``backend.advance`` (= one ``step_dt``) per 50 Hz policy tick. The same class
drives the CENIC adaptive solver or stock fixed-step ``SolverMuJoCo`` via the
``BackendSpec``, so swapping the integrator is the only difference between an A and
a B training run.

Hot-path discipline (CLAUDE.md): the only per-tick host transfer is ``step_dt``'s
internal 4-byte boundary flag. No ``.numpy()`` in ``step``/obs/reward.
"""

from __future__ import annotations

import math

import torch  # noqa: TID253
import warp as wp

import newton

from . import anymal
from .anymal import build_anymal_model, compute_base_frame, compute_obs, compute_reward, compute_termination
from .backends import BackendSpec, make_backend
from .config import EnvConfig
from .domain_rand import DomainRandomizer, DRConfig


def _control_target_view(control, num_worlds: int):
    """Return (torch view (N, slots_per_world), free_slots) for the PD target array.

    Handles both the legacy DOF-shaped ``joint_target_pos`` and the newer
    coord-shaped ``joint_target_q``; the actuated 12 joints occupy the trailing
    slots, the free-base slots (6 dof or 7 coord) lead and stay zero.
    """
    arr = getattr(control, "joint_target_q", None)  # coord-shaped, aligned with joint_q
    if arr is None:
        arr = getattr(control, "joint_target_pos", None)  # deprecated DOF-shaped alias
    if arr is None:
        raise AttributeError("Control exposes neither joint_target_q nor joint_target_pos")
    view = wp.to_torch(arr).view(num_worlds, -1)
    free_slots = view.shape[1] - anymal.ACT_DIM
    return view, free_slots


class CenicLocomotionEnv:
    """ANYmal-C velocity-tracking env implementing the rsl_rl v1.x VecEnv contract."""

    def __init__(
        self,
        env_cfg: EnvConfig,
        backend_spec: BackendSpec,
        dr_cfg: DRConfig | None,
        device: str = "cuda",
        headless: bool = True,
    ):
        self.cfg = env_cfg
        self.device = device
        self.num_envs = env_cfg.num_envs
        self.num_obs = anymal.OBS_DIM
        self.num_privileged_obs = None
        self.num_actions = anymal.ACT_DIM
        self.max_episode_length = env_cfg.max_episode_length
        self.control_dt = env_cfg.control_dt

        # Paired-eval RNG: when ``eval_seed`` is set, initial-condition and command
        # draws come from dedicated generators seeded identically across backends, so
        # world w sees byte-identical ICs+commands regardless of how many domain-
        # randomization randoms a reference consumes. Training leaves this None and
        # keeps the global-RNG path unchanged. See project_eval_rng_confound.
        eval_seed = getattr(env_cfg, "eval_seed", None)
        self._ic_gen: torch.Generator | None = None
        self._cmd_gen: torch.Generator | None = None
        if eval_seed is not None:
            self._ic_gen = torch.Generator(device=device).manual_seed(int(eval_seed))
            self._cmd_gen = torch.Generator(device=device).manual_seed(int(eval_seed) + 1)

        wp_device = wp.get_device() if device == "cpu" else device
        with wp.ScopedDevice(wp_device):
            self.model, self.meta = build_anymal_model(self.num_envs, spacing=env_cfg.spacing, device=wp_device)
        self.backend = make_backend(self.model, backend_spec)

        self.s0 = self.model.state()
        self.s1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)

        N = self.num_envs
        self.coords = self.meta.coords_per_world
        self.dofs = self.meta.dofs_per_world
        self._rebind_views()
        self.tgt, self.free_slots = _control_target_view(self.control, N)
        self.tgt[:] = 0.0

        # Default standing pose (all worlds identical at init) for reset ICs.
        self.default_full_q = self.jq[0].clone()  # (coords,)
        self.default_q = self.default_full_q[7:].clone()  # (12,) actuated, Newton order

        # Index/const tensors.
        self.lab_idx = torch.tensor(anymal.LAB_TO_MUJOCO, device=device)
        self.mujoco_idx = torch.tensor(anymal.MUJOCO_TO_LAB, device=device)
        self.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=device, dtype=torch.float32)

        # Episode buffers.
        self.episode_length_buf = torch.zeros(N, dtype=torch.long, device=device)
        self.last_action = torch.zeros(N, anymal.ACT_DIM, device=device)
        self.last_qvel = torch.zeros(N, anymal.ACT_DIM, device=device)
        self.obs_buf = torch.zeros(N, anymal.OBS_DIM, device=device)
        self.extras: dict = {}
        self._step_count = 0

        self.command = anymal.VelocityCommand(
            N, device, vx=env_cfg.command_vx, vy=env_cfg.command_vy, wz=env_cfg.command_wz,
            generator=self._cmd_gen,
        )
        self.dr = DomainRandomizer(dr_cfg or DRConfig(), self.meta, self.model, device)

        # Live views of the per-world actuated PD gains (reflect DR scaling when
        # on; nominal otherwise) so the torque-penalty reward matches the gains
        # actually driving the sim.
        self.act_ke = self.dr.joint_ke.view(N, self.dofs)[:, 6:]
        self.act_kd = self.dr.joint_kd.view(N, self.dofs)[:, 6:]

        self.reset()

    # ---- rsl_rl VecEnv interface ----
    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh_obs()
        return self.obs_buf, None

    def step(self, actions: torch.Tensor):
        actions = torch.clip(actions, -100.0, 100.0)

        # Apply DR push to the live velocity view before integrating.
        self.dr.maybe_push(self._step_count, self.jqd)

        targets_actuated = anymal.action_to_targets(actions, self.default_q, self.mujoco_idx)
        self.tgt[:, : self.free_slots] = 0.0
        self.tgt[:, self.free_slots :] = targets_actuated

        self.s0, self.s1 = self.backend.advance(self.control_dt, self.s0, self.s1, self.control, apply_forces=None)
        self._rebind_views()
        self._step_count += 1
        self.episode_length_buf += 1

        bf = compute_base_frame(self.jq, self.jqd, self.gravity_vec)
        rew = compute_reward(
            bf,
            self.command.cmd,
            actions,
            self.last_action,
            targets_actuated,
            self.last_qvel,
            self.control_dt,
            ke=self.act_ke,
            kd=self.act_kd,
        )
        fallen = compute_termination(bf)
        time_out = self.episode_length_buf >= self.max_episode_length
        dones = fallen | time_out

        # History update (before reset overwrites state).
        self.last_action = actions.clone()
        self.last_qvel = bf.qvel.clone()

        self.extras["time_outs"] = time_out
        self.extras["episode"] = {
            "rew_mean": rew.mean().detach(),
            "lin_vel_error": (self.command.cmd[:, :2] - bf.vel_b[:, :2]).norm(dim=1).mean().detach(),
        }

        # Periodic command resample.
        due = ((self.episode_length_buf % self.cfg.command_resample_steps) == 0).nonzero(as_tuple=False).flatten()
        if len(due) > 0:
            self.command.resample(due)

        env_ids = dones.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)

        self._refresh_obs()
        return self.obs_buf, None, rew, dones, self.extras

    # ---- internals ----
    def _rebind_views(self):
        self.jq = wp.to_torch(self.s0.joint_q).view(self.num_envs, self.coords)
        self.jqd = wp.to_torch(self.s0.joint_qd).view(self.num_envs, self.dofs)

    def _refresh_obs(self):
        bf = compute_base_frame(self.jq, self.jqd, self.gravity_vec)
        obs = compute_obs(bf, self.command.cmd, self.last_action, self.default_q, self.lab_idx)
        self.obs_buf = self.dr.obs_noise_vec(obs)

    def _reset_idx(self, env_ids: torch.Tensor):
        mask_t = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask_t[env_ids] = True
        mask_wp = wp.from_torch(mask_t.contiguous(), dtype=wp.bool)

        # Physical-parameter DR (no-op when disabled).
        self.dr.sample_and_apply(env_ids, self.backend.solver)

        # Randomized initial conditions: standing pose + yaw + small joint noise.
        self.jq[env_ids] = self.default_full_q
        self.jqd[env_ids] = 0.0
        k = len(env_ids)
        if self.cfg.ic_yaw:
            yaw = (2.0 * torch.rand(k, device=self.device, generator=self._ic_gen) - 1.0) * math.pi
            self.jq[env_ids, 3] = 0.0
            self.jq[env_ids, 4] = 0.0
            self.jq[env_ids, 5] = torch.sin(yaw * 0.5)
            self.jq[env_ids, 6] = torch.cos(yaw * 0.5)
        if self.cfg.ic_joint_noise > 0:
            noise = (
                2.0 * torch.rand(k, anymal.ACT_DIM, device=self.device, generator=self._ic_gen) - 1.0
            ) * self.cfg.ic_joint_noise
            self.jq[env_ids, 7:] += noise

        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        # flags=0: keep our custom ICs, only clear MuJoCo's internal buffers.
        self.backend.solver.reset(self.s0, world_mask=mask_wp, flags=0)

        self.episode_length_buf[env_ids] = 0
        self.last_action[env_ids] = 0.0
        self.last_qvel[env_ids] = 0.0
        self.command.resample(env_ids)
