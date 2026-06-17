"""Per-world domain randomization for the sim-to-real arm of the study.

Physical-parameter DR (mass, friction, motor strength) is applied at reset by
writing Newton-side ``Model`` arrays then calling ``solver.notify_model_changed``
— the public path that re-pushes to MuJoCo-Warp and recomputes derived inertial
fields. Velocity pushes and observation noise are free (state/obs writes) and run
per step.

DR-on reset cost is a known validate-first item (notify_model_changed may re-sync
all worlds); see the build spec §6.4. Defaults to disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # noqa: TID253
import warp as wp

from newton import ModelFlags

from .anymal import TaskMeta


@dataclass
class DRConfig:
    enable: bool = False
    friction: tuple[float, float] = (0.6, 1.25)  # scale of nominal shape mu
    base_mass_add: tuple[float, float] = (-1.0, 2.0)  # kg added to trunk body
    motor_strength: tuple[float, float] = (0.9, 1.1)  # scale joint_target_ke (Kp)
    kd_scale: tuple[float, float] = (0.8, 1.2)  # scale joint_target_kd
    push_vel_xy: float = 0.8  # m/s base-velocity kick magnitude
    push_interval_steps: int = 500  # ~10 s @ 50 Hz
    obs_noise: bool = True
    obs_noise_std: float = 0.05

    @classmethod
    def preset(cls, name: str) -> DRConfig:
        if name in ("off", "none", None):
            return cls(enable=False)
        if name == "on":
            return cls(enable=True)
        if name == "ood":
            # Out-of-distribution perturbation for the ref_perturbed eval backend:
            # wider than training DR, to probe transfer.
            return cls(
                enable=True,
                friction=(0.4, 1.5),
                base_mass_add=(-2.0, 4.0),
                motor_strength=(0.75, 1.25),
                kd_scale=(0.7, 1.3),
                push_vel_xy=1.2,
            )
        raise ValueError(f"unknown DR preset {name!r}")

    # Single-term knobs at training-band magnitude, used to attribute the transfer
    # gap to one channel at a time (error-budget ranking, Phase 0a/0b). Each opens
    # exactly one term; obs_noise and pushes are OFF unless they ARE the term.
    SINGLE_TERMS = ("friction", "mass", "kp", "kd", "push", "obsnoise")

    @classmethod
    def single(cls, term: str) -> DRConfig:
        """One-knob preset: identity everywhere except ``term``.

        Starting from a fully-disabled randomizer (all ranges pinned to identity,
        obs noise and pushes off), open a single channel at its nominal training
        magnitude. ``friction``/``mass``/``kp``/``kd``/``push`` are dynamics terms;
        ``obsnoise`` is a SENSING term (perturbs the observation, not the sim) and
        must be ranked on a separate axis, not as a physical-dynamics term.
        """
        base = dict(
            enable=True,
            friction=(1.0, 1.0),
            base_mass_add=(0.0, 0.0),
            motor_strength=(1.0, 1.0),
            kd_scale=(1.0, 1.0),
            push_vel_xy=0.0,
            push_interval_steps=0,
            obs_noise=False,
            obs_noise_std=0.0,
        )
        knobs = {
            "friction": dict(friction=(0.6, 1.25)),
            "mass": dict(base_mass_add=(-1.0, 2.0)),
            "kp": dict(motor_strength=(0.9, 1.1)),
            "kd": dict(kd_scale=(0.8, 1.2)),
            "push": dict(push_vel_xy=0.8, push_interval_steps=500),
            "obsnoise": dict(obs_noise=True, obs_noise_std=0.05),
        }
        if term not in knobs:
            raise ValueError(f"unknown single-term {term!r}; choose from {cls.SINGLE_TERMS}")
        return cls(**{**base, **knobs[term]})


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(n, device=device)


class DomainRandomizer:
    """Samples and applies per-world physical-parameter randomization.

    Args:
        cfg: DR configuration.
        meta: Task layout.
        model: Finalized Newton model (arrays are written in place).
        device: Torch device.
    """

    def __init__(self, cfg: DRConfig, meta: TaskMeta, model, device):
        self.cfg = cfg
        self.meta = meta
        self.model = model
        self.device = device

        # Zero-copy torch views of the model arrays we perturb.
        self.body_mass = wp.to_torch(model.body_mass)
        self.shape_mu = wp.to_torch(model.shape_material_mu)
        self.joint_ke = wp.to_torch(model.joint_target_ke)
        self.joint_kd = wp.to_torch(model.joint_target_kd)

        # Nominal snapshots for scale-from-nominal.
        self.nominal_mass = self.body_mass.clone()
        self.nominal_mu = self.shape_mu.clone()
        self.nominal_ke = self.joint_ke.clone()
        self.nominal_kd = self.joint_kd.clone()

        n = meta.num_worlds
        self.bodies_per_world = meta.bodies_per_world
        self.dofs_per_world = meta.dofs_per_world
        # One shared ground-plane shape trails the per-world robot shapes
        # (build_anymal_model always adds it once after replicate); exclude it.
        total_shapes = self.shape_mu.shape[0]
        self.shapes_per_world = (total_shapes - 1) // n

    def sample_and_apply(self, env_ids: torch.Tensor, solver):
        """Randomize physical params for ``env_ids`` and push to the solver."""
        if not self.cfg.enable or len(env_ids) == 0:
            return
        c = self.cfg
        k = len(env_ids)
        n = self.meta.num_worlds

        # Trunk mass: nominal + additive offset.
        trunk = env_ids * self.bodies_per_world + self.meta.trunk_local_index
        self.body_mass[trunk] = self.nominal_mass[trunk] + _uniform(*c.base_mass_add, k, self.device)

        # Friction: one per-world factor over the robot shapes (trailing ground
        # shape excluded). Vectorized — env_ids stays on device, no host sync.
        spw = self.shapes_per_world
        fric = _uniform(*c.friction, k, self.device)
        mu_view = self.shape_mu[: n * spw].view(n, spw)
        nom_mu = self.nominal_mu[: n * spw].view(n, spw)
        mu_view[env_ids] = nom_mu[env_ids] * fric[:, None]

        # Motor strength: scale actuated-dof PD gains (dofs 6: of each world).
        dpw = self.dofs_per_world
        ke_s = _uniform(*c.motor_strength, k, self.device)
        kd_s = _uniform(*c.kd_scale, k, self.device)
        ke_view = self.joint_ke.view(n, dpw)
        kd_view = self.joint_kd.view(n, dpw)
        nke = self.nominal_ke.view(n, dpw)
        nkd = self.nominal_kd.view(n, dpw)
        ke_view[env_ids, 6:] = nke[env_ids, 6:] * ke_s[:, None]
        kd_view[env_ids, 6:] = nkd[env_ids, 6:] * kd_s[:, None]

        solver.notify_model_changed(
            ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.SHAPE_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES
        )

    def maybe_push(self, step: int, jqd: torch.Tensor):
        """Apply a random planar base-velocity kick on the push interval."""
        if not self.cfg.enable or self.cfg.push_interval_steps <= 0:
            return
        if step % self.cfg.push_interval_steps != 0 or step == 0:
            return
        n = jqd.shape[0]
        kick = (2.0 * torch.rand(n, 2, device=self.device) - 1.0) * self.cfg.push_vel_xy
        jqd[:, 0:2] += kick

    def obs_noise_vec(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.cfg.enable or not self.cfg.obs_noise:
            return obs
        return obs + self.cfg.obs_noise_std * torch.randn_like(obs)
