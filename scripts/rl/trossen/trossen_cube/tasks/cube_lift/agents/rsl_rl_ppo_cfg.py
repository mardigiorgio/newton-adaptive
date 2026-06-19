"""Stage-1 PPO teacher runner cfg (privileged-state teacher, SYMMETRIC actor-critic).

Both the actor AND the critic see proprioception (``policy``) plus the ground-truth object pose +
command (``privileged``). Putting ``privileged`` in the ACTOR's groups is load-bearing -- do NOT
"simplify" it back to actor-``["policy"]``: that blinds the actor to the randomized cube so it only
reaches the mean spawn point and never grasps (the single biggest bug we fixed). The deployable
ASYMMETRIC split arrives in Phase 4, where the vision STUDENT swaps ``privileged`` for depth images
while the teacher keeps it. Keys are ``"actor"``/``"critic"`` -- the rsl_rl 5.x ``OnPolicyRunner``
convention (NOT ``"policy"``; that's an env obs-group name). Hyperparameters mirror the reference
Franka/WXAI lift PPO cfg. See IMPL_GROUND_TRUTH.md.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class StationaryAILiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3500
    save_interval = 50
    experiment_name = "stationary_ai_lift_teacher"

    # PRIVILEGED TEACHER: the actor MUST see the object pose (privileged) -- otherwise it's blind to
    # the randomized cube and can only reach the average spawn point, never grasping. (The reference
    # lift task puts object_position directly in its single policy obs group.) The Phase-4 vision
    # student replaces "privileged" with depth images in its own actor group.
    obs_groups = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged"]}

    # Mirrors the reference Franka PPO cfg (init_std=1.0, no obs normalization, [256,128,64] elu). Only
    # deviation: the new rsl-rl 5.x model API requires RslRlMLPModelCfg, and we use std_type="log"
    # (std = exp(clamp(log_std)), always > 0) -- the crash-safe equivalent of the old ActorCritic std.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
