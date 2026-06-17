"""Gym task registrations for the Trossen Stationary AI cube-lift task.

Importing this (via ``import trossen_cube``) registers the gym ids. The env-cfg
modules import ``isaaclab.*``, so this must run AFTER ``AppLauncher`` has started.
The ``rsl_rl`` agent entry points are resolved lazily at train time (Phase 3/4).
"""

import gymnasium as gym

gym.register(
    id="Isaac-Lift-Cube-StationaryAI-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_lift.cube_lift_env_cfg:StationaryAiCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.cube_lift.agents.rsl_rl_ppo_cfg:StationaryAILiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cube_lift.cube_lift_env_cfg:StationaryAiCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.cube_lift.agents.rsl_rl_ppo_cfg:StationaryAILiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
