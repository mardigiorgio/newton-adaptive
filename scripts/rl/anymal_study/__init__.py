"""Adaptive-vs-fixed RL locomotion workflow (ANYmal-C velocity tracking).

Tests the hypothesis that training an RL locomotion policy with the adaptive
timestep solver as the physics backend increases policy performance
and closes the sim-to-real gap, compared to fixed-timestep physics.

Stack: a custom vectorized ``VecEnv`` (:mod:`scripts.rl.anymal_study.adaptive_env`) wrapping
``SolverMuJoCoAdaptive.step_dt`` (or stock fixed-step ``SolverMuJoCo`` behind the
same env for an apples-to-apples A/B), trained with rsl_rl PPO.

Entrypoints::

    uv run -m scripts.rl.anymal_study.smoke_test                    # CPU plumbing check (Mac-runnable)
    uv run -m scripts.rl.anymal_study.train --backend adaptive ...  # train on the 4070
    uv run -m scripts.rl.anymal_study.eval_transfer ...             # cross-physics transfer matrix
"""
