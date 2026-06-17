"""CENIC-vs-fixed RL locomotion workflow (ANYmal-C velocity tracking).

Tests the hypothesis that training an RL locomotion policy with the CENIC
adaptive-timestep solver as the physics backend increases policy performance
and closes the sim-to-real gap, compared to fixed-timestep physics.

Stack: a custom vectorized ``VecEnv`` (:mod:`scripts.rl.cenic_env`) wrapping
``SolverMuJoCoCENIC.step_dt`` (or stock fixed-step ``SolverMuJoCo`` behind the
same env for an apples-to-apples A/B), trained with rsl_rl PPO.

Entrypoints::

    uv run -m scripts.rl.smoke_test                 # CPU plumbing check (Mac-runnable)
    uv run -m scripts.rl.train --backend cenic ...  # train on the 4070
    uv run -m scripts.rl.eval_transfer ...          # cross-physics transfer matrix
"""
