"""CPU plumbing smoke test (Mac-runnable) for the RL env contract.

Validates build -> reset -> obs shape -> step -> reward/done shapes -> finite,
for both backends, on a tiny 2-world model. The rsl_rl handshake test is skipped
when rsl_rl is not installed (e.g. on the Mac).

    uv run -m scripts.rl.smoke_test
"""

from __future__ import annotations

import unittest

import torch  # noqa: TID253

from .backends import BackendSpec
from .cenic_env import CenicLocomotionEnv
from .config import EnvConfig, make_ppo_cfg


def _tiny_env(kind: str):
    cfg = EnvConfig(num_envs=2)
    return CenicLocomotionEnv(cfg, BackendSpec(kind=kind), None, device="cpu", headless=True)


class SmokeTest(unittest.TestCase):
    def _run_contract(self, kind: str):
        env = _tiny_env(kind)
        obs, priv = env.reset()
        self.assertEqual(tuple(obs.shape), (2, 48))
        self.assertIsNone(priv)
        action = torch.zeros(2, 12)
        for _ in range(5):
            obs, priv, rew, done, extras = env.step(action)
            self.assertEqual(tuple(obs.shape), (2, 48))
            self.assertEqual(tuple(rew.shape), (2,))
            self.assertEqual(tuple(done.shape), (2,))
            self.assertIsNone(priv)
            self.assertIn("time_outs", extras)
            self.assertTrue(torch.isfinite(obs).all(), "non-finite obs")
            self.assertTrue(torch.isfinite(rew).all(), "non-finite reward")

    def test_env_contract_fixed(self):
        self._run_contract("fixed")

    def test_env_contract_cenic(self):
        self._run_contract("cenic")

    def test_rsl_rl_handshake(self):
        try:
            from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415
        except ImportError:
            self.skipTest("rsl_rl not installed")
        import tempfile  # noqa: PLC0415

        env = _tiny_env("fixed")
        cfg = make_ppo_cfg(max_iterations=2, seed=0)
        cfg["runner"]["num_steps_per_env"] = 4
        with tempfile.TemporaryDirectory() as logdir:
            runner = OnPolicyRunner(env, cfg, log_dir=logdir, device="cpu")
            runner.learn(num_learning_iterations=2, init_at_random_ep_len=True)


if __name__ == "__main__":
    unittest.main()
