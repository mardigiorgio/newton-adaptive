"""Instrument the still-hold to find the ROOT CAUSE of the jitter (no guessing).

Rolls out the deterministic-mean policy and logs, per control step, the signal at each
layer of the control stack so we can see WHERE the chatter originates:

  raw policy action [7]  ->  processed arm targets [6] + binary gripper cmd  ->  joint_pos/vel [7]

From this we can tell whether the policy is COMMANDING the oscillation (action chatters),
the BINARY GRIPPER is toggling (gripper cmd flips 0<->0.044), or the PHYSICS/PD is ringing
(action steady but joint_vel oscillates). Writes one .npz per checkpoint + prints a hold-window
summary (per-DOF std + sign-flip rate). Headless, no cameras -> fast (~1-2 min total).

    scripts/rl/trossen/docker/run.sh scripts/rl/trossen/diag_jitter.py --headless \
      --ckpts 3498 3897 --steps 250
"""

import argparse
import glob
import importlib.metadata as metadata
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=250, help="rollout steps (full episode ~250)")
parser.add_argument("--ckpts", type=int, nargs="+", default=[3498, 3897])
parser.add_argument("--log_dirs", nargs="+",
                    default=["/isaac/logs/trossen/stationary_ai_lift_teacher.rails_v1",
                             "/isaac/logs/trossen/stationary_ai_lift_teacher.jv03"],
                    help="dirs to search for model_<iter>.pt (any ckpt found in any dir is used)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--hold_from", type=int, default=160, help="first step of the settled-hold window for the summary")
parser.add_argument("--out_dir", default="/isaac/artifacts/diag")
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

ARM = [f"follower_left_joint_{i}" for i in range(6)]
GRIP = "follower_left_left_carriage_joint"


def _find(it):
    for d in args.log_dirs:
        p = os.path.join(d, f"model_{it}.pt")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"model_{it}.pt not in any of {args.log_dirs}")


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    ver = metadata.version("rsl-rl-lib")
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, ver)

    env = gym.make(args.task, cfg=env_cfg, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    robot = env.unwrapped.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM, preserve_order=True)
    grip_ids, _ = robot.find_joints([GRIP])
    sel = list(arm_ids) + list(grip_ids)
    names = ARM + [GRIP]
    am = env.unwrapped.action_manager
    print(f"[diag] joint order logged: {names}")
    print(f"[diag] action_manager terms: {am.active_terms}")
    # Baked PD gains (Lever-2 / physics-ringing hypothesis): if action is steady but joint_vel
    # oscillates, underdamped PD is the cause and these are what we'd raise damping on.
    print(f"[diag] baked stiffness {names}: {robot.data.joint_stiffness[0, sel].cpu().numpy().round(2)}")
    print(f"[diag] baked damping   {names}: {robot.data.joint_damping[0, sel].cpu().numpy().round(3)}")

    for it in args.ckpts:
        runner.load(_find(it))
        policy = runner.get_inference_policy(device=agent_cfg.device)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        obs, _ = wrapped.reset()

        raw_a, proc_a, jpos, jvel = [], [], [], []
        with torch.no_grad():
            for _ in range(args.steps):
                a = policy(obs)                                  # raw network mean output [1,7]
                raw_a.append(a[0].detach().cpu().numpy().copy())
                obs, _, _, _ = wrapped.step(a)
                proc_a.append(am.action[0].detach().cpu().numpy().copy())   # processed full action
                jpos.append(robot.data.joint_pos[0, sel].detach().cpu().numpy().copy())
                jvel.append(robot.data.joint_vel[0, sel].detach().cpu().numpy().copy())

        raw_a = np.stack(raw_a); proc_a = np.stack(proc_a); jpos = np.stack(jpos); jvel = np.stack(jvel)
        np.savez(os.path.join(args.out_dir, f"jitter_{it}.npz"),
                 raw_action=raw_a, proc_action=proc_a, joint_pos=jpos, joint_vel=jvel, names=np.array(names))

        h = slice(args.hold_from, args.steps)
        print(f"\n===== ckpt {it}  hold window steps {args.hold_from}-{args.steps} =====")
        print(f"{'DOF':>34} {'jvel_std':>9} {'jpos_std':>9} {'rawA_std':>9} {'rawA_signflips':>14}")
        for j, nm in enumerate(names):
            # raw action dim: arm dims 0-5 map to arm joints; dim 6 is gripper
            ra = raw_a[h, j] if j < 7 else None
            flips = int(np.sum(np.diff(np.sign(raw_a[h, j])) != 0)) if j < raw_a.shape[1] else -1
            print(f"{nm:>34} {np.std(jvel[h, j]):>9.4f} {np.std(jpos[h, j]):>9.4f} "
                  f"{np.std(raw_a[h, j]):>9.4f} {flips:>14}")
        # gripper-specific: how often does the BINARY command region flip, and where does raw dim 6 sit?
        g = raw_a[h, 6]
        print(f"[diag] gripper raw action dim6: mean={g.mean():.3f} std={g.std():.4f} "
              f"min={g.min():.3f} max={g.max():.3f} sign-flips={int(np.sum(np.diff(np.sign(g))!=0))}/{len(g)}")

    wrapped.close()


main()
os._exit(0)
