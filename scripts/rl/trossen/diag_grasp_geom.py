"""Geometry sanity check: does the cube fit the gripper, and is the TCP between the fingers?

No reward shaping fixes a geometric mismatch, so verify before training. Run in the container:
    /opt/venv/bin/python scripts/rl/trossen/diag_grasp_geom.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Cube-StationaryAI-Teacher-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import trossen_cube  # noqa: F401,E402

cfg = parse_env_cfg(args.task, num_envs=1)
cfg.scene.num_envs = 1
env = gym.make(args.task, cfg=cfg)
env.reset()
u = env.unwrapped
robot = u.scene["robot"]
obj = u.scene["object"]
ee = u.scene["ee_frame"]

jid = robot.find_joints("follower_left_left_carriage_joint")[0][0]


def bpos(name):
    i = robot.find_bodies(name)[0][0]
    return robot.data.body_pos_w[0, i]


def report(tag):
    gl, gr = bpos("follower_left_gripper_left"), bpos("follower_left_gripper_right")
    sep = torch.norm(gl - gr).item() * 100.0
    tcp = ee.data.target_pos_w[0, 0]
    cube = obj.data.root_pos_w[0, :3]
    mid = (gl + gr) / 2.0
    print(
        f"[{tag}] carriage={robot.data.joint_pos[0, jid].item():.4f}  finger_sep={sep:.2f}cm  "
        f"TCP->finger_mid={torch.norm(tcp - mid).item()*100:.2f}cm  TCP->cube={torch.norm(tcp - cube).item()*100:.2f}cm"
    )


# arm holds default; toggle the binary gripper action (+1 open / -1 close) and settle
act = torch.zeros((1, u.action_manager.total_action_dim), device=u.device)
act[0, -1] = 1.0
for _ in range(40):
    env.step(act)
report("OPEN ")
act[0, -1] = -1.0
for _ in range(40):
    env.step(act)
report("CLOSE")

# cube width: world AABB of the manipuland prim (compare to the closed gripper finger_sep above)
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

stage = omni.usd.get_context().get_stage()
cube_prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
if cube_prim and cube_prim.IsValid():
    rng = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(cube_prim)
        .ComputeAlignedRange()
    )
    sz = rng.GetSize()
    w = max(sz[0], sz[1]) * 100.0
    verdict = "FITS (> closed 4.83cm)" if w > 4.83 else "TOO SMALL (< closed gripper, fingers won't touch)"
    print(f"[CUBE ] x={sz[0]*100:.1f}cm y={sz[1]*100:.1f}cm z={sz[2]*100:.1f}cm -> grip width ~{w:.1f}cm  {verdict}")
else:
    print("[CUBE ] object prim not found")

import sys  # noqa: E402

sys.stdout.flush()  # os._exit below does NOT flush buffers -- flush or the prints are lost
app.close()
import os  # noqa: E402

os._exit(0)
