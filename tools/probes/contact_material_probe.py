"""Certify that the authored per-body contact material reaches the MuJoCo model.

Oracle argument: the expected values are not computed here, they are READ from
upstream models that ship with the hardware and the engine --

  * gripper / arm colliders: friction 1.0, condim 6
    google-deepmind/mujoco_menagerie ``trossen_wxai/wxai_follower.xml``, which
    models this exact arm: collision class ``friction="1 5e-3 5e-4"``, gripper
    subclass ``condim="6"``. The same triple appears in ``aloha/aloha.xml``.
  * mug: friction 0.2
    the LBM asset's own ``drake:mu_static`` / ``drake:mu_dynamic``.
  * tabletop: friction 0.6

so a PASS says our config produces the upstream numbers, and a FAIL names the
body that did not. The task config is ours and therefore unvalidated; the
mixing rule that MuJoCo then applies to a PAIR (element-wise maximum of the two
geoms' coefficients at equal priority, maximum of their condim) is upstream
engine behaviour and is NOT re-derived here -- this probe constrains only the
per-geom inputs that rule consumes.

What a PASS does NOT certify: that a grasp holds, that impratio/cone are right,
or anything about the SAP backend, which reads none of these ``mjc:`` arrays and
combines friction by its own rule.

Run (single line, from the newton-adaptive repo root):

    VIRTUAL_ENV=$HOME/Documents/code/IsaacLabRubato/.venv $HOME/Documents/code/IsaacLabRubato/.venv/bin/python tools/probes/contact_material_probe.py
"""

from __future__ import annotations

import argparse
import re
import sys

TASK = "IsaacContrib-Lift-Spatula-Trossen-v0"

# Body class -> (prim-path regex, expected sliding friction, expected condim).
# Geoms are named by their USD prim path, so the class is read off the path.
# condim None means "not asserted": only the rig authors an mjc:condim.
EXPECTED = (
    ("gripper", re.compile(r"/Robot/follower_left_(carriage|gripper)_"), 1.0, 6),
    ("arm link", re.compile(r"/Robot/follower_left_(link|base)_"), 1.0, 6),
    ("mug", re.compile(r"/Object/"), 0.2, None),
    ("table", re.compile(r"/TableGuard"), 0.6, None),
)

TOL = 1e-6


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    app = AppLauncher(args).app  # noqa: F841

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import numpy as np
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(TASK, num_envs=1)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()

    import mujoco

    from isaaclab_newton.physics.mjwarp_manager import NewtonManager

    solver = NewtonManager._solver

    geom_friction = solver.mjw_model.geom_friction.numpy()
    geom_condim = solver.mjw_model.geom_condim.numpy()

    # Both arrays may or may not carry a leading world axis; this probe reads
    # world 0, and the config authors no per-world material variation.
    if geom_friction.ndim == 3:
        geom_friction = geom_friction[0]
    if geom_condim.ndim == 2:
        geom_condim = geom_condim[0]

    # Geoms are named by their USD prim path, which is what the class patterns
    # match on -- Newton leaves Model.shape_key unset on this path.
    # solref[0] is the contact response TIME CONSTANT [s]. MuJoCo's own
    # guidance is that it should not fall below 2*dt; below that the contact
    # responds faster than the integrator can represent, which a fixed step
    # absorbs silently and an error-controlled step pays for in substeps.
    geom_solref = solver.mjw_model.geom_solref.numpy()
    if geom_solref.ndim == 3:
        geom_solref = geom_solref[0]

    rows = []
    for geom_idx in range(solver.mj_model.ngeom):
        name = mujoco.mj_id2name(solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_idx) or f"<unnamed {geom_idx}>"
        rows.append((name, float(geom_friction[geom_idx][0]), int(geom_condim[geom_idx]),
                     float(geom_solref[geom_idx][0])))

    if not rows:
        print("FAIL: model carries no geoms; probe cannot certify anything")
        return 1

    failures = []
    dt = float(env_cfg.sim.dt)
    print(f"sim.dt = {dt:g} s; MuJoCo guidance: solref timeconst >= 2*dt = {2*dt:g} s")
    print(f"{'body class':<10} {'n':>4}  {'friction':>10}  {'condim':>7}  {'solref t':>12}  {'vs 2dt':>8}")
    print("-" * 62)
    for label, pattern, want_mu, want_condim in EXPECTED:
        matched = [r for r in rows if pattern.search(r[0])]
        if not matched:
            failures.append(f"{label}: no collider matched /{pattern.pattern}/")
            print(f"{label:<10} {0:>4}  {'NO MATCH':>18}  {'-':>12}")
            continue
        mus = sorted({round(r[1], 9) for r in matched})
        dims = sorted({r[2] for r in matched})
        mu_txt = ",".join(f"{m:g}" for m in mus)
        dim_txt = ",".join(str(d) for d in dims)
        tcs = sorted({round(r[3], 9) for r in matched})
        tc_txt = ",".join(f"{t:.3g}" for t in tcs)
        ratio = min(tcs) / (2 * dt) if tcs else float("nan")
        print(f"{label:<10} {len(matched):>4}  {mu_txt:>10}  {dim_txt:>7}  {tc_txt:>12}  {ratio:>7.3f}x")

        off = [r for r in matched if abs(r[1] - want_mu) > TOL]
        if off:
            failures.append(f"{label}: friction {off[0][1]:g} on '{off[0][0]}', authored {want_mu:g} ({len(off)} geoms)")
        if want_condim is not None:
            bad = [r for r in matched if r[2] != want_condim]
            if bad:
                failures.append(f"{label}: condim {bad[0][2]} on '{bad[0][0]}', authored {want_condim}")

    print()
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        print("\nall geoms:")
        for name, mu, dim, tc in rows:
            print(f"  mu={mu:<6g} condim={dim} solref_t={tc:<10.4g}  {name}")
        return 1

    # The mug must NOT have been given the fallback coefficient, and the rig
    # must NOT have been given the mug's: that collapse is the failure this
    # probe exists to catch, and equal expected values would hide it.
    mug_mu = {r[1] for r in rows if EXPECTED[2][1].search(r[0])}
    rig_mu = {r[1] for r in rows if EXPECTED[0][1].search(r[0])}
    if mug_mu & rig_mu:
        print(f"FAIL\n  - mug and rig share a coefficient ({sorted(mug_mu & rig_mu)}); per-body material did not apply")
        return 1

    print("PASS: authored per-body friction and condim reach the MuJoCo model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
