# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the four-arm contact budgets against measured demand.

A contact the pipeline generates but the solver cannot scan is dropped
silently, and every number downstream is physics of a different scene.
This probe drives each CENIC scene for 2 s at a 1 ms fixed step, records
the PEAK per-world contact demand on both backends (Newton's collision
pipeline for the ICF arms; MuJoCo's active-contact and constraint counts),
and fails unless every budget in four_arms.py holds at least 2x the peak.
Run it before any sweep whose scene or budgets changed:

    uv run python scripts/bench/verify_contact_budgets.py
"""

from __future__ import annotations

import sys

import newton
import numpy as np

from scripts.bench.four_arms import ICF_MAX_RIGID_CONTACT, NCONMAX, NJMAX, build_model, make_arm
from scripts.scenes.cenic_scenes import SCENES

MARGIN = 2.0
BOUNDARIES = 200  # 2 s


def demand(scene: str, n: int) -> dict:
    m = build_model(n, scene=scene)
    pipe = newton.CollisionPipeline(m, rigid_contact_max=200000)
    contacts = pipe.contacts()
    a = make_arm(m, "icf", scene=scene, n_sub=10)
    s0, s1, c = m.state(), m.state(), m.control()
    peak = 0
    for _ in range(BOUNDARIES):
        s0, s1 = a.boundary(s0, s1, c)
        pipe.collide(s0, contacts)
        peak = max(peak, int(contacts.rigid_contact_count.numpy()[0]))
    m2 = build_model(n, scene=scene)
    a2 = make_arm(m2, "mujoco", scene=scene, n_sub=10)
    s0, s1, c = m2.state(), m2.state(), m2.control()
    d = a2.solver.mjw_data
    pk_con, pk_efc = 0, 0
    for _ in range(BOUNDARIES):
        s0, s1 = a2.boundary(s0, s1, c)
        pk_con = max(pk_con, int(np.asarray(d.nacon.numpy()).max()))
        if hasattr(d, "nefc"):
            pk_efc = max(pk_efc, int(np.asarray(d.nefc.numpy()).max()))
    return {"icf_per_world": peak / n, "mj_con_per_world": pk_con / n, "mj_efc_per_world": pk_efc / n}


def main() -> int:
    ok = True
    for scene in sorted(SCENES):
        for n in (1, 64):
            dm = demand(scene, n)
            checks = [
                ("ICF_MAX_RIGID_CONTACT", ICF_MAX_RIGID_CONTACT, dm["icf_per_world"]),
                ("NCONMAX", NCONMAX, dm["mj_con_per_world"]),
                ("NJMAX", NJMAX, dm["mj_efc_per_world"]),
            ]
            for name, budget, need in checks:
                good = budget >= MARGIN * need
                ok &= good
                print(f"  {'ok  ' if good else 'FAIL'} {scene:13s} N={n:<3d} {name:22s} budget {budget:6d}  peak demand/world {need:7.0f}  (need >= {MARGIN * need:.0f})")
    print("VERIFY PASSED: budgets hold >= 2x measured demand" if ok else "VERIFY FAILED: raise the budgets in four_arms.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
