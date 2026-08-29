# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Zero-gravity momentum conservation for the four arms (SimBenchmark's
ANYmal-momentum test, Erez 2015's momentum drift): two spheres collide
head-on with gravity off; total linear momentum must be conserved by the
contact solve regardless of step or accuracy. Certifies that the
per-world adaptive controller injects no momentum.

    uv run python scripts/bench/probe_momentum.py
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from scripts.bench.four_arms import _make_icf, _make_icf_adaptive, _make_mujoco, _make_mujoco_adaptive
from scripts.scenes.cenic_scenes import SCENES


def build(n=2):
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1e5, kd=2e3, mu=0.5, margin=0.0, density=1000.0)
    for x, vx in ((-0.1, 1.0), (0.1, -0.5)):
        b = t.add_body(xform=wp.transform(p=wp.vec3(x, 0.0, 0.5), q=wp.quat_identity()))
        t.add_shape_sphere(b, radius=0.025, cfg=cfg)
    builder = newton.ModelBuilder()
    builder.replicate(t, n)
    m = builder.finalize()
    # gravity is per world: zero EVERY world (a (1, 3) assign left world 1 in free fall)
    m.gravity.zero_()
    assert float(np.abs(m.gravity.numpy()).max()) == 0.0
    return m


def main() -> int:
    scene = "hard-clutter"
    icf, solref = SCENES[scene].icf, SCENES[scene].mujoco_solref
    print(f"{'arm':16s} {'setting':>10} {'|p_end - p0| / |p0|':>20} {'finite':>7}")
    for backend in ("icf", "mujoco"):
        for kind, knob in (("fixed", 1e-2), ("fixed", 1e-3), ("adaptive", 1e-2), ("adaptive", 1e-4)):
            m = build()
            if backend == "icf":
                a = (
                    _make_icf(m, int(round(0.01 / knob)), 0.01, icf)
                    if kind == "fixed"
                    else _make_icf_adaptive(m, knob, 0.01, icf, 4096)
                )
            else:
                a = (
                    _make_mujoco(m, int(round(0.01 / knob)), 0.01, solref)
                    if kind == "fixed"
                    else _make_mujoco_adaptive(m, knob, 0.01, 4096, solref)
                )
            s0, s1, c = m.state(), m.state(), m.control()
            # initial velocities on BOTH layouts the solvers read (free joints: joint_qd [lin, ang]; bodies: body_qd)
            v0 = np.array([[1.0, 0, 0], [-0.5, 0, 0]] * 2, dtype=np.float32)
            qd = s0.joint_qd.numpy().reshape(-1, 6)
            qd[:, :3] = v0
            s0.joint_qd.assign(qd.reshape(-1))
            bqd = s0.body_qd.numpy().reshape(-1, 6)
            bqd[:, :3] = v0
            s0.body_qd.assign(bqd.reshape(-1))
            mass = m.body_mass.numpy()
            # p0 from the state through the same accessor used at the end (guards the layout)
            v_state = s0.joint_qd.numpy().reshape(-1, 6)[:, :3]
            assert np.allclose(v_state, v0), v_state
            p0 = (mass[:, None] * v_state).reshape(2, 2, 3).sum(axis=1)
            assert np.linalg.norm(p0, axis=1).min() > 0, p0
            for _ in range(60):  # 0.6 s: the spheres meet at ~0.1 s
                s0, s1 = a.boundary(s0, s1, c)
            v = s0.joint_qd.numpy().reshape(-1, 6)[:, :3]
            p = (mass[:, None] * v).reshape(2, 2, 3).sum(axis=1)
            err = np.linalg.norm(p - p0, axis=1) / np.linalg.norm(p0, axis=1)
            arm = backend if kind == "fixed" else f"{backend}-adaptive"
            lab = f"dt={knob * 1e3:g}ms" if kind == "fixed" else f"eps={knob:g}"
            print(f"{arm:16s} {lab:>10} {err.max():>20.2e} {bool(np.isfinite(v).all())!s:>7}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
