# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Four-arm solver harness for the Part-1 (pure solver) benchmarks.

One factory builds any of the paper's four arms on the same
``contact_objects`` scene model, with matched contact budgets:

* ``mujoco``          -- fixed-step SolverMuJoCo, n substeps per boundary
* ``mujoco-adaptive`` -- SolverMuJoCoAdaptive (CENIC per-world step doubling)
* ``icf``             -- fixed-step SolverICF, n substeps per boundary
* ``icf-adaptive``    -- SolverICFAdaptive (same controller over ICF), with a
  Newton collision pipeline attached for the paper's two-query cadence

Every arm exposes the same drive: ``arm.boundary(s0, s1, ctrl)`` advances one
outer boundary of ``DT_OUTER`` and returns the (s0, s1) pair to continue
with. Fixed arms subdivide the boundary into ``n_sub`` equal substeps —
their accuracy knob; adaptive arms take ``tol`` — theirs. The ICF arms run
Newton's CollisionPipeline per substep (fixed) or attached (adaptive); the
MuJoCo arms collide internally exactly as in the existing benches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import newton

from scripts.scenes.contact_objects import (
    DT_INNER_MIN,
    DT_OUTER,
    build_model_randomized,
)

ARMS = ("mujoco", "mujoco-adaptive", "icf", "icf-adaptive")

# Matched per-world contact budgets. The scene's demo sizing (128/640)
# under-allocates on chaotic 64-world layouts — piled objects exceed it
# and MuJoCo's constraint arrays go out of bounds (CUDA 700, observed in
# the penetration sweep). Doubled with matched ICF capacity.
NCONMAX = 256
NJMAX = 1280
ICF_MAX_RIGID_CONTACT = 256


@dataclass
class Arm:
    name: str
    solver: object
    boundary: Callable  # (s0, s1, ctrl) -> (s0, s1)
    iteration_count: Callable  # () -> int, inner iterations of the last boundary


def _make_mujoco(model: newton.Model, n_sub: int) -> Arm:
    solver = newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=NCONMAX, njmax=NJMAX
    )
    contacts = model.contacts()
    dt = DT_OUTER / n_sub

    def boundary(s0, s1, ctrl):
        # step() writes state_out in place and returns None
        for _ in range(n_sub):
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0
        return s0, s1

    return Arm("mujoco", solver, boundary, lambda: n_sub)


def _make_mujoco_adaptive(model: newton.Model, tol: float) -> Arm:
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode="per_world",
        nconmax=NCONMAX,
        njmax=NJMAX,
    )

    def boundary(s0, s1, ctrl):
        return solver.step_dt(DT_OUTER, s0, s1, ctrl)

    return Arm(
        "mujoco-adaptive",
        solver,
        boundary,
        lambda: int(solver.iteration_count.numpy()[0]),
    )


def _icf():
    """Import icf_warp the way the IsaacLab manager does: pip-less, from
    ICF_WARP_PATH (default: the icf_warp_adaptive checkout)."""
    try:
        import icf_warp
    except ModuleNotFoundError:
        import os
        import sys

        root = os.path.expanduser(os.environ.get("ICF_WARP_PATH", "~/Documents/code/icf_warp_adaptive"))
        sys.path.insert(0, root)
        import icf_warp
    return icf_warp


def _icf_params():
    return replace(_icf().IcfParams(), max_rigid_contact=ICF_MAX_RIGID_CONTACT)


def _make_icf(model: newton.Model, n_sub: int) -> Arm:
    solver = _icf().SolverICF(model, params=_icf_params())
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = DT_OUTER / n_sub

    def boundary(s0, s1, ctrl):
        for _ in range(n_sub):
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, ctrl, contacts, dt)
            s0, s1 = s1, s0
        return s0, s1

    return Arm("icf", solver, boundary, lambda: n_sub)


def _make_icf_adaptive(model: newton.Model, tol: float) -> Arm:
    icf = _icf()
    solver = icf.SolverICFAdaptive(
        model,
        params=_icf_params(),
        adaptive=icf.IcfAdaptiveParams(
            tol=tol,
            dt_inner_init=DT_OUTER,
            dt_inner_min=DT_INNER_MIN,
            dt_inner_max=DT_OUTER,
        ),
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver.attach_collision_pipeline(pipeline)

    def boundary(s0, s1, ctrl):
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, DT_OUTER)
        return s1, s0

    def iterations() -> int:
        count = getattr(solver, "iteration_count", None)
        return int(count.numpy()[0]) if count is not None else -1

    return Arm("icf-adaptive", solver, boundary, iterations)


def make_arm(model: newton.Model, name: str, *, n_sub: int = 1, tol: float = 1e-3) -> Arm:
    """Build one arm on ``model``. Fixed arms take ``n_sub``, adaptive ``tol``."""
    if name == "mujoco":
        return _make_mujoco(model, n_sub)
    if name == "mujoco-adaptive":
        return _make_mujoco_adaptive(model, tol)
    if name == "icf":
        return _make_icf(model, n_sub)
    if name == "icf-adaptive":
        return _make_icf_adaptive(model, tol)
    raise ValueError(f"unknown arm {name!r}; choose from {ARMS}")


def build_model(n: int, seed: int = 42) -> newton.Model:
    """The shared benchmark scene at ``n`` worlds (randomized, fixed seed)."""
    return build_model_randomized(n, seed=seed)
