# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Solver factory builders shared across all scenes.

Each builder returns a ``(solver, step_fn)`` pair given a Newton ``Model``.
``step_fn`` has signature ``(model, s0, s1, ctrl) -> (s0_new, s1_old)`` so the
bench's ``infra.measure`` loop can call it uniformly across solver kinds.

Why factories are parameterised by scene-specific knobs (``nconmax``,
``njmax``, ``tol``, ``dt_outer``): different scenes have wildly different
contact / constraint budgets and outer cadences, so each scene constructs
its own factory dict via these builders rather than hardcoding values here.

``mjwarp``'s ``nconmax``/``njmax`` are per-world; it multiplies by ``nworld``
internally.  Pass per-world values, not totals.
"""
from __future__ import annotations

from collections.abc import Callable

import newton
import newton.solvers

# A step_fn advances state from s0 to s1 by one DT_OUTER period and returns
# the (latest, previous) state pair so the caller can rotate buffers cheaply.
StepFn = Callable[[newton.Model, newton.State, newton.State, newton.Control],
                  tuple[newton.State, newton.State]]
SolverBuilder = Callable[[newton.Model], tuple[object, StepFn]]


def _resolve_buf(x, n: int) -> int:
    """Resolve a buffer size that may be a plain int or a callable of N.

    ``nconmax``/``njmax`` can be passed as ``int`` (same for every world count)
    or as ``Callable[[int], int]`` so a scene can hand each N its own
    finely-tuned, minimal-safe buffer (see ``contact_objects.buffer_sizes``).
    """
    return int(x(n)) if callable(x) else int(x)


# --- MuJoCo Adaptive (per-world step-doubling) ---------------------------

def mujoco_adaptive_factory(
    *, tol: float, nconmax: int, njmax: int, dt_outer: float,
    dt_inner_min: float = 1e-6, dt_inner_max: float | None = None,
    use_mujoco_contacts: bool = False,
) -> SolverBuilder:
    """Adaptive MuJoCo solver. Inner loop is owned by ``solver.step``.

    Args:
        use_mujoco_contacts: If True, route contacts through mjwarp's native
            broadphase+narrowphase (avoids the Newton SAP pipeline allocation,
            which is O(N*nconmax) and hits int32 overflow at high N). Use True
            for scenes with only primitives (sphere/box/plane/cylinder).
            Leave False for scenes with non-convex meshes that need Newton's
            collision pipeline (e.g. CoACD-decomposed assets).
    """
    inner_max = dt_outer if dt_inner_max is None else dt_inner_max

    def build(model):
        nc, nj = _resolve_buf(nconmax, model.world_count), _resolve_buf(njmax, model.world_count)
        solver = newton.solvers.SolverMuJoCoAdaptive(
            model, tol=tol, dt_init=dt_outer, dt_min=dt_inner_min,
            dt_max=inner_max,
            nconmax=nc, njmax=nj,
            use_mujoco_contacts=use_mujoco_contacts,
        )

        def step_fn(model, s0, s1, ctrl):
            # Adaptive solver updates s0 in place; return (s0, s1) unchanged.
            solver.step(s0, s1, ctrl, None, dt_outer)
            return s0, s1

        return solver, step_fn

    return build


# --- MuJoCo fixed step ---------------------------------------------------

def mujoco_fixed_factory(
    *, dt: float, nconmax: int, njmax: int, dt_outer: float,
) -> SolverBuilder:
    """Fixed-step MuJoCo (SolverMuJoCo). Internal substeps so each step_fn
    advances exactly one ``dt_outer``."""
    n_sub = max(1, round(dt_outer / dt))

    def build(model):
        nc, nj = _resolve_buf(nconmax, model.world_count), _resolve_buf(njmax, model.world_count)
        solver = newton.solvers.SolverMuJoCo(
            model, separate_worlds=True, nconmax=nc, njmax=nj,
        )
        contacts = model.contacts()  # MuJoCo populates via its own pipeline.

        def step_fn(model, s0, s1, ctrl):
            # solver.step writes into state_out (s1) in place. Some Newton
            # solvers (SemiImplicit, VBD) return None; don't reassign.
            for _ in range(n_sub):
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            return s0, s1

        return solver, step_fn

    return build


# --- Featherstone (generalized-coord articulations) ----------------------

def featherstone_factory(*, dt: float, dt_outer: float) -> SolverBuilder:
    """Fixed-step Featherstone. Requires ``model.collide()`` before each step."""
    n_sub = max(1, round(dt_outer / dt))

    def build(model):
        solver = newton.solvers.SolverFeatherstone(model)
        contacts = model.contacts()

        def step_fn(model, s0, s1, ctrl):
            for _ in range(n_sub):
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            return s0, s1

        return solver, step_fn

    return build


# --- Semi-implicit Euler (maximal coords) --------------------------------

def semi_implicit_factory(*, dt: float, dt_outer: float) -> SolverBuilder:
    """Fixed-step semi-implicit Euler. Maximal coords; needs collide() per step."""
    n_sub = max(1, round(dt_outer / dt))

    def build(model):
        solver = newton.solvers.SolverSemiImplicit(model)
        contacts = model.contacts()

        def step_fn(model, s0, s1, ctrl):
            for _ in range(n_sub):
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            return s0, s1

        return solver, step_fn

    return build


# --- VBD (vertex block descent, implicit) --------------------------------

def vbd_factory(*, dt: float, dt_outer: float) -> SolverBuilder:
    """Fixed-step VBD. Implicit integrator; rigid contacts via collide()."""
    n_sub = max(1, round(dt_outer / dt))

    def build(model):
        solver = newton.solvers.SolverVBD(model)
        contacts = model.contacts()

        def step_fn(model, s0, s1, ctrl):
            for _ in range(n_sub):
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            return s0, s1

        return solver, step_fn

    return build


# --- XPBD (position-based dynamics, implicit) ----------------------------

def xpbd_factory(*, dt: float, dt_outer: float) -> SolverBuilder:
    """Fixed-step XPBD. Maximal coords, implicit, needs collide() per step."""
    n_sub = max(1, round(dt_outer / dt))

    def build(model):
        solver = newton.solvers.SolverXPBD(model)
        contacts = model.contacts()

        def step_fn(model, s0, s1, ctrl):
            for _ in range(n_sub):
                model.collide(s0, contacts)
                solver.step(s0, s1, ctrl, contacts, dt)
                s0, s1 = s1, s0
            return s0, s1

        return solver, step_fn

    return build


# --- Catalog of all builders for introspection ---------------------------

ALL_BUILDERS = {
    "mujoco_adaptive": mujoco_adaptive_factory,
    "mujoco_fixed": mujoco_fixed_factory,
    "featherstone_fixed": featherstone_factory,
    "semi_implicit_fixed": semi_implicit_factory,
    "vbd_fixed": vbd_factory,
    "xpbd_fixed": xpbd_factory,
}
