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

CUDA-graph parity: wall time is only comparable if every arm launches the
same way. SolverMuJoCoAdaptive captures its march internally; the other
three run eagerly when driven directly, paying per-kernel launch overhead
the MuJoCo-adaptive arm does not (measured: a captured adaptive boundary
timed BELOW one eager fixed substep on a small scene). So every boundary
here replays a captured graph — the first call runs eagerly to load
modules, the next captures, the rest replay — which is also how the
IsaacLab manager drives both ICF arms in training.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import warp as wp

import newton

from scripts.scenes.cenic_scenes import SCENES
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


class _CapturedBoundary:
    """Replay ``run(a, b, ctrl)`` as a captured CUDA graph, one graph per
    starting buffer. ``run`` advances the state held in ``a`` using ``b`` as
    scratch and leaves the result in ``b`` when ``ends_in_other`` else in
    ``a``; the caller keeps the returned (current, other) order."""

    def __init__(self, run: Callable, ends_in_other: bool):
        self._run = run
        self._ends_in_other = ends_in_other
        self._graphs: dict[int, object] = {}
        self._warm = False

    def __call__(self, s0, s1, ctrl):
        if not self._warm:
            self._run(s0, s1, ctrl)  # eager: compiles and loads every module
            self._warm = True
        else:
            graph = self._graphs.get(id(s0))
            if graph is None:
                with wp.ScopedCapture() as cap:
                    self._run(s0, s1, ctrl)
                graph = self._graphs[id(s0)] = cap.graph
            wp.capture_launch(graph)
        return (s1, s0) if self._ends_in_other else (s0, s1)


def _make_mujoco(model: newton.Model, n_sub: int) -> Arm:
    solver = newton.solvers.SolverMuJoCo(
        model, separate_worlds=True, nconmax=NCONMAX, njmax=NJMAX
    )
    contacts = model.contacts()
    dt = DT_OUTER / n_sub

    def run(a, b, ctrl):
        # step() writes state_out in place and returns None; ping-pong
        for _ in range(n_sub):
            solver.step(a, b, ctrl, contacts, dt)
            a, b = b, a

    return Arm("mujoco", solver, _CapturedBoundary(run, n_sub % 2 == 1), lambda: n_sub)


def _make_mujoco_adaptive(model: newton.Model, tol: float, max_substeps: int | None = None) -> Arm:
    extra = {"max_substeps": int(max_substeps)} if max_substeps else {}
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
        dt_mode="per_world",
        nconmax=NCONMAX,
        njmax=NJMAX,
        **extra,
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


def _icf_params(overrides: dict | None = None):
    return replace(_icf().IcfParams(), max_rigid_contact=ICF_MAX_RIGID_CONTACT, **(overrides or {}))


def _make_icf(model: newton.Model, n_sub: int, icf: dict | None = None) -> Arm:
    solver = _icf().SolverICF(model, params=_icf_params(icf))
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    dt = DT_OUTER / n_sub

    def run(a, b, ctrl):
        for _ in range(n_sub):
            pipeline.collide(a, contacts)
            solver.step(a, b, ctrl, contacts, dt)
            a, b = b, a

    return Arm("icf", solver, _CapturedBoundary(run, n_sub % 2 == 1), lambda: n_sub)


def _make_icf_adaptive(
    model: newton.Model, tol: float, icf_overrides: dict | None = None, max_substeps: int | None = None
) -> Arm:
    icf = _icf()
    extra = {"max_substeps": int(max_substeps)} if max_substeps else {}
    solver = icf.SolverICFAdaptive(
        model,
        params=_icf_params(icf_overrides),
        adaptive=icf.IcfAdaptiveParams(
            tol=tol,
            dt_inner_init=DT_OUTER,
            dt_inner_min=DT_INNER_MIN,
            dt_inner_max=DT_OUTER,
            **extra,
        ),
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    solver.attach_collision_pipeline(pipeline)

    def run(a, b, ctrl):
        # under the outer capture the march records as a conditional
        # while-node (the manager's mode); eagerly on the warm-up call
        pipeline.collide(a, contacts)
        solver.step(a, b, ctrl, contacts, DT_OUTER)

    def iterations() -> int:
        count = getattr(solver, "iteration_count", None)
        return int(count.numpy()[0]) if count is not None else -1

    return Arm("icf-adaptive", solver, _CapturedBoundary(run, True), iterations)


def make_arm(
    model: newton.Model,
    name: str,
    *,
    n_sub: int = 1,
    tol: float = 1e-3,
    scene: str | None = None,
    max_substeps: int | None = None,
) -> Arm:
    """Build one arm on ``model``. Fixed arms take ``n_sub`` (dt = DT_OUTER /
    n_sub), adaptive arms ``tol`` (the paper's accuracy eps_acc) and an
    optional ``max_substeps`` march budget. ``scene`` selects the ICF
    material overrides the scene declares (MuJoCo reads its materials from
    the shapes)."""
    icf = SCENES[scene].icf if scene in SCENES else None
    if name == "mujoco":
        return _make_mujoco(model, n_sub)
    if name == "mujoco-adaptive":
        return _make_mujoco_adaptive(model, tol, max_substeps)
    if name == "icf":
        return _make_icf(model, n_sub, icf)
    if name == "icf-adaptive":
        return _make_icf_adaptive(model, tol, icf, max_substeps)
    raise ValueError(f"unknown arm {name!r}; choose from {ARMS}")


@wp.kernel
def _or_bool_kernel(src: wp.array(dtype=wp.bool), acc: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if src[i]:
        acc[i] = 1


@wp.kernel
def _or_int_kernel(src: wp.array(dtype=wp.int32), acc: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if src[i] != 0:
        acc[i] = 1


class ExhaustionTracker:
    """Accumulates, on the device, whether an adaptive arm ever latched a
    world as diverged / exhausted its march budget — the paper's "solver
    failure" — without a host sync inside the timed loop. Call ``tick()``
    after every boundary; read ``fraction()`` once at the end."""

    def __init__(self, arm: Arm):
        self._srcs = []
        div = getattr(arm.solver, "diverged", None)
        if div is not None:
            self._srcs.append((_or_bool_kernel, div, wp.zeros(div.shape[0], dtype=wp.int32, device=div.device)))
        flag = getattr(arm.solver, "_boundary_flag", None)
        if flag is not None:
            self._srcs.append((_or_int_kernel, flag, wp.zeros(flag.shape[0], dtype=wp.int32, device=flag.device)))

    def tick(self) -> None:
        for kern, src, acc in self._srcs:
            wp.launch(kern, dim=src.shape[0], inputs=[src, acc], device=src.device)

    def fraction(self) -> float:
        """Max over sources of the fraction of entries that ever latched."""
        best = 0.0
        for _, _, acc in self._srcs:
            a = acc.numpy()
            best = max(best, float(a.mean()) if a.size else 0.0)
        return best


def build_model(n: int, seed: int = 42, scene: str = "contact-objects") -> newton.Model:
    """A benchmark scene at ``n`` worlds: one of the CENIC paper's scenes
    (``SCENES``) or the repo's randomized ``contact-objects`` pile."""
    if scene == "contact-objects":
        return build_model_randomized(n, seed=seed)
    return SCENES[scene].build(n)
