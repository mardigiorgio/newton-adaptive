# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Registry of CENIC scenes for the bench / test / demo infrastructure.

Each :class:`SceneEntry` exposes:

- ``build_model`` / ``build_model_randomized`` — model constructors
- ``solver_factories`` — dict ``{kind: builder}`` where ``builder(model)``
  returns a ``(solver, step_fn)`` pair. Bench iterates over this dict to
  plot one curve per supported solver kind.
- ``make_solver`` / ``make_fixed_solver`` — legacy single-solver factories
  kept for demo backward compat. Prefer ``solver_factories`` for new code.

Scenes are imported lazily on first ``get(name)`` to keep startup low.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import newton
import newton.solvers


@dataclass
class SceneEntry:
    """Uniform interface to a CENIC scene module."""

    name: str
    module_path: str
    build_model: Callable[[int], newton.Model]
    build_model_randomized: Callable[..., newton.Model]
    dt_outer: float
    solver_factories: dict[str, Callable] = field(default_factory=dict)
    # Legacy single-solver constructors (still used by some demos).
    make_solver: Callable[..., newton.solvers.SolverMuJoCoAdaptive] | None = None
    make_fixed_solver: Callable[..., newton.solvers.SolverMuJoCo] | None = None
    bench_capable: bool = True
    notes: str = ""

    def solver_kinds(self) -> list[str]:
        """Names of all solver kinds this scene supports."""
        return list(self.solver_factories.keys())


def _load(module_path: str) -> SceneEntry:
    """Import a scene module and pack into a SceneEntry."""
    mod = importlib.import_module(module_path)
    name = module_path.rsplit(".", 1)[-1]
    return SceneEntry(
        name=name,
        module_path=module_path,
        build_model=mod.build_model,
        build_model_randomized=mod.build_model_randomized,
        dt_outer=mod.DT_OUTER,
        solver_factories=getattr(mod, "SOLVER_FACTORIES", {}),
        make_solver=getattr(mod, "make_solver", None),
        make_fixed_solver=getattr(mod, "make_fixed_solver", None),
    )


# --- Scene catalog --------------------------------------------------------

_BENCH_SCENES: dict[str, str] = {
    "contact_objects": "scripts.scenes.contact_objects",
    "falling_cylinder": "scripts.scenes.falling_cylinder",
    "falling_gripper": "scripts.scenes.falling_gripper",
    "anymal_clutter": "scripts.scenes.anymal_clutter",
}

_NON_BENCH_SCENES: dict[str, str] = {
    "franka_dish_rack": "scripts.scenes.franka_dish_rack",
}


_CACHE: dict[str, SceneEntry] = {}


def get(name: str) -> SceneEntry:
    """Resolve a scene by name. Imports the module on first call."""
    if name not in _CACHE:
        if name in _BENCH_SCENES:
            entry = _load(_BENCH_SCENES[name])
        elif name in _NON_BENCH_SCENES:
            entry = _load(_NON_BENCH_SCENES[name])
            entry.bench_capable = False
            entry.notes = (
                "Requires per-step helper calls (update_held_objects / "
                "update_franka_targets); not compatible with bench step_fn."
            )
        else:
            raise KeyError(
                f"Unknown scene {name!r}. Available: "
                f"{sorted(_BENCH_SCENES) + sorted(_NON_BENCH_SCENES)}"
            )
        _CACHE[name] = entry
    return _CACHE[name]


def bench_scenes() -> list[str]:
    """Names of scenes that are bench-compatible."""
    return list(_BENCH_SCENES)


def all_scenes() -> list[str]:
    """All known scenes (bench-compatible plus non-bench)."""
    return list(_BENCH_SCENES) + list(_NON_BENCH_SCENES)
