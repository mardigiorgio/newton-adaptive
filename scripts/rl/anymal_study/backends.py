"""Physics-backend abstraction: adaptive vs stock fixed-step, same model.

The whole A/B rests on driving identical worlds, control, and gains through
either ``SolverMuJoCoAdaptive.step_dt`` or a fixed-step ``SolverMuJoCo`` substep
loop behind one ``advance()`` call, so the only thing that differs between
training runs is the integrator.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton


@dataclass
class BackendSpec:
    """Solver configuration for one physics backend.

    Attributes:
        kind: ``"adaptive"``, ``"fixed"`` (stock fixed-step),
            ``"ref_tol"`` (tight-tolerance adaptive reference), or ``"ref_dt"``
            (tiny fixed-dt reference). The ``ref_*`` kinds are eval-only
            high-fidelity backends for the transfer matrix.
        tol: Adaptive inf-norm error tolerance [m or rad].
        dt_inner_init: Adaptive initial inner timestep [s].
        dt_inner_min: Adaptive minimum inner timestep [s] (must be < dt_inner_init).
        dt_inner_max: Adaptive maximum inner timestep [s].
        dt_mode: ``"per_world"`` or ``"global"``.
        fixed_dt: Inner substep for the fixed-step family [s].
        solver: MuJoCo constraint solver (``"newton"`` or ``"cg"``).
        ls_iterations: Linesearch iterations.
        njmax / nconmax: MuJoCo constraint/contact capacities.
    """

    kind: str = "adaptive"
    # adaptive / ref_tol
    tol: float = 1e-3
    dt_inner_init: float = 5e-3
    dt_inner_min: float = 2e-7
    dt_inner_max: float = 0.02
    dt_mode: str = "per_world"
    # fixed / ref_dt
    fixed_dt: float = 5e-3
    # shared SolverMuJoCo kwargs
    solver: str = "newton"
    ls_iterations: int = 50
    njmax: int = 50
    nconmax: int = 100

    @property
    def is_adaptive(self) -> bool:
        return self.kind in ("adaptive", "ref_tol")


# Canonical high-fidelity reference backends for the transfer matrix (§5).
REF_SPECS: dict[str, BackendSpec] = {
    "ref_tol": BackendSpec(kind="ref_tol", tol=1e-5, dt_inner_min=5e-8, dt_inner_max=2e-3),
    "ref_dt": BackendSpec(kind="ref_dt", fixed_dt=5e-4),
}


class Backend:
    """Uniform ``advance()`` over adaptive ``step_dt`` and a fixed-step substep loop.

    Args:
        model: Finalized Newton model (already replicated to ``num_worlds``).
        spec: Backend configuration.
    """

    def __init__(self, model, spec: BackendSpec):
        self.spec = spec
        self.model = model
        if spec.is_adaptive:
            self.solver = newton.solvers.SolverMuJoCoAdaptive(
                model,
                tol=spec.tol,
                dt_inner_init=spec.dt_inner_init,
                dt_inner_min=spec.dt_inner_min,
                dt_inner_max=spec.dt_inner_max,
                dt_mode=spec.dt_mode,
                njmax=spec.njmax,
                nconmax=spec.nconmax,
                solver=spec.solver,
                ls_iterations=spec.ls_iterations,
            )
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                model,
                solver=spec.solver,
                ls_iterations=spec.ls_iterations,
                njmax=spec.njmax,
                nconmax=spec.nconmax,
            )

    @property
    def is_adaptive(self) -> bool:
        return self.spec.is_adaptive

    def advance(self, control_dt: float, s0, s1, control, apply_forces=None):
        """Advance every world by exactly ``control_dt`` seconds.

        Returns the (state_0, state_1) pair. The live (post-step) state is the
        FIRST element for both backends; callers must rebind tensor views from
        it because the fixed backend ping-pongs the two states internally.
        """
        if self.is_adaptive:
            # The adaptive solver owns the substep loop, clear_forces, and apply_forces order.
            return self.solver.step_dt(control_dt, s0, s1, control, apply_forces=apply_forces)

        # Fixed-step: integer number of substeps per control tick.
        n = max(1, round(control_dt / self.spec.fixed_dt))
        for _ in range(n):
            s0.clear_forces()
            if apply_forces is not None:
                apply_forces(s0)
            self.solver.step(s0, s1, control, None, self.spec.fixed_dt)
            s0, s1 = s1, s0
        return s0, s1

    def diagnostics(self) -> dict | None:
        """Adaptive-only adaptive-stepping diagnostics. Reads device arrays — call
        OUTSIDE the hot path (e.g. gated logging only)."""
        if not self.is_adaptive:
            return None
        return self.solver.get_status_summary()


def make_backend(model, spec: BackendSpec) -> Backend:
    return Backend(model, spec)
