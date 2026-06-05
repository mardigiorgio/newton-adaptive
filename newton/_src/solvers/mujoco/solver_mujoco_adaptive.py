# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""SolverMuJoCoAdaptive -- adaptive-step MuJoCo solver.

v2 thin shim that delegates step-doubling, controller, and error control to
:class:`scripts.adaptive.factories.AdaptiveCompactionWrapper`.
"""

from __future__ import annotations

import warp as wp

import newton

from ...sim import Contacts
from .solver_mujoco import SolverMuJoCo


def _import_wrapper_bits():
    """Lazy import to avoid circular dependency at module load.

    ``scripts.adaptive`` lives outside the ``newton`` package and pulls in the
    public ``newton.solvers`` interface, which in turn re-exports this module.
    Importing at function call time avoids that cycle.
    """
    from scripts.adaptive.factories import (
        AdaptiveCompactionWrapper,
        _build_mujoco_q_weights,
    )

    return AdaptiveCompactionWrapper, _build_mujoco_q_weights


class SolverMuJoCoAdaptive(SolverMuJoCo):
    """Adaptive-step MuJoCo solver.

    Uses step doubling (3 MuJoCo evals per attempt) with a Drake-style PI
    controller to adapt the timestep per world.  Delegates all step-doubling
    work to :class:`scripts.adaptive.factories.AdaptiveCompactionWrapper`.

    Example:

    .. code-block:: python

        solver = newton.solvers.SolverMuJoCoAdaptive(model, tol=1e-3)
        state_0, state_1 = model.state(), model.state()

        while viewer.is_running():
            solver.step(state_0, state_1, control, None, DT)
            # state_0 updated in place; state_1 is unused scratch.
            viewer.render(state_0, solver.sim_time.numpy().min())
    """

    def __init__(
        self,
        model,
        *,
        tol: float = 1e-3,
        dt_init: float = 0.01,
        dt_min: float = 1e-6,
        dt_max: float | None = None,
        **kwargs,
    ):
        """
        Args:
            model: The model to simulate.
            tol: Inf-norm error tolerance on joint_q per world
                [m or rad, depending on joint type].  Error is ``max|Δq|``
                between the full step and the doubled half-step.  Worlds with
                error > tol are rejected and retry with a smaller dt.
            dt_init: Initial inner (adaptive physics) timestep [s].
            dt_min: Minimum allowed inner timestep [s].
            dt_max: Maximum allowed inner timestep [s].  If ``None``,
                defaults to ``dt_init``; per-call outer dt clamps the
                effective max so the inner step never overshoots the boundary.
            **kwargs: Forwarded to :class:`SolverMuJoCo`.
        """
        super().__init__(
            model,
            separate_worlds=True,
            use_mujoco_cpu=False,
            use_mujoco_contacts=False,
            **kwargs,
        )

        # Display-only attrs accessed directly by callers for banner strings.
        # ``_dt_max`` is a plain float; ``_tol`` is a plain float.
        self._tol = float(tol)
        self._dt_max = float(dt_max) if dt_max is not None else float(dt_init)

        # Stable buffer for ``mjw_model.opt.timestep``; per-substep
        # ``wp.copy()`` always targets a known warp array.
        # The wrapper expects this to already be in place.
        world_count = model.world_count
        device = model.device
        self._timestep_buf = wp.full(world_count, dt_init, dtype=wp.float32, device=device)
        self.mjw_model.opt.timestep = self._timestep_buf

        # Build the adaptive wrapper that owns the step-doubling loop.
        self._wrapper = self._build_wrapper(
            tol=tol,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=self._dt_max,
        )

    def _build_wrapper(self, *, tol, dt_init, dt_min, dt_max):
        """Construct the underlying AdaptiveCompactionWrapper with MuJoCo hooks."""
        AdaptiveCompactionWrapper, _build_mujoco_q_weights = _import_wrapper_bits()

        # Dedicated SAP collision pipeline sized to MJWarp's max contact count
        # (CENIC v1 lines 415-418).  This is the buffer callers read via
        # :attr:`contacts`; it must persist for the solver's lifetime.
        self._pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=self.mjw_data.naconmax,
        )
        self._contacts_start = self._pipeline.contacts()

        # Per-coord error weights from ``dof_invweight0`` (paper Sec V-E).
        q_weights = _build_mujoco_q_weights(self.model, self.mjw_model)

        def _step_fn(model_arg, state_in, state_out, ctrl, contacts_arg, dt_array, dt_scalar_buf):
            """One MuJoCo substep: sync state -> set per-world dt -> step -> read back.

            Matches :meth:`SolverMuJoCoAdaptive._run_substep` from v1 exactly.
            Contacts are already populated in MJWarp format by ``pre_iter_hook``
            (using state_cur transforms) -- this shim must not touch them, or
            the third substep would see different body transforms and corrupt
            the step-doubling error estimate.
            """
            self._update_mjc_data(self.mjw_data, model_arg, state_in)
            wp.copy(self.mjw_model.opt.timestep, dt_array)
            with wp.ScopedDevice(model_arg.device):
                self._mujoco_warp_step()
            self._update_newton_state(model_arg, state_out, self.mjw_data)

        def _pre_boundary_hook(model_arg, state_0, control, contacts_arg):
            """Once per outer step: apply control, enable RNE, run broad-phase
            collision detection.  Matches CENIC v1.step_dt lines 730-746."""
            self._apply_mjc_control(model_arg, state_0, control, self.mjw_data)
            self._enable_rne_postconstraint(state_0)
            if not self.mjw_model.opt.run_collision_detection:
                self._pipeline.collide(state_0, contacts_arg)

        def _pre_iter_hook(model_arg, state_cur, contacts_arg):
            """Once per inner iteration (before the 3 substeps): re-transform
            body-frame contacts to world frame using the current state.  Matches
            CENIC v1._run_iteration_body lines 469-471."""
            if not self.mjw_model.opt.run_collision_detection:
                self._convert_contacts_to_mjwarp(model_arg, state_cur, contacts_arg)

        return AdaptiveCompactionWrapper(
            model=self.model,
            step_fn=_step_fn,
            tol=tol,
            dt_init=dt_init,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_outer=dt_init,  # safe default; step_dt() overrides per call
            needs_collide=False,
            contacts=self._contacts_start,
            q_weights=q_weights,
            pre_boundary_hook=_pre_boundary_hook,
            pre_iter_hook=_pre_iter_hook,
        )

    # ----- Public API methods -----

    def step(self, state_in, state_out, control, contacts, dt):
        """Advance every world by exactly ``dt`` seconds of sim time.

        Matches the :class:`~newton.solvers.SolverBase` signature.  Updates
        ``state_in`` in place (the adaptive wrapper writes results back to its
        input buffer).  ``state_out`` is accepted for API compatibility but is
        unused by this solver.  Returns ``None``.

        Args:
            state_in: Current state (updated in place on return).
            state_out: Unused scratch buffer (accepted for API compatibility).
            control: Control inputs (applied once, persists across substeps).
            contacts: Ignored (the solver owns its internal collision pipeline).
            dt: Outer control/render period [s].
        """
        del contacts  # owned internally
        # The adaptive wrapper updates its first arg (state_0 / _state_cur)
        # in place, which is state_in here.  state_out is ignored by the
        # wrapper (it is the scratch buffer for the caller's double-buffer
        # rotation).  After this call, state_in holds the updated physics
        # state.  Callers rotate buffers themselves without a swap.
        self._wrapper.step_dt(dt, state_in, state_out, control)

    def get_status_summary(self) -> dict[str, float]:
        """Reduce per-world arrays to a 6-scalar summary via one GPU transfer."""
        return self._wrapper.status_summary()

    # ----- Public properties (forward to wrapper) -----

    @property
    def iteration_count(self) -> wp.array:
        """Iteration count from the most recent :meth:`step`, shape ``[1]``, int32, on device."""
        return self._wrapper._iteration_count_buf

    @property
    def dt(self) -> wp.array:
        """Current per-world timestep [s], shape ``[world_count]``, float32, on device."""
        return self._wrapper._dt

    @property
    def sim_time(self) -> wp.array:
        """Per-world simulation time [s], shape ``[world_count]``, float32, on device.

        Only advances for accepted steps.
        """
        return self._wrapper._sim_time

    @property
    def last_error(self) -> wp.array:
        """Inf-norm state error from the most recent accepted step,
        shape ``[world_count]``, float32, on device."""
        return self._wrapper._accepted_error

    @property
    def accepted(self) -> wp.array:
        """Per-world accept flags from the most recent step,
        shape ``[world_count]``, bool, on device."""
        return self._wrapper._accepted

    @property
    def contacts(self) -> Contacts:
        """Contacts from the most recent :meth:`step` boundary.

        Populated once per outer step by the solver's internal
        :class:`~newton.CollisionPipeline` and reused across all inner
        iterations.  Pass to ``viewer.log_contacts`` for rendering without
        duplicating the collision pass.
        """
        return self._contacts_start

    # v1 also exposed ``_dt`` as a wp.array (display-only callers do
    # ``solver._dt.numpy()[0]``).  Forward to the wrapper's ``_dt`` so that
    # surface remains usable.
    @property
    def _dt(self) -> wp.array:
        return self._wrapper._dt
