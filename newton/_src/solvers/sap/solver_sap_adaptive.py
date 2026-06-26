# SPDX-License-Identifier: Apache-2.0
"""Error-controlled (step-doubling) SAP solver: ``SolverSAPAdaptive``.

True error-controlled CENIC with the convex SAP contact solver as the inner
step, under the SAME Drake step-doubling controller used by
``SolverMuJoCoAdaptive``. The controller leaf kernels are single-sourced from
:mod:`newton._src.solvers.adaptive.controller_kernels`; only the thin even-tiling
orchestration body is copied here, with the two SAP-specific seams:

  1. ``_run_substep`` takes a SCALAR dt (``SolverSAP.step`` drives every world
     with one dt) and restores the contact-solve warm-start before each eval so
     the three step-doubling evals are independent (proven feasible: warm-start
     changes iteration count, not the converged velocity of the strictly convex
     SAP minimization).
  2. contacts are converted once per boundary iteration into a ``SapContacts``
     bundle reused across all three evals (error reflects integration error only).

Because ``SolverSAP.step`` takes a single scalar dt for all worlds, this solver
runs EVEN + GLOBAL tiling only: all worlds share one substep count N and one
inner dt = dt_outer / N. Per-world ragged dt is structurally inapplicable to SAP.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

import newton

from ..adaptive.controller_kernels import (
    _advance_sim_time,
    _apply_dt_cap,
    _boundary_advance,
    _broadcast_error,
    _calc_even_step,
    _finish_diverged_worlds,
    _inf_norm_state_error_kernel,
    _iter_count_increment,
    _rebase_time,
    _reduce_max_error,
    _reset_error_scalar,
    _reset_worlds,
    _select_even_dt,
    _select_float_kernel,
    _select_spatial_vector_kernel,
    _select_transform_kernel,
    _set_uniform_even_dt,
)

# sys.path is configured by the package __init__ before this module is imported.
from sim.solver_sap import SolverSAP  # noqa: E402
from sim.sap_runtime import (  # noqa: E402
    sap_contacts_from_newton,
    sap_control_from_newton,
    sap_model_from_newton,
    sap_state_from_newton,
)


class SolverSAPAdaptive:
    """Adaptive-step (step-doubling) SAP solver. EVEN + GLOBAL tiling only.

    Drop-in mirror of ``SolverMuJoCoAdaptive(model, ...)``: takes the Newton
    ``Model``, builds the ``SapModel`` + inner ``SolverSAP`` internally, and
    exposes ``step_dt`` / ``reset`` / ``diverged`` / ``dt`` / ``cumulative_substeps``.
    """

    def __init__(
        self,
        model,
        *,
        tol: float = 1e-3,
        dt_inner_init: float = 0.01,
        dt_inner_min: float = 1e-6,
        dt_inner_max: float | None = None,
        max_substeps: int = 256,
        max_rigid_contact: int = 128,
        max_iterations: int = 30,
        contact_preset_variant: str = "drake",
        line_search_variant: str = "armijo_decay",
        contact_tau_d: float = 0.01,
        **kwargs,
    ):
        self.model = model
        device = model.device
        wc = int(model.world_count)

        # ---- inner SAP solver + model ----
        self._sap_model = sap_model_from_newton(model)
        self._sap = SolverSAP(
            self._sap_model,
            max_rigid_contact=int(max_rigid_contact),
            max_iterations=int(max_iterations),
            contact_tau_d=float(contact_tau_d),
            contact_preset_variant=str(contact_preset_variant),
            line_search_variant=str(line_search_variant),
        )

        # ---- scratch SapStates (independent backing arrays) ----
        self._scratch_full = self._sap_model.state()
        self._scratch_mid = self._sap_model.state()
        self._scratch_double = self._sap_model.state()
        self._state_cur = self._sap_model.state()
        self._state_saved = self._sap_model.state()

        # ---- warm-start snapshot buffers (one wp.copy of dof_count each) ----
        self._wstart_v_flat = wp.clone(self._sap.contact_solve.v_flat)
        self._wstart_active = wp.clone(self._sap._contact_solve_v_guess_active)

        # ---- controller buffers (Newton-model-level; same as MuJoCo-adaptive) ----
        self._dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._ideal_dt = wp.full(wc, dt_inner_init, dtype=wp.float32, device=device)
        self._dt_half = wp.full(wc, dt_inner_init * 0.5, dtype=wp.float32, device=device)
        self._sim_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._next_time = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted = wp.zeros(wc, dtype=wp.bool, device=device)
        self._commit = wp.zeros(wc, dtype=wp.bool, device=device)
        self._diverged = wp.zeros(wc, dtype=wp.bool, device=device)
        self._last_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._accepted_error = wp.zeros(wc, dtype=wp.float32, device=device)
        self._N = wp.zeros(wc, dtype=wp.int32, device=device)
        self._error_scalar = wp.zeros(1, dtype=wp.float32, device=device)

        self._tol = float(tol)
        self._dt_min = float(dt_inner_min)
        self._dt_max = float(dt_inner_max) if dt_inner_max is not None else float("inf")
        self._dt_inner_init = float(dt_inner_init)
        self._max_substeps = int(max_substeps)
        self._divergence_threshold = float(1.0e9)
        self._full_world_mask = wp.full(wc, True, dtype=wp.bool, device=device)

        self._coords_per_world = int(model.joint_coord_count) // wc
        self._dofs_per_world = int(model.joint_dof_count) // wc
        self._bodies_per_world = int(model.body_count) // wc
        self._world_count = wc
        self._max_rigid_contact = int(max_rigid_contact)

        # Accuracy-metric scaling S = identity (per PI directive); overwrite after
        # construction for expert per-coordinate scales.
        self._state_scale = wp.array(
            np.ones((wc, self._coords_per_world), dtype=np.float32),
            dtype=wp.float32,
            device=device,
        )

        self._iteration_count_buf = wp.zeros(1, dtype=wp.int32, device=device)
        self._cum_iters = wp.zeros(1, dtype=wp.int32, device=device)

        # ---- own collision pipeline (fixed contact set across the 3 evals) ----
        self._pipeline = newton.CollisionPipeline(
            model,
            broad_phase="sap",
            rigid_contact_max=int(max_rigid_contact) * wc,
        )
        self._contacts = self._pipeline.contacts()
        # Newton State used only as the body_q source for collide() (SapState has
        # no particle_q); synced from _state_cur once per boundary iteration.
        self._collide_state = model.state()
        self._sap_contacts = None
        self._sap_control = None

        # ---- L1: solver-internal per-N CUDA-graph capture of the fixed-N loop ----
        # The even+global tiling yields a STATIC kernel sequence per substep count
        # n_max (and inner dt = dt_outer/n_max). At our world counts the SAP contact
        # solve is launch/host-bound (GPU ~60% util), so collapsing the n_max-body
        # Python launch loop into one captured graph replay recovers the idle + the
        # entire per-launch driver-dispatch cost. The one host sync
        # (int(self._N.numpy())) stays eager OUTSIDE the captured region.
        # Cache keyed by (n_max, dt_inner). Gated by NEWTON_SAP_ADAPTIVE_GRAPH (default
        # on) and CUDA device; CPU unit tests fall back to the eager loop.
        try:
            _is_cuda = bool(wp.get_device(device).is_cuda)
        except Exception:
            _is_cuda = False
        self._graph_enabled = (
            _is_cuda and os.environ.get("NEWTON_SAP_ADAPTIVE_GRAPH", "1") != "0"
        )
        self._graph_cache: dict = {}
        # Modules/allocations must be warm before capture (a launch that triggers a
        # lazy module load syncs the stream and aborts capture). Run the FIRST step_dt
        # eagerly, then capture from the second frame on.
        self._graph_warmed = False

    # ---------------------------------------------------------------- properties
    @property
    def diverged(self) -> wp.array:
        return self._diverged

    @property
    def dt(self) -> wp.array:
        return self._dt

    @property
    def sim_time(self) -> wp.array:
        return self._sim_time

    @property
    def last_error(self) -> wp.array:
        return self._accepted_error

    @property
    def accepted(self) -> wp.array:
        return self._accepted

    @property
    def tiling(self) -> str:
        return "even"

    @property
    def contacts(self):
        return self._contacts

    def cumulative_substeps(self) -> int:
        """Total SAP opt-steps since the last reset_compute_counter (= iterations * 3)."""
        return int(self._cum_iters.numpy()[0]) * 3

    def get_max_contact_count(self) -> int:
        """Per-batch rigid-contact capacity (for manager-level sensor buffer sizing)."""
        return self._max_rigid_contact * self._world_count

    def update_contacts(self, contacts, state) -> None:
        """No-op: SAP-adaptive owns its internal contact set; contact-sensor
        writeback from SAP is not yet wired (documented limitation for v1)."""
        return None

    @property
    def cumulative_iterations(self) -> wp.array:
        return self._cum_iters

    def reset_compute_counter(self) -> None:
        self._cum_iters.fill_(0)

    def notify_model_changed(self, flags: int) -> None:
        """Forward model-change notifications to the inner SAP solver.

        The Isaac Lab manager calls this on the first step after env reset/randomization
        (newton_manager.step iterates pending model changes). The controller's own state
        is per-world scalars (dt/sim_time/latches) unaffected by model-array changes, so
        we only refresh the inner SolverSAP's topology-dependent caches.
        """
        self._sap.notify_model_changed(flags)

    # ----------------------------------------------------------- state copy utils
    @staticmethod
    def _copy_state(dst, src) -> None:
        wp.copy(dst.joint_q, src.joint_q)
        wp.copy(dst.joint_qd, src.joint_qd)
        if src.body_q is not None and dst.body_q is not None:
            wp.copy(dst.body_q, src.body_q)
        if src.body_qd is not None and dst.body_qd is not None:
            wp.copy(dst.body_qd, src.body_qd)

    # ---------------------------------------------------------- warm-start seam
    def _save_warmstart(self) -> None:
        wp.copy(self._wstart_v_flat, self._sap.contact_solve.v_flat)
        wp.copy(self._wstart_active, self._sap._contact_solve_v_guess_active)

    def _restore_warmstart(self) -> None:
        wp.copy(self._sap.contact_solve.v_flat, self._wstart_v_flat)
        wp.copy(self._sap._contact_solve_v_guess_active, self._wstart_active)

    def _run_substep(self, state_in, state_out, dt: float) -> None:
        """One SAP step_in->out at SCALAR dt. Restores the warm-start first so the
        three step-doubling evals from one input state are independent."""
        self._restore_warmstart()
        self._sap.step(state_in, state_out, self._sap_control, self._sap_contacts, float(dt))

    # ----------------------------------------------------------- iteration body
    def _run_even_iteration_body(self, dt_inner: float) -> None:
        """One even-tiling substep at shared scalar ``dt_inner`` (global tiling):
        snapshot -> save warm-start -> 3 evals -> error -> global reduce/broadcast ->
        even controller -> commit-gated select rollback -> advance."""
        n = self._world_count
        dev = self.model.device
        dt_half = dt_inner * 0.5

        wp.launch(_iter_count_increment, dim=1, inputs=[self._iteration_count_buf], device=dev)
        wp.launch(_iter_count_increment, dim=1, inputs=[self._cum_iters], device=dev)

        # Snapshot _state_cur -> _state_saved for the commit-gated rollback.
        self._copy_state(self._state_saved, self._state_cur)

        # Save warm-start ONCE so all three evals restore to the same guess.
        self._save_warmstart()

        # 3 SAP evals: full dt, half dt, half dt from mid.
        self._run_substep(self._state_cur, self._scratch_full, dt_inner)
        self._run_substep(self._state_cur, self._scratch_mid, dt_half)
        self._run_substep(self._scratch_mid, self._scratch_double, dt_half)

        wp.launch(
            _inf_norm_state_error_kernel,
            dim=n,
            inputs=[
                self._scratch_full.joint_q,
                self._scratch_double.joint_q,
                self._state_scale,
                self._coords_per_world,
            ],
            outputs=[self._last_error],
            device=dev,
        )

        # Global tiling: collapse per-world error to the worst case, broadcast back
        # so the controller produces one shared decision driving the shared N.
        wp.launch(_reset_error_scalar, dim=1, inputs=[self._error_scalar], device=dev)
        wp.launch(_reduce_max_error, dim=n, inputs=[self._last_error, self._error_scalar], device=dev)
        wp.launch(_broadcast_error, dim=n, inputs=[self._error_scalar, self._last_error], device=dev)

        wp.launch(
            _calc_even_step,
            dim=n,
            inputs=[
                self._last_error,
                self._dt,
                self._ideal_dt,
                self._accepted,
                self._commit,
                self._diverged,
                self._tol,
                self._dt_min,
                self._divergence_threshold,
            ],
            device=dev,
        )

        # State select: committed worlds get scratch_double, the rest hold saved.
        wp.launch(
            _select_float_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self._scratch_double.joint_q, self._state_saved.joint_q, self._commit, self._coords_per_world],
            outputs=[self._state_cur.joint_q],
            device=dev,
        )
        wp.launch(
            _select_float_kernel,
            dim=self.model.joint_dof_count,
            inputs=[self._scratch_double.joint_qd, self._state_saved.joint_qd, self._commit, self._dofs_per_world],
            outputs=[self._state_cur.joint_qd],
            device=dev,
        )
        if self._state_cur.body_q is not None:
            wp.launch(
                _select_transform_kernel,
                dim=self.model.body_count,
                inputs=[self._scratch_double.body_q, self._state_saved.body_q, self._commit, self._bodies_per_world],
                outputs=[self._state_cur.body_q],
                device=dev,
            )
        if self._state_cur.body_qd is not None:
            wp.launch(
                _select_spatial_vector_kernel,
                dim=self.model.body_count,
                inputs=[self._scratch_double.body_qd, self._state_saved.body_qd, self._commit, self._bodies_per_world],
                outputs=[self._state_cur.body_qd],
                device=dev,
            )

        wp.launch(
            _advance_sim_time,
            dim=n,
            inputs=[self._sim_time, self._dt, self._accepted, self._last_error, self._accepted_error],
            device=dev,
        )
        wp.launch(
            _finish_diverged_worlds,
            dim=n,
            inputs=[self._sim_time, self._next_time, self._diverged],
            device=dev,
        )

    # ------------------------------------------------------- fixed-N substep loop
    def _run_substep_loop(self, n_max: int, dt_inner: float) -> None:
        """Execute the n_max even+global substeps. When graph capture is enabled and
        warm, capture the fixed-N body sequence once per (n_max, dt_inner) and replay
        it with ``wp.capture_launch`` (driver-speed dispatch); otherwise run eagerly.

        Capture safety: during ``wp.ScopedCapture`` kernel launches are RECORDED, not
        executed -- the subsequent ``capture_launch`` runs exactly n_max bodies (no
        overshoot). All scratch/warm-start buffers are pre-allocated; the per-frame
        conversion wrappers reference the same underlying arrays, so the captured graph
        stays valid across frames (collide/control refill those arrays eagerly upstream).
        """
        if not (self._graph_enabled and self._graph_warmed):
            for _ in range(n_max):
                self._run_even_iteration_body(dt_inner)
            self._graph_warmed = True
            return

        key = (int(n_max), round(float(dt_inner), 12))
        graph = self._graph_cache.get(key)
        if graph is None:
            try:
                with wp.ScopedCapture() as cap:
                    for _ in range(n_max):
                        self._run_even_iteration_body(dt_inner)
                graph = cap.graph
                self._graph_cache[key] = graph
            except Exception:
                # Capture failed (e.g. an unexpected sync); fall back to eager for this
                # frame and disable capture to avoid repeated failures. Recorded ops are
                # NOT executed on a failed capture, so re-running eagerly is correct.
                self._graph_enabled = False
                for _ in range(n_max):
                    self._run_even_iteration_body(dt_inner)
                return
        wp.capture_launch(graph)

    # ---------------------------------------------------------------- step_dt
    def step_dt(self, dt_outer: float, state_0, state_1, control, apply_forces=None):
        """Advance all worlds by exactly ``dt_outer`` of sim time via even+global
        step-doubling SAP. ``state_0`` (Newton State) is read and written in place.
        """
        device = self.model.device
        n = self._world_count

        self._sap_control = sap_control_from_newton(control)
        effective_dt_max = min(self._dt_max, dt_outer)

        wp.launch(
            _apply_dt_cap,
            dim=n,
            inputs=[self._ideal_dt, self._dt_min, effective_dt_max, self._dt, self._dt_half],
            device=device,
        )

        # Load the incoming Newton state into the internal SapState working buffer.
        self._copy_state(self._state_cur, sap_state_from_newton(state_0))

        if apply_forces is not None:
            apply_forces(state_0)

        # Fix B time-rebase, then advance the per-world boundary by dt_outer.
        wp.launch(_rebase_time, dim=n, inputs=[self._sim_time, self._next_time], device=device)
        wp.launch(_boundary_advance, dim=n, inputs=[self._next_time, dt_outer], device=device)

        self._iteration_count_buf.fill_(0)
        self._diverged.fill_(False)

        # Detect contacts once per boundary using _state_cur's body transforms,
        # reused across all 3 evals of every substep (cartpole: empty -> no-op).
        wp.copy(self._collide_state.body_q, self._state_cur.body_q)
        self._pipeline.collide(self._collide_state, self._contacts)
        self._sap_contacts = sap_contacts_from_newton(self._contacts)

        # Even+global: choose one shared N from the carried ideal_dt, uniform dt.
        wp.launch(
            _select_even_dt,
            dim=n,
            inputs=[self._ideal_dt, dt_outer, self._dt_min, effective_dt_max,
                    self._max_substeps, self._N, self._dt, self._dt_half],
            device=device,
        )
        n_max = int(self._N.numpy().max())
        n_max = max(n_max, 1)
        wp.launch(
            _set_uniform_even_dt,
            dim=n,
            inputs=[n_max, dt_outer, self._N, self._dt, self._dt_half],
            device=device,
        )

        dt_inner = float(dt_outer) / float(n_max)
        self._run_substep_loop(n_max, dt_inner)

        # Write the result back into the Newton state_0.
        self._copy_state(sap_state_from_newton(state_0), self._state_cur)
        return state_0, state_1

    # ------------------------------------------------------------------- reset
    def reset(self, state, world_mask: wp.array | None = None, flags=0) -> None:
        """Restore per-world controller state for reset worlds (Fix C) and clear the
        SAP contact-solve warm-start at the boundary."""
        mask = self._full_world_mask if world_mask is None else world_mask
        self._sap.reset_runtime_state()
        self._save_warmstart()
        wp.launch(
            _reset_worlds,
            dim=self._world_count,
            inputs=[
                mask,
                self._dt_inner_init,
                self._ideal_dt,
                self._dt,
                self._dt_half,
                self._sim_time,
                self._next_time,
                self._diverged,
                self._accepted,
            ],
            device=self.model.device,
        )
