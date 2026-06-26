# SPDX-License-Identifier: Apache-2.0
"""Shared building blocks for error-controlled (step-doubling) adaptive solvers.

The leaf ``@wp.kernel`` controller primitives live in :mod:`controller_kernels`
so they can be single-sourced across the MuJoCo and SAP adaptive solvers (the
Drake CalcAdjustedStepSize controller, the inf-norm accuracy metric, the
commit-gated state-select rollback, the even-tiling N-selection, and the
Fix A/B/C latches). Both ``SolverMuJoCoAdaptive`` and ``SolverSAPAdaptive``
import them, so the subtle controller logic never drifts between the two.
"""
