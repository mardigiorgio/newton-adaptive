# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Boundary-exit check shared by the step-doubling adaptive solvers.

Both :class:`~newton.solvers.SolverMuJoCoAdaptive` and
:class:`~newton.solvers.SolverSAPAdaptive` march a per-world adaptive timestep
until EVERY world reaches the control boundary: full-integrity bookkeeping,
no abandonment, no forced completion — every committed step satisfied the
error test (or a floor/divergence path that never commits garbage), so the
per-world error budget holds for the whole batch. Tail cost is bounded by
tail compaction (work restricted to still-active worlds), never by giving up
on stragglers.

Usage: reset the flag to 0 each iteration, launch the exit kernel (dim =
world_count) so any world short of its boundary raises the flag, then apply
the ``max_substeps`` safety cap LAST so the cap always wins. A solver whose
inner solve reports convergence status has two exits for a failed solve:
:func:`mark_unfinished_with_status` folds it into the same one-int32 poll
(status 2 = "converge or throw", no extra host sync), while
:func:`mark_unfinished_contained` keeps the failure per-world (the
controller's own reject/shrink/latch machinery absorbs it) and only latches
a sticky per-boundary side flag plus a per-world event counter for the
caller's rare-path readout.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def mark_unfinished(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    flag: wp.array[wp.int32],
):
    """Raise ``flag[0]`` to 1 while any world is short of its boundary."""
    i = wp.tid()
    if sim_time[i] < next_time[i]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def mark_unfinished_with_status(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    solve_ok: wp.array[int],
    flag: wp.array[wp.int32],
):
    """Boundary-exit check that also folds a per-world solve status.

    ``flag[0]``: 0 done, 1 unfinished, 2 solve failed. Status 2 dominates
    (atomic max) so a non-converged inner solve is read back on the same
    one-int32 poll the loop already performs ("converge or throw" without
    another host sync).
    """
    i = wp.tid()
    if solve_ok[i] == 0:
        wp.atomic_max(flag, 0, 2)
    elif sim_time[i] < next_time[i]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def mark_unfinished_with_status_target(
    sim_time: wp.array[wp.float32],
    target: wp.array[wp.float32],
    solve_ok: wp.array[int],
    flag: wp.array[wp.int32],
):
    """:func:`mark_unfinished_with_status` against a per-call scalar target.

    ``target[0]`` (device, written by the caller at boundary open -- graph-replay
    safe) replaces the per-world ``next_time`` in the finish test, so a world
    whose own boundary clock has RUN AHEAD of the call target still counts as
    finished for this call. Used by the SAP run-ahead march, where per-world
    boundary targets advance inside the call while the call itself only owes
    the batch ``target[0]`` of sim time. The solve-status fold is identical to
    the per-world kernel's.
    """
    i = wp.tid()
    if solve_ok[i] == 0:
        wp.atomic_max(flag, 0, 2)
    elif sim_time[i] < target[0]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def mark_unfinished_contained_target(
    sim_time: wp.array[wp.float32],
    target: wp.array[wp.float32],
    solve_ok: wp.array[int],
    fail_world: wp.array[wp.int32],
    flag: wp.array[wp.int32],
):
    """:func:`mark_unfinished_contained` against a per-call scalar target.

    Same containment semantics (failures stay per-world; ``flag[1]`` sticky
    latch + per-world event counter), with the finish test taken against the
    device scalar ``target[0]`` instead of per-world ``next_time`` -- see
    :func:`mark_unfinished_with_status_target` for why the run-ahead march
    needs the scalar form.
    """
    i = wp.tid()
    if solve_ok[i] == 0:
        fail_world[i] = fail_world[i] + 1
        wp.atomic_max(flag, 1, 1)
    if sim_time[i] < target[0]:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def mark_unfinished_contained(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    solve_ok: wp.array[int],
    fail_world: wp.array[wp.int32],
    flag: wp.array[wp.int32],
):
    """Boundary-exit check that CONTAINS a per-world solve failure.

    A non-converged inner solve is not escalated to a batch-fatal status:
    the failing world's error estimate was already forced to the divergence
    sentinel, so its own controller rejects the attempt (the committed state
    is untouched by construction) and retries at a shrunken dt, latching the
    world ``diverged`` if the failure persists to the dt floor -- the same
    per-world path a NaN state takes. ``flag[0]`` therefore carries only
    0 (done) / 1 (unfinished): a failing world short of its boundary is
    simply unfinished, and a floor-latched world froze its clock to the
    boundary and exits the march like any landed world.

    ``flag[1]`` is a sticky per-boundary failure latch (the caller zeroes it
    at boundary open, never per iteration) so the single post-march host
    read still learns a failure happened even when every failing world
    recovered or latched before the final iteration. ``fail_world``
    accumulates per-world failure events (one per failing solve attempt,
    never reset here) for the rare-path host readout that names the worlds.
    """
    i = wp.tid()
    if solve_ok[i] == 0:
        fail_world[i] = fail_world[i] + 1
        wp.atomic_max(flag, 1, 1)
    if sim_time[i] < next_time[i]:
        wp.atomic_max(flag, 0, 1)
