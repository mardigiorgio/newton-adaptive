# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Quantile boundary stop shared by the step-doubling adaptive solvers.

Both :class:`~newton.solvers.SolverMuJoCoAdaptive` and
:class:`~newton.solvers.SolverSAPAdaptive` march a per-world adaptive timestep until every
world reaches the control boundary. That makes the loop length the MAXIMUM over worlds of
the per-world attempt count, and every world pays for it: a batched step costs the same
whether all worlds need it or one does, so worlds that landed early are re-stepped at full
price while the slowest finishes.

The maximum grows with world count; a quantile does not. Stopping once ``landed_fraction``
of worlds have arrived therefore makes the loop length -- and the cost -- independent of
batch size. Measured on an Allegro in-hand scene at 4096 worlds, ``landed_fraction=0.95``
cut the loop from 20.3 attempts to 2.2 and the straggler waste from 10.4x to 1.0x.

The abandoned tail is not left behind: :meth:`QuantileBoundaryStop.force_complete` gives
each remaining world its whole boundary remainder in one unchecked step, so every world
lands at the exact right TIME and only that step's local error exceeds tolerance. Worlds
abandoned repeatedly are the ones already stuck in a pathological contact state; consumers
should latch them for termination rather than let them keep contributing.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import warp as wp


@wp.kernel
def count_unfinished(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    count: wp.array[wp.int32],
):
    """Accumulate the NUMBER of worlds short of their boundary into ``count[0]``."""
    i = wp.tid()
    if sim_time[i] < next_time[i]:
        wp.atomic_add(count, 0, 1)


@wp.kernel
def apply_quantile_threshold(
    count: wp.array[wp.int32],
    flag: wp.array[wp.int32],
    max_unfinished: int,
):
    """Set ``flag[0]`` to 1 only while MORE than ``max_unfinished`` worlds are short.

    A ``flag[0] >= 2`` status (SolverSAPAdaptive's non-converged inner solve) is left
    alone: abandoning stragglers must never mask a solver failure.
    """
    if flag[0] >= 2:
        return
    if count[0] <= max_unfinished:
        flag[0] = 0
    else:
        flag[0] = 1


@wp.kernel
def force_complete_dt(
    sim_time: wp.array[wp.float32],
    next_time: wp.array[wp.float32],
    dt: wp.array[wp.float32],
    dt_half: wp.array[wp.float32],
):
    """Give every world still short of its boundary the whole remaining span; others 0."""
    i = wp.tid()
    r = next_time[i] - sim_time[i]
    if r > wp.float32(0.0):
        dt[i] = r
        dt_half[i] = r * wp.float32(0.5)
    else:
        dt[i] = wp.float32(0.0)
        dt_half[i] = wp.float32(0.0)


@wp.kernel
def force_complete_land(sim_time: wp.array[wp.float32], next_time: wp.array[wp.float32]):
    """Land the forced worlds exactly on their boundary (time exact, accuracy degraded)."""
    i = wp.tid()
    if sim_time[i] < next_time[i]:
        sim_time[i] = next_time[i]


class QuantileBoundaryStop:
    """Per-solver state for the quantile stop and its forced-completion pass.

    ``landed_fraction = 1.0`` (the default) disables the whole mechanism: :attr:`enabled`
    is False, no extra kernel launches occur, and the owning solver keeps its
    wait-for-every-world behavior exactly.

    Args:
        world_count: Number of simulated worlds.
        device: Warp device the solver's arrays live on.
        landed_fraction: Fraction of worlds that must reach the boundary before the march
            may stop, in ``(0, 1]``.
        graph_enabled: Whether the owning solver permits CUDA-graph capture. The forced
            completion is captured when allowed -- an eager evaluation costs roughly 78x a
            replayed one at 4096 worlds, which is enough to erase the entire saving.
    """

    def __init__(
        self,
        world_count: int,
        device,
        landed_fraction: float = 1.0,
        graph_enabled: bool = True,
    ):
        if not (0.0 < float(landed_fraction) <= 1.0):
            raise ValueError(f"landed_fraction must be in (0, 1], got {landed_fraction!r}")
        self.landed_fraction = float(landed_fraction)
        self.max_unfinished = int(world_count * (1.0 - self.landed_fraction))
        self.world_count = int(world_count)
        self.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.force_accept = wp.zeros(1, dtype=wp.int32, device=device)
        self._graph_enabled = bool(graph_enabled)
        self._graph = None
        self._graph_failed = False

    @property
    def enabled(self) -> bool:
        """True when the stop actually abandons a tail (``landed_fraction < 1``)."""
        return self.max_unfinished > 0

    def mark_boundary(self, sim_time, next_time, flag, device) -> None:
        """Replace a wait-for-everyone boundary check with the quantile test.

        Counts the worlds short of their boundary and lowers ``flag[0]`` to 0 once at most
        ``max_unfinished`` remain. Call in place of the solver's own "any world short"
        kernel, after that solver has recorded any failure status into ``flag``.
        """
        self.count.zero_()
        wp.launch(count_unfinished, dim=self.world_count, inputs=[sim_time, next_time, self.count], device=device)
        wp.launch(apply_quantile_threshold, dim=1, inputs=[self.count, flag, self.max_unfinished], device=device)

    def force_complete(self, sim_time, next_time, dt, dt_half, eval_fn: Callable[[], None], device) -> None:
        """Land every abandoned world on its boundary with one unchecked step.

        A force-accepted step is unconditional, so it needs no error estimate and costs a
        single evaluation rather than a step-doubling attempt's three. ``eval_fn`` performs
        that one evaluation at the per-world ``dt`` this method writes.
        """

        def body():
            wp.launch(force_complete_dt, dim=self.world_count, inputs=[sim_time, next_time, dt, dt_half], device=device)
            eval_fn()
            wp.launch(force_complete_land, dim=self.world_count, inputs=[sim_time, next_time], device=device)

        if self._graph is not None:
            wp.capture_launch(self._graph)
            return
        if not self._graph_enabled or self._graph_failed:
            body()
            return
        try:
            with wp.ScopedCapture(device=device) as cap:
                body()
            self._graph = cap.graph
            wp.capture_launch(self._graph)
        except Exception as exc:
            self._graph_failed = True
            warnings.warn(
                f"QuantileBoundaryStop: forced-completion capture failed ({exc}); running it eagerly.",
                stacklevel=2,
            )
            body()
