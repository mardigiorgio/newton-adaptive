# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture-safe chunked prefix scan for int32 arrays.

``wp.utils.array_scan`` allocates temporary device memory on every call,
which a conditional CUDA-graph body (a launch stream recorded inside
``wp.capture_if`` / ``wp.capture_while``) cannot contain. The three
fixed-dim kernels here compute exactly the same integer prefix sums with no
per-call allocation: per-chunk sums, a serial scan of the (small) chunk
sums, then a per-chunk emit. Integer arithmetic, so outputs are
byte-identical to the scan they replace.

Scratch buffers are cached per (device, chunk count) at first use; callers
must run one eager pass before recording (the collision pipeline's boundary
pass does) so the cache is warm when a capture records the launches.
"""

from __future__ import annotations

import warp as wp

SCAN_CHUNK = 256

_scratch: dict = {}


@wp.kernel(enable_backward=False)
def _cs_scan_chunk_reduce(
    vals: wp.array[wp.int32],
    n: int,
    chunk: int,
    chunk_sums: wp.array[wp.int32],
):
    b = wp.tid()
    lo = b * chunk
    hi = wp.min(lo + chunk, n)
    s = int(0)
    for i in range(lo, hi):
        s = s + vals[i]
    chunk_sums[b] = s


@wp.kernel(enable_backward=False)
def _cs_scan_chunk_offsets(
    chunk_sums: wp.array[wp.int32],
    n_chunks: int,
    chunk_offsets: wp.array[wp.int32],
):
    """Serial exclusive scan over the (small) per-chunk sums (dim=1)."""
    acc = int(0)
    for b in range(n_chunks):
        chunk_offsets[b] = acc
        acc = acc + chunk_sums[b]


@wp.kernel(enable_backward=False)
def _cs_scan_chunk_emit_inclusive(
    vals: wp.array[wp.int32],
    n: int,
    chunk: int,
    chunk_offsets: wp.array[wp.int32],
    out: wp.array[wp.int32],
):
    b = wp.tid()
    lo = b * chunk
    hi = wp.min(lo + chunk, n)
    acc = chunk_offsets[b]
    for i in range(lo, hi):
        acc = acc + vals[i]
        out[i] = acc


@wp.kernel(enable_backward=False)
def _cs_scan_chunk_emit_exclusive(
    vals: wp.array[wp.int32],
    n: int,
    chunk: int,
    chunk_offsets: wp.array[wp.int32],
    out: wp.array[wp.int32],
):
    """Exclusive emit; safe for in-place use (``out is vals``): each element
    is read before its slot is overwritten within the same serial loop."""
    b = wp.tid()
    lo = b * chunk
    hi = wp.min(lo + chunk, n)
    acc = chunk_offsets[b]
    for i in range(lo, hi):
        v = vals[i]
        out[i] = acc
        acc = acc + v


def capture_safe_scan_int32(
    vals: wp.array,
    out: wp.array,
    *,
    inclusive: bool,
    device=None,
    record_tape: bool = False,
) -> None:
    """Prefix-scan ``vals`` into ``out`` (int32, same length; in-place OK for
    the exclusive form) with no per-call allocation after first use."""
    n = int(vals.shape[0])
    if n == 0:
        return
    if device is None:
        device = vals.device
    n_chunks = (n + SCAN_CHUNK - 1) // SCAN_CHUNK
    key = (str(device), n_chunks)
    scratch = _scratch.get(key)
    if scratch is None:
        scratch = (
            wp.zeros(n_chunks, dtype=wp.int32, device=device),
            wp.zeros(n_chunks, dtype=wp.int32, device=device),
        )
        _scratch[key] = scratch
    sums, offsets = scratch
    wp.launch(
        kernel=_cs_scan_chunk_reduce,
        dim=n_chunks,
        inputs=[vals, n, SCAN_CHUNK, sums],
        device=device,
        record_tape=record_tape,
    )
    wp.launch(
        kernel=_cs_scan_chunk_offsets,
        dim=1,
        inputs=[sums, n_chunks, offsets],
        device=device,
        record_tape=record_tape,
    )
    wp.launch(
        kernel=_cs_scan_chunk_emit_inclusive if inclusive else _cs_scan_chunk_emit_exclusive,
        dim=n_chunks,
        inputs=[vals, n, SCAN_CHUNK, offsets, out],
        device=device,
        record_tape=record_tape,
    )
