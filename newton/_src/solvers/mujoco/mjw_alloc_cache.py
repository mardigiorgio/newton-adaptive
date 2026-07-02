"""Scoped allocation cache for mujoco_warp's per-step scratch buffers.

CUDA conditional body graphs (``wp.capture_while``) may not contain memory-allocation
nodes ("Conditional body graph contains an unsupported operation (memory allocation)").
mujoco_warp allocates fixed-shape scratch on every ``step`` call -- the solver context
(``solver.py: create_solver_context``, ``step_size_cost``, ``nsolving``), implicitfast
integrator temporaries (``forward.py``), and narrowphase buffers
(``collision_convex.py``) -- which is legal inside REGULAR graph captures (alloc nodes)
but fatal inside a conditional body.

This shim makes a mjw step allocation-free after warmup WITHOUT copying or forking any
mujoco_warp code: for the duration of one step it monkeypatches ``wp.empty`` /
``wp.zeros`` / ``wp.full`` / ``wp.clone`` on the ``warp`` module and returns cached,
per-call-site buffers instead of allocating. The intercepted functions' contracts are
preserved on every call:

* ``wp.empty``  -> cached buffer as-is (contents undefined, same contract)
* ``wp.zeros``  -> cached buffer + ``zero_()``            (memset: records in any graph)
* ``wp.full``   -> cached buffer + ``fill_(value)``       (memset: records in any graph)
* ``wp.clone``  -> cached buffer + ``wp.copy(dst, src)``  (copy: records in any graph)

Cache keying is ``(fn, call site file:line, occurrence-within-step, shape/dtype/device
signature)``. The occurrence index means a site hit N times within one step gets N
distinct buffers, so live scratch never aliases. Reuse ACROSS steps is safe because
these are transient per-step buffers: persistent MuJoCo state lives in ``mjw_data``
(allocated at ``make_data``), never in these temporaries.

Known limitation: allocations made inside Warp's C++ core are invisible to Python-level
patching -- relevant only to ``wp.utils.array_scan`` / ``segmented_sort_pairs`` on the
SAP broadphase path. Scenes small enough for NXN broadphase (< 250k candidate pairs,
e.g. in-hand manipulation) never execute that path.

Requests with ``requires_grad=True`` or unhashable key components bypass the cache.
"""

from __future__ import annotations

import contextlib
import sys

import warp as wp

_PATCHED_FUNCS = ("empty", "zeros", "full", "clone")


def _freeze(value):
    """Best-effort hashable normalization of a key component (raises if impossible)."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    hash(value)
    return value


class MjwStepAllocCache:
    """Reusable-buffer allocator, activated around one mujoco_warp step via :meth:`scope`."""

    def __init__(self):
        self._cache: dict = {}
        self._site_counts: dict = {}
        self._orig: dict = {}
        self._active = False
        self.hits = 0
        self.misses = 0
        self.bypasses = 0

    # ------------------------------------------------------------------ scope
    @contextlib.contextmanager
    def scope(self):
        """Activate the cache for the duration of one mjw step (re-entrant safe)."""
        if self._active:
            yield
            return
        self._site_counts.clear()  # per-step occurrence indices
        self._orig = {name: getattr(wp, name) for name in _PATCHED_FUNCS}
        wp.empty = self._make_wrapper("empty")
        wp.zeros = self._make_wrapper("zeros")
        wp.full = self._make_wrapper("full")
        wp.clone = self._make_clone_wrapper()
        self._active = True
        try:
            yield
        finally:
            for name, fn in self._orig.items():
                setattr(wp, name, fn)
            self._active = False

    def clear(self) -> None:
        """Drop all cached buffers (e.g. after a model/data rebuild)."""
        self._cache.clear()

    # ------------------------------------------------------------------ internals
    def _next_site(self):
        """(file, line, occurrence-within-step) of the intercepted call."""
        f = sys._getframe(2)
        site = (f.f_code.co_filename, f.f_lineno)
        idx = self._site_counts.get(site, 0)
        self._site_counts[site] = idx + 1
        return site, idx

    def _make_wrapper(self, name: str):
        orig = self._orig[name]

        def wrapper(*args, **kwargs):
            if kwargs.get("requires_grad"):
                self.bypasses += 1
                return orig(*args, **kwargs)
            site, idx = self._next_site()
            try:
                if name == "full":
                    # value affects CONTENT, not identity: exclude it from the key and
                    # refill on every call (a varying value must not grow the cache).
                    if "value" in kwargs:
                        value = kwargs["value"]
                        key_args = args
                        key_kwargs = {k: v for k, v in kwargs.items() if k != "value"}
                    else:
                        value = args[1]
                        key_args = args[:1] + args[2:]
                        key_kwargs = kwargs
                    key = (name, site, idx, _freeze(key_args), _freeze(tuple(sorted(key_kwargs.items()))))
                else:
                    key = (name, site, idx, _freeze(args), _freeze(tuple(sorted(kwargs.items()))))
            except Exception:
                self.bypasses += 1
                return orig(*args, **kwargs)

            arr = self._cache.get(key)
            if arr is None:
                arr = orig(*args, **kwargs)  # orig initializes contents correctly
                self._cache[key] = arr
                self.misses += 1
                return arr
            self.hits += 1
            if name == "zeros":
                arr.zero_()
            elif name == "full":
                arr.fill_(value)
            return arr

        return wrapper

    def _make_clone_wrapper(self):
        orig = self._orig["clone"]
        copy_fn = wp.copy  # not patched; bind for clarity

        def wrapper(src, *args, **kwargs):
            if kwargs.get("requires_grad"):
                self.bypasses += 1
                return orig(src, *args, **kwargs)
            site, idx = self._next_site()
            try:
                key = (
                    "clone",
                    site,
                    idx,
                    tuple(src.shape),
                    str(src.dtype),
                    str(src.device),
                    _freeze(args),
                    _freeze(tuple(sorted(kwargs.items()))),
                )
            except Exception:
                self.bypasses += 1
                return orig(src, *args, **kwargs)

            arr = self._cache.get(key)
            if arr is None:
                arr = orig(src, *args, **kwargs)
                self._cache[key] = arr
                self.misses += 1
                return arr
            self.hits += 1
            copy_fn(arr, src)
            return arr

        return wrapper
