"""Contract tests for the mjw step allocation cache (see mjw_alloc_cache.py).

The cache must return the SAME buffer for the same call site across steps (so a
captured conditional body contains no allocation nodes) while preserving each
function's content contract (zeros zeroed, full refilled, clone copied) and never
aliasing distinct live buffers within one step (occurrence indexing).

Pure CPU: no GPU / MuJoCo needed.
"""

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.mjw_alloc_cache import MjwStepAllocCache

wp.init()

DEV = "cpu"


def _alloc_pair_same_line(cache):
    with cache.scope():
        a, b = wp.zeros(4, dtype=wp.float32, device=DEV), wp.zeros(4, dtype=wp.float32, device=DEV)
    return a, b


def test_same_site_reuses_buffer_across_steps():
    """The same call site returns the identical buffer on the second step (cache hit)."""
    cache = MjwStepAllocCache()
    ptrs = []
    for _ in range(3):
        with cache.scope():
            arr = wp.empty(8, dtype=wp.float32, device=DEV)
        ptrs.append(arr.ptr)
    assert ptrs[0] == ptrs[1] == ptrs[2], "same site must reuse one buffer"
    assert cache.misses == 1 and cache.hits == 2


def test_zeros_rezeroed_on_reuse():
    """A cached zeros buffer is re-zeroed even if the previous step dirtied it.

    Allocation calls go through a helper so both "steps" hit the SAME call site,
    exactly like a fixed line inside mujoco_warp does on every step.
    """
    cache = MjwStepAllocCache()

    def step_alloc():
        with cache.scope():
            return wp.zeros(4, dtype=wp.float32, device=DEV)

    arr = step_alloc()
    arr.fill_(7.0)
    arr2 = step_alloc()
    assert arr2.ptr == arr.ptr, "same site must reuse the buffer"
    assert np.all(arr2.numpy() == 0.0), "reused zeros buffer must be zeroed"


def test_full_refilled_with_current_value():
    """full() refills the cached buffer with THIS call's value (value not in key)."""
    cache = MjwStepAllocCache()

    def step_alloc(val):
        with cache.scope():
            return wp.full(shape=(3,), value=val, dtype=wp.int32, device=DEV)

    a = step_alloc(5)
    b = step_alloc(9)
    assert b.ptr == a.ptr, "value change must not allocate a new buffer"
    assert np.all(b.numpy() == 9)


def test_clone_copies_current_source():
    """clone() reuses the buffer but copies the CURRENT source contents."""
    cache = MjwStepAllocCache()
    src = wp.array(np.arange(4, dtype=np.float32), dtype=wp.float32, device=DEV)

    def step_alloc():
        with cache.scope():
            return wp.clone(src)

    a = step_alloc()
    src.fill_(3.0)
    b = step_alloc()
    assert b.ptr == a.ptr, "same site must reuse the buffer"
    assert np.all(b.numpy() == 3.0)


def test_same_line_twice_in_one_step_gets_distinct_buffers():
    """Two calls from ONE line within one step must not alias (occurrence index)."""
    cache = MjwStepAllocCache()
    a, b = _alloc_pair_same_line(cache)
    assert a.ptr != b.ptr, "live scratch buffers within a step must not alias"
    # and the pairing is stable across steps
    a2, b2 = _alloc_pair_same_line(cache)
    assert a2.ptr == a.ptr and b2.ptr == b.ptr


def test_requires_grad_bypasses_cache():
    """requires_grad allocations pass through untouched.

    Each call bypasses at least twice: warp's zeros() internally calls wp.empty
    (also patched), so the outer AND inner interceptions both bypass.
    """
    cache = MjwStepAllocCache()
    with cache.scope():
        a = wp.zeros(4, dtype=wp.float32, device=DEV, requires_grad=True)
        b = wp.zeros(4, dtype=wp.float32, device=DEV, requires_grad=True)
    assert a.ptr != b.ptr
    assert cache.bypasses >= 2 and cache.misses == 0


def test_originals_restored_outside_scope():
    """Outside the scope, wp.zeros allocates fresh buffers again."""
    cache = MjwStepAllocCache()
    with cache.scope():
        wp.zeros(4, dtype=wp.float32, device=DEV)
    x = wp.zeros(4, dtype=wp.float32, device=DEV)
    y = wp.zeros(4, dtype=wp.float32, device=DEV)
    assert x.ptr != y.ptr, "patching must not leak outside the scope"


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
