# Empirical kernel-level verification of the adaptive controller (2026-07-01):
#   1. NaN detection in _inf_norm_state_error_kernel (FIX KEPT: wp.max/fmaxf drops NaN
#      operands, so the old kernel reported error~0 for NaN states and committed them).
#   2. Boundary-landing behavior of _calc_adjusted_step (FIX REVERTED after A/B: pins
#      the RETAINED behavior — an accepted landing sliver rewrites ideal_dt relative to
#      the clamped remainder. The Drake-style fix left iterations unchanged (the loop is
#      gated by the error-limited max-substep world) while raising wall time 5.2% at
#      2048 envs / tol=1e-3, because non-binding worlds ran larger, costlier solves.
#      The collapse acts as a free dt-limiter at batch scale.)
#
# Run: uv run python verify_kernel_fixes.py   (from the newton-adaptive repo)

import numpy as np
import warp as wp

from newton._src.solvers.mujoco.solver_mujoco_adaptive import (
    _calc_adjusted_step,
    _inf_norm_state_error_kernel,
)

wp.init()

TOL = 1.0e-3
DT_MIN = 1.0e-6
DIVERGENCE = 1.0e9

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  {status}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# The OLD error-kernel logic (pre-fix), copied verbatim, to demonstrate the bug is real.
@wp.kernel
def _old_inf_norm_kernel(
    qpos_full: wp.array2d[wp.float32],
    qpos_double: wp.array2d[wp.float32],
    state_scale: wp.array2d[wp.float32],
    nq: int,
    error_out: wp.array[wp.float32],
):
    world = wp.tid()
    max_err = float(0.0)
    for i in range(nq):
        d = wp.abs(qpos_double[world, i] - qpos_full[world, i])
        max_err = wp.max(max_err, state_scale[world, i] * d)
    if wp.isnan(max_err) or wp.isinf(max_err):
        max_err = float(1.0e10)
    error_out[world] = max_err


def run_error_kernel(kernel, qpos_double_rows, device):
    n = len(qpos_double_rows)
    nq = len(qpos_double_rows[0])
    full = wp.array(np.zeros((n, nq), dtype=np.float32), dtype=wp.float32, device=device)
    double = wp.array(np.asarray(qpos_double_rows, dtype=np.float32), dtype=wp.float32, device=device)
    scale = wp.array(np.ones((n, nq), dtype=np.float32), dtype=wp.float32, device=device)
    err = wp.zeros(n, dtype=wp.float32, device=device)
    wp.launch(kernel, dim=n, inputs=[full, double, scale, nq], outputs=[err], device=device)
    return err.numpy()


def run_controller(e, dt, ideal0, device):
    err = wp.array(np.float32([e]), dtype=wp.float32, device=device)
    dt_a = wp.array(np.float32([dt]), dtype=wp.float32, device=device)
    ideal = wp.array(np.float32([ideal0]), dtype=wp.float32, device=device)
    acc = wp.zeros(1, dtype=wp.bool, device=device)
    com = wp.zeros(1, dtype=wp.bool, device=device)
    div = wp.zeros(1, dtype=wp.bool, device=device)
    wp.launch(
        _calc_adjusted_step,
        dim=1,
        inputs=[err, dt_a, ideal, acc, com, div, TOL, DT_MIN, DIVERGENCE],
        device=device,
    )
    return bool(acc.numpy()[0]), bool(com.numpy()[0]), float(ideal.numpy()[0])


def main():
    devices = ["cpu"]
    if wp.is_cuda_available():
        devices.append("cuda:0")

    NAN, INF = float("nan"), float("inf")
    for dev in devices:
        print(f"\n=== device: {dev} ===")

        print("[1] NaN detection in the error kernel")
        old = run_error_kernel(_old_inf_norm_kernel, [[0.0, NAN, 0.0, 0.0]], dev)
        check("old kernel DROPS NaN (bug demo: error==0, would commit NaN)", old[0] == 0.0, f"err={old[0]:.3e}")
        new = run_error_kernel(_inf_norm_state_error_kernel, [[0.0, NAN, 0.0, 0.0]], dev)
        check("fixed kernel flags single-NaN component -> 1e10", new[0] == 1.0e10, f"err={new[0]:.3e}")
        new_all = run_error_kernel(_inf_norm_state_error_kernel, [[NAN, NAN, NAN, NAN]], dev)
        check("fixed kernel flags all-NaN row -> 1e10", new_all[0] == 1.0e10, f"err={new_all[0]:.3e}")
        new_inf = run_error_kernel(_inf_norm_state_error_kernel, [[0.0, INF, 0.0, 0.0]], dev)
        check("inf still detected -> 1e10", new_inf[0] == 1.0e10, f"err={new_inf[0]:.3e}")
        new_ok = run_error_kernel(_inf_norm_state_error_kernel, [[0.0, 2.0e-4, 1.0e-4, 0.0]], dev)
        check("healthy row unchanged (max|d|)", abs(new_ok[0] - 2.0e-4) < 1e-9, f"err={new_ok[0]:.3e}")

        print("[2] boundary-landing behavior (RETAINED after A/B: see header)")
        # A: accepted landing sliver rewrites ideal_dt relative to the clamped remainder
        #    (5x grow cap anchored to the sliver). Retained: acts as a free dt-limiter
        #    on non-binding worlds at batch scale.
        acc, com, ideal = run_controller(e=1e-8, dt=1e-4, ideal0=4e-3, device=dev)
        check(
            "accepted landing re-anchors ideal_dt to 5x remainder",
            acc and com and abs(ideal - 5e-4) < 1e-9,
            f"ideal={ideal:.3e}",
        )
        # B: rejected sliver shrinks (safety direction)
        acc, com, ideal = run_controller(e=0.1, dt=1e-4, ideal0=4e-3, device=dev)
        check("rejected landing shrinks", (not acc) and (not com) and ideal <= 1.1e-5, f"ideal={ideal:.3e}")
        # C: normal accept grows 5x
        acc, com, ideal = run_controller(e=1e-8, dt=1e-3, ideal0=1e-3, device=dev)
        check("normal accept grows 5x", acc and abs(ideal - 5e-3) < 1e-8, f"ideal={ideal:.3e}")
        # D: normal reject shrinks
        acc, com, ideal = run_controller(e=0.1, dt=1e-3, ideal0=1e-3, device=dev)
        check("normal reject shrinks", (not acc) and abs(ideal - 1e-4) < 1e-8, f"ideal={ideal:.3e}")
        # E: floor step over tol pins ideal_dt to dt_min (accept progress)
        acc, com, ideal = run_controller(e=10 * TOL, dt=DT_MIN, ideal0=4e-3, device=dev)
        check("floor step over tol pins to dt_min", acc and com and abs(ideal - DT_MIN) < 1e-12, f"ideal={ideal:.3e}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
