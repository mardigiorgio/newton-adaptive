"""Fast trap rig: manufacture the quantization-pinned-error trap on demand.

Oracle argument (no incubation needed): rigid-body dynamics are
translation-invariant, so an identical scene parked at a large world-frame
offset must cost the same to march -- the physics cannot tell the difference.
What DOES differ is the fp32 grid: at |x| ~ 8192 one ulp of the position
coordinate is 2^-10 ~ 0.00098, just under tol=1e-3, which is exactly the
condition of the process-age trap observed in the benchmark (error pinned at
2^-10, dt frozen in the controller deadband, boundaries capped). This rig
creates that condition in the first second of a run instead of after ~4100
boundaries.

Cells (2x2): offset in {0 m, 8192 m} x quantization floor {off, on}.
PASS requires all three:
  T1 (trap armed, vacuity): offset-8192/floor-off costs >= 2x iters of
      offset-0/floor-off. If not, the rig manufactured nothing and proves
      nothing.
  T2 (fix works): offset-8192/floor-on within 25% of offset-0/floor-on.
      Translation invariance restored.
  T3 (fix free when healthy): offset-0/floor-on within 25% of
      offset-0/floor-off.
Also prints the armed cell's post-march error values: the trap signature is
values pinned at ~9.77e-4.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp

wp.init()

import newton  # noqa: E402
import newton.solvers  # noqa: E402

N_WORLDS = 4
DT_OUTER = 1.0 / 60.0
K_BOUNDARIES = 120
TOL = 1e-3
SEED = 20260811
FORCE_SCALE = 0.5
OFFSET = 8192.0


def build_model(offset: float):
    """Slab + resting box + falling box (impact drives dt down, which is the
    state the trap freezes), all shifted by `offset` on x."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoAdaptive.register_custom_attributes(t)
    cfg = newton.ModelBuilder.ShapeConfig(ke=1.0e4, kd=100.0, kf=0.0, mu=0.5, margin=0.005)
    t.add_shape_box(-1, xform=wp.transform(p=wp.vec3(offset, 0.0, 0.05)), hx=0.2, hy=0.2, hz=0.05, cfg=cfg)
    rest = t.add_body(xform=wp.transform(p=wp.vec3(offset - 0.08, 0.0, 0.13)))
    t.add_shape_box(rest, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
    fall = t.add_body(xform=wp.transform(p=wp.vec3(offset + 0.08, 0.0, 0.16)))
    t.add_shape_box(fall, hx=0.03, hy=0.03, hz=0.03, cfg=cfg)
    b = newton.ModelBuilder()
    b.replicate(t, N_WORLDS)
    b.add_ground_plane()
    return b.finalize()


def run_cell(offset: float, floor_on: bool) -> tuple[float, np.ndarray, np.ndarray]:
    os.environ["NEWTON_ADAPTIVE_ERR_QUANT_FLOOR"] = "1" if floor_on else "0"
    os.environ.pop("NEWTON_ADAPTIVE_TWIN_EVAL", None)
    os.environ.pop("NEWTON_ADAPTIVE_TAIL_COMPACT", None)
    model = build_model(offset)
    solver = newton.solvers.SolverMuJoCoAdaptive(
        model,
        tol=TOL,
        dt_inner_init=DT_OUTER,
        dt_inner_min=1e-6,
        dt_inner_max=DT_OUTER,
        nconmax=32,
        njmax=128,
        use_newton_contacts=False,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    rng = np.random.default_rng(SEED)
    forces = rng.normal(0.0, FORCE_SCALE, size=(K_BOUNDARIES, model.body_count, 6)).astype(np.float32)
    iters = np.zeros(K_BOUNDARIES, dtype=np.int64)
    for k in range(K_BOUNDARIES):
        state_0.body_f.assign(forces[k])
        state_0, state_1 = solver.step_dt(DT_OUTER, state_0, state_1, control)
        iters[k] = int(solver._iteration_count_buf.numpy()[0])
    return float(iters[K_BOUNDARIES // 2 :].mean()), iters, solver._last_error.numpy().copy()


def main() -> int:
    cells = {}
    for offset in (0.0, OFFSET):
        for floor_on in (False, True):
            key = (offset, floor_on)
            mean_iters, iters, err = run_cell(offset, floor_on)
            cells[key] = (mean_iters, iters, err)
            tag = f"offset={offset:>6.0f} floor={'on ' if floor_on else 'off'}"
            print(f"{tag}: steady iters/boundary {mean_iters:6.2f}  max {iters.max():4d}  err {err}")

    base_off = cells[(0.0, False)][0]
    base_on = cells[(0.0, True)][0]
    far_off = cells[(OFFSET, False)][0]
    far_on = cells[(OFFSET, True)][0]

    ok = True
    if not far_off >= 2.0 * base_off:
        print(f"VACUOUS: trap not armed (far/off {far_off:.2f} < 2x near/off {base_off:.2f})")
        return 3
    if not far_on <= 1.25 * base_on:
        print(f"FAIL T2: floor does not release the trap (far/on {far_on:.2f} vs near/on {base_on:.2f})")
        ok = False
    if not (0.75 * base_off <= base_on <= 1.25 * base_off):
        print(f"FAIL T3: floor not free near origin ({base_on:.2f} vs {base_off:.2f})")
        ok = False
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
