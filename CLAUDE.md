## CENIC simulation loop pattern

All scripts using `SolverMuJoCoAdaptive` (the adaptive step-doubling solver; true CENIC = adaptive + convex ICF contact, not yet built) must use `step_dt` — never reimplement the inner loop manually.

```python
DT = 0.002  # 500 Hz — default control and render period [s]

while viewer.is_running():
    state_0, state_1 = solver.step_dt(
        DT, state_0, state_1, control,
        apply_forces=viewer.apply_forces,
    )
    # control / policy updates go here — once per DT boundary
    t += DT
    viewer.render(state_0, t)  # begin_frame(t) + log_state + end_frame
```

`step_dt` owns the inner loop, the GPU boundary kernels, `clear_forces`, and `apply_forces` ordering. Never call `viewer.begin_frame`, policy updates, or `for _ in range(N)` inside the inner loop.

---

## CRITICAL: Zero device transfers in the hot path

**Every `.numpy()` call on a GPU array is a full CUDA device synchronization.** In the inner physics loop this fires on every substep — thousands of times per frame during dense contact. This is the single most destructive performance pattern in CENIC scripts.

### The rule: `.numpy()` must never appear inside the inner physics loop.

Wrong — stalls the GPU on every substep, O(N) data transferred per stall:
```python
while True:
    solver.step(...)
    if np.all(solver.sim_time.numpy() >= next_time):  # FULL GPU SYNC + N floats
        break
```

Correct — use `step_dt`, which handles the inner loop internally:
```python
# step_dt replays a captured per-iteration CUDA graph and checks a 4-byte
# boundary flag via .numpy() -- one int32 per iteration. With
# NEWTON_MJ_ADAPTIVE_CONDITIONAL=1 the whole loop is one conditional graph
# node (zero host syncs). Do not reimplement this loop manually.
state_0, state_1 = solver.step_dt(DT, state_0, state_1, control)
```

### Why N worlds do not hurt physics throughput — but do hurt render throughput

MuJoCo Warp batches all N worlds into a single GPU kernel per step. For small N (≤ ~64 worlds of simple geometry) the GPU is not saturated — physics throughput scales sub-linearly with N, approaching free.

The viewer is the bottleneck at large N. `log_state` calls `wp.synchronize()` once per frame to flush the VBO copy. With N worlds doing more GPU work, that sync takes longer. **For data collection at N > 1, always run `--headless`.**

### Acceptable `.numpy()` call-sites (outside the inner loop)

- `_print_status()` — status grid, gated behind `step % LOG_EVERY`
- Startup banners before the loop (e.g. reading `solver._dt.numpy()` once)
- End-of-run summaries (wall time, FPS)

Any `.numpy()` inside `while True: … solver.step(…)` must be rejected in review.

---

## dt parameter rules

`dt_inner_min` must always be strictly less than `dt_inner_init`. If `dt_inner_min >= dt_inner_init`, any rejected step clamps dt *upward*, which is physically wrong and causes oscillation.

```python
# Correct relationship:  dt_inner_min < dt_inner_init <= dt_inner_max
solver = SolverMuJoCoAdaptive(
    model,
    dt_inner_init=1e-3,
    dt_inner_min=5e-4,   # floor — must be < dt_inner_init
    dt_inner_max=0.008,
)
```

---

## Viewer sim-time integration

`viewer.render(state, sim_time)` drives the camera and UI from **simulation time**, not wall clock. This prevents camera jumps during dense contact substeps where many physics steps fire between renders. No additional timing logic is needed in scripts — `render()` handles it.

Identically-seeded runs are NOT bit-reproducible (measured 2026-07-02, Allegro reorient, 64 envs, bit-identical inputs): trajectories diverge at step 0 in BOTH the fixed-step and adaptive solvers (max|dq| ~1e-6..1e-5 after one control step), because MuJoCo-Warp's contact pipeline uses non-associative GPU reductions. Contact-rich dynamics then amplify chaotically (~O(1) rad by ~20 control steps). The adaptive controller's accept/reject decisions also diverge (iteration counts split within a few steps), but that is a consequence of already-divergent states, not the cause -- the controller-free fixed solver diverges identically. Implications: datasets/demos must be reproduced by RECORDING, never by replay; use `--num-worlds 1` for visualization and `--headless` for data collection.

---

## Plotting conventions

All benchmark and scaling plots must use log-log axes unless the plot is a time series (x-axis is simulation time) or log scale does not make sense for the data. Time series plots use linear x with log y where appropriate (e.g. error traces).

---

## Benchmark command

Scaling measurement:
```
uv run -m scripts.bench --only scaling --ns 1 4 16 64 256 --steps 50 --warmup 20
```

@AGENTS.md
