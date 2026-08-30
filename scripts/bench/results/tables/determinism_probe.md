# Run-to-run determinism (scripts/bench/probe_determinism.py)

Two identical arms on the same model from the same initial state, 8 worlds, 1 s; max |Δx| over bodies between the two runs, every 0.1 s. A deterministic solver gives 0.

```
ball          icf              max |dx| after 1 s: 0.00e+00 m   (every 0.1 s: 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00)
ball          mujoco           max |dx| after 1 s: 0.00e+00 m   (every 0.1 s: 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00)
ball          icf-adaptive     max |dx| after 1 s: 0.00e+00 m   (every 0.1 s: 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00)
ball          mujoco-adaptive  max |dx| after 1 s: 0.00e+00 m   (every 0.1 s: 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00 0.0e+00)
soft-clutter  icf              max |dx| after 1 s: 2.38e-02 m   (every 0.1 s: 3.0e-08 7.2e-04 2.2e-03 6.4e-03 7.7e-03 2.3e-02 2.6e-02 2.6e-02 2.4e-02 2.4e-02)
soft-clutter  mujoco           max |dx| after 1 s: 2.68e-06 m   (every 0.1 s: 0.0e+00 7.5e-09 7.5e-08 1.9e-07 3.3e-07 3.1e-07 7.2e-07 1.2e-06 1.9e-06 2.7e-06)
soft-clutter  icf-adaptive     max |dx| after 1 s: 3.65e-02 m   (every 0.1 s: 0.0e+00 8.2e-04 3.4e-03 2.0e-02 4.2e-02 4.3e-02 4.2e-02 4.0e-02 3.9e-02 3.7e-02)
soft-clutter  mujoco-adaptive  max |dx| after 1 s: 4.23e-04 m   (every 0.1 s: 0.0e+00 1.5e-08 2.6e-07 1.0e-06 2.3e-06 1.8e-06 5.8e-05 1.2e-04 1.3e-04 4.2e-04)
hard-clutter  icf              max |dx| after 1 s: 1.50e-01 m   (every 0.1 s: 1.5e-08 9.4e-03 7.8e-02 1.0e-01 1.1e-01 1.4e-01 1.4e-01 1.5e-01 1.5e-01 1.5e-01)
hard-clutter  mujoco           max |dx| after 1 s: 9.39e-02 m   (every 0.1 s: 0.0e+00 1.2e-03 1.4e-02 4.4e-02 5.9e-02 6.6e-02 6.5e-02 8.4e-02 9.3e-02 9.4e-02)
hard-clutter  icf-adaptive     max |dx| after 1 s: 1.12e-01 m   (every 0.1 s: 0.0e+00 5.9e-03 3.9e-02 5.7e-02 1.0e-01 1.1e-01 1.1e-01 1.1e-01 1.1e-01 1.1e-01)
hard-clutter  mujoco-adaptive  max |dx| after 1 s: 7.59e-02 m   (every 0.1 s: 0.0e+00 6.3e-06 5.4e-03 2.2e-02 4.1e-02 4.5e-02 4.9e-02 5.8e-02 6.7e-02 7.6e-02)
NOT bit-reproducible
```

Reading: the ball (one body) is bit-reproducible on every arm; both contact solvers are not on clutter — the order of the GPU reductions in the contact solve differs run to run, and the pile amplifies the difference to millimetres within 0.3 s on soft clutter (ICF; MuJoCo stays at micrometres there) and to centimetres within 0.5 s on hard clutter (both). Consequences: (i) a restarted-window comparison against a reference (part1_consistency.py) has a floor equal to this noise — the reference restarted against itself is now a row of that bench and is drawn as the floor; (ii) two training runs with the same seed are not the same run on either backend under clutter contact.

## Self-check of the consistency bench (`part1_consistency.py --self-check`)

Three properties per backend, from two runs of the restart floor (the
reference solver at 0.1 ms restarted from its own snapshots) and one run of
the coarsest knob (10 ms): every window runs; the floor reproduces in
magnitude across the two runs (within 10x -- the solvers are not bitwise
run-to-run deterministic on clutter, above, so exact equality is not a
property this system has); the floor's mean deviation sits >= 10x below the
coarsest knob's, so reported deviations are signal, not restart noise. The
floor VALUE is reported as the bench's reference-dt row, not asserted.

| scene | ICF floor (mean) | ICF coarse | MuJoCo floor | MuJoCo coarse | verdict |
|---|---|---|---|---|---|
| ball | 0 | 9.1e-3 m | 0 | 7.7e-3 m | PASS, PASS |
| soft clutter | 4.6e-6 m | 5.1e-3 m | 2.3e-7 m | 1.7e-3 m | PASS, PASS |
| hard clutter | 1.4e-3 m | 1.5e-2 m | 4.3e-4 m | 1.0e-2 m | PASS, PASS |

The ball's floor is exactly zero on both backends (the restart wiring is
right); the clutter floors are the solvers' own run-to-run noise amplified
over the window, 11-24x below the coarsest knob's deviation.
