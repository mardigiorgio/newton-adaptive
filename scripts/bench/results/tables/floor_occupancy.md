# Floor occupancy of the error-controlled arms (dt_inner_min = 1e-6 s, budget 65536, one world, 2 s)

A floor hit is an inner step selected at the floor; an accepted floor step skips the accuracy test.

| scene | arm | eps_acc | attempts | floor hits | budget-exhausted |
|---|---|---|---|---|---|
| hard-clutter | mujoco-adaptive | 0.0001 | 2900 | 0 | 0.00 |
| hard-clutter | mujoco-adaptive | 1e-05 | 6061 | 0 | 0.00 |
| hard-clutter | mujoco-adaptive | 1e-06 | 19817 | 0 | 0.00 |
| hard-clutter | icf-adaptive | 0.0001 | 2523 | 0 | 0.00 |
| hard-clutter | icf-adaptive | 1e-05 | 6950 | 2 | 0.00 |
| hard-clutter | icf-adaptive | 1e-06 | 18236 | 0 | 0.00 |
| soft-clutter | mujoco-adaptive | 0.0001 | 865 | 0 | 0.00 |
| soft-clutter | mujoco-adaptive | 1e-05 | 2565 | 0 | 0.00 |
| soft-clutter | mujoco-adaptive | 1e-06 | 7274 | 0 | 0.00 |
| soft-clutter | icf-adaptive | 0.0001 | 1399 | 0 | 0.00 |
| soft-clutter | icf-adaptive | 1e-05 | 3927 | 0 | 0.00 |
| soft-clutter | icf-adaptive | 1e-06 | 11499 | 0 | 0.00 |
