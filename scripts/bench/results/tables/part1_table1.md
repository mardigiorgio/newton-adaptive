# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world penetration run: any ejection, or max penetration > the model's impact depth v·√(m/k))

### Soft clutter  (artifact if max penetration > 22.7 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 1024% | 504% | 206% | 83% |
| ICF artifacts | No (pen 4.1 mm, eject 0.0%) | No (pen 3.9 mm, eject 0.0%) | No (pen 3.9 mm, eject 0.0%) | No (pen 3.6 mm, eject 0.0%) |
| MuJoCo real-time rate | 2421% | 760% | 333% | 126% |
| MuJoCo artifacts | No (pen 4.9 mm, eject 0.0%) | No (pen 5.7 mm, eject 0.0%) | No (pen 6.2 mm, eject 0.0%) | No (pen 6.4 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 897% | 534% | 237% | 130% |
| ICF artifacts | No (pen 4.2 mm, eject 0.0%) | No (pen 4.5 mm, eject 0.0%) | No (pen 4.5 mm, eject 0.0%) | No (pen 4.6 mm, eject 0.0%) |
| MuJoCo real-time rate | 2426% | 1199% | 493% | 248% |
| MuJoCo artifacts | No (pen 3.8 mm, eject 0.0%) | No (pen 4.5 mm, eject 0.0%) | No (pen 5.8 mm, eject 0.0%) | No (pen 6.5 mm, eject 0.0%) |

### Hard clutter  (artifact if max penetration > 2.27 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 263% | 166% | 74% | 35% |
| ICF artifacts | Yes (pen 7.0 mm, eject 0.0%) | No (pen 0.8 mm, eject 0.0%) | No (pen 0.3 mm, eject 0.0%) | No (pen 0.3 mm, eject 0.0%) |
| MuJoCo real-time rate | 1090% | 332% | 96% | 34% |
| MuJoCo artifacts | Yes (pen 10.1 mm, eject 0.0%) | Yes (pen 3.9 mm, eject 0.0%) | No (pen 1.1 mm, eject 0.0%) | No (pen 0.7 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 489% | 340% | 170% | 88% |
| ICF artifacts | Yes (pen 27.8 mm, eject 1.6%) | Yes (pen 9.4 mm, eject 0.0%) | Yes (pen 3.9 mm, eject 0.0%) | No (pen 0.9 mm, eject 0.0%) |
| MuJoCo real-time rate | 1569% | 799% | 296% | 151% |
| MuJoCo artifacts | Yes (pen 6.8 mm, eject 0.0%) | Yes (pen 3.1 mm, eject 0.0%) | No (pen 0.9 mm, eject 0.0%) | No (pen 0.7 mm, eject 0.0%) |

# Fixed-step reference levels — wall time [s] per simulated second

| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|---|---|
| soft-clutter | ICF fixed | 1 | 0.111 | 0.187 | 0.422 | 0.767 |
| soft-clutter | MuJoCo fixed | 1 | 0.0412 | 0.0834 | 0.203 | 0.403 |
| soft-clutter | ICF fixed | 1024 | 3.91 | 6.66 | 14.3 | 25.1 |
| soft-clutter | MuJoCo fixed | 1024 | 0.0835 | 0.169 | 0.417 | 0.844 |
| hard-clutter | ICF fixed | 1 | 0.204 | 0.294 | 0.588 | 1.13 |
| hard-clutter | MuJoCo fixed | 1 | 0.0637 | 0.125 | 0.338 | 0.663 |
| hard-clutter | ICF fixed | 1024 | 7.47 | 10.9 | 20.9 | 38.2 |
| hard-clutter | MuJoCo fixed | 1024 | 0.183 | 0.389 | 1.04 | 2.11 |

# Bouncing ball (Fig. 8 scene): energy change after 10 s and rebounds (paper: 11)

| arm | δt or ε_acc | energy change [%] | bounces | status |
|---|---|---|---|---|
| icf | δt = 0.01 s | -100.10 | 2 | ok |
| icf | δt = 0.005 s | -100.10 | 4 | ok |
| icf | δt = 0.002 s | -100.10 | 7 | ok |
| icf | δt = 0.001 s | -100.10 | 11 | ok |
| icf | δt = 0.0005 s | -97.66 | 21 | ok |
| icf | δt = 0.0002 s | -57.09 | 13 | ok |
| icf | δt = 0.0001 s | -29.98 | 12 | ok |
| icf | δt = 5e-05 s | -16.28 | 11 | ok |
| icf | δt = 2e-05 s | -6.25 | 11 | ok |
| icf | δt = 1e-05 s | -3.17 | 11 | ok |
| mujoco | δt = 0.01 s | -7.08 | 10 | ok |
| mujoco | δt = 0.005 s | -7.33 | 10 | ok |
| mujoco | δt = 0.002 s | +0.78 | 10 | ok |
| mujoco | δt = 0.001 s | -0.02 | 10 | ok |
| mujoco | δt = 0.0005 s | -0.00 | 10 | ok |
| mujoco | δt = 0.0002 s | -0.00 | 10 | ok |
| mujoco | δt = 0.0001 s | -0.00 | 10 | ok |
| mujoco | δt = 5e-05 s | +0.00 | 10 | ok |
| mujoco | δt = 2e-05 s | +0.00 | 10 | ok |
| mujoco | δt = 1e-05 s | -0.00 | 10 | ok |
| icf-adaptive | ε = 0.1 | -100.10 | 3 | ok |
| icf-adaptive | ε = 0.01 | -100.10 | 3 | ok |
| icf-adaptive | ε = 0.001 | -100.10 | 5 | ok |
| icf-adaptive | ε = 0.0001 | -100.10 | 14 | ok |
| icf-adaptive | ε = 1e-05 | -51.01 | 11 | budget-exhausted |
| icf-adaptive | ε = 1e-06 | -91.70 | 0 | budget-exhausted |
| mujoco-adaptive | ε = 0.1 | +0.77 | 9 | ok |
| mujoco-adaptive | ε = 0.01 | +0.77 | 9 | ok |
| mujoco-adaptive | ε = 0.001 | +57.27 | 9 | ok |
| mujoco-adaptive | ε = 0.0001 | +22.34 | 10 | ok |
| mujoco-adaptive | ε = 1e-05 | +12.71 | 10 | ok |
| mujoco-adaptive | ε = 1e-06 | +3.97 | 9 | ok |
