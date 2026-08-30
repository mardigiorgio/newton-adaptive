# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world penetration run: any ejection, or max penetration > the model's impact depth v·√(m/k))

### Soft clutter  (artifact if max penetration > 22.7 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 765% | 488% | 175% | 68% |
| ICF artifacts | No (pen 4.5 mm, eject 0.0%) | No (pen 7.2 mm, eject 0.0%) | No (pen 7.2 mm, eject 0.0%) | No (pen 8.7 mm, eject 0.0%) |
| MuJoCo real-time rate | 2238% | 831% | 400% | 157% |
| MuJoCo artifacts | No (pen 22.3 mm, eject 0.0%) | Yes (pen 22.9 mm, eject 0.0%) | Yes (pen 23.7 mm, eject 0.0%) | No (pen 20.0 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 894% | 482% | 235% | 122% |
| ICF artifacts | No (pen 5.3 mm, eject 0.0%) | No (pen 6.9 mm, eject 0.0%) | No (pen 8.3 mm, eject 0.0%) | No (pen 9.3 mm, eject 0.0%) |
| MuJoCo real-time rate | 2297% | 1043% | 468% | 244% |
| MuJoCo artifacts | No (pen 21.9 mm, eject 0.0%) | No (pen 15.8 mm, eject 0.0%) | No (pen 19.2 mm, eject 0.0%) | No (pen 18.1 mm, eject 0.0%) |

### Hard clutter  (artifact if max penetration > 2.27 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 329% | 181% | 85% | 30% |
| ICF artifacts | Yes (pen 3.1 mm, eject 0.0%) | No (pen 1.1 mm, eject 0.0%) | No (pen 0.8 mm, eject 0.0%) | No (pen 0.6 mm, eject 0.0%) |
| MuJoCo real-time rate | 1080% | 335% | 96% | 36% |
| MuJoCo artifacts | Yes (pen 10.1 mm, eject 0.0%) | No (pen 2.1 mm, eject 0.0%) | No (pen 0.9 mm, eject 0.0%) | No (pen 0.7 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 541% | 346% | 172% | 93% |
| ICF artifacts | Yes (pen 4.1 mm, eject 0.0%) | Yes (pen 5.7 mm, eject 0.0%) | No (pen 2.1 mm, eject 0.0%) | No (pen 1.6 mm, eject 0.0%) |
| MuJoCo real-time rate | 1545% | 806% | 294% | 159% |
| MuJoCo artifacts | Yes (pen 13.4 mm, eject 0.0%) | Yes (pen 2.8 mm, eject 0.0%) | No (pen 1.5 mm, eject 0.0%) | No (pen 0.7 mm, eject 0.0%) |

# Fixed-step reference levels — wall time [s] per simulated second

| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|---|---|
| soft-clutter | ICF fixed | 1 | 0.112 | 0.207 | 0.426 | 0.822 |
| soft-clutter | MuJoCo fixed | 1 | 0.0435 | 0.0958 | 0.214 | 0.409 |
| soft-clutter | ICF fixed | 1024 | 3.99 | 7.36 | 14.7 | 29 |
| soft-clutter | MuJoCo fixed | 1024 | 0.0875 | 0.19 | 0.435 | 0.838 |
| hard-clutter | ICF fixed | 1 | 0.185 | 0.289 | 0.582 | 1.07 |
| hard-clutter | MuJoCo fixed | 1 | 0.0647 | 0.124 | 0.34 | 0.628 |
| hard-clutter | ICF fixed | 1024 | 6.31 | 10.4 | 21.3 | 38.3 |
| hard-clutter | MuJoCo fixed | 1024 | 0.183 | 0.387 | 1.04 | 2.11 |

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
