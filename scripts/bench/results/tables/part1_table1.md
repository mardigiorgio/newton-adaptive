# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world penetration run: any ejection, or max penetration > the model's impact depth v·√(m/k))

### Soft clutter  (artifact if max penetration > 22.7 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 917% | 619% | 242% | 95% |
| ICF artifacts | No (pen 3.8 mm, eject 0.0%) | No (pen 4.2 mm, eject 0.0%) | No (pen 3.7 mm, eject 0.0%) | No (pen 3.6 mm, eject 0.0%) |
| MuJoCo real-time rate | 1845% | 676% | 325% | 136% |
| MuJoCo artifacts | Yes (pen 31.3 mm, eject 0.0%) | Yes (pen 35.3 mm, eject 0.0%) | Yes (pen 36.7 mm, eject 0.0%) | Yes (pen 37.8 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 1129% | 682% | 280% | 150% |
| ICF artifacts | No (pen 3.9 mm, eject 0.0%) | No (pen 4.5 mm, eject 0.0%) | No (pen 4.8 mm, eject 0.0%) | No (pen 4.4 mm, eject 0.0%) |
| MuJoCo real-time rate | 1685% | 915% | 390% | 222% |
| MuJoCo artifacts | Yes (pen 26.8 mm, eject 0.0%) | Yes (pen 35.0 mm, eject 0.0%) | Yes (pen 36.4 mm, eject 0.0%) | Yes (pen 37.1 mm, eject 0.0%) |

### Hard clutter  (artifact if max penetration > 2.27 mm = the model's impact depth v·√(m/k), or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 298% | 148% | 112% | 49% |
| ICF artifacts | Yes (pen 5.5 mm, eject 0.0%) | No (pen 1.1 mm, eject 0.0%) | No (pen 0.3 mm, eject 0.0%) | No (pen 0.3 mm, eject 0.0%) |
| MuJoCo real-time rate | 885% | 265% | 98% | 30% |
| MuJoCo artifacts | Yes (pen 12.0 mm, eject 0.0%) | Yes (pen 9.8 mm, eject 0.0%) | Yes (pen 4.2 mm, eject 0.0%) | Yes (pen 4.8 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 512% | 433% | 216% | 117% |
| ICF artifacts | Yes (pen 28.3 mm, eject 1.4%) | Yes (pen 11.4 mm, eject 0.0%) | Yes (pen 3.6 mm, eject 0.0%) | No (pen 1.1 mm, eject 0.0%) |
| MuJoCo real-time rate | 1256% | 875% | 355% | 143% |
| MuJoCo artifacts | Yes (pen 10.9 mm, eject 0.0%) | Yes (pen 4.1 mm, eject 0.0%) | Yes (pen 4.0 mm, eject 0.0%) | Yes (pen 3.9 mm, eject 0.0%) |

# Fixed-step reference levels — wall time [s] per simulated second

| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|---|---|
| soft-clutter | ICF fixed | 1 | 0.0886 | 0.147 | 0.357 | 0.669 |
| soft-clutter | MuJoCo fixed | 1 | 0.0593 | 0.109 | 0.256 | 0.451 |
| soft-clutter | ICF fixed | 1024 | 3.94 | 6.67 | 14.3 | 25.2 |
| soft-clutter | MuJoCo fixed | 1024 | 0.116 | 0.215 | 0.535 | 0.927 |
| hard-clutter | ICF fixed | 1 | 0.195 | 0.231 | 0.463 | 0.855 |
| hard-clutter | MuJoCo fixed | 1 | 0.0796 | 0.114 | 0.282 | 0.7 |
| hard-clutter | ICF fixed | 1024 | 7.46 | 11 | 20.9 | 38.6 |
| hard-clutter | MuJoCo fixed | 1024 | 0.23 | 0.393 | 0.973 | 2.25 |

# Bouncing ball (Fig. 8 scene): energy change after 10 s and rebounds (paper: 11)

| arm | δt or ε_acc | energy change [%] | bounces | status |
|---|---|---|---|---|
| icf | δt = 0.01 s | -100.10 | 2 | ok |
| icf | δt = 0.005 s | -100.10 | 4 | ok |
| icf | δt = 0.002 s | -100.10 | 7 | ok |
| icf | δt = 0.001 s | -100.10 | 11 | ok |
| icf | δt = 0.0005 s | -98.05 | 21 | ok |
| icf | δt = 0.0002 s | -57.09 | 13 | ok |
| icf | δt = 0.0001 s | -32.21 | 12 | ok |
| icf | δt = 5e-05 s | -16.29 | 11 | ok |
| icf | δt = 2e-05 s | -6.85 | 11 | ok |
| icf | δt = 1e-05 s | -3.48 | 11 | ok |
| mujoco | δt = 0.01 s | -100.04 | 1 | ok |
| mujoco | δt = 0.005 s | -100.04 | 1 | ok |
| mujoco | δt = 0.002 s | -100.04 | 0 | ok |
| mujoco | δt = 0.001 s | -100.04 | 1 | ok |
| mujoco | δt = 0.0005 s | -100.04 | 1 | ok |
| mujoco | δt = 0.0002 s | -100.04 | 1 | ok |
| mujoco | δt = 0.0001 s | -100.04 | 1 | ok |
| mujoco | δt = 5e-05 s | -100.04 | 1 | ok |
| mujoco | δt = 2e-05 s | -100.04 | 1 | ok |
| mujoco | δt = 1e-05 s | -100.04 | 1 | ok |
| icf-adaptive | ε = 0.1 | -100.10 | 3 | ok |
| icf-adaptive | ε = 0.01 | -100.10 | 3 | ok |
| icf-adaptive | ε = 0.001 | -100.10 | 5 | ok |
| icf-adaptive | ε = 0.0001 | -100.10 | 14 | ok |
| icf-adaptive | ε = 1e-05 | -52.45 | 11 | budget-exhausted |
| icf-adaptive | ε = 1e-06 | -91.70 | 0 | budget-exhausted |
| mujoco-adaptive | ε = 0.1 | -100.04 | 1 | ok |
| mujoco-adaptive | ε = 0.01 | -100.04 | 1 | ok |
| mujoco-adaptive | ε = 0.001 | -100.04 | 1 | ok |
| mujoco-adaptive | ε = 0.0001 | -100.04 | 0 | ok |
| mujoco-adaptive | ε = 1e-05 | -100.04 | 1 | ok |
| mujoco-adaptive | ε = 1e-06 | -100.04 | 1 | ok |
