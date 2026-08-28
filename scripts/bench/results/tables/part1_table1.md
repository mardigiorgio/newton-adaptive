# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world penetration run: any ejection, or max penetration > 10× the scene's static penetration m·g/k)

### Soft clutter  (artifact if max penetration > 6.42 mm = 10× m·g/k, or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 765% | 764% | 640% | 555% |
| ICF artifacts | No (pen 1.9 mm, eject 0.0%) | No (pen 1.9 mm, eject 0.0%) | No (pen 2.0 mm, eject 0.0%) | No (pen 2.2 mm, eject 0.0%) |
| MuJoCo real-time rate | 1017% | 1018% | 923% | 539% |
| MuJoCo artifacts | Yes (pen 21.9 mm, eject 0.0%) | Yes (pen 21.9 mm, eject 0.0%) | Yes (pen 23.3 mm, eject 0.0%) | Yes (pen 24.3 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 2425% | 1371% | 605% | 321% |
| ICF artifacts | No (pen 1.7 mm, eject 0.0%) | No (pen 1.9 mm, eject 0.0%) | No (pen 2.2 mm, eject 0.0%) | No (pen 2.2 mm, eject 0.0%) |
| MuJoCo real-time rate | 2082% | 1707% | 707% | 382% |
| MuJoCo artifacts | Yes (pen 33.8 mm, eject 0.0%) | Yes (pen 21.9 mm, eject 0.0%) | Yes (pen 23.6 mm, eject 0.0%) | Yes (pen 23.3 mm, eject 0.0%) |

### Hard clutter  (artifact if max penetration > 0.0642 mm = 10× m·g/k, or any ejection)

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 430% | 433% | 436% | 356% |
| ICF artifacts | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) |
| MuJoCo real-time rate | 699% | 293% | 824% | 381% |
| MuJoCo artifacts | Yes (pen 6.5 mm, eject 0.0%) | Yes (pen 7.4 mm, eject 0.0%) | Yes (pen 3.8 mm, eject 0.0%) | Yes (pen 7.5 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 2232% | 844% | 341% | 308% |
| ICF artifacts | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) |
| MuJoCo real-time rate | 926% | 773% | 607% | 296% |
| MuJoCo artifacts | Yes (pen 13.6 mm, eject 0.0%) | Yes (pen 8.0 mm, eject 0.0%) | Yes (pen 5.2 mm, eject 0.0%) | Yes (pen 4.9 mm, eject 0.0%) |

# Fixed-step reference levels — wall time [s] per simulated second

| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|---|---|
| soft-clutter | ICF fixed | 1 | 0.0412 | 0.0729 | 0.165 | 0.311 |
| soft-clutter | MuJoCo fixed | 1 | 0.048 | 0.0586 | 0.141 | 0.262 |
| soft-clutter | ICF fixed | 1024 | 0.845 | 1.41 | 2.76 | 5.03 |
| soft-clutter | MuJoCo fixed | 1024 | 0.14 | 0.143 | 0.346 | 0.664 |
| hard-clutter | ICF fixed | 1 | 0.0448 | 0.118 | 0.293 | 0.325 |
| hard-clutter | MuJoCo fixed | 1 | 0.108 | 0.129 | 0.165 | 0.338 |
| hard-clutter | ICF fixed | 1024 | 0.542 | 1.73 | 4.12 | 4.26 |
| hard-clutter | MuJoCo fixed | 1024 | 0.325 | 0.534 | 0.76 | 1.62 |

# Bouncing ball (Fig. 8 scene): energy change after 10 s and rebounds (paper: 11)

| arm | δt or ε_acc | energy change [%] | bounces | status |
|---|---|---|---|---|
| icf | δt = 0.01 s | -99.60 | 2 | ok |
| icf | δt = 0.005 s | -99.60 | 3 | ok |
| icf | δt = 0.002 s | -99.60 | 8 | ok |
| icf | δt = 0.001 s | -99.60 | 12 | ok |
| icf | δt = 0.0005 s | -97.58 | 21 | ok |
| icf | δt = 0.0002 s | -56.81 | 13 | ok |
| icf | δt = 0.0001 s | -32.07 | 11 | ok |
| icf | δt = 5e-05 s | -16.22 | 10 | ok |
| icf | δt = 2e-05 s | -6.84 | 11 | ok |
| icf | δt = 1e-05 s | -3.48 | 11 | ok |
| mujoco | δt = 0.01 s | -99.54 | 1 | ok |
| mujoco | δt = 0.005 s | -99.54 | 1 | ok |
| mujoco | δt = 0.002 s | -99.54 | 1 | ok |
| mujoco | δt = 0.001 s | -99.54 | 0 | ok |
| mujoco | δt = 0.0005 s | -99.54 | 0 | ok |
| mujoco | δt = 0.0002 s | -99.54 | 0 | ok |
| mujoco | δt = 0.0001 s | -99.54 | 0 | ok |
| mujoco | δt = 5e-05 s | -99.54 | 0 | ok |
| mujoco | δt = 2e-05 s | -99.54 | 0 | ok |
| mujoco | δt = 1e-05 s | -99.54 | 0 | ok |
| icf-adaptive | ε = 0.1 | -99.60 | 3 | ok |
| icf-adaptive | ε = 0.01 | -99.60 | 3 | ok |
| icf-adaptive | ε = 0.001 | -99.60 | 6 | ok |
| icf-adaptive | ε = 0.0001 | -99.60 | 15 | ok |
| icf-adaptive | ε = 1e-05 | -61.48 | 12 | ok |
| icf-adaptive | ε = 1e-06 | -22.11 | 11 | ok |
| mujoco-adaptive | ε = 0.1 | -99.54 | 1 | ok |
| mujoco-adaptive | ε = 0.01 | -99.54 | 1 | ok |
| mujoco-adaptive | ε = 0.001 | -99.54 | 1 | ok |
| mujoco-adaptive | ε = 0.0001 | -99.54 | 1 | ok |
| mujoco-adaptive | ε = 1e-05 | -99.54 | 0 | ok |
| mujoco-adaptive | ε = 1e-06 | -99.54 | 0 | ok |
