# Table I analog — real-time rate and artifacts (N = 1 GPU world; artifacts from the 64-world penetration run: any ejection or max penetration > 1 mm)

### Soft clutter

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 392% | 387% | 337% | 281% |
| ICF artifacts | — | — | — | — |
| MuJoCo real-time rate | 672% | 657% | 402% | 172% |
| MuJoCo artifacts | — | — | — | — |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 1493% | 808% | 348% | 181% |
| ICF artifacts | — | — | — | — |
| MuJoCo real-time rate | 1917% | 1094% | 480% | 272% |
| MuJoCo artifacts | — | — | — | — |

### Hard clutter

| Error control ε_acc | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| ICF real-time rate | 147% | 78% | 28% | 9% |
| ICF artifacts | Yes (pen 3.1 mm, eject 2.7%) | Yes (pen 14.4 mm, eject 0.0%) | No (pen 0.0 mm, eject 0.0%) | Yes (pen 0.0 mm, eject 0.1%) |
| MuJoCo real-time rate | 525% | 388% | 836% | 369% |
| MuJoCo artifacts | Yes (pen 6.6 mm, eject 0.0%) | Yes (pen 7.7 mm, eject 0.0%) | Yes (pen 10.2 mm, eject 0.0%) | Yes (pen 8.3 mm, eject 0.0%) |

| Fixed step δt | 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|
| ICF real-time rate | 432% | 343% | 221% | 125% |
| ICF artifacts | Yes (pen 5.2 mm, eject 10.2%) | Yes (pen 5.6 mm, eject 8.3%) | Yes (pen 1.1 mm, eject 5.2%) | Yes (pen 0.0 mm, eject 4.8%) |
| MuJoCo real-time rate | 948% | 579% | 619% | 283% |
| MuJoCo artifacts | Yes (pen 13.2 mm, eject 0.0%) | Yes (pen 6.6 mm, eject 0.0%) | Yes (pen 5.2 mm, eject 0.0%) | Yes (pen 4.9 mm, eject 0.0%) |

# Fixed-step reference levels — wall time [s] per simulated second

| scene | arm | N | δt = 10 ms | 5 ms | 2 ms | 1 ms |
|---|---|---|---|---|---|---|
| soft-clutter | ICF fixed | 1 | 0.067 | 0.124 | 0.287 | 0.551 |
| soft-clutter | MuJoCo fixed | 1 | 0.0522 | 0.0914 | 0.208 | 0.368 |
| soft-clutter | ICF fixed | 1024 | 2.41 | 4.24 | 8.77 | 15 |
| soft-clutter | MuJoCo fixed | 1024 | 0.152 | 0.255 | 0.605 | 1.15 |
| hard-clutter | ICF fixed | 1 | 0.231 | 0.292 | 0.452 | 0.8 |
| hard-clutter | MuJoCo fixed | 1 | 0.105 | 0.173 | 0.161 | 0.353 |
| hard-clutter | ICF fixed | 1024 | 7.26 | 10.1 | 19.9 | 35.7 |
| hard-clutter | MuJoCo fixed | 1024 | 0.3 | 0.494 | 0.688 | 1.49 |

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
