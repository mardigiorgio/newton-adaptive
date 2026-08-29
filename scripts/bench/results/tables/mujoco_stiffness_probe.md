# MuJoCo contact solref: conversion vs explicit (one 65 g sphere resting on the plane; model m·g/k = 6.4 µm at k = 1e5)

Newton's `convert_solref(ke, kd, 1, 1)` gives (timeconst, dampratio) = (1 ms, 3.16) for ke = 1e5, kd = 0.02 ke; (0.1 s, 0.32) for the soft clutter; and (20 ms, **1.0**) for the ball that asked for kd = 0. MuJoCo's stiffness scales as 1/(timeconst² · dampratio²) and timeconst is clamped to ≥ 2 dt.

```
                      solref     dt rest pen [um]
     converted (0.001, 3.16) 10.00ms         877.9
     converted (0.001, 3.16)  1.00ms         415.6
     converted (0.001, 3.16)  0.25ms         415.5
       explicit (0.001, 1.0) 10.00ms         367.2
       explicit (0.001, 1.0)  1.00ms           4.4
       explicit (0.001, 1.0)  0.25ms           1.1
      explicit (0.0005, 1.0) 10.00ms         367.2
      explicit (0.0005, 1.0)  1.00ms           4.4
      explicit (0.0005, 1.0)  0.25ms           0.3
       explicit (0.002, 1.0) 10.00ms         367.2
       explicit (0.002, 1.0)  1.00ms           4.4
       explicit (0.002, 1.0)  0.25ms           4.4
```

Resting penetration scales as timeconst²: 2 ms → 4.4 µm, so timeconst 2.4 ms realizes k = 1e5 at rest once dt ≤ 1.2 ms (24 ms for k = 1e3). Adopted per scene as `SceneSpec.mujoco_solref`, dampratio 1 on the clutters, 0 for the ball.

## Ball with explicit solref (τ = 31.6 ms)

```
  converted                      dt= 1.0 ms: energy change -100.04 %  rebounds  0  (solref now [0.02 1.  ])
  converted                      dt= 0.1 ms: energy change -100.04 %  rebounds  1  (solref now [0.02 1.  ])
  explicit tau=31.6ms zeta=0     dt= 1.0 ms: energy change    +nan %  rebounds  0  (solref now [0.0316 0.    ])
  explicit tau=31.6ms zeta=0     dt= 0.1 ms: energy change    +nan %  rebounds  0  (solref now [0.0316 0.    ])
  explicit tau=31.6ms zeta=0.05  dt= 1.0 ms: energy change -100.00 %  rebounds 25  (solref now [0.0316 0.05  ])
  explicit tau=31.6ms zeta=0.05  dt= 0.1 ms: energy change -100.00 %  rebounds 25  (solref now [0.0316 0.05  ])
  explicit tau=10ms zeta=0       dt= 1.0 ms: energy change    +nan %  rebounds  0  (solref now [0.01 0.  ])
  explicit tau=10ms zeta=0       dt= 0.1 ms: energy change    +nan %  rebounds  0  (solref now [0.01 0.  ])
```

MuJoCo admits no zero damping ratio (ζ = 0 diverges to NaN); with the smallest stable ζ the ball rebounds 25 times in 10 s but still loses all of its energy — MuJoCo's soft-constraint contact dissipates at every impact regardless of solref, so a conservative contact cannot be represented.
