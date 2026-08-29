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

In the reference solref format (timeconst, dampratio) MuJoCo admits no zero damping ratio (ζ = 0 diverges to NaN); with the smallest stable ζ the ball rebounds 25 times in 10 s but still loses all of its energy. The direct format below does represent the undamped contact.

## Direct solref format (−stiffness, −damping)

MuJoCo's direct format takes the constraint stiffness and damping as numbers
(a_ref = −stiffness · r − damping · v, in acceleration units) and applies no
refsafe clamp. Resting sphere, k = 1e5 requested, damping −632 (ζ = 1 at that
stiffness), stiffness interpolated for m g / k = 6.4 µm at δt = 1 ms:

```
interpolated direct stiffness for m*g/k at 1 ms: 1.703e+05
  calibrated direct solref, dt=10.00 ms: rest -26114759.1 um (target 6.4)   launched: unstable
  calibrated direct solref, dt= 5.00 ms: rest  -1561953.0 um (target 6.4)   launched: unstable
  calibrated direct solref, dt= 2.00 ms: rest        6.4 um (target 6.4)
  calibrated direct solref, dt= 1.00 ms: rest        6.4 um (target 6.4)
  calibrated direct solref, dt= 0.25 ms: rest        6.4 um (target 6.4)
```

The clamp in the reference format is the guard against this: the soft
constraint is stable only while ω δt stays below ~2 (ω = √(k/m_eff) = 1240
rad/s for the 65 g sphere at k = 1e5 → δt < 1.6 ms), and the reference
format enforces it by softening (τ ≥ 2 δt) instead of exploding. Either
way a fixed-step MuJoCo arm at δt = 10 ms realizes at most k ≈ m_eff / (4 δt²)
of order 1e2–1e3 N/m per 65 g body. The clutter arms keep the reference
format (MuJoCo's default and its own safety rule).

Undamped ball (k = 1e3, zero dissipation) in the direct format, stiffness
interpolated for the resting depth m g / k = 0.98 mm (measured 1.01 mm at
−2.24e3):

```
 dt [ms]  energy change %  rebounds   (direct solref (-2.24e+03, 0))
   10.00            -3.28        10
    5.00          -104.60        10
    2.00             0.51        10
    1.00            -0.16        10
    0.50            -0.08        10
    0.20            -0.03        10
```

ω δt = 100 × 0.01 = 1 at δt = 10 ms, inside the stable range: the undamped
soft constraint conserves the ball's energy to 0.2 % at δt ≤ 1 ms (the
5 ms row is the one exception on the ladder and is reported as measured).
The ball's MuJoCo arm uses this solref (`MUJOCO_BALL_DIRECT_STIFFNESS`); the
earlier ζ = 0.05 rows above are superseded.
