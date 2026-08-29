# Actuated push — trajectory forensics (scripts/bench/probe_actuated_trace.py)

One world, k = 1e5 N/m, μ = 0.5, push at 300 mm/s; per-boundary trace of tip and box. 'PUSH FROZEN' holds the x target at 0 (box at rest, tip beside it).

```
== mujoco fixed dt=10 ms Kp=100
   box z: min 48.697 mm max 70.573 mm (rest 50.0);  box vz RMS push 0.0943 / settle 0.0018 m/s;  pitch rate RMS push 1.449 rad/s
   tip z: min 39.7 mm max 110.8 mm (box mid 50);  box x end 0.247 m;  tip pen max 13.23 mm at t=2.01s (xt=0.1849 xb=0.2316 zt=0.1091 zb=0.0602 target x=0.2200)
   tip pen > 1 mm at 21 boundaries, t in [0.91, 2.01] s
== mujoco fixed dt=1 ms Kp=100
   box z: min 49.978 mm max 70.706 mm (rest 50.0);  box vz RMS push 0.0568 / settle 0.0681 m/s;  pitch rate RMS push 0.883 rad/s
   tip z: min 40.2 mm max 112.4 mm (box mid 50);  box x end 0.267 m;  tip pen max 18.18 mm at t=2.30s (xt=0.2078 xb=0.2497 zt=0.1124 zb=0.0627 target x=0.2200)
   tip pen > 1 mm at 27 boundaries, t in [2.04, 2.30] s
== mujoco fixed dt=1 ms Kp=100 PUSH FROZEN (box at rest, tip beside it)
   box z: min 49.997 mm max 49.998 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.000 rad/s
   tip z: min 40.2 mm max 99.6 mm (box mid 50);  box x end 0.000 m;  tip pen max 0.00 mm at t=0.00s (xt=-0.0800 xb=0.0000 zt=0.0996 zb=0.0500 target x=-0.0800)
== mujoco fixed dt=10 ms Kp=10000
   box z: min 46.787 mm max 59.092 mm (rest 50.0);  box vz RMS push 0.1428 / settle 0.0000 m/s;  pitch rate RMS push 3.431 rad/s
   tip z: min 49.1 mm max 99.9 mm (box mid 50);  box x end 0.280 m;  tip pen max 3.48 mm at t=1.63s (xt=0.1837 xb=0.2403 zt=0.0500 zb=0.0506 target x=0.1840)
   tip pen > 1 mm at 47 boundaries, t in [0.82, 1.87] s
== mujoco fixed dt=1 ms Kp=10000
   box z: min 49.991 mm max 52.292 mm (rest 50.0);  box vz RMS push 0.0593 / settle 0.0000 m/s;  pitch rate RMS push 0.695 rad/s
   tip z: min 49.7 mm max 99.9 mm (box mid 50);  box x end 0.280 m;  tip pen max 0.26 mm at t=1.72s (xt=0.2097 xb=0.2695 zt=0.0498 zb=0.0505 target x=0.2104)
== mujoco fixed dt=1 ms Kp=10000 PUSH FROZEN (box at rest, tip beside it)
   box z: min 49.997 mm max 49.998 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.000 rad/s
   tip z: min 49.9 mm max 99.9 mm (box mid 50);  box x end 0.000 m;  tip pen max 0.00 mm at t=0.00s (xt=-0.0800 xb=0.0000 zt=0.0999 zb=0.0500 target x=-0.0800)
== icf fixed dt=10 ms Kp=100
   box z: min 49.974 mm max 49.976 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.002 rad/s
   tip z: min 40.2 mm max 100.0 mm (box mid 50);  box x end 0.235 m;  tip pen max 0.08 mm at t=0.92s (xt=-0.0593 xb=0.0006 zt=0.0402 zb=0.0500 target x=-0.0290)
== icf fixed dt=1 ms Kp=100
   box z: min 49.975 mm max 49.977 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.009 rad/s
   tip z: min 40.2 mm max 99.6 mm (box mid 50);  box x end 0.236 m;  tip pen max 0.13 mm at t=0.91s (xt=-0.0594 xb=0.0005 zt=0.0402 zb=0.0500 target x=-0.0320)
== icf fixed dt=1 ms Kp=100 PUSH FROZEN (box at rest, tip beside it)
   box z: min 49.975 mm max 49.977 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.000 rad/s
   tip z: min 40.2 mm max 99.6 mm (box mid 50);  box x end 0.000 m;  tip pen max 0.00 mm at t=0.00s (xt=-0.0800 xb=0.0000 zt=0.0996 zb=0.0500 target x=-0.0800)
== icf fixed dt=10 ms Kp=10000
   box z: min 49.975 mm max 49.976 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.003 rad/s
   tip z: min 49.9 mm max 100.0 mm (box mid 50);  box x end 0.280 m;  tip pen max 0.17 mm at t=0.84s (xt=-0.0564 xb=0.0035 zt=0.0499 zb=0.0500 target x=-0.0530)
== icf fixed dt=1 ms Kp=10000
   box z: min 49.975 mm max 49.977 mm (rest 50.0);  box vz RMS push 0.0001 / settle 0.0000 m/s;  pitch rate RMS push 0.014 rad/s
   tip z: min 49.9 mm max 99.9 mm (box mid 50);  box x end 0.280 m;  tip pen max 0.20 mm at t=0.82s (xt=-0.0597 xb=0.0001 zt=0.0499 zb=0.0500 target x=-0.0590)
== icf fixed dt=1 ms Kp=10000 PUSH FROZEN (box at rest, tip beside it)
   box z: min 49.975 mm max 49.977 mm (rest 50.0);  box vz RMS push 0.0000 / settle 0.0000 m/s;  pitch rate RMS push 0.000 rad/s
   tip z: min 49.9 mm max 99.9 mm (box mid 50);  box x end 0.000 m;  tip pen max 0.00 mm at t=0.00s (xt=-0.0800 xb=0.0000 zt=0.0999 zb=0.0500 target x=-0.0800)
```

Reading: MuJoCo's pushed box rocks and hops (pitch rate 0.7–3.4 rad/s RMS, centre up to 2 cm above rest at K_p = 1e2, 2.3 mm at K_p = 1e4 and δt = 1 ms) and at K_p = 1e2 the tip climbs onto it (tip z 112 mm > box top 100 mm) — the '18 mm tip penetration' of the first sweep was the horizontal overlap of a tip riding on the box, which the tightened metric now excludes. ICF's box slides flat (pitch rate ≤ 0.014 rad/s, z within 2 µm of its 25 µm resting depth = m g / (4 k)) at both steps. With the push frozen neither backend moves the box: the hop is a sliding-contact behaviour of MuJoCo's soft constraint, not a resting one.
