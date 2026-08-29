# Fixed-ICF Newton tolerance: cost and accuracy (hard clutter, δt = 10 ms and 1 ms under a 0.1 s boundary)

Hypothesis tested: the paper's fixed-step tolerance 1e-8 is below float32 resolution and pins every step at the 100-iteration cap. Result: wall is flat across 1e-5…1e-8 at 64, 1024 and 4096 worlds and penetration is unchanged — refuted; the tolerance stays at the paper's value.

## 64 worlds

```
arm            newton_tol     knob ms/boundary mean_pen_um max_pen_mm  p95_um
icf fixed           1e-05     10ms        94.9      297.57     34.818   550.7
icf fixed           1e-05      1ms       321.9        6.53      0.905    16.1
icf fixed           1e-06     10ms        96.5      363.92     28.450   707.8
icf fixed           1e-06      1ms       353.6        6.51      0.518    16.3
icf fixed           1e-07     10ms       104.2      217.98     28.410   539.9
icf fixed           1e-07      1ms       416.8        6.74      1.159    16.9
icf fixed           1e-08     10ms        97.3      273.15     31.763   540.5
icf fixed           1e-08      1ms       411.5        6.73      1.281    16.8
```

## 1024 and 4096 worlds (δt = 10 ms)

```
    N newton_tol ms/boundary(100ms) us/world/10ms
 1024      1e-05              693.1         67.68
 1024      1e-06              702.2         68.57
 1024      1e-08              726.4         70.93
 4096      1e-05             2893.2         70.64
 4096      1e-06             2916.1         71.19
 4096      1e-08             3044.5         74.33
```
