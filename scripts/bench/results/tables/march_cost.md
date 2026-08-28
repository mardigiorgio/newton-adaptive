# March cost vs requested accuracy (probe_march_cost.py, one world, idle GPU)

iters = march iterations (step-doubling attempts) per simulated second; wall per simulated second; µs per iteration; exhausted fraction.

## hard-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01         100         0.24     2433      0.00
icf-adaptive      1e-02         100         0.25     2480      0.00
icf-adaptive      1e-03         100         0.25     2474      0.00
icf-adaptive      1e-04         171         0.39     2255      0.00
icf-adaptive      1e-05         342         0.65     1903      0.00
icf-adaptive      1e-06        1027         1.62     1580      0.00
mujoco-adaptive   1e-01         100         0.15     1510      0.00
mujoco-adaptive   1e-02         118         0.18     1549      0.00
mujoco-adaptive   1e-03         127         0.14     1125      0.00
mujoco-adaptive   1e-04         239         0.29     1216      0.00
mujoco-adaptive   1e-05        1220         1.74     1424      0.00
mujoco-adaptive   1e-06        3812         4.97     1304      0.00
```

## soft-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01         100         0.18     1802      0.00
icf-adaptive      1e-02         100         0.18     1839      0.00
icf-adaptive      1e-03         128         0.24     1858      0.00
icf-adaptive      1e-04         166         0.28     1658      0.00
icf-adaptive      1e-05         359         0.51     1414      0.00
icf-adaptive      1e-06         908         1.15     1261      0.00
mujoco-adaptive   1e-01         100         0.11     1114      0.00
mujoco-adaptive   1e-02         100         0.11     1115      0.00
mujoco-adaptive   1e-03         130         0.13     1010      0.00
mujoco-adaptive   1e-04         296         0.26      874      0.00
mujoco-adaptive   1e-05         841         0.63      751      0.00
mujoco-adaptive   1e-06        2509         1.79      713      0.00
```

