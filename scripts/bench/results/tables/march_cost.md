# March cost vs requested accuracy (probe_march_cost.py, one world, idle GPU, dt_max 0.1 s on clutter)

## hard-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01          90         0.58     6414      0.00
icf-adaptive      1e-02         271         1.13     4162      0.00
icf-adaptive      1e-03         696         2.19     3150      0.00
icf-adaptive      1e-04        1732         4.49     2590      0.00
icf-adaptive      1e-05        4662        12.79     2743      0.00
icf-adaptive      1e-06       14970        43.28     2891      0.00
mujoco-adaptive   1e-01          61         0.15     2464      0.00
mujoco-adaptive   1e-02         244         0.55     2268      0.00
mujoco-adaptive   1e-03         876         1.72     1957      0.00
mujoco-adaptive   1e-04        2095         3.92     1871      0.00
mujoco-adaptive   1e-05        7016        13.55     1932      0.00
mujoco-adaptive   1e-06       15059        26.08     1732      0.00
```

## soft-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01          26         0.18     6933      0.00
icf-adaptive      1e-02         126         0.40     3143      0.00
icf-adaptive      1e-03         360         0.82     2275      0.00
icf-adaptive      1e-04         979         1.91     1951      0.00
icf-adaptive      1e-05        2848         6.39     2246      0.00
icf-adaptive      1e-06        8680        21.96     2531      0.00
mujoco-adaptive   1e-01          40         0.08     2056      0.00
mujoco-adaptive   1e-02         122         0.23     1916      0.00
mujoco-adaptive   1e-03         289         0.49     1695      0.00
mujoco-adaptive   1e-04         809         1.15     1427      0.00
mujoco-adaptive   1e-05        2389         2.60     1088      0.00
mujoco-adaptive   1e-06        7579         6.48      854      0.00
```

