# March cost vs requested accuracy (probe_march_cost.py, one world, idle GPU, dt_max 0.1 s on clutter; calibrated MuJoCo solref)

## hard-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01          72         0.50     6917      0.00
icf-adaptive      1e-02         264         1.09     4132      0.00
icf-adaptive      1e-03         736         2.27     3088      0.00
icf-adaptive      1e-04        1621         4.30     2653      0.00
icf-adaptive      1e-05        4759        13.07     2746      0.00
icf-adaptive      1e-06       14408        42.01     2916      0.00
mujoco-adaptive   1e-01          54         0.11     2062      0.00
mujoco-adaptive   1e-02         259         0.50     1916      0.00
mujoco-adaptive   1e-03         699         1.50     2145      0.00
mujoco-adaptive   1e-04        2348         4.68     1995      0.00
mujoco-adaptive   1e-05        5330         8.68     1628      0.00
mujoco-adaptive   1e-06       11611        18.34     1580      0.00
```

## soft-clutter

```
arm                 eps iters/sim_s wall_s/sim_s  us/iter exhausted
icf-adaptive      1e-01          32         0.20     6276      0.00
icf-adaptive      1e-02         126         0.42     3320      0.00
icf-adaptive      1e-03         382         0.86     2258      0.00
icf-adaptive      1e-04         968         1.91     1977      0.00
icf-adaptive      1e-05        2856         6.35     2222      0.00
icf-adaptive      1e-06        8732        22.24     2547      0.00
mujoco-adaptive   1e-01          40         0.07     1776      0.00
mujoco-adaptive   1e-02         141         0.24     1694      0.00
mujoco-adaptive   1e-03         320         0.53     1663      0.00
mujoco-adaptive   1e-04         890         1.37     1545      0.00
mujoco-adaptive   1e-05        2489         3.16     1270      0.00
mujoco-adaptive   1e-06        7691         7.32      952      0.00
```

