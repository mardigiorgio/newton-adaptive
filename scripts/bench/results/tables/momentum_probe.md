# Zero-gravity momentum conservation (scripts/bench/probe_momentum.py)

Two 65 g spheres (k = 1e5 N/m, μ = 0.5) collide head-on with gravity zero in
every world, 0.6 s; |p_end − p_0| / |p_0| of the pair's linear momentum. Every
arm conserves momentum to solver precision: neither contact solve nor the
per-world step controller injects momentum.

| arm | setting | momentum drift |
|---|---|---|
| ICF fixed | δt = 10 ms | 1.1e-7 |
| ICF fixed | δt = 1 ms | 1.1e-7 |
| ICF error control | ε = 1e-2 | 0 |
| ICF error control | ε = 1e-4 | 0 |
| MuJoCo fixed | δt = 10 ms | 1.0e-5 |
| MuJoCo fixed | δt = 1 ms | 4.6e-7 |
| MuJoCo error control | ε = 1e-2 | 8.9e-6 |
| MuJoCo error control | ε = 1e-4 | 3.9e-6 |
