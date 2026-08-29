# Part-1 results tables (generated from the CSVs — do not edit)

## Work-precision — wall time [s] per simulated second (δt_max = 100 ms)

| scene, N | arm | ε = 10^-1 | ε = 10^-2 | ε = 10^-3 | ε = 10^-4 | ε = 10^-5 | ε = 10^-6 |
|---|---|---|---|---|---|---|---|
| soft, 1 | ICF error control | 0.109 | 0.161 | 0.413 | 1.05 | 3.01 | 8.55 |
| soft, 1 | MuJoCo error control | 0.0542 | 0.148 | 0.307 | 0.737 | 1.74 | 4.43 |
| soft, 1024 | ICF error control | 3.84 | 7.84 | 17 | 39.6 | 164 | 462 |
| soft, 1024 | MuJoCo error control | 0.107 | 0.296 | 0.717 | 1.96 | 5.59 | 14 |
| hard, 1 | ICF error control | 0.336 | 0.677 | 0.894 | 2.02 | 5.75 | 13.8 |
| hard, 1 | MuJoCo error control | 0.113 | 0.377 | 1.02 | 3.31 | 7.85 | 22.9 |
| hard, 1024 | ICF error control | 14.6 | 23.9 | 50 | 118 | 348 | 887 |
| hard, 1024 | MuJoCo error control | 0.531 | 1.73 | 6.71 | 19 | 49.1 | budget-exhausted |

| scene, N | arm | δt = 10 ms | δt = 5 ms | δt = 2 ms | δt = 1 ms |
|---|---|---|---|---|---|
| soft, 1 | ICF fixed step | 0.0886 | 0.147 | 0.357 | 0.669 |
| soft, 1 | MuJoCo fixed step | 0.0593 | 0.109 | 0.256 | 0.451 |
| soft, 1024 | ICF fixed step | 3.94 | 6.67 | 14.3 | 25.2 |
| soft, 1024 | MuJoCo fixed step | 0.116 | 0.215 | 0.535 | 0.927 |
| hard, 1 | ICF fixed step | 0.195 | 0.231 | 0.463 | 0.855 |
| hard, 1 | MuJoCo fixed step | 0.0796 | 0.114 | 0.282 | 0.7 |
| hard, 1024 | ICF fixed step | 7.46 | 11 | 20.9 | 38.6 |
| hard, 1024 | MuJoCo fixed step | 0.23 | 0.393 | 0.973 | 2.25 |

## Penetration and ejections — 64 worlds, 200 boundaries

| scene | arm | setting | mean [µm] | max [mm] | p95 [µm] | ejected | wall/boundary [ms] |
|---|---|---|---|---|---|---|---|
| soft | ICF error control | ε = 1e-01 | 662.66 | 3.770 | 836.5 | 0.0% | 17.64 |
| soft | ICF error control | ε = 1e-02 | 662.47 | 4.172 | 959.3 | 0.0% | 52.76 |
| soft | ICF error control | ε = 1e-03 | 664.05 | 3.707 | 997.1 | 0.0% | 103.12 |
| soft | ICF error control | ε = 1e-04 | 663.03 | 3.555 | 975.4 | 0.0% | 234.58 |
| soft | MuJoCo error control | ε = 1e-01 | 1728.93 | 31.288 | 3015.8 | 0.0% | 5.25 |
| soft | MuJoCo error control | ε = 1e-02 | 1800.94 | 35.268 | 5838.3 | 0.0% | 13.84 |
| soft | MuJoCo error control | ε = 1e-03 | 1786.19 | 36.695 | 5929.3 | 0.0% | 32.16 |
| soft | MuJoCo error control | ε = 1e-04 | 1839.51 | 37.794 | 7028.7 | 0.0% | 106.15 |
| soft | ICF fixed step | δt = 10 ms | 667.74 | 3.896 | 965.8 | 0.0% | 36.06 |
| soft | ICF fixed step | δt = 5 ms | 663.99 | 4.504 | 968.7 | 0.0% | 60.02 |
| soft | ICF fixed step | δt = 2 ms | 659.64 | 4.799 | 991.7 | 0.0% | 122.03 |
| soft | ICF fixed step | δt = 1 ms | 658.22 | 4.398 | 1009.6 | 0.0% | 204.41 |
| soft | MuJoCo fixed step | δt = 10 ms | 1498.70 | 26.844 | 4238.8 | 0.0% | 7.15 |
| soft | MuJoCo fixed step | δt = 5 ms | 1656.49 | 35.032 | 5853.7 | 0.0% | 12.35 |
| soft | MuJoCo fixed step | δt = 2 ms | 1783.38 | 36.426 | 7104.7 | 0.0% | 34.03 |
| soft | MuJoCo fixed step | δt = 1 ms | 1831.64 | 37.087 | 6457.6 | 0.0% | 58.84 |
| hard | ICF error control | ε = 1e-01 | 31.39 | 5.487 | 108.9 | 0.0% | 165.97 |
| hard | ICF error control | ε = 1e-02 | 7.28 | 1.081 | 16.8 | 0.0% | 195.44 |
| hard | ICF error control | ε = 1e-03 | 6.15 | 0.302 | 15.7 | 0.0% | 355.01 |
| hard | ICF error control | ε = 1e-04 | 6.14 | 0.283 | 15.2 | 0.0% | 720.06 |
| hard | MuJoCo error control | ε = 1e-01 | 2057.32 | 12.006 | 4884.8 | 0.0% | 28.11 |
| hard | MuJoCo error control | ε = 1e-02 | 516.02 | 9.778 | 1137.1 | 0.0% | 75.02 |
| hard | MuJoCo error control | ε = 1e-03 | 378.52 | 4.229 | 978.4 | 0.0% | 265.58 |
| hard | MuJoCo error control | ε = 1e-04 | 369.06 | 4.766 | 977.8 | 0.0% | 848.30 |
| hard | ICF fixed step | δt = 10 ms | 250.00 | 28.269 | 487.8 | 1.4% | 102.70 |
| hard | ICF fixed step | δt = 5 ms | 35.86 | 11.448 | 78.7 | 0.0% | 151.46 |
| hard | ICF fixed step | δt = 2 ms | 8.47 | 3.589 | 16.3 | 0.0% | 266.66 |
| hard | ICF fixed step | δt = 1 ms | 6.57 | 1.105 | 16.2 | 0.0% | 415.74 |
| hard | MuJoCo fixed step | δt = 10 ms | 883.00 | 10.897 | 2058.3 | 0.0% | 13.66 |
| hard | MuJoCo fixed step | δt = 5 ms | 327.33 | 4.096 | 638.8 | 0.0% | 20.37 |
| hard | MuJoCo fixed step | δt = 2 ms | 350.50 | 3.953 | 799.1 | 0.0% | 52.93 |
| hard | MuJoCo fixed step | δt = 1 ms | 356.08 | 3.852 | 862.8 | 0.0% | 127.21 |

## Wall time per 10 ms boundary [ms] vs parallel worlds — median of 3 runs (spread in brackets)

**soft-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 101 [1e+02–1.2e+02] | 150 [1.4e+02–1.6e+02] | 226 [2.2e+02–2.5e+02] | 393 [3.7e+02–4e+02] | 741 [7.3e+02–7.5e+02] | 1.44e+03 [1.4e+03–1.4e+03] | 2.83e+03 [2.8e+03–2.9e+03] | 5.59e+03 [5.6e+03–5.6e+03] |
| MuJoCo error control | 34.5 [34–35] | 35 [35–35] | 38 [38–38] | 43 [43–43] | 54.1 [54–54] | 91.1 [90–94] | 151 [1.5e+02–1.5e+02] | 271 [2.7e+02–2.7e+02] |
| ICF fixed step | 36.5 [36–37] | 54.1 [54–54] | 86.6 [87–88] | 152 [1.5e+02–1.5e+02] | 316 [3.1e+02–3.2e+02] | 631 [6.3e+02–6.4e+02] | 1.27e+03 [1.3e+03–1.3e+03] | 2.5e+03 [2.5e+03–2.5e+03] |
| MuJoCo fixed step | 7.17 [7.1–7.2] | 7.3 [7.3–7.3] | 7.77 [7.8–7.8] | 8.85 [8.8–8.8] | 11.5 [11–12] | 20.1 [20–20] | 34.2 [34–34] | 62.3 [62–62] |

**hard-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 355 [3.2e+02–3.6e+02] | 473 [4.7e+02–4.8e+02] | 704 [6e+02–7.4e+02] | 1.08e+03 [1.1e+03–1.1e+03] | 1.77e+03 [1.8e+03–1.8e+03] | 3.38e+03 [3.4e+03–3.4e+03] | 6.59e+03 [6.3e+03–6.7e+03] | 1.25e+04 [1.2e+04–1.2e+04] |
| MuJoCo error control | 273 [2.6e+02–3e+02] | 308 [3.1e+02–3.2e+02] | 378 [3.8e+02–3.8e+02] | 445 [4.4e+02–4.5e+02] | 607 [5.9e+02–6.2e+02] | 967 [9.4e+02–9.7e+02] | 1.65e+03 [1.6e+03–1.7e+03] | 3.03e+03 [3e+03–3.2e+03] |
| ICF fixed step | 104 [1e+02–1e+02] | 142 [1.4e+02–1.4e+02] | 202 [2e+02–2.1e+02] | 331 [3.3e+02–3.4e+02] | 629 [6.2e+02–6.5e+02] | 1.27e+03 [1.3e+03–1.3e+03] | 2.56e+03 [2.5e+03–2.6e+03] | 5.09e+03 [5.1e+03–6.1e+03] |
| MuJoCo fixed step | 13.5 [13–14] | 14.2 [14–14] | 15.8 [16–16] | 18.6 [19–19] | 24 [24–24] | 38.5 [38–39] | 63.9 [64–64] | 114 [1.1e+02–1.1e+02] |


## Bouncing ball — energy change after 10 s and rebounds

| arm | setting | energy change [%] | rebounds | status |
|---|---|---|---|---|
| ICF fixed step | δt = 0.01 s | -100.10 | 2 | ok |
| ICF fixed step | δt = 0.005 s | -100.10 | 4 | ok |
| ICF fixed step | δt = 0.002 s | -100.10 | 7 | ok |
| ICF fixed step | δt = 0.001 s | -100.10 | 11 | ok |
| ICF fixed step | δt = 0.0005 s | -98.05 | 21 | ok |
| ICF fixed step | δt = 0.0002 s | -57.09 | 13 | ok |
| ICF fixed step | δt = 0.0001 s | -32.21 | 12 | ok |
| ICF fixed step | δt = 5e-05 s | -16.29 | 11 | ok |
| ICF fixed step | δt = 2e-05 s | -6.85 | 11 | ok |
| ICF fixed step | δt = 1e-05 s | -3.48 | 11 | ok |
| MuJoCo fixed step | δt = 0.01 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 0.005 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 0.002 s | -100.04 | 0 | ok |
| MuJoCo fixed step | δt = 0.001 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 0.0005 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 0.0002 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 0.0001 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 5e-05 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 2e-05 s | -100.04 | 1 | ok |
| MuJoCo fixed step | δt = 1e-05 s | -100.04 | 1 | ok |
| ICF error control | ε = 1e-01 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-02 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-03 | -100.10 | 5 | ok |
| ICF error control | ε = 1e-04 | -100.10 | 14 | ok |
| ICF error control | ε = 1e-05 | -52.45 | 11 | budget-exhausted |
| ICF error control | ε = 1e-06 | -91.70 | 0 | budget-exhausted |
| MuJoCo error control | ε = 1e-01 | -100.04 | 1 | ok |
| MuJoCo error control | ε = 1e-02 | -100.04 | 1 | ok |
| MuJoCo error control | ε = 1e-03 | -100.04 | 1 | ok |
| MuJoCo error control | ε = 1e-04 | -100.04 | 0 | ok |
| MuJoCo error control | ε = 1e-05 | -100.04 | 1 | ok |
| MuJoCo error control | ε = 1e-06 | -100.04 | 1 | ok |
