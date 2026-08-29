# Part-1 results tables (generated from the CSVs — do not edit)

## Work-precision — wall time [s] per simulated second (δt_max = 100 ms)

| scene, N | arm | ε = 10^-1 | ε = 10^-2 | ε = 10^-3 | ε = 10^-4 | ε = 10^-5 | ε = 10^-6 |
|---|---|---|---|---|---|---|---|
| soft, 1 | ICF error control | 0.0977 | 0.199 | 0.484 | 1.2 | 3.87 | 11.1 |
| soft, 1 | MuJoCo error control | 0.0413 | 0.132 | 0.301 | 0.796 | 1.76 | 4 |
| soft, 1024 | ICF error control | 3.83 | 7.78 | 16.9 | 39.5 | 162 | 458 |
| soft, 1024 | MuJoCo error control | 0.0767 | 0.277 | 0.667 | 2.4 | 5.31 | 11.5 |
| hard, 1 | ICF error control | 0.38 | 0.603 | 1.35 | 2.89 | 7.91 | 18.2 |
| hard, 1 | MuJoCo error control | 0.0917 | 0.301 | 1.04 | 2.92 | 5.63 | 13.8 |
| hard, 1024 | ICF error control | 15 | 23.8 | 49.3 | 115 | 343 | 856 |
| hard, 1024 | MuJoCo error control | 0.47 | 1.54 | 5.02 | 15.8 | 37.8 | 88.4 |

| scene, N | arm | δt = 10 ms | δt = 5 ms | δt = 2 ms | δt = 1 ms |
|---|---|---|---|---|---|
| soft, 1 | ICF fixed step | 0.111 | 0.187 | 0.422 | 0.767 |
| soft, 1 | MuJoCo fixed step | 0.0412 | 0.0834 | 0.203 | 0.403 |
| soft, 1024 | ICF fixed step | 3.91 | 6.66 | 14.3 | 25.1 |
| soft, 1024 | MuJoCo fixed step | 0.0835 | 0.169 | 0.417 | 0.844 |
| hard, 1 | ICF fixed step | 0.204 | 0.294 | 0.588 | 1.13 |
| hard, 1 | MuJoCo fixed step | 0.0637 | 0.125 | 0.338 | 0.663 |
| hard, 1024 | ICF fixed step | 7.47 | 10.9 | 20.9 | 38.2 |
| hard, 1024 | MuJoCo fixed step | 0.183 | 0.389 | 1.04 | 2.11 |

## Penetration and ejections — 64 worlds, 200 boundaries

| scene | arm | setting | mean [µm] | max [mm] | p95 [µm] | ejected | wall/boundary [ms] |
|---|---|---|---|---|---|---|---|
| soft | ICF error control | ε = 1e-01 | 664.39 | 4.059 | 873.6 | 0.0% | 18.32 |
| soft | ICF error control | ε = 1e-02 | 662.21 | 3.923 | 961.7 | 0.0% | 56.37 |
| soft | ICF error control | ε = 1e-03 | 663.35 | 3.856 | 994.9 | 0.0% | 97.13 |
| soft | ICF error control | ε = 1e-04 | 662.87 | 3.556 | 971.6 | 0.0% | 234.49 |
| soft | MuJoCo error control | ε = 1e-01 | 3094.81 | 4.918 | 4895.3 | 0.0% | 1.98 |
| soft | MuJoCo error control | ε = 1e-02 | 1018.17 | 5.714 | 2800.5 | 0.0% | 13.54 |
| soft | MuJoCo error control | ε = 1e-03 | 622.94 | 6.174 | 915.9 | 0.0% | 29.20 |
| soft | MuJoCo error control | ε = 1e-04 | 577.37 | 6.391 | 1064.8 | 0.0% | 95.85 |
| soft | ICF fixed step | δt = 10 ms | 669.38 | 4.247 | 974.3 | 0.0% | 36.40 |
| soft | ICF fixed step | δt = 5 ms | 664.47 | 4.523 | 971.4 | 0.0% | 62.55 |
| soft | ICF fixed step | δt = 2 ms | 659.97 | 4.479 | 991.4 | 0.0% | 119.56 |
| soft | ICF fixed step | δt = 1 ms | 658.67 | 4.629 | 1007.8 | 0.0% | 213.85 |
| soft | MuJoCo fixed step | δt = 10 ms | 516.53 | 3.840 | 619.7 | 0.0% | 4.63 |
| soft | MuJoCo fixed step | δt = 5 ms | 550.38 | 4.453 | 856.8 | 0.0% | 8.80 |
| soft | MuJoCo fixed step | δt = 2 ms | 561.16 | 5.779 | 942.6 | 0.0% | 21.48 |
| soft | MuJoCo fixed step | δt = 1 ms | 562.75 | 6.515 | 1003.1 | 0.0% | 46.88 |
| hard | ICF error control | ε = 1e-01 | 30.05 | 6.971 | 95.9 | 0.0% | 174.03 |
| hard | ICF error control | ε = 1e-02 | 7.09 | 0.828 | 16.7 | 0.0% | 227.72 |
| hard | ICF error control | ε = 1e-03 | 6.18 | 0.334 | 15.3 | 0.0% | 367.14 |
| hard | ICF error control | ε = 1e-04 | 6.12 | 0.320 | 14.8 | 0.0% | 693.60 |
| hard | MuJoCo error control | ε = 1e-01 | 549.78 | 10.054 | 1237.8 | 0.0% | 22.75 |
| hard | MuJoCo error control | ε = 1e-02 | 149.99 | 3.854 | 444.5 | 0.0% | 60.55 |
| hard | MuJoCo error control | ε = 1e-03 | 20.30 | 1.114 | 51.9 | 0.0% | 188.09 |
| hard | MuJoCo error control | ε = 1e-04 | 7.04 | 0.714 | 10.0 | 0.0% | 716.91 |
| hard | ICF fixed step | δt = 10 ms | 225.99 | 27.753 | 475.0 | 1.6% | 97.43 |
| hard | ICF fixed step | δt = 5 ms | 34.58 | 9.439 | 79.6 | 0.0% | 150.86 |
| hard | ICF fixed step | δt = 2 ms | 8.36 | 3.904 | 17.1 | 0.0% | 250.04 |
| hard | ICF fixed step | δt = 1 ms | 6.44 | 0.865 | 15.5 | 0.0% | 407.36 |
| hard | MuJoCo fixed step | δt = 10 ms | 336.14 | 6.791 | 527.2 | 0.0% | 10.11 |
| hard | MuJoCo fixed step | δt = 5 ms | 100.21 | 3.070 | 152.3 | 0.0% | 21.45 |
| hard | MuJoCo fixed step | δt = 2 ms | 17.00 | 0.866 | 40.6 | 0.0% | 56.99 |
| hard | MuJoCo fixed step | δt = 1 ms | 6.55 | 0.719 | 8.4 | 0.0% | 115.21 |

## Wall time per boundary [ms] (δt_max = 100 ms on the clutters) vs parallel worlds — median of 3 runs (spread in brackets)

**soft-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 107 [1e+02–1.1e+02] | 145 [1.4e+02–1.5e+02] | 219 [2.2e+02–2.2e+02] | 398 [3.8e+02–4e+02] | 729 [7.3e+02–7.5e+02] | 1.46e+03 [1.4e+03–1.5e+03] | 2.86e+03 [2.9e+03–2.9e+03] | 5.58e+03 [5.5e+03–5.6e+03] |
| MuJoCo error control | 28.4 [28–29] | 29.6 [29–32] | 32.5 [32–33] | 37.2 [37–37] | 50.2 [49–51] | 83.2 [80–90] | 139 [1.4e+02–1.4e+02] | 281 [2.8e+02–2.8e+02] |
| ICF fixed step | 36 [36–36] | 54.3 [54–55] | 85.6 [85–86] | 153 [1.5e+02–1.5e+02] | 317 [3.1e+02–3.2e+02] | 629 [6.3e+02–6.3e+02] | 1.27e+03 [1.3e+03–1.3e+03] | 2.5e+03 [2.5e+03–2.5e+03] |
| MuJoCo fixed step | 4.66 [4.7–4.7] | 4.74 [4.7–4.7] | 5.09 [5.1–5.1] | 5.91 [5.9–5.9] | 7.85 [7.8–7.8] | 13.8 [14–14] | 24.2 [24–24] | 45.2 [45–45] |

**hard-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 322 [3.2e+02–3.4e+02] | 474 [4.6e+02–5.4e+02] | 651 [6.1e+02–6.9e+02] | 1.03e+03 [1e+03–1e+03] | 1.79e+03 [1.7e+03–1.8e+03] | 3.28e+03 [3.2e+03–3.4e+03] | 6.32e+03 [6.3e+03–6.4e+03] | 1.27e+04 [1.3e+04–1.3e+04] |
| MuJoCo error control | 174 [1.5e+02–1.9e+02] | 224 [2.2e+02–2.4e+02] | 263 [2.6e+02–2.7e+02] | 316 [3.1e+02–3.2e+02] | 415 [4.1e+02–4.3e+02] | 719 [7.2e+02–7.3e+02] | 1.22e+03 [1.1e+03–1.2e+03] | 2.19e+03 [2.2e+03–2.3e+03] |
| ICF fixed step | 101 [99–1e+02] | 142 [1.4e+02–1.4e+02] | 211 [2e+02–2.1e+02] | 338 [3.3e+02–3.4e+02] | 613 [6.1e+02–6.2e+02] | 1.26e+03 [1.3e+03–1.3e+03] | 2.54e+03 [2.5e+03–2.6e+03] | 5.06e+03 [5e+03–5.1e+03] |
| MuJoCo fixed step | 10.1 [10–10] | 10.7 [11–11] | 12 [12–12] | 14.4 [14–14] | 18.6 [19–19] | 30.2 [30–30] | 50.4 [50–50] | 90.7 [91–91] |


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
| MuJoCo fixed step | δt = 0.01 s | +56956.91 | 1 | ok |
| MuJoCo fixed step | δt = 0.005 s | +51887.05 | 2 | ok |
| MuJoCo fixed step | δt = 0.002 s | -78.88 | 10 | ok |
| MuJoCo fixed step | δt = 0.001 s | -100.00 | 25 | ok |
| MuJoCo fixed step | δt = 0.0005 s | -100.00 | 25 | ok |
| MuJoCo fixed step | δt = 0.0002 s | -100.00 | 22 | ok |
| MuJoCo fixed step | δt = 0.0001 s | -100.00 | 24 | ok |
| MuJoCo fixed step | δt = 5e-05 s | -100.00 | 23 | ok |
| MuJoCo fixed step | δt = 2e-05 s | -100.00 | 25 | ok |
| MuJoCo fixed step | δt = 1e-05 s | -100.00 | 25 | ok |
| ICF error control | ε = 1e-01 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-02 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-03 | -100.10 | 5 | ok |
| ICF error control | ε = 1e-04 | -100.10 | 14 | ok |
| ICF error control | ε = 1e-05 | -52.45 | 11 | budget-exhausted |
| ICF error control | ε = 1e-06 | -91.70 | 0 | budget-exhausted |
| MuJoCo error control | ε = 1e-01 | +1517.79 | 4 | ok |
| MuJoCo error control | ε = 1e-02 | +2635.09 | 5 | ok |
| MuJoCo error control | ε = 1e-03 | +4920.87 | 3 | ok |
| MuJoCo error control | ε = 1e-04 | +162.83 | 9 | ok |
| MuJoCo error control | ε = 1e-05 | -88.17 | 17 | ok |
| MuJoCo error control | ε = 1e-06 | -100.00 | 26 | ok |
