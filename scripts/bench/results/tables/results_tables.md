# Part-1 results tables (generated from the CSVs — do not edit)

## Work-precision — wall time [s] per simulated second (δt_max = 100 ms)

| scene, N | arm | ε = 10^-1 | ε = 10^-2 | ε = 10^-3 | ε = 10^-4 | ε = 10^-5 | ε = 10^-6 |
|---|---|---|---|---|---|---|---|
| soft, 1 | ICF error control | 0.131 | 0.205 | 0.571 | 1.47 | 5.17 | 14.3 |
| soft, 1 | MuJoCo error control | 0.0447 | 0.12 | 0.25 | 0.635 | 1.46 | 3.41 |
| soft, 1024 | ICF error control | 4.59 | 8.25 | 19.8 | 53 | 214 | 608 |
| soft, 1024 | MuJoCo error control | 0.0817 | 0.239 | 0.592 | 1.68 | 4.41 | 9.91 |
| hard, 1 | ICF error control | 0.304 | 0.553 | 1.17 | 3.29 | 9.27 | 25.4 |
| hard, 1 | MuJoCo error control | 0.0926 | 0.299 | 1.04 | 2.81 | 5.78 | 15.2 |
| hard, 1024 | ICF error control | 11.5 | 21.9 | 45.8 | 129 | 422 | 1.13e+03 |
| hard, 1024 | MuJoCo error control | 0.513 | 1.61 | 5.25 | 15.6 | 37.2 | 91 |

| scene, N | arm | δt = 10 ms | δt = 5 ms | δt = 2 ms | δt = 1 ms |
|---|---|---|---|---|---|
| soft, 1 | ICF fixed step | 0.112 | 0.207 | 0.426 | 0.822 |
| soft, 1 | MuJoCo fixed step | 0.0435 | 0.0958 | 0.214 | 0.409 |
| soft, 1024 | ICF fixed step | 3.99 | 7.36 | 14.7 | 29 |
| soft, 1024 | MuJoCo fixed step | 0.0875 | 0.19 | 0.435 | 0.838 |
| hard, 1 | ICF fixed step | 0.185 | 0.289 | 0.582 | 1.07 |
| hard, 1 | MuJoCo fixed step | 0.0647 | 0.124 | 0.34 | 0.628 |
| hard, 1024 | ICF fixed step | 6.31 | 10.4 | 21.3 | 38.3 |
| hard, 1024 | MuJoCo fixed step | 0.183 | 0.387 | 1.04 | 2.11 |

## Penetration and ejections — 64 worlds, 200 boundaries

| scene | arm | setting | mean [µm] | max [mm] | p95 [µm] | ejected | wall/boundary [ms] |
|---|---|---|---|---|---|---|---|
| soft | ICF error control | ε = 1e-01 | 650.03 | 4.546 | 1005.8 | 0.0% | 31.18 |
| soft | ICF error control | ε = 1e-02 | 653.86 | 7.230 | 1058.7 | 0.0% | 78.60 |
| soft | ICF error control | ε = 1e-03 | 662.75 | 7.244 | 1182.7 | 0.0% | 159.27 |
| soft | ICF error control | ε = 1e-04 | 670.29 | 8.746 | 1484.7 | 0.0% | 437.51 |
| soft | MuJoCo error control | ε = 1e-01 | 3560.42 | 22.330 | 4904.3 | 0.0% | 2.28 |
| soft | MuJoCo error control | ε = 1e-02 | 1442.18 | 22.888 | 3938.8 | 0.0% | 8.54 |
| soft | MuJoCo error control | ε = 1e-03 | 1139.25 | 23.662 | 3359.4 | 0.0% | 26.09 |
| soft | MuJoCo error control | ε = 1e-04 | 1093.88 | 19.955 | 3536.8 | 0.0% | 66.10 |
| soft | ICF fixed step | δt = 10 ms | 683.41 | 5.317 | 1475.9 | 0.0% | 38.62 |
| soft | ICF fixed step | δt = 5 ms | 680.37 | 6.876 | 1367.0 | 0.0% | 75.86 |
| soft | ICF fixed step | δt = 2 ms | 680.19 | 8.320 | 1301.5 | 0.0% | 138.69 |
| soft | ICF fixed step | δt = 1 ms | 683.86 | 9.292 | 1404.9 | 0.0% | 254.32 |
| soft | MuJoCo fixed step | δt = 10 ms | 963.35 | 21.949 | 3087.4 | 0.0% | 4.83 |
| soft | MuJoCo fixed step | δt = 5 ms | 1012.66 | 15.764 | 3489.2 | 0.0% | 11.59 |
| soft | MuJoCo fixed step | δt = 2 ms | 1068.38 | 19.175 | 3638.4 | 0.0% | 24.75 |
| soft | MuJoCo fixed step | δt = 1 ms | 1089.66 | 18.069 | 3665.9 | 0.0% | 47.11 |
| hard | ICF error control | ε = 1e-01 | 14.20 | 3.135 | 44.9 | 0.0% | 135.59 |
| hard | ICF error control | ε = 1e-02 | 7.37 | 1.069 | 16.1 | 0.0% | 197.08 |
| hard | ICF error control | ε = 1e-03 | 7.46 | 0.833 | 12.0 | 0.0% | 398.89 |
| hard | ICF error control | ε = 1e-04 | 6.26 | 0.632 | 10.2 | 0.0% | 967.34 |
| hard | MuJoCo error control | ε = 1e-01 | 503.61 | 10.055 | 1229.2 | 0.0% | 23.11 |
| hard | MuJoCo error control | ε = 1e-02 | 131.00 | 2.125 | 405.6 | 0.0% | 67.19 |
| hard | MuJoCo error control | ε = 1e-03 | 19.33 | 0.890 | 52.9 | 0.0% | 210.05 |
| hard | MuJoCo error control | ε = 1e-04 | 7.48 | 0.726 | 10.3 | 0.0% | 738.40 |
| hard | ICF fixed step | δt = 10 ms | 18.38 | 4.098 | 61.8 | 0.0% | 84.43 |
| hard | ICF fixed step | δt = 5 ms | 17.03 | 5.677 | 38.3 | 0.0% | 144.56 |
| hard | ICF fixed step | δt = 2 ms | 9.29 | 2.108 | 16.1 | 0.0% | 248.25 |
| hard | ICF fixed step | δt = 1 ms | 9.05 | 1.577 | 13.9 | 0.0% | 450.01 |
| hard | MuJoCo fixed step | δt = 10 ms | 385.32 | 13.403 | 962.8 | 0.0% | 10.29 |
| hard | MuJoCo fixed step | δt = 5 ms | 115.90 | 2.783 | 231.3 | 0.0% | 21.52 |
| hard | MuJoCo fixed step | δt = 2 ms | 21.58 | 1.530 | 43.0 | 0.0% | 57.10 |
| hard | MuJoCo fixed step | δt = 1 ms | 7.75 | 0.722 | 9.5 | 0.0% | 119.76 |

## Wall time per boundary [ms] (δt_max = 100 ms on the clutters) vs parallel worlds — median of 3 runs (spread in brackets)

**soft-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 121 [1.2e+02–1.3e+02] | 183 [1.7e+02–1.9e+02] | 268 [2.6e+02–2.7e+02] | 454 [4.5e+02–4.6e+02] | 835 [8.2e+02–8.7e+02] | 1.61e+03 [1.6e+03–1.6e+03] | 3.15e+03 [3.1e+03–3.2e+03] | 6.22e+03 [6.2e+03–6.3e+03] |
| MuJoCo error control | 26.2 [26–26] | 26.8 [27–29] | 30.7 [31–31] | 35.1 [32–36] | 45.4 [45–45] | 73.5 [73–75] | 122 [1.2e+02–1.2e+02] | 222 [2.2e+02–2.2e+02] |
| ICF fixed step | 33.7 [33–36] | 51.6 [50–52] | 79.8 [79–82] | 143 [1.4e+02–1.4e+02] | 293 [2.9e+02–2.9e+02] | 588 [5.9e+02–5.9e+02] | 1.18e+03 [1.2e+03–1.2e+03] | 2.33e+03 [2.3e+03–2.3e+03] |
| MuJoCo fixed step | 4.63 [4.6–4.6] | 4.7 [4.7–4.7] | 5.04 [5–5] | 5.89 [5.9–5.9] | 7.79 [7.8–7.8] | 13.7 [14–14] | 24 [24–24] | 44.9 [45–45] |

**hard-clutter**

| arm | 2^6 | 2^7 | 2^8 | 2^9 | 2^10 | 2^11 | 2^12 | 2^13 |
|---|---|---|---|---|---|---|---|---|
| ICF error control | 344 [3.1e+02–3.5e+02] | 413 [4e+02–4.2e+02] | 595 [5.7e+02–6.3e+02] | 878 [8.6e+02–9.3e+02] | 1.55e+03 [1.5e+03–1.6e+03] | 2.93e+03 [2.9e+03–3e+03] | 5.64e+03 [5.6e+03–5.8e+03] | 1.11e+04 [1.1e+04–1.1e+04] |
| MuJoCo error control | 159 [1.6e+02–1.8e+02] | 209 [1.8e+02–2.1e+02] | 255 [2.5e+02–2.6e+02] | 305 [3e+02–3.4e+02] | 432 [4.2e+02–4.4e+02] | 705 [6.9e+02–7.1e+02] | 1.17e+03 [1.2e+03–1.2e+03] | 2.29e+03 [2.2e+03–2.3e+03] |
| ICF fixed step | 75.9 [74–80] | 112 [1.1e+02–1.1e+02] | 170 [1.6e+02–1.7e+02] | 275 [2.7e+02–2.8e+02] | 513 [5.1e+02–5.1e+02] | 1.01e+03 [1e+03–1e+03] | 2.01e+03 [2e+03–2e+03] | 3.96e+03 [3.9e+03–4e+03] |
| MuJoCo fixed step | 10.3 [10–10] | 10.7 [11–11] | 12.1 [12–12] | 14.4 [14–14] | 18.4 [18–19] | 30.3 [30–30] | 50.5 [50–51] | 90.8 [90–91] |


## Bouncing ball — energy change after 10 s and rebounds

| arm | setting | energy change [%] | rebounds | status |
|---|---|---|---|---|
| ICF fixed step | δt = 0.01 s | -100.10 | 2 | ok |
| ICF fixed step | δt = 0.005 s | -100.10 | 4 | ok |
| ICF fixed step | δt = 0.002 s | -100.10 | 7 | ok |
| ICF fixed step | δt = 0.001 s | -100.10 | 11 | ok |
| ICF fixed step | δt = 0.0005 s | -97.66 | 21 | ok |
| ICF fixed step | δt = 0.0002 s | -57.09 | 13 | ok |
| ICF fixed step | δt = 0.0001 s | -29.98 | 12 | ok |
| ICF fixed step | δt = 5e-05 s | -16.28 | 11 | ok |
| ICF fixed step | δt = 2e-05 s | -6.25 | 11 | ok |
| ICF fixed step | δt = 1e-05 s | -3.17 | 11 | ok |
| MuJoCo fixed step | δt = 0.01 s | -7.08 | 10 | ok |
| MuJoCo fixed step | δt = 0.005 s | -7.33 | 10 | ok |
| MuJoCo fixed step | δt = 0.002 s | +0.78 | 10 | ok |
| MuJoCo fixed step | δt = 0.001 s | -0.02 | 10 | ok |
| MuJoCo fixed step | δt = 0.0005 s | -0.00 | 10 | ok |
| MuJoCo fixed step | δt = 0.0002 s | -0.00 | 10 | ok |
| MuJoCo fixed step | δt = 0.0001 s | -0.00 | 10 | ok |
| MuJoCo fixed step | δt = 5e-05 s | +0.00 | 10 | ok |
| MuJoCo fixed step | δt = 2e-05 s | +0.00 | 10 | ok |
| MuJoCo fixed step | δt = 1e-05 s | -0.00 | 10 | ok |
| ICF error control | ε = 1e-01 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-02 | -100.10 | 3 | ok |
| ICF error control | ε = 1e-03 | -100.10 | 5 | ok |
| ICF error control | ε = 1e-04 | -100.10 | 14 | ok |
| ICF error control | ε = 1e-05 | -51.01 | 11 | budget-exhausted |
| ICF error control | ε = 1e-06 | -91.70 | 0 | budget-exhausted |
| MuJoCo error control | ε = 1e-01 | +0.77 | 9 | ok |
| MuJoCo error control | ε = 1e-02 | +0.77 | 9 | ok |
| MuJoCo error control | ε = 1e-03 | +57.27 | 9 | ok |
| MuJoCo error control | ε = 1e-04 | +22.34 | 10 | ok |
| MuJoCo error control | ε = 1e-05 | +12.71 | 10 | ok |
| MuJoCo error control | ε = 1e-06 | +3.97 | 9 | ok |
