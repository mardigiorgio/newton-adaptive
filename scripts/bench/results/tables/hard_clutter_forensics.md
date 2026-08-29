# Hard-clutter fixed-ICF artifacts at 10 ms: forensics and the two authoring faults (2026-08-29)

Trigger: fixed ICF at δt = 10 ms read max ground penetration 27.8 mm and 1.6 % ejections
on hard clutter while MuJoCo read 6.8 mm and 0 %. Per-body forensics (64 worlds, seed 42,
0.2 s warm-up + 2 s): every deep ICF penetration was a CUBE, nearly at rest (|v| ≈ 0.01 m/s),
sunk to its centre (z ≈ −2 mm), later inside a bin wall. Its contact dump showed four
BOX–BOX contacts with a static wall, normal ≈ +z, d = −45.7 mm: the wall's bottom face
(coincident with the floor, z = 0) had become the box–box SAT reference face for a rotated
cube driven into the wall, wedging the cube between that face and the floor. The ICF solve
converged in every world (≤ 24 Newton iterations); nconmax/ICF_MAX_RIGID_CONTACT were not
the cause (peak 342/1024 and 360/2048 per world; 4096 and 65536 change nothing); the
isolated cube (flat, tilted, loaded) rests at 0.00 mm at 10 ms.

## Fault 1 — bin walls with their bottom face on the floor (fixed: walls span z ∈ [−h, h])

| ICF fixed 10 ms, hard clutter, 64 worlds | max pen | body-boundaries > 2.3 mm | ejected | mean pen (end) |
|---|---|---|---|---|
| walls on the floor (old) | 27.8 mm | 312 | 1.6 % | 230 µm |
| walls through the floor (new) | 7.8 mm | 78 | 0.6 % | 16 µm |
| MuJoCo fixed 10 ms (unchanged) | 6.8 mm | 259 | 0 % | 247 µm |

## Fault 2 — an assumed Hunt & Crossley dissipation (icf_warp default d = 10 s/m)

(1 − d v_n)_+ removes the force once separation exceeds 1/d = 0.1 m/s; at a coarse step the
lagged force cannot expel a body a pile drives into a wall, and it goes through (all
"ejections" were beyond the outer wall face, none embedded).

| ICF fixed 10 ms, new walls | max pen | body-boundaries > 2.3 mm | beyond the outer wall face | mean pen (end) |
|---|---|---|---|---|
| d = 10 s/m | 7.5 mm | 62 | 1.56 % | 11.8 µm |
| d = 1 s/m (adopted; stated as assumed) | 4.4 mm | 2 | 0 | 7.3 µm |
| d = 0 | 1.27 mm | 0 | 0 | 5.5 µm |

Wall thickness (40 vs 100 mm) does not change the d = 10 passthrough. The paper states zero
dissipation only for the ball; the clutters' d is our assumption — confirm with the authors.
Every clutter number in PART1.md dated before this table is superseded by the rerun.
