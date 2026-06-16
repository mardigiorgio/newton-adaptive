# CENIC in Newton — presentation (speaker notes)

Deck: `results/cenic_deck.html` (self-contained; open in a browser, or Print -> Save as PDF).
Regenerate: `uv run python scripts/make_deck.py`.

**Audience = a CENIC author.** Framing is a measurement post-mortem + implementation status,
NOT a tutorial on the method. Do not explain step-doubling, why coarse dt tunnels, or the §V-E
metric back to its author. The one load-bearing finding is the closed-loop measurement artifact;
the work-precision win is validation that our port behaves as expected; the S note is housekeeping.

## Slide flow
1. **Scope** — our CENIC-in-Newton work for sim-to-real RL data; this is a post-mortem + status.
2. **The surprising null** [error-budget figure] — integrator ranked last of 8 transfer factors.
3. **Why it was an artifact** — the metric was a closed-loop policy's tracking error; a stiff Kp=150
   controller rejects dynamics deviations, so it masks integration error. Only sensor noise (the
   controller's input) survived. LESSON: don't evaluate integrator accuracy through a closed-loop
   policy metric. (This is the genuinely useful slide for the audience.)
4. **Open-loop, the port is sound** [work-precision figure] — stiff impact, ~3.5x less compute at
   equal accuracy, dt collapses at impact. Validation, brief. Caveat: explicit euler.
5. **Spec note + question** — our S had drifted from §V-E (mass-weighted, clipped); restored to
   identity; ablation confirms it's housekeeping. Ask the author: coordinate-type S vs identity for
   floating-base + revolute robots?
6. **Status & next** — re-run transfer open-loop with the corrected solver; quantify euler vs
   implicitfast. The integrator-for-transfer question is open and now being measured correctly.
7. **Backup** [penetration trace] — contact-resolution intuition; only if asked.

## 30-second version
"We tried to measure whether the integrator's accuracy helps RL sim-to-real transfer and got a null.
The null was a measurement artifact: the metric was a trained policy's tracking error, and a stiff
feedback controller masks integration error before you can read it. Open-loop, our CENIC port shows
the expected ~3.5x work-precision advantage. We also realigned our error scaling S to §V-E. Next: the
transfer study, re-run open-loop."

## Q&A prep
- Paper: Kurtz & Castro, arXiv:2511.08771. §V-E metric: e=||S(q-q̂)||_inf, position-only, no clipping.
- The closed-loop-masking claim: obs-noise ranked top precisely because it perturbs the policy input,
  not the plant; all plant-side perturbations (integrator/friction/mass) were rejected by feedback.
- Honest caveats to volunteer: explicit euler (implicitfast gap is smaller); single-impact is a clean
  unit test, not the full robot; the S deviation was ours and is fixed (one file, net deletion, nothing
  committed); the work-precision case uses contact-already-engaged so it is not a collision-cadence effect.
- Do NOT claim the S deviation caused the null — a direct ablation refutes that.
