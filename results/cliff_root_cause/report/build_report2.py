"""Rebuild the meeting-brief artifact, v8: per defect, labeled beats
(Trigger/Mistake/Damage/Cost) -> diagram -> Fix -> Proof; airy spacing;
notation guide; dt names; no em dashes; no br in mermaid.
Args: <steady_mean> <steady_best> <steady_worst> <n_iters>
"""

import sys

SRC = "/home/mdigiorgio/.claude/projects/-home-mdigiorgio-Documents-code/843c73ab-c03d-45ba-8af0-818bb827bb7d/tool-results/artifact-adc9468a-1786420927-436b.html"
OUT = "/tmp/claude-1002/-home-mdigiorgio-Documents-code/843c73ab-c03d-45ba-8af0-818bb827bb7d/scratchpad/report_updated.html"

mean, best, worst, niters = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

html = open(SRC, encoding="utf-8").read()
content = html[html.find("<body>") + len("<body>"):].replace("</body></html>", "").rstrip()


def replace_once(hay, old, new):
    assert hay.count(old) == 1, f"anchor bad: {old[:70]!r} count={hay.count(old)}"
    return hay.replace(old, new)


content = replace_once(
    content,
    '<div class="stat"><b>13.3 → 55 s</b><span>solver iteration: 13.3 s fresh-process (−24%, verified) but degrades ~4× after ~90 iters in long runs — open bug; segmented-restart workaround in production</span></div>',
    f'<div class="stat"><b>{mean} s steady</b><span>solver iteration at 2048 envs, flat through {niters} iterations (best {best}, worst {worst}). The old ~90-iteration jump to 55-65 s: root-caused, 3 solver defects, all fixed.</span></div>',
)

content = replace_once(
    content,
    "definitive pair in flight</p>",
    "definitive pair in flight · <b>process-age cliff root-caused and fixed</b></p>",
)


def delete_figure(hay, caption_snippet):
    cap_at = hay.find(caption_snippet)
    assert cap_at != -1, f"caption not found: {caption_snippet[:60]!r}"
    p_start = hay.rfind('<p class="cap">', 0, cap_at)
    fig_start = hay.rfind('<div class="fig">', 0, p_start)
    p_end = hay.find("</p>", cap_at) + len("</p>")
    assert 0 < fig_start < p_start < p_end
    return hay[:fig_start] + hay[p_end:]


content = delete_figure(content, "fixed arms die at ×; the adaptive run continues")
content = delete_figure(content, "Possession reward for the running adaptive arm")
content = delete_figure(content, "Right: the open item — same state costs 16 s in a fresh process")
content = content.replace("✝ ", "")

NEW = f"""
  <style>
    .mermaid {{ display:flex; justify-content:center; padding:16px 8px; }}
    .bug {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:20px 22px; margin:22px 0; }}
    .bug h3 {{ margin:0 0 14px; font-size:17px; }}
    .beats {{ display:grid; grid-template-columns:auto 1fr; column-gap:14px; row-gap:8px; margin:0 0 8px; font-size:15px; line-height:1.5; }}
    .beats b {{ color:var(--ink2); font-weight:700; }}
    .fixrow {{ margin-top:12px; font-size:15px; }}
    .fixrow b {{ color:var(--blueink); }}
    .proofrow {{ margin-top:6px; font-size:14px; color:var(--ink3); }}
    .notation {{ background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 18px; margin:14px 0 22px; }}
    .notation td:first-child {{ font-weight:700; white-space:nowrap; font-variant-numeric:normal; }}
  </style>

  <h2>The live-process degradation: three defects, three fixes</h2>

  <p class="cap">One training iteration = 48 frames of 10 ms. Inside a frame, every world advances by adaptive sub-steps until all 2048 reach the mark. Notation (the solver's own):</p>

  <div class="notation">
    <table>
      <tr><td>frame</td><td>10 ms of simulated time; hard cap 256 attempts</td></tr>
      <tr><td>dt</td><td>a world's current sub-step size</td></tr>
      <tr><td>carried dt</td><td>the dt a world remembers between frames (code: ideal_dt)</td></tr>
      <tr><td>e</td><td>error of an attempt: one step of dt vs two of dt/2, largest position disagreement</td></tr>
      <tr><td>tol</td><td>1e-3; accept iff e ≤ tol, else roll back</td></tr>
      <tr><td>resize rule</td><td>dt_next = 0.9 · dt · √(tol / e); grow ≤ 5x; floor dt_min = 1 µs</td></tr>
      <tr><td>deadband</td><td>dt_next within 0.9 to 1.2 of dt: keep dt (prevents churn)</td></tr>
      <tr><td>ulp</td><td>one float32 bit at a number's magnitude; smaller differences don't exist</td></tr>
    </table>
  </div>

  <div class="bug">
    <h3>Defect 1 · a frame-edge sliver poisons the carried dt</h3>
    <div class="beats">
      <b>Trigger</b><span>the frame's last step is clamped to the leftover time, sometimes 2 ns</span>
      <b>Mistake</b><span>the sliver accepts, and the resize rule treats 2 ns as a real dt</span>
      <b>Damage</b><span>carried dt overwritten to 5 × 2 ns = 1e-8 s; next frame restarts at dt_min</span>
      <b>Cost</b><span>6 to 8 extra attempts per event; caught 41 times in telemetry</span>
    </div>
    <div class="fig"><pre class="mermaid">
flowchart LR
  A["last step clamped to 2 ns"] --> B["accepts: e ≤ tol"]
  B -- before --> C["resize writes carried dt = 1e-8 s"] --> D["next frame climbs from dt_min"]
  B -- "after fix" --> E["clamped accepts skip the write"] --> F["carried dt keeps its value"]
    </pre></div>
    <div class="fixrow"><b>Fix</b> · a step shortened by the calendar says nothing about the physics: frame-clamped accepts never write carried dt.</div>
    <div class="proofrow"><b>Proof</b> · sub-dt_min writes 41 → 0; onset moved from iteration 86 to 130 (a channel, not the root).</div>
  </div>

  <div class="bug">
    <h3>Defect 2 · time debt from a truncated frame compounds forever</h3>
    <div class="beats">
      <b>Trigger</b><span>a world misses the 10 ms mark within the 256-attempt cap</span>
      <b>Mistake</b><span>the shortfall is added to the next frame's target</span>
      <b>Damage</b><span>11 ms fits in 256 attempts even less; debt grows every frame, no exit</span>
      <b>Cost</b><span>attempts pinned at 256; debt measured 8.3 → 706 ms, monotone</span>
    </div>
    <div class="fig"><pre class="mermaid">
flowchart LR
  A["frame ends short"] --> B["shortfall carried forward"]
  B -- before --> C["target = 10 ms + debt, cap hit again"] --> A
  B -- "after fix" --> D["debt capped at one frame, controller reset"] --> E["climbs out next frame"]
    </pre></div>
    <div class="fixrow"><b>Fix</b> · carried debt clamped to one frame; the world's controller memory resets so recovery runs at full 5x growth.</div>
    <div class="proofrow"><b>Proof</b> · residual bounded at 9.5 ms with the guard; unbounded in every control run.</div>
  </div>

  <div class="bug">
    <h3>Defect 3 · root cause: e reads the float grid, dt freezes</h3>
    <div class="beats">
      <b>Trigger</b><span>any coordinate near magnitude 8192, where one ulp = 0.00098 = 0.977 · tol</span>
      <b>Mistake</b><span>e cannot read below one ulp, so e ≈ tol at every dt: the grid, not the physics</span>
      <b>Damage</b><span>resize returns 0.911 · dt, the deadband holds it, dt freezes tiny</span>
      <b>Cost</b><span>~2000 attempts needed vs 256 allowed; one frozen world stalls all 2048</span>
      <b>Why restarts "cured" it</b><span>fresh processes have no frozen worlds: 13-17 s young, 55-65 s old</span>
      <b>Smoking gun</b><span>e's per-frame max bit-exact at 2⁻¹⁰ for 181 straight frames</span>
    </div>
    <div class="fig"><pre class="mermaid">
flowchart LR
  A["ulp of a coordinate = 0.977 · tol"] --> B["e reads 0.977 · tol at every dt"]
  B -- before --> C["deadband holds dt forever"] --> D["frozen world pins the fleet"]
  B -- "after fix" --> E["sub-ulp differences count as zero"] --> F["e reads real error, dt grows out"]
    </pre></div>
    <div class="fixrow"><b>Fix</b> · a per-coordinate difference within 4 ulps of that coordinate's magnitude is representation noise: it counts as zero.</div>
    <div class="proofrow"><b>Proof</b> · the same scene translated to x = 8192 (physics unchanged, ulp = 0.977 · tol by construction) freezes in 90 seconds: 147 attempts/frame, 24 with the fix, 24-25 at the origin either way.</div>
  </div>

  <h2>The cure, measured</h2>
  <table>
    <tr><th>run (all seed 42, Trossen lift)</th><th>iter 86 (historical cliff)</th><th>steady state</th><th>outcome</th></tr>
    <tr><td>2048 envs, graphs off, control ×2</td><td class="dead">143.9 / 145.9 s</td><td>21-24 s, then 159-163 s</td><td class="dead">cliff</td></tr>
    <tr><td>256-env assay, control</td><td class="dead">117.5 s</td><td>9-10 s, then pinned at cap</td><td class="dead">cliff</td></tr>
    <tr><td>256-env assay, all 3 fixes</td><td class="live">9.8 s</td><td>9.6 s mean over 150 iters</td><td class="live">flat</td></tr>
    <tr><td><b>2048 envs, graphs on, production cfg + fixes</b></td><td class="live"><b>15.2 s</b></td><td><b>{mean} s mean (best {best}, worst {worst}), {niters} iters</b></td><td class="live"><b>flat</b></td></tr>
  </table>
  <p class="cap">Adaptation intact under the fixes: ~18 attempts/frame, ~1,600 fleet-wide rejections/frame, accepted e ≤ tol on 6,089 of 6,089 deep-contact frames, zero guard activations at full scale.</p>
"""

NEW = NEW.replace("<br/>", " ")

start = content.find("  <h2>Corrections ledger — retracted intermediate claims</h2>")
end_marker = "earlier-era diagnostics are excluded by policy.</p>"
end = content.find(end_marker)
assert start != -1 and end != -1 and end > start
content = content[:start] + NEW + "\n" + content[end + len(end_marker):]

content = replace_once(
    content,
    '<td class="live">training now</td><td>error-controlled per-world steps; full integrity</td>',
    '<td class="live">completed 300 iters</td><td>error-controlled per-world steps; full integrity</td>',
)

content = replace_once(
    content,
    "Open: definitive pair completing now · live-process degradation root cause · lift-and-carry demonstration · single seed · ~45 commits unpushed (GitHub auth).",
    "Open: lift-and-carry demonstration · single seed · defect-probe gating of the three cliff fixes · nconmax headroom · ~48 commits unpushed (GitHub auth).",
)

open(OUT, "w", encoding="utf-8").write(content)
print(f"written {OUT} ({len(content)} chars); v8 beats layout")
