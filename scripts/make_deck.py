"""Generate a self-contained HTML slide deck (inline CSS, base64-embedded figures) for
the CENIC-in-Newton status report. Audience: a CENIC author. Framing: a measurement
post-mortem + implementation status, NOT a tutorial on the method.

    uv run python scripts/make_deck.py   ->   results/cenic_deck.html
"""

from __future__ import annotations

import base64
import html
import os

PLOTS = "results/plots"


def img(name: str) -> str:
    path = os.path.join(PLOTS, name)
    if not os.path.exists(path):
        return '<div class="missing">[missing figure: %s]</div>' % html.escape(name)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{html.escape(name)}"/>'


# (title, body_html, figure_or_None, speaker_notes)
SLIDES = [
    (
        "CENIC in Newton: a measurement post-mortem",
        "<p class='lead'>We integrated CENIC into Newton to generate higher-fidelity data for "
        "sim-to-real RL. This is the status of that work: a surprising null in our integrator "
        "evaluation, what it actually was, and where we stand.</p>"
        "<p class='sub'>Not a recap of the method &mdash; a report on our use of it.</p>",
        None,
        "Set expectations: this is about our evaluation and implementation, not the CENIC method.",
    ),
    (
        "The question, and a surprising null",
        "<p>We asked: does the integrator's accuracy measurably help RL sim-to-real transfer? "
        "Our first measurement ranked eight error sources by their transfer-gap contribution. "
        "<b>The integrator (red) came out last &mdash; slightly negative.</b> Sensor noise dominated.</p>"
        "<p class='sub'>Naive read: 'integration doesn't matter.' We did not accept it &mdash; here's why.</p>",
        "error_budget.png",
        "This is the result that kicked off the investigation. Don't oversell it; it's the thing we then explain away.",
    ),
    (
        "Why the null was a measurement artifact",
        "<p>The metric was a trained policy's velocity-tracking error. A stiff feedback controller "
        "(K<sub>p</sub>=150) rejects dynamics deviations every control tick, so it <b>regulates away the "
        "integration error before the metric reads it</b>. We measured the controller, not the integrator.</p>"
        "<p>Tell: the only channel that survived &mdash; sensor noise &mdash; is the one that corrupts the "
        "controller's <i>input</i>, which feedback cannot reject. Every plant-side perturbation "
        "(integrator, friction, mass) was masked. <b>Lesson: you cannot evaluate integrator accuracy "
        "through a closed-loop policy's tracking error.</b></p>",
        None,
        "This is the genuinely useful point for the audience: a methodological gotcha in evaluating integrators via RL policies.",
    ),
    (
        "Open-loop: the implementation behaves as expected",
        "<p>Open-loop work-precision on a single stiff impact (no policy, contact already engaged so "
        "collision cadence is not a factor). Left: the adaptive solver reaches ~1.3&nbsp;mm accuracy in ~228 solver "
        "steps where fixed-step needs ~800 &mdash; <b>~3.5x less compute at equal accuracy</b>. Right: the "
        "inner dt collapses at the impact and relaxes back. The port reproduces the expected behavior.</p>",
        "v1_single_drop.png",
        "Validation, not a discovery. Keep it brief for this audience. Note it is explicit euler (caveat on the next slide).",
    ),
    (
        "Spec note + a question for you",
        "<p>Aligning our implementation with &sect;V-E: our error scaling <code>S</code> had drifted to a "
        "mass-weighted, clipped form (<code>diag(M)<sup>-1/2</sup></code>, normalized, clip[1,10]) that the "
        "paper does not specify. We restored <code>S = identity</code>. (A direct ablation confirmed this "
        "is housekeeping &mdash; it does not change the unit-test result.)</p>"
        "<p class='q'>Question: for our floating-base + revolute robots, would you recommend a "
        "coordinate-type <code>S</code> (e.g. length-scaling the translational coordinates) over identity?</p>",
        None,
        "Engage the author as the expert on S. Be transparent that the deviation was ours and is fixed.",
    ),
    (
        "Status &amp; next",
        "<ul>"
        "<li>The integrator-accuracy-for-transfer question is <b>still open</b> &mdash; we were measuring it wrong.</li>"
        "<li>Re-running the transfer study <b>open-loop</b> (frozen control, no feedback) with the corrected solver.</li>"
        "<li>Quantifying explicit euler vs the implicitfast default (A-stable &rarr; smaller dt sensitivity).</li>"
        "<li>Implementation validated on the open-loop unit test; <code>S</code> aligned to &sect;V-E.</li>"
        "</ul>",
        None,
        "Honest close. The headline is the methodological correction, not an 'adaptive solver works' claim.",
    ),
]


def build() -> str:
    css = """
    :root{--fg:#16202c;--mut:#5b6b7b;--ac:#1f6feb;--bg:#fff;--card:#f6f8fa;}
    *{box-sizing:border-box}
    body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--fg);background:#e9edf1}
    .slide{max-width:1000px;margin:26px auto;background:var(--bg);border-radius:14px;
      box-shadow:0 2px 16px rgba(20,32,44,.10);padding:42px 52px;min-height:560px;display:flex;flex-direction:column}
    .num{color:var(--mut);font-size:13px;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px}
    h1{font-size:30px;line-height:1.18;margin:0 0 18px}
    .slide.title h1{font-size:38px;margin-top:40px}
    p,li{font-size:19px;line-height:1.5;margin:0 0 14px}
    .lead{font-size:22px}.sub{color:var(--mut)}
    .q{background:var(--card);border-left:4px solid var(--ac);padding:14px 18px;border-radius:6px;font-size:20px}
    code{background:var(--card);padding:1px 6px;border-radius:5px;font-size:.92em}
    img{max-width:100%;max-height:430px;display:block;margin:8px auto 0;border:1px solid #e3e8ee;border-radius:8px}
    .figwrap{flex:1;display:flex;align-items:center;justify-content:center;margin-top:6px}
    .notes{margin-top:18px;color:var(--mut);font-size:13px;border-top:1px dashed #d6dde4;padding-top:10px}
    .notes b{color:var(--fg)}
    .missing{color:#b00;padding:40px;text-align:center}
    @media print{body{background:#fff}.slide{box-shadow:none;page-break-after:always;margin:0;border-radius:0;min-height:96vh}}
    """
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'><title>CENIC in Newton</title><style>{css}</style></head><body>"
    ]
    n = len(SLIDES)
    for i, (title, body, fig, notes) in enumerate(SLIDES, 1):
        cls = "slide title" if i == 1 else "slide"
        figwrap = f"<div class='figwrap'>{img(fig)}</div>" if fig else ""
        parts.append(
            f"<section class='{cls}'><div class='num'>{i} / {n}</div><h1>{title}</h1>{body}{figwrap}"
            f"<div class='notes'><b>Say:</b> {html.escape(notes)}</div></section>"
        )
    parts.append("</body></html>")
    return "".join(parts)


def main():
    os.makedirs("results", exist_ok=True)
    out = "results/cenic_deck.html"
    with open(out, "w") as f:
        f.write(build())
    kb = os.path.getsize(out) / 1024
    print(f"wrote {out} ({kb:.0f} KB, {len(SLIDES)} slides)")


if __name__ == "__main__":
    main()
