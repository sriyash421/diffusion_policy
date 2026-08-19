"""Render the 30/100-demo sweep's success rates as a standalone HTML page.

Companion to build_30_100_success_doc.py: same data, same refusal to nominate a best
checkpoint, but laid out as six heatmap matrices arranged in the experiment's own 3x2 shape
(policy family down, demo budget across) so the comparison the sweep exists to make is the
thing the page shows.

    python scripts/build_30_100_success_artifact.py -o out.html
"""
import argparse
import json
import os
import pathlib
import re

ROOT = pathlib.Path(os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
BASE = ROOT / 'pusht_search' / 'pusht_image_search'
NS = [1, 2, 4, 8, 16, 32, 64]
STEPS = [10000 * k for k in range(1, 11)]

FAMILIES = [
    ('ST-diffusion k=16 <span class="arch">4/4/256 · 17.1M</span>', 'Search transformer, diffusion head. Candidate k is conditioned on '
                     'candidates 0..k-1 and their verifier values.',
     {30: BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-30_seed-42',
      100: BASE / 'outer_inner' / 'value_k16_corrupt-False_demos-100_seed-42'}),
    ('ST-gaussian k=16 <span class="arch">4/4/256 · 14.6M</span>', 'Same search procedure, Gaussian head: a candidate is one rsample '
                    'rather than a denoising loop.',
     {30: BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-30_seed-42',
      100: BASE / 'offline' / 'gaussian_k16_corrupt-False_demos-100_seed-42'}),
    ('ST-diffusion k=1 <span class="arch">4/4/256 · 17.1M</span>', 'The same policy class as k=16, trained at width 1 — the search '
              'context is always empty, so n>1 is best-of-n over i.i.d. samples at a '
              'matched compute budget.',
     {30: BASE / 'offline' / 'bc_demos-30_seed-42',
      100: BASE / 'offline' / 'bc_demos-100_seed-42'}),
]

# 30 demos only, so they do not belong in the grid above (whose second column IS the
# 100-demo budget). Same manifest, seed, protocol and n grid, so the numbers compare
# directly with the 30-demo column.
EXTRA = [
    ('UNet BC <span class="arch">293.4M</span>', 'A different architecture: the diffusion UNet, not a transformer. No '
                'search context at all, so every n is i.i.d. best-of-n.',
     BASE / 'unet_bc' / 'unetbc_demos-30_seed-42'),
    ('ST-diffusion k=1 <span class="arch">6/8/1024 · 137.8M</span>', 'The same search transformer widened to a 126.6M trunk against the '
                   '5.9M above, trained at width 1.',
     BASE / 'offline' / 'value_k1_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
    ('ST-diffusion k=16 <span class="arch">6/8/1024 · 137.8M</span>', 'The same wide trunk trained at width 16. Against k=1 it isolates '
                    'search from capacity.',
     BASE / 'outer_inner' / 'value_k16_arch-6x8x1024_corrupt-False_demos-30_seed-42'),
]

# Sequential ramp, one hue light->dark (dataviz: magnitude is never categorical). Light mode
# reads low->high as surface->deep; dark mode gets its OWN steps against the dark ground
# rather than an inversion, so "near zero recedes" holds in both.
# The steps skip the luminance band where NEITHER ink clears 4.5:1 against the fill --
# a plain even ramp puts a step right on that crossover and its numbers go unreadable.
# Verified: every step >= 4.5:1 with the ink its tier selects, and both ramps stay
# monotonic in luminance so magnitude still reads as darkness.
RAMP_LIGHT = ['#eef4fd', '#dcebfc', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef',
              '#6da7ec', '#5598e7', '#256abf', '#1c5cab', '#184f95', '#104281']
RAMP_DARK = ['#171a20', '#191d24', '#1b2735', '#1d3149', '#1f3c5e', '#214873',
             '#245489', '#4184c9', '#5598e7', '#6da7ec', '#86b6ef', '#a8c9f2']
# index at which the cell switches to the contrasting ink, per mode
CUT_LIGHT, CUT_DARK = 8, 7


def read_rows(run):
    p = run / 'bon_search' / 'success_curves.jsonl'
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def grid(rows, field='success_rate'):
    agg = {}
    for r in rows:
        m = re.search(r'step_(\d+)', r.get('checkpoint', ''))
        step = int(m.group(1)) if m else r.get('step')
        if step is None:
            continue
        agg.setdefault(int(step), {}).update(
            dict(zip(r.get('n') or [], r.get(field) or [])))
    return agg


def n_ckpt(run):
    d = run / 'checkpoints'
    return len(list(d.glob('step_*.ckpt'))) if d.is_dir() else 0


def cell(v):
    if v is None:
        return '<td class="c empty" aria-label="not evaluated">·</td>'
    i = min(len(RAMP_LIGHT) - 1, max(0, int(round(v * (len(RAMP_LIGHT) - 1)))))
    # value text stays an ink token; the tier class only drives which ink (dataviz:
    # text never wears the series color)
    tier = 'hi' if i >= CUT_LIGHT else 'lo'
    return (f'<td class="c s{i} {tier}" title="success rate {v:.2f}">'
            f'{v:.2f}</td>')


def _plain(label):
    """Labels carry an inline <span> for the arch chip; the a11y caption wants text."""
    return re.sub(r'<[^>]+>', '', label)


def matrix(run, label):
    rows = read_rows(run)
    g = grid(rows)
    if not g:
        return ('<div class="mx-empty"><p>No checkpoint evaluated yet.</p>'
                f'<p class="mono tiny">{label}</p></div>')
    head = ''.join(f'<th scope="col">{n}</th>' for n in NS)
    body = []
    for s in STEPS:
        if s not in g:
            continue
        tds = ''.join(cell(g[s].get(n)) for n in NS)
        body.append(f'<tr><th scope="row">{s // 1000}k</th>{tds}</tr>')
    return (f'<table class="mx"><caption class="vh">Success rate by checkpoint and search '
            f'width for {_plain(label)}</caption><thead><tr><th scope="col" class="corner">'
            f'<span class="vh">gradient step</span></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def peak(run):
    g = grid(read_rows(run))
    vals = [v for row in g.values() for v in row.values() if v is not None]
    return max(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--generated', default='', help='timestamp string for the header')
    args = ap.parse_args()

    cards = []
    for fam, blurb, runs in FAMILIES:
        cols = []
        for demos in (30, 100):
            run = runs[demos]
            ck = n_ckpt(run)
            done = len([s for s, v in grid(read_rows(run)).items()
                        if set(NS) <= set(v)])
            pk = peak(run)
            cols.append(
                '<div class="cell">'
                f'<div class="cell-hd"><span class="demos">{demos}<span class="unit">'
                f' demos</span></span>'
                f'<span class="prog mono">{done}<span class="sep">/</span>10 swept</span>'
                '</div>'
                f'{matrix(run, f"{fam} at {demos} demos")}'
                f'<div class="cell-ft mono tiny">{ck}/10 checkpoints written'
                + (f' · highest cell {pk:.2f}' if pk is not None else '')
                + '</div></div>')
        cards.append(
            '<section class="arm">'
            f'<div class="arm-hd"><h2>{fam}</h2><p>{blurb}</p></div>'
            f'<div class="arm-grid">{"".join(cols)}</div>'
            '</section>')

    extra_cells = []
    for label, blurb, run in EXTRA:
        ck = n_ckpt(run)
        done = len([s for s, v in grid(read_rows(run)).items() if set(NS) <= set(v)])
        pk = peak(run)
        extra_cells.append(
            '<div class="cell">'
            f'<div class="cell-hd"><span class="demos">{label}</span>'
            f'<span class="prog mono">{done}<span class="sep">/</span>10 swept</span></div>'
            f'<p class="blurb">{blurb}</p>'
            f'{matrix(run, label)}'
            f'<div class="cell-ft mono tiny">{ck}/10 checkpoints written'
            + (f' · highest cell {pk:.2f}' if pk is not None else '') + '</div></div>')
    cards.append(
        '<section class="arm">'
        '<div class="arm-hd"><h2>30-demo additions</h2><p>A different architecture and a '
        'wider trunk, run only at 30 demos. Same manifest, seed, protocol and n grid as '
        'the 30-demo column above, so these compare directly with it.</p></div>'
        f'<div class="arm-grid arm-grid-3">{"".join(extra_cells)}</div>'
        '</section>')

    legend = ''.join(f'<i class="s{i}"></i>' for i in range(len(RAMP_LIGHT)))
    stamp = args.generated or 'sweep in progress'

    css_ramp_light = '\n'.join(
        f'.s{i}{{--cell:{c};}}' for i, c in enumerate(RAMP_LIGHT))
    css_ramp_dark_media = '\n'.join(
        f'  :root:not([data-theme="light"]) .s{i}{{--cell:{c};}}'
        for i, c in enumerate(RAMP_DARK))
    css_ramp_dark_attr = '\n'.join(
        f':root[data-theme="dark"] .s{i}{{--cell:{c};}}'
        for i, c in enumerate(RAMP_DARK))

    html = f"""<title>PushT Search Sweep</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#fcfcfb; --panel:#ffffff; --ink:#14161a; --ink-2:#54585f; --ink-3:#878c93;
  --rule:#e6e6e3; --rule-2:#f0f0ed; --accent:#256abf; --warn:#8a5a12;
  --cell-ink:#14161a; --cell-ink-hi:#ffffff;
}}
{css_ramp_light}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground:#141618; --panel:#191b1e; --ink:#f2f3f4; --ink-2:#b3b8bf; --ink-3:#7b818a;
    --rule:#2a2e33; --rule-2:#212529; --accent:#6da7ec; --warn:#d8a24a;
    --cell-ink:#c8d4e2; --cell-ink-hi:#0b0f14;
  }}
{css_ramp_dark_media}
  /* dark crosses to the contrasting ink one step earlier than light */
  :root:not([data-theme="light"]) td.c.s7{{color:var(--cell-ink-hi);}}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground:#141618; --panel:#191b1e; --ink:#f2f3f4; --ink-2:#b3b8bf; --ink-3:#7b818a;
  --rule:#2a2e33; --rule-2:#212529; --accent:#6da7ec; --warn:#d8a24a;
  --cell-ink:#c8d4e2; --cell-ink-hi:#0b0f14;
}}
{css_ramp_dark_attr}
:root[data-theme="dark"] td.c.s7{{color:var(--cell-ink-hi);}}
*{{box-sizing:border-box;}}
body{{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:1120px; margin:0 auto; padding:56px 28px 88px;
  display:flex; flex-direction:column; gap:44px;}}
.mono{{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums;}}
.tiny{{font-size:12px;}}
.vh{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
  clip-path:inset(50%);white-space:nowrap;}}

header h1{{
  font-family:Newsreader,Georgia,"Times New Roman",serif;
  font-weight:400; font-size:clamp(34px,5vw,52px); line-height:1.06;
  letter-spacing:-0.015em; margin:0 0 14px; text-wrap:balance;
}}
header h1 em{{font-style:italic; color:var(--accent);}}
.lede{{max-width:64ch; color:var(--ink-2); margin:0; font-size:16.5px;}}
.eyebrow{{
  font-size:11.5px; letter-spacing:0.14em; text-transform:uppercase;
  color:var(--ink-3); margin:0 0 16px;
}}

.legend{{display:flex; align-items:center; gap:14px; flex-wrap:wrap;
  padding:14px 16px; border:1px solid var(--rule); border-radius:3px;
  background:var(--panel);}}
.legend .ramp{{display:flex; gap:2px;}}
.legend i{{width:26px; height:11px; background:var(--cell); display:block;}}
.legend .lab{{font-size:12px; color:var(--ink-3);}}

.arm{{display:flex; flex-direction:column; gap:18px;}}
.arm-hd{{border-top:2px solid var(--ink); padding-top:14px; max-width:74ch;}}
.arm-hd h2{{
  font-family:Newsreader,Georgia,serif; font-weight:500; font-size:26px;
  margin:0 0 6px; letter-spacing:-0.01em;
}}
.arm-hd p{{margin:0; color:var(--ink-2); font-size:14.5px;}}
.arm-grid{{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px;}}
@media (max-width:760px){{.arm-grid{{grid-template-columns:1fr;}}}}

.arm-grid-3{{grid-template-columns:repeat(3,minmax(0,1fr));}}
@media (max-width:1000px){{.arm-grid-3{{grid-template-columns:1fr;}}}}
.arch{{display:inline-block; font-family:"IBM Plex Mono",monospace; font-size:11.5px;
  font-weight:400; color:var(--ink-3); letter-spacing:0.01em; margin-left:7px;
  white-space:nowrap;}}
.blurb{{margin:0; font-size:12.5px; color:var(--ink-3); line-height:1.45;}}
.cell{{border:1px solid var(--rule); border-radius:3px; background:var(--panel);
  padding:14px 14px 12px; display:flex; flex-direction:column; gap:10px;
  overflow-x:auto;}}
.cell-hd{{display:flex; align-items:baseline; justify-content:space-between; gap:10px;}}
.demos{{font-size:19px; font-weight:600; letter-spacing:-0.01em;}}
.demos .unit{{font-size:13px; font-weight:400; color:var(--ink-3);}}
.prog{{font-size:12px; color:var(--ink-3);}}
.prog .sep{{opacity:.5;}}
.cell-ft{{color:var(--ink-3); border-top:1px solid var(--rule-2); padding-top:8px;}}
.mx-empty{{color:var(--ink-3); font-size:14px;}}
.mx-empty p{{margin:0 0 4px;}}

table.mx{{border-collapse:separate; border-spacing:2px; width:100%;}}
table.mx th{{
  font-family:"IBM Plex Mono",monospace; font-weight:500; font-size:11.5px;
  color:var(--ink-3); font-variant-numeric:tabular-nums; padding:2px 4px;
}}
table.mx thead th{{text-align:center;}}
table.mx tbody th{{text-align:right; white-space:nowrap;}}
td.c{{
  background:var(--cell); color:var(--cell-ink);
  font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums;
  font-size:12.5px; text-align:center; padding:6px 4px; border-radius:2px;
  min-width:44px;
}}
td.c.hi{{color:var(--cell-ink-hi);}}
td.c.empty{{background:transparent; color:var(--ink-3); border:1px dashed var(--rule);}}

.notes{{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:18px;}}
.note{{border-top:1px solid var(--rule); padding-top:12px;}}
.note h3{{font-size:12px; letter-spacing:0.1em; text-transform:uppercase;
  color:var(--ink-3); margin:0 0 8px; font-weight:600;}}
.note p{{margin:0 0 8px; font-size:14px; color:var(--ink-2);}}
.note code{{font-family:"IBM Plex Mono",monospace; font-size:12.5px;
  background:var(--rule-2); padding:1px 4px; border-radius:2px; color:var(--ink);}}
.caveat{{border-left:2px solid var(--warn); padding:2px 0 2px 14px; color:var(--ink-2);
  font-size:14.5px; max-width:70ch;}}
footer{{color:var(--ink-3); font-size:12.5px; border-top:1px solid var(--rule);
  padding-top:14px;}}
</style>

<div class="wrap">
<header>
  <p class="eyebrow">PushT · 100k steps · 50 held-out test episodes</p>
  <h1>Does search beat <em>more samples?</em></h1>
  <p class="lede">Three policy families at two demo budgets, plus three 30-demo
  additions, every 10k-step checkpoint
  read out at seven search widths. Each matrix is one arm: gradient step down, search
  width <span class="mono">n</span> across. Darker means a higher success rate.</p>
</header>

<div class="legend">
  <span class="lab mono">0.00</span>
  <span class="ramp">{legend}</span>
  <span class="lab mono">1.00</span>
  <span class="lab">success rate over 50 test episodes</span>
</div>

<p class="caveat"><strong>Read columns, not cells.</strong> At 50 episodes a single
cell carries a 95% CI of roughly ±0.13 near 0.5, so differences under about 0.15 are
not separable. No checkpoint or width is marked “best” here — selection is never done
on the test split.</p>

{''.join(cards)}

<div class="notes">
  <div class="note">
    <h3>Choice mechanism</h3>
    <p>Candidates are ranked by a scalar verifier value — a simulated PushT rollout of
    the proposed action chunk — and the <code>argmax</code> candidate is executed.
    Recorded as <code>selection: argmax</code> in every curve row; no softmax, no
    temperature.</p>
  </div>
  <div class="note">
    <h3>Sampler</h3>
    <p>DDIM, 100 train timesteps, <strong>8 inference steps</strong>,
    <code>eta = 0.0</code> — the deterministic ODE, no noise injected while denoising.
    The initial latent is a fresh draw per candidate, which is what makes the
    <span class="mono">n</span> candidates differ.</p>
  </div>
  <div class="note">
    <h3>Arm labels</h3>
    <p>Each label carries <code>n_layer/n_head/n_emb</code> and the whole policy's
    parameter count, read off the checkpoints. Every arm shares the same 11.2M
    ResNet-18 encoder, so the trunk alone is 5.9M at 4/4/256, 126.6M at 6/8/1024
    and 282.2M for the UNet — these arms are <em>not</em> parameter-matched.</p>
  </div>
  <div class="note">
    <h3>Data</h3>
    <p>The 30-demo train set is the first 30 episodes of the 100-demo list in its own
    order; val and test are copied verbatim between them, so 30-vs-100 varies training
    set size alone.</p>
  </div>
  <div class="note">
    <h3>What n costs</h3>
    <p><span class="mono">n</span> is test-time compute, not training: the same weights
    read out <span class="mono">n</span> ways. Cost is linear in
    <span class="mono">n</span>, so the <span class="mono">n=64</span> column is half
    the price of the whole sweep.</p>
  </div>
</div>

<footer class="mono">Generated from each run's bon_search/success_curves.jsonl ·
{stamp} · regenerate with scripts/build_30_100_success_artifact.py</footer>
</div>
"""
    pathlib.Path(args.out).write_text(html)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
