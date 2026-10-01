"""Why best-of-n under a waypoint verifier loses: plots from render_search_videos.py logs.

Reads the steps_<label>_step<STEP>_n<N>.jsonl that `render_search_videos.py
--shadow-values ...` writes (one row per rendered decision: every candidate's score under
the steering value and each shadow value, the executed index, coverage, tracker progress)
and draws, per episode:

  timeline   coverage and tracker progress vs env step, one line per steering value, plus
             the per-decision Spearman rank agreement between the steering score and
             t_goal over the n candidates (+1 = the waypoint verifier orders candidates as
             t_goal does, 0 = unrelated, -1 = opposite).
  scatter    every candidate's steering score (x) against the T-to-goal distance it reaches
             (y, px; lower is better), executed candidates filled, others hollow.
  sheet      a contact sheet of ~12 evenly spaced decision frames per steering value.

    python scripts/wp_rollout_plots.py --run-dir analysis/wp_rollout_viz/step_0060000 \
        --labels bc60k_wp_v3,bc60k_wp_v5 --n 16 --out docs/reports/media/wp_rollout_viz
"""
import json
import pathlib

import click
import cv2
import numpy as np

# colour + line style + marker per steering label, so nothing relies on hue alone
STYLES = [('#D55E00', '--', 's'), ('#009E73', ':', '^'), ('#0072B2', '-', 'o')]


def load_steps(path):
    rows = [json.loads(l) for l in open(path)]
    by_ep = {}
    for r in rows:
        by_ep.setdefault(r['episode_idx'], []).append(r)
    return {e: sorted(v, key=lambda r: r['decision']) for e, v in by_ep.items()}


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def steer_name(row):
    """The steering value is the scores key that is not a shadow (it is listed first)."""
    return next(iter(row['scores']))


def timeline(ax_cov, ax_rank, runs, ep):
    for (label, steps), (col, ls, mk) in zip(runs.items(), STYLES):
        rows = steps.get(ep)
        if not rows:
            continue
        sv = steer_name(rows[0])
        x = [r['env_step'] for r in rows]
        ax_cov.plot(x, [r['max_coverage'] for r in rows], ls, color=col, marker=mk, ms=3,
                    lw=1.6, label=f'{sv}: max coverage')
        prog = [r['progress'][sv][0] for r in rows]
        ax_cov.plot(x, prog, ls, color=col, lw=0.9, alpha=0.6)
        ax_cov.annotate(f'{sv} pusher progress', (x[-1], prog[-1]), fontsize=7, color=col,
                        xytext=(3, 0), textcoords='offset points', va='center')
        rho = [spearman(r['scores'][sv], r['scores']['t_goal']) if 't_goal' in r['scores']
               else np.nan for r in rows]
        ax_rank.plot(x, rho, ls, color=col, marker=mk, ms=3, lw=1.2, label=sv)
    ax_cov.axhline(0.95, color='0.4', lw=0.8, ls='-.')
    ax_cov.annotate('success (0.95)', (0, 0.95), fontsize=7, color='0.3',
                    xytext=(2, 2), textcoords='offset points')
    ax_cov.set_ylim(0, 1.02)
    ax_cov.set_ylabel('coverage / progress')
    ax_cov.legend(fontsize=7, loc='lower right')
    ax_cov.set_title(f'episode {ep}', fontsize=10)
    ax_rank.axhline(0, color='0.4', lw=0.8)
    ax_rank.set_ylim(-1.05, 1.05)
    ax_rank.set_ylabel('rank agreement\nwith t_goal (Spearman)', fontsize=8)
    ax_rank.set_xlabel('env step')
    ax_rank.legend(fontsize=7, loc='lower right')


def scatter(ax, runs, ep):
    for (label, steps), (col, ls, mk) in zip(runs.items(), STYLES):
        rows = steps.get(ep)
        if not rows or 't_goal' not in rows[0]['scores']:
            continue
        sv = steer_name(rows[0])
        xs, ys, ex, ey = [], [], [], []
        for r in rows:
            s = np.asarray(r['scores'][sv])
            d = -np.asarray(r['scores']['t_goal'])
            k = r['executed']
            xs += s.tolist(); ys += d.tolist()
            ex.append(s[k]); ey.append(d[k])
        ax.scatter(xs, ys, s=8, marker=mk, facecolors='none', edgecolors=col, lw=0.5,
                   alpha=0.5, label=f'{sv}: candidate')
        ax.scatter(ex, ey, s=22, marker=mk, color=col, edgecolors='black', lw=0.4,
                   label=f'{sv}: executed')
    ax.set_xlabel('steering verifier score (0..1)')
    ax.set_ylabel('T-to-goal distance reached (px)')
    ax.legend(fontsize=7)


def contact_sheet(frames_dir, label, ep, n, k=12):
    paths = sorted(frames_dir.glob(f'ep{ep}_{label}_n{n}_dec*.png'))
    if not paths:
        return None
    pick = [paths[i] for i in np.linspace(0, len(paths) - 1, min(k, len(paths))).astype(int)]
    tiles = [cv2.imread(str(p)) for p in pick]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(24, 24, 24)) for t in tiles]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 24))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


@click.command()
@click.option('--run-dir', required=True, help='render_search_videos --out-dir')
@click.option('--labels', required=True, help='comma-separated render labels to overlay')
@click.option('--step', default=60000, show_default=True)
@click.option('--n', default=16, show_default=True)
@click.option('--out', required=True)
def main(run_dir, labels, step, n, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run_dir, out = pathlib.Path(run_dir), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for lb in labels.split(','):
        p = run_dir / f'steps_{lb}_step{step}_n{n}.jsonl'
        runs[lb] = load_steps(p)
        print(f'{lb}: {len(runs[lb])} episodes from {p}')
    eps = sorted(set().union(*[set(v) for v in runs.values()]))
    for ep in eps:
        fig = plt.figure(figsize=(12, 6.2))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[2, 1])
        ax_cov, ax_rank = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
        ax_sc = fig.add_subplot(gs[:, 1])
        timeline(ax_cov, ax_rank, runs, ep)
        scatter(ax_sc, runs, ep)
        fig.tight_layout()
        fig.savefig(out / f'ep{ep}_timeline_scatter.png', dpi=130)
        plt.close(fig)
        for lb in runs:
            sheet = contact_sheet(run_dir, lb, ep, n)
            if sheet is not None:
                cv2.imwrite(str(out / f'ep{ep}_{lb}_sheet.png'), sheet)
        # one line per run: where it ended and how often it agreed with t_goal
        for lb, steps in runs.items():
            rows = steps.get(ep, [])
            if not rows:
                continue
            sv = steer_name(rows[0])
            rho = [spearman(r['scores'][sv], r['scores']['t_goal']) for r in rows
                   if 't_goal' in r['scores']]
            print(f'ep{ep} {lb}: final max coverage {rows[-1]["max_coverage"]:.2f}, '
                  f'{len(rows)} decisions, mean rank agreement with t_goal '
                  f'{np.nanmean(rho):+.2f}')
    print(f'-> {out}/')


if __name__ == '__main__':
    main()
