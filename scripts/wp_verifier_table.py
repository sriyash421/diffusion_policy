"""Markdown readout of the waypoint-verifier evaluation on one checkpoint.

Reads, per verifier value, the best-of-n curve eval_search_pusht.py wrote to
<run>/bon_search_ver-<value>/step_XXXXXXX/success_curve.json and the a*-walk JSON
scripts/astar_uniform_walk.py wrote to <analysis>/<run>/step_XXXXXXX_<value>.json, and
prints three tables plus one success-vs-n figure. Reports measurements only: nothing here
picks a best n or a best value.

    python scripts/wp_verifier_table.py --run <run dir> --step 60000 \
        --fig docs/reports/media/wp_verifier_bon.png
"""
import json
import pathlib

import click

VALUES = ('t_goal', 'wp_v3', 'wp_v5')
# colour + marker + line style per value, so no row is told apart by hue alone
STYLE = {'t_goal': ('#0072B2', 'o', '-'),
         'wp_v3': ('#D55E00', 's', '--'),
         'wp_v5': ('#009E73', '^', ':')}


def _pct(v, ci=None):
    if v is None:
        return '--'
    s = f'{100 * v:.0f}'
    if ci:
        s += f' [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]'
    return s


def _load(path):
    return json.loads(pathlib.Path(path).read_text()) if pathlib.Path(path).is_file() else None


def bon_table(curves):
    ns = sorted({n for c in curves.values() if c for n in c['n']})
    lines = ['| value | ' + ' | '.join(f'n={n}' for n in ns) + ' | wp progress (pusher / T) |',
             '|---|' + '---|' * (len(ns) + 1)]
    for v, c in curves.items():
        if not c:
            lines.append(f'| {v} | ' + ' | '.join('--' for _ in ns) + ' | -- |')
            continue
        by_n = dict(zip(c['n'], zip(c['success_rate'], c['success_ci'])))
        cells = [_pct(*by_n[n]) if n in by_n else '--' for n in ns]
        prog = '--'
        if c.get('mean_wp_pusher_progress'):
            p = dict(zip(c['n'], c['mean_wp_pusher_progress']))
            t = dict(zip(c['n'], c.get('mean_wp_t_progress') or [None] * len(c['n'])))
            prog = ', '.join(f'n{n}: {p[n]:.2f}' + (f'/{t[n]:.2f}' if t.get(n) is not None else '')
                             for n in ns if n in p)
        lines.append(f'| {v} | ' + ' | '.join(cells) + f' | {prog} |')
    return '\n'.join(lines)


def astar_table(walks):
    lines = ['| value | decisions | a* p_best (all) | a* mean rank / n (all) | '
             'a* p_best (informative) | a* mean rank / n (informative) |',
             '|---|---|---|---|---|---|']
    for v, w in walks.items():
        if not w:
            lines.append(f'| {v} | -- | -- | -- | -- | -- |')
            continue
        n = w['n']
        a, i = w['astar']['all'], w['astar'].get('informative', {})

        def rank(d):
            return (f'{d["mean_rank"]["v"] / n:.2f} '
                    f'[{d["mean_rank"]["ci"][0] / n:.2f}, {d["mean_rank"]["ci"][1] / n:.2f}]')
        lines.append(
            f'| {v} | {w["n_decisions"]} | {_pct(a["p_best"]["v"], a["p_best"]["ci"])} | '
            f'{rank(a)} | '
            + (f'{_pct(i["p_best"]["v"], i["p_best"]["ci"])} | {rank(i)} |' if 'p_best' in i
               else f'-- ({i.get("n_points", 0)} pts) | -- |'))
    return '\n'.join(lines)


def nontouch_table(walks):
    lines = ['| value | decisions w/ 2+ non-movers | distinct scores among non-movers | '
             'fraction ranked | a* non-touch decisions | a* p_best among non-movers | '
             'a* mean rank frac (0.5 = null) |',
             '|---|---|---|---|---|---|---|']
    for v, w in walks.items():
        nt = (w or {}).get('nontouch')
        if not nt:
            lines.append(f'| {v} | -- | -- | -- | -- | -- | -- |')
            continue
        p, a = nt['policy'], nt['astar_nontouch']
        fmt = lambda x, f='.2f': '--' if x is None else format(x, f)
        lines.append(f'| {v} | {p["n_decisions_2plus_nonmovers"]} | '
                     f'{fmt(p["mean_distinct"])} of {fmt(p["mean_nonmovers"], ".1f")} | '
                     f'{fmt(p["frac_ranked"])} | {a["n_decisions"]} | '
                     f'{fmt(a["p_best"])} | {fmt(a["mean_rank_frac"])} |')
    return '\n'.join(lines)


def plot(curves, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v, c in curves.items():
        if not c:
            continue
        col, mk, ls = STYLE[v]
        lo = [ci[0] for ci in c['success_ci']]
        hi = [ci[1] for ci in c['success_ci']]
        ax.fill_between(c['n'], lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(c['n'], c['success_rate'], ls, marker=mk, color=col, lw=1.8, ms=6, label=v)
        ax.annotate(v, (c['n'][-1], c['success_rate'][-1]), xytext=(5, 0),
                    textcoords='offset points', color=col, fontsize=9, va='center')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('n candidates (best-of-n, argmax)')
    ax.set_ylabel('test success rate (30 episodes, Wilson 95%)')
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


@click.command()
@click.option('--run', required=True, help='training run dir (holds bon_search_ver-*/)')
@click.option('--step', required=True, type=int)
@click.option('--analysis', default='analysis/wp_verifier', show_default=True)
@click.option('--values', default=','.join(VALUES), show_default=True)
@click.option('--fig', default=None, help='write the success-vs-n figure here')
def main(run, step, analysis, values, fig):
    run = pathlib.Path(run)
    values = values.split(',')
    tag = f'step_{step:07d}'
    curves = {v: _load(run / f'bon_search_ver-{v}' / tag / 'success_curve.json') for v in values}
    walks = {v: _load(pathlib.Path(analysis) / run.name / f'{tag}_{v}.json') for v in values}
    print(f'run: {run.name}  step {step}\n')
    print('### Best-of-n test success (%)\n')
    print(bon_table(curves))
    print('\n### a* rank among n=16 policy candidates (cluster-bootstrap 95% CI)\n')
    print(astar_table(walks))
    print('\n### NO-TOUCH: ranking among chunks that never move the T\n')
    print(nontouch_table(walks))
    if fig:
        plot(curves, fig, f'{run.name} @ {step}')
        print(f'\n-> {fig}')


if __name__ == '__main__':
    main()
