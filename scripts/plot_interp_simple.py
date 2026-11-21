"""Simple figures for the PushT interpolation ST runs (one demo count per call), from the stage candidates
(<run>/bon_search/stage_eval/pi_k<k>/{train,test}_candidates.npz) and the U-Net BC's. Every coverage figure has
three panels -- train starts, all test starts, BC-failure test starts; y = max T coverage, mean over the starts.

  overview.png       clean / flat 400 / ramp 600-200 ST against context length: x = candidates n, STk with k = n-1
                     own attempts in context, scored on candidate n; star = the stage picked by train coverage;
                     BC = its candidate 1 (test: mean over its 3 eval seeds)
  flat.png, ramp.png the same, clean vs noised ST
  candidates.png     BC, clean STk7 and the two no-context controls at 7,200 updates (with / without per-step
                     feedback), each drawing n = 1..32 candidates (<run>/bon_search/cands32: the controls and BC
                     without context, STk7 in its trained window); solid = the verifier-picked candidate,
                     dashed = the max-coverage candidate among the n
  startmaps.png      every model at its train-picked stage: coverage per start at the T start position (the
                     controls: candidate 1)
  summary.json       per model and stage: coverage per group; the train-picked and the test-picked stage

    python scripts/plot_interp_simple.py --demos 50 --out-dir docs/reports/figures/interp_rand50/simple50
"""
import json
import pathlib
import sys

import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from plot_st_candidates import bc_candidates
from plot_st_positions import init_states
from make_geometric_splits import WS

OUT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search_imgonly')
SPLITS = ROOT / 'diffusion_policy/config/splits'
N = 8
GROUPS = ('train starts', 'all test starts', 'BC-failure test starts')
# label: (run-name infix, feedback, scored candidate, colour, marker, line style)
MODELS = {
    'clean ST': ('on-none', 'step', 'k+1', '#0072B2', 'o', '-'),
    'flat 400 ST': ('on-flat400', 'step', 'k+1', '#D55E00', 's', '-'),
    'ramp 600-200 ST': ('on-ramp600to200', 'step', 'k+1', '#009E73', '^', '-'),
    'no-context control': ('nocollect1opt_on-none', 'step', '1', '#CC79A7', 'D', '--'),
    'no-context, no-feedback control': ('nocollect1opt_on-none', 'traj', '1', '#E69F00', 'v', ':'),
}
FIGS = {'overview': ('clean / flat / ramp ST', ['clean ST', 'flat 400 ST', 'ramp 600-200 ST']),
        'flat': ('clean vs flat-noise (t = 400) ST', ['clean ST', 'flat 400 ST']),
        'ramp': ('clean vs ramp-noise (t = 600 -> 200) ST', ['clean ST', 'ramp 600-200 ST'])}


def run_dir(d, infix, fb):
    return (OUT / 'dt_asteroid' / f'dt-gpt2_asteroid8x30_s30_b128_cps1_{infix}_ctx-state_fb-{fb}_tf-cov_ver-wp_v3_la_tgn'
            f'_enc-resnet18_demos-{d}_split-rand{d}cov30_seed-42')


def load_groups(d):
    side = json.loads((SPLITS / f'pusht_rand{d}_train{d}_bccov30_25x25.bcfail.json').read_text())
    per = {int(e): v for e, v in side['per_start'].items()}
    return per, sorted(per), sorted(e for e, v in per.items() if v['bc_fail'])


def model_stages(d, label, test_all, test_fail):
    """{k: {'cov': {group: mean}, 'per': {split: (episodes, coverage)}}} for the stages evaluated so far."""
    infix, fb, cand = MODELS[label][:3]
    out = {}
    for k in range(N):
        z = {s: run_dir(d, infix, fb) / 'bon_search' / 'stage_eval' / f'pi_k{k}' / f'{s}_candidates.npz'
             for s in ('train', 'test')}
        if not all(f.is_file() for f in z.values()):
            continue
        z = {s: np.load(f, allow_pickle=True) for s, f in z.items()}
        c = k if cand == 'k+1' else 0
        per = {s: (z[s]['episode_idxs'].astype(int), z[s]['max_coverage'][:, c]) for s in z}
        row = {int(e): i for i, e in enumerate(per['test'][0])}
        te = per['test'][1]
        out[k] = {'cov': {GROUPS[0]: float(per['train'][1].mean()),
                          GROUPS[1]: float(te[[row[e] for e in test_all]].mean()),
                          GROUPS[2]: float(te[[row[e] for e in test_fail]].mean())},
                  'per': per}
    return out


def bc_stats(d, per, test_all, test_fail):
    bc_run = OUT / 'unet_bc' / f'unetbc-full300_ver-t_goal_enc-resnet18_demos-{d}_split-rand{d}-50k_seed-42'
    tr = bc_candidates(bc_run, 50000, 'train')
    tr_e, tr_c = tr['episode_idxs'].astype(int), tr['max_coverage'][:, 0]
    te = np.array([np.mean(per[e]['cov']) for e in test_all])           # mean over BC's 3 eval seeds
    cov = {GROUPS[0]: float(tr_c.mean()), GROUPS[1]: float(te.mean()),
           GROUPS[2]: float(np.mean([np.mean(per[e]['cov']) for e in test_fail]))}
    return cov, {'train': (tr_e, tr_c), 'test': (np.array(test_all), te)}


def picks(st):
    """(stage picked by train coverage, stage picked by all-test coverage)."""
    return max(st, key=lambda k: st[k]['cov'][GROUPS[0]]), max(st, key=lambda k: st[k]['cov'][GROUPS[1]])


def plot_lines(d, name, title, labels, stages, bc, path):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for g, ax in zip(GROUPS, axs):
        ax.axhline(bc[g], color='black', ls='--', lw=1.8)
        ax.annotate(f'BC {bc[g]:.2f}', (N - .6, bc[g]), xytext=(0, 3), textcoords='offset points', fontsize=8,
                    ha='right', va='bottom')
        for label in labels:
            st = stages[label]
            col, mk, ls = MODELS[label][3:]
            if not st:
                ax.plot([], [], color=col, marker=mk, ls=ls, label=f'{label}: not evaluated yet')
                continue
            ks = sorted(st)
            y = [st[k]['cov'][g] for k in ks]
            kt, kv = picks(st)
            lab = label
            if g == GROUPS[1]:
                lab += f': train-picked STk{kt} {st[kt]["cov"][g]:.2f}, test-picked STk{kv} {st[kv]["cov"][g]:.2f}'
            else:
                lab += f': train-picked STk{kt} {st[kt]["cov"][g]:.2f}'
            ax.plot(ks, y, color=col, marker=mk, ls=ls, lw=2, label=lab)
            ax.plot([kt], [st[kt]['cov'][g]], marker='*', ms=17, color=col, markeredgecolor='black', ls='none')
            ax.annotate(label.replace(' ST', ''), (ks[-1], y[-1]), xytext=(5, 0), textcoords='offset points',
                        color=col, fontsize=8, va='center')
        ax.set_title(g, fontsize=10)
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'{k + 1}\nSTk{k}' for k in range(N)], fontsize=8)
        ax.set_xlabel('candidates n  (STk, k = n - 1 own attempts in context; candidate n scored)')
        ax.set_xlim(-.3, N - .3); ax.set_ylim(0, 1); ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc='upper left')
    axs[0].set_ylabel('max T coverage of the scored attempt\n(mean over starts)')
    fig.suptitle(f'{d} demos -- {title}  (star = stage picked by train coverage; BC dashed)', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


# label: (run dir, scored dir under bon_search, colour, marker)
def cand_models(d):
    """label -> (sources, colour, marker); sources = [(train dir, test dir)], tried in order (32-candidate eval first)."""
    bc = OUT / 'unet_bc' / f'unetbc-full300_ver-t_goal_enc-resnet18_demos-{d}_split-rand{d}-50k_seed-42' / 'bon_search'
    st = run_dir(d, 'on-none', 'step') / 'bon_search'
    c32 = lambda r: (r / 'cands32/pi_k7',) * 2
    return {'BC': ([(bc / 'cands32/step_0050000',) * 2,
                                           (bc / 'dt_eval_train/step_0050000', bc / 'dt_eval/step_0050000')], 'black', 'x'),
            'STk7': ([c32(st), (st / 'stage_eval/pi_k7',) * 2], '#0072B2', 'o'),
            'STk0 with action feedback': ([c32(run_dir(d, 'nocollect1opt_on-none', 'step') / 'bon_search')],
                                                  '#CC79A7', 'D'),
            'STk0 without action feedback': (
                [c32(run_dir(d, 'nocollect1opt_on-none', 'traj') / 'bon_search')], '#E69F00', 'v')}


def cand_curves(sources, test_all, test_fail):
    """{group: {rule: [coverage at n = 1, 2, 4, ...]}} from the first evaluated source; None if none is."""
    for tr, te in sources:
        z = {'train': tr / 'train_candidates.npz', 'test': te / 'test_candidates.npz'}
        if all(f.is_file() for f in z.values()):
            break
    else:
        return None
    z = {s: np.load(f, allow_pickle=True) for s, f in z.items()}
    row = {int(e): i for i, e in enumerate(z['test']['episode_idxs'])}
    rows = {GROUPS[0]: (z['train'], slice(None)), GROUPS[1]: (z['test'], [row[e] for e in test_all]),
            GROUPS[2]: (z['test'], [row[e] for e in test_fail])}
    out = {}
    for g, (zz, r) in rows.items():
        cov, val = zz['max_coverage'][r], zz['value'][r]
        ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= cov.shape[1]]
        idx = np.arange(len(cov))
        out[g] = {'n': ns, 'verifier': [float(cov[idx, val[:, :n].argmax(1)].mean()) for n in ns],
                  'max_coverage': [float(cov[:, :n].max(1).mean()) for n in ns]}
    return out


def plot_candidates(d, test_all, test_fail, path):
    models = {label: (cand_curves(p, test_all, test_fail), c, m) for label, (p, c, m) in cand_models(d).items()}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
    for g, ax in zip(GROUPS, axs):
        for label, (cv, c, m) in models.items():
            if cv is None:
                ax.plot([], [], color=c, marker=m, label=f'{label}: not evaluated yet')
                continue
            ns = cv[g]['n']
            ax.plot(ns, cv[g]['verifier'], color=c, marker=m, lw=2, ls='-',
                    label=f'{label}: n={ns[-1]} verifier {cv[g]["verifier"][-1]:.2f}, max {cv[g]["max_coverage"][-1]:.2f}')
            ax.plot(ns, cv[g]['max_coverage'], color=c, marker=m, lw=1.4, ls='--', alpha=.8)
        ax.set_xscale('log', base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlabel('candidates drawn n'); ax.set_title(g, fontsize=10)
        ax.set_ylim(0, 1); ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(.5, -.16))
    axs[0].set_ylabel('max T coverage of the picked candidate\n(mean over starts)')
    fig.suptitle(f'{d} demos -- n candidates: solid = verifier picks one, dashed = best coverage among the n '
                 '(BC and STk0 without context, STk7 with its own attempts in context)', fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)
    return {label: cv for label, (cv, _, _) in models.items()}


def plot_startmaps(d, stages, bc_per, test_fail, path):
    xy = init_states()[:, 2:4]
    panels = [('BC (UNet), candidate 1 (test: mean of 3 seeds)', bc_per)]
    for label, st in stages.items():
        if st:
            kt = picks(st)[0]
            panels.append((f'{label}, STk{kt} (train-picked), '
                           f'{"attempt " + str(kt + 1) if MODELS[label][2] == "k+1" else "attempt 1"}', st[kt]['per']))
    fig, axs = plt.subplots(2, 3, figsize=(13, 9.5), constrained_layout=True)
    sc = None
    for ax, (title, per) in zip(axs.flat, panels):
        means = []
        tr_e, tr_c = per['train']; te_e, te_c = per['test']
        f = np.isin(te_e, test_fail)
        for e, cov, mk, lw, name in ((tr_e, tr_c, 'o', .6, 'train'), (te_e[~f], te_c[~f], 's', .6, None),
                                     (te_e[f], te_c[f], 'D', 1.8, 'BC-fail')):
            sc = ax.scatter(xy[e, 0], xy[e, 1], c=cov, cmap='cividis', vmin=0, vmax=1, marker=mk, s=55,
                            edgecolor='black', lw=lw, zorder=3)
            if name == 'BC-fail':
                means.append(f'test {te_c.mean():.2f}')
            if name:
                means.append(f'{name} {cov.mean():.2f}')
        ax.set_xlim(0, WS); ax.set_ylim(WS, 0); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{title}\n{"  ·  ".join(means)}', fontsize=8.5)
    for ax in list(axs.flat)[len(panels):]:
        ax.axis('off')
    if sc is not None:
        fig.colorbar(sc, ax=axs, shrink=.6, label='max T coverage of the scored attempt')
    fig.suptitle(f'{d} demos -- coverage per start at the T start position (circle = train start, square = other test '
                 'start, thick diamond = BC-failure test start; 1 seed)', fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)
    print('wrote', path)


@click.command()
@click.option('--demos', required=True, type=int)
@click.option('--out-dir', required=True)
def main(demos, out_dir):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    per, test_all, test_fail = load_groups(demos)
    bc, bc_per = bc_stats(demos, per, test_all, test_fail)
    stages = {label: model_stages(demos, label, test_all, test_fail) for label in MODELS}
    for name, (title, labels) in FIGS.items():
        plot_lines(demos, name, title, labels, stages, bc, out / f'{name}.png')
    plot_startmaps(demos, stages, bc_per, test_fail, out / 'startmaps.png')
    cands = plot_candidates(demos, test_all, test_fail, out / 'candidates.png')
    summary = {'bc': bc, 'candidates': cands}
    for label, st in stages.items():
        if not st:
            summary[label] = None
            continue
        kt, kv = picks(st)
        summary[label] = {'scored': MODELS[label][2], 'stages': {k: v['cov'] for k, v in st.items()},
                          'train_picked': kt, 'test_picked': kv}
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print('wrote', out / 'summary.json')


if __name__ == '__main__':
    main()
