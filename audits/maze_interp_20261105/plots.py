"""Interp30 figures, in the styles of maze_rounds_20261103/plot_clear.py. Everything comes from runs/<tag>/bon_seeds8.npz
(STk: R × 8 attempts × starts × 8 seeds; BC: starts × 8 seeds × 8 iid candidates, fresh seeds, not the scan's) and log.jsonl.
  MAZE_SETTING=interp30 python plots.py [tags...]      (CPU job; the first call caches the training starts)
"""
import json, os, sys, textwrap
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; REP = HERE / 'report'; RUNS = HERE / 'runs'
os.environ.setdefault('MAZE_SETTING', 'interp30')
sys.path[:0] = [str(HERE.parent / 'maze_rounds_20261103')]
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

FB = {'int_s30_bfs': ('BFS', '#000000', 'o', '-'), 'int_s30_bfsdir': ('BFS + dx, dy', '#0072B2', 's', '--'),
      'int_s30_dir': ('dx, dy', '#D55E00', '^', ':'),
      'int_s30_nofb': ('BFS, zeroed for attempts in context', '#009E73', 'D', '-.'),  # ablation: --fb bfs --no-ctx-feedback
      'int_s30_on400': ('BFS + dx, dy, obs noise flat t=400', '#CC79A7', 'v', '-'),     # --obs-noise flat --obs-noise-t 400
      'int_s30_onramp': ('BFS + dx, dy, obs noise ramp t=600→200', '#E69F00', 'P', '-'),  # --obs-noise ramp --obs-noise-ramp 600,200
      'int_k0long': ('BFS + dx, dy, round 0 only, 7200 steps', '#56B4E9', 'X', ':'),   # --rounds 1 --iters 240 (= STk7's total steps)
      'int_k0long_nofb': ('no feedback, round 0 only, 7200 steps', '#D55E00', 'X', ':'),    # --fb none --rounds 1 --iters 240
      'int_k1long': ('BFS + dx, dy, 2 rounds × 3600 steps', '#009E73', 'D', ':'),            # --rounds 2 --iters 120
      'int_k2long': ('BFS + dx, dy, 3 rounds × 2400 steps', '#CC79A7', 'P', ':'),
      'int_s30_bfsdir_ol': ('BFS + dx, dy, open-loop obs (start cell)', '#E69F00', 'v', '-.'),   # --open-loop
      'int_s30_relabel': ('BFS + dx, dy, DAgger: BFS expert labels own attempt', '#CC79A7', 'P', '-'),
      # hidden goal (asteroid.py --hide-goal --fb reward): the policy never sees the goal; feedback = success flag of the previous attempt
      'int_s30_hg_demo': ('hidden goal, ASTEROID (repo): expert path after own attempts', '#D55E00', 'o', '-'),   # --target demo
      'int_s30_hg_relabel': ('hidden goal, DAgger: BFS expert labels own attempt', '#009E73', 'D', '-')}         # --target relabel   # --target relabel           # --rounds 3 --iters 80
MODES = ['int_s30_bfs', 'int_s30_bfsdir', 'int_s30_dir']; ABL = ['int_s30_bfs', 'int_s30_nofb']
NOISE = ['int_s30_bfsdir', 'int_s30_on400', 'int_s30_onramp']; K0LONG = 'int_k0long'
BCCL = [('bc64_closed', 'BC 64-step chunk, re-planned every step', '#000000', '-.'),        # bc_closed.py --exec 1, step 7000
        ('bc_h4_closed', 'BC horizon 4, re-planned every step', '#8B4513', (0, (1, 1)))]  # run_experiment --horizon 4, bc_closed --select
VARIANTS = ['int_s30_bfsdir', 'int_s30_bfsdir_ol', 'int_s30_relabel']
DAGBC = [('dagbc_int_s30_relabel_h64_open', 'BC 64-step chunk on the DAgger data, open loop', '#009E73', '--'),   # dagger_bc.sbatch
         ('dagbc_int_s30_relabel_h64_closed', 'BC 64-step chunk on the DAgger data, re-planned', '#009E73', '-.'),
         ('dagbc_int_s30_relabel_h4_closed', 'BC horizon 4 on the DAgger data, re-planned', '#56B4E9', (0, (1, 1)))]
ALLPV = [('int_s30_bfsdir', None, 'A* demo target, closed-loop obs (STk = after round k)'),
         ('int_s30_bfsdir_ol', None, 'A* demo target, open-loop obs (start cell)'),
         ('int_s30_relabel', None, 'DAgger: BFS expert labels own attempt')]
HIDDEN = ['int_s30_hg_demo', 'int_s30_hg_relabel', 'int_s30_bfsdir', 'int_s30_relabel']
ALLPH = [('int_s30_hg_demo', None, 'hidden goal, ASTEROID (repo)'), ('int_s30_hg_relabel', None, 'hidden goal, DAgger'),
         ('int_s30_bfsdir', None, 'goal visible, BFS + dx, dy, A* demo target (= ASTEROID scheme)'),
         ('int_s30_relabel', None, 'goal visible, BFS + dx, dy, DAgger')]


def allp_file(d):
    """runs/<tag>/bon_seeds<s>_allpairs.npz: 8 seeds, else 3 (allpairs_eval.sbatch default since the hidden-goal runs); None if absent."""
    for s_ in (8, 3):
        if (d / f'bon_seeds{s_}_allpairs.npz').exists():
            return d / f'bon_seeds{s_}_allpairs.npz'
    return None
BC_C = '#555555'; CACHE = HERE / 'starts_cache.npz'


def bon(success, bfs, axis):
    """Best of the first n (n = 1..A) with the verifier (-BFS of the final cell, first on ties); as plot_clear.bon."""
    s, b = np.moveaxis(success, axis, 0), np.moveaxis(bfs, axis, 0); out = []
    for n in range(1, s.shape[0] + 1):
        i = np.argmin(b[:n], 0); out.append(np.take_along_axis(s[:n], i[None], 0)[0])
    return np.stack(out)


def starts():
    if CACHE.exists():
        z = np.load(CACHE); return {k: z[k] for k in z}
    import rounds
    pc = json.loads((rounds.ST_RUN / 'config.json').read_text())['policy']
    s, g, free = rounds.train_starts(rounds.load_data(pc['shape_meta']))
    te = json.loads((HERE / 'test_episodes.json').read_text())['episodes']
    out = dict(train_start=s, train_goal=g, free=free, test_start=np.array([e['start'] for e in te]), test_goal=np.array([e['goal'] for e in te]),
               bc_fail=np.array([e['bc_fail'] for e in te]), bc_scan=np.array([e['bc_scan_successes'] for e in te]))
    np.savez(CACHE, **out); return out


def subsets(st):
    f = st['bc_fail']; out = {'train': ('TRAIN starts: the training episodes (30 train goals)', None),
                              'test': (f'TEST: all {len(f)} held-out-goal pairs', np.ones(len(f), bool)),
                              'fail': (f'TEST, BC-fail pairs ({f.sum()}: BC solved ≤ 1 of 3 scan seeds)', f),
                              'ok': (f'TEST, other pairs ({(~f).sum()}: BC solved ≥ 2 of 3 scan seeds)', ~f)}
    out = {k: v for k, v in out.items() if v[1] is None or v[1].any()}   # 'ok' is empty when the 50 pairs are all BC failures
    out['ctrl'] = ('CONTROL: 50 held-out-goal pairs BC solved on 3 of 3 scan seeds', 'ctrl')
    return out


def load(tag, st):
    z = np.load(RUNS / tag / 'bon_seeds8.npz'); out = {}
    zc = RUNS / tag / 'bon_seeds8_control.npz'
    if zc.exists():   # INTERP_TEST=control bon_eval: the same checkpoints on control_episodes.json (test split only)
        c = np.load(zc); out['ctrl'] = dict(S=c['test_st_success'], B=c['test_st_bfs'], C=c['test_bc_success'], Cb=c['test_bc_bfs'])
    for name, (_, m) in subsets(st).items():
        if name == 'ctrl':
            continue
        sp = 'train' if m is None else 'test'; sl = slice(None) if m is None else m
        if f'{sp}_st_success' not in z:
            continue
        out[name] = dict(S=z[f'{sp}_st_success'][:, :, sl], B=z[f'{sp}_st_bfs'][:, :, sl], C=z[f'{sp}_bc_success'][sl], Cb=z[f'{sp}_bc_bfs'][sl])
    return out


def perf_bon(tag, st, path):
    """Left: perf@n (attempt n alone; STk with n-1 own attempts in context; BC its n-th iid sample). Right: best of n."""
    d = load(tag, st); sub = subsets(st); rows_ = [k for k in ['train', 'test'] + (['fail'] if 'ok' in sub else []) + ['ctrl'] if k in d]   # fail == test when all pairs are BC failures
    cols = plt.cm.viridis(np.linspace(0, .92, 8))
    f, a = plt.subplots(len(rows_), 2, figsize=(13, 4.8 * len(rows_)), gridspec_kw={'width_ratios': [2.3, 1]}, sharey=True, squeeze=False)
    for r, k in enumerate(rows_):
        S, Bf, C, Cb = d[k]['S'], d[k]['B'], d[k]['C'], d[k]['Cb']
        perf, perf_bc = S.mean((-1, -2)), C.mean((0, 1)); best, best_bc = bon(S, Bf, 1).mean((-1, -2)), bon(C, Cb, 2).mean((-1, -2))
        R, A = perf.shape; n = np.arange(1, A + 1)
        for m in range(R):
            a[r, 0].plot(n, perf[m], 'o-', ms=4, lw=1.9, color=cols[m], label=f'STk{m} (trained with {m} attempts in context)')
            a[r, 1].plot(n, best[:, m], 'o-', ms=2.5, lw=1.2, color=cols[m])
        a[r, 0].plot(n, perf_bc, 's--', ms=4.5, lw=2.2, color=BC_C, label='BC (UNet): n-th iid sample / best of n iid samples')
        a[r, 1].plot(n, best_bc, 's--', ms=3, lw=1.6, color=BC_C)
        a[r, 0].set_title('perf@n — ' + sub[k][0], fontsize=9); a[r, 1].set_title('best of n with the verifier', fontsize=9)
        a[r, 0].set_xlabel('attempt n (n − 1 own earlier attempts in context; no selection)', fontsize=8.5)
        a[r, 1].set_xlabel('n candidates (verifier picks one, it is executed)', fontsize=8)
        for ax in a[r]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(n)
        a[r, 0].set_ylabel('success')
    f.legend(*a[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, fontsize=8, frameon=False)
    f.suptitle('\n'.join(textwrap.wrap(f'feedback {FB[tag][0]} ({tag}), interpolation split: perf@n (left) and best of n (right), '
                                       f'STk0 … STk{R - 1} and BC', 120)), fontsize=10.5)
    f.tight_layout(rect=(0, .05, 1, 1)); f.savefig(path, dpi=140); plt.close(f); print(tag, 'perf + bon written')


def goal_map(tag, st, path):
    """BC and STk0..STk7 (STk on its attempt k+1): every goal cell filled with the success of the episodes with that goal.
    Held-out goals outlined (dashed); goals of BC-fail test pairs hatched; held-out goals not in the test set light grey."""
    d = load(tag, st)
    if 'train' not in d:
        print(tag, 'goal map skipped: no train split'); return
    free = st['free']; H, W = free.shape; g = json.loads((HERE / 'goals.json').read_text())
    R = d['train']['S'].shape[0]; A = d['train']['S'].shape[1]; tr = np.stack([d['train']['S'][m, min(m, A - 1)] for m in range(R)]).mean(-1)
    te = np.stack([d['test']['S'][m, min(m, A - 1)] for m in range(R)]).mean(-1)
    pols = [('BC (UNet, demos only)', d['train']['C'][..., 0].mean(-1), d['test']['C'][..., 0].mean(-1))] + \
           [(f'STk{k} (context = {k})', tr[k], te[k]) for k in range(R)]
    draw_goal_maps(pols, st, path, f'feedback {FB[tag][0]} ({tag}): success per goal cell (black = wall; dashed = held-out goal; hatched = goal of a '
                   'BC-fail test pair; grey = held-out goal not in the test set)'); print(tag, 'goal map written')


def goal_map_compare(entries, st, path, title):
    """Goal maps of chosen checkpoints side by side: entries = [(tag, round or 'bc', label)], STk on its attempt k+1."""
    pols = []
    for tag, k, label in entries:
        d = load(tag, st)
        if k == 'bc':
            pols.append((label, d['train']['C'][..., 0].mean(-1), d['test']['C'][..., 0].mean(-1)))
        else:
            A = d['train']['S'].shape[1]; pols.append((label, d['train']['S'][k, min(k, A - 1)].mean(-1), d['test']['S'][k, min(k, A - 1)].mean(-1)))
    draw_goal_maps(pols, st, path, title); print(path.name, 'written')


def draw_goal_maps(pols, st, path, title, hatch=None, test_label='test'):
    """One maze panel per (name, train success per train start, test success per test pair); grid sized to the panels.
    hatch = set of (x, y) goals to hatch (default: goals of the BC-fail test pairs)."""
    free = st['free']; H, W = free.shape; g = json.loads((HERE / 'goals.json').read_text())
    fail_goals = hatch if hatch is not None else {tuple(c) for c in np.round(st['test_goal'][st['bc_fail']]).astype(int)}
    nc = min(3, len(pols)) if len(pols) != 4 else 4; nr = -(-len(pols) // nc)
    cmap = plt.get_cmap('RdYlGn'); fig, axs = plt.subplots(nr, nc, figsize=(4.5 * nc + .8, 4.6 * nr + .6), layout='constrained', squeeze=False)
    axs = axs.ravel()
    for ax in axs[len(pols):]:
        ax.set_visible(False)
    for ax, (name, s_tr, s_te) in zip(axs, pols):
        img = np.ones((H, W, 3)); img[~free] = 0
        for (x, y) in g['test']:
            img[y, x] = (.85, .85, .85)
        for G, s in ((st['train_goal'], s_tr), (st['test_goal'], s_te)):
            gg = np.round(G).astype(int)
            for c in np.unique(gg, axis=0):
                img[c[1], c[0]] = cmap(float(s[(gg == c).all(1)].mean()))[:3]
        ax.imshow(img, origin='upper', extent=(-.5, W - .5, H - .5, -.5), interpolation='nearest')
        for (x, y) in g['test']:
            ax.add_patch(Rectangle((x - .45, y - .45), .9, .9, fill=False, ec='#3A1F00', lw=1.4, ls='--',
                                   hatch='///' if (x, y) in fail_goals else None))
        for v in range(W + 1):
            ax.axvline(v - .5, color='#999999', lw=.4); ax.axhline(v - .5, color='#999999', lw=.4)
        ax.set_xticks([]); ax.set_yticks([])
        f = st['bc_fail']
        ax.set_title(f'{name}\ntrain {s_tr.mean():.2f} · {test_label} {s_te.mean():.2f} · BC-fail pairs {s_te[f].mean():.2f}', fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=axs[:len(pols)].tolist(), shrink=.6 if nr > 1 else .9, pad=.01, label='success of episodes with this goal\n(red = fail, green = success)')
    fig.suptitle('\n'.join(textwrap.wrap(title, 50 * nc)), fontsize=10.5)
    fig.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)


def feedback_compare(tags, st, path, title=None):
    """Rows: BC-fail pairs, other test pairs, train. Cols: STk on attempt k+1 vs k; STk7 perf@n; STk7 best of n. BC dashed."""
    ds = {t: load(t, st) for t in tags}; sub = subsets(st); rows_ = [k for k in ('fail', 'ok', 'ctrl', 'train') if all(k in ds[t] for t in tags)]; out = {}
    f, a = plt.subplots(len(rows_), 3, figsize=(15, 4.3 * len(rows_)), sharey=True, squeeze=False)
    for r, k in enumerate(rows_):
        for t in tags:
            name, c, mk, ls = FB[t]; S, Bf = ds[t][k]['S'], ds[t][k]['B']; R, A = S.shape[:2]
            diag = np.array([S[m, min(m, A - 1)].mean() for m in range(R)]); perf7 = S[R - 1].mean((-1, -2)); best7 = bon(S[R - 1], Bf[R - 1], 0).mean((-1, -2))
            a[r, 0].plot(range(R), diag, marker=mk, ls=ls, color=c, lw=2, ms=5, label=f'ST, {name}' if name.startswith(('hidden', 'goal')) else f'ST, feedback {name}')
            a[r, 1].plot(range(1, A + 1), perf7, marker=mk, ls=ls, color=c, lw=2, ms=5)
            a[r, 2].plot(range(1, A + 1), best7, marker=mk, ls=ls, color=c, lw=2, ms=5)
            out.setdefault(k, {})[t] = dict(stk_diag=diag.round(3).tolist(), stk7_perf=perf7.round(3).tolist(), stk7_bon=best7.round(3).tolist())
        C, Cb = ds[tags[0]][k]['C'], ds[tags[0]][k]['Cb']; A = C.shape[-1]
        a[r, 0].axhline(C[..., 0].mean(), color=BC_C, ls='--', lw=2, label='BC (UNet, demos only), 1 sample')
        a[r, 1].plot(range(1, A + 1), C.mean((0, 1)), 's--', color=BC_C, lw=2, ms=4, label='BC: n-th iid sample / best of n')
        a[r, 2].plot(range(1, A + 1), bon(C, Cb, 2).mean((-1, -2)), 's--', color=BC_C, lw=2, ms=4)
        out[k]['bc'] = dict(first=round(float(C[..., 0].mean()), 3), bon=bon(C, Cb, 2).mean((-1, -2)).round(3).tolist())
        a[r, 0].set_title(f'{sub[k][0]}\nSTk on its attempt k+1 (k own attempts in context)', fontsize=9)
        a[r, 1].set_title('STk7: perf@n (attempt n alone, n − 1 own attempts in context)', fontsize=9)
        a[r, 2].set_title('STk7: best of n with the verifier', fontsize=9)
        a[r, 0].set_xlabel('k (rounds of own context in training)'); a[r, 1].set_xlabel('attempt n'); a[r, 2].set_xlabel('n candidates')
        a[r, 0].set_ylabel('success')
        for ax in a[r]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    h0, l0 = a[0, 0].get_legend_handles_labels(); h1, l1 = a[0, 1].get_legend_handles_labels()
    f.legend(h0 + h1[-1:], l0 + l1[-1:], loc='lower center', ncol=3, fontsize=9, frameon=False)
    f.suptitle('\n'.join(textwrap.wrap((title or 'interpolation split: does feedback in context fix the pairs BC fails on? (BFS / BFS + dx, dy / dx, dy)')
               + f'; rows: {", ".join(rows_)}', 150)), fontsize=11)
    f.tight_layout(rect=(0, .05, 1, .97)); f.savefig(path, dpi=140); plt.close(f)
    path.with_suffix('.json').write_text(json.dumps(out, indent=1)); print(path.name, 'written')


def pair_table(tags, st, path):
    """Per test pair: start, goal, BFS length, BC scan successes /3, fresh BC (8 seeds), STk7 attempt 1 and 8 per feedback mode."""
    from plot_clear import bfs_len
    ds = {t: np.load(RUNS / t / 'bon_seeds8.npz') for t in tags}; free = st['free']; lines = []
    hdr = ['#', 'start', 'goal', 'BFS len', 'BC scan /3', 'BC fresh'] + [f'{FB[t][0]} STk7 a{a}' for t in tags for a in (1, 8)]
    for i in range(len(st['test_start'])):
        s, g = st['test_start'][i].astype(int), st['test_goal'][i].astype(int)
        row = [i, tuple(s), tuple(g), bfs_len(free, s, g), int(st['bc_scan'][i]), round(float(ds[tags[0]]['test_bc_success'][i, :, 0].mean()), 2)]
        row += [round(float(ds[t]['test_st_success'][-1, min(a, ds[t]['test_st_success'].shape[1]) - 1, i].mean()), 2) for t in tags for a in (1, 8)]
        lines.append(row)
    import csv
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(hdr); w.writerows(lines)
    print('pair table written', path); return hdr, lines


def train_curves(tags, st, path):
    """Test success logged during training (every 30 steps, 8 seeds): STk = round k on attempt k+1, over the round's iterations."""
    f, axs = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4.4), sharey=True); axs = np.atleast_1d(axs)
    cols = plt.cm.viridis(np.linspace(0, .92, 8)); bc = None
    for ax, t in zip(axs, tags):
        rr = [json.loads(l) for l in (RUNS / t / 'log.jsonl').read_text().splitlines() if 'test_success' in l]
        for n in range(8):
            q = sorted({r['update']: r for r in rr if r['round'] == n}.values(), key=lambda r: r['update'])
            if q:
                ax.plot([r['update'] for r in q], [r['test_success'] for r in q], '-', lw=1.6, color=cols[n], label=f'STk{n}')
        if (RUNS / t / 'bon_seeds8.npz').exists():
            bc = float(np.load(RUNS / t / 'bon_seeds8.npz')['test_bc_success'][..., 0].mean()); ax.axhline(bc, color=BC_C, ls='--', lw=2, label='BC')
        ax.set_title(f'feedback {FB[t][0]}: all 50 test pairs', fontsize=9); ax.set_xlabel('gradient steps within round k (eval every 30)')
        ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axs[0].set_ylabel('test success (attempt k+1)'); axs[-1].legend(fontsize=7.5, ncol=2)
    f.tight_layout(); f.savefig(path, dpi=140); plt.close(f); print('train curves written')


def noise_arrays(tag, split, noisy=False):
    """(S, B, source) for a run: STk success / final BFS (R × A × starts × seeds). Clean: bon_seeds8[_control].npz, or for the train
    split before bon_eval has written it, eval_seeds8.npz (success only, B = None). Noisy (--eval-obs-noise): the final file, else
    the partial one (rounds saved so far). None if missing."""
    d = RUNS / tag
    if split == 'ctrl':
        f = d / 'bon_seeds8_control.npz'
        return (np.load(f)['test_st_success'], np.load(f)['test_st_bfs'], f.name) if f.exists() and not noisy else None
    files = (['bon_seeds8_noisyobs.npz', 'bon_seeds8_noisyobs_partial.npz'] if noisy else ['bon_seeds8.npz'])
    for fn in files:
        if (d / fn).exists():
            z = np.load(d / fn)
            if f'{split}_st_success' in z:
                return z[f'{split}_st_success'], z[f'{split}_st_bfs'], fn
    if not noisy and (d / 'eval_seeds8.npz').exists():
        z = np.load(d / 'eval_seeds8.npz'); return z[f'{split}_success'], None, 'eval_seeds8.npz'
    return None


def noise_compare(st, path):
    """Observation-noise runs vs the same run without noise (BFS + dx, dy) and the matched-steps STk0 (round 0 only, 7200 steps).
    Rows: train starts, test pairs (all BC failures), control pairs. Cols: STk on attempt k+1; STk(last) perf@n; STk(last) best of n.
    Noise runs: solid = clean eval, hollow markers + dashed = eval with the slot's training noise on the attempt being rolled out."""
    rows_ = [('train', 'TRAIN starts (642 training episodes, 30 train goals)'), ('test', 'TEST: 50 held-out-goal pairs, all BC failures'),
             ('ctrl', 'CONTROL: 50 held-out-goal pairs BC solved on 3/3 scan seeds')]
    rows_ = [r for r in rows_ if any(noise_arrays(t, r[0]) is not None for t in NOISE[1:])]
    bcz = np.load(RUNS / NOISE[0] / 'bon_seeds8.npz'); out = {}
    f, a = plt.subplots(len(rows_), 3, figsize=(16, 4.5 * len(rows_)), sharey=True, squeeze=False)
    for r, (k, title) in enumerate(rows_):
        for t in NOISE:
            name, c, mk, ls = FB[t]
            for noisy in ((False, True) if t != NOISE[0] else (False,)):
                x = noise_arrays(t, k, noisy)
                if x is None:
                    continue
                S, Bf, src = x; R, A = S.shape[:2]
                diag = np.array([S[m, min(m, A - 1)].mean() for m in range(R)]); perf = S[R - 1].mean((-1, -2))
                kw = dict(marker=mk, color=c, ms=6 if not noisy else 5, lw=2.2 if not noisy else 1.3, ls=ls if not noisy else (0, (4, 2)),
                          mfc=c if not noisy else 'white')
                lab = f'ST, {name}' + (' — eval with its training noise' if noisy else (' — clean eval' if t != NOISE[0] else ' (no noise)'))
                a[r, 0].plot(range(R), diag, label=lab, **kw); a[r, 1].plot(range(1, A + 1), perf, **kw)
                rec = dict(stk_diag=diag.round(3).tolist(), last_perf=perf.round(3).tolist(), rounds=R, source=src)
                if Bf is not None:
                    best = bon(S[R - 1], Bf[R - 1], 0).mean((-1, -2)); a[r, 2].plot(range(1, A + 1), best, **kw); rec['last_bon'] = best.round(3).tolist()
                out.setdefault(k, {})[t + ('_noisy' if noisy else '')] = rec
        x = noise_arrays(K0LONG, k)
        if x is not None:
            S, Bf, src = x; name, c, mk, ls = FB[K0LONG]; perf = S[0].mean((-1, -2))
            a[r, 0].axhline(S[0, 0].mean(), color=c, ls=ls, lw=2.4, label=f'{name} (STk0 trained as long as STk7), attempt 1')
            a[r, 1].plot(range(1, len(perf) + 1), perf, marker=mk, ls=ls, color=c, lw=2.2, ms=6)
            rec = dict(first=round(float(S[0, 0].mean()), 3), perf=perf.round(3).tolist(), source=src)
            if Bf is not None:
                best = bon(S[0], Bf[0], 0).mean((-1, -2)); a[r, 2].plot(range(1, len(best) + 1), best, marker=mk, ls=ls, color=c, lw=2.2, ms=6)
                rec['bon'] = best.round(3).tolist()
            out.setdefault(k, {})[K0LONG] = rec
        if k == 'ctrl':
            zc = RUNS / NOISE[0] / 'bon_seeds8_control.npz'; C, Cb = (np.load(zc)['test_bc_success'], np.load(zc)['test_bc_bfs']) if zc.exists() else (None, None)
        else:
            C, Cb = (bcz[f'{k}_bc_success'], bcz[f'{k}_bc_bfs']) if f'{k}_bc_success' in bcz else (None, None)
        if C is not None:
            a[r, 0].axhline(C[..., 0].mean(), color=BC_C, ls='--', lw=2, label='BC (UNet, demos only), 1 sample')
            a[r, 1].plot(range(1, C.shape[-1] + 1), C.mean((0, 1)), 's--', color=BC_C, lw=2, ms=4)
            a[r, 2].plot(range(1, C.shape[-1] + 1), bon(C, Cb, 2).mean((-1, -2)), 's--', color=BC_C, lw=2, ms=4)
            out.setdefault(k, {})['bc'] = round(float(C[..., 0].mean()), 3)
        a[r, 0].set_title(f'{title}\nSTk on its attempt k+1 (k own attempts in context)', fontsize=9)
        a[r, 1].set_title('last saved STk: perf@n (attempt n alone, n − 1 own attempts in context)', fontsize=9)
        a[r, 2].set_title('last saved STk: best of n with the verifier', fontsize=9)
        a[r, 0].set_xlabel('k (rounds of own context in training)'); a[r, 1].set_xlabel('attempt n'); a[r, 2].set_xlabel('n candidates')
        a[r, 0].set_ylabel('success')
        for ax in a[r]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
    hl = {}
    for ax in a[:, 0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    f.legend(list(hl.values()), list(hl.keys()), loc='lower center', ncol=2, fontsize=8.5, frameon=False)
    f.suptitle('observation noise in training (DDPM, on the attempt being predicted; context attempts stay clean): '
               'flat t=400 and ramp t=600→200 (slot 0 → 7) vs the same run without noise', fontsize=11)
    f.tight_layout(rect=(0, (len(hl) + 1) // 2 * .32 / (4.5 * len(rows_)), 1, .97)); f.savefig(path, dpi=140); plt.close(f)
    path.with_suffix('.json').write_text(json.dumps(out, indent=1)); print(path.name, 'written')


ALLP = [('int_s30_bfsdir', None, 'BFS + dx, dy, 8 rounds × 900 steps (STk = after round k)'), ('int_k0long', 0, 'STk0, 7200 steps (round 0 only)'),
        ('int_k0long_nofb', 0, 'STk0, 7200 steps, no feedback at all'), ('int_k1long', 1, 'STk1, 7200 steps total (2 rounds × 3600)'),
        ('int_k2long', 2, 'STk2, 7200 steps total (3 rounds × 2400)')]


def allpairs(st, path, allp=None, bccl=None, title=None, maps=None, map_bcs=(('bc_h4_closed', 'BC horizon 4, re-planned every step'),)):
    """Every (start, held-out goal) pair (bc_scan.npz: 2870 pairs, 41 goals). Panels: all pairs / pairs BC fails (≤ 1 of 3 scan seeds) /
    pairs BC solves (≥ 2 of 3). x = k: STk on attempt k+1 (k own attempts in context); the 8-round run as a line (attempt 1 hollow),
    the 7200-step runs as single markers at their k. BC = scan success (3 seeds). Also a goal map over all 41 held-out goals."""
    z = np.load(HERE / 'bc_scan.npz'); P = z['pairs']; bc = z['success'].mean(1); k_bc = z['success'].sum(1)
    groups = [('all', f'ALL {len(P)} (start, held-out goal) pairs, 41 goals', np.ones(len(P), bool)),
              ('bcfail', f'pairs BC fails (≤ 1 of 3 scan seeds): {(k_bc <= 1).sum()}', k_bc <= 1),
              ('bcok', f'pairs BC solves (≥ 2 of 3 scan seeds): {(k_bc >= 2).sum()}', k_bc >= 2)]
    allp = allp or ALLP
    data = {t: np.load(allp_file(RUNS / t))['test_st_success'] for t, _, _ in allp if allp_file(RUNS / t) is not None}
    if not data:
        print('allpairs: no data yet'); return
    out = {}; f, a = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, (gk, gtitle, m) in zip(a, groups):
        ax.axhline(bc[m].mean(), color=BC_C, ls='--', lw=2, label='BC (UNet, 64-step chunk, open loop), scan, 3 seeds'); out.setdefault(gk, {})['bc'] = round(float(bc[m].mean()), 3)
        for t, name, c, ls in (BCCL if bccl is None else bccl):
            if (RUNS / t / 'bc_closed.npz').exists() and 'allpairs_success' in np.load(RUNS / t / 'bc_closed.npz').files:
                zc = np.load(RUNS / t / 'bc_closed.npz'); assert np.array_equal(zc['allpairs_pairs'], P)
                v = zc['allpairs_success'][m].mean(); ax.axhline(v, color=c, ls=ls, lw=2, label=f'{name}, 8 seeds'); out[gk][t] = round(float(v), 3)
        for t, k, name in allp:
            if t not in data:
                continue
            S = data[t][..., m, :]; R, A = S.shape[:2]; lab, c, mk, ls = FB.get(t, (name, '#000000', 'o', '-'))
            if k is None:
                diag = np.array([S[j, min(j, A - 1)].mean() for j in range(R)]); first = S[:, 0].mean((-1, -2))
                ax.plot(range(R), diag, marker=mk, ls=ls, color=c, lw=2, ms=6, label=f'{name}, attempt k+1')
                ax.plot(range(R), first, marker=mk, ls=':', color=c, lw=1.3, ms=5, mfc='white', label=f'{name}, attempt 1 (no context)')
                out[gk][t] = dict(attempt_k1=diag.round(3).tolist(), attempt1=first.round(3).tolist())
            else:
                v = S[k, min(k, A - 1)].mean(); v1 = S[k, 0].mean()
                ax.plot([k], [v], marker=mk, color=c, ms=11, ls='none', mec='black', label=name)
                if k > 0:
                    ax.plot([k], [v1], marker=mk, color=c, ms=9, ls='none', mfc='white', label=f'{name}, attempt 1')
                out[gk][t] = dict(k=k, attempt_k1=round(float(v), 3), attempt1=round(float(v1), 3))
        ax.set_title(gtitle, fontsize=9.5); ax.set_xlabel('k (rounds of own context in training)'); ax.set_ylim(0, 1.02); ax.grid(alpha=.25)
        ax.set_xticks(range(8))
    a[0].set_ylabel('success (8 seeds; hidden-goal runs 3)' if any(t.startswith('int_s30_hg') for t in data) else 'success (8 seeds)')
    hl = {}
    for ax in a:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hl.setdefault(l, h)
    f.legend(list(hl.values()), list(hl.keys()), loc='lower center', ncol=3, fontsize=8.5, frameon=False)
    f.suptitle(title or ('all held-out goals: STk of the 8-round run vs runs trained the same 7200 total steps (STk0 / STk1 / STk2), BFS + dx, dy feedback '
               'unless noted'), fontsize=11)
    f.tight_layout(rect=(0, .07 + .03 * ((len(hl) + 2) // 3), 1, 1)); f.savefig(path, dpi=140); plt.close(f)
    path.with_suffix('.json').write_text(json.dumps(out, indent=1)); print(path.name, 'written')
    # goal map over all 41 held-out goals (BC-fail = scan ≤ 1 of 3; hatched = goals where BC's mean scan success < 0.5)
    st2 = dict(st, test_goal=P[:, 2:].astype(float), bc_fail=k_bc <= 1)
    gb = {tuple(gg): bc[(P[:, 2:] == gg).all(1)].mean() for gg in np.unique(P[:, 2:], axis=0)}
    tr_bc = np.load(RUNS / 'int_s30_bfsdir' / 'bon_seeds8.npz')['train_bc_success'][..., 0].mean(-1)
    pols = [('BC (UNet, demos only), scan', tr_bc, bc)]
    for t_, name_ in map_bcs:
        if (RUNS / t_ / 'bc_closed.npz').exists() and 'allpairs_success' in np.load(RUNS / t_ / 'bc_closed.npz').files:
            zc = np.load(RUNS / t_ / 'bc_closed.npz'); assert np.allclose(zc['train_pairs'], np.concatenate([st['train_start'], st['train_goal']], 1))
            pols.append((name_, zc['train_success'].mean(1), zc['allpairs_success'].mean(1)))
    for t, k, name in maps or [('int_s30_bfsdir', 0, 'STk0, 900 steps'), ('int_k0long', 0, 'STk0, 7200 steps'), ('int_k0long_nofb', 0, 'STk0, 7200 steps, no feedback'),
                       ('int_k1long', 1, 'STk1, 7200 steps total'), ('int_k2long', 2, 'STk2, 7200 steps total'), ('int_s30_bfsdir', 7, 'STk7, 7200 steps total')]:
        if t not in data or not (RUNS / t / 'bon_seeds8.npz').exists() or 'train_st_success' not in np.load(RUNS / t / 'bon_seeds8.npz'):
            continue
        Tr = np.load(RUNS / t / 'bon_seeds8.npz')['train_st_success']; A = Tr.shape[1]
        pols.append((f'{name} (attempt {min(k, A - 1) + 1})', Tr[k, min(k, A - 1)].mean(-1), data[t][k, min(k, data[t].shape[1] - 1)].mean(-1)))
    draw_goal_maps(pols, st2, path.with_name(path.stem + '_goalmap.png'),
                   'all (start, held-out goal) pairs: success per goal cell, averaged over all starts (black = wall; dashed = held-out goal; '
                   'hatched = held-out goal BC solves < 50% of the time in the scan)', hatch={g_ for g_, v in gb.items() if v < .5}, test_label='all held-out pairs')
    print(path.stem + '_goalmap.png written')


def matched_table(st, path, st_rows=None, bccl=None):
    """Train / 50 test (BC-fail) / 50 control / all held-out pairs (+ BC-fail / BC-ok by scan) for BC (open loop, re-planned) and the
    matched-steps ST runs; STk on its attempt k+1, 8 seeds. -> path (.json); None where the eval has not finished."""
    z = np.load(HERE / 'bc_scan.npz'); k_bc = z['success'].sum(1); scan = z['success'].mean(1)
    groups = {'all': np.ones(len(k_bc), bool), 'all_bcfail': k_bc <= 1, 'all_bcok': k_bc >= 2}
    f = lambda v: None if v is None else round(float(np.mean(v)), 3)
    ref = np.load(RUNS / 'int_s30_bfsdir' / 'bon_seeds8.npz'); refc = np.load(RUNS / 'int_s30_bfsdir' / 'bon_seeds8_control.npz')
    rows = [dict(name='BC, 64-step chunk, open loop (the BC everywhere above)', train=f(ref['train_bc_success'][..., 0]),
                 test=f(ref['test_bc_success'][..., 0]), control=f(refc['test_bc_success'][..., 0]), **{g: f(scan[m]) for g, m in groups.items()},
                 note='all-pairs columns: the scan (3 seeds)')]
    for t, name, _, _ in (BCCL if bccl is None else bccl):
        if (RUNS / t / 'bc_closed.npz').exists() and 'allpairs_success' in np.load(RUNS / t / 'bc_closed.npz').files:
            c = np.load(RUNS / t / 'bc_closed.npz')
            rows.append(dict(name=name, train=f(c['train_success']), test=f(c['test_success']), control=f(c['control_success']),
                             **{g: f(c['allpairs_success'][m]) for g, m in groups.items()}))
    for t, k, name in st_rows or [('int_s30_bfsdir', 0, 'STk0, 900 steps (8-round run, round 0)'), ('int_s30_bfsdir', 7, 'STk7, 7200 steps total (8 rounds × 900)'),
                       ('int_k0long', 0, 'STk0, 7200 steps'), ('int_k0long_nofb', 0, 'STk0, 7200 steps, NO feedback (all channels zero)'),
                       ('int_k1long', 1, 'STk1, 7200 steps total (2 rounds × 3600)'), ('int_k2long', 2, 'STk2, 7200 steps total (3 rounds × 2400)')]:
        d = RUNS / t; r = dict(name=name)
        for fn, key, sp in [('bon_seeds8.npz', 'train', 'train'), ('bon_seeds8.npz', 'test', 'test'), ('bon_seeds8_control.npz', 'control', 'test')]:
            if (d / fn).exists() and f'{sp}_st_success' in np.load(d / fn).files:
                S = np.load(d / fn)[f'{sp}_st_success']; r[key] = f(S[k, min(k, S.shape[1] - 1)])
        if allp_file(d) is not None:
            S = np.load(allp_file(d))['test_st_success']; v = S[k, min(k, S.shape[1] - 1)].mean(-1)
            r.update({g: f(v[m]) for g, m in groups.items()})
        rows.append(r)
    path.write_text(json.dumps(dict(rows=rows, n_pairs={g: int(m.sum()) for g, m in groups.items()}), indent=1)); print(path.name, 'written')
    for r in rows:
        print(f"{r['name'][:60]:60s}", *[r.get(c) for c in ('train', 'test', 'control', 'all', 'all_bcfail', 'all_bcok')])


LOOP = [('int_s30_bfsdir', 'ST grid, closed loop (current-cell obs)', '#0072B2', 's', '-'),
        ('int_s30_bfsdir_ol', 'ST grid, open-loop obs (start-cell obs; feedback per step)', '#D55E00', 'v', '--'),
        ('c_s30_bfsdir', 'ST continuous, closed loop (real-valued actions)', '#009E73', 'o', '-.')]
LOOP_BC = [('bc64_closed', 'BC grid, 64-step chunk re-planned every step', '#000000', '-.'),
           ('bc_h4_closed', 'BC grid, horizon 4 closed loop', '#8B4513', (0, (1, 1))),
           ('c_bc_h4', 'BC continuous, horizon 4 closed loop', '#009E73', (0, (3, 1, 1, 1)))]


def loop_compare(st, path):
    """Open vs closed loop and grid vs continuous (all BFS + dx, dy feedback, ast_s30 schedule). Panels: train starts / the 50 BC-fail test
    pairs / the 50 control pairs / all 2870 held-out pairs. ST: STk on attempt k+1 (filled) and on attempt 1 (hollow); BC: horizontal lines
    (open-loop grid BC dashed grey: fresh 8 seeds on train / test / control, the 3-seed scan on all pairs)."""
    def st_arr(t, split):
        d = RUNS / t
        f = {'train': ['bon_seeds8.npz'], 'test': ['bon_seeds8.npz', 'bon_seeds8_partial.npz'], 'ctrl': ['bon_seeds8_control.npz'],
             'all': ['bon_seeds8_allpairs.npz']}[split]
        for name in f:
            if (d / name).exists():
                z = np.load(d / name); k = 'train_st_success' if split == 'train' else 'test_st_success'
                if k in z.files:
                    return z[k]
        return None
    ref = np.load(RUNS / 'int_s30_bfsdir' / 'bon_seeds8.npz'); ctl = np.load(RUNS / 'int_s30_bfsdir' / 'bon_seeds8_control.npz')
    scan = np.load(HERE / 'bc_scan.npz')
    bc_open = {'train': ref['train_bc_success'][..., 0].mean(), 'test': ref['test_bc_success'][..., 0].mean(),
               'ctrl': ctl['test_bc_success'][..., 0].mean(), 'all': scan['success'].mean()}
    panels = [('train', 'TRAIN starts (30 train goals)'), ('test', 'TEST: 50 BC-fail held-out pairs'),
              ('ctrl', 'CONTROL: 50 pairs open-loop BC solves'), ('all', 'ALL 2870 held-out pairs (41 goals)')]
    out = {}; f, a = plt.subplots(1, 4, figsize=(22, 5.4), sharey=True)
    for ax, (sp, title) in zip(a, panels):
        o = out.setdefault(sp, {}); ax.axhline(bc_open[sp], color=BC_C, ls='--', lw=2, label='BC grid, 64-step chunk, open loop'); o['bc_open'] = round(float(bc_open[sp]), 3)
        for t, name, c, ls in LOOP_BC:
            if (RUNS / t / 'bc_closed.npz').exists():
                zc = np.load(RUNS / t / 'bc_closed.npz'); key = {'ctrl': 'control', 'all': 'allpairs'}.get(sp, sp) + '_success'
                v = float(zc[key].mean()); ax.axhline(v, color=c, ls=ls, lw=2, label=name); o[t] = round(v, 3)
        for t, name, c, mk, ls in LOOP:
            S = st_arr(t, sp)
            if S is None:
                continue
            R, A = S.shape[:2]; diag = np.array([S[j, min(j, A - 1)].mean() for j in range(R)]); first = S[:, 0].mean((-1, -2))
            ax.plot(range(R), diag, marker=mk, ls=ls, color=c, lw=2, ms=6, label=f'{name}: attempt k+1')
            ax.plot(range(R), first, marker=mk, ls=':', color=c, lw=1.2, ms=5, mfc='white', label=f'{name}: attempt 1')
            o[t] = dict(attempt_k1=diag.round(3).tolist(), attempt1=first.round(3).tolist(), rounds=R)
        ax.set_title(title, fontsize=10); ax.set_xlabel('round k (STk; k own attempts in context)'); ax.set_ylim(-.03, 1.03); ax.grid(alpha=.3)
    a[0].set_ylabel('success (8 seeds)'); h, l = a[1].get_legend_handles_labels()
    f.legend(h, l, loc='lower center', ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(.5, -.02))
    f.suptitle('Open vs closed loop, grid vs continuous actions (BFS + dx, dy feedback; ST 8 rounds × 900 steps; BC 10k updates)', fontsize=11)
    f.tight_layout(rect=(0, .14, 1, 1)); f.savefig(path, dpi=120, bbox_inches='tight'); plt.close(f)
    (path.with_suffix('.json')).write_text(json.dumps(out, indent=1)); print(json.dumps(out))

N16_ST = [('int_k0long', 0, 'STk0, 7200 steps, BFS + dx, dy feedback (1 round; no context: 16 iid attempts)', '#0072B2', 'o', '-'),
          ('int_s30_bfsdir', 7, 'STk7, 7200 steps total (8 rounds × 900), feedback, attempt n sees the last min(7, n−1) attempts', '#009E73', 's', '--'),
          ('int_k0long_nofb', 0, 'STk0, 7200 steps, NO feedback (all channels zero; 16 iid attempts)', '#D55E00', '^', ':')]
N16_BC = [('bc_h4_closed', 'BC horizon 4, re-planned every step (step 10000; n-th / best of n iid rollouts)', '#8B4513', 'D', '-.'),
          ('bc64_closed', 'BC 64-step chunk, re-planned every step (step 7000)', '#000000', 'v', '-.')]


def n16(st, path, st_entries=None, bc_entries=None, fname=lambda t, k: f'bon_seeds8_n16_r{k}', title_extra=''):
    """perf@n (top) and best of n with the verifier (bottom), n = 1..16, on train starts, all held-out pairs BC fails (scan <= 1/3)
    and all held-out pairs (2870). ST: runs/<tag>/bon_seeds8_n16_r<k>[_allpairs].npz (8 seeds x 16 attempts in a row);
    BC: runs/<tag>/bc_n16.npz (128 iid closed-loop rollouts per pair = 8 seeds x 16 candidates)."""
    k_bc = np.load(HERE / 'bc_scan.npz')['success'].sum(1); fail = k_bc <= 1
    cols = [('train', None, f'TRAIN starts ({len(st["train_start"])} pairs, 30 train goals)'),
            ('all', fail, f'ALL held-out pairs BC fails ({fail.sum()}: BC ≤ 1 of 3 scan seeds)'),
            ('all', np.ones(len(fail), bool), f'ALL held-out pairs ({len(fail)}, 41 held-out goals)')]
    curves = []   # (label, color, marker, ls, {col: (perf, best)})
    for t, k, lab, c, mk, ls in (st_entries or N16_ST):
        d = {}
        for j, (sp, m, _) in enumerate(cols):
            f = RUNS / t / (fname(t, k) + ('_allpairs' if sp == 'all' else '') + '.npz')
            if not f.exists():
                continue
            z = np.load(f); key = 'train' if sp == 'train' else 'test'
            S, B = z[f'{key}_st_success'][0], z[f'{key}_st_bfs'][0]   # (16, pairs, seeds)
            if m is not None:
                assert S.shape[1] == len(m), (S.shape, len(m)); S, B = S[:, m], B[:, m]
            d[j] = (S.mean((1, 2)), bon(S, B, 0).mean((1, 2)))
        curves.append((lab, c, mk, ls, d))
    for t, lab, c, mk, ls in (bc_entries or N16_BC):
        d = {}; f = RUNS / t / 'bc_n16.npz'
        if f.exists():
            z = np.load(f)
            for j, (sp, m, _) in enumerate(cols):
                key = 'train' if sp == 'train' else 'allpairs'
                S, B = z[f'{key}_success'], z[f'{key}_bfs']; S, B = S.reshape(len(S), -1, 16), B.reshape(len(B), -1, 16)   # (pairs, seeds, 16)
                if m is not None:
                    assert len(S) == len(m); S, B = S[m], B[m]
                d[j] = (S.mean((0, 1)), bon(S, B, 2).mean((1, 2)))
        curves.append((lab, c, mk, ls, d))
    f, a = plt.subplots(2, 3, figsize=(17, 9.5), sharey=True); n = np.arange(1, 17)
    for j, (_, _, title) in enumerate(cols):
        for lab, c, mk, ls, d in curves:
            if j in d:
                a[0, j].plot(n, d[j][0], marker=mk, ls=ls, ms=4, lw=1.9, color=c, label=lab)
                a[1, j].plot(n, d[j][1], marker=mk, ls=ls, ms=4, lw=1.9, color=c, label=lab)
        a[0, j].set_title('perf@n — ' + title, fontsize=9); a[1, j].set_title('best of n with the verifier — ' + title, fontsize=9)
        a[0, j].set_xlabel('attempt n (ST: n − 1 own earlier attempts before it; BC: n-th iid rollout)', fontsize=8)
        a[1, j].set_xlabel('n candidates (verifier = −BFS of the final cell picks one; it is executed)', fontsize=8)
        for ax in a[:, j]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(n)
    a[0, 0].set_ylabel('success'); a[1, 0].set_ylabel('success')
    h, l = [], []
    for ax in a.flat:   # union of the labels (some evals may be missing from some panels)
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    f.legend(h, l, loc='lower center', ncol=2, fontsize=8.5, frameon=False)
    f.suptitle('Interpolation split, n = 1..16: closed-loop BC vs ST at 7200 updates (STk0 with / without feedback, STk7 of the 8-round run); '
               '8 seeds per pair (BC 8 × 16 iid rollouts)' + title_extra, fontsize=10.5)
    f.tight_layout(rect=(0, .07, 1, .97)); f.savefig(path, dpi=140); plt.close(f); print(path.name, 'written')
    out = {lab: {cols[j][2]: dict(perf=np.round(v[0], 3).tolist(), best=np.round(v[1], 3).tolist()) for j, v in d.items()} for lab, _, _, _, d in curves}
    path.with_suffix('.json').write_text(json.dumps(out, indent=1))


C_BC = '#8B4513'


def cont_load():
    """Continuous (interp30c) results, all from the n = 1..16 evals (8 seeds x 16 attempts; held-out = every (start, held-out goal) pair).
    BC-fail pairs = the pairs the continuous closed-loop BC (c_bc_h4, bc_closed.npz, 8 seeds) solves <= 1/3 of the time; the BC curves
    themselves use the 128 fresh rollouts of bc_n16.npz. Returns sets, BC (S, B as (pairs, 8, 16)) and ST {(tag, k): {split: (S, B)}}
    with S, B (16 attempts, pairs, 8 seeds)."""
    zb = np.load(RUNS / 'c_bc_h4' / 'bc_closed.npz'); P = np.load(HERE / 'bc_scan.npz')['pairs']; assert np.array_equal(zb['allpairs_pairs'], P)
    fail = zb['allpairs_success'].mean(1) <= 1 / 3
    sets = [('train', None, 'TRAIN starts (642 pairs, 30 train goals)'),
            ('fail', fail, f'held-out pairs continuous BC fails ({fail.sum()})'),
            ('all', np.ones(len(P), bool), f'ALL held-out pairs ({len(P)}, 41 goals)')]
    z = np.load(RUNS / 'c_bc_h4' / 'bc_n16.npz'); bc = {}
    for sp, m, _ in sets:
        k = 'train' if sp == 'train' else 'allpairs'
        S, B = z[f'{k}_success'].reshape(-1, 8, 16), z[f'{k}_bfs'].reshape(-1, 8, 16)
        bc[sp] = (S, B) if m is None else (S[m], B[m])
    stres = {}
    for t, ks in (('c_k0long', [0]), ('c_k0long_nofb', [0]), ('c_s30_bfsdir', range(8))):
        for k in ks:
            d = {}
            for sp, m, _ in sets:
                f = RUNS / t / (f'bon_seeds8_n16_r{k}' + ('' if sp == 'train' else '_allpairs') + '.npz')
                if f.exists():
                    zz = np.load(f); key = 'train' if sp == 'train' else 'test'
                    S, B = zz[f'{key}_st_success'][0], zz[f'{key}_st_bfs'][0]
                    if m is not None:
                        assert S.shape[1] == len(m), (f, S.shape); S, B = S[:, m], B[:, m]
                    d[sp] = (S, B)
            stres[(t, k)] = d
    return sets, fail, P, bc, stres


def cont_section(st):
    """Continuous-action section of the report: cont_demos.png, cont_rounds.png, cont_goalmap.png, cont_table.json."""
    sets, fail, P, bc, stres = cont_load(); out = {}
    # 1. table numbers (attempt 1 / attempt k+1 / attempt 16 / best of 16 per policy and set)
    for sp, _, _ in sets:
        o = out.setdefault(sp, {}); S, B = bc[sp]
        o['bc'] = dict(first=round(float(S[..., 0].mean()), 3), attempt16=round(float(S[..., 15].mean()), 3), best16=round(float(bon(S, B, 2)[-1].mean()), 3))
        for (t, k), d in stres.items():
            if sp in d:
                S, B = d[sp]; o[f'{t}_r{k}'] = dict(first=round(float(S[0].mean()), 3), attempt_k1=round(float(S[min(k, 15)].mean()), 3),
                                                   attempt16=round(float(S[-1].mean()), 3), best16=round(float(bon(S, B, 0)[-1].mean()), 3))
    out['n_pairs'] = dict(train=int(len(bc['train'][0])), fail=int(fail.sum()), all=int(len(fail)))
    (REP / 'cont_table.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:1500])
    # 2. goal maps over all held-out pairs (hatched = goals the continuous BC solves < 50%)
    zc = np.load(RUNS / 'c_bc_h4' / 'bc_closed.npz'); assert np.allclose(zc['train_pairs'][:, 2:], st['train_goal'])
    st2 = dict(st, test_goal=P[:, 2:].astype(float), bc_fail=fail); bcr = zc['allpairs_success'].mean(1)
    hatch = {tuple(g) for g in np.unique(P[:, 2:], axis=0) if bcr[(P[:, 2:] == g).all(1)].mean() < .5}
    pols = [('BC horizon 4, closed loop', bc['train'][0].mean((1, 2)), np.load(RUNS / 'c_bc_h4' / 'bc_n16.npz')['allpairs_success'].mean(1))]
    for t, k, name in (('c_k0long', 0, 'STk0, 7200 steps'), ('c_k0long_nofb', 0, 'STk0, 7200 steps, no feedback'),
                       ('c_s30_bfsdir', 7, 'STk7 (8 rounds, 7200 steps), attempt 8')):
        d = stres[(t, k)]
        if 'train' in d and 'all' in d:
            Sa = np.load(RUNS / t / f'bon_seeds8_n16_r{k}_allpairs.npz')['test_st_success'][0]
            pols.append((name, d['train'][0][k].mean(-1), Sa[k].mean(-1)))
    draw_goal_maps(pols, st2, REP / 'cont_goalmap.png', 'CONTINUOUS actions, all (start, held-out goal) pairs: success per goal cell averaged over all '
                   'starts (dashed = held-out goal; hatched = held-out goal the continuous BC solves < 50%)', hatch=hatch, test_label='all held-out')
    print('cont_goalmap.png written')
    # 3. demos
    import zarr
    spec = json.loads((HERE.parent / 'maze_quadrants_delta_20261019' / 'datasets.json').read_text())['interp30c']; z = zarr.open(spec['dataset'], 'r')
    Pd = np.asarray(z['data/obs/agent_pos']); Aa = np.asarray(z['data/actions']); ends = np.asarray(z['meta/episode_ends']); begs = np.r_[0, ends[:-1]]
    free = st['free'].astype(bool); f = plt.figure(figsize=(17, 5.6)); gs = f.add_gridspec(1, 4, width_ratios=[1.5, 1.5, 1, 1]); ax = f.add_subplot(gs[0, 0:2])
    ax.imshow(~free, cmap='Greys', origin='upper', extent=(-.5, free.shape[1] - .5, free.shape[0] - .5, -.5), vmin=0, vmax=1.6)
    rng = np.random.default_rng(1); pick = rng.choice(len(ends), 14, replace=False); cols = plt.cm.tab20(np.linspace(0, 1, len(pick)))
    for c, e in zip(cols, pick):
        pts = np.concatenate([Pd[begs[e]:ends[e]], Pd[ends[e] - 1:ends[e]] + Aa[ends[e] - 1:ends[e]]])
        ax.plot(pts[:, 0], pts[:, 1], '-', color=c, lw=1.8); ax.plot(pts[:, 0], pts[:, 1], '.', color=c, ms=5)
        ax.plot(*pts[0], 'o', color=c, ms=8, mec='black'); ax.plot(*pts[-1], '*', color=c, ms=13, mec='black')
    ax.set_title('14 random smoothed-A* demos (circle = jittered start, star = goal centre, dots = steps; walls grey)', fontsize=10)
    ax.set_xticks(range(free.shape[1])); ax.set_yticks(range(free.shape[0])); ax.set_aspect('equal')
    ax = f.add_subplot(gs[0, 2]); nrm = np.linalg.norm(Aa, axis=1)
    ax.hist(nrm, bins=40, color='#009E73'); ax.set_xlabel('demo action length |a| (cells)'); ax.set_ylabel('steps')
    ax.set_title(f'action length: mean {nrm.mean():.2f}, min {nrm.min():.2f}\n({len(nrm)} steps, {len(ends)} demos)', fontsize=10)
    ax = f.add_subplot(gs[0, 3], projection='polar'); ang = np.arctan2(-Aa[:, 1], Aa[:, 0])   # y down in the image -> flip for compass
    ax.hist(ang, bins=72, color='#0072B2'); ax.set_yticklabels([])
    diag = float(((np.abs(Aa) > 1e-4).all(1)).mean()); ax.set_title(f'action direction ({diag:.0%} not axis-aligned)', fontsize=10, pad=14)
    f.tight_layout(); f.savefig(REP / 'cont_demos.png', dpi=120); plt.close(f); print('cont_demos.png written')


def n16c(st, path):
    """Continuous setting (interp30c), n = 1..16: perf@n (top) and best of n with the verifier (bottom) on train starts, the held-out
    pairs the CONTINUOUS closed-loop BC fails (c_bc_h4 bc_closed.npz, all pairs, 8 seeds: success <= 1/3) and all held-out pairs.
    ST: STk0 7200 steps with feedback (c_k0long) / without (c_k0long_nofb), STk1..7 of the 8-round run (c_s30_bfsdir; attempt n sees
    the last min(7, n-1) attempts); BC: c_bc_h4 (horizon 4 closed loop, 128 iid rollouts per pair = 8 seeds x 16, fresh seeds)."""
    zb = np.load(RUNS / 'c_bc_h4' / 'bc_closed.npz'); P = np.load(HERE / 'bc_scan.npz')['pairs']; assert np.array_equal(zb['allpairs_pairs'], P)
    bcr = zb['allpairs_success'].mean(1); fail = bcr <= 1 / 3
    cols = [('train', None, 'TRAIN starts (642 pairs, 30 train goals; jittered continuous starts)'),
            ('all', fail, f'held-out pairs continuous BC fails ({fail.sum()}: ≤ 1/3 of 8 seeds)'),
            ('all', np.ones(len(fail), bool), f'ALL held-out pairs ({len(fail)}, 41 held-out goals)')]
    vir = plt.cm.viridis(np.linspace(.15, .9, 7))
    sts = [('c_k0long', 0, 'STk0, 7200 steps, BFS + dx, dy feedback (no context: 16 iid attempts)', '#0072B2', 'o', '-', 2.6),
           ('c_k0long_nofb', 0, 'STk0, 7200 steps, NO feedback (16 iid attempts)', '#D55E00', '^', '-', 2.6)]
    sts += [('c_s30_bfsdir', 7, 'STk7 (8 rounds × 900 = 7200 steps; attempt n sees its last min(7, n − 1) attempts)', '#009E73', 's', '--', 2.6)]
    curves = []
    for t, k, lab, c, mk, ls, lw in sts:
        d = {}
        for j, (sp, m, _) in enumerate(cols):
            f = RUNS / t / (f'bon_seeds8_n16_r{k}' + ('_allpairs' if sp == 'all' else '') + '.npz')
            if not f.exists():
                continue
            z = np.load(f); key = 'train' if sp == 'train' else 'test'
            S, B = z[f'{key}_st_success'][0], z[f'{key}_st_bfs'][0]
            if m is not None:
                assert S.shape[1] == len(m); S, B = S[:, m], B[:, m]
            d[j] = (S.mean((1, 2)), bon(S, B, 0).mean((1, 2)))
        curves.append((lab, c, mk, ls, lw, d))
    d = {}; f = RUNS / 'c_bc_h4' / 'bc_n16.npz'
    if f.exists():
        z = np.load(f)
        for j, (sp, m, _) in enumerate(cols):
            key = 'train' if sp == 'train' else 'allpairs'
            S, B = z[f'{key}_success'], z[f'{key}_bfs']; S, B = S.reshape(len(S), -1, 16), B.reshape(len(B), -1, 16)
            if m is not None:
                S, B = S[m], B[m]
            d[j] = (S.mean((0, 1)), bon(S, B, 2).mean((1, 2)))
    curves.append(('BC horizon 4, closed loop (n-th / best of n iid rollouts)', '#8B4513', 'D', '-.', 2.6, d))
    f, a = plt.subplots(2, 3, figsize=(18, 10), sharey=True); n = np.arange(1, 17)
    for j, (_, _, title) in enumerate(cols):
        for lab, c, mk, ls, lw, d in curves:
            if j in d:
                for r in (0, 1):
                    a[r, j].plot(n, d[j][r], marker=mk, ls=ls, ms=4, lw=lw, color=c, label=lab)
        a[0, j].set_title('perf@n — ' + title, fontsize=9); a[1, j].set_title('best of n with the verifier — ' + title, fontsize=9)
        a[0, j].set_xlabel('attempt n (ST: n − 1 own earlier attempts before it; BC: n-th iid rollout)', fontsize=8)
        a[1, j].set_xlabel('n candidates (verifier = −BFS of the final cell picks one)', fontsize=8)
        for ax in a[:, j]:
            ax.set_ylim(0, 1.02); ax.grid(alpha=.25); ax.set_xticks(n)
    a[0, 0].set_ylabel('success'); a[1, 0].set_ylabel('success'); h, l = [], []
    for ax in a.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    f.legend(h, l, loc='lower center', ncol=2, fontsize=9, frameon=False)
    f.suptitle('CONTINUOUS actions (point agent, |a| ≤ 1, smoothed-A* demos), n = 1..16, all at 7200 training steps: closed-loop BC, STk0 with / '
               'without feedback, STk7; 8 seeds per pair', fontsize=10.5)
    f.tight_layout(rect=(0, .09, 1, .97)); f.savefig(path, dpi=130); plt.close(f); print(path.name, 'written')
    out = {lab: {cols[j][2]: dict(perf=np.round(v[0], 3).tolist(), best=np.round(v[1], 3).tolist()) for j, v in d.items()} for lab, _, _, _, _, d in curves}
    path.with_suffix('.json').write_text(json.dumps(out, indent=1)); print(json.dumps({k: list(v) for k, v in out.items()}))


def main(tags):
    REP.mkdir(exist_ok=True); st = starts(); done = [t for t in tags if (RUNS / t / 'bon_seeds8.npz').exists()]
    for t in done:
        perf_bon(t, st, REP / f'perf_bon_{t}.png'); goal_map(t, st, REP / f'goalmap_{t}.png')
    if done:
        modes = [t for t in done if t in MODES]
        if modes:
            feedback_compare(modes, st, REP / 'feedback_compare.png')
        if all(t in done for t in ABL):
            feedback_compare(ABL, st, REP / 'ablation_nofb.png', 'ablation: BFS feedback vs the same run with the feedback of the attempts in '
                             'context zeroed (cells and moves kept)')
        if any(t in MODES + ABL[1:] for t in done):
            pair_table([t for t in done if t in MODES + ABL[1:]], st, REP / 'pair_table.csv')
    if any(t in tags for t in NOISE[1:]):
        noise_compare(st, REP / 'noise_compare.png')
    if 'n16' in tags:
        n16(st, REP / 'n16.png'); return
    if 'n16sel' in tags:   # retrained runs (ck_run.sbatch), the best checkpoint of the round by train_heldout or by test success
        for crit, what in [('train_heldout', 'train_heldout (100 unseen pairs of train goals)'), ('test', 'test (the 50 BC-fail test pairs; optimistic)')]:
            ste = [(t + '_ck', k, lab.split(' (')[0] + f', best ckpt by {crit}' + (' (16 iid attempts)' if k == 0 else ''), c, mk, ls) for t, k, lab, c, mk, ls in N16_ST]
            bce = [(f'{t}_sel{crit}', lab.split(' (')[0] + f', best ckpt by {crit}', c, mk, ls) for t, lab, c, mk, ls in N16_BC]
            n16(st, REP / f'n16_sel{crit}.png', ste, bce, lambda t, k, c=crit: f'bon_seeds8_n16_sel{c}_r{k}', f'\nevery policy at its best checkpoint by {what}')
        return
    if 'n16c' in tags:
        n16c(st, REP / 'n16_cont.png'); return
    if 'cont' in tags:
        cont_section(st); n16c(st, REP / 'n16_cont.png'); return   # n16_cont: BC, STk0 +- feedback, STk7
    if 'loop' in tags:
        loop_compare(st, REP / 'loop_compare.png'); return
    if 'allpairs' in tags:
        allpairs(st, REP / 'allpairs.png'); matched_table(st, REP / 'matched_table.json'); return
    if 'hidden' in tags:   # hidden goal: repo ASTEROID (expert path after own attempts) vs DAgger, + the goal-visible runs for reference
        ht = [t for t in HIDDEN if (RUNS / t / 'bon_seeds8.npz').exists() and 'train_st_success' in np.load(RUNS / t / 'bon_seeds8.npz').files]
        for t in ht[:2]:
            perf_bon(t, st, REP / f'perf_bon_{t}.png'); goal_map(t, st, REP / f'goalmap_{t}.png')
        feedback_compare(ht, st, REP / 'hidden_compare.png', 'HIDDEN goal (policy never sees it; feedback = success flag of the previous attempt) '
                         'vs goal visible (BFS + dx, dy feedback), 8 rounds × 900 steps: ASTEROID (repo: expert path after own attempts) vs '
                         'DAgger (BFS expert labels own attempt); BC sees the goal')
        allpairs(st, REP / 'allpairs_hidden.png', allp=ALLPH, bccl=[],
                 title='all held-out goals, HIDDEN goal: ASTEROID (repo) vs DAgger, with the goal-visible runs for reference (BC sees the goal)',
                 maps=[('int_s30_hg_demo', 0, 'STk0, hidden goal (round 0 = expert demos)'), ('int_s30_hg_demo', 7, 'STk7, hidden goal, ASTEROID (repo)'),
                       ('int_s30_hg_relabel', 7, 'STk7, hidden goal, DAgger')], map_bcs=[])
        matched_table(st, REP / 'hidden_table.json', bccl=[],
                      st_rows=[(t, k, f'STk{k}, {n}') for t, _, n in ALLPH for k in (0, 7)])
        return
    if 'variants' in tags:   # A* demo target (closed / open-loop obs) vs DAgger relabel, + BC retrained on the relabel run's data
        vt = [t for t in VARIANTS if (RUNS / t / 'bon_seeds8.npz').exists() and 'train_st_success' in np.load(RUNS / t / 'bon_seeds8.npz').files]
        for t in vt[1:]:
            perf_bon(t, st, REP / f'perf_bon_{t}.png'); goal_map(t, st, REP / f'goalmap_{t}.png')
        if len(vt) > 1:
            feedback_compare(vt, st, REP / 'variants_compare.png', 'BFS + dx, dy feedback, 8 rounds × 900 steps: A* demo target (closed-loop obs) vs '
                             'open-loop obs (start cell, feedback still per step) vs DAgger (the BFS expert labels the policy\'s own last attempt)')
        allpairs(st, REP / 'allpairs_variants.png', allp=ALLPV, bccl=BCCL + DAGBC,
                 title='all held-out goals: A* demo target vs open-loop obs vs DAgger relabel (8 rounds × 900 steps, BFS + dx, dy); BC on demos and on the DAgger data',
                 maps=[('int_s30_bfsdir', 7, 'STk7, A* demo target'), ('int_s30_bfsdir_ol', 7, 'STk7, open-loop obs'),
                       ('int_s30_relabel', 0, 'STk0, DAgger (round 0 = BFS expert rollouts)'), ('int_s30_relabel', 7, 'STk7, DAgger relabel')],
                 map_bcs=[(t, n) for t, n, _, _ in DAGBC if not t.endswith('h64_closed')])
        matched_table(st, REP / 'variants_table.json', bccl=BCCL + DAGBC,
                      st_rows=[(t, k, f'STk{k}, {n}') for t, _, n in ALLPV for k in (0, 7)])
        return
    if K0LONG in tags and K0LONG in done:
        goal_map_compare([(NOISE[0], 'bc', 'BC (UNet, demos only)'), (NOISE[0], 0, 'STk0, 900 steps (no noise run)'),
                          (K0LONG, 0, 'STk0, 7200 steps (round 0 only)'), (NOISE[0], 7, 'STk7, 7200 steps total (no noise run)')],
                         st, REP / 'goalmap_k0long_compare.png',
                         'BFS + dx, dy: success per goal cell, STk0 trained 900 vs 7200 steps vs STk7 (STk on attempt k+1; 8 seeds). '
                         'Black = wall; dashed = held-out goal; hatched = goal of a BC-fail test pair; grey = held-out goal not in the test set')
    logged = [t for t in tags if (RUNS / t / 'log.jsonl').exists() and t in MODES]
    if logged:
        train_curves(logged, st, REP / 'train_curves.png')


if __name__ == '__main__':
    main(sys.argv[1:] or list(FB))
