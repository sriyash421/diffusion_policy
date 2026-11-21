"""Simple-plots section (top of the report): discrete grid maze, continuous maze, online expert (DAgger relabel).
x = number of candidates n = 1..16. Top row: perf@n (success of candidate n alone); bottom row: best of n (the verifier, -BFS of the final
cell, picks one of the first n). ST candidates = 16 attempts in a row (attempt n sees its last min(R-1, n-1) attempts; STk0 has no
context, so its 16 are iid); BC candidates = 16 iid rollouts.
Every ST policy is at its best checkpoint of the round by train_heldout (select_ckpt.py: 100 unseen pairs of train goals; held-out goals
never used; checkpoints every 60 steps). Every BC is closed loop (horizon 4, re-planned every step, as ST), checkpoint picked by the same
rule. Columns: train starts (642), ALL held-out pairs (2870 = every start x 41 held-out goals), the held-out pairs closed-loop BC fails
(<= 1/3 of 8 seeds in bc_closed.npz; the BC curves use 128 fresh rollouts = 8 seeds x 16 candidates).
Goal maps: STk on its attempt k+1 (k own attempts in context), BC on its first rollout.
  MAZE_SETTING=interp30 python simple.py      (CPU job)  -> report/simple_<setting>_<plot>[_goals].png + simple.json
"""
import json
import numpy as np
from plots import HERE, REP, RUNS, starts, draw_goal_maps, bon, plt

CRIT = 'train_heldout'; N = 16
RAMP_T = [int(round(600 + (200 - 600) * k / 7)) for k in range(8)]   # asteroid.obs_noise_t, ramp 600 -> 200 over slots 0..7
C_BC, C_BC2, C_K0, C_NOFB, C_K7 = '#8B4513', '#56B4E9', '#0072B2', '#D55E00', '#009E73'
SETTINGS = {
    'grid': dict(name='Discrete grid maze', rounds='int_s30_bfsdir_ck', k0='int_k0long_ck', k0nofb='int_k0long_nofb_ck',
                 bc=[('BC (closed loop)', 'bc_h4_closed_seltrain_heldout', C_BC)], fail='bc_h4_closed'),
    'cont': dict(name='Continuous maze', rounds='c_s30_bfsdir_ck', k0='c_k0long_ck', k0nofb='c_k0long_nofb_ck',
                 bc=[('BC (closed loop)', 'c_bc_h4', C_BC)], fail='c_bc_h4'),
    'online': dict(name='Online expert (DAgger: the BFS expert labels the policy\'s own attempts)', rounds='int_s30_relabel_ck',
                   k0='int_k0long_relabel_ck', k0nofb='int_k0long_relabel_nofb_ck',
                   bc=[('BC on demos (closed loop)', 'bc_h4_closed_seltrain_heldout', C_BC),
                       ('BC on the DAgger data (closed loop)', 'dagbc_int_s30_relabel_h4_closed', C_BC2)], fail='bc_h4_closed'),
}
NOISE = [('flat', 'int_s30_on400_ck', 'flat noise t = 400', [400] * 8), ('ramp', 'int_s30_onramp_ck', 'ramped noise t = 600 → 200', RAMP_T)]
BC_NOISE = 'bc_h4_noise128'


def st(tag, k, noisy=False):
    """{'train', 'all'}: (S, B) as (16 candidates, pairs, seeds) for the round-k pick, or None if not evaluated yet."""
    n = '_noisyobs' if noisy else ''; out = {}
    for sp, key, x in (('train', 'train', ''), ('all', 'test', '_allpairs')):
        f = RUNS / tag / f'bon_seeds8{n}_n16_sel{CRIT}_r{k}{x}.npz'
        if f.exists():
            z = np.load(f); out[sp] = (z[f'{key}_st_success'][0], z[f'{key}_st_bfs'][0])
    return out if len(out) == 2 else None


def bc(tag, fname='bc_n16.npz', ts=None):
    """BC (S, B) as (16 candidates, pairs, seeds) from 8 seeds x 16 iid rollouts. ts = noise timestep of each candidate (None = clean)."""
    p = RUNS / tag / fname
    if not p.exists():
        return None
    z = np.load(p); out = {}
    for sp, key in (('train', 'train'), ('all', 'allpairs')):
        def get(t, what):
            a = z[f'{key}{"" if t is None else f"_t{t}"}_{what}']; return np.moveaxis(a.reshape(len(a), -1, N), 2, 0)
        if ts is None:
            out[sp] = (get(None, 'success'), get(None, 'bfs'))
        else:
            if any(f'{key}_t{t}_success' not in z for t in set(ts)):
                return None
            out[sp] = tuple(np.stack([get(ts[n], w)[n] for n in range(N)]) for w in ('success', 'bfs'))
    return out


def curves(d, m):
    """(perf@n, best of n) over the pairs in mask m (None = all)."""
    S, B = d
    if m is not None:
        S, B = S[:, m], B[:, m]
    return S.mean((1, 2)), bon(S, B, 0).mean((1, 2))


def figure(lines, cols, title, path):
    """lines: (label, data, colour, marker, ls, hollow). 2 x 3: perf@n / best of n by column set."""
    lines = [l for l in lines if l[1] is not None]; out = {}
    if not lines:
        return out
    f, a = plt.subplots(2, 3, figsize=(17, 8.6), sharey=True); n = np.arange(1, N + 1)
    for j, (sp, m, ctitle) in enumerate(cols):
        for lab, d, c, mk, ls, hollow in lines:
            pf, bt = curves(d[sp], m)
            for r, y in enumerate((pf, bt)):
                a[r, j].plot(n, y, marker=mk, ls=ls, color=c, mfc='white' if hollow else c, ms=4, lw=2.2, label=lab)
            out.setdefault(lab, {})[sp if j < 2 else 'fail'] = dict(perf1=round(float(pf[0]), 3), perf16=round(float(pf[-1]), 3),
                                                                     best16=round(float(bt[-1]), 3))
        a[0, j].set_title(f'{ctitle}\nperf@n: candidate n alone', fontsize=9.5); a[1, j].set_title('best of n with the verifier', fontsize=9.5)
        a[1, j].set_xlabel('number of candidates n')
        for ax in a[:, j]:
            ax.set_ylim(0, 1.03); ax.grid(alpha=.25); ax.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])
    a[0, 0].set_ylabel('success'); a[1, 0].set_ylabel('success'); h, l = [], []
    for ax in a.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    nc = 3 if len(l) > 4 else len(l); f.legend(h, l, loc='lower center', ncol=nc, fontsize=9, frameon=False); f.suptitle(title, fontsize=11)
    f.tight_layout(rect=(0, .03 + .035 * -(-len(l) // nc), 1, .96)); f.savefig(path, dpi=125); plt.close(f); print(path.name, 'written', flush=True)
    return out


def goalmaps(pols, S, fail, title, path):
    """pols: (label, data, attempt index). Success per goal cell at that candidate."""
    pols = [(lab, d['train'][0][i].mean(-1), d['all'][0][i].mean(-1)) for lab, d, i in pols if d is not None]
    if not pols:
        return
    P = np.load(HERE / 'bc_scan.npz')['pairs']; s2 = dict(S, test_goal=P[:, 2:].astype(float), bc_fail=fail)
    hatch = {tuple(g) for g in np.unique(P[:, 2:], axis=0) if fail[(P[:, 2:] == g).all(1)].mean() > .5}
    draw_goal_maps(pols, s2, path, title + ' — success per goal cell (STk on attempt k+1, BC on 1 rollout); red = fail, green = success; '
                   'dashed = held-out goal; hatched = most of its pairs are closed-loop BC failures', hatch=hatch, test_label='held-out')
    print(path.name, 'written', flush=True)


def fail_mask(tag, P):
    z = np.load(RUNS / tag / 'bc_closed.npz'); assert np.array_equal(z['allpairs_pairs'], P)
    return z['allpairs_success'].mean(1) <= 1 / 3


def main():
    S = starts(); out = {}; P = np.load(HERE / 'bc_scan.npz')['pairs']
    vir = plt.cm.viridis(np.linspace(0, .92, 8))
    for key, cfg in SETTINGS.items():
        fail = fail_mask(cfg['fail'], P); o = out.setdefault(key, {'n_fail': int(fail.sum())})
        cols = [('train', None, 'train starts (642)'), ('all', None, 'all held-out pairs (2870)'),
                ('all', fail, f'held-out pairs closed-loop BC fails ({fail.sum()})')]
        bcs = [(lab, bc(t), c, 's', '--', False) for lab, t, c in cfg['bc']]
        stk = [st(cfg['rounds'], k) for k in range(8)]
        k0, k0n, k7 = st(cfg['k0'], 0), st(cfg['k0nofb'], 0), stk[7]
        if any(x is not None for x in stk):   # figures whose ST runs are still training are skipped (the report shows 'running')
          # 1. STk0..7 at their best checkpoints
          o['stk'] = figure(bcs + [(f'STk{k}', stk[k], vir[k], 'o', '-', False) for k in range(8)], cols,
                          f'{cfg["name"]}: STk0..7 (8 rounds × 900 steps), each at its best checkpoint', REP / f'simple_{key}_stk.png')
          for k in range(8):
            p = RUNS / cfg['rounds'] / f'selection_r{k}.json'
            if p.exists() and f'STk{k}' in o['stk']:
                o['stk'][f'STk{k}']['update'] = json.loads(p.read_text())['pick'][CRIT]['update']
          goalmaps([(lab, d, 0) for lab, d, *_ in bcs] + [(f'STk{k}', stk[k], k) for k in range(8)], S, fail,
                 f'{cfg["name"]}: BC and STk0..7', REP / f'simple_{key}_stk_goals.png')
        # 2. same number of training steps (7200): STk0 in one round vs STk7 over 8 rounds vs BC
        L = bcs + [('STk0, 7200 steps (one round)', k0, C_K0, 'o', '-', False), ('STk7, 8 rounds × 900 = 7200 steps', k7, C_K7, '^', '-', False)]
        if k7 is not None:
          o['steps'] = figure(L, cols, f'{cfg["name"]}: same number of training steps (7200), best checkpoints', REP / f'simple_{key}_steps.png')
          goalmaps([(lab, d, 7 if 'STk7' in lab else 0) for lab, d, *_ in L], S, fail, f'{cfg["name"]}: same training steps', REP / f'simple_{key}_steps_goals.png')
        # 3. feedback: STk0 7200 steps with vs without feedback vs BC
        L = bcs + [('STk0 with feedback (BFS + dx, dy)', k0, C_K0, 'o', '-', False), ('STk0 without feedback', k0n, C_NOFB, 'v', '-', False)]
        o['fb'] = figure(L, cols, f'{cfg["name"]}: STk0 (7200 steps) with vs without feedback, best checkpoints', REP / f'simple_{key}_fb.png')
        goalmaps([(lab, d, 0) for lab, d, *_ in L], S, fail, f'{cfg["name"]}: feedback', REP / f'simple_{key}_fb_goals.png')
        # 3b. discrete grid only: STk7 (8 rounds, feedback) vs STk0 (7200 steps) with and without feedback
        if key == 'grid' and k7 is not None:
          L = bcs + [('STk7, 8 rounds × 900 steps (with feedback)', k7, C_K7, '^', '-', False),
                     ('STk0, 7200 steps, with feedback (BFS + dx, dy)', k0, C_K0, 'o', '-', False),
                     ('STk0, 7200 steps, without feedback', k0n, C_NOFB, 'v', '-', False)]
          o['k7fb'] = figure(L, cols, f'{cfg["name"]}: STk7 vs STk0 with and without feedback, best checkpoints', REP / f'simple_{key}_k7fb.png')
          goalmaps([(lab, d, 7 if 'STk7' in lab else 0) for lab, d, *_ in L], S, fail, f'{cfg["name"]}: STk7 vs STk0 with / without feedback',
                   REP / f'simple_{key}_k7fb_goals.png')
    # 4. observation noise (discrete grid): STk7 of the noise run and BC, each evaluated clean and with the noise
    fail = fail_mask('bc_h4_closed', P)
    cols = [('train', None, 'train starts (642)'), ('all', None, 'all held-out pairs (2870)'), ('all', fail, f'held-out pairs closed-loop BC fails ({fail.sum()})')]
    bcc = bc(BC_NOISE, 'bc_noisy.npz') or bc('bc_h4_closed_seltrain_heldout')
    for nk, tag, lab, ts in NOISE:
        tn = [ts[min(n, 7)] for n in range(N)]   # candidate n+1 of STk7 has min(7, n) attempts in context -> that slot's t
        L = [('STk7, clean eval', st(tag, 7), C_K7, 'o', '-', False), ('STk7, noised eval (its training noise)', st(tag, 7, True), C_K7, 'o', '--', True),
             ('BC (closed loop), clean eval', bcc, C_BC, 's', '-', False),
             ('BC (closed loop), noised eval (same t as ST\'s candidate n)', bc(BC_NOISE, 'bc_noisy.npz', tn), C_BC, 's', '--', True)]
        if L[0][1] is None:
            continue
        out.setdefault('noise', {})[nk] = figure(L, cols, f'Discrete grid maze, ST trained with {lab} on its observation features: clean vs noised '
                                                  'evaluation (best checkpoints)', REP / f'simple_noise_{nk}.png')
        goalmaps([(l, d, 7 if l.startswith('STk7') else 0) for l, d, *_ in L], S, fail, f'Discrete grid maze, {lab}', REP / f'simple_noise_{nk}_goals.png')
    (REP / 'simple.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out)[:2500])


if __name__ == '__main__':
    main()
