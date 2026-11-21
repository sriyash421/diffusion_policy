"""Report figures (CPU): maze split; per-round curves of the train_asteroid runs; per-goal success maps; same-total-steps
controls; BC vs 1 vs 20 steps. Uses whatever has been logged so far."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import plot_clear as p
R = p.HERE / 'report'; p.maze_split(R / 'maze_split.png')
RUNS = {'ast_s20': '20 gradient steps per batch', 'ast_s1': '1 gradient step per batch', 'ast_s30': '30 gradient steps per batch',
        'ast_s20_nofb': '20 gradient steps per batch, context without feedback',
        'ast_s20_on400': '20 gradient steps per batch, training obs noise flat t = 400',
        'ast_s20_onramp': '20 gradient steps per batch, training obs noise ramp t = 600 (no context) → 200 (7 attempts)',
        'ast_main_curves': '10k updates per collect-and-train cycle',
        'ast_s30_bfsdir_on400': '30 steps per batch, step feedback [−BFS, 0, dx, dy], training obs noise flat t = 400',
        'ast_s30_bfsdir_onramp': '30 steps per batch, step feedback [−BFS, 0, dx, dy], training obs noise ramp t = 600 → 200',
        'ast_main_on400': '10k updates per cycle, BFS feedback, training obs noise flat t = 400',
        'ast_main_onramp': '10k updates per cycle, BFS feedback, training obs noise ramp t = 600 → 200'}
have = {}
for t, lab in RUNS.items():
    if (p.HERE / 'runs' / t / 'log.jsonl').exists() and p.rows(t):
        done = (p.HERE / 'runs' / t / 'TRAINED').exists(); have[t] = lab
        p.round_curves(t, f'train_asteroid, {lab}' + ('' if done else '   [training in progress]'), R / f'curves_{t}.png')
if {'ast_s20', 'ast_s1'} <= set(have):
    p.compare({t: have[t] for t in ('ast_s20', 'ast_s1')}, R / 'compare.png')
ctrl = [t for t in p.CTRL if (p.HERE / 'runs' / t / 'log.jsonl').exists() and p.rows(t)]
if len(ctrl) > 1:
    p.steps_compare(ctrl, R / 'steps_compare.png')
for t in ('ast_s20', 'ast_main'):
    if (p.HERE / 'runs' / t / 'eval_seeds8.npz').exists():
        p.goal_map(t, R / f'goalmap_{t}.png')
if {'ast_s20', 'ast_s20_nofb'} <= set(have):
    p.feedback_compare(R / 'feedback_compare.png')
if (p.HERE / 'runs' / 'ast_s30' / 'bon_seeds8.npz').exists():
    p.bon_plots('ast_s30', R / 'bon_curves_ast_s30.png', R / 'bon_line_ast_s30.png', '30 gradient steps per batch')
_b = p.HERE / 'runs' / 'ast_s30' / 'bon_seeds8.npz'
if _b.exists() and 'train_st_success' in p.np.load(_b):
    p.goal_map('ast_s30', R / 'goalmap_ast_s30.png', src='bon_seeds8.npz')
_m = p.HERE / 'runs' / 'ast_main' / 'bon_seeds8.npz'
if _m.exists():
    p.bon_plots('ast_main', R / 'bon_curves_ast_main.png', R / 'bon_line_ast_main.png', '10k updates per collect-and-train cycle')
if (p.HERE / 'runs' / 'ast_s30' / 'bon_seeds8.npz').exists():
    p.perf_at_n('ast_s30', R / 'perf_curves_ast_s30.png', '30 gradient steps per batch')
if _m.exists():
    p.perf_at_n('ast_main', R / 'perf_curves_ast_main.png', '10k updates per collect-and-train cycle')
for _t, _lab in (('ast_s30', '30 gradient steps per batch'), ('ast_main', '10k updates per collect-and-train cycle')):
    if (p.HERE / 'runs' / _t / 'bon_seeds8.npz').exists():
        p.perf_bon(_t, R / f'perf_bon_{_t}.png', _lab)
if all((p.HERE / 'runs' / t / 'bon_seeds8.npz').exists() for t in ('ast_s30', 'ast_main')):
    p.bfs_per_attempt(R / 'bfs_per_attempt.png')
for _t, _lab in (('ast_s20_on400', '20 steps per batch, training obs noise flat t = 400'), ('ast_s20_onramp', '20 steps per batch, training obs noise ramp t = 600 → 200')):
    _f = p.HERE / 'runs' / _t / 'bon_seeds8.npz'
    if _f.exists() and 'train_st_success' in p.np.load(_f):
        p.perf_bon(_t, R / f'perf_bon_{_t}.png', _lab)
for _t, _lab in (('ast_s20_on400', '20 steps per batch, obs noise flat t = 400, eval observations noised as in training'),
                 ('ast_s20_onramp', '20 steps per batch, obs noise ramp t = 600 → 200, eval observations noised as in training')):
    _f = p.HERE / 'runs' / _t / 'bon_seeds8_noisyobs.npz'
    if _f.exists() and 'train_st_success' in p.np.load(_f):
        p.perf_bon(_t, R / f'perf_bon_{_t}_noisyobs.png', _lab, src='bon_seeds8_noisyobs.npz')
def _eval_src(t, base):
    """Final eval file if it has both splits, else the --partial one (saved cycles only, training still running)."""
    for name in (base + '.npz', base + '_partial.npz'):
        f = p.HERE / 'runs' / t / name
        if f.exists():
            z = p.np.load(f)
            if 'test_st_success' in z and (name.endswith('_partial.npz') or 'train_st_success' in z):
                note = f' [cycles 0–{int(z["cycles_evaluated"]) - 1}, training in progress]' if name.endswith('_partial.npz') else ''
                return name, note + ('' if 'train_st_success' in z else ' [train split eval running]')
    return None, ''
for _t in ('ast_s30_bfsdir_on400', 'ast_s30_bfsdir_onramp', 'ast_main_on400', 'ast_main_onramp'):
    for _base, _suf, _ev in (('bon_seeds8', '', 'clean eval'), ('bon_seeds8_noisyobs', '_noisyobs', 'eval observations noised as in training')):
        _src, _note = _eval_src(_t, _base)
        if _src:
            p.perf_bon(_t, R / f'perf_bon_{_t}{_suf}.png', RUNS[_t] + ', ' + _ev + _note, src=_src)
            if 'train_st_success' in p.np.load(p.HERE / 'runs' / _t / _src):
                p.goal_map(_t, R / f'goalmap_{_t}{_suf}.png', src=_src)
