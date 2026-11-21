"""INTERIM grid figure while the best-checkpoint STk7 eval runs: STk7 = LAST saved checkpoint (round_7.pt) of the original 8-round run
int_s30_bfsdir (same config as int_s30_bfsdir_ck), 16 attempts on train starts + all 2870 held-out pairs; STk0 lines and BC as in simple.py.
  MAZE_SETTING=interp30 python simple_k7fb_interim.py   (CPU job) -> report/simple_grid_k7fb_interim[_goals].png"""
import json
import numpy as np
import simple as s
from plots import HERE, REP, RUNS, starts


def last_k7():
    out = {}
    for sp, key, x in (('train', 'train', ''), ('all', 'test', '_allpairs')):
        z = np.load(RUNS / 'int_s30_bfsdir' / f'bon_seeds8_n16_r7{x}.npz'); out[sp] = (z[f'{key}_st_success'][0], z[f'{key}_st_bfs'][0])
    return out


cfg = s.SETTINGS['grid']; P = np.load(HERE / 'bc_scan.npz')['pairs']; fail = s.fail_mask(cfg['fail'], P)
cols = [('train', None, 'train starts (642)'), ('all', None, 'all held-out pairs (2870)'), ('all', fail, f'held-out pairs closed-loop BC fails ({fail.sum()})')]
k7 = last_k7(); print({sp: v[0].shape for sp, v in k7.items()})
L = [(lab, s.bc(t), c, 's', '--', False) for lab, t, c in cfg['bc']] + [
    ('STk7, 8 rounds × 900 steps (with feedback), LAST checkpoint', k7, s.C_K7, '^', '-', False),
    ('STk0, 7200 steps, with feedback (BFS + dx, dy), best checkpoint', s.st(cfg['k0'], 0), s.C_K0, 'o', '-', False),
    ('STk0, 7200 steps, without feedback, best checkpoint', s.st(cfg['k0nofb'], 0), s.C_NOFB, 'v', '-', False)]
o = s.figure(L, cols, 'INTERIM — Discrete grid maze: STk7 (last checkpoint; best-checkpoint eval running) vs STk0 with and without feedback',
             REP / 'simple_grid_k7fb_interim.png')
s.goalmaps([(lab, d, 7 if 'STk7' in lab else 0) for lab, d, *_ in L], starts(), fail, 'INTERIM — Discrete grid maze: STk7 (last ckpt) vs STk0 with / without feedback',
           REP / 'simple_grid_k7fb_interim_goals.png')
print(json.dumps(o, indent=1))
