"""Save the (start, goal) pairs of each eval_ctx split, in eval_ctx's order, for the position plots (plots_positions.py).
  MAZE_SETTING=interp30c INTERP_TEST=all python dump_pairs.py      (CPU)  -> report/pairs_interp30c.npz
  MAZE_SETTING=11brc python dump_pairs.py                                  -> report/pairs_11brc.npz
"""
import os
from pathlib import Path
import numpy as np
import rounds

REP = Path(__file__).resolve().parent / 'report'
_, pc = rounds.make_policy(8, 'cpu'); ds = rounds.load_data(pc['shape_meta'])
tr_s, tr_g, free = rounds.train_starts(ds)
te = rounds.test_episodes(); th = rounds.heldout_train_episodes()
te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
th_s = np.stack([e['start'] for e in th]).astype(np.float32); th_g = np.stack([e['goal'] for e in th]).astype(np.float32)
out = REP / f'pairs_{rounds.SETTING}.npz'
np.savez(out, free=free, train_start=tr_s, train_goal=tr_g, train_heldout_start=th_s, train_heldout_goal=th_g, test_start=te_s, test_goal=te_g)
print('saved', out, {k: len(v) for k, v in dict(train=tr_s, train_heldout=th_s, test=te_s).items()}, 'INTERP_TEST', os.environ.get('INTERP_TEST'))
