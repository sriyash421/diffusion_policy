"""Interpolation split on the 11x11 layout 0: 30 random train goal cells, the other 41 held out (goal_split 'random30').

Writes goals.json, the 800-episode BFS demo zarr, the 'interp30' entry of maze_quadrants_delta_20261019/datasets.json,
split_checks.json and maze_split.png. setup_data.goal_pool reads goals.json for 'random30'.
  python setup_interp.py
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; DELTA = HERE.parent / 'maze_quadrants_delta_20261019'
sys.path[:0] = [str(DELTA)]
from setup_data import ROOT, maze   # noqa: E402  (patches maze.goal_pool)

LAYOUT = ROOT / 'data/procgen_mazes/layouts.npz'; DATA = HERE / 'datasets/interp30/data.zarr'; N_TRAIN = 30


def main():
    import zarr
    nav = np.load(LAYOUT)['nav'][0]; free = np.argwhere(nav)[:, [1, 0]].astype(int)        # (x, y)
    gf = HERE / 'goals.json'
    if not gf.exists():
        idx = np.random.default_rng(0).permutation(len(free))
        tr, te = sorted(map(tuple, free[idx[:N_TRAIN]].tolist())), sorted(map(tuple, free[idx[N_TRAIN:]].tolist()))
        gf.write_text(json.dumps(dict(seed=0, layout=str(LAYOUT), layout_index=0, train=tr, test=te)))
    g = json.loads(gf.read_text()); tr, te = set(map(tuple, g['train'])), set(map(tuple, g['test']))
    assert len(tr) == N_TRAIN and len(te) == len(free) - N_TRAIN and not tr & te and tr | te == set(map(tuple, free.tolist()))
    assert set(map(tuple, maze.goal_pool(nav, 'random30', 'train').tolist())) == tr
    assert set(map(tuple, maze.goal_pool(nav, 'random30', 'test').tolist())) == te
    if not DATA.exists():
        info = maze.make_dataset(maze.MazeEnv(str(LAYOUT), seed=0, split='train', layout_index=0, goal_split='random30'), str(DATA), 800, 0)
        print(json.dumps(info))
    z = zarr.open(str(DATA), 'r'); ends = z['meta/episode_ends'][:]; lengths = np.diff(np.r_[0, ends])
    starts = z['meta/start_xy'][:]; goals = z['meta/goal_xy'][:]
    assert len(ends) == 800 and lengths.max() <= 64 and set(map(tuple, goals.tolist())) <= tr
    # the 80/20 pair split of Sim2RealImageDataset(val_split='pair', seed 0), as setup_data.prepare checks it
    ps = np.concatenate([starts, goals], 1).astype(int); uniq, inv = np.unique(ps, axis=0, return_inverse=True)
    chosen = np.random.default_rng(0).choice(len(uniq), size=round(.2 * len(uniq)), replace=False); mask = np.isin(inv, chosen)
    checks = dict(train_goal_cells=len(tr), test_goal_cells=len(te), episodes=800, unique_pairs=len(uniq),
                  train_episodes=int((~mask).sum()), validation_episodes=int(mask.sum()),
                  train_goals_used=len(set(map(tuple, goals.tolist()))), expert_len_mean=float(lengths.mean()),
                  expert_len_max=int(lengths.max()))
    (HERE / 'split_checks.json').write_text(json.dumps(checks, indent=1)); print(json.dumps(checks))
    dj = DELTA / 'datasets.json'; specs = json.loads(dj.read_text())
    specs['interp30'] = dict(layouts=str(LAYOUT), dataset=str(DATA), goal_split='random30', shape=list(nav.shape), horizon=64,
                             expert_max=int(lengths.max()), diameter=32, free_cells=int(nav.sum()), dataset_seed=0, split_seed=0, episodes=800)
    dj.write_text(json.dumps(specs, indent=2))
    split_fig(nav, tr, te)


def split_fig(nav, tr, te, path=HERE / 'report' / 'maze_split.png'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    H, W = nav.shape; fig, ax = plt.subplots(figsize=(5.2, 5.6))
    img = np.where(nav[..., None] > 0, 1., 0.) * np.ones(3); ax.imshow(img, interpolation='nearest')
    for (x, y) in tr:
        ax.add_patch(Rectangle((x - .5, y - .5), 1, 1, color='#0072B2', alpha=.85)); ax.text(x, y, 'T', ha='center', va='center', color='w', fontsize=8)
    for (x, y) in te:
        ax.add_patch(Rectangle((x - .45, y - .45), .9, .9, fill=False, ec='#D55E00', lw=1.8, ls='--'))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'interpolation split: {len(tr)} train goals (blue, T), {len(te)} held-out goals (dashed orange)\n'
                 'black = wall; starts: any free cell', fontsize=8)
    path.parent.mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig); print('split figure', path)


if __name__ == '__main__':
    main()
