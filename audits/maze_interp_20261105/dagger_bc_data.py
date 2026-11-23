"""BC baseline data for an ASTEROID / DAgger relabel run (asteroid.py --target relabel): the BFS expert's demos from every state the
learner visited. For each item of runs/<tag>/state.pt (aggregated over all rounds):
  round 0 (expert rollout from a training start):        that expert path, one episode (as the original demos)
  round n >= 1 (the learner's attempt n+1, relabelled):  for every cell the learner acted from, the BFS expert's path from that cell
                                                         to the goal, one episode each (the expert's chunk labels for that state)
The context attempts are not used (BC has no context). Written in the interp30 demo zarr layout -> runs/<tag>/bc_data.zarr, with
meta start_xy = the episode's first cell, goal_xy = its goal. Only training goals appear (the learner only rolls out training starts).

  python dagger_bc_data.py --tag int_s30_relabel [--rounds 0,1,...]      (CPU)
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, '/gscratch/robotics/harine/codex_audits/maze_rounds_20261103')
import run_setting; run_setting.resolve(run_setting.tag_from_argv(sys.argv), allowed=('interp30',))   # DISCRETE only (grid BFS demos)
os.environ.setdefault('MAZE_SETTING', 'interp30')
sys.path.insert(0, '/gscratch/robotics/harine/codex_audits/maze_rounds_20261103')
import numpy as np
import torch
import zarr
from numcodecs import Blosc
import rounds
from rounds import maze, expert_rollout, SPEC

I = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--out', default='bc_data.zarr')
    args = ap.parse_args(); run = I / 'runs' / args.tag; cfg = json.loads((run / 'config.json').read_text())
    assert cfg.get('target') == 'relabel', 'not a relabel run'
    st = torch.load(run / 'state.pt', map_location='cpu', weights_only=False); items = st['items']
    print('items', len(items), 'trained through round', st['round'], flush=True)
    src = zarr.open(SPEC['dataset'], 'r'); free = np.asarray(src['data/obs/image'][0, :, :, 0]) < 128     # wall plane 255 = wall
    z0 = rounds.load_data(json.loads((rounds.ST_RUN / 'config.json').read_text())['policy']['shape_meta'])
    S0, G0, free_np = rounds.train_starts(z0); assert np.array_equal(free, free_np)
    cache = {}
    def path(c, g):
        k = (int(c[0]), int(c[1]), int(g[0]), int(g[1]))
        if k not in cache:
            cache[k] = expert_rollout(free_np, c, g).astype(np.int64)
        return cache[k]
    eps, n_rel, n_exp = [], 0, 0
    for it in items:
        assert len(it) == 3, 'demo item in a relabel run'
        i, ctx, lab = it; g = np.rint(G0[i]).astype(int); lab = np.asarray(lab).astype(int)
        if len(ctx) == 0 and np.array_equal(lab, path(lab[0], g)):   # round 0: the expert's own rollout
            eps.append((lab, g)); n_exp += 1
        else:
            for c in lab[:-1]:
                eps.append((path(c, g), g)); n_rel += 1
    print(f'episodes: {n_exp} expert rollouts + {n_rel} relabelled learner states; unique (cell, goal) {len(cache)}', flush=True)
    L = np.array([len(p) - 1 for p, _ in eps]); ends = np.cumsum(L); N = int(ends[-1])
    img = np.zeros((N, *free.shape, 3), np.uint8); pos = np.zeros((N, 2), np.float32); gp = np.zeros((N, 2), np.float32)
    act = np.zeros((N, 2), np.float32); rew = np.zeros(N, np.float32); done = np.zeros(N, bool)
    nav = free.astype(np.uint8); o = 0
    for (p, g), l in zip(eps, L):
        sl = slice(o, o + l)
        img[sl] = np.stack([maze.build_image(nav, q, g) for q in p[:-1]]); pos[sl] = p[:-1]; gp[sl] = g; act[sl] = np.diff(p, axis=0)
        rew[o + l - 1] = 1; done[o + l - 1] = True; o += l
    assert np.abs(act).sum(-1).max() == 1 and np.abs(act).sum(-1).min() == 1
    # the demo zarr's first image equals build_image (the same planes / values)
    assert np.array_equal(np.asarray(src['data/obs/image'][0]), maze.build_image(nav, np.asarray(src['data/obs/agent_pos'][0]).astype(int),
                                                                                  np.asarray(src['data/obs/goal_pos'][0]).astype(int)))
    out = run / args.out; root = zarr.open(str(out), 'w'); comp = Blosc(cname='lz4', clevel=5, shuffle=Blosc.NOSHUFFLE)
    d = root.create_group('data'); ob = d.create_group('obs')
    ob.array('image', img, chunks=(min(N, 5510), *img.shape[1:]), compressor=comp); ob.array('agent_pos', pos, chunks=pos.shape, compressor=comp)
    ob.array('goal_pos', gp, chunks=gp.shape, compressor=comp)
    d.array('actions', act, chunks=act.shape); d.array('reward', rew, chunks=rew.shape); d.array('done', done, chunks=done.shape)
    d.array('expert_mask', np.ones((N, 1), np.float32), chunks=(N, 1))
    m = root.create_group('meta'); m.array('episode_ends', ends.astype(np.int64))
    m.array('start_xy', np.stack([p[0] for p, _ in eps]).astype(np.int32)); m.array('goal_xy', np.stack([g for _, g in eps]).astype(np.int32))
    m.attrs.update(dict(src['meta'].attrs))
    m.attrs['dagger_source'] = str(run); m.attrs['expert_rollouts'] = n_exp; m.attrs['relabelled_states'] = n_rel
    held = set(map(tuple, json.loads((I / 'goals.json').read_text())['test']))
    assert not any(tuple(g) in held for _, g in eps), 'a held-out goal in the BC data'
    print(f'wrote {out}: {len(eps)} episodes, {N} frames, mean length {L.mean():.1f}', flush=True)


if __name__ == '__main__':
    main()
