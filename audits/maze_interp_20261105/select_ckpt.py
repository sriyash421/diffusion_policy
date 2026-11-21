"""Checkpoint selection for ST runs trained with --save-every: every saved weight of round k (ckpt/round_<k>_u*.pt and the end of the
round, round_<k>.pt) makes k own attempts and then attempt k+1 (= how STk is scored) on
  train_heldout: 100 unseen (start, train goal) pairs (the BC selection set; held-out goals never used), --seeds seeds
  test:          the 50 test pairs (BC-fail), --seeds seeds
and is picked by each (max success of attempt k+1; ties -> the later update).  -> runs/<tag>/selection_r<k>.json
  MAZE_SETTING=interp30 python select_ckpt.py --tag int_k0long_ck --rounds 0
"""
import argparse, json, os, sys
from pathlib import Path
os.environ.setdefault('MAZE_SETTING', 'interp30')
sys.path.insert(0, '/gscratch/robotics/harine/codex_audits/maze_rounds_20261103')
import numpy as np
import torch
import rounds
from rounds import SPEC, maze, make_policy, load_data, train_starts, build_episodes
from asteroid import chain, test_set, set_fb, drop_context_feedback

I = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--rounds', default='last')
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--chunk', type=int, default=512)
    a = ap.parse_args(); device = 'cuda'; torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))
    out = I / 'runs' / a.tag; cfg = json.loads((out / 'config.json').read_text()); set_fb(cfg.get('fb'), cfg.get('open_loop'), cfg.get('hide_goal'))
    if cfg.get('no_ctx_feedback'):
        drop_context_feedback()
    R = cfg['rounds']; W = R - 1; model, pc = make_policy(R, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer()); model.to(device).eval()
    _, _, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]
    _, te_s, te_g = test_set(free_np)
    th = build_episodes(SPEC['layouts'], 'train_heldout', 0, 100, SPEC['goal_split'], 0, maze.load_train_pairs(SPEC['dataset']))
    sets = {'train_heldout': (np.stack([e['start'] for e in th]).astype(np.float32), np.stack([e['goal'] for e in th]).astype(np.float32)),
            'test': (te_s, te_g)}
    ks = [R - 1] if a.rounds == 'last' else list(range(R)) if a.rounds == 'all' else [int(k) for k in a.rounds.split(',')]
    upd = cfg['iters'] * cfg['steps_per_iter']
    for k in ks:
        files = sorted((out / 'ckpt').glob(f'round_{k}_u*.pt'), key=lambda p: int(p.stem.split('_u')[1]))
        cands = [(int(p.stem.split('_u')[1]), str(p.relative_to(out))) for p in files] + [(upd, f'round_{k}.pt')]
        by = {}
        for u, rel in cands:
            model.load_state_dict(torch.load(out / rel, map_location='cpu', weights_only=False)['model']); model.to(device).eval(); model._feat_std = None
            r = dict(update=u, file=rel)
            for name, (S, G) in sets.items():
                torch.manual_seed(20261117)
                with torch.no_grad():
                    _, succ, _ = chain(model, free1, np.repeat(S, a.seeds, 0), np.repeat(G, a.seeds, 0), k + 1, W, a.chunk, device)
                r[name] = round(float(succ[k].mean()), 4)
            by[u] = r; print(k, u, r['train_heldout'], r['test'], flush=True)
        pick = {c: by[max(by, key=lambda u: (by[u][c], u))] for c in sets}
        (out / f'selection_r{k}.json').write_text(json.dumps(dict(
            round=k, rule='max success of attempt k+1 (k own attempts in context), ties -> later update', seeds=a.seeds,
            sets=dict(train_heldout='100 unseen pairs of train goals', test='the 50 test pairs (BC-fail)'),
            pick=pick, by_update=list(by.values())), indent=1))
        print('round', k, 'picks', {c: (p['update'], p['train_heldout'], p['test']) for c, p in pick.items()}, flush=True)


if __name__ == '__main__':
    main()
