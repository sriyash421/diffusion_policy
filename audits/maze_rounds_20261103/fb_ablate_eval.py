"""One set of weights (STk of a run trained WITH feedback), evaluated twice on the same episodes and seeds:
  fb    the context carries the feedback it was trained with;
  zero  the same weights, but every feedback channel of the earlier attempts in context is zero (drop_context_feedback: the
        per-step -BFS and the end-of-attempt [-BFS, -BFS, dx, dy]); their cells and moves stay, the current attempt keeps its
        own per-step -BFS.
A attempts in a row per (start, goal, seed); attempt j has attempts 1..j-1 in context. Saves success and the verifier value
(-BFS of the final cell) per attempt: runs/<tag>/fb_ablate_r<round>.npz with {mode}_{split}_success / _bfs (A, starts, seeds).
  python fb_ablate_eval.py --tag ast_s20 --round 7
"""
import argparse, json
import numpy as np
import torch
from rounds import HERE, make_policy, load_data, train_starts
from asteroid import chain, test_set, drop_context_feedback
from bon_eval import bfs_of


def main(args):
    device = 'cuda'; out = HERE / 'runs' / args.tag; cfg = json.loads((out / 'config.json').read_text())
    assert not cfg.get('no_ctx_feedback'), 'pick a run trained with feedback'
    R = cfg['rounds']; A = args.attempts; W = R - 1; model, pc = make_policy(R, device)
    ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
    model.load_state_dict(torch.load(out / f'round_{args.round}.pt', map_location='cpu', weights_only=False)['model']); model.to(device).eval()
    tr_s, tr_g, free_np = train_starts(ds); free1 = torch.as_tensor(free_np, device=device)[None]; free = free_np.astype(bool)
    te, te_s, te_g = test_set(free_np); cache = {}; res = dict(round=args.round, attempts=A, seeds=args.seeds)
    for mode in ('fb', 'zero'):                       # 'zero' patches build_context for the rest of the process, so it runs second
        if mode == 'zero':
            drop_context_feedback()
        for split, (S0, G0) in {'test': (te_s, te_g), 'train': (tr_s, tr_g)}.items():
            if split not in args.splits.split(','):
                continue
            torch.manual_seed(20261105)                # same sampling noise for both modes
            S = np.repeat(S0, args.seeds, 0); G = np.repeat(G0, args.seeds, 0)
            with torch.no_grad():
                mem, succ, _ = chain(model, free1, S, G, A, W, args.chunk, device)
            bfs = np.stack([bfs_of(free, G, [mem[r][a][-1] for r in range(len(S))], cache) for a in range(A)])
            res[f'{mode}_{split}_success'] = succ.reshape(A, len(S0), args.seeds); res[f'{mode}_{split}_bfs'] = bfs.reshape(A, len(S0), args.seeds)
            print(mode, split, 'success by attempt', succ.mean(1).round(3).tolist(), flush=True)
            np.savez_compressed(out / f'fb_ablate_r{args.round}.npz', **res)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', default='ast_s20'); ap.add_argument('--round', type=int, default=7)
    ap.add_argument('--seeds', type=int, default=8); ap.add_argument('--attempts', type=int, default=8)
    ap.add_argument('--chunk', type=int, default=512); ap.add_argument('--splits', default='train,test')
    main(ap.parse_args())
