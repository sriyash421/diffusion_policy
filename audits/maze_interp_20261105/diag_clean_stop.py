"""Why the clean-image (goal visible) no-context policy fails on train starts (2026-11-23).
Rolls out the final-round policy of each tag (one maze) on train / held-out pairs under 4 test-time variants:
  as trained   sampled action, stop head on
  mean action  the Gaussian's mean instead of a sample
  no stop      stop head never fires (attempt ends at the goal or at the step budget)
  mean+no stop
and for attempts that ended by the stop head away from the goal: Euclidean and BFS (cells) distance to the goal where it stopped.
  python diag_clean_stop.py --tags c_dag_clean_noctx      (GPU; MAZE_SETTING from the run)
-> runs/<tag>/diag_stop.json
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--tags', required=True); ap.add_argument('--pairs', type=int, default=200)
ap.add_argument('--seeds', type=int, default=2); args = ap.parse_args(); TAGS = args.tags.split(',')
import run_setting
S0 = run_setting.resolve(TAGS[0]); assert all(run_setting.infer(t) == S0 for t in TAGS)
if S0 in ('interp30', 'interp30c'):
    os.environ['INTERP_TEST'] = 'all'
import rounds, asteroid
from rounds import make_policy, load_data, train_starts, test_episodes, build_context
import torch
dev = 'cuda'; torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 1)))


def patch(model, mean, nostop):
    orig = model.__class__._heads.__get__(model)
    def heads(h):
        d, s = orig(h)
        if mean:
            m = d.mean; d.sample = lambda *a, **k: m
        return d, (torch.full_like(s, -1e9) if nostop else s)
    model._heads = heads


def main():
    for tag in TAGS:
        out = rounds.HERE / 'runs' / tag; cfg = json.loads((out / 'config.json').read_text()); R = cfg['rounds']
        asteroid.apply_cfg(cfg); model, pc = make_policy(R, dev); ds = load_data(pc['shape_meta']); model.set_normalizer(ds.get_normalizer())
        model.load_state_dict(torch.load(out / f'round_{R - 1}.pt', map_location='cpu', weights_only=False)['model']); model.to(dev).eval()
        tr_s, tr_g, free_np = train_starts(ds); te = test_episodes()
        te_s = np.stack([e['start'] for e in te]).astype(np.float32); te_g = np.stack([e['goal'] for e in te]).astype(np.float32)
        mv, dist = rounds.expert_tables(free_np); guard = int(free_np.sum()); res = {}
        for split, (Ss, Gs) in (('train', (tr_s, tr_g)), ('test', (te_s, te_g))):
            idx = np.repeat(np.sort(np.random.default_rng(0).choice(len(Ss), min(args.pairs, len(Ss)), replace=False)), args.seeds)
            S, G = Ss[idx], Gs[idx]; N = len(idx); fr = torch.as_tensor(free_np, device=dev)[None].repeat(N, 1, 1)
            exp_len = np.array([len(rounds.expert_rollout(free_np, s, g)) - 1 for s, g in zip(S, G)])
            for name, mean, nostop in (('as trained', 0, 0), ('mean action', 1, 0), ('no stop', 0, 1), ('mean+no stop', 1, 1)):
                patch(model, mean, nostop); torch.manual_seed(20261123)
                t, o, n = rounds.rollout(model, fr, torch.as_tensor(S, device=dev), torch.as_tensor(G, device=dev),
                                         build_context(model, fr, torch.as_tensor(G, device=dev), [[] for _ in range(N)]))
                end = np.where(o, 0, np.where(n >= guard, 2, 1)); last = np.stack([tr[-1] for tr in t])
                eu = np.linalg.norm(last - G, axis=1); c = np.rint(last).astype(int); g = np.rint(G).astype(int)
                bfs = dist[g[:, 1], g[:, 0], c[:, 1], c[:, 0]]; st = end == 1
                r = dict(success=float(o.mean()), stopped=float(st.mean()), budget=float((end == 2).mean()),
                         stop_eu_median=float(np.median(eu[st])) if st.any() else None,
                         stop_bfs_quartiles=np.percentile(bfs[st], [25, 50, 75]).tolist() if st.any() else None,
                         stop_frac_within_2=float((eu[st] <= 2).mean()) if st.any() else None,
                         steps_success_over_expert=float(np.median(n[o] / exp_len[o])) if o.any() else None)
                res[f'{split}/{name}'] = r; print(tag, split, name, json.dumps(r), flush=True)
        (out / 'diag_stop.json').write_text(json.dumps(res, indent=1)); print('saved', out / 'diag_stop.json', flush=True)


if __name__ == '__main__':
    main()
