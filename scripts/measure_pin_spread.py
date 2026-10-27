"""Within-decision spread of the Veritas pin cost and of T-to-goal, on dataset windows.

The two constants of `wp_v3_la_tgn` come from here: PIN_COST_SPREAD (the pin cost's std
across the n candidates of one decision, the role T_GOAL_SPREAD = 13.6 plays for t_goal)
and the value's mean magnitude (the role ARM_TN_CONTEXT_SCALE plays for armTn's context).
Same protocol as measure_q_spread.py: random training windows, n candidates each, the
candidates' reached states scored. Both spreads are measured on the SAME decisions, so
their ratio is what weights the two terms; t_goal's is reported so it can be checked
against 13.6.

Candidates come from the checkpoint's own search: a UNet BC arm draws them i.i.d.; an ST
arm conditions each on the previous ones, so it is steered by its OWN trained value
(``--steer``, default: the checkpoint's) and the candidates are scored under wp_v3_la by a
shadow verifier -- the renderer's --shadow-values mechanism -- so the generator never sees
a context it was not trained on.

    python scripts/measure_pin_spread.py -c <bc run>/checkpoints/step_0040000.ckpt [-c ...]
    python scripts/measure_pin_spread.py -c <st run>/checkpoints/step_0100000.ckpt --steer native

Needs a GPU.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply


def measure(checkpoint, n, windows, batch, device, seed, steer):
    """-> (W, n, 3) raw-px terms [d_T->goal, d_arm->T, pin cost]."""
    import hydra
    from torch.utils.data.dataloader import default_collate

    from eval_search_pusht import load_policy, override_verifier_value, _verifier_value_of
    from diffusion_policy.env.pusht.pusht_verifier import is_waypoint_value
    from diffusion_policy.env.pusht.veritas.tracker import (
        TrackerSnapshotStore, WAYPOINT_PLAN_DIRS, load_split_plans)

    policy, cfg = load_policy(checkpoint, device)
    policy.eval()
    steer = _verifier_value_of(policy) if steer == 'native' else steer
    shadow = None
    if steer == 'wp_v3_la':
        override_verifier_value(policy, 'wp_v3_la')
    else:
        override_verifier_value(policy, steer)
        # scored under wp_v3_la on its own sim pool; the steering verifier keeps the
        # policy's trackers (same pattern as render_search_videos --shadow-values)
        shadow = policy._build_verifier(**{**policy.search_kwargs, 'verifier_value': 'wp_v3_la'})
    print(f'steer={steer}  score=wp_v3_la{" (shadow)" if shadow else ""}')

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    eps = [int(e) for e in np.nonzero(dataset.train_pool)[0]]
    plans = dict(zip(eps, load_split_plans(WAYPOINT_PLAN_DIRS['wp_v3_la'], eps, 'train',
                                           'wp_v3_la')))
    store = TrackerSnapshotStore.from_replay_buffer(dataset.replay_buffer, plans)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(windows, len(dataset)), replace=False)
    out = []
    for start in range(0, len(idxs), batch):
        chunk = [int(i) for i in idxs[start:start + batch]]
        obs = default_collate([dataset[i] for i in chunk])['obs']
        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
        trackers = [store.for_dataset_index(dataset, i) for i in chunk]
        if is_waypoint_value(steer):
            policy.set_waypoint_trackers(trackers)
        with torch.no_grad():
            res = policy.search_candidates(obs, verifier=policy.verifier, n_actions=n,
                                           return_scores=True, return_terms=True)
            if shadow is None:
                terms = res[-1]                                              # (B, n, 3)
            else:
                shadow.set_trackers(trackers)
                actions = res[0]                                             # (B, n, H, Da)
                cols = []
                for k in range(n):
                    _, value, _, t2 = policy._score_candidates(shadow, obs, actions[:, k])
                    # the pin cost as the third raw-px term, as _score_candidates appends
                    # it when the POLICY's own value is a lookahead one (value == -cost)
                    cols.append(torch.cat([t2, -value.unsqueeze(-1).to(t2)], dim=-1))
                terms = torch.stack(cols, dim=1)                             # (B, n, 3)
        out.append(terms.double().cpu().numpy())
        print(f'  {start + len(chunk)}/{len(idxs)} windows', flush=True)
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--checkpoint', action='append', required=True)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--windows', type=int, default=256)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('-d', '--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--steer', default='wp_v3_la',
                    help="value that generates the candidates: 'native' = the checkpoint's own")
    ap.add_argument('-o', '--out', required=True)
    args = ap.parse_args()

    report = {'n': args.n, 'windows': args.windows, 'seed': args.seed, 'steer': args.steer, 'arms': {}}
    pooled = []
    for ck in args.checkpoint:
        print(f'[{ck}]', flush=True)
        terms = measure(ck, args.n, args.windows, args.batch, args.device, args.seed, args.steer)
        pooled.append(terms)
        sd = terms.std(axis=1)                                       # (W, 3)
        report['arms'][ck] = {
            'n_decisions': int(len(terms)),
            'mean_sd_t_goal': float(sd[:, 0].mean()), 'mean_sd_pin': float(sd[:, 2].mean()),
            'median_sd_t_goal': float(np.median(sd[:, 0])), 'median_sd_pin': float(np.median(sd[:, 2])),
            'mean_t_goal': float(terms[..., 0].mean()), 'mean_pin': float(terms[..., 2].mean()),
        }
        print(json.dumps(report['arms'][ck], indent=2), flush=True)

    t = np.concatenate(pooled, axis=0)
    sd = t.std(axis=1)
    report['pooled'] = {
        'n_decisions': int(len(t)),
        'mean_sd_t_goal': float(sd[:, 0].mean()), 'mean_sd_pin': float(sd[:, 2].mean()),
        'mean_t_goal': float(t[..., 0].mean()), 'mean_pin': float(t[..., 2].mean()),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['pooled'], indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
