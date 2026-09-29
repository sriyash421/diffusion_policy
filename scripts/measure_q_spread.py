"""What scale is the learned Q on, and how far apart are the candidates of one decision?

THE FINDING, measured over 512 real decisions on blq137 and brd100 at k=16:

    Q runs 0.34 .. 0.93, mean 0.77          -- a discounted sparse-reward success probability,
    within-decision spread  0.0011 mean     -- three orders of magnitude below that range
                            0.0005 median
    exact ties              0.0             -- candidates are always separated, just barely

That spread is why `PushTSearchMixin._normalize_q` exists. Concatenated raw into
`action_value_emb` beside an O(1) embedding, a signal of 0.001 is a dead input: it would
train, teach nothing, and no loss or metric would report it. The running normalizer turns it
into a z-score instead. Nothing here is a constant the runtime reads -- this script is the
evidence for that design, re-runnable, and its output is analysis/q_spread.json.

MEASURED WITH A t_goal-TRAINED POLICY, and it has to be: the Q-trained arms do not exist yet,
so the candidate distribution comes from an arm trained under a different verifier. Fine for
establishing the scale; re-run it on a Q-trained arm before drawing a conclusion about how
the Q separates ITS OWN candidates.

    python scripts/measure_q_spread.py --checkpoint <t_goal arm>/checkpoints/step_0100000.ckpt

Needs a GPU. Run it on a compute node -- the login node is one shared ~10GB cgroup.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_verifier import q_verifier_spec


DEFAULT_CKPTS = (
    'value_k16_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42',
    'value_k16_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42',
)


def measure(checkpoint, value, n, windows, batch, device, seed):
    """-> (per-decision stds (W,), raw scores (W, n))."""
    import hydra
    from torch.utils.data.dataloader import default_collate

    from eval_search_pusht import load_policy
    from sac.score import PushTQVerifier

    policy, cfg = load_policy(checkpoint, device)
    policy.eval()

    ckpt_path, rung = q_verifier_spec(value)
    q = PushTQVerifier(ckpt_path, rung=rung, value_fn=value, device=device)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=min(windows, len(dataset)), replace=False)

    scores = list()
    for start in range(0, len(idxs), batch):
        chunk = idxs[start:start + batch]
        obs = default_collate([dataset[int(i)] for i in chunk])['obs']
        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
        # no_grad EXPLICITLY: `search_candidates` does not wrap itself (its training caller
        # `generate_search_context` does), so calling it directly keeps a graph alive across
        # all n candidate samples -- ~22GB and an OOM at n=16, batch 32.
        #
        # return_scores gives the RAW ranking scalar, (B, n) -- not the context copy, which
        # is the thing this constant is used to build and so cannot define it.
        with torch.no_grad():
            out = policy.search_candidates(obs, verifier=q, n_actions=n, return_scores=True)
        scores.append(out[2].float().cpu().numpy())
        print(f'  {start + len(chunk)}/{len(idxs)} windows', flush=True)

    scores = np.concatenate(scores, axis=0)                 # (W, n)
    return scores.std(axis=1, ddof=0), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--checkpoint', action='append', default=None,
                    help='repeatable; defaults to the blq137 and brd100 k16 t_goal arms')
    ap.add_argument('--value', default='q_sac_all', help='a pusht_verifier.Q_VERIFIERS name')
    ap.add_argument('--n', type=int, default=16, help='candidates per decision')
    ap.add_argument('--windows', type=int, default=256, help='dataset windows per arm')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('-d', '--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('-o', '--out', default='analysis/q_spread.json')
    args = ap.parse_args()

    ckpts = args.checkpoint
    if not ckpts:
        root = pathlib.Path('ckpts/pusht_search/pusht_image_search_imgonly/outer_inner')
        ckpts = [str(root / r / 'checkpoints' / 'step_0100000.ckpt') for r in DEFAULT_CKPTS]

    report = {'value': args.value, 'n': args.n, 'windows': args.windows,
              'seed': args.seed, 'arms': {}}
    pooled = list()
    for ck in ckpts:
        print(f'[{ck}]', flush=True)
        stds, scores = measure(ck, args.value, args.n, args.windows, args.batch,
                               args.device, args.seed)
        pooled.append(stds)
        report['arms'][ck] = {
            'mean_within_decision_std': float(stds.mean()),
            'median_within_decision_std': float(np.median(stds)),
            'p10_within_decision_std': float(np.percentile(stds, 10)),
            'frac_decisions_tied': float((stds == 0).mean()),
            'score_mean': float(scores.mean()),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'n_decisions': int(len(stds)),
        }
        print(json.dumps(report['arms'][ck], indent=2), flush=True)

    allstds = np.concatenate(pooled)
    report['pooled_mean_within_decision_std'] = float(allstds.mean())
    report['pooled_median_within_decision_std'] = float(np.median(allstds))
    report['n_decisions'] = int(len(allstds))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f'\nQ_CONTEXT_SPREAD = {report["pooled_mean_within_decision_std"]:.4g}'
          f'   (pooled mean over {report["n_decisions"]} decisions)')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
