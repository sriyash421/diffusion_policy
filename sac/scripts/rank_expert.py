"""EVAL TEST 2 -- where does the expert action a* rank, and does Q rank it where t_goal cannot?

Best-of-n is only as good as the thing doing the ranking, and the only ground truth available
for "good action" is the recorded demonstration. At a demo timestep the expert chunk a* is well
defined -- which is why the measurement is taken here and not on a policy rollout, where the
state has drifted and there is no expert action for it.

THE POINT OF THIS SCRIPT over scripts/verifier_ranks_expert.py, which already does the above for
the sim heuristic: both verifiers are run on **the same decisions, the same candidates and the
same a***, and the results are split by the existing blind/partial/informative classes. The
headline is the BLIND column -- the 28-35% of decisions (68-73% during approach) where
`value_t_goal` returns the identical number for every candidate and argmax is a tie-break. A Q
that merely matches the heuristic overall, but ranks a* well on those, is already the win.

    python sac/scripts/rank_expert.py -c <st_or_bc.ckpt> --q logs/sac/keypoint/<run>/model.zip
"""

import json
import pathlib
import sys

import click
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from diffusion_policy.common.replay_buffer import ReplayBuffer                # noqa: E402
from diffusion_policy.env.pusht.feedback_util import t_goal_distance          # noqa: E402
from eval_search_pusht import get_split_states, load_policy                   # noqa: E402
from sac.score import PushTQVerifier                                          # noqa: E402
from scripts.verifier_ranks_expert import (BLIND, CLASSES, INFORMATIVE,       # noqa: E402
                                           T_GOAL, _stats, build_batch, classify,
                                           sample_points)


def _q_scores(q, obs, chunks, To, Ta):
    """Q on the EXECUTED window of each candidate -- the window `_verifier_inputs` slices.

    Scoring the whole horizon instead would score actions the policy never executes, and
    scoring `[:Ta]` would replay one already-past step and miss the last executed one.
    """
    now = {k: v[:, To - 1:To] for k, v in obs.items()}
    flat = chunks.reshape(-1, *chunks.shape[-2:])[:, To - 1:To - 1 + Ta]
    rep = {k: v.repeat_interleave(chunks.shape[1], dim=0) for k, v in now.items()} \
        if chunks.ndim == 4 else now
    return q.get_value(rep, flat).float().cpu().numpy().reshape(chunks.shape[:-2])


@click.command()
@click.option('-c', '--checkpoint', required=True, help='the ST or BC checkpoint')
@click.option('--q', 'q_ckpt', required=True, help='the SAC checkpoint holding the learned Q')
@click.option('--arm', default='unknown')
@click.option('--n', 'n_actions', default=16, show_default=True)
@click.option('--episodes', default=20, show_default=True)
@click.option('--per-episode', default=8, show_default=True)
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('--batch', default=8, show_default=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('--out', default=None, help='write the stats as JSON here')
def main(checkpoint, q_ckpt, arm, n_actions, episodes, per_episode, split, batch, device, seed, out):
    policy, cfg = load_policy(checkpoint, device)
    q = PushTQVerifier(q_ckpt, device=device)
    To, Ta, H = policy.n_obs_steps, policy.n_action_steps, cfg.policy.horizon

    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    _, ep_idxs = get_split_states(cfg, split, run_dir=run_dir)      # ST's OWN held-out episodes
    ep_idxs = list(ep_idxs)[:episodes]
    rb = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path,
                                     keys=['img', 'agent_pos', 'action', 'block_pos'])
    pts = sample_points(rb, ep_idxs, per_episode, To, H, np.random.default_rng(seed))
    print(f'{arm}: {len(pts)} decision points from {len(ep_idxs)} {split} episodes, n={n_actions}')

    sim_star, sim_cand, q_star, q_cand, t_cand, ep_of, ref_of = [], [], [], [], [], [], []
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        for b0 in range(0, len(pts), batch):
            chunk = pts[b0:b0 + batch]
            obs, star = build_batch(rb, chunk, To, H, torch.device(device))
            ref_of.append(t_goal_distance(obs['feedback'][:, To - 1].cpu().numpy()))
            seeder = getattr(policy, 'set_sample_seeds', None)
            if seeder is not None:
                seeder([seed * 1_000_003 + b0 + k for k in range(len(chunk))])
            with torch.no_grad(), policy._crop_scope():
                feats = policy._encode_obs_features(obs)
                acts, _, sc, tm = policy.predict_n_actions(
                    obs, verifier=policy.verifier, n_actions=n_actions, return_scores=True,
                    obs_features=feats, return_terms=True)
                # a* through THE SAME call the candidates go through -- same verifier, same
                # window. A separate scoring path could drift from the ranking actually applied.
                out_star = policy._score_candidates(policy.verifier, obs, star)
            sim_star.append(out_star[1].float().cpu().numpy())
            sim_cand.append(sc.float().cpu().numpy())
            t_cand.append(tm.float().cpu().numpy())
            q_star.append(_q_scores(q, obs, star, To, Ta))
            q_cand.append(_q_scores(q, obs, acts, To, Ta))
            ep_of += [e for e, _ in chunk]
            print(f'  {min(b0 + batch, len(pts))}/{len(pts)}', end='\r', flush=True)
    finally:
        for obj in (policy, q):
            close = getattr(obj, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f'warning: close failed: {exc}')

    eps = np.array(ep_of)
    refs = np.concatenate(ref_of)
    cls, n_movers, _ = classify(np.concatenate(t_cand)[:, :, T_GOAL], refs)
    scores = {'sim': (np.concatenate(sim_star), np.concatenate(sim_cand)),
              'q': (np.concatenate(q_star), np.concatenate(q_cand))}

    report = {'checkpoint': checkpoint, 'q': q_ckpt, 'arm': arm, 'n': n_actions,
              'split': split, 'episodes': len(ep_idxs), 'decisions': int(len(eps)),
              'class_fractions': {c: float((cls == c).mean()) for c in CLASSES}}
    print(f"\ndecisions: {len(eps)}   " + "  ".join(
        f"{c}={float((cls == c).mean()):.1%}" for c in CLASSES))
    print("  (blind = the sim verifier gave EVERY candidate the same score; argmax there is a "
          "tie-break)\n")

    hdr = f"{'subset':<14}{'n':>6}  {'verifier':<10}{'p_best':>18}{'mean_rank':>18}"
    print(hdr)
    print('-' * len(hdr))
    for subset in ('all',) + CLASSES:
        sel = np.ones(len(eps), bool) if subset == 'all' else (cls == subset)
        if sel.sum() < 2 or len(np.unique(eps[sel])) < 2:
            continue
        report.setdefault('by_class', {})[subset] = {}
        for name in ('sim', 'q'):
            s, c = scores[name]
            st = _stats(s[sel], c[sel], higher_is_better=True, ep_ids=eps[sel], seed=seed)
            report['by_class'][subset][name] = st
            pb, mr = st['p_best'], st['mean_rank']
            print(f"{subset:<14}{int(sel.sum()):>6}  {name:<10}"
                  f"{pb['v']:>8.3f} [{pb['ci'][0]:.2f},{pb['ci'][1]:.2f}]"
                  f"{mr['v']:>8.2f} [{mr['ci'][0]:.1f},{mr['ci'][1]:.1f}]")
        print()
    print(f"mean_rank is out of {n_actions} candidates; 0 means a* beat every one, "
          f"{n_actions / 2:.1f} is the exchangeability null (ties count as half).")

    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(report, indent=2))
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
