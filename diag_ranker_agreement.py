"""Does a learned ranker agree with the t_goal heuristic it replaces, where the heuristic can see?

Two questions, one run:

  * CONTACT decisions -- some candidate moved the T, so t_goal has real signal. Does the learned
    ranker order candidates the same way? This is the question the heuristic can be checked
    against; disagreement here is the learned value contradicting a verifier that knows.
  * NO-CONTACT decisions -- no candidate moved the T, so every t_goal score is bit-identical and
    argmax degenerates to numpy's first-maximizer tie-break. The heuristic is blind by
    construction, so agreement is undefined and only the learned ranker's own spread is
    meaningful. Reported separately, never pooled.

Contact is decided on the heuristic's OWN scores, at the same BLIND_EPS the repo uses to call a
decision blind -- so "contact" here means exactly "t_goal could tell the candidates apart".

BOTH SCALARS ON THE SAME CANDIDATES. Running two rankers as separate rollouts would not answer
this: a different ranker takes different actions, so by the second decision the two runs are at
different states scoring different candidate sets. Instead score each candidate set twice inside
ONE rollout and return the sim value, so the trajectory is exactly t_goal's.

  python diag_ranker_agreement.py --ckpt <ST.ckpt> --q <sac.zip> --out <out>.json
  python diag_ranker_agreement.py --ckpt <ST.ckpt> --v <ppo.zip> --out <out>.json
"""
import argparse, json, pathlib
import numpy as np
import torch

from diffusion_policy.common.slot_stats_util import BLIND_EPS
from sac.eval import load_policy
from eval_search_pusht import _eval_split_at_n, build_envs, get_split_states


def _summary(pairs):
    """Rank agreement between two scalars over candidate sets. Spearman, not Pearson: selection
    reads ORDER, and a monotone but curved value would look bad under Pearson while ranking
    identically."""
    from scipy.stats import spearmanr

    rho, agree, spread = [], [], []
    for x, y in pairs:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) != len(y) or len(x) < 2:
            continue
        agree.append(int(np.argmax(x) == np.argmax(y)))
        spread.append(float(y.std()))
        if x.std() > 1e-12 and y.std() > 1e-12:
            rho.append(float(spearmanr(x, y).correlation))
    rho = np.asarray(rho)
    n = len(pairs[0][0]) if pairs else 0
    return {
        'decisions': len(pairs),
        'decisions_with_spread': int(len(rho)),
        'spearman_mean': float(rho.mean()) if len(rho) else None,
        'spearman_median': float(np.median(rho)) if len(rho) else None,
        'frac_negative': float((rho < 0).mean()) if len(rho) else None,
        'argmax_agreement': float(np.mean(agree)) if agree else None,
        'argmax_agreement_chance': (1.0 / n) if n else None,
        'learned_spread_median': float(np.median(spread)) if spread else None,
    }


# UNDER `if __name__ == '__main__'`, and that is load-bearing, not style. PushTVerifier's rollout
# pool spawns rather than forks, so the child re-imports __main__ -- and with this work at module
# level the child re-ran the whole script and tried to spawn again, which multiprocessing catches
# as "a new process before the current process has finished its bootstrapping phase".
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--q', default=None, help='SAC checkpoint')
    p.add_argument('--v', default=None, help='PPO checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--split', default='test')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if bool(a.q) == bool(a.v):
        raise SystemExit('pass exactly one of --q / --v')

    policy, cfg = load_policy(a.ckpt, a.device)
    states, idxs = get_split_states(cfg, a.split, pathlib.Path(a.ckpt).resolve().parent.parent)
    states, idxs = states[:a.episodes], idxs[:a.episodes]
    n_envs = min(a.episodes, 8)
    env = build_envs(n_envs, policy.n_obs_steps, policy.n_action_steps, 300)

    if a.q:
        from sac.score import PushTQVerifier
        scorer, kind = PushTQVerifier(a.q, device=a.device), 'q'
    else:
        from recurrent_ppo.value_score import PushTVVerifier
        scorer, kind = PushTVVerifier(a.v, policy.verifier, device=a.device), 'v'

    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ('agent_pos', 'feedback', 'image')
    original = policy._score_candidates
    pairs, buf = [], []

    # `_score_candidates` IS CALLED ONCE PER CANDIDATE, not once per decision: search_procedure
    # loops `for i in range(n_actions)` and passes one candidate for the whole env batch, so the
    # array it returns is (B_envs,) -- one score per ENVIRONMENT for candidate i. Treating that
    # array as "the candidate set" compares scores at B different states, which is what the
    # first version of this file did and why its argmax-agreement number meant nothing.
    # Buffer n consecutive calls and transpose instead: (n, B) -> B rows of n real candidates,
    # each row one environment's candidate set at one decision.
    def patched(verifier, obs_dict, action, want_subgoals=False):
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        now = {k: val[:, To - 1:To] for k, val in obs_dict.items() if k in keys}
        learned = scorer.get_value(now, action[:, To - 1:To - 1 + Ta])
        buf.append((value.detach().cpu().numpy().ravel(),
                    learned.detach().cpu().numpy().ravel()))
        if len(buf) == a.n:
            sim = np.stack([b[0] for b in buf], axis=1)        # (B, n)
            lrn = np.stack([b[1] for b in buf], axis=1)        # (B, n)
            for r in range(sim.shape[0]):
                pairs.append((sim[r].tolist(), lrn[r].tolist()))
            buf.clear()
        return context, value, subgoal, terms     # sim value: the trajectory stays t_goal's

    policy._score_candidates = patched
    try:
        _eval_split_at_n(env, policy, states, a.device, a.n, n_envs, a.seed, f'{kind}-agree')
    finally:
        policy._score_candidates = original
        env.close()

    if buf:
        # A partial group means the call pattern is not what this file assumes; refuse rather
        # than silently average over a ragged set.
        raise SystemExit(f'{len(buf)} unpaired _score_candidates calls -- the search width is '
                         f'not {a.n}; rerun with the matching --n')
    contact = [pr for pr in pairs if np.asarray(pr[0], float).std() > BLIND_EPS]
    blind = [pr for pr in pairs if np.asarray(pr[0], float).std() <= BLIND_EPS]
    res = {'checkpoint': a.ckpt, kind: a.q or a.v, 'ranker': kind, 'n': a.n, 'split': a.split,
           'episodes': [int(i) for i in idxs], 'blind_eps': BLIND_EPS,
           'frac_no_contact': (len(blind) / len(pairs)) if pairs else None,
           'contact': _summary(contact) if contact else None,
           'no_contact': _summary(blind) if blind else None}
    print(json.dumps(res, indent=2))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
