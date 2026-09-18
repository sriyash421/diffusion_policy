"""Is the learned V anti-correlated with the t_goal heuristic it replaces?

The best-of-N curves rise in n under `t_goal` and fall under `v`, which is the signature of a
ranker that prefers the wrong candidate. Under `--reward delta` the return telescopes to
(d_here - d_end) plus a success bonus, so for a policy whose achievable d_end barely depends on
the state -- and ppo_plain_keypoint finished 10M steps at success_rate 0, so its bonus term is
~0 everywhere -- V should be an INCREASING function of the current T-to-goal distance, while
t_goal is its negation. Selection is argmax, so V would pick the candidate landing furthest
from the goal.

BOTH SCALARS ON THE SAME CANDIDATES. Running the two rankers as separate rollouts would not
answer this: a different ranker takes different actions, so by the second decision the two runs
are at different states scoring different candidate sets. Instead score each candidate set
twice inside ONE rollout and return the sim value, so the trajectory is exactly t_goal's.
"""
import json, pathlib, sys
import numpy as np
import torch

from sac.eval import load_policy
from eval_search_pusht import _eval_split_at_n, build_envs, get_split_states
from recurrent_ppo.value_score import PushTVVerifier

# UNDER `if __name__ == '__main__'`, and that is load-bearing, not style. PushTVerifier's
# rollout pool spawns rather than forks, so the child re-imports __main__ -- and with this
# work at module level the child re-ran the whole script and tried to spawn again, which
# multiprocessing catches as "a new process before the current process has finished its
# bootstrapping phase". Four files in this repo were guarded for the same reason already.
def main():
    CK, V, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    EPS = int(sys.argv[5]) if len(sys.argv) > 5 else 12
    DEV, SEED = 'cuda:0', 42

    policy, cfg = load_policy(CK, DEV)
    states, idxs = get_split_states(cfg, 'test', pathlib.Path(CK).resolve().parent.parent)
    states, idxs = states[:EPS], idxs[:EPS]
    n_envs = min(EPS, 6)
    env = build_envs(n_envs, policy.n_obs_steps, policy.n_action_steps, 300)
    vv = PushTVVerifier(V, policy.verifier, device=DEV)

    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ('agent_pos', 'feedback', 'image')
    original = policy._score_candidates
    pairs = []

    def patched(verifier, obs_dict, action, want_subgoals=False):
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        now = {k: val[:, To - 1:To] for k, val in obs_dict.items() if k in keys}
        vscore = vv.get_value(now, action[:, To - 1:To - 1 + Ta])
        pairs.append((value.detach().cpu().numpy().ravel().tolist(),
                      vscore.detach().cpu().numpy().ravel().tolist()))
        return context, value, subgoal, terms   # sim value: the trajectory stays t_goal's

    policy._score_candidates = patched
    try:
        _eval_split_at_n(env, policy, states, DEV, N, n_envs, SEED, 'paired')
    finally:
        policy._score_candidates = original
        env.close()

    # WITHIN-DECISION is the correlation that matters: argmax ranks candidates against each other
    # at one state, so a pooled correlation would be dominated by between-state variation that
    # selection never sees.
    per, agree = [], []
    for x, y in pairs:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) != len(y) or len(x) < 2:
            continue
        agree.append(int(np.argmax(x) == np.argmax(y)))
        if x.std() > 1e-12 and y.std() > 1e-12:
            per.append(np.corrcoef(x, y)[0, 1])

    per = np.asarray(per)
    fx = np.concatenate([np.asarray(x, float) for x, _ in pairs])
    fy = np.concatenate([np.asarray(y, float) for _, y in pairs])
    res = {'checkpoint': CK, 'v': V, 'n': N, 'episodes': [int(i) for i in idxs],
           'decisions': len(pairs), 'decisions_with_spread': int(len(per)),
           'within_decision_corr_mean': float(per.mean()) if len(per) else None,
           'within_decision_corr_median': float(np.median(per)) if len(per) else None,
           'within_decision_frac_negative': float((per < 0).mean()) if len(per) else None,
           'pooled_corr': float(np.corrcoef(fx, fy)[0, 1]),
           'argmax_agreement': float(np.mean(agree)) if agree else None,
           'argmax_agreement_chance': 1.0 / N}
    print(json.dumps(res, indent=2))
    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(OUT).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
