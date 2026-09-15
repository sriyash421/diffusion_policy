"""EVAL TEST 4 -- does the learned Q improve a policy through best-of-N?

The headline number. `n in {1,2,4,8,16,32,64}` on ST k=1 and BC, ranked by the learned Q and by
the sim heuristics it replaces, on the SAME held-out episodes with the SAME per-episode sampling
noise -- so the arms are paired episode by episode and the only thing that differs is the
ranker.

WHAT IS SWAPPED, AND WHAT IS DELIBERATELY NOT. `_score_candidates` returns
`(context, value, subgoal, terms)`: `context` is what ST conditions its NEXT candidate on,
`value` is what argmax ranks by. This replaces **only `value`**. ST was trained with a
`t_goal`-shaped search context, so handing it a Q-shaped context at eval would put it off its
training distribution and confound "Q ranks better" with "ST was given an input it has never
seen". The sim verifier therefore still runs, purely to produce the context ST expects.

BC (`PushTUNetSearchPolicy`) ignores the context entirely, so for BC the two are identical and
`--skip-context-sim` drops the sim call as pure overhead. That also makes BC the cleaner first
read of whether the Q ranks better at all.

    python sac/scripts/bon_sweep.py -c <bc_30k.ckpt> --q logs/sac/keypoint/<run>/model.zip \
        --rankers q,t_goal,armTn --max-n 16
"""

import json
import pathlib
import sys

import click
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from eval_search_pusht import (_eval_split_at_n, build_envs, get_split_states,   # noqa: E402
                               load_policy)
from sac.score import PushTQVerifier                                             # noqa: E402


def install_q_ranker(policy, q, skip_context_sim=False):
    """Replace ONLY the ranking scalar in `_score_candidates`. Returns a restore callable.

    Monkeypatched on the instance rather than edited into `pusht_search_mixin`, so nothing that
    ST and BC are trained and evaluated with changes on disk, and an ordinary run is bit-for-bit
    what it was.
    """
    original = policy._score_candidates
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ("agent_pos", "feedback", "image")

    def patched(verifier, obs_dict, action, want_subgoals=False):
        now = {k: v[:, To - 1:To] for k, v in obs_dict.items() if k in keys}
        exec_action = action[:, To - 1:To - 1 + Ta]
        qv = q.get_value(now, exec_action)
        if skip_context_sim:
            # BC never reads the context, so the sim call is pure overhead. `terms` is only
            # used by cross-candidate values, which are not in play here.
            zeros = torch.zeros_like(qv)
            return zeros, qv.to(zeros.dtype), None, None
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        return context, qv.to(value.device).to(value.dtype), subgoal, terms

    policy._score_candidates = patched
    return lambda: setattr(policy, "_score_candidates", original)


def set_sim_value(policy, value_fn):
    """Point the sim verifier at one of VALUE_FNS, both places that must move together."""
    policy.search_kwargs["verifier_value"] = value_fn
    built = policy.__dict__.get("_verifier") or policy.__dict__.get("verifier")
    if built is not None:
        built.value_fn = value_fn


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--q', 'q_ckpt', default=None, help='SAC checkpoint; required if `q` is a ranker')
@click.option('--rankers', default='q,t_goal,armTn', show_default=True,
              help='comma-separated: any of q, t_goal, d_t_goal, armTn')
@click.option('--max-n', default=16, show_default=True)
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('--n-envs', default=25, show_default=True)
@click.option('--max-steps', default=300, show_default=True)
@click.option('--episodes', default=None, type=int, help='cap, for a smoke run')
@click.option('--skip-context-sim', is_flag=True, help='BC only: it ignores the search context')
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('-o', '--out', default='sac_eval/bon_sweep', show_default=True)
def main(checkpoint, q_ckpt, rankers, max_n, split, n_envs, max_steps, episodes,
         skip_context_sim, device, seed, out):
    outdir = pathlib.Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    policy, cfg = load_policy(checkpoint, device)
    if not hasattr(policy, 'predict_action_best'):
        raise SystemExit(f'{type(policy).__name__} has no predict_action_best, so best-of-n is '
                         'undefined for it.')
    rankers = tuple(r.strip() for r in str(rankers).split(',') if r.strip())
    states, idxs = get_split_states(cfg, split, pathlib.Path(checkpoint).resolve().parent.parent)
    if episodes:
        states, idxs = states[:episodes], idxs[:episodes]
    n_list = [2 ** k for k in range(int(np.log2(max_n)) + 1)]
    print(f'[INFO] {len(idxs)} {split} episodes, n in {n_list}, rankers {list(rankers)}')

    q = PushTQVerifier(q_ckpt, device=device) if 'q' in rankers else None
    if 'q' in rankers and q_ckpt is None:
        raise SystemExit('--q is required when `q` is among the rankers')

    env = build_envs(min(n_envs, len(idxs)), policy.n_obs_steps, policy.n_action_steps, max_steps)
    curves = {}
    try:
        for ranker in rankers:
            restore = None
            if ranker == 'q':
                restore = install_q_ranker(policy, q, skip_context_sim)
            else:
                set_sim_value(policy, ranker)
            try:
                rates, cis = [], []
                for n in n_list:
                    # the SAME per-episode noise under every ranker, so the arms are paired
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    rate, ci, _ = _eval_split_at_n(env, policy, states, device, n,
                                                   min(n_envs, len(idxs)), seed,
                                                   label=f'{ranker} n={n}')
                    rates.append(float(rate))
                    cis.append([float(ci[0]), float(ci[1])])
                    print(f'  {ranker:<8} n={n:<3} success {rate:.3f} '
                          f'[{ci[0]:.3f}, {ci[1]:.3f}]')
                curves[ranker] = {'n': n_list, 'success': rates, 'ci': cis}
            finally:
                if restore is not None:
                    restore()
    finally:
        env.close()
        for obj in (policy, q):
            close = getattr(obj, 'close', None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    print(f'warning: close failed: {exc}')

    report = {'checkpoint': checkpoint, 'q': q_ckpt, 'split': split,
              'episodes': [int(i) for i in idxs], 'curves': curves}
    (outdir / 'bon_curves.json').write_text(json.dumps(report, indent=2))
    _plot(curves, outdir / 'bon_curves.png', checkpoint)
    print(f'wrote {outdir}/bon_curves.json, bon_curves.png')


def _plot(curves, path, title):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # style AND colour AND a direct label, so the figure survives greyscale (repo rule)
    styles = {'q': ('-', 'o', '#0072B2'), 't_goal': ('--', 's', '#D55E00'),
              'armTn': (':', '^', '#009E73'), 'd_t_goal': ('-.', 'D', '#CC79A7')}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, c in curves.items():
        ls, mk, col = styles.get(name, ('-', 'x', '#444444'))
        lo = [a for a, _ in c['ci']]
        hi = [b for _, b in c['ci']]
        ax.plot(c['n'], c['success'], ls, marker=mk, color=col, label=name, lw=2)
        ax.fill_between(c['n'], lo, hi, color=col, alpha=0.13, lw=0)
        ax.annotate(name, (c['n'][-1], c['success'][-1]), color=col, fontsize=10,
                    xytext=(7, 0), textcoords='offset points', va='center')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('n candidates')
    ax.set_ylabel('success rate (max coverage >= 0.95)')
    ax.set_title(f'Best-of-N by ranker\n{pathlib.Path(title).parent.parent.name}', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
