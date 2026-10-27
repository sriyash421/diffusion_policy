"""Full-trajectory best-of-N eval for PushT: N sequential in-context attempts per episode.

Mirrors L2S's maze eval (l2s/eval_openloop.py: rollout_dt / eval_candidates /
select_and_summarize) on the live PushT env. For every held-out episode the policy stream is
reset once, then N attempts run one after another from the recorded initial state; attempt k of
an in-context policy sees attempts < k. Each attempt runs until the policy stops it, the env
reports done (coverage > 0.95, or its step cap) or the guard trips. The per-step feedback the
policy is handed -- and the value a candidate is ranked by -- is pusht_step_feedback.step_feedback
at the REALIZED state (veritas pin cost + T-to-goal), computed here, so it is the same verifier
for every arm.

Policy interface (duck-typed): reset(), begin_attempt(), predict_step(obs_dict) -> (action (B, 2),
active (B,) bool); obs_dict carries the env's (B, 1, ...) frame plus `fb` (B,) and `done` (B,).

Per (episode, candidate): success (max reward >= 1 within the attempt), max coverage, final
reward, n_steps, stop reason, final value, the feedback trace and the actions. Per n in
{1, 2, 4, .., N}: argmax@n (highest final value among the first n), any@n (perf@n: any of the
first n succeeded) and last@n. Outputs under <run>/bon_search/dt_eval/<step>/.

    python eval_dt_pusht.py <ckpt> [--n 8] [--split test|val|both] [--n-envs 50]
    python eval_dt_pusht.py --watch --run-dir <run>           # every step_*.ckpt as it appears
"""
import csv
import json
import os
import pathlib
import time

import click
import numpy as np
import torch
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.stats_util import wilson_interval
from diffusion_policy.env.pusht.pusht_feedback import (
    build_parallel, set_reset_states, set_veritas_plans)
from diffusion_policy.env.pusht.pusht_step_feedback import step_feedback
from diffusion_policy.env.pusht.veritas.tracker import WAYPOINT_PLAN_DIRS, load_split_plans
from eval_search_pusht import get_split_states, load_policy, step_from_ckpt

# Workspaces that never update their EMA copy: under pusht_base's `use_ema: True` their
# checkpoint carries an untrained ema_model (an init-time clone with an empty normalizer),
# so the live weights are the trained ones.
_NO_EMA_WORKSPACES = ('TrainOnlineSearchWorkspace',)


def load_eval_policy(checkpoint, device):
    """load_policy, except that a no-EMA workspace's checkpoint is read at its live weights."""
    import dill
    import omegaconf
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    if not str(cfg._target_).endswith(_NO_EMA_WORKSPACES) or not cfg.training.get('use_ema', False):
        return load_policy(checkpoint, device)
    import hydra
    with omegaconf.open_dict(cfg):
        cfg.training.use_ema = False
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=('ema_model',), include_keys=None)
    policy = workspace.model
    policy.to(torch.device(device))
    policy.eval()
    print(f'[load] {cfg._target_} never steps its EMA; evaluating the live weights')
    return policy, cfg

FEEDBACK_VALUE = 'wp_v3_la_tgn'
STOP_REASONS = {0: 'running', 1: 'model', 2: 'env', 3: 'guard'}
# argmax ranks by the verifier value at the final state; argmax_cov by the env's max coverage of
# the executed attempt (observed, so deployable in the sequential protocol; equals the oracle
# on that coverage by construction). any = perf@n, last = the n-th attempt.
RULES = ('argmax', 'argmax_cov', 'any', 'last')


class ChunkPolicyAdapter:
    """A chunk policy (UNet BC: predict_action on an n_obs_steps window -> n_action_steps
    actions, replan when the chunk is spent) behind the per-step interface, so the BC baseline
    is evaluated by exactly the same 8-attempt protocol. Attempts are i.i.d. (no context, no
    stop head): every attempt replans closed-loop until env done or the step cap, as the
    chunked evals do. The obs window is rebuilt from the frames seen, padded by repeating the
    first frame as MultiStepWrapper does, so the policy sees what it would in eval_search_pusht.
    """

    def __init__(self, policy):
        self.policy = policy
        self.To, self.Ta = int(policy.n_obs_steps), int(policy.n_action_steps)
        self.frames, self.queue = [], []

    def reset(self):
        self.frames, self.queue = [], []

    def begin_attempt(self):
        self.frames, self.queue = [], []

    def predict_step(self, obs_dict):
        frame = {k: v[:, -1] for k, v in obs_dict.items() if k not in ('fb', 'done', 'coverage', 'attention_mask')}
        self.frames.append(frame)
        B = next(iter(frame.values())).shape[0]
        if not self.queue:
            window = (self.frames[:1] * (self.To - len(self.frames)) + self.frames)[-self.To:]
            obs = {k: torch.stack([f[k] for f in window], dim=1) for k in frame}
            action = self.policy.predict_action(obs)['action']          # (B, Ta, Da)
            self.queue = [action[:, i] for i in range(action.shape[1])]
        return self.queue.pop(0), torch.ones(B, dtype=torch.bool, device=next(iter(frame.values())).device)


def rollout_attempt(env, policy, states, plans, device, max_steps):
    """One attempt for every env row. Returns dict of per-row arrays (see module doc)."""
    set_reset_states(env, list(states))
    set_veritas_plans(env, list(plans))
    obs = env.reset()
    policy.begin_attempt()
    B = len(states)
    was_active = np.ones(B, dtype=bool)
    done = np.zeros(B, dtype=bool)
    n_steps = np.zeros(B, dtype=np.int64)
    traces = [[] for _ in range(B)]
    actions = [[] for _ in range(B)]
    hold = np.asarray(obs['agent_pos'][:, -1], dtype=np.float32)   # action for inactive rows
    for _ in range(max_steps + 2):
        trackers = env.call('get_attr', 'tracker')
        fb = step_feedback(obs['agent_pos'][:, -1], obs['feedback'][:, -1], trackers)
        for i in np.flatnonzero(was_active):
            traces[i].append(float(fb[i]))
        obs_t = dict_apply(obs, lambda x: torch.from_numpy(np.asarray(x)).to(device))
        obs_t['fb'] = torch.from_numpy(fb).to(device)
        obs_t['done'] = torch.from_numpy(done).to(device)
        # the env's max coverage so far: the terminal_feedback=coverage policies read it at an
        # attempt's end; recorded for every arm as the argmax_cov ranking signal
        obs_t['coverage'] = torch.as_tensor(
            np.asarray(env.call('get_attr', 'max_coverage'), dtype=np.float32), device=device)
        with torch.inference_mode():
            action, active = policy.predict_step(obs_t)
        action = action.detach().cpu().numpy().astype(np.float32)
        active = active.detach().cpu().numpy().astype(bool)
        # the harness ends a row the env finished or that hit the guard, whether or not the
        # policy noticed (a policy without a stop head keeps every row active)
        active &= ~done & (n_steps < max_steps)
        if not active.any():
            break
        # a finished row holds its pusher still, so it cannot disturb its own final state
        action = np.where(active[:, None], action, hold)
        hold = np.where(active[:, None], action, hold)
        for i in np.flatnonzero(active):
            actions[i].append(action[i])
        n_steps += active
        was_active = active
        obs, _, step_done, _ = env.step(action[:, None, :])
        done |= np.asarray(step_done, dtype=bool)
    reason = getattr(policy, 'attempt_stop_reason', None)
    reason = np.where(done, 2, 3) if reason is None else reason.detach().cpu().numpy()
    reason = np.where(reason == 0, np.where(done, 2, 3), reason)   # rows the harness ended
    rewards = env.call('get_attr', 'reward')              # per-env per-step list
    # rewards are read over the row's OWN steps; a finished row holds still, so the env's
    # max_coverage (accumulated over every step it was driven) is the attempt's as well
    n_steps = np.minimum(n_steps, [len(r) for r in rewards])
    max_r = np.array([max(r[:n]) if n else 0.0 for r, n in zip(rewards, n_steps)])
    final_r = np.array([r[n - 1] if n else 0.0 for r, n in zip(rewards, n_steps)])
    return {
        'success': max_r >= 1.0, 'max_reward': max_r, 'final_reward': final_r,
        'max_coverage': np.asarray(env.call('get_attr', 'max_coverage'), dtype=np.float64),
        'n_steps': n_steps, 'stop_reason': np.asarray(reason),
        'value': np.array([t[-1] for t in traces], dtype=np.float32),
        'traces': traces, 'actions': actions,
    }


def eval_candidates(env, policy, states, plans, device, n_candidates, max_steps, n_envs, label):
    """(E, N) arrays over all episodes of a split."""
    E = len(states)
    out = {k: [] for k in ('success', 'max_reward', 'final_reward', 'max_coverage', 'n_steps',
                           'stop_reason', 'value')}
    traces, actions = [], []
    for start in tqdm.tqdm(range(0, E, n_envs), desc=label, leave=False):
        chunk_states = list(states[start:start + n_envs])
        chunk_plans = list(plans[start:start + n_envs])
        m = len(chunk_states)
        pad = n_envs - m
        chunk_states += [chunk_states[0]] * pad
        chunk_plans += [chunk_plans[0]] * pad
        policy.reset()
        per_k = []
        t0 = time.perf_counter()
        for k in range(n_candidates):
            per_k.append(rollout_attempt(env, policy, chunk_states, chunk_plans, device, max_steps))
        tqdm.tqdm.write(f'  {label} chunk {start}: {n_candidates} attempts in '
                        f'{time.perf_counter() - t0:.1f}s')
        for key in out:
            out[key].append(np.stack([r[key][:m] for r in per_k], axis=1))
        traces += [[r['traces'][i] for r in per_k] for i in range(m)]
        actions += [[r['actions'][i] for r in per_k] for i in range(m)]
    cands = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    cands['traces'] = traces
    cands['actions'] = actions
    return cands


def select(cands, n, rule):
    """Chosen candidate index per episode among the first n."""
    E = cands['value'].shape[0]
    succ = cands['success'][:, :n]
    if rule == 'argmax':
        return cands['value'][:, :n].argmax(axis=1)
    if rule == 'argmax_cov':
        return cands['max_coverage'][:, :n].argmax(axis=1)
    if rule == 'last':
        return np.full(E, n - 1)
    if rule == 'any':   # perf@n: the first success, else the argmax
        first = succ.argmax(axis=1)
        return np.where(succ.any(axis=1), first, cands['value'][:, :n].argmax(axis=1))
    raise ValueError(rule)


def summarize(cands, n, rule):
    E = cands['value'].shape[0]
    chosen = select(cands, n, rule)
    rows = np.arange(E)
    success = cands['success'][rows, chosen]
    k = int(success.sum())
    lo, hi = wilson_interval(k, E)
    return {
        'n': n, 'rule': rule, 'episodes': E, 'successes': k,
        'success_rate': k / E, 'ci_lo': lo, 'ci_hi': hi,
        'mean_max_coverage': float(cands['max_coverage'][rows, chosen].mean()),
        'mean_final_reward': float(cands['final_reward'][rows, chosen].mean()),
        'mean_n_steps': float(cands['n_steps'][rows, chosen].mean()),
        'mean_value': float(cands['value'][rows, chosen].mean()),
        'value_std_across_candidates': float(cands['value'][:, :n].std(axis=1).mean()),
        'frac_model_stop': float((cands['stop_reason'][:, :n] == 1).mean()),
        'frac_env_done': float((cands['stop_reason'][:, :n] == 2).mean()),
    }


def plot_curve(rows, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'argmax': ('#0072B2', 'o', '-'), 'argmax_cov': ('#CC79A7', 'D', '-.'),
             'any': ('#D55E00', 's', '--'), 'last': ('#009E73', '^', ':')}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for rule in RULES:
        r = [x for x in rows if x['rule'] == rule]
        if not r:
            continue
        ns = [x['n'] for x in r]
        c, m, ls = style[rule]
        axes[0].errorbar(ns, [x['success_rate'] for x in r],
                         yerr=[[x['success_rate'] - x['ci_lo'] for x in r],
                               [x['ci_hi'] - x['success_rate'] for x in r]],
                         color=c, marker=m, linestyle=ls, label=f'{rule}@n', capsize=3)
        axes[1].plot(ns, [x['mean_max_coverage'] for x in r], color=c, marker=m, linestyle=ls,
                     label=f'{rule}@n')
    for ax, ylabel in zip(axes, ('success rate', 'mean max coverage')):
        ax.set_xscale('log', base=2); ax.set_xlabel('n candidates'); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def eval_checkpoint(checkpoint, device, splits, n_candidates, n_envs, max_steps, seed, out_dir,
                    plan_dir=None, max_episodes=None):
    policy, cfg = load_eval_policy(checkpoint, device)
    if not hasattr(policy, 'predict_step'):
        print(f'[load] {type(policy).__name__} is a chunk policy: wrapping it for per-step, '
              f'i.i.d. attempts (n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps})')
        policy = ChunkPolicyAdapter(policy)
    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    plan_dir = plan_dir or WAYPOINT_PLAN_DIRS[FEEDBACK_VALUE]
    env = build_parallel(n_envs=n_envs, use_async=True, n_obs_steps=1, n_action_steps=1,
                         max_episode_steps=max_steps, legacy=False, tracker=True)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    try:
        for split in splits:
            states, idxs = get_split_states(cfg, split, run_dir=run_dir)
            if max_episodes:
                states, idxs = states[:max_episodes], idxs[:max_episodes]
            if len(states) == 0:
                print(f'{split}: empty split, skipped')
                continue
            plans = load_split_plans(plan_dir, idxs, split, FEEDBACK_VALUE)
            torch.manual_seed(seed); np.random.seed(seed)
            cands = eval_candidates(env, policy, states, plans, device, n_candidates, max_steps,
                                    n_envs, f'{split}')
            n_list = [n for n in (2 ** k for k in range(8)) if n <= n_candidates]
            rows = [dict(split=split, checkpoint=os.path.basename(checkpoint), **summarize(cands, n, rule))
                    for n in n_list for rule in RULES]
            for r in rows:
                print(f"{split} n={r['n']:>2} {r['rule']:>6}: success={r['success_rate']:.3f} "
                      f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] ({r['successes']}/{r['episodes']}) "
                      f"cov={r['mean_max_coverage']:.3f} steps={r['mean_n_steps']:.0f} "
                      f"stop(model/env)={r['frac_model_stop']:.2f}/{r['frac_env_done']:.2f}")
            all_rows += rows
            np.savez(out_dir / f'{split}_candidates.npz', episode_idxs=np.asarray(idxs),
                     **{k: v for k, v in cands.items() if k not in ('traces', 'actions')},
                     traces=np.array(cands['traces'], dtype=object),
                     actions=np.array(cands['actions'], dtype=object))
            with open(out_dir / 'candidate_scores.jsonl', 'a') as f:
                for e, ep in enumerate(idxs):
                    for k in range(n_candidates):
                        f.write(json.dumps({
                            'checkpoint': os.path.basename(checkpoint), 'split': split,
                            'episode': int(ep), 'candidate': k,
                            'success': bool(cands['success'][e, k]),
                            'max_coverage': round(float(cands['max_coverage'][e, k]), 4),
                            'final_reward': round(float(cands['final_reward'][e, k]), 4),
                            'n_steps': int(cands['n_steps'][e, k]),
                            'stop_reason': STOP_REASONS[int(cands['stop_reason'][e, k])],
                            'value': round(float(cands['value'][e, k]), 5)}) + '\n')
            plot_curve(rows, out_dir / f'{split}_curve.png',
                       f'{run_dir.name} {os.path.basename(checkpoint)} {split}')
    finally:
        env.close()
    with open(out_dir / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    return all_rows


@click.command()
@click.argument('checkpoint', required=False)
@click.option('--output-dir', default=None, help='default <run>/bon_search/dt_eval/<step>')
@click.option('--watch', is_flag=True, help='evaluate every step_*.ckpt under --run-dir as it appears')
@click.option('--run-dir', default=None)
@click.option('--device', default='cuda:0')
@click.option('--n', 'n_candidates', type=int, default=8)
@click.option('--n-envs', type=int, default=50)
@click.option('--max-steps', type=int, default=300)
@click.option('--seed', type=int, default=0)
@click.option('--split', type=click.Choice(['test', 'val', 'both']), default='both')
@click.option('--plan-dir', default=None)
@click.option('--poll-sec', type=int, default=300)
@click.option('--max-episodes', type=int, default=None, help='smoke tests: first K episodes of each split')
def main(checkpoint, output_dir, watch, run_dir, device, n_candidates, n_envs, max_steps, seed,
         split, plan_dir, poll_sec, max_episodes):
    splits = ['val', 'test'] if split == 'both' else [split]

    def _out(ckpt):
        if output_dir is not None and not watch:
            return output_dir
        run = pathlib.Path(ckpt).resolve().parent.parent
        return run / 'bon_search' / 'dt_eval' / f'step_{step_from_ckpt(ckpt) or 0:07d}'

    if not watch:
        assert checkpoint, 'checkpoint required without --watch'
        eval_checkpoint(checkpoint, device, splits, n_candidates, n_envs, max_steps, seed,
                        _out(checkpoint), plan_dir, max_episodes)
        return
    assert run_dir, '--run-dir required with --watch'
    ckpt_dir = pathlib.Path(run_dir) / 'checkpoints'
    while True:
        pending = [p for p in sorted(ckpt_dir.glob('step_*.ckpt'))
                   if not (_out(p) / 'metrics.csv').exists()]
        if not pending:
            time.sleep(poll_sec)
            continue
        ckpt = str(pending[0])
        print(f'[watch] evaluating {ckpt}')
        eval_checkpoint(ckpt, device, splits, n_candidates, n_envs, max_steps, seed, _out(ckpt), plan_dir)


if __name__ == '__main__':
    main()
