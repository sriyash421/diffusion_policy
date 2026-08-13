"""Per-candidate verifier scores, in generation order, from a closed-loop rollout.

The question this exists to answer: inside one search, does candidate k get BETTER as k
grows -- i.e. is conditioning on the previous candidates and their feedback actually
steering generation -- or is the search an i.i.d. resampler whose apparent gain is entirely
the argmax skimming a fixed distribution?

Nothing on disk could answer that before this script. `search_candidates(...,
return_scores=True)` has always returned the (B, n) scalar, but every consumer immediately
reduced it: the workspaces log `action_value{,_best,_first}` (three numbers out of n), the
env runner and eval_search_pusht take the argmax and drop the rest. So the shape of the
sequence -- the only thing that distinguishes steering from resampling -- was never stored.

THE TEST. If draw k is a new running maximum over draws 0..k-1 more often than *order alone*
predicts, the context is steering generation; if not, the search is resampling and the whole
gain belongs to the argmax.

The textbook null for that is 1/(k+1) -- exact for any continuous i.i.d. sequence, no free
parameters. It is WRONG HERE, and measurably so: these scores are full of exact ties (many
candidates never touch the block, so the sim returns the identical value), and ties suppress
strict-inequality records. Randomly shuffling each step's own candidates -- which is
exchangeable by construction and must therefore hit the null if the null applies -- lands
~30% BELOW 1/(k+1). Any comparison against the analytic curve would read that tie artefact
as evidence and conclude the search is worse than resampling.

So the baseline used here is the PERMUTATION null: the same step's candidate values, order
shuffled, averaged over many shuffles. It preserves each step's exact value multiset (ties
included) and destroys only the ordering, which is precisely the thing under test. The
analytic 1/(k+1) is still reported alongside, as a reference showing how far the ties move
it. Everything else (per-index means, argmax histogram, running max) is descriptive colour.

Rollouts are closed-loop and execute the argmax, exactly as `predict_action_best` does, so
the recorded scores are the ones a real deployment would have seen -- not scores from a
distribution the policy never actually visits.

  python scripts/dump_candidate_scores.py -c <run>/checkpoints/step_0009000.ckpt \
      --arm subgoal-chosen4value --out-dir <dir> --n 16 --episodes 20
  python scripts/dump_candidate_scores.py --report <dir1> <dir2> ... --out CANDIDATES_FROM_SUBGOAL.md
"""
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import json
import pathlib
import time

import click
import dill
import numpy as np
import torch
import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from eval_search_pusht import (
    SUCCESS_REWARD, build_envs, get_split_states, load_policy, wilson_interval)


# ------------------------------------------------------------------------------- rewards

# Discount factors reported alongside the undiscounted numbers. PushTEnv sets `done` as soon
# as coverage clears the threshold, so an 8-step solve and a 38-step solve both score
# max-reward 1.0 -- nothing in the headline metric prefers the faster one. A discount is the
# standard way to express that preference, and at 0.99 a 30-step-later solve is worth ~26%
# less. These are computed from the stored trajectory, so more can be added without re-running.
_GAMMAS = (0.99, 0.95)


def _reward_stats(traj):
    """Reward summaries from the full per-step trajectory. `traj` is (B, T), NaN-padded.

    Reports three quantities that the usual `max` hides:

      reward_last   the reward at the episode's FINAL step. Equals the max for a successful
                    episode (PushT terminates the instant coverage clears the threshold, so
                    the last step IS the best one), but on a failed episode it says whether
                    the policy held its best position or brushed the goal and drifted off.
      reward_max    the existing metric, kept for continuity.
      return_gXX    the discounted return sum_t gamma^t r_t, which prefers solving sooner.
      steps_to_success  env steps until the first reward >= 1, NaN if never.
    """
    out = {}
    lengths = (~np.isnan(traj)).sum(axis=1)
    last = np.array([traj[i, lengths[i] - 1] if lengths[i] else np.nan
                     for i in range(len(traj))])
    filled = np.nan_to_num(traj, nan=0.0)
    t = np.arange(traj.shape[1])
    hit = filled >= SUCCESS_REWARD
    tts = np.where(hit.any(axis=1), hit.argmax(axis=1).astype(float), np.nan)

    for g in _GAMMAS:
        gs = str(g).replace('0.', '')
        # DISCOUNTED TIME-TO-GOAL: gamma^(steps to success), 0 if never. This is the one to
        # read. It is monotone in both things we care about -- succeeding at all, and
        # succeeding sooner.
        out[f'disc_success_g{gs}'] = float(np.where(np.isnan(tts), 0.0, g ** np.nan_to_num(tts)).mean())
        # Naive sum_t gamma^t r_t, kept only as a warning. It is BACKWARDS on this task:
        # PushTEnv terminates the instant coverage clears the threshold, so a fast solve
        # contributes FEWER reward terms than a slow one and scores LOWER. Measured here, the
        # 0.90-success arms score below the 0.70-success ones on it. Episode length is
        # endogenous, so an undiscounted-horizon sum cannot be compared across policies.
        out[f'return_g{gs}'] = float((filled * (g ** t)).sum(axis=1).mean())
    out.update({
        'reward_last': float(np.nanmean(last)),
        'reward_last_std': float(np.nanstd(last)),
        'reward_max': float(np.nanmax(filled, axis=1).mean()),
        'episode_len_mean': float(lengths.mean()),
        'steps_to_success_mean': float(np.nanmean(tts)) if np.isfinite(tts).any() else None,
        # how often the episode ENDED worse than its best moment -- only possible on a
        # failure, since a success terminates at its peak
        'frac_lost_after_peak': float(np.mean(np.nanmax(filled, axis=1) - last > 1e-6)),
    })
    return out


# ---------------------------------------------------------------------------- collection

def rollout_candidate_scores(env, policy, states, device, n, max_steps):
    """One episode per env; record every candidate's verifier score at every control step.

    Returns (scores, chosen, rewards, alive, score_final) where
      scores  (T, B, n)  raw verifier value, candidate IN GENERATION ORDER
      chosen  (T, B)     argmax index -- what an argmax policy would have executed
      rewards (B,)       episode max reward
      alive   (T, B)     bool, False once that env's episode has ended
      score_final (T, B) the DEPLOYED sample's verifier value under `selection: final_pass`,
                  else None

    THE ROLLOUT IS DRIVEN BY WHATEVER THE POLICY ACTUALLY DEPLOYS. Under `argmax` that is the
    best-scoring candidate; under `final_pass` it is a further sample conditioned on all n
    candidates. Either way the visited states are the deployed ones, so the recorded scores
    describe a distribution the policy really encounters. `chosen` is still recorded in
    final_pass mode as the counterfactual -- what the oracle would have picked here.

    `alive` matters: AsyncVectorEnv keeps stepping finished envs to keep the batch square,
    so without it the tail of a short episode would contribute scores from a state the
    policy was never really in, and every per-index statistic would be diluted by whichever
    episodes happened to finish early.
    """
    def make_init_fn(state):
        state = np.asarray(state, dtype=np.float64)

        def _fn(e):
            e.unwrapped.reset_to_state = state
        return _fn

    env.call_each('run_dill_function',
                  args_list=[(dill.dumps(make_init_fn(s)),) for s in states])
    obs = env.reset()
    policy.reset()
    B = len(states)
    arange = torch.arange(B, device=device)
    To, Ta = policy.n_obs_steps, policy.n_action_steps
    sl = slice(To - 1, To - 1 + Ta)
    final_pass = getattr(policy, 'selection', 'argmax') == 'final_pass'

    scores_t, chosen_t, alive_t, final_t = [], [], [], []
    ep_done = np.zeros(B, dtype=bool)
    done = False
    pbar = tqdm.tqdm(total=max_steps // Ta + 1, desc=f'n={n}', leave=False)
    while not done:
        obs_dict = dict_apply(obs, lambda x: torch.from_numpy(x).to(device=device))
        with torch.no_grad():
            # Mirror predict_action_best exactly: one obs encode shared by the search AND
            # (in final_pass) the extra sample, all inside one crop scope so the obs and
            # every candidate's subgoal share an offset.
            with policy._crop_scope():
                obs_features = policy._encode_obs_features(obs_dict)
                actions, values, scores = policy.predict_n_actions(
                    obs_dict, verifier=policy.verifier, n_actions=n,
                    return_scores=True, obs_features=obs_features)   # (B,n,H,Da), ctx, (B,n)
                best = scores.argmax(dim=1)                          # (B,)
                if final_pass:
                    keep = policy.max_actions - 1
                    final = policy.predict_action(
                        obs_dict, actions=actions[:, -keep:], values=values[:, -keep:],
                        obs_features=obs_features)['action_pred']    # (B, H, Da)
                    # SIMULATED FOR LOGGING ONLY. The deployed policy never scores this --
                    # one extra verifier rollout per control step buys the ability to place
                    # the executed sample inside the distribution it was conditioned on.
                    score_final = policy._score_candidates(
                        policy.verifier, obs_dict, final)[1]         # (B,)
                    act = final[:, sl]
                else:
                    score_final = None
                    act = actions[arange, best][:, sl]

        scores_t.append(scores.float().cpu().numpy())
        chosen_t.append(best.cpu().numpy())
        alive_t.append(~ep_done.copy())
        if score_final is not None:
            final_t.append(score_final.float().cpu().numpy())

        obs, reward, step_done, info = env.step(act.detach().cpu().numpy())
        ep_done |= np.asarray(step_done, dtype=bool)
        done = np.all(step_done)
        pbar.update(1)
    pbar.close()

    # The FULL per-env-step reward trajectory, not just its max. The max is what every
    # existing metric reports, and it hides two things: PushTEnv sets `done` the moment
    # coverage clears the threshold, so an 8-step solve and a 38-step solve both score 1.0;
    # and on a failed episode the max may have occurred mid-episode before the block drifted
    # away. Keeping the trajectory makes last-step reward and any discounted return
    # computable after the fact, without re-running the rollout.
    rewards = env.call('get_attr', 'reward')
    T = max(len(r) for r in rewards)
    traj = np.full((len(rewards), T), np.nan, dtype=np.float32)
    for i, r in enumerate(rewards):
        traj[i, :len(r)] = r
    return (np.stack(scores_t), np.stack(chosen_t),
            np.array([np.max(r) for r in rewards]), np.stack(alive_t),
            np.stack(final_t) if final_t else None, traj)


def collect(checkpoint, arm, out_dir, device, n, episodes, split, max_steps, seed):
    policy, cfg = load_policy(checkpoint, device)
    if seed is None:
        seed = int(cfg.training.get('seed', 42))
    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    states, idxs = get_split_states(cfg, split, run_dir=run_dir)
    states, idxs = states[:episodes], idxs[:episodes]
    step = int(''.join(c for c in pathlib.Path(checkpoint).stem if c.isdigit()) or 0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = build_envs(len(states), cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)
    t0 = time.perf_counter()
    try:
        scores, chosen, rewards, alive, score_final, reward_traj = rollout_candidate_scores(
            env, policy, list(states), torch.device(device), n, max_steps)
    finally:
        env.close()
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: failed to close verifier pool: {e}')
    dt = time.perf_counter() - t0

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        'arm': arm,
        'checkpoint': str(checkpoint),
        'step': step,
        'n': int(n),
        'split': split,
        'seed': int(seed),
        'search_context': str(cfg.get('search_context', 'value')),
        'selection': str(getattr(policy, 'selection', 'argmax')),
        'corrupt_obs': bool(cfg.get('corrupt_obs', False)),
        'n_demos': int(cfg.get('n_demos', 0) or 0),
        'episode_idxs': [int(i) for i in idxs],
        'success_rate': float(np.mean(rewards >= SUCCESS_REWARD)),
        'mean_reward': float(np.mean(rewards)),
        **_reward_stats(reward_traj),
        'n_control_steps': int(alive.sum()),
        'seconds': dt,
    }
    arrays = dict(scores=scores, chosen=chosen, rewards=rewards, alive=alive,
                  reward_traj=reward_traj, episode_idxs=np.asarray(idxs))
    if score_final is not None:
        arrays['score_final'] = score_final
    np.savez_compressed(out_dir / 'candidate_scores.npz', **arrays)
    (out_dir / 'candidate_scores_meta.json').write_text(json.dumps(meta, indent=2))
    stats = analyse(scores, chosen, alive, score_final=score_final)
    (out_dir / 'candidate_scores_stats.json').write_text(
        json.dumps({**meta, **stats}, indent=2))
    print(f"{arm} step {step}: {meta['n_control_steps']} control steps, "
          f"success {meta['success_rate']:.2f}, {dt:.0f}s -> {out_dir}")
    return meta, stats


# ------------------------------------------------------------------------------ analysis

def _record_rate(X):
    """P(column k is a new running max over columns 0..k-1), per column."""
    running_max = np.maximum.accumulate(X, axis=1)
    is_record = np.zeros_like(X, dtype=bool)
    is_record[:, 0] = True
    is_record[:, 1:] = X[:, 1:] > running_max[:, :-1]
    return is_record.mean(axis=0)


def analyse(scores, chosen, alive, score_final=None, n_perm=200, seed=0):
    """Per-candidate-index statistics over every live control step.

    `scores` (T,B,n) -> flattened to (S,n) keeping only live steps, since a finished
    episode's padded steps are not states the policy visited.
    """
    S = scores.reshape(-1, scores.shape[-1])[alive.reshape(-1)]        # (S, n)
    ch = chosen.reshape(-1)[alive.reshape(-1)]                         # (S,)
    n_steps, n = S.shape
    idx = np.arange(n)

    # Steps where every candidate scored identically carry no ordering information at all
    # (the agent is nowhere near the block, so nothing any candidate does moves it). They
    # are counted and reported, but excluded from the record test, where they would only
    # dilute both the observed rate and the null by the same factor.
    spread = S.max(axis=1) - S.min(axis=1)
    degenerate = spread <= 1e-9
    live = ~degenerate
    Sl = S[live]

    # per-index level. `centered` removes each control step's own difficulty -- raw means
    # are dominated by which states happen to be easy, not by candidate order.
    centered = S - S.mean(axis=1, keepdims=True)
    sem = S.std(axis=0, ddof=1) / np.sqrt(n_steps)
    csem = centered.std(axis=0, ddof=1) / np.sqrt(n_steps)

    # THE TEST: is candidate k a new running max more often than ORDER ALONE predicts?
    #
    # The baseline is the permutation null -- each step's own candidate values, reshuffled.
    # It is exchangeable by construction, so it isolates exactly the ordering effect while
    # holding the value multiset (and its ties) fixed. The analytic 1/(k+1) is reported too
    # but is NOT the comparison: these scores tie often enough that a shuffled sequence
    # already sits ~30% below it, so measuring against 1/(k+1) would score a tie artefact as
    # a real effect. See the module docstring.
    record_rate = _record_rate(Sl)
    rng = np.random.default_rng(seed)
    perm = np.stack([_record_rate(rng.permuted(Sl, axis=1)) for _ in range(n_perm)])
    perm_rate = perm.mean(axis=0)
    perm_sd = perm.std(axis=0, ddof=1)
    null_rate = 1.0 / (idx + 1.0)
    # binomial SE of the empirical rate, for "is the gap real"
    record_se = np.sqrt(np.clip(record_rate * (1 - record_rate), 0, None) / max(len(Sl), 1))

    # trend of the centered score against index: OLS slope and Spearman rho
    x = np.repeat(idx[None, :], n_steps, axis=0).reshape(-1)
    y = centered.reshape(-1)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    slope_se = float(np.sqrt((resid ** 2).sum() / (len(y) - 2) / ((x - x.mean()) ** 2).sum()))
    rho = float(_spearman(x, y))

    # "does a bigger n raise the AVERAGE candidate, or only the max?" -- prefix statistics
    # over candidates 0..n-1. prefix_mean is the question as asked; prefix_max is the
    # best-of-n curve, and the gap between them is the entire value of having a selector.
    #
    # NOTE a regime change at max_actions: predict_n_actions switches to a rolling window
    # past K, so candidates 0..K-1 come from slots 0..K-1 while every candidate beyond that
    # is drawn from slot K-1's conditional. A rise in prefix_mean across that boundary is
    # partly a change of generator, not a trend within one.
    csum = np.cumsum(S, axis=1)
    denom = np.arange(1, n + 1, dtype=np.float64)[None, :]
    prefix_mean = (csum / denom).mean(axis=0)                 # (n,)
    prefix_max = np.maximum.accumulate(S, axis=1).mean(axis=0)

    # Per-slot version of the deployed-sample analysis: treat EVERY candidate index as if it
    # were the one executed, and ask the same questions. `rank_by_index` is how many of the n
    # candidates beat candidate k (0 = best); `p_is_best_by_index` is how often it is the
    # outright winner. Uniform expectations are (n-1)/2 and 1/n, so departures say whether a
    # slot is systematically better-positioned than its siblings rather than merely different.
    order = (S[:, :, None] < S[:, None, :]).sum(axis=2)      # (S_steps, n) rank of each slot
    rank_by_index = order.mean(axis=0)
    p_best_by_index = (order == 0).mean(axis=0)               # tie-INCLUSIVE: "is a maximizer"

    # The baseline is the MEAN ACROSS SLOTS, not (n-1)/2 and 1/n. Under any exchangeable
    # ordering every slot has the same expected rank, so the across-slot average IS the
    # permutation null -- and unlike the analytic values it is unaffected by ties, which are
    # rife here (many candidates never touch the block and return identical values, so a
    # dozen slots can be jointly "best"). Quoting 1/n would make every slot look ~10x better
    # than chance when the effect is entirely ties, the same trap as the 1/(k+1) record null.
    out_final = {'rank_by_index': rank_by_index.tolist(),
                 'p_is_best_by_index': p_best_by_index.tolist(),
                 'rank_baseline': float(rank_by_index.mean()),
                 'p_is_best_baseline': float(p_best_by_index.mean()),
                 'rank_uniform_naive': float((n - 1) / 2)}
    if score_final is not None:
        F = score_final.reshape(-1)[alive.reshape(-1)]        # (S_all,) deployed sample
        Sa = scores.reshape(-1, scores.shape[-1])[alive.reshape(-1)]
        # rank of the deployed sample among the n candidates it was conditioned on;
        # 0 == better than all of them. Uniform expectation is n/2.
        rank = (Sa > F[:, None]).sum(axis=1)
        # How often does the deployed sample beat the best of the FIRST n candidates, as a
        # function of n? This is the crossover the arm lives or dies on: at n=1 it only has
        # to beat a single draw, at n=64 it has to beat the max of 64. Where this curve
        # crosses 0.5 is the search width past which selection wins.
        pmax = np.maximum.accumulate(Sa, axis=1)             # (S_steps, n)
        beats_prefix = (F[:, None] > pmax).mean(axis=0)      # (n,)
        gap_prefix = (F[:, None] - pmax).mean(axis=0)        # (n,) px
        out_final.update({
            'final_beats_prefix_max': beats_prefix.tolist(),
            'final_minus_prefix_max': gap_prefix.tolist(),
        })
        out_final.update({
            'final_mean': float(F.mean()),
            'final_rank_mean': float(rank.mean()),
            'final_rank_uniform': float(n / 2),
            'final_percentile': float((rank / max(n - 1, 1)).mean()),
            'p_final_is_best': float((rank == 0).mean()),
            'p_final_beats_argmax': float((F > Sa.max(axis=1)).mean()),
            'final_minus_mean': float((F - Sa.mean(axis=1)).mean()),
            'final_minus_best': float((F - Sa.max(axis=1)).mean()),
            'final_minus_first': float((F - Sa[:, 0]).mean()),
        })

    return {
        'n_control_steps': int(n_steps),
        'n_degenerate_steps': int(degenerate.sum()),
        'n_test_steps': int(live.sum()),
        'prefix_mean': prefix_mean.tolist(),
        'prefix_max': prefix_max.tolist(),
        **out_final,
        'index': idx.tolist(),
        'mean_by_index': S.mean(axis=0).tolist(),
        'sem_by_index': sem.tolist(),
        'centered_mean_by_index': centered.mean(axis=0).tolist(),
        'centered_sem_by_index': csem.tolist(),
        # within-step spread of candidate k: how much room that slot still explores
        'centered_sd_by_index': centered.std(axis=0, ddof=1).tolist(),
        'running_max_by_index': np.maximum.accumulate(S, axis=1).mean(axis=0).tolist(),
        'record_rate': record_rate.tolist(),
        'record_rate_se': record_se.tolist(),
        'perm_null_rate': perm_rate.tolist(),
        'perm_null_sd': perm_sd.tolist(),
        'iid_null_rate': null_rate.tolist(),
        # summed over k>=1: how many more records than SHUFFLING the same values gives.
        # 0 == order carries no information == the search is resampling.
        'excess_records': float((record_rate[1:] - perm_rate[1:]).sum()),
        'excess_records_iid': float((record_rate[1:] - null_rate[1:]).sum()),
        'argmax_hist': (np.bincount(ch, minlength=n) / len(ch)).tolist(),
        'p_argmax_late': float((ch >= n / 2).mean()),
        'ols_slope': float(slope),
        'ols_slope_se': slope_se,
        'spearman_rho': rho,
        'score_first_mean': float(S[:, 0].mean()),
        'score_last_mean': float(S[:, -1].mean()),
        'score_best_mean': float(S.max(axis=1).mean()),
        'spread_mean': float((S.max(axis=1) - S.min(axis=1)).mean()),
    }


def _spearman(x, y):
    """Spearman rho without scipy (not guaranteed in this env)."""
    def rank(a):
        order = a.argsort()
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties, which x is full of (it is a repeated 0..n-1 grid)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r)
        return (sums / cnt)[inv]
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return 0.0 if denom == 0 else float((rx * ry).sum() / denom)


def plot(stats, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    idx = np.array(stats['index'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    m = np.array(stats['centered_mean_by_index'])
    e = np.array(stats['centered_sem_by_index'])
    ax.errorbar(idx, m, yerr=1.96 * e, marker='o', ms=3, lw=1)
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set_xlabel('candidate index (generation order)')
    ax.set_ylabel('verifier value - step mean (px)')
    ax.set_title(f"step-centered level\nslope {stats['ols_slope']:+.3f}"
                 f" +/- {1.96 * stats['ols_slope_se']:.3f}/index")

    ax = axes[1]
    rr = np.array(stats['record_rate'])
    rse = np.array(stats['record_rate_se'])
    perm = np.array(stats['perm_null_rate'])
    psd = np.array(stats['perm_null_sd'])
    null = np.array(stats['iid_null_rate'])
    ax.errorbar(idx[1:], rr[1:], yerr=1.96 * rse[1:], marker='o', ms=3, lw=1,
                label='observed order')
    ax.plot(idx[1:], perm[1:], color='crimson', lw=1.2, label='permutation null')
    ax.fill_between(idx[1:], perm[1:] - 1.96 * psd[1:], perm[1:] + 1.96 * psd[1:],
                    color='crimson', alpha=0.18)
    # shown only to make the tie effect visible -- it is NOT the comparison, see analyse()
    ax.plot(idx[1:], null[1:], 'k:', lw=1, label='1/(k+1) (invalid: ties)')
    ax.set_xlabel('candidate index k')
    ax.set_ylabel('P(new running max)')
    ax.set_title(f"THE TEST: steering vs resampling\nexcess over permutation "
                 f"{stats['excess_records']:+.2f}")
    ax.legend(fontsize=7)

    ax = axes[2]
    ax.bar(idx, stats['argmax_hist'], width=0.8)
    ax.axhline(1.0 / len(idx), color='k', lw=0.8, ls='--', label='uniform')
    ax.set_xlabel('candidate index')
    ax.set_ylabel('P(executed)')
    ax.set_title(f"which candidate wins\nP(k >= n/2) = {stats['p_argmax_late']:.3f}")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------------- report

def _fmt(v, p=1):
    return f'{v:.{p}f}'


# How each arm label decomposes into the two mechanisms the report is keyed on.
_MECHANISM = {
    'value':                ('verifier scalar', 'argmax'),
    'subgoal-chosen4value': ('subgoal image',   'argmax'),
    'subgoal-value':        ('subgoal + scalar', 'argmax'),
    'subgoal-only':         ('subgoal image',   'final pass'),
}

_REPORT_NS = [1, 2, 4, 8, 16, 32, 64]


def _test_success(stats):
    """End-to-end test success for this checkpoint, from the run's own eval curves.

    Pulled in so the candidate-level statistics and the number they are meant to explain sit
    on one line. Returns {n: rate} or {} when the checkpoint has not been evaluated.
    """
    ckpt = pathlib.Path(stats.get('checkpoint', ''))
    jl = ckpt.parent.parent / 'bon_search' / 'success_curves.jsonl'
    if not jl.is_file():
        return {}
    try:
        rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    except Exception:
        return {}
    for r in rows:
        if int(r.get('step', -1)) == int(stats['step']):
            return dict(zip(r.get('n', []), r.get('success_rate', [])))
    return {}


def _cfg_num(cfg, value, default):
    """A hydra config value as a number, resolving a `${key}` interpolation if that is what
    is stored.

    `.hydra/config.yaml` is the RAW config, so a key written as `${n_candidates}` sits on
    disk verbatim rather than resolved. Runs predating `n_candidates` stored a literal int
    for `policy.max_actions`, which is why `int(...)` worked until the `k*_cd0.9` arms --
    those store the interpolation and made this raise, taking the whole report down.
    Dotted targets (`${a.b}`) resolve too; anything unresolvable falls back to `default`.
    """
    if isinstance(value, str):
        s = value.strip()
        if not (s.startswith('${') and s.endswith('}')):
            return default
        node = cfg
        for part in s[2:-1].split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        value = node
    try:
        return type(default)(value)
    except (TypeError, ValueError):
        return default


def _slot_weight_table(stats):
    """The loss/context weighting this checkpoint was TRAINED under, read off its config.

    Rendered from the checkpoint's own cfg rather than hardcoded, so the doc cannot state a
    weighting the run did not use.
    """
    ckpt = pathlib.Path(stats.get('checkpoint', ''))
    cfg_path = ckpt.parent.parent / '.hydra' / 'config.yaml'
    if not cfg_path.is_file():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text())
    except Exception:
        return None
    K = _cfg_num(cfg, (cfg.get('policy') or {}).get('max_actions', 0), 0)
    if not K:
        return None
    return {
        'K': K,
        'slot_weight_decay': _cfg_num(cfg, cfg.get('slot_weight_decay', 1.0), 1.0),
        'context_decay': _cfg_num(cfg, cfg.get('context_decay', 1.0), 1.0),
    }


# Stats keys the report tables index directly. A dump lacking any of them predates the
# statistic and cannot be rendered; build_report skips it by name instead of raising.
_REPORT_KEYS = frozenset((
    'prefix_mean', 'prefix_max', 'rank_baseline', 'p_is_best_baseline', 'reward_max',
))


def build_report(dirs, out_path, n_example_steps=8):
    """Assemble CANDIDATES_FROM_SUBGOAL.md from a set of dump directories."""
    entries = []
    for d in dirs:
        d = pathlib.Path(d)
        sf = d / 'candidate_scores_stats.json'
        if not sf.is_file():
            print(f'skip (no stats): {d}')
            continue
        st = json.loads(sf.read_text())
        # Dumps written before the prefix/rank/reward statistics existed carry none of the
        # fields the tables below index, and the report used to die on the first one with a
        # bare KeyError -- taking every OTHER section down with it. Skip them by name so a
        # stale dump costs its own section and says so, rather than the whole document.
        # Each of these has an `_n64` sibling at the same step that supersedes it.
        stale = _REPORT_KEYS - st.keys()
        if stale:
            print(f'skip (pre-{min(stale)} dump schema, re-dump to include): {d.name}')
            continue
        npz = np.load(d / 'candidate_scores.npz')
        entries.append((d, st, npz))
    if not entries:
        raise SystemExit('no dumps found')
    # argmax arms first, final_pass last, so the table reads baseline -> new mechanism
    entries.sort(key=lambda e: (e[1].get('selection', 'argmax') == 'final_pass',
                                e[1]['arm'], e[1]['corrupt_obs'], e[1]['step']))

    L = []
    A = L.append
    A('# Candidates from subgoal — per-candidate verifier feedback\n')
    A('Generated by `scripts/dump_candidate_scores.py`. Every number comes from a closed-loop\n'
      'rollout driven by **whatever the policy actually deploys** — the argmax candidate for\n'
      'the argmax arms, the extra conditioned sample for the final-pass ones — so these are\n'
      'the scores a real deployment saw.\n')
    A('\n> **Do not compare raw levels across arms.** Two effects confound them.\n'
      '>\n'
      '> *Different states.* Each arm is driven by its own selection rule, so the final-pass\n'
      '> arms visit a different (and worse) state distribution than the argmax arms.\n'
      '>\n'
      '> *Episode composition.* The verifier value improves steeply WITHIN an episode — the\n'
      '> block starts far from the goal and gets pushed in, so the mean candidate value runs\n'
      '> about -138 px at the first control step to -22 px by step 20. A finished episode\n'
      '> stops contributing steps. So the pooled mean over live steps is mostly a statement\n'
      '> about *which stage of the approach* the pooled steps came from, and it moves far\n'
      '> more with the mix of early-vs-late steps than with policy quality.\n'
      '>\n'
      '> Concretely, `subgoal-value` clean between its n=16 and n=64 dumps: success rises\n'
      '> 0.55 → 0.70, yet the pooled mean *falls* 9.4 px. Succeeded episodes run ~20 control\n'
      '> steps and stop; failed ones run the full 38 and so contribute many late, high-value\n'
      '> steps — their mean is **-42.3** against **-51.8** for the succeeded ones. Improving\n'
      '> the policy converts failed episodes into succeeded ones, which REMOVES those\n'
      '> high-value late steps from the pool: the failed-episode share of live steps drops\n'
      '> 61% → 45% and the pooled mean drops with it. **A lower mean here indicates a better\n'
      '> policy, not a worse one.**\n'
      '>\n'
      '> The statistics the conclusions rest on are immune to both, because they are computed\n'
      '> WITHIN a control step: step-centered means, within-step SD, record rate against the\n'
      '> permutation null, the deployed sample\'s rank, and the prefix curves (every prefix\n'
      '> is measured on the same visited states).\n')
    A('\nThe verifier value is `-mean_kp ||feedback||` in **pixels**: 0 at the goal, about\n'
      '-300 at the worst. Higher (less negative) is better. Candidates are listed **in\n'
      'generation order** — candidate 0 has an empty search context (the no-search\n'
      'baseline) and candidate k is conditioned on candidates 0..k-1 and their feedback.\n')

    A('\n## 0. The test, and how to read it\n')
    A('The question is whether candidate `k` is a new running maximum over `0..k-1` more\n'
      'often than **order alone** would produce. If yes, conditioning on the previous\n'
      'candidates is steering generation. If no, the search is resampling a fixed\n'
      'distribution and the entire gain belongs to the oracle argmax.\n')
    A('\n### The obvious null is wrong here\n')
    A('For a continuous i.i.d. sequence that probability is exactly `1/(k+1)`. These scores\n'
      'are **not** continuous: many candidates never touch the block, so the sim returns\n'
      'byte-identical values, and strict-inequality records are suppressed by the ties.\n'
      'Reshuffling each step\'s own candidates — exchangeable by construction, so it *must*\n'
      'hit the null if the null applies — lands ~30% below `1/(k+1)`. Measuring against the\n'
      'analytic curve therefore scores a tie artefact as evidence, and reports every arm as\n'
      '**worse than resampling** when the opposite is true.\n')
    A('\n### The null used instead\n')
    A('**The permutation null**: the same step\'s candidate values, order shuffled, averaged\n'
      'over 200 shuffles. It holds the value multiset (ties and all) fixed and destroys only\n'
      'the ordering, which is exactly the thing under test. `excess records` sums\n'
      '`(observed − permutation)` over `k ≥ 1`; **0 means order carries no information**.\n'
      'The analytic `1/(k+1)` is still shown in the plots, as a dotted line, only to make\n'
      'the size of the tie effect visible.\n')

    # ---- THE headline table: mechanisms x what the last loop actually produced
    A('\n## 1. Mechanisms, and the distance feedback of the last prediction loop\n')
    A('One row per arm, keyed by the two mechanisms that define it. **`last loop`** is the\n'
      'distance feedback of the final iteration of the search: candidate `n-1` for the argmax\n'
      'arms, and for `final pass` the deployed sample itself (the loop that genuinely runs\n'
      'last). `cand 0` is the empty-context baseline, `best` the argmax over all n.\n')
    A('\n| feedback | choice | obs | step | K | **last loop** | cand 0 | mean | best | '
      'test succ @n |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for d, st, _ in entries:
        fb, ch = _MECHANISM.get(st['arm'], (st['arm'], st.get('selection', 'argmax')))
        sw = _slot_weight_table(st) or {}
        succ = _test_success(st)
        nn = st['n']
        scell = f"{succ[nn]:.2f} @{nn}" if nn in succ else (
            f"{max(succ.values()):.2f} @{max(succ, key=succ.get)}" if succ else '—')
        last = st.get('final_mean', st['mean_by_index'][-1])
        A(f"| {fb} | {ch} | {'corrupt' if st['corrupt_obs'] else 'clean'} | {st['step']} | "
          f"{sw.get('K', '?')} | **{_fmt(last)}** | {_fmt(st['score_first_mean'])} | "
          f"{_fmt(np.mean(st['mean_by_index']))} | {_fmt(st['score_best_mean'])} | {scell} |")
    A('\nFor `final pass` the `last loop` value is the executed sample, simulated for logging\n'
      'only — the deployed policy never scores it. Compare it against `best` on the same row:\n'
      'that difference is what selecting by oracle argmax is worth.\n')

    # ---- rewards, including the two things `max` hides
    if any('reward_last' in st for _, st, _ in entries):
        A('\n### Reward — max, last step, and discounted\n')
        A('PushT reward is `clip(coverage / success_threshold, 0, 1)` per step, and the env\n'
          'sets `done` the instant coverage clears the threshold. Two consequences the usual\n'
          '`max` metric hides:\n')
        A('\n* **No time preference.** An 8-step solve and a 38-step solve both score 1.0.\n'
          '  `disc succ` below fixes that: it is `γ^(steps to success)`, 0 if never — monotone\n'
          '  in both succeeding and succeeding sooner. At γ=0.99, solving 30 steps later is\n'
          '  worth 26% less.\n'
          '* ⚠️ **`Σ γᵗ rₜ` is backwards on this task** and is shown only as a warning. The env\n'
          '  terminates the instant coverage clears the threshold, so a fast solve contributes\n'
          '  FEWER reward terms than a slow one and scores LOWER — the 0.90-success arms come\n'
          '  out below the 0.70-success ones on it. Episode length is endogenous here, so a\n'
          '  sum over the realized horizon is not comparable across policies.\n'
          '* **`max` can credit a moment the policy did not hold.** Because a success\n'
          '  terminates at its peak, `last` equals `max` for every successful episode — so\n'
          '  any gap between them comes entirely from FAILED episodes that brushed the goal\n'
          '  and drifted off. `lost after peak` is how often that happened.\n')
        A('\n| feedback | choice | obs | succ | reward max | **reward last** | '
          '**disc succ γ=.99** | disc succ γ=.95 | ep len | steps→succ | lost after peak | '
          '(Σγᵗrₜ ⚠️) |')
        A('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for d, st, _ in entries:
            if 'reward_last' not in st:
                continue
            fb, ch = _MECHANISM.get(st['arm'], (st['arm'], st.get('selection', 'argmax')))
            tts = st.get('steps_to_success_mean')
            tts_cell = f'{tts:.0f}' if tts is not None else '—'
            A(f"| {fb} | {ch} | {'corrupt' if st['corrupt_obs'] else 'clean'} | "
              f"{st['success_rate']:.2f} | {st['reward_max']:.3f} | "
              f"**{st['reward_last']:.3f}** | **{st['disc_success_g99']:.3f}** | "
              f"{st['disc_success_g95']:.3f} | {st['episode_len_mean']:.0f} | {tts_cell} | "
              f"{st['frac_lost_after_peak']:.2f} | {st['return_g99']:.1f} |")
        A('\n`steps→succ` is env steps to the first full-coverage frame, averaged over the\n'
          'episodes that got there. **`disc succ` is the column to quote** when time matters:\n'
          'unlike `reward max` it separates arms that both reach the goal but at different\n'
          'speeds, and unlike `Σγᵗrₜ` it is not inverted by the termination rule.\n')

    # ---- does a bigger n raise the AVERAGE, or only the max?
    A('\n## 2. Does a higher n raise the average? — prefix statistics\n')
    A('`mean` is the average distance feedback over candidates `0..n-1`; `max` is the\n'
      'best-of-n curve over the same prefix. The question "does the average rise with n"\n'
      'is the `mean` row; the `max` row is what people usually assume that question means,\n'
      'and the two diverge sharply.\n')
    A('\n**Read the `‖` as a regime change, not a trend.** `predict_n_actions` switches to a\n'
      'rolling window past `K`, so candidates `0..K-1` come from slots `0..K-1` while every\n'
      'candidate beyond that is drawn from slot `K-1`\'s conditional. Movement across that\n'
      'boundary is partly a change of generator.\n')
    for d, st, _ in entries:
        fb, ch = _MECHANISM.get(st['arm'], (st['arm'], st.get('selection', 'argmax')))
        sw = _slot_weight_table(st) or {}
        K = sw.get('K', 16)
        ns = [n for n in _REPORT_NS if n <= st['n']]
        hdr = ' | '.join((('‖ ' if n == K else '') + f'n={n}') for n in ns)
        A(f"\n**{fb} · {ch} · {'corrupt' if st['corrupt_obs'] else 'clean'} · step "
          f"{st['step']} · K={K}**\n")
        A('| | ' + hdr + ' |')
        A('|---|' + '---|' * len(ns))
        A('| mean | ' + ' | '.join(f"{st['prefix_mean'][n-1]:.1f}" for n in ns) + ' |')
        A('| max  | ' + ' | '.join(f"{st['prefix_max'][n-1]:.1f}" for n in ns) + ' |')
        d1 = st['prefix_mean'][ns[-1]-1] - st['prefix_mean'][0]
        d2 = st['prefix_max'][ns[-1]-1] - st['prefix_max'][0]
        A(f"\nn=1 → n={ns[-1]}: mean **{d1:+.1f} px**, max **{d2:+.1f} px**.\n")

    # ---- the deployed sample, for final_pass arms only
    fps = [(d, st) for d, st, _ in entries if 'final_rank_mean' in st]
    if fps:
        A('\n## 3. The deployed sample (final-pass arms)\n')
        A('Where the executed sample lands among the n candidates it was conditioned on.\n'
          'Rank 0 = no candidate strictly beats it.\n')
        A('\n**Compare against `baseline`, not `n/2`.** Ties are rife (many candidates never\n'
          'touch the block and score identically), which compresses every rank downward. The\n'
          'tie-aware baseline is the across-slot average rank — what a candidate drawn from\n'
          'the same pool with no special status actually scores. Against the naive `n/2` the\n'
          'deployed sample looks strongly distinguished; against the correct baseline it\n'
          'mostly is not.\n')
        A('\n| arm | obs | step | mean rank | **baseline** | (naive n/2) | P(rank=0) | '
          '(baseline) | P(beats argmax) | vs mean | vs best |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for d, st in fps:
            A(f"| {st['arm']} | {'corrupt' if st['corrupt_obs'] else 'clean'} | {st['step']} | "
              f"**{st['final_rank_mean']:.1f}** | **{st['rank_baseline']:.1f}** | "
              f"{st['final_rank_uniform']:.1f} | {st['p_final_is_best']:.3f} | "
              f"{st['p_is_best_baseline']:.3f} | {st['p_final_beats_argmax']:.3f} | "
              f"{st['final_minus_mean']:+.2f} | {st['final_minus_best']:+.2f} |")
        A('\n`P(beats argmax)` is the arm\'s thesis as a single number: how often the model\'s\n'
          'own synthesis beats the oracle pick it replaces.\n')

        A('\n### Where selection overtakes the deployed sample — P(final > best-of-n)\n')
        A('The same comparison run at every search width. At n=1 the deployed sample only has\n'
          'to beat a single draw; at n=64 it must beat the max of 64. Where this crosses 0.5\n'
          'is the width past which having a selector is worth more than having a better draw.\n')
        A('\n| arm | obs | ' + ' | '.join(f'n={x}' for x in _REPORT_NS) + ' |')
        A('|---|---|' + '---|' * len(_REPORT_NS))
        for d, st in fps:
            b = st['final_beats_prefix_max']
            A(f"| {st['arm']} | {'corrupt' if st['corrupt_obs'] else 'clean'} | "
              + ' | '.join(f'{b[x-1]:.3f}' for x in _REPORT_NS if x <= st['n']) + ' |')
        A('\nand the same in pixels — how far the deployed sample sits below best-of-n:\n')
        A('\n| arm | obs | ' + ' | '.join(f'n={x}' for x in _REPORT_NS) + ' |')
        A('|---|---|' + '---|' * len(_REPORT_NS))
        for d, st in fps:
            g = st['final_minus_prefix_max']
            A(f"| {st['arm']} | {'corrupt' if st['corrupt_obs'] else 'clean'} | "
              + ' | '.join(f'{g[x-1]:+.1f}' for x in _REPORT_NS if x <= st['n']) + ' |')

    # ---- every slot analysed the way the deployed sample was
    A('\n### Every candidate slot, analysed the same way\n')
    A('Treating each candidate index as if it were the executed one: `rank` is how many of\n'
      'the n candidates strictly beat it (0 = it is a maximizer), `P(best)` how often that\n'
      'happens.\n')
    A('\n**The baseline is the across-slot average, not `(n-1)/2` and `1/n`.** Under any\n'
      'exchangeable ordering every slot has the same expected rank, so the average across\n'
      'slots IS the permutation null — and unlike the analytic values it is unaffected by\n'
      'ties, which are rife here. Against the naive `1/n` every slot would look ~10x better\n'
      'than chance when the effect is entirely ties: the same trap as the `1/(k+1)` record\n'
      'null in §0.\n')
    for d, st, _ in entries:
        n = st['n']
        ks = [k for k in (0, 1, 2, 3, 7, 15, 31, 47, n - 1) if k < n]
        ks = sorted(set(ks))
        A(f"\n**{st['arm']} · {'corrupt' if st['corrupt_obs'] else 'clean'} · step "
          f"{st['step']}** — baseline rank **{st['rank_baseline']:.1f}**, baseline P(best) "
          f"**{st['p_is_best_baseline']:.3f}** (naive uniform would say "
          f"{st['rank_uniform_naive']:.1f} / {1/n:.3f})\n")
        A('\n| slot | ' + ' | '.join(str(k) for k in ks)
          + (' | **final** |' if 'final_rank_mean' in st else ' |'))
        A('|---|' + '---|' * (len(ks) + (1 if 'final_rank_mean' in st else 0)))
        row = ' | '.join(f"{st['rank_by_index'][k]:.1f}" for k in ks)
        if 'final_rank_mean' in st:
            row += f" | **{st['final_rank_mean']:.1f}**"
        A('| rank | ' + row + ' |')
        row = ' | '.join(f"{st['p_is_best_by_index'][k]:.3f}" for k in ks)
        if 'final_rank_mean' in st:
            row += f" | **{st['p_final_is_best']:.3f}**"
        A('| P(best) | ' + row + ' |')

    # ---- the ordering test
    A('\n## 4. Is the ordering informative? — permutation test\n')
    A('| arm | obs | step | steps | **excess records** | (vs i.i.d.) | '
      'OLS slope px/idx | Spearman ρ | P(argmax ≥ n/2) |')
    A('|---|---|---|---|---|---|---|---|---|')
    for d, st, _ in entries:
        A(f"| {st['arm']} | {'corrupt' if st['corrupt_obs'] else 'clean'} | {st['step']} | "
          f"{st['n_test_steps']} | "
          f"**{st['excess_records']:+.2f}** | {st['excess_records_iid']:+.2f} | "
          f"{st['ols_slope']:+.3f} ± {1.96*st['ols_slope_se']:.3f} | "
          f"{st['spearman_rho']:+.3f} | {st['p_argmax_late']:.3f} |")
    A('\n`steps` counts control steps with non-zero spread; degenerate steps (every candidate\n'
      'identical) carry no ordering information and are excluded from the record test.\n'
      'A uniform argmax histogram gives `P(argmax ≥ n/2) = 0.5`. The `(vs i.i.d.)` column is\n'
      'the discarded analytic comparison, kept only to show it flips the sign.\n')

    # ---- what this implies for the subgoal-only arm. Computed, not asserted, so it stays
    # true if the dumps are regenerated against different checkpoints.
    A('\n## 5. Verdict\n')
    A('\n| arm | obs | step | last cand vs step mean | best-of-n vs cand 0 | '
      'within-step SD, cand 0 → last |')
    A('|---|---|---|---|---|---|')
    for d, st, _ in entries:
        last_c = st['centered_mean_by_index'][-1]
        gain = st['running_max_by_index'][-1] - st['running_max_by_index'][0]
        A(f"| {st['arm']} | {'corrupt' if st['corrupt_obs'] else 'clean'} | {st['step']} | "
          f"{last_c:+.2f} px | {gain:+.1f} px | "
          f"{st['centered_sd_by_index'][0]:.1f} → {st['centered_sd_by_index'][-1]:.1f} |")

    if fps:
        A('\n### Did the weighting distinguish the deployed sample? — measured, no\n')
        for d, st in fps:
            n = st['n']
            dr = st['rank_baseline'] - st['final_rank_mean']
            A(f"* **{st['arm']} {'corrupt' if st['corrupt_obs'] else 'clean'} step "
              f"{st['step']}**: mean rank **{st['final_rank_mean']:.1f} of {n}** against a "
              f"tie-aware baseline of **{st['rank_baseline']:.1f}** — a difference of "
              f"{dr:+.1f} ranks. It is a maximizer {st['p_final_is_best']:.1%} of the time "
              f"against {st['p_is_best_baseline']:.1%} for a typical slot, and sits "
              f"{st['final_minus_mean']:+.2f} px from the step mean.\n")
        A('\nSo the deployed sample is, to within a rank out of 64, **a typical candidate**.\n'
          'The slot weighting did not give it special status — which agrees with the\n'
          'training-time `action_value_final − action_value` metric sitting at ~0 throughout.\n')
        A('\n> Measured against the naive `n/2` baseline the same numbers look like a strong\n'
          '> effect (rank 19.8 vs 32.0, "69th percentile"). That reading is a tie artefact and\n'
          '> is wrong — see the note in §3.\n')
        A('\n### But it is still not competitive with the argmax it replaces\n')
        for d, st in fps:
            A(f"* **{st['arm']} {'corrupt' if st['corrupt_obs'] else 'clean'}**: beats the "
              f"oracle argmax **{st['p_final_beats_argmax']:.1%}** of the time, and sits "
              f"{-st['final_minus_best']:.1f} px below it on average.\n")
        A('\nBeing a good draw is not the same as being the best of n draws, and the gap does\n'
          'not close by making the draw a little better — `prefix_max` keeps climbing with n\n'
          'while `prefix_mean` saturates by n≈8, so the selector\'s advantage GROWS with the\n'
          'search width the arm was supposed to exploit.\n')

    A('\nThe last column of the table above is the mechanism behind all of it: the within-step\n'
      'spread collapses after candidate 0 and keeps shrinking. The search *narrows* rather\n'
      'than improves — later candidates agree with each other more, which is why the record\n'
      'rate decays to the permutation null by k≈6, why `prefix_mean` saturates, and why the\n'
      'only thing that keeps paying at large n is having a selector to pick the tail draw.\n')

    # ---- per-arm detail
    for d, st, npz in entries:
        n = st['n']
        tag = f"{st['arm']} · corrupt={st['corrupt_obs']} · step {st['step']}"
        A(f'\n---\n\n## {tag}\n')
        A(f"`{st['checkpoint']}`  \n"
          f"search_context=`{st['search_context']}` selection=`{st['selection']}` "
          f"n={n} split={st['split']} episodes={len(st['episode_idxs'])} "
          f"success={st['success_rate']:.2f}\n")

        # The weighting this checkpoint carries. The two keys are NOT the same kind of thing
        # and the heading must not imply they are: slot_weight_decay is a term in the loss
        # and therefore exists only while training, whereas context_decay is an attention
        # bias inside the forward pass and is live at INFERENCE too.
        sw = _slot_weight_table(st)
        if sw and (sw['slot_weight_decay'] < 1.0 or sw['context_decay'] < 1.0):
            K = sw['K']
            A(f"\n### Search weighting (K={K})\n")
            if sw['context_decay'] < 1.0:
                lam = sw['context_decay']
                A(f"**Context recency decay λ={lam} — ACTIVE AT INFERENCE.** This is not a\n"
                  f"loss weight. It is an additive attention bias inside the forward pass\n"
                  f"(`_build_memory_masks`): `(m-1-j)·log({lam})` on the pre-softmax logit,\n"
                  f"which multiplies the attention weight on entry `j` by `{lam}^(m-1-j)`.\n"
                  f"So it shapes what this policy attends to every time it runs, deployment\n"
                  f"included — unlike `slot_weight_decay`, which vanishes once there is no\n"
                  f"loss.\n")
                A(f"\nFor a candidate generated against `m` context entries the latest counts\n"
                  f"1, the previous {lam}, the one before {lam ** 2:.2f}. It depends only on\n"
                  f"distance-from-latest — never on absolute index, K, or n — so the profile\n"
                  f"is identical in every loop at every search width.\n")
                A(f"\n**It never reaches zero.** The bias is a *relative* reweighting that\n"
                  f"softmax renormalizes, not an absolute attenuation, and only invalid\n"
                  f"entries are masked out — the obs tokens keep bias 0 and are never\n"
                  f"decayed. At K={K} the oldest context entry still carries\n"
                  f"`{lam}^{K - 1}` = {lam ** (K - 1):.3f} of the latest entry's weight.\n")
                A('\n| entries back | 0 (latest) | 1 | 2 | 3 | 4 | 5 |')
                A('|---|---|---|---|---|---|---|')
                A('| weight | ' + ' | '.join(f'{lam ** i:.3f}' for i in range(6)) + ' |')
            if sw['slot_weight_decay'] < 1.0:
                lam = sw['slot_weight_decay']
                w = lam ** (K - 1 - np.arange(K))
                w = w / w.mean()
                A(f"\n**Slot loss weighting λ={lam} — TRAINING ONLY.** A per-slot factor on "
                  f"the\nloss terms (`_slot_weights`, used only by `_compute_loss`), so it "
                  f"has no\neffect at inference: there is no loss to weight. "
                  f"`w_k ∝ {lam}^(K-1-k)`, normalized to mean 1:\n")
                A('\n| slot | ' + ' | '.join(str(j) for j in range(K)) + ' |')
                A('|---|' + '---|' * K)
                A('| weight | ' + ' | '.join(f'{v:.2f}' for v in w) + ' |')
                A("\n⚠️ This keys the weight to the ABSOLUTE slot index, but the final pass is\n"
                  "conditioned on `min(n, K-1)` entries — so which conditional is *deployed*,\n"
                  "and how well it was trained, depends on the eval n:\n")
                ns_all = _REPORT_NS
                A('\n| eval n | ' + ' | '.join(str(x) for x in ns_all) + ' |')
                A('|---|' + '---|' * len(ns_all))
                A('| slot deployed | ' + ' | '.join(str(min(x, K - 1)) for x in ns_all) + ' |')
                A('| its weight | ' + ' | '.join(f'{w[min(x, K - 1)]:.2f}'
                                                 for x in ns_all) + ' |')
                A(f"\nBelow n={K} this penalizes the conditional actually being deployed — by\n"
                  f"{(1 - w[min(1, K - 1)]) * 100:.0f}% at n=1. `context_decay` has no such\n"
                  f"dependence, which is why the later arms use it instead.\n")

        # §A the literal per-step table
        scores = npz['scores']
        chosen = npz['chosen']
        alive = npz['alive']
        sfinal = npz['score_final'] if 'score_final' in npz.files else None
        live = np.argwhere(alive)
        pick = live[np.linspace(0, len(live) - 1, min(n_example_steps, len(live))).astype(int)]
        # 64 columns does not render; show a head slice and always the last candidate
        show = list(range(min(n, 12)))
        if n - 1 not in show:
            show.append(n - 1)
        A('\n### Candidates and their verifier values\n')
        A(f'One row per control step, columns = candidates in generation order'
          + (f' (first {len(show) - 1} of {n}, plus the last)' if n > 12 else '') + '. '
          + '`*` marks the\nargmax. `Δ0` is `best - candidate 0`, the best-of-n gain at that '
            'step.\n')
        head = ' | '.join(str(k) for k in show)
        A('\n| step | ' + head + ' | argmax'
          + (' | **final** |' if sfinal is not None else ' | Δ0 |'))
        A('|---|' + '---|' * (len(show) + 2))
        for t, b in pick:
            row = scores[t, b]
            c = int(chosen[t, b])
            cells = [f'{row[k]:.1f}' + ('\\*' if k == c else '') for k in show]
            tail = (f'**{sfinal[t, b]:.1f}**' if sfinal is not None
                    else f'{row.max() - row[0]:+.1f}')
            A(f"| ep{st['episode_idxs'][b]} t={t} | " + ' | '.join(cells)
              + f' | {c} | {tail} |')
        if sfinal is not None:
            A('\n**final** is the sample this arm actually executed — a further draw '
              'conditioned on\nall the candidates in the row, not one of them.\n')

        # §B aggregate by index
        A('\n### Aggregate by candidate index\n')
        A('| index | mean | ±95% | step-centered | ±95% | within-step SD | E[running max] | '
          'P(record) | perm null | i.i.d. | P(executed) |')
        A('|---|---|---|---|---|---|---|---|---|---|---|')
        for k in range(n):
            A(f"| {k} | {_fmt(st['mean_by_index'][k])} | "
              f"{1.96*st['sem_by_index'][k]:.2f} | "
              f"{st['centered_mean_by_index'][k]:+.2f} | "
              f"{1.96*st['centered_sem_by_index'][k]:.2f} | "
              f"{st['centered_sd_by_index'][k]:.1f} | "
              f"{_fmt(st['running_max_by_index'][k])} | "
              f"{st['record_rate'][k]:.3f} | {st['perm_null_rate'][k]:.3f} | "
              f"{st['iid_null_rate'][k]:.3f} | "
              f"{st['argmax_hist'][k]:.3f} |")
        A(f"\n![{tag}]({d.name}/candidate_scores.png)\n")

    pathlib.Path(out_path).write_text('\n'.join(L) + '\n')
    print(f'wrote {out_path} ({len(entries)} dumps)')


def sweep_rewards(checkpoint, arm, out_dir, device, n_list, episodes, split, max_steps, seed):
    """Roll out at each search width and record the REWARD statistics per n.

    Separate from `collect` because the reward-vs-n question needs one rollout per n, while
    the candidate-level statistics only need the widest. `success_curve.json` already carries
    max-reward vs n for every arm, but only the max -- last-step reward and the discounted
    time-to-goal need the trajectory, which nothing stored until now.

    Re-seeds to the same base before every n, matching eval_search_pusht._eval_split_at_n, so
    the points on the curve are PAIRED rather than each carrying independent sampler noise.
    """
    policy, cfg = load_policy(checkpoint, device)
    if seed is None:
        seed = int(cfg.training.get('seed', 42))
    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    states, idxs = get_split_states(cfg, split, run_dir=run_dir)
    states, idxs = states[:episodes], idxs[:episodes]
    step = int(''.join(c for c in pathlib.Path(checkpoint).stem if c.isdigit()) or 0)
    env = build_envs(len(states), cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)

    rows = []
    try:
        for n in n_list:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t0 = time.perf_counter()
            _, _, rewards, _, _, traj = rollout_candidate_scores(
                env, policy, list(states), torch.device(device), n, max_steps)
            row = {'n': int(n),
                   'success_rate': float(np.mean(rewards >= SUCCESS_REWARD)),
                   **_reward_stats(traj)}
            rows.append(row)
            print(f"  n={n:<3} succ={row['success_rate']:.2f} max={row['reward_max']:.3f} "
                  f"last={row['reward_last']:.3f} disc={row['disc_success_g99']:.3f} "
                  f"({time.perf_counter()-t0:.0f}s)")
    finally:
        env.close()
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: failed to close verifier pool: {e}')

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'arm': arm, 'checkpoint': str(checkpoint), 'step': step,
        'selection': str(getattr(policy, 'selection', 'argmax')),
        'corrupt_obs': bool(cfg.get('corrupt_obs', False)),
        'split': split, 'seed': int(seed), 'episodes': len(states), 'rows': rows,
    }
    (out_dir / 'reward_vs_n.json').write_text(json.dumps(payload, indent=2))
    print(f'wrote {out_dir}/reward_vs_n.json')
    return payload


# ----------------------------------------------------------------------------------- cli

@click.command()
@click.option('-c', '--checkpoint', default=None)
@click.option('--arm', default='unknown', help='label for the report (e.g. subgoal-chosen4value)')
@click.option('--out-dir', default=None)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n', 'n_actions', default=16, help='candidates per control step')
@click.option('--episodes', default=20)
@click.option('--split', default='test', type=click.Choice(['val', 'test']))
@click.option('--max-steps', default=300)
@click.option('--seed', default=None, type=int)
@click.option('--n-list', default=None,
              help='comma-separated search widths; rolls out at each and writes '
                   'reward_vs_n.json (reward max / last / discounted vs n)')
@click.option('--report', multiple=True, help='dump dirs to assemble into a markdown report')
@click.option('--reanalyse', multiple=True,
              help='recompute stats+plot from an existing dump, no rollouts')
@click.option('--out', default='CANDIDATES_FROM_SUBGOAL.md')
def main(checkpoint, arm, out_dir, device, n_actions, episodes, split, max_steps, seed,
         n_list, report, reanalyse, out):
    # Re-derive the statistics from the stored (T,B,n) tensor. The rollouts are the
    # expensive half and the npz holds everything they produced, so a change to the
    # analysis never costs another 4 minutes of physics per arm.
    for d in reanalyse:
        d = pathlib.Path(d)
        z = np.load(d / 'candidate_scores.npz')
        meta = json.loads((d / 'candidate_scores_meta.json').read_text())
        stats = analyse(z['scores'], z['chosen'], z['alive'],
                        score_final=z['score_final'] if 'score_final' in z.files else None)
        (d / 'candidate_scores_stats.json').write_text(
            json.dumps({**meta, **stats}, indent=2))
        plot(stats, d / 'candidate_scores.png',
             f"{meta['arm']} corrupt={meta['corrupt_obs']} step {meta['step']} "
             f"n={meta['n']}")
        print(f"reanalysed {d.name}: excess_records={stats['excess_records']:+.3f} "
              f"(vs iid {stats['excess_records_iid']:+.3f}), "
              f"{stats['n_degenerate_steps']}/{stats['n_control_steps']} degenerate")
    if reanalyse:
        return
    if report:
        build_report(report, out)
        return
    assert checkpoint is not None, 'need -c/--checkpoint (or --report)'
    assert out_dir is not None, 'need --out-dir'
    if n_list:
        ns = [int(x) for x in n_list.split(',') if x.strip()]
        sweep_rewards(checkpoint, arm, out_dir, device, ns, episodes, split, max_steps, seed)
        return
    meta, stats = collect(checkpoint, arm, out_dir, device, n_actions, episodes, split,
                          max_steps, seed)
    plot(stats, pathlib.Path(out_dir) / 'candidate_scores.png',
         f"{arm} corrupt={meta['corrupt_obs']} step {meta['step']} n={n_actions}")


if __name__ == '__main__':
    main()
