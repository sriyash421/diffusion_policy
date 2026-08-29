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

if __name__ == "__main__":
    # `python scripts/dump_candidate_scores.py` puts scripts/ on sys.path, not the repo
    # root, so `import diffusion_policy` raised ModuleNotFoundError unless the caller
    # remembered PYTHONPATH=$PWD. Same preamble render_search_videos.py uses; it makes the
    # command in the docs work as written.
    import os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

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
    SUCCESS_REWARD, build_envs, get_split_states, load_policy, wilson_interval,
    _episode_seed)


def resolved_verifier_value(policy, cfg):
    """Which VALUE_FNS key this rollout actually scores with.

    Recorded in every artifact because the scoring rule changed on 2026-08-19 (`t_goal` ->
    `armT`, adding an arm-to-T approach term) and the two are NOT comparable: `t_goal` is
    flat across candidates until the arm touches the block, `armT` is not. A checkpoint
    saved before the cutover carries no `verifier_value` key, so it resolves to the
    pre-cutover default and is scored on exactly what it was trained on -- but an artifact
    that does not SAY so cannot be told apart from one produced under armT.
    """
    from diffusion_policy.env.pusht.pusht_verifier import DEFAULT_VALUE_FN, VALUE_FNS
    mode = None
    try:
        mode = cfg.policy.get('verifier_value', None)
    except Exception:
        pass
    mode = str(mode or DEFAULT_VALUE_FN)
    assert mode in VALUE_FNS, f'unknown verifier_value {mode!r}'
    return mode


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

    Returns (scores, chosen, rewards, alive, score_final, reward_traj, terms) where
      scores  (T, B, n)  raw verifier value, candidate IN GENERATION ORDER
      chosen  (T, B)     argmax index -- what an argmax policy would have executed
      rewards (B,)       episode max reward
      alive   (T, B)     bool, False once that env's episode has ended
      score_final (T, B) the DEPLOYED sample's verifier value under `selection: final_pass`,
                  else None
      terms   (T, B, n, 2) raw-PIXEL [d_T->goal, d_arm->T], or None if the verifier does
                  not decompose its value (only PushT's does)

    WHY `terms` IS RECORDED. Under `armTn` the score is -(d_T->goal/13.6 + d_arm->T/52.1):
    two different distances, summed after each is divided by its own within-step candidate
    spread. A slot can therefore out-score another purely by parking the arm nearer the T
    while making no task progress at all -- the exact failure mode that retired the raw
    `armT` value (it cost the UNet BC arm its best-of-n gain, 0.460 -> 0.060 at n=8). Any
    claim that candidate k beats candidate 0 is uninterpretable without the split, so the
    terms are recorded at rollout time rather than reconstructed later: they come straight
    out of the verifier that produced the score, so the two cannot drift apart.

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

    scores_t, chosen_t, alive_t, final_t, terms_t = [], [], [], [], []
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
                # optional returns are APPENDED to the tuple, never inserted; with
                # return_subgoals off the terms land in slot 3.
                actions, values, scores, terms = policy.predict_n_actions(
                    obs_dict, verifier=policy.verifier, n_actions=n,
                    return_scores=True, obs_features=obs_features,
                    return_terms=True)          # (B,n,H,Da), ctx, (B,n), (B,n,2)|None
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
        if terms is not None:
            terms_t.append(terms.float().cpu().numpy())
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
            np.stack(final_t) if final_t else None, traj,
            np.stack(terms_t) if terms_t else None)


def write_per_step_json(out_dir, meta, scores, chosen, alive, reward_traj, idxs,
                        n_action_steps, score_final=None, max_episodes=0, terms=None):
    """Objective 2a: every candidate's distance value at every control step of a rollout.

    One file per episode. Written for ALL episodes by default because the arrays are
    already in memory and the whole set costs ~1 MB -- picking "the one episode" before
    seeing the data would be choosing the example to fit the story.

    Both `value` (verifier units, higher is better, 0 is perfect) and `neg_value`
    (= -value, lower is better) are stored. They are the same number; the pair exists so a
    reader never has to remember which way the sign runs.

    `neg_value` REPLACES the old `distance_px` key, which was a misnomer under any value but
    `t_goal`. Under `armTn` the score is -(d_t_goal/13.6 + d_arm_t/52.1) -- a sum of two
    distances each divided by its own within-step candidate spread -- so its negation is a
    normalized composite running to about 32, not a pixel distance. The actual pixel
    distances are `d_t_goal` and `d_arm_t`, present whenever the verifier decomposes.
    """
    per = pathlib.Path(out_dir) / 'per_step'
    per.mkdir(parents=True, exist_ok=True)
    T, B, n = scores.shape
    Ta = int(n_action_steps)
    written = []
    limit = B if not max_episodes else min(B, int(max_episodes))
    for b in range(limit):
        steps = []
        for t in range(T):
            if not bool(alive[t, b]):
                continue
            v = scores[t, b].astype(np.float64)
            mx = float(v.max())
            tie = v >= (mx - TIE_TAU)
            order = np.sort(v)
            env_step = t * Ta
            # reward at the END of this decision's action window, clipped to the episode
            row = reward_traj[b]
            fin = np.flatnonzero(~np.isnan(row))
            last = int(fin[-1]) if fin.size else 0
            rw = float(row[min(env_step + Ta - 1, last)]) if fin.size else float('nan')
            steps.append({
                't': int(t), 'env_step': int(env_step), 'alive': True,
                'value': [round(float(x), 4) for x in v],
                'neg_value': [round(float(-x), 4) for x in v],
                'argmax_first': int(v.argmax()),
                'argmax_last': int(n - 1 - tie[::-1].argmax()),
                'n_argmax_tied': int(tie.sum()),
                'argmax_margin_px': round(float(order[-1] - order[-2]), 6) if n >= 2 else 0.0,
                'degenerate': bool(mx - float(v.min()) <= BLIND_EPS),
                'executed': int(chosen[t, b]),
                'executed_value': round(float(v[int(chosen[t, b])]), 4),
                'final_slot_value': round(float(v[-1]), 4),
                'min': round(float(v.min()), 4), 'max': round(mx, 4),
                'mean': round(float(v.mean()), 4),
                'spread': round(float(mx - v.min()), 4),
                'reward': round(rw, 4) if rw == rw else None,
            })
            if score_final is not None:
                steps[-1]['deployed_final_value'] = round(float(score_final[t, b]), 4)
            if terms is not None:
                # RAW PIXELS, per candidate. Kept separate from `value` because a slot can
                # out-score another on the arm-reach term alone while making no task
                # progress, and the composite cannot show that.
                steps[-1]['d_t_goal'] = [round(float(x), 4) for x in terms[t, b, :, 0]]
                steps[-1]['d_arm_t'] = [round(float(x), 4) for x in terms[t, b, :, 1]]

        deg = np.array([x['degenerate'] for x in steps], dtype=bool)
        tied = np.array([x['n_argmax_tied'] > 1 for x in steps], dtype=bool)
        first = np.array([x['argmax_first'] for x in steps], dtype=float)
        uniq = ~tied
        traj = reward_traj[b][~np.isnan(reward_traj[b])]
        doc = {
            # v2: `distance_px` -> `neg_value` (the old name was a misnomer under armTn),
            # plus per-candidate `d_t_goal` / `d_arm_t` in raw pixels.
            'schema': 'pusht_candidate_values/v2',
            **{k: meta[k] for k in ('arm', 'checkpoint', 'step', 'split', 'n',
                                    'max_actions', 'selection', 'seed', 'paired_seeds',
                                    'slot_semantics', 'slot_index_meaningful', 'slot_note',
                                    'n_obs_steps', 'n_action_steps') if k in meta},
            'split_pos': int(b), 'episode_idx': int(idxs[b]),
            'episode_seed': int(meta['episode_seeds'][b]) if meta.get('episode_seeds') else None,
            'value_units': f"verifier value under {meta.get('verifier_value', '?')}; "
                           'higher is better, 0 is perfect. Under armTn this is '
                           '-(d_t_goal/13.6 + d_arm_t/52.1) -- a spread-normalized '
                           'composite, NOT a pixel distance.',
            'neg_value_units': '= -value; lower is better. Same units as `value`.',
            'term_units': ('d_t_goal, d_arm_t: raw pixels, per candidate'
                           if terms is not None else None),
            'alive_note': 'alive is captured BEFORE the env step, so the terminal decision '
                          'of a successful episode is included.',
            'steps': steps,
            'summary': {
                'n_decisions': len(steps), 'n_alive': len(steps),
                'success': bool(traj.max() >= SUCCESS_REWARD) if traj.size else False,
                'max_reward': float(traj.max()) if traj.size else 0.0,
                'n_degenerate_steps': int(deg.sum()),
                'n_tied_argmax_steps': int(tied.sum()),
                'argmax_slot_mean_first': float(first.mean()) if len(first) else None,
                'argmax_slot_median_first': float(np.median(first)) if len(first) else None,
                'argmax_slot_mean_unique': float(first[uniq].mean()) if uniq.any() else None,
                'argmax_slot_median_unique': float(np.median(first[uniq])) if uniq.any() else None,
                'n_unique_steps': int(uniq.sum()),
            },
        }
        name = f'ep{b:02d}_idx{int(idxs[b])}.json'
        (per / name).write_text(json.dumps(doc, indent=2))
        written.append(name)
    print(f'  per-step JSON: {len(written)} episodes -> {per}')
    return written


def collect(checkpoint, arm, out_dir, device, n, episodes, split, max_steps, seed,
            per_step_json=False, per_step_episodes=0, pair_seeds=True,
            verifier_value=None):
    policy, cfg = load_policy(checkpoint, device)
    # SCORING RULE OVERRIDE, so arms trained under different verifier eras can be compared.
    #
    # The value is a scoring instrument, not trained weights -- it is what RANKS candidates
    # -- so it is swappable on a loaded checkpoint. Same mechanism eval_search_pusht.py's
    # --verifier-value uses, deliberately: one implementation, not two that drift.
    #
    # THIS CHANGES THE ROLLOUT, and that is not a side effect to gloss over. The executed
    # candidate is the argmax under this value, so a different value picks a different
    # action and the trajectory diverges. A dump written with an override is a measurement
    # of "this policy scored/selected under THAT rule", which is exactly what makes a
    # t_goal-era arm usable as a null for armTn arms -- and is not the same experiment as
    # its native dump. Both the native and the used value are recorded below so the two can
    # never be silently mixed.
    native_value = resolved_verifier_value(policy, cfg)
    if verifier_value is not None:
        search_kwargs = getattr(policy, '_search_kwargs', None)   # UNet BC arm
        target = search_kwargs if search_kwargs is not None else policy.kwargs
        target['verifier_value'] = verifier_value
        # The transformer arms build the verifier in __init__, so theirs must be swapped in
        # place. The UNet arm builds lazily (to keep training from forking a 32-process sim
        # pool) and will read the kwarg above when it does -- so do not touch its `verifier`
        # property here, which would force that fork now.
        built = (policy.__dict__.get('_verifier') if search_kwargs is not None
                 else getattr(policy, 'verifier', None))
        if built is not None:
            built.value_fn = verifier_value
    if seed is None:
        seed = int(cfg.training.get('seed', 42))
    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    states, idxs = get_split_states(cfg, split, run_dir=run_dir)
    states, idxs = states[:episodes], idxs[:episodes]
    step = int(''.join(c for c in pathlib.Path(checkpoint).stem if c.isdigit()) or 0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Per-EPISODE noise streams keyed on the episode's position in the split -- the same key
    # _eval_split_at_n uses -- so an episode sees the same trajectory noise under every arm
    # and at every checkpoint, and the arms are paired episode-by-episode rather than merely
    # both random. This CHANGES the numbers relative to dumps written before it existed
    # (those drew from the unseeded global RNG), which is why `paired_seeds` is recorded and
    # why paired and unpaired dumps must not be compared.
    episode_seeds = None
    if pair_seeds:
        seeder = getattr(policy, 'set_sample_seeds', None)
        if seeder is not None:
            episode_seeds = [_episode_seed(seed, n, i) for i in range(len(states))]
            seeder(episode_seeds)
        else:
            print('warning: policy has no set_sample_seeds; --pair-seeds is a no-op')
    env = build_envs(len(states), cfg.policy.n_obs_steps, cfg.policy.n_action_steps, max_steps)
    t0 = time.perf_counter()
    try:
        (scores, chosen, rewards, alive, score_final, reward_traj,
         terms) = rollout_candidate_scores(
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
    slot_sem, slot_note = _slot_semantics(getattr(policy, 'max_actions', 1), n)
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
        'max_actions': int(getattr(policy, 'max_actions', 0) or 0),
        'n_obs_steps': int(cfg.policy.n_obs_steps),
        'n_action_steps': int(cfg.policy.n_action_steps),
        'paired_seeds': bool(episode_seeds is not None),
        'episode_seeds': episode_seeds,
        'slot_semantics': slot_sem,
        'slot_index_meaningful': slot_sem == 'trained_staircase',
        'slot_note': slot_note,
        # what this dump was SCORED with, which is what every downstream comparison keys on
        'verifier_value': verifier_value or native_value,
        # what the checkpoint's own config says, and whether they differ. An artifact that
        # does not say it was overridden cannot be told apart from a native one.
        'verifier_value_native': native_value,
        'verifier_value_overridden': bool(
            verifier_value is not None and verifier_value != native_value),
        'success_rate': float(np.mean(rewards >= SUCCESS_REWARD)),
        'mean_reward': float(np.mean(rewards)),
        **_reward_stats(reward_traj),
        'n_control_steps': int(alive.sum()),
        # Whether the value could be split into its distance components. False for any
        # verifier that does not decompose; downstream term analysis must skip those dumps
        # rather than silently reporting nothing.
        'has_terms': terms is not None,
        'term_names': ['d_t_goal_px', 'd_arm_t_px'] if terms is not None else None,
        'term_units': ('raw pixels, NOT the verifier value. armTn ranks on '
                       '-(d_t_goal/13.6 + d_arm_t/52.1), so `value` is a normalized '
                       'composite of these two and is not in pixels.'
                       if terms is not None else None),
        'seconds': dt,
    }
    arrays = dict(scores=scores, chosen=chosen, rewards=rewards, alive=alive,
                  reward_traj=reward_traj, episode_idxs=np.asarray(idxs))
    if score_final is not None:
        arrays['score_final'] = score_final
    if terms is not None:
        arrays['terms'] = terms                      # (T, B, n, 2) raw px
    np.savez_compressed(out_dir / 'candidate_scores.npz', **arrays)
    (out_dir / 'candidate_scores_meta.json').write_text(json.dumps(meta, indent=2))
    if per_step_json:
        write_per_step_json(out_dir, meta, scores, chosen, alive, reward_traj, idxs,
                            cfg.policy.n_action_steps, score_final=score_final,
                            max_episodes=per_step_episodes, terms=terms)
    stats = analyse(scores, chosen, alive, score_final=score_final, seed=seed)
    (out_dir / 'candidate_scores_stats.json').write_text(
        json.dumps({**meta, **stats}, indent=2))
    print(f"{arm} step {step}: {meta['n_control_steps']} control steps, "
          f"success {meta['success_rate']:.2f}, {dt:.0f}s -> {out_dir}")
    print(f"  verifier value: {meta['verifier_value']}"
          + (f" (OVERRIDDEN; checkpoint's own is {native_value})"
             if meta['verifier_value_overridden'] else ' (from checkpoint cfg)'))
    print(f"  slot semantics: {slot_sem}  |  blind {stats['blind_rate_overall']:.1%} of live "
          f"steps, {stats['frac_blind_in_leading_run']:.0%} of those in the leading run")
    print(f"  argmax slot (discriminating steps): unique "
          f"{stats.get('argmax_slot_mean_unique', float('nan')):.2f} "
          f"| first {stats.get('argmax_slot_mean_first', float('nan')):.2f} "
          f"| perm null {stats.get('argmax_slot_perm_null', float('nan')):.2f} "
          f"| uniform {stats.get('argmax_slot_uniform_null', float('nan')):.2f}")
    return meta, stats


# ------------------------------------------------------------------------------ analysis

# The per-slot statistics live in diffusion_policy/common/slot_stats_util so the
# read-only reporting scripts (scripts/slot_context_analysis.py) can share these exact
# definitions without importing the simulator through this module. Imported here rather
# than defined here, and re-exported, because several call sites reach for them by name.
from diffusion_policy.common.slot_stats_util import (   # noqa: E402
    BLIND_EPS, TIE_TAU, _argmax_slot_stats, _blindness_profile, _record_rate,
    _slot_semantics, _spearman, analyse,
)

def plot(stats, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    idx = np.array(stats['index'])
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

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

    ax = axes[3]
    br = stats.get('blind_rate_by_step_index')
    if br is not None:
        t = np.arange(len(br))
        ax.plot(t, br, marker='o', ms=2.5, lw=1, color='darkorange')
        ax.axhline(stats['blind_rate_overall'], color='k', lw=0.9, ls='--',
                   label=f"overall {stats['blind_rate_overall']:.1%}")
        ax.set_ylim(0, 1)
        ax.set_xlabel('control step index')
        ax.set_ylabel('P(all n candidates identical)')
        # frac_in_leading_run is the number that kills the "blind only during the approach"
        # reading: measured ~0.17, i.e. ~83% of blind steps are MID-EPISODE contact losses.
        ax.set_title('verifier blind: no candidate touches the T\n'
                     f"{stats['frac_blind_in_leading_run']:.0%} of blind steps are in the "
                     f"leading run")
        ax.legend(fontsize=8)
    else:
        ax.set_axis_off()

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
    # Two config surfaces: the nested `slot_weights` block (current) and the legacy
    # `slot_weight_decay` scalar. Prefer the block when it names a real profile, and fall
    # back to the scalar otherwise, so a checkpoint from either generation renders. Wrapped
    # because .hydra/config.yaml stores the RAW config -- nested values may be unresolved
    # `${...}` strings, and a raise here took the whole report down once before.
    sw = cfg.get('slot_weights') or {}
    mode = sw.get('mode') if isinstance(sw, dict) else None
    if isinstance(mode, str) and mode.startswith('${'):
        mode = None
    sn = cfg.get('slot_loss_norm') or {}
    norm_mode = sn.get('mode') if isinstance(sn, dict) else None
    if isinstance(norm_mode, str) and norm_mode.startswith('${'):
        norm_mode = None
    out = {
        'K': K,
        'slot_weight_decay': _cfg_num(cfg, cfg.get('slot_weight_decay', 1.0), 1.0),
        'context_decay': _cfg_num(cfg, cfg.get('context_decay', 1.0), 1.0),
        'slot_weights_mode': mode or ('geometric'
                                      if cfg.get('slot_weight_decay') not in (None, False, 1.0)
                                      else 'uniform'),
        # Orthogonal to the weights: which norm each slot's term used. Same defensive read
        # -- a raw '${...}' means the config never resolved, not that the run used it.
        'slot_loss_norm_mode': norm_mode or 'l2',
    }
    if isinstance(sw, dict):
        for k, default in (('decay', 0.0), ('ratio', 0.0)):
            v = sw.get(k)
            if v is not None:
                out[f'slot_weights_{k}'] = _cfg_num(cfg, v, default)
        if sw.get('schedule'):
            out['slot_weights_schedule'] = str(sw.get('schedule'))
    return out


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
    A('\nThe verifier value is in **pixels**, ≤ 0, higher (less negative) is better. Which\n'
      'value depends on the run (`pusht_verifier.VALUE_FNS`):\n'
      '`t_goal` = `-mean_kp||feedback||` (0 on goal, ~-300 worst) — every run before\n'
      '2026-08-19; `armT` = `-(mean_kp||feedback|| + ||agent - T_centre||)` (0 only with the\n'
      'T on goal AND the arm at its centre, ~-800 worst) — runs after. The two are not\n'
      'comparable; the run directory carries `ver-<value>` from the cutover onward.\n'
      'Candidates are listed **in generation order** — candidate 0 has an empty search\n'
      'context (the no-search baseline) and candidate k is conditioned on candidates\n'
      '0..k-1 and their feedback.\n')

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
    # ---- which slot the argmax lands on, and whether that beats the i.i.d. null
    if any('argmax_slot_mean_unique' in st for _, st, _ in entries):
        A('\n## 3b. Which candidate slot does the argmax land on?\n')
        A('Scoped to **discriminating** control steps -- those where the candidates do not\n'
          'all score identically. On a blind step every slot is a maximizer and `np.argmax`\n'
          'returns 0 every time, so including them measures numpy\'s tiebreak rule rather\n'
          'than the policy.\n')
        A('\n**Two columns, two different questions.** `first` is `np.argmax`, which is what\n'
          '`select_candidate` uses, so it is literally the slot the deployed policy executed\n'
          '-- but it is a fact about the tiebreak rule wherever candidates tie. `unique`\n'
          'restricts to steps with a single maximizer and answers which slot genuinely wins.\n')
        A('\n**Compare against `perm null`, never against `uniform`.** The permutation null\n'
          'reshuffles each step\'s own value multiset, holding the ties fixed and destroying\n'
          'only the ordering -- the one thing under test. Ties are rife here, so a shuffled\n'
          'sequence already departs from `(n-1)/2`.\n')
        A('\n⚠️ **The i.i.d. arms are not baselines OF this statistic -- they ARE its null\n'
          'distribution**, measured on real data with the real tie structure. UNet BC discards\n'
          'the search context and ST k=1 has no context capacity, so neither has any reason to\n'
          'prefer a slot. If a k>1 arm sits inside their band, the search context does not move\n'
          'the argmax slot.\n')
        A('\n| arm | step | slot semantics | n | **unique** | first | **perm null** | uniform | tied | blind |')
        A('|---|---|---|---|---|---|---|---|---|---|')
        for d, st, _ in entries:
            if 'argmax_slot_mean_unique' not in st:
                continue
            sem = st.get('slot_semantics', '?')
            note = '' if sem == 'trained_staircase' else ' *(null)*'
            A(f"| {st['arm']} | {st['step']} | {sem}{note} | {st['n']} | "
              f"**{st['argmax_slot_mean_unique']:.2f}** | "
              f"{st['argmax_slot_mean_first']:.2f} | "
              f"**{st['argmax_slot_perm_null']:.2f}** | "
              f"{st['argmax_slot_uniform_null']:.1f} | "
              f"{st['frac_steps_tied']:.0%} | "
              f"{st.get('blind_rate_overall', float('nan')):.0%} |")
        A('\nA `unique` value ABOVE `perm null` means later slots win more often than order\n'
          'alone predicts; BELOW means earlier slots do. Either is a finding; equality means\n'
          'the slot index carries no information.\n')

        # blindness, which bounds what best-of-n can buy no matter how good the candidates are
        if any('blind_rate_overall' in st for _, st, _ in entries):
            A('\n### How often the verifier cannot discriminate at all\n')
            A('The value is a function of where the T ends up, so when no candidate contacts\n'
              'the block every candidate returns the identical current distance and the search\n'
              'carries zero information. `in leading run` is the fraction of those steps that\n'
              'sit in the episode\'s opening run: **a low number means blindness is a\n'
              'mid-episode phenomenon** (the agent backing off, repositioning, circling the T),\n'
              'not an approach-phase artifact that training will remove.\n')
            A('\n| arm | step | blind | in leading run | mean blind/episode | median spread when discriminating |')
            A('|---|---|---|---|---|---|')
            for d, st, _ in entries:
                if 'blind_rate_overall' not in st:
                    continue
                A(f"| {st['arm']} | {st['step']} | **{st['blind_rate_overall']:.1%}** | "
                  f"{st['frac_blind_in_leading_run']:.0%} | "
                  f"{st['blind_total_mean']:.1f} | "
                  f"{st['spread_median_discriminating']:.1f} px |")

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
        norm_mode = (sw or {}).get('slot_loss_norm_mode', 'l2')
        if sw and (sw['slot_weight_decay'] < 1.0 or sw['context_decay'] < 1.0
                   or norm_mode != 'l2'):
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
            if norm_mode != 'l2':
                # A norm per slot, not a scale per slot -- it changes the SHAPE of each
                # term, so it is not comparable with the weights above and gets its own
                # table. Training only, like the slot weights.
                al = (np.arange(K) / max(K - 1, 1) if norm_mode == 'l2tol1'
                      else np.ones(K))
                A(f"\n**Slot loss norm `{norm_mode}` — TRAINING ONLY.** Slot `k`'s term is\n"
                  f"`(1-α_k)·(pred-target)² + α_k·|pred-target|` (`_slot_norm_alphas`), so\n"
                  f"α is the L1 fraction: 0 = pure L2, 1 = pure L1. Not renormalized — it\n"
                  f"picks a norm rather than splitting a fixed loss budget.\n")
                A('\n| slot | ' + ' | '.join(str(j) for j in range(K)) + ' |')
                A('|---|' + '---|' * K)
                A('| α (L1) | ' + ' | '.join(f'{v:.3f}' for v in al) + ' |')
                A('| 1-α (L2) | ' + ' | '.join(f'{1 - v:.3f}' for v in al) + ' |')

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

        # The same numbers transposed -- slot across the columns, metric down the rows. This
        # orientation answers "how does the value move ALONG the slots", which is the question
        # the decay arms exist to ask; the per-index table above answers "what is slot k like".
        # Wide searches are truncated exactly as the per-step table is: 64 columns will not
        # render, so show a head slice and always the last slot.
        cols = list(range(min(12, n))) + ([n - 1] if n > 12 else [])
        A('\n### Value by candidate slot\n')
        A('Same statistics as above, transposed. The verifier value is in PIXELS and less '
          'negative\nis better; see the header for which of `t_goal` / `armT` this run '
          'used.\n')
        A('\n| metric | ' + ' | '.join(f'slot {k}' for k in cols) + ' |')
        A('|---|' + '---|' * len(cols))
        for label, fn in (
            ('mean (px)',      lambda k: _fmt(st['mean_by_index'][k])),
            ('±95%',           lambda k: f"{1.96 * st['sem_by_index'][k]:.2f}"),
            ('step-centered',  lambda k: f"{st['centered_mean_by_index'][k]:+.2f}"),
            ('±95%',           lambda k: f"{1.96 * st['centered_sem_by_index'][k]:.2f}"),
            ('within-step SD', lambda k: f"{st['centered_sd_by_index'][k]:.1f}"),
            ('E[running max]', lambda k: _fmt(st['running_max_by_index'][k])),
            ('P(record)',      lambda k: f"{st['record_rate'][k]:.3f}"),
            ('perm null',      lambda k: f"{st['perm_null_rate'][k]:.3f}"),
            ('i.i.d.',         lambda k: f"{st['iid_null_rate'][k]:.3f}"),
            ('P(executed)',    lambda k: f"{st['argmax_hist'][k]:.3f}"),
        ):
            A(f'| {label} | ' + ' | '.join(fn(k) for k in cols) + ' |')

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
            _, _, rewards, _, _, traj, _ = rollout_candidate_scores(
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
@click.option('--per-step-json/--no-per-step-json', default=False,
              help="write per_step/ep<NN>_idx<M>.json: every candidate's distance value at "
                   'every control step of the rollout')
@click.option('--per-step-episodes', default=0,
              help='cap how many episodes get a per-step file; 0 = all')
@click.option('--verifier-value', default=None,
              help='score with this VALUE_FNS key instead of the one in the checkpoint cfg '
                   '(e.g. armTn). The value RANKS candidates, so this changes which one is '
                   'executed and therefore the trajectory -- it is how a pre-cutover arm is '
                   'made comparable to armTn arms, not a cosmetic relabel. Both the used '
                   'and the native value are recorded in the meta.')
@click.option('--pair-seeds/--no-pair-seeds', default=True,
              help='give each episode its own noise stream keyed on its split position, so '
                   'arms are paired episode-by-episode. Changes the numbers relative to '
                   'dumps predating it -- see `paired_seeds` in the meta.')
def main(checkpoint, arm, out_dir, device, n_actions, episodes, split, max_steps, seed,
         n_list, report, reanalyse, out, per_step_json, per_step_episodes, pair_seeds,
         verifier_value):
    # Re-derive the statistics from the stored (T,B,n) tensor. The rollouts are the
    # expensive half and the npz holds everything they produced, so a change to the
    # analysis never costs another 4 minutes of physics per arm.
    for d in reanalyse:
        d = pathlib.Path(d)
        z = np.load(d / 'candidate_scores.npz')
        meta = json.loads((d / 'candidate_scores_meta.json').read_text())
        stats = analyse(z['scores'], z['chosen'], z['alive'],
                        score_final=z['score_final'] if 'score_final' in z.files else None,
                        seed=int(meta.get('seed', 0) or 0))
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
                          max_steps, seed, per_step_json=per_step_json,
                          per_step_episodes=per_step_episodes, pair_seeds=pair_seeds,
                          verifier_value=verifier_value)
    plot(stats, pathlib.Path(out_dir) / 'candidate_scores.png',
         f"{arm} corrupt={meta['corrupt_obs']} step {meta['step']} n={n_actions}")


if __name__ == '__main__':
    main()
