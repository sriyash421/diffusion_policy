"""Does the armTn verifier rank the EXPERT action above the policy's own candidates?

Best-of-n search is only as good as the thing doing the ranking. Every result in
slot_context_aug27_tier*.md measures how candidates compare TO EACH OTHER under the
verifier; none of them asks whether the verifier's ordering tracks "good action" at all.
This script asks that directly, against the only ground truth available: the recorded
demonstration.

THE SETUP. At a recorded demo timestep i, the expert action chunk a* is well defined --
which is exactly why the measurement is taken here and not on a policy rollout. Once the
policy diverges from the demo there is no expert action for the state it is actually in, so
a per-state comparison there would be a claim about a different state (the same reason
render_search_videos draws the whole demo path rather than a per-state arrow).

  * observations come straight from the zarr, so they are the frames the expert actually saw
  * a* is laid out exactly like `action_pred`, so `_verifier_inputs` slices the SAME window
    from it that it slices from a candidate: action[:, To-1 : To-1+Ta]
  * a* is scored by `_score_candidates` -- the same call, the same verifier, the same sim
    pool that scores the candidates. Not a reimplementation.

WHAT IT REPORTS. a*'s rank among the n candidates, how often it is the outright best, and
the value gap to the best and mean candidate -- then the same three under each raw distance
term separately. The split is the point: armTn is -(d_t_goal/13.6 + d_arm_t/52.1), and the
expert routinely swings the arm AROUND the T to set up the next push, which raises d_arm_t
while lowering d_t_goal. If a* ranks well on d_t_goal and badly on the composite, the
verifier is penalising the expert for repositioning, and best-of-n is optimising away from
the demonstrated behaviour rather than toward it.

WHAT IT DOES NOT SHOW. This is the expert's state distribution, which is the TRAINING
distribution -- not the states a policy actually visits at eval, which drift away from it.
A verifier that ranks a* well here can still rank badly off-distribution; a verifier that
ranks a* badly here is broken in the easiest case available.

    python scripts/verifier_ranks_expert.py -c <ckpt> --arm lin100-l2 --n 16 \
        --episodes 20 --per-episode 8
"""
import json
import pathlib
import sys

import click
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from diffusion_policy.common.replay_buffer import ReplayBuffer        # noqa: E402
from diffusion_policy.env.pusht.feedback_util import (                # noqa: E402
    compute_feedback_from_pose)
from eval_search_pusht import get_split_states, load_policy           # noqa: E402
from scripts.dump_candidate_scores import resolved_verifier_value     # noqa: E402

T_GOAL, ARM_T = 0, 1
TERM = {T_GOAL: 'd_t_goal', ARM_T: 'd_arm_t'}


def sample_points(rb, ep_idxs, per_episode, To, H, rng):
    """(episode, absolute index) pairs where a full obs window AND a full a* chunk exist.

    Both windows are required intact rather than padded: the dataset pads a short tail by
    repeating the last action, and a repeated action is not something the expert did -- it
    would put a synthetic chunk on the ground-truth side of the comparison.
    """
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    out = []
    for e in ep_idxs:
        s, t = int(starts[e]), int(ends[e])
        lo, hi = s + To - 1, t - H + To - 1      # i-To+1 >= s  and  i-To+1+H <= t
        if hi <= lo:
            continue
        k = min(per_episode, hi - lo)
        out += [(int(e), int(i)) for i in rng.choice(np.arange(lo, hi), size=k, replace=False)]
    return out


def build_batch(rb, pts, To, H, device):
    """obs_dict (B,To,...) and the expert chunk (B,H,2), aligned as `action_pred` is.

    Layout matters and is not free choice: `_verifier_inputs` reads obs step To-1 as "now"
    and simulates action[To-1 : To-1+Ta]. Writing the chunk as action[i-To+1 : i-To+1+H]
    puts the expert's command AT state i exactly where a candidate's executed window sits,
    so the two are scored over the same timesteps.
    """
    img = np.asarray(rb['img'])
    ap = np.asarray(rb['agent_pos'])
    ac = np.asarray(rb['action'])
    bp = np.asarray(rb['block_pos'])
    im, pos, fb, star = [], [], [], []
    for _, i in pts:
        j = i - To + 1
        # Built exactly as PushTImageDataset._sample_to_data builds it -- same moveaxis,
        # same /255, same compute_feedback_from_pose on the stored block_pos. The obs dict
        # must be normalizer-COMPLETE (the encoder KeyErrors on a missing low-dim key) and
        # must carry no extra keys, so this mirrors that construction rather than inventing
        # a parallel one that could drift from what the policy was trained on.
        im.append(np.moveaxis(img[j:i + 1], -1, 1) / 255)
        pos.append(ap[j:i + 1])
        fb.append(compute_feedback_from_pose(bp[j:i + 1].astype(np.float32)))
        star.append(ac[j:j + H])
    t = lambda x: torch.tensor(np.stack(x), dtype=torch.float32, device=device)  # noqa: E731
    return {'image': t(im), 'agent_pos': t(pos), 'feedback': t(fb)}, t(star)


def _stats(star, cand, higher_is_better, ep_ids, n_boot=4000, seed=0):
    """a* against the n candidates, with a cluster bootstrap over episodes.

    `higher_is_better` flips the raw distance terms so "a* wins" always means "a* is the
    better action" whichever quantity is read.
    """
    s = star if higher_is_better else -star                 # (S,)
    c = cand if higher_is_better else -cand                 # (S, n)
    beats = (s[:, None] > c)                                # (S, n)
    rank = (~beats).sum(axis=1)                             # 0 == better than every candidate
    rows = np.stack([rank == 0, rank, s - c.max(axis=1), s - c.mean(axis=1),
                     beats.mean(axis=1)], axis=1)           # (S, 5)
    eps = np.unique(ep_ids)
    sums = np.stack([rows[ep_ids == e].sum(axis=0) for e in eps])
    cnts = np.array([(ep_ids == e).sum() for e in eps], dtype=float)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(eps), np.full(len(eps), 1 / len(eps)), size=n_boot).astype(float)
    draws = (W @ sums) / (W @ cnts)[:, None]
    pt = sums.sum(0) / cnts.sum()
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    keys = ['p_best', 'mean_rank', 'gap_to_best', 'gap_to_mean', 'frac_candidates_beaten']
    return {k: {'v': float(pt[i]), 'ci': [float(lo[i]), float(hi[i])]}
            for i, k in enumerate(keys)}


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--arm', default='unknown')
@click.option('--n', 'n_actions', default=16, show_default=True)
@click.option('--episodes', default=20, show_default=True)
@click.option('--per-episode', default=8, show_default=True)
@click.option('--split', type=click.Choice(['val', 'test', 'train']), default='test',
              show_default=True)
@click.option('--batch', default=8, show_default=True,
              help='decision points scored at once; each costs batch*(n+1) verifier sims.')
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('--verifier-value', default=None, help='override the scoring rule')
@click.option('--out', default=None, help='write the stats as JSON here')
def main(checkpoint, arm, n_actions, episodes, per_episode, split, batch, device, seed,
         verifier_value, out):
    policy, cfg = load_policy(checkpoint, device)
    if verifier_value is not None:
        policy.search_kwargs['verifier_value'] = verifier_value
        # __dict__, not the attribute: the UNet arm's `verifier` is a lazy property.
        built = policy.__dict__.get('_verifier') or policy.__dict__.get('verifier')
        if built is not None:
            built.value_fn = verifier_value
    native = resolved_verifier_value(policy, cfg)
    vv = verifier_value or native
    To, Ta, H = policy.n_obs_steps, policy.n_action_steps, cfg.policy.horizon
    step = int(''.join(ch for ch in pathlib.Path(checkpoint).stem if ch.isdigit()) or 0)

    run_dir = pathlib.Path(checkpoint).resolve().parent.parent
    _, ep_idxs = get_split_states(cfg, split, run_dir=run_dir)
    ep_idxs = list(ep_idxs)[:episodes]
    rb = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path,
                                     keys=['img', 'agent_pos', 'action', 'block_pos'])
    pts = sample_points(rb, ep_idxs, per_episode, To, H, np.random.default_rng(seed))
    print(f'{arm} step {step}: {len(pts)} decision points from {len(ep_idxs)} {split} '
          f'episodes, n={n_actions}, verifier={vv}'
          + ('  (OVERRIDDEN)' if verifier_value and verifier_value != native else ''))

    v_star, v_cand, t_star, t_cand, ep_of = [], [], [], [], []
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        for b0 in range(0, len(pts), batch):
            chunk = pts[b0:b0 + batch]
            obs, star = build_batch(rb, chunk, To, H, torch.device(device))
            seeder = getattr(policy, 'set_sample_seeds', None)
            if seeder is not None:
                seeder([seed * 1_000_003 + b0 + k for k in range(len(chunk))])
            with torch.no_grad(), policy._crop_scope():
                feats = policy._encode_obs_features(obs)
                _, _, sc, tm = policy.predict_n_actions(
                    obs, verifier=policy.verifier, n_actions=n_actions,
                    return_scores=True, obs_features=feats, return_terms=True)
                # THE SAME CALL THE CANDIDATES GO THROUGH -- same verifier, same sim pool,
                # same window. A separate scoring path here could drift from the ranking
                # the policy actually applies, which is the thing being tested.
                out_star = policy._score_candidates(policy.verifier, obs, star)
            v_star.append(out_star[1].float().cpu().numpy())
            t_star.append(out_star[3].float().cpu().numpy())
            v_cand.append(sc.float().cpu().numpy())
            t_cand.append(tm.float().cpu().numpy())
            ep_of += [e for e, _ in chunk]
            print(f'  {min(b0 + batch, len(pts))}/{len(pts)}', end='\r', flush=True)
    finally:
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: verifier close failed: {e}')

    vs = np.concatenate(v_star)                     # (S,)
    vc = np.concatenate(v_cand)                     # (S, n)
    ts = np.concatenate(t_star)                     # (S, 2)
    tc = np.concatenate(t_cand)                     # (S, n, 2)
    eps = np.array(ep_of)
    res = {'arm': arm, 'step': step, 'checkpoint': str(checkpoint), 'split': split,
           'n': n_actions, 'n_points': int(len(vs)), 'n_episodes': int(len(set(ep_of))),
           'verifier_value': vv, 'verifier_value_native': native, 'seed': seed,
           'value': _stats(vs, vc, True, eps)}
    for ax in (T_GOAL, ARM_T):
        res[TERM[ax]] = _stats(ts[:, ax], tc[:, :, ax], False, eps)

    print(f'\n{"quantity":12s} {"mean rank/" + str(n_actions):>20s} '
          f'{"cands a* beats":>20s} {"a* - MEAN cand":>22s} {"a* is best":>20s} '
          f'{"a* - best cand":>22s}')
    for k in ('value', TERM[T_GOAL], TERM[ARM_T]):
        r = res[k]
        f = (lambda d, p=3: f"{d['v']:.{p}f} [{d['ci'][0]:.{p}f},{d['ci'][1]:.{p}f}]")
        print(f'{k:12s} {f(r["mean_rank"], 2):>20s} {f(r["frac_candidates_beaten"]):>20s} '
              f'{f(r["gap_to_mean"], 2):>22s} {f(r["p_best"]):>20s} '
              f'{f(r["gap_to_best"], 2):>22s}')
    print(f'\nUnder the null that a* is exchangeable with the candidates: mean rank '
          f'{(n_actions) / 2:.1f}, fraction beaten 0.500, gap 0.00, p_best '
          f'{1 / (n_actions + 1):.3f}.')
    print('READ THE FIRST THREE COLUMNS. `a* - best cand` and `a* is best` compare one '
          'action against the MAX of n, an order statistic that beats a typical draw by '
          'construction -- they are reported for completeness but are biased against a* '
          'and get more so as n grows. Rank, fraction-beaten and gap-to-MEAN are the '
          'unbiased comparisons.')
    print('d_t_goal / d_arm_t are raw pixels, sign-flipped so "a* is best" always means '
          'the expert action is the better one.')
    if out:
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
        print(f'-> {out}')


if __name__ == '__main__':
    main()
