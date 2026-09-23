"""Pool every arm's predictions into ONE verifier ranking, split by whether the block moves.

    python scripts/rank_arms_by_blockmotion.py --step 100000 \
        --out analysis/arm_ranking/step_0100000.json

WHAT IS NEW HERE. Everything in docs/reports/archive/ASTAR_RECALL.md measured one arm at a time, against a* and
against a uniform sampler. No arm was ever ranked against another arm's predictions, and
nothing conditioned on whether the block was actually in play. This does both at once: at
each offline decision state every arm proposes 16 candidates, they all go into a single pool
with 16 block-directed uniform samples and a*, ONE shared verifier scores the lot, and the
pool is ranked.

THE SPLIT. `expert_moved` asks whether the DEMONSTRATOR's own block pose changes over this
chunk's executed window. It is a property of the STATE, not of any policy, so the same
decisions land in each subset for every arm and the comparison stays paired:

    block_moving   the expert is pushing the block here
    block_still    the expert is repositioning; nothing moves the T

They behave completely differently and must never be pooled. Under `t_goal` the value is a
function of where the T ends up, so on a still-block state most candidates score IDENTICALLY
(measured: 68-73% of them are fully blind). A naive "who ranked first" there is decided by
argmax's tie-break toward index 0, not by the verifier -- which is why every tally below uses
FRACTIONAL CREDIT under ties and prints the blind fraction beside itself.

TWO RANKINGS OVER THE SAME POOL, and the pair is the point:

    verifier value   what best-of-n actually selects on.        higher is better
    RMSE to a*       how close the prediction is to the demo.   lower is better

Both are taken over the EXECUTED window action[To-1 : To-1+Ta] -- the same 8 waypoints
`_verifier_inputs` simulates -- so they score the identical slice. a* is excluded from the
RMSE ranking (its RMSE to itself is 0, so it would win every decision by construction) but
kept in the verifier ranking, where its position is a real measurement.

The headline is the AGREEMENT between the two. A per-arm version of this correlation was
already ~0, but that could be dismissed as one collapsed policy's candidates being
interchangeable. Here the ranking has 96 genuinely different actions from five policies plus
a blind sampler to separate; if it is still orthogonal to expert-proximity, it is not an
artefact of any one arm.

ONE SHARED VERIFIER, owned here and not by any policy. PushTVerifier is pure
(obs, action) -> value with no policy state, and `_score_candidates(verifier, ...)` takes it
as an argument. Sharing is what makes this paired rather than five runs stapled together.
`policy.close()` would force_close the pool for everyone, so it is never called; the pool is
closed once at the end.
"""
import os
import pathlib as _pl
import sys

# UNCONDITIONAL: forkserver children rebuild __main__ by re-importing this file as
# `__mp_main__`, so anything behind an `if __name__ == '__main__'` guard does not run in the
# child. Without the repo root on its sys.path the child dies importing diffusion_policy and
# the parent exits 120 with no traceback. See scripts/dump_candidate_scores.py.
sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

if __name__ == '__main__':
    os.chdir(str(_pl.Path(__file__).resolve().parent.parent))

import hashlib
import json
import pathlib

import click
import numpy as np
import torch

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.feedback_util import t_center_from_feedback, t_goal_distance
from diffusion_policy.env.pusht.pusht_verifier import PushTVerifier
from eval_search_pusht import get_split_states, load_policy
from scripts.astar_uniform_walk import (
    build_batch, decision_points, expert_moved, measure_rmax, uniform_chunks)
from scripts.verifier_ranks_expert import MOVE_EPS_PX, classify

TASK_SUB = 'pusht_search/pusht_image_search_imgonly'
# subdir | run stem | label.  BC first in every set: its checkpoint is ~4.4GB, so a memory
# failure should happen immediately rather than after four successful loads.
#
# NAMED SETS, because the pool is only meaningful when every arm in it contributes the same
# number of candidates. A K=4 arm cannot produce 16, so the widths are read in two passes:
# `moving4` pairs all seven arms at n=4, and `moving16` reads the k=16 arms plus BC at full
# width (and is the set directly comparable with the blq137 numbers in
# docs/reports/archive/ARM_RANKING_BY_BLOCKMOTION.md). Mixing widths in one pool would give
# the wider arms more entries and bias every mid-rank toward them.
ARM_SETS = {
    # the 2026-09-04 geometric generation, unfiltered; the default, so existing calls are
    # unchanged
    'blq137': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
               ('offline', 'value_k1_ver-t_goal', 'ST k=1'),
               ('outer_inner', 'value_k16_ver-t_goal', 'ST k=16'),
               ('outer_inner', 'value_k16_ver-t_goal_son-flat400', 'flat t=400'),
               ('outer_inner', 'value_k16_ver-t_goal_son-ramp400to200', 'ramp 400->200')],
    # the t_goal-moving generation: run at --n 16
    'moving16': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
                 ('outer_inner', 'value_k16_ver-t_goal', 'k16 uniform'),
                 ('outer_inner', 'value_k16_ver-t_goal_son-flat400', 'k16 flat400'),
                 ('outer_inner', 'value_k16_ver-t_goal_son-ramp400to200', 'k16 ramp400to200')],
    # ...and at --n 4, where all seven are paired
    'moving4': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
                ('outer_inner', 'value_k16_ver-t_goal', 'k16 uniform'),
                ('outer_inner', 'value_k16_ver-t_goal_son-flat400', 'k16 flat400'),
                ('outer_inner', 'value_k16_ver-t_goal_son-ramp400to200', 'k16 ramp400to200'),
                ('outer_inner', 'value_k4_ver-t_goal', 'k4 uniform'),
                ('outer_inner', 'value_k4_ver-t_goal_son-flat400', 'k4 flat400'),
                ('outer_inner', 'value_k4_ver-t_goal_son-ramp400to200', 'k4 ramp400to200')],
    # the 2026-09-19 follow-up pair, at t=600. Both widths together at n=4 (paired), and the
    # k=16 half at n=16. BC and the uniform arms ride along as the reference the 400-level
    # generation is already read against.
    'moving600_16': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
                     ('outer_inner', 'value_k16_ver-t_goal', 'k16 uniform'),
                     ('outer_inner', 'value_k16_ver-t_goal_son-flat600', 'k16 flat600'),
                     ('outer_inner', 'value_k16_ver-t_goal_son-ramp600to200', 'k16 ramp600to200')],
    'moving600_4': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
                    ('outer_inner', 'value_k16_ver-t_goal', 'k16 uniform'),
                    ('outer_inner', 'value_k16_ver-t_goal_son-flat600', 'k16 flat600'),
                    ('outer_inner', 'value_k16_ver-t_goal_son-ramp600to200', 'k16 ramp600to200'),
                    ('outer_inner', 'value_k4_ver-t_goal', 'k4 uniform'),
                    ('outer_inner', 'value_k4_ver-t_goal_son-flat600', 'k4 flat600'),
                    ('outer_inner', 'value_k4_ver-t_goal_son-ramp600to200', 'k4 ramp600to200')],
    # ALL ELEVEN arms in one pool, at n=4 -- the only width every arm can contribute, so the
    # pool is paired. Used for the per-checkpoint sweep that picks which arms and which
    # checkpoint to carry into the OOD evals.
    'moving_all': [('unet_bc', 'unetbc_ver-t_goal', 'BC'),
                   ('outer_inner', 'value_k16_ver-t_goal', 'k16 uniform'),
                   ('outer_inner', 'value_k16_ver-t_goal_son-flat400', 'k16 flat400'),
                   ('outer_inner', 'value_k16_ver-t_goal_son-ramp400to200', 'k16 ramp400'),
                   ('outer_inner', 'value_k16_ver-t_goal_son-flat600', 'k16 flat600'),
                   ('outer_inner', 'value_k16_ver-t_goal_son-ramp600to200', 'k16 ramp600'),
                   ('outer_inner', 'value_k4_ver-t_goal', 'k4 uniform'),
                   ('outer_inner', 'value_k4_ver-t_goal_son-flat400', 'k4 flat400'),
                   ('outer_inner', 'value_k4_ver-t_goal_son-ramp400to200', 'k4 ramp400'),
                   ('outer_inner', 'value_k4_ver-t_goal_son-flat600', 'k4 flat600'),
                   ('outer_inner', 'value_k4_ver-t_goal_son-ramp600to200', 'k4 ramp600')],
    # the 2026-09-21 blq137-FILTERED generation: the same four 600-level ladders as
    # moving600_*, trained on blq137's own moving transitions and scored on its 50
    # bottom-left-quadrant test episodes. Paired at --n 4; the k=16 pair also runs at --n 16.
    # No BC and no uniform arm: neither exists at split-blq-mv, so the pool here is the four
    # trained arms plus the block-directed uniform samples and a*.
    'blqmv4': [('outer_inner', 'value_k16_ver-t_goal_son-flat600', 'k16 flat600'),
               ('outer_inner', 'value_k16_ver-t_goal_son-ramp600to200', 'k16 ramp600'),
               ('outer_inner', 'value_k4_ver-t_goal_son-flat600', 'k4 flat600'),
               ('outer_inner', 'value_k4_ver-t_goal_son-ramp600to200', 'k4 ramp600')],
    'blqmv16': [('outer_inner', 'value_k16_ver-t_goal_son-flat600', 'k16 flat600'),
                ('outer_inner', 'value_k16_ver-t_goal_son-ramp600to200', 'k16 ramp600')],
}
# The default dataset for each set, so --arms carries its own splits rather than needing
# three flags kept in step by hand.
ARM_SET_DATA = {'blq137': (137, 'split-blq'),
                'moving16': (176, 'split-mv'),
                'moving4': (176, 'split-mv'),
                'moving600_16': (176, 'split-mv'),
                'moving600_4': (176, 'split-mv'),
                'moving_all': (176, 'split-mv'),
                'blqmv4': (137, 'split-blq-mv'),
                'blqmv16': (137, 'split-blq-mv')}
ARMS = ARM_SETS['blq137']
UNIFORM, ASTAR = 'uniform', 'a*'
SUBSETS = ('block_moving', 'block_still')


def midrank(x, higher_is_better):
    """Competition-free mid-ranks along the last axis; rank 0 == best. Ties share their mean.

    Ties are not a detail here. On a still-block decision most of the pool scores IDENTICALLY,
    and any rank rule that breaks ties by index would hand the win to whichever source happens
    to be laid out first. Mid-ranks give every tied candidate the same rank, which is the only
    honest answer when the verifier has expressed no preference between them.
    """
    v = -x if higher_is_better else x
    order = v.argsort(axis=-1)
    r = np.empty_like(order, dtype=float)
    np.put_along_axis(r, order, np.arange(v.shape[-1], dtype=float), axis=-1)
    out = np.empty_like(r)
    for i in range(v.shape[0]):
        _, inv, cnt = np.unique(v[i], return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, r[i])
        out[i] = (sums / cnt)[inv]
    return out


def topk_credit(x, higher_is_better, k):
    """(S, P) fractional credit for being in the top k. Each row sums to exactly k.

    A candidate tied with m others across the top-k boundary gets the fraction of the k slots
    that the tie group is entitled to, split evenly. So a fully-tied decision spreads k/P to
    every candidate instead of awarding k winners by sort order.
    """
    v = -x if higher_is_better else x
    S, P = v.shape
    out = np.zeros((S, P))
    for i in range(S):
        vals, inv, cnt = np.unique(v[i], return_inverse=True, return_counts=True)
        # groups arrive sorted best-first; walk them filling the k slots
        filled, share = 0, np.zeros(len(vals))
        for g in range(len(vals)):
            take = min(cnt[g], max(0, k - filled))
            if take <= 0:
                break
            share[g] = take / cnt[g]
            filled += cnt[g]
        out[i] = share[inv]
    return out


def spearman(a, b):
    """Rank correlation along the last axis, row by row. NaN where a row is constant.

    Constant rows are the norm on still-block decisions (every verifier value identical), and
    a correlation is genuinely undefined there rather than zero -- so they are dropped from
    the mean rather than counted as evidence of no relationship.
    """
    ra, rb = midrank(a, False), midrank(b, False)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))
    return np.where(den > 1e-12, (ra * rb).sum(-1) / np.where(den > 1e-12, den, 1), np.nan)


def rmse_to_star(cand, star, sl):
    """(S, P) RMSE and mean euclidean offset to a*, over the EXECUTED window. Both in px.

    `sl` is action[To-1 : To-1+Ta] -- exactly what _verifier_inputs simulates -- so these and
    the verifier score describe the same 8 waypoints.

    THE TWO SCALES DIFFER BY sqrt(2) AND IT MATTERS FOR READING THE NUMBERS. `rmse` averages
    the squared error over (timestep, x, y), so a candidate sitting a constant euclidean D px
    away at every waypoint scores D/sqrt(2), not D. That is the ordinary definition of RMSE
    and it is what was asked for; it ranks identically to the euclidean offset (a positive
    monotone rescale), so every ranking below is unaffected by the choice. `euclid` is
    reported next to it because it is the one that can be compared against arena distances,
    the candidate cloud spread, and the r_max of the uniform sampler.
    """
    d = cand[:, :, sl, :] - star[:, None, sl, :]
    rmse = np.sqrt((d ** 2).mean(axis=(-1, -2)))
    euclid = np.linalg.norm(d, axis=-1).mean(axis=-1)
    return rmse, euclid


def load_arms(root, step, device, shared, n_demos=137, tag='split-blq', enc='enc-resnet18',
              arms=None):
    """Load every arm and point them all at ONE verifier. -> [(label, policy, cfg)]

    The injection is not cosmetic. `PushTUNetSearchPolicy.verifier` is a read-only property
    backed by `_verifier` and built lazily, so assigning `policy.verifier` on the BC arm
    raises, and leaving it alone forks a SECOND pool of 32 processes the first time it scores.
    The transformer arms hold `verifier` as a plain attribute and are assigned directly.
    """
    out = []
    for sub, stem, label in (ARMS if arms is None else arms):
        run = f'{stem}_{enc}_demos-{n_demos}_{tag}_seed-42'
        ckpt = pathlib.Path(root) / TASK_SUB / sub / run / 'checkpoints' / f'step_{step:07d}.ckpt'
        assert ckpt.is_file(), f'missing {ckpt}'
        print(f'  loading {label:14s} {ckpt.parent.parent.name}', flush=True)
        policy, cfg = load_policy(str(ckpt), device)
        # drop whatever pool this policy built or would build, and hand it the shared one
        own = policy.__dict__.pop('_verifier', None) or policy.__dict__.pop('verifier', None)
        if own is not None and own is not shared:
            try:
                own.close()
            except Exception:
                pass
        policy.__dict__['_verifier' if hasattr(type(policy), 'verifier') else 'verifier'] = shared
        assert policy.verifier is shared, f'{label}: verifier injection failed'
        out.append((label, policy, cfg))
    return out


def candidates_for(policy, obs, n, shared, seeds):
    """(B, n, H, 2) and (B, n) -- one arm's candidates and their verifier values.

    Runs inside the policy's OWN crop and corrupt scopes: the corrupt scope pins one
    obs-noise sample across the decision, without which the `son-*` arms' spread would be
    inflated by fresh noise per candidate rather than by the policy.
    """
    seeder = getattr(policy, 'set_sample_seeds', None)
    if seeder is not None:
        seeder(list(seeds))
    try:
        with torch.no_grad(), policy._crop_scope(), policy._corrupt_scope():
            feats = policy._encode_obs_features(obs)
            acts, _, sc, tm = policy.predict_n_actions(
                obs, verifier=shared, n_actions=n, return_scores=True,
                obs_features=feats, return_terms=True)
    finally:
        if seeder is not None:
            seeder(None)
    return (acts.float().cpu().numpy(), sc.float().cpu().numpy(),
            tm.float().cpu().numpy())


def slot_curves(v, eu, src, labels, n_arm):
    """Per-source curves over search width n = 1..n_arm, as FINAL candidate vs BEST-of-n.

    ONE WIDTH-16 RUN GIVES THE WHOLE CURVE. `search_candidates` builds candidate i
    conditioned on exactly the i already-scored candidates before it (search_procedure.py
    :1021), so the first k+1 candidates of a width-16 search are bit-identical to a
    width-(k+1) search under the same seeds. Slot index and search width are the same axis.

        final(n)  the candidate a `final_pass` policy would execute at width n: slot n-1,
                  the most-conditioned draw, chosen WITHOUT consulting the verifier.
        best(n)   what argmax executes at width n: the max over slots 0..n-1.

    The pair separates two things that best-of-n conflates. `best` rises with n for ANY
    source, including i.i.d. noise, purely because it is a running maximum over more draws.
    `final` rises only if conditioning on the scored context actually steers generation. For
    BC and ST k=1 -- no context, i.i.d. slots -- `final` MUST be flat, which is the control
    that says the axis is wired up correctly.
    """
    out = {}
    for j, lab in enumerate(labels):
        cols = np.flatnonzero(src == lab)
        if len(cols) != n_arm:            # a* holds a single column; no width axis
            continue
        va, ea = v[:, cols], eu[:, cols]                       # (S, n) in GENERATION order
        out[lab] = {
            'final_value': [float(va[:, k].mean()) for k in range(n_arm)],
            'best_value': [float(va[:, :k + 1].max(axis=1).mean()) for k in range(n_arm)],
            'final_euclid_px': [float(ea[:, k].mean()) for k in range(n_arm)],
            'best_euclid_px': [float(ea[:, :k + 1].min(axis=1).mean()) for k in range(n_arm)],
        }
    return out


def summarise(v, rm, eu, src, cls, n_pool, labels):
    """Per-source tallies over one subset of decisions. `src` is (P,) source label per column.

    Every tally is fractional-credit under ties, so each decision contributes exactly k to the
    top-k totals no matter how degenerate its verifier values are.
    """
    star_col = np.array([s == ASTAR for s in src])
    out = {'n_decisions': int(len(v)),
           'p_blind': float((cls == 'blind').mean()) if len(cls) else None,
           'by_source': {}}
    # verifier ranking over the whole pool; RMSE ranking excludes a* (its RMSE is 0)
    vr = midrank(v, True)
    credit = {k: topk_credit(v, True, k) for k in (1, 5, 10)}
    rr = midrank(rm[:, ~star_col], False)
    rcredit = {k: topk_credit(rm[:, ~star_col], False, k) for k in (1, 5, 10)}
    for lab in labels:
        m = np.array([s == lab for s in src])
        d = {'n_candidates': int(m.sum()),
             'verifier_mean_rank': float(vr[:, m].mean()),
             'verifier_value_mean': float(v[:, m].mean()),
             'verifier_value_best': float(v[:, m].max(axis=1).mean()),
             'rmse_mean': float(rm[:, m].mean()),
             'rmse_best': float(rm[:, m].min(axis=1).mean()),
             'euclid_mean_px': float(eu[:, m].mean()),
             'euclid_best_px': float(eu[:, m].min(axis=1).mean())}
        for k in (1, 5, 10):
            d[f'verifier_top{k}_share'] = float(credit[k][:, m].sum() / (k * len(v)))
        if lab != ASTAR:
            mr = np.array([s == lab for s in src[~star_col]])
            d['rmse_mean_rank'] = float(rr[:, mr].mean())
            for k in (1, 5, 10):
                d[f'rmse_top{k}_share'] = float(rcredit[k][:, mr].sum() / (k * len(v)))
        out['by_source'][lab] = d
    # Agreement between the two rankings, BOTH taken over the pool minus a*. Ranking the
    # verifier over the full pool here and the RMSE over the reduced one would compare
    # positions in two different pools -- a* scores well on the verifier, so it would shift
    # every rank by one and make a perfect agreement read as 1.0 instead of 0.0.
    vsub, rsub = v[:, ~star_col], rm[:, ~star_col]
    rho = spearman(-vsub, rsub)
    vr_sub = midrank(vsub, True)
    top_v, top_r = topk_credit(vsub, True, 1), topk_credit(rsub, False, 1)
    has = np.isfinite(rho)
    out['by_slot'] = slot_curves(v, eu, src, labels, int((src == labels[0]).sum()))
    out['agreement'] = {
        # None, not NaN: on a fully blind subset no decision has an ordering, the correlation
        # is undefined rather than zero, and NaN is not valid JSON.
        'spearman_verifier_vs_rmse': float(np.nanmean(rho)) if has.any() else None,
        'n_decisions_with_a_ranking': int(has.sum()),
        # both are fractional, so this is the expected overlap under each decision's own ties
        'top1_overlap': float(np.minimum(top_v, top_r).sum() / len(v)),
        'rmse_best_mean_verifier_rank': float(
            vr_sub[np.arange(len(v)), rsub.argmin(axis=1)].mean()),
        'pool_size': int(n_pool), 'ranked_pool_size': int(vsub.shape[1])}
    return out


@click.command()
@click.option('--step', required=True, type=int, help='checkpoint step, shared by every arm')
@click.option('--root', default=os.environ.get(
    'DP_OUTPUT_ROOT', '/gscratch/robotics/harine/diffusion_policy_outputs'))
@click.option('--n', 'n_actions', default=16, show_default=True,
              help='candidates per arm; the uniform budget matches it')
@click.option('--episodes', default=50, show_default=True)
@click.option('--sectors', default=16, show_default=True)
@click.option('--split', type=click.Choice(['train', 'val', 'test']), default='test', show_default=True)
@click.option('--batch', default=32, show_default=True,
              help='decisions per verifier call; 32 matches verifier_n_envs so no sim is wasted')
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('--out', default=None, help='write the stats as JSON here')
@click.option('--arms', 'arm_set', type=click.Choice(sorted(ARM_SETS)), default='blq137',
              show_default=True,
              help='which named arm set to pool; each carries its own n_demos/tag')
@click.option('--n-demos', default=None, type=int, help='override the set default')
@click.option('--tag', default=None, help='split tag, e.g. split-mv; overrides the set default')
@click.option('--corrupt-obs-eval/--no-corrupt-obs-eval', 'corrupt_obs_eval', default=False,
              show_default=True,
              help='evaluate each ladder arm under the conditional it was FITTED to. '
                   'load_policy calls policy.eval(), and the PushT config leaves '
                   'corrupt_obs_eval null, so by default the slot ladder is the identity '
                   'here and every arm is read on CLEAN observations. A no-op on an arm '
                   'with no ladder, exactly as in eval_search_pusht.')
def main(step, root, n_actions, episodes, sectors, split, batch, device, seed, out,
         arm_set, n_demos, tag, corrupt_obs_eval):
    # OWNED HERE, never by a policy: policy.close() force_closes the pool, which with one
    # shared instance would tear it down for every other arm mid-run.
    arms_spec = ARM_SETS[arm_set]
    d_demos, d_tag = ARM_SET_DATA[arm_set]
    n_demos = n_demos if n_demos is not None else d_demos
    tag = tag if tag is not None else d_tag
    shared = PushTVerifier(n_envs=32, value_fn='t_goal', use_async=True)
    print(f'shared verifier: t_goal, 32 envs')
    print(f'arm set {arm_set}: {len(arms_spec)} arms on demos-{n_demos}_{tag}, n={n_actions}')
    arms = load_arms(root, step, device, shared, n_demos=n_demos, tag=tag, arms=arms_spec)
    if corrupt_obs_eval:
        # Set on the POLICY, after load_policy's .eval(), the same way eval_search_pusht
        # does it. corrupt_obs_features_slotwise gates on `not self.training and not
        # self.corrupt_obs_eval`, so without this the ladder is the identity at eval.
        laddered = 0
        for lab, pol, _ in arms:
            pol.corrupt_obs_eval = True
            laddered += bool(getattr(pol, 'slot_ladder_on', False))
        print(f'corrupt-obs-eval ON: {laddered}/{len(arms)} arms have a ladder to apply')
    labels = [a[0] for a in arms] + [UNIFORM, ASTAR]

    cfg = arms[0][2]
    To, Ta, H = arms[0][1].n_obs_steps, arms[0][1].n_action_steps, cfg.policy.horizon
    sl = slice(To - 1, To - 1 + Ta)
    for lab, pol, c in arms[1:]:
        assert (pol.n_obs_steps, pol.n_action_steps, c.policy.horizon) == (To, Ta, H), \
            f'{lab} has a different window; states would not align'

    run_dir = pathlib.Path(root) / TASK_SUB / arms_spec[0][0] / (
        f'{arms_spec[0][1]}_enc-resnet18_demos-{n_demos}_{tag}_seed-42')
    _, ep_idxs = get_split_states(cfg, split, run_dir=str(run_dir))
    ep_idxs = list(ep_idxs)[:episodes]
    rb = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path,
                                     keys=['img', 'agent_pos', 'action', 'block_pos'])
    # COMPUTED ONCE and reused by every arm -- the states cannot drift between arms.
    pts = decision_points(rb, ep_idxs, To, H, Ta)
    r_max = measure_rmax(rb, ep_idxs, To, Ta, H)
    rng = np.random.default_rng(seed)
    n_pool = n_actions * len(arms) + n_actions + 1
    print(f'step {step}: {len(pts)} decisions over {len(ep_idxs)} {split} episodes\n'
          f'pool per decision: {len(arms)} arms x {n_actions} + {n_actions} uniform + a* '
          f'= {n_pool} candidates\n'
          f'uniform: {sectors} wedges, block-facing half, r_max={r_max:.1f}px')

    V, RM, EU, T, PH, EP, RF = [], [], [], [], [], [], []
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        for b0 in range(0, len(pts), batch):
            chunk = pts[b0:b0 + batch]
            obs, star, poses = build_batch(rb, chunk, To, H, torch.device(device))
            now = obs['feedback'][:, To - 1].cpu().numpy()
            agent = obs['agent_pos'][:, To - 1].cpu().numpy()
            h = hashlib.md5(obs['image'].cpu().numpy().tobytes()).hexdigest()
            seeds = [seed * 1_000_003 + b0 + k for k in range(len(chunk))]

            acts, vals, terms = [], [], []
            for lab, pol, _ in arms:
                a, v, t = candidates_for(pol, obs, n_actions, shared, seeds)
                acts.append(a); vals.append(v); terms.append(t)
                # every arm must see the byte-identical observation batch
                assert hashlib.md5(obs['image'].cpu().numpy().tobytes()).hexdigest() == h, \
                    f'{lab} mutated the obs batch'

            uni, _, _ = uniform_chunks(agent, t_center_from_feedback(now), n_actions,
                                       r_max, To, Ta, H, sectors, rng)
            uni_t = torch.tensor(uni, dtype=torch.float32, device=device)
            # scored through ANY arm's _score_candidates -- the call only forwards to the
            # verifier, which is the same object for every arm, so the choice of arm here
            # cannot bias the reference rows.
            scorer = arms[0][1]
            uv, ut = [], []
            with torch.no_grad():
                for k in range(n_actions):
                    o = scorer._score_candidates(shared, obs, uni_t[:, k])
                    uv.append(o[1].float().cpu().numpy())
                    ut.append(o[3].float().cpu().numpy())
                o = scorer._score_candidates(shared, obs, star)
            acts.append(uni); vals.append(np.stack(uv, axis=1)); terms.append(np.stack(ut, axis=1))
            sn = star.float().cpu().numpy()
            acts.append(sn[:, None]); vals.append(o[1].float().cpu().numpy()[:, None])
            terms.append(o[3].float().cpu().numpy()[:, None])

            cand = np.concatenate(acts, axis=1)              # (B, P, H, 2)
            V.append(np.concatenate(vals, axis=1))           # (B, P)
            T.append(np.concatenate(terms, axis=1))          # (B, P, 2)
            rm, eu = rmse_to_star(cand, sn, sl)
            RM.append(rm); EU.append(eu)
            PH.append(expert_moved(poses, Ta))
            # the no-contact reference: the CURRENT T-to-goal distance, straight from the
            # observation, so classify() costs no extra simulation
            RF.append(t_goal_distance(now))
            EP += [e for e, _, _ in chunk]
            print(f'  {min(b0 + batch, len(pts))}/{len(pts)}', end='\r', flush=True)
    finally:
        # closed ONCE, here -- never via policy.close()
        try:
            shared.close()
        except Exception as e:
            print(f'warning: verifier close failed: {e}')

    v = np.concatenate(V); rm = np.concatenate(RM); eu = np.concatenate(EU)
    t = np.concatenate(T); phase = np.concatenate(PH)
    src = np.array(sum([[lab] * n_actions for lab, _, _ in arms], [])
                   + [UNIFORM] * n_actions + [ASTAR])
    assert v.shape[1] == len(src) == n_pool

    ref = np.concatenate(RF)
    res = {'step': step, 'split': split, 'n_per_arm': n_actions, 'pool_size': n_pool,
           'n_decisions': int(len(v)), 'n_episodes': int(len(set(EP))),
           'r_max_px': r_max, 'sectors': sectors, 'seed': seed,
           'verifier_value': 't_goal', 'arm_labels': labels,
           # WHICH READOUT. Absent in files written before 2026-09-21, which were all clean.
           'corrupt_obs_eval': bool(corrupt_obs_eval), 'subsets': {}}
    for name, m in (('block_moving', phase), ('block_still', ~phase)):
        if not m.any():
            continue
        # BLIND HERE MEANS THE WHOLE POOL IS BLIND: no candidate from ANY of the five arms
        # moves the T. That is the right test for a pooled ranking -- it asks whether this
        # ranking can discriminate at all -- but it is STRICTER than the per-arm blind rate
        # in docs/reports/archive/ASTAR_RECALL.md and comes out much lower (five policies rarely all miss the
        # block at once). The two numbers answer different questions; do not compare them.
        # The reference rows are excluded so the tally describes the trained arms.
        pol_cols = np.array([s not in (UNIFORM, ASTAR) for s in src])
        cls, _, _ = classify(t[m][:, pol_cols, 0], ref[m])
        res['subsets'][name] = summarise(v[m], rm[m], eu[m], src, cls, n_pool, labels)
    # WRITE FIRST, REPORT SECOND. These runs cost ~2 GPU-hours and load seven policies; on
    # 2026-09-19 a hardcoded width in report() raised IndexError at --n 4 and threw the whole
    # completed result away. Formatting must never be able to destroy a measurement.
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
        print(f'\n-> {out}')
    try:
        report(res)
    except Exception as e:
        print(f'\nreport() failed ({type(e).__name__}: {e}) -- the JSON above is complete '
              f'and unaffected.')


def report(res):
    P = res['pool_size']
    labels = res['arm_labels']
    for name in SUBSETS:
        s = res['subsets'].get(name)
        if not s:
            continue
        head = ('the expert IS pushing the block' if name == 'block_moving'
                else 'the expert is NOT moving the block')
        print(f'\n{"="*92}\n{name.upper()}  --  {head}')
        print(f'{s["n_decisions"]} decisions   pooled-blind {s["p_blind"]:.1%} '
              f'(no candidate from ANY of the 5 arms moves the T,\n'
              f'{"":33s}so the verifier ranks nothing -- stricter than the per-arm rate '
              f'in docs/reports/archive/ASTAR_RECALL.md)\n{"="*92}')
        print(f'{"source":15s}{"verif top1":>11s}{"top5":>8s}{"top10":>8s}'
              f'{"vrank/" + str(P):>11s}{"rmse top1":>11s}{"rmse rank":>11s}'
              f'{"eucl best":>11s}{"verif best":>12s}')
        for lab in labels:
            d = s['by_source'][lab]
            rt = (f"{d['rmse_top1_share']:>11.1%}" if 'rmse_top1_share' in d else f"{'ref':>11s}")
            rr = (f"{d['rmse_mean_rank']:>11.1f}" if 'rmse_mean_rank' in d else f"{'—':>11s}")
            print(f'{lab:15s}{d["verifier_top1_share"]:>11.1%}{d["verifier_top5_share"]:>8.1%}'
                  f'{d["verifier_top10_share"]:>8.1%}{d["verifier_mean_rank"]:>11.1f}'
                  f'{rt}{rr}{d["euclid_best_px"]:>11.1f}{d["verifier_value_best"]:>12.2f}')
        n_arm = res['n_per_arm']
        print(f'\n  Shares are FRACTIONAL under ties, so each decision contributes exactly 1.0\n'
              f'  to top1. A source holding {n_arm} of {P} columns owns {n_arm/P:.1%} by chance;\n'
              f'  a* holds 1 column, so its chance share is {1/P:.1%}. Mean verifier rank is\n'
              f'  out of {P} (chance {(P-1)/2:.1f}); RMSE rank excludes a* so it is out of {P-1}.')
        bs = s.get('by_slot') or {}
        if bs:
            # Derived from the data, not hardcoded: a --n 4 pool has four slots, and
            # [1,2,4,8,16] indexes slot 15 of a length-4 list.
            width = min(len(d['final_value']) for d in bs.values() if d)
            NS = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= width]
            print(f'\n  FINAL CANDIDATE vs BEST-OF-N, as search width n grows')
            print(f'  final(n) = slot n-1, the most-conditioned draw, executed WITHOUT '
                  f'consulting the verifier.\n  best(n) = max over slots 0..n-1, what argmax '
                  f'executes. One width-16 run gives\n  both curves because candidate i '
                  f'conditions on exactly the i candidates before it.')
            print(f'\n  {"source":15s}' + ''.join(f'n={n}'.rjust(9) for n in NS)
                  + '   |' + ''.join(f'n={n}'.rjust(9) for n in NS))
            print(f'  {"":15s}' + f'{"--- final(n) verifier value ---":>45s}'
                  + f'   |{"--- best-of-n verifier value ---":>45s}')
            for lab in labels:
                d = bs.get(lab)
                if not d:
                    continue
                f_ = ''.join(f'{d["final_value"][n-1]:9.2f}' for n in NS)
                b_ = ''.join(f'{d["best_value"][n-1]:9.2f}' for n in NS)
                print(f'  {lab:15s}{f_}   |{b_}')
            print(f'\n  {"source":15s}' + ''.join(f'n={n}'.rjust(9) for n in NS)
                  + '   |' + ''.join(f'n={n}'.rjust(9) for n in NS))
            print(f'  {"":15s}' + f'{"--- final(n) euclid to a* px ---":>45s}'
                  + f'   |{"--- best-of-n euclid to a* px ---":>45s}')
            for lab in labels:
                d = bs.get(lab)
                if not d:
                    continue
                f_ = ''.join(f'{d["final_euclid_px"][n-1]:9.1f}' for n in NS)
                b_ = ''.join(f'{d["best_euclid_px"][n-1]:9.1f}' for n in NS)
                print(f'  {lab:15s}{f_}   |{b_}')
            print('\n  best(n) rises with n for ANY source, i.i.d. noise included -- it is a\n'
                  '  running maximum. final(n) rises only if the scored context steers\n'
                  '  generation. BC, ST k=1 and uniform have no usable context, so their\n'
                  '  final(n) MUST be flat; that is the control on the axis.')

        a = s['agreement']
        Q = a['ranked_pool_size']
        print(f'\n  DO THE TWO RANKINGS AGREE?')
        rho = a['spearman_verifier_vs_rmse']
        print(f'    spearman(verifier, -RMSE to a*)   '
              + (f'{rho:+.3f}' if rho is not None else '  undefined')
              + f'   over {a["n_decisions_with_a_ranking"]} decisions that had any ordering')
        print(f'    verifier top-1 is also RMSE top-1 {a["top1_overlap"]:.1%}')
        print(f'    the RMSE-best candidate sits at verifier rank '
              f'{a["rmse_best_mean_verifier_rank"]:.1f} of {Q} (chance {(Q-1)/2:.1f})')
        print('    ~0 spearman means the verifier orders the pool without regard to how close\n'
              '    a candidate is to what the demonstrator actually did.')


if __name__ == '__main__':
    main()
