"""Is a* generated, and does the verifier rank it above the policy's own candidates?

    python scripts/astar_uniform_walk.py -c <run>/checkpoints/step_0100000.ckpt \
        --arm k16-uniform --out analysis/astar_recall/<canonical>/step_0100000.json

OPEN LOOP, ALONG THE DEMO. a* -- the demonstrator's action chunk -- exists only at recorded
demo states. The moment a rollout diverges there is no expert action for the state it is
actually in, so a per-state comparison there would be a claim about a different state. This
walks a test episode's OWN states in order and asks, at every decision:

  1. does the policy's candidate set contain a*?              (is a* sampled)
  2. where does a* rank among the candidates under the verifier?   (does the verifier prefer it)
  3. could BLIND UNIFORM SAMPLING have beaten both?           (is the policy's support the
                                                               binding constraint, or the ranking)

THE UNIFORM COMPETITOR. 16 policy candidates against 16 hand-built chunks that owe nothing to
the policy: directions around the arm are cut into S wedges, the half facing the BLOCK is
kept, and each kept wedge contributes straight-line chunks at a uniform angle and radius. The
budget is matched deliberately, so under the null "the verifier's top pick is equally likely
to come from either source" the provenance tally is a coin flip and any skew reads directly.
If the verifier's favourite is routinely a blind uniform sample, best-of-n is limited by what
the policy can generate, not by the ranking.

Aimed at the BLOCK, not at the goal pose: under `t_goal` the value is a function of where the
T ends up, so a chunk that never contacts the block returns the identical value as every
other such chunk. Sampling away from the block manufactures ties, not information.

WHEN IS THE VERIFIER USEFUL AT ALL. That same fact makes most of the ranking questions
conditional. Every decision is classified by how many of its candidates actually move the T:

    blind        0 movers   -- every score identical. The verifier expresses NO preference;
                              argmax is an arbitrary tie-break toward index 0.
    partial      some       -- movers ranked against a tied floor.
    informative  all n      -- a fully distinct set.

The reference for "did it move" costs no simulation: a chunk that never touches the block
scores exactly the CURRENT T-to-goal distance, which is `t_goal_distance(obs['feedback'])`.
Every statistic here is reported stratified by that class, because on a blind decision a*'s
rank is a property of the tie rule rather than of the verifier. `p_blind` is also the honest
denominator for the provenance tally and the reason `n` is only a real test-time-compute axis
on part of the episode.

OUTPUT (--out JSON):
    informativeness   the class split, movers/n, effective search width, by phase
    astar             a* vs policy candidates, stratified by class
    provenance        where the verifier's top pick came from, on non-tied decisions
    recall            distance from the candidate cloud to a*, against the cloud's own spread
    per_decision      the raw per-frame record
"""
import os
import pathlib as _pl
import sys

# UNCONDITIONAL, not under an `if __name__ == '__main__'` guard. The verifier's sim pool is
# an AsyncVectorEnv on the 'forkserver' context, and a forkserver child rebuilds __main__ by
# RE-IMPORTING this file -- as `__mp_main__`, so anything hidden behind that guard does not
# run in the child. With the repo root missing from its sys.path the child dies importing
# `diffusion_policy`, the parent sees only a closed pipe, and the process exits 120 with no
# traceback on either side.
sys.path.append(str(_pl.Path(__file__).resolve().parent.parent))

if __name__ == '__main__':
    os.chdir(str(_pl.Path(__file__).resolve().parent.parent))

import json
import pathlib

import click
import numpy as np
import torch

from diffusion_policy.env.pusht.feedback_util import (
    compute_feedback_from_pose, t_center_from_feedback, t_goal_distance)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.env.pusht.pusht_verifier import is_waypoint_value
from diffusion_policy.env.pusht.veritas.tracker import DualTracker
from eval_search_pusht import (
    _WAYPOINT_PLAN_DIRS, get_split_states, load_policy, load_split_plans)
from scripts.dump_candidate_scores import resolved_verifier_value
from scripts.verifier_ranks_expert import (
    BLIND, CLASSES, INFORMATIVE, MOVE_EPS_PX, PARTIAL, T_GOAL, classify,
    nontouch_stats, _stats)

POLICY, UNIFORM, ASTAR, TIE = 'policy', 'uniform', 'astar', 'tie'
SOURCES = (POLICY, UNIFORM, ASTAR)


def decision_points(rb, ep_idxs, To, H, Ta):
    """Ordered (episode, absolute index, decision number) for every decision of every episode.

    Stride is Ta, not 1: a decision is one complete search, and consecutive control steps
    advance by the executed window. Both the obs window and the a* chunk must fit intact --
    the dataset pads a short tail by repeating the last action, and a repeated action is not
    something the expert did, so a padded chunk cannot sit on the ground-truth side.
    """
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    out = []
    for e in ep_idxs:
        s, t = int(starts[e]), int(ends[e])
        lo, hi = s + To - 1, t - H + To - 1
        out += [(int(e), int(i), k) for k, i in enumerate(range(lo, hi, Ta))]
    return out


def measure_rmax(rb, ep_idxs, To, Ta, H, q=95):
    """One GLOBAL radius for the uniform competitor: the q-th percentile of the expert's own
    net displacement over an executed window, in arena px.

    Global and expert-derived on purpose. Deriving it per-arm from that arm's own candidates
    would make the baseline a different baseline for every arm and destroy exactly the
    cross-arm comparison it exists to support.
    """
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    ac = np.asarray(rb['action'])
    d = []
    for e in ep_idxs:
        s, t = int(starts[e]), int(ends[e])
        for i in range(s + To - 1, t - H + To - 1, Ta):
            j = i - To + 1
            w = ac[j + To - 1: j + To - 1 + Ta]
            d.append(np.linalg.norm(w[-1] - w[0]))
    return float(np.percentile(d, q))


def uniform_chunks(agent, block_c, n_samples, r_max, To, Ta, H, sectors, rng):
    """(B, n_samples, H, 2) straight-line chunks into the block-facing half. Owes nothing
    to the policy.

    Laid out exactly like `action_pred`, because `_verifier_inputs` simulates
    action[To-1 : To-1+Ta] and nothing else: index 0 holds the current arm position, the
    executed window sweeps to the drawn endpoint, and the tail holds it there. The chunk is
    plotted whole but only that window is scored.
    """
    B = len(agent)
    to_block = block_c - agent                                    # (B, 2)
    base = np.arctan2(to_block[:, 1], to_block[:, 0])             # (B,)
    edges = np.arange(sectors + 1) * 2 * np.pi / sectors          # wedge boundaries
    centres = (edges[:-1] + edges[1:]) / 2
    keep = np.abs(np.angle(np.exp(1j * (centres[None] - base[:, None])))) <= np.pi / 2
    per = np.zeros(B, dtype=int)
    out = np.empty((B, n_samples, H, 2), dtype=np.float32)
    meta = np.empty((B, n_samples), dtype=int)
    for b in range(B):
        idx = np.flatnonzero(keep[b])
        assert len(idx), 'no block-facing wedge'
        reps = int(np.ceil(n_samples / len(idx)))
        wedges = np.tile(idx, reps)[:n_samples]
        theta = edges[wedges] + rng.random(n_samples) * (2 * np.pi / sectors)
        r = rng.random(n_samples) * r_max
        end = agent[b] + np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
        # fractions 1/Ta .. 1 -- the first waypoint already moves, the last IS the endpoint
        f = np.arange(1, Ta + 1)[:, None] / Ta
        out[b, :, :To - 1] = agent[b]
        out[b, :, To - 1:To - 1 + Ta] = agent[b] + f[None] * (end - agent[b])[:, None]
        out[b, :, To - 1 + Ta:] = end[:, None]
        meta[b], per[b] = wedges, len(idx)
    return out, meta, per


def tracker_snapshots(rb, pts, plans_by_ep, To):
    """{(episode, i): DualTracker} -- the live tracker state at each decision point,
    from one demo replay per episode.

    CONVENTION (must match VeritasTrackerWrapper, which is what eval's verifier branches
    from): anchor at the episode's reset row ``s`` with ZERO updates, then one ``update``
    per POST-STEP row ``s+1 .. i``. Every recorded row between decisions is replayed --
    the tracker's hold/stall counters are per-frame, so a Ta-strided replay would be a
    different tracker. (scripts/veritas_tracker_demo_check.py also updates on the anchor
    row; that is a diagnostic-only deviation -- do not copy it here.)
    """
    ends = np.asarray(rb.episode_ends[:])
    starts = np.concatenate([[0], ends[:-1]])
    ap, bp = np.asarray(rb['agent_pos']), np.asarray(rb['block_pos'])
    want = {}
    for e, i, _ in pts:
        want.setdefault(e, []).append(i)
    snaps = {}
    for e, idxs in want.items():
        s = int(starts[e])
        tracker = DualTracker(plans_by_ep[e], ap[s], bp[s])
        idxs = sorted(set(idxs))
        j = 0
        if idxs[0] == s:                       # decision at the reset row itself
            snaps[(e, s)] = tracker.copy()
            j = 1
        for f in range(s + 1, idxs[-1] + 1):
            tracker.update(ap[f], bp[f])
            if j < len(idxs) and f == idxs[j]:
                snaps[(e, f)] = tracker.copy()
                j += 1
    return snaps


def save_decision_viz(path, image, a_pol, a_uni, a_star, v_pol, v_uni, v_star,
                      To, Ta, title):
    """One decision: every scored chunk over the obs frame, coloured by verifier value.

    Colourblind-safe by construction: the VALUE is the colour (viridis + colourbar), the
    SOURCE is the line style (policy solid, uniform dotted), and the two highlighted
    chunks -- a* and the argmax pick -- get distinct markers, heavier lines and their
    values written in the legend, so nothing is told apart by hue alone. Only the
    executed window [To-1, To-1+Ta) is drawn: it is the only part the verifier scored.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    px = 96.0 / 512.0
    sl = slice(To - 1, To - 1 + Ta)
    vals = np.concatenate([v_pol, v_uni, [v_star]])
    norm = colors.Normalize(vmin=float(vals.min()), vmax=float(vals.max()))
    cmap = cm.viridis
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(np.moveaxis(np.asarray(image), 0, -1), extent=[0, 96, 96, 0])
    best_src, best_k = max(
        [('policy', int(v_pol.argmax())), ('uniform', int(v_uni.argmax()))],
        key=lambda sk: {'policy': v_pol, 'uniform': v_uni}[sk[0]][sk[1]])
    for name, chunks, vv, style in (('policy', a_pol, v_pol, '-'),
                                    ('uniform', a_uni, v_uni, ':')):
        for k in range(len(chunks)):
            w = chunks[k][sl] * px
            hot = (name == best_src and k == best_k)
            ax.plot(w[:, 0], w[:, 1], style, color=cmap(norm(vv[k])),
                    lw=2.6 if hot else 1.0, alpha=1.0 if hot else 0.75, zorder=4 if hot else 2)
            if hot:
                ax.plot(w[-1, 0], w[-1, 1], 'o', ms=9, mfc='none', mew=2.2,
                        color=cmap(norm(vv[k])), zorder=5)
    w = a_star[sl] * px
    ax.plot(w[:, 0], w[:, 1], '--', color=cmap(norm(v_star)), lw=2.6, zorder=6)
    ax.plot(w[-1, 0], w[-1, 1], '*', ms=15, color=cmap(norm(v_star)),
            mec='black', mew=0.8, zorder=7)
    best_v = {'policy': v_pol, 'uniform': v_uni}[best_src][best_k]
    handles = [
        plt.Line2D([], [], color='0.3', ls='-', label='policy candidate'),
        plt.Line2D([], [], color='0.3', ls=':', label='uniform sample'),
        plt.Line2D([], [], color=cmap(norm(v_star)), ls='--', marker='*', ms=11,
                   mec='black', label=f'a* (expert)  v={v_star:.4g}'),
        plt.Line2D([], [], color=cmap(norm(best_v)), ls='-', marker='o', mfc='none',
                   mew=2, label=f'argmax ({best_src})  v={best_v:.4g}'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.85)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046,
                 label='verifier value')
    ax.set_xlim(0, 96); ax.set_ylim(96, 0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_batch(rb, pts, To, H, device):
    """obs_dict (B,To,...), a* (B,H,2), and the demo block poses over the executed window.

    Mirrors PushTImageDataset._sample_to_data -- same moveaxis, same /255, same
    compute_feedback_from_pose on the stored block_pos -- rather than inventing a parallel
    construction that could drift from what the policy was trained on. The obs dict must be
    normalizer-complete and carry no extra keys.
    """
    img, ap = np.asarray(rb['img']), np.asarray(rb['agent_pos'])
    ac, bp = np.asarray(rb['action']), np.asarray(rb['block_pos'])
    im, pos, fb, star, poses = [], [], [], [], []
    for _, i, _ in pts:
        j = i - To + 1
        im.append(np.moveaxis(img[j:i + 1], -1, 1) / 255)
        pos.append(ap[j:i + 1])
        fb.append(compute_feedback_from_pose(bp[j:i + 1].astype(np.float32)))
        star.append(ac[j:j + H])
        poses.append(bp[i:i + H - To + 2].astype(np.float32))
    t = lambda x: torch.tensor(np.stack(x), dtype=torch.float32, device=device)  # noqa: E731
    return {'image': t(im), 'agent_pos': t(pos), 'feedback': t(fb)}, t(star), poses


def expert_moved(poses, Ta, eps_px=1e-3):
    """Did the DEMO's own block move over this executed window? Defines the episode phase.

    Taken from the demonstration rather than from the policy's candidates so the phase label
    is a property of the state, identical across arms.
    """
    out = []
    for p in poses:
        w = p[:min(Ta + 1, len(p))]
        out.append(bool(np.linalg.norm(w[-1, :2] - w[0, :2]) > eps_px
                        or abs(float(w[-1, 2] - w[0, 2])) > 1e-4))
    return np.array(out)


def reach_stats(cand, uni, star, agent, r_max):
    """How far each source actually moves the arm. A FAIRNESS AUDIT, not a result.

    The uniform competitor draws its radius from U(0, r_max) with r_max fixed at the 95th
    percentile of the expert's own net move. If the policy's chunks are much shorter than
    that, uniform reaches further, contacts the block more often, and wins on reach rather
    than on being better aimed -- so any provenance number has to be read next to these.
    """
    def net(x):
        return np.linalg.norm(x[..., -1, :] - agent.reshape(-1, *([1] * (x.ndim - 3)), 2),
                              axis=-1)
    dp, du = net(cand), net(uni)
    ds = np.linalg.norm(star[:, -1, :] - agent, axis=-1)
    q = lambda v: [float(np.percentile(v, k)) for k in (5, 50, 95)]      # noqa: E731
    return {'r_max_px': float(r_max), 'policy_net_move_px_p5_50_95': q(dp),
            'uniform_net_move_px_p5_50_95': q(du), 'astar_net_move_px_p5_50_95': q(ds),
            'policy_median_over_astar_median': float(np.median(dp) / max(np.median(ds), 1e-9))}


def recall_stats(cand, star):
    """How close does the candidate cloud get to a*, against its OWN spread.

    cand (S, n, H, 2), star (S, H, 2). A raw distance means nothing on its own: a wide cloud
    lands near a* by luck, so every distance is also reported divided by the cloud's spread
    at that decision.
    """
    ce, se = cand[:, :, -1, :], star[:, None, -1, :]          # chunk endpoints
    d_end = np.linalg.norm(ce - se, axis=-1)                  # (S, n)
    d_chunk = np.linalg.norm(cand - star[:, None], axis=-1).mean(axis=-1)
    centroid = ce.mean(axis=1, keepdims=True)
    spread = np.linalg.norm(ce - centroid, axis=-1).mean(axis=1)      # (S,)
    nearest = d_end.min(axis=1)
    safe = np.where(spread > 1e-9, spread, np.nan)
    return {
        # THE RECALL NUMBER: does ANY of the n draws land on the expert?
        'min_endpoint_dist_px': float(nearest.mean()),
        'min_endpoint_dist_px_median': float(np.median(nearest)),
        'min_chunk_dist_px': float(d_chunk.min(axis=1).mean()),
        'mean_endpoint_dist_px': float(d_end.mean()),
        'cloud_spread_px': float(np.nanmean(spread)),
        # < 1 means the cloud reaches a* more closely than its own members reach its centre
        'min_dist_over_spread': float(np.nanmean(nearest / safe)),
        'frac_decisions_a_star_inside_spread': float((nearest <= spread).mean()),
    }


def _spearman(a, b):
    """Rank correlation along the last axis, row by row. NaN where a row is constant.

    Written out rather than pulled from scipy: the rows are tiny (n=16) and a third of them
    are CONSTANT, because every candidate that misses the block scores identically. scipy
    would warn and return nan per row; doing it here makes that case explicit.
    """
    def rank(x):
        o = x.argsort(axis=-1)
        r = np.empty_like(o, dtype=float)
        np.put_along_axis(r, o, np.arange(x.shape[-1], dtype=float), axis=-1)
        # average ranks over ties, or every tied row would get a spurious ordering
        for i in range(x.shape[0]):
            _, inv, cnt = np.unique(x[i], return_inverse=True, return_counts=True)
            sums = np.zeros(len(cnt))
            np.add.at(sums, inv, r[i])
            r[i] = (sums / cnt)[inv]
        return r
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean(-1, keepdims=True)
    rb = rb - rb.mean(-1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))
    return np.where(den > 1e-12, (ra * rb).sum(-1) / np.where(den > 1e-12, den, 1), np.nan)


def steering(v_cand, cand, star, cls):
    """Does the verifier's ranking point AT a*, and do later slots get closer to it?

    This is the "guide the next prediction" half of the question. Candidate i is generated
    conditioned on the scored candidates before it, so if the ranking signal tracked
    closeness-to-expert the context would be steering generation toward a*.

      spearman  per decision, across the n candidates, between the verifier score and
                MINUS the distance to a*. Positive => the verifier prefers candidates nearer
                the expert. ~0 => its ordering is orthogonal to the expert.
      by_slot   mean distance to a* at each slot index. For ST k=1 and UNet BC the slots are
                i.i.d. draws, so this MUST come out flat; a trend there is a bug, not a result.
    """
    d = np.linalg.norm(cand[:, :, -1, :] - star[:, None, -1, :], axis=-1)   # (S, n)
    rho = _spearman(v_cand, -d)
    out = {'spearman_score_vs_astar': float(np.nanmean(rho)),
           'n_decisions_with_a_ranking': int(np.isfinite(rho).sum()),
           'dist_to_astar_by_slot_px': [float(v) for v in d.mean(axis=0)]}
    m = cls == INFORMATIVE
    if m.sum() > 1:
        out['spearman_informative_only'] = float(np.nanmean(rho[m]))
        out['dist_to_astar_by_slot_px_informative'] = [float(v) for v in d[m].mean(axis=0)]
    return out


def informativeness(t_cand, ref, v_cand, n, phase):
    """The blind / partial / informative split, plus the effective search width."""
    cls, movers, _ = classify(t_cand, ref)
    distinct = np.array([len(np.unique(r)) for r in v_cand], dtype=float)
    out = {'move_eps_px': MOVE_EPS_PX,
           'n_decisions': int(len(cls)),
           **{f'p_{c}': float((cls == c).mean()) for c in CLASSES},
           'n_points': {c: int((cls == c).sum()) for c in CLASSES},
           'mean_movers_frac': float((movers / n).mean()),
           'mean_distinct': float(distinct.mean()),
           'mean_distinct_frac': float((distinct / n).mean()),
           'by_phase': {}}
    # NAMED FOR WHAT THE TEST ACTUALLY IS. `phase` comes from expert_moved(), which asks
    # whether the DEMO's block pose changes over THIS chunk's executed window -- a per-decision
    # property of the state. It is not a before/after-first-contact split, and the old
    # `approach`/`engaged` labels asserted exactly that.
    for name, m in (('block_still', ~phase), ('block_moving', phase)):
        if not m.any():
            continue
        out['by_phase'][name] = {
            'n_decisions': int(m.sum()),
            **{f'p_{c}': float((cls[m] == c).mean()) for c in CLASSES},
            'mean_movers_frac': float((movers[m] / n).mean()),
            'mean_distinct': float(distinct[m].mean())}
    return out, cls, movers, distinct


def provenance(v_pol, v_uni, v_star):
    """Where did the verifier's favourite come from, over policy + uniform + a*?

    A decision whose top value is tied ACROSS sources is counted as `tie` and excluded from
    the tally: there the winner is whichever index the argmax happened to reach first, and
    counting it would let the tie-break manufacture the result.
    """
    best = {POLICY: v_pol.max(1), UNIFORM: v_uni.max(1), ASTAR: v_star}
    top = np.max(np.stack(list(best.values())), axis=0)
    at = {k: np.isclose(v, top, rtol=0, atol=MOVE_EPS_PX) for k, v in best.items()}
    n_at = np.sum(np.stack(list(at.values())), axis=0)
    tied = n_at > 1
    lab = np.full(len(top), TIE, dtype=object)
    for k in SOURCES:
        lab[~tied & at[k]] = k
    dec = ~tied
    out = {'n_decisions': int(len(top)), 'n_tied_excluded': int(tied.sum()),
           'n_decided': int(dec.sum()),
           'note': 'uniform vs policy budgets are matched, so the null is 0.5 each'}
    for k in SOURCES:
        out[f'p_{k}'] = float((lab[dec] == k).mean()) if dec.any() else None
    # head to head, ignoring a*: the question best-of-n actually turns on
    h2h = v_uni.max(1) - v_pol.max(1)
    live = np.abs(h2h) > MOVE_EPS_PX
    out['uniform_beats_policy'] = float((h2h[live] > 0).mean()) if live.any() else None
    out['n_head_to_head'] = int(live.sum())
    out['mean_uniform_minus_policy'] = float(h2h.mean())
    return out, lab


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('--arm', default='unknown')
@click.option('--n', 'n_actions', default=16, show_default=True,
              help='policy candidates per decision; the uniform budget matches it')
@click.option('--episodes', default=50, show_default=True, help='test episodes to walk')
@click.option('--sectors', default=16, show_default=True,
              help='wedges around the arm; the block-facing half is kept')
@click.option('--split', type=click.Choice(['val', 'test']), default='test', show_default=True)
@click.option('--batch', default=8, show_default=True,
              help='decisions scored at once; each costs batch*(2n+1) verifier sims')
@click.option('-d', '--device', default='cuda:0')
@click.option('--seed', default=42, show_default=True)
@click.option('--verifier-value', default=None, help='override the scoring rule')
@click.option('--plan-dir', default=None,
              help='ep{idx}.json VLM plans for the wp_* values; default: the value\'s own '
                   'prompt dir (eval_search_pusht._WAYPOINT_PLAN_DIRS)')
@click.option('--out', default=None, help='write the stats as JSON here')
@click.option('--viz', default=10, show_default=True,
              help='save this many decision visualizations: every scored chunk over the '
                   'obs frame, coloured by verifier value, a* and the argmax pick '
                   'highlighted. 0 disables.')
@click.option('--viz-dir', default=None,
              help='where the visualizations go; default: <out stem>_viz/ beside --out, '
                   'or astar_walk_viz/ in the cwd without one')
def main(checkpoint, arm, n_actions, episodes, sectors, split, batch, device, seed,
         verifier_value, plan_dir, out, viz, viz_dir):
    policy, cfg = load_policy(checkpoint, device)
    if verifier_value is not None:
        policy.search_kwargs['verifier_value'] = verifier_value
        built = policy.__dict__.get('_verifier') or policy.__dict__.get('verifier')
        if built is not None:
            if is_waypoint_value(verifier_value) and not hasattr(built, 'set_trackers'):
                # a wp value cannot be assigned onto a plain PushTVerifier (its setter
                # refuses); rebuild through the policy's factory, exactly as
                # eval_search_pusht.eval_checkpoint does.
                built.close()
                rebuilt = policy._build_verifier(**policy.search_kwargs)
                if '_verifier' in policy.__dict__:
                    policy._verifier = rebuilt
                else:
                    policy.verifier = rebuilt
            else:
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
    pts = decision_points(rb, ep_idxs, To, H, Ta)
    r_max = measure_rmax(rb, ep_idxs, To, Ta, H)
    rng = np.random.default_rng(seed)
    snaps = None
    if is_waypoint_value(vv):
        # per-episode VLM plans + the live tracker state at every decision, from a demo
        # replay -- the same state eval's VeritasTrackerWrapper would hold there. The
        # WALKED episodes (post-truncation), not the whole split: load_split_plans
        # refuses any gap, and only these episodes are scored.
        if plan_dir is None:
            plan_dir = _WAYPOINT_PLAN_DIRS[vv]
        plans = load_split_plans(plan_dir, ep_idxs, split, vv)
        snaps = tracker_snapshots(rb, pts, dict(zip(ep_idxs, plans)), To)
        print(f'waypoint plans: {plan_dir} ({len(plans)} episodes)')
    print(f'{arm} step {step}: {len(pts)} decisions over {len(ep_idxs)} {split} episodes, '
          f'n={n_actions} policy + {n_actions} uniform, verifier={vv}'
          + ('  (OVERRIDDEN)' if verifier_value and verifier_value != native else ''))
    print(f'uniform competitor: {sectors} wedges, block-facing half kept, '
          f'r_max={r_max:.1f}px (95th pct of the expert net move)')

    V_pol, V_uni, V_star, T_pol, T_uni, T_star = [], [], [], [], [], []
    A_pol, A_uni, A_star, refs, phases, eps_of, agents = [], [], [], [], [], [], []
    progress = []      # (pusher, t) tracker progress at each decision, wp values only
    # decisions to visualize: evenly spaced across the whole walk, so the panel spans
    # approach and contact rather than sampling one episode's opening moves
    viz_idx = (set(np.linspace(0, len(pts) - 1, min(viz, len(pts))).astype(int).tolist())
               if viz else set())
    viz_rows = []
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        for b0 in range(0, len(pts), batch):
            chunk = pts[b0:b0 + batch]
            obs, star, poses = build_batch(rb, chunk, To, H, torch.device(device))
            now = obs['feedback'][:, To - 1].cpu().numpy()
            refs.append(t_goal_distance(now))
            agent = obs['agent_pos'][:, To - 1].cpu().numpy()
            uni, _, _ = uniform_chunks(agent, t_center_from_feedback(now), n_actions,
                                       r_max, To, Ta, H, sectors, rng)
            uni_t = torch.tensor(uni, dtype=torch.float32, device=device)
            seeder = getattr(policy, 'set_sample_seeds', None)
            if seeder is not None:
                seeder([seed * 1_000_003 + b0 + k for k in range(len(chunk))])
            if snaps is not None:
                # one call covers everything scored in this batch (policy candidates,
                # uniform chunks and a* share the same B rows, and the verifier copies
                # per candidate). Hand in copies so the stored snapshots stay pristine.
                policy.set_waypoint_trackers(
                    [snaps[(e, i)].copy() for e, i, _ in chunk])
                progress += [snaps[(e, i)].progress for e, i, _ in chunk]
            with torch.no_grad(), policy._crop_scope(), policy._corrupt_scope():
                feats = policy._encode_obs_features(obs)
                acts, _, sc, tm = policy.predict_n_actions(
                    obs, verifier=policy.verifier, n_actions=n_actions,
                    return_scores=True, obs_features=feats, return_terms=True)
                # a* and every uniform chunk go through THE SAME call the candidates do --
                # same verifier, same sim pool, same window. A separate scoring path could
                # drift from the ranking the policy actually applies, which is under test.
                o_star = policy._score_candidates(policy.verifier, obs, star)
                u_v, u_t = [], []
                for k in range(n_actions):
                    o = policy._score_candidates(policy.verifier, obs, uni_t[:, k])
                    u_v.append(o[1].float().cpu().numpy())
                    u_t.append(o[3].float().cpu().numpy())
            V_pol.append(sc.float().cpu().numpy())
            T_pol.append(tm.float().cpu().numpy())
            V_star.append(o_star[1].float().cpu().numpy())
            T_star.append(o_star[3].float().cpu().numpy())
            V_uni.append(np.stack(u_v, axis=1))
            T_uni.append(np.stack(u_t, axis=1))
            A_pol.append(acts.float().cpu().numpy())
            A_uni.append(uni)
            A_star.append(star.float().cpu().numpy())
            agents.append(agent)
            phases.append(expert_moved(poses, Ta))
            eps_of += [e for e, _, _ in chunk]
            for r, (e, i, _) in enumerate(chunk):
                if b0 + r not in viz_idx:
                    continue
                viz_rows.append({
                    'ep': e, 'frame': i,
                    'image': obs['image'][r, To - 1].cpu().numpy(),
                    'a_pol': A_pol[-1][r], 'a_uni': A_uni[-1][r], 'a_star': A_star[-1][r],
                    'v_pol': V_pol[-1][r], 'v_uni': V_uni[-1][r],
                    'v_star': float(V_star[-1][r]),
                })
            print(f'  {min(b0 + batch, len(pts))}/{len(pts)}', end='\r', flush=True)
    finally:
        close = getattr(policy, 'close', None)
        if close is not None:
            try:
                close()
            except Exception as e:
                print(f'warning: verifier close failed: {e}')

    cat = np.concatenate
    v_pol, v_uni, v_star = cat(V_pol), cat(V_uni), cat(V_star)
    t_pol, t_uni, t_star = cat(T_pol), cat(T_uni), cat(T_star)
    ref, phase, ep = cat(refs), cat(phases), np.array(eps_of)
    a_pol, a_uni, a_star, agent_all = cat(A_pol), cat(A_uni), cat(A_star), cat(agents)

    inf_pol, cls, _, _ = informativeness(t_pol[:, :, T_GOAL], ref, v_pol, n_actions, phase)
    inf_uni, cls_u, _, _ = informativeness(t_uni[:, :, T_GOAL], ref, v_uni, n_actions, phase)
    prov, _ = provenance(v_pol, v_uni, v_star)

    # NO-TOUCH DISCRIMINATION: how the verifier orders the chunks that never move the T.
    # Under t_goal this is degenerate by construction (the ties classify keys on); under
    # the wp values it is the question -- the pusher track varies on exactly these chunks.
    _, _, mv_pol = classify(t_pol[:, :, T_GOAL], ref)
    _, _, mv_uni = classify(t_uni[:, :, T_GOAL], ref)
    star_mover = np.abs(t_star[:, T_GOAL] - ref) > MOVE_EPS_PX
    nontouch = nontouch_stats(v_pol, v_uni, v_star, mv_pol, mv_uni, star_mover)

    res = {'arm': arm, 'step': step, 'checkpoint': str(checkpoint), 'split': split,
           'n': n_actions, 'sectors': sectors, 'r_max_px': r_max,
           'n_decisions': int(len(v_star)), 'n_episodes': int(len(set(eps_of))),
           'verifier_value': vv, 'verifier_value_native': native, 'seed': seed,
           'plan_dir': (str(plan_dir) if snaps is not None else None),
           'wp_progress': ([[float(p), float(t)] for p, t in progress]
                           if snaps is not None else None),
           'nontouch': nontouch,
           'informativeness': {'policy': inf_pol, 'uniform': inf_uni},
           'provenance': prov,
           'reach': reach_stats(a_pol, a_uni, a_star, agent_all, r_max),
           'recall': recall_stats(a_pol, a_star),
           'steering': steering(v_pol, a_pol, a_star, cls),
           'astar': {'all': _stats(v_star, v_pol, True, ep)}}
    for c in CLASSES:
        m = cls == c
        if m.sum() < 2 or len(np.unique(ep[m])) < 2:
            res['astar'][c] = {'n_points': int(m.sum()), 'note': 'too few to bootstrap'}
            continue
        res['astar'][c] = {'n_points': int(m.sum()),
                           **_stats(v_star[m], v_pol[m], True, ep[m])}
        res['recall_' + c] = recall_stats(a_pol[m], a_star[m])
    report(res, n_actions)
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out).write_text(json.dumps(res, indent=2))
        print(f'\n-> {out}')
    if viz_rows:
        vdir = pathlib.Path(viz_dir) if viz_dir else (
            pathlib.Path(out).parent / (pathlib.Path(out).stem + '_viz')
            if out else pathlib.Path('astar_walk_viz'))
        vdir.mkdir(parents=True, exist_ok=True)
        for row in viz_rows:
            name = f'ep{row["ep"]}_frame{row["frame"]}_{vv}.png'
            save_decision_viz(
                vdir / name, row['image'], row['a_pol'], row['a_uni'], row['a_star'],
                row['v_pol'], row['v_uni'], row['v_star'], To, Ta,
                f'{arm} step {step} | ep {row["ep"]} frame {row["frame"]} | '
                f'verifier {vv}')
        print(f'-> {len(viz_rows)} decision visualizations in {vdir}/')


def report(res, n):
    ip, iu = res['informativeness']['policy'], res['informativeness']['uniform']
    print(f'\n{"="*78}\nWHEN IS THE VERIFIER USEFUL?  ({ip["n_decisions"]} decisions, '
          f'{res["n_episodes"]} episodes)\n{"="*78}')
    print(f'{"":28s}{"policy cands":>16s}{"uniform samples":>18s}')
    for c in CLASSES:
        print(f'  {c:<26s}{ip[f"p_{c}"]:>15.1%}{iu[f"p_{c}"]:>18.1%}')
    print(f'  {"mean candidates moving T":<26s}{ip["mean_movers_frac"]:>15.1%}'
          f'{iu["mean_movers_frac"]:>18.1%}')
    print(f'  {"effective search width":<26s}{ip["mean_distinct"]:>12.2f}/{n}'
          f'{iu["mean_distinct"]:>15.2f}/{n}')
    if res.get('verifier_value', '').startswith('wp_'):
        print(f'\n  BLIND here means no candidate MOVED THE T -- under this waypoint value the\n'
              f'  scores there are NOT tied (the pusher track still varies); see the NO-TOUCH\n'
              f'  section below for how much ordering survives on those decisions.')
    else:
        print(f'\n  BLIND means every score is identical -- the verifier ranks nothing there, '
              f'and\n  best-of-n buys nothing. n is a real compute axis only on the other '
              f'{1 - ip["p_blind"]:.0%}.')
    if ip['by_phase']:
        print(f'\n  split by whether the DEMO\'s own chunk moves the block (a property of the\n'
              f'  state, so the same decisions fall in each subset for every arm):')
        for k, d in ip['by_phase'].items():
            print(f'    {k:<13s} {d["n_decisions"]:5d} decisions   blind {d["p_blind"]:6.1%}'
                  f'   movers {d["mean_movers_frac"]:5.1%}')

    p = res['provenance']
    print(f'\n{"="*78}\nWHOSE ACTION DOES THE VERIFIER PREFER?  '
          f'({p["n_decided"]} decided, {p["n_tied_excluded"]} tied and excluded)\n{"="*78}')
    for k in SOURCES:
        v = p[f'p_{k}']
        print(f'  top pick is {k:<10s} {"n/a" if v is None else f"{v:6.1%}"}')
    u = p['uniform_beats_policy']
    print(f'  uniform beats policy head-to-head: '
          f'{"n/a" if u is None else f"{u:.1%}"} of {p["n_head_to_head"]} live decisions '
          f'(null 50%)')
    print('  Budgets are matched at 16 v 16, so 50% is the coin flip. Well above it means\n'
          '  best-of-n is limited by what the policy GENERATES, not by the ranking.')

    nt = res.get('nontouch')
    if nt:
        print(f'\n{"="*78}\nNO-TOUCH DISCRIMINATION: ranking among chunks that never move '
              f'the T\n{"="*78}')
        for k in ('policy', 'uniform'):
            d = nt[k]
            if d['mean_distinct'] is None:
                print(f'  {k:<10s} no decision had 2+ non-mover chunks')
                continue
            print(f'  {k:<10s} {d["n_decisions_2plus_nonmovers"]:4d} decisions with 2+ '
                  f'non-movers   distinct {d["mean_distinct"]:5.2f} of '
                  f'{d["mean_nonmovers"]:.1f}   ranked {d["frac_ranked"]:6.1%}   '
                  f'spread {d["mean_spread"]:.4f}')
        a = nt['astar_nontouch']
        if a['mean_rank_frac'] is not None:
            print(f'  a* among non-mover policy chunks ({a["n_decisions"]} decisions where '
                  f'a* is a non-mover):\n'
                  f'    mean rank fraction {a["mean_rank_frac"]:.3f} (0 = best, null 0.500)'
                  f'   p_best {a["p_best"]:.3f}')
        pn = nt['provenance_nontouch']
        if pn['n_decided']:
            print(f'  top non-mover pick: policy {pn["p_policy"]:.1%}  uniform '
                  f'{pn["p_uniform"]:.1%}  a* {pn["p_astar"]:.1%}  '
                  f'({pn["n_decided"]} decided, {pn["n_tied_excluded"]} tied excluded)')
        print('  Under t_goal non-movers tie by construction, so distinct~1 and "ranked" ~0%\n'
              '  IS the t_goal result; a waypoint value earns its keep by ranking here.')

    rc = res['reach']
    print(f'\n{"="*78}\nFAIRNESS AUDIT: how far does each source move the arm?\n{"="*78}')
    for k, lab in [('policy_net_move_px_p5_50_95', 'policy candidates'),
                   ('uniform_net_move_px_p5_50_95', 'uniform samples'),
                   ('astar_net_move_px_p5_50_95', 'a* (expert)')]:
        a, b, c = rc[k]
        print(f'  {lab:<20s} net move px   p5 {a:6.1f}   median {b:6.1f}   p95 {c:6.1f}')
    print(f'  uniform radius cap r_max = {rc["r_max_px"]:.1f}px; policy median is '
          f'{rc["policy_median_over_astar_median"]:.2f}x the expert median.')
    print('  If the policy moves much less than uniform, part of any uniform win is REACH,\n'
          '  not aim -- read the provenance number against these three rows.')

    print(f'\n{"="*78}\nIS a* SAMPLED?  distance from the candidate cloud to the expert\n{"="*78}')
    for key, lab in [('recall', 'all decisions'), ('recall_' + INFORMATIVE, 'informative only')]:
        r = res.get(key)
        if not r:
            continue
        print(f'  {lab:<20s} nearest of {n} draws {r["min_endpoint_dist_px"]:6.1f}px   '
              f'cloud spread {r["cloud_spread_px"]:6.1f}px   '
              f'ratio {r["min_dist_over_spread"]:.2f}   '
              f'a* within spread {r["frac_decisions_a_star_inside_spread"]:.0%}')
    print('  ratio < 1 means the cloud reaches a* more closely than its own members reach\n'
          '  its centre -- i.e. a* is inside the support, not merely near a wide blob.')

    s = res['steering']
    print(f'\n{"="*78}\nDOES THE RANKING STEER TOWARD a*?\n{"="*78}')
    print(f'  spearman(verifier score, -distance to a*) across the {n} candidates:')
    print(f'    all decisions     {s["spearman_score_vs_astar"]:+.3f}   '
          f'({s["n_decisions_with_a_ranking"]} decisions had any ordering at all)')
    if 'spearman_informative_only' in s:
        print(f'    informative only  {s["spearman_informative_only"]:+.3f}')
    print('    positive => the verifier prefers candidates nearer the expert; ~0 => its\n'
          '    ordering is orthogonal to a*, so the context cannot be steering toward it.')
    bs = s.get('dist_to_astar_by_slot_px_informative') or s['dist_to_astar_by_slot_px']
    print(f'  distance to a* by slot: '
          f'{bs[0]:.1f} -> {bs[len(bs)//2]:.1f} -> {bs[-1]:.1f}px (slot 0 / mid / {len(bs)-1})')
    print('    for ST k=1 and UNet BC the slots are i.i.d. draws, so this MUST be flat.')

    print(f'\n{"="*78}\nDOES THE VERIFIER RANK a* ABOVE THE CANDIDATES?\n{"="*78}')
    print(f'{"stratum":<16s}{"pts":>7s}{"mean rank/" + str(n):>16s}{"cands beaten":>15s}'
          f'{"a* is best":>13s}')
    for c in ('all',) + CLASSES:
        b = res['astar'].get(c)
        if not b or 'mean_rank' not in b:
            print(f'{c:<16s}{b.get("n_points", 0) if b else 0:>7d}   '
                  f'{(b or {}).get("note", "absent")}')
            continue
        pts = b.get('n_points', res['n_decisions'])
        print(f'{c:<16s}{pts:>7d}{b["mean_rank"]["v"]:>16.2f}'
              f'{b["frac_candidates_beaten"]["v"]:>15.3f}{b["p_best"]["v"]:>13.3f}')
    print(f'\n  Null if a* were exchangeable with the candidates: rank {n/2:.1f}, '
          f'beaten 0.500, best {1/(n+1):.3f}.')
    print('  READ THE `informative` ROW. On blind decisions a* ties every candidate by\n'
          '  construction, so mid-rank puts it at chance there and pooling drags every\n'
          '  stratum toward the null.')


if __name__ == '__main__':
    main()
