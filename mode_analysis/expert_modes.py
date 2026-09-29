"""Is the DEMONSTRATOR multimodal at the states where the policy's candidates are not?

    python mode_analysis/expert_modes.py --split-file diffusion_policy/config/splits/<M>.json

candidate_modes.py answers "how spread are the policy's candidates at this state" by decoding
one observation R times. It draws the demonstrator action a* on each panel as a single dashed
reference -- ONE trajectory, which cannot show whether the data itself had a choice to make.
This script supplies the missing half.

HOW A DISTRIBUTION IS OBTAINED FROM SINGLE-ACTION DEMOS. Each window carries exactly one
expert action, so multimodality only appears by POOLING NEARBY STATES: every window in the
dataset whose conditioning state is within epsilon of the query is treated as a draw from the
expert's conditional. Nearness is measured on the FULL state the policy conditions on --
agent position AND block pose -- so the pooled set is the choice the demonstrator faced from
the same configuration, not merely from the same block pose reached by a different approach.

    |agent_pos - q_agent| <= eps_agent      (euclidean, arena px)
    |block_xy  - q_block| <= eps_block      (euclidean, arena px)
    |wrap(block_theta - q_theta)| <= eps_theta   (radians)

NEIGHBOURS COME FROM ALL 206 EPISODES, train/val/test alike. The question here is what the
DATA contains, not what one split contains, and pooling the whole set is what makes the
neighbourhoods dense enough to be a distribution rather than a handful of points. The
consequence is that a mode the policy never saw still counts, so a gap between this cloud and
the candidate cloud is NOT by itself evidence that the policy dropped a mode it was taught.

EPSILON IS ARBITRARY AND THE RESULT DEPENDS ON IT: a larger ball pools states that are
genuinely different and inflates the spread. There is no correct value, so rather than pick
one, --eps-sweep reports dispersion as a function of the radius and the headline figure is
that curve. Read it against the candidate dispersion from modes.json at the same state.

Windows come from the project's own SequenceSampler over a replay buffer loaded WITHOUT
`img`, so this is the dataset's real windowing at a few MB instead of 2.8GB, and needs no
checkpoint and no GPU.

OUTPUT under --outdir:
    expert_modes_state<i>.png    trajectories + endpoint scatter for each eps in the sweep
    expert_dispersion.png        expert dispersion vs eps, averaged over states
    expert_modes.json            the numbers, including neighbour counts per state per eps
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, MUTED = '#1a1a19', '#8a8880'
C_EXP, C_STAR = '#1baf7a', '#eb6834'      # expert cloud vs the query's own a*
ARENA = (5, 506)


def wrap(a):
    """Signed angle difference in (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def load_windows(zarr_path, horizon, pad_before, pad_after, n_obs_steps):
    """(states, actions) for EVERY window of every episode.

    states:  (M, 5) [agent_x, agent_y, block_x, block_y, block_theta] at the LAST observed
             step of the window (index n_obs_steps-1), which is what the policy conditions on.
    actions: (M, horizon, 2) agent targets in arena pixels, the same units and frame the
             candidates are plotted in.
    starts:  (M,) buffer index each window begins at, used to decide split membership.
    """
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    from diffusion_policy.common.sampler import SequenceSampler
    # No 'img': the 2.8GB array is never read here, and dropping it is what lets this run
    # on a login node.
    rb = ReplayBuffer.copy_from_path(str(zarr_path),
                                     keys=['agent_pos', 'block_pos', 'action'])
    sampler = SequenceSampler(replay_buffer=rb, sequence_length=horizon,
                              pad_before=pad_before, pad_after=pad_after,
                              episode_mask=np.ones(rb.episode_ends[:].shape, dtype=bool))
    obs_i = int(n_obs_steps) - 1
    states, actions = [], []
    for i in range(len(sampler)):
        s = sampler.sample_sequence(i)
        states.append(np.concatenate([s['agent_pos'][obs_i], s['block_pos'][obs_i]]))
        actions.append(s['action'])
    return (np.asarray(states, dtype=np.float64),
            np.asarray(actions, dtype=np.float64),
            np.asarray(sampler.indices[:, 0], dtype=np.int64))


def test_windows(split_file, zarr_path, starts, n_states):
    """The same TEST windows candidate_modes.py analyses, as indices into `states`.

    candidate_modes builds a test-split dataset and takes linspace(0, len-1, n_states) over
    ITS WINDOWS. The sampler here runs over all episodes, so test windows are recovered by
    episode membership -- a window belongs to the split its starting frame does, windows
    never crossing an episode boundary -- and the same linspace is then applied, giving the
    identical states to overlay against. Selecting on frames rather than windows would be
    off: this split has 6701 test frames but only 6424 test windows.
    """
    from diffusion_policy.dataset.pusht_image_dataset import (
        load_split_manifest, masks_from_manifest, episode_frame_mask)
    from diffusion_policy.common.replay_buffer import ReplayBuffer
    ends = ReplayBuffer.copy_from_path(str(zarr_path), keys=['agent_pos']).episode_ends[:]
    man = load_split_manifest(str(split_file), episode_ends=ends)
    _, _, test_mask = masks_from_manifest(man, len(ends))
    in_test = episode_frame_mask(ends, test_mask)[starts]
    win = np.nonzero(in_test)[0]
    return win[np.linspace(0, len(win) - 1, n_states).astype(int)], len(win)


def neighbours(states, q, eps_agent, eps_block, eps_theta):
    """Boolean mask of windows within epsilon of q on the full state."""
    da = np.linalg.norm(states[:, 0:2] - q[0:2], axis=1)
    db = np.linalg.norm(states[:, 2:4] - q[2:4], axis=1)
    dt = np.abs(wrap(states[:, 4] - q[4]))
    return (da <= eps_agent) & (db <= eps_block) & (dt <= eps_theta)


def dispersion(endpoints):
    """Mean pairwise distance between chunk endpoints -- the SAME statistic
    candidate_modes.dispersion computes, so the two are directly comparable."""
    n = len(endpoints)
    if n < 2:
        return 0.0
    d = np.linalg.norm(endpoints[:, None, :] - endpoints[None, :, :], axis=-1)
    return float(d[np.triu_indices(n, 1)].mean())


def frame(ax, b):
    ax.set_xlim(b[0], b[1]); ax.set_ylim(b[3], b[2])      # y down, as in the env
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_state(per_eps, a_star, out, title):
    """One column per epsilon: trajectories on top, endpoint scatter below."""
    allpts = [c.reshape(-1, 2) for _, c, _ in per_eps if len(c)] + [a_star.reshape(-1, 2)]
    pts = np.concatenate(allpts, axis=0)
    lo, hi = pts.min(0), pts.max(0)
    cx, cy = (lo + hi) / 2
    half = max(hi[0] - lo[0], hi[1] - lo[1], 8.0) * 0.68
    b = (cx - half, cx + half, cy - half, cy + half)

    fig, axes = plt.subplots(2, len(per_eps), figsize=(2.7 * len(per_eps) + 0.8, 6.4),
                             squeeze=False)
    for c_i, (eps, cand, _) in enumerate(per_eps):
        ax = axes[0][c_i]
        for r in range(len(cand)):
            ax.plot(cand[r, :, 0], cand[r, :, 1], color=C_EXP, lw=0.7, alpha=0.45)
        ax.plot(a_star[:, 0], a_star[:, 1], color=C_STAR, lw=2.0, ls=(0, (4, 2)),
                marker='o', ms=2.5, label="a* (this window)")
        frame(ax, b)
        ax.set_title(f'eps {eps[0]:.0f}px / {np.degrees(eps[2]):.0f}°\n{len(cand)} demos',
                     fontsize=9.5, color=INK)
        if c_i == 0:
            ax.set_ylabel('expert trajectories', fontsize=9, color=INK)
        ax2 = axes[1][c_i]
        if len(cand):
            ax2.scatter(cand[:, -1, 0], cand[:, -1, 1], s=14, color=C_EXP,
                        edgecolor=INK, lw=0.3, alpha=0.8)
        ax2.scatter([a_star[-1, 0]], [a_star[-1, 1]], s=70, marker='*', color=C_STAR,
                    edgecolor=INK, lw=0.5, zorder=5)
        frame(ax2, b)
        if c_i == 0:
            ax2.set_ylabel('chunk endpoints', fontsize=9, color=INK)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left',
                      bbox_to_anchor=(0, -0.02))
    fig.suptitle(title + f'   —   view is {2*half:.0f}px across (arena is 512px)',
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=140, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-file', required=True,
                    help='manifest whose TEST windows are the query states')
    ap.add_argument('--zarr', default=str(ROOT / 'data/pusht_cchi_v7_replay.zarr'))
    ap.add_argument('--horizon', type=int, default=16)
    ap.add_argument('--n-obs-steps', type=int, default=2)
    ap.add_argument('--n-action-steps', type=int, default=8)
    ap.add_argument('--n-states', type=int, default=3)
    ap.add_argument('--eps-sweep', default='2,5,10,20,40',
                    help='agent/block radius in arena px; theta scales with it')
    ap.add_argument('--eps-theta-per-px', type=float, default=0.02,
                    help='radians of block rotation allowed per px of the positional radius')
    ap.add_argument('--modes-json', default=None,
                    help="a run's modes.json; its candidate dispersion is drawn as reference "
                         'lines so policy and demonstrator spread share one axis')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    split_file = pathlib.Path(args.split_file)
    out = pathlib.Path(args.outdir or
                       f'/gscratch/robotics/harine/mode_analysis/expert/{split_file.stem}')
    out.mkdir(parents=True, exist_ok=True)

    states, actions, starts = load_windows(args.zarr, args.horizon, args.n_obs_steps - 1,
                                           args.n_action_steps - 1, args.n_obs_steps)
    print(f'{len(states)} windows over all episodes')
    q_idx, n_test = test_windows(split_file, args.zarr, starts, args.n_states)
    print(f'{n_test} of them are TEST windows; querying {len(q_idx)}')

    eps_list = [float(x) for x in args.eps_sweep.split(',')]
    summary = {'split_file': str(split_file), 'n_windows': int(len(states)),
               'eps_theta_per_px': args.eps_theta_per_px, 'states': {}}
    disp_by_eps = {e: [] for e in eps_list}

    for si, qi in enumerate(q_idx):
        q, a_star = states[qi], actions[qi]
        per_eps, rec = [], {}
        for e in eps_list:
            et = e * args.eps_theta_per_px
            m = neighbours(states, q, e, e, et)
            cand = actions[m]
            d = dispersion(cand[:, -1, :]) if len(cand) > 1 else 0.0
            per_eps.append(((e, e, et), cand, d))
            rec[f'{e:g}'] = {'n_demos': int(m.sum()), 'endpoint_dispersion_px': d}
            disp_by_eps[e].append(d)
            print(f'  state {si} eps {e:5.1f}px/{np.degrees(et):4.1f}°  '
                  f'{int(m.sum()):5d} demos  dispersion {d:7.2f}px')
        summary['states'][str(si)] = {'window': int(qi),
                                      'state': [float(v) for v in q], 'eps': rec}
        fig_state(per_eps, a_star, out / f'expert_modes_state{si}.png',
                  f'{split_file.stem} — expert actions pooled by state proximity, '
                  f'state {si} (all 206 episodes)')

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    m = [float(np.mean(disp_by_eps[e])) for e in eps_list]
    ax.plot(eps_list, m, color=C_EXP, marker='o', ms=5, lw=1.7, label='expert (pooled demos)')
    if args.modes_json:
        # The policy's candidate spread at the SAME statistic, as a reference. Flat in eps
        # because it does not depend on the neighbourhood at all -- it is one number per
        # checkpoint, and the point of the plot is where the expert curve crosses it.
        md = json.loads(pathlib.Path(args.modes_json).read_text())
        styles = [(0, (5, 2)), (0, (1.5, 1.5)), (0, (6, 2, 1, 2)), (0, (3, 1, 1, 1))]
        for i, (step, rec) in enumerate(sorted(md.get('steps', {}).items(), key=lambda kv: int(kv[0]))):
            v = rec.get('endpoint_dispersion_px')
            if not v:
                continue
            ax.axhline(float(np.mean(v)), color=MUTED, lw=1.3, ls=styles[i % len(styles)],
                       label=f'candidates, {int(step)//1000}k (mean over slots)')
        summary['modes_json'] = str(args.modes_json)
    ax.set_xlabel('neighbourhood radius (arena px, agent and block)', color=INK)
    ax.set_ylabel('mean pairwise endpoint distance (px)', color=INK)
    ax.set_title(f'{split_file.stem} — expert dispersion vs neighbourhood radius\n'
                 f'averaged over {len(q_idx)} test states; compare to candidate dispersion '
                 f'in modes.json', fontsize=10, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(out / 'expert_dispersion.png', dpi=150, facecolor='white',
                bbox_inches='tight')
    plt.close(fig)

    summary['mean_dispersion_by_eps'] = {f'{e:g}': v for e, v in zip(eps_list, m)}
    (out / 'expert_modes.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'wrote {out}/')


if __name__ == '__main__':
    main()
