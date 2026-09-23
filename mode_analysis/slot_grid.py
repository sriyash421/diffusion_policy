"""How each policy samples a0..a_{K-1} at ONE decision state, with a* on every panel.

    python mode_analysis/slot_grid.py --step 100000 --state 0 --out media/slot_grid.png

WHAT THIS ADDS over modes_step<N>_state<i>.png. That figure is one arm, faceted into K
panels side by side -- fine for reading a single slot, unreadable as a 16-wide strip, and it
cannot put two arms next to each other. This is one panel per ARM with every slot overlaid
and coloured by slot index, which is the view that answers "does the candidate cloud move as
context accumulates, and do the arms differ in how it moves".

SAME STATE FOR EVERY ARM. All seven arms share one split manifest and one transition manifest,
so `test_windows` builds the identical dataset and `--state i` indexes the identical window.
The comparison is therefore paired; without that it would be seven arms at seven states.

ON THE HELD-OUT MOVING TRANSITIONS. test_windows reads PushTImageDataset(split='test'), which
under this generation's transition_file is the filtered moving set -- the same windows §2 and
§3 of the report are measured on.

COLOUR IS ORDINAL, not categorical: slot index runs 0 -> K-1 on viridis, which is
perceptually uniform and colourblind-safe, and the two ends are labelled directly on the
panel as well, so the ordering survives greyscale.
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mode_analysis'))
from candidate_modes import load, sample_candidates, test_windows  # noqa: E402

INK, MUTED = '#1a1a19', '#8a8880'
C_STAR = '#eb6834'
SUB = 'pusht_search/pusht_image_search_imgonly'
ARMS = [('unet_bc', 'unetbc_ver-t_goal', 'BC  (no context)', 16),
        ('outer_inner', 'value_k16_ver-t_goal', 'ST k=16 uniform', None),
        ('outer_inner', 'value_k16_ver-t_goal_son-flat400', 'ST k=16 flat400', None),
        ('outer_inner', 'value_k16_ver-t_goal_son-ramp400to200', 'ST k=16 ramp400to200', None),
        ('outer_inner', 'value_k4_ver-t_goal', 'ST k=4 uniform', None),
        ('outer_inner', 'value_k4_ver-t_goal_son-flat400', 'ST k=4 flat400', None),
        ('outer_inner', 'value_k4_ver-t_goal_son-ramp400to200', 'ST k=4 ramp400to200', None)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/gscratch/robotics/harine/diffusion_policy_outputs')
    ap.add_argument('--step', type=int, default=100000)
    ap.add_argument('--state', type=int, default=0)
    ap.add_argument('--n-states', type=int, default=3)
    ap.add_argument('--repeats', type=int, default=32)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='media/slot_grid.png')
    args = ap.parse_args()

    panels = []
    for sub, stem, label, n_force in ARMS:
        run = (pathlib.Path(args.root) / SUB / sub /
               f'{stem}_enc-resnet18_demos-176_split-mv_seed-42')
        if not (run / 'checkpoints' / f'step_{args.step:07d}.ckpt').is_file():
            print(f'skip {label}: no step {args.step}'); continue
        policy, cfg = load(args.step, run, args.device)
        K = int(getattr(policy, 'max_actions', 1) or 1)
        n_act = n_force or K
        states = test_windows(cfg, str(run), args.n_states, args.device)
        obs, a_star = states[args.state]
        cand = sample_candidates(policy, obs, args.repeats, n_act, args.device)
        panels.append((label, cand, a_star, n_act))
        print(f'{label}: {cand.shape} (repeats, slots, horizon, 2)')
        del policy
        torch.cuda.empty_cache()

    if not panels:
        print('nothing to plot'); return

    n = len(panels)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.0 * nrow), squeeze=False)
    # ONE zoom for every panel, so cloud sizes are comparable across arms by eye.
    allpts = np.concatenate([c[:, :, -1, :].reshape(-1, 2) for _, c, _, _ in panels]
                            + [panels[0][2][None, -1, :]])
    ctr = allpts.mean(axis=0)
    half = max(float(np.abs(allpts - ctr).max()) * 1.15, 12.0)

    for ax, (label, cand, a_star, n_act) in zip(axes.ravel(), panels):
        cmap = plt.get_cmap('viridis')
        for s in range(n_act):
            e = cand[:, s, -1, :]
            ax.scatter(e[:, 0], e[:, 1], s=11,
                       color=cmap(s / max(n_act - 1, 1)), alpha=0.75,
                       linewidths=0, zorder=2)
        # a*, on every panel, deliberately a different SHAPE as well as colour
        ax.scatter(*a_star[-1], marker='*', s=260, color=C_STAR,
                   edgecolor='white', linewidths=0.8, zorder=4, label='a* (demo)')
        ax.annotate('slot 0', xy=cand[:, 0, -1, :].mean(axis=0), fontsize=7.5,
                    color=cmap(0.0), weight='bold',
                    xytext=(4, 6), textcoords='offset points', zorder=5)
        ax.annotate(f'slot {n_act-1}', xy=cand[:, -1, -1, :].mean(axis=0), fontsize=7.5,
                    color=cmap(1.0), weight='bold',
                    xytext=(4, -10), textcoords='offset points', zorder=5)
        ax.set_xlim(ctr[0] - half, ctr[0] + half)
        ax.set_ylim(ctr[1] - half, ctr[1] + half)
        ax.set_aspect('equal')
        ax.set_title(f'{label}   (K={n_act})', fontsize=10, color=INK, loc='left')
        ax.tick_params(labelsize=7, colors=MUTED)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cb.set_label('candidate slot  (0 = no context  ->  K-1 = full context)', color=INK,
                 fontsize=9)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    axes.ravel()[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left')
    fig.suptitle(f'Chunk ENDPOINTS, {args.repeats} seeds per slot — step {args.step//1000}k, '
                 f'held-out moving transition #{args.state}\n'
                 f'one shared zoom across panels; arena is 512px',
                 color=INK, fontsize=12, x=0.01, ha='left')
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
