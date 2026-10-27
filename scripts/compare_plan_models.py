"""Overlay several Gemini models' v3 pusher plans on the same start frame, one tile per episode.

Each `--plan` is LABEL=DIR, a folder of ep{idx}.json written by veritas_pusht_overlays.py.
Models are told apart by line style + marker + a direct label (not colour alone); the
expert pusher path is drawn dashed grey. Episodes are taken from the split in its order.

    python scripts/compare_plan_models.py \
        --split-file diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json \
        --plan flash=media/veritas_pusht/v3_end_outside --plan pro=<dir> --plan er=<dir> \
        --n-episodes 10 --out compare.png
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.veritas_pusht_overlays import (  # noqa: E402
    demo_path, load_replay, split_starts, start_obs)
from diffusion_policy.env.pusht.veritas.pusht_client import (  # noqa: E402
    WORKSPACE_SIZE, obs_to_gemini_inputs)

# Okabe-Ito colours, each paired with its own line style and marker
STYLES = [('#E69F00', '-', 'o'), ('#0072B2', '--', 's'), ('#D55E00', ':', '^'),
          ('#009E73', '-.', 'D')]


def waypoints(plan_dir, ep):
    p = pathlib.Path(plan_dir) / f'ep{ep}.json'
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    if 'plan' not in rec:
        return None
    return np.array([w['uv'] for w in rec['plan']['waypoints']], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--plan', action='append', required=True, help='LABEL=DIR, repeatable')
    ap.add_argument('--n-episodes', type=int, default=10)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plans = [tuple(s.split('=', 1)) for s in args.plan]
    rb = load_replay(args.zarr)
    eps = split_starts(rb, args.split_file, args.split)[:args.n_episodes]
    rows = int(np.ceil(len(eps) / args.cols))
    fig, axes = plt.subplots(rows, args.cols, figsize=(3.2 * args.cols, 3.4 * rows))
    for ax in np.ravel(axes)[len(eps):]:
        ax.axis('off')
    for ax, (ep, state) in zip(np.ravel(axes), eps):
        image, agent_pos, feedback = start_obs(ep, state)
        g = obs_to_gemini_inputs(image, agent_pos, feedback)
        frame = g['frame']
        size = frame.shape[0]
        ax.imshow(frame, extent=(0, size, size, 0))
        demo = demo_path(rb, ep) * (size / WORKSPACE_SIZE)
        ax.plot(demo[:, 0], demo[:, 1], color='0.35', ls=(0, (2, 2)), lw=1.2, label='expert')
        start = np.asarray(g['pusher_uv'], float)
        for (label, d), (col, ls, mk) in zip(plans, STYLES):
            wp = waypoints(d, ep)
            if wp is None:
                ax.text(2, 4 + 6 * plans.index((label, d)), f'{label}: no plan', color=col,
                        fontsize=7, va='top')
                continue
            path = np.vstack([start, wp])
            ax.plot(path[:, 0], path[:, 1], color='white', lw=3.0, ls=ls, alpha=0.8)
            ax.plot(path[:, 0], path[:, 1], color=col, lw=1.6, ls=ls, marker=mk, ms=4,
                    markevery=list(range(1, len(path))), label=f'{label} ({len(wp)} wp)')
            ax.annotate(label, path[-1], color=col, fontsize=7, fontweight='bold',
                        xytext=(3, -3), textcoords='offset points')
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'ep{ep}', fontsize=9)
        ax.legend(fontsize=6, loc='lower right', framealpha=0.85)
    fig.tight_layout()
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
