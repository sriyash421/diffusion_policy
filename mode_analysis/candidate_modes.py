"""Do the search candidates become MULTIMODAL as the slot index grows, or as obs noise rises?

    python mode_analysis/candidate_modes.py --run-dir <RUN> --steps 10000,50000,100000

WHAT IS PLOTTED. `search_candidates` generates candidate i conditioned on the i previous
ones, so i IS the slot index (`_slot_kwargs(i)`) and n and slot are the same axis. Each
candidate is a (horizon, 2) chunk of agent targets in ARENA PIXELS -- it comes from
`predict_action(...)['action_pred']`, already unnormalized -- so it overlays directly on the
demonstrator's action from the dataset, which is the `a*` reference here.

ONE ROLLOUT GIVES ONE SAMPLE PER SLOT, which cannot show a mode. So each state is decoded
`--repeats` times with DIFFERENT seeds. `set_sample_seeds` makes the diffusion noise a pure
function of (batch row, draw index) and DDIM at eta=0 draws no noise in `step`, so the
initial draw is the only stochastic input: R rows of the SAME observation with R different
seeds are R independent samples, obtainable in ONE batched forward pass rather than R
sequential rollouts.

A* IS ALSO MEASURED, not just drawn. The panels show a* as a reference line, which shows
whether the cloud sits on the expert but cannot be read off a page or compared across arms.
`astar_min_dist_px[slot]` -- the nearest of the R draws -- is the recall number, and
`astar_min_dist_norm` divides it by that slot's own dispersion, because a raw distance is
unreadable without the cloud's scale: a wide cloud lands near a* by luck.

OUTPUT per (run, step), under --outdir:
    modes_step<N>_state<i>.png   left: candidate trajectories; right: endpoint scatter,
                                 both faceted by slot, with a* drawn on each
    dispersion_step<N>.png       endpoint spread vs slot index, averaged over states
    modes.json                   the numbers: endpoint_dispersion_px plus the four
                                 astar_* vectors, each one value per slot
"""
import argparse
import json
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, MUTED = '#1a1a19', '#8a8880'
C_CAND, C_STAR = '#2a78d6', '#eb6834'
ARENA = (5, 506)


def merge_summary(path, summary, params):
    """Fold `summary` into an existing json instead of replacing it.

    These analyses are run as checkpoints LAND, not once at the end, so a run's json is
    written many times with a few new steps each time. Overwriting would silently drop every
    previously computed step, and recomputing them all each round costs far more GPU than the
    new ones.

    MERGING IS ONLY SOUND WHEN THE SETTINGS MATCH. `params` names the knobs that change what
    the numbers MEAN (how many redraws, how many states, how many candidates, whether the obs
    ladder is applied). If the file on disk was produced under different ones, its steps are
    not comparable with these, so it is replaced wholesale and the reason is printed -- a
    quietly mixed file would be worse than a recomputed one.
    """
    merged = dict(summary)
    if path.is_file():
        try:
            prev = json.loads(path.read_text())
        except Exception:
            prev = None
        if prev is not None:
            differing = {k: (prev.get(k), v) for k, v in params.items() if prev.get(k) != v}
            if differing:
                print(f'  {path.name}: settings changed {differing}; replacing rather than '
                      f'merging -- old steps are not comparable with these')
            else:
                steps = dict(prev.get('steps', {}))
                steps.update(summary.get('steps', {}))
                merged['steps'] = dict(sorted(steps.items(), key=lambda kv: int(kv[0])))
    path.write_text(json.dumps(merged, indent=2) + '\n')
    return merged


def load(step, run_dir, device):
    from eval_search_pusht import load_policy
    ckpt = pathlib.Path(run_dir) / 'checkpoints' / f'step_{int(step):07d}.ckpt'
    if not ckpt.is_file():
        return None, None
    return load_policy(str(ckpt), device)


def test_windows(cfg, run_dir, n_states, device):
    """(obs_dict, a_star) for a few held-out TEST windows, from the run's own manifest."""
    import hydra
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    ds_cfg = dict(cfg.task.dataset)
    ds_cfg.pop('_target_', None)
    ds_cfg['split'] = 'test'
    ds_cfg.pop('goal_mask_noise', None)          # a* must be the clean demonstrator action
    ds = PushTImageDataset(**ds_cfg)
    idxs = np.linspace(0, len(ds) - 1, n_states).astype(int)
    out = []
    To = int(cfg.n_obs_steps)
    for i in idxs:
        s = ds[int(i)]
        obs = {k: v[:To].unsqueeze(0).to(device) for k, v in s['obs'].items()}
        out.append((obs, s['action'].numpy()))   # a*: (horizon, 2) in arena pixels
    return out


def sample_candidates(policy, obs, repeats, n_actions, device):
    """(repeats, n_actions, horizon, 2): R independent draws per slot, one forward pass.

    MIRRORS predict_action_best's search block exactly. The two scopes are load-bearing
    here, not decoration:
      _crop_scope    pins ONE crop offset for the observation and every candidate, as in
                     training. Without it each candidate would see a different crop.
      _corrupt_scope pins ONE obs-noise sample across the decision, so the slots are one
                     observation seen at graded levels. Without it every candidate would
                     draw fresh noise and the spread this script measures would be inflated
                     by the corruption, not by the policy.
    The verifier is required because candidate i conditions on the SCORED context of
    candidates 0..i-1; there is no unscored path through the search.
    """
    batched = {k: v.expand(repeats, *v.shape[1:]).contiguous() for k, v in obs.items()}
    if hasattr(policy, 'set_sample_seeds'):
        policy.set_sample_seeds(list(range(repeats)))
    try:
        with torch.no_grad(), policy._crop_scope(), policy._corrupt_scope():
            obs_features = policy._encode_obs_features(batched)
            out = policy.predict_n_actions(
                batched, verifier=policy.verifier, n_actions=n_actions,
                return_scores=True, obs_features=obs_features)
        actions = out[0]
    finally:
        if hasattr(policy, 'set_sample_seeds'):
            policy.set_sample_seeds(None)
    return actions.detach().cpu().numpy()


def frame(ax, bounds=None):
    """Draw the arena, or ZOOM to `bounds` when given.

    An action chunk spans ~30px of a 512px arena, so a full-arena view collapses every
    candidate into one blob and shows nothing about modes. `bounds` is shared across the
    slots of a state, so the slots stay comparable to each other while still being legible.
    """
    if bounds is None:
        ax.plot([ARENA[0], ARENA[1], ARENA[1], ARENA[0], ARENA[0]],
                [ARENA[0], ARENA[0], ARENA[1], ARENA[1], ARENA[0]], color=MUTED, lw=0.8)
        ax.set_xlim(-20, 532); ax.set_ylim(532, -20)
    else:
        x0, x1, y0, y1 = bounds
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)      # y is down, as in the env
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def data_bounds(cand, a_star, pad=0.18):
    """Square bounds covering every candidate and a*, with a margin. Square so the aspect
    is honest -- a stretched box would make an elongated cloud look isotropic."""
    pts = np.concatenate([cand.reshape(-1, 2), a_star.reshape(-1, 2)], axis=0)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, 8.0) * (0.5 + pad)
    return cx - half, cx + half, cy - half, cy + half


def fig_state(cand, a_star, slots, out, title):
    """cand: (R, n, H, 2). Top row trajectories, bottom row endpoint scatter."""
    b = data_bounds(cand[:, slots], a_star)
    span = b[1] - b[0]
    fig, axes = plt.subplots(2, len(slots), figsize=(2.7 * len(slots) + 0.8, 6.4),
                             squeeze=False)
    for c, s in enumerate(slots):
        ax = axes[0][c]
        for r in range(cand.shape[0]):
            ax.plot(cand[r, s, :, 0], cand[r, s, :, 1], color=C_CAND, lw=0.7, alpha=0.45)
        ax.plot(a_star[:, 0], a_star[:, 1], color=C_STAR, lw=2.0, ls=(0, (4, 2)),
                marker='o', ms=2.5, label='a* (demonstrator)')
        frame(ax, b)
        ax.set_title(f'slot {s}', fontsize=10, color=INK)
        if c == 0:
            ax.set_ylabel('trajectories', fontsize=9, color=INK)
        ax2 = axes[1][c]
        ax2.scatter(cand[:, s, -1, 0], cand[:, s, -1, 1], s=14, color=C_CAND,
                    edgecolor=INK, lw=0.3, alpha=0.8)
        ax2.scatter([a_star[-1, 0]], [a_star[-1, 1]], s=70, marker='*', color=C_STAR,
                    edgecolor=INK, lw=0.5, zorder=5)
        frame(ax2, b)
        if c == 0:
            ax2.set_ylabel('chunk endpoints', fontsize=9, color=INK)
    axes[0][0].legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left',
                      bbox_to_anchor=(0, -0.02))
    fig.suptitle(title + f'   —   view is {span:.0f}px across '
                 f'(arena is 512px); all slots share one zoom',
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=140, facecolor='white')
    plt.close(fig)


def dispersion(cand):
    """Mean pairwise distance between the R endpoints, per slot. (n,)

    A spread statistic, NOT a mode count: it rises both when a unimodal cloud widens and
    when it splits. The scatter panels are what show which. Reported because it is the one
    number that is comparable across arms without fitting anything.
    """
    R, n = cand.shape[0], cand.shape[1]
    out = np.zeros(n)
    for s in range(n):
        p = cand[:, s, -1, :]
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        out[s] = d[np.triu_indices(R, 1)].mean() if R > 1 else 0.0
    return out


def astar_recall(cand, a_star, disp):
    """Does any of the R draws at slot s actually land on a*?  cand (R,n,H,2), a* (H,2).

    `disp` is that slot's own endpoint dispersion, and dividing by it is the point: a raw
    px distance is unreadable without the cloud's scale, because a WIDE cloud lands near a*
    by luck while a tight one that happens to sit on a* is the real thing. A ratio below 1
    means the draws reach a* more closely than they reach each other.

    The chunk distance is kept alongside the endpoint distance so a candidate that arrives
    at the same place along a different path is not scored as a hit.
    """
    R, n = cand.shape[0], cand.shape[1]
    end = cand[:, :, -1, :]                                   # (R, n, 2)
    d_end = np.linalg.norm(end - a_star[-1], axis=-1)         # (R, n)
    d_chunk = np.linalg.norm(cand - a_star[None, None], axis=-1).mean(axis=-1)
    safe = np.where(disp > 1e-9, disp, np.nan)
    return {'astar_min_dist_px': d_end.min(axis=0),           # (n,) THE recall number
            'astar_mean_dist_px': d_end.mean(axis=0),
            'astar_min_chunk_dist_px': d_chunk.min(axis=0),
            'astar_min_dist_norm': d_end.min(axis=0) / safe}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--steps', default='10000,50000,100000')
    ap.add_argument('--repeats', type=int, default=32)
    ap.add_argument('--n-actions', type=int, default=None)
    ap.add_argument('--n-states', type=int, default=3)
    ap.add_argument('--slots', default='0,1,3,7,15')
    ap.add_argument('--corrupt-obs-eval', dest='corrupt', action='store_true', default=None,
                    help='force the training obs corruption ON at sampling time')
    ap.add_argument('--no-corrupt-obs-eval', dest='corrupt', action='store_false',
                    help='force it OFF; default leaves the checkpoint own setting')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out = pathlib.Path(args.outdir or f'/gscratch/robotics/harine/mode_analysis/{run_dir.name}')
    out.mkdir(parents=True, exist_ok=True)
    summary = {'run': run_dir.name, 'repeats': args.repeats,
               'n_states': args.n_states, 'n_actions': args.n_actions,
               'corrupt': args.corrupt, 'steps': {}}

    for step in [int(s) for s in args.steps.split(',') if s.strip()]:
        policy, cfg = load(step, run_dir, args.device)
        if policy is None:
            print(f'step {step}: no checkpoint, skipping'); continue
        if args.corrupt is not None and hasattr(policy, 'corrupt_obs_eval'):
            policy.corrupt_obs_eval = bool(args.corrupt)
        K = int(getattr(policy, 'max_actions', 1) or 1)
        n_act = args.n_actions or K
        slots = [s for s in (int(x) for x in args.slots.split(',')) if s < n_act]
        states = test_windows(cfg, str(run_dir), args.n_states, args.device)
        disp, rec = [], []
        for i, (obs, a_star) in enumerate(states):
            cand = sample_candidates(policy, obs, args.repeats, n_act, args.device)
            fig_state(cand, a_star, slots, out / f'modes_step{step}_state{i}.png',
                      f'{run_dir.name}\nstep {step//1000}k, state {i}, '
                      f'{args.repeats} seeds per slot')
            ds = dispersion(cand)
            disp.append(ds)
            rec.append(astar_recall(cand, a_star, ds))
        d = np.mean(disp, axis=0)
        entry = {'n_actions': n_act, 'endpoint_dispersion_px': [float(v) for v in d]}
        for k in rec[0]:
            entry[k] = [float(v) for v in np.nanmean([r[k] for r in rec], axis=0)]
        summary['steps'][str(step)] = entry
        print(f'step {step}: n={n_act}, dispersion slot0={d[0]:.1f}px '
              f'slot{n_act-1}={d[-1]:.1f}px')
        a = entry['astar_min_dist_px']
        print(f'    nearest draw to a*: slot0={a[0]:.1f}px slot{n_act-1}={a[-1]:.1f}px '
              f'(as a fraction of the cloud spread: {entry["astar_min_dist_norm"][0]:.2f} -> '
              f'{entry["astar_min_dist_norm"][-1]:.2f})')
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(np.arange(len(d)), d, color=C_CAND, marker='o', ms=4, lw=1.6)
        ax.set_xlabel('candidate slot (= n index)', color=INK)
        ax.set_ylabel('mean pairwise endpoint distance (px)', color=INK)
        ax.set_title(f'{run_dir.name} — step {step//1000}k, {args.repeats} seeds/slot',
                     fontsize=10, color=INK, loc='left')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        fig.savefig(out / f'dispersion_step{step}.png', dpi=150, facecolor='white',
                    bbox_inches='tight')
        plt.close(fig)
        del policy
        torch.cuda.empty_cache()

    merge_summary(out / 'modes.json', summary,
                  {k: summary[k] for k in ('repeats', 'n_states', 'n_actions', 'corrupt')})
    print(f'wrote {out}/')


if __name__ == '__main__':
    main()
