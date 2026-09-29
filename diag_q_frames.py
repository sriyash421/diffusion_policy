"""Where t_goal is blind, what does Q actually prefer?

On no-contact decisions every candidate leaves the T where it is, so every t_goal score is
bit-identical and argmax falls through to numpy's first-maximizer tie-break -- the heuristic is
not choosing at all. Those decisions are the whole reason for a learned verifier, and an
aggregate p_best cannot show whether Q's preference is sensible there. This draws them: the
decision-state frame, every candidate chunk on top of it, coloured and labelled by its Q.

Read it for whether Q prefers candidates that move the arm TOWARD the T. That is the behaviour a
useful pre-contact verifier must have, and it is visible by eye long before it is significant in
a success rate.

  python diag_q_frames.py --ckpt <ST.ckpt> --q <sac.zip> --out frames.png
"""
import argparse, pathlib
import numpy as np
import torch

from diffusion_policy.common.slot_stats_util import BLIND_EPS
from sac.eval import load_policy
from eval_search_pusht import _eval_split_at_n, build_envs, get_split_states

WS = 512.0   # PushT workspace, and the units candidate targets are in


def draw(panels, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis('off')

    for i, pan in enumerate(panels):
        ax = axes[i // cols][i % cols]
        img, agent, cand, q = pan['image'], pan['agent'], pan['cand'], pan['q']
        H = img.shape[0]
        s = H / WS                                   # workspace px -> image px
        ax.imshow(img, interpolation='nearest')
        ax.axis('off')

        # viridis is perceptually uniform and colourblind-safe, and the best and worst
        # candidates additionally carry a linestyle, a width and a printed value -- colour is
        # never the only channel (repo rule).
        lo, hi = float(q.min()), float(q.max())
        rng = max(hi - lo, 1e-12)
        order = np.argsort(q)
        cmap = matplotlib.cm.get_cmap('viridis')
        for j in order:
            xy = np.concatenate([agent[None, :], cand[j]], axis=0) * s
            best, worst = (j == order[-1]), (j == order[0])
            ax.plot(xy[:, 0], xy[:, 1],
                    color=cmap((q[j] - lo) / rng),
                    lw=2.4 if best else (1.8 if worst else 0.9),
                    ls='-' if best else ('--' if worst else ':'),
                    alpha=1.0 if (best or worst) else 0.65, zorder=3 if best else 2)
            if best or worst:
                ax.annotate(f'{q[j]:+.3f}', xy[-1][:2], fontsize=7,
                            color='white', zorder=4,
                            bbox=dict(fc=cmap((q[j] - lo) / rng), ec='none',
                                      pad=1.0, alpha=0.9))
        ax.plot(agent[0] * s, agent[1] * s, 'o', ms=5, mfc='none', mec='white', mew=1.6, zorder=5)
        ax.set_title(f"ep {pan['episode']} step {pan['step']}\n"
                     f"Q spread {q.std():.4f}  (solid=argmax, dashed=min)", fontsize=7)

    fig.suptitle(f'No-contact decisions: candidate chunks coloured by Q\n{title}', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--q', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=10)
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--split', default='test')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    policy, cfg = load_policy(a.ckpt, a.device)
    states, idxs = get_split_states(cfg, a.split, pathlib.Path(a.ckpt).resolve().parent.parent)
    states, idxs = states[:a.episodes], idxs[:a.episodes]
    n_envs = min(a.episodes, 8)
    env = build_envs(n_envs, policy.n_obs_steps, policy.n_action_steps, 300)

    from sac.score import PushTQVerifier
    q_scorer = PushTQVerifier(a.q, device=a.device)

    To, Ta = policy.n_obs_steps, policy.n_action_steps
    keys = ('agent_pos', 'feedback', 'image')
    original = policy._score_candidates
    panels, seen, buf = [], [0], []

    # ONE CALL PER CANDIDATE, batched over ENVIRONMENTS -- search_procedure loops
    # `for i in range(n_actions)`. A single call therefore holds candidate i at B different
    # states, not the candidate set at one state. Buffer n calls and transpose.
    def patched(verifier, obs_dict, action, want_subgoals=False):
        context, value, subgoal, terms = original(verifier, obs_dict, action, want_subgoals)
        now = {k: val[:, To - 1:To] for k, val in obs_dict.items() if k in keys}
        chunk = action[:, To - 1:To - 1 + Ta]
        qv = q_scorer.get_value(now, chunk)
        img = obs_dict['image'][:, To - 1] if 'image' in obs_dict else None
        buf.append((value.detach().cpu().numpy().ravel(),
                    qv.detach().cpu().numpy().ravel(),
                    chunk.detach().cpu().numpy().astype(float),
                    now['agent_pos'][:, 0].detach().cpu().numpy().astype(float),
                    None if img is None else img.detach().cpu().numpy()))
        if len(buf) == a.n:
            sim = np.stack([b[0] for b in buf], axis=1)          # (B, n)
            qs = np.stack([b[1] for b in buf], axis=1)           # (B, n)
            seen[0] += sim.shape[0]
            for r in range(sim.shape[0]):
                if len(panels) >= a.frames:
                    break
                # NO CONTACT = t_goal could not tell these candidates apart, at the same
                # threshold the repo uses to call a decision blind.
                if sim[r].std() > BLIND_EPS or buf[0][4] is None:
                    continue
                im = buf[0][4][r]
                im = np.transpose(im, (1, 2, 0)) if im.shape[0] in (1, 3) else im
                im = np.clip(im if im.max() > 1.5 else im * 255, 0, 255).astype(np.uint8)
                panels.append({
                    'image': im,
                    'agent': buf[0][3][r],
                    'cand': np.stack([b[2][r] for b in buf], axis=0),   # (n, Ta, 2)
                    'q': qs[r],
                    'episode': r, 'step': seen[0]})
            buf.clear()
        return context, value, subgoal, terms

    policy._score_candidates = patched
    try:
        _eval_split_at_n(env, policy, states, a.device, a.n, n_envs, a.seed, 'q-frames')
    finally:
        policy._score_candidates = original
        env.close()

    print(f'[INFO] {seen[0]} decisions seen, {len(panels)} no-contact frames captured')
    if buf:
        print(f'[WARN] {len(buf)} unpaired calls at exit -- search width may not be {a.n}')
    if not panels:
        raise SystemExit('no no-contact decisions found -- nothing to draw')
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    draw(panels, a.out, f'{pathlib.Path(a.ckpt).parent.parent.name}\nQ: {a.q}')


if __name__ == '__main__':
    main()
