"""Visualise the search transformer's cross-attention: does the policy use its search context?

    python attention_analysis/visualize_attention.py --run-dir <RUN> --steps 10000,50000,100000
    python attention_analysis/visualize_attention.py --run-dir <RUN> --steps 100000 --mode episode

WHAT IS BEING MEASURED. SearchTransformerForDiffusion decodes the horizon trajectory against
a memory of [obs tokens | action_value tokens]; the second block IS the search context (the
previous candidates and their verifier values). So the attention mass a query puts on that
block is, literally, how much this policy is using its context.

    ST k=16 -> 2 obs + 15 context tokens.   ST k=1 -> 2 obs + 0 context (nothing to compare).
    UNet BC -> no attention at all; the script refuses it with an explanation.

TWO CONDITIONING MODES, because attention depends on both the input and the denoising step:
  --mode batch    (default) average over --n-episodes real held-out TEST observations, and
                  report the first / middle / last DDIM step. Representative.
  --mode episode  one episode, EVERY DDIM step. Sharp on timestep dependence, one sample.

The test observations come from the checkpoint's OWN split manifest, so an arm is never
analysed on episodes it trained on.

OUTPUT (under attention_analysis/<run_name>/):
    matrices_step<N>.png   layer x head grid of the (H x S) map, one figure per step
    slots_step<N>.png      context-vs-obs mass per candidate slot
    context_vs_step.png    the slot summary across every --steps value, one panel per step
    attention.json         the numbers behind the figures
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
sys.path.insert(0, str(ROOT / 'attention_analysis'))
from attention_hooks import capture_cross_attention, context_mass, split_slots
sys.path.insert(0, str(ROOT / 'mode_analysis'))
from candidate_modes import merge_summary  # one definition, shared

INK, MUTED = '#1a1a19', '#8a8880'
C_OBS, C_CTX = '#2a78d6', '#eb6834'      # paired with hatch/marker below, never hue alone


def load(step, run_dir, device):
    from eval_search_pusht import load_policy
    ckpt = pathlib.Path(run_dir) / 'checkpoints' / f'step_{int(step):07d}.ckpt'
    if not ckpt.is_file():
        raise FileNotFoundError(f'no checkpoint at {ckpt}')
    return (*load_policy(str(ckpt), device), ckpt)


def test_window_observations(cfg, n, device):
    """A batch of held-out TEST WINDOWS, straight from the dataset the run was trained with.

    WHY THIS EXISTS. `episode_init_observations` below probes at each test episode's t=0,
    where the T has not been touched -- so every probe state is one where the demonstrator's
    block does not move. For an arm trained only on the MOVING transitions (transition_file),
    that is precisely the distribution it never saw, and "does it attend to its context" would
    be answered off-distribution. Reading windows out of PushTImageDataset(split='test')
    instead gives the held-out transitions themselves, and matches what
    mode_analysis/candidate_modes.test_windows() already probes, so the two analyses of one
    arm describe the same states.

    Costs the 2.8 GB image array, which episode_init_observations deliberately avoids -- fine
    inside run_attention.sbatch, not on a login node.
    """
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    # Built exactly as mode_analysis/candidate_modes.test_windows builds it -- split on the
    # constructor, goal mask dropped -- so the two analyses probe the same windows of the
    # same split. `transition_file` rides along in cfg, which is what makes these the
    # held-out MOVING transitions on a filtered arm and every test window otherwise.
    ds_cfg = dict(cfg.task.dataset)
    ds_cfg.pop('_target_', None)
    ds_cfg['split'] = 'test'
    ds_cfg.pop('goal_mask_noise', None)      # probe a clean observation, as eval always does
    ds = PushTImageDataset(**ds_cfg)
    if len(ds) == 0:
        raise RuntimeError('the test split has no windows; nothing to probe')
    To = int(cfg.n_obs_steps)
    # Spread the probes over the whole split rather than taking a prefix: consecutive windows
    # overlap by To-1 frames, so the first n would be one moment of one episode.
    idxs = np.linspace(0, len(ds) - 1, num=min(n, len(ds))).round().astype(int)
    keys = ('image', 'agent_pos', 'feedback')
    batch = {k: torch.stack([ds[int(i)]['obs'][k][:To] for i in idxs]).float().to(device)
             for k in keys}
    return batch


def episode_init_observations(cfg, run_dir, n, device):
    """A batch of held-out TEST episode INITIAL states, shaped for the policy's obs dict.

    Every attention.json written before 2026-09-18 used this, so it is what an existing arm
    must keep being probed at for its numbers to stay comparable. See test_window_observations
    for why a filtered arm should not use it.
    """
    from eval_search_pusht import get_split_states
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose
    states, _ = get_split_states(cfg, 'test', run_dir=run_dir)
    states = np.asarray(states)[:n]
    env = PushTImageEnv(legacy=cfg.task.env_runner.get('legacy', False),
                        render_size=96)
    imgs, agents = [], []
    for s in states:
        env.reset_to_state = np.asarray(s)
        obs = env.reset()
        imgs.append(obs['image']); agents.append(obs['agent_pos'])
    To = int(cfg.n_obs_steps)
    img = torch.from_numpy(np.asarray(imgs)).float()             # (N, 3, 96, 96)
    img = img.unsqueeze(1).repeat(1, To, 1, 1, 1)                # (N, To, 3, 96, 96)
    agent = torch.from_numpy(np.asarray(agents)).float().unsqueeze(1).repeat(1, To, 1)
    # `feedback` is REQUIRED even though the encoder never reads it: the search path scores
    # every candidate with PushTVerifier, and _reset_states_from_obs seeds its pymunk sim
    # from obs_dict['feedback'] (KeyError without it). It is an exact invertible transform
    # of the block pose, which is states[:, 2:5] of the reset state -- so this is the same
    # tensor PushTImageDataset._sample_to_data would emit, without loading the 2.8GB zarr.
    fb = torch.from_numpy(
        compute_feedback_from_pose(np.asarray(states)[:, 2:5].astype(np.float32))
    ).float().unsqueeze(1).repeat(1, To, 1)                      # (N, To, 16)
    return {'image': img.to(device), 'agent_pos': agent.to(device),
            'feedback': fb.to(device)}


def run_capture(policy, obs_dict, n_actions):
    """One SEARCH readout with the hooks armed, plus the capture's slot/denoise layout.

    `predict_action` is the single non-searched chunk -- it builds NO context, so it would
    show nothing. `predict_action_best` runs `search_candidates`, whose loop is
    `for i in range(n_actions): ... **self._slot_kwargs(i)`: candidate i is decoded AT SLOT i,
    each with its own full DDIM loop. So a layer's capture list is blocks of
    `num_inference_steps` entries, one block per slot, in slot order.

    The block count is DERIVED, not assumed: under `selection: final_pass`
    `predict_action_best` draws one extra sample after the search, which appends a trailing
    block that is not a search slot.
    """
    with capture_cross_attention(policy) as cap:
        with torch.no_grad():
            policy.predict_action_best(obs_dict, n_actions=n_actions)
    n_d = int(getattr(policy, 'num_inference_steps', 0) or 0)
    total = len(cap[0]) if cap else 0
    if n_d <= 0 or total % n_d != 0:
        raise RuntimeError(
            f'captured {total} forward passes but num_inference_steps={n_d}; cannot split '
            f'the capture into per-slot blocks. Inspect search_candidates before trusting '
            f'any slot axis.')
    return cap, n_d, total // n_d


def fig_matrices(attn, n_obs, out, title):
    """layer x head grid of the (H queries x S memory) map for one denoising step."""
    L, Hh = attn.shape[0], attn.shape[1]
    fig, axes = plt.subplots(L, Hh, figsize=(2.5 * Hh + 1.2, 2.5 * L + 1.0), squeeze=False)
    vmax = float(attn.max())
    for l in range(L):
        for h in range(Hh):
            ax = axes[l][h]
            im = ax.imshow(attn[l, h], cmap='magma', vmin=0, vmax=vmax, aspect='auto')
            # the obs|context boundary, so the two blocks are never read as one
            ax.axvline(n_obs - 0.5, color='#ffffff', lw=1.6)
            ax.axvline(n_obs - 0.5, color=INK, lw=0.7, ls=(0, (3, 2)))
            if l == 0:
                ax.set_title(f'head {h}', fontsize=9, color=INK)
            if h == 0:
                ax.set_ylabel(f'layer {l}\nquery (horizon)', fontsize=8, color=INK)
            if l == L - 1:
                ax.set_xlabel('memory token', fontsize=8, color=INK)
            ax.tick_params(labelsize=6, colors=MUTED)
    fig.suptitle(title + '\nleft of the dashed line = obs tokens, right = search context',
                 fontsize=12, color=INK)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label='attention weight')
    fig.savefig(out, dpi=140, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def fig_slots(per_slot, out, title):
    """Context-vs-obs mass per candidate slot. per_slot: (K,) context fraction in [0,1]."""
    K = len(per_slot)
    x = np.arange(K)
    fig, ax = plt.subplots(figsize=(max(6.5, 0.45 * K + 3), 4.2))
    ax.bar(x, per_slot, color=C_CTX, edgecolor=INK, lw=0.5, label='search context')
    ax.bar(x, 1 - per_slot, bottom=per_slot, color=C_OBS, edgecolor=INK, lw=0.5,
           hatch='//', label='observation')
    ax.set_xlabel('candidate slot (0 = no context, K-1 = full context)', color=INK)
    ax.set_ylabel('share of cross-attention mass', color=INK)
    ax.set_xticks(x); ax.set_ylim(0, 1); ax.set_xlim(-0.6, K - 0.4)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_title(title, fontsize=11, color=INK, loc='left')
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, ncol=2, loc='upper left',
              bbox_to_anchor=(0, -0.16))
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True, help='training output dir (has checkpoints/)')
    ap.add_argument('--steps', default='100000', help='comma-separated checkpoint steps')
    ap.add_argument('--mode', choices=('batch', 'episode'), default='batch')
    ap.add_argument('--n-episodes', type=int, default=16, help='--mode batch only')
    ap.add_argument('--n-actions', type=int, default=None,
                    help='search width for the capture; defaults to the policy max_actions')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--outdir', default=None)
    # WHERE the attention is probed. `auto` reads the run's own config: an arm trained on a
    # filtered set of transitions is probed on the held-out transitions, and everything else
    # keeps the episode-t=0 states every attention.json before 2026-09-18 was measured at.
    # Recorded in the json, because the two probe sets are not comparable with each other.
    ap.add_argument('--states', choices=('auto', 'episode-init', 'test-windows'),
                    default='auto')
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    out = pathlib.Path(args.outdir or (ROOT / 'attention_analysis' / run_dir.name))
    out.mkdir(parents=True, exist_ok=True)
    steps = [int(s) for s in args.steps.split(',') if s.strip()]
    summary = {'run': run_dir.name, 'mode': args.mode, 'states': args.states,
               'n_episodes': args.n_episodes, 'n_actions': args.n_actions, 'steps': {}}

    across = []
    # The probe states are a property of the RUN, not of the checkpoint, so they are built
    # once and reused across steps. Worth being explicit about: test_window_observations
    # loads the 2.8 GB image array, and rebuilding it per step would pay that ten times over
    # a full 10k..100k grid for a byte-identical batch.
    obs_cache = {}
    for step in steps:
        policy, cfg, ckpt = load(step, run_dir, args.device)
        K = int(getattr(policy, 'max_actions', 1) or 1)
        n_obs = int(cfg.n_obs_steps)
        mca = int(getattr(policy.model, 'max_context_actions', 0))
        if mca == 0:
            print(f'step {step}: max_context_actions == 0 (ST k=1) -- memory is obs only, '
                  f'so there is no context to attend to. Matrices still written.')
        n_act = args.n_actions or K
        n_ep = 1 if args.mode == 'episode' else args.n_episodes
        states = args.states
        if states == 'auto':
            states = ('test-windows'
                      if cfg.task.dataset.get('transition_file', None) else 'episode-init')
        summary['states'] = states
        if (states, n_ep) not in obs_cache:
            obs_cache[(states, n_ep)] = (
                test_window_observations(cfg, n_ep, args.device) if states == 'test-windows'
                else episode_init_observations(cfg, str(run_dir), n_ep, args.device))
            print(f'built {n_ep} {states} probe observations')
        obs = obs_cache[(states, n_ep)]

        cap, n_dstep, n_blocks = run_capture(policy, obs, n_act)
        n_layers = len(cap)
        n_slots = min(n_blocks, n_act)     # a trailing final_pass block is not a slot
        print(f'step {step}: captured {n_layers} layers x {n_blocks} blocks x {n_dstep} '
              f'denoising steps; slots={n_slots}, memory = {n_obs} obs + {mca} context, K={K}')
        if n_blocks > n_act:
            print(f'  note: {n_blocks - n_act} trailing block(s) beyond n_actions -- the '
                  f'final_pass extra sample; excluded from the slot axis.')

        # which denoising steps to draw: all of them in episode mode, else first/mid/last
        picks = range(n_dstep) if args.mode == 'episode' else \
            sorted({0, n_dstep // 2, n_dstep - 1})
        step_rec = {'n_obs_tokens': n_obs, 'max_context_actions': mca, 'K': K,
                    'n_denoise_steps': n_dstep, 'n_slots': n_slots, 'denoise': {}}

        def block(layer, slot, di):
            """capture index for (slot, denoise step): blocks of n_dstep, in slot order."""
            return cap[layer][slot * n_dstep + di]

        for di in picks:
            # average over the episode batch; keep layer and head. Slot K-1 has the most
            # context, so the matrices are drawn there -- slot 0 has none by construction.
            top = n_slots - 1
            a = torch.stack([block(l, top, di) for l in range(n_layers)]).mean(1).numpy()
            tag = f'step{step}_d{di}'
            fig_matrices(a, n_obs, out / f'matrices_{tag}.png',
                         f'{run_dir.name}\nstep {step//1000}k, denoise {di}/{n_dstep-1}, '
                         f'slot {top} (most context), {args.mode} mode')
            rec = {'slot_shown': top,
                   'context_mass_mean': float(context_mass(torch.from_numpy(a), n_obs).mean())}
            if mca > 0:
                per_slot = np.array([
                    float(context_mass(torch.stack([block(l, s_, di) for l in range(n_layers)]),
                                       n_obs).mean())
                    for s_ in range(n_slots)])
                fig_slots(per_slot, out / f'slots_{tag}.png',
                          f'{run_dir.name} — step {step//1000}k, denoise {di}')
                rec['context_mass_per_slot'] = [float(v) for v in per_slot]
                across.append((step, di, per_slot))
            step_rec['denoise'][str(di)] = rec
        summary['steps'][str(step)] = step_rec
        del policy
        torch.cuda.empty_cache()

    if across:
        # One ROW per denoising step, one COLUMN per training step: the two axes are
        # different questions ("does context use develop with training?" vs "does it change
        # through denoising?") and a single strip of 9 panels conflates them.
        steps_seen = sorted({s for s, _, _ in across})
        ds_seen = sorted({d for _, d, _ in across})
        fig, axes = plt.subplots(len(ds_seen), len(steps_seen),
                                 figsize=(3.3 * len(steps_seen) + 1.0, 2.6 * len(ds_seen) + 1.2),
                                 squeeze=False, sharey=True, sharex=True)
        by = {(s, d): ps for s, d, ps in across}
        for r, di in enumerate(ds_seen):
            for c, st in enumerate(steps_seen):
                ax = axes[r][c]
                ps = by.get((st, di))
                if ps is None:
                    ax.set_visible(False); continue
                ax.bar(np.arange(len(ps)), ps, color=C_CTX, edgecolor=INK, lw=0.4)
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)))
                if r == 0:
                    ax.set_title(f'{st//1000}k steps', fontsize=10, color=INK)
                if c == 0:
                    ax.set_ylabel(f'denoise {di}\ncontext share', fontsize=9, color=INK)
                if r == len(ds_seen) - 1:
                    ax.set_xlabel('candidate slot', fontsize=9, color=INK)
                for sp in ('top', 'right'):
                    ax.spines[sp].set_visible(False)
        fig.suptitle(f'{run_dir.name}\ncontext share of cross-attention '
                     '(slot 0 has no context by construction; dashed line = parity with obs)',
                     color=INK, fontsize=11, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / 'context_vs_step.png', dpi=150, facecolor='white')
        plt.close(fig)

    # `states` is in the guard: the episode-t=0 and held-out-transition probes are not
    # comparable, so a file must never hold both.
    merge_summary(out / 'attention.json', summary,
                  {k: summary[k] for k in ('mode', 'states', 'n_episodes', 'n_actions')})
    print(f'\nwrote {out}/')


if __name__ == '__main__':
    main()
