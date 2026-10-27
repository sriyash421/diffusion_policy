"""Paired denoising measurements and live/EMA rollouts; no weight updates."""
import argparse
import gc
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import dill
import hydra
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from scripts.eval_pusht_capacity import evaluate
from scripts.make_blq137_holdout import SOURCE, validate_holdout


def predict_fixed_noise(policy, batch, noise, times):
    """Native k=1 denoising computation with externally fixed noise/timesteps."""
    target = policy.normalizer['action'].normalize(batch['action'])
    is_unet = isinstance(policy.model, ConditionalUnet1D)
    if not is_unet:
        features = policy._encode_obs_features(batch['obs'])
    else:
        obs = policy.normalizer.normalize(policy._select_obs(batch['obs']))
        obs = dict_apply(obs, lambda x: x[:, :policy.n_obs_steps].reshape(-1, *x.shape[2:]))
        features = policy._encode_images(obs, len(target)).reshape(len(target), policy.n_obs_steps, -1)
    noisy = policy.noise_scheduler.add_noise(target, noise, times)
    if not is_unet:
        return policy.model(noisy, times, obs_cond=features)
    return policy.model(noisy, times, global_cond=features.flatten(1))


def paired_losses(checkpoint, device, windows_per_episode=4):
    payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill)
    cfg = payload['cfg']
    assert cfg.policy.noise_scheduler.prediction_type == 'epsilon'
    assert cfg.policy.get('max_actions', 1) == 1
    ds = hydra.utils.instantiate(cfg.task.dataset)
    manifest = json.loads(Path(cfg.task.dataset.split_file).read_text())
    validate_holdout(manifest, json.loads(SOURCE.read_text()))
    policy = hydra.utils.instantiate(cfg.policy).to(device).eval()
    result = dict(checkpoint=str(checkpoint), crop='center', dropout=False,
        windows_per_episode=windows_per_episode, timesteps=[0, 10, 30, 60, 90],
        holdout_checksum=manifest['holdout_checksum'], weights={})
    try:
        # Replay membership and the quantiles within each episode do not depend on model.
        for weights, state_key in [('live', 'model'), ('ema', 'ema_model')]:
            policy.load_state_dict(payload['state_dicts'][state_key])
            # LinearNormalizer materializes parameters during state loading.
            policy.to(device)
            result['weights'][weights] = {}
            for split in ('train', 'val', 'id_test', 'test'):
                mask = np.zeros(ds.replay_buffer.n_episodes, dtype=bool)
                mask[manifest[split]] = True
                # No transition filter is enabled; this builds an evaluation-only view.
                assert cfg.task.dataset.get('transition_file') is None
                subset = ds._split_copy(mask, 'test')
                episode_ids = np.searchsorted(ds.replay_buffer.episode_ends[:],
                    subset.sampler.indices[:, 0], side='right')
                selected = []
                window_ids = []
                for ep in manifest[split]:
                    candidates = np.flatnonzero(episode_ids == ep)
                    assert len(candidates) >= windows_per_episode
                    picks = candidates[np.linspace(0, len(candidates)-1, windows_per_episode).astype(int)]
                    selected.extend(map(int, picks))
                    window_ids.extend([[int(ep), *map(int, subset.sampler.indices[i])] for i in picks])
                metrics = {str(t): [] for t in result['timesteps']}
                gradient_norms = []
                for offset in range(0, len(selected), 16):
                    batch = default_collate([subset[i] for i in selected[offset:offset+16]])
                    batch = dict_apply(batch, lambda x: x.to(device))
                    target = policy.normalizer['action'].normalize(batch['action'])
                    # Generate on CPU so the exact paired noise is independent of GPU type.
                    generator = torch.Generator().manual_seed(42000 + offset)
                    noise = torch.randn(target.shape, generator=generator).to(device)
                    for timestep in result['timesteps']:
                        policy.zero_grad(set_to_none=True)
                        # Gradient diagnostic on one fixed batch per split, t=30 only.
                        grad = offset == 0 and timestep == 30
                        with torch.set_grad_enabled(grad):
                            times = torch.full((len(target),), timestep, dtype=torch.long, device=device)
                            pred = predict_fixed_noise(policy, batch, noise, times)
                            error = (pred-noise).square()
                            metrics[str(timestep)].extend(error.detach().mean(dim=2).cpu().tolist())
                            if grad:
                                error.mean().backward()
                                norms = {}
                                for name, module in [('encoder', policy.obs_encoder), ('denoiser', policy.model)]:
                                    norms[name] = float(torch.sqrt(sum(p.grad.detach().square().sum()
                                        for p in module.parameters() if p.grad is not None)))
                                gradient_norms.append(dict(timestep=timestep, **norms))
                result['weights'][weights][split] = dict(windows=window_ids,
                    epsilon_mse_by_timestep_and_horizon={t: np.mean(v, axis=0).tolist() for t, v in metrics.items()},
                    epsilon_mse_by_timestep={t: float(np.mean(v)) for t, v in metrics.items()},
                    gradient_norms_eval_mode_first_batch=gradient_norms)
    finally:
        policy.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--windows-per-episode', type=int, default=4)
    args = parser.parse_args()
    summary = json.loads((args.run_dir/'capacity_eval_id_ood/summary.json').read_text())
    steps = sorted({summary['selected']['step'], 100000})
    out = args.run_dir/'capacity_diagnostics'
    out.mkdir(exist_ok=True)
    for step in steps:
        checkpoint = args.run_dir/'checkpoints'/f'step_{step:07d}.ckpt'
        dest = out/f'loss_step_{step:07d}.json'
        if not dest.exists():
            result = paired_losses(checkpoint, args.device, args.windows_per_episode)
            temp = dest.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2)); temp.replace(dest)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if not args.loss_only:
            for weights in ('live', 'ema'):
                dest = out/f'rollout_{weights}_step_{step:07d}.json'
                if not dest.exists():
                    result = evaluate(checkpoint, args.run_dir, args.device, 50, weights=weights, include_train=True)
                    temp = dest.with_suffix('.tmp'); temp.write_text(json.dumps(result, indent=2)); temp.replace(dest)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f'Diagnostics complete: {out}', flush=True)


if __name__ == '__main__':
    main()
