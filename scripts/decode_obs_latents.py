"""Decode the observation latents back to images -- clean, and at every corruption slot.

    python scripts/decode_obs_latents.py --config-name train_pusht_diffusion_search \
        -o slot_obs_noise.mode=linear_signal
    python scripts/decode_obs_latents.py -c <run>/checkpoints/step_0050000.ckpt

WHY THIS EXISTS. `slot_obs_noise` grades the encoded observation by a DDPM forward marginal
at a per-slot timestep, and the whole argument for doing that in the SD VAE latent space is
that the corrupted latent should still BE a plausible latent -- a blurred, uncertain view of
the same scene, not noise. Nothing in the repo could check that: SDVAEEncoder drops the
decoder at construction (49M parameters that would otherwise sit in the model, its EMA copy
and both Adam moments of every checkpoint), so the latents were write-only. This loads a
stock `AutoencoderKL` for its decoder alone and renders the round trip.

Read the output as a gate before spending GPU time on a ladder arm: if slot 0 decodes to
noise rather than to a blurred T, the ladder's floor is too aggressive and the shape (or,
under `random_base`, the base range) needs moving first.

THE SCALING IS REVERSED IN ORDER, and each step is the exact inverse of a forward one:

    feature[:, :D_rgb]                 the rgb block; _forward concatenates rgb -> depth ->
                                       low_dim, so it leads. D_rgb comes from the backbone's
                                       own output_shape(), never a hardcoded 324.
    .reshape(B, 4, H/8, W/8)           undoes SDVAEEncoder's flatten(start_dim=1)
    / scaling_factor                   undoes  * 0.18215
    post_quant_conv -> decoder         the half SDVAEEncoder drops
    (x + 1) / 2                        undoes normalize_util.get_image_range_normalizer,
                                       which is the ONLY image normalization on this path
                                       (imagenet_norm must stay False)

No checkpoint is required. Without one the policy is built from the config with a pretrained
VAE and an untrained head -- which is enough, because the encoder is FROZEN: the latents an
untrained run produces are the latents a finished one produces.
"""
import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dill
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'diffusion_policy', 'config')


def _build_from_config(config_name, overrides, device):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    policy = hydra.utils.instantiate(cfg.policy)
    # The normalizer is fitted, not learned, so it can be built without training. Everything
    # downstream needs it: _encode_obs_features normalizes before it encodes.
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device).eval()
    return policy, cfg, dataset


def _build_from_checkpoint(checkpoint, device):
    """Same rebuild path as eval_search_pusht.load_policy, EMA copy included."""
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.get('use_ema', False) and getattr(workspace, 'ema_model', None) is not None:
        policy = workspace.ema_model
    policy.to(device).eval()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    return policy, cfg, dataset


def seed_obs_feature_std(policy, dataset, n_windows, device, seed=0):
    """Fill `obs_feature_std` from real encoded observations. Returns the per-dim std.

    THE LADDER SCALES ITS NOISE BY THIS, per dimension:
    `eps = randn_like(x) * self.obs_feature_std`. In training the buffer is an EMA seeded
    from the first batch, but `_update_obs_feature_std` begins `if not self.training:
    return` -- and this script runs the policy in eval(). Without this function the buffer
    keeps its constructor value, `torch.ones(D)`, and every panel is rendered at sigma = 1
    on every dimension instead of the measured latent scale. That is not what training
    applies, and it silently overstates the corruption wherever the true std is below 1.

    Sampled over `n_windows` windows rather than the handful being rendered: a std taken
    over 4 samples is itself noise. Encoded in chunks so a large sample does not have to fit
    in memory at once.
    """
    idxs = np.random.default_rng(seed).choice(
        len(dataset), size=min(n_windows, len(dataset)), replace=False)
    rgb_key = policy.obs_encoder.rgb_keys[0]
    chunks = []
    with torch.no_grad(), policy._crop_scope():
        for start in range(0, len(idxs), 32):
            sel = idxs[start:start + 32]
            obs = {rgb_key: torch.stack(
                [dataset[int(i)]['obs'][rgb_key] for i in sel]).to(device)}
            f = policy._encode_obs_features(obs)          # (b, To, D)
            chunks.append(f.reshape(-1, f.shape[-1]).cpu())
    flat = torch.cat(chunks, dim=0)                     # (n_windows * To, D)
    std = flat.std(dim=0).clamp_min(1e-6)
    if getattr(policy, 'obs_feature_std', None) is not None:
        policy.obs_feature_std.copy_(std.to(policy.obs_feature_std.device))
        policy.obs_feature_std_inited.fill_(True)
    return std, flat


def _rgb_backbone(policy):
    """The one rgb model in the obs encoder, and the width of its slice of the feature."""
    enc = policy.obs_encoder
    assert len(enc.rgb_keys) == 1, \
        f'expected exactly one rgb key, got {enc.rgb_keys}'
    key = enc.rgb_keys[0]
    model = enc.key_model_map['rgb' if enc.share_rgb_model else key]
    assert hasattr(model, 'scaling_factor'), (
        f'{type(model).__name__} is not an SD VAE encoder -- this script inverts '
        f'SDVAEEncoder specifically (its scaling factor and its 4-channel latent). Run it '
        f'on an encoder_tag=vae arm.')
    return model


@torch.no_grad()
def decode(latent_flat, vae, scaling_factor, latent_hw):
    """(B, 4*h*w) scaled latent mean -> (B, 3, H, W) image in [-1, 1].

    The exact inverse of SDVAEEncoder.forward: unflatten, undo the 0.18215 scale, then run
    the half of the VAE that encoder drops. `decoder` is called directly rather than through
    `vae.decode`, which would apply post_quant_conv a second time.
    """
    b = latent_flat.shape[0]
    z = latent_flat.reshape(b, 4, *latent_hw).to(vae.dtype) / scaling_factor
    return vae.decoder(vae.post_quant_conv(z))


def _psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return float('inf') if mse == 0 else 10.0 * float(np.log10(1.0 / mse))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--checkpoint', default=None,
                    help='a step_*.ckpt. Omit to build from --config-name instead; the '
                         'encoder is frozen, so the latents are the same either way.')
    ap.add_argument('--config-name', default='train_pusht_diffusion_search',
                    help='used when --checkpoint is not given')
    ap.add_argument('-o', '--override', action='append', default=[],
                    help='hydra override, repeatable (e.g. -o slot_obs_noise.mode=linear_signal). '
                         'Override the TOP-LEVEL key, not policy.slot_obs_noise -- the policy '
                         'block interpolates it and hydra cannot reach inside an interpolation.')
    ap.add_argument('--split', default='train', choices=('train', 'val', 'test'))
    ap.add_argument('-n', '--n-samples', type=int, default=4, help='rows in the panel')
    ap.add_argument('--slots', default=None,
                    help='comma-separated slot indices; default every slot')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--std-samples', type=int, default=256,
                    help='windows used to seed obs_feature_std when building from a config. '
                         'Ignored with --checkpoint, whose trained buffer is authoritative.')
    ap.add_argument('-d', '--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='media/obs_latents',
                    help='directory for the png and its json sidecar')
    ap.add_argument('--name', default='obs_latents',
                    help='filename stem inside --out, so several ladders can share one '
                         'directory (e.g. --out media/obs_latent_linear_t --name obs_latents40)')
    args = ap.parse_args()

    device = torch.device(args.device)
    if args.checkpoint:
        policy, cfg, dataset = _build_from_checkpoint(args.checkpoint, device)
        source = args.checkpoint
    else:
        policy, cfg, dataset = _build_from_config(args.config_name, args.override, device)
        source = f'{args.config_name} {" ".join(args.override)}'.strip()

    if args.split != 'train':
        dataset = dataset.get_validation_dataset() if args.split == 'val' \
            else dataset._split_copy(dataset.test_pool)

    backbone = _rgb_backbone(policy)
    rgb_key = policy.obs_encoder.rgb_keys[0]

    # The DECODER half, which the policy deliberately does not carry.
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(backbone.model_name if hasattr(backbone, 'model_name')
                                        else 'stabilityai/sd-vae-ft-mse')
    vae.to(device).eval()

    # Seed the per-dimension noise scale BEFORE anything is corrupted. With --checkpoint the
    # trained buffer rode in with the payload and is left alone; from a config it is
    # torch.ones(D) and would mis-scale every panel (see seed_obs_feature_std).
    std_stats = None
    if getattr(policy, 'obs_feature_std', None) is not None:
        if args.checkpoint:
            std = policy.obs_feature_std.detach().cpu()
            src = 'checkpoint'
        else:
            std, _ = seed_obs_feature_std(policy, dataset, args.std_samples, device,
                                          seed=args.seed)
            src = f'measured over {min(args.std_samples, len(dataset))} windows'
        std_stats = {
            'source': src,
            'min': float(std.min()), 'median': float(std.median()),
            'mean': float(std.mean()), 'max': float(std.max()),
        }
        print(f'obs_feature_std ({src}): min {std.min():.4f}  median {std.median():.4f}  '
              f'mean {std.mean():.4f}  max {std.max():.4f}')

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False)
    batch = [dataset[int(i)] for i in idxs]
    obs = {k: torch.stack([b['obs'][k] for b in batch]).to(device)
           for k in batch[0]['obs']}

    K = int(getattr(policy, 'max_actions', 1) or 1)
    ladder_on = (getattr(policy, 'slot_obs_t', None) is not None
                 or getattr(policy, 'slot_obs_shape', None) is not None)
    slots = ([int(x) for x in args.slots.split(',')] if args.slots
             else (list(range(K)) if ladder_on else []))

    # Corruption is gated on `training or corrupt_obs_eval`; the policy is in eval here, so
    # ask for the eval branch explicitly rather than flipping it into train mode (which would
    # also re-enable dropout and start moving the obs_feature_std EMA).
    prev_eval_flag = getattr(policy, 'corrupt_obs_eval', None)
    policy.corrupt_obs_eval = True

    with torch.no_grad(), policy._crop_scope(), policy._corrupt_scope():
        feats = policy._encode_obs_features(obs)          # (B, To, D)
        views = {'clean': feats}
        for k in slots:
            views[f'slot{k}'] = policy.corrupt_obs_features_slotwise(feats, slot=k)
        # What the VAE actually saw: the normalized image put through the encoder's own
        # transform chain (resize / crop / normalize). Rebuilt from the encoder rather than
        # cropped by hand, so the reference cannot drift from the input. The policy is in
        # eval here, so CropRandomizer center-crops -- which is also what a rollout does.
        nobs = policy.normalizer.normalize(obs)
        chain = policy.obs_encoder.key_transform_map[rgb_key]
        seen = chain(nobs[rgb_key][:, feats.shape[1] - 1])          # (B, 3, h, w) in [-1,1]
        # The per-slot levels this panel was rendered at, taken from the policy inside the
        # SAME scope that pinned them. Under random_base they are only defined here: the base
        # is drawn per decision, so there is no buffer to read afterwards. Row 0 of the (B, K)
        # ladder -- with base_range pinned to [N, N] every row is identical.
        slot_levels = (policy._decision_slot_timesteps(feats)[0].tolist()
                       if ladder_on else [])
    policy.corrupt_obs_eval = prev_eval_flag

    # The obs window's LAST step is the one the policy acts on; one frame per row keeps the
    # panel about corruption rather than about time.
    t_idx = feats.shape[1] - 1
    # The rgb block leads the concatenation, so its width is the feature width minus the
    # low_dim keys' -- which is the whole feature under the image-only task, but derived
    # rather than assumed so the script still works on a config that declares low_dim keys.
    enc = policy.obs_encoder
    d_low = sum(int(np.prod(enc.key_shape_map[k])) for k in enc.low_dim_keys)
    d_rgb = int(feats.shape[-1]) - d_low
    side = int(round((d_rgb / 4) ** 0.5))
    assert 4 * side * side == d_rgb, (
        f'rgb feature width {d_rgb} is not 4*h*w for a square latent -- this script assumes '
        f'the SD VAE\'s 4-channel latent and a square crop.')
    cols = ['input']
    images = [((seen + 1.0) / 2.0).clamp(0, 1).cpu()]
    for name, f in views.items():
        img = decode(f[:, t_idx, :d_rgb], vae, backbone.scaling_factor, (side, side))
        images.append(((img + 1.0) / 2.0).clamp(0, 1).cpu())
        cols.append(name)

    # Two references, and they answer different questions. Against the INPUT: how much the
    # VAE round trip itself costs (~29.6 dB when this encoder was adopted) plus the
    # corruption. Against the CLEAN RECONSTRUCTION: the corruption alone, which is the
    # ladder's own signal and the column to read down.
    inp = images[0]
    clean = images[1]
    abars = policy.obs_noise_scheduler.alphas_cumprod
    rows = []
    for j, name in enumerate(cols):
        entry = {'column': name,
                 'psnr_vs_input': _psnr(images[j], inp),
                 'psnr_vs_clean_recon': _psnr(images[j], clean)}
        if name.startswith('slot'):
            k = int(name[4:])
            # `slot_levels` is the ladder ACTUALLY applied to the rendered batch, read back
            # from the policy rather than recomputed here -- under random_base there is no
            # `slot_obs_t` buffer to read, and a panel that cannot say what level it rendered
            # is exactly the failure the obs_feature_std bug was.
            t = int(slot_levels[min(k, len(slot_levels) - 1)])
            entry['timestep'] = t
            entry['sqrt_alpha_bar'] = float(abars[t].sqrt())
        rows.append(entry)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import torchvision
    grid = torchvision.utils.make_grid(
        torch.cat([im for im in images], dim=0), nrow=len(batch), padding=2)
    png = out / f'{args.name}.png'
    torchvision.utils.save_image(grid, png)
    meta = {
        'source': source,
        'task': cfg.task_name,
        'split': args.split,
        'episode_sample_idxs': [int(i) for i in idxs],
        'obs_feature_dim': int(feats.shape[-1]),
        'rgb_latent_dim': d_rgb,
        'latent_shape': [4, side, side],
        'scaling_factor': backbone.scaling_factor,
        # The per-dimension noise scale the ladder actually multiplied eps by. Recorded so a
        # panel can never again be silently rendered at the wrong magnitude.
        'obs_feature_std': std_stats,
        'slot_obs_noise': OmegaConf.to_container(cfg.policy.slot_obs_noise, resolve=True)
                          if 'slot_obs_noise' in cfg.policy else None,
        'obs_noise_scheduler': {
            'num_train_timesteps': int(policy.obs_noise_scheduler.config.num_train_timesteps),
            'beta_start': float(policy.obs_noise_scheduler.config.beta_start),
            'beta_end': float(policy.obs_noise_scheduler.config.beta_end),
        },
        'columns': rows,
        'note': 'grid rows are columns of this list, in order; each row of the image is one '
                'view, each column one sample.',
    }
    js = out / f'{args.name}.json'
    js.write_text(json.dumps(meta, indent=2))
    print(f'wrote {png} and {js}')
    print(f"  {'column':>8}  {'PSNR vs input':>14}  {'vs clean recon':>14}")
    for r in rows:
        extra = ''
        if 'timestep' in r:
            extra = f"   t={r['timestep']}"
            if 'sqrt_alpha_bar' in r:
                extra += f"  sqrt(abar)={r['sqrt_alpha_bar']:.3f}"
        def _f(v):
            return '   inf' if v == float('inf') else f'{v:6.2f}'
        print(f"  {r['column']:>8}  {_f(r['psnr_vs_input']):>14}  "
              f"{_f(r['psnr_vs_clean_recon']):>14}{extra}")


if __name__ == '__main__':
    main()
