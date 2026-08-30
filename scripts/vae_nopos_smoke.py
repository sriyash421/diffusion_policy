"""Preflight for the VAE_no_pos generation. Cheap, CPU-only, no training.

    python scripts/vae_nopos_smoke.py

Checks the four invariants the whole generation rests on, for EVERY arm, because each one
fails silently if it breaks:

  1. IMAGE-ONLY OBSERVATION. shape_meta.obs is exactly {image}, the encoder builds no
     low_dim keys, and obs_feature_dim is the bare VAE latent. If agent_pos or feedback
     leaked back in, the corruption ladder could not bite -- slot 0 would still know its own
     position and the exact goal-relative error -- and the arm would be a null by
     construction while every label still said otherwise.
  2. AND IT CANNOT BE PUT BACK. Re-adding a low_dim obs key must RAISE at construction.
  3. THE ENCODER IS FROZEN. After policy.train(), the backbone must still report
     training=False with no parameter requiring grad. The DDPM forward marginal is only the
     right operator on an SD latent while the encoder still produces SD latents, and
     obs_feature_std -- the scale the ladder measures its SNR against -- is only a fixed
     quantity while the encoder is.
  4. THE VERIFIER STILL SEES EVERYTHING. agent_pos and feedback must still be in the sample
     dict and the normalizer must still fit feedback: PushTVerifier resets a pymunk sim from
     them, and _normalize_value rescales the context scalar by the fitted feedback scale.
     This is the direction the check exists to catch -- removing too much is as wrong as
     removing too little, and it would quietly turn the search into noise.

Succeeds silently apart from its report; exits non-zero on the first failure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'diffusion_policy', 'config')
# config | overrides. The k=1 arm needs n_candidates=1 explicitly: ..._single pins the
# single-step TRAINER, not width one, and inherits n_candidates: 16.
ARMS = [
    ('train_pusht_unet_bc', []),
    ('train_pusht_diffusion_search_single', ['n_candidates=1']),
    ('train_pusht_diffusion_search', ['n_candidates=16']),
]
VERIFIER_KEYS = ('agent_pos', 'feedback')

failures = []


def check(cond, msg):
    print(f'    {"ok  " if cond else "FAIL"}  {msg}')
    if not cond:
        failures.append(msg)


def build(name, overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name=name, overrides=list(overrides))
    return cfg, hydra.utils.instantiate(cfg.policy)


def main():
    dataset = None
    for name, ov in ARMS:
        print(f'\n{name}  {" ".join(ov)}')
        cfg, policy = build(name, ov)
        enc = policy.obs_encoder
        backbone = enc.key_model_map['rgb' if enc.share_rgb_model else enc.rgb_keys[0]]

        check(sorted(cfg.task.shape_meta.obs.keys()) == ['image'],
              f'shape_meta.obs == {{image}} (got {sorted(cfg.task.shape_meta.obs.keys())})')
        check(list(enc.low_dim_keys) == [],
              f'encoder builds no low_dim keys (got {list(enc.low_dim_keys)})')
        check(enc.output_shape()[0] == 324,
              f'obs_feature_dim == 324, the bare VAE latent (got {enc.output_shape()[0]})')
        check(cfg.task_name == 'pusht_image_search_imgonly',
              f'task_name == pusht_image_search_imgonly (got {cfg.task_name})')

        # The obs corruption schedule sets the FLOOR on how corrupted any slot can be, and
        # it is separate from the action scheduler. On the legacy T=100 default slot 0 can
        # only ever be blurred (59% of the signal survives); TMRL's VLA schedule reaches
        # 0.6%. Asserted because nothing else would notice a silent revert to the default.
        if hasattr(policy, 'obs_noise_scheduler'):
            osc = policy.obs_noise_scheduler
            floor = float(osc.alphas_cumprod[-1].sqrt())
            check(osc.config.num_train_timesteps == 1000,
                  f'obs schedule T == 1000 (got {osc.config.num_train_timesteps})')
            check(floor < 0.01,
                  f'obs schedule reaches the marginal: sqrt(abar) floor {floor:.4f} < 0.01')

        policy.train()
        check(not backbone.training, 'backbone still in eval after policy.train()')
        n_grad = sum(p.requires_grad for p in backbone.parameters())
        check(n_grad == 0, f'no backbone parameter requires grad (got {n_grad})')
        frozen = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f'          {trainable:,} trainable / {frozen:,} frozen')

        # The dataset is identical across arms (none overrides task.dataset), so build it
        # once -- it copies the whole zarr into memory.
        if dataset is None:
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            sample = dataset[0]
            norm = dataset.get_normalizer()
            print('\n  dataset / verifier contract')
            for k in VERIFIER_KEYS:
                check(k in sample['obs'], f'{k} still emitted in the sample dict')
            check('feedback' in norm.params_dict,
                  'normalizer still fits feedback (_normalize_value reads its scale)')
            check('expert_mask' not in sample,
                  'expert_mask is gone (it was a constant nothing read)')
            si = dataset.get_split_indices()
            print(f'          split {len(si["train"])} train / {len(si["val"])} val / '
                  f'{len(si["test"])} test, checksum {si["checksum"]}')

    print('\nobs-encode caching and crop sharing (UNet BC arm)')
    cfg, policy = build('train_pusht_unet_bc', [])
    policy.set_normalizer(dataset.get_normalizer())
    policy.eval()
    B, To = 6, policy.n_obs_steps
    obs = {'image': torch.stack([dataset[i]['obs']['image'][:To] for i in range(B)])}
    with torch.no_grad():
        gc = policy._encode_obs_features(obs)
        # The diffusion sampler draws its own noise, so the two calls are only comparable
        # under the same RNG state. Seeding is what makes this a test of the CACHE rather
        # than a test of whether two samples from one conditional happen to coincide.
        torch.manual_seed(0)
        a = policy.predict_action(obs, obs_features=gc)['action']
        torch.manual_seed(0)
        b = policy.predict_action(obs)['action']
    check(tuple(gc.shape) == (B, 324 * To),
          f'cached global_cond is (B, To*324) (got {tuple(gc.shape)})')
    # Bit-identical, not merely close: the encoder is frozen and in eval, so the cached
    # encode and the recomputed one are the same tensor and the sampler sees the same input.
    check(torch.equal(a, b),
          f'cached encode gives the identical action (max|diff| {float((a-b).abs().max()):.2e})')

    # Each draw inside its own _crop_scope: outside one, `_crop_offsets_for` caches the
    # offsets indefinitely and the second call would return the first call's, which is the
    # very leak the scope exists to prevent (see unit_tests/test_crop_scope.py).
    centre = (96 - 72) // 2
    with policy._crop_scope():
        off_eval = policy._crop_offsets_for(B, repeat=To)
    policy.train()
    policy.set_crop_step(0, 0)
    with policy._crop_scope():
        off_train = policy._crop_offsets_for(B, repeat=To)
    policy.eval()
    # None is the centre crop: `_crop_offsets_for` declines to materialise offsets outside
    # training so CropRandomizer takes its own center_crop slice, which is the same pixels
    # without the per-call device syncs the forced-offset path costs.
    check(off_eval is None, 'eval delegates to the deterministic centre crop (None)')
    check(bool((policy._draw_crop_offsets(B, *policy._crop_input_hw) == centre).all()),
          'and the offsets it declines to build are the centre')
    shared = all(torch.equal(off_train[i * To], off_train[i * To + j])
                 for i in range(B) for j in range(To))
    check(shared, 'train crop: the frames of one sample share ONE offset')
    distinct = {tuple(off_train[i * To].tolist()) for i in range(B)}
    check(len(distinct) > 1,
          f'train crop: samples get different offsets (got {len(distinct)} distinct of {B})')

    print('\nre-adding a low_dim obs key must raise')
    try:
        build('train_pusht_diffusion_search',
              ['+task.shape_meta.obs.agent_pos.shape=[2]',
               '+task.shape_meta.obs.agent_pos.type=low_dim'])
        check(False, 'construction raised')
    except Exception as e:
        check('image only' in str(e), f'construction raised: {str(e)[:60]}...')

    print()
    if failures:
        print(f'{len(failures)} FAILED:')
        for f in failures:
            print('  -', f)
        sys.exit(1)
    print('all invariants hold')


if __name__ == '__main__':
    main()
