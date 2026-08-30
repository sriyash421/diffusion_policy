"""Runtime check that observations are normalized ONCE on the search policy's path.

Static reading says images are mapped to [-1,1] by the LinearNormalizer and that
`imagenet_norm: False` stops the encoder doing it again. This asserts it on real data,
because the failure mode is silent: with both normalizations on, the encoder saw roughly
[-6.5, +2.25] and nothing raised -- the run just trained badly (the 2026-08-03 encoder/crop fix).

Checks:
  1. raw dataset images are in [0,1]
  2. after `policy.normalizer.normalize`, images are in [-1,1] -- normalized exactly once
  3. the obs encoder is not configured to normalize again
  4. actions round-trip: unnormalize(normalize(a)) == a

Usage: python scripts/check_normalization.py [--config-name train_pusht_diffusion_search]
"""
import sys

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, '/mmfs1/home/harine/diffusion_policy_standalone')
OmegaConf.register_new_resolver("eval", eval, replace=True)

CFG_DIR = '/mmfs1/home/harine/diffusion_policy_standalone/diffusion_policy/config'


def main(config_name='train_pusht_diffusion_search', overrides=()):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = compose(config_name=config_name, overrides=list(overrides))

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy.eval()

    batch = torch.utils.data.default_collate([dataset[i] for i in range(8)])
    fail = []

    # 1. raw images out of the dataset
    img = batch['obs']['image']
    lo, hi = float(img.min()), float(img.max())
    ok = -1e-4 <= lo and hi <= 1 + 1e-4
    print(f'raw image range          [{lo:.4f}, {hi:.4f}]  expect [0,1]        '
          f'{"OK" if ok else "FAIL"}')
    fail += [] if ok else ['raw image range']

    # 2. after the policy's normalizer -- exactly one application
    with torch.no_grad():
        nobs = policy.normalizer.normalize(batch['obs'])
    nimg = nobs['image']
    lo, hi = float(nimg.min()), float(nimg.max())
    ok = -1 - 1e-3 <= lo and hi <= 1 + 1e-3
    print(f'normalized image range   [{lo:.4f}, {hi:.4f}]  expect [-1,1]       '
          f'{"OK" if ok else "FAIL"}')
    if not ok:
        print('   -> outside [-1,1] means the image was normalized MORE THAN ONCE '
              '(double-normalizing lands it near [-6.5, 2.25])')
        fail.append('normalized image range')

    # 3. the encoder must not normalize a second time
    enc_norm = cfg.policy.obs_encoder.get('imagenet_norm', None)
    ok = enc_norm is False
    print(f'obs_encoder.imagenet_norm {str(enc_norm):<22} expect False        '
          f'{"OK" if ok else "FAIL"}')
    fail += [] if ok else ['imagenet_norm']

    # 4. action round trip
    a = batch['action']
    with torch.no_grad():
        rt = policy.normalizer['action'].unnormalize(
            policy.normalizer['action'].normalize(a))
    err = float((rt - a).abs().max())
    ok = err < 1e-4
    print(f'action round-trip max err {err:<22.3e} expect ~0           '
          f'{"OK" if ok else "FAIL"}')
    fail += [] if ok else ['action round trip']

    na = policy.normalizer['action'].normalize(a)
    print(f'   (normalized action range [{float(na.min()):.3f}, {float(na.max()):.3f}])')

    print('\nRESULT:', 'PASS' if not fail else f'FAIL {fail}')
    return 0 if not fail else 1


if __name__ == '__main__':
    args = sys.argv[1:]
    name = 'train_pusht_diffusion_search'
    if args and args[0] == '--config-name':
        name, args = args[1], args[2:]
    sys.exit(main(name, args))
