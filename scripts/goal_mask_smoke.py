"""Assert the part-3 goal-only corruption actually reaches the training samples -- and ONLY
those. Loads the real zarr, so run it on a compute node (the img array is ~2.8GB).

    python scripts/goal_mask_smoke.py
"""
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
from diffusion_policy.env.pusht.goal_mask import goal_mask

SPLIT = 'diffusion_policy/config/splits/pusht_blockquad_bottomleft_train137.json'
GM = {'t': 800, 'margin_px': 3}

ds = PushTImageDataset(zarr_path='data/pusht_cchi_v7_replay.zarr', horizon=16,
                       n_test_episodes=50, n_val_episodes=19, n_train_episodes=137,
                       split='train', split_file=SPLIT, goal_mask_noise=GM)
clean = PushTImageDataset(zarr_path='data/pusht_cchi_v7_replay.zarr', horizon=16,
                          n_test_episodes=50, n_val_episodes=19, n_train_episodes=137,
                          split='train', split_file=SPLIT, goal_mask_noise=None)
m = torch.from_numpy(goal_mask(GM['margin_px']))

a, b = ds[0]['obs']['image'], clean[0]['obs']['image']
assert a.shape == b.shape, (a.shape, b.shape)
# outside the mask the two must be bit-identical; inside, they must differ
out_same = torch.allclose(a[..., ~m], b[..., ~m], atol=1e-6)
in_diff = not torch.allclose(a[..., m], b[..., m], atol=1e-2)
print(f'train sample: outside-mask identical={out_same}  inside-mask corrupted={in_diff}')
assert out_same and in_diff

# fresh noise per __getitem__, not a fixed field baked once
a2 = ds[0]['obs']['image']
redrawn = not torch.allclose(a[..., m], a2[..., m], atol=1e-3)
print(f'noise redrawn on re-read: {redrawn}')
assert redrawn

# val and test must be CLEAN -- _split_copy clears the flag
for name, sub in (('val', ds.get_validation_dataset()), ('test', ds.get_test_dataset())):
    assert sub._goal_mask_on is False, f'{name} split still has the corruption on'
    if len(sub) == 0:
        print(f'{name} split: empty, flag off (ok)'); continue
    v = sub[0]['obs']['image']
    vc = {'val': clean.get_validation_dataset(), 'test': clean.get_test_dataset()}[name][0]['obs']['image']
    assert torch.allclose(v, vc, atol=1e-6), f'{name} split is NOT clean'
    print(f'{name} split: clean (identical to the no-corruption dataset)')

print(f'\nmask covers {m.float().mean().item()*100:.1f}% of the frame')
print('range check: image stays in [0,1] ->', float(a.min()), float(a.max()))
assert a.min() >= 0.0 and a.max() <= 1.0

# ---- exclude_block: the repair for the arm that scored ~0.00 -------------------------
# The fixed mask sits at the goal, and the task is to push the block ONTO the goal, so at
# t=800 the block was destroyed exactly during the endgame. exclude_block subtracts the
# block's own pixels per frame; the check is that corruption stays INSIDE the goal mask but
# no longer touches the block.
from diffusion_policy.env.pusht.goal_mask import goal_mask_excluding_block

xb = PushTImageDataset(zarr_path='data/pusht_cchi_v7_replay.zarr', horizon=16,
                       n_test_episodes=50, n_val_episodes=19, n_train_episodes=137,
                       split='train', split_file=SPLIT,
                       goal_mask_noise={**GM, 'exclude_block': True})
assert xb.goal_mask_noise['exclude_block'] is True
assert ds.goal_mask_noise['exclude_block'] is False, 'default must stay off'

raw = xb.sampler.sample_sequence(0)
pose = np.asarray(raw['block_pos'], dtype=np.float32)[0]
kept = torch.from_numpy(goal_mask_excluding_block(goal_mask(GM['margin_px']), pose))
dropped = m & ~kept                      # goal pixels the block covers on this frame

x0, c0 = xb[0]['obs']['image'][0], clean[0]['obs']['image'][0]
assert torch.allclose(x0[..., ~m], c0[..., ~m], atol=1e-6), 'corruption escaped the goal mask'
if dropped.any():
    assert torch.allclose(x0[..., dropped], c0[..., dropped], atol=1e-6), \
        'block pixels were corrupted despite exclude_block'
    print(f'exclude_block: {int(dropped.sum())} block-covered px left clean, '
          f'{int(kept.sum())} goal px still corrupted')
else:
    print(f'exclude_block: block does not overlap the goal on frame 0 '
          f'({int(kept.sum())} goal px corrupted, mask unchanged)')
assert not torch.allclose(x0[..., kept], c0[..., kept], atol=1e-2), 'nothing was corrupted'

print('\nALL GOAL-MASK SMOKE CHECKS PASSED')
