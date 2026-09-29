"""Generate pusht_seed42_paper90.json -- the ORIGINAL Diffusion Policy PushT split.

THE PAPER'S SETUP, reproduced by calling its own functions rather than reimplementing them:

    val_mask   = get_val_mask(206, val_ratio=0.02, seed=42)   -> 4 val episodes
    train_mask = downsample_mask(~val_mask, max_n=90, seed=42) -> 90 of the remaining 202

So every reported sim PushT number is trained on 90 demos, with 4 for validation loss and
the other 112 DISCARDED. That matters for sample-efficiency claims: work that trains on all
206 and compares against the paper's PushT numbers is not matching the original setup.

THE TEST SPLIT IS EMPTY, and that is not an oversight. The paper never evaluates on held-out
EPISODES -- it rolls out 50 fresh environment seeds from test_start_seed=100000, whose
initial states PushTEnv generates itself (`reset_to_state=None` -> RandomState(seed), see
pusht_env.reset). Those states are not in the dataset at all, so there is nothing to hold
out. Our other manifests reserve 50 test episodes because our runner resets the sim to a
held-out episode's initial state instead; both readouts run on these checkpoints and land in
separate directories.

    python scripts/make_paper_split.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.common.sampler import get_val_mask, downsample_mask   # the paper's own
from diffusion_policy.dataset.pusht_image_dataset import split_checksum

SPLITS = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/config/splits'
REF = 'pusht_seed42_train100.json'      # only for n_episodes and the zarr checksum


def main():
    ref = json.loads((SPLITS / REF).read_text())
    n = ref['n_episodes']

    val_mask = get_val_mask(n_episodes=n, val_ratio=0.02, seed=42)
    train_mask = downsample_mask(~val_mask, max_n=90, seed=42)

    val = sorted(int(i) for i in np.nonzero(val_mask)[0])
    train = sorted(int(i) for i in np.nonzero(train_mask)[0])
    discarded = sorted(set(range(n)) - set(train) - set(val))

    out = {
        'generated_by': 'scripts/make_paper_split.py',
        'zarr_path': ref['zarr_path'],
        'n_episodes': n,
        'episode_ends_checksum': ref['episode_ends_checksum'],
        'derivation': {
            'method': 'ORIGINAL Diffusion Policy: get_val_mask(206, val_ratio=0.02, '
                      'seed=42) then downsample_mask(~val, max_n=90, seed=42); '
                      'no test episodes -- the paper evaluates on 50 fresh env seeds '
                      'from test_start_seed=100000',
            'seed': 42,
            'val_ratio': 0.02,
            'max_train_episodes': 90,
            'n_train': len(train),
            'n_val': len(val),
            'n_test': 0,
            'n_discarded': len(discarded),
            'test_start_seed': 100000,
            'n_test_seeds': 50,
        },
        'train': train,
        'val': val,
        'test': [],
        'discarded': discarded,
    }
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])

    assert len(train) == 90, len(train)
    assert len(val) == 4, len(val)
    assert not (set(train) & set(val)), 'train overlaps val'
    assert len(train) + len(val) + len(discarded) == n
    assert len(set(train)) == len(train) and len(set(val)) == len(val)

    path = SPLITS / 'pusht_seed42_paper90.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')
    print(f'  train {len(train)}  val {len(val)}  test 0 (fresh seeds)  '
          f'discarded {len(discarded)}')
    print(f'  val episodes: {val}')


if __name__ == '__main__':
    main()
