"""Build the split manifest for the Round-8 29-demo runs.

Goal: a 29-demo generation that differs from the 100-demo one ONLY in the demo count, so
the two are directly comparable, while still training on the SAME 29 episodes as the legacy
29-demo arms so old-vs-new isolates the config changes (EMA, policy-level crop, 100k steps).

  train  the legacy 29 episodes, verbatim. Contrary to the note this was written to fix,
         these ARE exactly reproducible: `downsample_mask(get_split_masks_3way(...),
         max_n=round(0.2*n_train), seed=42)` regenerates them index for index, and
         pusht_seed42_legacy_val10_train29.json records the result (verified equal). What
         was never recorded is anything IN THE RUN DIRECTORIES -- those runs predate
         splits.json, so nothing on disk ties a legacy checkpoint to the episodes behind
         it. That is a provenance gap, not an irreproducible split.
  test   the standard 50. Identical in every manifest, so test numbers are comparable
         across every section of SUCCESS_RATES.md.
  val    30 episodes, up from the legacy 10. This is the one deliberate departure. At 10
         episodes SE is ~9.5pp at p=0.9, and three legacy arms tied at 9/10 while their
         test numbers were 84 / 70 / 32% -- the selector could not tell them apart. val is
         only ever used to CHOOSE a checkpoint, never reported, so widening it changes no
         published number; it just stops selection being a coin flip.

val is drawn from the episodes used by neither train nor test, in the Round-7 order
(`perm[n_test:n_test+n_val]` under seed 42) minus anything the legacy train set claims, so
it is deterministic and derived the same way the 100-demo val was.

    python scripts/make_29demo_parity_split.py [--force]
"""
import hashlib
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy.dataset.pusht_image_dataset import (  # noqa: E402
    episode_ends_checksum, episode_frame_mask, get_split_masks_3way, split_checksum)

SPLITS = pathlib.Path('diffusion_policy/config/splits')
LEGACY = SPLITS / 'pusht_seed42_legacy_val10_train29.json'
OUT = SPLITS / 'pusht_seed42_train29_val30.json'
ZARR = 'data/pusht_cchi_v7_replay.zarr'
SEED = 42
N_VAL = 30


def main():
    legacy = json.loads(LEGACY.read_text())
    rb = ReplayBuffer.copy_from_path(ZARR, keys=['agent_pos', 'block_pos'])
    ends = rb.episode_ends[:]
    n_ep = len(ends)

    train = sorted(int(i) for i in legacy['train'])
    test = sorted(int(i) for i in legacy['test'])

    # Round-7 val order: the same permutation the 100-demo manifest used, so the new val is
    # drawn the same way rather than by a fresh ad-hoc rule.
    _, r7_val_mask, r7_test_mask = get_split_masks_3way(
        n_episodes=n_ep, n_test_episodes=len(test), n_val_episodes=N_VAL, seed=SEED,
        n_train_episodes=None)
    assert sorted(np.nonzero(r7_test_mask)[0].tolist()) == test, \
        'test split disagrees with the legacy manifest; the two generations would not be ' \
        'comparable on the only split that is ever reported'

    claimed = set(train) | set(test)
    val = [int(i) for i in np.nonzero(r7_val_mask)[0] if int(i) not in claimed]
    # top up deterministically from the unclaimed remainder if train stole any val slots
    if len(val) < N_VAL:
        rest = [i for i in range(n_ep) if i not in claimed and i not in val]
        val += rest[:N_VAL - len(val)]
    val = sorted(val[:N_VAL])

    assert not (set(train) & set(val)), 'train/val overlap'
    assert not (set(val) & set(test)), 'val/test overlap'
    assert len(train) == 29 and len(test) == 50 and len(val) == N_VAL

    masks = {}
    for name, idxs in (('train', train), ('val', val), ('test', test)):
        m = np.zeros(n_ep, dtype=bool)
        m[idxs] = True
        masks[name] = m

    manifest = {
        'generated_by': 'scripts/make_29demo_parity_split.py',
        'zarr_path': ZARR,
        'n_episodes': int(n_ep),
        'episode_ends_checksum': episode_ends_checksum(ends),
        'derivation': {
            'method': 'train = the legacy 29 (downsample_mask(train_ratio=0.2, seed=42), '
                      'byte-identical to pusht_seed42_legacy_val10_train29.json); '
                      'test = the standard 50; val = Round-7 perm order, 30 episodes, '
                      'minus anything the legacy train set claims',
            'seed': SEED,
            'n_test_episodes': len(test),
            'n_val_episodes': N_VAL,
            'n_train_episodes': len(train),
            'train_source': LEGACY.name,
        },
    }
    for name in ('train', 'val', 'test'):
        manifest[name] = [int(i) for i in np.nonzero(masks[name])[0]]
        manifest[f'{name}_frames'] = int(episode_frame_mask(ends, masks[name]).sum())
    manifest['checksum'] = split_checksum(
        manifest['train'], manifest['val'], manifest['test'])

    if OUT.exists() and '--force' not in sys.argv:
        old = json.loads(OUT.read_text())
        same = old.get('checksum') == manifest['checksum']
        print(f'{OUT} exists ({"identical" if same else "DIFFERENT"}); pass --force to rewrite')
        return 0 if same else 1
    OUT.write_text(json.dumps(manifest, indent=2))
    print(f'wrote {OUT}')
    print(f"  train {len(manifest['train'])} ({manifest['train_frames']} frames)  "
          f"val {len(manifest['val'])}  test {len(manifest['test'])}")
    print(f"  train identical to legacy: {manifest['train'] == sorted(legacy['train'])}")
    print(f"  test  identical to legacy: {manifest['test'] == sorted(legacy['test'])}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
