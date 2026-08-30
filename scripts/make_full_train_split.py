"""Generate pusht_seed42_train126.json -- every episode that is neither test nor val.

WHAT "ALL THE DEMOS" MEANS HERE. The zarr has 206 episodes. The upstream PushT config
(`config/task/pusht_image.yaml`) holds out 50 test and trains on the remaining 156, running
VALIDATION ON THE TEST EPISODES. That is the practice the committed split manifests exist to
remove, so "all for train" under a clean three-way split is 206 - 50 test - 30 val = 126, not
156. Same 50 test and same 30 val as every other manifest, so a 30-vs-100-vs-126 comparison
varies the training budget alone and every number stays comparable on an identical test set.

DERIVED, NOT RE-DRAWN. train126 = train100's episodes plus the 26 that the 100-demo manifest
left unused (206 minus test, val and train100). `get_split_masks_3way` -- the RNG that
produced the original permutation -- was deleted with the no-manifest branch, so re-drawing is
not possible and would not be reproducible anyway; taking the complement of the committed sets
needs no randomness and is exactly verifiable. train100 is therefore a strict subset of
train126, exactly as train30 is a strict subset of train100.

    python scripts/make_full_train_split.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.dataset.pusht_image_dataset import split_checksum

SPLITS = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/config/splits'
PARENT = 'pusht_seed42_train100.json'


def main():
    parent = json.loads((SPLITS / PARENT).read_text())
    n = parent['n_episodes']
    val, test = parent['val'], parent['test']
    held = set(val) | set(test)
    # The parent's own order first, then the leftovers in index order, so train100 is a
    # PREFIX of train126 and the subset relation is visible rather than merely true.
    leftover = sorted(set(range(n)) - held - set(parent['train']))
    train = list(parent['train']) + leftover

    out = {
        'generated_by': 'scripts/make_full_train_split.py',
        'zarr_path': parent['zarr_path'],
        'n_episodes': n,
        'episode_ends_checksum': parent['episode_ends_checksum'],
        'derivation': {
            'method': f"{PARENT}'s train list in its order, then every remaining episode "
                      f"that is neither val nor test, in index order; val and test verbatim",
            'parent': PARENT,
            'parent_checksum': parent['checksum'],
            'seed': parent['derivation'].get('seed', 42),
            'n_train': len(train),
            'n_val': len(val),
            'n_test': len(test),
        },
        'train': train,
        'val': val,
        'test': test,
    }
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])

    # every invariant this file rests on
    assert len(set(train)) == len(train), 'duplicate train episode'
    assert not (set(train) & held), 'train overlaps val or test'
    assert set(train) | held == set(range(n)), 'splits do not cover every episode'
    assert set(parent['train']) <= set(train), 'train100 is not a subset'
    assert train[:len(parent['train'])] == parent['train'], 'train100 is not a prefix'
    assert out['val'] == val and out['test'] == test

    path = SPLITS / f"pusht_seed42_train{len(train)}.json"
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')
    print(f'  train {len(train)} (= {len(parent["train"])} from {PARENT} + {len(leftover)} '
          f'previously unused)  val {len(val)}  test {len(test)}')
    print(f'  covers all {n} episodes exactly once: True')


if __name__ == '__main__':
    main()
