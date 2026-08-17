"""Generate pusht_seed42_train30.json as a strict SUBSET of the 100-demo train set.

The existing small-demo manifests are not nested inside train100 and do not share its val
set, so a 29-vs-100 comparison varies which episodes AND which val set, not just the count.
This one takes the first 30 of train100's train episodes in its own permutation order and
reuses its val (30) and test (50) verbatim, so 30-vs-100 isolates dataset size alone.

    python scripts/make_30demo_split.py [--n 30]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.dataset.pusht_image_dataset import split_checksum

SPLITS = pathlib.Path(__file__).resolve().parents[1] / 'diffusion_policy/config/splits'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--parent', default='pusht_seed42_train100.json')
    args = ap.parse_args()

    parent = json.loads((SPLITS / args.parent).read_text())
    assert len(parent['train']) >= args.n, \
        f"parent has only {len(parent['train'])} train episodes, need {args.n}"

    # Prefix of the parent's own order -- NOT a re-draw, so the subset relation is exact
    # and reproducible from the parent alone.
    train = parent['train'][:args.n]

    out = {
        'generated_by': 'scripts/make_30demo_split.py',
        'zarr_path': parent['zarr_path'],
        'n_episodes': parent['n_episodes'],
        'episode_ends_checksum': parent['episode_ends_checksum'],
        'derivation': {
            'method': f"first {args.n} of {args.parent}'s train list, in its order; "
                      f"val and test copied verbatim",
            'parent': args.parent,
            'parent_checksum': parent['checksum'],
            'seed': parent['derivation'].get('seed', 42),
            'n_train': args.n,
            'n_val': len(parent['val']),
            'n_test': len(parent['test']),
        },
        'train': train,
        'val': parent['val'],
        'test': parent['test'],
    }
    # the loader's own function, not a re-implementation -- it validates against this
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])

    # the invariants the whole point of this file rests on
    assert set(train) <= set(parent['train'])
    assert out['val'] == parent['val'] and out['test'] == parent['test']
    assert len(set(train)) == args.n
    assert not (set(train) & set(out['val'])) and not (set(train) & set(out['test']))

    path = SPLITS / f'pusht_seed42_train{args.n}.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')
    print(f'  train {len(train)} (subset of {args.parent}: True)'
          f'  val {len(out["val"])}  test {len(out["test"])}')


if __name__ == '__main__':
    main()
