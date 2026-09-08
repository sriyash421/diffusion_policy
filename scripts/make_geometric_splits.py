"""Generate the block-start GEOMETRIC split manifests -- quadrant and border families.

Every other manifest in config/splits partitions episodes at random (a seed-42
permutation). These five partition by WHERE THE T STARTS, so train and eval differ by
region rather than by draw, and a policy is measured on start states it never saw.

Both families key on the T's start pose, data/block_pos[episode_start][:2] -- NOT the
agent's. The two disagree: agent and block start in the same 256-px quadrant in only
51 of 206 episodes.

Family 1 -- quadrant. Eval is one quadrant of the 512x512 arena, train is everything
else. The two datasets are not size-matched, because the demos are not uniform over the
arena: 69 episodes start with the T bottom-left, only 33 top-right.

Family 2 -- border. `d = min(x, y, 512-x, 512-y)` is the T's distance to the nearest
arena wall. Train is the k episodes with the SMALLEST d (a band hugging the walls);
test is the 50 with the LARGEST d (the interior core), shared verbatim by all three
budgets so the three runs are directly comparable. Val is the ring the wider bands
train on and this one does not -- band_100 \\ band_k -- so the 30-demo dataset is
validated on ground the 100-demo dataset treats as training data, and the val sets nest
(val_100 subset of val_60 subset of val_30). The 56 episodes between the widest band and
the core are deliberately unused: they belong to neither the train region nor the shared
eval region.

Selection is by RANK, not by a width threshold: the 60-demo cut falls between two
episodes 0.002 px apart, so a float cutoff would be a coin flip. Ties in d break by
episode index (a stable argsort), so the lists are reproducible.

    python scripts/make_geometric_splits.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import zarr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from diffusion_policy.dataset.pusht_image_dataset import (
    episode_ends_checksum, episode_frame_mask, split_checksum)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLITS = ROOT / 'diffusion_policy/config/splits'
WS = 512                         # PushTEnv.window_size
SEED = 42                        # only used to pick WHICH quadrant episodes are val

BORDER_BUDGETS = (30, 60, 100)
CORE_TEST = 50                   # the shared interior eval set
QUADRANTS = {
    'bottomleft': lambda x, y: (x < WS / 2) & (y >= WS / 2),   # y is down (see frame())
    'topright':   lambda x, y: (x >= WS / 2) & (y < WS / 2),
}


def block_starts(zarr_path):
    root = zarr.open(str(zarr_path), 'r')
    ends = np.asarray(root['meta']['episode_ends'][:])
    starts = np.concatenate([[0], ends[:-1]])
    return ends, np.asarray(root['data']['block_pos'][:])[starts][:, :2]


def wall_distance(xy):
    x, y = xy[:, 0], xy[:, 1]
    return np.minimum.reduce([x, y, WS - x, WS - y])


def finish(name, method, extra, train, val, test, ends, zarr_path):
    """Assemble, self-check and write one manifest."""
    idxs = {'train': sorted(int(i) for i in train),
            'val': sorted(int(i) for i in val),
            'test': sorted(int(i) for i in test)}
    all_idxs = [i for v in idxs.values() for i in v]
    assert len(all_idxs) == len(set(all_idxs)), f'{name}: splits overlap'
    assert len(idxs['test']) >= CORE_TEST or name.startswith('pusht_blockquad'), \
        f'{name}: test set shrank below {CORE_TEST}'

    out = {
        'generated_by': 'scripts/make_geometric_splits.py',
        'zarr_path': str(zarr_path),
        'n_episodes': int(len(ends)),
        'episode_ends_checksum': episode_ends_checksum(ends),
        'derivation': dict(method=method, key='block_pos[episode_start][:2]', **extra),
    }
    out.update(idxs)
    for split in ('train', 'val', 'test'):
        mask = np.zeros(len(ends), dtype=bool)
        mask[idxs[split]] = True
        out[f'{split}_frames'] = int(episode_frame_mask(ends, mask).sum())
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])

    path = SPLITS / f'{name}.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path.name:42s} train {len(idxs["train"]):3d}  '
          f'val {len(idxs["val"]):3d}  test {len(idxs["test"]):3d}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zarr', default='data/pusht_cchi_v7_replay.zarr')
    args = ap.parse_args()

    ends, xy = block_starts(ROOT / args.zarr)
    n = len(ends)
    d = wall_distance(xy)
    order = np.argsort(d, kind='stable')          # ties -> episode index order

    # --- family 1: quadrant --------------------------------------------------------
    for qname, pred in QUADRANTS.items():
        evalset = np.nonzero(pred(xy[:, 0], xy[:, 1]))[0]
        train = np.setdiff1d(np.arange(n), evalset)
        # Keep the rollout set at CORE_TEST where the quadrant is big enough to spare
        # episodes for val; top-right holds only 33, so it is all test and val is empty
        # (the workspace's val loop is guarded on len(val_losses) > 0).
        if len(evalset) > CORE_TEST:
            perm = np.random.default_rng(SEED).permutation(evalset)
            test, val = perm[:CORE_TEST], perm[CORE_TEST:]
        else:
            test, val = evalset, np.array([], dtype=int)
        finish(
            f'pusht_blockquad_{qname}_train{len(train)}',
            f'eval = episodes whose T starts in the {qname} quadrant of the '
            f'{WS}x{WS} arena (x/y split at {WS // 2}, y down); train = all others. '
            + (f'{CORE_TEST} of the {len(evalset)} eval episodes are test, the rest '
               f'val, chosen by a seed-{SEED} permutation.'
               if len(val) else
               f'all {len(evalset)} eval episodes are test; the quadrant holds fewer '
               f'than {CORE_TEST + 1}, so there is no val set.'),
            {'quadrant': qname, 'n_eval_region': int(len(evalset)),
             'val_seed': SEED if len(val) else None},
            train, val, test, ends, args.zarr)

    # --- family 2: border band -----------------------------------------------------
    core = order[n - CORE_TEST:]                  # largest d -> interior
    widest = set(int(i) for i in order[:max(BORDER_BUDGETS)])
    for k in BORDER_BUDGETS:
        train = order[:k]
        val = np.array(sorted(widest - set(int(i) for i in train)), dtype=int)
        finish(
            f'pusht_blockborder_train{k}_core{CORE_TEST}',
            f'train = the {k} episodes whose T starts nearest an arena wall '
            f'(d = min(x, y, {WS}-x, {WS}-y), ascending, ties by episode index); '
            f'test = the {CORE_TEST} with the largest d, identical across budgets; '
            f'val = band_{max(BORDER_BUDGETS)} minus band_{k}.',
            {'n_train': k, 'train_d_max_px': round(float(d[order[k - 1]]), 3),
             'test_d_min_px': round(float(d[core[0]]), 3),
             'n_unused': int(n - max(BORDER_BUDGETS) - CORE_TEST)},
            train, val, core, ends, args.zarr)


if __name__ == '__main__':
    main()
