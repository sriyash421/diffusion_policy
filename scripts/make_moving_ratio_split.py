"""Generate pusht_seed42_train176.json -- 176 train / 0 val / 30 test, all 206 episodes.

WHAT IS DIFFERENT ABOUT THIS SPLIT. Every other manifest in config/splits/ partitions
EPISODES and is agnostic to what is inside them. This one is built for a generation that
trains on a SUBSET of each episode's transitions -- only the windows where the demonstrator's
T actually moves, which is where the `t_goal` verifier has signal -- so the quantity that has
to come out right is the ratio of moving TRANSITIONS, not of episodes.

Episodes are not interchangeable for that purpose: they hold 22 to 145 moving windows each
(mean 75.6). Holding out any 30 episodes gives 30 * 75.6 ~ 2270 moving test windows against
13300 train, i.e. 100:17.5 -- and which 30 moves that by several points. So the 30 are
SELECTED to hit the target ratio rather than drawn and accepted.

    python scripts/make_moving_ratio_split.py

NO VAL SPLIT. This generation logs no val_loss and selects no checkpoint; the analysis steps
are named a priori. `val: []` is an already-exercised path -- both training workspaces compute
`has_val = len(val_dataset) > 0` and skip the validation block (README_pusht.md 2.3).

THE SELECTION IS DETERMINISTIC AND CARRIES ITS OWN INPUTS. There is no RNG. The rule is a
seed-free greedy plus an exhaustive first-improvement repair, both scanning in episode-index
order, so re-running reproduces the same 30 episodes exactly. `moving_per_episode` is written
into the manifest, so the choice can be re-derived and checked WITHOUT the zarr -- which is
what makes this auditable rather than merely reproducible.

THE COUNTS DEPEND ON THE WINDOW GEOMETRY, so horizon / n_obs_steps / n_action_steps and the
eps values are recorded in `predicate` and must match the transition manifest built on top of
this one (scripts/make_moving_transitions.py). A ratio targeted under one geometry is not the
same ratio under another.
"""
import json
import pathlib
import sys

import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.dataset.pusht_image_dataset import (  # noqa: E402
    episode_ends_checksum, split_checksum)
from scripts.moving_transitions_util import moving_frames, predicate_spec  # noqa: E402

SPLITS = ROOT / 'diffusion_policy/config/splits'
ZARR = 'data/pusht_cchi_v7_replay.zarr'
PARENT = 'pusht_seed42_train126.json'          # for zarr_path / n_episodes provenance only

N_TEST = 30
RATIO = (100, 20)                              # train : test, over MOVING transitions


def select_test(moving, n_test, target):
    """`n_test` episode indices whose moving counts sum as close to `target` as possible.

    Deterministic: a greedy seeded by per-episode closeness to the mean share, then
    first-improvement swaps scanned in index order until nothing improves. No RNG, no
    tie-break that depends on iteration order of a set.
    """
    per = target / n_test
    order = sorted(range(len(moving)), key=lambda e: (abs(moving[e] - per), e))
    test = sorted(order[:n_test])
    rest = sorted(set(range(len(moving))) - set(test))

    def err(sel):
        return abs(sum(moving[e] for e in sel) - target)

    improved = True
    while improved:
        improved = False
        best = (err(test), None)
        for i in test:
            for j in rest:
                cand = [e for e in test if e != i] + [j]
                d = err(cand)
                if d < best[0]:
                    best = (d, (i, j))
        if best[1] is not None:
            i, j = best[1]
            test = sorted([e for e in test if e != i] + [j])
            rest = sorted([e for e in rest if e != j] + [i])
            improved = True
    return test


def main():
    parent = json.loads((SPLITS / PARENT).read_text())
    root = zarr.open(str(ROOT / ZARR), 'r')
    ends = np.asarray(root['meta/episode_ends'])
    block_pos = np.asarray(root['data/block_pos'])
    n = len(ends)

    moving = [len(moving_frames(block_pos, ends, e)[0]) for e in range(n)]
    total = sum(moving)
    target = total * RATIO[1] / sum(RATIO)

    test = select_test(moving, N_TEST, target)
    train = sorted(set(range(n)) - set(test))
    te = sum(moving[e] for e in test)
    tr = total - te

    out = {
        'generated_by': 'scripts/make_moving_ratio_split.py',
        'zarr_path': parent['zarr_path'],
        'n_episodes': n,
        'episode_ends_checksum': episode_ends_checksum(ends),
        'derivation': {
            'method': 'test = the n_test episodes whose MOVING-transition counts sum nearest '
                      'to total*20/120, by index-ordered greedy + first-improvement swaps; '
                      'train = every other episode; val is empty by design',
            'seed': None,
            'n_test_episodes': N_TEST,
            'n_val_episodes': 0,
            'n_train_episodes': len(train),
            'target_ratio_train_test': list(RATIO),
            'target_test_moving': round(target, 1),
            'achieved_train_moving': tr,
            'achieved_test_moving': te,
            'achieved_ratio_test_per_100_train': round(100 * te / tr, 2),
            # Written so the selection can be re-derived and checked without the zarr.
            'predicate': predicate_spec(),
            'moving_per_episode': moving,
        },
        'train': train,
        'val': [],
        'test': test,
    }
    out['checksum'] = split_checksum(out['train'], out['val'], out['test'])

    # every invariant this file rests on
    assert len(set(train)) == len(train) and len(set(test)) == len(test), 'duplicate episode'
    assert not (set(train) & set(test)), 'train overlaps test'
    assert set(train) | set(test) == set(range(n)), 'splits do not cover every episode'
    assert len(test) == N_TEST, f'{len(test)} test episodes, expected {N_TEST}'
    assert tr + te == total, 'moving counts do not add up'
    # The ratio is the whole point of this generator, so hold it to a tight tolerance.
    assert abs(100 * te / tr - RATIO[1]) < 0.5, f'ratio 100:{100*te/tr:.2f}, wanted 100:{RATIO[1]}'
    # Selection must be a pure function of `moving` -- re-run it and demand the same answer.
    assert select_test(moving, N_TEST, target) == test, 'selection is not deterministic'

    path = SPLITS / f'pusht_seed42_train{len(train)}.json'
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(f'wrote {path}')
    print(f'  train {len(train)} eps / val 0 / test {len(test)} eps   (covers all {n})')
    print(f'  moving transitions: train {tr}  test {te}  -> 100:{100*te/tr:.2f} '
          f'(target 100:{RATIO[1]})')


if __name__ == '__main__':
    main()
