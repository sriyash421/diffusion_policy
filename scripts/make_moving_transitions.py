"""Generate the transition manifest: which windows of each split the T actually moves on.

    python scripts/make_moving_transitions.py
    python scripts/make_moving_transitions.py --split-file <episode manifest>

The episode manifest says which EPISODES a run sees; this says which TRANSITIONS inside them
it trains and is scored on. `t_goal` scores a candidate by where the T ends up, so on a
decision where nothing moves the block every candidate ties and the verifier expresses no
preference at all -- measured at 15-25% of decisions in
docs/reports/ppo_sac_lstm-bc_eval_2026-09-17.md. This drops those windows.

WHY A COMMITTED FILE rather than a predicate evaluated at dataset construction. Same reason
the split manifest is one: a rule recomputed at runtime is a rule that can change underneath
a half-trained run with nothing on disk recording that it did. The file is checksummed and
pinned to its parent split, and <run_dir>/splits.json records which one a checkpoint used.

READS block_pos AND episode_ends ONLY -- a few hundred KB, never the 2.8 GB image array -- so
it is safe on a login node.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import zarr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.dataset.pusht_image_dataset import (  # noqa: E402
    SPLIT_NAMES, episode_ends_checksum, load_transition_manifest, transition_checksum)
from scripts.moving_transitions_util import (  # noqa: E402
    N_ACT, moving_frames_for, predicate_spec)

SPLITS = ROOT / 'diffusion_policy/config/splits'
# Their OWN directory: these index FRAMES, not episodes, and unit_tests/test_splits.py
# globs config/splits/*.json and loads every hit as an episode manifest.
TRANS_DIR = SPLITS / 'transitions'
DEFAULT_SPLIT = 'pusht_seed42_train176.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-file', default=DEFAULT_SPLIT,
                    help='episode manifest under config/splits/ (name or path)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    sp = pathlib.Path(args.split_file)
    if not sp.is_file():
        sp = SPLITS / args.split_file
    split = json.loads(sp.read_text())

    root = zarr.open(str(ROOT / split['zarr_path']), 'r')
    ends = np.asarray(root['meta/episode_ends'])
    block_pos = np.asarray(root['data/block_pos'])

    ck = episode_ends_checksum(ends)
    if split.get('episode_ends_checksum') not in (None, ck):
        raise SystemExit(f'{sp.name} was built for a different zarr '
                         f'({split["episode_ends_checksum"]} != {ck})')

    out = {
        'generated_by': 'scripts/make_moving_transitions.py',
        'source_split_file': f'diffusion_policy/config/splits/{sp.name}',
        'source_split_checksum': split['checksum'],
        'zarr_path': split['zarr_path'],
        'episode_ends_checksum': ck,
        'predicate': predicate_spec(),
    }
    kept_by_split = {}
    for name in SPLIT_NAMES:
        kept, total = moving_frames_for(block_pos, ends, split.get(name, []))
        kept_by_split[name] = kept
        out[name] = kept
        out[f'{name}_total'] = total
        out[f'{name}_kept'] = len(kept)
    out['checksum'] = transition_checksum(kept_by_split)

    # every invariant this file rests on
    all_frames = [i for v in kept_by_split.values() for i in v]
    assert len(set(all_frames)) == len(all_frames), 'a decision frame is in two splits'
    for name in SPLIT_NAMES:
        assert kept_by_split[name] == sorted(kept_by_split[name]), f'{name} not sorted'
        assert out[f'{name}_kept'] <= out[f'{name}_total'], f'{name} kept > total'

    TRANS_DIR.mkdir(parents=True, exist_ok=True)
    path = pathlib.Path(args.out) if args.out else (
        TRANS_DIR / f'{sp.stem}_moving_ta{N_ACT}.json')
    path.write_text(json.dumps(out, indent=2) + '\n')

    # Read it straight back through the real loader, so a manifest that this repo cannot
    # load is never left on disk looking valid.
    load_transition_manifest(path, ends, split, n_action_steps=N_ACT)

    print(f'wrote {path}')
    for name in SPLIT_NAMES:
        t, k = out[f'{name}_total'], out[f'{name}_kept']
        pct = f'{k / t:6.1%}' if t else '     --'
        print(f'  {name:5s} {t:6d} windows -> {k:6d} moving ({pct})')
    tr, te = out['train_kept'], out['test_kept']
    if tr:
        print(f'  ratio train:test = 100:{100 * te / tr:.2f}')


if __name__ == '__main__':
    main()
