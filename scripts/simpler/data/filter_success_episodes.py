#!/usr/bin/env python3
"""Success-filtered dataset from a task's raw collection shards, in the layout of the strict
datasets (one zarr with every key, ``meta/episode_ends``, and a ``filter_manifest.json`` whose
``episodes`` carry src_shard / src_episode / seed / out_start / out_end, which the seed gate and
split builder read). Selection: ``max(reward) > 0`` in the episode.

    python scripts/simpler/data/filter_success_episodes.py --task carrot

Writes /gscratch/robotics/harine/data/<out>/<out>.zarr and filter_manifest.json. Seeds follow
the collector's rule ``100000 + START_INDEX[shard] + src_episode``
(scripts/simpler/analysis/reconstruct_demo_seeds.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts/simpler' / 'analysis'))
from reconstruct_demo_seeds import START_INDEX, START_SEED  # noqa: E402

DATA = Path('/gscratch/robotics/harine/data')
TASKS = {
    'carrot': dict(in_dir=DATA / 'carrot_on_plate', shards=['state0.zarr', 'state1.zarr', 'state2.zarr', 'state3.zarr'],
                   out='carrot_on_plate_strict', seed_task='carrot'),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    t = TASKS[a.task]
    out_dir = DATA / t['out']
    out_zarr = out_dir / f"{t['out']}.zarr"
    if out_zarr.exists():
        sys.exit(f'{out_zarr} exists; remove it to rebuild')
    selected = []   # (shard, src_episode, start, end)
    n_input = 0
    for shard in t['shards']:
        z = zarr.open(str(t['in_dir'] / shard), 'r')
        ends = z['meta/episode_ends'][:]
        starts = np.concatenate([[0], ends[:-1]])
        rewards = z['data/rewards'][:]
        n_input += len(ends)
        for k, (s, e) in enumerate(zip(starts, ends)):
            if rewards[s:e].max() > 0:
                selected.append((shard, int(k), int(s), int(e)))
    print(f'{a.task}: {len(selected)} successful of {n_input} episodes', flush=True)
    if a.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out = zarr.open(str(out_zarr), 'w')
    src0 = zarr.open(str(t['in_dir'] / t['shards'][0]), 'r')
    out.attrs.update(dict(src0.attrs))
    arrays = {}   # key -> (src path, out array)

    def ensure(group_in, group_out, prefix):
        for key, arr in group_in.items():
            if isinstance(arr, zarr.Array):
                total = sum(e - s for _, _, s, e in selected)
                o = group_out.create_dataset(key, shape=(total, *arr.shape[1:]), chunks=arr.chunks,
                                             dtype=arr.dtype, compressor=arr.compressor)
                arrays[f'{prefix}/{key}'] = o
            else:
                ensure(arr, group_out.require_group(key), f'{prefix}/{key}')
    ensure(src0['data'], out.require_group('data'), 'data')

    episodes, ends_out, cursor = [], [], 0
    open_shards = {}
    for i, (shard, k, s, e) in enumerate(selected):
        z = open_shards.setdefault(shard, zarr.open(str(t['in_dir'] / shard), 'r'))
        n = e - s
        for path, o in arrays.items():
            o[cursor:cursor + n] = z[path][s:e]
        cursor += n
        ends_out.append(cursor)
        episodes.append(dict(out_episode=i, in_episode=i, src_shard=shard, src_episode=k,
                             seed=START_SEED + START_INDEX[t['seed_task']][shard] + k,
                             n_steps=n, out_start=cursor - n, out_end=cursor))
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(selected)} episodes, {cursor} steps', flush=True)
    out.require_group('meta').create_dataset('episode_ends', data=np.asarray(ends_out, dtype=np.int64))
    manifest = dict(in_dir=str(t['in_dir']), out_zarr=str(out_zarr), n_input=n_input, n_episodes=len(selected),
                    n_steps=cursor, selection='max(reward) > 0', episodes=episodes)
    (out_dir / 'filter_manifest.json').write_text(json.dumps(manifest, indent=1))
    (out_dir / 'README.md').write_text(
        f"# {t['out']}\n\nSuccessful episodes (max reward > 0) of `{t['in_dir']}` shards {t['shards']}: "
        f"{len(selected)} of {n_input}, {cursor} steps, all zarr keys. Seeds in filter_manifest.json "
        f"follow 100000 + START_INDEX[shard] + src_episode. Built by scripts/simpler/data/filter_success_episodes.py.\n")
    print(f'wrote {out_zarr} ({len(selected)} episodes, {cursor} steps)')


if __name__ == '__main__':
    main()
