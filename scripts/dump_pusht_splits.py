"""Generate / verify the committed PushT split manifest.

The three splits used to be derived at runtime, independently, in three places
(``PushTImageDataset``, ``PushTSearchImageRunner``, ``eval_search_pusht``) from five keys:
``seed``, ``n_test_episodes``, ``n_val_episodes``, ``n_train_episodes``, ``train_ratio``.
Nothing recorded which episodes a checkpoint had actually been trained on, so changing any
one of those keys silently repartitioned the data -- which is what happened when
``n_val_episodes`` went 10 -> 30 and the training budget fell 29 -> 25 episodes unnoticed.

This script is the ONLY thing that writes the manifest, so regenerating it is an explicit,
reviewable act that shows up as a diff. Everything else reads it.

    # generate (writes diffusion_policy/config/splits/pusht_seed42.json)
    python scripts/dump_pusht_splits.py

    # re-derive and diff against the stored file; exits non-zero on mismatch
    python scripts/dump_pusht_splits.py --verify

    # confirm the stored episode_ends_checksum still matches the zarr on disk
    python scripts/dump_pusht_splits.py --check-zarr

``--verify`` is the guard against the manifest and the derivation drifting apart;
``--check-zarr`` is the guard against the *data* changing under fixed indices (README_pusht
documents a heredoc that mutates the zarr in place, so this is a real failure mode).
"""
import sys
import os
import pathlib

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)

import json
import click
import numpy as np
import zarr

from diffusion_policy.dataset.pusht_image_dataset import (
    build_split_manifest, episode_ends_checksum, SPLIT_NAMES)

DEFAULT_ZARR = 'data/pusht_cchi_v7_replay.zarr'
DEFAULT_OUT = 'diffusion_policy/config/splits/pusht_seed42.json'


def read_episode_ends(zarr_path):
    """Episode boundaries only -- avoids materializing the ~2.8 GB image array."""
    path = pathlib.Path(zarr_path)
    if not path.is_absolute():
        path = pathlib.Path(ROOT_DIR).joinpath(path)
    if not path.exists():
        raise FileNotFoundError(f'zarr not found: {path}')
    return np.asarray(zarr.open(str(path), 'r')['meta/episode_ends'][:])


def summarize(manifest):
    lines = [
        f"zarr            {manifest['zarr_path']}",
        f"n_episodes      {manifest['n_episodes']}",
        f"episode_ends    {manifest['episode_ends_checksum']}",
        f"derivation      {json.dumps(manifest['derivation'])}",
    ]
    total = 0
    for name in SPLIT_NAMES:
        n_ep, n_fr = len(manifest[name]), manifest[f'{name}_frames']
        total += n_ep
        lines.append(f"{name:<15} {n_ep:>4} episodes  {n_fr:>6} frames")
    lines.append(f"{'unused':<15} {manifest['n_episodes'] - total:>4} episodes")
    lines.append(f"checksum        {manifest['checksum']}")
    return '\n'.join(lines)


@click.command()
@click.option('--zarr-path', default=DEFAULT_ZARR, show_default=True)
@click.option('-o', '--out', default=DEFAULT_OUT, show_default=True)
@click.option('--seed', default=42, show_default=True)
@click.option('--n-test-episodes', default=50, show_default=True)
@click.option('--n-val-episodes', default=30, show_default=True)
@click.option('--n-train-episodes', default=100, show_default=True,
              help='absolute episode budget; a PREFIX of the permuted train pool, so '
                   'budgets are nested (25 is a strict subset of 100)')
@click.option('--verify', is_flag=True,
              help='re-derive and diff against the stored file; exit 1 on mismatch')
@click.option('--check-zarr', is_flag=True,
              help='recompute episode_ends_checksum against the zarr on disk')
def main(zarr_path, out, seed, n_test_episodes, n_val_episodes, n_train_episodes,
         verify, check_zarr):
    out_path = pathlib.Path(out)
    if not out_path.is_absolute():
        out_path = pathlib.Path(ROOT_DIR).joinpath(out_path)

    if check_zarr:
        if not out_path.is_file():
            raise SystemExit(f'no manifest at {out_path}; generate it first')
        stored = json.loads(out_path.read_text())
        actual = episode_ends_checksum(read_episode_ends(zarr_path))
        if stored.get('episode_ends_checksum') != actual:
            print(f'MISMATCH: manifest {stored.get("episode_ends_checksum")} != '
                  f'zarr {actual}\n'
                  f'The episode boundaries changed, so the stored indices no longer refer '
                  f'to the same frames. Every number reported against this manifest was '
                  f'measured on different data.')
            raise SystemExit(1)
        print(f'zarr matches manifest (episode_ends {actual})')
        return

    episode_ends = read_episode_ends(zarr_path)
    manifest = build_split_manifest(
        episode_ends=episode_ends,
        zarr_path=zarr_path,
        seed=seed,
        n_test_episodes=n_test_episodes,
        n_val_episodes=n_val_episodes,
        n_train_episodes=n_train_episodes)

    if verify:
        if not out_path.is_file():
            raise SystemExit(f'no manifest at {out_path}; generate it first')
        stored = json.loads(out_path.read_text())
        problems = list()
        for key in ('n_episodes', 'episode_ends_checksum', 'checksum'):
            if stored.get(key) != manifest[key]:
                problems.append(f'  {key}: stored={stored.get(key)!r} '
                                f'derived={manifest[key]!r}')
        for name in SPLIT_NAMES:
            if list(stored.get(name, [])) != manifest[name]:
                s, d = set(stored.get(name, [])), set(manifest[name])
                problems.append(
                    f'  {name}: {len(s)} stored vs {len(d)} derived; '
                    f'only-stored={sorted(s - d)[:5]} only-derived={sorted(d - s)[:5]}')
        if problems:
            print('MISMATCH between the stored manifest and the derivation:')
            print('\n'.join(problems))
            print('\nEither the derivation arguments changed (pass the same --seed / '
                  '--n-*-episodes the manifest was built with, see its "derivation" '
                  'block) or the file was edited by hand.')
            raise SystemExit(1)
        print(f'{out_path} matches the derivation\n')
        print(summarize(manifest))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(manifest, indent=2) + '\n')
    os.replace(tmp, out_path)
    print(f'wrote {out_path}\n')
    print(summarize(manifest))


if __name__ == '__main__':
    main()
