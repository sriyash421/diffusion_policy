"""Dataset -> env-seed provenance for the SIMPLER tasks.

No dataset stores its seeds, but every filter/subsample stage keeps the raw-shard
coordinates (``src_shard``, ``src_episode``) in ``filter_manifest.json``, and collection
used ``env.reset(seed=START_SEED + global_idx)``, so a seed is
``START_SEED + start_index[src_shard] + src_episode``. The split manifests under ``config/splits/simpler/<task>/`` carry that seed per episode
and are the eval seed source; ``scripts/simpler/eval_search/demoseeds/<task>_<split>.json``
(``reconstruct_demo_seeds.py``) predates them and only describes the archived sub50 sets.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# env task name -> demoseeds task short name
TASK_SHORT = {
    'widowx_put_eggplant_in_basket': 'eggplant',
    'widowx_spoon_on_towel': 'spoon',
    'widowx_stack_cube': 'stack_cube',
}
SPLITS = ('test', 'train')


def demoseed_dir() -> Path:
    env = os.environ.get('DEMOSEED_DIR')
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / 'scripts' / 'simpler' / 'eval_search' / 'demoseeds'


def file_md5(path) -> Optional[str]:
    try:
        return hashlib.md5(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def load_split(env_task: str, split: str) -> Dict[str, Any]:
    """The demoseeds manifest for (task, split), with ``path``/``md5`` added."""
    assert split in SPLITS, f'split must be one of {SPLITS}, got {split!r}'
    short = TASK_SHORT.get(env_task, env_task)
    path = demoseed_dir() / f'{short}_{split}.json'
    if not path.exists():
        raise FileNotFoundError(
            f'no demo-seed manifest for task={env_task!r} split={split!r}: {path}. '
            f'Run scripts/simpler/analysis/reconstruct_demo_seeds.py first.')
    d = json.loads(path.read_text())
    d['path'] = str(path)
    d['md5'] = file_md5(path)
    return d


def split_seeds(manifest: Dict[str, Any]) -> List[int]:
    return [int(e['seed']) for e in manifest['episodes']]


def load_split_file(split_file) -> Dict[str, Any]:
    """A validated split manifest (disjoint lists, self-checksum), with ``path``/``md5``
    and ``seeds``: {split name: [env seed of each listed episode, in list order]}."""
    from diffusion_policy.common.split_manifest import SPLIT_NAMES, load_split_manifest
    m = load_split_manifest(split_file)
    path = Path(split_file).resolve()
    seeds: Dict[str, List[int]] = {}
    for name in SPLIT_NAMES:
        rows = [m['episodes'][str(i)] for i in m[name]]
        missing = [i for i, r in zip(m[name], rows) if r.get('seed') is None]
        if missing:
            raise ValueError(f'{path}: {len(missing)} {name} episode(s) have no '
                             f'reconstructable seed: {missing[:10]}')
        seeds[name] = [int(r['seed']) for r in rows]
    m['path'] = str(path)
    m['md5'] = file_md5(path)
    m['seeds'] = seeds
    return m


def split_file_provenance(split_file) -> Dict[str, Any]:
    """What a checkpoint records about the split it trained on (``split_manifest``).
    Seeds, not indices, are the identity: a manifest remapped onto a smaller dataset keeps
    every split's seeds but renumbers the episodes."""
    m = load_split_file(split_file)
    out = {'path': m['path'], 'md5': m['md5'], 'checksum': m.get('checksum'),
           'rule': (m.get('derivation') or {}).get('rule')}
    for name, seeds in m['seeds'].items():
        out[name] = {'n': len(seeds), 'seeds': seeds}
    return out


def dataset_provenance(dataset_dir, split_file=None) -> Dict[str, Any]:
    """Everything that identifies the data a policy was trained on.

    Walks the ``filter_manifest.json`` chain (``in_dir`` links) back to the raw shards and
    records each stage's manifest md5, so two checkpoints trained on the same directory
    name but different contents are distinguishable. With ``split_file``, records that
    manifest (``split_manifest``: path, md5, per-split seeds) and takes ``train_split``
    from it; otherwise ``train_split`` is the demoseeds train set matching this directory.
    """
    out: Dict[str, Any] = {'dataset_dir': str(dataset_dir), 'chain': []}
    if split_file is not None:
        import hydra.utils
        sm = split_file_provenance(hydra.utils.to_absolute_path(str(split_file)))
        out['split_manifest'] = sm
        out['train_split'] = {'path': sm['path'], 'md5': sm['md5'],
                              'n': sm['train']['n'], 'seeds': sm['train']['seeds']}
    seen, cur = set(), Path(str(dataset_dir))
    while cur and str(cur) not in seen:
        seen.add(str(cur))
        mpath = cur / 'filter_manifest.json'
        stage: Dict[str, Any] = {'dir': str(cur), 'manifest_md5': file_md5(mpath)}
        if not mpath.exists():
            stage['zarrs'] = sorted(p.name for p in cur.glob('*.zarr')) if cur.exists() else None
            out['chain'].append(stage)
            break
        try:
            m = json.loads(mpath.read_text())
        except (OSError, ValueError) as e:
            stage['error'] = str(e)
            out['chain'].append(stage)
            break
        for k in ('n_input', 'n_episodes', 'n_steps', 'seed', 'in_dir', 'sub_dir'):
            if k in m:
                stage[k] = m[k]
        out['chain'].append(stage)
        nxt = m.get('in_dir')
        cur = Path(nxt) if nxt else None
    for p in ([] if split_file is not None else sorted(demoseed_dir().glob('*_train.json'))):
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if str(d.get('dataset_dir', '')).rstrip('/') == str(dataset_dir).rstrip('/'):
            out['train_split'] = {'path': str(p), 'md5': file_md5(p), 'n': d.get('n'),
                                  'start_seed': d.get('start_seed'),
                                  'start_index': d.get('start_index'),
                                  'seeds': [int(e['seed']) for e in d['episodes']]}
            break
    return out
