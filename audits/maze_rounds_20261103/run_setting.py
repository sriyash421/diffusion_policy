"""Pick MAZE_SETTING from the run being evaluated, BEFORE `rounds` is imported (rounds binds SPEC / HERE / CONT at import).

The default setting is continuous (interp30c, rounds.py). Discrete runs must not be re-evaluated in the continuous maze by accident, so
eval scripts call `resolve(tag)` first:
  - the run dir is looked up under maze_interp_20261105/runs/<tag> and maze_rounds_20261103/runs/<tag>;
  - its config.json 'setting' wins (written by asteroid.py train since 2026-11-22);
  - older runs: INTERP runs are interp30c if the policy has snap_cardinal False (continuous) else interp30 (discrete);
    runs under maze_rounds_20261103 are 11br (discrete);
  - an explicitly set MAZE_SETTING that disagrees with the run raises.
"""
import json, os
from pathlib import Path

AUD = Path(__file__).resolve().parent.parent
ROOTS = {'interp': AUD / 'maze_interp_20261105' / 'runs', 'rounds': AUD / 'maze_rounds_20261103' / 'runs'}
ACTIONS = {'11br': 'discrete', 'interp30': 'discrete', 'interp30c': 'continuous', '11brc': 'continuous'}


def action_label(setting):
    """'discrete' / 'continuous' for a MAZE_SETTING (datasets.json carries the same field as 'actions')."""
    return ACTIONS[setting]


def infer(tag):
    hits = [(k, r / tag) for k, r in ROOTS.items() if (r / tag / 'config.json').exists()]
    if not hits:
        return None
    if len(hits) > 1:
        raise SystemExit(f'run {tag!r} exists under both {hits}; set MAZE_SETTING explicitly')
    root, d = hits[0]; cfg = json.loads((d / 'config.json').read_text())
    if cfg.get('setting'):
        return cfg['setting']
    if root == 'rounds':
        return '11br'
    return 'interp30c' if cfg.get('policy', {}).get('snap_cardinal') is False else 'interp30'


def resolve(tag, allowed=None):
    """Set os.environ['MAZE_SETTING'] from the run `tag` (if it exists); returns the setting."""
    s = infer(tag) if tag else None
    env = os.environ.get('MAZE_SETTING')
    if s is None:
        return env or 'interp30c'
    if env and env != s:
        raise SystemExit(f'run {tag!r} was trained with MAZE_SETTING={s} ({ACTIONS.get(s, "?")}), but MAZE_SETTING={env} is set')
    if allowed is not None and s not in allowed:
        raise SystemExit(f'run {tag!r} is {s} ({ACTIONS.get(s, "?")}); this script supports {allowed} only')
    os.environ['MAZE_SETTING'] = s
    return s


def tag_from_argv(argv):
    for i, a in enumerate(argv):
        if a == '--tag' and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith('--tag='):
            return a.split('=', 1)[1]
    return None
