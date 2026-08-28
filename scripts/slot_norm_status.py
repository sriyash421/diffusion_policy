"""One-line-per-arm status for the per-slot weighting experiment.

    python scripts/slot_norm_status.py

Steps come from each run's logs.json.txt (the trainer's own global_step, live), checkpoints
from checkpoints/step_*.ckpt, and the sweep count from bon_search/success_curves.jsonl --
an arm is 'swept' only when a checkpoint has every n in the 1..64 grid. Read-only.
"""
import json
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_30_100_success_doc import BASE, NS, by_step, read_rows   # noqa: E402

TARGET = 100000
_K16 = 'outer_inner/value_k16_ver-armTn%s_corrupt-False_demos-30_seed-42'
_DTG = 'outer_inner/value_k16_ver-d_t_goal%s_corrupt-False_demos-30_seed-42'
ARMS = [
    # round 1 -- finished, kept so a regression in the shared tooling shows up here
    ('k=1  l2tol1', 'offline/value_k1_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42'),
    ('k=16 l2tol1', _K16 % '_l2tol1'),
    ('k=16 lin4857', _K16 % '_sw-lin4857'),
    # round 2 under armTn -- STOPPED 2026-08-27 before 100k; evals finished on what existed
    ('armTn lin100 l2', _K16 % '_sw-lin100-l2'),
    ('armTn lin100 l2tol1', _K16 % '_sw-lin100-l2tol1'),
    ('armTn geo735 l2', _K16 % '_sw-geo735-l2'),
    ('armTn geo735 l2tol1', _K16 % '_sw-geo735-l2tol1'),
    ('armTn curr-lin100 l2', _K16 % '_sw-curr-lin100-l2'),
    ('armTn curr-lin100 l2tol1', _K16 % '_sw-curr-lin100-l2tol1'),
    # round 2 under d_t_goal -- the live ones
    ('dtg lin100 l2', _DTG % '_sw-lin100-l2'),
    ('dtg lin100 l2tol1', _DTG % '_sw-lin100-l2tol1'),
    ('dtg geo735 l2', _DTG % '_sw-geo735-l2'),
    ('dtg geo735 l2tol1', _DTG % '_sw-geo735-l2tol1'),
    ('dtg curr-lin100 l2', _DTG % '_sw-curr-lin100-l2'),
    ('dtg curr-lin100 l2tol1', _DTG % '_sw-curr-lin100-l2tol1'),
]


# Longest first: a non-greedy `_ver-(.+?)_` split would take 'd' out of 'd_t_goal'.
# read_rows FILTERS on this, so passing the wrong one silently reports 0 swept for an arm
# whose curves are all on disk -- which is exactly what a hardcoded 'armTn' did to the
# d_t_goal arms here for a whole evening.
_VERIFIERS = ('d_t_goal', 'armTn', 'armTd', 'armT', 't_goal')


def _ver_of(rel):
    """The verifier a run dir was trained/swept under, read off its own name."""
    for v in _VERIFIERS:
        if f'_ver-{v}_' in rel:
            return v
    return 't_goal'          # pre-cutover runs carry no tag; see DEFAULT_VALUE_FN


def squeue_states():
    """{job_name: state}. Empty if squeue is unavailable -- never fatal."""
    try:
        out = subprocess.run(['squeue', '-u', 'harine', '-h', '-o', '%j %t'],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    return dict(l.rsplit(' ', 1) for l in out.splitlines() if ' ' in l)


def main():
    live = squeue_states()
    print(f'{"arm":<22} {"steps":>14} {"ckpt":>6} {"swept":>6}  {"train":<9} watcher')
    for label, rel in ARMS:
        run = BASE / rel
        name = rel.split('/')[-1]
        try:
            last = [l for l in (run / 'logs.json.txt').read_text().splitlines() if l.strip()][-1]
            step = json.loads(last).get('global_step')
        except Exception:
            step = None
        ck = len(list((run / 'checkpoints').glob('step_*.ckpt'))) \
            if (run / 'checkpoints').is_dir() else 0
        agg = by_step(read_rows(run, 'bon_search', _ver_of(rel)), 'success_rate')
        swept = sum(1 for v in agg.values() if set(NS) <= set(v))
        tr = live.get(f'tr_{name}', 'done' if step == TARGET else '—')
        ev = live.get(f'ev_{name}', '—')
        pct = f'{step:,}/{TARGET:,} ({100 * step // TARGET}%)' if step else '—'
        print(f'{label:<22} {pct:>14} {ck:>4}/10 {swept:>4}/10  {tr:<9} {ev}')
    print('\ntrain/watcher: R=running, PD=pending, done/— = not in queue')


if __name__ == '__main__':
    main()
