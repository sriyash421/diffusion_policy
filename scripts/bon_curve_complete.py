"""Exit 0 if a bon_search directory holds a full n-sweep for every checkpoint on disk.

    python scripts/bon_curve_complete.py <BON_DIR> <N_CKPT> <N_GRID>

Used by the readout launchers to decide whether an arm still needs a watcher. Split out of
submit_30_100_watchers.sh's inline heredoc because the noised-obs grid asks it about FOUR
directories per run (argmax/final_pass x corrupt/clean rollouts), each with its own expected
grid -- the heredoc hard-coded both `bon_search` and the 7-level grid.
"""
import json
import os
import re
import sys

bon_dir, n_ckpt, grid = sys.argv[1], int(sys.argv[2]), sys.argv[3]
want = {int(x) for x in grid.split(',')}
path = os.path.join(bon_dir, 'success_curves.jsonl')
if not os.path.exists(path):
    sys.exit(1)
agg = {}
for line in open(path):
    if not line.strip():
        continue
    r = json.loads(line)
    m = re.search(r'step_(\d+)', r['checkpoint'])
    if m:
        agg.setdefault(int(m.group(1)), set()).update(r['n'])
# complete == one fully-swept row per checkpoint on disk
sys.exit(0 if len(agg) >= n_ckpt and all(want <= v for v in agg.values()) else 1)
