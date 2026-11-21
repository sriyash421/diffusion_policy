"""Pick the BC checkpoint by train_heldout success at n=1 (unseen pairs of TRAIN goals; test goals never used).
Ties go to the later step. Writes bc_selection.json and prints every checkpoint's n=1 success on the three splits."""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent; RUN = HERE / 'runs/interp30/bc_unet__clean/seed_0'
rows = {}
for f in sorted(RUN.glob('checkpoint_evals_n64/step_*/metrics.csv')):
    for r in csv.DictReader(f.open()):
        if int(r['n']) == 1 and r['strategy'] == 'last':
            rows.setdefault(int(r['checkpoint']), {})[r['split']] = float(r['success_rate'])
for s in sorted(rows):
    print(s, rows[s])
best = max(rows, key=lambda s: (rows[s]['train_heldout'], s))
sel = dict(step=best, rule='max train_heldout success at n=1 (ties: later step)', run=str(RUN), by_step=rows)
(HERE / 'bc_selection.json').write_text(json.dumps(sel, indent=1)); print('selected', best, rows[best])
