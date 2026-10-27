"""rank_arms_by_blockmotion.py outputs (analysis/nsweep_ckpt/*/step_*.json) -> one CSV.

One row per (run dir, step, subset, source, n): best-of-n and final-candidate `t_goal`
value and distance to a*, read from `by_slot` (one width-16 run gives every n <= 16).
Every arm is read on the readout named by its directory (clean / corrupt); the same arm
appears under both arm sets, which score it in a different pool.

    python scripts/export_value_vs_n_csv.py --out docs/reports/summaries/value_vs_n.csv
"""
import csv
import glob
import json
import os
import pathlib

import click

FIELDS = ['arm_set', 'split', 'readout', 'step', 'subset', 'n_decisions', 'source', 'n',
          'best_value', 'final_value', 'best_euclid_px', 'final_euclid_px', 'source_file']


@click.command()
@click.option('--root', default='analysis/nsweep_ckpt', show_default=True)
@click.option('--out', required=True, type=click.Path())
def main(root, out):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, '*', 'step_*.json'))):
        d = json.load(open(f))
        split, readout = d['split'], 'corrupt' if d.get('corrupt_obs_eval') else 'clean'
        for subset, s in d['subsets'].items():
            for src, v in s['by_slot'].items():
                for i in range(len(v['best_value'])):
                    rows.append({
                        'arm_set': pathlib.Path(f).parent.name.rsplit('_', 2)[0], 'split': split,
                        'readout': readout, 'step': d['step'], 'subset': subset,
                        'n_decisions': s['n_decisions'], 'source': src, 'n': i + 1,
                        'best_value': v['best_value'][i], 'final_value': v['final_value'][i],
                        'best_euclid_px': v['best_euclid_px'][i],
                        'final_euclid_px': v['final_euclid_px'][i], 'source_file': f})
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {out}')


if __name__ == '__main__':
    main()
