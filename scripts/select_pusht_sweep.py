"""Select BC at N=1 and search over its sweep, without matching checkpoint steps."""
import json
from pathlib import Path


def update_selection(audit):
    audit = Path(audit)
    reruns = json.loads((audit / 'retrain_manifest.json').read_text())
    allowed_search = {r[key] for r in reruns for key in ('old', 'new')}
    readouts = {r['output']: r['run']
                for r in json.loads((audit / 'redo_manifest.json').read_text())}
    for run in reruns:
        for path in Path(run['new']).glob('bon_search*'):
            readouts[str(path)] = run['new']
    cells = {'bc': [], 'search': []}
    for directory, run in readouts.items():
        is_bc = Path(run).parent.name == 'unet_bc'
        if not is_bc and run not in allowed_search:
            continue
        for path in Path(directory).glob('step_*/success_curve.json'):
            try:
                curve = json.loads(path.read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            # Current comparison only: do not select from obsolete RNG/scoring runs.
            if (curve.get('candidate_rng') != 'paired_decision_v3'
                    or not curve.get('terminal_scoring', False)):
                continue
            for i, n in enumerate(curve['n']):
                if is_bc and n != 1:
                    continue
                cells['bc' if is_bc else 'search'].append(dict(
                    run=run, checkpoint=path.parent.name, n=n,
                    selection=curve.get('selection', 'argmax'),
                    success_rate=curve['success_rate'][i], episodes=curve['n_episodes'],
                    corrupt_obs_eval=curve.get('corrupt_obs_eval'), source=str(path)))
    selected = {}
    for model, rows in cells.items():
        selected[model] = max(rows, key=lambda r: (
            r['success_rate'], -r['n'], -int(r['checkpoint'].split('_')[1]),
            r['selection'] == 'argmax')) if rows else None
    result = dict(rule='BC: best checkpoint at N=1; search: best checkpoint/N/selection in sweep',
                  candidate_rng='paired_decision_v3', selected=selected,
                  completed_cells={k: len(v) for k, v in cells.items()},
                  interpretation='Observed selection scores; fresh holdout evaluation is separate.')
    target = audit / 'selected_best_sweep.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    return result


if __name__ == '__main__':
    import sys
    print(json.dumps(update_selection(sys.argv[1]), indent=2))
