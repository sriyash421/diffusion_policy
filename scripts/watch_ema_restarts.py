"""CPU watcher: allocate evaluation GPUs only when a stable checkpoint exists."""
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from select_pusht_sweep import update_selection

audit = Path(sys.argv[1])
sizes = {}
while True:
    update_selection(audit)
    runs = json.loads((audit / 'retrain_manifest.json').read_text())
    registry = audit / 'fresh_evaluations.json'
    done = json.loads(registry.read_text()) if registry.exists() else {}
    for run in runs:
        for ckpt in sorted((Path(run['new']) / 'checkpoints').glob('step_*.ckpt')):
            step = int(ckpt.stem.split('_')[1])
            if step % 10000 or step > 100000:
                continue
            key = run['name'] + '_' + str(step)
            size = ckpt.stat().st_size
            previous = sizes.get(str(ckpt))
            sizes[str(ckpt)] = size
            if key in done or size != previous or not zipfile.is_zipfile(ckpt):
                continue
            folder = audit / 'fresh' / key
            folder.mkdir(parents=True, exist_ok=True)
            cells = []
            for noisy in ((True, False) if run.get('noisy', True) else (False,)):
                for mode in (('argmax',) if run.get('model') == 'bc' else ('argmax', 'final_pass')):
                    verifier = run.get('verifier_override')
                    tag = f'_ver-{verifier}' if verifier else ''
                    output = Path(run['new']) / f'bon_search_sel-{mode}{tag}_obs-{"corrupt" if noisy else "clean"}_terminal-success'
                    cells.append(dict(run=run['new'], output=str(output), corrupt_obs_eval=noisy,
                        entries=[dict(step=ckpt.stem, checkpoint=str(ckpt), n=[1,2,4,8,16,32,64],
                                      n_envs=50, seed=42, selection=mode, terminal_scoring=True,
                                      verifier_value=verifier,
                                      plan_dir=run.get('plan_dir'), store_scores=True)]))
            (folder / 'redo_manifest.json').write_text(json.dumps(cells, indent=2))
            job = subprocess.check_output(['sbatch', '--parsable', f'--array=0-{len(cells)-1}%2', '--export=NONE',
                '--job-name=eval_fixed_' + key, 'scripts/slurm/redo_noisy_evals.sbatch', str(folder)], text=True).strip()
            done[key] = dict(job=job, manifest=str(folder / 'redo_manifest.json'))
            temp = registry.with_suffix('.tmp')
            temp.write_text(json.dumps(done, indent=2)); temp.replace(registry)
    if all(run['name'] + '_' + str(step) in done
           for run in runs for step in range(10000, 100001, 10000)):
        break
    time.sleep(60)
