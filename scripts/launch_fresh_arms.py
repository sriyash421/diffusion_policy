"""Launch the corrected-protocol PushT arms (BC, ST k1, ST k16 uniform/flat600/ramp) for a
split x seeds grid, the way the 2026-10-03 far50 retrain was submitted, and start the
checkpoint watcher that evaluates every 10k checkpoint.

The override sets are copied verbatim from pusht_far50_20261003/retrain_manifest.json so a
new seed or a new quadrant differs from the report's seed-42 blq50far arms in exactly
`training.seed` and the four dataset keys. The audit directory gets the same files that
directory has -- retrain_manifest.json, redo_manifest.json ([]), jobs.json -- so
scripts/watch_ema_restarts.py and scripts/select_pusht_sweep.py run on it unchanged.

    python scripts/launch_fresh_arms.py --audit pusht_tlq50far_20261006 \
        --split-tag _split-tlq50far \
        --split-file diffusion_policy/config/splits/pusht_blockquad_topleft_train50_far_valpool19.json \
        --n-test 46 --n-val 19 --seeds 42 43 44                 # dry run: resolves names
    python scripts/launch_fresh_arms.py ... --submit            # sbatch + watcher

Every run_name is checked against what hydra resolves for the same overrides, so a
template drift fails here instead of resuming the wrong directory.
"""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_ROOT = pathlib.Path('/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search')
LOGS = pathlib.Path('/gscratch/robotics/harine/slurm_logs')
TRAIN_SBATCH = ['sbatch', '--parsable', '--account=ckpt-robotics', '--partition=ckpt',
                '--qos=ckpt-gpu', '--constraint=a40|l40s|l40|a100', '--time=9:00:00',
                '--signal=B:USR1@600', '--requeue', '--open-mode=append']

ST_COMMON = ['training.device=cuda:0', 'verifier_tag=wp_v3_la_tgn',
             'training.rollout_every_steps=null', 'verifier_terminal_scoring=True',
             'ts_suffix=_ts-term', 'training.fresh_context=true', 'training.inner_epochs=1',
             '+training.checkpoint_steps=[1,1000,2000,5000]']
FLAT600 = ['slot_obs_noise.mode=list', '+slot_obs_noise.timesteps=' + str([600] * 16).replace(' ', ''),
           'son_suffix=_son-flat600', 'son_tag=flat-600', 'obs_noise_tag=obs_noised']
RAMP = ['slot_obs_noise.mode=list',
        '+slot_obs_noise.timesteps=[200,227,253,280,307,333,360,387,413,440,467,493,520,547,573,600]',
        'son_suffix=_son-ramp200to600', 'son_tag=ramp-200-600', 'obs_noise_tag=obs_noised']

# arm key -> (config, job suffix, trainer dir, run_name template, overrides, noisy)
ARMS = {
    'bc': ('train_pusht_unet_bc', 'bc', 'unet_bc',
           'unetbc_ver-t_goal_enc-resnet18_demos-50{split}_seed-{seed}',
           ['training.max_gradient_steps=100000', 'training.rollout_every=null',
            'training.rollout_every_steps=null'], False),
    'k1': ('train_pusht_diffusion_search', 'st_k1', 'outer_inner',
           'value_k1_ver-wp_v3_la_tgn_ts-term_enc-resnet18_demos-50{split}_seed-{seed}_fresh_context_v3',
           ['n_candidates=1', 'slot_obs_noise.mode=uniform'] + ST_COMMON
           + ['training.max_gradient_steps=100000'], False),
    'k16': ('train_pusht_diffusion_search', 'uniform', 'outer_inner',
            'value_k16_ver-wp_v3_la_tgn_ts-term_enc-resnet18_demos-50{split}_seed-{seed}_fresh_context_v3',
            ['n_candidates=16', 'slot_obs_noise.mode=uniform'] + ST_COMMON, False),
    'flat600': ('train_pusht_diffusion_search', 'flat600', 'outer_inner',
                'value_k16_ver-wp_v3_la_tgn_ts-term_son-flat600_enc-resnet18_demos-50{split}_seed-{seed}_fresh_context_v3',
                ['n_candidates=16'] + FLAT600 + ST_COMMON, True),
    'ramp': ('train_pusht_diffusion_search', 'ramp200to600', 'outer_inner',
             'value_k16_ver-wp_v3_la_tgn_ts-term_son-ramp200to600_enc-resnet18_demos-50{split}_seed-{seed}_fresh_context_v3',
             ['n_candidates=16'] + RAMP + ST_COMMON, True),
}


def resolved_run_name(config, overrides):
    out = subprocess.run([sys.executable, 'train.py', f'--config-name={config}', *overrides,
                          '--cfg', 'job', '--resolve'], cwd=ROOT, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f'config failed to resolve for {config} {overrides}:\n{out.stderr[-2000:]}')
    names = [l.split(': ', 1)[1] for l in out.stdout.splitlines() if l.startswith('run_name: ')]
    return names[0] if names else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit', required=True, help='directory name under ' + str(OUT_ROOT))
    ap.add_argument('--split-tag', required=True, help='e.g. _split-tlq50far')
    ap.add_argument('--split-file', required=True)
    ap.add_argument('--n-test', type=int, required=True)
    ap.add_argument('--n-val', type=int, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', required=True)
    ap.add_argument('--arms', nargs='+', default=list(ARMS), choices=list(ARMS))
    ap.add_argument('--submit', action='store_true')
    ap.add_argument('--no-watcher', action='store_true')
    a = ap.parse_args()

    split_file = pathlib.Path(a.split_file)
    assert (ROOT / split_file).is_file(), f'missing {split_file}'
    manifest_split = json.loads((ROOT / split_file).read_text())
    assert len(manifest_split['train']) == 50, 'these arms are demos-50 arms'
    assert (len(manifest_split['test']), len(manifest_split['val'])) == (a.n_test, a.n_val), \
        f'{split_file}: test/val {len(manifest_split["test"])}/{len(manifest_split["val"])} ' \
        f'!= --n-test/--n-val {a.n_test}/{a.n_val}'
    dataset_ov = [f'task.dataset.split_file={split_file}', 'n_demos=50',
                  f'split_suffix={a.split_tag}', f'task.dataset.n_test_episodes={a.n_test}',
                  f'task.dataset.n_val_episodes={a.n_val}']

    audit = OUT_ROOT / a.audit
    entries = json.loads((audit / 'retrain_manifest.json').read_text()) \
        if (audit / 'retrain_manifest.json').exists() else []
    known = {e['name'] for e in entries}
    live = set(subprocess.run(['squeue', '-u', 'harine', '-h', '-o', '%j'],
                              capture_output=True, text=True).stdout.split())
    jobs = json.loads((audit / 'jobs.json').read_text()) if (audit / 'jobs.json').exists() else {}
    prefix = a.audit.split('_')[1]   # pusht_<prefix>_<date> -> slurm log / job-name prefix

    new = []
    for seed in a.seeds:
        for arm in a.arms:
            config, suffix, trainer, tmpl, arm_ov, noisy = ARMS[arm]
            name = tmpl.format(split=a.split_tag, seed=seed)
            run_dir = OUT_ROOT / 'pusht_image_search_imgonly' / trainer / name
            # ORDER MATTERS ONLY FOR READABILITY; the seed goes last so a diff against the
            # far50 manifest is one line.
            ov = arm_ov + dataset_ov + [f'run_name={name}', f'logging.name={name}',
                                        f'logging.id={name}', f'training.seed={seed}']
            if arm == 'bc':
                ov = [o for o in ov if o != 'training.device=cuda:0']
            job_name = f'{prefix}_{suffix}_s{seed}'
            status = ('in manifest' if name in known else 'queued' if job_name in live
                      else 'FINISHED' if (run_dir / 'checkpoints' / 'step_0100000.ckpt').exists()
                      else 'new')
            got = resolved_run_name(config, ov)
            assert got == name, f'run_name template drift: hydra resolves {got!r}, expected {name!r}'
            print(f'{status:12s} {job_name:22s} {name}')
            if status != 'new':
                continue
            entry = dict(name=name, old=str(run_dir), new=str(run_dir), config=config,
                         overrides=ov, noisy=noisy, model='bc' if arm == 'bc' else 'st')
            if arm == 'bc':
                entry.update(verifier_override='wp_v3_la_tgn',
                             plan_dir='media/veritas_pusht/v3_end_outside')
            new.append((job_name, entry))

    print(f'\n{len(new)} new arm(s)' + ('' if a.submit else ' -- dry run, add --submit'))
    if not a.submit or not new:
        return
    audit.mkdir(parents=True, exist_ok=True)
    if not (audit / 'redo_manifest.json').exists():
        (audit / 'redo_manifest.json').write_text('[]\n')
    for job_name, entry in new:
        cmd = TRAIN_SBATCH + [f'--export=NONE,CONFIG_NAME={entry["config"]}',
                              f'--job-name={job_name}', f'--output={LOGS}/{prefix}-%j.out',
                              'scripts/slurm/train_pusht_search.sbatch', *entry['overrides']]
        entry['job'] = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
        entries.append(entry)
        print(f'submitted {entry["job"]}  {job_name}')
        # write after every submission so a failure mid-loop leaves a truthful manifest
        (audit / 'retrain_manifest.json').write_text(json.dumps(entries, indent=1) + '\n')
    (audit / 'split_file.json').write_text(json.dumps(
        dict(split_file=str(split_file), split_tag=a.split_tag,
             checksum=manifest_split['checksum']), indent=1) + '\n')
    if a.no_watcher:
        return
    if 'fresh_checkpoint_watcher' in jobs and f'watch_{prefix}' in live:
        print(f'watcher already running: {jobs["fresh_checkpoint_watcher"]} (it re-reads the manifest)')
        return
    jobs['fresh_checkpoint_watcher'] = subprocess.check_output(
        ['sbatch', '--parsable', '--export=NONE', f'--job-name=watch_{prefix}',
         'scripts/slurm/watch_ema_restarts.sbatch', str(audit)], cwd=ROOT, text=True).strip()
    (audit / 'jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
    print(f'watcher {jobs["fresh_checkpoint_watcher"]}')


if __name__ == '__main__':
    main()
