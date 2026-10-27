"""Atomically refresh complete evaluation tables, excluding invalidated results."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import time

from export_bon_csv import ROOT, MODELS, TAG_TO_SPLIT, FIELDS, rows_for, invalidated_readouts, _RUN_CFG, run_noise

SIMPLER = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/veritas_h16_20261002')
OUTPUT = Path(ROOT) / 'live_reports'
CORRECTED_NAME = 'pusht_simpler_corrected_verifier_results.csv'
METRIC_PILOT = Path('/gscratch/robotics/harine/RoboMonkey_data/experiments/metric_verifier_pilot_20261004')
EXTRA = ['checkpoint', 'candidate_rng', 'obs_noise_rng', 'ema_noise_buffers_repaired',
         'terminal_scoring', 'seed', 'n_generations', 'coverage_metric',
         'mean_reward_max', 'mean_reward_final', 'mean_reward_discounted', 'source_updated_utc',
         'training_observation_noise', 'evaluation_observation_noise',
         'results_status', 'verifier_name', 'verifier_settings', 'verifier_note']
SIMPLE_FIELDS = ['task', 'demos', 'model', 'variant', 'run', 'step', 'selection', 'n',
                 'condition', 'split', 'n_episodes', 'successes', 'success_rate',
                 'checkpoint', 'use_ema', 'source', 'source_updated_utc']


def utc_stamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def write_csv(path, fields, rows):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def corrected_row(row):
    """Strict current comparison subset; missing protocol evidence is not accepted."""
    if row.get('results_status') in ('archived_original_verifier', 'pilot_superseded_verifier_v1'):
        return None
    if row.get('results_status') == 'pilot_30_episodes':
        # The pilot loader below validates its explicit protocol and completeness.
        return dict(row)
    if row['environment'] == 'pusht':
        if row.get('candidate_rng') != 'paired_decision_v3' or row.get('terminal_scoring') is not True:
            return None
        # N=1 BC does not use verifier ranking; retain that baseline even if its
        # checkpoint names t_goal. BC best-of-N requires the correct verifier too.
        baseline = row['model'] == 'bc' and row['n'] == 1
        if row['value'] != 'wp_v3_la_tgn' and not baseline:
            return None
        run_dir = Path(row['source']).parents[2]
        if (run_noise(run_dir)[0] and row['obs_eval'] != 'clean'
                and row.get('obs_noise_rng') != 'per_episode_v2'):
            return None
        return dict(row, verifier_name='not_used_BC_N1' if baseline else 'wp_v3_la_tgn',
                    verifier_settings='terminal_scoring=true; candidate_rng=paired_decision_v3')
    try:
        config = json.loads(Path(row['source']).with_name('run_config.json').read_text())
        env = config['env']
        expected = {'VERIFIER_TYPE': 'veritas', 'VERITAS_PLAN_DONE_MODE': 'zero',
                    'VERITAS_HORIZON': '8', 'VERITAS_ALLOW_LIVE': '0'}
        if any(str(env.get(k, {}).get('value')) != v for k, v in expected.items()):
            return None
        if config['args'].get('score_window') != 'executed':
            return None
        if row['split'] != 'test' or row['n_episodes'] != 50:
            return None
        return dict(row, verifier_name='veritas / VeritasPixelVerifier',
                    verifier_settings='plan_done_mode=zero; horizon=8; score_window=executed; cached_plans')
    except (FileNotFoundError, KeyError, ValueError):
        return None


def export(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        invalidated_readouts.cache_clear()
        _RUN_CFG.clear()
        pusht, simpler, skipped = [], [], []
        for run in sorted(Path(ROOT).glob('*/*/*')):
            task, trainer, name = run.parts[-3:]
            match = re.search(r'_(demos-\d+(?:_split-[a-z0-9]+)?)_seed-', name)
            if trainer not in MODELS or match is None:
                continue
            split = TAG_TO_SPLIT.get(match.group(1), match.group(1))
            for path in sorted(run.glob('bon_search*/step_*/success_curve.json')):
                try:
                    curve = json.loads(path.read_text())
                    rows = list(rows_for(path, task, MODELS[trainer], split, curve))
                    if not rows:
                        skipped.append(dict(source=str(path), reason='invalidated noisy EMA evaluation'))
                    for i, row in enumerate(rows):
                        row.update({k: curve.get(k, '') for k in EXTRA[:7]})
                        row['candidate_rng'] = curve.get('candidate_rng', 'legacy_unpaired')
                        generations = curve.get('n_generations')
                        row['n_generations'] = (generations[i] if generations is not None
                                                and len(generations) == len(curve['n']) else '')
                        row['coverage_metric'] = ('mean_episode_max_raw_coverage'
                                                  if row['coverage'] != '' else 'reward_derived_bounds_only')
                        for label, key in [('mean_reward_max', 'mean_reward'),
                                           ('mean_reward_final', 'mean_reward_final'),
                                           ('mean_reward_discounted', 'mean_reward_discounted')]:
                            values = curve.get(key)
                            row[label] = (values[i] if values is not None
                                          and len(values) == len(curve['n']) else '')
                        row['source_updated_utc'] = utc_stamp(path.stat().st_mtime)
                        pusht.append(row)
                except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
                    skipped.append(dict(source=str(path), reason=str(error)))
        campaign = json.loads((SIMPLER / 'campaign.json').read_text())
        archive_path = SIMPLER / 'pause_archive.json'
        archive = json.loads(archive_path.read_text()) if archive_path.exists() else {}
        removed = set(json.loads((SIMPLER / 'noise_restart_audit.json').read_text())['removed_runs'])
        for group in ('eval', 'eval_clean', 'eval_repaired'):
            for path in sorted((SIMPLER / group).glob('*/*/*/*/eval_log.json')):
                run, step, mode, n = path.relative_to(SIMPLER / group).parts[:4]
                if run in removed or run not in campaign['runs']:
                    skipped.append(dict(source=str(path), reason='removed or inactive Simpler run'))
                    continue
                try:
                    result = json.loads(path.read_text())
                    spec = campaign['runs'][run]
                    episodes = int(result['num_episodes'])
                    if episodes <= 0:
                        continue
                    simpler.append(dict(task=spec['task'], demos=spec['demos'],
                        model='bc' if spec['kind'] == 'bc' else 'st', variant=spec['kind'],
                        run=run, step=int(step.replace('step_', '')), selection=mode,
                        n=int(n.lstrip('n')), condition=group, split=result.get('split', 'test'),
                        n_episodes=episodes, successes=result['num_successes'],
                        success_rate=result['success_rate'], checkpoint=result.get('checkpoint', ''),
                        use_ema=result.get('use_ema', ''), source=str(path),
                        source_updated_utc=utc_stamp(path.stat().st_mtime)))
                except (ValueError, KeyError, FileNotFoundError) as error:
                    skipped.append(dict(source=str(path), reason=str(error)))
        pusht.sort(key=lambda r: (r['task'], r['split'], r['run'], r['readout'], r['step'], r['n']))
        simpler.sort(key=lambda r: (r['task'], r['demos'], r['run'], r['step'], r['condition'], r['selection'], r['n']))
        for row in pusht:
            row['environment'] = 'pusht'
            trained_noisy = run_noise(Path(row['source']).parents[2])[0]
            match = re.search(r'_son-(.*?)_enc-', row['run'])
            row['variant'] = ('bc' if row['model'] == 'bc' else 'st_k1' if '_k1_' in row['run'] else
                              match.group(1) if match else 'observation_noise' if trained_noisy else 'uniform')
            row['training_observation_noise'] = row['variant'] if trained_noisy else 'none'
            row['obs_eval'] = 'noised' if trained_noisy and row['obs_eval'] != 'clean' else 'clean'
            row['evaluation_observation_noise'] = row['obs_eval']
            row['condition'] = row['obs_eval']
            demos = re.search(r'_demos-(\d+)', row['run'])
            row['demos'] = int(demos.group(1)) if demos else ''
        for row in simpler:
            row['environment'] = 'simpler'
            trained_noisy = row['variant'] not in ('bc', 'uniform')
            row['training_observation_noise'] = row['variant'] if trained_noisy else 'none'
            row['obs_eval'] = 'noised' if trained_noisy and row['condition'] != 'eval_clean' else 'clean'
            row['evaluation_observation_noise'] = row['obs_eval']
            verified = corrected_row(row)
            row['verifier_name'] = 'veritas / VeritasPixelVerifier'
            row['verifier_settings'] = (verified['verifier_settings'] if verified
                                        else 'See source run_config.json; protocol not verified')
            row['results_status'] = archive.get('results_status', 'active')
            row['verifier_note'] = archive.get('verifier_note', '')
        pilot_paths=list((METRIC_PILOT / 'eval').glob('*/*/eval_log.json'))
        pilot_paths+=list((METRIC_PILOT / 'eval_v2').glob('*/*/eval_log.json'))
        for group in ('eval_pixel_w0','eval_pixel_w0p1'):
            pilot_paths+=list((METRIC_PILOT/group).glob('*/*/eval_log.json'))
        for path in sorted(pilot_paths):
            try:
                result = json.loads(path.read_text())
                config = json.loads(path.with_name('run_config.json').read_text())
                env = config['env']
                n = int(path.parent.name[1:])
                revision='v2' if path.parents[2].name=='eval_v2' else 'v1'
                pixel=path.parents[2].name.startswith('eval_pixel_w')
                if pixel:revision=path.parents[2].name[len('eval_'):]
                verifier = 'veritas' if n == 1 else ('veritas_pixel_gripper' if pixel else 'veritas_metric')
                expected = {'VERIFIER_TYPE': verifier, 'VERITAS_HORIZON': '8',
                            'BC_CANDIDATE_POOL': '16', 'BC_BATCH_CANDIDATES': '1'}
                if result['num_episodes'] != 30 or any(
                        str(env.get(k, {}).get('value')) != v for k, v in expected.items()):
                    continue
                if revision=='v2' and n>1 and env.get('METRIC_VERIFIER_REVISION',{}).get('value')!='v2':
                    continue
                if pixel and n>1 and str(env.get('VERITAS_GRIPPER_WEIGHT',{}).get('value'))!=revision.split('_w')[1].replace('p','.'):
                    continue
                records = [json.loads(line) for line in path.with_name('episodes.jsonl').read_text().splitlines()]
                if len({r['seed'] for r in records}) != 30 or any(r.get('error') for r in records):
                    continue
                checkpoint = result['checkpoint']
                simpler.append(dict(environment='simpler', task=path.parents[1].name,
                    demos=100, model='bc', variant='bc', run=Path(checkpoint).parents[1].name,
                    step=int(Path(checkpoint).stem.split('_')[1]), selection='argmax', n=n,
                    condition='metric_verifier_pilot_'+revision, split='test_subset_first30', n_episodes=30,
                    successes=result['num_successes'], success_rate=result['success_rate'],
                    checkpoint=checkpoint, use_ema=result['use_ema'], source=str(path),
                    source_updated_utc=utc_stamp(path.stat().st_mtime), obs_eval='clean',
                    training_observation_noise='none', evaluation_observation_noise='clean',
                    results_status='pilot_superseded_verifier_v1' if revision=='v1' and n>1 else 'pilot_30_episodes',
                    verifier_name='unused_N1' if n == 1 else ('veritas_pixel_gripper_v1_'+revision.split('_')[1] if pixel else 'veritas_metric_'+revision),
                    verifier_settings='predict16; score8; execute8; fixed_candidate_pool16; fresh_reset_plan'+
                        ('; pixel_cost_scale=105; gripper_weight='+revision.split('_w')[1].replace('p','.') if pixel else ''),
                    verifier_note=('Pixel ablation: unchanged corrected Veritas pixel scorer and waypoint tracker; '
                                  'only additive gripper mismatch penalty. No metric-phase logic or calibration. '
                                  'Initial planning hints from policy-accessible object poses. Reused development seeds.' if pixel else
                                  'Pilot: VLM plan plus state metrics using policy-accessible object poses; '
                                  'no per-decision VLM calls. Benchmark stacking success can precede release. '+
                                  ('Superseded: within-chunk grasp penalty and actor-center assumption.' if revision=='v1' and n>1
                                   else 'V2 uses train50-only geometry/controller calibration; reused pilot seeds are exploratory.' if revision=='v2'
                                   else 'Verifier-independent N1 baseline.'))))
            except (ValueError, KeyError, FileNotFoundError) as error:
                skipped.append(dict(source=str(path), reason=str(error)))
        fields = ['environment'] + list(dict.fromkeys(FIELDS + EXTRA + SIMPLE_FIELDS))
        write_csv(output / 'evaluation_results.csv', fields, pusht + simpler)
        corrected = [result for row in pusht + simpler
                     if (result := corrected_row(row)) is not None]
        write_csv(output / CORRECTED_NAME, fields, corrected)
        status = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                      corrected_csv=CORRECTED_NAME, corrected_rows=len(corrected),
                      corrected_pusht_rows=sum(r['environment'] == 'pusht' for r in corrected),
                      corrected_simpler_rows=sum(r['environment'] == 'simpler' for r in corrected),
                      pusht_rows=len(pusht), simpler_rows=len(simpler), excluded_or_unreadable=skipped,
                      notes=['One row per completed checkpoint/selection/N/observation condition.',
                             'Unevaluated checkpoints are not assigned zero success.',
                             'Legacy valid evaluation protocols are labeled, not silently mixed.',
                             'Simpler five-episode diagnostics are excluded from sweep results.',
                             'Missing raw PushT coverage stays blank; reward-derived bounds are labeled.'])
        temporary = output / 'export_status.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(output / 'export_status.json')
        print(json.dumps({k: v for k, v in status.items() if k != 'excluded_or_unreadable'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        export()
        if not args.watch:
            break
        time.sleep(600)
