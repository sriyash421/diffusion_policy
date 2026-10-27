"""Single-factor ablations on the resized BLQ137 holdout pair, seed 42."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from run_pusht_capacity import REPO, configs

ARMS = [('unet_clip1', 'train_pusht_unet_bc_capacity', 'training.gradient_clip_norm=1.0'),
        ('st_noclip', 'train_pusht_st_k1_capacity', 'training.gradient_clip_norm=null'),
        ('st_dropout0', 'train_pusht_st_k1_capacity', 'policy.p_drop_attn=0.0'),
        ('st_noncausal', 'train_pusht_st_k1_capacity', 'policy.causal_attn=False')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--account', default='weirdlab')
    parser.add_argument('--partition', default='gpu-l40')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--arm', action='append', choices=[a[0] for a in ARMS],
                        help='Select arms; default is the original clipping/dropout trio.')
    args = parser.parse_args()
    os.chdir(REPO)
    overrides, pairs = configs('blq137_holdout50', 42)
    baselines = dict(pairs)
    if args.submit and args.manifest.exists():
        raise SystemExit(f'Submission record already exists: {args.manifest}')
    resolved = []
    with initialize_config_dir(config_dir=str(REPO/'diffusion_policy/config'), version_base=None):
        selected = set(args.arm or [a[0] for a in ARMS[:3]])
        for label, config, change in ARMS:
            if label not in selected:
                continue
            base = baselines[config]
            arm_overrides = overrides + [change, f'run_name={base.run_name}_abl-{label}']
            cfg = compose(config_name=config, overrides=arm_overrides)
            OmegaConf.resolve(cfg)
            expected = OmegaConf.to_container(base, resolve=True)
            actual = OmegaConf.to_container(cfg, resolve=True)
            expected['run_name'] = actual['run_name']
            # This display name embeds ${now:...}; it is not a training setting.
            expected['logging']['name'] = actual['logging']['name']
            group, key = change.split('=')[0].split('.')
            expected[group][key] = actual[group][key]
            assert actual == expected, f'Unexpected additional change in {label}'
            run = Path(cfg.output_root)/cfg.exp_name/cfg.task_name/cfg.trainer/cfg.run_name
            if args.submit and run.exists():
                raise SystemExit(f'Run already exists: {run}')
            resolved.append((label, config, arm_overrides, run, change))
    env = dict(os.environ, DP_REPO=str(REPO), TORCH_HOME='/gscratch/robotics/harine/.cache/torch',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    netrc = Path.home()/'.netrc'
    if netrc.exists() and 'api.wandb.ai' in netrc.read_text(): env.pop('WANDB_API_KEY', None)
    records = []
    for label, config, overrides, run, change in resolved:
        rec = dict(label=label, config=config, change=change, run=str(run), overrides=overrides)
        records.append(rec)
        base = ['sbatch', '--parsable', f'--account={args.account}', f'--partition={args.partition}', '--export=ALL']
        train = base + ['--time=1-00:00:00', f'--job-name=capacity_{label}', 'scripts/slurm/train_pusht_search.sbatch', *overrides]
        print(shlex.join(train), flush=True)
        if args.submit:
            rec['train_job'] = subprocess.check_output(train, env=dict(env, CONFIG_NAME=config), text=True).strip().split(';')[0]
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(json.dumps(records, indent=2))
        evaluation = base + [f'--dependency=afterok:{rec.get("train_job", "TRAIN_JOB")}',
            '--time=8:00:00', f'--job-name=capacity_eval_{label}', 'scripts/slurm/eval_pusht_capacity.sbatch', str(run)]
        rec['eval_command'] = evaluation
        print(shlex.join(evaluation), flush=True)
        if args.submit:
            rec['eval_job'] = subprocess.check_output(evaluation, env=env, text=True).strip().split(';')[0]
            args.manifest.write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
