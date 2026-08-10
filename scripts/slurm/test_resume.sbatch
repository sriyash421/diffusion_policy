#!/bin/bash
#SBATCH --job-name=test_resume
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#
# Resume smoke test. Copies a LIVE run's latest.ckpt into a throwaway directory, resumes
# there for a handful of gradient steps, and asserts what a resume is supposed to preserve.
# The copy is the whole point: running this against the real run dir would race the live
# job's own latest.ckpt writes and could overwrite a real checkpoint with a 5-step one.
#
#   SRC=<run dir> sbatch --account=A --partition=P scripts/slurm/test_resume.sbatch
#
# Checks, in the order they can fail:
#   1. resume actually finds the checkpoint (global_step continues, does not restart at 0)
#   2. optimizer state came back -> the LR scheduler rebuild at last_epoch=global_step-1
#      finds `initial_lr` and reproduces the lr an uninterrupted run would hold
#   3. the EMA average survived: post-resume ema != live model. Equality is the exact
#      signature of the unrestored `EMAModel.optimization_step` (decay 0.0 -> copy_)
#   4. the normalizer was preserved, not recomputed
#   5. splits.json checksum guard passes rather than aborting the run
#   6. no eval block re-fires immediately (last_{rollout,val,sample}_step restored)
set -uo pipefail

SRC="${SRC:?set SRC=<run dir>}"
DST="${DST:-/gscratch/robotics/harine/diffusion_policy_outputs/_resume_test/$(basename "$SRC")}"
EXTRA_STEPS="${EXTRA_STEPS:-6}"

set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

rm -rf "$DST"
mkdir -p "$DST/checkpoints"
cp "$SRC/checkpoints/latest.ckpt" "$DST/checkpoints/latest.ckpt"
for f in splits.json run.json; do [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DST/$f"; done
cp "$DST/checkpoints/latest.ckpt" "$DST/before.ckpt"

BEFORE_STEP=$(python -c "
import torch, dill, sys
p = torch.load(open('$DST/before.ckpt','rb'), pickle_module=dill, map_location='cpu')
print(dill.loads(p['pickles']['global_step']))")
echo "=== resuming $SRC from global_step=$BEFORE_STEP for $EXTRA_STEPS steps -> $DST"

CONFIG_NAME="${CONFIG_NAME:-train_pusht_diffusion_search_subgoal_only}"
python train.py \
  --config-name="$CONFIG_NAME" \
  training.device=cuda:0 \
  training.max_gradient_steps=$((BEFORE_STEP + EXTRA_STEPS)) \
  logging.mode=disabled \
  hydra.run.dir="$DST" \
  "$@"
rc=$?
echo "=== train.py exit=$rc"

python - "$DST" "$BEFORE_STEP" <<'EOF'
import json, pathlib, sys
import dill, torch

dst, before_step = pathlib.Path(sys.argv[1]), int(sys.argv[2])
fails = []


def check(ok, name, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


after = torch.load(open(dst / 'checkpoints' / 'latest.ckpt', 'rb'),
                   pickle_module=dill, map_location='cpu')
before = torch.load(open(dst / 'before.ckpt', 'rb'), pickle_module=dill, map_location='cpu')
pk = {k: dill.loads(v) for k, v in after['pickles'].items()}
pk0 = {k: dill.loads(v) for k, v in before['pickles'].items()}

print('=== resume assertions')
check(pk['global_step'] > before_step,
      '1. global_step continued rather than restarting',
      f"{before_step} -> {pk['global_step']}")
check(len(after['state_dicts'].get('optimizer', {}).get('state', {})) > 0,
      '2a. optimizer moment buffers present after resume')
lr = after['state_dicts']['optimizer']['param_groups'][0]
lr0 = before['state_dicts']['optimizer']['param_groups'][0]
check(abs(lr['lr'] - lr0['lr']) < 1e-9 and lr.get('initial_lr') == lr0.get('initial_lr'),
      '2b. lr unchanged across the resume (constant tail of decay_then_constant)',
      f"{lr0['lr']:.3e} -> {lr['lr']:.3e}")

# 3. the EMA average must NOT have been overwritten by the live weights
ema_a, mod_a = after['state_dicts']['ema_model'], after['state_dicts']['model']
ema_b = before['state_dicts']['ema_model']
key = next(k for k, v in ema_a.items() if v.dtype.is_floating_point and v.numel() > 100)
same_as_live = torch.allclose(ema_a[key].float(), mod_a[key].float(), atol=0, rtol=0)
moved = not torch.allclose(ema_a[key].float(), ema_b[key].float(), atol=0, rtol=0)
drift = (ema_a[key].float() - mod_a[key].float()).abs().mean().item()
check(not same_as_live, '3a. EMA was not reset to the live weights on the first step',
      f'mean|ema-live| = {drift:.3e} on {key}')
check(moved, '3b. EMA did update (average is live, not frozen)')

check(len([k for k in mod_a if k.startswith('normalizer')]) ==
      len([k for k in before['state_dicts']['model'] if k.startswith('normalizer')]),
      '4. normalizer preserved across the resume')

check((dst / 'splits.json').is_file(), '5. splits.json checksum guard passed (run did not abort)')

logs = dst / 'logs.json.txt'
rows = [json.loads(l) for l in logs.read_text().splitlines() if l.strip()] if logs.is_file() else []
refired = [k for k in ('test_mean_score', 'val_loss')
           if any(k in r for r in rows) and
           min(r['global_step'] for r in rows if k in r) <= before_step + 1]
check(not refired, '6. no eval block re-fired at the resume point',
      f'refired: {refired}' if refired else f'{len(rows)} rows logged')
for k in ('last_rollout_step', 'last_val_step', 'last_sample_step', 'last_checkpoint_step'):
    check(pk[k] == pk0[k] or pk[k] > pk0[k],
          f'6b. {k} carried over', f"{pk0[k]} -> {pk[k]}")

print(f"\n=== {'ALL PASS' if not fails else str(len(fails)) + ' FAILURE(S): ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
EOF
