#!/bin/bash
#SBATCH --job-name=dt_smoke_gpt2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Round-2 smoke: GPT-2 DT k=1 debug training, then one timed k=8 compute_loss (fresh context).
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone_pushT
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate "${DP_CONDA_ENV:-vae_pushT_l2s}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export WANDB_MODE=offline HYDRA_FULL_ERROR=1
export DP_OUTPUT_ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/_dt_smoke

echo "=== 1. GPT-2 DT k=1 debug training"
[ -n "${SKIP1:-}" ] && echo "skipped" || python train.py --config-name=train_pusht_dt_bc training.debug=True training.max_gradient_steps=6 \
  training.val_every_steps=3 training.checkpoint_every=6 dataloader.num_workers=2 val_dataloader.num_workers=2 \
  logging.mode=offline \
  && echo "VERDICT 1: GPT-2 k=1 debug OK" || echo "VERDICT 1: GPT-2 k=1 debug FAILED"

echo "=== 2. timed k=8 compute_loss (batch 2, fresh context, 7 x 300-step attempts)"
python - <<'PY' && echo "VERDICT 2: k=8 loss OK" || echo "VERDICT 2: k=8 loss FAILED"
import time, hydra, torch, numpy as np
from omegaconf import OmegaConf
OmegaConf.register_new_resolver('eval', eval, replace=True)
from hydra import compose, initialize_config_dir
from diffusion_policy.common.sampler import get_collate_fn
with initialize_config_dir(version_base=None, config_dir='/mmfs1/home/harine/diffusion_policy_standalone_pushT/diffusion_policy/config'):
    cfg = compose(config_name='train_pusht_dt_search', overrides=['dataloader.num_workers=0'])
dataset = hydra.utils.instantiate(cfg.task.dataset)
policy = hydra.utils.instantiate(cfg.policy)
policy.set_normalizer(dataset.get_normalizer())
policy.cuda().train()
batch = get_collate_fn()([dataset[0], dataset[1]])
batch = {k: ({kk: vv.cuda() for kk, vv in v.items()} if isinstance(v, dict) else v.cuda()) for k, v in batch.items()}
t0 = time.perf_counter()
loss = policy.compute_loss(batch)
t1 = time.perf_counter()
loss.backward()
print(f'k=8 fresh-context compute_loss: {t1 - t0:.1f}s, backward {time.perf_counter() - t1:.1f}s, loss {loss.item():.3f}')
ctx = policy._last_context
print('context n_steps', ctx['n_steps'].tolist(), 'values', [round(v, 3) for v in ctx['values'].flatten().tolist()])
print('peak GPU mem GB', torch.cuda.max_memory_allocated() / 1e9)
policy.close()
PY
