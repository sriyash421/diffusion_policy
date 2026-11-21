#!/bin/bash
# Smoke for ck_run.sbatch rounds=all (+ noisy evals, + continuous maze) and bc_closed --obs-noise-t, on tiny 2-round runs (removed after).
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue
#SBATCH --job-name=smoke_all
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105; PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
T=(--rounds 2 --batch 32 --steps-per-iter 10 --iters 4 --test-seeds 1 --save-every 10)
rm -rf $I/runs/int_smk_noise $I/runs/c_smk $I/runs/smk_bcnoise
SEEDS=1 CRITS=train_heldout bash $I/ck_run.sbatch int_smk_noise all bfs_dir "${T[@]}" --obs-noise ramp
ls $I/runs/int_smk_noise
for f in a_seltrain_heldout_r0 a_seltrain_heldout_r1 a_seltrain_heldout_r0_allpairs noisyobs_a_seltrain_heldout_r1_allpairs; do
  $PY -c "import numpy as np; z=np.load('$I/runs/int_smk_noise/bon_seeds1_$f.npz'); k=[x for x in z if x.endswith('st_success')][0]; print('$f', k, z[k].shape, z[k].mean().round(3), z.get('eval_obs_noise_t'))"
done
SEEDS=1 bash $I/ck_run.sbatch c_smk all bfs_dir "${T[@]}"
ls $I/runs/c_smk
$PY -c "import numpy as np; z=np.load('$I/runs/c_smk/bon_seeds1_a_seltest_r1_allpairs.npz'); print(z['test_st_success'].shape, z['test_st_success'].mean())"
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30; cd $I
$PY -u bc_closed.py --tag smk_bcnoise --run runs/interp30/bc_unet__clean__h4/seed_0 --step 10000 --exec 1 --seeds 1 --splits train,allpairs --obs-noise-t 0,400,900 --out bc_noisy.npz 2>&1 | grep -v Warning
$PY -c "import numpy as np; z=np.load('runs/smk_bcnoise/bc_noisy.npz'); print(sorted(z.keys()))"
rm -rf $I/runs/int_smk_noise $I/runs/c_smk $I/runs/smk_bcnoise; echo SMOKE_ALL_OK
