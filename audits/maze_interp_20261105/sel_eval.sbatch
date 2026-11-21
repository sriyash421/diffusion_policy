#!/bin/bash
# n = 1..16 eval of one round's best checkpoint (by train_heldout; selection_r<k>.json from ck_run.sbatch ... sel), train starts + all
# 2870 held-out pairs, 8 seeds x 16 attempts in a row; runs trained with --obs-noise also get the eval with their training noise.
#   sbatch --job-name=se_<tag>_r<k> sel_eval.sbatch <tag> <k>   -> runs/<tag>/bon_seeds8[_noisyobs]_n16_seltrain_heldout_r<k>[_allpairs].npz
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
[[ $1 == c_* ]] && export MAZE_SETTING=interp30c
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105; tag=$1; k=$2; D=$I/runs/$tag; crit=train_heldout
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag round=$k"
[[ -f $D/selection_r$k.json ]] || $PY -u $I/select_ckpt.py --tag $tag --rounds $k 2>&1 | grep -v Warning | tail -1
rel=$($PY -c "import json; print(json.load(open('$D/selection_r$k.json'))['pick']['$crit']['file'])")
NOISE=$($PY -c "import json; print(json.load(open('$D/config.json')).get('obs_noise', 'none'))")
sfx=_n16_sel${crit}_r$k
for nz in '' --eval-obs-noise; do
  [[ -n $nz && $NOISE == none ]] && continue
  n=${nz:+_noisyobs}; a=(--tag $tag --only-round $k --ckpt $rel --suffix $sfx --seeds 8 --attempts 16 --no-bc $nz)
  [[ -f $D/bon_seeds8$n$sfx.npz ]] || $PY -u bon_eval.py "${a[@]}" --splits train 2>&1 | grep -v Warning | tail -1
  [[ -f $D/bon_seeds8$n${sfx}_allpairs.npz ]] || INTERP_TEST=all $PY -u bon_eval.py "${a[@]}" --splits test 2>&1 | grep -v Warning | tail -1
done
echo "== done $(date)"
