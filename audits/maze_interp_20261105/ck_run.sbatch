#!/bin/bash
# Retrain an ST run with mid-round checkpoints (--save-every 60), pick the best checkpoint of the given rounds by train_heldout and
# by test success (select_ckpt.py), then evaluate each pick on train starts + all 2870 held-out pairs (8 seeds):
#   sbatch --job-name=ck_<tag> ck_run.sbatch <tag> <rounds: last|all|0,7> <fb> [extra train flags]
#   rounds last / 0,7: 16 attempts in a row  -> runs/<tag>/bon_seeds8_n16_sel<crit>_r<k>[_allpairs].npz
#   rounds sel:        train + select every round only (then sel_eval.sbatch per round)
#   rounds all:        k+1 attempts (STk is scored on attempt k+1; skipped where the n16 file exists) -> bon_seeds8_a_sel<crit>_r<k>[...]
#   runs trained with --obs-noise also get the eval with the training noise (bon_eval --eval-obs-noise) -> bon_seeds8_noisyobs_<same>
#   env CRITS (default "train_heldout test"): which picks to evaluate.  Tags c_* run on the continuous maze (interp30c).
#SBATCH --account=cse
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/robotics/harine/codex_audits/maze_interp_20261105/logs/%x-%j.out
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MPLBACKEND=Agg
export PYTHONPATH=/gscratch/robotics/harine/L2S MAZE_SETTING=interp30
[[ $1 == c_* ]] && export MAZE_SETTING=interp30c
cd /gscratch/robotics/harine/codex_audits/maze_rounds_20261103
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
tag=$1; rk=$2; fb=$3; shift 3; D=$I/runs/$tag; CRITS=${CRITS:-train_heldout test}; SEEDS=${SEEDS:-8}   # SEEDS: smoke tests only
echo "== start $(date) on $(hostname) restart=${SLURM_RESTART_COUNT:-0} tag=$tag rounds=$rk crits=$CRITS setting=$MAZE_SETTING"
[[ -f $D/TRAINED ]] || $PY -u asteroid.py train --tag "$tag" --fb "$fb" --rounds 8 --batch 128 --micro 32 --lr 1e-4 \
    --steps-per-iter 30 --iters 30 --test-every 30 --test-seeds 8 --train-seeds 1 --save-every 60 "$@" 2>&1 | grep -v Warning | grep -E "ROUND|resumed|Traceback|Error"
[[ -f $D/TRAINED ]]
R=$($PY -c "import json; print(json.load(open('$D/config.json'))['rounds'])")
NOISE=$($PY -c "import json; print(json.load(open('$D/config.json')).get('obs_noise', 'none'))")
case $rk in last) ks=$((R - 1));; all|sel) ks=$(seq 0 $((R - 1)));; *) ks=${rk//,/ };; esac
if [[ $rk == sel ]]; then   # train + checkpoint selection only; the evals run as one sel_eval.sbatch job per round
  for k in $ks; do [[ -f $D/selection_r$k.json ]] || $PY -u $I/select_ckpt.py --tag $tag --rounds $k 2>&1 | grep -v Warning | tail -1; done
  echo "== done $(date)"; exit 0
fi
for k in $ks; do
  [[ -f $D/selection_r$k.json ]] || $PY -u $I/select_ckpt.py --tag $tag --rounds $k 2>&1 | grep -v Warning | tail -2
  prev=; pcrit=
  $PY -c "import json; [print(c, p['file']) for c, p in json.load(open('$D/selection_r$k.json'))['pick'].items()]" | while read -r crit rel; do
    [[ " $CRITS " == *" $crit "* ]] || continue
    if [[ $rk == all && -f $D/bon_seeds${SEEDS}_n16_sel${crit}_r${k}_allpairs.npz ]]; then sfx=_n16_sel${crit}_r$k; A=16; else
      [[ $rk == all ]] && { sfx=_a_sel${crit}_r$k; A=$((k + 1)); } || { sfx=_n16_sel${crit}_r$k; A=16; }; fi
    if [[ $rel == "$prev" ]]; then   # both criteria picked the same weights: reuse the eval
      for f in $D/bon_seeds$SEEDS*${psfx}*.npz; do g=${f/$psfx/$sfx}; [[ -f $g ]] || cp $f $g; done; continue
    fi
    prev=$rel; pcrit=$crit; psfx=$sfx
    for nz in '' --eval-obs-noise; do
      [[ -n $nz && $NOISE == none ]] && continue
      n=${nz:+_noisyobs}
      a=(--tag $tag --only-round $k --ckpt $rel --suffix $sfx --seeds $SEEDS --attempts $A --no-bc $nz)
      [[ -f $D/bon_seeds$SEEDS$n$sfx.npz ]] || $PY -u bon_eval.py "${a[@]}" --splits train 2>&1 | grep -v Warning | tail -1
      [[ -f $D/bon_seeds$SEEDS$n${sfx}_allpairs.npz ]] || INTERP_TEST=all $PY -u bon_eval.py "${a[@]}" --splits test 2>&1 | grep -v Warning | tail -1
    done
  done
done
echo "== done $(date)"
