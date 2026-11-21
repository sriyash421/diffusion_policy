#!/bin/bash
# Interpolation experiment, after BC: build the BC-failure test manifest, then submit the in-context ST
# (train_pusht_dt_asteroid, seed 42) on it and its stage-candidate eval. Run on a login node once
# scripts/slurm/eval_bc_fulltraj.sbatch has finished for the BC run.
#   bash scripts/run_interp_rand50.sh            # dry run: writes the manifest, prints the sbatch lines
#   SUBMIT=1 bash scripts/run_interp_rand50.sh
#   DEMOS=75 SUBMIT=1 bash scripts/run_interp_rand50.sh   # the 75-demo split (pusht_rand75_train75_test131)
#   EXTRA="asteroid.obs_noise=flat asteroid.on_suffix=400" TAG=_fl400 SUBMIT=1 bash scripts/run_interp_rand50.sh
#     # training noise on the demo's image features (flat t=400; or obs_noise=ramp on_suffix=600to200)
# Earlier steps:
#   python scripts/dump_pusht_splits.py --seed 42 --n-train-episodes 50 --n-test-episodes 156 \
#     --n-val-episodes 0 -o diffusion_policy/config/splits/pusht_rand50_train50_test156.json
#   train_pusht_unet_bc_fulltraj on it (split_suffix=_split-rand50-50k, max_gradient_steps=50000,
#   rollout_every_steps=null; the blq50far BC's schedule), then
#   sbatch --export=ALL,RUN_DIR=<bc run>,STEP=50000 scripts/slurm/eval_bc_fulltraj.sbatch
# (A 10k-step BC, split_suffix=_split-rand50, was too weak: 1/50 train and 1/156 held-out successes.)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
OUT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search_imgonly
DEMOS=${DEMOS:-50}
PARENT=pusht_rand${DEMOS}_train${DEMOS}_test$((206 - DEMOS))
BC_RUN=${BC_RUN:-$OUT/unet_bc/unetbc-full300_ver-t_goal_enc-resnet18_demos-${DEMOS}_split-rand${DEMOS}-50k_seed-42}
BC_STEP=${BC_STEP:-50000}
MIN_FAIL_STARTS=${MIN_FAIL_STARTS:-10}
# BC (50k) solves none of the 156 held-out starts, so failure = first-sample coverage < 0.3 under >= 2 of
# 3 eval seeds, and test = 25 such starts + 25 others (user's choice, 2026-11-06)
NAME=${NAME:-pusht_rand${DEMOS}_train${DEMOS}_bccov30_25x25}
SPLIT_ARGS=${SPLIT_ARGS:---fail-cov 0.3 --n-fail 25}
SUFFIX=${SUFFIX:-_split-rand${DEMOS}cov30}
SPLIT=diffusion_policy/config/splits/$NAME.json
# not ckpt-all: it can land on an 11 GB 2080 Ti, where the asteroid ST runs out of memory (2026-11-06)
A=${A:-robotics} P=${P:-gpu-a100} TIME=${TIME:-3-00:00:00}
# on ckpt-all, keep off the small GPUs: A=ckpt-robotics P=ckpt-all TIME=08:30:00 CONSTRAINT='l40s|l40|a40|a100|h200'
GPU_ARGS=${CONSTRAINT:+--constraint=$CONSTRAINT}
PY=/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python
export OMP_NUM_THREADS=1

[ -f "$SPLIT" ] || $PY scripts/make_interp_split.py --bc-run "$BC_RUN" --step "$BC_STEP" --parent "$PARENT" \
  --name "$NAME" $SPLIT_ARGS
n_fail=$($PY -c "import json; print(len(json.load(open('${SPLIT%.json}.bcfail.json'))['bc_fail_test']))")
echo "BC-failure starts in the test set: $n_fail"
if [ "$n_fail" -lt "$MIN_FAIL_STARTS" ]; then
  echo "fewer than $MIN_FAIL_STARTS BC failures: stopping before ST (MIN_FAIL_STARTS=0 to go on)"; exit 1
fi
if [ "$n_fail" -ge 50 ] && [ -z "${NO_CONTRAST_OK:-}" ]; then
  echo "every test start is a BC failure, no contrast group: stopping before ST (NO_CONTRAST_OK=1 to go on)"; exit 1
fi

OV="task.dataset.split_file=$SPLIT n_demos=$DEMOS task.dataset.n_test_episodes=50 task.dataset.n_val_episodes=0"
OV="$OV split_suffix=$SUFFIX training.seed=42 ${EXTRA:-}"
run=$($PY train.py --config-name=train_pusht_dt_asteroid $OV --cfg job --resolve | sed -n 's/^run_name: //p')
[ -n "$run" ] || { echo "config failed to resolve" >&2; exit 1; }
ST_RUN=$OUT/dt_asteroid/$run
echo "ST run: $ST_RUN"
if [ -z "${SUBMIT:-}" ]; then echo "dry run; SUBMIT=1 to sbatch"; exit 0; fi
# USR1 ten minutes before --time makes train_pusht_search.sbatch requeue and resume
jid=$(sbatch --parsable --account="$A" --partition="$P" --requeue --open-mode=append --time="$TIME" $GPU_ARGS \
      --signal=B:USR1@600 --job-name=tr_st${SUFFIX#_split-}${TAG:-} --export=ALL,CONFIG_NAME=train_pusht_dt_asteroid \
      scripts/slurm/train_pusht_search.sbatch $OV)
echo "ST training: $jid"
eid=$(sbatch --parsable --account="$A" --partition="$P" --requeue $GPU_ARGS --dependency=afterok:$jid \
      --job-name=ev_st${SUFFIX#_split-}${TAG:-} --export=ALL,RUN_DIR="$ST_RUN" scripts/slurm/eval_stage_candidates.sbatch)
echo "stage eval: $eid"
