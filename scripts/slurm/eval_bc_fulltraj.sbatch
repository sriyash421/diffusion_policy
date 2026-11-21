#!/bin/bash
#SBATCH --job-name=ev_bc_full
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# U-Net BC full300 at one step, sampled under 3 eval seeds (eval_dt_pusht.py; --seed only moves the
# diffusion noise). Seed 0 gives 8 candidates on test and train, in the dirs the plot scripts read
# (plot_dt_rounds.BC_DIRS). Seeds 1, 2 give candidate 1 on test, for the BC-failure rule
# (scripts/make_interp_split.py). Skips finished outputs, so a requeue resumes.
#   sbatch --export=ALL,RUN_DIR=<run>,STEP=10000 scripts/slurm/eval_bc_fulltraj.sbatch
set -eo pipefail
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate vae_pushT_l2s
cd "${SLURM_SUBMIT_DIR:-/mmfs1/home/harine/diffusion_policy_standalone_pushT}"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$PWD
: "${RUN_DIR:?set RUN_DIR}" "${STEP:?set STEP}"
ck="$RUN_DIR/checkpoints/step_$(printf %07d "$STEP").ckpt"
s=step_$(printf %07d "$STEP")
run() {  # <out dir> <split> <seed> <n>
  [[ -f "$1/$2_candidates.npz" ]] && return 0
  python eval_dt_pusht.py "$ck" --n "$4" --split "$2" --seed "$3" --output-dir "$1"
}
run "$RUN_DIR/bon_search/dt_eval/$s" test 0 8
run "$RUN_DIR/bon_search/dt_eval_train/$s" train 0 8
run "$RUN_DIR/bon_search/dt_eval_seed1/$s" test 1 1
run "$RUN_DIR/bon_search/dt_eval_seed2/$s" test 2 1
