#!/bin/bash
#SBATCH --job-name=nopos_smoke
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --requeue
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
# logs on /gscratch, not home: home is a 10G hard quota (see README_pusht.md)
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Preflight for the image-only ablation, on the preemptible checkpoint partition so the
# guaranteed robotics/weirdlab GPUs stay free for training.
#
# WHY NOT THE LOGIN NODE. This was first run there and sat 28 minutes inside one
# compute_loss: at k=16 the context search is 15 candidates x 8 DDIM steps plus a sim
# rollout each, unbatched, and CPU makes every one of those serial. On a GPU it is minutes.
# (The login node is also a shared ~10G cgroup -- three policy builds there is how a smoke
# run gets OOM-killed at exit 137 and looks like a hang.)
#
# All three arms in one job, so a single log answers "is the deletion threaded through" for
# the whole ablation. Widths are passed EXPLICITLY: n_candidates lives in
# train_pusht_diffusion_search.yaml and defaults to 16, which ..._single.yaml inherits --
# the config name selects the trainer, not the width.
#
#   sbatch scripts/slurm/nopos_smoke.sbatch
set -euo pipefail

# see eval_ckpt_pusht_search.sbatch: conda's deactivate hook reads an unset var,
# which kills the job under `set -u` if sbatch was run from an active robodiff shell
set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

rc=0
run_one () {   # $1 label  $2 config  $3... extra args
    local label="$1" cfg="$2"; shift 2
    echo "############ $label ############"
    # `|| rc=1` not `set -e`: every arm must run even if an earlier one fails, or a k=1
    # failure hides whether k=16 and the UNet are also broken.
    python -u scripts/nopos_smoke.py --config-name "$cfg" --device cuda:0 "$@" || rc=1
    echo
}

run_one "ST k=1"    train_pusht_diffusion_search_single -o n_candidates=1
run_one "ST k=16"   train_pusht_diffusion_search        -o n_candidates=16
run_one "UNet BC"   train_pusht_unet_bc

echo "############ preflight rc=$rc ############"
exit $rc
