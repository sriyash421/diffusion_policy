#!/bin/bash
#SBATCH --job-name=tgoal_moving_loop
#SBATCH --partition=ckpt-all
#SBATCH --account=weirdlab
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Run the analysis autoupdate loop AS A SLURM JOB rather than a detached shell.
#
# WHY NOT `setsid nohup ... &`. That is what scripts/slurm/autoupdate_geometric_readouts.sh
# documents, and it works from a human's login shell -- but launched from an agent session it
# dies with the session: measured on 2026-09-18, the loop logged two passes and was gone when
# the session ended, leaving four checkpoints unanalysed. A batch job outlives any shell.
#
# NO GPU. The loop only runs squeue, a little json reading, and sbatch; the analysis work
# happens in the jobs it submits. ckpt-all is preemptible, which is fine -- requeue picks the
# loop back up and the submitter is idempotent, so a preemption costs nothing.
#SBATCH --requeue
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone
exec env SUBMIT=1 INTERVAL="${INTERVAL:-1800}" HOURS="${HOURS:-48}" \
     bash scripts/slurm/autoupdate_tgoal_moving_analysis.sh
