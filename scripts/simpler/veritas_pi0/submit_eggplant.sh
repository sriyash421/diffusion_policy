#!/bin/bash
# Submit the step-1 eggplant arms (paper protocol, 240 episodes each):
#   raw_n1      plain open-pi-zero, 1 sample per decision
#   veritas_n5  5 samples, Veritas pixel verifier picks; verified rollouts collected
#
#   SUBMIT=1 bash scripts/simpler/veritas_pi0/submit_eggplant.sh            # both arms
#   SUBMIT=1 ARMS="veritas_n5" bash scripts/simpler/veritas_pi0/submit_eggplant.sh
#
# Picks a non-preemptible robotics/weirdlab GPU partition with scripts/slurm/pick_gpu.sh
# (PARTITION=ckpt to use the preemptible queue once resume is trusted). OUT_ROOT defaults to
# a dated campaign dir on gscratch; an existing OUT_DIR resumes.
set -euo pipefail
cd "$(dirname "$0")/../../.."
DATE="${DATE:-$(date +%Y%m%d)}"
OUT_ROOT="${OUT_ROOT:-/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_${DATE}/eggplant}"
TASK="${TASK:-widowx_put_eggplant_in_basket}"
ARMS="${ARMS:-raw_n1 veritas_n5}"
N_EPISODES="${N_EPISODES:-240}"

if [ "${PARTITION:-}" = ckpt ]; then
  SLURM_OPTS=(--account=ckpt-robotics --partition=ckpt --qos=ckpt-gpu --requeue)
else
  read -r A P < <(NEED_CPUS=6 NEED_MEM_G=48 NEED_GPUS=1 bash scripts/slurm/pick_gpu.sh)
  SLURM_OPTS=(--account="$A" --partition="$P" --qos=normal --no-requeue)
fi
echo "slurm: ${SLURM_OPTS[*]}   out: $OUT_ROOT"

for arm in $ARMS; do
  case "$arm" in
    raw_n1)     VERIFIER=0; N=1 ;;
    veritas_n5) VERIFIER=1; N=5 ;;
    raw_n*)     VERIFIER=0; N="${arm#raw_n}" ;;
    veritas_n*) VERIFIER=1; N="${arm#veritas_n}" ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
  cmd=(sbatch "${SLURM_OPTS[@]}" --job-name="pi0_${arm}" \
       --export=ALL,OUT_DIR="$OUT_ROOT/$arm",TASK="$TASK",VERIFIER="$VERIFIER",N="$N",ARM="$arm",N_EPISODES="$N_EPISODES" \
       scripts/simpler/veritas_pi0/eval_pi0.sbatch)
  echo "${cmd[*]}"
  if [ "${SUBMIT:-0}" = 1 ]; then "${cmd[@]}"; fi
done
[ "${SUBMIT:-0}" = 1 ] || echo "(dry run; SUBMIT=1 to submit)"
