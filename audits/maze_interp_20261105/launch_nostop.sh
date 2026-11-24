#!/bin/bash
# No-stop retrain (2026-11-23): the same 8-round DAgger runs as c_dag / br_dag / pg_dag (window and DDPM t=400 x fbR / fbB / nofb / noctx,
# plus the clean-image no-context reference) with --no-stop: the stop head is not trained and never ends an attempt, so an attempt ends
# only at the goal radius (success) or the step budget. Tags <pre>ns_<obs>_<arm>, i.e. c_dagns_* / br_dagns_* / pg_dagns_*.
#   launch_nostop.sh <interp30c|11brc|pg> <obs: window|ddpm400|clean> <arm: fbR|fbB|nofb|noctx> [sbatch placement]
# ckpt-* caps the wall time at 9:05: with FOLLOW=1 the job gets a resume job (afternotok) on FOLLOW_P / FOLLOW_A (state.pt per round).
set -euo pipefail
A=/gscratch/robotics/harine/codex_audits; I=$A/maze_interp_20261105
setting=$1; obs=$2; arm=$3; shift 3
case $arm in fbR) fb=reward; X="";; fbB) fb=bfs; X="";; nofb) fb=none; X="";; noctx) fb=reward; X="--no-context";; *) echo bad arm; exit 1;; esac
case $obs in window) O="--obs window";; ddpm400) O="--obs ddpm --obs-t 400";; clean) O="--obs none"; [[ $arm = noctx ]] || { echo clean is noctx only; exit 1; };; *) echo bad obs; exit 1;; esac
case $setting in
  interp30c) tag=c_dagns_${obs}_${arm}; args="$I/dag_run.sbatch $tag $setting $fb $O $X --no-stop";;
  11brc) tag=br_dagns_${obs}_${arm}; args="$I/dag_run.sbatch $tag $setting $fb $O $X --no-stop";;
  pg) tag=pg_dagns_${obs}_${arm}; args="$A/maze_procgen_ast_20261121/pg_run.sbatch $tag $fb $O $X --no-stop";;
  *) echo bad setting; exit 1;;
esac
j=$(sbatch --parsable --job-name=$tag "$@" $args)
echo "$(date '+%F %T') $tag $j $setting fb=$fb placement=$* args=$args" | tee -a $I/logs/dag_jobs.txt
if [[ ${FOLLOW:-0} = 1 ]]; then
  f=$(sbatch --parsable --job-name=$tag -p ${FOLLOW_P:-gpu-a40} -A ${FOLLOW_A:-robotics} --time=24:00:00 \
      --dependency=afternotok:$j --kill-on-invalid-dep=yes $args)
  echo "$(date '+%F %T') $tag $f RESUBMIT dep=$j p=${FOLLOW_P:-gpu-a40}" | tee -a $I/logs/dag_jobs.txt
fi
