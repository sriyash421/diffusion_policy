#!/bin/bash
# 3-round DAgger runs (2026-11-23): at most 2 earlier attempts in context (3 episodes in the stream), DDPM t=400, all 4 arms,
# random split (interp30c) and held-out quadrant (11brc). Eval: 3 attempts per pair.
#   launch_r3.sh <sched: r3 (3 x 2400 = 7200 updates, same total as the 8-round runs) | r3x3 (3 x 7200 = 21600, 3x longer)> <setting> [sbatch placement]
# ckpt-all caps the wall time at 9:05: with FOLLOW=1 each job gets a resume job (afternotok) on FOLLOW_P / FOLLOW_A.
set -euo pipefail
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
sched=$1; setting=$2; shift 2
case $sched in r3) IT=80;; r3x3) IT=240;; *) echo bad sched; exit 1;; esac
case $setting in interp30c) pre=c_dag;; 11brc) pre=br_dag;; *) echo bad setting; exit 1;; esac
for arm in fbR fbB nofb noctx; do
  case $arm in fbR) fb=reward; X="";; fbB) fb=bfs; X="";; nofb) fb=none; X="";; noctx) fb=reward; X="--no-context";; esac
  tag=${pre}_ddpm400${sched}_${arm}
  args="$I/dag_run.sbatch $tag $setting $fb --obs ddpm --obs-t 400 --rounds 3 --iters $IT $X"
  j=$(sbatch --parsable --export=ALL,EVAL_ATTEMPTS=3 --job-name=$tag "$@" $args)
  echo "$(date '+%F %T') $tag $j $setting fb=$fb sched=$sched placement=$* args=$args" | tee -a $I/logs/dag_jobs.txt
  if [[ ${FOLLOW:-0} = 1 ]]; then
    f=$(sbatch --parsable --export=ALL,EVAL_ATTEMPTS=3 --job-name=$tag -p ${FOLLOW_P:-gpu-a40} -A robotics --time=24:00:00 \
        --dependency=afternotok:$j --kill-on-invalid-dep=yes $args)
    echo "$(date '+%F %T') $tag $f RESUBMIT dep=$j p=${FOLLOW_P:-gpu-a40}" | tee -a $I/logs/dag_jobs.txt
  fi
done
