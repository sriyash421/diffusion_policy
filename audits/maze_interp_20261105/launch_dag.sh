#!/bin/bash
# Submit DAgger runs (dag_run.sbatch). Usage: launch_dag.sh <obs: window|ddpm200|ddpm400|ddpm600> <setting: interp30c|11brc> [sbatch placement flags...]
# Env EVAL_ATTEMPTS (default 8) = attempts per pair in evalctx (use <= --rounds). Submits the 4 arms: fbR (reward), fbB (bfs, leak control), nofb (none), noctx (reward, --no-context). Job ids -> logs/dag_jobs.txt
set -euo pipefail
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
obs=$1; setting=$2; shift 2
case $obs in window) OF="--obs window";; ddpm*) OF="--obs ddpm --obs-t ${obs#ddpm}";; *) echo bad obs; exit 1;; esac
case $setting in interp30c) pre=c_dag;; 11brc) pre=br_dag;; *) echo bad setting; exit 1;; esac
for arm in fbR fbB nofb noctx; do
  case $arm in fbR) fb=reward; X="";; fbB) fb=bfs; X="";; nofb) fb=none; X="";; noctx) fb=reward; X="--no-context";; esac
  tag=${pre}_${obs}_${arm}
  j=$(sbatch --parsable --job-name=$tag "$@" $I/dag_run.sbatch $tag $setting $fb $OF $X)
  echo "$(date '+%F %T') $tag $j $setting fb=$fb $OF $X placement=$*" | tee -a $I/logs/dag_jobs.txt
done
