#!/bin/bash
# 2026-11-24: 8 DAgger rounds with at most 2 earlier attempts in context (--max-ctx 2), DDPM t=400, stop head on:
#   pg      new mazes:                         pg_dag_ddpm400r8c2_{fbR,fbB,nofb}
#   d160    random split, 160 of the 642 train pairs (--n-demos 160): c_dag_ddpm400d160r8c2_{fbR,fbB,nofb}, c_dag_ddpm400d160_noctx,
#           c_dag_cleand160_noctx (clean image, no context)
# Each on ckpt-all with an afternotok resume job on robotics gpu-a100 (state.pt per round). launch_c2.sh <pg|d160>
set -euo pipefail
A=/gscratch/robotics/harine/codex_audits; I=$A/maze_interp_20261105
sub() {  # tag script args...
  local tag=$1; shift
  j=$(sbatch --parsable --job-name=$tag -p ckpt-all -A robotics --constraint='a40|l40|l40s|a100' --time=9:00:00 "$@")
  f=$(sbatch --parsable --job-name=$tag -p gpu-a100 -A robotics --constraint=a100 --time=24:00:00 --dependency=afternotok:$j --kill-on-invalid-dep=yes "$@")
  echo "$(date '+%F %T') $tag $j resume=$f args=$*" | tee -a $I/logs/dag_jobs.txt
}
case $1 in
  pg) for a in fbR:reward fbB:bfs nofb:none; do
        sub pg_dag_ddpm400r8c2_${a%%:*} $A/maze_procgen_ast_20261121/pg_run.sbatch pg_dag_ddpm400r8c2_${a%%:*} ${a#*:} --obs ddpm --obs-t 400 --max-ctx 2; done;;
  d160) export EVAL_ATTEMPTS=3
        for a in fbR:reward fbB:bfs nofb:none; do
          sub c_dag_ddpm400d160r8c2_${a%%:*} --export=ALL $I/dag_run.sbatch c_dag_ddpm400d160r8c2_${a%%:*} interp30c ${a#*:} --obs ddpm --obs-t 400 --max-ctx 2 --n-demos 160; done
        sub c_dag_ddpm400d160_noctx --export=ALL $I/dag_run.sbatch c_dag_ddpm400d160_noctx interp30c reward --obs ddpm --obs-t 400 --no-context --n-demos 160
        sub c_dag_cleand160_noctx --export=ALL $I/dag_run.sbatch c_dag_cleand160_noctx interp30c reward --obs none --no-context --n-demos 160;;
  *) echo 'usage: launch_c2.sh <pg|d160>'; exit 1;;
esac
