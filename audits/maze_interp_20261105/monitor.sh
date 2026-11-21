#!/bin/bash
# Watches ST runs (one line per run in monitor_runs.txt: "tag|train_job|ctrl_job|st_run args"), every 10 min:
#  - a new round_k.pt and no TRAINED yet -> submit partial_eval.sbatch (one at a time per run);
#  - a training job that ended without COMPLETED: no Traceback in its log and < 3 restarts -> resubmit st_run.sbatch with the same
#    args (it resumes from state.pt / skips finished steps) and a new control eval after it; Traceback or OOM -> stop and report;
#  - a control eval that ended without COMPLETED (training done) -> resubmit it once.
# Exits (for a human / Claude to look) when every run has its final files, or on anything it should not retry.
#   bash monitor.sh            (light: sacct / squeue / ls every 10 min; run from a login node in the background)
I=/gscratch/robotics/harine/codex_audits/maze_interp_20261105
cd $I; RUNS=${RUNS_FILE:-$I/monitor_runs.txt}; LOG=$I/logs/monitor.log
R="-A robotics -p gpu-a40 --gres=gpu:a40:1 --constraint=a40 --no-requeue"
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
state() { sacct -j "$1" -X -n -o State%20 2>/dev/null | head -1 | tr -d ' '; }
final_ok() {   # tag -> 0 if all final outputs exist
  local t=$1 d=$I/runs/$1
  [[ -f $d/TRAINED && -f $d/bon_seeds8_control.npz ]] || return 1
  /gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python -c "import numpy as np,sys; sys.exit(0 if 'train_st_success' in np.load('$d/bon_seeds8.npz').files else 1)" 2>/dev/null || return 1
  grep -q '"obs_noise": "\(flat\|ramp\)"' $d/config.json && [[ ! -f $d/bon_seeds8_noisyobs.npz ]] && return 1
  return 0
}
declare -A RESTARTS
while true; do
  all=1; tmp=$(mktemp -p $I/logs)
  while IFS="|" read -r -u 9 tag tj cj args; do
    [[ -z $tag || $tag == \#* ]] && { echo "$tag|$tj|$cj|$args" >> $tmp; continue; }
    d=$I/runs/$tag
    if final_ok $tag; then echo "$tag|$tj|$cj|$args" >> $tmp; continue; fi
    all=0; ts=$(state $tj); cs=$(state $cj)
    # partial evals while training
    if [[ ! -f $d/TRAINED ]]; then
      n=$(ls $d/round_*.pt 2>/dev/null | wc -l); last=$(cat $d/.partial_rounds 2>/dev/null || echo 0)
      if (( n > last )) && ! squeue -u $USER -h -n pe_$tag -o %i | grep -q .; then
        j=$(sbatch --parsable --job-name=pe_$tag $I/partial_eval.sbatch $tag) && echo $n > $d/.partial_rounds && log "$tag: $n rounds saved -> partial eval $j"
      fi
    fi
    # training job ended badly
    case $ts in
      PENDING|RUNNING|REQUEUED|COMPLETED|"") ;;
      *)
        lf=$(ls -t $I/logs/$tag-$tj.out 2>/dev/null | head -1)
        if [[ $ts == OUT_OF_ME* ]] || grep -q Traceback "$lf" 2>/dev/null; then
          log "STOP $tag: train job $tj $ts with a Traceback/OOM (log $lf), not retrying"; rm -f $tmp; exit 2
        fi
        k=${RESTARTS[$tag]:-0}; (( k >= 3 )) && { log "STOP $tag: 3 restarts used"; rm -f $tmp; exit 3; }
        [[ $cs == PENDING ]] && scancel $cj
        nj=$(sbatch --parsable $R --job-name=$tag $I/st_run.sbatch $tag $args) || { log "sbatch failed for $tag, retry next tick"; echo "$tag|$tj|$cj|$args" >> $tmp; continue; }
        nc=$(sbatch --parsable $R --job-name=ctrl_$tag --dependency=afterok:$nj $I/control_eval.sbatch $tag)
        RESTARTS[$tag]=$((k + 1)); log "$tag: train job $tj ended $ts -> restarted as $nj (ctrl $nc), restart $((k + 1))"
        printf "%s %s restarted by monitor.sh: train %s ctrl %s\n" "$(date '+%F %T')" "$tag" "$nj" "$nc" >> $I/logs/submissions.txt
        tj=$nj; cj=$nc ;;
    esac
    # control eval ended badly after a completed training job
    if [[ $ts == COMPLETED ]]; then
      case $cs in
        PENDING|RUNNING|COMPLETED|"") ;;
        *) if [[ -z ${RESTARTS[ctrl_$tag]} ]]; then
             nc=$(sbatch --parsable $R --job-name=ctrl_$tag $I/control_eval.sbatch $tag) && RESTARTS[ctrl_$tag]=1 && log "$tag: control eval $cj ended $cs -> resubmitted $nc" && cj=$nc
           else log "STOP $tag: control eval failed twice ($cs)"; rm -f $tmp; exit 4; fi ;;
      esac
      if [[ $cs == COMPLETED ]] && ! final_ok $tag; then log "STOP $tag: jobs COMPLETED but final files missing"; rm -f $tmp; exit 5; fi
    fi
    echo "$tag|$tj|$cj|$args" >> $tmp
  done 9< $RUNS
  mv $tmp $RUNS
  (( all )) && { log "all runs done"; exit 0; }
  [[ -n $ONCE ]] && exit 0; sleep 600
done
