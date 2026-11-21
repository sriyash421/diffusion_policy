from pathlib import Path
import json,subprocess,datetime
OUT=Path(__file__).resolve().parent
if (OUT/'submission.json').exists():print((OUT/'submission.json').read_text());raise SystemExit
assert (OUT/'DATA_READY').exists()
rows=[]
for setting in ['11tr','11bl']:
 for algo in ['bc_unet','dt_bon','dt_st_traj','dt_st_step']:rows.append([setting,algo,0,'clean'])
 for algo in ['dt_st_traj','dt_st_step']:rows.append([setting,algo,0,'noisy'])
for setting in ['11br','11tl']:
 for algo in ['dt_st_traj','dt_st_step']:rows.append([setting,algo,3,'noisy'])
for algo in ['bc_unet','dt_bon','dt_st_traj','dt_st_step']:rows.append(['11br',algo,3,'clean'])
for setting in ['11tl','11br']:
 for algo in ['dt_st_traj','dt_st_step']:rows.append([setting,algo,0,'delta'])
assert len(rows)==24 and len({tuple(r) for r in rows})==24
(OUT/'train.tsv').write_text(''.join(' '.join(map(str,r))+'\n' for r in rows))
# Exercise each architecture/feedback mode plus both new datasets and both delta quadrants.
smoke=[rows[i] for i in [0,1,2,3,4,5,6,20,21,22,23]]
(OUT/'smoke.tsv').write_text(''.join(' '.join(map(str,r))+'\n' for r in smoke))
py='/gscratch/robotics/harine/miniconda3/envs/vae_pushT_l2s/bin/python'
header='''#!/bin/bash
#SBATCH --account=robotics
#SBATCH --partition=ckpt-all
#SBATCH --constraint=a40|a100|l40|l40s|h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
export OMP_NUM_THREADS=4
'''+f'ROOT={OUT}\nPY={py}\n'
train=header+'''read -r setting algo seed variant < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$ROOT/${TASK_LIST:-train.tsv}")
extra=(); if [[ "${TASK_LIST:-train.tsv}" == smoke.tsv ]]; then extra+=(--smoke); fi
"$PY" -u "$ROOT/run_experiment.py" --setting "$setting" --algo "$algo" --seed "$seed" --variant "$variant" "${extra[@]}"
if [[ "${TASK_LIST:-train.tsv}" == smoke.tsv ]]; then
 checkpoint="$ROOT/smoke/$setting/${algo}__${variant}/seed_${seed}/step_0000002.pt"
 if [[ "$variant" == delta ]]; then "$PY" -u "$ROOT/test_delta.py" "$checkpoint"; fi
 "$PY" -u "$ROOT/evaluate.py" --checkpoint "$checkpoint" --smoke
fi
'''
(OUT/'train.sbatch').write_text(train)
# Each evaluation owns one checkpoint, so interruption never blocks unrelated checkpoints.
evalscript=header.replace('--mem=64G','--mem=48G')+'''train_task=$((SLURM_ARRAY_TASK_ID / 10))
step=$(((SLURM_ARRAY_TASK_ID % 10 + 1) * 1000))
read -r setting algo seed variant < <(sed -n "$((train_task+1))p" "$ROOT/train.tsv")
printf -v step_tag '%07d' "$step"
checkpoint="$ROOT/runs/$setting/${algo}__${variant}/seed_${seed}/step_${step_tag}.pt"
if [[ ! -f "$checkpoint" ]]; then echo "Checkpoint missing: $checkpoint"; exit 42; fi
"$PY" -u "$ROOT/evaluate.py" --checkpoint "$checkpoint"
'''
(OUT/'eval.sbatch').write_text(evalscript)
def submit(args):return subprocess.check_output(['sbatch','--parsable']+args,text=True).strip().split(';')[0]
sm=submit([f'--array=0-{len(smoke)-1}%4','--time=02:00:00','--export=ALL,TASK_LIST=smoke.tsv','--job-name=maze_new_smoke',f'--output={OUT}/smoke_%A_%a.log',str(OUT/'train.sbatch')])
tr=submit(['--array=0-23%6',f'--dependency=afterok:{sm}','--export=ALL,TASK_LIST=train.tsv','--job-name=maze_new_train',f'--output={OUT}/train_%A_%a.log',str(OUT/'train.sbatch')])
# Submit each training run's ten checkpoints as a dependent array; global concurrency is managed by watcher.
evals=[]
for i,row in enumerate(rows):
 j=submit([f'--array={i*10}-{i*10+9}%1',f'--dependency=afterok:{tr}_{i}','--job-name=maze_new_eval',f'--output={OUT}/eval_%A_%a.log',str(OUT/'eval.sbatch')]);evals.append(dict(job=j,train_task=i,task=row))
record=dict(submitted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),smoke_job=sm,train_job=tr,eval_jobs=evals,train_tasks=rows,smoke_tasks=smoke,planned_checkpoints=240,partition='ckpt-all',n_values=[1,2,4,8,16,32,64],strategies=['last','max'],splits=['eval','train_heldout','validation'])
(OUT/'submission.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
