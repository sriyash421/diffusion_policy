#!/bin/bash
#SBATCH --job-name=check_cfg
#SBATCH --partition=ckpt-all
#SBATCH --account=robotics
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Dry-run every PushT config: resolve it fully and print the identity keys + the derived
# run directory and split manifest. Catches interpolation breakage (n_demos -> split_file
# -> run_name) BEFORE eight trainings are launched against it, and makes the run-directory
# collision check a five-second read instead of an archaeology exercise.
set -uo pipefail
cd /mmfs1/home/harine/diffusion_policy_standalone
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
set +u; conda activate robodiff; set -u

# label|config|overrides. The six argmax arms no longer have a config file each -- they are
# train_pusht_diffusion_search plus three overrides (config/archive/README.md), so what has
# to be checked is the OVERRIDE LINE, which is the thing a launcher can now get wrong.
# Commas separate overrides; they are split on whitespace below.
# slot_weight_decay is part of the run IDENTITY (it changes the objective and is a component
# of run_name), so both values of it must appear here -- the uniqueness check below is the
# thing that would have caught the two swd variants of an arm resolving to one directory.
CONFIGS="
value_clean|train_pusht_diffusion_search|search_context=value,arm=value,corrupt_obs=False
value_corrupt|train_pusht_diffusion_search|search_context=value,arm=value,corrupt_obs=True,corrupt_obs_eval=False
subgoal_clean|train_pusht_diffusion_search|search_context=subgoal,arm=subgoal-chosen4value,corrupt_obs=False
subgoal_corrupt|train_pusht_diffusion_search|search_context=subgoal,arm=subgoal-chosen4value,corrupt_obs=True,corrupt_obs_eval=False
subgoalval_clean|train_pusht_diffusion_search|search_context=subgoal_value,arm=subgoal-value,corrupt_obs=False
subgoalval_corrupt|train_pusht_diffusion_search|search_context=subgoal_value,arm=subgoal-value,corrupt_obs=True,corrupt_obs_eval=False
value_clean_30|train_pusht_diffusion_search|search_context=value,arm=value,corrupt_obs=False,n_demos=30
subgoal_clean_30|train_pusht_diffusion_search|search_context=subgoal,arm=subgoal-chosen4value,corrupt_obs=False,n_demos=30
subgoal_only|train_pusht_diffusion_search_subgoal_only|
subgoal_only_cd_k4|train_pusht_diffusion_search_subgoal_only_cd|n_candidates=4
train_pusht_bc|train_pusht_bc|
train_pusht_bc_30|train_pusht_bc|n_demos=30
train_pusht_gaussian_search|train_pusht_gaussian_search|
train_pusht_st_n1|train_pusht_st_n1|
"
fail=0
for spec in $CONFIGS; do
    label="${spec%%|*}"; rest="${spec#*|}"; c="${rest%%|*}"; ovr="${rest##*|}"
    out=$(python train.py --config-name="$c" --cfg job --resolve ${ovr//,/ } 2>&1)
    if [ $? -ne 0 ]; then
        echo "FAIL $label"; echo "$out" | tail -5; fail=1; continue
    fi
    printf '%-52s ' "$label"
    echo "$out" | python -c "
import sys, yaml
c = yaml.safe_load(sys.stdin.read())
d = c['task']['dataset']
print(f\"demos={c['n_demos']:>3} ctx={c['search_context']:<13} corrupt={str(c['corrupt_obs']):<5} \"
      f\"swd={c['slot_weight_decay']:<4} \"
      f\"max_actions={c['policy']['max_actions']:<3} n_search={c['task']['env_runner']['n_search_actions']:<2} \"
      f\"steps={c['training']['max_gradient_steps']}\")
print(f\"{'':52} wandb_name={c['logging']['name']} group={c['logging']['group']} \"
      f\"ckpt_every={c['training']['checkpoint_every']} extra={c['training'].get('checkpoint_steps')}\")
print(f\"{'':52} manifest={d['split_file'].split('/')[-1]} n_train={d['n_train_episodes']} \"
      f\"n_val={d['n_val_episodes']} n_test={d['n_test_episodes']}\")
print(f\"{'':52} run_name={c['run_name']}\")
" || { echo "  (parse failed)"; fail=1; }
done
echo
echo "=== run-directory uniqueness ==="
# Two r8 rows share an arm with a 100-demo row on purpose, so this must compare the
# RESOLVED run_name (which carries demos-29-r8), not the label.
for spec in $CONFIGS; do
    rest="${spec#*|}"; c="${rest%%|*}"; ovr="${rest##*|}"
    python train.py --config-name="$c" --cfg job --resolve ${ovr//,/ } 2>/dev/null \
      | python -c "import sys,yaml;c=yaml.safe_load(sys.stdin.read());print(c['run_name'])"
done | sort | uniq -c | awk '$1>1{print "COLLISION:",$0; f=1} END{if(!f) print "OK: all run_names distinct"}'
exit $fail
