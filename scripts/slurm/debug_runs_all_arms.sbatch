#!/bin/bash
#SBATCH --job-name=debug_all_arms
#SBATCH --partition=ckpt
#SBATCH --account=robotics
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/gscratch/robotics/harine/slurm_logs/%x-%j.out
# Short `training.debug=True` run of ALL THREE training arms, to prove each still starts,
# steps, evaluates, checkpoints and exits 0 after the workspace changes -- and that each
# writes the run.json / splits.json provenance files.
#
# Runs on a compute node, NOT the login node: home there is one shared ~10G cgroup and these
# get OOM-killed (exit 137) whenever an editor or agent is also resident.
#
# DP_OUTPUT_ROOT IS REDIRECTED, and that is the whole reason this is a script rather than
# three commands. hydra.run.dir is derived from the run IDENTITY, so a debug run of
# train_pusht_diffusion_search resolves to the SAME directory a real run of that arm will
# use -- and since `training.resume: True`, the debug run's latest.ckpt (6 gradient steps,
# random weights) would be found and resumed by the real launch. Redirecting the root keeps
# throwaway runs out of the real tree entirely.
set -uo pipefail

export DP_OUTPUT_ROOT=/gscratch/robotics/harine/_debug_runs
rm -rf "$DP_OUTPUT_ROOT"
mkdir -p "$DP_OUTPUT_ROOT"

set +u
source /gscratch/robotics/harine/miniconda3/etc/profile.d/conda.sh
conda activate robodiff
set -u
cd /mmfs1/home/harine/diffusion_policy_standalone

# label|config|overrides   -- one per arm/workspace
ARMS="
outer_inner|train_pusht_diffusion_search|search_context=value arm=value slot_weight_decay=0.9
offline|train_pusht_st_n1|
unet|train_pusht_unet_bc|
"

fail=0
for spec in $(echo "$ARMS" | tr ' ' '\a'); do
    spec="${spec//$'\a'/ }"
    [ -z "${spec// /}" ] && continue
    label="${spec%%|*}"; rest="${spec#*|}"; cfg="${rest%%|*}"; ovr="${rest##*|}"
    echo "================================================================"
    echo "=== $label ($cfg) $ovr"
    echo "================================================================"
    # shellcheck disable=SC2086
    if python train.py --config-name="$cfg" training.debug=True training.device=cuda:0 $ovr; then
        echo "--- $label: EXIT 0"
    else
        echo "--- $label: FAILED (exit $?)"
        fail=1
    fi
done

echo
echo "=== provenance files written by each debug run ==="
# Both must exist for every arm. splits.json is what eval_search_pusht cross-checks a
# checkpoint's partition against, and it only performs that check `if recorded.is_file()` --
# so a missing file SKIPS the guard silently rather than failing.
for d in $(find "$DP_OUTPUT_ROOT" -name checkpoints -type d | xargs -r -n1 dirname); do
    printf '%-72s run.json=%s splits.json=%s ckpts=%s\n' \
        "${d#$DP_OUTPUT_ROOT/}" \
        "$([ -f "$d/run.json" ] && echo yes || echo NO)" \
        "$([ -f "$d/splits.json" ] && echo yes || echo NO)" \
        "$(ls "$d"/checkpoints/*.ckpt 2>/dev/null | wc -l)"
    [ -f "$d/run.json" ] || fail=1
    [ -f "$d/splits.json" ] || fail=1
    # topk was removed from the PushT configs; nothing should write epoch=*.ckpt any more
    if ls "$d"/checkpoints/epoch=*.ckpt >/dev/null 2>&1; then
        echo "  UNEXPECTED: topk checkpoints present in $d"; fail=1
    fi
done

echo
echo "=== train splits identical across arms? (all use n_demos-appropriate manifests) ==="
find "$DP_OUTPUT_ROOT" -name splits.json -print0 \
  | xargs -0 -r -n1 sh -c 'printf "%s  " "$1"; python -c "
import json,sys,hashlib
d=json.load(open(sys.argv[1]))
tr=d.get(\"train\") or []
print(f\"n_train={len(tr)} sha={hashlib.md5(str(sorted(tr)).encode()).hexdigest()[:10]}\")
" "$1"' _

echo
[ $fail -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $fail
