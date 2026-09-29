#!/bin/bash
# Extra READ-OUT rules for the two k=16 slot-weighting variant arms, on weights already
# trained. Selection is a pure read-out -- which of the n scored candidates is executed --
# so all three rules run on the same checkpoint without retraining.
#
#   argmax      best-of-n over the verifier. THE PAIRED CONTROL, and it is not optional:
#               the native bon_search/ curve is NOT a valid control for a re-run, because
#               the search is stochastic (every candidate is a fresh DDIM draw). A
#               --selection argmax re-run of the SAME weights disagrees with the native
#               curve at 20 of 24 checkpoints, by up to 22pp -- see the header of
#               submit_selection_sweep.sh. Compare final_pass/index against THIS, never
#               against the native column.
#   final_pass  n-1 candidates are scored and the n'th is generated conditioned on them and
#               executed unsimulated, so the verifier never touches selection.
#   index 8     the 8th candidate in generation order, 1-based, scores ignored entirely.
#               What one slot of the search is worth with no selection on top.
#
# n = 8 and 16 (the powers of two in [MIN_N, MAX_N]): 8 is the smallest width at which an
# 8th candidate exists, and 16 is the width the model is actually TRAINED at (compute_loss
# conditions on max_actions-1 = 15 context entries).
#
# Each rule writes to its own bon_search_sel-<rule>/ so it can never merge with another
# rule's curve or with the native one.
#
# IDEMPOTENT: a (run, step, rule) already on disk is skipped, so re-run it as the last
# checkpoints land and it submits only the new tail.
#
#   bash scripts/slurm/submit_slot_norm_readouts.sh          # dry run
#   SUBMIT=1 bash scripts/slurm/submit_slot_norm_readouts.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search
PY="${PY_BIN:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"
SUBMIT="${SUBMIT:-}"
MIN_N="${MIN_N:-8}"
MAX_N="${MAX_N:-16}"

# All three k=16 arms: the uniform CONTROL plus the two variants. The control was added
# once the read-out table needed to be rectangular -- without it, final_pass has no
# same-protocol uniform baseline to be read against at n=8 and 16.
RUNS="
outer_inner/value_k16_ver-armTn_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42
"

# The UNet BC reference, under the SAME armTn verifier the rest of the doc uses so its rows
# sit in one table with the search arms.
#
# ⚠️ final_pass MEANS SOMETHING DIFFERENT HERE. PushTUNetSearchPolicy.predict_action accepts
# the search context and DISCARDS it -- deliberately; that is what makes best-of-n on this
# arm n i.i.d. draws. So its "final pass" is an UNCONDITIONED sample, not a synthesis over
# the n-1 scored candidates, and the n-1 verifier sims that precede it are thrown away. The
# number should therefore reproduce this arm's own n=1 column to within noise; if it does
# not, something is wrong. Under this rule the verifier never touches the executed action
# at all (final_pass does not select, and the context is ignored), so --verifier-value here
# only decides which directory the rows land in.
BC_RUN="unet_bc/unetbc_demos-30_seed-42"
BC_RULES=(
  "argmax:--selection argmax --verifier-value armTn"
  "final_pass:--selection final_pass --verifier-value armTn"
)

# label:flags. The label is the bon_search_sel-<label> directory eval_search_pusht writes.
RULES=(
  "argmax:--selection argmax"
  "final_pass:--selection final_pass"
  "index8:--selection index --selection-index 8"
)

# Preemptible pools only -- the guaranteed robotics/weirdlab GPUs are what the two training
# jobs are still running on, and parking evals there would starve them.
ACCTS=(ckpt-robotics ckpt-cse)
PARTS=(ckpt         ckpt)

# Jobs already IN THE QUEUE, by name. The on-disk check below cannot see these: a job
# submitted five minutes ago has written nothing yet, so a second invocation (this script
# runs from a 30-min loop) would resubmit the entire batch. That is exactly what happened
# once -- 47 duplicates in one tick. One squeue call, not one per job: on a busy queue the
# per-job form is both slow and racy.
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

n_sub=0; n_skip=0; n_live=0; i=0

# group 2 is BC with its own rule list; the body below is shared via GROUP_RULES
run_group () {   # $1 = run dirs, $2 = rule array name, $3/$4 = min/max n
    local runs="$1"; local -n rules=$2
    local MIN_N="${3:-$MIN_N}" MAX_N="${4:-$MAX_N}"
    for rel in $runs; do
    dir="$ROOT/$rel"
    [ -d "$dir/checkpoints" ] || { echo "no checkpoints dir: $rel"; continue; }
    for ckpt in "$dir"/checkpoints/step_*.ckpt; do
        [ -e "$ckpt" ] || continue
        step=$(basename "$ckpt" .ckpt); step=${step#step_}
        for rule in "${rules[@]}"; do
            label="${rule%%:*}"; flags="${rule#*:}"
            # eval_search_pusht writes to bon_search_sel-<mode>[_ver-<value>]/ (see
            # _bon_subdir). MIRROR that here: reading bon_search_sel-<mode>/ for a group that
            # passes --verifier-value checks a directory that does not exist, so the arm
            # never looks done and the whole group is resubmitted on every tick. The
            # verifier also has to be in the JOB NAME, or a d_t_goal job is dropped as a
            # duplicate of the in-flight armTn job for the same (rule, arm, step).
            ver=$(sed -n 's/.*--verifier-value \([A-Za-z_]*\).*/\1/p' <<<"$flags")
            # The corruption axis is part of the output dir too, and part of the job name --
            # otherwise the corrupted and clean rules for one (arm, step) share a name and
            # the live-queue guard drops the second as a duplicate, exactly as the verifier
            # axis did before it.
            #
            # The spelling MUST be _bon_subdir's (`_obs-corrupt` / `_obs-clean`). It was
            # `_corrupt-on` / `_corrupt-off` here from the day the axis was added until
            # 2026-08-29, so the probe below stat'd a path that never exists and every
            # finished corruption readout looked un-done and was resubmitted on every tick.
            cor=""
            case "$flags" in
                *--no-corrupt-obs-eval*) cor="clean"   ;;
                *--corrupt-obs-eval*)    cor="corrupt" ;;
            esac
            sub="bon_search_sel-${label}${ver:+_ver-$ver}${cor:+_obs-$cor}"
            tag="${label}${ver:+-$ver}${cor:+-$cor}"
            # Present already? One fully-swept row for this step in that rule's jsonl.
            if "$PY" - "$dir/$sub/success_curves.jsonl" "$step" \
                      "$MIN_N" "$MAX_N" <<'EOF'
import json, os, re, sys
path, step, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
want = {n for n in (2 ** k for k in range(31)) if lo <= n <= hi}
if not os.path.exists(path):
    sys.exit(1)
have = set()
for line in open(path):
    if not line.strip():
        continue
    r = json.loads(line)
    m = re.search(r'step_(\d+)', r.get('checkpoint', ''))
    if m and int(m.group(1)) == step:
        have.update(r.get('n') or [])
sys.exit(0 if want <= have else 1)
EOF
            then
                n_skip=$((n_skip + 1)); continue
            fi
            name="ro_${tag}_n${MIN_N}-${MAX_N}_$(basename "$rel")_${step}"
            if grep -qxF "$name" <<<"$LIVE"; then
                n_live=$((n_live + 1)); continue
            fi
            if [ -z "$SUBMIT" ]; then
                echo "WOULD SUBMIT $name"
                n_sub=$((n_sub + 1)); continue
            fi
            a=${ACCTS[$((i % ${#ACCTS[@]}))]}; p=${PARTS[$((i % ${#PARTS[@]}))]}
            i=$((i + 1))
            jid=$(sbatch --parsable --account="$a" --partition="$p" --job-name="$name" \
                  ${NICE:+--nice=$NICE} \
                  scripts/slurm/eval_ckpt_pusht_search.sbatch "$ckpt" \
                  --min-n "$MIN_N" --max-n "$MAX_N" --n-envs 50 --skip-val $flags)
            echo "submitted $jid  $name"
            n_sub=$((n_sub + 1))
        done
    done
    done
}

run_group "$RUNS" RULES
run_group "$BC_RUN" BC_RULES

# n=1 final_pass on all THREE k=16 policies, the control included. At n=1 final_pass takes
# the empty-context branch (n_search = 0), so the executed action is the unconditioned
# sample -- the same action argmax and softmax return there, since with one candidate there
# is nothing to select. That makes this row the "no search at all" anchor for the read-out
# table, and it should agree with each arm's own n=1 column in the n-sweep to within noise.
N1_RUNS="
outer_inner/value_k16_ver-armTn_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42
"
N1_RULES=("final_pass:--selection final_pass")
run_group "$N1_RUNS" N1_RULES 1 1

# BC at n=1, kept separate ONLY because it needs --verifier-value armTn to land in the same
# directory as its n=8/16 rows. Without that it would write to bon_search_sel-final_pass/
# (the checkpoint's own t_goal default) and the doc would render a hole at n=1.
BC_N1_RULES=("final_pass:--selection final_pass --verifier-value armTn")
run_group "$BC_RUN" BC_N1_RULES 1 1

# argmax at n=1, the last cell of the read-out table. At n=1 argmax, softmax and final_pass
# all return the SAME action -- with one candidate there is nothing to select -- so this is
# the same quantity as the final_pass n=1 row above, measured a second time. It fills the
# column and doubles as a noise replicate: the two should agree to within rollout noise, and
# the control's smoke run already matched its native n=1 column exactly (0.180 vs 0.18).
N1_ARGMAX_RULES=("argmax:--selection argmax")
run_group "$N1_RUNS" N1_ARGMAX_RULES 1 1
BC_N1_ARGMAX_RULES=("argmax:--selection argmax --verifier-value armTn")
run_group "$BC_RUN" BC_N1_ARGMAX_RULES 1 1

# ------------------------------------------------------------- TMRL corruption round ----
# Each arm evaluated FOUR ways per checkpoint: {argmax, final_pass} x {corrupted, clean}.
#
# --corrupt-obs-eval reproduces the slot->corruption ladder the loss trained under, so the
# rollout conditional matches the training one. --no-corrupt-obs-eval evaluates every slot
# clean, which is a legitimate arm (how much of the benefit survives without deployment-time
# corruption) but is NOT the conditional that was trained -- so the two must never share a
# curve. _bon_subdir appends _obs-corrupt/_obs-clean and corrupt_obs_eval is in _IDENTITY;
# before that fix these four rows merged n-by-n into two.
#
# TMRL arms PARKED 2026-08-29: the SD-VAE encoder could not import in `robodiff` (diffusers
# 0.36 against accelerate 0.13.2), so all three died at startup with 0 steps. The fix is the
# `vae_pushT_l2s` environment (conda_environment_vae_pusht.yaml) -- and note those three ran
# against the task that still had agent_pos and feedback in the observation, where the ladder
# cannot bite at all, so they are superseded rather than resumable. Re-add run dirs here once
# the replacement arms are training.
TMRL_RUNS="
"
TMRL_RULES=(
  "argmax:--selection argmax --corrupt-obs-eval"
  "argmax:--selection argmax --no-corrupt-obs-eval"
  "final_pass:--selection final_pass --corrupt-obs-eval"
  "final_pass:--selection final_pass --no-corrupt-obs-eval"
)
run_group "$TMRL_RUNS" TMRL_RULES 1 1
run_group "$TMRL_RUNS" TMRL_RULES 8 16

# The _nopos arms, tracked the same way. No corruption axis -- they were trained clean, so
# only the two selection rules apply.
NOPOS_RUNS="
offline/value_k1_ver-t_goal_nopos_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_nopos_corrupt-False_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_nopos_demos-30_seed-42
"
NOPOS_RULES=(
  "argmax:--selection argmax"
  "final_pass:--selection final_pass"
)
run_group "$NOPOS_RUNS" NOPOS_RULES 1 1
run_group "$NOPOS_RUNS" NOPOS_RULES 8 16

# ---------------------------------------------------------------- d_t_goal round -----
# argmax and final_pass at n = 1, 8, 16. No cand-8: this round asks for those two rules.
#
# `d_t_goal` RANKS IDENTICALLY to `t_goal` -- it is that term over its 13.6px spread, and a
# positive divisor is monotone (see value_d_t_goal). So the three REFERENCE arms below are a
# re-measurement of curves that already exist under t_goal, which is exactly what makes them
# a correctness check on the new value: argmax on them must reproduce their bon_search/ rows
# to within rollout noise. It is also why they are LOWER PRIORITY -- see NICE.
DT_RUNS="
outer_inner/value_k16_ver-d_t_goal_sw-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-geo735-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42
"
DT_RULES=(
  "argmax:--selection argmax --verifier-value d_t_goal"
  "final_pass:--selection final_pass --verifier-value d_t_goal"
)
run_group "$DT_RUNS" DT_RULES 1 1
run_group "$DT_RUNS" DT_RULES 8 16

# The three reference arms, re-measured under d_t_goal. NICE drops them behind everything
# else of ours in the queue (higher = lower priority) so 120 jobs re-measuring existing
# weights cannot delay the six new arms' training or their read-outs. Override with NICE=0
# to promote them once the new arms are done.
DT_REF_RUNS="
offline/value_k1_demos-30_seed-42
outer_inner/value_k16_corrupt-False_demos-30_seed-42
unet_bc/unetbc_demos-30_seed-42
"
NICE="${NICE:-10000}" run_group "$DT_REF_RUNS" DT_RULES 1 1
NICE="${NICE:-10000}" run_group "$DT_REF_RUNS" DT_RULES 8 16
echo
echo "submitted/pending: $n_sub   already on disk: $n_skip   already queued: $n_live"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
