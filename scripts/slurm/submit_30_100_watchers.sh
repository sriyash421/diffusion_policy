#!/usr/bin/env bash
# Top up the eval watchers for the 30/100-demo 3x2 (ST-diffusion, ST-gaussian, BC).
#
# Each watcher (eval_watch_pusht_search.sbatch) polls its run's checkpoints/ and sweeps
# n = 1 2 4 8 16 32 64 over the 50 test episodes for every new step_*.ckpt. Its wall clock
# is 12h but the search arms train for days, so ONE watcher per run is not enough: this is
# the idempotent top-up. It skips any run that already has a live `ev_<run_name>` job, so
# running it repeatedly (by hand, from cron, or from /loop) is safe and never doubles up --
# two watchers on one run dir would contend for the same success_curve.json lock.
#
#   bash scripts/slurm/submit_30_100_watchers.sh            # dry run: show what is missing
#   SUBMIT=1 bash scripts/slurm/submit_30_100_watchers.sh   # ...and sbatch those
#
# Dry-run by default, matching fill_eval_gaps.sh: most gaps close on their own while a
# watcher is still alive.
set -uo pipefail
cd "$(dirname "$0")/../.."

ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search
SUBMIT="${SUBMIT:-}"
# --skip-val: the ask is the 50 TEST episodes; the 30 val ones would roughly double the
# per-checkpoint cost for a number nothing here reports.
EVAL_ARGS="${EVAL_ARGS:---skip-val}"
PY_BIN="${PY_BIN:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"

# The `_l2tol1` and `_sw-lin4857` entries are the slot-weighting experiment: two knobs
# varied one at a time off the `_ver-armTn_` k=16 arm directly above them, which is their
# shared control (uniform slot weights, plain L2) on the same manifest, seed and protocol.
# See success_rates_slot_weights_armTn.md.
#
# The three `_nopos_` entries at the bottom are the image-only ablation (scripts/
# run_nopos_30demo.sh): the same three headline arms with `agent_pos` and `feedback` deleted
# from shape_meta.obs, so the policy conditions on pixels alone. Their with-pos controls --
# outer_inner/value_k16_corrupt-False_demos-30_seed-42, offline/value_k1_demos-30_seed-42 and
# unet_bc/unetbc_demos-30_seed-42 -- are already fully scored under t_goal and are
# DELIBERATELY NOT relisted here: re-running them would spend GPU reproducing numbers that
# already exist. See success_rates_no_pos.md.
# TMRL arms PARKED 2026-08-29: the SD-VAE encoder cannot import in robodiff
# (diffusers 0.36 requires accelerate>=0.31, env has 0.13.2), so all three died at
# startup with 0 steps. Re-add the three run dirs here to resume tracking them.
# TMRL arms PARKED 2026-08-29: the SD-VAE encoder cannot import in robodiff
# (diffusers 0.36 requires accelerate>=0.31, the env has 0.13.2), so all three died
# at startup with 0 steps. Re-add the three run dirs to the list to resume tracking.
RUNS="
outer_inner/value_k16_corrupt-False_demos-30_seed-42
outer_inner/value_k16_corrupt-False_demos-100_seed-42
offline/gaussian_k16_corrupt-False_demos-30_seed-42
offline/gaussian_k16_corrupt-False_demos-100_seed-42
offline/value_k1_demos-30_seed-42
offline/value_k1_demos-100_seed-42
unet_bc/unetbc_demos-30_seed-42
offline/value_k1_ver-armTn_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_corrupt-False_demos-30_seed-42
offline/value_k1_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-armTn_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-lin100-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-geo735-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-geo735-l2tol1_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-curr-lin100-l2_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-d_t_goal_sw-curr-lin100-l2tol1_corrupt-False_demos-30_seed-42
offline/value_k1_ver-t_goal_nopos_corrupt-False_demos-30_seed-42
outer_inner/value_k16_ver-t_goal_nopos_corrupt-False_demos-30_seed-42
unet_bc/unetbc_ver-t_goal_nopos_demos-30_seed-42
"

# One squeue call, not one per run: on a busy queue the per-run form is both slow and
# racy (a job can appear between calls and get a second watcher).
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

for rel in $RUNS; do
    name="${rel##*/}"
    dir="$ROOT/$rel"
    if ! [ -d "$dir" ]; then
        printf '%-58s %s\n' "$name" "no run dir yet -- training has not started it"
        continue
    fi
    if grep -qxF "ev_$name" <<<"$LIVE"; then
        printf '%-58s %s\n' "$name" "watcher already live, skipping"
        continue
    fi
    n_ckpt=$(ls "$dir"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    # Nothing to score yet. A watcher started now would just poll an empty checkpoints/
    # until its 12h wall expired, holding a ckpt GPU for no work -- and the search arms
    # are hours from their first step_*.ckpt. The next top-up picks it up instead.
    if [ "$n_ckpt" -eq 0 ]; then
        printf '%-58s %s\n' "$name" "no checkpoints yet -- nothing to evaluate"
        continue
    fi
    # Already fully scored: every checkpoint has every n level. Without this the script
    # resurrects a watcher for a FINISHED arm on every invocation -- it only ever asked
    # "has checkpoints, has no live watcher", which stays true forever once training ends.
    # That watcher then idle-polls a completed run until its wall clock expires.
    if "$PY_BIN" - "$dir" "$n_ckpt" <<'EOF'
import json, os, re, sys
run, n_ckpt = sys.argv[1], int(sys.argv[2])
GRID = {1, 2, 4, 8, 16, 32, 64}
path = os.path.join(run, 'bon_search', 'success_curves.jsonl')
if not os.path.exists(path):
    sys.exit(1)
agg = {}
for line in open(path):
    if not line.strip():
        continue
    r = json.loads(line)
    m = re.search(r'step_(\d+)', r['checkpoint'])
    if m:
        agg.setdefault(int(m.group(1)), set()).update(r['n'])
# complete == one fully-swept row per checkpoint on disk
sys.exit(0 if len(agg) >= n_ckpt and all(GRID <= v for v in agg.values()) else 1)
EOF
    then
        printf '%-58s %s\n' "$name" "fully evaluated (${n_ckpt} checkpoints x 7 n) -- nothing to do"
        continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-58s %s\n' "$name" "WOULD SUBMIT (${n_ckpt} checkpoints present)"
        continue
    fi
    jid=$(sbatch --parsable --job-name="ev_$name" \
          scripts/slurm/eval_watch_pusht_search.sbatch "$dir" $EVAL_ARGS)
    printf '%-58s %s\n' "$name" "submitted job $jid (${n_ckpt} checkpoints present)"
done

[ -z "$SUBMIT" ] && echo && echo "dry run; re-run with SUBMIT=1 to sbatch the above"
exit 0
