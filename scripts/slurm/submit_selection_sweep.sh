#!/bin/bash
# argmax vs softmax SELECTION, same weights, across training, on the 50 test episodes.
#
# Selection is a pure readout rule -- which of the n scored candidates gets executed -- so
# both modes can be run on an already-trained checkpoint. That makes this a controlled
# comparison: the only thing that differs between the two runs of a given checkpoint is how
# the verifier's ranking is consumed. `final_pass` cannot isolate that axis, because it also
# changes WHO produces the action (the model synthesizes a fresh one instead of picking).
#
# Softmax samples from softmax(z / T) where z is the score standardized across the n
# candidates; T=1. Standardizing is what makes one T mean the same thing for every arm --
# the raw score is -mean keypoint distance in pixels and its spread varies by arm, by
# checkpoint and with n. T->0 is exactly argmax; T->inf is a uniform pick.
#
# --skip-val: the 50 test episodes only. val costs the SAME as test despite being 10-30
# episodes (they pad out to n_envs=50), so dropping it halves every job. Safe here because
# nothing is being SELECTED -- the checkpoints are named up front. Do not use it for a
# sweep whose curves you intend to choose a checkpoint from: with no val curve, that choice
# is made on test.
#
# Results go to bon_search_sel-<mode>/ per run, never bon_search/, so they cannot merge
# with the native-mode curves already on disk.
#
# CHECKPOINT GRID: every multiple of STRIDE (default 10000) that a run actually saved.
#
# 10000 rather than 5000 because it is the one stride every cadence hits EXACTLY, so the
# x-axis is the same on every arm. The cadences are 1000 (the 20k argmax arms), 2000
# (subgoal-only, r8, bc-29) and 10000 (BC@100): at a 5000 stride the 2000-cadence runs have
# nothing at 5k/15k/25k, so half the grid would silently fall back to a neighbouring step
# and the arms would no longer be sampled at comparable steps.
#
# IDEMPOTENT: a (run, step, mode) already present in bon_search_sel-<mode>/ is skipped, so
# this can be re-run as training advances and it only submits the new tail. That also makes
# it safe to run alongside the earlier partial sweep still sitting in the queue.
#
#   bash scripts/slurm/submit_selection_sweep.sh
#   DRY=1 bash scripts/slurm/submit_selection_sweep.sh
#   STRIDE=5000 bash scripts/slurm/submit_selection_sweep.sh    # denser, uneven across arms
#   MODES=softmax bash scripts/slurm/submit_selection_sweep.sh  # skip the paired argmax
#   MODES=argmax  bash scripts/slurm/submit_selection_sweep.sh  # the paired twin, after
#
# The twin is NOT optional if the softmax numbers are going to be compared to anything.
# The native bon_search curve is argmax for the argmax arms, but it is not a usable
# control: measured at the 24 checkpoints carrying both, the native curve and a
# `--selection argmax` re-run of the SAME weights disagree at 20 of them, by up to 22pp at
# n=8. Nothing is wrong with either -- the search is stochastic (every candidate is a fresh
# DDIM draw) and the native run consumed RNG on the val split first, so the two are
# independent samples at roughly the binomial SE. Which is exactly why softmax must be read
# against an argmax run under the same protocol, not against the native column.
#
#   MODES=final_pass MIN_N=16 MAX_N=16 ONLY_K=16 ...   # see ONLY_K below
set -uo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
MAX_N="${MAX_N:-64}"
MIN_N="${MIN_N:-1}"
TEMP="${TEMP:-1.0}"
DRY="${DRY:-}"
STRIDE="${STRIDE:-10000}"
MODES="${MODES:-argmax softmax}"

# Restrict to runs of one search width. Empty = every run.
#
# The motivating case is `final_pass` at n=16 on the K=16 arms:
#   MODES=final_pass MIN_N=16 MAX_N=16 ONLY_K=16 bash scripts/slurm/submit_selection_sweep.sh
# That is the one width the model is actually TRAINED at -- compute_loss conditions on
# max_actions-1 = 15 context entries -- and README_pusht.md "Known limitations" flags it as the width never
# evaluated (the in-training rollout uses 8, the eval sweep sweeps 1..64). Running it under
# final_pass asks that arm's whole question: does the model's own synthesis, conditioned on
# all 15 scored candidates, beat the oracle argmax it replaces?
#
# K is read from the run's own saved config, not guessed from the name: `n_candidates` is
# absent in the pre-Round-8 configs, where max_actions is the only place the width appears.
ONLY_K="${ONLY_K:-}"

# BC is skipped unless BC_GRID=1, which puts all three BC runs on the same stride as the
# search arms. That is +96 jobs at STRIDE=10000 (BC@100 alone saves 30 checkpoints, all of
# them multiples of 10000), which is why it is off by default.
#
# BC@100 used to fall back to a single val-selected step instead of being skipped, read out
# of best.json. That file is gone -- it recorded one selection rule as though it were a
# result -- so there is no fallback step source and BC is now all-or-nothing. BC also has no
# selection rule of its own to sweep: with max_actions=1 the n candidates are i.i.d. samples,
# so argmax vs softmax is a readout over unstructured scores, not over a learned search.
BC_GRID="${BC_GRID:-}"

# Rotate across three pools so a 100+ job sweep does not all queue behind one. Never the
# *-weirdlab L40/L40S accounts: those GPUs are what the trainers run on, and parking an
# eval sweep there would starve them.
#
# The first two are the preemptible checkpoint partition under the two associations we
# hold; the third is CSE's guaranteed L40S. Preemption is safe here -- the eval sbatch sets
# --requeue and eval_search_pusht merges under a lock after every n, so a requeued job
# keeps the levels it already wrote.
#
# ACCOUNT and PARTITION are different namespaces: the account is `ckpt-robotics`
# (`sacctmgr show assoc` lists it; bare `robotics` happens to resolve to it, but naming it
# exactly is what the running ev_* jobs report) and `gpu-l40s-cse`, while the partitions
# are `ckpt` and `gpu-l40s`. Passing one as the other fails with "invalid partition".
ACCTS=(ckpt-robotics ckpt-cse gpu-l40s-cse)
PARTS=(ckpt          ckpt     gpu-l40s)

submitted=0
shopt -s nullglob
for d in "$ROOT"/*_seed-42; do
    run=$(basename "$d")
    [ -L "$d" ] && continue
    case "$run" in ctx-*) continue;; esac
    [ -d "$d/checkpoints" ] || continue
    # BC takes the grid only under BC_GRID, and is skipped entirely otherwise -- see above.
    case "$run" in
        bc_*) [ -n "$BC_GRID" ] || continue;;
    esac
    if [ -n "$ONLY_K" ]; then
        runk=$("$PY" - "$d" <<'EOF'
import pathlib, sys, yaml
c = yaml.safe_load(open(pathlib.Path(sys.argv[1]) / '.hydra' / 'config.yaml'))
# max_actions is `${n_candidates}` in the post-Round-8 configs and a literal in the older
# ones, so try both and take whichever resolves to a number.
for v in ((c.get('policy') or {}).get('max_actions'), c.get('n_candidates')):
    if isinstance(v, int):
        print(v); break
    if isinstance(v, str) and v.strip().isdigit():
        print(int(v)); break
else:
    print('')
EOF
)
        [ "$runk" = "$ONLY_K" ] || { echo "  skip K=$runk $run"; continue; }
    fi
    steps=$("$PY" - "$d" "$STRIDE" <<'EOF'
import pathlib, re, sys
d = pathlib.Path(sys.argv[1])
stride = int(sys.argv[2])
have = sorted(int(re.search(r'step_(\d+)\.ckpt', f.name).group(1))
              for f in (d / 'checkpoints').glob('step_*.ckpt'))
# EXACT multiples only. Snapping to the nearest neighbour instead would put different arms
# on different steps while labelling them the same grid point.
print(' '.join(str(s) for s in have if s % stride == 0))
EOF
)
    [ -n "$steps" ] || { echo "  no checkpoints on the grid for $run"; continue; }

    for step in $steps; do
        ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "$d" "$step")
        [ -f "$ckpt" ] || continue
        for sel in $MODES; do
            # skip what is already on disk: re-running a (run, step, mode) would re-append
            # a duplicate row to the same success_curves.jsonl, not overwrite it.
            # "Already done" means the row COVERS every n this invocation asks for, not
            # merely that the step appears -- otherwise raising MAX_N (or asking for a
            # single high level via MIN_N) would silently skip every step that already had
            # a narrower curve, and the sweep would submit nothing while looking complete.
            done_already=$("$PY" - "$d/bon_search_sel-$sel/success_curves.jsonl" \
                               "$step" "$MIN_N" "$MAX_N" <<'EOF'
import json, sys
path, step, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
want = {n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024) if lo <= n <= hi}
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    have = {n for r in rows if r['step'] == step for n in r['n']}
    print('yes' if want and want <= have else 'no')
except Exception:
    print('no')
EOF
)
            if [ "$done_already" = yes ]; then
                echo "  have $sel step=$step $run"
                continue
            fi
            tag="sel_${sel:0:3}_$(echo "$run" | sed 's/_demos-100//;s/_seed-42//;s/subgoal-/sg-/')_$step"
            if [ -n "$DRY" ]; then
                echo "  would submit  $sel  step=$step  $run"
                submitted=$((submitted+1)); continue
            fi
            k=$((submitted % ${#ACCTS[@]}))     # round-robin the pools; see ACCTS above
            acct=${ACCTS[$k]}; part=${PARTS[$k]}
            # A single-level request (MIN_N == MAX_N) is a fraction of a full sweep, whose
            # cost is ~2x its top level, so ask for less walltime -- shorter requests
            # schedule sooner on the preemptible ckpt partition.
            [ "$MIN_N" = "$MAX_N" ] && tlim=2:00:00 || tlim=8:00:00
            jid=$(sbatch --parsable --account="$acct" --partition="$part" \
                    --time="$tlim" --job-name="${tag:0:60}" \
                    "$HERE/eval_ckpt_pusht_search.sbatch" "$ckpt" \
                    --min-n "$MIN_N" --max-n "$MAX_N" --skip-val \
                    --selection "$sel" --selection-temperature "$TEMP")
            echo "  $jid  $sel  step=$step  $run"
            submitted=$((submitted+1))
        done
    done
done
echo "${DRY:+DRY-RUN: }$submitted job(s)"
