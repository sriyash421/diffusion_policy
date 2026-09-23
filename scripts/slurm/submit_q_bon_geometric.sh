#!/usr/bin/env bash
# Does the learned Q beat t_goal at best-of-N, on the GEOMETRIC splits' UNet BC arms?
#
#   bash scripts/slurm/submit_q_bon_geometric.sh           # dry run
#   SUBMIT=1 bash scripts/slurm/submit_q_bon_geometric.sh  # ...and sbatch it
#
# BC FIRST, AND BC IS THE CLEAN READ. `install_q_ranker` replaces only the ranking scalar and
# leaves the search context alone, because a t_goal-TRAINED ST arm handed a Q-shaped context is
# off its training distribution -- which would confound "Q ranks better" with "ST was given an
# input it has never seen". BC ignores the context entirely, so for BC there is no confound to
# have, and `--skip-context-sim` additionally drops the sim call as pure overhead.
#
# The ST arms of this generation (scripts/run_q_geometric.sh) do not need that patch at all:
# they TRAIN under the Q, carry `verifier_tag` in their own config, and rebuild the same
# verifier and the same context at eval. Sweep them with STEPS=100000 and ARMS pointed at the
# ver-q_sac_all dirs, WITHOUT --skip-context-sim.
#
# ⚠️ NOTHING HERE IS HELD OUT FROM q_sac_all. sac_keypoint ran with `demo_episodes: all`, so
# all 206 episodes seeded its buffer -- including all 50 test episodes of both splits.
# `assert_held_out` refuses this by design; --allow-contaminated is how the number gets taken
# anyway, and it is RECORDED in bon_curves.json rather than merely permitted, so a
# contaminated curve stays distinguishable from a clean one on a shared plot.
#
# TWO STEPS. 100k is the end of training; 30k is what scripts/slurm/q_eval_arms.sh already
# targets, so these numbers stay commensurable with anything run there. Neither is nominated
# as best -- both are reported.
set -uo pipefail

source "$(dirname "$0")/q_arms.sh"

B_GEOM="${B_GEOM:-ckpts/pusht_search/pusht_image_search_imgonly}"
OUT="${OUT:-/gscratch/robotics/harine/value_arms/bon_q_geom}"
RANKERS="${RANKERS:-t_goal,q_sac_all}"
MAX_N="${MAX_N:-16}"
SPLIT="${SPLIT:-test}"
STEPS="${STEPS:-100000 30000}"

# label | run dir | extra flags.  BOTH BC arms already exist at 100k; nothing is trained here.
# blq137 = pusht_blockquad_bottomleft_train137 (137 train / 19 val / 50 test);
# brd100  = pusht_blockborder_train100_core50   (100 train /  0 val / 50 test, the shared core).
GEOM_ARMS="${GEOM_ARMS:-
unetbc_blq137 | $B_GEOM/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-137_split-blq_seed-42  | --skip-context-sim
unetbc_brd100 | $B_GEOM/unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-100_split-brd100_seed-42 | --skip-context-sim
}"

while IFS='|' read -r label run flags; do
    label="$(echo "$label" | xargs)"; run="$(echo "$run" | xargs)"; flags="$(echo "$flags" | xargs)"
    [ -z "$label" ] && continue
    for step in $STEPS; do
        ck=$(printf '%s/checkpoints/step_%07d.ckpt' "$run" "$step")
        [ -f "$ROOT/$ck" ] || { echo "[SKIP $label @${step}] no $ck"; continue; }
        # --rankers carries a registry NAME, so no --q: the name resolves its own checkpoint
        # and lands in bon_curves.json's q_checkpoints, which is what lets a curve be
        # attributed to a Q later without trusting the output directory's name.
        cmd="python -u sac/eval.py bon-sweep -c $ck --rankers $RANKERS \
--max-n $MAX_N --split $SPLIT --n-envs ${CPUS:-8} --allow-contaminated $flags \
-o $OUT/${label}_step$((step/1000))k"
        submit_or_echo "qbong_${label}_$((step/1000))k" "$cmd"
    done
done <<< "$GEOM_ARMS"

[ "${SUBMIT:-0}" = "1" ] || echo $'\nDry run. SUBMIT=1 to sbatch.'
