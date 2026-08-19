#!/usr/bin/env bash
# Pull the analysis material behind SUCCESS_RATES.md off Hyak -- see aug10_results2copy.md.
# Everything except checkpoints/, ~2.3 GB. Re-run with STEPS=(...) to add specific weights.
set -euo pipefail

REMOTE=${REMOTE:-hyak}
SRC=${SRC:-/gscratch/robotics/harine/diffusion_policy_outputs}
DST=${DST:-/home/harine/diffusion_policy/hyak_results}
CKPT=${CKPT:-none}                       # none | all | a step list, e.g. CKPT="2000 8000"

# The generator reads $DP_OUTPUT_ROOT/pusht_search/pusht_image_search/offline, so the copy
# has to keep that prefix -- not the flattened offline/ the doc's recipe writes.
OFF_REL=pusht_search/pusht_image_search/offline

# -a without -L: the 12 ctx-* entries are back-symlinks, and following them copies every
# run twice. They are excluded outright since nothing off-cluster resolves them.
# --partial-dir keeps interrupted transfers out of the way and resumes them -- worth having
# when CKPT=all moves 213 GB and a dropped master would otherwise restart a 260 MB file.
RS=(rsync -a --info=progress2 --partial-dir=.rsync-partial)
[ "$CKPT" = all ] || RS+=(--exclude='checkpoints/')

mkdir -p "$DST/$OFF_REL"
"${RS[@]}" --exclude='/ctx-*' "$REMOTE:$SRC/$OFF_REL/" "$DST/$OFF_REL/"
"${RS[@]}" "$REMOTE:$SRC/runs" "$DST/"
rsync -a --info=progress2 --partial-dir=.rsync-partial "$REMOTE:$SRC/candidate_scores" "$DST/"

# Named steps only: CKPT=all already came across with the runs above.
case "$CKPT" in
    all|none) steps=() ;;
    *)        read -r -a steps <<<"$CKPT" ;;
esac
for s in "${steps[@]}"; do
    printf 'checkpoints: step %s\n' "$s"
    ssh "$REMOTE" "ls -d $SRC/$OFF_REL/*/ | grep -v /ctx-" | while read -r d; do
        name=$(basename "$d")
        ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "${d%/}" "$s")
        mkdir -p "$DST/$OFF_REL/$name/checkpoints"
        rsync -a --partial --ignore-missing-args \
            "$REMOTE:$ckpt" "$DST/$OFF_REL/$name/checkpoints/" || true
    done
done

printf '\ndone -> %s\n' "$DST"
du -sh "$DST"
