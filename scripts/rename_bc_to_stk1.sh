#!/usr/bin/env bash
# Rename the ST k=1 run directories off the misleading `bc_` prefix.
#
# "BC" means the diffusion UNet and nothing else -- train_pusht_diffusion_search.yaml,
# scripts/st_k1_smoke.py and scripts/render_search_videos.py all say so. The width-1
# TRANSFORMER is ST k=1. But the k=1 directories on disk are still `offline/bc_demos-*`,
# left over from before that convention, so the one place the name actually matters
# contradicts it -- and the doc builders each special-case `bc_` to relabel it back.
#
# WHAT IT DOES. `bc_<rest>` -> `value_k1_<rest>`, replacing ONLY the family token. The rest
# of the name is left byte-identical rather than reconstructed into today's spelling: these
# are pre-`ver-`, pre-`enc-`, ResNet-era runs, and guessing what run_name would have
# produced at the time is how a directory ends up claiming a configuration it never had.
#
# A back-symlink is left at the old path, exactly as scripts/rename_ctx_dirs.sh did for the
# ctx-* rename, so absolute paths held by running jobs and by already-written JSON keep
# resolving. Drop the symlinks once nothing reads through them.
#
# GATED on squeue: any directory with a live tr_/ev_/ro_ job naming it is SKIPPED and
# reported, not moved.
#
#   bash scripts/rename_bc_to_stk1.sh            # dry run
#   SUBMIT=1 bash scripts/rename_bc_to_stk1.sh   # ...and actually move them
set -uo pipefail
cd "$(dirname "$0")/.."

SUBMIT="${SUBMIT:-}"
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sort -u)

n=0; skipped=0
while IFS= read -r -d '' dir; do
    base=$(basename "$dir")
    parent=$(dirname "$dir")
    new="value_k1_${base#bc_}"
    # A symlink is what a previous run of this script left behind; do not re-rename it.
    if [ -L "$dir" ]; then continue; fi
    if [ -e "$parent/$new" ]; then
        printf '%-52s %s\n' "$base" "target $new already exists, skipping"; skipped=$((skipped+1)); continue
    fi
    # Any live job naming this run, by any of the three job-name prefixes the launchers use.
    if grep -qE "^(tr|ev|ro)_${base}(_|$)" <<<"$LIVE"; then
        printf '%-52s %s\n' "$base" "LIVE JOB, skipping"; skipped=$((skipped+1)); continue
    fi
    if [ -z "$SUBMIT" ]; then
        printf '%-52s WOULD RENAME -> %s\n' "$base" "$new"; n=$((n+1)); continue
    fi
    mv "$dir" "$parent/$new" && ln -s "$new" "$dir" \
        && printf '%-52s renamed -> %s (back-symlink left)\n' "$base" "$new" \
        && n=$((n+1))
done < <(find "$ROOT" -mindepth 3 -maxdepth 3 -name 'bc_*' -print0 2>/dev/null)

echo
echo "renamed: $n   skipped: $skipped"
[ -z "$SUBMIT" ] && echo "dry run; re-run with SUBMIT=1 to move the above"
exit 0
