#!/bin/bash
# Regenerate SUCCESS_RATES.md from whatever evals have landed, then commit and push.
#
# Scheduled with `at` on the LOGIN NODE, not as a cloud routine and not as a SLURM job:
#   * the eval results live under /gscratch, which only the cluster can see;
#   * `git push` needs the SSH key on the login node (verified to work with BatchMode=yes,
#     so no agent prompt at run time).
#
#   at 08:30 -f scripts/slurm/commit_eval_results.sh
#   atq / atrm <id>    to inspect or cancel
#
# Idempotent and safe to re-run: if the regenerated doc is unchanged and nothing else is
# dirty, it commits nothing and exits 0.
set -uo pipefail

REPO=/mmfs1/home/harine/diffusion_policy_standalone
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
LOG=/gscratch/robotics/harine/slurm_logs/commit_eval_results.log
cd "$REPO" || exit 1

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
say "start"

# THE GUARD, and the reason this script exists rather than a one-line `git commit -am`.
# build_success_rates_doc.py reads DP_OUTPUT_ROOT (default /gscratch/...). If that tree is
# missing or unreadable -- a bad mount, a wrong env, a cloud checkout -- every rows_for()
# returns [] and the script still WRITES the file, with every table empty and a cheerful
# "0 checkpoints" on stdout. Committing that would silently delete 400+ KB of results that
# exist nowhere else in git. So: count checkpoints before and after, and refuse to commit a
# doc that lost any.
count_of() { sed -n 's/.*(\([0-9]*\) bytes, \([0-9]*\) checkpoints.*/\2/p' <<<"$1"; }

before=$(grep -c '^| ' SUCCESS_RATES.md 2>/dev/null || echo 0)
out=$("$PY" scripts/build_success_rates_doc.py 2>&1 | tail -1)
say "generator: $out"
after_ck=$(count_of "$out")
after=$(grep -c '^| ' SUCCESS_RATES.md 2>/dev/null || echo 0)

if [ -z "$after_ck" ] || [ "$after_ck" -lt 1 ]; then
    say "ABORT: generator reported no checkpoints -- results tree unreadable? Nothing committed."
    git checkout -- SUCCESS_RATES.md 2>/dev/null
    exit 1
fi
if [ "$after" -lt "$before" ]; then
    say "ABORT: table rows fell $before -> $after. Reverting the regenerated doc, nothing committed."
    git checkout -- SUCCESS_RATES.md 2>/dev/null
    exit 1
fi
say "rows $before -> $after, $after_ck checkpoints"

if [ -z "$(git status --porcelain)" ]; then
    say "clean tree, nothing to commit"
    exit 0
fi
git status --short >> "$LOG"

git add -A
git commit -q -F - <<EOF
Refresh eval results: $after_ck checkpoints

Regenerated SUCCESS_RATES.md from the eval output on gscratch, plus any working-tree
changes outstanding at the time of the run. Committed by
scripts/slurm/commit_eval_results.sh, scheduled with \`at\`.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
EOF
say "committed: $(git log --oneline -1)"

if GIT_SSH_COMMAND="ssh -o BatchMode=yes" git push -q origin HEAD 2>>"$LOG"; then
    say "pushed to $(git rev-parse --abbrev-ref '@{u}' 2>/dev/null || echo origin/HEAD)"
else
    say "PUSH FAILED -- the commit is local, push by hand"
    exit 1
fi
say "done"
