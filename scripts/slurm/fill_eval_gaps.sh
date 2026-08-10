#!/bin/bash
# Find every checkpoint/n that has no 50-episode best-of-N eval, and optionally submit it.
#
# This is the executable half of aug9_analysis.md section 2: the doc is a snapshot, this
# recomputes the same gap set live. Dry-run by default -- nothing is submitted without
# SUBMIT=1, because most gaps close on their own (the `ev_*` watchers are still running).
#
#   bash scripts/slurm/fill_eval_gaps.sh              # print the gap table
#   SUBMIT=1 bash scripts/slurm/fill_eval_gaps.sh     # ...and sbatch the fillable ones
#   GRID="1 2 4 8 16 32 64" bash scripts/slurm/fill_eval_gaps.sh
#
# What counts as a gap here is only what a single eval job can fix: a checkpoint with no
# row, or a row missing an n level. The other four coverage holes (missing mean_reward,
# absent final/discounted series, the unprovenanced outer/inner runs, the empty large-n
# tail) are printed as advisories -- they are a backfill, a full re-eval and two cost
# decisions respectively, not something to fire off automatically.
set -uo pipefail

ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python
GRID="${GRID:-1 2 4 8 16 32 64}"
SUBMIT="${SUBMIT:-}"

# Runs whose watcher is alive are skipped: it will evaluate the checkpoint itself, and a
# second job on the same run dir contends for the same success_curve.json lock.
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | sed -n 's/^ev_//p' | sort -u)

PLAN=$(mktemp)
trap 'rm -f "$PLAN"' EXIT

GRID="$GRID" LIVE="$LIVE" ROOT="$ROOT" "$PY" - "$PLAN" <<'EOF' || { echo "gap detection failed"; exit 1; }
import json, os, re, sys

ROOT = os.environ['ROOT']
GRID = [int(x) for x in os.environ['GRID'].split()]
LIVE = set(filter(None, os.environ['LIVE'].splitlines()))

def rows_of(run):
    p = os.path.join(run, 'bon_search', 'success_curves.jsonl')
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]

def at_n(row, key, n):
    return dict(zip(row['n'], row.get(key) or [])).get(n)

plan, advis, no_tail = [], [], []
for name in sorted(os.listdir(ROOT)):
    run = os.path.join(ROOT, name)
    # symlinks are the ctx-* back-aliases from the 2026-08-05 rename (AUDIT 9.9); visiting
    # them would plan every gap twice, once under each name.
    if os.path.islink(run) or not os.path.isdir(run):
        continue
    rows = rows_of(run)
    evaluated = {r['step']: r for r in rows}
    ckdir = os.path.join(run, 'checkpoints')
    ck = sorted(int(m.group(1)) for f in os.listdir(ckdir)
                if (m := re.match(r'step_(\d+)\.ckpt$', f))) if os.path.isdir(ckdir) else []

    # Preserve the run's own val convention. The r8 watchers ran --skip-val, so every row
    # there has val_n_episodes 0; filling a gap WITH val would leave one run half scored on
    # a split the rest of its curve never saw.
    skip_val = bool(rows) and all(not r.get('val_success_rate') for r in rows)

    # BC is swept at every n only at its best-val-n=1 checkpoint -- best-of-N over i.i.d.
    # samples costs the same as a search arm, so the whole curve is not worth it. The target
    # is re-derived, never hardcoded: a later checkpoint can overtake and move it.
    is_bc = name.startswith('bc_')
    bc_target = None
    if is_bc:
        cand = [(at_n(r, 'val_success_rate', 1), r['step']) for r in rows
                if at_n(r, 'val_success_rate', 1) is not None]
        bc_target = max(cand)[1] if cand else None

    for step in ck:
        row = evaluated.get(step)
        if row is None:
            want = [1] if is_bc else list(GRID)       # off-target BC needs only n=1
            gap = 'G1'
        elif is_bc and step != bc_target:
            continue                                  # BC off-target: n=1 only, by design
        else:
            want = sorted(set(GRID) - set(row['n']))
            gap = 'G3' if is_bc else 'G2'
        if not want:
            continue
        # one job for a whole sweep, one job per level otherwise -- levels are independent
        # since eval_search_pusht writes and merges under a lock after every n
        chunks = [(min(want), max(want))] if want == GRID else [(n, n) for n in want]
        for lo, hi in chunks:
            plan.append(dict(gap=gap, run=run, name=name, step=step, lo=lo, hi=hi,
                             skip_val=skip_val, live=name in LIVE))

    # --- advisories: real coverage holes, but not fixable by one eval job ---
    miss_mr = [r['step'] for r in rows if not r.get('mean_reward')]
    if miss_mr:
        advis.append(f'G4 {name}: {len(miss_mr)} row(s) without mean_reward '
                     f'(steps {miss_mr}) -- scripts/backfill_mean_reward.py')
    n_fin = sum(1 for r in rows if r.get('mean_reward_final'))
    if rows and n_fin < len(rows):
        advis.append(f'G5 {name}: mean_reward_final/_discounted on {n_fin}/{len(rows)} rows '
                     f'-- per-step rewards discarded pre-2026-08-05, re-eval only')
    if rows and not any(n > 64 for r in rows for n in r['n']):
        no_tail.append(name)

# one line, not one per run: the tail is missing almost everywhere, and 20 identical lines
# bury the advisories that name specific steps.
if no_tail:
    advis.append(f'G7 no n>64 on {len(no_tail)} run(s) -- the large-n tail exists only on '
                 f'the legacy-29 generation. scripts/slurm/submit_large_n_evals.sh. Runs: '
                 + ', '.join(no_tail))

for name in ('train_pusht_search_outer_inner', 'train_pusht_search_outer_inner_subgoal',
             'train_pusht_search_outer_inner_subgoal_verifier'):
    advis.append(f'G6 {name}: rows predate the n_episodes field (null) -- the 50-episode '
                 f'test set CANNOT be confirmed; 10/50 checkpoints evaluated, no splits.json')

json.dump({'plan': plan, 'advisories': advis}, open(sys.argv[1], 'w'))
EOF

"$PY" - "$PLAN" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print('=== eval gaps: checkpoints/levels with no 50-episode result ===')
print(f'{"GAP":<4}{"RUN":<58}{"STEP":>8}  {"N":<7}{"SKIPVAL":<9}WATCHER')
for j in d['plan']:
    n = str(j['lo']) if j['lo'] == j['hi'] else f'{j["lo"]}-{j["hi"]}'
    live = 'LIVE (skipped)' if j['live'] else '-'
    print(f'{j["gap"]:<4}{j["name"]:<58}{j["step"]:>8}  {n:<7}'
          f'{str(j["skip_val"]).lower():<9}{live}')
n_live = sum(1 for j in d['plan'] if j['live'])
print(f'\n{len(d["plan"])} job(s) planned, {n_live} skipped as live, '
      f'{len(d["plan"]) - n_live} submittable')
print('\n=== advisories: coverage holes NOT auto-filled ===')
for a in d['advisories']:
    print(' ', a)
EOF

if [ -z "$SUBMIT" ]; then
    echo
    echo "dry run. SUBMIT=1 bash scripts/slurm/fill_eval_gaps.sh to submit the non-LIVE jobs."
    exit 0
fi

submitted=0
while IFS=$'\t' read -r name run step lo hi skip_val live; do
    [ "$live" = "True" ] && continue
    ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "$run" "$step")
    [ -f "$ckpt" ] || { echo "  missing $ckpt, skipping"; continue; }
    extra=()
    [ "$skip_val" = "True" ] && extra+=(--skip-val)
    # a full 1..64 sweep costs roughly 2x its top level; a single level is far cheaper.
    # Shorter requests schedule sooner on the preemptible ckpt partition.
    if [ "$lo" = "$hi" ]; then tlim=1:30:00; else tlim=3:00:00; fi
    jid=$(sbatch --parsable --time="$tlim" \
            --job-name="fill_n${hi}_$(echo "$name" | sed 's/_seed-42//')" \
            "$HERE/eval_ckpt_pusht_search.sbatch" "$ckpt" \
            --min-n "$lo" --max-n "$hi" ${extra[@]+"${extra[@]}"})
    echo "  submitted $jid  n=$lo-$hi  step=$step  $name"
    submitted=$((submitted + 1))
done < <("$PY" - "$PLAN" <<'EOF'
import json, sys
for j in json.load(open(sys.argv[1]))['plan']:
    print('\t'.join(map(str, (j['name'], j['run'], j['step'], j['lo'], j['hi'],
                              j['skip_val'], j['live']))))
EOF
)
echo "submitted $submitted job(s)"
