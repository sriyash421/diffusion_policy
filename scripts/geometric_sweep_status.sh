#!/usr/bin/env bash
# Status of the geometric-split sweep: training progress and eval coverage, one screen.
#
# STEP comes from <run>/logs.json.txt, which carries `global_step` on EVERY logged line, so
# it is the live count -- checkpoints land only every 10k and a job three hours in with no
# checkpoint looks identical to one that never started. CKPT is the largest step_*.ckpt on
# disk, which is what the eval watchers actually consume.
#
#   bash scripts/geometric_sweep_status.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="${DP_OUTPUT_ROOT:-/gscratch/robotics/harine/diffusion_policy_outputs}"/pusht_search/pusht_image_search_imgonly
PY="${DP_PY:-/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python}"
MAXSTEP=100000

date '+=== geometric-split sweep, %F %T ==='
printf '%-40s %-10s %8s %7s %6s %s\n' RUN STATE STEP CKPT PCT ELAPSED
running=0; done_n=0; failed=0

# The work list: the 15 baseline arms, then the 12 obs-noise arms, which live on a subset of
# the datasets and carry an extra suffix between ver-t_goal and enc-.
WORK=()
for tag in _split-blq:137 _split-trq:173 _split-brd30:30 _split-brd60:60 _split-brd100:100; do
  t=${tag%%:*}; d=${tag##*:}
  for a in unet_bc:unetbc_ver-t_goal offline:value_k1_ver-t_goal outer_inner:value_k16_ver-t_goal; do
    sub=${a%%:*}; stem=${a##*:}
    WORK+=("$sub|${stem}_enc-resnet18_demos-${d}${t}_seed-42")
  done
done
for tag in _split-blq:137 _split-brd60:60; do
  t=${tag%%:*}; d=${tag##*:}
  for infix in _son-flat800 _son-flat400 _son-flat200 _son-ramp800to400 _son-ramp400to200 _gm-t800-T3px; do
    WORK+=("outer_inner|value_k16_ver-t_goal${infix}_enc-resnet18_demos-${d}${t}_seed-42")
  done
done

for entry in "${WORK[@]}"; do
    sub=${entry%%|*}; run=${entry##*|}
    dir="$ROOT/$sub/$run"
    line=$(squeue -u "$USER" -h -o "%j|%t|%M" | awk -F'|' -v n="tr_$run" '$1==n{print $2"|"$3}')
    if [ -n "$line" ]; then st=${line%%|*}; el=${line##*|}
    else
      st=$(sacct -u "$USER" -n -X --starttime=2026-09-03 -o JobName%70,State 2>/dev/null \
           | awk -v n="tr_$run" '$1==n{s=$2} END{print (s?s:"—")}'); el="—"
    fi
    step=$(tail -1 "$dir/logs.json.txt" 2>/dev/null \
           | $PY -c "import sys,json
try: print(json.loads(sys.stdin.read()).get('global_step','—'))
except Exception: print('—')" 2>/dev/null)
    ck=$(ls "$dir"/checkpoints/step_*.ckpt 2>/dev/null | sed 's/.*step_0*//;s/\.ckpt//' | sort -n | tail -1)
    pct='—'; [ "${step:-—}" != "—" ] && pct=$(( step * 100 / MAXSTEP ))%
    # A just-finished job can return an empty state from sacct while the accounting db
    # catches up, which counted it as neither done nor running and made the tally short.
    # The checkpoint on disk is the ground truth.
    if [ "${ck:-0}" = "$MAXSTEP" ] && [ "$st" != "R" ]; then st=COMPLETED; fi
    case "$st" in R) running=$((running+1));; COMPLETED) done_n=$((done_n+1)); step=$MAXSTEP; pct=100%;;
                    FAILED|TIMEOUT|CANCELLED*|NODE_FAIL|PREEMPTED) failed=$((failed+1));; esac
    printf '%-40s %-10s %8s %7s %6s %s\n' \
      "$(echo "$run" | sed 's/_ver-t_goal//;s/_enc-resnet18//;s/_seed-42//')" \
      "$st" "${step:-—}" "${ck:-none}" "$pct" "$el"
done
echo "training: $done_n done / $running running / $failed failed-or-preempted (of ${#WORK[@]})"

echo
echo "--- eval coverage (steps with a curve, per rule) ---"
printf '%-40s %-14s %s\n' RUN RULE STEPS
nfiles=0
while read -r f; do
  [ -z "$f" ] && continue
  nfiles=$((nfiles+1))
  rule=$(basename "$(dirname "$f")" | sed 's/bon_search_sel-//;s/_obs-/ /')
  r=$(basename "$(dirname "$(dirname "$f")")" | sed 's/_ver-t_goal//;s/_enc-resnet18//;s/_seed-42//')
  steps=$($PY -c "
import json
s=sorted({json.loads(l)['step'] for l in open('$f') if l.strip()})
print(','.join(str(x//1000)+'k' for x in s) or '—')" 2>/dev/null)
  printf '%-40s %-14s %s\n' "$r" "$rule" "$steps"
done < <(find "$ROOT" -path "*split-*" -name success_curves.jsonl 2>/dev/null | sort)
echo "curve files: $nfiles / 74   eval jobs queued: $(squeue -u "$USER" -h -o '%j' | grep -c '^ev_.*split-')"
pgrep -f autoupdate_geometric_readouts.sh >/dev/null \
  && echo "autoupdate loop: running" || echo "autoupdate loop: NOT RUNNING -- restart it"

# --- results ----------------------------------------------------------------------------
# The LATEST evaluated checkpoint per (run, rule), at three widths. This is a progress
# readout, not a result to quote: a row at step 40k is a partially-trained model, and no
# step is nominated as best -- selection is never done on test. Full per-step tables with
# every n are in SUCCESS_RATES_GEOMETRIC.md.
echo
echo "--- test success rate, LATEST evaluated step per run ---"
$PY - "$ROOT" <<'PYEOF_INNER'
import json, pathlib, sys, re
root = pathlib.Path(sys.argv[1])
NS = [1, 2, 4, 8, 16]
# GROUPED BY DATASET, with each dataset's baselines first and its noise arms under them.
# An arm is only ever comparable to the baseline it shares a split with, so putting the
# three baselines beside their own ladders is the arrangement that makes the comparison
# readable; sorting by run name scatters them.
DATASETS = [('_split-trq', 'trq-173  (eval: 33 top-right-quadrant episodes)'),
            ('_split-blq', 'blq-137  (eval: 50 bottom-left-quadrant episodes)'),
            ('_split-brd100', 'brd100   (eval: 50 interior-core episodes)'),
            ('_split-brd60', 'brd60    (eval: same 50 interior-core episodes)'),
            ('_split-brd30', 'brd30    (eval: same 50 interior-core episodes)')]
# order within a dataset: baselines, then ladders by severity, then the goal mask
ORDER = ['unetbc', 'value_k1', 'k16-base', '_son-flat200', '_son-flat400', '_son-flat800',
         '_son-ramp400to200', '_son-ramp800to400', '_gm-t400', '_gm-t800']
def family(run):
    if run.startswith('unetbc'): return 'unetbc', 'BC (UNet)'
    if run.startswith('value_k1_'): return 'value_k1', 'ST k=1'
    for key in ORDER[3:]:
        if key in run:
            lab = key.lstrip('_').replace('son-', '').replace('ramp', 'ramp ').replace('to', '->')
            return key, f'k16 {lab}'
    return 'k16-base', 'ST k=16 (baseline)'
rows = {}
for f in sorted(root.glob('*/*split-*/bon_search_sel-*/success_curves.jsonl')):
    d = f.parent.name
    sel = 'final_pass' if 'final_pass' in d else 'argmax'
    obs = 'corrupt' if 'obs-corrupt' in d else 'clean'
    run = f.parent.parent.name
    per = {}
    for line in f.read_text().splitlines():
        if line.strip():
            x = json.loads(line)
            per[int(x['step'])] = dict(zip(x['n'] or [], x['success_rate'] or []))
    if per:
        st = max(per)
        rows.setdefault((run, obs), {})[sel] = (per[st], st)
def cells(v):
    return ' '.join(f'{v[n]:5.2f}' if v and v.get(n) is not None else f'{"-":>5}' for n in NS)
for tag, title in DATASETS:
    present = [r for (r, o) in rows if tag in r]
    if not present:
        continue
    print(f'\n  === {title} ===')
    print(f'  {"arm":<22} {"obs":<8} {"step":>5}  {"ARGMAX  n=1/2/4/8/16":<30}{"FINAL CAND  n=1/2/4/8/16"}')
    seen = sorted({r for r in present},
                  key=lambda r: (ORDER.index(family(r)[0]), r))
    for run in seen:
        _, lab = family(run)
        for obs in ('clean', 'corrupt'):
            rec = rows.get((run, obs))
            if not rec:
                continue
            a, st = rec.get('argmax', (None, None))
            fp, st2 = rec.get('final_pass', (None, None))
            st = st if st is not None else st2
            print(f'  {lab:<22} {obs:<8} {st//1000:4d}k  {cells(a):<30}{cells(fp)}')
PYEOF_INNER
