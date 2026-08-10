#!/bin/bash
# Run status: every run directory, every eval watcher, the large-n grid, doc row count.
#
# GLOBS the run directory rather than listing runs. The hardcoded version silently omitted
# arms launched outside a launcher -- a `subgoal-only_k16_cd0.9` variant trained
# unreported because it was not in the list. A run that exists on disk should appear here
# whether or not this script knew about it in advance.
#
# `ctx-*` entries are skipped: after the 2026-08-05 rename (AUDIT.md 9.9) those are
# back-symlinks to the arm-labelled directories, so counting them would double every run.
set -uo pipefail
ROOT=/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline
PY=/gscratch/robotics/harine/miniconda3/envs/robodiff/bin/python

printf '%-52s %-9s %-8s %-9s %-9s %s\n' RUN STATE STEP TRAIN VAL 'CKPT saved/eval'
for d in "$ROOT"/*/; do
    r=$(basename "$d")
    [ -L "${d%/}" ] && continue                      # back-symlink, not a run
    case "$r" in ctx-*|bon_search*) continue;; esac
    [ -d "$d/checkpoints" ] || [ -f "$d/logs.json.txt" ] || continue
    st=$(squeue -u "$USER" -h -o "%T" -n "tr_$r" 2>/dev/null | head -1); st=${st:-ENDED}
    nck=$(ls "$d"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    nev=$( [ -f "$d/bon_search/success_curves.jsonl" ] && wc -l < "$d/bon_search/success_curves.jsonl" || echo 0 )
    watch=$(squeue -u "$USER" -h -o "%j" | grep -cx "ev_$r" || true)
    "$PY" - "$d" "$r" "$st" "$nck" "$nev" "$watch" <<'EOF'
import json, pathlib, sys
d, r, st, nck, nev, watch = sys.argv[1:7]
p = pathlib.Path(d, 'logs.json.txt'); gs = tl = vl = None
if p.is_file():
    for line in p.read_text().splitlines()[::-1]:
        try: x = json.loads(line)
        except Exception: continue
        if gs is None: gs = x.get('global_step'); tl = x.get('train_loss')
        if vl is None and 'val_loss' in x: vl = x['val_loss']
        if gs is not None and vl is not None: break
f = lambda v: f'{v:.4f}' if isinstance(v, (int, float)) else '-'
short = r.replace('_demos-100_seed-42', '').replace('_seed-42', '')
flag = '' if watch != '0' or st == 'ENDED' else '  [NO WATCHER]'
print(f'{short:<52} {st:<9} {str(gs or "-"):<8} {f(tl):<9} {f(vl):<9} {nck}/{nev}{flag}')
EOF
done

echo
echo "=== training jobs not RUNNING ==="
squeue -u "$USER" -h -o "%j %T %R" | grep "^tr_" | grep -v RUNNING | sed 's/^/  /' || echo "  (none)"
echo
echo "=== runs with checkpoints but NO eval watcher ==="
miss=0
for d in "$ROOT"/*/; do
    r=$(basename "$d"); [ -L "${d%/}" ] && continue
    case "$r" in ctx-*) continue;; esac
    nck=$(ls "$d"/checkpoints/step_*.ckpt 2>/dev/null | wc -l)
    nev=$( [ -f "$d/bon_search/success_curves.jsonl" ] && wc -l < "$d/bon_search/success_curves.jsonl" || echo 0 )
    [ "$nck" -le "$nev" ] && continue
    squeue -u "$USER" -h -o "%j" | grep -qx "ev_$r" && continue
    echo "  $r  ($nck saved, $nev evaluated)"; miss=1
done
[ "$miss" -eq 0 ] && echo "  (none - every run with unevaluated checkpoints has a watcher)"
echo
echo "=== job mix ==="
squeue -u "$USER" -h -o "%j %P" | awk '{split($1,a,"_"); print "  "a[1]"  "$2}' | sort | uniq -c
