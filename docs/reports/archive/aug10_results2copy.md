# Results to copy for offline analysis — 2026-08-10

Which directories on `/gscratch` back the numbers in `SUCCESS_RATES.md` and `aug9_analysis.md`,
and what inside them is worth moving.

Nothing here is in git — the repo holds the code and the analysis, the cluster holds the runs.
This doc is the join between the two. `aug9_analysis.md` §1b says *what each run is*; this says
*what to carry off the cluster*.

**Source root** — `$DP_OUTPUT_ROOT` = `/gscratch/robotics/harine/diffusion_policy_outputs`

## What to copy from each run

Take **everything except `checkpoints/`** — measured across all 26 runs, that is only ~2.3 GB,
so there is no reason to be selective about the small stuff:

```
bon_search/                      success_curves.jsonl  (one row per checkpoint)
                                 step_*/success_curve.json -- per_n_rewards, the full
                                 50-episode reward vector at each n, plus episode_idxs.
                                 This is what any re-analysis beyond the summary rates
                                 needs: per-episode outcomes, not just the mean.
bon_search_sel-argmax/           the argmax-vs-softmax selection probe, where it exists
bon_search_sel-softmax/
run.json  splits.json            what was run, and on exactly which episodes
logs.json.txt  train.log         every training metric per step, and the run record
.hydra/config.yaml               provenance -- aug9_analysis.md reads THIS, not config/
media/                           96 rendered rollout mp4s per run -- qualitative failure
                                 modes, which no scalar in the curves captures
wandb/                           full run history and debug logs; the metrics are also in
                                 logs.json.txt, so this is the one you can drop first
checkpoints/step_<chosen>.ckpt   only the step(s) you decide to keep -- see below
```

| | all 26 runs |
|---|--:|
| curves, configs, logs, splits | 408 MB |
| `media/` | 463 MB |
| `wandb/` | 1.4 GB |
| **everything except checkpoints** | **~2.3 GB** |
| one checkpoint per run | +6.5 GB (~200–265 MB each) |
| every checkpoint | 213 GB |

**Nothing on disk names a winner.** `bon_search/best.json` used to record one — highest val
success at the largest n common to every evaluated checkpoint — and it was removed, along with
the code that wrote it, because that rule is an analysis choice and having it on disk made it
look like a result. Read `success_curves.jsonl`, decide which steps matter, and copy those.

## The 26 runs under `pusht_search/pusht_image_search/offline/`

Every checkpoint is ~262–264M, except the four legacy-29 arms at ~197–198M.

```
bc_demos-100_seed-42                                    bc_demos-25_seed-42
bc_demos-29_seed-42

value_corrupt-{False,True}_demos-100_seed-42            argmax, verifier scalar
value_corrupt-{False,True}_demos-29_seed-42               legacy 29 (no EMA, split crop)
value_corrupt-{False,True}_demos-29-r8_seed-42            r8 29 (no val curve)

subgoal-chosen4value_corrupt-{False,True}_demos-100_seed-42
subgoal-chosen4value_corrupt-{False,True}_demos-29_seed-42
subgoal-chosen4value_corrupt-{False,True}_demos-29-r8_seed-42

subgoal-value_corrupt-{False,True}_demos-100_seed-42
subgoal-value_corrupt-{False,True}_demos-29_seed-42
subgoal-value_corrupt-{False,True}_demos-29-r8_seed-42

subgoal-only_corrupt-{False,True}_demos-100_seed-42     final_pass, 100k steps
subgoal-only_k{4,8,16}_cd0.9_corrupt-False_demos-100_seed-42   context decay, clean only
```

## Also copy

| path | what it is | size |
|---|---|--:|
| `runs/train_pusht_search_outer_inner` | outer/inner trainer, value ctx | ~264M + artifacts |
| `runs/train_pusht_search_outer_inner_subgoal` | outer/inner, subgoal ctx | ~264M + artifacts |
| `runs/train_pusht_search_outer_inner_subgoal_verifier` | outer/inner, subgoal+value | ~264M + artifacts |
| `candidate_scores/` | 10 dumps backing `CANDIDATES_FROM_SUBGOAL.md` | 8.7M |

The three `outer_inner` runs are **archive-grade**: ImageNet weights paired with GroupNorm and no
augmentation, overfit by ~3k steps (`SUCCESS_RATES.md` Appendix A). Copy them for completeness,
not to quote from.

## Two things that will bite

**The 12 `ctx-*` entries in `offline/` are back-symlinks, not runs.** They point at the renamed
directories (2026-08-05, `AUDIT.md` §9.9) and exist so live jobs holding the old path keep
working. Use `rsync -a` **without** `-L`, or every run copies twice under two names and the
totals above double.

**The six `*_demos-29-r8_*` runs have no val curve at all.** Their watchers ran `--skip-val`,
so the whole budget went to the 50 test episodes. Any step you pick from those curves is
picked on test, and a test number read at a test-chosen step is not held out. Re-evaluate on
val first if you need one (`SUCCESS_RATES.md` §2, `aug9_analysis.md` §2).

## Recipe

```bash
SRC=/gscratch/robotics/harine/diffusion_policy_outputs
DST=/path/to/destination     # ~2.3 GB for everything below, +~260 MB per checkpoint kept
STEPS=()                     # e.g. STEPS=(2000 8000); empty = no checkpoints this pass
SKIP=(--exclude='checkpoints')          # add --exclude='wandb' to save 1.4 GB

OFF=$SRC/pusht_search/pusht_image_search/offline
for d in "$OFF"/*/; do
    [ -L "${d%/}" ] && continue                      # skip the ctx-* back-symlinks
    name=$(basename "$d")
    rsync -a "${SKIP[@]}" "$d" "$DST/offline/$name/"
    for s in "${STEPS[@]}"; do
        ckpt=$(printf '%s/checkpoints/step_%07d.ckpt' "${d%/}" "$s")
        [ -f "$ckpt" ] || continue
        mkdir -p "$DST/offline/$name/checkpoints"
        rsync -a "$ckpt" "$DST/offline/$name/checkpoints/"
    done
done

rsync -a "${SKIP[@]}" "$SRC/runs" "$DST/"
rsync -a "$SRC/candidate_scores" "$DST/"
```

Run it once with `STEPS=()` to get all 2.3 GB of analysis material, decide which steps are
worth the weights, then re-run with those steps — rsync is incremental, so the second pass
moves only the checkpoints.

Then point the generator at the copy and the tables rebuild off-cluster:

```bash
export DP_OUTPUT_ROOT=$DST
python scripts/build_success_rates_doc.py
```

One caveat for anything that matches run directories by string: the `checkpoint` paths recorded
*inside* `success_curves.jsonl` are absolute cluster paths, and for runs evaluated before the
2026-08-05 rename they point through the `ctx-*` alias. Resolve them yourself; nothing reads
them back automatically.

