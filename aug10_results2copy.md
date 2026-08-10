# Results to copy for offline analysis — 2026-08-10

Which directories on `/gscratch` back the numbers in `SUCCESS_RATES.md` and `aug9_analysis.md`,
and what inside them is worth moving.

Nothing here is in git — the repo holds the code and the analysis, the cluster holds the runs.
This doc is the join between the two. `aug9_analysis.md` §1b says *what each run is*; this says
*what to carry off the cluster*.

**Source root** — `$DP_OUTPUT_ROOT` = `/gscratch/robotics/harine/diffusion_policy_outputs`

## What to copy from each run

```
bon_search/                      success_curves.jsonl, best.json, step_*/success_curve.json
bon_search_sel-argmax/           the argmax/softmax selection sweep, where it exists
bon_search_sel-softmax/
run.json  splits.json            what was run, and on which episodes
logs.json.txt  train.log         loss curves and the training record
.hydra/config.yaml               provenance -- aug9_analysis.md reads THIS, not config/
checkpoints/step_<best>.ckpt     the one step named in bon_search/best.json ("checkpoint")
```

**Skip** `checkpoints/` beyond that single file, `wandb/`, and `media/`. Those are 99% of the
213 GB and none of it is needed to reproduce a table or re-plot a curve.

Everything except the checkpoint is ~250 MB across all runs. Adding the one selected checkpoint
per run brings the total to **≈7.5 GB**.

## The 26 runs under `pusht_search/pusht_image_search/offline/`

`step` is the checkpoint `best.json` selects; `sel` is what it was selected on.

| run directory | best step | sel | ckpt |
|---|--:|---|--:|
| `bc_demos-100_seed-42` | 290000 | val | 262M |
| `bc_demos-25_seed-42` | 90000 | val | 262M |
| `bc_demos-29_seed-42` | 96000 | val | 262M |
| `value_corrupt-False_demos-100_seed-42` | 8000 | val | 262M |
| `value_corrupt-True_demos-100_seed-42` | 5000 | val | 262M |
| `value_corrupt-False_demos-29_seed-42` | 1000 | val | 197M |
| `value_corrupt-True_demos-29_seed-42` | 7000 | val | 197M |
| `value_corrupt-False_demos-29-r8_seed-42` | 2000 | **test** | 262M |
| `value_corrupt-True_demos-29-r8_seed-42` | 2000 | **test** | 262M |
| `subgoal-chosen4value_corrupt-False_demos-100_seed-42` | 6000 | val | 264M |
| `subgoal-chosen4value_corrupt-True_demos-100_seed-42` | 15000 | val | 264M |
| `subgoal-chosen4value_corrupt-False_demos-29_seed-42` | 9000 | val | 198M |
| `subgoal-chosen4value_corrupt-True_demos-29_seed-42` | 5000 | val | 198M |
| `subgoal-chosen4value_corrupt-False_demos-29-r8_seed-42` | 8000 | **test** | 264M |
| `subgoal-chosen4value_corrupt-True_demos-29-r8_seed-42` | 2000 | **test** | 264M |
| `subgoal-value_corrupt-False_demos-100_seed-42` | 7000 | val | 264M |
| `subgoal-value_corrupt-True_demos-100_seed-42` | 5000 | val | 264M |
| `subgoal-value_corrupt-False_demos-29_seed-42` | 2000 | val | 198M |
| `subgoal-value_corrupt-True_demos-29_seed-42` | 1000 | val | 198M |
| `subgoal-value_corrupt-False_demos-29-r8_seed-42` | 2000 | **test** | 264M |
| `subgoal-value_corrupt-True_demos-29-r8_seed-42` | 2000 | **test** | 264M |
| `subgoal-only_corrupt-False_demos-100_seed-42` | 52000 | val | 264M |
| `subgoal-only_corrupt-True_demos-100_seed-42` | 50000 | val | 264M |
| `subgoal-only_k4_cd0.9_corrupt-False_demos-100_seed-42` | 98000 | val | 264M |
| `subgoal-only_k8_cd0.9_corrupt-False_demos-100_seed-42` | 96000 | val | 264M |
| `subgoal-only_k16_cd0.9_corrupt-False_demos-100_seed-42` | 46000 | val | 264M |

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

**The six `*_demos-29-r8_*` rows are selected on test, not val.** Their watchers ran
`--skip-val`, so those runs have no val curve at all and `best.json` records
`selected_on: "test (legacy row)"`. The checkpoint the table names for them is the best *test*
step — fine to copy, but it is not a held-out pick and must not be quoted as one
(`SUCCESS_RATES.md` §2, `aug9_analysis.md` §2).

## Recipe

```bash
SRC=/gscratch/robotics/harine/diffusion_policy_outputs
DST=/path/to/destination            # needs ~8 GB

OFF=$SRC/pusht_search/pusht_image_search/offline
for d in "$OFF"/*/; do
    [ -L "${d%/}" ] && continue                      # skip the ctx-* back-symlinks
    name=$(basename "$d")
    mkdir -p "$DST/offline/$name/checkpoints"
    rsync -a --exclude='checkpoints' --exclude='wandb' --exclude='media' \
          "$d" "$DST/offline/$name/"
    ckpt=$(python3 -c "import json;print(json.load(open('$d/bon_search/best.json'))['checkpoint'])")
    rsync -a "$ckpt" "$DST/offline/$name/checkpoints/"
done

rsync -a --exclude='checkpoints' --exclude='wandb' --exclude='media' "$SRC/runs" "$DST/"
rsync -a "$SRC/candidate_scores" "$DST/"
```

Re-derive this table rather than trusting it if runs have been added since: the `step` column is
just `bon_search/best.json`'s `step`, and evals were still landing when this was written.
