# ST-diffusion k=16 with linear slot weights — success rates (drumkit)

One arm and its control: the 30-demo search transformer at 4/4/256, trained to 100k gradient steps with a checkpoint every 10k, where the K=16 candidate slots carry **linear** rather than uniform loss weights. Every checkpoint is swept over `n = 1, 2, 4, 8, 16, 32, 64` under all three selection rules, on the same 50 held-out test episodes.

Launched by `scripts/run_st_k16_linear_drumkit.sh`. Regenerate this doc with `python scripts/build_linear_weights_doc.py`.

**Source is the driver log, not `success_curves.jsonl`.** `eval_search_pusht.py` keys its curve on the checkpoint rather than on (checkpoint, selection), so one checkpoint's three selection rules overwrite each other on disk and only the last to run — `final_pass` — survives, wearing all seven n values. The pre-existing `search` arm shows the same collapse (10 rows, all `final_pass`). The per-n numbers are printed to stdout for all three rules, so the log is the only complete record; `scripts/bon_grid_table.py` parses it for the same reason.

## What is being varied

One forward decodes all K=16 candidate slots against the same expert action, and the staircase memory mask lets slot k attend to exactly the first k scored candidates — so the model fits a family of conditionals from *no context* (slot 0) to *15 scored candidates* (slot 15). `slot_weights` is the SCALE of each of those slots' loss terms, always renormalized to mean 1 so switching profiles cannot move the loss scale that `gradient_clip_norm` and the effective step size both read.

`mode: linear` makes the weight affine in the slot index, `w_k ∝ 1 + (ratio-1)·k/(K-1)`:

| slot k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w_k | 0.341 | 0.429 | 0.517 | 0.605 | 0.693 | 0.780 | 0.868 | 0.956 | 1.044 | 1.132 | 1.220 | 1.307 | 1.395 | 1.483 | 1.571 | 1.659 |

**Why ratio 4.857.** `linear` has no default — the resolver raises without a ratio — so the number is a choice. 4.857 = `0.9^-(K-1)` at K=16, which is exactly the endpoint spread of the legacy `slot_weight_decay: 0.9` geometric profile (`w_last/w_first` = 4.857 for both). Holding the spread fixed is what isolates **curvature** — geometric front-loads its down-weighting into the low-context slots, linear spreads it evenly — from the much larger effect of simply tilting harder. Any other ratio confounds the two.

`val` stays `uniform`, so `val_loss` is computed under the canonical objective (uniform weights, plain L2) and remains a fixed cross-arm yardstick instead of moving with the weighting. `slot_loss_norm` stays `l2`. The two knobs are orthogonal and only the first is exercised here.

### The caveat this arm runs into

Under `argmax` **every slot is deployed** — all n candidates come from slots 0..K-1 and the executed action is the best of them. The objective is a good *max over the pool*, not a good final conditional, so up-weighting the high-context slots may be the wrong direction there; the last-slot-heavy argument is a `final_pass` argument, where slot K-1 *is* the deployment condition. That is why the grid sweeps all three rules rather than argmax alone — the columns should not be expected to move together.

## Comparability — read this before using the numbers

`verifier_tag` is **armTn** here, and the slot-weight code path requires it. The 4/4/256 uniform k=16 run these tables would naturally be read against — `outer_inner/value_k16_corrupt-False_demos-30_seed-42`, the `search` arm of `SUCCESS_RATES_30_100.md` — carries no `verifier_value` in its `.hydra/config.yaml` at all: it predates the cutover and trained under `t_goal`. That is a different scoring rule, and it does not only rescore at eval — it feeds the search context the model conditions on during training. `pusht_base.yaml` states outright that runs across the two are not comparable, which is why `ver-` is in `run_name`.

**So do not diff these tables against `SUCCESS_RATES_30_100.md`.** The valid control is the uniform armTn arm below; until it is trained (`bash scripts/run_st_k16_linear_drumkit.sh --uniform-control`) the linear numbers describe one arm in isolation and carry no claim about the weighting.

**The overfitting below is the 30-demo regime, not something the weighting did.** The uniform `t_goal` k=16 arm on the same 30 demos runs the same curve — `val_loss` min 0.1029 at step 4,096 rising to 0.3174 by 99,328, 3.08x — against the linear arm's 0.1006 → 0.2850, 2.83x. Near-identical shape, marginally *less* drift under linear weights. `val_loss` is computed under the canonical objective in both, so it is comparable across them even though their verifiers are not.

At 50 test episodes a single cell carries a 95% CI of roughly ±0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not separable**; read down a column or across several checkpoints, never one cell.

## Linear slot weights, ratio 4.857

`outer_inner/value_k16_ver-armTn_sw-lin4857_corrupt-False_demos-30_seed-42` → log arm `st-k16-lin4857` — w_k affine in the slot index, mean 1, w_last/w_first = 4.857

_10/10 checkpoints written; 9/10 fully swept (all three rules, n=1..64)._

**Fit.** `val_loss` bottoms at **0.1006** by step 4,096 and rises to **0.2850** by 99,328 — 2.83x off its minimum. Read the step rows below with that in mind: the late checkpoints are more overfit than the early ones, and a success rate that falls down the column is the expected shape here, not a surprise.

### Test success rate — `argmax`

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.06 | 0.14 | 0.20 | 0.40 | 0.22 | 0.30 |
| 20,000 | 0.04 | 0.14 | 0.14 | 0.34 | 0.30 | 0.32 | 0.42 |
| 30,000 | 0.04 | 0.18 | 0.22 | 0.26 | 0.38 | 0.38 | 0.40 |
| 40,000 | 0.22 | 0.20 | 0.42 | 0.34 | 0.38 | 0.40 | 0.36 |
| 50,000 | 0.10 | 0.18 | 0.30 | 0.28 | 0.28 | 0.34 | 0.46 |
| 60,000 | 0.14 | 0.24 | 0.28 | 0.36 | 0.26 | 0.34 | 0.32 |
| 70,000 | 0.20 | 0.20 | 0.32 | 0.36 | 0.44 | 0.34 | 0.36 |
| 80,000 | 0.20 | 0.28 | 0.34 | 0.32 | 0.30 | 0.42 | 0.36 |
| 90,000 | 0.14 | 0.30 | 0.26 | 0.32 | 0.24 | 0.28 | 0.34 |
| 100,000 | 0.10 | 0.26 | 0.34 | 0.26 | 0.40 | 0.22 | 0.26 |

### Test success rate — `softmax`

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.08 | 0.22 | 0.20 | 0.16 | 0.18 | 0.16 |
| 20,000 | 0.04 | 0.16 | 0.20 | 0.26 | 0.24 | 0.24 | 0.28 |
| 30,000 | 0.04 | 0.14 | 0.22 | 0.22 | 0.38 | 0.22 | 0.24 |
| 40,000 | 0.22 | 0.26 | 0.28 | 0.26 | 0.28 | 0.38 | 0.26 |
| 50,000 | 0.10 | 0.32 | 0.16 | 0.26 | 0.30 | 0.26 | 0.48 |
| 60,000 | 0.14 | 0.24 | 0.22 | 0.24 | 0.34 | 0.34 | 0.32 |
| 70,000 | 0.20 | 0.16 | 0.28 | 0.14 | 0.30 | 0.26 | 0.22 |
| 80,000 | 0.20 | 0.30 | 0.08 | 0.24 | 0.26 | 0.34 | 0.18 |
| 90,000 | 0.14 | 0.30 | 0.26 | 0.24 | 0.20 | 0.34 | 0.24 |
| 100,000 | 0.10 | 0.30 | 0.26 | 0.18 | 0.20 | 0.32 | 0.32 |

### Test success rate — `final_pass`

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.00 | 0.06 | 0.18 | 0.12 | 0.12 | 0.04 | 0.10 |
| 20,000 | 0.04 | 0.14 | 0.14 | 0.16 | 0.10 | 0.16 | 0.16 |
| 30,000 | 0.04 | 0.16 | 0.14 | 0.20 | 0.08 | 0.14 | 0.20 |
| 40,000 | 0.22 | 0.26 | 0.22 | 0.22 | 0.20 | 0.22 | 0.18 |
| 50,000 | 0.10 | 0.24 | 0.18 | 0.20 | 0.20 | 0.30 | 0.26 |
| 60,000 | 0.14 | 0.20 | 0.20 | 0.24 | 0.14 | 0.12 | 0.22 |
| 70,000 | 0.20 | 0.20 | 0.12 | 0.26 | 0.22 | 0.16 | – |
| 80,000 | 0.20 | 0.24 | 0.14 | 0.20 | 0.22 | 0.26 | 0.24 |
| 90,000 | 0.14 | 0.24 | 0.18 | 0.18 | 0.08 | 0.18 | 0.14 |
| 100,000 | 0.10 | 0.18 | 0.18 | 0.14 | 0.18 | 0.18 | 0.20 |

### Val success rate — `argmax` (30 episodes)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.03 | 0.00 | 0.23 | 0.23 | 0.23 | 0.37 | 0.43 |
| 20,000 | 0.03 | 0.07 | 0.37 | 0.27 | 0.20 | 0.23 | 0.23 |
| 30,000 | 0.10 | 0.13 | 0.23 | 0.30 | 0.33 | 0.30 | 0.33 |
| 40,000 | 0.10 | 0.23 | 0.30 | 0.23 | 0.43 | 0.27 | 0.37 |
| 50,000 | 0.17 | 0.10 | 0.27 | 0.37 | 0.30 | 0.30 | 0.27 |
| 60,000 | 0.20 | 0.07 | 0.37 | 0.23 | 0.33 | 0.23 | 0.40 |
| 70,000 | 0.23 | 0.20 | 0.17 | 0.20 | 0.23 | 0.23 | 0.60 |
| 80,000 | 0.17 | 0.17 | 0.37 | 0.27 | 0.20 | 0.37 | 0.27 |
| 90,000 | 0.13 | 0.23 | 0.27 | 0.20 | 0.30 | 0.37 | 0.27 |
| 100,000 | 0.20 | 0.20 | 0.30 | 0.33 | 0.33 | 0.33 | 0.47 |

Val is the split a checkpoint and an n may honestly be chosen on; the test tables above are the read-off, not the search space.

## Uniform slot weights (matched control)

`outer_inner/value_k16_ver-armTn_corrupt-False_demos-30_seed-42` → log arm `st-k16-unif-armTn` — the same arm with slot_weights untouched -- every slot at weight 1

_Not trained yet — no checkpoints on disk and no evaluated cells._

