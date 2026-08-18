# Latest PushT success rates

The current generation: the **30-demo search-vs-BC pair**, trained 2026-08-17/18 after the
search-procedure unification. Older generations are in `SUCCESS_RATES.md` (section B holds
what used to be in this file, with the list of changes that invalidated it).

`success = coverage >= 95%`; `mean reward = clip(coverage/0.95, 0, 1)` averaged over
episodes. **At 50 test episodes the SE is ~7pp, so gaps under ~14pp are not resolvable.**
Most differences below are inside that band -- read the caveats before quoting anything.

## The two arms

Both arms are **diffusion** policies (`PushTDiffusionSearchPolicy`, DDIM at 8 steps).
"SEARCH" here means search width 16, not the `train_pusht_st_n1` arm -- that is a separate
width-1 config with `num_inference_steps: 100`, and it was not part of these runs. The
Gaussian arm (`train_pusht_gaussian_search`) exists in the tree but has never been run.

Both are `PushTDiffusionSearchPolicy` with a **byte-identical** obs encoder and denoiser
(verified from each run's `.hydra/config.yaml`): ResNet18 `IMAGENET1K_V1`, `crop_shape
[76,76]`, `random_crop`, `use_group_norm`, `imagenet_norm: False`; 4 layers / 2 cond layers
/ 4 heads / 256 emb, `p_drop_attn 0.2`, `causal_attn: True`, `cond_encoder: gpt2`, DDIM at
8 inference steps. **The only difference is `max_actions`: 16 vs 1.**

| | SEARCH (k=16) | BC (k=1) |
|---|---|---|
| config | `train_pusht_diffusion_search n_demos=30` | `train_pusht_bc n_demos=30` |
| `max_actions` / eval n | 16 / 16 | 1 / 1 |
| trainer | `outer_inner` | `offline` |
| run dir | `outer_inner/value_k16_corrupt-False_demos-30_seed-42` | `offline/bc_demos-30_seed-42` |
| wandb | `5ufduqb2` | `mzatxyrb` |
| wall clock | 12h26m | 44m |
| final train / val loss | 0.0093 / 0.3174 | 0.0212 / 0.2005 |

Split: `pusht_seed42_train30.json` -- 30 train episodes, a strict subset of
`train100`'s train list, reusing its val (30) and test (50) verbatim, so 30-vs-100 isolates
dataset size. Seed 42, single seed.

## 1. In-training rollouts (mean reward, max)

**These are NOT a matched comparison**: SEARCH evaluated at n=16 (search active), BC at n=1.
Use section 2 for like-for-like.

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| SEARCH test | 0.7208 | 0.7755 | 0.8023 | 0.745 | 0.6143 |
| SEARCH val  | 0.8143 | 0.7574 | 0.7539 | 0.7708 | 0.7362 |
| BC test  | 0.4957 | 0.612 | 0.6029 | 0.6382 | - |
| BC val   | 0.4577 | 0.5681 | 0.6342 | 0.6397 | - |

BC's 100k row is blank because **the offline trainer skips its final rollout**: its rollouts
drift +1 per fire (20001, 40002, 60003, 80004) so the next trigger lands at 100005, past the
`max_gradient_steps: 100000` cap. Evaluated separately from `step_0100000.ckpt`: test mean
reward 0.593, success 0.240. `TrainSearchOuterInnerWorkspace` hits its boundaries exactly and
does not have this bug.

**Both arms are over-trained at 100k.** SEARCH peaks at 60k (0.8023) and falls 19pp by 100k; BC
peaks at 80k. 60-80k would have been the right budget at this demo count.

## 2. Best-of-n x selection rule (matched n, both arms)

SEARCH from `step_0060000.ckpt`, BC from `step_0080000.ckpt` -- each arm's best in-training
checkpoint. n in {1,2,4,8,16} x {argmax, softmax, final_pass}, `eval_search_pusht.py`,
seed 42.

> **These tables predate the 2026-08-18 selection fixes and must be read with the two
> caveats in section 3a.** In short: their n=1 argmax-vs-softmax gap is a reseeded
> replicate, not a selection effect, and their `final_pass` column was measured at n+1
> generations rather than n. Section 6 is being regenerated under the corrected semantics.

### TEST -- mean reward (max)

| n | SEARCH/argmax | SEARCH/softmax | SEARCH/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.599 | 0.668 | 0.663 | 0.624 | 0.590 | 0.585 |
| 2 | 0.670 | 0.731 | 0.709 | 0.647 | 0.612 | 0.613 |
| 4 | 0.757 | 0.670 | 0.734 | 0.660 | 0.679 | 0.576 |
| 8 | 0.707 | 0.667 | 0.728 | 0.728 | 0.618 | 0.602 |
| 16 | 0.724 | 0.790 | 0.660 | 0.644 | 0.637 | 0.664 |

### TEST -- success rate

| n | SEARCH/argmax | SEARCH/softmax | SEARCH/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.120 | 0.160 | 0.120 | 0.260 | 0.200 | 0.180 |
| 2 | 0.200 | 0.340 | 0.160 | 0.240 | 0.300 | 0.240 |
| 4 | 0.340 | 0.240 | 0.100 | 0.300 | 0.380 | 0.240 |
| 8 | 0.180 | 0.200 | 0.140 | 0.400 | 0.180 | 0.200 |
| 16 | 0.300 | 0.340 | 0.200 | 0.460 | 0.420 | 0.280 |

### VAL -- mean reward (max)

| n | SEARCH/argmax | SEARCH/softmax | SEARCH/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.680 | 0.683 | 0.723 | 0.677 | 0.644 | 0.601 |
| 2 | 0.743 | 0.769 | 0.740 | 0.663 | 0.725 | 0.641 |
| 4 | 0.783 | 0.731 | 0.760 | 0.686 | 0.670 | 0.703 |
| 8 | 0.739 | 0.732 | 0.732 | 0.777 | 0.719 | 0.613 |
| 16 | 0.780 | 0.751 | 0.735 | 0.786 | 0.772 | 0.643 |

### VAL -- success rate

| n | SEARCH/argmax | SEARCH/softmax | SEARCH/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.333 | 0.200 | 0.133 | 0.267 | 0.133 | 0.033 |
| 2 | 0.233 | 0.133 | 0.200 | 0.167 | 0.267 | 0.233 |
| 4 | 0.333 | 0.167 | 0.200 | 0.200 | 0.167 | 0.200 |
| 8 | 0.200 | 0.300 | 0.167 | 0.200 | 0.367 | 0.200 |
| 16 | 0.233 | 0.367 | 0.233 | 0.400 | 0.267 | 0.267 |

## 3. Two artifacts in the tables above (fixed 2026-08-18)

Both were found by storing the per-candidate verifier distances and comparing the n=1
column, where all three selection rules must agree -- with one candidate there is nothing
to select.

**1. argmax vs softmax at n=1 was a reseeded replicate, not a comparison.** The decision was
always identical (one candidate z-scores to 0, so `Categorical` returns index 0). But
`.sample()` drew from the *global* torch stream, and so does the sampler's
`torch.randn(..., generator=None)`; DDIM at `eta=0` adds nothing further, so that one extra
draw shifted every subsequent noise vector. Measured at `step_0010000`/test/n=1: all rules
agreed at step 0, diverged at step 1, and only **86/1900** rows matched after.

That makes the n=1 gaps in the tables above a **direct measurement of the noise floor** --
two runs of identical weights differing only in RNG:

| | argmax | softmax | gap |
|---|---|---|---|
| 60k, test reward | 0.599 | 0.668 | 0.069 |
| 60k, test success | 0.120 | 0.160 | 0.040 |
| 10k, test reward | 0.486 | 0.477 | 0.009 |
| 10k, test success | 0.060 | 0.120 | 0.060 |

**No gap anywhere in these tables smaller than ~0.07 reward / ~0.06 success is resolvable.**
That is a stronger and better-founded bound than the 14pp Wilson figure at the top, because
it is measured on this exact policy rather than assumed.

Fixed by giving selection its own `torch.Generator` (the pattern `CropScopeMixin` already
used). Verified end-to-end: n=1 argmax and softmax now produce **400/400** identical
candidate-score rows.

**2. `final_pass` was measured at n+1 generations.** It searched at n and then drew one
more, so n=1 executed a slot-1 conditional over two generations. Training does the opposite
-- `generate_search_context` runs at `max_actions - 1` and the loss covers slot
`max_actions - 1`, i.e. K generations total -- so eval was off-by-one from training at every
n, not just n=1.

Fixed by making n the total generation count under every rule: `final_pass` now searches at
n-1 and returns the n'th sample. At n=1 it skips the search entirely and returns the
empty-context conditional, the same action the other two rules return. At n = `max_actions`
it conditions on all K-1 context candidates, exactly the training configuration.

Both fixes change results, so the full grid was relaunched; sections 1-2 above are retained
under these caveats until it lands.

## 4. What this supports, and what it does not

**Supported:**
* **Best-of-n gives a real gain on mean reward** for both arms -- roughly +0.06 to +0.16
  from n=1 to n=16 in most columns.
* **`final_pass` is the weakest selection rule.** Lowest or near-lowest in most columns on
  both arms; it never wins on test success. Note this was measured while `final_pass` was
  spending n+1 generations to the others' n (see section 3), i.e. it lost while doing MORE work --
  so the corrected grid should not overturn this, only sharpen it.

**NOT supported by this data:**
* **That the search-trained policy beats BC.** At matched n the search advantage largely
  disappears -- on test success at n=16, BC/argmax (0.460) is the best cell in the table,
  above SEARCH/argmax (0.300). The apparent +0.164 in section 1 is search-at-16 vs BC-at-1, which
  conflates the trained policy with eval-time search budget.
* **Any ranking of argmax vs softmax.** They trade places by n and by split.
* **Any single cell.** CIs are ~+/-0.12 at 50 episodes and nearly every pairwise difference
  overlaps.

**Open discrepancy:** the same search checkpoint at n=16 scored **0.8023** in training but
**0.724** in the standalone sweep. Same weights, same n, same runner. Most likely EMA vs raw
weights (training rollouts use the EMA model; the eval script may not), but unconfirmed. **Do
not quote the search column until this is resolved** -- it would shift every search number.

**Other caveats:** single seed, one checkpoint per arm, and success rate is non-monotonic in
n (SEARCH/argmax test: 0.12, 0.20, 0.34, 0.18, 0.30), which is impossible without noise and
indicates the noise dominates the effect at this episode count.

## 5. Where the raw results are

| what | path (under `$DP_OUTPUT_ROOT/pusht_search/pusht_image_search`) |
|---|---|
| best-of-n curves + per-episode rewards | `bon_30demo/{search,bc}/step_00*/success_curve.json` |
| all selection modes merged | `bon_30demo/{search,bc}/success_curves.jsonl` |
| plotted curves | `bon_30demo/{search,bc}/step_00*/success_curve.png` |
| in-training metrics | `{outer_inner,offline}/*/logs.json.txt` |
| resolved configs | `{outer_inner,offline}/*/.hydra/config.yaml` |
| sweep stdout | `logs/bon_sweep_30demo.log` (repo) |
| full grid (corrected semantics) | `bon_grid_30demo/{search,bc}/step_00*/`, `logs/bon_grid_30demo.log` |
| per-candidate verifier distances | `bon_grid_30demo/*/step_00*/candidate_scores.jsonl` -- one row per policy call, so any selection rule can be replayed offline without re-running the sim |
| grid as measured BEFORE the 2026-08-18 fixes | `bon_grid_30demo.pre-nfix-*/`, `logs/bon_grid_30demo.log.pre-nfix-*` (kept as provenance for section 3; do not quote) |

Reproduce: `bash scripts/run_30demo_2x2.sh` then `bash scripts/bon_sweep_30demo.sh`.

<!-- BEGIN grid (generated by scripts/update_latest_with_grid.py) -->

## 6. Full checkpoint x selection x n grid

Every `step_*.ckpt` on the 10k grid x {argmax, softmax, final_pass} x n in {1,2,4,8,16}, both arms. **1/60 combos complete -- sweep still running, table is partial.**

This exists so a checkpoint never has to be pre-selected: pick on **val** and read **test** at the same row. Selecting by maximising test (which is how the section 2 checkpoints were chosen) inflates the reported number by 1-2 SE.

**n counts GENERATIONS, identically under all three rules**, so the columns are matched on compute. At n=1 there is nothing to select, so all three rules run the same empty-context conditional and the n=1 rows must agree exactly -- a disagreement there is a bug, not a result. (Grids generated before 2026-08-18 do not have this property: `final_pass` cost n+1 generations, and softmax drew from the sampler's RNG stream, so its n=1 column was a reseeded replicate rather than the same rollout.)

Regenerate: `python scripts/update_latest_with_grid.py`

<!-- END grid -->
