# Latest PushT success rates

The current generation: the **30-demo search-vs-BC pair**, trained 2026-08-17/18 after the
search-procedure unification. Older generations are in `SUCCESS_RATES.md` (section B holds
what used to be in this file, with the list of changes that invalidated it).

`success = coverage >= 95%`; `mean reward = clip(coverage/0.95, 0, 1)` averaged over
episodes. **At 50 test episodes the SE is ~7pp, so gaps under ~14pp are not resolvable.**
Most differences below are inside that band -- read the caveats before quoting anything.

## The two arms

Both are `PushTDiffusionSearchPolicy` with a **byte-identical** obs encoder and denoiser
(verified from each run's `.hydra/config.yaml`): ResNet18 `IMAGENET1K_V1`, `crop_shape
[76,76]`, `random_crop`, `use_group_norm`, `imagenet_norm: False`; 4 layers / 2 cond layers
/ 4 heads / 256 emb, `p_drop_attn 0.2`, `causal_attn: True`, `cond_encoder: gpt2`, DDIM at
8 inference steps. **The only difference is `max_actions`: 16 vs 1.**

| | ST (search) | BC |
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

**These are NOT a matched comparison**: ST evaluated at n=16 (search active), BC at n=1.
Use section 2 for like-for-like.

| arm | 20k | 40k | 60k | 80k | 100k |
|---|---|---|---|---|---|
| ST test  | 0.7208 | 0.7755 | 0.8023 | 0.745 | 0.6143 |
| ST val   | 0.8143 | 0.7574 | 0.7539 | 0.7708 | 0.7362 |
| BC test  | 0.4957 | 0.612 | 0.6029 | 0.6382 | - |
| BC val   | 0.4577 | 0.5681 | 0.6342 | 0.6397 | - |

BC's 100k row is blank because **the offline trainer skips its final rollout**: its rollouts
drift +1 per fire (20001, 40002, 60003, 80004) so the next trigger lands at 100005, past the
`max_gradient_steps: 100000` cap. Evaluated separately from `step_0100000.ckpt`: test mean
reward 0.593, success 0.240. `TrainSearchOuterInnerWorkspace` hits its boundaries exactly and
does not have this bug.

**Both arms are over-trained at 100k.** ST peaks at 60k (0.8023) and falls 19pp by 100k; BC
peaks at 80k. 60-80k would have been the right budget at this demo count.

## 2. Best-of-n x selection rule (matched n, both arms)

ST from `step_0060000.ckpt`, BC from `step_0080000.ckpt` -- each arm's best in-training
checkpoint. n in {1,2,4,8,16} x {argmax, softmax, final_pass}, `eval_search_pusht.py`,
seed 42.

### TEST -- mean reward (max)

| n | ST/argmax | ST/softmax | ST/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.599 | 0.668 | 0.663 | 0.624 | 0.590 | 0.585 |
| 2 | 0.670 | 0.731 | 0.709 | 0.647 | 0.612 | 0.613 |
| 4 | 0.757 | 0.670 | 0.734 | 0.660 | 0.679 | 0.576 |
| 8 | 0.707 | 0.667 | 0.728 | 0.728 | 0.618 | 0.602 |
| 16 | 0.724 | 0.790 | 0.660 | 0.644 | 0.637 | 0.664 |

### TEST -- success rate

| n | ST/argmax | ST/softmax | ST/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.120 | 0.160 | 0.120 | 0.260 | 0.200 | 0.180 |
| 2 | 0.200 | 0.340 | 0.160 | 0.240 | 0.300 | 0.240 |
| 4 | 0.340 | 0.240 | 0.100 | 0.300 | 0.380 | 0.240 |
| 8 | 0.180 | 0.200 | 0.140 | 0.400 | 0.180 | 0.200 |
| 16 | 0.300 | 0.340 | 0.200 | 0.460 | 0.420 | 0.280 |

### VAL -- mean reward (max)

| n | ST/argmax | ST/softmax | ST/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.680 | 0.683 | 0.723 | 0.677 | 0.644 | 0.601 |
| 2 | 0.743 | 0.769 | 0.740 | 0.663 | 0.725 | 0.641 |
| 4 | 0.783 | 0.731 | 0.760 | 0.686 | 0.670 | 0.703 |
| 8 | 0.739 | 0.732 | 0.732 | 0.777 | 0.719 | 0.613 |
| 16 | 0.780 | 0.751 | 0.735 | 0.786 | 0.772 | 0.643 |

### VAL -- success rate

| n | ST/argmax | ST/softmax | ST/final_pass | BC/argmax | BC/softmax | BC/final_pass |
|---|---|---|---|---|---|---|
| 1 | 0.333 | 0.200 | 0.133 | 0.267 | 0.133 | 0.033 |
| 2 | 0.233 | 0.133 | 0.200 | 0.167 | 0.267 | 0.233 |
| 4 | 0.333 | 0.167 | 0.200 | 0.200 | 0.167 | 0.200 |
| 8 | 0.200 | 0.300 | 0.167 | 0.200 | 0.367 | 0.200 |
| 16 | 0.233 | 0.367 | 0.233 | 0.400 | 0.267 | 0.267 |

## 3. What this supports, and what it does not

**Supported:**
* **Best-of-n gives a real gain on mean reward** for both arms -- roughly +0.06 to +0.16
  from n=1 to n=16 in most columns.
* **`final_pass` is the weakest selection rule.** Lowest or near-lowest in most columns on
  both arms; it never wins on test success.

**NOT supported by this data:**
* **That the search-trained policy beats BC.** At matched n the ST advantage largely
  disappears -- on test success at n=16, BC/argmax (0.460) is the best cell in the table,
  above ST/argmax (0.300). The apparent +0.164 in section 1 is ST-at-16 vs BC-at-1, which
  conflates the trained policy with eval-time search budget.
* **Any ranking of argmax vs softmax.** They trade places by n and by split.
* **Any single cell.** CIs are ~+/-0.12 at 50 episodes and nearly every pairwise difference
  overlaps.

**Open discrepancy:** the same ST checkpoint at n=16 scored **0.8023** in training but
**0.724** in the standalone sweep. Same weights, same n, same runner. Most likely EMA vs raw
weights (training rollouts use the EMA model; the eval script may not), but unconfirmed. **Do
not quote the ST column until this is resolved** -- it would shift every ST number.

**Other caveats:** single seed, one checkpoint per arm, and success rate is non-monotonic in
n (ST/argmax test: 0.12, 0.20, 0.34, 0.18, 0.30), which is impossible without noise and
indicates the noise dominates the effect at this episode count.

## 4. Where the raw results are

| what | path (under `$DP_OUTPUT_ROOT/pusht_search/pusht_image_search`) |
|---|---|
| best-of-n curves + per-episode rewards | `bon_30demo/{search,bc}/step_00*/success_curve.json` |
| all selection modes merged | `bon_30demo/{search,bc}/success_curves.jsonl` |
| plotted curves | `bon_30demo/{search,bc}/step_00*/success_curve.png` |
| in-training metrics | `{outer_inner,offline}/*/logs.json.txt` |
| resolved configs | `{outer_inner,offline}/*/.hydra/config.yaml` |
| sweep stdout | `logs/bon_sweep_30demo.log` (repo) |

Reproduce: `bash scripts/run_30demo_2x2.sh` then `bash scripts/bon_sweep_30demo.sh`.

