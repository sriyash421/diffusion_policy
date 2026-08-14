# Latest PushT success rates

Running record of the **most recent, post-bugfix** runs. `SUCCESS_RATES.md` is the full historical archive across every arm and generation; this file is the short list of what is currently trustworthy, so a number here can be quoted without first checking which generation it came from.

**Add new runs to this file as they land.** Anything superseded should move out rather than accumulate — the point of a separate doc is that everything in it is current.

---

## What these runs incorporate

All runs below were trained after the fixes in `b487ec0`, each of which silently produced wrong numbers rather than raising:

| fix | what it silently did before |
|---|---|
| `lr_scheduler_kwargs` not forwarded (UNet workspace) | `decay_then_constant` used its `decay_steps: 10000` default whatever the config said — a 100k run configured to decay over 77k hit its floor at 13k and sat there for 87% of training |
| `max_gradient_steps` checked only between epochs | overshot by up to a full epoch, and never fired at all when the last epoch began under the cap (a 300,000 cap ended at 300,252) |
| no `max_gradient_steps` in the UNet workspace | `num_epochs` was the only bound, and an epoch is `ceil(n_windows/batch)` steps, so it needed hand-recomputing per demo budget |
| normalizer refit on resume | a resumed run re-derived statistics that `load_checkpoint` had already restored |
| `train_action_mse_error` over all 16 horizon steps | not comparable to the search workspace, which scores the 8 executed steps |
| eval harness required `predict_action_best` | could not score a policy without a search interface at all |

## The runs

| | search transformer (ST n=1) | diffusion UNet (BC) |
|---|---|---|
| policy | `PushTDiffusionSearchPolicy`, `max_actions: 1` | `DiffusionUnetImagePolicy` |
| config | `train_pusht_st_n1{,_29}` | `train_pusht_unet_bc{,_29}` |
| denoiser | transformer, 4 layer x 4 head x 256 emb | UNet, `down_dims [512,1024,2048]` |
| params (denoiser / total) | 5.9M / 17.1M | 282M / 293M |
| scheduler | DDIM | DDPM |

All four are **search width 1** — one action sampled, no candidates, no verifier — so the contrast is architecture and demo budget. Evaluated at **n=1 only**: the UNet has no verifier and cannot rank candidates, so best-of-n is undefined for it.

**Held equal**: committed split manifests (identical 29 / 100 training episodes, and the same 50 test episodes used everywhere else), LR (3k warmup to 1e-4, cosine to a 1e-5 floor at step 80k, then held), 100k steps, batch 32, `num_inference_steps: 100`, `diffusion_step_embed_dim: 256`, ResNet18 + GroupNorm encoder with `imagenet_norm: False`, EMA 0.995, eval protocol.

**Not equalisable**: denoiser capacity (48x) and sampler family (DDIM vs DDPM). See `diffusion_policy/ARCH_2x2_PARITY.md`.

> Evaluated `--skip-val`, so any checkpoint picked from these tables is picked on test, not held out.

---

## Binary success rate — TEST (50 episodes), n=1

| checkpoint | ST n=1 @29 | UNet BC @29 | ST n=1 @100 | UNet BC @100 |
|---|---|---|---|---|
| 20000 | 2% | 16% | 2% | 36% |
| 40000 | 0% | 12% | 4% | 56% |
| 60000 | 4% | 18% | 18% | 62% |
| 80000 | 2% | 18% | 28% | 56% |
| 100000 | 8% | 20% | 26% | 62% |

## Mean reward, episode max — TEST, n=1

The success column thresholds this at 1.0 (`coverage >= 95%`), so it is the same measurement without the cliff. Read it for trend: one episode is 2%, and much of the test set sits just under the threshold, which makes the binary column swing over a nearly-flat policy.

| checkpoint | ST n=1 @29 | UNet BC @29 | ST n=1 @100 | UNet BC @100 |
|---|---|---|---|---|
| 20000 | 0.370 | 0.628 | 0.265 | 0.912 |
| 40000 | 0.448 | 0.616 | 0.690 | 0.959 |
| 60000 | 0.494 | 0.606 | 0.780 | 0.915 |
| 80000 | 0.521 | 0.558 | 0.795 | 0.915 |
| 100000 | 0.546 | 0.585 | 0.784 | 0.912 |

## Mean reward, FINAL step — TEST, n=1

The same quantity at the last step rather than the episode maximum — it catches the policy that reaches the goal and then leaves.

| checkpoint | ST n=1 @29 | UNet BC @29 | ST n=1 @100 | UNet BC @100 |
|---|---|---|---|---|
| 20000 | 0.154 | 0.574 | 0.097 | 0.838 |
| 40000 | 0.256 | 0.467 | 0.432 | 0.919 |
| 60000 | 0.377 | 0.514 | 0.574 | 0.901 |
| 80000 | 0.416 | 0.486 | 0.673 | 0.888 |
| 100000 | 0.433 | 0.510 | 0.634 | 0.877 |

## Best within the matched 100k budget

| | 29 demos | 100 demos |
|---|---|---|
| ST n=1 | **18%** @ 70000 (rew 0.558) | **28%** @ 80000 (rew 0.795) |
| UNet BC | **20%** @ 100000 (rew 0.585) | **62%** @ 60000 (rew 0.915) |

## ST n=1 @29 beyond 100k — supplementary

This run reached 300k before the budget was capped at 100k; the other three stop there. Not part of the matched comparison. Included because it answers whether the transformer was simply undertrained — **it was not**: it never beats its own 18% at 70k, and mean reward decays from 0.558 to ~0.50.

| checkpoint | n=1 success | mean reward (max) |
|---|---|---|
| 120001 | 8% | 0.540 |
| 140001 | 6% | 0.549 |
| 160001 | 6% | 0.547 |
| 180001 | 8% | 0.506 |
| 200001 | 12% | 0.538 |
| 220001 | 10% | 0.529 |
| 240001 | 2% | 0.497 |
| 260001 | 10% | 0.475 |
| 280001 | 4% | 0.500 |
| 300001 | 4% | 0.534 |

---

## Sampler ablation — UNet BC at 8 inference steps

**Same weights, same 50 test episodes, eval-time only.** No retraining: both schedulers derive `alphas_cumprod` from the same betas, so `add_noise` — the only scheduler call `compute_loss` makes — is bit-identical between them (verified, `max|diff| = 0.0`). Only the reverse loop differs: DDPM injects noise at every step, DDIM is deterministic.

Verified from the recorded `episode_idxs`: all three settings, both budgets, all five checkpoints each — the same 50 episodes, identical to the manifest test split.

**n=1 success rate — TEST (50 episodes)**

| checkpoint | @29 DDPM 100 | @29 DDPM 8 | @29 DDIM 8 | @100 DDPM 100 | @100 DDPM 8 | @100 DDIM 8 |
|---|---|---|---|---|---|---|
| 20000 | 16% | 0% | 16% | 36% | 0% | 30% |
| 40000 | 12% | 0% | 14% | 56% | 0% | 52% |
| 60000 | 18% | 0% | 22% | 62% | 0% | 58% |
| 80000 | 18% | 0% | 8% | 56% | 0% | 42% |
| 100000 | 20% | 0% | 20% | 62% | 0% | 46% |

**Mean reward, episode max — TEST, n=1**

| checkpoint | @29 DDPM 100 | @29 DDPM 8 | @29 DDIM 8 | @100 DDPM 100 | @100 DDPM 8 | @100 DDIM 8 |
|---|---|---|---|---|---|---|
| 20000 | 0.628 | 0.131 | 0.610 | 0.912 | 0.130 | 0.850 |
| 40000 | 0.616 | 0.130 | 0.565 | 0.959 | 0.124 | 0.902 |
| 60000 | 0.606 | 0.129 | 0.584 | 0.915 | 0.138 | 0.866 |
| 80000 | 0.558 | 0.119 | 0.564 | 0.915 | 0.127 | 0.854 |
| 100000 | 0.585 | 0.130 | 0.587 | 0.912 | 0.132 | 0.847 |

**The collapse at 8 steps is the SCHEDULER, not the step count.** DDPM at 8 scores **0% at every checkpoint at both budgets** — final-step reward is exactly 0.000, i.e. the policy does not move the block toward the goal at all, and the ~0.13 max reward is incidental contact. Swapping to DDIM at the same 8 steps restores it.

At **29 demos DDIM-8 is indistinguishable from DDPM-100** (16/14/22/8/20 against 16/12/18/18/20). At **100 demos it costs a real but modest amount** — 46–58% against 56–62%, and ~6% on mean reward.

Mechanism: DDPM needs a dense trajectory to converge because it re-injects noise each reverse step, so truncating 100 → 8 leaves the sample unresolved. DDIM is built to subsample the trajectory, which is why the search transformer runs at 8 with no loss (the repo measured n=1 success identical at 8/16/32/100 on those weights).

**Practical upshot.** The UNet BC can run at **8 DDIM steps for a 12.5x sampling speedup**, free at 29 demos and ~10 points at 100. 16 or 32 DDIM steps would likely close that gap and are cheap to check, since this is eval-only.

It also means the headline tables above, which use the UNet's native DDPM at 100, are not penalising it for the sampler — a DDIM reading is at best equal and at 100 demos slightly worse.

Reproduce (the `--noise-scheduler` flag is a sampling-time override; results go to a separate directory because `save_outputs` merges by step and would overwrite the 100-step rows):

```bash
python eval_search_pusht.py -c <run>/checkpoints/step_XXXXXXX.ckpt \
    --min-n 1 --max-n 1 --skip-val --n-envs 50 --device cuda:0 \
    --num-inference-steps 8 --noise-scheduler ddim \
    -o <run>/bon_search_ddim8/step_XXXXXXX
```

## Reading these

**The UNet wins decisively at 100 demos** — 62% against 28%, and 0.959 against 0.796 on mean reward, on identical episodes with a matched schedule. It also learns much faster: 56% by step 40k, a level the transformer never reaches at any checkpoint.

**At 29 demos they converge** (20% vs 18%, indistinguishable at 50 episodes). So the architecture gap is a function of DATA BUDGET rather than a fixed offset: 29 -> 100 demos takes the UNet 20% -> 62% and the transformer only 18% -> 28%.

**Neither is usable at n=1.** The best cell here is 62%; the width-16 search arms in `SUCCESS_RATES.md` section 2 reach 78% at n=64 by step 1000. Nearly all the performance in this project comes from search width, not the single-sample policy.

**Noise floor.** 50 test episodes, so one episode is 2% and the standard error near 50% is ~7pp. Differences under ~14pp are not resolvable; read the mean-reward tables for trend and the success tables for magnitude.

**Known gap.** UNet BC @29's `val_loss` bottoms at step ~7k, before the first eval point at 20k, so its true peak may be unmeasured. Its 10k checkpoint is on disk if that needs closing.

## Provenance

| run | directory |
|---|---|
| ST n=1 @29 | `data/outputs/pusht_search/pusht_image_search/offline/stn1_demos-29-r8_seed-42` |
| UNet BC @29 | `data/outputs/pusht_search/pusht_image_search/unet_bc/unetbc_demos-29-r8_seed-42` |
| ST n=1 @100 | `data/outputs/pusht_search/pusht_image_search/offline/stn1_demos-100_seed-42` |
| UNet BC @100 | `data/outputs/pusht_search/pusht_image_search/unet_bc/unetbc_demos-100_seed-42` |

Curves are `<run>/bon_search/success_curves.jsonl`. Regenerate a row with:

```bash
python eval_search_pusht.py -c <run>/checkpoints/step_XXXXXXX.ckpt \
    --min-n 1 --max-n 1 --skip-val --n-envs 50 --device cuda:0
```

