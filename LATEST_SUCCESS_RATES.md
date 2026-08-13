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

