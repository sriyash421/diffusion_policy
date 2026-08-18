# ARCHIVED success rates (as of 2026-08-17)

Everything below was produced BEFORE the search-procedure unification and the config
changes of 2026-08-17, and is not comparable to anything measured after it. Preserved
verbatim for provenance; `LATEST_SUCCESS_RATES.md` carries the current numbers.

## What invalidated these

1. **obs encoder** `weights: null` -> `IMAGENET1K_V1` on all three PushT arms (search,
   Gaussian, UNet BC). The crop and `imagenet_norm: False` are unchanged.
2. **`train_pusht_st_n1` width** 6 layers / 8 heads / 1024 emb -> 4 / 4 / 256 (126.58M ->
   5.94M denoiser), matching online search and the search arm.
3. **`corrupt_obs_eval: False`** on the six corruption arms. They previously corrupted
   observations at EVAL as well as at train, with a fresh random noise level per candidate,
   because only `OnlineSearchPolicy` guarded on `self.training`. Corruption is now a
   training augmentation only.
4. **Eval width** `n_search_actions` 8 -> `${n_candidates}`. Every arm now evaluates at the
   width it trained at. In particular `_cd_k4` trained at K=4 but evaluated at n=8, i.e.
   in the rolling-window regime its training never covered, so its published k=4 numbers
   were not a clean K=4 measurement.
5. **Demo budgets** 25 / 29 -> 30 / 100, with the new 30 a strict subset of the 100-demo
   train set (see below).
6. **maze `monitor_key`** `train_loss` -> `val_loss`, so maze stops selecting its most
   overfit checkpoint. Checkpoint `format_str` precision `.3f` -> `.4f`.
7. **`slot_weight_decay`** `1.0` -> `False`. Numerically identical (both mean uniform
   slots); listed only so the config diff is accounted for.

## Caveats that were always true and never recorded

* **The 29-vs-100 columns are not a clean data-scaling comparison.** Measured on the
  manifests: the 29-demo train set is NOT a subset of the 100-demo train set, the 25-demo
  set is not a subset of the 29-demo set, and the three do not share a val set. Only the
  50-episode test set is common. So those columns vary *which* episodes and *which* val
  set as well as how many. The replacement `pusht_seed42_train30.json` is a strict subset
  of `train100`'s train episodes and reuses its val (30) and test (50) verbatim, so
  30-vs-100 isolates dataset size alone.
* **`pusht_seed42.json` was an exact duplicate** of `pusht_seed42_train100.json` (identical
  train/val/test lists).
* **The UNet arm logs none of the search metrics.** `nrmse_*` and `action_value_*` come
  from `_search_action_nrmse`, which only runs for policies passing `_is_search_policy`. So
  the 2x2 was only ever comparable on `train_loss`, `val_loss`, `train_action_mse_error`
  and rollout success rate; `nrmse_min` has no UNet counterpart by construction.
* **`nrmse_*` is normalised RMSE; `train_action_mse_error_*` is MSE**, both over the same
  executed window. Different units under similar names -- never plot them on one axis.
* **maze has no environment evaluation at all.** Both maze tasks use `DummyRunner`, whose
  `run()` returns `{}`. Every maze number is a loss, never a success rate.

## Provenance

Split manifests for the retired budgets are kept on disk even though no config points at
them: `pusht_seed42_train25.json`, `pusht_seed42_train29_val30.json`,
`pusht_seed42_legacy_val10_train29.json`. The configs that used them are in
`diffusion_policy/config/archive_config/`.

---

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

## ST n=1 @100 with a causal GPT-2 conditioning encoder

A re-run of `train_pusht_st_n1` after two changes to `train_pusht_diffusion_search`, which
every PushT ST config inherits. Same 100 training episodes, same 50 test episodes, same LR
schedule, optimizer, EMA, batch size, encoder and `num_inference_steps: 100`.

| change | before | after |
|---|---|---|
| `policy.cond_encoder` | *(absent)* — `nn.TransformerEncoder` under `_build_encoder_mask` | `gpt2` — a causal `GPT2Model` over the same token stream, no hand-built mask, causality from GPT-2's own triangular mask and a padding mask as the only input. Matches how `search_policy.py` / `online_search_policy.py` encode their context |
| `policy.causal_attn` | `False` | `True` — within a candidate the 16 horizon steps go from bidirectional to autoregressive |

**Parameter-neutral**: GPT-2 at `n_embd 256 / n_layer 2` is 1.58M, the same as the
`nn.TransformerEncoder` it replaces (total 5.9483M vs 5.9482M). One connectivity edge is
lost — obs step 0 can no longer see obs step 1; everything else was already what causal
masking gives.

| checkpoint | ST n=1 @100 (old) | **ST n=1 @100 (gpt2+causal)** | rew max, old → new | rew final, old → new |
|---|---|---|---|---|
| 20000 | 2% | **4%** | 0.265 → 0.417 | 0.097 → 0.199 |
| 40000 | 4% | **20%** | 0.690 → 0.651 | 0.432 → 0.493 |
| 60000 | 18% | **24%** | 0.780 → 0.755 | 0.574 → 0.641 |
| 80000 | 28% | **30%** | 0.795 → 0.752 | 0.673 → 0.642 |
| 100000 | 26% | **32%** | 0.784 → 0.758 | 0.634 → 0.673 |

**Success is up at all ten checkpoints** (the 10k-grid rows omitted above: 30k 2%→10%,
50k 16%→26%, 70k 24%→26%, 90k 24%→30%), and it reaches the old run's *final* 26% by step
50k — half the compute. It is also **still rising at 100k** (30%→32%, val 43%→47%) where the
old run peaked at 80k and declined. A 300k extension is running to settle that.

**Read the reward columns against the success column, because they disagree.** Final-step
reward improves nearly everywhere, but episode-max reward is a wash and slightly *worse*
past 60k. On this doc's own reading — max is what the policy achieves, final is whether it
stays there — the change buys goal-*holding*, not a higher peak. Success thresholds the max,
so some of the success gain is episodes crossing 95% coverage that were already close.

**Two caveats before quoting this.** (1) By this file's own noise floor — 50 episodes, ~7pp
SE, differences under ~14pp unresolvable — only the 40k row (4%→20%) clears the bar on its
own. The claim worth making is the consistent direction across all ten checkpoints plus the
still-rising trend, not any single cell. (2) Two variables moved at once, so this arm cannot
attribute the gain between the encoder swap and the decoder causality.

Unlike every other run in this file, this one was evaluated **with** val (30 episodes), so it
is the only arm here that can be checkpoint-selected on held-out data: val peaks at 46.7% at
step 100k, giving a val-selected test of **32%**. That column is not comparable to the
`--skip-val` arms above.

Requires `transformers` (newly declared in `conda_environment.yaml`; the import is lazy, so
arms not using `cond_encoder: gpt2` are unaffected). Checkpoints are not interchangeable with
the pre-change architecture — `causal_attn: True` adds a `model.mask` buffer and the trunk
swaps `model.encoder.layers.*` for `model.encoder.h.*`, so an older run cannot be resumed
across this, only evaluated from its own embedded cfg.

---

## ST n=1 @100 at 21x denoiser capacity — no measurable gain

Trained 2026-08-14/15, 100k steps, all 10 checkpoints evaluated.

**Result: 21x the denoiser parameters bought nothing at 100 demos.** Final checkpoint is
**32% against the baseline's 32%**, peak is 32% on both, and the 60k-100k plateau means are
24.4% against 28.4% — a difference well inside this file's ~14pp resolution bar, in the
baseline's favour. Mean reward finishes at 0.770 against 0.758. The capacity branch of the
question is answered in the negative; see "What this rules out" below.

`train_pusht_st_n1` itself moved, so this is now the DEFAULT transformer arm rather than a
side experiment, and `train_pusht_st_n1_29` follows it. The right baseline is the
**gpt2+causal** arm directly above, not the original: against that one only two things
change, both in this file.

| change | before (gpt2+causal) | after |
|---|---|---|
| `policy.n_layer` | 4 | **6** |
| `policy.n_head` | 4 | **8** |
| `policy.n_emb` | 256 | **1024** |
| `policy.p_drop_attn` | 0.2 | **0.1** |

Everything else is inherited untouched: same 100 training episodes and same 50 test episodes
from `pusht_seed42_train100.json`, `cond_encoder: gpt2`, `causal_attn: True`, DDIM at
`num_inference_steps: 100`, `max_actions: 1`, AdamW 1e-4 / betas [0.95, 0.999] / wd 1e-6,
`decay_then_constant` with 3k warmup and `decay_steps: 77000`, 100k steps, batch 32, EMA
0.995, `gradient_clip_norm: 1.0`, ResNet18 + GroupNorm encoder at crop [76,76] with
`imagenet_norm: False`.

**What the capacity change actually is**, measured by instantiating from the config:

| | obs encoder | denoiser | decoder | cond trunk | total | ckpt | vs UNet denoiser |
|---|--:|--:|--:|--:|--:|--:|--:|
| ST n=1, before | 11.18M | 5.94M | 4.21M | 1.58M | 17.12M | 0.27 GB | 0.021x |
| **ST n=1, this run** | 11.18M | **126.58M** | 100.78M | 25.20M | **137.76M** | 2.20 GB | **0.449x** |
| UNet BC | 11.18M | 282.18M | — | — | 293.36M | 4.40 GB | 1.000x |

This is the first ST arm in the same capacity *order* as the UNet rather than 47x below it.
It still does not reach parity, deliberately — matching 282M needs ~12 layers at n_emb 1024
(227M), a model neither paper used. The question is whether the *slope* of success against
capacity is steep, not whether parity is reachable.

The GPT-2 trunk stays parameter-neutral at the new width too: **25.20M** against the
`nn.TransformerEncoder`'s 25.19M, the same like-for-like the 256-wide arm above reports.

**Two things to hold against the result.**

1. **Two variables moved.** Dropout dropped with capacity, so a gain does not attribute
   between them. `train_pusht_st_n1_drop001` — 0.01 dropout pinned to the OLD architecture —
   is the other half of that pair.
2. **Head dim changed, 64 → 128** (1024/8, where it was 256/4). This is therefore not a pure
   widening: each head attends in a 128-d subspace, so the inherited LR schedule transfers
   less cleanly than it would at `n_head: 16`, which would hold head dim at 64 for the same
   width and the *same* parameter count. `n_head: 16` is the first thing to try if this
   trains badly.

**Prediction, stated before the result.** If capacity is what limits this policy, success
rises materially above the gpt2+causal arm's 32% at 100k. If nothing moves, capacity is not
the driver and what remains is conditioning *bandwidth* — 6 cross-attention read-outs from a
2-token memory against the UNet's 12 FiLM injections of an unprojected 1060-d vector — which
points at the obs projection and `n_cond_layers`, not at model size. If it gets **worse**,
suspect overfitting before concluding anything about capacity: 126.58M parameters against
12,199 training windows, with dropout reduced.

Both arms are evaluated **with** val (30 episodes), so unlike the four `--skip-val` arms
higher up, these two are directly comparable on both columns.

| checkpoint | base success | **this success** | base rew max | **this rew max** | base val | **this val** |
|---|---|---|---|---|---|---|
| 10000 | 0% | **0%** | 0.180 | **0.161** | 0.0% | **0.0%** |
| 20000 | 4% | **2%** | 0.417 | **0.262** | 3.3% | **0.0%** |
| 30000 | 10% | **6%** | 0.625 | **0.485** | 23.3% | **3.3%** |
| 40000 | 20% | **4%** | 0.651 | **0.541** | 33.3% | **10.0%** |
| 50000 | 26% | **10%** | 0.680 | **0.562** | 23.3% | **10.0%** |
| 60000 | 24% | **16%** | 0.755 | **0.717** | 36.7% | **26.7%** |
| 70000 | 26% | **20%** | 0.767 | **0.719** | 33.3% | **23.3%** |
| 80000 | 30% | **30%** | 0.752 | **0.771** | 40.0% | **40.0%** |
| 90000 | 30% | **24%** | 0.800 | **0.762** | 43.3% | **43.3%** |
| 100000 | 32% | **32%** | 0.758 | **0.770** | 46.7% | **30.0%** |
| _60k-100k mean_ | _28.4%_ | _**24.4%**_ | _0.766_ | _**0.748**_ | | |

### Reading the curve

**It converged much more slowly, then landed in the same place.** The mean-reward gap is the
cleanest view — it opens early, closes monotonically, and ends at zero:

| step | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| rew max gap | -0.155 | -0.140 | -0.110 | -0.118 | -0.038 | -0.048 | **+0.019** | -0.038 | **+0.012** |

The 40k row (20% vs 4%) looked like a divergence at the time and was a lag: this arm needed
~80k steps to reach what the baseline had at 60k. That is the expected shape for a 21x
parameter jump inheriting warmup and an LR schedule tuned for 5.94M, and it means the early
checkpoints of a scaled arm cannot be read as a result.

**Final-step reward is the one place it is consistently worse** — 0.649 against 0.673 at 100k,
and further behind at every earlier checkpoint (0.347 vs 0.593 at 50k). Episode-max reward is
level while final-step is not, which is the "approaches the goal and drifts off rather than
committing" signature `train_pusht_st_n1_drop001.yaml` describes for the original arm. More
capacity did not fix it.

**val_loss overfits, and it does not matter.** It bottomed at **0.0562 at 35k** and rose to
~0.077 by 80k while train_loss sat at 0.034 — a 2.2x train/val gap, textbook for 126.58M
parameters on 12,199 windows. Success rose 4% -> 30% over exactly that interval. This is the
same anti-correlation the UNet shows (val_loss 0.042 -> 0.107 while success climbed to 62%)
and the reason this file says never to select on it.

### What this rules out

The prediction stated before the run had three branches. The second one is what happened:
success did not rise, so **capacity is not what limits this policy at 100 demos**. What
remains is conditioning *bandwidth* — 6 cross-attention read-outs from a 2-token memory,
against the UNet's 12 FiLM injections of an unprojected 1060-d vector — which points at the
obs projection and `n_cond_layers`, not at model size. Note the 25.20M GPT-2 trunk in this
arm is spent encoding a memory that is **two tokens** wide.

It does *not* rule out capacity mattering at a larger demo budget, at a different LR schedule,
or past 100k. It rules out capacity being the free win at this budget on this schedule.

### Follow-ups this points at, cheapest first

1. **Hold `p_drop_attn: 0.2`** with the capacity increase. Dropout dropped alongside capacity
   here, and the val_loss evidence says that was the wrong direction — this is the cheaper
   half of the two-variable change to undo, and it makes the arm one-variable.
2. **`n_head: 16`** — restores head dim to 64 at an identical parameter count, removing the
   one respect in which this was not a pure widening.
3. **`n_cond_layers`** rather than `n_emb`, if bandwidth is the real constraint.

Given the null result, **reverting `train_pusht_st_n1` to n_emb 256 is defensible**: the
larger model costs 21x the parameters, 8x the checkpoint (2.20 GB vs 0.27 GB) and a slower
step rate for no measured gain. It is left in place only because 1 and 2 above are untested.

**val_loss has turned, but that is not decisive here.** It bottomed at ~0.0593 (32k) and has
risen since — 0.0617 (51k), 0.0644 (52k), 0.0630 (56k), 0.0685 (57k) — which is the
overfitting signature this section said to check for, and it appeared later than the flat
trace through 32k suggested. It is *not* conclusive, because on these arms val_loss is
anti-correlated with success: the UNet's rose 0.042 → 0.107 while its success climbed to 62%,
which is why this file says not to select on it. Success on this arm is still climbing over
the same interval (4% at 40k → 10% at 50k), so the turn is consistent with the documented
pattern rather than with a model coming apart. Worth watching, not yet worth acting on.

Working reading: the larger model **learns more slowly**, consistent with a 21x parameter jump
inheriting warmup and an LR schedule tuned for 5.94M, and with head dim moving 64 → 128. The
baseline plateaus at 26–32% from 50k on, so convergence by 100k is still open — but at 50k
this arm is at 10% against 26%, so closing that needs a steeper climb than it has shown yet.
If it lands materially short at 100k, the schedule is the first suspect and `n_head: 16` —
head dim back to 64 at an identical parameter count — is the cheapest next test.

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

**The UNet wins decisively at 100 demos** — 62% against 28%, and 0.959 against 0.796 on mean reward, on identical episodes with a matched schedule. It also learns much faster: 56% by step 40k, a level the transformer never reaches at any checkpoint. (The gpt2+causal ST above lifts that 28% to 32%, which narrows the gap without changing this conclusion — the UNet is still ~2x at the same budget. Raising the ST denoiser 21x on top of that, to 126.58M, moves it no further: 32% again. So the remaining gap is **not** explained by the 47x capacity difference that `ARCH_2x2_PARITY.md` lists as the largest confound in the 2x2 — the one thing that had never been tested is now tested, and it is not the answer.)

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
| ST n=1 @100 (gpt2+causal) | `data/outputs/pusht_search/pusht_image_search/offline/stn1gpt2_demos-100_seed-42` |
| ST n=1 @100 (gpt2+causal+21x) | `data/outputs/pusht_search/pusht_image_search/offline/stn1gpt2big_demos-100_seed-42` |
| UNet BC @100 | `data/outputs/pusht_search/pusht_image_search/unet_bc/unetbc_demos-100_seed-42` |

Curves are `<run>/bon_search/success_curves.jsonl`. Regenerate a row with:

```bash
python eval_search_pusht.py -c <run>/checkpoints/step_XXXXXXX.ckpt \
    --min-n 1 --max-n 1 --skip-val --n-envs 50 --device cuda:0
```

