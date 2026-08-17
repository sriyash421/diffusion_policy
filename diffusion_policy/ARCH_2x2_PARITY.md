# The architecture 2x2: what is held equal and what is not

> **2026-08-17 — superseded numbers.** Everything measured before the search-procedure
> unification of 2026-08-17 is archived in `ARCHIVED_SUCCESS_RATES_AUG17.md`, which lists
> the seven changes that invalidated it. The parity notes below have been updated to the
> current arms; the "Not equal, deliberately" rows in particular changed.
>
> Corrections landed at the same time:
> * `train_pusht_bc_29_r8` (referenced below) never existed; `train_pusht_bc_29` has been
>   retired to `archive_config/` along with every other 25/29-demo config.
> * `checkpoint_every` is 10,000 on BOTH arms, not 1,000 vs 10,000.
> * `max_gradient_steps` is 100,000, not 300,000.
> * `causal_attn` and `cond_encoder` still exist; `search_context` lost its `state` and
>   `state_value` modes (no config ever used them), leaving value / subgoal / subgoal_value.
> * The 29-vs-100 comparison carried an unrecorded confound: the 29-demo train set is not a
>   subset of the 100-demo one and the two do not share a val set. The replacement 30-demo
>   split is a strict subset of the 100 with identical val and test.


Four runs, all at **search width 1**, varying architecture and demo budget:

| config | policy | demos | run dir |
|---|---|---|---|
| `train_pusht_st_n1` | `PushTDiffusionSearchPolicy` (transformer, DDIM) | 100 | `offline/stn1gpt2big_demos-100_seed-42` |
| `train_pusht_st_n1_29` | same | 29 (r8) | `offline/stn1gpt2big_demos-29-r8_seed-42` |
| `train_pusht_unet_bc` | `DiffusionUnetImagePolicy` (UNet, DDPM) | 100 | `unet_bc/unetbc_demos-100_seed-42` |
| `train_pusht_unet_bc_29` | same | 29 (r8) | `unet_bc/unetbc_demos-29-r8_seed-42` |

The point of the 2x2 is that **architecture** is the variable. Anything else that differs
between the two columns is a confound, so it is either equalised or written down here.

## The transformer side changed on 2026-08-14. Read this before comparing anything.

`train_pusht_st_n1` is not the model that produced the transformer columns of
LATEST_SUCCESS_RATES.md. **Four** things moved at once, and no single result attributes
among them:

| | was | now | set in |
|---|---|---|---|
| capacity | 4 layers x 4 heads x 256 emb, 5.94M denoiser | 6 x 8 x 1024, **126.58M** | `train_pusht_st_n1` |
| `p_drop_attn` | 0.2 | **0.1** | `train_pusht_st_n1` |
| `causal_attn` | False | **True** | `train_pusht_diffusion_search` |
| `cond_encoder` | `transformer` (`nn.TransformerEncoder`) | **`gpt2`** (causal `GPT2Model`) | `train_pusht_diffusion_search` |

Two of those reach the *decoder* and two reach the *conditioning memory*:

- **`causal_attn: True`** switches on the second clause of `_build_tgt_mask`, so within a
  candidate horizon step *i* attends only to *j <= i*. The 16 action steps were
  bidirectional and are now autoregressive. It also adds a `model.mask` buffer, which is why
  pre-2026-08-14 checkpoints cannot be *resumed* under these configs (they remain evaluable —
  `create_from_checkpoint` rebuilds from the cfg in the payload).
- **`cond_encoder: gpt2`** replaces the memory encoder with a causal `GPT2Model` over one
  flat token stream, with no hand-built mask. **Like-for-like on parameters** at both
  capacities — 1.58M vs 1.58M at n_emb 256, 25.20M vs 25.19M at 1024 — because the model
  passes `vocab_size=1`; `GPT2Config`'s 50257 default would have added a dead `wte`. Exactly
  one edge of connectivity disappears: obs step 0 can no longer see obs step 1. Everything
  else was already what causality gives.

Neither reaches the UNet arms (`DiffusionUnetImagePolicy` has no such knob), so **the 2x2 now
varies horizon causality and memory trunk as well as architecture**. Runs from here are a new
generation; the run dirs carry `gpt2big` to keep them separable from the reported ones.

`train_pusht_st_n1_drop001` is pinned back to the old architecture in its own policy block,
because its premise is one variable against `stn1_demos-100_seed-42`.

## Held equal

| | value | note |
|---|---|---|
| data | the committed manifests | `pusht_seed42_train100.json` / `pusht_seed42_train29_val30.json`; the 50 test episodes are byte-identical in both |
| task | `pusht_image_search` | all four share one dataset, one env runner, one split source. **Not** `pusht_image`, which has no `split_file` and derives a 2-way split from the seed at `train_ratio: 0.2` (~31 episodes) |
| LR | 1e-4 max, 3k warmup, cosine to a 1e-5 floor reached at step 80,000, then held | `decay_then_constant`, decay_steps 77000, min_lr_ratio 0.1. A pure function of the step index, so it is safe to resume and to extend |
| budget | **exactly 100,000 gradient steps**, all four | `training.max_gradient_steps`, enforced mid-epoch. See below — this was not exact until 2026-08-12, and the cap was 300,000 when that section was written |
| optimizer | AdamW, betas [0.95, 0.999], wd 1e-6 | |
| EMA | flat decay 0.995 | the UNet's `EMAModel` is clamped `min_value == max_value` to match the search line's `training.ema_decay`, instead of its step-dependent warmup curve |
| obs encoder | ResNet18, random init, GroupNorm, crop 76x76, `imagenet_norm: False` | |
| horizon / n_obs_steps / n_action_steps | 16 / 2 / 8 | |
| **`num_inference_steps`** | **100** | changed 2026-08-11, see below |
| **batch size** | **32** | changed 2026-08-11, see below |
| `diffusion_step_embed_dim` | **256** | the UNet's, matched to the transformer's `n_emb` (`SinusoidalPosEmb(n_emb)`). ARCHITECTURE-CHANGING: checkpoints written at the old 128 cannot be loaded, and the runs behind SUCCESS_RATES.md section 5 used 128 (`data/outputs/_discarded/`). since 2026-08-14 `train_pusht_st_n1` runs `n_emb: 1024` and so no longer matches this — the UNet needs no retraining, but the *reason* for this row no longer holds |
| eval | n=1, 50 test episodes, `eval_search_pusht.py --skip-val` | |

## Changes made 2026-08-11 for parity

**`num_inference_steps`: 8 -> 100 on the search transformer.** Was 8 across the whole
search line, because there a training step samples `max_actions-1` candidates and 100 steps
would be 12.5x the dominant cost. That constraint does not exist at width 1: no candidates
are generated, `conditional_sample` is never called by the loss, so the gradient step is
unchanged. The UNet's DDPM stays at its native 100 — DDPM at 8 steps is not the same kind of
setting that DDIM at 8 is — so raising the transformer is what makes the samplers equal
rather than lowering the UNet.

- Costs: rollout and eval are 12.5x slower. Training is not affected.
- **Consequence:** the width-1 arms are no longer sampler-matched to the width-16 arm
  `value (oi)`, which runs at 8. Width-1-vs-width-16 comparisons now carry that difference.
  The repo measured n=1 success identical at 8/16/32/100 on frozen BC weights
  (`scripts/slurm/diag_bc_n1.sbatch`), so the effect is expected to be nil — but that was
  measured on the transformer, never on the UNet.

**Batch size: 64 -> 32 on the UNet.** 64 is the upstream UNet recipe; 32 is the search
line's, pinned there originally because the verifier dominated step cost. Matched to 32.

- At the time this made `num_epochs` load-bearing, because it was the only thing bounding
  the run. That is no longer true — see the next section.

## Changes made 2026-08-12: the step count is now a setting

The run length used to be an *epoch* count that had to be recomputed by hand, because an
epoch is `ceil(n_windows / batch_size)` steps and so moves with both the batch size and the
demo budget. Two defects came out of that:

- `TrainDiffusionUnetImageWorkspace` had **no `max_gradient_steps` at all**. `num_epochs`
  was its only bound.
- `TrainMLPImageWorkspace` had one, but checked it **only between epochs** — so it
  overshot by up to a full epoch, and whenever the last epoch began below the cap it never
  fired. At 382 steps/epoch the 786th epoch began at step 299,870, under the 300,000 cap, so
  the cap did nothing and the run ended at 300,252.

Both now enforce the cap **mid-epoch**, and every config sets `training.max_gradient_steps`
explicitly with `num_epochs: 100000` as a non-binding safety bound (the pattern
`train_pusht_bc.yaml` already used). Changing the batch size or the demo budget no longer
changes the run length, and no config comment carries arithmetic that can drift.

The cap was **300000** when this section was written; it is **100000** on all four arms today,
which is the grid LATEST_SUCCESS_RATES.md reports on. The numbers below are the 2026-08-12
measurement at the then-current 300k and are kept as the record of the bug, not as the
current setting.

| | windows | steps/epoch @32 | old end | new end |
|---|--:|--:|--:|--:|
| 100 demos | 12,199 | 382 | 300,252 | 300,000 |
| 29 demos | 3,504 | 110 | 300,080 | 300,000 |

Two related keys that the UNet workspace was also silently ignoring were fixed at the same
time. Neither changes what these four runs do:

- **`training.lr_scheduler_kwargs`** was not forwarded to `get_scheduler` until 2026-08-12,
  so `decay_then_constant` silently used its defaults (`decay_steps=10000`) — a run
  configured to decay over 77k instead hit its floor by 13k.
- **`training.gradient_clip_norm`** had no `clip_grad_norm_` call behind it, so setting it
  did nothing. Now honored, but deliberately left unset on the UNet arms (see below).
- **The normalizer was refit from the dataset on every resume**, overwriting the statistics
  `load_checkpoint` had just restored. Harmless while the data is fixed
  (`get_normalizer(mode='limits')` is deterministic over a manifest-pinned train set), but a
  silent corruption if the split or the zarr ever changed under a `resume: True` run. The
  resumed normalizer now wins, matching `TrainMLPImageWorkspace`.

## Not equal, deliberately

- **Model capacity.** The UNet is `down_dims: [512, 1024, 2048]`; the transformer is 4
  layers x 4 heads x 256 emb. These are not matched, and cannot be without changing one
  architecture into something neither paper used. Measured, and to be reported alongside
  any result:

  | | obs encoder | denoiser | total | vs UNet denoiser |
  |---|--:|--:|--:|--:|
  | ST-n1, pre-2026-08-14 | 11.18M | 5.94M | **17.12M** | 0.021x |
  | ST-n1, current | 11.18M | 126.58M | **137.76M** | 0.449x |
  | UNet BC | 11.18M | 282.18M | **293.36M** | 1.000x |

  The encoders are identically configured. The denoiser was **47x** larger on the UNet side;
  after the 2026-08-14 capacity raise it is **2.2x**. Within the current transformer the
  126.58M splits 100.78M decoder / 25.20M GPT-2 conditioning trunk / 0.60M embeddings.

  Re-measured 2026-08-14 by instantiating each `obs_encoder` from its own config and the
  denoiser from the resolved arch keys. Two corrections to the figures previously here: the
  UNet denoiser is **282.18M**, not 278.1M — that number was taken before
  `diffusion_step_embed_dim` went 128 -> 256, which widens the FiLM conditioning vector
  1188 -> 1316 and so every `cond_encoder` projection (278.12M at 128, confirmed). Checkpoint
  sizes follow the same ratio: 0.27 / 2.20 / 4.40 GB, each carrying weights + EMA + two AdamW
  moments in fp32 (the optimizer is dropped from the payload only when `training.resume` is
  False, `train_mlp_image_workspace.py:165`).

- **Sampler stochasticity — this survives the `num_inference_steps` match.** Both run 100
  steps, but DDIM at its default `eta=0` is *deterministic* given the initial noise
  (`set_alpha_to_one: True`), while DDPM injects fresh noise at every step
  (`variance_type: fixed_small`). Raising the transformer 8 -> 100 equalized the step
  *count*, not the sampler. Equal steps are not equal stochasticity.

- **Regularization.** The transformer has `p_drop_attn: 0.2` and `gradient_clip_norm: 1.0`;
  the UNet configs set no dropout and no clip norm, so it trains unregularized apart from
  the random crop and weight decay. (Until 2026-08-12 the UNet workspace could not have
  clipped even if asked — the key was dead there. It is now honored but left unset, so this
  arm is unchanged. Equalizing it is a decision about the experiment, not a bug fix.)

  `p_drop_attn` is **no longer one value across the transformer arms**: 0.1 on `st_n1` since
  2026-08-14 (0.2 before), 0.01 on `st_n1_drop001`. Quote the arm and the generation, not
  "the transformer", when reporting.

- **Crop offset granularity.** The transformer owns its crop: one offset per *sample*,
  reused across the whole obs window, drawn as a pure function of
  `(training.seed, global_step)` via `set_crop_step` / `_draw_crop_offsets`. The UNet leaves
  it to `CropRandomizer` inside `MultiImageObsEncoder`, which draws one offset per *image*
  from the global RNG — so its two obs steps are cropped differently, and the offsets are
  not reproducible across restarts.

- **Trainer.** The transformer runs `TrainMLPImageWorkspace` (the offline loop), the UNet
  `TrainDiffusionUnetImageWorkspace`. Neither is a choice, and **neither can be
  outer/inner**:
  - The UNet policy has none of the methods that workspace calls
    (`generate_search_context`, `predict_epsilon`, `search_candidates`, `set_crop_step`).
  - The transformer at width 1 has nothing to amortize. `generate_search_context` asks for
    `max_actions - 1 = 0` candidates and returns `(None, None)`, which the context buffer
    cannot concatenate. `TrainSearchOuterInnerWorkspace._fill_context_buffer` now asserts
    `max_actions > 1` and names the offline trainer, instead of failing on a bare NoneType.

  So the width-1 arms differ from the width-16 arm `value (oi)` in **trainer as well as
  search width**. This costs nothing on the objective — the two trainers compute the same
  loss; outer/inner only changes when the search context is regenerated, and at width 1
  there is no search context. It does mean the two loops draw data differently
  (full-dataset epochs vs a resampled 256-window pool with 4 passes).

## Different, but not confounds

These affect the eval grid and checkpoint selection, not what is optimized:

- **`checkpoint_every`** — 1,000 on the transformer, 10,000 on the UNet. Not a preference: a
  UNet checkpoint is ~4.4 GB against the transformer's 261 MB (~278M params, plus the EMA
  copy and AdamW moments), and at a 1k cadence the two UNet arms filled a 1.9 TB disk at
  step 115k. 10k matches the eval grid exactly, so nothing is ever evaluated off it.
- **Validation cadence** — the transformer validates on a step schedule
  (`val_every_steps: 1000`); the UNet on an epoch schedule (`val_every: 1`), which is every
  382 steps at 100 demos and every 110 at 29. Both feed the same `topk` monitor, so the
  UNet's `val_loss` curve is denser and its top-5 is selected from more candidates.
- **`num_workers`** — 8 vs 4. No effect on the objective.

## Verified equal, so it does not need re-checking

- **Demo budgets.** All 19 `train_pusht_*` configs were resolved and their `task.dataset`
  compared: the four arms here sit on byte-identical manifests at both budgets, with
  `train_ratio: null` and `seed: 42` throughout, and the env runner interpolates
  `split_file` from `task.dataset` so rollouts cannot drift from training. The only PushT
  config on a different split is `train_pusht_bc_29`
  (`pusht_seed42_legacy_val10_train29.json`, val=10) — the superseded pre-Round-8 split,
  replaced by `train_pusht_bc_29_r8`.
- **Normalization.** Both arms use the same `LinearNormalizer.fit(last_n_dims=1,
  mode='limits')` over `action` / `agent_pos` / `feedback`, fit on the *train* episodes only,
  plus a fixed `get_image_range_normalizer()` ([0,1] -> [-1,1], not data-fit) for images. No
  workspace passes a `mode` override. Both policies normalize *before* the encoder, which is
  why `imagenet_norm` must be `False` on both.
- **Dataset caching.** Neither arm caches, and `use_cache` would not help. It is a disk cache
  of the *preprocessed* replay buffer (keyed by an md5 of `shape_meta`, written as
  `<hash>.zarr.zip` under a `FileLock`) and exists only on `RealPushTImageDataset`,
  `RobomimicReplayImageDataset` and `Sim2RealImageMultiDataset`, which decode and resize
  source data. `PushTImageDataset` has no such parameter: its zarr is already the
  preprocessed form and `ReplayBuffer.copy_from_path` just reads the arrays. Both arms pay
  the same ~2.8 GB load.
