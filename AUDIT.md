# Audit — PushT offline diffusion search

**Scope: the PushT offline search arm only.** That is the six `train_pusht_diffusion_search*`
configs, `PushTDiffusionSearchPolicy` / `DiffusionTransformerSearchPolicy`, `PushTVerifier`,
`PushTImageDataset`, `TrainMLPImageWorkspace`, `PushTSearchImageRunner` and
`eval_search_pusht.py`. Findings about the maze/procgen/robomimic tasks, the online-search
arm, the outer/inner trainer, and the other twelve workspaces have been dropped from this
document; they are preserved in git history at the previous revision of this file and are
summarised in one line each in §12.

What the pipeline is meant to be: **offline distillation from expert data with a transformer
and a verifier in the loop**, built to the same shape as any standard ML pipeline — one
policy interface, one dataset contract, one training loop, one checkpoint/eval path, and
training and evaluation that agree.

**Status** ✅ fixed · ⬜ open · 📄 design-level, documented deliberately
**Confidence** **[V]** read in the code this pass or run · **[R]** reported, not re-verified this pass

---

## 1. The pipeline, in one page

Written down first because most findings below are "X and Y disagree", and that only means
something against a stated reference.

**Data.** `pusht_cchi_v7_replay.zarr`, 206 episodes, partitioned **50 test / 30 val / 100
train** (26 unused) — **12,899 frames → 12,199 training windows, 382 batches/epoch at batch 32, ~52 epochs
over a 20k-step run**. The partition is *pinned* by a committed manifest
(`config/splits/pusht_seed42.json`), not derived at runtime; see §3.1. Each sample is a
16-step window: `obs = {image (3,96,96) in [0,1], agent_pos (2), feedback (16)}` and
`action (16,2)`. `feedback` is the goal-relative keypoint displacement — an exact invertible
function of the block pose, which is why the block pose itself is not carried in the obs dict.

**Model.** `PushTDiffusionSearchPolicy`: a `MultiImageObsEncoder` (ResNet18, random init,
GroupNorm) producing **obs_feature_dim = 530** (512 ResNet + 2 agent_pos + 16 feedback),
feeding a 4-layer DDIM transformer that denoises a 16-step action chunk. The 76×76 random
crop still happens in the encoder's `CropRandomizer`, but the *policy* chooses the offsets:
one per sample, shared by its obs window and all its subgoal images (§4.2).

**Search.** `search_candidates` draws `max_actions = 16` candidates *sequentially*. After each
candidate, `_score_candidates` slices the current obs step and the executed action window
(`action_pred[:, To-1 : To-1+Ta]` = steps 1..8), hands them to `PushTVerifier.rollout`, which
resets a pool of 32 real PushT sims to `[agent_pos, block_pose]`, replays the 8 actions, and
returns `value = −mean keypoint distance` plus the reached state (and, in the `subgoal*`
modes, one rendered frame). That feedback becomes the context the *next* candidate is
conditioned on.

**Loss.** `compute_loss` generates a 15-candidate context under `no_grad`, then trains all 16
decode slots to reconstruct the **same** dataset expert action. See §9.1 — this is the single
most important structural fact about the arm.

**Eval.** Two paths. `PushTSearchImageRunner` rolls out val+test every 2000 gradient steps
with `n_search_actions = 8`. `eval_search_pusht.py --watch` evaluates every `step_*.ckpt` over
`n ∈ {1,2,4,...,64}`, selects on **val**, reports **test** at the val-selected step, and
maintains `bon_search/success_curves.jsonl` + `best.json`.

---

## 2. Results, and exactly what they license

From `SUCCESS_RATES.md` (20–21 checkpoints per arm, 50 test episodes, val-selected).

⚠️ **These numbers describe a configuration that no longer exists.** They were produced at a
**25-episode** training budget (~220 epochs over ~2,900 windows), with **per-image** random
crops and **no EMA**, selected on a **10**-episode val split. The current configuration is 100
episodes (~52 epochs over 12,199 windows), a shared per-sample crop, EMA 0.995, and a
30-episode selector. Every row below must be re-measured; in particular the headline in (a)
is confounded with the old data budget. Test *episodes* are unchanged, so re-runs remain
directly comparable to these figures.

| arm | sel. step | n=1 | n=8 | **n=64** | val n=64 |
|---|---|---|---|---|---|
| value / clean | 1000 | 0% | 2% | **84%** | 90% |
| subgoal_value / clean | 2000 | 0% | 24% | **70%** | 90% |
| subgoal / corrupt | 2000 | 0% | 12% | **68%** | 80% |
| subgoal_value / corrupt | 1000 | 0% | 20% | **56%** | 70% |
| subgoal / clean | 8000 | 2% | 20% | **36%** | 60% |
| value / corrupt | 7000 | 0% | 18% | **32%** | 90% |

Three readings, in decreasing order of how well supported they are.

**(a) `n=1` is 0–2% everywhere. The entire result is best-of-N over an oracle simulator.**
The single-sample policy is far below the BC baseline on this task. The model is not learning
to *produce* good actions; it is learning a distribution diverse enough that a ground-truth
physics verifier can pick a good one out of 64. This is §9.1 made concrete, and it is the
headline — a research finding, not a bug.

**(b) The search gain decays with training.** value/clean at n=64 goes **84% (step 1000) → 26%
(step 20000)** while its n=8 goes 2% → 16%. Single-sample quality rises; candidate diversity
dies faster. Every arm's selected step is 1000–8000, i.e. the useful region is the first ~40%
of a run that goes to 20000.

**(c) The arms are NOT separable, and no arm-vs-arm claim should be made.** Selection ran over
~20 checkpoints against a 10-episode val split (SE ≈ 9.5pp at p=0.9). Three arms tied at val
9/10 — Wilson CI [0.60, 0.98] — and their test numbers were 84 / 70 / 32%. On a steeply
declining curve the ranking is determined by where each arm's noise peak happened to land.

**And a fourth thing that invalidates the clean-vs-corrupt comparison outright:** the corrupt
arms are corrupted **at evaluation time too** (P0-1 below). They are not the same arms
measured under a different training signal; they are solving a strictly harder task. The
clean and corrupt rows of that table are not comparable.

---

## 3. CONFIGS

### 3.1 Inventory — every config in the PushT offline search arm

| file | role |
|---|---|
| `config/train_pusht_diffusion_search.yaml` | **base.** All policy/optimizer/training/logging/hydra settings live here. `search_context: value`, `corrupt_obs: False` |
| `config/train_pusht_diffusion_search_corrupt.yaml` | inherits base; sets `corrupt_obs: True` |
| `config/train_pusht_diffusion_search_subgoal.yaml` | inherits base; sets `search_context: subgoal` |
| `config/train_pusht_diffusion_search_subgoal_corrupt.yaml` | inherits `_subgoal`; sets `corrupt_obs: True` |
| `config/train_pusht_diffusion_search_subgoal_verifier.yaml` | inherits base; sets `search_context: subgoal_value` |
| `config/train_pusht_diffusion_search_subgoal_verifier_corrupt.yaml` | inherits `_subgoal_verifier`; sets `corrupt_obs: True` |
| `config/task/pusht_image_search.yaml` | shape_meta, dataset, env_runner. Shared by all six |
| `config/splits/pusht_seed42.json` | **the committed split manifest** — the exact train/val/test episode indices, plus a checksum of the zarr's `episode_ends`. Read by the dataset, the env runner and the eval script; regenerated only by `scripts/dump_pusht_splits.py` |

**This inheritance structure is correct and worth keeping.** Each of the five derived configs
overrides only `name`, the two identity keys, and `logging.tags` — nothing else. Verified by
diff: no derived config touches the encoder, the horizons, or the training length. An earlier
version pinned `max_gradient_steps: 100000` in the `_subgoal*` files while the base ran 20k;
that made the arms incomparable and is now explicitly warned against in their headers.

### 3.2 The resolved block every arm shares **[V]**

| group | key | value | note |
|---|---|---|---|
| horizons | `horizon` / `n_obs_steps` / `n_action_steps` / `n_latency_steps` | 16 / 2 / 8 / 0 | policy gets `n_action_steps + n_latency_steps`; runner gets the bare `n_action_steps` — equal only because latency is 0 (P2-6) |
| encoder | `rgb_model.weights` | `null` | random init |
| | `use_group_norm` | `True` | self-consistent with random init |
| | `crop_shape` / `random_crop` | `[76,76]` / `True` | the crop happens here; the offsets come from the policy — see the next row |
| | `imagenet_norm` | `False` | required: `LinearNormalizer` already maps to [−1,1] |
| crop | policy `crop_shape` / `random_crop` | `[76,76]` / `True` | the policy draws one offset per **sample**, shared by its obs window and every subgoal it generates, and passes them to the encoder; a pure function of `(seed, global_step)` (§4.2) |
| | `share_rgb_model` | `False` | one image key, so no effect |
| diffusion | DDIM, `num_train_timesteps` 100, `num_inference_steps` 8 | | `squaredcos_cap_v2`, `clip_sample: True`, `epsilon` |
| transformer | `n_layer` 4, `n_cond_layers` 2, `n_head` 4, `n_emb` 256 | | `p_drop_emb` 0.0, `p_drop_attn` **0.2**, `causal_attn` False |
| search | `max_actions` | 16 | 15 context candidates + 1 |
| verifier | `verifier_n_envs` / `legacy` / `use_async` / `verifier_steps` | 32 / False / True / null | `render_size` **not passed** — defaults to 96 (P2-2) |
| data | `batch_size` | 32 | pinned by verifier cost, see P2-1 |
| | `split_file` | `config/splits/pusht_seed42.json` | **source of truth**; the count keys are validated against it, not used to generate it |
| | train / val / test | **100** / 30 / 50 eps (26 unused) | 12,199 / 3,376 / 5,882 windows (12,899 / 3,586 / 6,232 frames); `train_ratio` and `max_train_episodes` are `null` and rejected alongside a manifest |
| optim | AdamW `lr` 1e-4, `weight_decay` 1e-6, `betas` [0.95, 0.999] | | |
| schedule | `decay_then_constant`, warmup 500, decay 10000, floor 0.1× | | |
| length | `max_gradient_steps` | 20000 | `num_epochs: 1000` is the outer bound only |
| cadence | `checkpoint_every` / `rollout_every_steps` | 1000 / 2000 | both in **gradient steps**, and 2000 is a multiple of 1000, so every rollout number belongs to saved weights |
| | `val_every_steps` / `sample_every_steps` | 1000 / 1000 | also gradient steps; the epoch-based `val_every`/`sample_every` remain only as the fallback for older configs |
| | `nrmse_max_batches` | 4 | 128 windows, drawn as a seeded spanning subset |
| | `use_ema` / `ema_decay` | **True** / **0.995** | constant decay (a 200-step window), not the repo's step-dependent warmup curve; EMA weights are what gets rolled out, validated and shipped |
| ckpt | `topk` | `val_loss`, min, k=5 | safety net only; real selection is `best.json` on val success |
| runner | `n_search_actions` / `max_steps` / `n_envs` | 8 / 300 / `null` | `null` → **80** persistent env subprocesses (P2-1) |
| paths | `hydra.run.dir` | `${output_root}/${exp_name}/${task_name}/${trainer}/ctx-${search_context}_corrupt-${corrupt_obs}_seed-${training.seed}` | identity-keyed, so relaunch resumes |

### 3.3 What actually differs between the six arms

Only these, and they drive both the policy and the run directory:

| arm | `search_context` | `corrupt_obs` | resulting `context_dim` |
|---|---|---|---|
| value / clean | `value` | False | 1 |
| value / corrupt | `value` | True | 1 |
| subgoal / clean | `subgoal` | False | 530 |
| subgoal / corrupt | `subgoal` | True | 530 |
| subgoal_value / clean | `subgoal_value` | False | 531 |
| subgoal_value / corrupt | `subgoal_value` | True | 531 |

Note the arms are not equal-capacity: the `value` arm's per-candidate context embedding takes
a 1-dim input, the `subgoal*` arms a 530-dim one. `SearchTransformerForDiffusion` sizes
`action_value_emb` from `context_dim`, so the subgoal arms have materially more parameters on
the context path. That is inherent to the ablation, not a defect — but it means "subgoal beats
value" would be confounded with "more parameters", and should be stated whenever the arms are
compared.

`state` and `state_value` are implemented in `PushTDiffusionSearchPolicy` and reachable, but
**no config uses them**. They are the cheap middle ground between the scalar and the rendered
subgoal (the reached `[agent_pos, feedback]`, no render, no encoder pass) and would isolate
"does the *outcome state* help" from "does the *rendered image* help". Worth a config.

### 3.4 Differences vs the simpler offline search on `harine/improved_verifier`

Reference: `diffusion_policy/config/maze_search.yaml` on that branch — the closest analogue,
since it targets the **same workspace** (`TrainMLPImageWorkspace`). Its siblings are
`maze_search_{corruption,concat_obs,mask_obs,concat_obs_corruption,mask_obs_corruption}.yaml`;
`train_search.yaml` targets `train_search.TrainSearchWorkspace`, a module that does not exist,
so it is dead and not a useful reference.

| axis | `maze_search.yaml` (improved_verifier) | `train_pusht_diffusion_search.yaml` | why it differs |
|---|---|---|---|
| workspace | `TrainMLPImageWorkspace` | **same** | ✅ the shared-structure goal is met at the workspace level |
| policy | `SearchPolicy` — MLP, `hidden_dim` 256, `hidden_depth` 4, deterministic regressor | `PushTDiffusionSearchPolicy` — DDIM transformer | task needs a multimodal action distribution |
| obs encoder | `FlattenObsEncoder`, **zero parameters** | `MultiImageObsEncoder` ResNet18 + crop + GroupNorm | maze obs is a 4-vector; PushT is 96×96 RGB |
| verifier | `MazeVerifier` (analytic, from `l2s`) | `PushTVerifier` (32-process pymunk pool) | **the dominant cost difference** — analytic vs physics-simulated |
| search context | scalar only, `context_dim = 1`, not overridable | 5 modes, `context_dim` 1…531 | the PushT ablation axis |
| ablation axes | `corrupt_obs`, plus `concat_obs` / `mask_obs` variants | `corrupt_obs` × `search_context` (3×2) | **`concat_obs` / `mask_obs` have no PushT counterpart** — see below |
| `batch_size` | 256 | **32** | verifier sim cost; 8× smaller |
| `num_epochs` | 100, no step cap | 1000 + `max_gradient_steps: 20000` | PushT needed a hard stop at the val optimum |
| `lr_scheduler` | `cosine` | `decay_then_constant` + `lr_scheduler_kwargs` | custom schedule, PushT only |
| `topk.monitor_key` | `train_loss` | `val_loss` | maze selects on a *training* metric — it will pick the most-overfit checkpoint |
| EMA | `use_ema: False`, no `ema_decay` | `use_ema: True`, constant `ema_decay: 0.995` | shared implementation; see P1-5 |
| crop | encoder-owned `CropRandomizer` (n/a — `FlattenObsEncoder`, no images) | policy-owned, one offset per sample | see P1-1 |
| `rollout_every_steps` | absent (epoch-based `rollout_every: 50`) | 2000, aligned to `checkpoint_every` | PushT fix; maze still on the never-coinciding cadence |
| `nrmse_max_batches` | absent → workspace default 4 | 4, explicit | same behaviour |
| env runner | `DummyRunner` — **no rollout at all** | `PushTSearchImageRunner`, 80 env subprocesses | maze has no in-training success metric |
| dataset | `Sim2RealImageMultiDataset`, `val_ratio: 0.05`, **2-way** split | `PushTImageDataset`, **3-way** split pinned by a committed manifest (100/30/50) | PushT needs an unbiased selector split, and a partition that cannot move silently |
| `hydra.run.dir` | `data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_...` — **timestamped** | identity-keyed under `${output_root}` | timestamped dirs mean `resume: True` never finds `latest.ckpt` |
| `multi_run:` block | present | removed | dead config; nothing reads it |
| identity keys | `exp_name: "default"`, read by nothing | `exp_name`/`trainer`/`search_context`/`corrupt_obs` drive the path | arm readable off the path |
| run manifest | none | `run.json` (git sha, dirty, SLURM id, host, arm, seed) | |

**Three things to take from this comparison.**

1. **The divergence is almost entirely justified.** Every structural difference traces to
   either "PushT has images and maze does not" or "PushT's verifier is a physics sim and
   maze's is analytic". The workspace, dataset contract, policy interface, checkpoint format
   and eval entry points are shared. That was the goal and it holds.

2. **Four PushT-only improvements should be backported, because the maze configs are
   currently wrong in ways PushT already fixed [V]:** the timestamped `hydra.run.dir` (silently
   restarts from scratch on every relaunch despite `resume: True`), `topk.monitor_key:
   train_loss` (selects the most-overfit checkpoint by construction), epoch-based
   `rollout_every` against step-based `checkpoint_every` (no rollout number is reproducible
   from a checkpoint), and the **pinned split manifest** — `Sim2RealImageMultiDataset` derives
   its split from `val_ratio: 0.05` at runtime, so it has exactly the drift exposure that
   P0-5 describes, with nothing recording what any maze checkpoint trained on. None of these
   are PushT-specific.

3. **`concat_obs` and `mask_obs` have no PushT counterpart, and that is a real gap.** On the
   maze branch those two flags change *how the context enters the model* — concatenated onto
   the obs vector, or with the observation masked out entirely. `mask_obs` in particular is
   the clean test of §9.1: with the observation removed, the context is no longer conditionally
   redundant, so the model *must* read it. PushT's only lever in that direction is
   `corrupt_obs`, which as implemented is uncalibrated (P0-2). A PushT `mask_obs` arm would be
   a more interpretable experiment than the corruption axis and is cheap to add.

### 3.5 Config-level findings

- ⬜ **`config/task/pusht_image_search.yaml:6-8` comment is stale [V]** — it still says
  "block_pos is emitted by the dataset (reset-only)". It has not been since `28e196b`.
- ⬜ **`training.device` is declared and ignored [V]** — every accelerate workspace takes the
  device from `Accelerator()`. Setting `cuda:1` here does nothing.
- ⬜ **`verifier_steps: null` and `n_search_actions` are the only two knobs with no
  `shape_meta` or code-side validation** — `**kwargs` transports `search_context`,
  `verifier_*`, `max_actions`, `scheduler_step_kwargs` into the policy with no schema, so a
  typo'd key is silently ignored rather than raising (P1-6).
- ⬜ **`n_envs: null` in the runner** now resolves to 80 rather than 60, because
  `n_val_episodes` went 10 → 30. Not a config error, but it is a resource change that the
  config does not make visible (P2-1).

---

## 4. Train/eval parity

The requirement is "training and eval should match fully". Here is where they do and do not.

### 4.1 Environment — in-training rollout vs `eval_search_pusht.py` **[V]**

| | `PushTSearchImageRunner` | `eval_search_pusht.py` | match? |
|---|---|---|---|
| env | `PushTImageEnv(legacy=False, render_size=96)` | same | ✅ |
| wrappers | Feedback → **VideoRecording** → MultiStep | Feedback → MultiStep | ⚠️ benign; the recorder only renders |
| `n_obs_steps` / `n_action_steps` | 2 / `${n_action_steps}` = 8 | 2 / `cfg.policy.n_action_steps` = 8 | ✅ equal **only because `n_latency_steps` is 0** |
| `max_steps` | 300, from the task config | 300, **hardcoded default** | ⚠️ desyncs silently if the config changes |
| reset states | `get_episode_init_states` over the 3-way masks | same function, same masks | ✅ |
| splits | val + test | val + test | ✅ |
| success | `max reward ≥ 1.0` | `max reward ≥ 1.0` | ✅ |
| `env.seed()` | `episode_idx` | not called | ✅ irrelevant — see §6 |
| candidates per decision | **8** | **swept 1…64** | ❌ different quantity, same name |
| seeding | unseeded global RNG | re-seeded per `n` from `cfg.training.seed` | ❌ rollout is not reproducible |
| obs corruption | ON in corrupt arms | ON in corrupt arms | ⚠️ consistent, but see P0-1 |

The two `❌` rows are why `test/success_rate` in wandb and the n=64 point of the eval curve
must never be quoted as the same measurement.

### 4.2 The encoder is shared, and it is normalized exactly once **[V]**

One `MultiImageObsEncoder` instance serves all three consumers, with no duplicate parameters
anywhere:

| consumer | path |
|---|---|
| observation conditioning | `_encode_obs_features` → `normalizer.normalize` → `[:, :To]` → `_encode_obs` |
| subgoal context (`subgoal*`) | `_encode_subgoal` → `normalizer.normalize` → `unsqueeze(1)` → `_encode_obs` |
| loss | `compute_loss` calls `_encode_obs_features` **once** and reuses it for the grad-tracked forward and every context candidate |

`_encode_obs(nobs)` takes an already-normalized dict and is the single encode point, matching
`OnlineSearchPolicy._encode_obs`'s contract. There is no second projection: online's trailing
`obs_projection` has its offline counterpart inside the transformer as `obs_emb`.

**Images are normalized once.** `LinearNormalizer` maps [0,1]→[−1,1] via
`get_image_range_normalizer`, and with `imagenet_norm: False` the encoder's `this_normalizer`
is `nn.Identity()` (`multi_image_obs_encoder.py:174-180`). The previous double normalization
(landing inputs at ~[−6.5, +2.25]) is gone.

**The crop is applied identically to obs and subgoal.** Cropping still happens where the rest
of the repo expects it — inside the encoder's `CropRandomizer` — but the *offsets* now come
from the policy, which is the only party that knows which images belong to the same sample.
`_draw_crop_offsets` produces one per sample (seeded from `(training.seed, global_step)`),
`_crop_scope` caches them for the forward pass, `_encode_obs` repeats them across the obs
timesteps, and `_encode_subgoal` reuses the very same ones for every candidate's subgoal.
`MultiImageObsEncoder.forward(obs_dict, crop_offsets=...)` pins them onto its randomizers for
the duration of one call and clears them in a `finally`; passing `None` restores the original
per-image sampling exactly, so no other task changes behaviour.

Verified: obs and subgoal offsets equal per batch element, the encoder crops at exactly those
offsets, the same `(seed, step)` reproduces them and a different step does not, eval yields
`[10, 10]`, output is 76×76 and `obs_feature_dim` is still 530.

### 4.3 The context distribution differs between train and eval **[V]**

| | training (`compute_loss`) | rollout / eval |
|---|---|---|
| model mode | `train()` | `eval()` |
| `p_drop_attn` | **0.2 active** while candidates are drawn | 0 |
| image crop | **random** 76×76, one offset per sample shared by obs and all its subgoals | center 76×76 (same relative geometry) |
| candidates | 15, fresh each step | 8 (rollout) or 1…64 (eval), rolling window past 16 |
| obs corruption | fresh draw per candidate + a 16th for `obs_cond` | fresh draw per candidate |

Dropout is now the only row that differs in kind: the search context the model is *trained*
to consume is drawn with `p_drop_attn` active and consumed at deployment without it. This is
P1-2, the last remaining train/eval context mismatch.

---

## 5. Normalization ledger — every site, and the count **[V]**

Traced end to end. Everything is normalized exactly once.

| # | site | input | direction | verdict |
|---|---|---|---|---|
| 1 | `_encode_obs_features:568` | raw obs from dataset/env | → normalized | once ✓ |
| 2 | `_encode_subgoal:167` | raw subgoal obs from the verifier | → normalized | once ✓ |
| 3 | `_normalize_state:123-124` | raw rollout state (px) | → normalized | once ✓ |
| 4 | `_normalize_value:144` | raw feedback (px) | → scale-only | once ✓, offset deliberately dropped |
| 5 | `_normalize_context_actions:537` | raw context actions | → normalized | once ✓, at the model boundary only |
| 6 | `predict_action:666` | model sample | → **un**normalized | correct direction ✓ |
| 7 | `compute_loss:890` | raw `batch['action']` | → normalized | once ✓ |
| 8 | `_search_action_nrmse:193-194` | raw pred + raw GT | → normalized | once ✓, both sides |

**Are the subgoal encodings normalized? Yes** — through the *same* fitted normalizer and the
*same* encoder as the observation conditioning, at site 2. No separate stats, no extra
parameters.

Two things worth knowing, neither a bug:

- **`_normalize_value` drops the offset on purpose.** The value's defining property is that it
  is 0 exactly at the goal; applying the normalizer's offset would make `‖feedback‖` nonzero
  there and destroy that. Only the *context* copy is rescaled — the raw scalar still ranks
  candidates, so `predict_action_best` and the `*_action_value*` metrics are unaffected.
- **In `state_value` / `subgoal_value` the same quantity enters the context twice** under two
  different normalizations: once inside `nstate` (range-normalized, with offset) and once as
  `nvalue` (scale-only). Redundant, not wrong. Only reachable in `subgoal_value`, since no
  config uses `state_value`.

The one remaining normalization hazard is P2-4: `LinearNormalizer._normalize` does
`x.reshape(-1, scale.shape[0])` with no last-dim assertion, and `normalizer['image']` has
`scale.shape == (1,)`, so it accepts a tensor of literally any width. A slicing typo in
`_normalize_state` would be silently absorbed rather than raising.

---

## 6. Stochasticity ledger

**The PushT env itself is deterministic on the path we use. [V]** The only RNG in
`pusht_env.py` is at `:98-102`, inside `if reset_to_state is None`. Every path here sets
`reset_to_state` — the runner via `init_fn`, `eval_search_pusht` via `make_init_fn`, the
verifier via `_set_reset_states`. `PushTEnv.__init__` calls `self.seed()` which draws from the
global numpy RNG, but `self.np_random` is then never read on the reset-to-state path. So
`env.seed(episode_idx)` in the runner and its absence in the eval script are both immaterial.
The physics (`space.step`, PD control at `k_p=100, k_v=20`, 10 substeps of `dt=0.01`) is fully
deterministic.

Everything that *is* stochastic:

| source | seeded? | consequence |
|---|---|---|
| workspace init (`torch`/`np`/`random`) | ✅ `cfg.training.seed` = 42 | |
| DataLoader shuffle | ✅ via the torch seed | |
| crop offsets | ✅ a pure function of `(seed, global_step)`, off the global RNG entirely | one per sample, shared by its obs window and subgoals; reproducible across restarts |
| `p_drop_attn` dropout | ✅ same | active during context generation |
| diffusion initial noise (`conditional_sample:607`) | `generator=None` → global RNG | reproducible in-process only |
| `corrupt_obs_features` noise + timestep | global RNG, **fresh per `predict_action` call** | P0-1, P0-2, P0-3 |
| `compute_loss` noise + timesteps | global RNG | fine |
| nrmse subset indices | ✅ fixed seeds 0 (val) / 1 (test) | comparable across steps |
| `eval_search_pusht` per-`n` re-seed | ✅ `cfg.training.seed` | points on the curve are paired |
| in-training rollout | ✅ re-seeded from `cfg.training.seed` before `env_runner.run()` | |
| verifier sim | deterministic | ✓ |
| `AsyncVectorEnv` worker ordering | deterministic (indexed, not first-come) | ✓ |

One gap remains, and it is not crop-specific: `torch.get_rng_state()` is never checkpointed,
so after a *resume* the dropout masks and diffusion noise diverge from an uninterrupted run.
The crop is immune by construction (it is derived from the step, not the stream), and both
eval paths re-seed, so this affects only mid-run reproducibility of the training trajectory
itself. ~10 lines in `BaseWorkspace` whenever it is wanted.

---

## 7. Metrics logged during training

| metric | when | computed on | reliable? |
|---|---|---|---|
| `train_loss` | every optimizer step + epoch mean | current batch | ✅ |
| `val_loss` | every 1000 steps | **full** val split, EMA weights | ✅ |
| `lr`, `global_step`, `epoch` | every step | | ✅ `global_step` == optimizer steps exactly |
| `val_nrmse_{min,avg,first}` | every 1000 steps | 128 windows, seeded spanning subset, **10/10 val episodes** | ✅ since §10 |
| `test_nrmse_{min,avg,first}` | every 1000 steps | 128 windows, **46/50 test episodes** | ✅ since §10 |
| `{val,test}_action_value{,_best,_first}` | every 1000 steps | same subsets | ✅ |
| `train_action_mse_error_{min,avg,first}` | every 1000 steps | a **fresh** training batch (32 windows) | ⚠️ it is a *fit* statistic by design — quote the `val_*`/`test_*` versions |
| `train_action_value{,_best,_first}` | every 1000 steps | same | ⚠️ same |
| `train_subgoals` (image panel) | every 1000 steps | same | ✅ good diagnostic; candidate order left→right, argmax starred |
| `{val,test}/mean_score` | every 2000 steps | full split rollout, **n=8** | ⚠️ `mean(min(coverage/0.95, 1))` — a *capped* score, not mean coverage |
| `{val,test}/success_rate` | every 2000 steps | full split rollout, **n=8** | ⚠️ not the same quantity as the eval curve's n=64, though now seeded and reproducible |
| `{val,test}/T_distance` | every 2000 steps | **final** step | ⚠️ disagrees with `mean_score` by construction (P2-5) |

**The diagnostic that answers "does search help" is `nrmse_first − nrmse_min`** — candidate 0
is drawn with an *empty* context, so it is the no-search baseline, and the gap to the best
candidate is the best-of-n gain. Its value-space twin is `action_value_best − action_value_first`.
Both now come from episode-spanning subsets, so for the first time they are split-level
statistics rather than single-episode ones.

Two more exist under `selection: final_pass` only (the `subgoal-only` arm), because none of the
above describes the action that arm executes — it deploys a further sample conditioned on all K
candidates, not any of them:

| metric | when | computed on | reliable? |
|---|---|---|---|
| `{val,test}_nrmse_final` | every 1000 steps | same subsets | ✅ the deployed sample vs the expert action |
| `{val,test}_action_value_final` | every 1000 steps | same subsets, **one extra sim** | ✅ monitoring only — the deployed policy never scores it |
| `train_action_{mse_error,value}_final` | every 1000 steps | fresh training batch | ⚠️ fit statistic, same caveat as the other `train_*` |

`action_value_final − action_value_best` is that arm's whole question: does the model's own
synthesis beat the oracle argmax it replaces?

**What is missing:** no gradient norm, no parameter norm. Per-candidate diversity is no longer
missing — §2(b) says collapsing diversity is the mechanism behind the success decay, and
`scripts/dump_candidate_scores.py` now measures it directly, off-line rather than in-training:
it stores the full `(T, B, n)` score tensor from a closed-loop rollout and reports within-step
SD per candidate index alongside a permutation test for whether generation order carries any
information at all. Results in `CANDIDATES_FROM_SUBGOAL.md`; the headline is that within-step SD
collapses after candidate 0 (e.g. 13.3 → 5.9 px on value/clean) and the ordering signal decays
to the permutation null by k≈6, so beyond ~6 candidates the search is resampling.

⚠️ **If you re-run that test, do not compare against `1/(k+1)`.** These scores tie constantly
(candidates that never touch the block return byte-identical values), which suppresses
strict-inequality records; a within-step *shuffle* — exchangeable by construction — already lands
~30% below the analytic curve. Measured that way every arm scores −0.5 to −0.8 and reads as
worse than resampling, when against the correct permutation null every arm is positive
(+0.14 to +0.38).

---

## 8. Findings

### P0 — corrupts a result that is currently being reported

#### ⬜ P0-1. Obs corruption is applied at evaluation, not just at training **[V]**
`diffusion_transformer_search_policy.py:579-595`

```python
def corrupt_obs_features(self, obs_features):
    if not self.corrupt_obs:
        return obs_features
    ...
```

The gate is `self.corrupt_obs`, a constructor flag. There is **no `self.training` check**.
`predict_action_best` → `predict_n_actions` → `search_candidates` → `predict_action` →
`corrupt_obs_features`, so the in-training rollout, `PushTSearchImageRunner`, and
`eval_search_pusht.py` all run the three corrupt arms with **noised observations**.

Consequence: the corrupt arms are not "the same task with a different training signal", they
are a strictly harder task. The clean-vs-corrupt rows of the §2 table are not comparable, and
the corrupt arms' success rates are not comparable to the BC baseline either.

This may well be the intended reading — "the policy that operates under a degraded
observation" is a coherent arm. But nothing in the six configs or the policy docstring says
so; the config headers describe corruption purely as the mechanism that makes the observation
partially informative *so the search context has something to add*, which reads as a training
device. **Decide and document which it is**, then either gate on `self.training` or state the
harder-task framing wherever the numbers appear.

#### ⬜ P0-2. The corruption magnitude is uncalibrated **[V]**
Same function. The noise is `torch.randn_like(obs_features)` — **absolute** std 1, independent
of the feature scale — and the timestep is drawn uniformly from all 100 scheduler steps. Two
consequences:

- **Nothing controls the corruption strength.** `obs_features` is a raw concatenation of
  ResNet18 GroupNorm activations (unnormalized, scale unknown) and the two low_dim keys
  (normalized to ~[−1,1]). Unit-variance noise is negligible for the former if its activations
  are large, and severe for the latter regardless. So the corruption is applied *unevenly
  across the feature vector* and at an unknown overall SNR.
- **There is no single "corruption level" to report.** At `t=0` the obs is nearly clean; at
  `t=99` the signal is scaled by `sqrt(ᾱ) ≈ 0.59` with noise of std ≈ 0.81 on top. The
  reported corrupt-arm success rate is an average over that whole range.

Fix direction: draw the timestep from a *fixed* value (or a narrow, configured band), scale
the noise relative to the per-feature std, and expose the level as a config key so the corrupt
row means something specific. As it stands, "corrupt" is not a measurable experimental
condition. Compare `mask_obs` on `harine/improved_verifier`, which is unambiguous.

#### ⬜ P0-3. Every candidate in one search sees a different corruption draw **[V]**
`predict_action:659` calls `corrupt_obs_features` per invocation, and `search_candidates`
calls `predict_action` once per candidate. So within a single decision, candidate 0 may see a
near-clean observation and candidate 5 a heavily corrupted one — and their verifier feedback
is then concatenated into one context as though all candidates were evaluated from the same
observation. `compute_loss` draws a 16th independent corruption for the `obs_cond` the
denoiser is trained against.

The docstring at `:653-654` notes the per-call draw deliberately ("caching the corrupted
features instead would make every candidate share one noise sample") — but sharing one sample
*within a decision* is arguably the correct semantics: the agent has one observation, not 16.
Worth revisiting alongside P0-2.

#### ✅ P0-5. The training data budget was a fraction of a pool that moves, and no split was recorded anywhere **[V]**

`config/task/pusht_image_search.yaml`, `pusht_image_dataset.py`

`train_ratio: 0.2` is a fraction of *whatever is left after the test and val splits are
taken*. So raising `n_val_episodes` 10 → 30 — a change made to fix the checkpoint selector,
with no apparent connection to the training set — shrank the train pool 146 → 126 and
silently cut the training budget from **29 episodes to 25**. The config comment still claimed
"~29 episodes, matching online search's data budget"; online actually resolves to 30.

The result: the six arms in §2 trained on **25 episodes / ~2,900 windows / 12% of the dataset,
for ~220 epochs**, and `val_loss` bottomed at step 3410 ≈ 35 epochs — exactly where
memorization would be expected on that budget. That confounds the §2(a) headline: "the model
only learns to cover a distribution, not to produce good actions" cannot be separated from
"the model saw 3.1k windows 208 times" until it is measured at a budget where single-sample
BC works.

The underlying defect is worse than the arithmetic. All three splits were **derived** at
runtime, independently, in the dataset, the env runner and `eval_search_pusht` — from five
keys, with nothing on disk recording which episodes a checkpoint had trained on. Derivation
from a seed is recreatable but not *pinned*: any of those five keys changing silently
repartitions the data, and nothing notices.

**Fixed** by making the partition an artifact rather than a computation:
- `config/splits/pusht_seed42.json` names the exact indices — **50 test / 30 val / 100 train**
  (26 unused; **12,199** / 3,376 / 5,882 windows) — plus an md5 of the zarr's `episode_ends`,
  because an episode index only means something relative to a particular dataset and
  `README_pusht.md` documents a heredoc that mutates the zarr in place (P2-11).
- The dataset, the env runner and the eval script all read it; the `n_*_episodes` keys are
  **validated** against it and generate nothing. `train_ratio` / `max_train_episodes`
  alongside a manifest is refused outright rather than applied on top.
- The budget is now absolute (100), so it does not move when a split size changes. It is a
  *prefix* of the permuted pool, so budgets nest — the old 25-episode set is a strict subset.
- Each run writes `<run_dir>/splits.json` and **refuses to resume** if the partition changed.
- `scripts/dump_pusht_splits.py` is the only writer; `--verify` / `--check-zarr` re-derive and
  re-fingerprint.

**Test and val are byte-identical to before** (`perm[:50]` and `perm[50:80]`; verified against
both the pre-change 3-way and the legacy 2-way derivation), so previously-collected test
numbers remain measured on the same episodes. All eight guards verified to raise.

#### ⬜ P0-4. DDP is configured and non-functional **[V]**
`train_mlp_image_workspace.py:394` — `self.accelerator.unwrap_model(self.model).compute_loss(batch)`
bypasses `DistributedDataParallel.forward`, so the all-reduce hook never fires and replicas
silently diverge. Any `world_size > 1` run is wrong. Compounding it, `env_runner.run()`
(`:480`) is not guarded by `is_main_process`, so every rank builds its own pool: **80 rollout
env subprocesses + 32 verifier workers = 112 processes per rank.** Single-GPU runs are
unaffected; this is a live trap the moment anyone launches with more than one.

### P1 — train/eval mismatch and measurement quality

#### ✅ P1-1. Obs and subgoal got independent random crops in training, identical center crops at eval **[V]**
`CropRandomizer.forward_in` samples an offset per **image**. The subgoal path goes through the
same encoder, and `_encode_obs` reshapes `(B,T,…)` → `(B*T,…)` before it — so the observation's
two timesteps and every candidate's subgoal each got their own 76×76 window out of 96×96,
while at eval everything was center-cropped and aligned.

Scoped honestly: independent crops *across the obs window* are ordinary augmentation (upstream
PushT BC does exactly that), so that half was over-stated when first written. The part specific
to the search policy is obs↔subgoal — the model is meant to compare "the scene now" against
"where this candidate lands it", and those two views were relatively translated at train and
aligned at eval. The magnitude was bounded even so: `agent_pos` (2) and `feedback` (16) are
concatenated onto the ResNet feature **uncropped and exact** on both sides, so the geometry the
search needs was always at full precision; only the ResNet channel was perturbed, and it
carries position weakly after global average pooling.

**Fixed** by keeping the crop in the encoder but letting the policy choose the offsets — it is
the only party that knows which images belong to the same sample. `CropRandomizer` gained an
optional `_forced_offsets`, and `MultiImageObsEncoder.forward` an optional `crop_offsets` that
it pins onto its randomizers for one call; both default to the previous per-image sampling, so
no other task is affected. Pre-cropping in the policy was tried first and rejected: the encoder
asserts its input matches `shape_meta` (96×96), and handing it 76×76 breaks that contract.

`_draw_crop_offsets` produces **one offset per sample**, `_crop_scope` caches it for the
forward pass, and it is reused by `_encode_obs` (repeated across the obs timesteps) and by
`_encode_subgoal` for every candidate's subgoal. Eval draws nothing and center-crops, exactly
as before. The augmentation is retained deliberately — §10 identifies it as load-bearing.

The offsets are a pure function of `(training.seed, global_step)` rather than of the global
RNG, so they are identical across restarts and machines with no RNG state to persist, and are
no longer interleaved with the diffusion-noise stream. All of this is asserted, not eyeballed:
obs and subgoal offsets equal per batch element, same step reproduces, different steps differ,
eval yields `[10, 10]`, output 76×76, `obs_feature_dim` still 530.

#### ⬜ P1-2. The context is generated with dropout active **[V]**
`compute_loss` runs while the model is in `train()`, and `generate_search_context` →
`search_candidates` → `predict_action` therefore draws every context candidate with
`p_drop_attn = 0.2` active. At rollout and eval the model is in `eval()`.

With P1-1 fixed this is **the only remaining train/eval context-distribution mismatch**. The
outer/inner trainer explicitly switches to `eval()` for context generation, which is arguably
the right call. Left open deliberately: changing it alongside the data-budget and EMA changes
would make the first re-run uninterpretable.

#### ✅ P1-3. Validation and sampling were on epochs while rollout/checkpoint were on gradient steps **[V]**
An epoch is not a stable unit here — its length depends on the dataset size and the training
budget. `val_every: 10` fired every ~910 steps at a 25-episode budget and would have fired
every ~3820 at 100 episodes: the same config value silently meaning 4× less validation, and
4× fewer points on every search-quality curve.

**Fixed**: `val_every_steps: 1000` and `sample_every_steps: 1000` join the existing
`rollout_every_steps: 2000` and `checkpoint_every: 1000`, so all four are in gradient steps and
rollout stays a multiple of checkpointing. `last_val_step` / `last_sample_step` were added to
`include_keys` so the cadences survive a resume instead of re-firing immediately, and the debug
override sets the step-based keys too (they take precedence, so a debug run would otherwise
have fired no eval block at all). The epoch keys remain as the fallback for older configs.

#### ✅ P1-4. The in-training rollout was unseeded **[V]**
Nothing re-seeded before `env_runner.run(policy)`. The env is deterministic (§6), but
`conditional_sample` passes `generator=None`, so candidates depended on whatever global RNG
state the preceding training batches left behind and the same checkpoint re-rolled gave a
different `test/success_rate`.

**Fixed**: re-seeded from `cfg.training.seed` immediately before the rollout, the same pattern
`eval_search_pusht._eval_split` already uses per `n`.

#### ✅ P1-5. `use_ema` was a silent no-op **[V]**
`grep -n ema train_mlp_image_workspace.py` matched nothing, yet every config declared
`training.use_ema` — so setting it `True` changed nothing about training and then made
`eval.py:40` (`policy = workspace.ema_model`) raise `AttributeError` on the resulting
checkpoint.

**Fixed** by backporting the outer/inner trainer's EMA rather than writing a second one, which
also closes R13 (the two trainers disagreeing on `use_ema` made the arms non-comparable despite
the config claiming only the loop differs). `use_ema: True`, `ema_decay: 0.995`. Three details
carried over because each is silent when wrong:
- **constant decay** via `min_value = max_value = decay`, which clamps `get_decay`'s warmup
  curve flat. The repo default is step-dependent (0.99438 at step 1000 for these settings), so
  the averaging window would otherwise depend on where in the run you are.
- **`ema.optimization_step = self.global_step` on resume.** `EMAModel` exposes no `state_dict`,
  so the counter is never checkpointed; left at 0, `get_decay` returns 0.0 and the update
  degenerates to `ema_param.copy_(param)` — **verified to destroy the restored average on the
  first post-resume step**.
- **`clone_policy`** (now shared in `base_workspace.py`) nulls the verifier before
  `copy.deepcopy`, so the EMA copy does not fork a second 32-process sim pool.
- **Resuming a pre-EMA checkpoint now raises.** `load_payload` only restores keys the
  payload contains, so an older checkpoint would leave `ema_model` at its random
  initialization while `model` got the trained weights — and since only the EMA weights are
  rolled out and shipped, every metric for the next ~200 steps would be silently wrong.

The EMA weights are what gets rolled out, validated, sampled and shipped;
`eval_search_pusht.load_policy` already preferred `workspace.ema_model`. Verified: decay is
0.995 at every step past the first, and with the counter restored the average is preserved and
still moves (`before 1.01700 → after 1.02200` against a live `2.01700`).

#### ✅ P1-6. `**kwargs` was an untyped config transport **[V]**
`DiffusionTransformerSearchPolicy.__init__` took `**kwargs` and read `max_actions`,
`search_context`, `verifier_*`, `scheduler_step_kwargs` with none of them validated, so a
typo'd config key was **silently ignored** — exactly how an ablation arm ends up secretly
identical to its sibling.

**Fixed**: `_validate_kwargs` raises naming the unknown keys and listing the known set.
Verified with a `verifier_n_env` (missing `s`) typo, which now raises instead of quietly
leaving the verifier on its 32-env default.

### P2 — efficiency, resources, metric hygiene

#### ⬜ P2-1. Process and memory footprint of one run **[V]**
Per training process: 80 rollout env subprocesses (`n_envs: null` → 30 val + 50 test) held for
the whole run, 32 verifier sim workers, and 8 (train) + 8 (val) + 8 (test) + 16 (two nrmse
loaders) dataloader workers. The zarr `img` array is stored as `<f4` (~2.8 GB) and is
copy-on-write shared across forked workers, so the memory is not multiplied — but the
**process count is ~150**, which is what actually trips node limits. The 80 rollout envs in
particular sit idle between rollouts 2000 steps apart.

Separately, `batch_size: 32` is pinned by verifier cost, not by memory: `compute_loss` runs 15
sequential candidates, each a full 8-step DDIM loop plus a 32-env 8-step sim. That is the
dominant term in step time and the reason a 20k-step run is expensive.

#### ⬜ P2-2. `render_size` is not plumbed into the verifier **[V]**
`PushTDiffusionSearchPolicy._build_verifier` passes `verifier_n_envs`, `verifier_legacy`,
`verifier_use_async`, `verifier_steps` — **not** `render_size`, which falls back to its default
of 96. Correct today because `shape_meta` says 96. Silently wrong the moment the image shape
changes: the subgoal frames would be a different size from what the encoder expects, and
`_encode_subgoal` would fail (or worse, resize silently).

#### ⬜ P2-3. 8× redundant image decoding **[V]**
`PushTImageDataset` has no `key_first_k`, so all 16 horizon steps of `img` are decompressed
and collated per sample while only `[:, :2]` is ever used. At `batch_size: 32` that is ~56 MB
moved per batch where ~7 MB is needed. `robomimic_replay_image_dataset.py` avoids exactly
this. Coupled to the uint8 storage change below.

#### ⬜ P2-4. `LinearNormalizer._normalize` accepts wrong-width tensors **[V]**
`x.reshape(-1, scale.shape[0])` with no last-dim assertion. For `normalizer['image']`,
`scale.shape == (1,)`, so it accepts any tensor at all. One slicing typo in `_normalize_state`
away from silently corrupting the context.

#### ⬜ P2-5. `T_distance` and `mean_score` disagree by construction **[V]**
`pusht_search_image_runner.py:224` reads `feedback` from the **final** step, while
`mean_score`/`success_rate` are the **max** over the episode. For sub-threshold episodes that
transiently approach the goal and then push the T away, the two metrics point in opposite
directions. (For *successful* episodes there is no disagreement: `pusht_env.py:133` returns
`done = coverage > threshold` and `MultiStepWrapper` stops on the first `done`.) Minor
adjacent off-by-one: success is `>= 0.95` in the runner and `> 0.95` for env termination.

#### ⬜ P2-6. Latency-step mismatch is silent **[V]**
`pusht_image_search.yaml:39` gives the runner `n_action_steps` without `n_latency_steps`, while
the policy gets `n_action_steps + n_latency_steps`. `MultiStepWrapper.step` iterates whatever
it is handed, so a nonzero latency yields a wrong control cadence with no exception.
`eval_search_pusht.py:204` guards this explicitly (it reads `cfg.policy.n_action_steps`); the
runner does not. Inert while latency is 0.

#### ⬜ P2-7. Three different search widths, one of which is never measured **[V]**
Training conditions on `max_actions - 1 = 15` candidates; the in-training rollout uses
`n_search_actions: 8`; `eval_search_pusht` sweeps 1…64. The training regime — 15 — is the one
that determines what the model learns and is the one never evaluated.

#### ⬜ P2-8. `eval_search_pusht.py` hardcodes the env instead of instantiating `cfg.task.env_runner` **[V]**
`build_envs` fixes `legacy=False`, `render_size=96`, and `main` defaults `--max-steps 300`,
`--n-envs 50`. Any change to `task.env_runner` therefore desyncs eval from the in-training
rollout with no error. Concretely today: `--n-envs 50` against a 30-episode val split pads with
20 copies of `states[0]` (`_eval_split:171-174`) and runs 20 wasted envs per val chunk.

#### ⬜ P2-9. `_curve_key` selects on `sr[-1]` **[V]**
`eval_search_pusht.py:315-316` selects the best checkpoint by val success **at the largest `n`
in the sweep**. With `--max-n` variable between invocations, rows in one `success_curves.jsonl`
can be selected on different quantities (val@64 vs val@1024) and compared as though they were
the same. Record `max_n` in the row and require it to match, or select on a fixed `n`.

#### ⬜ P2-10. Pretrained-finetune silently refits the normalizer **[V]**
`train_mlp_image_workspace.py:230` loads pretrained weights only when `not checkpoint_loaded`,
which is correct — but `checkpoint_loaded` stays `False` on that branch, so `:267` falls
through to `dataset.get_normalizer()` and overwrites the pretrained run's statistics. Not hit
by the six PushT configs (none set `pretrained_ckpt_path`), but it is a trap for the finetune
path the code otherwise supports.

#### ⬜ P2-11. The zarr keys the data path needs are not in the published dataset **[V]**
`data/agent_pos` and `data/block_pos` are required by six call sites, but the published
`pusht.zip` ships only `img, state, action, keypoint, n_contacts`. The only recipe is a
heredoc in `README_pusht.md` that mutates the zarr in place. An unpatched checkout fails with a
bare `KeyError` inside `ReplayBuffer.copy_from_store`. Commit
`scripts/prepare_pusht_zarr.py` (`agent_pos = state[:, :2]`, `block_pos = state[:, 2:5]`,
verified against `pusht_env.py:154-158`) and raise an actionable error in the dataset.

#### ⬜ P2-12. `img` stored as float32; dtype follows the zarr with no cast **[V]**
`pusht_image_dataset.py:239` — `image = np.moveaxis(sample['img'],-1,1)/255` with no
`.astype(np.float32)`, unlike `agent_pos`/`action` on the adjacent lines. Correct only because
the zarr happens to store `<f4`. Storing as `uint8` would be 4× smaller — but `uint8 / 255`
promotes to **float64**, so the two changes must be made together or every image tensor
silently doubles in size and reaches the ResNet as a `DoubleTensor`.

#### ⬜ P2-13. Small ones, verified **[V]**
- `expert_mask` is all-ones, collated into every batch, and consumed by nobody.
- `dataset.return_sequences` (`:263`) and `cfg.training.freeze_encoder` (`:373`) are bare
  attribute accesses — any dataset or config predating them raises before training starts.
- `val_pool is not test_pool` (`:288`) is an **identity** comparison on numpy arrays. It works
  only because the 2-way path aliases the same object; a refactor to `.copy()` would silently
  start duplicating test metrics under both prefixes. Branch on `n_val_episodes > 0`.
- `torch.load(open(path,'rb'))` with no `with` in `eval_search_pusht.py:79` — one leaked fd
  per checkpoint, forever, in `--watch` mode.
- `mkdir(parents=False)` at `pusht_search_image_runner.py:134`.
- Dead `test()` with a hardcoded `~/Projects/...` path in `pusht_image_dataset.py`.

---

## 9. Design findings 📄 — documented deliberately, no code change

### 📄 9.1 The search context is conditionally uninformative under the current loss **[V]**
`diffusion_transformer_search_policy.py:907`

```python
trajectory = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1)
```

All 16 decode slots are trained to reconstruct the *same* dataset expert action. The
conditioning is `(obs, candidate actions, verifier feedback on those candidates)`, and the
candidates are themselves generated from `obs` alone — so the context is a deterministic
function of `obs`, and

```
p(a* | obs, context) = p(a* | obs)      exactly
```

The Bayes-optimal model **ignores the context**, and no gradient pressure exists for the
feedback channel to matter.

**This is task-dependent, and PushT clean is the worst case.** PushT is fully observed — image
+ agent_pos + feedback determine the state, and the expert action is a function of state — so
the redundancy is provable. Partially-observed tasks (maze, procgen) are the setting where the
context genuinely carries information one observation does not. So this is not "the
architecture is broken"; it is "PushT with `corrupt_obs: False` is the one setting where
redundancy is provable". That is the entire reason the corrupt row exists — and why P0-1/P0-2
matter so much, because as implemented the corrupt row does not cleanly deliver the partial
observability it was built for.

Three directions, not ranked:
1. **A `mask_obs` arm** (as on `harine/improved_verifier`) — unambiguous partial observability,
   unlike the uncalibrated corruption.
2. **Supervise only the final slot**, or weight later slots more — the model is then trained to
   *improve given context* rather than to reproduce a context-independent target.
3. **Make the target the searched best-of-N action** — true offline distillation, which is what
   the project description says the pipeline is for. Note that today **nothing writes the
   searched best-of-N action anywhere**: a repo-wide grep for `distill` returns only wandb
   project names.

### 📄 9.2 `corrupt_obs` means opposite things in the two search paths **[V]**
Offline, `encode_obs_cond` corrupts the **observation** (so the context has something to add).
Online, `OnlineSearchPolicy.forward` corrupts the **context** (a robustness measure). Same
method name, inverted target. Rename one.

### 📄 9.3 Search requires a ground-truth simulator at inference **[V]**
`PushTVerifier.rollout` resets a real PushT sim to the exact state and steps true dynamics. The
block *state* is not privileged — it comes from `feedback`, a declared obs key — but the
*simulator* is. The `subgoal*` modes go further, putting a ResNet embedding of a sim-rendered
future frame into the model's input distribution, so the trained model needs the oracle at
inference, not merely for ranking. State this alongside any result.

### 📄 9.4 Verifier objective ≠ eval metric **[V]**
The verifier scores the final state of a chunk by `−mean keypoint distance`; success is the
`max` **coverage** over the episode. Keypoint distance is not monotone in coverage, and a
candidate that passes through the goal mid-chunk and slides off scores badly. So
argmax-verifier ≠ argmax-success even with a perfect simulator.

### 📄 9.5 Verifier dynamics gap — MEASURED, and it does not matter **[V]**
Each rollout starts with **zero agent velocity** (a fresh body), so overshoot is systematically
under-predicted; `_set_state` also runs one extra `space.step` the real state never
experienced; and stepping continues past `done`.

Quantified by `scripts/measure_verifier_fidelity.py` (read-only; replays recorded test
episodes). 200 decision points, 25 test episodes, 15-step replay, against a ground truth
replayed from the episode start so its velocity is exact:

| variant | block pos err px (mean / median / p95) | angle err deg | keypoint-dist err (mean / p95) |
|---|---|---|---|
| **today** (zero velocity + settle step) | 0.51 / 0.03 / 1.36 | 0.34 | **0.180 / 0.956** |
| **k=1 warm-up + snap-back** (proposed fix) | 0.41 / 0.00 / 1.38 | 0.29 | 0.132 / 0.866 |

The reset discards a real **9.5 px** of agent momentum (p95 22.6 px), yet after 15 steps the
block lands **0.51 px** from truth on average, median 0.03 px. The keypoint-distance error —
the quantity the verifier's value is built from — is **0.18 against a between-candidate value
spread of ~5 units**, i.e. ~3% of the signal being ranked on.

**Conclusion: the k=1 warm-up + snap-back is NOT worth implementing.** It reduces mean position
error by 20% (p95 unchanged) and costs ~2× verifier time, which dominates the training loop.
PushT's block is quasi-static enough over a 15-step horizon that agent momentum does not
propagate meaningfully into block pose. The design concern was real; the magnitude is not.
Revisit only for a task with lighter or faster-moving objects.

### 📄 9.6 Search is fully sequential **[V]**
N candidates = N sequential 8-step DDIM loops with `K_decode = 1`, even though
`SearchTransformerForDiffusion.forward` accepts `(B, K, H, Da)`. At n=64 that is 64 sequential
diffusion loops *per control step*, which is what makes the top of the eval sweep expensive.
Batching N into the batch dim would change the sequential-conditioning semantics, so this is
design-level rather than a straight optimization.

### 📄 9.7 The rolling window drops the incumbent best **[R]**
Past `max_actions`, `predict_n_actions` evicts the *oldest* candidate, so conditioning never
sees the current leader; only the final `argmax` over all scores recovers it.

### 📄 9.8 Boundary padding is supervised as real data **[V]**
`sampler.py:194-202` repeats the last frame, and with `pad_after = n_action_steps - 1 = 7`, up
to 7 of the 16 target steps in an end-of-episode window are duplicates while `expert_mask` is
all-ones. Worse on the search path: the verifier physics-simulates a chunk of repeated targets
that never occurred, and that fabricated outcome becomes the model's context.

### ✅ 9.9 Run directories renamed from `ctx-*` to arm labels — DONE 2026-08-05 **[V]**

The arms now have labels naming what they *do*, not just which context they feed back, because
`selection` was added as a second axis and `ctx-<search_context>` no longer identifies an arm:

| `search_context` | `selection` | arm label |
|---|---|---|
| `value` | `argmax` | `value` |
| `subgoal` | `argmax` | `subgoal-chosen4value` |
| `subgoal_value` | `argmax` | `subgoal-value` |
| `subgoal` | `final_pass` | `subgoal-only` |

Asserted in `train_mlp_image_workspace._ARM_LABELS` against `cfg.arm`, so a config cannot file
its results under the wrong arm.

**DONE.** All twelve directories were renamed on 2026-08-05 by
`scripts/rename_ctx_dirs.sh` (dry-run first, 12/12 matched this map), and `run_name` in
`train_pusht_diffusion_search.yaml` is now `${arm}_corrupt-${corrupt_obs}_demos-${n_demos}_seed-${training.seed}`.

**Preconditions checked before moving anything**: every `ctx-*` run ENDED with its
checkpoints 20/20 evaluated, no `tr_ctx-*` remained, and the six idle `ev_ctx-*` watchers
were cancelled (they were polling for checkpoints that would never arrive). Six `cand64_*`
jobs were live and hold **absolute** `ctx-*/checkpoints/step_*.ckpt` paths in their
`--export`; they requeue on preemption and re-resolve, which is exactly why the back-symlink
is not optional.

**Verified after**: old path and new path both list 20 checkpoints (same inode); the
regenerated `SUCCESS_RATES.md` holds 1,092 table rows, unchanged from before the rename, so
no run was orphaned or double-counted.

**Trigger — do the rename when `squeue -u $USER` shows no `tr_ctx-*`, no `ev_ctx-*` and no
`bon_n*` job.** Until then the `ctx-*` names are load-bearing: `training.resume: True` locates
`latest.ckpt` by a path derived from `run_name`, so flipping the template while a trainer is
live sends any relaunch (requeue, node failure, a rerun of `launch_round7.sh`) to a *fresh empty
directory* — the run silently restarts from scratch keeping neither weights nor step count.

**The map**, under
`/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline/`:

| current | becomes |
|---|---|
| `ctx-value_corrupt-False_demos-100_seed-42` | `value_corrupt-False_demos-100_seed-42` |
| `ctx-value_corrupt-True_demos-100_seed-42` | `value_corrupt-True_demos-100_seed-42` |
| `ctx-subgoal_corrupt-False_demos-100_seed-42` | `subgoal-chosen4value_corrupt-False_demos-100_seed-42` |
| `ctx-subgoal_corrupt-True_demos-100_seed-42` | `subgoal-chosen4value_corrupt-True_demos-100_seed-42` |
| `ctx-subgoal_value_corrupt-False_demos-100_seed-42` | `subgoal-value_corrupt-False_demos-100_seed-42` |
| `ctx-subgoal_value_corrupt-True_demos-100_seed-42` | `subgoal-value_corrupt-True_demos-100_seed-42` |
| `ctx-value_corrupt-False_seed-42` (29 demos) | `value_corrupt-False_demos-29_seed-42` |
| `ctx-value_corrupt-True_seed-42` | `value_corrupt-True_demos-29_seed-42` |
| `ctx-subgoal_corrupt-False_seed-42` | `subgoal-chosen4value_corrupt-False_demos-29_seed-42` |
| `ctx-subgoal_corrupt-True_seed-42` | `subgoal-chosen4value_corrupt-True_demos-29_seed-42` |
| `ctx-subgoal_value_corrupt-False_seed-42` | `subgoal-value_corrupt-False_demos-29_seed-42` |
| `ctx-subgoal_value_corrupt-True_seed-42` | `subgoal-value_corrupt-True_demos-29_seed-42` |

Already on the target form (new runs, nothing to resume): `subgoal-only_corrupt-{False,True}_demos-100_seed-42`.
Unchanged: `bc_demos-{100,29,25}_seed-42` (BC is not a search arm), `runs/train_pusht_search_outer_inner*`, `_diag/`, `_smoke_r4/`.

**Recipe used** — `mv old new && ln -s new old`, dry-run first
(`bash scripts/rename_ctx_dirs.sh --dry`). The back-symlink means a live job holding the old
absolute path keeps hitting the same inode, so nothing had to be cancelled.

**REMAINING**: the twelve `ctx-*` back-symlinks are still in place. Drop them once no job
resolves through them — currently the six `cand64_*` jobs do. Until then `ls` shows each run
twice, once per name; the tooling globs arm labels only, so nothing double-counts.

**Same-commit code changes — all applied:**
1. ✅ `train_pusht_diffusion_search.yaml`: `run_name` → `${arm}_corrupt-${corrupt_obs}_demos-${n_demos}_seed-${training.seed}`.
2. ✅ `train_pusht_diffusion_search_subgoal_only.yaml`: `run_name` override deleted, now inherits the template. (`_subgoal_only_corrupt.yaml` never had one.)
3. ✅ `scripts/build_success_rates_doc.py`: `D100`/`D29` built from an `ARMS` tuple of arm labels; the 29-demo dirs now carry an explicit `demos-29`.
4. ✅ `scripts/slurm/submit_large_n_evals.sh`: globs the four arm labels, **not** `ctx-*` — globbing both forms would visit every run twice, once per name, and the second visit reads as a separate arm rather than a duplicate.

**Caveat to carry into the report**: the 29-demo runs trained against a *different split
manifest and a 10-episode val split* than the 100-demo runs. `demos-29` names their budget; it
is not a claim that the two budgets are drop-in comparable. Same note belongs in `SUCCESS_RATES.md` §2.

---

## 10. Fixed — changelog

### ROUND 7 (2026-08-03) — the data, the crop, and EMA

| # | was | now |
|---|---|---|
| P0-5 | `train_ratio: 0.2` of a pool that shrinks when the val split grows → the budget silently fell 29 → **25 episodes** (~2,900 windows, ~220 epochs), and no split was recorded anywhere | committed manifest `config/splits/pusht_seed42.json` names the exact indices: **100 train / 30 val / 50 test** (12,199 / 3,376 / 5,882 windows, 26 unused, ~52 epochs). Read by the dataset, runner and eval script; the count keys are validated against it; `<run_dir>/splits.json` per run, and resume refuses on a changed partition |
| P1-1 | `CropRandomizer` drew an offset per **image**, so the obs timesteps and each subgoal were relatively translated at train and aligned at eval | crop stays in the encoder, but the policy supplies the offsets: **one per sample**, shared by its obs window and all its subgoals, derived from `(seed, global_step)` so it survives restarts and leaves the global RNG alone |
| P1-3 | `val_every`/`sample_every` in **epochs** while rollout/checkpoint were in gradient steps — an epoch grows ~91 → 382 batches at the new budget, so the same value meant 4× less validation | `val_every_steps`/`sample_every_steps: 1000`; all four cadences in gradient steps, resume-safe via `include_keys` |
| P1-4 | in-training rollout unseeded → `test/success_rate` not reproducible from its checkpoint | re-seeded from `cfg.training.seed` before `env_runner.run()` |
| P1-5 | `use_ema` a silent no-op; setting it `True` then broke `eval.py` | EMA backported from the outer/inner trainer at **constant decay 0.995**, with the resume counter restored and a verifier-sharing `clone_policy`. Also closes R13 |
| P1-6 | typo'd policy config keys silently ignored | `_validate_kwargs` raises naming them |

Verified: all six configs resolve; the test and val splits are byte-identical to before (so
prior test numbers stay measured on the same episodes); **8/8 split guards raise**; crop
offsets are shared obs↔subgoal, reproducible per step, and center at eval; EMA decay is flat at
0.995 and survives resume.

### ROUND 6 (2026-08-03) — measurements

| # | was | now |
|---|---|---|
| ✅ | `nrmse_max_batches: 4` + `shuffle: False` drew all 128 windows from the split's **first episode** (1/10 val, 2/50 test) | `_make_nrmse_loader` draws a fixed seeded `Subset` spanning the split: **10/10** val, **46/50** test, identical cost |
| ✅ | `train_sampling_batch` pinned the **first batch of the first epoch** and reused that GPU tensor for the whole run | refreshed every step |
| ✅ | `rollout_every: 50` (epochs) vs `checkpoint_every: 1000` (steps) — the two sets never intersected | `rollout_every_steps: 2000`, a multiple of `checkpoint_every`; `last_rollout_step` in `include_keys` so it survives resume |
| ✅ | `n_val_episodes: 10` → selector SE ≈ 9.5pp, three arms tied at 9/10 | 30 → SE ≈ 5.5pp. Deliberate side effect: val comes out of the **train** pool (146 → 126, so `train_ratio: 0.2` is ~25 not ~29 episodes), so runs before/after are not comparable — but **test is provably unaffected** (`get_split_masks_3way` draws test first from the same seeded permutation; index checksum 5067 at both settings) |
| 📄 | verifier zero-velocity reset | measured, not worth fixing — §9.5 |

A seeded *fixed* subset rather than a reshuffling loader, on purpose: reshuffling would span
the split but inject fresh sampling noise into a curve read across training steps.

### ROUND 4/5 — the encoder fix, which is why there are results at all

Root cause of the earlier "no successes": severe overfitting from three compounding causes.
`crop_shape: null` + `random_crop: False` removed the primary regularizer for a ~29-episode
train set; `weights: IMAGENET1K_V1` + `use_group_norm: True` + `imagenet_norm: True` do not
compose (group norm discards the pretrained BatchNorm statistics the weights depend on, and
ImageNet stats assume [0,1] while `LinearNormalizer` had already mapped to [−1,1], landing
inputs at ~[−6.5, +2.25]); and nothing selected a best checkpoint, so the good model at step
~3k was overwritten while the run continued to 100k.

| | before | after |
|---|---|---|
| `val_loss` min | 0.0886 @ step 3072 | **0.0598 @ step 3410** |
| `val_loss` end | 0.2896 (3.3× min) | 0.1111 (**1.9×** min) |
| test success | peaked 14%, decayed to ~0 | **peaks 32–84%** |

Applied: random init + group norm + 76×76 random crop + `imagenet_norm: False`;
`max_gradient_steps` 100000 → 20000; `checkpoint_every` 2000 → 1000; `TopKCheckpointManager`
wired into the workspace (it was imported and never instantiated, so `checkpoint.topk` was dead
config in ~30 files).

### Results-integrity fixes

| # | was | now |
|---|---|---|
| ✅ | best checkpoint chosen by max **test** success over ~50 ckpts × 7 `n` values, and that same number reported → +10–15pp optimistic bias | `_curve_key` selects on **val**; test reported at the val-selected step, never selected on |
| ✅ | eval entirely unseeded; the `for n in n_list` loop consumed one continuous RNG stream, so curve points were un-paired | re-seeded from `cfg.training.seed` before every `n` |
| ✅ | no error bars | Wilson 95% CI on every rate, plotted as a band, `n_episodes` recorded per row |
| ✅ | single-ckpt eval wrote a flat `bon_search/success_curve.{json,png}` — **12 concurrent jobs observed, 3 of every 4 curves lost** | per-step subdir, same convention as watch mode; `best.json` written atomically via `os.replace` |
| ✅ | `seen.add(step)` before the `try` → any transient failure permanently skipped that checkpoint | added after success, with 3 retries |
| ✅ | watcher never exited; `wandb.init(resume='allow')` with no `id` minted a new run per requeue | `--idle-exit-sec`; deterministic wandb id from an md5 of the run dir; `wandb.init` wrapped |
| ✅ | non-atomic checkpoint save vs the watcher's existence-only glob → torn reads | `_atomic_save` (`.tmp` + `os.replace`); also protects `latest.ckpt`, which `resume` depends on |

### Results-layout fixes

`hydra.run.dir` was timestamp-keyed, so relaunching never resumed; the hand-override to
`runs/<config-name>` swapped that for silent collision. Arm and seed were absent from the path
and the arm mapping was hardcoded in three places that already disagreed. Now: identity-keyed
run dirs (`ctx-{value,subgoal,subgoal_value}_corrupt-{False,True}_seed-42`), and
`BaseWorkspace.write_manifest()` appends one `run.json` entry per launch with git sha, dirty
flag, SLURM job id, hostname, arm, seed and `global_step`.

### Correctness fixes on the training path

| # | was |
|---|---|
| ✅ | Pretrained weights loaded **after** the resume block, so every restart discarded all progress while keeping the resumed optimizer state and step counter. Every search config sets `resume: True` |
| ✅ | Gradient accumulation keyed on `global_step`, which counts optimizer steps and cannot delimit its own windows; each epoch's final partial window was dropped with its gradients leaking into the next |
| ✅ | `global_step` incremented inside the batch loop **and** once per epoch, so it drifted +1/epoch — and `step_*.ckpt` names are parsed for the eval curve's x-axis |
| ✅ | `block_pos` in the obs dict with no normalizer entry → `KeyError` in every non-filtering policy. Removed entirely rather than relocated: `feedback` is an exact invertible function of it (round-trip verified at 1.5e-5 px / 2.0e-7 rad), which also made the verifier's train-time and eval-time resets identical |
| ✅ | `predict_action` violated `BaseImagePolicy` — required `verifier, n_actions` and returned a bare tuple, while the *private* `_predict_action` honoured the contract. Now layered: `predict_action` → `predict_action_best` → `search_candidates` / `predict_n_actions` |
| ✅ | Three inconsistent duck-typed probes for "is this a search policy", with two incompatible return contracts in one function → one `_is_search_policy()` helper |
| ✅ | Context actions were fed to the model **raw** (~[0,512]) while the denoised trajectory and target were normalized — one tensor serving both the verifier boundary (needs pixels) and the model boundary (needs normalized). `_normalize_context_actions` at the two model boundaries only; public returns stay raw |
| ✅ | The verifier scalar entered the context raw (~0…−300). `_normalize_value` rescales the **context** copy onto the fitted feedback scale; `score` stays raw so ranking and metrics are unchanged |
| ✅ | Two overlapping normalize+encode paths → one shared `_encode_obs(nobs)`, matching `OnlineSearchPolicy`'s contract |
| ✅ | `_score_candidates` branched to `get_value` (literally `rollout(...)[0]`) → always `rollout`, since every mode now needs the reached state |
| ✅ | Verifier subprocess pool leaked one full 32-process pool per evaluated checkpoint in `--watch` mode → `policy.close()` in the `finally` |
| ✅ | `TopKCheckpointManager.get_ckpt_path` raised `KeyError` on any epoch where checkpointing fired without a rollout → returns `None` |
| ✅ | `pdb.set_trace()` in `sample_sequence`'s exception handler — hangs a DataLoader worker on stdin |
| ✅ | Dropped `assert` in `sampler.py:14` (`episode_mask.shape == episode_ends.shape` as a bare expression) |
| ✅ | Checkpoint saves raced on an unjoined background thread; `run()` returned without joining, leaving the final checkpoint truncated |
| ✅ | Scheduler horizon overcounted: `len(dl) * num_epochs // accum`, but the last batch never stepped the scheduler → `ceil(len(dl)/accum) * num_epochs` |
| ✅ | In-place mutation of the persistent GPU sampling batch (`obs_dict['attention_mask'] = ...`) → shallow copy before mutation |
| ✅ | `CenterCrop` assigned to the normalizer slot and unconditionally overwritten → assigned to the randomizer slot |

---

## 11. Not bugs — recorded so they are not "fixed" by a later pass

- **The subgoal embedding is deliberately NOT noised in `corrupt_obs` mode.** `_encode_subgoal`
  calls `_encode_obs` directly, bypassing `corrupt_obs_features`, while `obs_cond` is
  corrupted. That is the intended design: the corruption degrades the *observation* so the
  feedback channel has something to add; noising the feedback too would defeat the point.
- **`last_epoch=self.global_step-1` does not raise `KeyError: 'initial_lr'` on resume.**
  `LambdaLR.__init__` with `last_epoch >= 0` does require `initial_lr` in every param group,
  but `optimizer.load_state_dict` restores `param_groups` including the `initial_lr` the
  previous run's scheduler set. Verified against the PyTorch source. Resume works.
- **`PushTVerifier.__init__` does not fork workers.** It only stores config
  (`pusht_verifier.py:109` sets `self._vec = None`); the pool is built lazily on the first
  rollout, which is why `_get_vec` can use a `forkserver` context after CUDA is initialized.
- **`PushTSearchImageRunner` does not inherit a broken `run()`.** It bypasses
  `PushTImageRunner.__init__` deliberately (to cover val+test rather than test only) but
  overrides `run()` at `:173`; it inherits only `_tile_videos`.
- **`float32` img is not multiplied across dataloader workers.** `copy_from_store` produces
  plain numpy and workers are forked, so the ~2.8 GB buffer is copy-on-write shared. The 4×
  storage/IO argument (P2-12) stands on its own.
- **Training does cover all context lengths.** `_build_memory_masks:255-261` assigns decode
  slot *k* exactly *k* visible context tokens, so lengths 0…15 are all trained. Only the
  *number of candidates* differs between regimes (P2-7).
- **`compute_loss` no longer produces inference tensors.** `generate_search_context` uses
  `no_grad` rather than `inference_mode`, precisely so the returned context can be buffered and
  fed to the model on a later step. The train-mode context generation (P1-2) is a separate
  issue.

---

## 12. Out of scope

Dropped from this document at the owner's request; all still live, all findings preserved in
git history at the previous revision.

- **Online search arm** (`OnlineSearchPolicy`, `TrainOnlineSearchWorkspace`,
  `train_online_search.yaml`) — trained with context and evaluated without it; context envs use
  `legacy=True` (every reset block pose off by ~90 px); still carries the broken encoder combo
  that §10 identifies as the root cause of the earlier collapse; writes only `latest.ckpt` so
  the watcher can never evaluate it. **Explicitly deferred, not being fixed.**
- **Outer/inner trainer** (`train_search_outer_inner_workspace.py`, 3 configs) — uncommitted
  WIP. Resume replays the in-flight outer step; drift measured post-step; `train_loss` logged
  twice at one step. Its `use_ema: True` no longer makes it non-comparable (the offline arms now
  match at 0.995, and both share one `clone_policy`); eval-mode context generation still does —
  that is the P1-2 decision, and whichever way it goes both trainers should follow it.
- **Maze / procgen** — `search_policy.py` has a module-level `from l2s.verifier import
  MazeVerifier` and `l2s` is installed in neither env, so 8 configs are unimportable; its
  `predict_action` still has the pre-fix inverted contract. See §3.4 for the three fixes worth
  backporting from PushT.
- **Other workspaces / tasks** — `eval.py`'s unguarded `workspace.ema_model`; `checkpoint_every`
  meaning epochs in most workspaces and steps here; `join_saving_thread` reaching only 5 of 14
  workspaces; `dataset.return_sequences` / `expert_mask` as undeclared hard requirements that
  make any non-PushT image dataset an immediate `AttributeError`.
