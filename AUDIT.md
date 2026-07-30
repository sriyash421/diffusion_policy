# Codebase audit

Full read of every workspace, dataset, sampler/normalizer, search policy, verifier, env
runner, and eval script, checking for correctness bugs and for divergence from a standard
ML pipeline. The target shape: **every task and model shares one structure** — same policy
interface, same dataset contract, same training loop, same checkpoint/eval path.

**Status legend**

| | meaning |
|---|---|
| ✅ | fixed |
| ⬜ | open |
| 📄 | design-level; documented deliberately, no code change |

**Confidence legend**

| | meaning |
|---|---|
| **[V]** | verified by reading the code directly, or by running it |
| **[R]** | reported by a reading pass, not independently re-verified |

Severity: **P0** correctness (silently corrupts runs) · **P1** standardization · **P2**
hygiene/efficiency/metric quality · **D** design.

---

## P0 — correctness

### ✅ 1. Pretrained checkpoint clobbered every resume **[V]**
`workspace/train_mlp_image_workspace.py` (was ~L190)

The `pretrained_ckpt_path` block ran *after* the `cfg.training.resume` block. On a restart,
`latest.ckpt` was loaded (weights + optimizer + `global_step`), then the pretrained weights
overwrote the model while the resumed optimizer state and step counter were kept — all
training progress silently discarded, with the run continuing as though nothing happened.
Every search config sets `resume: True`.

**Fixed:** gated on `not checkpoint_loaded`. Also changed `include_keys=['model']` →
`include_keys=[]`: `include_keys` filters `payload['pickles']`, not `state_dicts`, so the
original argument did not do what it read as, and would have let the pretrained run's
`global_step`/`epoch` leak into a fresh run.

### ✅ 2. Gradient accumulation keyed on `global_step` **[V]**
`workspace/train_mlp_image_workspace.py`, and the same pattern in 9 other workspaces

`if self.global_step % gradient_accumulate_every == 0` — but `global_step` counts
*optimizer steps*, so it cannot delimit its own accumulation windows. Compounded by
`global_step` being skipped on the last batch of each epoch and incremented once *extra* at
the epoch boundary. Net effect with `accumulate > 1`: the window shifts by one every epoch,
the first batch of training takes an optimizer step on a single micro-batch, and each
epoch's final partial window is dropped with its gradients leaking into the next epoch
(`zero_grad` only ran inside the branch). Inert today only because every config sets
`gradient_accumulate_every: 1`.

**Fixed:** boundary on `(batch_idx + 1) % accum == 0`, epoch-final and `max_train_steps`
windows always flushed, `global_step` incremented only on optimizer steps.

### ✅ 3. `global_step` ≠ optimizer steps **[V]**
`workspace/train_mlp_image_workspace.py`

Incremented inside `if not is_last_batch` *and* once more per epoch, so it drifted by +1 per
epoch. `step_*.ckpt` filenames are derived from it and `eval_search_pusht.py` parses them for
the success-curve x-axis — so that axis was systematically wrong (by ~166 steps at epoch 166).

**Fixed** with #2. The per-epoch increment is gone; the last batch's log slot is reserved for
the end-of-epoch summary so wandb steps stay strictly increasing.

### ✅ 4. `block_pos` in the obs dict with no normalizer entry **[V]**
`dataset/pusht_image_dataset.py`, `env/pusht/pusht_verifier.py`, `policy/pusht_diffusion_search_policy.py`, `policy/online_search_policy.py`

The dataset emitted `block_pos` in `obs` as a "reset carrier", but `get_normalizer` never fit
it. `LinearNormalizer._normalize_impl` iterates the *input* dict's keys, so any policy doing
an unfiltered `self.normalizer.normalize(obs_dict)` raised `KeyError: 'block_pos'` —
`ibc_dfo_hybrid_image_policy.py:171,256` and `diffusion_unet_hybrid_image_policy.py:222,287`.
`config/train_ibc_dfo_hybrid_workspace.yaml:3` defaults to `task: pusht_image`, so that
config was dead on its first batch. It also meant `obs` did not match `shape_meta`.

**Fixed by removing the carrier entirely**, not by relocating it. `feedback` is an exact,
invertible function of the block pose (`block_pose_from_feedback`) and is already a declared
`shape_meta` obs key, so the carrier was redundant. Removing it means:
- `obs` now equals `shape_meta['obs']` exactly; no policy needs to know about a special key
- the verifier's **train-time and eval-time resets become identical** (train previously used
  exact `block_pos`, eval used the feedback reconstruction — a real train/eval discrepancy
  that relocating the carrier would have preserved)

Round-trip verified over 20 random poses: **max position error 1.5e-5 px, max angle error
2.0e-7 rad** in a 512-px sim space.

### ✅ 5. `predict_action` violated the `BaseImagePolicy` contract **[V]**
`policy/diffusion_transformer_search_policy.py`, `policy/pusht_diffusion_search_policy.py`

The public `predict_action` took required `verifier, n_actions` args and returned a *tuple*
`(actions, values[, scores][, subgoals])`, while the *private* `_predict_action` was the one
returning the standard `{'action': (B,Ta,Da), 'action_pred': (B,H,Da)}`. The naming was
inverted, so `eval.py`, the plain env runners, and any generic consumer broke on this policy.

**Fixed:** `_predict_action` → `predict_action` (contract-honoring), search entry point →
`search_candidates`, and `predict_action_best` moved from the PushT subclass up to the base
search class so both variants share one readout. The interface is now layered:
`predict_action` (standard) → `predict_action_best` (best-of-n, same output format) →
`search_candidates` / `predict_n_actions` (raw generators).

### ✅ 6. Duck-typed policy dispatch in the workspace **[V]**
`workspace/train_mlp_image_workspace.py`

Three inconsistent probes for "is this a search policy": `hasattr(policy, 'verifier')`,
`getattr(policy, 'supports_return_scores', False)`, and positional tuple unpacking
(`result[0]`, `result[1]`, `result[2]`) in one branch against `result['action']` in the
other — two incompatible return contracts in a single function. `search_policy.py` never set
`supports_return_scores`, so the maze policies silently fell through to a code path where
`scores` was the *rollout state*, not a rankable scalar, making `val_action_value*`
meaningless rather than absent.

**Fixed:** one `_is_search_policy()` helper; all policies return dicts or named tuples;
`supports_return_scores` deleted.

### ✅ 7. Verifier subprocess pool leaked per checkpoint in eval **[V]**
`eval_search_pusht.py`

`eval_checkpoint` builds a fresh policy per checkpoint, each lazily forking 32 sim workers;
only `env.close()` was in the `finally`. `policy.close()` existed and was never called. In
`--watch` mode this leaks a full 32-process pool per evaluated checkpoint, indefinitely.

**Fixed:** `policy.close()` in the `finally`, guarded so a close failure cannot mask the
original error.

### ✅ 8. `OnlineSearchPolicy` had no `close()` at all **[V]**
`policy/online_search_policy.py`

The workspace teardown called `close()` on the policy, but `OnlineSearchPolicy` never defined
one — so its lazily-built context-rollout vec-env pool leaked regardless of the teardown.

**Fixed:** added `close()`, matching `PushTDiffusionSearchPolicy.close()`.

### ✅ 9. `TopKCheckpointManager.get_ckpt_path` KeyErrors on a missing metric **[V]**
`common/checkpoint_util.py`

Bare `data[self.monitor_key]`. `rollout_every`, `val_every`, and `checkpoint_every` are
independent, so on any epoch where checkpointing fires without a rollout, 7 workspaces crash.
Only cadences happening to be multiples of each other kept this quiet.

**Fixed:** return `None` when the key is absent (topk resumes next epoch that produces it);
a missing `format_str` key now raises an actionable message naming the available metrics.

### ✅ 10. `pdb.set_trace()` in the data path **[V]**
`common/sampler.py`

`except Exception as e: import pdb; pdb.set_trace()` inside `sample_sequence`. In a
`num_workers > 0` DataLoader this hangs the worker on a stdin read instead of surfacing the
error. **Fixed:** removed; the exception propagates.

### ✅ 11. Dropped assert **[V]**
`common/sampler.py:14` — `episode_mask.shape == episode_ends.shape` was a bare expression.
A mismatched mask/ends length silently indexes out of range or under-iterates.
**Fixed:** restored to an `assert`.

### ✅ 12. Model trained in `eval()` mode when `use_ema: False` **[V]**
`workspace/train_diffusion_unet_image_workspace.py`

One `.train()` call, inside the step-based rollout branch. The epoch-end block called
`policy.eval()` and never restored. With `use_ema: False`, `policy` *is* the online model, so
dropout/BN were silently disabled for the entire rest of the run from epoch 1.
`train_mlp_image_workspace.py` fixed this exact bug and it was never backported.

**Fixed:** the *online* model is returned to `train()` at epoch end (the EMA copy is
deliberately left in `eval()` — the transformer workspace's version of this fix incorrectly
puts the EMA model into train mode).

### ✅ 13. Shape mismatch in IBC sampling **[V]**
`workspace/train_ibc_dfo_hybrid_workspace.py`

`obs_dict` was sliced to `n_samples` but `gt_action` was not, so `F.mse_loss` compared
mismatched batch sizes whenever `sample_max_batch < dataloader.batch_size`. The lowdim
sibling slices both. **Fixed**, plus a dead `batch = train_sampling_batch` assignment removed.

### ✅ 14. BET checkpoints incompatible with every loader **[V]**
`workspace/train_bet_lowdim_workspace.py`

The model attribute was named `self.policy`. `BaseWorkspace.save_checkpoint` keys state dicts
by attribute name, so BET payloads contained `state_dicts['policy']` where every other
workspace emits `'model'` — any generic loader (`eval.py`, `eval_bon.py`) silently found
nothing. **Fixed:** renamed to `self.model` (13 sites).

### ✅ 15. Three workspaces `eval.py` cannot construct **[V]**
`train_diffusion_transformer_lowdim_workspace.py`, `train_bet_lowdim_workspace.py`,
`train_robomimic_lowdim_workspace.py` — `__init__(self, cfg)` with no `output_dir`, while
`eval.py:34` calls `cls(cfg, output_dir=output_dir)` → `TypeError`.
**Fixed:** `__init__(self, cfg, output_dir=None)` + `Optional` import.

### ✅ 16. Center crop silently dropped **[V]**
`model/vision/multi_image_obs_encoder.py`

In the `crop_shape is not None and not random_crop` branch, `CenterCrop` was assigned to
`this_normalizer`, which the next block unconditionally overwrote with `nn.Identity()`. Live
for `train_diffusion_unet_hybrid_workspace.yaml:41`. **Fixed:** assigned to the randomizer slot.

### ✅ 17. Checkpoint saves raced on an unjoined thread **[V]**
`workspace/base_workspace.py`

The save ran on a background thread; `_saving_thread` was overwritten without ever being
joined, and `run()` returned without joining — so two saves within one epoch raced on the same
`latest.ckpt`, and the final checkpoint could be left truncated at process exit.

**Fixed:** join the prior thread before starting a new one; `join_saving_thread()` called at
the end of `run()` in the three accelerate workspaces. Also `mkdir(parents=False)` →
`parents=True` (it failed if `output_dir` did not exist yet).

### ✅ 18. `train_online_search_workspace.py` — six issues **[V]**

- re-ran `accelerator.prepare(self.model)` on **every checkpoint**, building a new DDP wrapper
  on rank 0 only, unrelated to the optimizer prepared at the top of `run()` → **fixed** with
  the standard save/restore of the wrapped module
- never registered the parent's `_close_worker_pools` teardown → env-runner and policy pools
  leaked → **fixed** via `ExitStack`
- the outer loop restarted at 0 after a resume, doubling run length on every restart
  (`resume: True` is the config default) → **fixed**, resumes from `self.epoch`
- normalizer unconditionally recomputed after a resume, silently changing statistics →
  **fixed**, matches the parent's preserve-on-resume rule
- `rng.integers(lo, ...)` raised `ValueError` when `min_context_trajs` exceeded a pool entry's
  rollout count (which early episode termination can produce) → **fixed**, `lo` clamped
- the debug override ran *after* the scheduler was sized from `num_outer * num_inner` → **fixed**
- additionally: each outer iteration ended in `eval()` (rollout/checkpoint) and the next began
  training without restoring `train()` → **fixed**

### ✅ 19. Scheduler horizon overcounted **[V]**
`num_training_steps = len(dl) * num_epochs // accum` in every workspace, but the last batch of
each epoch never stepped the scheduler, so the true count was lower by `num_epochs` and the
cosine schedule never reached its floor. **Fixed** in the MLP workspace alongside #2:
`ceil(len(dl)/accum) * num_epochs`. ⬜ Still open in the other workspaces.

### ✅ 20. In-place mutation of a persistent GPU batch **[V]**
`workspace/train_mlp_image_workspace.py` — `obs_dict = batch['obs']` then
`obs_dict['attention_mask'] = ...`. `train_sampling_batch` is held on GPU for the whole run,
so anything inserted into its obs dict stuck forever. **Fixed:** shallow copy before mutation.

---

## P1 — pipeline standardization ⬜ open

### ⬜ 21. `train_mlp_image_workspace` has no EMA at all **[V]**
`grep -n ema` matches nothing in the file, yet ~30 configs targeting it declare
`training.use_ema`, and `eval_search_pusht.py` silently falls back to the raw model when it is
`True`. Every other workspace has EMA. All configs currently say `False`, so nothing is wrong
*today* — but setting it is a silent no-op.

**Proposed:** implement the standard EMA block (create `ema_model`, step once per *optimizer*
step, use it for rollout/val/checkpoint), gated on `cfg.training.use_ema` so behavior is
unchanged while it stays `False`. Old checkpoints load fine — `load_payload` iterates the
payload, not the workspace.

### ⬜ 22. `TopKCheckpointManager` imported and never instantiated **[V]**
`train_mlp_image_workspace.py:29`. `checkpoint.topk` is therefore dead config in ~30 files and
**no best-metric checkpoint is ever written** — only `latest.ckpt` and `step_*.ckpt`.
Same in `train_diffusion_unet_image_workspace.py`, where the `metric_dict` is still built and
then discarded. **Proposed:** wire it up (now safe, given #9).

### ⬜ 23. `checkpoint_every` has two incompatible meanings **[V]**
Step semantics in `train_mlp_image_workspace` / `train_online_search_workspace`, epoch
semantics everywhere else, and **both** in `train_diffusion_unet_image_workspace` (steps at
one site, epochs at another — so its epoch-level save fires exactly once per run, since
`self.epoch % 5000 == 0` is true only at epoch 0). The debug block sets it to `1`, which under
step semantics writes a full multi-GB checkpoint every batch.
**Proposed:** rename to `checkpoint_every_steps` in the step-based workspaces.

### ⬜ 24. Multi-GPU is configured but non-functional **[R]**
- `compute_loss` is called on `accelerator.unwrap_model(self.model)` in every accelerate
  workspace, bypassing DDP's forward hook → **gradients never sync**, replicas diverge, and
  `accelerator.prepare` is pure overhead
- `lr_scheduler` is passed to `accelerator.prepare` *and* stepped manually →
  `AcceleratedScheduler.step()` advances the wrapped scheduler `num_processes` times per call
- checkpoint/topk blocks unguarded by `is_main_process` in the two hybrid workspaces → every
  rank writes `latest.ckpt` concurrently through the background save thread
- `train_diffusion_unet_image_workspace` runs `env_runner.run()` on every rank

Inherited from upstream, but it means any >1 GPU run is silently wrong.

### ⬜ 25. Dead / unimportable code **[V for search_policy.py, R for the rest]**
`policy/search_policy.py:14` has a top-level `from l2s.verifier import MazeVerifier`, making
the module unimportable in this environment; its `predict_n_actions` is a near-verbatim
duplicate of the new policy's. Also: dead `test()` with a hardcoded `~/Projects/...` path in
`pusht_image_dataset.py`; unused `import shutil` in 7 workspaces; unused `optimizer_to` in
both robomimic workspaces; unused `EMAModel` imports in the IBC workspaces; and
`from diffusers.training_utils import EMAModel` — the **wrong** `EMAModel` — in three lowdim
workspaces (the real one comes from `cfg.ema._target_`).

### ⬜ 26. Runner bypasses its parent's `__init__` **[R]**
`env_runner/pusht_search_image_runner.py` calls `BaseImageRunner.__init__` directly and
re-implements ~80 lines of env construction, while still inheriting the parent's `run()`,
which would call the wrong policy contract. Any fix to the parent silently fails to propagate.

### ⬜ 27. Fork-vs-forkserver inconsistency **[R]**
`pusht_search_image_runner.py` creates `AsyncVectorEnv` with the default *fork* context after
`Accelerator()` has initialized CUDA in the parent; `pusht_verifier.py` deliberately uses
`forkserver` for exactly this reason. The two pools should match.

### ⬜ 28. Bare attribute/config access **[V]**
`train_mlp_image_workspace.py` reads `dataset.return_sequences` and
`cfg.training.freeze_encoder` without `getattr`/`.get`, so any dataset or config predating
them raises `AttributeError` before training starts.

### ⬜ 29. Identity comparison on numpy arrays **[V]**
`train_mlp_image_workspace.py` decides whether a test dataloader exists via
`val_pool is not test_pool`. It works only because the dataset aliases the *same object* in
the 2-way case; any refactor to `.copy()` silently starts duplicating test metrics under both
`val_` and `test_` prefixes. **Proposed:** branch on `n_val_episodes > 0`.

### ⬜ 30. `**kwargs` as an untyped config transport **[V]**
`diffusion_transformer_search_policy.py` does `kwargs['max_actions']` (bare `KeyError`), and
`kwargs` also carries `search_context`, `verifier_*`, `maze_path`, `scheduler_step_kwargs`
with no validation — **typo'd config keys are silently ignored**. Same for `pusht_verifier`'s
swallowed `**kwargs`, and `render_size` is not plumbed through `_build_verifier` at all, so a
non-96 `shape_meta` image would silently mismatch the encoder in the `subgoal*` modes.

### ⬜ 31. `transformer_image_policy.py` contract violations **[R]**
Mutates the caller's dict via `obs_dict.pop('attention_mask')`, returns `action` of shape
`(B, Da)` with no `Ta` axis, and returns a *normalized* `action_pred` next to an unnormalized
`action`. Same pop-mutation in `DPTImagePolicy`.

### ⬜ 32. Verifier constructed unconditionally in `__init__` **[R]**
So merely *loading* a checkpoint spawns a `PushTVerifier`. The maze base class does a lazy
in-method import of an optional dependency instead. Should be lazy/optional on first use.

---

## P2 — hygiene, efficiency, metric quality ⬜ open

### ✅ 33. Scale mismatch in the search context — actions *and* scalar **[V]**
`policy/diffusion_transformer_search_policy.py`, `policy/pusht_diffusion_search_policy.py`

Originally logged as "the raw verifier scalar dominates the normalized state/embedding".
Tracing it further found the **larger** half: the **context actions** were raw too.
`search_candidates` collects `predict_action(...)['action_pred']`, which is already
unnormalized because the *verifier* needs pixel coords for its sim reset — and that same
tensor was reused as **model context**. One tensor serving two boundaries with incompatible
unit requirements. So `action_value_emb`'s input was 32 raw-pixel action dims (~[0,512])
plus one raw scalar (~0…−300), while the `noisy_trajectory` being denoised and the
`trajectory` target were both normalized.

Severity qualifier, recorded so it is not overstated: the encoder uses `norm_first=True`, so
a LayerNorm hits `memory` at the start of every layer and largely rescues forward-pass
magnitude. What it does not fix: at init the context tokens are ~2 orders larger than
`obs_emb(obs_cond)` and the two are concatenated into one sequence with a **shared
positional embedding added pre-norm**; and the gradient w.r.t. `action_value_emb.weight`
scales with input magnitude, giving that layer an effective LR ~2 orders off the rest.

**Fixed:**
- `_normalize_context_actions` applied at the two model boundaries only (`predict_action`
  before `conditional_sample`, and `compute_loss` at the `self.model(...)` call). The public
  returns of `search_candidates` / `predict_n_actions` / `predict_action_best` stay in raw
  units, because nearly every consumer wants them there: `_verifier_inputs`, the env runner,
  `_search_action_nrmse` (normalizes its own input), and the sampling block.
- `_normalize_value` rescales the scalar onto the fitted feedback scale wherever it enters
  the **context**, using the normalizer's `scale` but deliberately **not** its offset, so the
  value stays 0 exactly at the goal. `score` stays raw, so ranking and `train_action_value*`
  are unchanged.
- `_score_candidates` now always calls `verifier.rollout` instead of branching to
  `get_value` (which is literally `rollout(...)[0]` — same cost), because every mode now
  needs the reached state to rescale from. Removes the duplicate path noted in old #16.

This restores the invariant `OnlineSearchPolicy` already maintains: actions exist only in
normalized space anywhere the model touches them, and are unnormalized strictly at the
env/verifier boundary. **Also fixed on `harine/improved_verifier`** across all five affected
classes (commit `7f06736`).

### ✅ 33b. Offline encode path aligned with online **[V]**
`_encode_obs(nobs)` added to the offline base with the same contract as
`OnlineSearchPolicy._encode_obs` (takes already-normalized obs). `_encode_obs_features` and
PushT's `_encode_subgoal` are now both thin wrappers over it, so the observation
conditioning and the subgoal context share one encode path instead of each re-implementing
normalize+encode. No architecture change — online's trailing `obs_projection` already has an
offline counterpart in the transformer's `obs_emb`.

### ✅ 33c. `corrupt_obs` ablation matrix **[V]**
Three new configs (`..._corrupt`, `..._subgoal_corrupt`, `..._subgoal_verifier_corrupt`),
each inheriting its clean sibling and setting `corrupt_obs: True` and nothing else. No
existing config's `search_context` changed, so the verifier-only / goal-only / goal+verifier
ablation is preserved and the corrupt axis is added orthogonally. See #46 for why the corrupt
row is the one that can produce a non-null result.

### ⬜ 34. The zarr keys the whole data path needs don't exist in the published dataset **[V]**
`data/agent_pos` and `data/block_pos` are required by 6 call sites, but the published
`pusht.zip` ships only `img, state, action, keypoint, n_contacts`. The only recipe is a
heredoc in `README_pusht.md` that mutates the zarr in place. No committed script, no runtime
check — an unpatched checkout fails with a bare `KeyError` deep inside
`ReplayBuffer.copy_from_store`.
**Proposed:** commit `scripts/prepare_pusht_zarr.py` (derivation `agent_pos = state[:, :2]`,
`block_pos = state[:, 2:5]` — verified correct against `pusht_env.py:154-158`) and raise an
actionable error in the dataset.

### ⬜ 35. `val` is `test` **[V]**
`config/task/pusht_image.yaml` sets no `n_val_episodes`, so the dataset aliases
`val_mask = test_mask` — every `val_*` metric is measured on the same 50 episodes reported as
`test/mean_score`. And even in the real 3-way config, `eval_search_pusht.py` selects the best
checkpoint from the **test** success curve, so the headline number is a selected-on maximum
and the `val` split is decorative.

### ⬜ 36. 8× redundant image loading **[R]**
`pusht_image_dataset.py` has no `key_first_k`, so all 16 horizon steps of `img` are
decompressed and collated per sample while only `x[:, :2]` is used. The reference
`robomimic_replay_image_dataset.py` avoids exactly this. ~56 MB/batch moved where ~7 MB is used.

### ⬜ 37. `img` stored as float32 **[R]**
The zarr holds 0-255 values as `<f4`, so `copy_from_path` resides ~2.8 GB per process, forked
across 8 train + 8 val + 8 test workers. uint8 would be 4× smaller with identical values.
Fold into #34.

### ⬜ 38. `nrmse` metrics computed on a fixed tiny subsample **[V]**
`nrmse_max_batches: 4` with `val_dataloader.shuffle: False` → always the same first ~128
windows, all from the first one or two episodes of the split. Not a split-level estimate.

### ⬜ 39. Metrics that disagree by construction **[R]**
`pusht_search_image_runner.py` reads `T_distance` from the *final* step while
`mean_score`/`success_rate` are the `max` over the episode — an episode that succeeds then
drifts logs `success=1` with a large distance.

### ⬜ 40. Three different search widths **[V]**
Training conditions on `max_actions - 1 = 15` candidates; the in-training rollout uses
`n_search_actions: 8`; `eval_search_pusht.py` sweeps 1…64. Three regimes, one of which
(training) is never the one measured.

### ⬜ 41. Latency-step mismatch is silent in the runner **[V]**
`pusht_image_search.yaml` gives the runner `n_action_steps` without `n_latency_steps` while
the policy gets both. `MultiStepWrapper.step` iterates whatever it is handed, so a nonzero
latency yields a wrong control cadence with no exception. `eval_search_pusht.py` guards this
explicitly; the runner does not.

### ⬜ 42. Double image normalization **[R]**
`LinearNormalizer` maps images to [−1,1], then `imagenet_norm: True` applies ImageNet stats
that assume [0,1], landing the encoder input in roughly [−6.5, 2.2]. Compounded by
`weights: IMAGENET1K_V1` with `use_group_norm: True`, which replaces every `BatchNorm2d` with
a freshly-initialized `GroupNorm` and discards the pretrained statistics the weights depend on.
Repo-wide convention; worth choosing deliberately rather than inheriting.

### ⬜ 43. Config / comment drift **[V]**
`train_pusht_diffusion_search_subgoal{,_verifier}.yaml` claim "checkpointing every 10k" but
inherit `checkpoint_every: 2000`. The base config has `num_epochs: 1000` and no
`max_gradient_steps` while the ablations have `100000` and do. `training.device` is declared
but ignored in all accelerate workspaces. `cfg.dataloader`/`cfg.val_dataloader` are entirely
unread by `train_online_search_workspace`, which builds batches synchronously in the inner
loop instead.

### ⬜ 44. `expert_mask` is all-ones and unused **[V]**
On the `return_sequences=False` path it is collated into every batch and consumed by nobody.

### ⬜ 45. Unguarded / hardcoded eval-script behavior **[V]**
`eval_search_pusht.py` — no overwrite guard (`eval.py` has one); env construction hardcodes
`legacy=False`, `render_size=96`, `max_steps=300`, `n_envs=50` instead of instantiating
`cfg.task.env_runner`, so it can silently drift from the in-training rollout; no torch
seeding, so the n=1…64 curve mixes search width with sampling noise; `seen.add(step)` precedes
the try/except, so a transient failure permanently skips that checkpoint.

---

## D — design-level findings 📄 documented deliberately, no code change

### 📄 46. The search context is conditionally uninformative under the current loss **[V]**
`policy/diffusion_transformer_search_policy.py`, in `compute_loss`:

```python
trajectory = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1)
```

All `max_actions` decode slots are trained to reconstruct the *same* dataset expert action.
The conditioning is `(obs, candidate actions, verifier feedback on those candidates)`, and the
candidates are themselves generated from `obs` alone — so the context is a deterministic
function of `obs`, and

```
p(a* | obs, context) = p(a* | obs)      exactly
```

The Bayes-optimal model therefore **ignores the context**, and no gradient pressure exists for
the feedback channel to matter. Supporting details: `corrupt_obs: False` in all three PushT
configs, and the existing `nrmse_first` vs `nrmse_min` metrics are precisely the diagnostic —
expect `first ≈ min` at convergence. Note also that **nothing currently writes the searched
best-of-N action anywhere**; a repo-wide grep for `distill` returns only wandb project names.

**Refinement: this is task-dependent, and PushT is the worst case.** The argument bites only
where the task is **fully observed**. PushT's obs (image + agent_pos + feedback) determines
the state and the expert action is a function of state, so `context ⊥ target | obs` and the
optimal model provably ignores the context. Maze/procgen are partially observed — one obs
does not reveal the layout but exploration rollouts do, which is the `in_context_exploration_*`
premise — and there the context genuinely carries information. So this is not "the
architecture is broken"; it is "PushT with `corrupt_obs: False` is the one setting where
redundancy is provable." Hence the 3×2 matrix in #33c: the clean row is expected to show
`nrmse_first ≈ nrmse_min` in *every* arm, and the corrupt row is where a real gap can appear.

**Related, and a genuine inconsistency:** `corrupt_obs` means **opposite things** in the two
paths. Offline, `encode_obs_cond` corrupts the **observation** (so the context has something
to add). Online, `forward` corrupts the **context** (a robustness measure). Same method name,
inverted target. Worth renaming one of them.

Three directions, not ranked:
1. **Enable `corrupt_obs`** — the context then carries information `obs` does not, which is
   what the corruption machinery already in the policy appears to have been built for.
2. **Supervise only the final slot**, or weight slots so later ones dominate — the model is
   then trained to *improve* given context rather than to reproduce a context-independent target.
3. **Make the target the searched best-of-N action** — true off-policy distillation, which is
   what the project description says the pipeline is for.

### 📄 47. Search requires a ground-truth simulator at inference **[V]**
`pusht_verifier.rollout` resets a real PushT sim to the exact state and steps true dynamics.
The block *state* is not privileged (it comes from `feedback`, a declared obs key) — but the
*simulator* is. The `subgoal`/`subgoal_value` modes go further, putting a ResNet embedding of
a sim-rendered future frame into the model's input distribution, so the trained model needs
the oracle at inference, not merely for ranking. Worth stating explicitly alongside any result.

### 📄 48. Verifier objective ≠ eval metric **[R]**
The verifier scores the *final* state of a chunk by `−mean keypoint distance`; success is the
`max` **coverage** over the whole episode. Keypoint distance is not monotone in coverage, and
a candidate that passes through the goal mid-chunk and slides off scores badly.

### 📄 49. Verifier dynamics gap **[R]**
Each rollout starts with **zero agent velocity** (a fresh body), so overshoot is systematically
under-predicted; `_set_state` also runs one extra `space.step` the real state never
experienced; and stepping continues past `done`.

### 📄 50. Search is fully sequential **[V]**
N candidates = N sequential 8-step DDIM loops with `K_decode=1`, even though
`SearchTransformerForDiffusion.forward` accepts `(B, K, H, Da)`. At n=64 that is 64 sequential
diffusion loops *per control step*, which likely makes the top of the eval sweep impractical.
Batching N into the batch dim would change the sequential-conditioning semantics, hence
design-level rather than a straight optimization.

### 📄 51. The rolling window drops the incumbent best **[R]**
Past `max_actions`, the window evicts the *oldest* candidate, so conditioning never sees the
current leader; only the final `argmax` over all scores recovers it.

### 📄 52. `torch.inference_mode()` fragility **[V]**
`compute_loss` produces inference tensors that flow into the grad-tracked forward. This works
only because they are consumed by `torch.cat`, which reallocates; feeding them into any op
that saves its input for backward would raise *"Inference tensors cannot be saved for
backward"*. Separately, the context search runs with the model in `train()` mode, so candidates
are drawn with `p_drop_attn=0.2` active — a train/rollout mismatch in the *context distribution*.

---

## False positive, recorded so it is not re-raised

`last_epoch=self.global_step-1` when building the LR scheduler after a resume does **not**
raise `KeyError: 'initial_lr'`. `LambdaLR.__init__` with `last_epoch >= 0` does require
`initial_lr` in every param group — but `optimizer.load_state_dict` restores `param_groups`
including the `initial_lr` the previous run's scheduler set, so the key is present by the time
the scheduler is constructed. Verified against the PyTorch source. Resume works.

---

## Verification performed on the P0 fixes

- `py_compile` clean on all 21 changed files.
- Static sweep for stale references after the renames: no remaining `_predict_action`,
  `supports_return_scores`, `hasattr(policy, 'verifier')`, positional `result[N]` unpacking, or
  `self.policy` in the BET workspace. Remaining `block_pos` hits are the zarr-array reads
  (`get_episode_init_states`, the `feedback` derivation), which are correct and intentional.
- `feedback_util` round-trip (`pose → feedback → pose`) over 20 random poses: max position
  error 1.5e-5 px, max angle error 2.0e-7 rad.

**Not yet run:** import and training/eval smoke tests. The login node cannot do it — a bare
`import diffusion_policy.policy...` sat 15+ minutes under the shared-cgroup contention and had
to be killed. These need a GPU node via `scripts/train_pusht_search.sbatch`:

1. `python train.py -cn train_pusht_diffusion_search training.debug=True` — confirms #2, #3,
   #21, #22 by checking that `global_step` equals the optimizer-step count in the JSON log.
2. Kill after one checkpoint, relaunch with `resume=True` — confirms #1 (resumed weights not
   overwritten) and that the step counter continues rather than restarting.
3. `python eval_search_pusht.py --checkpoint <ckpt> --n-list 1,2`, then `ps` — confirms #7, #8
   (no surviving verifier workers).
4. `python train.py --cfg job --resolve -cn train_ibc_dfo_hybrid_workspace` and one batch —
   confirms #4 (no `KeyError: 'block_pos'`).
