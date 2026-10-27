# Codebase audit — `harine/improved_verifier`

A read of the training, inference, dataset and evaluation path on this branch: every
workspace, search policy, verifier, dataset, sampler/normalizer and env runner that a
config here can reach, plus the eval entry points in `scriptsv2/eval_search`. The vendored
`SimplerEnv` / `openvla-mini` / `sglang-vla` forks are out of scope.

The target shape being measured against: **every task and model shares one structure** —
same policy interface, same dataset contract, same training loop, same checkpoint/eval
path, nothing hacky or divergent from a standard ML pipeline.

**Status** ✅ fixed · 🟡 partly addressed · ⬜ open · 📄 design-level, documented deliberately

**Confidence** **[V]** verified by reading the code directly or by running it · **[R]**
reported by a reading pass, not independently re-verified

**Severity** **P0** silently corrupts a run · **P1** standardization / structure · **P2**
hygiene, efficiency, metric quality · **D** design

---

## 1. What is actually live

Reachability matters throughout this document, because this repo is a fork of upstream
`diffusion_policy` and most of its surface is unreachable from any config here. Two
workspaces carry all of this branch's work:

| workspace | configs | what it trains |
|---|---|---|
| `TrainMLPImageWorkspace` | 49 total, **19** eggplant/sim2real | every search arm (`robomonkey_eggplant_search_*`), the sim2real transformer/MLP finetunes |
| `TrainDiffusionUnetImageWorkspace` | **5** eggplant/sim2real | the eggplant BC arms |
| `TrainSearchOuterInnerWorkspace` | **1** (new) | the outer/inner search arm added in this change |

Shared support on the live path: `common/sampler.py` (`SequenceSampler`, used by
`sim2real_image_dataset.py`), `workspace/base_workspace.py`, `model/common/lr_scheduler.py`,
`model/common/normalizer.py`, `policy/dqc_verifier.py`.

Everything else — the IBC, BET, robomimic, video, lowdim and hybrid workspaces — is
upstream code no config here targets. Findings in those files are recorded but were
deliberately **not** patched: the fix would be untestable here and would add churn to
vendored code.

---

## 2. Fixed in this change

### ✅ 2.1 Pretrained weights clobbered the resumed run **[V]** — P0, live
`workspace/train_mlp_image_workspace.py`

The `pretrained_ckpt_path` block ran *after* the `cfg.training.resume` block. On a restart,
`latest.ckpt` was loaded (weights + optimizer + `global_step`), then the pretrained weights
overwrote the model while the resumed optimizer state and step counter were kept — all
progress silently discarded, the run continuing as though nothing had happened.

Live: the eight `finetune_*_sim2real_image_pomdp_workspace.yaml` configs set
`pretrained_ckpt_path`, and every search config sets `resume: True`.

**Fixed:** gated on `not checkpoint_loaded`. Also `include_keys=['model']` → `include_keys=[]`:
`include_keys` filters `payload['pickles']`, not `state_dicts`, so the original argument never
selected the weights (those are governed by `exclude_keys`) and only let the pretrained run's
`global_step`/`epoch`/`last_checkpoint_step` leak into a fresh run.

### ✅ 2.2 `global_step` and gradient-accumulation bookkeeping **[V]** — P0 latent, **not** a live numerical bug
`train_mlp_image_workspace.py`, `train_diffusion_unet_image_workspace.py`

Both workspaces keyed accumulation on `if self.global_step % gradient_accumulate_every == 0`
— but `global_step` counts *optimizer* steps, so it cannot delimit its own windows. It was
also skipped on the last batch of each epoch and incremented once *extra* at the epoch
boundary.

**The consequence was overstated in the source audit this was backported from, and the
correction matters.** That audit claims `global_step` "drifted by +1 per epoch", making the
`step_*.ckpt` axis "systematically wrong". At `gradient_accumulate_every: 1` — which is what
**all 26** live configs set — the two errors cancel *exactly*: the skipped last-batch
increment (−1/epoch) offsets the stray epoch-end increment (+1/epoch). Simulated across
several loop geometries:

```
batches=100 epochs=50 accum=1: OLD global_step=5000 vs true opt steps=5000 (drift +0)
batches=100 epochs=50 accum=4: OLD global_step=5000 vs true opt steps=1250 (drift +3750)
batches= 37 epochs=200 accum=1: OLD global_step=7400 vs true opt steps=7400 (drift +0)
```

So **no checkpoint name and no success-vs-steps axis produced on this branch is wrong.** The
same applies to the LR-scheduler horizon (`len(dl) * epochs // accum`), which is also exact
at `accum == 1`.

**Fixed anyway**, because the invariant should hold structurally rather than by cancellation:
windows keyed on `(batch_idx + 1) % accum == 0`, the epoch-final and `max_train_steps`
windows always flushed, `global_step` incremented only on optimizer steps, scheduler horizon
`ceil(len/accum) * epochs`. At `accum == 1` this is a **no-op numerically** — the fix becomes
load-bearing the moment anyone raises `gradient_accumulate_every`, which today would silently
quarter the real step count while the logs claimed otherwise.

Also fixed alongside, in the UNet workspace: `ema.step()` ran once per *batch* rather than per
optimizer step, so the EMA decay schedule advanced faster than the weights (again inert at
`accum == 1`).

### ✅ 2.3 `pdb.set_trace()` in the data path **[V]** — P0, live
`common/sampler.py:195` — `except Exception: import pdb; pdb.set_trace()` inside
`sample_sequence`'s `key_first_k` branch. Under `num_workers: 8` (what every eggplant config
uses) this hangs the worker on a stdin read instead of surfacing the error.

Live: `sim2real_image_dataset.py:90-94` populates `key_first_k` for every rgb key, and
`:306`/`:760` call `sampler.sample_sequence`. **Fixed:** removed; the exception propagates.

### ✅ 2.4 Dropped assert **[V]** — P0 latent, live path
`common/sampler.py:14` — `episode_mask.shape == episode_ends.shape` was a bare expression. A
mismatched mask silently under-iterates or indexes out of range, i.e. trains on a different
set of episodes than the split intended. **Fixed** to a real `assert`; verified that numba
`nopython` mode both compiles it and raises on mismatch.

### ✅ 2.5 Checkpoint saves on an unjoined thread **[V]** — P2, not P0
`workspace/base_workspace.py`

`_saving_thread` was overwritten without ever being joined, and no workspace joined it before
returning.

**A second overstated claim from the source audit, corrected:** it says this leaves "the final
checkpoint truncated at process exit". It does not. The save thread is created non-daemon, and
CPython joins all non-daemon threads before the interpreter exits — verified directly. The
real (narrower) exposure is two saves of the **same tag** overlapping: `path.open('wb')` is
evaluated inside the thread, so two concurrent saves of `latest.ckpt` truncate and interleave
on one file. At `checkpoint_every: 1000` that does not arise today.

**Fixed** as hygiene, and because the outer/inner workspace needs the primitive: join the
prior thread before starting a new one, add `join_saving_thread()`, and call it at the end of
`run()` in the live workspaces. Also `mkdir(parents=False)` → `parents=True` (it failed if
`output_dir` did not exist yet).

### ✅ 2.6 In-place mutation of a persistent GPU batch **[V]** — P0, reachable
`train_mlp_image_workspace.py` sampling blocks — `obs_dict = batch['obs']` then
`obs_dict['attention_mask'] = ...`, where `train_sampling_batch` is held on GPU for the whole
run, so the insertion sticks forever. **Fixed:** shallow copy before mutating.

Scope note: the eggplant dataset emits `expert_mask`, never `attention_mask`, so this does not
fire on the eggplant path. It is reachable for the `return_sequences` datasets that go through
`get_collate_fn`.

### ✅ 2.7 Three resume defects **[V]** — P0, live
All three verified by running the scenario, not by reading.

**(a) A run launched with `resume: False` produced unresumable checkpoints.**
`BaseWorkspace.__init__` set `exclude_keys = ['optimizer']` in that case, so the checkpoint
carried no optimizer state. Relaunching the same output dir with `resume: True` then built
the LR scheduler at `last_epoch = global_step - 1` against a fresh optimizer and died:

```
KeyError: "param 'initial_lr' is not specified in param_groups[0] when resuming an optimizer"
```

Note this contradicts the "verified false positive" entry in the pushT audit, which asserts
the `initial_lr` KeyError cannot happen. That holds only when the *saving* run had
`resume: True`.

**Fixed** two ways, because both cases exist: the optimizer is now **always** persisted
(`resume` describes a launch, not a checkpoint — a checkpoint should always be resumable),
and `base_workspace.set_initial_lr` backfills `initial_lr` from each param group's
configured `lr` so **checkpoints already on disk**, which have no optimizer state, still
resume. Verified both: flip-the-flag resume continues 12→24, and a checkpoint with
`state_dicts == ['ema_model', 'model']` resumes 12→24 instead of raising.

**(b) `train_diffusion_unet_image_workspace` re-fit the normalizer on every resume.**
`dataset.get_normalizer()` → `set_normalizer()` ran unconditionally, *after*
`load_checkpoint`, so resumed weights were handed freshly-fitted statistics. The scaling
does move in practice: the split is derived at load time and `discover_zarr_files()` picks
up whatever is in `dataset_dir` today (P1-9), so adding a data file silently rescales a
resumed run's inputs and outputs. **Live for all five eggplant BC configs.**
`train_mlp_image_workspace` already had the preserve-on-resume guard; **fixed** by applying
the same rule.

**(c) `resume: True` with no checkpoint restarted silently.** `hydra.run.dir` is timestamped
in every config here, so a plain re-submit mints a new output dir, finds no `latest.ckpt`,
and starts from scratch without saying so. **Fixed:** both live workspaces now print a loud
`STARTING FROM SCRATCH` warning naming the path it looked in and how to point at the real
run dir.

### ✅ 2.8 Search context could not be buffered **[V]** — prerequisite for §3
Every search policy generated its training context under `torch.inference_mode()`. Inference
tensors carry a permanent restriction against participating in autograd-recorded operations,
so such a context **cannot** be stored and fed to the model on a later step. Verified directly:

```
inference_mode   RuntimeError: Inference tensors cannot be saved for backward...
no_grad          OK  (buffered context usable)
```

**Fixed:** the new `generate_search_context` uses `no_grad`. The offline path was previously
safe only by accident — the context reaches the model through a `torch.cat`, which reallocates
and launders the inference flag.

---

## 3. Added in this change — the outer/inner trainer

`workspace/train_search_outer_inner_workspace.py`, `config/robomonkey_eggplant_search_dqc_outer_inner.yaml`

The offline workspace regenerates the search context inside `compute_loss` on **every**
gradient step: `max_actions - 1 = 15` sequential DDIM samples plus 15 DQC verifier calls per
batch, discarded after one update. That is the dominant term in step time. The new workspace
generates the context once per outer step for a pool of `outer_batch_size` windows and reuses
it for `inner_epochs` passes.

Policy-side support was added to `SearchPolicyRoboMonkeyDiffusion`,
`DiffusionTransformerSearchPolicy` and (partially) `SearchPolicyRoboMonkey`:
`generate_search_context`, `compute_loss(batch, actions=, values=, return_aux=)`, and
`predict_epsilon`. `SearchPolicyRoboMonkeyDiffusionNoiseCond` **raises** instead of inheriting
these — its `tcont` couples the context to the observation encoding, and buffering one without
the other would train on mismatched pairs with no error.

Staleness is measured, not assumed: a frozen collector snapshot is kept for each inner loop and
`train_drift_mse_eps` reports the epsilon-space MSE between the live policy and that snapshot
at matched `(noisy_trajectory, timestep, obs, context)`. For fixed inputs the two snapshots'
reverse kernels are Gaussians sharing the scheduler variance, so that MSE is proportional to
the per-denoising-step KL.

Verified on CPU with a stub policy/dataset (the real arm needs the `dqc` env, a GPU and the
zarr):

- `global_step` equals the optimizer-step count exactly (24 requested → 24 taken)
- the amortization is real: 24 buffer-generating verifier calls vs 72 for the offline path at
  `inner_epochs: 3` — exactly 3.0x, as designed
- drift rises monotonically within an inner loop and **resets** at each new outer step
  (`0.000122 → 0.006666`, then `0.000102 → 0.009872`) — the intended read on whether
  `inner_epochs` is set too high
- resume continues (`12 → 24`) rather than restarting, weights keep moving, and the
  `last_val_step` / `last_rollout_step` cadence anchors survive

Launch path: `scriptsv2/train_search/train_search_outer_inner.sbatch` +
`launch_outer_inner_3task.sh` (eggplant / stack / spoon). Both set `hydra.run.dir` explicitly,
so `resume: True` finds `latest.ckpt` and the §2.7c silent-restart trap does not apply to them.
The LR schedule is matched to what the offline multitask arms *actually* ran, not to what their
sbatch header claims — see P2-9.

✅ **First three runs completed** (2026-08-06, 20k steps each, all `COMPLETED 0:0`):
eggplant 8h09m (a100), stack 6h44m (a100), spoon 7h00m (a40). `global_step` landed on exactly
20000 in all three, confirming the step accounting end-to-end. Drift stayed bounded
(max 0.020 / 0.048 / 0.044, means ~1e-3), so `inner_epochs: 4` is not over-reusing the buffer
at these settings.

| arm | critic match | val rmse `min` | val rmse `first` (no search) | best-of-n gain |
|---|---|---|---|---|
| eggplant | **matched** | 0.249 → 0.212 | 0.439 → 0.424 | 0.190 → **0.212** ↑ |
| stack | not transferring | 0.190 → 0.182 | 0.317 → 0.287 | 0.127 → **0.105** ↓ |
| spoon | not transferring | 0.222 → 0.193 | 0.371 → 0.327 | 0.149 → **0.134** ↓ |

⬜ **The parity claim is still design intent, not a measurement.** No matched offline baseline
was run alongside these, so the "equal loss trajectory at ~`inner_epochs`x lower wall-clock"
claim remains unverified. That comparison is the next thing to run.

⬜ **On stack and spoon the search context is built from a critic that does not transfer.** All
three arms bake the same base eggplant DQC critic, as `launch_multitask_search_trainings.sh`
already does. The BC target is the dataset expert action, so training is well-posed; what is
compromised is how informative the context is. Do not read those two arms as evidence about
search on stack/spoon without a task-appropriate critic.

---

## 4. Open findings

### P1 — structure and standardization

#### ⬜ P1-1. `use_ema` is a silent no-op in the workspace 49 configs use **[V]**
`grep -n ema train_mlp_image_workspace.py` matches nothing, yet the configs targeting it
declare `training.use_ema`. Setting it `True` changes nothing about training, and then makes
`eval.py:40` (`policy = workspace.ema_model`) raise `AttributeError` on the resulting
checkpoint. All live configs currently say `False`, so nothing is wrong *today*.

The new outer/inner workspace implements EMA properly and can be used as the reference for
backporting it. Three details there are silent when wrong: constant decay via
`min_value == max_value`; `ema.optimization_step = self.global_step` on resume (EMAModel has no
`state_dict`, so the counter is never checkpointed — left at 0, `get_decay` returns 0.0 and the
update degenerates to `ema_param.copy_(param)`, destroying the restored average); and
`clone_policy` to avoid duplicating the verifier.

#### ✅ P1-2. `checkpoint.topk` was dead config that looked like model selection **[V]**
`TopKCheckpointManager` is imported at `train_mlp_image_workspace.py:28` and never
instantiated; it *is* instantiated in 10 workspaces, none of which this branch runs. So the
`topk` block in every eggplant/sim2real config was read by nobody and **no best-metric
checkpoint was ever written** — only `latest.ckpt` and `step_*.ckpt`.

Worse than dead: it could not have selected the right checkpoint even if wired up. All 26
live blocks monitored a **loss** (`train_action_mse_error` ×16, `train_loss` ×5, `val_loss`
×5), and on this task loss does not rank policies — success rate does, and it is measured
out-of-band by `scriptsv2/eval_search`. Monitoring `train_loss` in particular selects the
most-overfit checkpoint by construction.

**Removed** from the 24 configs whose workspace never instantiates the manager, replaced by a
comment stating where selection actually happens. Deliberately **left intact** in
`train_diffusion_transformer_sim2real_image_workspace.yaml` and
`train_diffusion_unet_sim2real_lowdim_workspace.yaml`, whose workspaces really do construct a
`TopKCheckpointManager` — removing it there would raise. All 27 configs re-resolved clean.

This is also why the upstream `checkpoint_util.py:26` bug (`data[self.monitor_key]` KeyErrors
when the monitored metric is absent) was not patched: it cannot fire on this branch. Fix it
first if topk is ever wired up against a real success metric.

#### ⬜ P1-3. DDP is configured and non-functional **[V]**
`compute_loss` is called on `self.accelerator.unwrap_model(self.model)` in every accelerate
workspace, bypassing `DistributedDataParallel.forward` — so the all-reduce hook never fires and
replicas silently diverge. Compounding: `lr_scheduler` is passed to `accelerator.prepare` *and*
stepped manually, so `AcceleratedScheduler.step()` advances the wrapped scheduler
`num_processes` times per call; and `env_runner.run()` is not guarded by `is_main_process`.
Single-GPU runs are unaffected; any `world_size > 1` run is wrong. `Accelerator()` is otherwise
pure overhead.

#### ⬜ P1-4. Dead and unimportable code **[V]**
- `policy/search_policy.py:14` has a module-level `from l2s.verifier import MazeVerifier`.
  Confirmed by running: `ModuleNotFoundError: No module named 'l2s'`. The module is
  unimportable in every environment on this machine, so the **eight** configs targeting
  `SearchPolicy` (`maze_search*.yaml`, `procgen_maze_search.yaml`, `train_search.yaml`) cannot
  run. `DiffusionTransformerSearchPolicy` does the same import lazily inside `__init__`, which
  is the pattern to copy.
- `config/train_search.yaml` targets `train_search.TrainSearchWorkspace` — a module that does
  not exist anywhere in the repo. Verified by resolving every config's `_target_` against the
  filesystem; it is the only one that misses.

#### ⬜ P1-5. Three near-duplicate search-policy classes **[V]**
`search_policy_robomonkey.py` defines `SearchPolicyRoboMonkey`, `SearchPolicyRoboMonkeyDiffusion`
and `SearchPolicyRoboMonkeyDiffusionNoiseCond`, each re-implementing `search_candidates`,
`predict_n_actions` and `_normalize_context_actions` almost verbatim (the file is ~1,450 lines).
`DiffusionTransformerSearchPolicy` is a fourth copy of the same loop.

This is not style. It is the direct mechanism by which a fix lands in one class and not the
others — the `_normalize_context_actions` fix in `7f06736` had to be applied five times, and
adding `generate_search_context` in this change had to be done three times. The pushT branch
has already collapsed this into a base class with `_score_candidates` / `_context_dim` hooks;
that structure should come back here.

#### ⬜ P1-6. `**kwargs` as an untyped config transport **[V]**
`kwargs['max_actions']` raises a bare `KeyError` with no indication which config key is
missing, and `search_context`, `verifier_*`, `scheduler_step_kwargs`, `maze_path` all arrive
unvalidated — so a **typo'd config key is silently ignored** and the policy stays on its
defaults. That is exactly how an ablation arm ends up secretly identical to its sibling. The
pushT `_validate_kwargs` (raise, naming the unknown keys and listing the known set) is the fix.

#### 🟡 P1-7. Inference behavior configured by environment variable **[V]** — recorded, not yet moved
- `BCSearchWrapper.predict_n_actions` branches on `BC_VERIFIER_SCORE`, `BC_VERIFIER_STEP`,
  `BC_VERIFIER_AGG`, `BC_CHUNK_FROM_PRED`, `BC_VERIFIER_DISCOUNT` — five env vars selecting
  between materially different scoring semantics (per-step mean vs discounted sum vs a real
  chunk critic), with `assert B == 1` scattered through it.
- `_run_search` branches on `ROBOMONKEY_KV_PREFIX` / `ROBOMONKEY_KV_CHUNK`.
- `scriptsv2/eval_search/eval_search.py` (1,339 lines) reads **16 distinct** env vars across 32
  sites, including which verifier checkpoint to load and how candidates are selected.

The consequence was that an evaluation's configuration was not recorded anywhere: the same
command produces different numbers depending on the shell it ran in, and nothing in the output
said which. Two rows of the same results table could come from materially different scoring
procedures with nothing to distinguish them.

**Partly addressed.** `eval_search.py` now writes `<output_dir>/run_config.json` before the
first episode runs (so a preempted eval still records what it was attempting), containing:
the full argv and parsed args; the **effective** value of all 20 env vars in the new
`EVAL_ENV_VARS` registry, each tagged `source: env|default` so an unset variable is
unambiguous rather than absent; the checkpoint path, size, mtime and a head-md5 fingerprint
(cheap identity, catches "same path, different file"); git sha + branch + **dirty flag** for
both repos; SLURM job id, host, and library versions. Running without `--output-dir` now
warns that the run is unreproducible.

Still open: this **records** the sprawl, it does not remove it. The variables belong in Hydra
config where they would be serialized into the run by construction. See P2-8 for the one
concrete hazard the registry work surfaced.

#### ⬜ P1-8. Cadence semantics are inconsistent **[V]**
`checkpoint_every` means gradient steps in `train_mlp_image_workspace` and epochs in most
others. `val_every` / `sample_every` / `rollout_every` are epochs while `checkpoint_every` is
steps, so **no rollout number is reproducible from a checkpoint** — the two never coincide. The
debug block sets `checkpoint_every: 1`, which under step semantics writes a full checkpoint
every batch. The new outer/inner workspace puts all four on gradient steps; that is the shape
to converge on.

#### ⬜ P1-9. The train/val split is derived at runtime and recorded nowhere **[V]**
`Sim2RealImageMultiDataset` partitions from `val_ratio: 0.05` and `seed` at load time. Nothing
on disk records which episodes a given checkpoint trained on, so any change to the seed, the
ratio, or the set of zarr files discovered under `dataset_dir` silently repartitions the data
and nothing notices. `discover_zarr_files(dataset_dir)` in particular means **adding a file to
the data directory changes the split of every subsequent run**. The pushT branch fixed the
equivalent exposure with a committed split manifest plus a refuse-to-resume-if-changed guard.

#### ⬜ P1-10. Bare attribute and config access **[V]**
`train_mlp_image_workspace.py` reads `dataset.return_sequences` and
`cfg.training.freeze_encoder` without `getattr` / `.get`, so any dataset or config predating
those keys raises `AttributeError` before training starts rather than falling back.

#### ⬜ P1-11. `exclude_keys = ['optimizer']` survives in one unreachable workspace **[V]**
Fixed on the live path (§2.7a), but `train_diffusion_transformer_hybrid_workspace.py:69` still
drops the optimizer when `resume: False`, so checkpoints from that workspace remain
unresumable. Not patched: no config on this branch targets it. Same one-line change if it is
ever brought back into use.

### P2 — hygiene, efficiency, metric quality

#### ⬜ P2-1. `LinearNormalizer._normalize` accepts wrong-width tensors **[V]**
`normalizer.py:272` — `x.reshape(-1, scale.shape[0])` with no assertion that `x`'s last
dimension matches. For the rgb keys, `create_identity()` gives `scale.shape == (1,)`, so the
call accepts **any tensor at all**. One slicing typo away from silently corrupting the context
rather than raising.

#### ⬜ P2-2. Search is fully sequential **[V]**
`search_candidates` runs `max_actions - 1` sequential DDIM loops with `K_decode = 1`, even
though `SearchTransformerForDiffusion.forward` accepts `(B, K, H, Da)`. This is the reason the
outer/inner trainer exists; batching the candidate dimension would change the sequential
conditioning semantics, so it is a design decision rather than a free optimization.

#### ⬜ P2-3. The training search width is the one never measured **[V]**
Training conditions on `max_actions - 1 = 15` candidates; `eval_search.py` sweeps a different
range. The regime that determines what the model learns is not the regime that gets reported.

#### ⬜ P2-4. `nrmse` metrics come from a fixed tiny subsample **[V]**
`nrmse_max_batches: 4` against `val_dataloader.shuffle: False` always scores the same first few
windows, drawn from the first one or two episodes of the split. That is not a split-level
estimate, and it is what the search-quality curves are plotted from.

#### ⬜ P2-5. `DummyRunner` means there is no in-training success signal **[V]**
`config/task/robomonkey_eggplant_state.yaml` sets `env_runner: DummyRunner`, whose `run()`
returns `{}`. So `rollout_every` is inert and success rate is only ever measured out-of-band by
`scriptsv2/eval_search`. Worth stating explicitly wherever training curves are shown: nothing
in a training run reports task performance.

#### ⬜ P2-10. `persistent_workers: true` crashes any `num_workers: 0` override **[V]**
Every eggplant/sim2real config sets `persistent_workers: true`, and torch raises
`ValueError: persistent_workers option needs num_workers > 0` when the two disagree. So
overriding `dataloader.num_workers=0` — a debug run, or a low-CPU partition — turns into a
crash, and it is raised **after** the dataset has already been loaded and the DQC critic
built, i.e. minutes into the job, for what is purely a configuration inconsistency.

Caught by the GPU smoke run of the outer/inner trainer (job 38177669 failed at 11:31 on
exactly this). **Fixed in `train_search_outer_inner_workspace.py`** via `_dataloader_kwargs`,
which drops `persistent_workers` / `prefetch_factor` when `num_workers == 0` for both the
train and val loaders.

`TrainMLPImageWorkspace` and `TrainDiffusionUnetImageWorkspace` have the identical exposure
(`DataLoader(dataset, **cfg.dataloader)`), not patched because no committed config sets 0
workers there — but the same one-line guard applies.

#### ⬜ P2-9. `training.lr_end` and `training.lr_power` are silently ignored **[V]**
`scriptsv2/train_search/train_search_pg.sbatch` passes `+training.lr_end=1e-5` and
`+training.lr_power=1.0`, and its header documents the schedule as "polynomial: warmup 500 →
1e-4, linear decay to **1e-5** by 10k, then hold". But `TrainMLPImageWorkspace` forwards only
`training.lr_scheduler_kwargs` into `get_scheduler`; a repo-wide grep for `lr_end` / `lr_power`
finds them in **comments only**. The `+` prefix adds the keys to the config, where nothing
reads them.

So every PG and multitask search training actually ran polynomial decay to diffusers' default
`lr_end = 1e-7` — two orders of magnitude below the documented floor — and to `power=1.0`
only because that is also the default. The runs are internally consistent and comparable with
each other; the *description* of them is wrong. A concrete instance of P1-6 (untyped config
transport: a key that looks like configuration and is inert).

The new `train_search_outer_inner.sbatch` deliberately does **not** pass these, and its header
states the effective floor, so the outer/inner arms match what the offline arms really did
rather than what their header claims. Fix direction: route them through
`lr_scheduler_kwargs`, or drop them and document the real default.

#### ⬜ P2-8. `N_SAMPLES` is a floor, not a default, on the non-DQC BC path **[V]**
`scriptsv2/eval_search/eval_search.py:949-953`. `N_SAMPLES` is N in best-of-N — the candidates
drawn per control step, verifier-scored, one of which is executed. On the BC branch it sets
`BCSearchWrapper(max_actions=...)`, which becomes the default search width:

```python
if _vtype == "dqc":
    _max_actions = int(os.environ.get("N_SAMPLES", "16"))
else:
    _max_actions = max(128, int(os.environ.get("N_SAMPLES", "128")))   # <- floor
```

The branches are mutually exclusive, so there is no internal disagreement — but the `else`
arm **clamps N up to 128**. With the RoboMonkey/LLaVA verifier (`VERIFIER_TYPE` unset — the
default), `N_SAMPLES=4` silently runs 128 candidates, and every point below N=128 in a BoN
sweep would collapse onto the same number while the output directory name still says `n4`.

**Currently latent, verified:** the two BC sweeps that vary N (`launch_bc_v2_argmax.sh`
N=1…64 → `eval_bc_dqc_chunk.sbatch`, and `sweep_bc_dqc.sh` N=1…256 → `eval_bc_dqc.sbatch`)
both submit sbatch scripts that set `VERIFIER_TYPE=dqc`, so they take the honest branch. No
published BoN curve is affected. The trap is live for any LLaVA-verifier run that passes a
small `N_SAMPLES`. Make it a plain default, or raise if the requested N is overridden.

#### ⬜ P2-6. Another `pdb.set_trace()` **[V]**
`common/timestamp_accumulator.py:68`. Real-robot path, not reachable from any config here, so
not patched — but it is the same trap as §2.3.

#### ⬜ P2-7. Small ones **[V]**
- `expert_mask` is emitted into every eggplant batch and consumed by nobody.
- `base_workspace.save_checkpoint` passes `path.open('wb')` to `torch.save` with no `with`, so
  the handle is closed only by refcounting inside the save thread.
- Unused imports across the workspaces (`shutil`, `optimizer_to` in both robomimic workspaces,
  `EMAModel` in the IBC workspaces).
- `training.device` is declared in every config and ignored — accelerate takes the device from
  `Accelerator()`. Setting `cuda:1` does nothing.
- `multi_run:` blocks are present in the eggplant configs and read by nothing.

---

## 5. Design findings 📄

### 📄 5.1 The search context is conditionally uninformative under the current loss **[V]**
In `compute_loss`:

```python
trajectory = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1)
```

All `max_actions` decode slots are trained to reconstruct the **same** dataset expert action.
The conditioning is `(obs, candidate actions, verifier feedback on those candidates)` — and the
candidates are themselves generated from `obs` alone. So the context is a deterministic function
of `obs`, and

```
p(a* | obs, context) = p(a* | obs)      exactly
```

The Bayes-optimal model therefore **ignores the context**, and no gradient pressure exists for
the feedback channel to matter. The `*_first` vs `*_min` metrics added in this change are
precisely the diagnostic: expect `first ≈ min` at convergence wherever this bites.

How hard it bites is task-dependent — the argument is exact only when the task is fully
observed. It connects directly to the observed result that the DQC Q is exploitable
off-manifold (random action diversity → 0% success): if the context is ignored and the verifier
prefers off-manifold actions, search selects confidently among candidates the model was never
trained to rank.

**First evidence, and it does not support the strong form of this claim [V].** The three
outer/inner runs (§3) log `first` (candidate 0, empty context = the no-search readout) beside
`min`. If the model were simply ignoring the context, the gap should close as BC improves. It
closes on two arms and **widens on the third**:

- eggplant (critic matched to the task): gain 0.190 → **0.212**, because `min` improved
  (0.249 → 0.212) while `first` barely moved (0.439 → 0.424)
- stack: 0.127 → 0.105 · spoon: 0.149 → 0.134 — both narrow, both scored by the same
  eggplant critic, which is already known not to transfer to those tasks

So the narrowing tracks **which critic scores the candidates**, not the loss structure. That
is a different mechanism from §5.1's argument and partially counts against it.

⚠️ **Confounded, deliberately recorded as such:** critic-match is perfectly aligned with task
identity here (eggplant is both the matched-critic arm *and* a different dataset and
difficulty), and n=1 per task. The clean test is to break that alignment — run eggplant with a
deliberately wrong task embedding, or stack with a stack-finetuned critic — and see whether the
gain direction follows the critic rather than the task. Until then this is a hypothesis with a
successful prediction, not a result.

Three directions, not ranked:
1. **`corrupt_obs` / a `mask_obs` arm** — make the observation partially informative so the
   context carries something `obs` does not. `mask_obs` is the cleaner test (unambiguous,
   already implemented for the Gaussian policy) versus corruption, which is uncalibrated (5.2).
2. **Weight the later slots** — train the model to *improve* given context rather than to
   reproduce a context-independent target.
3. **Make the target the searched best-of-N action** — true off-policy distillation, which is
   what the project describes itself as doing. Worth noting that **nothing currently writes the
   searched best-of-N action anywhere**.

### 📄 5.2 `corrupt_obs` is uncalibrated, and means opposite things in the two paths **[V]**
`corrupt_obs_features` gates on `self.corrupt_obs` and `self.obs_noise_level`, not on
`self.training`, so with a fixed level the corruption is applied at **evaluation** too — the
corrupt arms are a strictly harder task, not the same task with a different training signal.
The noise is `torch.randn_like(obs_features)`: **absolute** std 1, independent of feature
scale, applied to a vector that concatenates unnormalized encoder activations with
gaussian-normalized low-dim keys — so the corruption strength differs across the feature vector
at an unknown overall SNR. With `obs_noise_level: None` the timestep is drawn uniformly over
all 100 scheduler steps, so there is no single corruption level to report.

Separately, each candidate in one search gets its own corruption draw, though the agent has one
observation, not 16.

And the same flag name means opposite things in the two search lines: offline it corrupts the
**observation** (so the context has something to add); in the online/pushT path it corrupts the
**context** (a robustness measure). One of them should be renamed.

### 📄 5.3 The verifier's objective is not the eval metric **[R]**
The DQC chunk critic scores `Q(frame, chunk)` under its own training distribution; success is
measured by SimplerEnv task completion. These are related but not the same ordering, and the
BoN selection optimizes the former.

---

## 6. Not bugs — recorded so a later pass does not "fix" them

- **The eggplant normalizer does not double-normalize images.** `get_normalizer` assigns
  `create_identity()` to the rgb keys and lets `imagenet_norm` in the encoder do the work. The
  equivalent path on the pushT branch *does* double-normalize; this one is correct.
- **`last_epoch = self.global_step - 1` on resume is fine** *when the saving run had*
  `resume: True` — `optimizer.load_state_dict` restores `initial_lr` into the param groups
  before the scheduler is constructed. The pushT audit records this as an unconditional false
  positive; it is not. See §2.7a for the case where it genuinely raises, and the two-part fix.
- **`torch.inference_mode()` in `predict_n_actions`** is correct and was deliberately left: that
  path is eval-only and its outputs are never buffered into a later gradient step. Only the
  training-context generator needed `no_grad` (§2.7).
- **`gradient_accumulate_every: 1` made §2.2 numerically inert.** Do not re-report the fixed
  bookkeeping as having corrected any published number.
