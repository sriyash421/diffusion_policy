# Push-T experiments (branch `harine/pushT`)

This branch adds a **feedback-conditioned** Push-T task, an **online-search** in-context
policy, and **best-of-N** evaluation/visualization on top of upstream Diffusion Policy.
The root [README.md](README.md) is still the upstream Columbia doc and does **not** cover
any of this — use this file for the Push-T work here.

What's new relative to upstream:

| Area | Files |
| --- | --- |
| Feedback obs (16-dim goal-vs-achieved keypoint displacement) | [`diffusion_policy/env/pusht/feedback_util.py`](diffusion_policy/env/pusht/feedback_util.py), [`pusht_feedback.py`](diffusion_policy/env/pusht/pusht_feedback.py) |
| Task configs | [`config/task/pusht_image.yaml`](diffusion_policy/config/task/pusht_image.yaml), [`config/task/pusht_image_online.yaml`](diffusion_policy/config/task/pusht_image_online.yaml) |
| Online-search policy + workspace | [`policy/online_search_policy.py`](diffusion_policy/policy/online_search_policy.py), [`workspace/train_online_search_workspace.py`](diffusion_policy/workspace/train_online_search_workspace.py), [`config/train_online_search.yaml`](diffusion_policy/config/train_online_search.yaml) |
| Best-of-N eval + video | [`eval_bon.py`](eval_bon.py), [`bon_video.py`](bon_video.py) |

---

## 1. Environment

Use the existing `robodiff` conda env (already has torch 2.4.0+cu121, diffusers,
hydra-core 1.2, zarr, pygame, pymunk, shapely, robomimic, wandb, dill — everything
these scripts import):

```bash
conda activate robodiff
```

If recreating from scratch: `mamba env create -f conda_environment.yaml`.

> **GPU required.** These are ~278M-param image UNets; a CPU run does not complete a
> single batch in practical time. Launch training/eval on a GPU node (`sbatch`/`srun`).

---

## 2. Dataset  ⚠️ needs a post-download patch

The Push-T dataset is `pusht_cchi_v7_replay.zarr` (206 episodes). It's the **same** file
whether you get it from Columbia or the "gym-pusht" folder the configs originally
referenced — there is no separate gym-pusht data release; that path was just where the
original author stored this file.

```bash
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip   # ~31 MB
unzip pusht.zip
mv pusht/pusht_cchi_v7_replay.zarr .        # configs expect it at data/<zarr>, not data/pusht/<zarr>
rm -rf pusht pusht.zip && cd ..
```

**The raw zarr ships a single 5-dim `data/state` array, but this branch's dataset and
env-runner expect separate `data/agent_pos` (2d) and `data/block_pos` (3d) arrays**
([`pusht_image_dataset.py:88`](diffusion_policy/dataset/pusht_image_dataset.py#L88),
[`pusht_image_runner.py:54`](diffusion_policy/env_runner/pusht_image_runner.py#L54)).
Loading the unpatched file fails with `KeyError: 'agent_pos'`. Derive them once:

```bash
python - <<'PY'
import zarr, numpy as np
g = zarr.open('data/pusht_cchi_v7_replay.zarr', 'a')
d = g['data']; s = np.asarray(d['state'])   # [agent_x, agent_y, block_x, block_y, theta]
for name, arr in [('agent_pos', s[:, :2]), ('block_pos', s[:, 2:5])]:
    d.array(name, arr.astype(np.float32), chunks=(161, arr.shape[1]),
            compressor=d['state'].compressor, overwrite=True)
print('data keys:', list(d.keys()))
PY
```

This split is exact: resetting `PushTImageEnv(legacy=False)` to a recorded state
round-trips with 0.0 error, and `compute_feedback_from_pose(block_pos)` equals the live
`PushTFeedbackWrapper` feedback with 0.0 error. Re-downloading the dataset means redoing
this step.

All Push-T configs on this branch already point at `data/pusht_cchi_v7_replay.zarr`
(relative to repo root — hydra does not chdir here, so run commands from the repo root).

### 2.1 The split manifests — how the 29 / 100 / 25-demo splits and the 50 test episodes were made

The 206 episodes are partitioned by a **committed manifest**, never at runtime. Every
manifest below is already in the repo, so **you do not need to regenerate anything to
reproduce a trained policy** — a run reads `task.dataset.split_file` and validates the
`n_*_episodes` keys against it. The commands are here so the derivation is auditable and so
a new budget can be cut the same way.

| manifest | train | val | test | used by |
|---|--:|--:|--:|---|
| `pusht_seed42_train100.json` | 100 (12,899 fr) | 30 | 50 | every 100-demo arm, `bc_demos-100` |
| `pusht_seed42_legacy_val10_train29.json` | 29 (3,707 fr) | **10** | 50 | the six **legacy-29** arms, `bc_demos-29` |
| `pusht_seed42_train29_val30.json` | the **same** 29 | 30 | 50 | the six **r8** arms |
| `pusht_seed42_train25.json` | 25 (3,325 fr) | 30 | 50 | `bc_demos-25` (archive) |
| `pusht_seed42.json` | 100 | 30 | 50 | identical to `train100`; the original default |

Two invariants hold across all five, and both are load-bearing:

- **The 50 test episodes are byte-identical in every manifest.** `test = perm[:50]` under
  seed 42 does not move when the val split or the training budget changes, which is what
  makes test numbers comparable across every section of `SUCCESS_RATES.md`.
- **Budgets nest.** Train is a *prefix* of the permuted remainder, so the 25 episodes are a
  strict subset of the 100.

All five also record `episode_ends_checksum: c88b56c9af8c5ee778293e50768ae0a5` — a hash of
the zarr's episode boundaries. If you re-download the data and redo the `agent_pos` patch
above, check this still matches before trusting any stored index.

#### Regenerating them

Three of the five are a prefix derivation and come straight from the CLI. `-o` is required
for anything other than the default path:

```bash
# 100-demo (the default: seed 42, 50 test, 30 val, 100 train)
python scripts/dump_pusht_splits.py -o diffusion_policy/config/splits/pusht_seed42_train100.json
python scripts/dump_pusht_splits.py -o diffusion_policy/config/splits/pusht_seed42.json

# 25-demo
python scripts/dump_pusht_splits.py --n-train-episodes 25 \
    -o diffusion_policy/config/splits/pusht_seed42_train25.json
```

**The legacy 29 cannot be regenerated by that CLI, and `--n-train-episodes 29` does not
reproduce it.** Those runs used `train_ratio: 0.2` — a fraction of what remained after test
and val — which routes through `downsample_mask`'s own `rng.choice` rather than taking a
prefix. The two sets **overlap in only 4 of 29 episodes**. The committed file records the
real derivation (`get_split_masks_3way(n_test=50, n_val=10)` then
`downsample_mask(train_ratio=0.2, seed=42)`) and is the source of truth; treat it as data,
not as something to re-derive.

#### The indices themselves

Since the 29 is data rather than a derivation, here it is. These are episode indices into
`data/pusht_cchi_v7_replay.zarr` (206 episodes) and mean nothing against any other zarr —
check `episode_ends_checksum` first, see *Verifying* below.

**train, 29 episodes (3,707 frames)** — byte-identical in `pusht_seed42_legacy_val10_train29.json`
and `pusht_seed42_train29_val30.json`; this is the set the CLI cannot reproduce:

```
15, 16, 23, 39, 72, 73, 74, 80, 87, 95, 97, 99, 114, 115, 125, 130, 133, 134, 138, 143,
144, 156, 161, 166, 172, 174, 181, 191, 194
```

**test, 50 episodes (6,232 frames)** — identical in *all five* manifests. This is the anchor
that makes test numbers comparable across every section of `SUCCESS_RATES.md`:

```
3, 7, 8, 9, 20, 24, 25, 26, 29, 33, 37, 38, 46, 50, 51, 54, 56, 67, 70, 75, 79, 84, 94,
96, 98, 101, 104, 106, 107, 108, 112, 119, 132, 146, 149, 150, 152, 154, 159, 167, 168,
178, 184, 195, 196, 197, 198, 201, 202, 203
```

**val** — the one split that differs between generations:

```
r8    (30, 3,462 fr)  0, 1, 2, 4, 5, 6, 12, 27, 28, 32, 52, 55, 62, 66, 68, 81, 82, 86,
                      88, 89, 92, 120, 123, 145, 153, 155, 157, 171, 180, 205
100   (30, 3,586 fr)  0, 2, 4, 12, 27, 28, 32, 52, 55, 62, 66, 68, 80, 81, 82, 86, 88,
                      89, 92, 114, 120, 123, 138, 145, 153, 155, 157, 171, 180, 205
legacy(10, 1,147 fr)  2, 27, 82, 86, 88, 123, 153, 155, 180, 205
```

Print any of them without loading the images:

```bash
python -c "import json;m=json.load(open('diffusion_policy/config/splits/pusht_seed42_train29_val30.json'));print({k:m[k] for k in ('train','val','test')})"
```

**The 29- and 100-demo manifests are different partitions, not nested budgets.** Each is
internally disjoint, but they disagree with each other, so a split label is only meaningful
relative to its own manifest:

| relation | result |
|---|---|
| `test` | **identical** — test numbers are comparable across every arm |
| train29 ∩ train100 | 23 of 29 — overlapping, *not* a subset |
| train29 ∩ val100 | **{80, 114, 138}** — trained on at 29 demos, held out at 100 |
| val29 ∩ train100 | **{1, 5}** — held out at 29 demos, trained on at 100 |
| val29 ∩ val100 | 27 of 30 |

Practical consequence: **never compare a `val_*` number across the two budgets**, and never
select a checkpoint for one budget using the other's val split. Within an arm it is fine —
val only ever chooses a checkpoint and is never reported. (`train25` *is* a strict prefix
subset of `train100`, so that pair does nest.)

For contrast, the prefix derivation the CLI implements gives a genuinely different 29:
`--n-train-episodes 29` overlaps the real set in only **4 of 29** episodes at `n_val 10`
(the like-for-like comparison against the legacy config), or 7 of 29 at `n_val 30`.

The r8 manifest is built *from* the legacy one, so the two generations share their 29
training episodes exactly:

```bash
python scripts/make_29demo_parity_split.py        # -> pusht_seed42_train29_val30.json
```

It takes the legacy 29 verbatim as train, the standard 50 as test, and draws 30 val
episodes in the Round-7 permutation order minus anything the legacy train set claims. Val
widened from 10 to 30 because at 10 episodes SE is ~9.5 pp at p=0.9 — three legacy arms
tied at 9/10 while their test numbers were 84 / 70 / 32%. Val only ever *chooses* a
checkpoint and is never reported, so widening it changes no published number.

#### Verifying

```bash
python scripts/dump_pusht_splits.py --verify \
    -o diffusion_policy/config/splits/pusht_seed42_train100.json   # file matches the derivation
python scripts/dump_pusht_splits.py --check-zarr \
    -o diffusion_policy/config/splits/pusht_seed42_train100.json   # zarr still matches the file
```

`--verify` guards against the manifest and the derivation drifting apart; `--check-zarr`
guards against the *data* changing under fixed indices, which the in-place `agent_pos`
heredoc above makes a real failure mode. Pass the same `--seed` / `--n-*-episodes` the
manifest was built with — they are recorded in its own `derivation` block. Neither flag
works on the legacy-29 or r8 manifests, whose derivations the CLI does not implement.

Those two can still be checked, in the two ways that matter. **Does the zarr still match?**
Every manifest records `episode_ends_checksum`, so this is manifest-agnostic:

```bash
python -c "
import zarr, numpy as np, hashlib
ee = np.asarray(zarr.open('data/pusht_cchi_v7_replay.zarr','r')['meta']['episode_ends'])
print(len(ee), hashlib.md5(np.ascontiguousarray(ee.astype(np.int64)).tobytes()).hexdigest())
"
# expect: 206 c88b56c9af8c5ee778293e50768ae0a5
```

The `agent_pos` / `block_pos` patch adds arrays under `data/` and leaves `meta/episode_ends`
untouched, so a patched and a freshly-patched zarr are interchangeable here.

**Does the legacy 29 still match its recorded derivation?** It does reproduce — "cannot be
regenerated" means the *CLI* has no flag for it, not that the derivation is lost:

```bash
python -c "
import json, numpy as np, zarr
from diffusion_policy.common.sampler import downsample_mask
from diffusion_policy.dataset.pusht_image_dataset import get_split_masks_3way
ee = zarr.open('data/pusht_cchi_v7_replay.zarr','r')['meta']['episode_ends'][:]
tr, _, _ = get_split_masks_3way(n_episodes=len(ee), n_test_episodes=50,
                                n_val_episodes=10, seed=42)
tr29 = downsample_mask(tr, max_n=round(0.2 * int(tr.sum())), seed=42)
got = sorted(int(i) for i in np.nonzero(tr29)[0])
want = sorted(json.load(open('diffusion_policy/config/splits/pusht_seed42_train29_val30.json'))['train'])
print('match:', got == want)
"
# expect: match: True   (pool after 50 test + 10 val = 146; round(0.2*146) = 29)
```

Why any of this exists: the splits used to be derived independently in three places from
`(seed, n_test_episodes, n_val_episodes, n_train_episodes, train_ratio)`, with nothing
recording which episodes a checkpoint had trained on. Raising `n_val_episodes` from 10 to
30 shrank the pool `train_ratio: 0.2` was a fraction *of*, silently cutting the training
budget from 29 episodes to 25 (`AUDIT.md` P0-5).

---

## 3. Behavior-cloning baseline (feedback-conditioned)

Standard diffusion-UNet BC on the modified Push-T task. The `task=pusht_image` override
is required (the workspace defaults to `lift_image_abs`).

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=pusht_image \
  training.device=cuda:0 training.seed=42
```

- Task adds a 16-dim `feedback` obs; splits into 50 held-out **test** episodes and 156
  train, of which `task.dataset.train_ratio=0.2` → 31 episodes are actually trained on
  (test is never subsampled). Override `train_ratio` to change the data budget. Note this
  2-way `pusht_image` task still derives its split from the seed; only the search task
  (`pusht_image_search`) uses the pinned manifest described below.
- Rollouts run every `training.rollout_every_steps=5000` steps; success logged as
  `test/mean_score`. Checkpoints keep top-5 by `test_mean_score` plus `latest.ckpt`.
- Output: `data/outputs/<date>/<time>_train_diffusion_unet_image_pusht_image/` — note this
  baseline uses the shared upstream config and so still writes to **home**, unlike the
  PushT search configs (section 7). Pass `hydra.run.dir=$DP_OUTPUT_ROOT/...` for a long run.
- W&B project defaults to `diffusion_policy_debug` (override `logging.project=...`).

---

## 4. Online-search policy (in-context)

Single-step policy that conditions on on-policy rollouts. Task is pinned to
`pusht_image_online` in the config's defaults — no `task=` override needed.

```bash
python train.py --config-name=train_online_search training.device=cuda:0
```

- `policy.n_trajs=8` context rollouts (built with `policy.n_envs=32` parallel Push-T
  envs, up to `policy.max_rollout_steps=300` each). Set **`policy.n_trajs=0`** for the
  pure-BC ablation (no env in the loop).
- Checkpoints keep top-5 by `train_action_mse_error`; W&B project `pusht_online_search`.

---

## 5. Training loop structures (read this before comparing runs)

The two search families on this branch train in structurally different ways, which is easy
to miss because both are called "search".

**Offline conditional-diffusion search** — `train_pusht_diffusion_search` and the two
`_subgoal*` ablations, via
[`TrainMLPImageWorkspace`](diffusion_policy/workspace/train_mlp_image_workspace.py):

```bash
read A P < <(bash scripts/slurm/pick_gpu.sh)
sbatch --account=$A --partition=$P scripts/slurm/train_pusht_search.sbatch
```

A plain offline epoch loop — one dataset batch, one `compute_loss`, one optimizer step.
The search is *inside* the loss
([`compute_loss`](diffusion_policy/policy/diffusion_transformer_search_policy.py#L825)):
each step generates `max_actions - 1` = 15 candidates per batch element from the **current**
weights, each an 8-step DDIM sample plus a physics-simulated `PushTVerifier` rollout, then
trains the denoiser to predict the GT expert action conditioned on prefixes of them. All of
it is discarded after the update. In outer/inner terms this is the maximally on-policy
extreme: outer loop = one batch, inner loop = one update.

Real hyperparameters: 206 episodes → **50 test / 30 val / 100 train** (26 unused), i.e.
**12,899 frames → 12,199 training windows, 382 batches/epoch at batch 32, ~52 epochs over a
20k-step run**.
The split is not derived at runtime — it is pinned by
[`diffusion_policy/config/splits/pusht_seed42.json`](diffusion_policy/config/splits/pusht_seed42.json),
which names the exact episode indices; see "Splits" below. Horizon 16, `n_obs_steps` 2,
`n_action_steps` 8, `max_actions` 16, ResNet18 encoder, 4-layer transformer at `n_emb` 256,
AdamW 1e-4, EMA 0.995. Cost per update: 32 × 15 = **480 candidate samples + 480 verifier
rollouts**, which is why the batch size is pinned at 32 — it is verifier-bound, not
memory-bound, so the training budget above costs nothing extra per step.

### Splits are pinned, not derived

`task.dataset.split_file` names a committed manifest holding the exact train/val/test
episode indices, plus a checksum of the zarr's `episode_ends`. The dataset, the env runner
and `eval_search_pusht.py` all read it, and the `n_*_episodes` config keys are *validated*
against it rather than generating anything — a disagreement raises.

This exists because the splits used to be derived independently in those three places from
`(seed, n_test_episodes, n_val_episodes, n_train_episodes, train_ratio)`, with nothing
recording which episodes a checkpoint had trained on. Raising `n_val_episodes` from 10 to 30
therefore shrank the pool that `train_ratio: 0.2` was a fraction *of*, silently cutting the
training budget from 29 episodes to 25.

```bash
python scripts/dump_pusht_splits.py              # regenerate (shows up as a diff)
python scripts/dump_pusht_splits.py --verify     # stored file still matches the derivation
python scripts/dump_pusht_splits.py --check-zarr # zarr's episodes still match the manifest
```

**See §2.1** for which manifest each generation of runs actually used, the exact command
that produced each one, and why the legacy 29-demo split is the one that cannot be
re-derived from the CLI.

Each training run also writes `<run_dir>/splits.json` recording the partition it resolved,
and refuses to resume if that partition has since changed.

**Online search (MLP)** — `train_online_search`, via
[`TrainOnlineSearchWorkspace`](diffusion_policy/workspace/train_online_search_workspace.py).
This one *does* have a genuine outer/inner loop: `num_outer_steps=100`, `outer_batch_size=128`
episodes rolled out per outer step, then `num_inner_steps=100` updates at `inner_batch_size=8`
that resample which of the `n_trajs=8` rollouts form the context.

**Neither is PPO-like.** The online-search loss is masked negative log-likelihood of the
*expert* action ([online_search_policy.py:346-350](diffusion_policy/policy/online_search_policy.py#L346-L350))
— no reward, advantage, ratio, clipping, or KL term anywhere, and no frozen old-policy
snapshot. The on-policy rollouts exist only to build the in-context conditioning. Note the
irony: that policy has a tractable Gaussian logprob and so *could* report a real
`KL(π_old‖π_θ)` cheaply, but does not.

~~Also note `training.use_ema` was **inert** in `TrainMLPImageWorkspace` — the workspace has
no EMA code — so no offline search run before section 6 has ever used EMA.~~

**Corrected (2026-08-10).** That was true when written and is not any more: the silent
no-op is `AUDIT.md` P1-5, fixed. `TrainMLPImageWorkspace` now honours `use_ema`, and every
current run trains with `ema_decay: 0.995`. The exception is the **six legacy-29 arms**,
which really did run with `use_ema: False` — their numbers come from the live weights while
every 100-demo and r8 number comes from the EMA average. That is one of the two defects the
`[no-EMA]` `[split-crop]` markers in `SUCCESS_RATES.md` §2 flag, and one of the three
changes the r8 re-runs isolate.

---

## 6. Outer/inner search trainer

`train_pusht_search_outer_inner`, via
[`TrainSearchOuterInnerWorkspace`](diffusion_policy/workspace/train_search_outer_inner_workspace.py).
Same policy, same data, same loss as the offline baseline — only the loop differs.

```bash
read A P < <(bash scripts/slurm/pick_gpu.sh)
CONFIG_NAME=train_pusht_search_outer_inner \
  sbatch --account=$A --partition=$P --export=ALL,CONFIG_NAME \
         scripts/slurm/train_pusht_search.sbatch
```

The search context is generated **once per outer step** for a pool of `outer_batch_size`
windows and reused for `inner_epochs` passes:

| | |
| --- | --- |
| `max_gradient_steps` | 100000 — **the only run-length knob** |
| `outer_batch_size` / `inner_batch_size` | 256 / 32 → 8 batches per pass |
| `inner_epochs` | 4 → **32 updates per outer step** |
| `drift_every` | 4 inner steps |
| `ema_decay` | 0.995, constant |

**The speedup is exactly `inner_epochs`, not the pool size.** 256 × 15 = 3,840 sims
amortized over 32 updates = 120 sims/update, against the baseline's 480 — 4×. Over 100k
steps that is 12.0M candidate sims instead of 48.0M. A larger pool does not change the
ratio; only `inner_epochs` does.

### Reading the drift metric

The price of amortizing is staleness: the buffered context comes from weights up to 32
updates old. So a frozen snapshot of the collector policy is kept for the whole inner loop
and compared against the live one at matched inputs.

A diffusion policy has no tractable `log π(a|s)` — sampling is an 8-step DDIM chain — so
PPO's stored-logprob/ratio machinery does not apply. But for a fixed
`(noisy_trajectory, timestep, obs, context)` the two snapshots' reverse transition kernels
are Gaussians sharing the scheduler's model-independent variance, differing only in a mean
that is affine in the predicted noise. Hence, exactly:

```
KL( p_old(a_{t-1}|·) ‖ p_θ(a_{t-1}|·) )  =  c(t) · ‖ε_θ − ε_old‖²
```

So **`train_drift_mse_eps` is a per-denoising-step KL**, at the cost of two extra forward
passes and no sampling. Log keys:

- `train_drift_mse_eps` — the primary signal. Plot it against `train_drift_inner_step`
  (position within the inner loop). Flat ⇒ the buffered context is still effectively
  on-policy and `inner_epochs` could go higher. Rising sharply by step 32 ⇒ the model is
  training on context from a policy it no longer resembles; lower `inner_epochs` or the lr.
- `train_drift_action_mse` — the same drift sampled in action space under a shared noise
  seed (PushT pixel units, interpretable). Logged only on the sampling cadence, since it
  costs two full sampling chains.

Both run the live policy in `eval()` — with `p_drop_attn: 0.2` active the difference would
otherwise be dominated by independent dropout draws rather than by real drift.

Because the regression target is always the GT expert action, there is no importance weight
to correct and no ratio to clip. "PPO-style" here means the monitoring and trust-region
half of PPO, not its surrogate objective. To turn the diagnostic into a soft trust region,
add `λ · train_drift_mse_eps` to the loss.

### EMA

EMA keeps a second copy of the weights updated after every optimizer step,

```
θ_ema ← d·θ_ema + (1−d)·θ
```

and **that copy is what gets evaluated and checkpointed** — rollouts, nRMSE, and every
`step_*.ckpt` the `--watch` eval scores. It never receives gradients. The effective
averaging window is `1/(1−d)`, so `ema_decay: 0.995` averages ~200 steps.

Why it matters here specifically: each training sample draws one random timestep out of 100
and one random noise vector, so consecutive gradients estimate different parts of the
denoising problem (t≈90 is coarse structure, t≈5 is fine detail) and the live iterate
rattles. That jitter is zero-mean and averaging cancels it. It also makes checkpoint
selection more honest — `eval_search_pusht --watch` picks a winner from per-checkpoint
scores, and noisy weights mean partly picking lucky noise.

**The decay here is constant, unlike the rest of the repo.** `EMAModel`'s default is a
warmup curve, `decay(step) = 1 − (1+step)^−0.75` capped at `max_value`
([get_decay](diffusion_policy/model/diffusion/ema_model.py#L43)) — 0.405 at step 1, 0.99 at
~460, 0.995 at ~1168, 0.999 at 10k; the upstream cap of 0.9999 needs 215k steps and is
never reached. Two reasons not to use it: the averaging window becomes a function of where
you are in the run, and the counter driving it **is not checkpointed**, so a resumed run
re-enters the curve at step 0. Setting `min_value == max_value` clamps the curve flat, which
is step-independent and identical across restarts. Override with
`training.ema_decay=0.999` for a 1000-step window.

### Changing the run length

`training.max_gradient_steps` is the only knob. The outer loop is
`while global_step < max_gradient_steps`; `num_inner = inner_epochs × ceil(outer_batch_size
/ inner_batch_size)` and there is no outer-step constant at all. Cadences
(`rollout_every_steps`, `val_every_steps`, `sample_every_steps`, `checkpoint_every`) are in
gradient steps, and both the LR and EMA schedules are defined independently of the total.

---

## 7. Run output, resume, and disk

**All run output goes to `/gscratch`, never home.** Home is a **10 GB hard quota** and one
100k-step run writes 50 `step_*.ckpt` plus `latest.ckpt`, each holding the policy, AdamW's
two moment buffers and the EMA copy — more than the entire quota on its own. The run
*directory* lives there, not just a symlinked `checkpoints/`, so wandb files,
`logs.json.txt` and the watcher's `bon_search/` all follow:

```yaml
output_root: ${oc.env:DP_OUTPUT_ROOT,/gscratch/robotics/harine/diffusion_policy_outputs}
```

Set `DP_OUTPUT_ROOT` to relocate. SLURM logs go to
`/gscratch/robotics/harine/slurm_logs`. [`scripts/slurm/monitor_pusht_search.sh`](scripts/slurm/monitor_pusht_search.sh)
warns if anything training-related starts accumulating in `data/outputs` on home.

**Resuming requires naming the old run directory.** `training.resume: True` looks for
`latest.ckpt` under the *current* Hydra output dir, and `hydra.run.dir` embeds
`${now:...}` — a new timestamped directory on every launch. A plain re-submit therefore
finds nothing and starts from scratch. The new workspace prints a loud warning when this
happens; to actually continue:

```bash
sbatch --account=$A --partition=$P scripts/slurm/train_pusht_search.sbatch \
  hydra.run.dir=$DP_OUTPUT_ROOT/<date>/<time>_<name>_<task>
```

What is and is not restored:

- **Restored:** model and EMA weights, optimizer state, `global_step`, `epoch`, and the
  `last_*_step` eval cadence counters. The normalizer is kept from the checkpoint rather
  than recomputed, so resumed weights keep the scaling they were trained under.
- **Rebuilt, but exactly:** the LR schedule. It is not checkpointed; it is reconstructed at
  `last_epoch=global_step-1`, which is exact because
  [`get_decay_then_constant_schedule`](diffusion_policy/model/common/lr_scheduler.py#L10-L31)
  is a `LambdaLR` whose lambda is a closed-form function of the step index alone. This holds
  provided the optimizer is restored *before* the scheduler is built (it supplies the
  `initial_lr` that `LambdaLR` requires at `last_epoch != -1`) and `lr`, `lr_warmup_steps`,
  `decay_steps` and `min_lr_ratio` are unchanged. `decay_then_constant` also ignores
  `num_training_steps`, so the curve is independent of `max_gradient_steps` — that stops
  being true if you switch to `cosine`/`linear`.
- **Not restored:** RNG state. Diffusion noise and timestep draws repeat their opening
  sequence, which is harmless (they are i.i.d. and carry no information). *Data selection*
  repeating would not be, so the outer/inner workspace derives its pool and shuffle RNG
  from **position** (`default_rng([seed, outer_idx])`) rather than from call count — a
  resumed run draws the pool that outer step would have drawn anyway, instead of replaying
  the pools of outer steps 0, 1, 2… `TrainOnlineSearchWorkspace` does *not* do this.
- **Not restored:** the context buffer. A resume begins a fresh outer step, discarding at
  most the tail of the interrupted inner loop.
- `use_ema` cannot be flipped on resume — `load_payload` assigns into `self.__dict__` for
  every saved `state_dict`, so a `use_ema: True` checkpoint needs a workspace that built an
  `ema_model`.

---

## 8. Evaluation

**Single rollout eval** (upstream script) — one rollout per held-out test reset:

```bash
python eval.py -c $DP_OUTPUT_ROOT/<run>/checkpoints/latest.ckpt \
  -o data/pusht_eval_output -d cuda:0
# -> eval_log.json (test/mean_score + rollout videos)
```

**Best-of-N** — roll each test reset out N times, report best-of-n and regret curves
([`eval_bon.py`](eval_bon.py)):

```bash
python eval_bon.py -c $DP_OUTPUT_ROOT/<run>/checkpoints/latest.ckpt \
  -o $DP_OUTPUT_ROOT/<run>/bon --n-samples 64 --n-envs 50
# -> bon_summary.json, bon_rewards.npz, bon_curves.png
```

Success = `max_reward >= 1.0` (coverage ≥ 95%). Options: `--n-resets` (default all
test resets), `--max-steps 300`, `--seed`.

**Best-of-N contact sheet** — one mp4, resets as rows × samples as columns, each cell
green-bordered on success ([`bon_video.py`](bon_video.py)):

```bash
python bon_video.py -c $DP_OUTPUT_ROOT/<run>/checkpoints/latest.ckpt \
  -o $DP_OUTPUT_ROOT/<run>/bon --n-resets 5 --n-samples 8
# -> bon_grid_5x8.mp4  (--reset-idxs 47,12,2,22,6 to pick specific resets)
```

### 8.1 Selection-criteria sweep (fixed n, six read-out rules)

The sweep above varies `n`. This one holds `n` FIXED at 16 and varies how the executed
candidate is picked out of the 16 that were generated and scored. All six are pure read-outs
on trained weights — generation is byte-identical across them — so this isolates selection
from everything else.

Candidate order carries meaning: candidate *k* is conditioned on candidates 0..*k*-1, so the
trailing candidates are the deeply-conditioned ones. That is what the six criteria separate:

| criterion | rule | pool |
|---|---|---|
| `cand-last` | fixed index -1 | — |
| `cand-8th-from-last` | fixed index -8 | — |
| `argmax-all` | argmax verifier value | all 16 |
| `argmax-last8` | argmax verifier value | last 8 |
| `softmax-all` | softmax(z/T) | all 16 |
| `softmax-last8` | softmax(z/T) | last 8 |

The `cand-*` pair ignores the verifier entirely, so `(argmax-all − cand-last)` is what the
ranking contributes on top of conditioning depth, and `(argmax-last8 − argmax-all)` says
whether restricting the oracle to well-conditioned candidates helps or merely removes
options.

```bash
python eval_search_pusht.py -c <run>/checkpoints/step_0020000.ckpt \
  --criteria-sweep --criteria-n 16 --n-envs 50 --skip-val
# -> <run>/criteria_search/step_0020000/<criterion>.json
#    <run>/criteria_search/step_0020000/traces/<criterion>.npz
#    <run>/criteria_search/criteria_curves.jsonl   (run-level index, one row per step+criterion)
```

Results go to `criteria_search/`, **not** `bon_search/`: these rows are keyed by criterion at
fixed n rather than by n, so merging them would put two different experiments in one file.

**Traces.** Each `.npz` holds the per-control-step search state — `scores` (n_episodes, T,
16) for *every* candidate, `chosen_idx` (n_episodes, T) for the one executed, `step_reward`,
and `valid_len` (episodes are padded to the longest in their chunk, with NaN, since every env
keeps being stepped until all are done). `chosen_idx` is `-1` under `final_pass` (the executed
action is not a candidate) and `-2` where the episode had already ended.

The verifier value is **negative mean per-keypoint distance to the goal**, so it is ≤ 0 and
*higher is better*. It is stored exactly as produced; label plot axes accordingly rather than
flipping the sign, so the stored array keeps agreeing with the `action_value*` series in the
training logs.

Validate a trace with `python scripts/check_criteria_traces.py <.../step_XXXXXXX>` — the
load-bearing assertion is that `argmax-all`'s recorded `chosen_idx` equals
`scores.argmax(-1)` at every step, which is the cheapest proof that the trace and the action
actually executed correspond.

On SLURM, one job per (run, checkpoint) runs all six criteria — they share a policy load and
an env pool:

```bash
bash scripts/slurm/submit_criteria_sweep.sh          # safe to re-run as checkpoints appear
DRY=1 bash scripts/slurm/submit_criteria_sweep.sh    # list what would be submitted
```

---

## 9. SLURM

[`job.sh`](job.sh) has a usable preamble (`module load conda; conda activate robodiff2`)
but its body targets an unrelated `peg_insertion` experiment on another account/path —
swap in one of the `python train.py ...` commands above and set your own
`--account`/paths before submitting.
```
