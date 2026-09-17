# PushT policy inventory and eval coverage — 2026-08-10

What every trained policy actually is, on every axis that varies, and whether each of its
checkpoints has a 50-episode best-of-N result.

`SUCCESS_RATES.md` reports the numbers; this says what produced them. Where the two
overlap, `SUCCESS_RATES.md` is authoritative for *values* and this doc for *provenance*.

**Snapshot.** All six `r8` trainers are running, so their rows keep growing. Every count
below is re-derivable: run `bash scripts/slurm/fill_eval_gaps.sh` for the coverage half, and
read each run's own `.hydra/config.yaml` for the inventory half. Every config fact in this
doc was read from the run's saved config, **not** from the current `config/` tree — those
have drifted apart, and the drift is itself a finding (§1e).

**Scope.** 29 policy directories: 26 under
`…/pusht_search/pusht_image_search/offline/` and 3 outer/inner runs under `…/runs/`.
The 12 `ctx-*` entries in `offline/` are back-symlinks from the 2026-08-05 rename
(`AUDIT.md` §9.9), not runs — §1f.

---

## 1. Inventory

### 1a. The design matrix, and its holes

`arm × obs × demo budget`. ✓ = a run exists.

| arm | clean 100 | corrupt 100 | clean 29 legacy | corrupt 29 legacy | clean 29 r8 | corrupt 29 r8 | 25 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| none (BC) | ✓ | · | ✓¹ | · | · | · | ✓ |
| value | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| subgoal-chosen4value | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| subgoal-value | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | · |
| subgoal-only (final_pass) | ✓ | ✓ | · | · | · | · | · |
| subgoal-only k4/k8/k16 cd0.9 | ✓ | · | · | · | · | · | · |

¹ `bc_demos-29` uses the legacy 29-episode *split* but the *current* config (EMA on,
shared crop), which is why `AUDIT.md` says BC@29 vs BC@100 isolates demo count cleanly.

The empty cells are the point:

- **There is no search arm at 25 demos.** The budget grid is BC{25, 100} × search{100}, so
  the only place demo count is varied is the baseline. Adding search@25 would complete a
  budget × search 2×2 and is the only way to answer *does search substitute for data* —
  currently the data-budget axis and the search axis are never crossed.
- **`k4 / k8 / k16_cd0.9` are clean-only.** The context-decay result rests on three runs
  with no corrupt counterpart, so it cannot be separated from the clean-obs regime — and
  clean obs is exactly the regime where `AUDIT.md` §9.1 proves the context is redundant.
- **`subgoal-only` exists only at 100 demos.** No `final_pass` run at any smaller budget.
- **`r8` covers only the three argmax arms** — no `subgoal-only`, no BC. So the crop/EMA
  fix has never been applied to the `final_pass` selection rule.
- **BC has no corrupt variant at any budget.** Corruption is a policy-side flag that would
  apply perfectly well to a context-free policy, so there is no clean-vs-corrupt baseline
  to read the search arms' corrupt numbers against.
- **25 demos is BC only.**

### 1b. Every run

`trainer` = which workspace trained it. **26 of 26 offline runs used `offline`
(`TrainMLPImageWorkspace`)** — every legacy-29 arm, every 100-demo arm, all six r8 and all
three BC. Only the three directories literally named `train_pusht_search_outer_inner*` used
`outer_inner` (`TrainSearchOuterInnerWorkspace`). The two differ in *when* the search
context is generated: `offline` regenerates it from the current weights on every gradient
step (on-policy, and the dominant cost — 32 × 15 = 480 candidate samples plus 480 verifier
rollouts per update), while `outer_inner` generates it once per pool of 256 windows and
reuses it across 4 inner epochs (far cheaper, but the context comes from stale weights).

`K` = `policy.max_actions` (search width; context is K−1 tokens). `ctx` = search context
and its per-candidate dim. `cd` / `swd` = `context_decay` / `slot_weight_decay`
(`—` = key absent from that run's config, see §1e). `ck/ev` = checkpoints on disk /
checkpoints with an eval row. `max n` = largest n evaluated anywhere in the run.

| run | gen | trainer | arm | obs | dem | val | ctx | K | selection | cd | swd | crop | EMA | steps/ckpt | ck/ev | max n | sel probe |
|---|---|---|---|---|--:|--:|---|--:|---|--:|--:|---|---|---|---|--:|---|
| `bc_demos-100_seed-42` | 100-demo | offline | none (BC) | clean | 100 | 30 | — | 1 | — | — | — | shared | 0.995 | 300k/10k | 30/30 | 64 | argmax+softmax |
| `subgoal-chosen4value_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-chosen4value | clean | 100 | 30 | subgoal 530 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `subgoal-chosen4value_corrupt-True_demos-100_seed-42` | 100-demo | offline | subgoal-chosen4value | corrupt | 100 | 30 | subgoal 530 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `subgoal-only_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-only | clean | 100 | 30 | subgoal 530 | 16 | final_pass | — | 0.9 | shared | 0.995 | 100k/2k | 29/29 | 64 | argmax+softmax |
| `subgoal-only_corrupt-True_demos-100_seed-42` | 100-demo | offline | subgoal-only | corrupt | 100 | 30 | subgoal 530 | 16 | final_pass | — | 0.9 | shared | 0.995 | 100k/2k | 30/30 | 64 | argmax+softmax |
| `subgoal-only_k16_cd0.9_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-only | clean | 100 | 30 | subgoal 530 | 16 | final_pass | 0.9 | 1.0 | shared | 0.995 | 100k/2k | 38/36 | 64 | argmax+softmax |
| `subgoal-only_k4_cd0.9_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-only | clean | 100 | 30 | subgoal 530 | 4 | final_pass | 0.9 | 1.0 | shared | 0.995 | 100k/2k | 50/50 | 64 | argmax+softmax |
| `subgoal-only_k8_cd0.9_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-only | clean | 100 | 30 | subgoal 530 | 8 | final_pass | 0.9 | 1.0 | shared | 0.995 | 100k/2k | 50/50 | 64 | argmax+softmax |
| `subgoal-value_corrupt-False_demos-100_seed-42` | 100-demo | offline | subgoal-value | clean | 100 | 30 | subgoal_value 531 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `subgoal-value_corrupt-True_demos-100_seed-42` | 100-demo | offline | subgoal-value | corrupt | 100 | 30 | subgoal_value 531 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `value_corrupt-False_demos-100_seed-42` | 100-demo | offline | value | clean | 100 | 30 | value 1 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `value_corrupt-True_demos-100_seed-42` | 100-demo | offline | value | corrupt | 100 | 30 | value 1 | 16 | argmax | — | — | shared | 0.995 | 20k/1k | 20/20 | 64 | argmax+softmax |
| `bc_demos-29_seed-42` | bc-29 | offline | none (BC) | clean | 29 | 10 | — | 1 | — | — | — | shared | 0.995 | 100k/2k | 50/50 | 512 | — |
| `subgoal-chosen4value_corrupt-False_demos-29_seed-42` | legacy-29 | offline | subgoal-chosen4value | clean | 29 | 10 | subgoal 530 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 512 | argmax+softmax |
| `subgoal-chosen4value_corrupt-True_demos-29_seed-42` | legacy-29 | offline | subgoal-chosen4value | corrupt | 29 | 10 | subgoal 530 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 512 | argmax+softmax |
| `subgoal-value_corrupt-False_demos-29_seed-42` | legacy-29 | offline | subgoal-value | clean | 29 | 10 | subgoal_value 531 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 512 | argmax+softmax |
| `subgoal-value_corrupt-True_demos-29_seed-42` | legacy-29 | offline | subgoal-value | corrupt | 29 | 10 | subgoal_value 531 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 1024 | — |
| `value_corrupt-False_demos-29_seed-42` | legacy-29 | offline | value | clean | 29 | 10 | value 1 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 1024 | argmax+softmax |
| `value_corrupt-True_demos-29_seed-42` | legacy-29 | offline | value | corrupt | 29 | 10 | value 1 | 16 | argmax | — | — | **split-crop** | **off** | 20k/1k | 20/20 | 1024 | argmax+softmax |
| `subgoal-chosen4value_corrupt-False_demos-29-r8_seed-42` | r8-29 | offline | subgoal-chosen4value | clean | 29 | 30 | subgoal 530 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 15/15 | 64 | argmax+softmax |
| `subgoal-chosen4value_corrupt-True_demos-29-r8_seed-42` | r8-29 | offline | subgoal-chosen4value | corrupt | 29 | 30 | subgoal 530 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 14/14 | 64 | — |
| `subgoal-value_corrupt-False_demos-29-r8_seed-42` | r8-29 | offline | subgoal-value | clean | 29 | 30 | subgoal_value 531 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 26/26 | 64 | — |
| `subgoal-value_corrupt-True_demos-29-r8_seed-42` | r8-29 | offline | subgoal-value | corrupt | 29 | 30 | subgoal_value 531 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 24/24 | 64 | — |
| `value_corrupt-False_demos-29-r8_seed-42` | r8-29 | offline | value | clean | 29 | 30 | value 1 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 27/27 | 64 | argmax+softmax |
| `value_corrupt-True_demos-29-r8_seed-42` | r8-29 | offline | value | corrupt | 29 | 30 | value 1 | 16 | argmax | 1.0 | 1.0 | shared | 0.995 | 100k/2k | 29/29 | 64 | argmax+softmax |
| `bc_demos-25_seed-42` | archive-25 | offline | none (BC) | clean | 25 | 30 | — | 1 | — | — | — | shared | 0.995 | 300k/10k | 9/9 | 64 | — |
| `runs/…_outer_inner` | outer/inner | **outer_inner** | value | clean | 29 | 10 | value 1 | 16 | argmax | — | — | none | 0.995 | 100k/2k | 50/10 | 64 | — |
| `runs/…_outer_inner_subgoal` | outer/inner | **outer_inner** | subgoal-chosen4value | clean | 29 | 10 | subgoal 530 | 16 | argmax | — | — | none | 0.995 | 100k/2k | 50/10 | 64 | — |
| `runs/…_outer_inner_subgoal_verifier` | outer/inner | **outer_inner** | subgoal-value | clean | 29 | 10 | subgoal_value 531 | 16 | argmax | — | — | none | 0.995 | 100k/2k | 50/10 | 64 | — |

Totals: **661 checkpoints / 659 eval rows** in `offline/`; **150 / 30** in `runs/`. These grow
while the r8 trainers and the selection sweep run — regenerate rather than quote.

Two rows in the r8 block deserve a note: all six r8 runs report `val: 30` because that is
the split their *training* used, but **none of them has a val eval curve** — the watchers
ran `--skip-val`. See §2.

### 1c. What the varying axes mean

**Demo budget and split.** Five committed manifests in
[diffusion_policy/config/splits/](diffusion_policy/config/splits/), all over the same
206-episode zarr (`episode_ends_checksum c88b56c9…`), and **all five carry the identical
50 test episodes**:

| manifest | train | val | test |
|---|--:|--:|--:|
| `pusht_seed42_train100.json` | 100 (12,899 frames) | 30 | 50 |
| `pusht_seed42_train25.json` | 25 (3,325) | 30 | 50 |
| `pusht_seed42_legacy_val10_train29.json` | 29 (3,707) | **10** | 50 |
| `pusht_seed42_train29_val30.json` | the same 29 (3,707) | 30 | 50 |
| `pusht_seed42.json` | 100 | 30 | 50 |

206 = 50 test + 30 val + 100 train + 26 unused. The legacy 29 and the r8 29 are the same
29 training episodes; only the val split differs (10 → 30). That is why the r8 generation
is a clean re-run and not a new data condition.

**Obs corruption — and its magnitude.** There is no corruption in the dataset at all
(`grep corrupt diffusion_policy/dataset/pusht_image_dataset.py` returns nothing). It is
forward-diffusion noise applied to the **encoded observation feature vector** — never the
pixels, never the actions, never the context or the subgoal embedding
([diffusion_transformer_search_policy.py:798-814](diffusion_policy/policy/diffusion_transformer_search_policy.py#L798-L814)):

```python
obs_noise = torch.randn_like(obs_features)
timesteps = torch.randint(0, 100, (bsz,), ...)          # uniform, per batch row, per call
return self.obs_noise_scheduler.add_noise(obs_features, obs_noise, timesteps)
```

DDPM, **linear β 0.001 → 0.02 over 100 steps**, so `x ← √ᾱ_t·x + √(1−ᾱ_t)·ε`:

| t | 0 | 25 | 50 | 75 | 99 |
|---|--:|--:|--:|--:|--:|
| √ᾱ (signal) | 0.9995 | 0.957 | 0.862 | 0.731 | **0.590** |
| √(1−ᾱ) (noise) | 0.032 | 0.291 | 0.507 | 0.682 | **0.808** |

`t` is redrawn uniformly on every call, so **there is no single corruption level to
quote** — the expected noise coefficient is **0.476**, and any given forward pass is
somewhere between nearly clean and `0.59·signal + 0.81·ε`. Three consequences, all from
`AUDIT.md`:

- **P0-1** — the flag has no `self.training` gate, so corruption is **active at eval and
  rollout too**. Corrupt arms are solving a strictly harder task than clean arms.
- **P0-2** — the noise is absolute unit variance against a feature vector that concatenates
  unnormalized ResNet18 activations with two low-dim keys normalized to ~[−1,1], so the SNR
  differs across the vector by an unknown factor.
- **P0-3** — each candidate in one search draws its own corruption, so within a single
  decision candidate 0 may see a near-clean observation and candidate 5 a heavily corrupted
  one, yet their scores are concatenated into one context as if comparable.

Why corruption exists at all (`AUDIT.md` §9.1): with clean obs on a fully observed task,
`p(a* | obs, context) = p(a* | obs)` exactly — the Bayes-optimal model ignores the context.
Corruption is what is supposed to break that. P0-1/P0-2 mean it does not do so cleanly.

**`legacy` = split-crop.** In the legacy-29 runs `policy.crop_shape` is unset, so cropping
fell through to the encoder's own `CropRandomizer`, which samples **one offset per image**
([crop_randomizer.py:86-118](diffusion_policy/model/vision/crop_randomizer.py#L86-L118)).
Both observation frames and every candidate's subgoal frame therefore got *different*
76×76 windows out of 96×96 — translated relative to each other at train time, while eval
center-crops all of them. It lands hardest on the subgoal arms, whose entire context is a
subgoal image that no longer registers with the observation it came from. Everything
newer has the **policy** draw one offset per sample from `(training.seed, global_step)`
and share it across the observation and all K−1 subgoal encodes
([diffusion_transformer_search_policy.py:632-685](diffusion_policy/policy/diffusion_transformer_search_policy.py#L632-L685),
[multi_image_obs_encoder.py:207-232](diffusion_policy/model/vision/multi_image_obs_encoder.py#L207-L232)).
Eval draws nothing and center-crops to offset `(10,10)`. The three outer/inner runs have
**no crop at all** (`random_crop: False`, `crop_shape: null`) — a third regime.

**Context.** K−1 tokens in generation order, oldest first. Each candidate's action chunk
is simulated in a real PushT sim from the current state, and what the context records is:

| `search_context` | dim | content |
|---|--:|---|
| — (BC) | — | nothing; `max_actions: 1`, context always empty |
| `value` | 1 | the verifier scalar |
| `subgoal` | 530 | the reached frame, through the policy's own obs encoder |
| `subgoal_value` | 531 | both |

`state` (18) and `state_value` (19) are implemented and reachable but **no run uses them**.
Note the arms are **not equal capacity** — `SearchTransformerForDiffusion` sizes
`action_value_emb` from `context_dim`, so "subgoal beats value" is confounded with "more
parameters".

**Selection — final candidate vs distance-chosen.** Three rules
([diffusion_transformer_search_policy.py:1013-1094](diffusion_policy/policy/diffusion_transformer_search_policy.py#L1013-L1094)):

- **`argmax`** — the executed candidate is the one maximizing the verifier scalar, and that
  scalar *is* a distance: `value = −mean keypoint distance to the goal T`, measured after
  simulating the candidate. This is the "distance chosen" rule. Every arm but `subgoal-only`.
- **`final_pass`** — after the n candidates are scored, **one more sample is drawn
  conditioned on all of them and executed unsimulated**. The verifier scalar takes no part
  in selection. This is the "final candidate" rule; the deployed slot is
  `min(n, max_actions-1)`, and the arm costs `n+1` samples to argmax's `n`.
- **`softmax`** — a post-hoc *readout only*, never a trained condition: resample the
  executed candidate from `softmax(z/T)`, `T = 1`, `z` standardized across the n candidates.
  `T→0` reproduces argmax candidate-for-candidate; **n=1 is identical to argmax by
  construction**. Lives in `bon_search_sel-softmax/`, never merged into the native curve.

**There is no distance-to-subgoal selector anywhere in the repo.** Grepping for
`nearest|distance_to_subgoal|closest` across `policy/` and `eval_search_pusht.py` returns
nothing. The subgoal arms feed the rendered reached frame into the *model*; it never enters
ranking. Worth stating because the arm names suggest otherwise.

Related caveat (`AUDIT.md` §9.4): the verifier optimizes mean keypoint distance while
success is thresholded on max *coverage*, and the two are not monotone — so
argmax-verifier ≠ argmax-success even with a perfect simulator.

**`cd0.9` vs `swd0.9` — two different 0.9s.**

- **`context_decay`** is an *attention* bias: for a candidate seeing `m` context entries,
  entry `j` gets post-softmax weight `0.9^(m-1-j)`. It depends only on distance-from-latest,
  never on absolute index, K or n.
- **`slot_weight_decay`** is a *loss* weight over candidate slots, `w_k ∝ 0.9^(K-1-k)`,
  keyed to the **absolute** slot index. Since `predict_action_best` deploys slot
  `min(n, max_actions-1)`, the deployed conditional's training weight depends on eval n —
  it is down-weighted by 55% at n=1. `context_decay` replaced it for exactly this reason;
  the `cd` arms set `slot_weight_decay: 1.0` (off), and `subgoal-only` base still has 0.9.

**`k4/k8/k16`** = `n_candidates`. There is no `_cd_k16.yaml` — k16 is the base `_cd.yaml`,
the control isolating the weighting change at fixed width. Cost scales `(K-1)/15`:
K=4 → 20%, K=8 → 47%.

**Three different search widths, one never measured** (`AUDIT.md` P2-7): training
conditions on **K−1** (15, 7 or 3); the in-training rollout uses **`n_search_actions: 8`**
(1 for BC); the eval sweep runs **1…64**. The width that shapes what the model learns is
the one never evaluated.

### 1d. Collapsed axes — identical on every run

These do not vary, and in one case that is the most important fact in the doc.

- **`training.seed: 42` on all 29 runs. There are no seed replicates anywhere**, so no arm
  has an across-seed variance estimate. Every difference in `SUCCESS_RATES.md` is a
  single-draw difference.
- horizon 16, `n_obs_steps` 2, `n_action_steps` 8; DDIM `num_inference_steps` 8 over 100
  training timesteps.
- AdamW lr 1e-4, wd 1e-6, `decay_then_constant`, warmup 500; batch 32 — pinned there
  because each update costs 32 × 15 = 480 candidate samples and 480 verifier rollouts.
- ResNet18 + GroupNorm encoder, 4-layer transformer, `n_emb` 256; crop 76×76 from 96×96.
- `verifier_n_envs: 32`, `verifier_legacy: False`, env `max_steps: 300`, `legacy: False`.
  `render_size` is **not plumbed into the verifier** and silently defaults to 96
  (`AUDIT.md` P2-2).
- **No online-search runs exist in this tree.** `README_pusht.md` §4 documents the online
  policy — which corrupts the *context* rather than the observation, the same method name
  with an inverted target (`AUDIT.md` §9.2) — but nothing was ever trained with it.

### 1e. The config schema has four generations, so no single field identifies an arm

Reading each run's own `.hydra/config.yaml` shows the knobs were introduced in layers.
This is why the inventory above cannot be built by reading one key:

| keys present | runs |
|---|---|
| no `arm`, `selection`, `n_candidates`, `context_decay`, `slot_weight_decay`; no `policy.crop_shape`; `train_ratio: 0.2`; no `split_file` | the 6 legacy-29 |
| still no `arm`/`selection`/`n_candidates`/`cd`/`swd`, but `policy.crop_shape: [76,76]`, EMA on, `split_file` set | the 6 argmax 100-demo runs, `bc_demos-{100,25,29}` |
| `arm`, `selection`, `slot_weight_decay` — but no `n_candidates` or `context_decay` | `subgoal-only_corrupt-{False,True}` |
| full key set | the 3 `cd0.9` runs and all 6 r8 runs |
| `search_context` nested under `policy`, no top-level axes at all | the 3 outer/inner runs |

For runs in the first two rows the arm must be inferred from `search_context` plus the
directory name — which is exactly the fragility the `ctx-*` → arm-label rename fixed
(`AUDIT.md` §9.9), and it only fixed it going forward.

### 1f. The 12 `ctx-*` back-symlinks are not runs

Left in place by the 2026-08-05 rename so absolute paths baked into already-queued jobs
still resolve. `build_success_rates_doc.py` and `fill_eval_gaps.sh` both skip
`d.is_symlink()`; anything that globs `offline/*` without that guard counts every renamed
run twice.

| symlink | → |
|---|---|
| `ctx-value_corrupt-{False,True}_demos-100_seed-42` | `value_corrupt-{False,True}_demos-100_seed-42` |
| `ctx-value_corrupt-{False,True}_seed-42` | `value_corrupt-{False,True}_demos-29_seed-42` |
| `ctx-subgoal_corrupt-{False,True}_demos-100_seed-42` | `subgoal-chosen4value_corrupt-{False,True}_demos-100_seed-42` |
| `ctx-subgoal_corrupt-{False,True}_seed-42` | `subgoal-chosen4value_corrupt-{False,True}_demos-29_seed-42` |
| `ctx-subgoal_value_corrupt-{False,True}_demos-100_seed-42` | `subgoal-value_corrupt-{False,True}_demos-100_seed-42` |
| `ctx-subgoal_value_corrupt-{False,True}_seed-42` | `subgoal-value_corrupt-{False,True}_demos-29_seed-42` |

One live trap: the `checkpoint` path recorded inside `bon_search/success_curves.jsonl` and
`step_*/success_curve.json` was written **through the alias** for runs evaluated before the
rename. Anything that string-matches a run directory against the `checkpoint` field must
resolve symlinks first.

---

## 2. Eval coverage — are the 50-episode results there?

**Yes, wherever an eval exists in `offline/`.** All **659 rows across all 26 offline runs
have `n_episodes == 50`** — there is not one row scored on a different-sized test set. And
because all five split manifests carry the identical 50 test episodes, those numbers are
comparable across every generation even though the training sets and selectors are not.

What is missing is *which checkpoints* and *which n*. Regenerate this table with
`bash scripts/slurm/fill_eval_gaps.sh`.

| # | gap | detail | fixable by |
|---|---|---|---|
| **G1** | checkpoint with no eval row | `subgoal-only_k16_cd0.9` steps **74000, 76000** | its watcher, already running |
| **G2** | row missing `n=64` | `k16@72000`, `k8@100000`, `subgoal-value_…-r8_False@46000` | watchers, already running |
| **G3** | BC n-sweep missing | **`bc_demos-25` @ 90000** — 0 of its 9 rows has any n>1 | 6 jobs, nothing running |
| **G4** | row missing `mean_reward` | `subgoal-only_corrupt-False` steps 26000, 28000; `k4_cd0.9` steps 12000, 14000 | `scripts/backfill_mean_reward.py` |
| **G5** | no `mean_reward_final` / `_discounted` | 1 of 20 rows on each of the six 100-demo argmax arms and each legacy-29 arm; **0 of 50** on `bc_demos-29` | re-eval only — per-step rewards were discarded before 2026-08-05, so there is nothing to backfill from |
| **G6** | outer/inner runs unverifiable | all 3 have **`n_episodes: null`** — the field postdates them, so **the 50-episode test set cannot be confirmed**. Also **10 of 50 checkpoints evaluated**, and no `run.json`, `splits.json` or `RUN_CAVEATS.json` | full re-eval + split reconstruction |
| **G7** | no large-n tail | `n>64` exists on **7 runs only, all legacy-29 or `bc_demos-29`, at one checkpoint each**. **No 100-demo run has any n>64** | `scripts/slurm/submit_large_n_evals.sh` |

G1–G3 total 11 jobs, of which 5 are suppressed because their watcher is live. G3 is the
only gap with nothing already working on it.

**The BC rule.** BC is swept at every n only at its **best-val-n=1** checkpoint — a
best-of-N curve for BC costs the same as one for a search arm, so the whole trajectory is
not worth it. That rule is satisfied for `bc_demos-100` (step 290000, unique val argmax
0.567) and `bc_demos-29` (step 96000, unique val argmax 0.400), and violated only by
`bc_demos-25` (target step 90000, never swept). The target is a **moving one** — a later
checkpoint can overtake — so `fill_eval_gaps.sh` re-derives it from the val curve on every
run rather than hardcoding a step.

### Not gaps — deliberate, but load-bearing

- **The six r8 runs have no val curve at all** (`val_n_episodes == 0` on every row). The
  watchers were launched `--skip-val`
  ([launch_round8_29demo.sh:70-77](scripts/slurm/launch_round8_29demo.sh#L70-L77)) so the budget
  went entirely to the 50 test episodes. Consequence: **any step picked from an r8 curve is
  picked on test**, so a test number read at that step is not a held-out estimate. (Under the
  val-success rule that `best.json` used to apply, all six landed on step 2000, the earliest
  checkpoint — consistent with the known decay of the search gain with training, but
  selected-on-test cannot be evidence for it.) `fill_eval_gaps.sh` preserves `--skip-val` per
  run so a filled gap cannot leave one curve half-scored on a split the rest of it never saw.
- **`bon_search_sel-{argmax,softmax}/` is a sparse probe**, 3–6 steps on 14 runs, test-only.
  It is kept in separate directories so a selection-override curve can never merge into the
  native one.

### In flight at snapshot

6 `tr_*` (all six r8 trainers, target 100k steps), 11 `ev_*` running plus 1 pending on a
dependency, 24 `sel_*` held (`JobHeldUser`). Two r8 `n=64` gaps closed while this document
was being written, which is the clearest argument for regenerating §2 from the script
rather than reading it here.

---

## 3. What this inventory licenses

- **Clean vs corrupt is not a controlled comparison.** Corruption runs at eval too (P0-1)
  and at an uncalibrated, per-call-random magnitude (P0-2), so a corrupt arm is a different
  task, not the same task with a noisier input.
- **`value` vs `subgoal` is confounded with model capacity** — 1-dim vs 530-dim context
  embedding input.
- **29 vs 100 demos is confounded** with EMA and with the crop fix, *except* for
  `bc_demos-29` vs `bc_demos-100`, which differ only in data.
- **legacy vs r8 changes three things at once** (EMA off→0.995, split-crop→shared offset,
  20k→100k steps), so it measures the bundle, not any one of them.
- **One seed everywhere.** No arm has a variance estimate. At 50 test episodes SE is ~7 pp
  near 50%, so gaps under ~14 pp are not resolvable even before seed variance.
- **`subgoal-only` changes what n means** — no candidate is selected by score, so its n is
  not the other arms' n.
- **The `cd0.9` result is clean-obs-only** and rests on three runs with no corrupt
  counterpart.
- **The r8 selection is on test.** Any r8 number quoted at its starred step is not held out.

---

## 4. Where the raw data is

```
<run>/checkpoints/step_0005000.ckpt        the policies (~275 MB each)
<run>/bon_search/success_curves.jsonl      one merged row per checkpoint — the source for every table
<run>/bon_search/step_XXXXXXX/success_curve.json   full curve + per_n_rewards + episode_idxs
<run>/bon_search_sel-{argmax,softmax}/     the selection probe, kept separate on purpose
<run>/splits.json                          the exact episodes this run trained/validated/tested on
<run>/run.json                             git sha, SLURM job id, host, per launch
<run>/RUN_CAVEATS.json                     legacy-29 only — the [no-EMA] / [split-crop] record
<run>/.hydra/config.yaml                   what the run ACTUALLY trained with
```

Run root: `/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline/`
· outer/inner: `/gscratch/robotics/harine/diffusion_policy_outputs/runs/`
· SLURM logs: `/gscratch/robotics/harine/slurm_logs/`

Regenerate: `python scripts/build_success_rates_doc.py` (the tables),
`bash scripts/slurm/fill_eval_gaps.sh` (section 2).
