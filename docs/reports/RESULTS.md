# Do these methods work on PushT?

One page, every arm the repo supports, measured. Written 2026-09-16.

**Short answer.** The offline arms work and are ordered
ST k=16 > ST k=1 > BC-UNet > BC-LSTM. The online RL arms are **being retrained** and have no
current number: an earlier generation of 18 arms reached 10M steps without one ever recording a
non-zero evaluation success rate, and was retired on 2026-09-16. SAC has **never been run**, and
is in any case built as a best-of-N *verifier* rather than a policy.

---

## Before the table: what is resolvable

`success` thresholds episode max coverage at 0.95; `mean reward` is
`clip(coverage/0.95, 0, 1)` averaged over episodes. The held-out test split is **50 episodes**,
val is 30. At n=50 a 95% Wilson interval is roughly **±13 points** at p=0.5, so
**two arms differing by less than ~15 points are not distinguishable here**, and a single seed
cannot settle a close call. `docs/reports/archive/LATEST_SUCCESS_RATES.md` measured the
run-to-run noise floor directly at **~0.07 mean reward / ~0.06 success**.

Nothing in this repo nominates a best checkpoint. Where a single number is quoted below it was
**selected on val and reported on test**, and the selection step is named.

## The arms

| arm | entry point | obs | encoder | split | eval protocol | result |
|---|---|---|---|---|---|---|
| **ST k=16** | `train.py --config-name=train_pusht_diffusion_search` | image | shared | `pusht_seed42_train30` | `eval_search_pusht`, in-the-loop search | test mean reward **0.72–0.80** over 20k–100k steps |
| **ST k=1** | `..._search_single n_candidates=1` | image | shared | same | same | **0.50–0.64** |
| **BC-UNet** | `train.py --config-name=train_pusht_unet_bc` | image | shared | same | val-selected, test reported | val 0.588 @ 40k → **test 0.555** |
| **BC-LSTM (image)** | `train.py --config-name=train_pusht_lstm_bc` | image | shared | same | val-selected, test reported | val 0.404 @ **step 4k** → **test 0.250** |
| **BC-LSTM (keypoint)** | `..._lstm_bc_keypoint` | keypoint 20-d | none (flatten) | `pusht_keypoint_manifest` | val-selected, test reported | val 0.246 @ 28k → **test 0.262** |
| **recurrent PPO** | `python -m recurrent_ppo.train` | keypoint / image | shared (image) | manifest episodes | — | **retraining** |
| **plain PPO** (`--n-stack 1`) | `python -m recurrent_ppo.ppo.train` | keypoint / image | shared (image) | manifest episodes | — | **retraining** |
| **SAC** | `python sac/runner.py` | kp / image | shared (image) | procedural (policy) / manifest (verifier) | — | keypoint **10M done**, image running; per-split arms (blq137, brd100) training |

"shared" = ResNet18 / IMAGENET1K_V1 / GroupNorm / 76px crop / 96px input / 512-d per frame,
image-only. Verified identical across every image arm; see *The encoder* below.

### Notes that change how the rows compare

**The ST rows are a different protocol from the BC rows.** ST's 0.72–0.80 is a range of
per-step test means, not a val-selected point, and it is `eval_search_pusht`'s deployable
in-the-loop search. `eval_bon`'s best-of-n (~0.85 at n=64) is a **post-hoc oracle max over
independent rollouts** and is not comparable to either — it answers "is a good action in the
candidate set", not "does the policy pick it".

**BC-LSTM overfits almost immediately on images**: val peaks at step 4,000, 4% of the way
through training, and declines for the remaining 96%. Its best *test* number (0.406) occurs at
step 38k where val has already collapsed, which is why the val-selected 0.250 is the honest
figure. The keypoint arm peaks later (56%) and lower.

**The PPO arms are mid-retrain and deliberately have no row.** The previous generation — 18
runs across reward shapes (dense, shaped, delta), curricula and both architectures — recorded
zero evaluation success in every one, and was deleted on 2026-09-16 rather than carried
forward. Its record is in git history. The current generation trains on `delta` only, evaluates
on the split manifests rather than procedural resets, and exists to supply **V** for the
verifier study, not to claim a success rate.

**SAC is a verifier.** It learns `Q(s, chunk)` to rank candidates a diffusion policy proposes,
as a drop-in for the hand-written distance heuristic. "Does SAC solve PushT" is the wrong
question for this repo; the right one — does the learned Q outrank `t_goal` — is also
unmeasured.

## Comparability caveats

These are the reasons two numbers in the table above may not be measured on the same thing.

1. **Different episode sets.** Every offline arm resets to the recorded initial states of the
   manifest's 50 held-out demo episodes. PPO and SAC *policy* evals sample fresh starts from
   `agent_start_range` / `block_start_range`. SAC's *verifier* evals do use the manifest, so
   SAC's two kinds of evaluation disagree with each other.
2. **Different eval crop, historically.** Until 2026-09-16 the PPO arms random-cropped at eval
   while the offline arms centre-cropped. PPO now centre-crops in both phases, so this is
   closed going forward — but it also means PPO image numbers recorded before that date were
   measured under a different transform, and the arms lost random-crop augmentation.
3. **Different episode length.** SAC runs 304 steps (the nearest multiple of its 8-step chunk);
   everything else runs 300.
4. **Different keypoint width.** PPO keypoint is 40-d and carries a visibility mask because
   that arm studies occlusion; BC-LSTM keypoint is 20-d without one, because the demonstrations
   are fully visible and the mask would be a constant.
5. **`goal_mask_noise` is train-split only** by design, so that arm is deliberately evaluated
   off its training distribution.
5b. **The online-search arm does not share the encoder.** It keeps a private block with no
   crop and `imagenet_norm: True` on a dataset already in [-1, 1]. It is not in the table above
   because it has produced no measured result, and if it does, that number is not comparable
   with the other image arms until the encoder question is settled. See `DEAD_CODE.md`.
6. **Pre-2026-09-16 `eval_bon` numbers on geometric splits are train-leaked.** `eval_bon.py`
   re-derived the split from the seed and ignored `split_file`. On the eight seed-42 manifests
   that is identical, so those numbers stand. On the five geometric manifests it is not:

   | manifest | "test" episodes that were actually TRAIN |
   |---|---|
   | `blockborder_train30` | 7 / 50 |
   | `blockborder_train60` | 11 / 50 |
   | `blockborder_train100` | 25 / 50 |
   | `blockquad_bottomleft_train137` | 38 / 50 |
   | `blockquad_topright_train173` | **42 / 50** |

   Any geometric-split number in `docs/reports/archive/SUCCESS_RATES_GEOMETRIC.md` produced by
   `eval_bon` should be re-measured. `eval_search_pusht` was never affected.

## The encoder

Every image arm runs the same encoder: ResNet18 with `IMAGENET1K_V1` weights trained end to
end, `BatchNorm` replaced by `GroupNorm`, a 76×76 crop of a 96×96 frame, 512-d per frame, image
scaled to [-1, 1], and **no `agent_pos` or other state concatenated**. Two implementations,
deliberately kept in step: `MultiImageObsEncoder` (offline arms, configured once in
`pusht_base.yaml`) and `STResNetExtractor` (`recurrent_ppo/corrupt_policy.py`, reused by SAC).

Image-only is *enforced*, not merely configured: `pusht_search_mixin` raises at startup if
`shape_meta` declares any `low_dim` key, so a hydra override cannot quietly train a privileged
model. The keypoint arms have no encoder at all — `nn.Flatten` / `FlattenObsEncoder` — because
the observation is already the state.

## Splits

One zarr (`data/pusht_cchi_v7_replay.zarr`, 206 episodes,
`episode_ends_checksum c88b56c9…`), 13 committed manifests in
`diffusion_policy/config/splits/`. The derivation is deterministic given a seed, the partition
is checksummed, and every training run writes `<run_dir>/splits.json` which is re-checked on
resume and by `eval_search_pusht`. `unit_tests/test_splits.py` covers all of it.

Train budgets are **nested**: the 25-episode set is a strict subset of the 30, which is a
strict subset of the 100, so a demo-budget comparison varies only the count. Val and test never
move with the budget.

## How to regenerate each row

```bash
# offline arms, in-the-loop search
python eval_search_pusht.py -c <ckpt> -o <out> --split test

# offline arms, post-hoc best-of-n oracle (NOT comparable to the above)
python eval_bon.py -c <ckpt> -o <out> --split test --n-samples 64

# BC-LSTM: predict_action returns the distribution MEAN, so draws are byte-identical
python eval_bon.py -c <ckpt> -o <out> --split test --n-samples 1

# PPO / recurrent PPO, offline over saved checkpoints
python -m recurrent_ppo.scripts.eval_checkpoints --run <run_dir>
python -m recurrent_ppo.scripts.runs_doc --logs <logs root> --out docs/reports/ppo/

# SAC's verifier (the Q as a ranker) -- on ST's own split
python sac/eval.py bon-sweep -c <ckpt>
```

Run provenance for every run referenced here is in `docs/reports/summaries/`; what stayed off
this repo, and where, is in `docs/reports/MANIFEST.md`.
