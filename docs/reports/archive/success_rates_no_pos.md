# PushT image-only ablation — success rates

What did `agent_pos` and `feedback` buy the policy? Each pair below is one arm trained twice: once on the full observation and once on **pixels alone**, with both low_dim keys deleted from `shape_meta.obs`. Every arm: 30 demos, seed 42, `t_goal` verifier, 4/4/256, 100k gradient steps with a checkpoint every 10k, every checkpoint swept over `n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.

_Generated 2026-08-29 from the on-disk eval output._

**What changed and what did not.** The ablation is two hydra key deletions, `~task.shape_meta.obs.agent_pos ~task.shape_meta.obs.feedback`, which shrink `MultiImageObsEncoder` from 530 to 512 outputs and touch nothing else. The dataset still emits all three keys and the normalizer still holds params for all three. Crucially the **verifier is unchanged**: `_verifier_inputs` reads `agent_pos` and `feedback` straight off the obs dict to reset its sim, so the search scores candidates exactly as well in both columns. A gap here is the policy losing the closed-form T pose, not the search losing its ground truth.

`feedback` is an exact, invertible transform of `block_pos`, so the with-pos column is a privileged-observation setting: it hands the policy the goal-relative block pose rather than making the ResNet extract it. The image-only column is the one comparable to published PushT-image baselines.

**Reading these tables.** Success rate is the fraction of the 50 test episodes reaching the coverage threshold. At 50 episodes a single cell carries a 95% CI of roughly ±0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not separable** — read down a column or across several checkpoints, never one cell. The delta tables are printed in full with no cell singled out; picking the largest one is selection on test.

## ST-diffusion k=1 (4/4/256)

### With `agent_pos` + `feedback` (control)

`offline/value_k1_demos-30_seed-42/bon_search` — image + agent_pos + feedback (530-d encoder)

_10/10 checkpoints written, 10 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.04 | 0.08 | 0.26 | 0.34 | 0.58 | 0.46 |
| 20,000 | 0.14 | 0.18 | 0.24 | 0.30 | 0.40 | 0.54 | 0.46 |
| 30,000 | 0.14 | 0.16 | 0.28 | 0.30 | 0.50 | 0.36 | 0.56 |
| 40,000 | 0.16 | 0.30 | 0.32 | 0.34 | 0.22 | 0.36 | 0.42 |
| 50,000 | 0.14 | 0.22 | 0.32 | 0.40 | 0.38 | 0.46 | 0.38 |
| 60,000 | 0.18 | 0.18 | 0.22 | 0.22 | 0.28 | 0.30 | 0.36 |
| 70,000 | 0.16 | 0.24 | 0.30 | 0.26 | 0.32 | 0.36 | 0.36 |
| 80,000 | 0.16 | 0.18 | 0.28 | 0.20 | 0.22 | 0.30 | 0.42 |
| 90,000 | 0.12 | 0.18 | 0.20 | 0.32 | 0.26 | 0.30 | 0.32 |
| 100,000 | 0.22 | 0.20 | 0.22 | 0.28 | 0.22 | 0.22 | 0.32 |

### Image only

`offline/value_k1_ver-t_goal_nopos_corrupt-False_demos-30_seed-42/bon_search` — image only (512-d encoder)

_10/10 checkpoints written, 10 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.04 | 0.16 | 0.18 | 0.32 | 0.34 | 0.50 |
| 20,000 | 0.06 | 0.12 | 0.18 | 0.18 | 0.26 | 0.24 | 0.28 |
| 30,000 | 0.10 | 0.16 | 0.26 | 0.32 | 0.24 | 0.38 | 0.38 |
| 40,000 | 0.18 | 0.20 | 0.22 | 0.30 | 0.28 | 0.32 | 0.26 |
| 50,000 | 0.16 | 0.10 | 0.20 | 0.26 | 0.32 | 0.26 | 0.28 |
| 60,000 | 0.14 | 0.16 | 0.30 | 0.28 | 0.34 | 0.26 | 0.38 |
| 70,000 | 0.08 | 0.14 | 0.20 | 0.26 | 0.28 | 0.18 | 0.30 |
| 80,000 | 0.16 | 0.14 | 0.18 | 0.26 | 0.32 | 0.30 | 0.42 |
| 90,000 | 0.12 | 0.18 | 0.18 | 0.22 | 0.30 | 0.24 | 0.38 |
| 100,000 | 0.06 | 0.14 | 0.20 | 0.32 | 0.26 | 0.28 | 0.38 |

## ST-diffusion k=16 (4/4/256)

### With `agent_pos` + `feedback` (control)

`outer_inner/value_k16_corrupt-False_demos-30_seed-42/bon_search` — image + agent_pos + feedback (530-d encoder)

_10/10 checkpoints written, 10 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.08 | 0.28 | 0.36 | 0.42 | 0.46 | 0.50 |
| 20,000 | 0.20 | 0.12 | 0.22 | 0.44 | 0.42 | 0.42 | 0.42 |
| 30,000 | 0.22 | 0.24 | 0.30 | 0.34 | 0.44 | 0.42 | 0.50 |
| 40,000 | 0.36 | 0.24 | 0.34 | 0.38 | 0.42 | 0.46 | 0.44 |
| 50,000 | 0.18 | 0.26 | 0.26 | 0.30 | 0.34 | 0.36 | 0.42 |
| 60,000 | 0.18 | 0.30 | 0.28 | 0.32 | 0.22 | 0.44 | 0.32 |
| 70,000 | 0.20 | 0.16 | 0.20 | 0.28 | 0.30 | 0.36 | 0.42 |
| 80,000 | 0.24 | 0.22 | 0.18 | 0.30 | 0.26 | 0.30 | 0.28 |
| 90,000 | 0.20 | 0.18 | 0.18 | 0.28 | 0.30 | 0.34 | 0.30 |
| 100,000 | 0.20 | 0.28 | 0.32 | 0.28 | 0.20 | 0.30 | 0.24 |

### Image only

`outer_inner/value_k16_ver-t_goal_nopos_corrupt-False_demos-30_seed-42/bon_search` — image only (512-d encoder)

_7/10 checkpoints written, 7 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.18 | 0.24 | 0.14 | 0.36 | 0.36 | 0.50 |
| 20,000 | 0.10 | 0.14 | 0.16 | 0.24 | 0.40 | 0.34 | 0.46 |
| 30,000 | 0.06 | 0.20 | 0.08 | 0.30 | 0.28 | 0.22 | 0.30 |
| 40,000 | 0.12 | 0.14 | 0.14 | 0.14 | 0.22 | 0.18 | 0.24 |
| 50,000 | 0.14 | 0.14 | 0.18 | 0.22 | 0.18 | 0.22 | 0.28 |
| 60,000 | 0.06 | 0.30 | 0.20 | 0.18 | 0.16 | 0.26 | 0.20 |
| 70,000 | 0.06 | 0.22 | 0.10 | 0.18 | 0.16 | 0.08 | 0.32 |

## UNet BC (293.4M)

### With `agent_pos` + `feedback` (control)

`unet_bc/unetbc_demos-30_seed-42/bon_search` — image + agent_pos + feedback (530-d encoder)

_10/10 checkpoints written, 10 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | 0.18 | 0.28 | 0.46 | 0.62 | 0.70 | 0.70 |
| 20,000 | 0.12 | 0.24 | 0.34 | 0.48 | 0.52 | 0.66 | 0.68 |
| 30,000 | 0.16 | 0.30 | 0.34 | 0.50 | 0.50 | 0.42 | 0.60 |
| 40,000 | 0.22 | 0.30 | 0.38 | 0.36 | 0.50 | 0.48 | 0.64 |
| 50,000 | 0.14 | 0.22 | 0.26 | 0.38 | 0.50 | 0.34 | 0.42 |
| 60,000 | 0.26 | 0.16 | 0.36 | 0.34 | 0.48 | 0.30 | 0.38 |
| 70,000 | 0.22 | 0.32 | 0.24 | 0.34 | 0.26 | 0.42 | 0.36 |
| 80,000 | 0.24 | 0.24 | 0.28 | 0.34 | 0.38 | 0.38 | 0.38 |
| 90,000 | 0.30 | 0.26 | 0.28 | 0.36 | 0.26 | 0.36 | 0.44 |
| 100,000 | 0.20 | 0.28 | 0.26 | 0.42 | 0.24 | 0.34 | 0.40 |

### Image only

`unet_bc/unetbc_ver-t_goal_nopos_demos-30_seed-42/bon_search` — image only (512-d encoder)

_10/10 checkpoints written, 10 fully swept._

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.18 | 0.28 | 0.48 | 0.58 | 0.68 | 0.84 |
| 20,000 | 0.16 | 0.24 | 0.22 | 0.40 | 0.42 | 0.64 | 0.62 |
| 30,000 | 0.10 | 0.14 | 0.40 | 0.38 | 0.38 | 0.50 | 0.54 |
| 40,000 | 0.12 | 0.24 | 0.32 | 0.38 | 0.44 | 0.36 | 0.44 |
| 50,000 | 0.14 | 0.28 | 0.28 | 0.30 | 0.26 | 0.46 | 0.46 |
| 60,000 | 0.20 | 0.30 | 0.24 | 0.34 | 0.40 | 0.18 | 0.28 |
| 70,000 | 0.16 | 0.20 | 0.24 | 0.28 | 0.30 | 0.28 | 0.40 |
| 80,000 | 0.20 | 0.20 | 0.12 | 0.22 | 0.24 | 0.18 | 0.22 |
| 90,000 | 0.12 | 0.20 | 0.28 | 0.20 | 0.26 | 0.32 | 0.32 |
| 100,000 | 0.10 | 0.18 | 0.24 | 0.12 | 0.22 | 0.18 | 0.22 |

## Image-only − with-pos, paired by step and n

Negative = removing the two keys cost success at that checkpoint and that search width. Blank where either side has no measurement; nothing is imputed.


### ST-diffusion k=1 (4/4/256)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | +0.00 | +0.00 | +0.08 | -0.08 | -0.02 | -0.24 | +0.04 |
| 20,000 | -0.08 | -0.06 | -0.06 | -0.12 | -0.14 | -0.30 | -0.18 |
| 30,000 | -0.04 | +0.00 | -0.02 | +0.02 | -0.26 | +0.02 | -0.18 |
| 40,000 | +0.02 | -0.10 | -0.10 | -0.04 | +0.06 | -0.04 | -0.16 |
| 50,000 | +0.02 | -0.12 | -0.12 | -0.14 | -0.06 | -0.20 | -0.10 |
| 60,000 | -0.04 | -0.02 | +0.08 | +0.06 | +0.06 | -0.04 | +0.02 |
| 70,000 | -0.08 | -0.10 | -0.10 | +0.00 | -0.04 | -0.18 | -0.06 |
| 80,000 | +0.00 | -0.04 | -0.10 | +0.06 | +0.10 | +0.00 | +0.00 |
| 90,000 | +0.00 | +0.00 | -0.02 | -0.10 | +0.04 | -0.06 | +0.06 |
| 100,000 | -0.16 | -0.06 | -0.02 | +0.04 | +0.04 | +0.06 | +0.06 |

### ST-diffusion k=16 (4/4/256)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | +0.00 | +0.10 | -0.04 | -0.22 | -0.06 | -0.10 | +0.00 |
| 20,000 | -0.10 | +0.02 | -0.06 | -0.20 | -0.02 | -0.08 | +0.04 |
| 30,000 | -0.16 | -0.04 | -0.22 | -0.04 | -0.16 | -0.20 | -0.20 |
| 40,000 | -0.24 | -0.10 | -0.20 | -0.24 | -0.20 | -0.28 | -0.20 |
| 50,000 | -0.04 | -0.12 | -0.08 | -0.08 | -0.16 | -0.14 | -0.14 |
| 60,000 | -0.12 | +0.00 | -0.08 | -0.14 | -0.06 | -0.18 | -0.12 |
| 70,000 | -0.14 | +0.06 | -0.10 | -0.10 | -0.14 | -0.28 | -0.10 |

### UNet BC (293.4M)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | -0.02 | +0.00 | +0.00 | +0.02 | -0.04 | -0.02 | +0.14 |
| 20,000 | +0.04 | +0.00 | -0.12 | -0.08 | -0.10 | -0.02 | -0.06 |
| 30,000 | -0.06 | -0.16 | +0.06 | -0.12 | -0.12 | +0.08 | -0.06 |
| 40,000 | -0.10 | -0.06 | -0.06 | +0.02 | -0.06 | -0.12 | -0.20 |
| 50,000 | +0.00 | +0.06 | +0.02 | -0.08 | -0.24 | +0.12 | +0.04 |
| 60,000 | -0.06 | +0.14 | -0.12 | +0.00 | -0.08 | -0.12 | -0.10 |
| 70,000 | -0.06 | -0.12 | +0.00 | -0.06 | +0.04 | -0.14 | +0.04 |
| 80,000 | -0.04 | -0.04 | -0.16 | -0.12 | -0.14 | -0.20 | -0.16 |
| 90,000 | -0.18 | -0.06 | +0.00 | -0.16 | +0.00 | -0.04 | -0.12 |
| 100,000 | -0.10 | -0.10 | -0.02 | -0.30 | -0.02 | -0.16 | -0.18 |

## Status

| arm | observation | checkpoints | fully swept | selection | episodes | seed |
|---|---|---:|---:|---|---:|---:|
| ST-diffusion k=1 (4/4/256) | image + agent_pos + feedback (530-d encoder) | 10/10 | 10 | argmax | 50 | 42 |
| ST-diffusion k=1 (4/4/256) | image only (512-d encoder) | 10/10 | 10 | argmax | 50 | 42 |
| ST-diffusion k=16 (4/4/256) | image + agent_pos + feedback (530-d encoder) | 10/10 | 10 | argmax | 50 | 42 |
| ST-diffusion k=16 (4/4/256) | image only (512-d encoder) | 7/10 | 7 | argmax | 50 | 42 |
| UNet BC (293.4M) | image + agent_pos + feedback (530-d encoder) | 10/10 | 10 | argmax | 50 | 42 |
| UNet BC (293.4M) | image only (512-d encoder) | 10/10 | 10 | argmax | 50 | 42 |

