# PushT 30/100-demo sweep — success rates

Three policy families x two demo budgets. Every arm trains to 100k gradient steps with a checkpoint every 10k; every checkpoint is swept over `n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.

Regenerate with `python scripts/build_30_100_success_doc.py`. Source of truth is each run's `bon_search/success_curves.jsonl`.

## What the numbers mean

`n` is the **search width**: n action candidates are generated for the current observation, each is rolled out in a PushT simulator (the "verifier") to a scalar value, and one is executed. So n is a *test-time compute* axis — the same weights are read out n different ways.

Success rate is the fraction of the 50 test episodes reaching the goal coverage threshold. At 50 episodes a single cell carries a 95% CI of roughly +/-0.13 near 0.5, so **cell-to-cell differences under ~0.15 are not separable**; read down a column or across several checkpoints, not one cell.

## Choice mechanism

How the executed action is picked out of the n scored candidates, and how each candidate is produced. Identical across all six arms, so the arms differ only in policy family and demo budget.

| property | value | where it comes from |
|---|---|---|
| selection rule | **`argmax`** over the verifier value | recorded in every curve row as `selection` |
| selection temperature | n/a (argmax is not sampled) | `selection_temperature` null in every row |
| ranking signal | scalar verifier value = simulated rollout reward | `search_context: value` |
| candidates per decision | n (the sweep axis), i.i.d. given the obs | `n_generations` equals n in every row |
| eval episodes | 50 test episodes, `--skip-val` | `n_episodes` = 50 |
| eval seed | 42 (= `training.seed`) | `seed` in every row |

**Sampler.** The diffusion arms use `DDIMScheduler`, 100 train timesteps, **8 inference steps**, `prediction_type: epsilon`, and no `scheduler_step_kwargs` — so `DDIMScheduler.step` runs at its default `eta = 0.0`, the deterministic DDIM ODE. **No noise is injected during denoising.** The initial latent *is* a fresh `randn` per candidate, which is exactly what makes the n candidates differ and what best-of-n exploits. The ST-gaussian arm instead draws one `rsample` from a Normal head per candidate. Evals passed no `--noise-scheduler` / `--num-inference-steps` override, so every number below used the trained configuration.

**Arms.** BC is *not* a UNet: it is the same `PushTDiffusionSearchPolicy` as ST-diffusion with `max_actions: 1`, so at n>1 it is best-of-n over i.i.d. samples with an empty search context. That is the point — it gives BC the same test-time budget, so any gap is attributable to the learned search context rather than to drawing more samples. (A separate UNet BC config, `train_pusht_unet_bc`, exists but is not part of this sweep.)

## Data

The 30-demo train set is the first 30 episodes of the 100-demo train list in its own order; val (30) and test (50) are copied verbatim between them, so 30-vs-100 isolates training-set size alone. Manifests: `config/splits/pusht_seed42_train{30,100}.json`.

## ST-diffusion — 30 demos

`outer_inner/value_k16_corrupt-False_demos-30_seed-42` — search transformer, diffusion head

_7/10 checkpoints written, 7 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.08 | 0.28 | 0.36 | 0.42 | 0.46 | 0.50 |
| 20,000 | 0.20 | 0.12 | 0.22 | 0.44 | 0.42 | 0.42 | 0.42 |
| 30,000 | 0.22 | 0.24 | 0.30 | 0.34 | 0.44 | 0.42 | 0.50 |
| 40,000 | 0.36 | 0.24 | 0.34 | 0.38 | 0.42 | 0.46 | 0.44 |
| 50,000 | 0.18 | 0.26 | 0.26 | 0.30 | 0.34 | 0.36 | 0.42 |
| 60,000 | 0.18 | 0.30 | 0.28 | 0.32 | 0.22 | 0.44 | 0.32 |
| 70,000 | 0.20 | 0.16 | 0.20 | 0.28 | 0.30 | 0.36 | 0.42 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.481 | 0.671 | 0.675 | 0.711 | 0.780 | 0.759 | 0.839 |
| 20,000 | 0.522 | 0.665 | 0.642 | 0.726 | 0.713 | 0.821 | 0.765 |
| 30,000 | 0.632 | 0.654 | 0.771 | 0.731 | 0.763 | 0.803 | 0.811 |
| 40,000 | 0.652 | 0.710 | 0.707 | 0.677 | 0.769 | 0.818 | 0.809 |
| 50,000 | 0.617 | 0.687 | 0.666 | 0.673 | 0.681 | 0.725 | 0.790 |
| 60,000 | 0.594 | 0.651 | 0.599 | 0.636 | 0.706 | 0.717 | 0.719 |
| 70,000 | 0.572 | 0.627 | 0.635 | 0.659 | 0.669 | 0.734 | 0.728 |

## ST-diffusion — 100 demos

`outer_inner/value_k16_corrupt-False_demos-100_seed-42` — search transformer, diffusion head

_7/10 checkpoints written, 7 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.10 | 0.48 | 0.54 | 0.70 | 0.58 | 0.68 | 0.76 |
| 20,000 | 0.28 | 0.56 | 0.70 | 0.80 | 0.72 | 0.78 | 0.84 |
| 30,000 | 0.38 | 0.38 | 0.60 | 0.70 | 0.88 | 0.70 | 0.86 |
| 40,000 | 0.38 | 0.52 | 0.66 | 0.64 | 0.68 | 0.80 | 0.72 |
| 50,000 | 0.30 | 0.42 | 0.54 | 0.54 | 0.68 | 0.70 | 0.64 |
| 60,000 | 0.34 | 0.50 | 0.56 | 0.64 | 0.62 | 0.64 | 0.62 |
| 70,000 | 0.32 | 0.62 | 0.62 | 0.64 | 0.62 | 0.60 | 0.72 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.732 | 0.952 | 0.984 | 0.993 | 0.988 | 0.991 | 0.987 |
| 20,000 | 0.879 | 0.947 | 0.975 | 0.976 | 0.958 | 0.980 | 0.974 |
| 30,000 | 0.908 | 0.888 | 0.932 | 0.993 | 0.997 | 0.963 | 0.999 |
| 40,000 | 0.869 | 0.923 | 0.921 | 0.954 | 0.943 | 0.975 | 0.965 |
| 50,000 | 0.870 | 0.913 | 0.927 | 0.937 | 0.942 | 0.940 | 0.979 |
| 60,000 | 0.897 | 0.939 | 0.926 | 0.972 | 0.949 | 0.954 | 0.984 |
| 70,000 | 0.888 | 0.924 | 0.955 | 0.969 | 0.971 | 0.969 | 0.963 |

## ST-gaussian — 30 demos

`offline/gaussian_k16_corrupt-False_demos-30_seed-42` — search transformer, Gaussian head

_5/10 checkpoints written, 5 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.08 | 0.26 | 0.16 | 0.30 | 0.38 | 0.42 | 0.48 |
| 20,000 | 0.22 | 0.14 | 0.30 | 0.28 | 0.30 | 0.34 | 0.42 |
| 30,000 | 0.14 | 0.22 | 0.20 | 0.34 | 0.28 | 0.40 | 0.34 |
| 40,000 | 0.22 | 0.14 | 0.18 | 0.16 | 0.18 | 0.24 | 0.28 |
| 50,000 | 0.08 | 0.20 | 0.16 | 0.22 | 0.30 | 0.26 | 0.26 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.551 | 0.616 | 0.642 | 0.736 | 0.669 | 0.707 | 0.710 |
| 20,000 | 0.667 | 0.585 | 0.657 | 0.676 | 0.712 | 0.684 | 0.762 |
| 30,000 | 0.541 | 0.609 | 0.603 | 0.583 | 0.632 | 0.622 | 0.597 |
| 40,000 | 0.571 | 0.506 | 0.521 | 0.560 | 0.598 | 0.600 | 0.568 |
| 50,000 | 0.555 | 0.545 | 0.557 | 0.581 | 0.607 | 0.601 | 0.571 |

## ST-gaussian — 100 demos

`offline/gaussian_k16_corrupt-False_demos-100_seed-42` — search transformer, Gaussian head

_3/10 checkpoints written, 3 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.20 | 0.42 | 0.58 | 0.64 | 0.76 | 0.78 | 0.84 |
| 20,000 | 0.38 | 0.60 | 0.72 | 0.64 | 0.76 | 0.74 | 0.88 |
| 30,000 | 0.42 | 0.40 | 0.70 | 0.62 | 0.78 | 0.84 | 0.82 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.895 | 0.967 | 0.992 | 0.992 | 0.994 | 0.997 | 0.981 |
| 20,000 | 0.901 | 0.972 | 0.970 | 0.970 | 0.995 | 0.972 | 0.989 |
| 30,000 | 0.944 | 0.909 | 0.933 | 0.937 | 0.953 | 0.977 | 0.943 |

## BC — 30 demos

`offline/bc_demos-30_seed-42` — same policy class as ST-diffusion, max_actions=1

_10/10 checkpoints written, 10 fully swept._

### Test success rate

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

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.301 | 0.405 | 0.519 | 0.679 | 0.699 | 0.805 | 0.835 |
| 20,000 | 0.563 | 0.600 | 0.626 | 0.704 | 0.745 | 0.738 | 0.808 |
| 30,000 | 0.570 | 0.546 | 0.626 | 0.700 | 0.766 | 0.651 | 0.774 |
| 40,000 | 0.559 | 0.646 | 0.671 | 0.735 | 0.733 | 0.762 | 0.753 |
| 50,000 | 0.602 | 0.603 | 0.656 | 0.703 | 0.746 | 0.736 | 0.676 |
| 60,000 | 0.596 | 0.629 | 0.610 | 0.615 | 0.679 | 0.643 | 0.697 |
| 70,000 | 0.602 | 0.656 | 0.683 | 0.654 | 0.683 | 0.677 | 0.753 |
| 80,000 | 0.580 | 0.599 | 0.601 | 0.645 | 0.652 | 0.662 | 0.695 |
| 90,000 | 0.607 | 0.581 | 0.656 | 0.635 | 0.633 | 0.640 | 0.679 |
| 100,000 | 0.591 | 0.602 | 0.585 | 0.679 | 0.655 | 0.698 | 0.659 |

## BC — 100 demos

`offline/bc_demos-100_seed-42` — same policy class as ST-diffusion, max_actions=1

_10/10 checkpoints written, 10 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.02 | 0.22 | 0.32 | 0.36 | 0.54 | 0.64 | 0.74 |
| 20,000 | 0.28 | 0.40 | 0.44 | 0.46 | 0.56 | 0.50 | 0.78 |
| 30,000 | 0.26 | 0.50 | 0.60 | 0.66 | 0.64 | 0.60 | 0.78 |
| 40,000 | 0.46 | 0.56 | 0.68 | 0.72 | 0.64 | 0.76 | 0.84 |
| 50,000 | 0.38 | 0.60 | 0.78 | 0.70 | 0.46 | 0.66 | 0.64 |
| 60,000 | 0.48 | 0.58 | 0.60 | 0.76 | 0.74 | 0.74 | 0.70 |
| 70,000 | 0.64 | 0.66 | 0.64 | 0.66 | 0.72 | 0.80 | 0.76 |
| 80,000 | 0.54 | 0.64 | 0.68 | 0.74 | 0.56 | 0.66 | 0.80 |
| 90,000 | 0.40 | 0.52 | 0.50 | 0.64 | 0.64 | 0.54 | 0.78 |
| 100,000 | 0.34 | 0.42 | 0.62 | 0.66 | 0.76 | 0.66 | 0.80 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.611 | 0.839 | 0.953 | 0.975 | 0.990 | 0.993 | 0.995 |
| 20,000 | 0.827 | 0.951 | 0.967 | 0.956 | 0.992 | 0.981 | 0.975 |
| 30,000 | 0.901 | 0.936 | 0.965 | 0.947 | 0.985 | 0.979 | 0.998 |
| 40,000 | 0.925 | 0.943 | 0.962 | 0.975 | 0.964 | 0.980 | 0.986 |
| 50,000 | 0.929 | 0.942 | 0.966 | 0.985 | 0.951 | 0.980 | 0.981 |
| 60,000 | 0.930 | 0.969 | 0.968 | 0.972 | 0.943 | 0.944 | 0.986 |
| 70,000 | 0.959 | 0.962 | 0.974 | 0.985 | 0.956 | 0.966 | 0.966 |
| 80,000 | 0.967 | 0.934 | 0.980 | 0.959 | 0.956 | 0.963 | 0.984 |
| 90,000 | 0.930 | 0.902 | 0.937 | 0.968 | 0.971 | 0.938 | 0.995 |
| 100,000 | 0.967 | 0.938 | 0.936 | 0.958 | 0.958 | 0.931 | 0.974 |

## Coverage

| arm | demos | checkpoints | fully swept | pending |
|---|---:|---:|---:|---|
| ST-diffusion | 30 | 7/10 | 7 | — |
| ST-diffusion | 100 | 7/10 | 7 | — |
| ST-gaussian | 30 | 5/10 | 5 | — |
| ST-gaussian | 100 | 3/10 | 3 | — |
| BC | 30 | 10/10 | 10 | — |
| BC | 100 | 10/10 | 10 | — |

_No checkpoint or n is nominated as best: every evaluated cell is printed and selection is never done on the test split._
