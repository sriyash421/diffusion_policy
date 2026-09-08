# PushT 30/100-demo sweep — success rates

Three policy families x two demo budgets, plus three 30-demo additions (a diffusion UNet baseline, and a ~24x wider search transformer at widths 1 and 16). Every arm trains to 100k gradient steps with a checkpoint every 10k; every checkpoint is swept over `n = 1, 2, 4, 8, 16, 32, 64` on the same 50 held-out test episodes.

_Generated 2026-08-23 from the on-disk eval output._

Regenerate with `python scripts/build_30_100_success_doc.py` — it rewrites this file in place. Source of truth is each run's `bon_search*/success_curves.jsonl`.

**Verifier: `t_goal`** — the candidate value is `-(mean per-keypoint distance of the T from the goal T)`. Every arm here was trained and evaluated under it, so all numbers are mutually comparable. Rows measured under any other verifier are filtered out, not averaged in: the verifier became selectable on 2026-08-19 (`pusht_verifier.VALUE_FNS`) and the choice moves success rates by tens of points, so the two eras must never share a table.

Arm labels carry `(n_layer/n_head/n_emb, total params)` — the transformer shape and the whole policy's parameter count including the shared 11.2M ResNet-18 encoder, counted from each run's checkpoint. The trunk alone is 5.9M at 4/4/256, 126.6M at 6/8/1024, and 282.2M for the UNet.

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

**The two width-1 baselines are different things, and neither is "BC" alone.**

`ST k=1` is the *same transformer* as `ST-diffusion k16` (`PushTDiffusionSearchPolicy`) trained at `max_actions: 1`. Its search context is always empty, so it isolates the *learned search context*: it shares architecture, encoder, scheduler, optimizer and data with k16, and differs only in whether candidates condition on each other during training. `ST-big k=1` is the same thing at a ~24x wider trunk.

`UNet BC` is a *different architecture* — `PushTUNetSearchPolicy`, a convolutional diffusion UNet with no transformer and no search context at all. It isolates the *backbone*. It is matched to the ST arms on everything outside the backbone: same 30-demo manifest, seed 42, 100k steps, DDIM at 8 inference steps, ResNet-18/ImageNet encoder with the same [76,76] random crop, batch 32, lr 1e-4, EMA 0.995.

At n>1 **both** are plain best-of-n over i.i.d. samples scored by the same verifier, so all arms are compared at a matched test-time budget and any gap is attributable to the trained policy rather than to drawing more samples.

## Data

The 30-demo train set is the first 30 episodes of the 100-demo train list in its own order; val (30) and test (50) are copied verbatim between them, so 30-vs-100 isolates training-set size alone. Manifests: `config/splits/pusht_seed42_train{30,100}.json`.

## ST-diffusion k=16 (4/4/256, 17.1M) — 30 demos

`outer_inner/value_k16_corrupt-False_demos-30_seed-42/bon_search` — search transformer, diffusion head

_10/10 checkpoints written, 10 fully swept._

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
| 80,000 | 0.24 | 0.22 | 0.18 | 0.30 | 0.26 | 0.30 | 0.28 |
| 90,000 | 0.20 | 0.18 | 0.18 | 0.28 | 0.30 | 0.34 | 0.30 |
| 100,000 | 0.20 | 0.28 | 0.32 | 0.28 | 0.20 | 0.30 | 0.24 |

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
| 80,000 | 0.585 | 0.603 | 0.607 | 0.615 | 0.663 | 0.694 | 0.717 |
| 90,000 | 0.628 | 0.576 | 0.612 | 0.616 | 0.672 | 0.706 | 0.650 |
| 100,000 | 0.605 | 0.644 | 0.594 | 0.709 | 0.649 | 0.646 | 0.614 |

## ST-diffusion k=1 (4/4/256, 17.1M) — 30 demos

`offline/bc_demos-30_seed-42/bon_search` — same class as k=16, width 1 (empty search context)

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

## UNet BC (293.4M) — 30 demos

`unet_bc/unetbc_demos-30_seed-42/bon_search` — diffusion UNet, i.i.d. best-of-n (no search context)

_10/10 checkpoints written, 10 fully swept._

### Test success rate

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

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.492 | 0.640 | 0.828 | 0.842 | 0.849 | 0.900 | 0.929 |
| 20,000 | 0.618 | 0.719 | 0.761 | 0.788 | 0.864 | 0.867 | 0.891 |
| 30,000 | 0.617 | 0.749 | 0.726 | 0.754 | 0.771 | 0.791 | 0.811 |
| 40,000 | 0.623 | 0.698 | 0.730 | 0.760 | 0.748 | 0.798 | 0.805 |
| 50,000 | 0.632 | 0.677 | 0.704 | 0.702 | 0.740 | 0.734 | 0.791 |
| 60,000 | 0.654 | 0.689 | 0.741 | 0.716 | 0.782 | 0.724 | 0.684 |
| 70,000 | 0.673 | 0.725 | 0.729 | 0.697 | 0.717 | 0.749 | 0.741 |
| 80,000 | 0.758 | 0.734 | 0.747 | 0.726 | 0.725 | 0.746 | 0.813 |
| 90,000 | 0.729 | 0.710 | 0.711 | 0.769 | 0.736 | 0.763 | 0.792 |
| 100,000 | 0.683 | 0.663 | 0.776 | 0.746 | 0.744 | 0.770 | 0.796 |

# Arm-distance verifiers (`armT`, `armTn`, `armTd`)

Everything above ranks candidates by `t_goal`. The arms below add the arm-to-T-centre distance to that value. **Do not read these tables against the ones above cell-by-cell as if they were the same experiment** — the episodes, protocol and seed match, but the ranking rule does not, and that is the whole variable.

| verifier | value | why |
|---|---|---|
| `t_goal` | `-(T-to-goal)` | the original. Before the arm touches the block no candidate can move the T, so this value is *identical* across candidates and argmax is a coin flip on every approach step. |
| `armT` | `-(T-to-goal + arm-to-T)` | raw sum. Superseded: the two terms have comparable marginal scale (80 vs 91 px) but the spread argmax actually sees — across candidates *within* one control step — is 13.6 vs 52.1 px, so the arm term outvotes task progress ~4:1. |
| `armTn` | `-(T-to-goal/13.6 + arm-to-T/52.1)` | each term divided by its own within-step spread, so a 1-sigma gain in either counts the same. |

| `armTd` | `-(z(T-to-goal) + z(arm-to-T))` | the two constants above replaced by the spread measured across the n candidates *of that control step*, so the weighting tracks what the choice can actually change now. Eval-only: it is defined over a candidate SET, so it has no per-candidate scalar to train against, and its score is standardized rather than pixels — not `<= 0`, and comparable only WITHIN a step. |
## ST-diffusion k=1 (4/4/256) — armTn — 30 demos

`offline/value_k1_ver-armTn_corrupt-False_demos-30_seed-42/bon_search` — trained AND selected under armTn

_10/10 checkpoints written, 10 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | 0.04 | 0.08 | 0.08 | 0.08 | 0.18 | 0.16 |
| 20,000 | 0.16 | 0.02 | 0.24 | 0.26 | 0.22 | 0.24 | 0.22 |
| 30,000 | 0.06 | 0.18 | 0.24 | 0.28 | 0.34 | 0.46 | 0.34 |
| 40,000 | 0.12 | 0.12 | 0.16 | 0.28 | 0.24 | 0.34 | 0.34 |
| 50,000 | 0.12 | 0.10 | 0.26 | 0.24 | 0.26 | 0.32 | 0.36 |
| 60,000 | 0.18 | 0.12 | 0.16 | 0.24 | 0.28 | 0.32 | 0.32 |
| 70,000 | 0.22 | 0.16 | 0.20 | 0.20 | 0.22 | 0.30 | 0.30 |
| 80,000 | 0.20 | 0.18 | 0.20 | 0.32 | 0.36 | 0.30 | 0.20 |
| 90,000 | 0.16 | 0.24 | 0.18 | 0.30 | 0.32 | 0.30 | 0.28 |
| 100,000 | 0.18 | 0.16 | 0.20 | 0.24 | 0.20 | 0.28 | 0.38 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.368 | 0.342 | 0.432 | 0.464 | 0.464 | 0.513 | 0.478 |
| 20,000 | 0.522 | 0.566 | 0.583 | 0.587 | 0.549 | 0.620 | 0.541 |
| 30,000 | 0.470 | 0.533 | 0.581 | 0.555 | 0.585 | 0.597 | 0.601 |
| 40,000 | 0.541 | 0.558 | 0.632 | 0.596 | 0.620 | 0.687 | 0.662 |
| 50,000 | 0.554 | 0.553 | 0.557 | 0.650 | 0.574 | 0.674 | 0.651 |
| 60,000 | 0.554 | 0.597 | 0.523 | 0.560 | 0.514 | 0.577 | 0.555 |
| 70,000 | 0.642 | 0.584 | 0.637 | 0.514 | 0.620 | 0.593 | 0.558 |
| 80,000 | 0.591 | 0.612 | 0.603 | 0.656 | 0.650 | 0.596 | 0.571 |
| 90,000 | 0.573 | 0.603 | 0.591 | 0.535 | 0.677 | 0.593 | 0.614 |
| 100,000 | 0.598 | 0.560 | 0.596 | 0.626 | 0.631 | 0.581 | 0.649 |

## ST-diffusion k=16 (4/4/256) — armTn — 30 demos

`outer_inner/value_k16_ver-armTn_corrupt-False_demos-30_seed-42/bon_search` — trained AND selected under armTn

_10/10 checkpoints written, 10 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.06 | 0.12 | 0.14 | 0.26 | 0.26 | 0.32 | 0.36 |
| 20,000 | 0.14 | 0.20 | 0.20 | 0.24 | 0.32 | 0.38 | 0.32 |
| 30,000 | 0.20 | 0.18 | 0.28 | 0.26 | 0.36 | 0.30 | 0.52 |
| 40,000 | 0.12 | 0.28 | 0.24 | 0.26 | 0.30 | 0.24 | 0.42 |
| 50,000 | 0.12 | 0.24 | 0.30 | 0.24 | 0.26 | 0.26 | 0.34 |
| 60,000 | 0.12 | 0.26 | 0.26 | 0.30 | 0.22 | 0.26 | 0.32 |
| 70,000 | 0.14 | 0.24 | 0.22 | 0.28 | 0.32 | 0.24 | 0.40 |
| 80,000 | 0.22 | 0.30 | 0.26 | 0.30 | 0.30 | 0.26 | 0.36 |
| 90,000 | 0.10 | 0.28 | 0.32 | 0.20 | 0.24 | 0.42 | 0.38 |
| 100,000 | 0.18 | 0.16 | 0.26 | 0.30 | 0.32 | 0.24 | 0.26 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.424 | 0.558 | 0.609 | 0.668 | 0.628 | 0.657 | 0.702 |
| 20,000 | 0.535 | 0.580 | 0.653 | 0.655 | 0.660 | 0.676 | 0.767 |
| 30,000 | 0.589 | 0.612 | 0.644 | 0.707 | 0.728 | 0.691 | 0.787 |
| 40,000 | 0.539 | 0.625 | 0.590 | 0.648 | 0.643 | 0.681 | 0.743 |
| 50,000 | 0.561 | 0.625 | 0.600 | 0.659 | 0.622 | 0.714 | 0.722 |
| 60,000 | 0.567 | 0.668 | 0.657 | 0.666 | 0.653 | 0.705 | 0.771 |
| 70,000 | 0.547 | 0.675 | 0.633 | 0.688 | 0.699 | 0.641 | 0.711 |
| 80,000 | 0.566 | 0.645 | 0.639 | 0.612 | 0.655 | 0.626 | 0.676 |
| 90,000 | 0.555 | 0.673 | 0.636 | 0.645 | 0.667 | 0.630 | 0.662 |
| 100,000 | 0.571 | 0.617 | 0.639 | 0.647 | 0.654 | 0.641 | 0.650 |

## UNet BC (293.4M) — armTn, re-ranked — 30 demos

`unet_bc/unetbc_demos-30_seed-42/bon_search_ver-armTn` — the t_goal UNet BC weights, re-scored under armTn

_10/10 checkpoints written, 10 fully swept._

### Test success rate

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.04 | 0.10 | 0.16 | 0.18 | 0.18 | 0.16 | 0.16 |
| 20,000 | 0.12 | 0.14 | 0.32 | 0.30 | 0.32 | 0.40 | 0.38 |
| 30,000 | 0.16 | 0.28 | 0.30 | 0.42 | 0.52 | 0.36 | 0.50 |
| 40,000 | 0.22 | 0.22 | 0.38 | 0.36 | 0.30 | 0.38 | 0.46 |
| 50,000 | 0.14 | 0.24 | 0.20 | 0.26 | 0.36 | 0.44 | 0.32 |
| 60,000 | 0.26 | 0.30 | 0.26 | 0.28 | 0.28 | 0.38 | 0.40 |
| 70,000 | 0.22 | 0.22 | 0.34 | 0.34 | 0.28 | 0.36 | 0.46 |
| 80,000 | 0.24 | 0.18 | 0.24 | 0.30 | 0.26 | 0.24 | 0.28 |
| 90,000 | 0.30 | 0.34 | 0.22 | 0.32 | 0.22 | 0.30 | 0.34 |
| 100,000 | 0.20 | 0.28 | 0.30 | 0.32 | 0.42 | 0.22 | 0.32 |

### Test mean reward (max coverage reached)

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.492 | 0.560 | 0.611 | 0.490 | 0.584 | 0.491 | 0.469 |
| 20,000 | 0.618 | 0.599 | 0.683 | 0.723 | 0.731 | 0.701 | 0.679 |
| 30,000 | 0.617 | 0.700 | 0.675 | 0.746 | 0.709 | 0.638 | 0.758 |
| 40,000 | 0.623 | 0.696 | 0.733 | 0.692 | 0.708 | 0.702 | 0.734 |
| 50,000 | 0.632 | 0.689 | 0.652 | 0.638 | 0.687 | 0.712 | 0.680 |
| 60,000 | 0.654 | 0.721 | 0.695 | 0.628 | 0.639 | 0.716 | 0.698 |
| 70,000 | 0.673 | 0.735 | 0.693 | 0.711 | 0.709 | 0.748 | 0.738 |
| 80,000 | 0.758 | 0.734 | 0.725 | 0.680 | 0.711 | 0.677 | 0.729 |
| 90,000 | 0.729 | 0.701 | 0.660 | 0.736 | 0.693 | 0.712 | 0.724 |
| 100,000 | 0.683 | 0.710 | 0.729 | 0.752 | 0.725 | 0.703 | 0.685 |

## Head-to-head: `armTn` vs `t_goal` at k=16

The controlled comparison — same 4/4/256 trunk, same 30 demos, same seed 42, same 50 test episodes, same argmax rule. Only the verifier differs.

| step | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 | mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | -0.02 | +0.04 | -0.14 | -0.10 | -0.16 | -0.14 | -0.14 | **-0.094** |
| 20,000 | -0.06 | +0.08 | -0.02 | -0.20 | -0.10 | -0.04 | -0.10 | **-0.063** |
| 30,000 | -0.02 | -0.06 | -0.02 | -0.08 | -0.08 | -0.12 | +0.02 | **-0.051** |
| 40,000 | -0.24 | +0.04 | -0.10 | -0.12 | -0.12 | -0.22 | -0.02 | **-0.111** |
| 50,000 | -0.06 | -0.02 | +0.04 | -0.06 | -0.08 | -0.10 | -0.08 | **-0.051** |
| 60,000 | -0.06 | -0.04 | -0.02 | -0.02 | +0.00 | -0.18 | +0.00 | **-0.046** |
| 70,000 | -0.06 | +0.08 | +0.02 | +0.00 | +0.02 | -0.12 | -0.02 | **-0.011** |
| 80,000 | -0.02 | +0.08 | +0.08 | +0.00 | +0.04 | -0.04 | +0.08 | **+0.031** |
| 90,000 | -0.10 | +0.10 | +0.14 | -0.08 | -0.06 | +0.08 | +0.08 | **+0.023** |
| 100,000 | -0.02 | -0.12 | -0.06 | +0.02 | +0.12 | -0.06 | +0.02 | **-0.014** |

_Cells are `armTn − t_goal` success rate. Across all 70 shared points: mean **-0.039**, armTn ahead at 19, behind at 47. At 50 episodes a single cell carries a 95% CI of about +/-0.13, so read the mean column, not one cell._

## Coverage

| arm | demos | checkpoints | fully swept | pending |
|---|---:|---:|---:|---|
| ST-diffusion k=16 (4/4/256, 17.1M) | 30 | 10/10 | 10 | — |
| ST-diffusion k=1 (4/4/256, 17.1M) | 30 | 10/10 | 10 | — |
| UNet BC (293.4M) | 30 | 10/10 | 10 | — |
| ST-diffusion k=1 (4/4/256) — armTn | 30 | 10/10 | 10 | — |
| ST-diffusion k=16 (4/4/256) — armTn | 30 | 10/10 | 10 | — |
| UNet BC (293.4M) — armTn, re-ranked | 30 | 10/10 | 10 | — |

_No checkpoint or n is nominated as best: every evaluated cell is printed and selection is never done on the test split._
