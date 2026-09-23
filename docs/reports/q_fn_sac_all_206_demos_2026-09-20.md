# `q_fn_sac_all_206_demos` — the Q as verifier on the geometric splits — 2026-09-20

**Tag:** `q_fn_sac_all_206_demos`. Every run on this page scores with **`q_sac_all`** =
`sac_keypoint/model_8500000_steps.zip`, registered in `pusht_verifier.Q_VERIFIERS`. That run
finished 10M with peak `eval/success_rate` 1.00 at 8.51M; the save grid is 100k, so 8.5M is the
checkpoint the name resolves to.

⚠️ **NOTHING ON THIS PAGE IS HELD OUT FROM THE Q.** `params/args.yaml` records
`demo_episodes: all`, so all 206 episodes seeded its replay buffer — including all 50 test
episodes of **both** splits. `sac/score.py::assert_held_out` refuses this; every sweep here ran
under `--allow-contaminated`, and each `bon_curves.json` records
`held_out=False, overlap=50/50, allowed=True`. For a Q that holds its test episodes out by
construction see [sac_per_split_q_2026-09-20.md](sac_per_split_q_2026-09-20.md).

Two separate questions, and they must not be conflated:

1. **§1 — can the Q RANK?** `install_q_ranker` swaps only the ranking scalar and leaves the
   `t_goal`-shaped context alone. Measured on UNet BC, which ignores the search context
   entirely and therefore has no train/eval mismatch to confound the answer. **Complete.**
2. **§2 — does a Q-shaped CONTEXT teach the policy anything?** 12 ST arms trained with the Q as
   their verifier from step 0 (`README_pusht.md` §2.6). A different question: the verifier's only
   route into training is the context. **Training complete, sweep in progress.**

---

## 1. Best-of-N under `q_sac_all` vs `t_goal` — UNet BC, 50 test episodes

`sac/eval.py bon-sweep --rankers t_goal,q_sac_all --skip-context-sim`, launched by
`scripts/slurm/submit_q_bon_geometric.sh`. Both rankers run in ONE job on ONE GPU with the same
per-episode noise, so the pair is exact; cross-arm cells are not (two jobs on different GPU
models differ by 1-2 episodes of 50 — see `scripts/slurm/q_arms.sh`).

| arm | ranker | n=1 | n=2 | n=4 | n=8 | n=16 | n1→n16 |
|---|---|--:|--:|--:|--:|--:|--:|
| unetbc blq137 @100k | `t_goal` | 0.420 | 0.560 | **0.820** | 0.760 | 0.760 | **+0.340** |
| | `q_sac_all` | 0.420 | 0.480 | 0.480 | 0.320 | 0.420 | 0.000 |
| unetbc blq137 @30k | `t_goal` | 0.240 | 0.420 | 0.840 | 0.820 | **0.920** | **+0.680** |
| | `q_sac_all` | 0.240 | 0.340 | 0.280 | 0.360 | 0.180 | −0.060 |
| unetbc brd100 @100k | `t_goal` | 0.620 | 0.660 | 0.680 | 0.680 | **0.740** | **+0.120** |
| | `q_sac_all` | 0.620 | 0.560 | 0.500 | 0.580 | 0.560 | −0.060 |
| unetbc brd100 @30k | `t_goal` | 0.360 | 0.500 | 0.780 | 0.800 | **0.880** | **+0.520** |
| | `q_sac_all` | 0.360 | 0.320 | 0.420 | 0.440 | 0.320 | −0.040 |

**`t_goal` rises with n on all four rows (+0.12…+0.68). `q_sac_all` rises on none of them** —
three of four end below their own n=1. 50 episodes puts the 95% Wilson interval near ±0.13, so
individual cells overlap; the 4-of-4 consistency and the paired-by-episode design carry it.

**The leak check passes 4/4.** At n=1 there is nothing to rank, so the two rankers must agree
exactly, and they do (0.420 / 0.240 / 0.620 / 0.360). That is what makes the rest meaningful.

**Q is not inverted — it is uninformative.** V collapsed to 0.000–0.060 under the same sweep
(`ppo_sac_lstm-bc_eval_2026-09-17.md` §6); Q instead lands about where not searching landed.
This contradicts that report's stated expectation — "Q is unaffected by this argument … fitted
with a Bellman max backup to terminal success, so it has no reason to invert". It does not
invert. It does not rank either.

**This is the optimistic case, twice over**, which is what makes the negative result solid:
BC ignores the search context so there is no distribution shift to blame, and the Q has seen
every episode it is being scored on.

`q_verifier_2026-09-19.md` reaches the same conclusion from the other side — Q ranks the expert
action 4–9× better than `t_goal` where `t_goal` is blind, and still loses every best-of-N row.
Ranking `a*` well is necessary, not sufficient.

## 2. The 12 Q-trained ST arms — training complete, sweep in progress

2 splits × 2 widths × 3 obs-noise ladders, all `verifier_tag=q_sac_all`, `trainer: outer_inner`,
100k steps. Launched by `scripts/run_q_geometric.sh`; mechanism in `README_pusht.md` §2.6.

| | blq137 (137 train / 50 test) | brd100 (100 train / 50 test) |
|---|---|---|
| k=16 | uniform ✅ · flat400 ✅ · ramp400to200 ✅ | uniform ✅ · flat400 ✅ · ramp400to200 *(training)* |
| k=4 | uniform ✅ · flat400 ✅ · ramp400to200 ✅ | uniform ✅ · flat400 ✅ · ramp400to200 ✅ |

No arm failed. The Q verifier forks no sim pool, so these ran markedly faster than their
`t_goal` counterparts — k=4 at ~450 steps/min against k=16 at ~150.

**Readout: all 12 arms complete** (`scripts/slurm/submit_q_st_sweep.sh`; collect with
`python scripts/q_st_sweep_status.py --step 100000 [--metric cov_max]`). Every 10k checkpoint ×
`--selection {argmax, final_pass}` × `n ∈ {1,2,4,8,16}`, each arm on its own 50-episode test
split.

**NO `--q` FLAG, AND NONE IS POSSIBLE.** These arms carry `verifier_tag=q_sac_all` in their own
config, so the eval rebuilds the same `PushTQVerifier` and `_normalize_q` reproduces the running
z-score from the checkpointed buffers — the eval prints `verifier value: q_sac_all (from
checkpoint cfg)`. **This is what the 12 arms exist for**: unlike `install_q_ranker` on a `t_goal`
arm (§1), there is no train/eval mismatch to confound the reading.

`argmax` is the deployed rule. **`final_pass` is the verifier switched OFF** — candidate n−1
executed as-is, having been generated conditioned on the other n−1 — so the pair separates
*search changed the SAMPLER* from *search changed the SELECTOR*. `_bon_subdir` gives each its own
directory (`bon_search_sel-<sel>/`; the `_ver-` suffix appears only under an explicit override,
which these runs do not use).

### Held-out check — verified, not assumed

Every number on this page was scored on the **50 test episodes of the arm's own manifest**,
with zero overlap against what that run trained on. Checked by reading the `episode_idxs`
recorded in each curve against both the committed manifest and the run's own `splits.json`
(written at training time), rather than trusting that `get_split_states` was passed the right
thing:

| | blq137 | brd100 |
|---|---|---|
| manifest | 137 train / 19 val / **50 test** | 100 train / 0 val / **50 test** |
| episodes scored | 50, == manifest test set | 50, == manifest test set |
| overlap with the run's train split | **0** | **0** |
| violations across 24 ST (arm, selection) pairs | **0** | **0** |
| violations across the 4 UNet BC sweeps | **0** | **0** |

⚠️ **That is held out from the POLICY, not from the Q.** `q_sac_all` was seeded with
`demo_episodes: all`, so all 100 of these test episodes are in its replay buffer
(`held_out=False, overlap=50/50` in every `bon_curves.json`). The policy never saw them; the
verifier saw all of them. For a Q that holds them out by construction, see
[sac_per_split_q_2026-09-20.md](sac_per_split_q_2026-09-20.md).

### Success rate at step 100k, n = 1…64 — every arm, both selection rules

Built by `python scripts/q_n64_table.py [--metric cov_max]`. **UNet BC carries
`verifier_tag=t_goal`**, so the Q must be installed over it and both rankers are shown on the
same episodes — `[t_goal]` is the reference the `[q_sac_all]` row is read against. **The ST arms
carry `verifier_tag=q_sac_all` natively**, so there is one ranker per row and no ranker column.

<!-- Q_N64_TABLE_START -->

#### Success rate (coverage > 0.95) — step 100,000, n = 1…64

| arm | sel | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| unetbc blq137 [q_sac_all] | `argmax` | 0.420 | 0.480 | 0.480 | 0.320 | 0.420 | 0.400 | 0.280 |
| unetbc blq137 [q_sac_all] | `final_pass` | 0.420 | 0.400 | 0.420 | 0.400 | 0.400 | 0.500 | 0.380 |
| unetbc blq137 [t_goal] | `argmax` | 0.420 | 0.560 | 0.820 | 0.760 | 0.760 | 0.800 | 0.880 |
| unetbc blq137 [t_goal] | `final_pass` | 0.420 | 0.400 | 0.420 | 0.400 | 0.400 | 0.500 | 0.380 |
| unetbc brd100 [q_sac_all] | `argmax` | 0.620 | 0.560 | 0.500 | 0.580 | 0.560 | 0.520 | 0.600 |
| unetbc brd100 [q_sac_all] | `final_pass` | 0.620 | 0.420 | 0.560 | 0.520 | 0.360 | 0.360 | 0.420 |
| unetbc brd100 [t_goal] | `argmax` | 0.620 | 0.660 | 0.680 | 0.680 | 0.740 | 0.840 | 0.860 |
| unetbc brd100 [t_goal] | `final_pass` | 0.620 | 0.420 | 0.560 | 0.520 | 0.360 | 0.360 | 0.420 |
| k16 blq137 | `argmax` | 0.240 | 0.300 | 0.220 | 0.140 | 0.240 | 0.200 | 0.240 |
| k16 blq137 | `final_pass` | 0.240 | 0.220 | 0.200 | 0.200 | 0.200 | 0.280 | 0.260 |
| k16 brd100 | `argmax` | 0.380 | 0.360 | 0.480 | 0.460 | 0.360 | 0.440 | 0.520 |
| k16 brd100 | `final_pass` | 0.360 | 0.240 | 0.320 | 0.480 | 0.280 | 0.400 | 0.400 |
| k16 flat400 blq137 | `argmax` | 0.060 | 0.080 | 0.160 | 0.020 | 0.120 | 0.180 | 0.100 |
| k16 flat400 blq137 | `final_pass` | 0.080 | 0.160 | 0.160 | 0.160 | 0.180 | 0.060 | 0.220 |
| k16 flat400 brd100 | `argmax` | 0.180 | 0.340 | 0.240 | 0.160 | 0.220 | 0.280 | 0.220 |
| k16 flat400 brd100 | `final_pass` | 0.160 | 0.280 | 0.240 | 0.240 | 0.180 | 0.180 | 0.260 |
| k16 ramp400to200 blq137 | `argmax` | 0.000 | 0.160 | 0.180 | 0.120 | 0.160 | 0.200 | 0.200 |
| k16 ramp400to200 blq137 | `final_pass` | 0.000 | 0.160 | 0.180 | 0.160 | 0.300 | 0.320 | 0.260 |
| k16 ramp400to200 brd100 | `argmax` | 0.180 | 0.220 | 0.320 | 0.260 | 0.300 | 0.400 | 0.280 |
| k16 ramp400to200 brd100 | `final_pass` | 0.180 | 0.240 | 0.300 | 0.440 | 0.280 | 0.320 | 0.260 |
| k4 blq137 | `argmax` | 0.100 | 0.120 | 0.200 | 0.160 | 0.200 | 0.220 | 0.280 |
| k4 blq137 | `final_pass` | 0.100 | 0.160 | 0.140 | 0.260 | 0.260 | 0.200 | 0.180 |
| k4 brd100 | `argmax` | 0.280 | 0.280 | 0.240 | 0.220 | 0.220 | 0.340 | 0.400 |
| k4 brd100 | `final_pass` | 0.280 | 0.300 | 0.320 | 0.340 | 0.220 | 0.320 | 0.240 |
| k4 flat400 blq137 | `argmax` | 0.140 | 0.120 | 0.080 | 0.060 | 0.040 | 0.040 | 0.020 |
| k4 flat400 blq137 | `final_pass` | 0.140 | 0.100 | 0.080 | 0.120 | 0.160 | 0.020 | 0.120 |
| k4 flat400 brd100 | `argmax` | 0.240 | 0.220 | 0.140 | 0.160 | 0.160 | 0.200 | 0.160 |
| k4 flat400 brd100 | `final_pass` | 0.240 | 0.140 | 0.120 | 0.080 | 0.100 | 0.120 | 0.240 |
| k4 ramp400to200 blq137 | `argmax` | 0.120 | 0.060 | 0.160 | 0.160 | 0.060 | 0.080 | 0.180 |
| k4 ramp400to200 blq137 | `final_pass` | 0.120 | 0.140 | 0.240 | 0.260 | 0.220 | 0.160 | 0.200 |
| k4 ramp400to200 brd100 | `argmax` | 0.180 | 0.140 | 0.140 | 0.200 | 0.240 | 0.200 | 0.160 |
| k4 ramp400to200 brd100 | `final_pass` | 0.180 | 0.120 | 0.140 | 0.300 | 0.100 | 0.080 | 0.180 |

<!-- 32 rows; full n grid -->


#### Mean MAX goal coverage, RAW (uncensored) — step 100,000, n = 1…64

*Raw goal coverage, uncensored. `mean_reward` saturates at 1.0 from coverage
0.95 up, so it cannot separate two arms among their solved episodes; this can.
A row of em dashes means that curve predates the 2026-09-21 change that began
persisting coverage and needs a re-run.*

| arm | sel | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| unetbc blq137 [q_sac_all] | `argmax` | 0.881 | 0.869 | 0.850 | 0.844 | 0.854 | 0.800 | 0.759 |
| unetbc blq137 [q_sac_all] | `final_pass` | 0.881 | 0.854 | 0.835 | 0.867 | 0.799 | 0.797 | 0.860 |
| unetbc blq137 [t_goal] | `argmax` | 0.881 | 0.917 | 0.938 | 0.875 | 0.930 | 0.935 | 0.937 |
| unetbc blq137 [t_goal] | `final_pass` | 0.881 | 0.854 | 0.835 | 0.867 | 0.799 | 0.797 | 0.860 |
| unetbc brd100 [q_sac_all] | `argmax` | 0.924 | 0.918 | 0.911 | 0.929 | 0.921 | 0.881 | 0.890 |
| unetbc brd100 [q_sac_all] | `final_pass` | 0.924 | 0.914 | 0.910 | 0.889 | 0.900 | 0.890 | 0.900 |
| unetbc brd100 [t_goal] | `argmax` | 0.924 | 0.919 | 0.914 | 0.935 | 0.928 | 0.927 | 0.944 |
| unetbc brd100 [t_goal] | `final_pass` | 0.924 | 0.914 | 0.910 | 0.889 | 0.900 | 0.890 | 0.900 |
| k16 blq137 | `argmax` | 0.644 | 0.651 | 0.667 | 0.678 | 0.612 | 0.610 | 0.606 |
| k16 blq137 | `final_pass` | 0.644 | 0.618 | 0.640 | 0.631 | 0.666 | 0.650 | 0.599 |
| k16 brd100 | `argmax` | 0.796 | 0.794 | 0.836 | 0.818 | 0.834 | 0.848 | 0.847 |
| k16 brd100 | `final_pass` | 0.796 | 0.798 | 0.814 | 0.837 | 0.757 | 0.828 | 0.808 |
| k16 flat400 blq137 | `argmax` | 0.565 | 0.519 | 0.524 | 0.437 | 0.498 | 0.479 | 0.469 |
| k16 flat400 blq137 | `final_pass` | 0.566 | 0.595 | 0.639 | 0.660 | 0.640 | 0.602 | 0.668 |
| k16 flat400 brd100 | `argmax` | 0.601 | 0.669 | 0.614 | 0.617 | 0.638 | 0.638 | 0.626 |
| k16 flat400 brd100 | `final_pass` | 0.598 | 0.678 | 0.657 | 0.683 | 0.696 | 0.706 | 0.645 |
| k16 ramp400to200 blq137 | `argmax` | 0.508 | 0.558 | 0.538 | 0.552 | 0.594 | 0.614 | 0.613 |
| k16 ramp400to200 blq137 | `final_pass` | 0.500 | 0.601 | 0.659 | 0.661 | 0.753 | 0.782 | 0.729 |
| k16 ramp400to200 brd100 | `argmax` | 0.722 | 0.762 | 0.742 | 0.737 | 0.801 | 0.816 | 0.827 |
| k16 ramp400to200 brd100 | `final_pass` | 0.722 | 0.731 | 0.782 | 0.780 | 0.773 | 0.812 | 0.774 |
| k4 blq137 | `argmax` | 0.644 | 0.689 | 0.650 | 0.643 | 0.727 | 0.690 | 0.651 |
| k4 blq137 | `final_pass` | 0.644 | 0.745 | 0.666 | 0.675 | 0.697 | 0.677 | 0.690 |
| k4 brd100 | `argmax` | 0.767 | 0.748 | 0.752 | 0.772 | 0.806 | 0.791 | 0.769 |
| k4 brd100 | `final_pass` | 0.767 | 0.782 | 0.816 | 0.792 | 0.759 | 0.800 | 0.763 |
| k4 flat400 blq137 | `argmax` | 0.555 | 0.548 | 0.578 | 0.533 | 0.504 | 0.515 | 0.485 |
| k4 flat400 blq137 | `final_pass` | 0.555 | 0.578 | 0.508 | 0.609 | 0.588 | 0.563 | 0.625 |
| k4 flat400 brd100 | `argmax` | 0.661 | 0.668 | 0.664 | 0.667 | 0.705 | 0.662 | 0.697 |
| k4 flat400 brd100 | `final_pass` | 0.661 | 0.656 | 0.655 | 0.613 | 0.710 | 0.672 | 0.710 |
| k4 ramp400to200 blq137 | `argmax` | 0.617 | 0.582 | 0.621 | 0.570 | 0.577 | 0.576 | 0.583 |
| k4 ramp400to200 blq137 | `final_pass` | 0.617 | 0.620 | 0.682 | 0.760 | 0.733 | 0.716 | 0.686 |
| k4 ramp400to200 brd100 | `argmax` | 0.657 | 0.638 | 0.693 | 0.663 | 0.720 | 0.734 | 0.691 |
| k4 ramp400to200 brd100 | `final_pass` | 0.657 | 0.708 | 0.715 | 0.765 | 0.701 | 0.727 | 0.783 |

<!-- 32 rows; full n grid -->


#### Mean MAX reward  = clip(coverage/0.95, 0, 1), censored at 1.0 — step 100,000, n = 1…64

| arm | sel | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| unetbc blq137 [q_sac_all] | `argmax` | 0.921 | 0.910 | 0.890 | 0.886 | 0.894 | 0.837 | 0.796 |
| unetbc blq137 [q_sac_all] | `final_pass` | 0.921 | 0.895 | 0.875 | 0.909 | 0.836 | 0.835 | 0.900 |
| unetbc blq137 [t_goal] | `argmax` | 0.921 | 0.960 | 0.976 | 0.912 | 0.972 | 0.972 | 0.974 |
| unetbc blq137 [t_goal] | `final_pass` | 0.921 | 0.895 | 0.875 | 0.909 | 0.836 | 0.835 | 0.900 |
| unetbc brd100 [q_sac_all] | `argmax` | 0.966 | 0.959 | 0.953 | 0.972 | 0.963 | 0.920 | 0.930 |
| unetbc brd100 [q_sac_all] | `final_pass` | 0.966 | 0.958 | 0.953 | 0.931 | 0.943 | 0.934 | 0.942 |
| unetbc brd100 [t_goal] | `argmax` | 0.966 | 0.958 | 0.956 | 0.977 | 0.965 | 0.962 | 0.980 |
| unetbc brd100 [t_goal] | `final_pass` | 0.966 | 0.958 | 0.953 | 0.931 | 0.943 | 0.934 | 0.942 |
| k16 blq137 | `argmax` | 0.676 | 0.682 | 0.700 | 0.711 | 0.642 | 0.639 | 0.635 |
| k16 blq137 | `final_pass` | 0.676 | 0.649 | 0.671 | 0.662 | 0.699 | 0.681 | 0.628 |
| k16 brd100 | `argmax` | 0.834 | 0.833 | 0.872 | 0.856 | 0.874 | 0.889 | 0.886 |
| k16 brd100 | `final_pass` | 0.834 | 0.837 | 0.854 | 0.874 | 0.793 | 0.866 | 0.846 |
| k16 flat400 blq137 | `argmax` | 0.594 | 0.545 | 0.550 | 0.460 | 0.523 | 0.502 | 0.492 |
| k16 flat400 blq137 | `final_pass` | 0.596 | 0.624 | 0.671 | 0.693 | 0.673 | 0.633 | 0.701 |
| k16 flat400 brd100 | `argmax` | 0.631 | 0.701 | 0.644 | 0.647 | 0.668 | 0.668 | 0.657 |
| k16 flat400 brd100 | `final_pass` | 0.629 | 0.711 | 0.690 | 0.717 | 0.730 | 0.741 | 0.676 |
| k16 ramp400to200 blq137 | `argmax` | 0.535 | 0.585 | 0.565 | 0.580 | 0.624 | 0.645 | 0.644 |
| k16 ramp400to200 blq137 | `final_pass` | 0.526 | 0.632 | 0.692 | 0.695 | 0.789 | 0.820 | 0.764 |
| k16 ramp400to200 brd100 | `argmax` | 0.758 | 0.800 | 0.777 | 0.773 | 0.839 | 0.854 | 0.867 |
| k16 ramp400to200 brd100 | `final_pass` | 0.758 | 0.768 | 0.821 | 0.816 | 0.811 | 0.851 | 0.813 |
| k4 blq137 | `argmax` | 0.677 | 0.724 | 0.682 | 0.675 | 0.764 | 0.724 | 0.682 |
| k4 blq137 | `final_pass` | 0.677 | 0.782 | 0.700 | 0.707 | 0.732 | 0.710 | 0.725 |
| k4 brd100 | `argmax` | 0.805 | 0.785 | 0.788 | 0.810 | 0.846 | 0.829 | 0.804 |
| k4 brd100 | `final_pass` | 0.805 | 0.821 | 0.855 | 0.830 | 0.797 | 0.839 | 0.802 |
| k4 flat400 blq137 | `argmax` | 0.583 | 0.575 | 0.607 | 0.560 | 0.530 | 0.541 | 0.510 |
| k4 flat400 blq137 | `final_pass` | 0.583 | 0.607 | 0.535 | 0.639 | 0.616 | 0.593 | 0.656 |
| k4 flat400 brd100 | `argmax` | 0.693 | 0.701 | 0.697 | 0.701 | 0.741 | 0.694 | 0.731 |
| k4 flat400 brd100 | `final_pass` | 0.693 | 0.690 | 0.689 | 0.644 | 0.747 | 0.705 | 0.745 |
| k4 ramp400to200 blq137 | `argmax` | 0.649 | 0.612 | 0.652 | 0.599 | 0.607 | 0.606 | 0.613 |
| k4 ramp400to200 blq137 | `final_pass` | 0.649 | 0.651 | 0.716 | 0.797 | 0.769 | 0.752 | 0.719 |
| k4 ramp400to200 brd100 | `argmax` | 0.690 | 0.669 | 0.728 | 0.696 | 0.756 | 0.770 | 0.725 |
| k4 ramp400to200 brd100 | `final_pass` | 0.690 | 0.743 | 0.751 | 0.802 | 0.737 | 0.764 | 0.823 |

<!-- 32 rows; full n grid -->

<!-- Q_N64_TABLE_END -->

### Success rate at step 100k, n=16

| arm | argmax | final_pass | | arm | argmax | final_pass |
|---|--:|--:|---|---|--:|--:|
| k16 blq137 uniform | **0.240** | 0.200 | | k4 blq137 uniform | 0.200 | **0.240** |
| k16 blq137 flat400 | 0.120 | **0.160** | | k4 blq137 flat400 | 0.040 | **0.160** |
| k16 blq137 ramp | 0.160 | **0.280** | | k4 blq137 ramp | 0.060 | **0.220** |
| k16 brd100 uniform | **0.360** | 0.280 | | k4 brd100 uniform | **0.260** | 0.240 |
| k16 brd100 flat400 | **0.220** | 0.200 | | k4 brd100 flat400 | **0.160** | 0.100 |
| k16 brd100 ramp | **0.300** | 0.260 | | k4 brd100 ramp | **0.240** | 0.100 |

**`argmax` wins 7, `final_pass` wins 5.** That is a coin flip, and it is the result: **selecting
by the Q is indistinguishable from not selecting at all.** The verifier the arms were trained
against contributes nothing at deployment — the same conclusion §1 reached on BC, now on the
arms built specifically to remove that confound. At n=1 the two rules are identical by
construction (nothing to select), and every row confirms it.

**Mean reward agrees and adds nothing to rescue it.** At n=16 it spans 0.523–0.876 and tracks
the success ordering; where `final_pass` wins on success it usually wins on reward too (k16
blq137 ramp 0.789 vs 0.625; k4 blq137 ramp 0.769 vs 0.606). So this is not a case of search
pushing the block most of the way and missing the threshold.

> ⚠️ **CORRECTED 2026-09-21: those are REWARDS, not coverages.** An earlier revision of this
> page called them "mean MAX coverage". `pusht_env.py:133` defines
> `reward = clip(coverage / 0.95, 0, 1)` and success is `coverage > 0.95`, so reward is an
> exact affine image of coverage *below* the threshold and **saturates at 1.0 above it** —
> right-censored precisely on the solved episodes. Below threshold `coverage = reward × 0.95`;
> at 1.0 it says only "solved". The numbers were always these numbers; the name was wrong.
>
> `eval_search_pusht` prints raw coverage and does not persist it, so **raw coverage for the ST
> arms would need a re-run**. `sac/eval.py bon-sweep` does persist it (`mean_max_coverage`),
> which is why the UNet BC rows below can show the uncensored quantity and the ST rows cannot.
> Read ST coverage off `scripts/q_n64_table.py --metric reward_max`, understanding the censoring.

**These arms are also weak in absolute terms.** Best success at n=16 is 0.360 (k16 brd100
uniform, argmax) against 0.740–0.920 for UNet BC under `t_goal` on the same splits (§1). The
uniform arms beat both ladder arms on every split and width — the reverse of what the
`t_goal`-trained ladder arms showed in
[tgoal_moving_arms_2026-09-19.md](tgoal_moving_arms_2026-09-19.md) §2.

⚠️ 50 episodes puts the 95% Wilson interval near ±0.13, so no single cell above is separable.
What carries the claim is the 7–5 split across all 12 arms and the n=1 identity check (the
two rules are identical by construction at n=1, and every row confirms it), not any single cell.

⚠️ **THE CONTROLS ARE PARTIAL.** Under `t_goal` only four matched twins exist at 100k —
blq137 k16 {uniform, flat400, ramp400to200} and brd100 k16 uniform. **Every k=4 comparison, and
the brd100 k16 ladders, have no twin**: those rows are across generations, not controlled, and
must be labelled unpaired wherever they are quoted.

## Reproduce

```bash
SUBMIT=1 bash scripts/run_q_geometric.sh              # the 12 arms
SUBMIT=1 bash scripts/slurm/submit_q_bon_geometric.sh # §1
SUBMIT=1 bash scripts/slurm/submit_q_st_sweep.sh      # §2, 22 watchers
python scripts/q_st_sweep_status.py --all-steps       # §2, the 10k grid
SUBMIT=1 bash scripts/slurm/submit_q_n64_extend.sh    # §2, extend to n=32,64
python scripts/q_n64_table.py                         # §2, the n=1..64 table
python scripts/measure_q_spread.py                    # why the context is a z-score, not raw Q
PYTHONPATH=$PWD python scripts/q_verifier_smoke.py    # the GPU-side contract checks
pytest unit_tests/test_q_context.py                   # the normalizer's arithmetic
```
