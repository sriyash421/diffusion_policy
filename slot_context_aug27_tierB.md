# Slot-context analysis — tier B

Does candidate k improve with the context of candidates 0..k-1? Measured at n=16 over the 50 test episodes, with the verifier value each dump was scored under.

Each arm at **its own** peak `final_pass` success at n=16 (later checkpoint breaking ties), as of 2026-08-27. Steps DIFFER PER ARM, so this tier is a within-arm check only — any cross-arm reading here is confounded with training duration. It is also mildly circular: `final_pass` at n=16 deploys slot 15, so the checkpoint was selected on a metric downstream of the quantity being measured. Tier A is the un-selected comparison.

## Arms

| arm | step | slot semantics | verifier | live steps | blind | success |
|---|---:|---|---|---:|---:|---:|
| curr-lin100-l2 | 60000 | `trained_staircase` | armTn | 1592 | 0.0% | 0.36 |
| curr-lin100-l2tol1 | 60000 | `trained_staircase` | armTn | 1678 | 0.0% | 0.26 |
| geo735-l2 | 40000 | `trained_staircase` | armTn | 1707 | 0.0% | 0.28 |
| geo735-l2tol1 | 50000 | `trained_staircase` | armTn | 1743 | 0.0% | 0.16 |
| lin100-l2 | 60000 | `trained_staircase` | armTn | 1694 | 0.0% | 0.26 |
| lin100-l2tol1 | 30000 | `trained_staircase` | armTn | 1866 | 0.0% | 0.06 |
| stk1 | 60000 | `iid_no_context` | armTn | 1701 | 0.0% | 0.28 |
| unetbc | 60000 | `iid_unbounded` | armTn | 1683 | 0.0% | 0.28 |

`iid_no_context` (ST k=1) and `iid_unbounded` (UNet BC) generate their 16 candidates independently **by construction**. They are not baselines of the statistic — they are its null distribution, measured on real data with the real tie structure. If a k=16 arm does not beat them, the context bought nothing.

## Q2 — which slot most often gets the highest value

Scoped to **discriminating** steps (some candidate differs) and read under the `unique` convention (a single maximizer), against a **permutation null** that reshuffles each step's own value multiset and so holds its ties fixed. The `first` column is what an argmax policy actually executed, which is a fact about the tiebreak rule as much as about the policy: on a fully degenerate step every slot is a maximizer and `np.argmax` returns 0.

| arm | mean slot (unique) | perm null | first | tied steps | unique steps |
|---|---:|---:|---:|---:|---:|
| curr-lin100-l2 | 4.17 | 7.50 | 4.17 | 0.0% | 1592 |
| curr-lin100-l2tol1 | 3.79 | 7.49 | 3.79 | 0.0% | 1678 |
| geo735-l2 | 4.36 | 7.50 | 4.36 | 0.0% | 1707 |
| geo735-l2tol1 | 4.38 | 7.51 | 4.38 | 0.0% | 1743 |
| lin100-l2 | 4.17 | 7.50 | 4.17 | 0.0% | 1694 |
| lin100-l2tol1 | 3.51 | 7.50 | 3.51 | 0.0% | 1866 |
| stk1 | 7.59 | 7.50 | 7.59 | 0.0% | 1701 |
| unetbc | 7.59 | 7.50 | 7.59 | 0.0% | 1683 |

A mean slot above the permutation null is a late-slot advantage; at or below it, slot order carries no information the value can see.

## Q3 — is slot 15 > slot 7 > slot 0?

Paired **within control step**. `delta` is the mean paired difference over all live steps; `win` is the fraction of **discriminating** steps the later slot wins. Intervals are a cluster bootstrap over the 50 episodes (4000 draws), because control steps within an episode are autocorrelated. `+` = interval excludes zero (or 0.5) on the better side, `-` = on the worse side, `.` = indistinguishable.

### On the verifier value

| arm | 0→7 delta | 0→7 win | 7→15 delta | 7→15 win | 0→15 delta | 0→15 win |
|---|---|---|---|---|---|---|
| curr-lin100-l2 | -0.019 [-0.072, +0.023] . | 0.477 [0.43,0.52] . | -0.004 [-0.009, +0.001] . | 0.467 [0.45,0.49] - | -0.023 [-0.072, +0.021] . | 0.478 [0.44,0.52] . |
| curr-lin100-l2tol1 | -0.030 [-0.106, +0.040] . | 0.482 [0.43,0.54] . | -0.000 [-0.009, +0.008] . | 0.477 [0.45,0.50] . | -0.030 [-0.110, +0.039] . | 0.480 [0.42,0.54] . |
| geo735-l2 | +0.145 [+0.080, +0.206] + | 0.562 [0.52,0.60] + | -0.025 [-0.039, -0.012] - | 0.479 [0.45,0.51] . | +0.119 [+0.055, +0.183] + | 0.553 [0.51,0.59] + |
| geo735-l2tol1 | +0.068 [-0.049, +0.177] . | 0.542 [0.48,0.60] . | -0.002 [-0.011, +0.007] . | 0.523 [0.50,0.55] + | +0.066 [-0.058, +0.174] . | 0.554 [0.49,0.61] . |
| lin100-l2 | +0.096 [+0.044, +0.145] + | 0.550 [0.50,0.60] + | -0.000 [-0.013, +0.019] . | 0.472 [0.45,0.50] . | +0.096 [+0.037, +0.154] + | 0.550 [0.50,0.60] + |
| lin100-l2tol1 | -0.230 [-0.379, -0.090] - | 0.396 [0.34,0.46] - | -0.018 [-0.031, -0.008] - | 0.465 [0.44,0.49] - | -0.248 [-0.403, -0.110] - | 0.394 [0.34,0.45] - |
| stk1 | -0.009 [-0.017, +0.000] . | 0.477 [0.45,0.50] . | +0.017 [+0.006, +0.032] + | 0.539 [0.52,0.56] + | +0.009 [-0.002, +0.021] . | 0.509 [0.49,0.53] . |
| unetbc | -0.009 [-0.020, +0.001] . | 0.500 [0.48,0.52] . | +0.007 [-0.004, +0.017] . | 0.513 [0.49,0.54] . | -0.002 [-0.014, +0.008] . | 0.510 [0.49,0.53] . |

### Decomposed into the two raw-pixel distances

`armTn` = `-(d_t_goal/13.6 + d_arm_t/52.1)`, so the value is a spread-normalized composite of two different distances. **A gain that lives entirely in `d_arm_t` is not task progress** — it is the arm parking closer to the T, the exact behaviour that retired the raw `armT` value. Both terms are reported as pixels REDUCED, so positive is always better.

| arm | term | blind | 0→15 Δpx | 0→15 win | 7→15 Δpx |
|---|---|---:|---|---|---|
| curr-lin100-l2 | `d_t_goal` | 21.5% | -0.18 [-0.78, +0.36] . | 0.478 [0.43,0.53] . | -0.03 [-0.09, +0.03] . |
| curr-lin100-l2 | `d_arm_t` | 0.0% | -0.54 [-1.51, +0.47] . | 0.491 [0.45,0.53] . | -0.10 [-0.22, +0.02] . |
| curr-lin100-l2tol1 | `d_t_goal` | 18.5% | +0.16 [-1.20, +1.30] . | 0.519 [0.45,0.59] . | +0.00 [-0.12, +0.11] . |
| curr-lin100-l2tol1 | `d_arm_t` | 0.0% | -2.19 [-3.97, -0.06] - | 0.454 [0.40,0.51] . | -0.01 [-0.14, +0.10] . |
| geo735-l2 | `d_t_goal` | 10.5% | +1.60 [+0.66, +2.50] + | 0.532 [0.47,0.59] . | -0.29 [-0.52, -0.09] - |
| geo735-l2 | `d_arm_t` | 0.0% | +0.07 [-1.98, +2.23] . | 0.529 [0.48,0.58] . | -0.19 [-0.41, +0.02] . |
| geo735-l2tol1 | `d_t_goal` | 11.7% | +0.61 [-1.10, +2.13] . | 0.544 [0.47,0.62] . | -0.05 [-0.16, +0.05] . |
| geo735-l2tol1 | `d_arm_t` | 0.0% | +1.10 [-0.79, +3.13] . | 0.586 [0.54,0.63] + | +0.09 [-0.08, +0.27] . |
| lin100-l2 | `d_t_goal` | 19.4% | +1.02 [+0.39, +1.65] + | 0.521 [0.46,0.58] . | +0.01 [-0.15, +0.23] . |
| lin100-l2 | `d_arm_t` | 0.0% | +1.07 [-0.94, +3.29] . | 0.535 [0.48,0.59] . | -0.05 [-0.24, +0.19] . |
| lin100-l2tol1 | `d_t_goal` | 12.1% | -2.20 [-4.54, -0.14] - | 0.433 [0.36,0.50] - | -0.23 [-0.39, -0.09] - |
| lin100-l2tol1 | `d_arm_t` | 0.0% | -4.48 [-7.61, -0.89] - | 0.463 [0.40,0.52] . | -0.08 [-0.28, +0.13] . |
| stk1 | `d_t_goal` | 24.6% | +0.10 [-0.04, +0.26] . | 0.476 [0.44,0.51] . | +0.16 [+0.02, +0.34] + |
| stk1 | `d_arm_t` | 0.0% | +0.08 [-0.10, +0.25] . | 0.494 [0.47,0.51] . | +0.29 [+0.09, +0.49] + |
| unetbc | `d_t_goal` | 26.0% | -0.03 [-0.17, +0.10] . | 0.449 [0.43,0.47] - | +0.06 [-0.07, +0.19] . |
| unetbc | `d_arm_t` | 0.0% | -0.01 [-0.24, +0.22] . | 0.510 [0.49,0.53] . | +0.12 [-0.15, +0.38] . |

The `blind` column is per term and is the point of the split: a step where no candidate can move the block has zero spread in `d_t_goal` while `d_arm_t` still varies freely. On those steps the ranking is decided by arm reach alone.

## Q4 — slot-to-slot delta, and where it stops improving

Values centered within each control step, then averaged; cluster-bootstrap CI over episodes. `saturation` is the largest k whose remaining gain (slot 15 − slot k) still has an interval excluding zero — i.e. improvement is no longer detectable past it. `None` means no slot showed a detectable remaining gain.

| arm | slot 0 | slot 7 | slot 15 | saturation | sd@0 | sd@15 |
|---|---|---|---|---:|---:|---:|
| curr-lin100-l2 | +0.017 [-0.022, +0.064] | -0.002 [-0.008, +0.003] | -0.006 [-0.012, -0.001] | – | 0.376 | 0.087 |
| curr-lin100-l2tol1 | +0.023 [-0.039, +0.091] | -0.007 [-0.016, +0.001] | -0.007 [-0.020, +0.004] | 15 | 0.528 | 0.111 |
| geo735-l2 | -0.129 [-0.186, -0.070] | +0.016 [+0.005, +0.027] | -0.009 [-0.024, +0.004] | 1 | 0.612 | 0.160 |
| geo735-l2tol1 | -0.069 [-0.160, +0.039] | -0.001 [-0.015, +0.011] | -0.003 [-0.021, +0.014] | – | 0.675 | 0.147 |
| lin100-l2 | -0.092 [-0.140, -0.044] | +0.004 [-0.007, +0.014] | +0.004 [-0.008, +0.017] | 1 | 0.536 | 0.182 |
| lin100-l2tol1 | +0.213 [+0.088, +0.348] | -0.017 [-0.031, -0.003] | -0.035 [-0.054, -0.018] | – | 0.826 | 0.217 |
| stk1 | -0.005 [-0.018, +0.005] | -0.014 [-0.030, -0.003] | +0.004 [-0.002, +0.010] | 8 | 0.166 | 0.167 |
| unetbc | +0.006 [-0.002, +0.013] | -0.003 [-0.011, +0.004] | +0.003 [-0.006, +0.012] | – | 0.161 | 0.162 |

**`sd@15` below `sd@0` means the context is narrowing exploration, not improving quality.** That produces a rising mean too, so the two must be read together: a later slot that is better *and* less varied has converged onto the region the earlier candidates already found, which is a different claim from having found something better.

> A guard on the plots: `prefix_max` (best-of-k) rises with k even under a pure i.i.d. resampler, so it is never evidence of context. It is plotted only against `prefix_mean`; the gap between them is the value of having a selector.

## Files

- `per_prediction_B.csv` — every prediction, every slot 0/7/15, argmax slot
- `slot_profile_B.png` — centered value vs slot, all arms
- `slot_profile_terms_B.png` — the same split into `d_t_goal` and `d_arm_t`
- `slot_stats_B.json` — every number above, machine-readable
