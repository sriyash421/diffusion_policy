# Slot-context analysis — tier A

Does candidate k improve with the context of candidates 0..k-1? Measured at n=16 over the 50 test episodes, with the verifier value each dump was scored under.

All arms at the **same** step (10000), so a cross-arm comparison of slot effects is not confounded with training duration. Chosen by no criterion.

## Arms

| arm | step | slot semantics | verifier | live steps | blind | success |
|---|---:|---|---|---:|---:|---:|
| curr-lin100-l2 | 10000 | `trained_staircase` | armTn | 1721 | 0.0% | 0.24 |
| curr-lin100-l2tol1 | 10000 | `trained_staircase` | armTn | 1654 | 0.0% | 0.34 |
| geo735-l2 | 10000 | `trained_staircase` | armTn | 1801 | 0.0% | 0.18 |
| geo735-l2tol1 | 10000 | `trained_staircase` | armTn | 1705 | 0.0% | 0.22 |
| lin100-l2 | 10000 | `trained_staircase` | armTn | 1739 | 0.0% | 0.20 |
| lin100-l2tol1 | 10000 | `trained_staircase` | armTn | 1867 | 0.0% | 0.04 |
| stk1 | 10000 | `iid_no_context` | armTn | 1889 | 0.0% | 0.06 |
| unetbc | 10000 | `iid_unbounded` | armTn | 1750 | 0.0% | 0.18 |

`iid_no_context` (ST k=1) and `iid_unbounded` (UNet BC) generate their 16 candidates independently **by construction**. They are not baselines of the statistic — they are its null distribution, measured on real data with the real tie structure. If a k=16 arm does not beat them, the context bought nothing.

## Q2 — which slot most often gets the highest value

Scoped to **discriminating** steps (some candidate differs) and read under the `unique` convention (a single maximizer), against a **permutation null** that reshuffles each step's own value multiset and so holds its ties fixed. The `first` column is what an argmax policy actually executed, which is a fact about the tiebreak rule as much as about the policy: on a fully degenerate step every slot is a maximizer and `np.argmax` returns 0.

| arm | mean slot (unique) | perm null | first | tied steps | unique steps |
|---|---:|---:|---:|---:|---:|
| curr-lin100-l2 | 6.15 | 7.50 | 6.15 | 0.0% | 1721 |
| curr-lin100-l2tol1 | 6.54 | 7.50 | 6.54 | 0.0% | 1654 |
| geo735-l2 | 5.18 | 7.50 | 5.18 | 0.0% | 1801 |
| geo735-l2tol1 | 4.68 | 7.52 | 4.68 | 0.0% | 1705 |
| lin100-l2 | 5.54 | 7.51 | 5.54 | 0.0% | 1739 |
| lin100-l2tol1 | 4.40 | 7.52 | 4.40 | 0.0% | 1867 |
| stk1 | 7.56 | 7.49 | 7.56 | 0.0% | 1889 |
| unetbc | 7.39 | 7.50 | 7.39 | 0.0% | 1750 |

A mean slot above the permutation null is a late-slot advantage; at or below it, slot order carries no information the value can see.

## Q3 — is slot 15 > slot 7 > slot 0?

Paired **within control step**. `delta` is the mean paired difference over all live steps; `win` is the fraction of **discriminating** steps the later slot wins. Intervals are a cluster bootstrap over the 50 episodes (4000 draws), because control steps within an episode are autocorrelated. `+` = interval excludes zero (or 0.5) on the better side, `-` = on the worse side, `.` = indistinguishable.

### On the verifier value

| arm | 0→7 delta | 0→7 win | 7→15 delta | 7→15 win | 0→15 delta | 0→15 win |
|---|---|---|---|---|---|---|
| curr-lin100-l2 | +0.082 [+0.030, +0.137] + | 0.540 [0.50,0.58] . | -0.009 [-0.030, +0.011] . | 0.503 [0.48,0.53] . | +0.074 [+0.022, +0.129] + | 0.544 [0.50,0.59] + |
| curr-lin100-l2tol1 | +0.135 [+0.078, +0.201] + | 0.580 [0.55,0.61] + | -0.002 [-0.020, +0.015] . | 0.502 [0.48,0.52] . | +0.133 [+0.079, +0.196] + | 0.568 [0.53,0.61] + |
| geo735-l2 | +0.047 [-0.089, +0.180] . | 0.530 [0.47,0.59] . | -0.059 [-0.092, -0.028] - | 0.455 [0.43,0.48] - | -0.012 [-0.166, +0.140] . | 0.495 [0.42,0.56] . |
| geo735-l2tol1 | -0.288 [-0.456, -0.131] - | 0.398 [0.34,0.46] - | -0.035 [-0.076, +0.003] . | 0.484 [0.45,0.51] . | -0.323 [-0.522, -0.144] - | 0.398 [0.34,0.46] - |
| lin100-l2 | +0.089 [-0.045, +0.212] . | 0.547 [0.49,0.60] . | +0.005 [-0.014, +0.023] . | 0.519 [0.50,0.54] . | +0.094 [-0.046, +0.221] . | 0.550 [0.49,0.60] . |
| lin100-l2tol1 | -0.290 [-0.466, -0.125] - | 0.397 [0.34,0.46] - | -0.045 [-0.074, -0.017] - | 0.476 [0.45,0.50] . | -0.335 [-0.522, -0.166] - | 0.396 [0.33,0.46] - |
| stk1 | +0.007 [-0.024, +0.040] . | 0.509 [0.49,0.53] . | +0.022 [-0.007, +0.051] . | 0.507 [0.49,0.53] . | +0.029 [-0.004, +0.062] . | 0.509 [0.49,0.53] . |
| unetbc | +0.003 [-0.021, +0.027] . | 0.517 [0.49,0.54] . | -0.003 [-0.034, +0.027] . | 0.505 [0.48,0.53] . | -0.000 [-0.030, +0.031] . | 0.504 [0.48,0.53] . |

### Decomposed into the two raw-pixel distances

`armTn` = `-(d_t_goal/13.6 + d_arm_t/52.1)`, so the value is a spread-normalized composite of two different distances. **A gain that lives entirely in `d_arm_t` is not task progress** — it is the arm parking closer to the T, the exact behaviour that retired the raw `armT` value. Both terms are reported as pixels REDUCED, so positive is always better.

| arm | term | blind | 0→15 Δpx | 0→15 win | 7→15 Δpx |
|---|---|---:|---|---|---|
| curr-lin100-l2 | `d_t_goal` | 12.7% | +1.19 [+0.44, +1.98] + | 0.527 [0.48,0.57] . | -0.20 [-0.45, +0.03] . |
| curr-lin100-l2 | `d_arm_t` | 0.0% | -0.74 [-1.82, +0.37] . | 0.494 [0.45,0.54] . | +0.30 [-0.05, +0.66] . |
| curr-lin100-l2tol1 | `d_t_goal` | 17.2% | +2.16 [+1.33, +3.13] + | 0.558 [0.51,0.61] + | -0.09 [-0.30, +0.12] . |
| curr-lin100-l2tol1 | `d_arm_t` | 0.0% | -1.34 [-2.40, -0.24] - | 0.483 [0.44,0.52] . | +0.24 [-0.09, +0.57] . |
| geo735-l2 | `d_t_goal` | 4.9% | -0.96 [-3.34, +1.34] . | 0.484 [0.40,0.56] . | -0.81 [-1.23, -0.42] - |
| geo735-l2 | `d_arm_t` | 0.0% | +3.05 [+1.06, +5.19] + | 0.585 [0.54,0.63] + | +0.03 [-0.44, +0.51] . |
| geo735-l2tol1 | `d_t_goal` | 5.0% | -5.07 [-8.22, -2.08] - | 0.383 [0.32,0.45] - | -0.54 [-1.10, -0.02] - |
| geo735-l2tol1 | `d_arm_t` | 0.0% | +2.61 [-1.20, +6.63] . | 0.578 [0.51,0.65] + | +0.24 [-0.33, +0.79] . |
| lin100-l2 | `d_t_goal` | 6.9% | +1.42 [-0.75, +3.37] . | 0.555 [0.49,0.62] . | -0.12 [-0.34, +0.10] . |
| lin100-l2 | `d_arm_t` | 0.0% | -0.55 [-2.51, +1.57] . | 0.533 [0.49,0.58] . | +0.71 [+0.33, +1.09] + |
| lin100-l2tol1 | `d_t_goal` | 5.8% | -5.01 [-7.92, -2.41] - | 0.387 [0.31,0.47] - | -0.69 [-1.02, -0.38] - |
| lin100-l2tol1 | `d_arm_t` | 0.0% | +1.76 [-2.46, +6.21] . | 0.570 [0.50,0.64] . | +0.30 [-0.22, +0.83] . |
| stk1 | `d_t_goal` | 2.6% | +0.39 [+0.01, +0.75] + | 0.459 [0.43,0.49] - | +0.13 [-0.17, +0.43] . |
| stk1 | `d_arm_t` | 0.0% | +0.03 [-0.69, +0.79] . | 0.503 [0.48,0.52] . | +0.65 [-0.15, +1.50] . |
| unetbc | `d_t_goal` | 4.7% | -0.16 [-0.52, +0.20] . | 0.426 [0.40,0.45] - | -0.21 [-0.58, +0.16] . |
| unetbc | `d_arm_t` | 0.0% | +0.61 [-0.11, +1.31] . | 0.517 [0.50,0.54] . | +0.64 [-0.12, +1.37] . |

The `blind` column is per term and is the point of the split: a step where no candidate can move the block has zero spread in `d_t_goal` while `d_arm_t` still varies freely. On those steps the ranking is decided by arm reach alone.

## Q4 — slot-to-slot delta, and where it stops improving

Values centered within each control step, then averaged; cluster-bootstrap CI over episodes. `saturation` is the largest k whose remaining gain (slot 15 − slot k) still has an interval excluding zero — i.e. improvement is no longer detectable past it. `None` means no slot showed a detectable remaining gain.

| arm | slot 0 | slot 7 | slot 15 | saturation | sd@0 | sd@15 |
|---|---|---|---|---:|---:|---:|
| curr-lin100-l2 | -0.076 [-0.124, -0.030] | +0.007 [-0.005, +0.019] | -0.002 [-0.019, +0.014] | 1 | 0.485 | 0.259 |
| curr-lin100-l2tol1 | -0.128 [-0.188, -0.077] | +0.007 [-0.004, +0.019] | +0.005 [-0.005, +0.015] | 1 | 0.506 | 0.205 |
| geo735-l2 | -0.045 [-0.163, +0.078] | +0.003 [-0.018, +0.022] | -0.057 [-0.097, -0.018] | – | 0.853 | 0.380 |
| geo735-l2tol1 | +0.250 [+0.104, +0.403] | -0.038 [-0.059, -0.016] | -0.072 [-0.127, -0.024] | – | 0.909 | 0.404 |
| lin100-l2 | -0.088 [-0.197, +0.033] | +0.001 [-0.020, +0.021] | +0.006 [-0.021, +0.029] | – | 0.867 | 0.313 |
| lin100-l2tol1 | +0.265 [+0.115, +0.420] | -0.024 [-0.043, -0.006] | -0.069 [-0.103, -0.037] | – | 0.964 | 0.400 |
| stk1 | -0.021 [-0.043, +0.001] | -0.014 [-0.034, +0.005] | +0.008 [-0.015, +0.029] | – | 0.531 | 0.530 |
| unetbc | -0.004 [-0.019, +0.012] | -0.001 [-0.018, +0.016] | -0.004 [-0.027, +0.019] | – | 0.418 | 0.420 |

**`sd@15` below `sd@0` means the context is narrowing exploration, not improving quality.** That produces a rising mean too, so the two must be read together: a later slot that is better *and* less varied has converged onto the region the earlier candidates already found, which is a different claim from having found something better.

> A guard on the plots: `prefix_max` (best-of-k) rises with k even under a pure i.i.d. resampler, so it is never evidence of context. It is plotted only against `prefix_mean`; the gap between them is the value of having a selector.

## Files

- `per_prediction_A.csv` — every prediction, every slot 0/7/15, argmax slot
- `slot_profile_A.png` — centered value vs slot, all arms
- `slot_profile_terms_A.png` — the same split into `d_t_goal` and `d_arm_t`
- `slot_stats_A.json` — every number above, machine-readable
