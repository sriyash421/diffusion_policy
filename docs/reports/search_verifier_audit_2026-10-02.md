# PushT search and verifier audit — 2026-10-02

The existing results support **goal-distance best-of-N**, but do **not** establish an incremental benefit from Gemini waypoints. BC versus ST also changes architecture and capacity. A best-of-N improvement is not itself evidence that the transformer learned to improve its next attempt.

## 1. What the existing experiments establish

Source: `summaries/bon_results.csv`, checked against its original `success_curve.json` files. Fixed checkpoint 40k; argmax; same BC weights, seed 42, and 50 test episode IDs within each split. These are success percentages, not coverage.

| Split | BC N=1 | Goal distance N=16 | Gemini pusher lookahead N=16 | Combined dynamic normalization N=16 |
|---|---:|---:|---:|---:|
| blq137 | 28% | **92%** | 24% | 68% |
| brd100 | 48% | **92%** | 28% | 72% |

The goal-only and combined columns use `t_goal` and `wp_v3_la_tgd`; the pusher-only column uses `wp_v3_la`. At N=16, goal-only succeeds where combined fails on **13 versus 1** episodes in blq137 and **13 versus 3** in brd100. Episode IDs, checkpoint paths and seeds were verified equal. The combined rule helps relative to one sample but hurts relative to goal-only at this budget. This is not a claim that every possible Gemini-based verifier is ineffective.

`wp_v3_la_tgn` is a different combined rule: fixed component scales instead of normalization across the current candidate set. In `tgn_vs_bc.csv`, blq137/brd100 BC uses **tgd** while ST uses **tgn**. Do not attribute that gap wholly to the generator. The new five-episode diagnostic uses tgn for both.

The Gemini component here is a **fixed per-episode waypoint plan**, evaluated by a simulator and geometric tracker. Gemini does not inspect and rank every newly sampled chunk. Following a pusher waypoint can look good while producing the wrong block rotation or pushing from the wrong side. The existing [waypoint report](wp_verifier_2026-09-30.md) separately observes high tracker progress with low task success.

A one-sample versus best-of-N result combines additional sampling with selection. Add a same-budget random-selection control for a clean end-to-end selector ablation. Compare goal-only, waypoint-only and combined on identical candidate sets for local ranking, and separately run closed-loop rollouts because their visited states diverge.

## 2. Why BC can outperform ST

**There is a gap even when both use goal-only scoring.** At 40k and N=16, BC/ST are 92%/64% on blq137 and 92%/48% on brd100. Across paired episodes at N=16, BC-only versus ST-only wins are 17 versus 3 on blq137 and 25 versus 3 on brd100. At N=1 they are 28%/28% and 48%/24%. Thus a weak base generator explains part of the brd100 gap, but cannot alone explain blq137's equal N=1 and different best-of-16 results. Candidate distribution, feedback conditioning and selection sensitivity remain relevant.

**The baseline is much larger and architecturally different.** The loaded BC has **292,327,746 total trainable parameters**, versus **17,120,834** for tgn ST (17.1×). BC's UNet denoiser alone has 281,151,200 parameters. These are not equal-capacity models. The saved observation-encoder and optimizer blocks match; ST clips gradients at 1.0 while BC leaves clipping unset. Both saved configs use 8-step DDIM and EMA, so a DDPM-versus-DDIM explanation does not apply to these particular runs. Compare ST K=1 and K=16 with the same encoder, size, optimizer and training pipeline before assigning the whole difference to search.

**The objective permits ignoring feedback.** `diffusion_transformer_search_policy.py::_compute_loss` expands the same expert action across all K slots and trains denoising at each slot. It does not require slot k to beat slot k−1 or score an improvement over a prior candidate. With a clean observation, previous attempts can be redundant; with heavy corruption, the model can become dependent on context that carries an imperfect scalar. This is a mechanism consistent with the code, not a measured causal attribution.

**Selection helps more reliably than taking the last attempt.** The clean tgn ST at 40k, N=16 has argmax/final-pass success 52%/24% on blq137 and 44%/40% on brd100. On blq137, final-pass is also below N=1's 30%. Thus these results do not demonstrate reliable monotonic refinement. Larger N is not guaranteed to help: above K=16 the code uses a rolling 15-attempt context, which drops older candidates, including potentially the best one.

**Training and deployment states differ.** Waypoint tracker snapshots during training follow recorded demonstrations; evaluation trackers follow the policy's executed trajectory. A static pusher plan can become inappropriate after the block deviates. The scalar merges block-goal and waypoint errors, so it also hides *why* an attempt failed.

## 3. What differs from `harine/maze`

Inspected branch `cbdd68a857c06ff55cb278564e2a264440b36fe1`, including `common/maze_verifier.py` and the diffusion-search config.

- Maze uses a shortest-path distance table over free cells and simulates collision-aware movement. It supplies obstacle-aware progress; PushT's keypoint distance is not a contact-aware cost-to-go. Moving away from the immediate target can be necessary to reposition the pusher.
- Maze's action effects are direct navigation. PushT action chunks must first establish contact, then translate and rotate a block. Many useful approach chunks have identical block-goal scores.
- The maze diffusion-search config sets `causal_attn: False`; the audited PushT config sets it to `True`. This mask is **within the denoised action chunk**. Disabling it does not require revealing future search attempts: the search-history mask is separate. Test this before treating the two implementations as equivalent.
- Maze config uses a flattening encoder; PushT uses a learned image encoder and a different baseline family. Successful maze results do not isolate these representation and architecture changes.

## 3a. Why the noise result does not transfer directly

The user clarified the reference as fuzzy DST versus DST. The located L2S launcher sets `policy.corrupt_obs=True` for `dst_fuzzy`; its metric exporter labels it `dst_obsnoise_ddpm100`, and the vendored implementation specifies 100 DDPM steps (not 10). At checkpoint 20k on held-out quadrant goals, three-seed mean N=16 last-candidate success is **88.7% fuzzy versus 52.3% DST**; max-selection success is **99.0% versus 67.3%**. N=1 is 15.0% versus 49.7%. This is a real last-candidate recovery pattern, unlike the audited PushT table. Source: `/mmfs1/home/harine/diffusion_policy_standalone_maze/maze_splits_all_metrics.csv`; launcher `/gscratch/robotics/harine/L2S/l2s/run_procgen_train.py`.

The current PushT arms do not implement the same noise experiment. The user is rerunning conditionally noised evaluation; no duplicate rerun was launched here.

| Noise property | `harine/maze` implementation | Audited PushT flat600 / ramp runs |
|---|---|---|
| At evaluation | Applied whenever `corrupt_obs=True` | Saved `corrupt_obs_eval: null` becomes `False`; clean evaluation |
| Previous attempts during training | Sampling keeps corruption active even in eval mode | Outer/inner context collection calls `policy.eval()`, so attempts use clean features |
| Query features in training loss | Noised | Noised by the slot ladder |
| Noise across attempts | Fresh noise and timestep each sampling call | One shared noise draw per decision when enabled |
| Noise strength | 100-step linear beta .001→.02; signal multiplier about 1.00→0.59 | 1000-step .0001→.02; t=200 multiplier 0.810, t=600 multiplier 0.160, scaled by feature std |

The `corrupt_obs_eval` requirement is enforced for `corrupt_obs=True`, but **not for the separate slot-noise path**. That is a configuration-contract gap: a noisy slot arm can silently default to clean context collection and clean evaluation. Existing result metadata records the *CLI override* (`null`), not the resolved runtime Boolean, making this hard to see from the CSV.

At the same 40k checkpoint and N=16 (50 episodes per row):

| Split | Training noise | N=1 | Argmax N=16 | Final-pass N=16 |
|---|---|---:|---:|---:|
| blq137 | none | 30% | 52% | 24% |
| blq137 | flat600 | 2% | 72% | 16% |
| blq137 | ramp200→600 | 10% | 44% | 6% |
| brd100 | none | 26% | 44% | 40% |
| brd100 | flat600 | 14% | 56% | 24% |
| brd100 | ramp200→600 | 20%* | 54% | 24% |

*The ramp brd100 final-pass file reports N=1=22%, while argmax reports 20%. This inconsistency needs an exact-code/seed/checkpoint provenance check; the methods should coincide at N=1. Do not silently average it away.*

Thus “noise has no effect” is too broad for these rows. It can improve best-of-N while making the standalone/final proposal worse. The hypothesized mechanism is increased candidate diversity combined with selection, or reliance on clean-generated history, rather than robust iterative improvement; the table alone cannot establish which.

For a fair transfer test, first match **effective signal-to-noise**, **train/context/eval gates**, and **shared versus independent noise**. Maze's fresh noisy view on every attempt can make history aggregate multiple views of the same observation; this is not solely feedback-driven search. PushT's shared-noise design removes that source of extra information. Run both variants intentionally.

Priority fix: add explicit context-collection and evaluation corruption choices for slot-noise runs, independent of disabling dropout via `.eval()`, and record their resolved values. Preserve legacy checkpoint behavior rather than globally flipping a default. For existing checkpoints, `--corrupt-obs-eval` is a useful diagnostic, but it also introduces a distribution shift relative to clean-generated training histories; it is not a conclusive repair. Retrain a matched arm with an explicit policy for all three stages, then repeat final-pass and shuffled-feedback controls.

## 4. Five-episode candidate audit

The diagnostic uses both 40k checkpoints on blq137, seed 42, N=16, `wp_v3_la_tgn`, and the first five test episodes, with no success-based selection of episodes. Every decision is shown, including tied scores. Raw action chunks and initial states are logged for replay.

Each candidate is evaluated by resetting to the episode's initial state and replaying the complete executed action history before branching. This preserves momentum and contact history without assuming a five-dimensional pose is the full simulator state. Replay poses are checked against the live logs at every decision. Candidates are scored by peak and final coverage over the executable chunk, stopping at success or the episode limit. Coverage spread above 0.001 defines an informative decision; ties within 0.001 count as equally best.

This answers whether the selector picked the best **short-horizon sampled action**, not the globally best action or the best long-horizon continuation. Before-contact actions can be equally good by coverage while having very different future consequences. The two policies visit different states, so comparing their aggregated candidate spreads is descriptive, not a paired state-controlled experiment. Five episodes are diagnostic examples, not a new success-rate benchmark.

### Measured results and videos

[Open the five paired video comparisons](../../videos/verifier_audit_20261002/index.html). Each episode has BC and ST videos with every sampled chunk, the verifier-selected action, a cyan coverage-best path, and candidate coverage bars. [Full candidate replay measurements](../../videos/verifier_audit_20261002/coverage_audit.json) and [derived findings](../../videos/verifier_audit_20261002/audit_findings.json) are beside the videos. Artifacts live on scratch storage through the workspace link because home-directory quota prevented reliable video writes.

| Episode | BC max coverage / outcome | ST max coverage / outcome |
|---|---|---|
| 1 | 95.29% / success | 92.42% / failure |
| 11 | 95.80% / success | 83.90% / failure |
| 15 | 45.19% / failure | 94.80% / failure |
| 18 | 93.42% / failure | 51.58% / failure |
| 22 | 93.81% / failure | 95.27% / success |

| Candidate diagnostic | BC | ST |
|---|---:|---:|
| Decisions replayed | 160 | 186 |
| Informative decisions (coverage spread > 0.001) | 82 | 77 |
| Coverage-best selection, within 0.001 | 45.1% | 57.1% |
| Mean coverage regret of selected action | 0.77 percentage points | 0.42 percentage points |
| Selected coverage advantage over uniformly random candidate | +2.27 percentage points | +1.30 percentage points |
| Pairwise score/coverage ordering accuracy, excluding coverage ties | 69.2% | 71.0% |
| Mean endpoint spread at the **same five reset states** | **30.51 px** | **9.36 px** |
| Episodes offering an immediate-success candidate anywhere on the recorded trajectory | 4/5 | 2/5 |
| Decisions offering success but selecting a non-success candidate | 3 | 1 |

Except where marked, ranking averages use informative decisions on each policy's own states. The same-reset comparison uses identical initial state arrays, verified equal. BC is wider on **all five** reset states, not just on average. Wider proposals are not automatically better, but the ST's narrow candidate sets are a concrete bottleneck worth testing. On informative ST decisions, the final candidate improves peak coverage over candidate 0 by only **0.145 percentage points on average**; BC's last i.i.d. draw differs by +0.108 points. Five episodes do not support a significance claim, but this is not a strong measured refinement effect.

The verifier has useful local ordering on both distributions. ST does **not** have worse coverage-selection accuracy in this sample. BC provides more diverse choices and encounters more immediate winning candidates, while ST also has trajectories where no sampled chunk gets close to success: maximum offered coverage is only 51.58% on episode 18 and 83.90% on episode 11. Conversely, BC's episode 15 proposals peak at only 46.51%. This is a proposal/recovery problem as well as a selection problem, and it is not uniform across episodes.

The shadow goal-only selector has lower mean short-horizon regret on BC candidates (0.366 versus combined 0.767 percentage points), but slightly higher on ST candidates (0.474 versus combined 0.420 points). This is a local, fixed-trajectory diagnostic; it does not contradict the 50-episode closed-loop goal-only advantage in section 1.

All replay state checks passed. On ST candidates that neither terminate nor hit the time limit, mean verifier versus exact-replay endpoint goal-distance discrepancy is 0.058 px, median 0.000008 px, p95 0.289 px (2,907 candidates). Reset fidelity matters for close rankings, but this sample does not suggest a generally broken simulator rollout.

### A verified missed success

ST, episode **15**, decision **32** (zero-based; environment step 256):

| Candidate | Combined verifier score (higher better) | First successful step in chunk | Peak executable coverage | Coverage if all eight steps continue |
|---|---:|---:|---:|---:|
| 8 | −31.0763 | 6 | **95.134%** | 92.104% |
| 15 (final sample) | −31.8122 | 5 | **95.025%** | 90.169% |
| 14 (selected) | **−30.5807** | none | 94.796% | 94.173% |

The environment terminates on the first successful step, but the verifier evaluates the endpoint after continuing. Both candidate 8 and candidate 15 would win this episode from the same actual state. The selected candidate does not, and the episode eventually fails. This demonstrates a consequential local selection mistake and exposes why a terminal-success flag is needed; it does not by itself establish how much of the aggregate BC/ST gap this fix recovers. The goal-only pick at this state is candidate 7, whose true peak is also below success (94.823%), so dropping Gemini alone would not repair this particular decision.

![ST episode 15 decision 32: verifier-selected candidate 14 versus successful candidate 8](../../videos/verifier_audit_20261002/audit_st_ep15_dec032.png)

The continuation trace is saved in `videos/verifier_audit_20261002/missed_success_ep15_dec32.json`. Prioritize successful simulated chunks explicitly and use terminal-aware rollout semantics, then rerun matched evaluation. Keep the feedback scale consumed by existing ST checkpoints controlled when testing a selection-only change.

## 5. Bugs corrected and remaining fixes

Changes in this audit:

- `scripts/render_search_videos.py`: native verifier validation used `VALUE_FNS`, which excludes waypoint verifiers and rejected `wp_v3_la_tgn` before rendering. It now uses `VERIFIER_VALUES`. Under `--execute final`, the step log incorrectly labeled the argmax index as executed. The log now records the final candidate. Added full-precision `chunks_exact` and `initial_state`; the old rounded action list cannot reliably reproduce a contact trajectory.
- Renderer labels now describe generic scores rather than incorrectly calling tgn costs pixel distances or equating every score tie with no block motion. Failed PNG writes now raise an error instead of silently leaving incomplete artifacts.
- `scripts/measure_verifier_fidelity.py`: reset variants started at recorded demo poses while the reference started at replayed poses. The branches now share the reference's actual replayed poses, separating reset error from dataset replay drift. Temporary environments are closed.
- `scripts/audit_candidate_coverage.py`: new complete-history branch replay, state consistency checks, tie-aware ranking/regret metrics, annotated videos and an HTML viewer.
- Terminal-scoring follow-up (opt-in): with `terminal_scoring=True`, verifier lanes freeze at first success, including the returned state/image, and waypoint progress stops advancing at that step. `rollout(..., return_success=True)` exposes a separate success mask. Selection gives successful chunks score 2 (all unfused simulated heuristics are <=1), then reapplies success priority after cross-candidate fusion. Raw heuristic feedback and its normalization remain separate from this priority; feedback now describes the terminal state. Argmax chooses a successful chunk whenever the simulator finds one. Softmax remains stochastic, with successful chunks receiving the highest score; fixed-index/final-pass execution still follows its configured behavior. Learned-Q scoring is unchanged.

**Legacy version retained for current tests:** legacy full-chunk simulation and scoring remain the default, including continued waypoint updates after simulated success. Existing eval commands keep this behavior. Both `eval_search_pusht.py` and `scripts/render_search_videos.py` accept `--terminal-scoring` for the corrected version and `--legacy-terminal-scoring` to explicitly pin the old one. Eval curves record `terminal_scoring`; corrected results use a separate `_terminal-success` output-directory suffix. Curves lacking the field are treated as legacy for merge checks. Policy config opt-in is `verifier_terminal_scoring: true`. TGD is outside this follow-up's validation scope.

Validation: all six tests in `tests/test_pusht_terminal_scoring.py` pass, covering legacy continuation/scores (including tgn), terminal state/image freezing, mixed batches and padding, waypoint stopping, context separation, fusion priority, and pool rebuilding when switching versions. Separate checks pass for legacy curve continuation, cross-version merge isolation, distinct output paths, eager/lazy overrides and CLI plumbing. Syntax and `git diff --check` pass. No new aggregate success-rate evaluation was run.

Terminal-fix replay: episode 15 / decision 32 candidates 8 and 15 now remain at their first-success coverage, 95.1343% (step 6) and 95.0253% (step 5), through the rest of the chunk. The production pose-reset verifier flags candidates 5, 6, 8 and 15 as successful and goal-only argmax selects the first tie, candidate 5. Full-history audit coverage for candidate 5 is only 94.9727% (candidate 6 does succeed at 95.0252%). Thus terminal handling is repaired, but the reset approximation still produces a near-threshold false positive: this is not evidence that the fix alone wins episode 15. Replay outputs: `videos/verifier_audit_20261002/terminal_scoring_regression.json`.

Remaining issues to address:

1. **Comparison/provenance:** include exact verifier rule, component scales, plan hashes, inference settings and code revision in result keys. `TGN_T_GOAL_SPREAD`, `TGN_PIN_SPREAD` and `TGN_CONTEXT_SCALE` are module constants; loading a checkpoint does not freeze their historical values. Changing them changes the model's feedback distribution without changing its saved verifier name.
2. **Simulator fidelity:** the production verifier resets from agent position and reconstructed block pose, losing velocity/contact state and applying a settle step. This is real, but do not assume it explains the main gap. The corrected small probe (20 decision points, five demo episodes, eight-step chunks) measured mean goal-distance error **0.040 px**, p95 **0.317 px**; warm-up reduced those to 0.006/0.025 px. This is only a small demo-state sample, and warm-up remains an approximation.
3. **Existing fidelity guidance is stale:** its printed “~5 units” spread gate is historical; current tgn uses 0.82 and 0.88 component scales. Compare simulation error against the actual top-two score margin on current candidate sets.
4. **Terminal-fix evaluation:** the termination mismatch is fixed in code (above). Measure how often the fix changes the winner and run matched closed-loop evaluations before attributing aggregate improvements to it.
5. **Static-plan failure handling:** reaching/skipping all pusher pins does not certify task success. For tgn, a completed plan makes the pin term zero and leaves goal-distance selection. Replan or disable inconsistent pins based on observed block state, and validate the change against goal-only.

## 6. Experiments to prioritize

1. **Evaluate the terminal-success fix first:** run a matched evaluation with frozen terminal branches and successful candidates ranked first. The success bonus is selection-only; the terminal state can still change the raw feedback consumed by later candidates. Keep this distinction when interpreting the ablation.

2. **Keep a strong reference:** BC + goal-distance best-of-16, fixed validation-selected checkpoint, same held-out episode list. Establish ST K=1 versus K=16 under matching architecture and optimization. Report success and candidate-budget/latency curves.
3. **Separate conditioning from selection:** at fixed N and controlled noise streams compare real history, empty history, shuffled scalar feedback and action–feedback pairing shuffles. Measure final-candidate success, same-state slot quality and diversity, and best-selected success. Retrained controls are needed to distinguish inference distribution shift from lack of useful conditioning.
4. **Test noncausal action-chunk attention:** match maze's `causal_attn: False` while retaining causal search history. This is a focused architecture ablation rather than a new verifier sweep.
5. **Make feedback more informative:** expose separately normalized change in block-goal distance, block rotation, contact/approach status and waypoint consistency instead of only their scalar sum. Center relative progress on the current state or best prior attempt. Fit scales on training data and freeze them in the checkpoint.
6. **Improve the proposal model first:** pretrain an ST K=1 proposal, then add search conditioning; alternatively distill a strong BC proposal into the same ST family before refinement training. Measure whether later slots improve and retain diversity rather than merely increasing verifier value.
7. **Train on policy mistakes and recovery:** mix on-policy failed states with demonstrations and use reliable recovery targets or longer-horizon rollout comparisons. Do not optimize aggressively against the current waypoint scalar: the existing waypoint-only curves already show that higher compliance can lower success.
8. **Use longer-horizon action evaluation where coverage is flat:** a short BC continuation from candidate endpoints can evaluate repositioning more fairly than immediate coverage. Use a fixed continuation policy and paired rollout noise, and report the extra cost. Prior Q results in this repo were poor, so substituting a learned Q without a ranking/closed-loop check is not an established fix.

Validation: Python compilation; synthetic two-decision replay with nonzero motion and duplicate-action ties; corrected 20-point fidelity probe. 346 decisions × 16 candidates replayed with live-state checks; all ten annotated videos decoded through their final frame at 2 fps, and the viewer contains five paired episodes. Videos and measured results are in section 4.
