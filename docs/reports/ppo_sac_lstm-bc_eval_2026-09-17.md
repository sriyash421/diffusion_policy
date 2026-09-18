# PPO / SAC / LSTM-BC as best-of-N verifiers — 2026-09-17

**Status: in progress.** The heuristic baseline, both BC stages, the learned-V ranking
evaluation (section 5) and **the best-of-N sweep it was built for (section 6)** are measured.
The answer to the headline question is **no**: best-of-N under the learned V falls from 0.240
to 0.000 where the heuristic rises to 0.920. The diagnosis -- that V is the heuristic's inverse
under `--reward delta` -- follows from the reward definition and is consistent with the curves,
but the diagnostic that claimed to confirm it measured the wrong axis and is retracted in
section 6. Q is not measured: no SAC run has finished.

The question: best-of-N on PushT ranks candidates with `t_goal`, a hand-written heuristic that
**ignores `agent_pos` by construction** — until a candidate actually moves the block, every
candidate scores identically. Can a learned value do better? Two sources, deliberately
different: **Q(s, chunk) from SAC**, and **V(s) from PPO**, applied by rolling the candidate
chunk out and evaluating V at the state it reaches.

---

## 1. The heuristic baseline — the number to beat

`scripts/verifier_ranks_expert.py`, the sim verifier only, at step 30k, n=16, 160 decision
points from 20 held-out episodes each. Where does the expert's own chunk `a*` rank among 16
sampled candidates?

| arm | split | blind | informative | p_best (inf) | mean_rank (inf) | p_best (blind) |
|---|---|---|---|---|---|---|
| unetbc | blq | 15.6% | 51.9% | 0.205 | 7.663 | 0.059 |
| unetbc | brd60 | 16.9% | 55.6% | 0.270 | 7.787 | 0.092 |
| value_k16 | blq | 22.5% | 61.3% | 0.347 | 7.990 | 0.214 |
| value_k16 | brd60 | 25.0% | 60.6% | 0.351 | 7.804 | 0.169 |
| value_k1 | blq | 15.0% | 59.4% | 0.284 | 7.379 | 0.135 |
| value_k1 | brd60 | 18.8% | 59.4% | 0.337 | 6.700 | 0.147 |

**Two findings, and both lower the bar:**

**The blind rate is 15–25%, not the 28–35% this repo documents.** These are 30k checkpoints on
geometric splits at n=16; the documented figure came from a different generation. It is a
different measurement, not a contradiction — but the learned verifier's headline claim is
"ranks `a*` well where the heuristic is silent", and the silent region is smaller here than the
README implies. **Use the numbers in this table as the baseline, not 28–35%.**

**`mean_rank` is 6.7–8.0 out of 16, against an exchangeability null of 8.0.** Even on the
*informative* stratum — where the heuristic has real signal by construction — `a*` lands
mid-pack. Only `value_k1` on brd60 (6.70) is meaningfully below null. So the heuristic barely
distinguishes the expert action from sampled candidates. That is a low bar, and also a caution:
at 160 decision points these are noisy.

## 2. The BC stages — warm-start sources

`recurrent_ppo/bc.py` trains the RecurrentPPO policy object itself, on 106 train episodes of
`pusht_seed42_train106_val50`, selecting on **val rollout max-coverage** over 50 episodes.

| arm | best val_max_reward | at epoch | success there | final | band across evals |
|---|---|---|---|---|---|
| bc_image | **0.5772** | 180 / 200 | **0.040** (best 0.080) | 0.3990 | 0.40–0.58 |
| bc_keypoint | 0.2228 | **10** / 200 | 0.000 | 0.1767 | 0.18–0.22 |

**bc_image is the first thing in this line of work to solve a PushT episode** — 4 of 50 at the
0.95 coverage threshold. The retired PPO generation recorded zero across 18 runs at 10M steps
each. This is a BC policy rather than PPO, so it is not the same claim, but it is not nothing.

**bc_keypoint did not learn anything transferable.** Its val peaked at its FIRST evaluation and
never improved across 190 further epochs while its nll fell steadily (−0.73). A warm start from
an epoch-10-of-200 checkpoint is close to a random initialisation, so `ppo_lstm_keypoint_bc` as
configured would measure the encoder-freeze schedule rather than the warm start. **It has not
been launched.**

**Both arms clipped 0.87% of action components** — against ~18.8% under the previous
`delta_scale`. See section 4.

## 3. Training — status

| arm | state | note |
|---|---|---|
| `ppo_plain_keypoint` | **COMPLETED** 10M | `ep_rew_mean` 5.75, **success_rate 0** |
| `ppo_plain_image` | running | 5.60M, `ep_rew_mean` -0.421 |
| `ppo_lstm_keypoint` | running | 4.85M, `ep_rew_mean` -0.522 |
| `ppo_lstm_image` | running | 2.05M, `ep_rew_mean` -0.109 |
| `sac_keypoint` | running | 1.29M; `reward_rate_tau0.95` 9.2e-4, `q_spread_zero_frac` 0.0 |
| `sac_image` | pending | `AssocGrpCpuLimit` |
| `ppo_lstm_keypoint_bc` | **not launched** | see bc_keypoint above |
| `ppo_lstm_image_bc` | not launched | pending the keypoint result |

All four running arms sit within noise of zero reward; none has begun to solve the task.
`ppo_plain_keypoint` reaching 10M steps at **zero success** is consistent with the retired
generation. Its V is still usable as a ranker — under `--reward delta` the return telescopes to
total progress, so V estimates coverage-still-to-gain even for a policy that never crosses the
threshold — but any V number must be reported as V^pi of a policy that does not solve the task.

## 4. Two measurement errors found and fixed

**`delta_scale` was calibrated on the wrong quantity.** A delta action commands
`agent_pos + a*scale`, but the calibration differenced consecutive *targets*. Measured per axis
over the demonstrations:

| quantity | p50 | p90 | p99 |
|---|---|---|---|
| target-to-target | 4.0 | ~13 | 33.0 |
| **target − agent_pos** | 10.8 | 33.2 | **61.2** |

33 px was the p99 of the first and only the **p90** of the second, so **18.8% of demonstration
steps commanded a target the action space could not reach in one step**. That is a ceiling on
every PPO run this repo has done. Now 61 px; the BC stages measured 0.87% clipping as a result.

**SAC seeded its replay buffer from all 206 episodes**, including the 50 the best-of-N sweep
scores it on — so "the learned Q beats the heuristic" would not have been a held-out claim,
while the heuristic has no such advantage. Now restricted to the train split. V was never
exposed this way: BC trains only the actor, and PPO's critic learns from sampled starts.

## 5. V as the ranker — the first learned-value result

Same six checkpoints, same 160 decision points, same `rank-expert` call as section 1, with
`--v /gscratch/robotics/harine/value_arms/ppo_plain_keypoint/model.zip` added. `PushTVVerifier`
rolls the candidate chunk out in the sim, converts the reached state to the PPO keypoint
observation, and evaluates V there, de-normalising by `reward_scale 1.975` recovered from
`model_vecnormalize.pkl`. `sim` rows are the heuristic; `v` rows are the learned value, on the
identical decisions.

`p_best` = fraction of decisions where `a*` outranks all 16 candidates. `mean_rank` is out of
16; **8.0 is the exchangeability null**, lower is better.

| checkpoint | all (sim → V) | block_still (sim → V) | block_moving (sim → V) |
|---|---|---|---|
| unetbc blq137 | 0.157 → 0.163 | 0.039 → **0.130** | 0.217 → 0.179 |
| unetbc brd60 | 0.200 → 0.206 | 0.060 → **0.220** | 0.264 → 0.200 |
| value_k16 blq137 | 0.299 → 0.306 | 0.051 → **0.241** | 0.425 → 0.340 |
| value_k16 brd60 | 0.291 → 0.312 | 0.072 → **0.400** | 0.391 → 0.273 |
| value_k1 blq137 | 0.221 → 0.188 | 0.025 → **0.148** | 0.321 → 0.208 |
| value_k1 brd60 | 0.258 → 0.212 | 0.085 → **0.280** | 0.336 → 0.182 |

`mean_rank` on the same rows, `block_still`: 9.81→6.52, 8.95→7.80, 9.81→6.80, 7.96→**8.40**,
10.48→7.78, 8.80→8.16 — five of six move toward 0, one against.

**The result: V beats the heuristic exactly where the heuristic is blind, and loses where it is
not.** `p_best` on `block_still` rises in **6 of 6** rows, by 2.3×–5.6×. `p_best` on
`block_moving` falls in **6 of 6**. The sign is the same across all three policy classes
(BC-UNet, ST k16, ST k1) and both split families. That is the mechanism the plan predicted:
`t_goal` reads T-to-goal distance only, so before contact all 16 candidates score identically
and `a*` lands at the 8.0 null or worse; V reads `agent_pos` and can prefer approaching. Once
the T is moving, the heuristic measures the thing being optimised directly and V — a coarse
learned scalar — does not improve on it.

**Overall it is a wash, not a win.** On `all`, `p_best` moves up in 4 of 6 rows by ≤0.021 and
down in 2 by 0.033–0.046; `mean_rank` improves in 3 and degrades in 3. `block_moving` is
66–69% of decisions, so its loss cancels the `block_still` gain. **The deployable claim — that
substituting V raises best-of-N success — is not supported by this evaluation and has not been
tested** (the sweep is section 6).

The class-based split (`blind` / `partial` / `informative`, which classifies by whether the
*candidates* moved the T rather than whether the demo did) shows the same effect more weakly:
`p_best` on `blind` rises in 4 of 6 rows. It is the noisier of the two — `blind` is 24–40
decisions per arm, and it depends on the candidate set, so the strata are not the same
decisions across arms. `partial` is erratic in both directions, including 0.128 → 0.000 on
unetbc blq137 over 52 decisions.

### What this result does not establish

- **One V, from a policy that never solved the task.** `ppo_plain_keypoint` finished 10M steps
  at success_rate 0. Under `--reward delta` its V is still a progress estimate, but this is
  V^pi of a failing policy, and nothing here separates "learned value" from "this particular
  learned value".
- **The six rows are not six independent trials.** They share one V and, within a split family,
  largely the same held-out episodes — three rows on blq137's 20, three on brd60's 20. The
  consistent sign is worth more than any single row, but it is closer to two samples than six.
- **Every confidence interval overlaps.** `block_still` is 50–54 decisions per row; e.g. unetbc
  blq137 is 0.039 [0.02,0.06] → 0.130 [0.05,0.22]. The 6-of-6 consistency carries the claim,
  not per-row significance.
- **No recurrent-V caveat applies here.** The V is feed-forward (`ppo_plain_keypoint`,
  `--n-stack 1`), so there is no zero-LSTM-state query. It will apply to the `ppo_lstm_*` arms.

## 6. The best-of-N sweep — V is an inverted ranker

Phase 5(a) asked where the expert chunk `a*` ranks. This asks the deployable question: run the
policy, let each verifier pick, count solved episodes. `sac/eval.py bon-sweep`, n = 1…16,
50 test episodes, the same per-episode sampling noise under both rankers so the arms are paired
episode by episode. Launched from `scripts/slurm/bon_sweep_arms.sh`.

**At n=1 both rankers must agree** — there is nothing to rank — and they do, on all six arms.
That is the harness's own leak check, and it passing is what makes the rest meaningful.

| checkpoint | ranker | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|---|---|---|---|---|
| value_k16 blq137 | `t_goal` | 0.180 | 0.400 | 0.520 | 0.540 | **0.580** |
| | `v` | 0.180 | 0.140 | 0.100 | 0.120 | **0.020** |
| value_k16 brd60 | `t_goal` | 0.100 | 0.240 | 0.220 | 0.380 | **0.320** |
| | `v` | 0.100 | 0.140 | 0.100 | 0.100 | **0.040** |
| value_k1 blq137 | `t_goal` | 0.160 | 0.300 | 0.500 | 0.620 | **0.520** |
| | `v` | 0.160 | 0.120 | 0.040 | 0.040 | **0.020** |
| value_k1 brd60 | `t_goal` | 0.100 | 0.160 | 0.260 | 0.300 | **0.340** |
| | `v` | 0.100 | 0.120 | 0.020 | 0.020 | **0.000** |
| unetbc blq137 | `t_goal` | 0.240 | 0.420 | 0.840 | 0.820 | **0.920** |
| | `v` | 0.240 | 0.220 | 0.180 | 0.060 | **0.000** |
| unetbc brd60 | `t_goal` | 0.240 | 0.360 | 0.460 | 0.560 | **0.620** |
| | `v` | 0.240 | 0.180 | 0.140 | 0.100 | **0.060** |

**All six arms complete.** `t_goal` rises 2-4x in n on every one of them; `v` falls on every
one of them, ending at
0.000–0.040 where the heuristic reaches 0.320–0.920. Searching harder under V is worse than not
searching at all, and worse the harder it searches. That is not a weak ranker; it is a ranker
pointed the wrong way.

### Why: the sign, measured

> **RETRACTED, 2026-09-18.** The numbers below were computed on the wrong axis.
> `_score_candidates` is called **once per candidate**, batched over ENVIRONMENTS
> ([search_procedure.py:1021-1032](diffusion_policy/policy/search_procedure.py#L1021-L1032)):
> each call returns `(B_envs,)`, one score per environment for candidate *i*, and the caller
> stacks those into `(B, n)`. `diag_v_sign.py` treated a single call's array as the candidate
> set, so every statistic below compares scores at **six different states**, not sixteen
> candidates at one state. The `argmax` agreement figure is the worst of it: which *environment*
> scored highest is not a selection, so "never selected the same candidate" was not measured.
> What survives is weaker and still consistent with the inversion -- across states `t_goal` is
> -distance and V rises with distance, so a strongly negative correlation is expected. The
> within-candidate-set question is re-measured by `diag_ranker_agreement.py`, which buffers n
> calls and transposes.
>
> | | |
> |---|---|
> | ~~within-decision correlation~~ | ~~mean -0.589, median -0.856, 82.4% negative~~ |
> | ~~`argmax` agreement~~ | ~~0.000, against 1/8 = 0.125 chance~~ |
>
> The best-of-N curves above are unaffected: they ran through the real selection machinery.

The cause is in the reward, not the code. Under `--reward delta`
([pusht_gym.py:376-384](recurrent_ppo/pusht_gym.py#L376-L384)) the return telescopes to
`(d_here - d_end)` plus a success bonus. This V comes from `ppo_plain_keypoint`, which finished
10M steps at **success_rate 0**, so the bonus term is ~0 everywhere and V reduces to roughly the
current T-to-goal distance -- the negation of `t_goal`. Selection is `argmax`
([pusht_search_mixin.py:28](diffusion_policy/policy/pusht_search_mixin.py#L28)), so it picks the
candidate landing **furthest** from the goal.

The plan's premise was "delta telescopes to total progress, so V estimates coverage still to
gain, which is what a ranker needs". That drops the `coverage_now` term. What a ranker needs is
expected *final* coverage, `coverage_now + V(s)`; `V(s)` alone is monotone in the wrong
direction.

### This also explains section 5

V beat the heuristic on `block_still` in 6 of 6 rows and lost on `block_moving` in 6 of 6. Both
follow: where the T does not move, `d` is identical across candidates, so V ranks on `agent_pos`
-- real signal the heuristic does not have. Where the T moves, `d` varies and the inverted term
dominates. Section 5's result was real and is not evidence that the value is usable.

### What this does and does not settle

- **It does not show that a learned value cannot rank best-of-N.** It shows that *this* value,
  under *this* reward, ranked by `argmax` on the raw scalar, is inverted.
- **The obvious repair is untested**: rank on `coverage_now + V`. One line in `install_v_ranker`.
- **Q is unaffected by this argument.** SAC's chunk Q is fitted with a Bellman max backup to
  terminal success, not to a telescoping progress signal, so it has no reason to invert. It is
  also the only one of the two with a non-trivial `bon/q_spread_zero_frac` reading: 0.0 at every
  probe, against `t_goal`'s measured 15-25% blind rate.
- **`t_goal` itself is non-monotonic at the top end** (k16 brd60 0.380 -> 0.320, k1 blq137
  0.620 -> 0.520, unetbc blq137 0.840 -> 0.820), so "more candidates is always better" does not
  hold even for the heuristic. At 50 episodes a 0.100 move is 5 episodes.

## 7. What is not yet measured

- **Q, on anything.** No SAC run has completed; `sac_keypoint` is at 1.29M steps.
- **`coverage_now + V` as the ranker** — the repair section 6 identifies, untested.
- **V from any arm but `ppo_plain_keypoint`** — the image and recurrent arms are still training.
  Every one of them trains under `--reward delta`, so the same inversion applies to all of them
  unless the ranking quantity changes.
- **Recurrent V's caveat**, when those arms are scored: V is queried at a state the chunk
  *reaches*, which has no history, so the LSTM starts from zeros and the query is
  off-distribution.

## Reproduce

```bash
# the baseline in section 1
python scripts/verifier_ranks_expert.py -c <ckpt> --arm <name> --n 16 --episodes 20 \
       --per-episode 8 --split test --out <out>.json

# the BC stages in section 2
SUBMIT=1 ARMS="bc_keypoint bc_image" bash scripts/slurm/train_value_arms.sh

# the training in section 3
SUBMIT=1 bash scripts/slurm/train_value_arms.sh

# section 5
python sac/eval.py rank-expert -c <ST ckpt> --arm <name> --n 16 --episodes 20 \
       --split test --v /gscratch/robotics/harine/value_arms/ppo_plain_keypoint/model.zip \
       --out /gscratch/robotics/harine/value_arms/v_rank_<arm>.json

# section 6
SUBMIT=1 bash scripts/slurm/bon_sweep_arms.sh
python diag_ranker_agreement.py --ckpt <ST ckpt> --v <ppo ckpt> --out <out>.json

# what section 7 is waiting on: a finished SAC run
SUBMIT=1 RANKERS=t_goal,v,q Q=<sac.zip> bash scripts/slurm/bon_sweep_arms.sh
```

Run outputs are on `/gscratch/robotics/harine/value_arms/`, not under the repo: `$HOME` is a
10 GB hard quota and four arms died on it once already.
