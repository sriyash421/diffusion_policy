# PPO / SAC / LSTM-BC as best-of-N verifiers — 2026-09-17

**Status: in progress.** The heuristic baseline, both BC stages, and **the first learned-V
ranking evaluation** (section 5) are measured. Q is not: no SAC run has finished. Five of the
six RL arms are still training.

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
| `ppo_plain_image` | running | 2.45M, `ep_rew_mean` -0.174 |
| `ppo_lstm_keypoint` | running | 2.07M, `ep_rew_mean` -0.414 |
| `ppo_lstm_image` | running | 890k, `ep_rew_mean` +0.009 |
| `sac_keypoint` | running | 610k; `reward_rate_tau0.95` 7.4e-4, `q_resolution` 0.024 |
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

## 6. What is not yet measured

- **Q, on anything.** No SAC run has completed; `sac_keypoint` is at 610k steps.
- **The best-of-N sweep**, which is the deployable question: does swapping the heuristic for a
  learned value make the policy solve more episodes. Section 5 measures ranking of `a*`, which
  is necessary but not sufficient.
- **V from any arm but `ppo_plain_keypoint`** — the image and recurrent arms are still training.
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

# what section 6 is waiting on
python sac/eval.py bon-sweep -c <ST ckpt> --rankers t_goal,v,q --v <ppo> --q <sac>
```

Run outputs are on `/gscratch/robotics/harine/value_arms/`, not under the repo: `$HOME` is a
10 GB hard quota and four arms died on it once already.
