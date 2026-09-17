# PPO / SAC / LSTM-BC as best-of-N verifiers — 2026-09-17

**Status: in progress.** The heuristic baseline and both BC stages are measured. No learned V or
Q has been evaluated yet, because five of the six RL arms are still training.

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
| `ppo_plain_image` | running | restarted 15:26 |
| `ppo_lstm_keypoint` | running | restarted 15:26 |
| `ppo_lstm_image` | running | restarted 15:26 |
| `sac_keypoint` | running | first SAC run ever completed in this repo would be this one |
| `sac_image` | pending | `AssocGrpCpuLimit` |
| `ppo_lstm_keypoint_bc` | **not launched** | see bc_keypoint above |
| `ppo_lstm_image_bc` | not launched | pending the keypoint result |

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

## 5. What is not yet measured

- **V or Q on any evaluation.** `rank-expert --v/--q` and `bon-sweep`'s `v` ranker are written
  and import cleanly but have never scored a real checkpoint.
- **The best-of-N sweep**, which is the deployable question: does swapping the heuristic for a
  learned value make the policy solve more episodes.
- **The contact/no-contact split for V.** `frames` is Q-only today.
- **Recurrent V's caveat**, when it is measured: V is queried at a state the chunk *reaches*,
  which has no history, so the LSTM starts from zeros and the query is off-distribution. The
  feed-forward arms have no such problem.

## Reproduce

```bash
# the baseline in section 1
python scripts/verifier_ranks_expert.py -c <ckpt> --arm <name> --n 16 --episodes 20 \
       --per-episode 8 --split test --out <out>.json

# the BC stages in section 2
SUBMIT=1 ARMS="bc_keypoint bc_image" bash scripts/slurm/train_value_arms.sh

# the training in section 3
SUBMIT=1 bash scripts/slurm/train_value_arms.sh

# what section 5 is waiting on
python sac/eval.py rank-expert -c <ST ckpt> --v <ppo ckpt> --q <sac ckpt> --split test
python sac/eval.py bon-sweep   -c <ST ckpt> --rankers t_goal,v,q --v <ppo> --q <sac>
```

Run outputs are on `/gscratch/robotics/harine/value_arms/`, not under the repo: `$HOME` is a
10 GB hard quota and four arms died on it once already.
