# Per-slot verifier scores under the observation ladder

_Generated 2026-09-03 by `scripts/build_slot_scores_doc.py`. Re-run to refresh._

Does slot 15 actually produce better candidates than slot 0? The ladder assumes so — slot 0 sees the most-corrupted observation and no context, slot 15 the cleanest and the full context. Success rates cannot answer it: they report what the argmax executed, not what each slot produced.

Candidate *i* conditions on exactly *i* scored context entries, so **i is the slot index** the training loss decodes. Scoring every candidate of a closed-loop rollout therefore reads the ladder directly.

Values are `t_goal` = −(mean per-keypoint distance of the T from the goal), in pixels: **negative, and higher is better**. 50 test episodes, n = 16, seed 42, ResNet18 end-to-end, 30 demos.

## Headline

| arm | step | slot 0 | slot 15 | Δ mean | slot 15 > slot 0 |
|---|---:|---:|---:|---:|---:|
| uniform control (no ladder) | 100,000 | -106.5 | -106.6 | -0.14 | **46.0%** |
| 1. linear in t, cap 999 | 100,000 | -118.7 | -117.2 | +1.58 | **54.5%** |
| 1. linear in t, cap 400 | 100,000 | – | – | – | _not dumped yet_ |
| 2. linear in a_bar, cap 999 | 100,000 | – | – | – | _not dumped yet_ |
| 2. linear in a_bar, cap 400 | 100,000 | – | – | – | _not dumped yet_ |
| 3. geometric in t, cap 999 | 100,000 | – | – | – | _not dumped yet_ |
| 3. geometric in t, cap 400 | 100,000 | – | – | – | _not dumped yet_ |
| 4. random base, noised rollouts | 100,000 | – | – | – | _not dumped yet_ |
| 4. random base, clean rollouts | 100,000 | – | – | – | _not dumped yet_ |

The win-rate excludes tied decisions (see Method). **50% is chance.** The control row is what every other row must be read against: it has the identical context structure and no ladder, so whatever it shows is the contribution of conditioning alone, and only the excess over it belongs to the ladder.

## uniform control (no ladder)

No ladder: `slot_obs_noise` uniform, so all 16 slots see the same clean observation. Slots still differ in SEARCH CONTEXT, so this measures what conditioning alone buys — and it is the row every other row must be read against. Rollouts are clean because there is no ladder to switch on.

<sub>`value_k16_ver-t_goal_enc-resnet18_demos-30_seed-42/bon_search/step_0100000`</sub>

1900 decisions.

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean `t_goal` | -106.5 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 | -106.6 |
| std | 79.7 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 | 79.9 |
| argmax share | 26.5% | 6.4% | 5.1% | 4.1% | 4.8% | 3.9% | 4.5% | 4.2% | 5.1% | 4.5% | 4.5% | 5.6% | 4.8% | 4.5% | 5.1% | 6.4% |

- slots 0–7 mean **-106.6**, slots 8–15 mean **-106.6** (Δ -0.02)
- slot 15 − slot 0: mean **-0.14**, slot 15 better in **46.0%** of the 1236 decisions where the two differ
- ties: 32.3% of decisions have all 16 scores equal; 34.9% have slot 0 = slot 15

## 1. linear in t, cap 999

`t_k = (15-k)/15 * 999`. Slots 0-3 sit at sqrt(alpha_bar) 0.01/0.01/0.02/0.04 — four near-identical near-blind slots, so most of the ladder's usable range is squeezed into the clean end.

<sub>`value_k16_ver-t_goal_son-lint-cap999_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

1900 decisions.

| slot | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean `t_goal` | -118.7 | -122.8 | -123.2 | -123.0 | -122.8 | -122.2 | -121.8 | -121.4 | -120.6 | -119.7 | -119.2 | -118.5 | -117.7 | -117.6 | -117.3 | -117.2 |
| std | 48.1 | 53.4 | 54.7 | 55.4 | 55.2 | 55.1 | 54.5 | 54.4 | 54.1 | 53.7 | 53.2 | 52.9 | 52.4 | 52.2 | 52.2 | 52.1 |
| argmax share | 16.9% | 8.6% | 6.9% | 6.1% | 4.9% | 4.4% | 4.1% | 3.9% | 4.7% | 4.9% | 4.4% | 4.3% | 5.3% | 5.5% | 7.0% | 7.9% |

- slots 0–7 mean **-122.0**, slots 8–15 mean **-118.5** (Δ +3.52)
- slot 15 − slot 0: mean **+1.58**, slot 15 better in **54.5%** of the 1054 decisions where the two differ
- ties: 36.9% of decisions have all 16 scores equal; 44.5% have slot 0 = slot 15

## 1. linear in t, cap 400

The same shape compressed into [0, 400]: slot 0 at sqrt(alpha_bar) 0.44 rather than 0.01, i.e. degraded rather than near-blind.

<sub>`value_k16_ver-t_goal_son-lint-cap400_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 2. linear in a_bar, cap 999

sqrt(alpha_bar) falls in equal ~0.066 steps from 0.01 to 1.00. The only shape whose 16 levels are all distinct, and the one the config recommends as default.

<sub>`value_k16_ver-t_goal_son-linsig-cap999_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 2. linear in a_bar, cap 400

The same even grading compressed into [0, 400]. The compression is even in TIMESTEP, so the retained-signal steps are no longer equal.

<sub>`value_k16_ver-t_goal_son-linsig-cap400_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 3. geometric in t, cap 999

`t_k = 999 * 0.85^k`. Decay 0.85 rather than slot_weights' 0.7, which at K=16 leaves 6 of 15 adjacent slots indistinguishable. The trade: slot 15 lands at t=87, so this arm's cleanest slot is not fully clean.

<sub>`value_k16_ver-t_goal_son-geo85-cap999_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 3. geometric in t, cap 400

The same decay compressed into [0, 400]; 2 of 15 adjacent pairs fall within 0.005, so it is mildly collapsed at the clean end where the 999 arm is not.

<sub>`value_k16_ver-t_goal_son-geo85-cap400_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 4. random base, noised rollouts

No fixed ladder: slot 0's timestep is drawn PER SAMPLE from [0, 999] and the linear_signal curve rescaled into [0, that draw], so what varies between samples is the ladder's EXTENT. **This row is a MECHANISM PROBE, outside the success-rate protocol**, which evaluates this arm clean only. It is included because a clean rollout does not apply the ladder at all, so it cannot answer whether the ladder built a slot gradient.

<sub>`value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42/bon_search_obs-corrupt/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## 4. random base, clean rollouts

The same weights read the way the success-rate tables read them — clean. With `corrupt_obs_eval` False the corruption is the identity, so all 16 slots see the same observation and this row should look like the uniform control. It is the check that the row above is measuring the ladder and not something else.

<sub>`value_k16_ver-t_goal_son-rndlinsig-cap999_enc-resnet18_demos-30_seed-42/bon_search_obs-clean/step_0100000`</sub>

_no candidate scores dumped for this arm yet_

## Method

### Producing the data

```
sbatch scripts/slurm/eval_ckpt_pusht_search.sbatch <ckpt> \
    --n-list 16 --store-scores --skip-val [--corrupt-obs-eval]
```

`--corrupt-obs-eval` is **required** on a ladder arm. Without it `corrupt_obs_eval` stays False, every slot is evaluated on the same clean observation, and the ladder under test is not applied at all. It is meaningless on the uniform arm.

Rollouts are closed-loop and execute the argmax, so the recorded scores are the ones a real deployment would have seen — not scores from a distribution the policy never visits.

### Why ties are handled explicitly

A candidate that never touches the block makes the simulator return the identical value, so roughly a third of decisions have all 16 scores exactly equal. That breaks two obvious statistics:

- **The median paired difference is 0 by construction.** The win-rate here is therefore computed only over decisions where slot 0 and slot 15 actually differ.
- **`argmax` breaks ties toward the lowest index**, manufacturing a slot-0 spike. This doc breaks them at random instead — but even then the control shows slot 0 far above the 1/16 = 6.25% baseline while its means are flat, so an ordering bias remains that has not been isolated. **Draw no conclusion from the argmax row**; it is descriptive colour. The mean profile and the tie-excluded win-rate are the numbers to read.

## Caveats

**Steps are not matched.** Each arm is read at whatever checkpoint existed when the dump ran, so cross-arm ABSOLUTE levels are indicative only. The within-arm slot profile — the thing under test — is unaffected, since all 16 slots come from one checkpoint and one set of decisions.

**The ladder arms are scored under noised rollouts and the control under clean ones.** That is not an oversight: the control has no ladder, so there is nothing to switch on. It does mean the absolute gap between control and ladder arms confounds "trained with a ladder" against "deployed on corrupted observations".

**One checkpoint per arm, 1900 decisions.** Enough to separate a 60% win-rate from chance, not enough to rank two arms a few points apart.

