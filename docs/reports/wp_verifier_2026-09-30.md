# The VLM waypoint verifiers (wp_v3, wp_v5) on PushT — 2026-09-30

**Status: complete.** One checkpoint, the 176-demo BC UNet expert at step 60k
(`unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-176_seed-42`, train 176 / val 0 / test 30),
measured three ways under three verifier values: the `t_goal` baseline and the two
Veritas-tracker values `wp_v3` (pusher waypoints only) and `wp_v5` (pusher waypoints + T poses,
averaged). 27 ckpt jobs, all COMPLETED. Numbers are recorded, not selected: the 60k step was
named up front, and nothing below nominates an n or a value.

**Headline: the waypoint verifiers do exactly what they were built for and it does not help.**
They give a strict ordering to the chunks `t_goal` cannot tell apart (§2), but that ordering
is not the expert's (§1), and best-of-n under either value is flat or *falls* with n while
the tracker's own progress metric rises (§3) — the sampler is optimising the plan, not the task.

Plans: the 30 test-episode plans committed under `media/veritas_pusht/{v3_end_outside,
v5_pusher_and_t}/` (gemini-3.8-flash, one call per episode). **No Gemini call was made for
this report**; the sunk cost of those 60 plans is $0.97 (v3 $0.55, v5 $0.42, from the
`usage` recorded in each `ep*.json`).

## 1. a* rank among n=16 policy candidates

`scripts/astar_uniform_walk.py` walks the 30 test demos open-loop, one decision every Ta=8
steps (472 decisions), and at each scores 16 policy candidates, 16 uniform straight-line
chunks and the recorded expert chunk a*. Mean rank / n is 0 when a* beats every candidate;
ties count half, so 0.5 is the exchangeability null. Cluster bootstrap over episodes.

| value | decisions | a* p_best (all) | a* mean rank / n (all) | a* p_best (informative) | a* mean rank / n (informative) |
|---|---|---|---|---|---|
| t_goal | 472 | 19 [16, 22] | 0.48 [0.44, 0.52] | 24 [19, 29] | 0.48 [0.42, 0.54] |
| wp_v3 | 472 | 16 [11, 20] | 0.58 [0.53, 0.62] | 14 [9, 19] | 0.61 [0.56, 0.67] |
| wp_v5 | 472 | 20 [15, 24] | 0.51 [0.47, 0.55] | 17 [12, 22] | 0.54 [0.48, 0.60] |

Neither waypoint value ranks a* better than `t_goal`. `wp_v3` ranks it *below* the null
(0.58; 0.61 where `t_goal` is informative, CI excluding 0.5): the pusher-only plan actively
prefers non-expert chunks. Spearman(score, −distance to a*) is ≈0 for all three (−0.012,
−0.010, +0.003), so none of them steers toward the expert in the continuous sense either.

Provenance of the top pick (policy vs uniform, matched budgets, null 0.5 each):
`t_goal` 0.30 / 0.65, `wp_v3` 0.16 / 0.81, `wp_v5` 0.19 / 0.75. The waypoint values prefer
the uniform straight-line chunks over the policy's even more strongly than `t_goal` does.

## 2. NO-TOUCH: can the verifier rank chunks that never move the T?

This is the question the pusher track was added for. Under `t_goal` every non-mover scores
the current T-to-goal distance bit-for-bit. 202 decisions have ≥2 non-moving policy
candidates (mean 13 of 16); a* itself is a non-mover on 165 decisions.

| value | decisions w/ 2+ non-movers | distinct scores among non-movers | fraction ranked | a* non-touch decisions | a* p_best among non-movers | a* mean rank frac (0.5 = null) |
|---|---|---|---|---|---|---|
| t_goal | 202 | 1.00 of 13.0 | 0.00 | 165 | 0.09 | 0.50 |
| wp_v3 | 202 | 12.94 of 13.0 | 1.00 | 165 | 0.21 | 0.55 |
| wp_v5 | 202 | 11.38 of 13.0 | 0.87 | 165 | 0.22 | 0.52 |

**Yes, mechanically.** `wp_v3` gives every non-mover a distinct score on every such decision;
`wp_v5` on 87% (the T track is flat on non-movers, so only the pusher half varies). Among the
uniform chunks the mean spread is 0.07 (`wp_v3`) / 0.03 (`wp_v5`) on a 0..1 scale.

**But the ordering is not the expert's.** a* is the top non-mover 21–22% of the time
(chance ≈ 1/14 ≈ 7%), yet its mean rank fraction is 0.55 / 0.52 — at or below the null.
The verifier sometimes puts a* first and otherwise scatters it. Among non-movers the top pick
is uniform 61% / 60% of the time, policy 30% / 28%, a* 9% / 11%.

Decision frames (same three decisions under each value; colour = value, line style = source,
star = a*, ring = argmax): [`media/wp_verifier_viz/`](media/wp_verifier_viz/). In
`ep13_frame1825` the pusher is below-right of the T; `wp_v3` scores every policy candidate
that heads up-left toward waypoint 1 nearly equally (0.20–0.22) and the uniform chunks
pointing away lowest — a clean ordering by "closer to the next pin", which is not the same
as "pushes the T".

## 3. Best-of-n on the BC UNet expert, n = 1 … 128

`eval_search_pusht.py --skip-val --n-envs 30 --n-list <n>`, argmax selection, one ckpt job per
(value, n). 30 test episodes, Wilson 95% CI in brackets; ±18 points at p=0.5, so read the
trend, not adjacent cells. n=1 is identical across rows by construction (same seed, no
selection).

| value | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|---|---|---|
| t_goal | 47 [30, 64] | 83 [66, 93] | 67 [49, 81] | 80 [63, 90] | 87 [70, 95] | 90 [74, 97] | 90 [74, 97] | 83 [66, 93] |
| wp_v3 | 47 [30, 64] | 47 [30, 64] | 27 [14, 44] | 43 [27, 61] | 27 [14, 44] | 20 [10, 37] | 20 [10, 37] | 23 [12, 41] |
| wp_v5 | 47 [30, 64] | 57 [39, 73] | 47 [30, 64] | 73 [56, 86] | 33 [19, 51] | 40 [25, 58] | 50 [33, 67] | 43 [27, 61] |

![success vs n](media/wp_verifier_bon_step60k.png)

Mean tracker progress at episode end (pusher / T track, 0..1), from the same rollouts:

| value | n=1 | n=2 | n=4 | n=8 | n=16 | n=32 | n=64 | n=128 |
|---|---|---|---|---|---|---|---|---|
| wp_v3 | 0.90 | 0.89 | 0.87 | 0.90 | 0.91 | 0.91 | 0.89 | 0.93 |
| wp_v5 | 0.91 / 0.88 | 0.90 / 0.89 | 0.89 / 0.89 | 0.94 / 0.87 | 0.92 / 0.86 | 0.96 / 0.94 | 0.98 / 0.89 | 0.98 / 0.93 |

- `t_goal` climbs from 47% to 83–90% by n≥8, as on every earlier BC arm.
- `wp_v3` **falls** with n: 47% → 20–27% from n=16 on, while its pusher-track progress rises
  to 0.93. Following the pusher pins more faithfully makes the episode *less* likely to
  succeed. The pins are where Gemini drew them, at one frame, with no dynamics.
- `wp_v5` is flat-to-noisy (33–73%, all CIs overlapping n=1) while both tracks reach 0.93–0.98.
  The T-pose track reports the T "at" its final pose (tolerance 2 px mean keypoint, plus the
  tracker's skip-ahead and hysteresis) on rollouts that do not reach the 95% coverage success
  needs, so a high T-track score is not a success proxy.

## 4. Cost

| | |
|---|---|
| Gemini | **$0** this report. Sunk: 60 test plans, $0.97 (v3 $0.55, v5 $0.42), 116k in / 36k out / 203k thinking tokens. |
| Compute, measured | 26 COMPLETED jobs = **10.8 GPU-h** on the ckpt partition (24 BON + 3 a* walks; the wp values cost less per n than `t_goal` because their rollouts end sooner). |
| Compute, overhead | 2.3 GPU-h: three smoke jobs and one batch of 24 jobs that died at launch (`$0` under sbatch is the spooled copy; fixed in `2015f14`). |
| Wall clock | 12:59 → 14:03 for the 27 real jobs (first job start to last job end), with gscratch reading at ~8 MB/s from the compute nodes all afternoon. |

## 5. Reproduce

```
CKPT=<run>/checkpoints/step_0060000.ckpt
unset WANDB_API_KEY
CKPT=$CKPT SUBMIT=1 bash scripts/slurm/submit_wp_bon.sh                     # 24 jobs
CKPT=$CKPT sbatch --account=robotics --partition=ckpt --array=0-2 \
    --time=4:00:00 scripts/slurm/wp_astar_walk.sbatch                       # 3 jobs
python scripts/wp_verifier_table.py --run <run> --step 60000 --fig docs/reports/media/wp_verifier_bon_step60k.png
```

Raw inputs are copied beside this page: [`summaries/from-analysis/wp_verifier_step0060000/`](summaries/from-analysis/wp_verifier_step0060000/) holds the
three `success_curve.json`s and the three a*-walk JSONs (the `analysis/` tree is gitignored).
