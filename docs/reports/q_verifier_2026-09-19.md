# The SAC Q as a best-of-N verifier — 2026-09-19

**Status: complete**, and superseded in part — §7 lists the follow-on arms training as of
2026-09-20. Both evaluations measured, plus a verifier-off control (§5) and a seed-replicated
diagnostic of the UNet BC n=16 collapse (§6). 19 jobs, all COMPLETED.
[ppo_sac_lstm-bc_eval_2026-09-17.md](ppo_sac_lstm-bc_eval_2026-09-17.md) §7 listed "**Q, on
anything**" as unmeasured — no SAC run had finished. Both keypoint arms have now produced
checkpoints, so this is that measurement.

**The two sections disagree, and that is the result.**

**Section 1: Q ranks the expert action far better than `t_goal` where `t_goal` is blind** —
`p_best` up **4–9×** on all six arm/Q rows with **non-overlapping confidence intervals**, which
§5's V never achieved. Overall `p_best` rises in 6 of 6 too, so unlike V it is not a wash. The
contamination control comes out clean: the Q that never saw these demos ranks **as well as or
better than** the one that did.

**Section 4: that advantage does not deploy.** Best-of-N under Q reaches 0.200–0.267 at n=16
where `t_goal` reaches 0.400–0.467 — **`t_goal` wins on all six rows**. Q is not *inverted* the
way V was (it improves in n on 5 of 6 rows, where V fell on 6 of 6), it is simply the worse
ranker for choosing actions to execute.

**The 09-17 report's warning holds: ranking `a*` is necessary, not sufficient.** A verifier can
dominate the expert-ranking benchmark and still lose the sweep it was built for.

---

## The setup

**Two Q arms, and the difference between them is the control.** Both trained under
`--split-file pusht_seed42_train106_val50.json` with **byte-identical `command.txt`**; they differ
only in `params/args.yaml`'s `demo_episodes`, added by `e6070f9`:

| run | demo seeding | steps | Q used here | peak uniform `eval/success_rate` |
|---|---|---|---|---|
| `sac_keypoint` (**sac206**) | **all 206** episodes | 10M, completed | `model_8500000_steps.zip` | 1.00 @ 8.51M |
| `sac_keypoint_demos106` (**sac106**) | **106 train** episodes | 5M, cancelled | `model_3300000_steps.zip` | 0.80 @ 3.36M |

**Three policies**, all step 30k, all trained on `pusht_seed42_train176.json` (176 train / 0 val /
30 test) filtered to the 12,972 moving train windows of
`transitions/pusht_seed42_train176_moving_ta8.json`, all at git `e6070f9`:

| arm | run | trained-with verifier | search in training | 30k ckpt written |
|---|---|---|---|---|
| ST k16 uniform | `outer_inner/value_k16_ver-t_goal_enc-resnet18_demos-176_split-mv_seed-42` | `t_goal` | k=16, context = `t_goal` value | 2026-09-18 14:01 |
| ST k16 flat400 | `outer_inner/value_k16_ver-t_goal_son-flat400_enc-resnet18_demos-176_split-mv_seed-42` | `t_goal` | k=16, context = `t_goal` value | 2026-09-18 19:02 |
| UNet BC | `unet_bc/unetbc_ver-t_goal_enc-resnet18_demos-176_split-mv_seed-42` | `t_goal` | none (`n_search_actions: 1`) | 2026-09-18 09:30 |

**Every arm was trained under `t_goal`** (`verifier_tag: t_goal`, the `ver-t_goal` in each run
name). Section 4 discusses what that does to the comparison.

There is no k1 arm in this generation — it trained k16 and k4 only.

---

## 1. Where the expert action ranks — the 100 demos outside the 106

`sac/eval.py rank-expert --episodes-from pusht_seed42_train106_val50.json --episodes-split
val,test`: the **100 episodes** (val ∪ test) that `demo_episodes=train` seeding never preloads,
**800 decision points**, n=16. The 09-17 baseline had 160 points from 20 episodes; this is 5×.

`t_goal` and Q score the **same decisions, the same candidates and the same `a*`** — the columns
differ only in the ranker. `p_best` = fraction of decisions where `a*` outranks all 16 candidates,
ties split. `mean_rank` is out of 16; **8.0 is the exchangeability null**, lower is better.
Brackets are 95% cluster-bootstrap intervals over episodes.

`block_still` / `block_moving` is the **demo's own** T motion (`expert_moved`), so the split is
byte-identical across arms. It is 34.0% / 66.0% on all three.

### `block_still` — the demo's chunk does not move the T. **The headline.**

| arm | `t_goal` | Q (sac206) | Q (sac106, clean) |
|---|---|---|---|
| **p_best** ||||
| ST k16 uniform | 0.090 [0.062, 0.122] | **0.452** [0.384, 0.520] | **0.485** [0.418, 0.552] |
| ST k16 flat400 | 0.055 [0.041, 0.071] | **0.390** [0.325, 0.459] | **0.382** [0.322, 0.446] |
| UNet BC | 0.037 [0.027, 0.050] | **0.349** [0.287, 0.417] | **0.346** [0.283, 0.412] |
| **mean_rank** ||||
| ST k16 uniform | 9.48 [8.93, 10.03] | 7.01 [5.97, 8.02] | **6.21** [5.22, 7.19] |
| ST k16 flat400 | 9.20 [8.77, 9.63] | 7.51 [6.38, 8.58] | **6.87** [5.92, 7.86] |
| UNet BC | 9.43 [8.98, 9.91] | 7.03 [6.04, 8.04] | **6.32** [5.40, 7.26] |

**6 of 6 rows rise, by 4.3×–9.4×, and not one confidence interval overlaps.** That is the
distinction from §5: V beat the heuristic on `block_still` in 6 of 6 too, but every interval
overlapped and the report had to lean on the consistency of the sign. Here the intervals separate.

### `all` — every decision

| arm | `t_goal` | Q (sac206) | Q (sac106, clean) |
|---|---|---|---|
| **p_best** ||||
| ST k16 uniform | 0.151 [0.122, 0.179] | 0.271 [0.236, 0.308] | **0.290** [0.255, 0.326] |
| ST k16 flat400 | 0.170 [0.142, 0.198] | 0.216 [0.186, 0.247] | **0.236** [0.206, 0.268] |
| UNet BC | 0.055 [0.038, 0.074] | **0.179** [0.150, 0.207] | 0.169 [0.140, 0.199] |
| **mean_rank** ||||
| ST k16 uniform | 8.64 [8.19, 9.09] | 7.71 [7.21, 8.24] | **7.26** [6.74, 7.78] |
| ST k16 flat400 | 7.46 [7.02, 7.90] | 8.31 [7.80, 8.84] | 7.33 [6.86, 7.77] |
| UNet BC | 8.76 [8.34, 9.20] | 7.66 [7.18, 8.16] | **7.52** [7.03, 7.99] |

**`p_best` rises in 6 of 6.** §5's verdict on V was "overall it is a wash" — 4 rows up by ≤0.021,
2 down. Q is not a wash: the smallest gain is +0.046 and the largest is +0.124. `mean_rank`
improves in 5 of 6; the exception is flat400 under sac206 (7.46 → 8.31).

### `block_moving` — the demo's chunk moves the T. **Q does not dominate here.**

| arm | `t_goal` | Q (sac206) | Q (sac106, clean) |
|---|---|---|---|
| **p_best** ||||
| ST k16 uniform | 0.182 [0.143, 0.222] | 0.178 [0.145, 0.213] | 0.189 [0.156, 0.225] |
| ST k16 flat400 | **0.229** [0.187, 0.271] | 0.127 [0.098, 0.157] | 0.161 [0.131, 0.193] |
| UNet BC | 0.063 [0.039, 0.089] | **0.091** [0.067, 0.116] | 0.078 [0.058, 0.100] |
| **mean_rank** ||||
| ST k16 uniform | 8.22 [7.58, 8.85] | 8.07 [7.52, 8.64] | **7.81** [7.28, 8.35] |
| ST k16 flat400 | **6.56** [5.98, 7.15] | 8.73 [8.17, 9.29] | 7.56 [7.06, 8.08] |
| UNet BC | 8.42 [7.86, 9.00] | **7.98** [7.47, 8.51] | 8.14 [7.61, 8.66] |

**`p_best` splits 3 up / 3 down, and `mean_rank` 4 up / 2 down** — against V's 6-of-6 loss. The one
clear loss is **ST k16 flat400**, and it is the arm where `t_goal` is strongest (0.229 `p_best`,
6.56 `mean_rank` — the only cell in the whole table meaningfully below the null). Where the
heuristic measures the optimised quantity directly *and well*, a coarse learned scalar does not
improve on it. Where `t_goal` is mediocre (uniform 0.182, BC 0.063), Q roughly matches it.

Because `block_moving` is 66% of decisions, this stratum is what keeps the `all` gains modest
relative to the `block_still` gains.

---

## 2. `t_goal` is *worse than chance* where the demo does not move the T

Read the `block_still` `mean_rank` column again: **9.20–9.48, with every interval entirely above
8.0.** This is not the heuristic being uninformative; it is the heuristic being **anti-correlated
with the expert**.

The mechanism is in the definition. `value_t_goal` is `-t_goal_distance(feedback)` — a function of
where the T ends up and nothing else. On these decisions the demonstrator's own chunk leaves the T
where it is, so `a*` scores exactly the reference value. Any candidate that happens to shove the
block goalward scores *better*. `argmax` therefore systematically prefers a candidate that deviates
from the demonstration, and does so on 34% of all decisions.

Two corollaries worth recording:

- The 09-17 report framed the problem as `t_goal` being **silent** before contact (the `blind`
  class, 15–25%). The state-defined `block_still` split says it is worse than silent on a larger
  stratum: silent would be 8.0, and this is 9.4.
- On the candidate-defined `blind` class the UNet BC row reads `p_best` **0.059 with a zero-width
  interval** — exactly 1/17. That is the arithmetic signature of `a*` tying all 16 candidates on
  every blind decision, which is what "blind" means and a useful check that the stratum is what it
  claims.

---

## 3. The contamination control — the leakage did **not** manufacture the result

sac206 had all 206 episodes in its replay buffer, so every one of these 100 episodes is training
data for it. sac106 never saw them. If the Q advantage were leakage, sac206 would beat sac106.

**It does not.** On `all` `p_best`, sac106 wins 2 of 3 (0.290 vs 0.271, 0.236 vs 0.216; BC is
0.169 vs 0.179). On `all` `mean_rank`, sac106 wins **3 of 3**. On `block_still` `mean_rank` it wins
**3 of 3**, by 0.64–0.80 rank positions.

This is the more striking because sac106 is the **weaker** model by every training-time measure: it
was cancelled at 5M against sac206's completed 10M, this checkpoint is from 3.3M, and its own
policy eval was collapsing around it (§ caveats). A half-trained Q from a failing run, with no
access to the evaluation demos, ranks the expert action better than a fully-trained Q that had them
all. **The `block_still` result is therefore not an artefact of demo leakage.**

It is *not* an explanation of why. Nothing here separates "contamination does not help this task"
from "the 8.5M Q has specialised toward its own actor's action distribution and away from the
demonstrator's". Both are consistent with the table.

---

## 4. Best-of-N success — `t_goal` wins on all six rows

`sac/eval.py bon-sweep`, n = 1…16, `argmax`, 30 held-out episodes of
`pusht_seed42_train176.json`, paired episode by episode: the same per-episode sampling noise
under both rankers, so the only thing that differs is the ranker. Episodes come from each
**policy's** own manifest here, which is correct — best-of-N must be scored on episodes the
policy held out.

Every cell is **success rate / mean max goal coverage**. Coverage is the raw
`intersection(block, goal) / goal_area` at its episode maximum, and success is coverage > 0.95;
the two are reported together because they can and do move in opposite directions.

`t_goal` is listed once per arm: with every job pinned to one GPU model, the two Q jobs of an
arm return the **same** `t_goal` curve — every success rate exactly equal, and coverage equal to
within 3e-16 (see below). That is the cross-job check.

| arm | ranker | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|---|--:|--:|--:|--:|--:|
| ST k16 uniform | `t_goal` | 0.233 / 0.534 | 0.167 / 0.646 | 0.333 / 0.652 | 0.333 / 0.655 | **0.433** / 0.725 |
| | `q` sac206 | 0.233 / 0.534 | 0.133 / 0.619 | 0.133 / 0.568 | 0.133 / 0.616 | **0.267** / 0.606 |
| | `q` sac106 | 0.233 / 0.534 | 0.100 / 0.585 | 0.267 / 0.689 | 0.200 / 0.709 | **0.300** / **0.756** |
| ST k16 flat400 | `t_goal` | 0.100 / 0.464 | 0.167 / 0.727 | 0.467 / 0.826 | 0.467 / 0.841 | **0.433** / 0.754 |
| | `q` sac206 | 0.100 / 0.464 | 0.133 / 0.592 | 0.167 / 0.674 | 0.100 / 0.524 | **0.200** / 0.637 |
| | `q` sac106 | 0.100 / 0.464 | 0.067 / 0.562 | 0.100 / 0.596 | 0.233 / 0.664 | **0.200** / 0.670 |
| UNet BC | `t_goal` | 0.100 / 0.485 | 0.333 / 0.671 | 0.367 / 0.760 | 0.600 / 0.863 | **0.467** / 0.849 |
| | `q` sac206 | 0.100 / 0.485 | 0.100 / 0.502 | 0.167 / 0.492 | 0.200 / 0.566 | **0.033** / 0.587 |
| | `q` sac106 | 0.100 / 0.485 | 0.067 / 0.617 | 0.333 / 0.658 | 0.200 / 0.608 | **0.267** / 0.739 |

**Q is a working ranker, and still the worse one.** Measured from each row's own n=1, `q` gains
in 5 of 6 rows; the exception is UNet BC under sac206, which falls to 0.033. That is the crucial
difference from V, which fell in **6 of 6** and ended at 0.000–0.060: V was pointed the wrong
way, Q is pointed the right way and is simply weaker. `t_goal` gains more on every arm, and at
n=16 it leads on success in **6 of 6 rows**, by 0.133 to 0.434.

**But on coverage the gap nearly closes, and on one row it inverts.** At n=16, `q` sac106 on
ST k16 uniform reaches **0.756 mean coverage against `t_goal`'s 0.725** — higher coverage, and
yet 0.300 success against 0.433. Q pushes the T closer to the goal on average and converts fewer
episodes across the 0.95 line. The same shape is visible on UNet BC (0.739 vs 0.849 coverage but
0.267 vs 0.467 success) and is what §6 pins down on the one row where it is extreme.

**This is why coverage is reported.** A success rate alone reads "Q is a much worse verifier";
success and coverage together read "Q optimises the right quantity and stops short of the
threshold", which is a different defect with a different fix.

**Section 1's advantage does not survive the trip.** Q's edge is concentrated on `block_still`,
where `t_goal` is blind — but those are the decisions where, by construction, no candidate moves
the T and so the choice matters least for the episode outcome. `block_moving` is 66% of
decisions and is where `t_goal` is at least as good; that is the stratum that decides episodes.

### What the policies were trained with — `t_goal` is not a neutral opponent here

**All three arms were trained with `t_goal` as their verifier.** Every run carries
`verifier_tag: t_goal` in its hydra config, which propagates to `policy.verifier_value`; it is
also the `ver-t_goal` in each run name. Section 4 is therefore not a symmetric contest between
two rankers — it asks whether Q can beat the heuristic **on candidates produced by a policy that
was trained under that heuristic**. Three ways that cuts:

| arm | `n_search_actions` in training | `search_context` | reads the context? |
|---|---|---|---|
| ST k16 uniform | **16** (`n_candidates`) | `value` = `t_goal`'s scalar | yes |
| ST k16 flat400 | **16** | `value` = `t_goal`'s scalar | yes |
| UNet BC | **1** — no search at all | — | no |

1. **The candidate distribution is `t_goal`-shaped for all three.** The sampler was trained to
   produce chunks that a `t_goal` search would select among, so the pool Q is handed is one the
   heuristic has already had a hand in choosing.
2. **The two ST arms were trained to condition on `t_goal`'s own value.** `search_context: value`
   means candidate *i* is generated conditioned on the `t_goal` scores of candidates `< i`. At
   eval, `install_q_ranker` ([sac/eval.py:85-110](sac/eval.py#L85-L110)) replaces **only the
   ranking scalar** — `return context, qv, subgoal, terms` — deliberately leaving the context
   `t_goal`-shaped, because feeding a Q-shaped context to a network never trained on one would
   confound "Q ranks worse" with "ST was handed an input it has never seen". So on the ST rows Q
   is choosing among candidates that were *generated under `t_goal`'s running commentary*. At
   n=16 the k16 arms are also at exactly their trained search width.
3. **UNet BC is the one clean read**, and it is the only arm with no `t_goal` machinery at eval:
   trained at `n_search_actions: 1` with no search, it ignores the context entirely, which is why
   it is run with `--skip-context-sim`. It is also where Q does **worst** — 0.033 at n=16 under
   sac206 against `t_goal`'s 0.467. Removing the structural advantage did not rescue Q.

That last point is what stops this from being an excuse. The confound is real and should be
stated, but the arm where it is absent is the arm where Q loses hardest, so "Q only lost because
the policies were trained under `t_goal`" is not supported by these six rows.

**Both leak checks pass.** At n=1 there is nothing to rank, and `t_goal` and `q` return identical
success on all six jobs. The contamination contrast is also flat here: sac106 (clean) and sac206
end at the same 0.267 / 0.200 on the ST arms, so nothing in section 4 is explained by leakage
either.

⚠️ **30 episodes** → 95% Wilson interval ≈ ±0.17, and one episode is 0.033. Only the trend in n
is paired and worth reading; the UNet BC `q` collapse to 0.033 at n=16 is a single episode.
⚠️ **20 of those 30 are inside the 106**, so this section is clean on 10 of 30 even for sac106.
**Section 1 is the clean comparison; this is not.**

### An empirical noise floor, and the re-run that closed it

`t_goal` is deterministic given seed, split and `n_envs`, so the two jobs of an arm must return
the identical `t_goal` curve. In the first pass **one of three pairs did**, and the two that did
not had been scheduled onto different GPU models — same model ⇒ bit-identical, different ⇒
differs, 3 for 3. [eval_search_pusht.py:73-87](eval_search_pusht.py#L73-L87) disables TF32 and
sets `cudnn.deterministic`, and its claim that the eval is "bit-identical run to run" is verified
here — **on fixed hardware**. Across GPU models the reduction order still differs and a 300-step
contact sim amplifies it.

Every sweep above was therefore re-run pinned to one GPU model, and **all three pairs now agree
on every success rate**, which is why §4 lists `t_goal` once per arm. Four of the twelve original
rows moved, each by one or two episodes:

| row | published | pinned re-run |
|---|---|---|
| k16 uniform / sac106 / `t_goal` | 0.167 0.133 0.333 0.333 0.400 | 0.233 0.167 0.333 0.333 0.433 |
| k16 uniform / sac106 / `q` | 0.167 0.100 0.300 0.200 0.267 | 0.233 0.100 0.267 0.200 0.300 |
| UNet BC / sac206 / `t_goal` | n=8 **0.633** | n=8 **0.600** |
| UNet BC / sac206 / `q` | n=2 **0.133** | n=2 **0.100** |

All four moved *toward* their pinned partner, and no conclusion in §4 or §6 changes. The useful
residue is a **measured run-to-run noise floor of 1–2 episodes of 30 (0.033–0.067)**, independent
of the Wilson interval: differences of one or two episodes anywhere in this report carry no
information. `GPUS=rtx6k:1` in `scripts/slurm/q_arms.sh` pins it.

**"Bit-identical" is slightly too strong, even on one GPU model.** On the pinned re-run the
k16 flat400 pair differs on **2 of 30 episodes by up to 3.3e-16** — a few float64 ULPs, on the
same node, so it is the CPU contact sim's summation order rather than anything on the GPU. It
changes no success rate and no figure reported here (the largest effect is the last bit of one
mean, 0.7543613977878553 vs ...54). `eval_search_pusht`'s own docstring claims the eval is
"bit-identical run to run (verified)"; at three decimals it is, at 1e-16 it is not.

## 5. The final-candidate readout — what the search is worth with nothing selected

`--selection final_pass`: `n-1` candidates are generated and scored, and the **n'th generation,
conditioned on them, is returned unexamined**. The verifier scalar never touches selection, so
this is the verifier-off control — it isolates what widening the search does to the *sampler*
from what it does to the *selector*. Every ranker gives the same curve under it, so one job per
arm. Seed 42, all three pinned to the same GPU model.

Cells are **success / mean max coverage**.

| arm | n=1 | n=2 | n=4 | n=8 | n=16 | `argmax`/`t_goal` at n=16 |
|---|--:|--:|--:|--:|--:|--:|
| ST k16 uniform | 0.233 / 0.534 | 0.167 / 0.642 | 0.100 / 0.644 | 0.100 / 0.607 | 0.167 / 0.571 | 0.433 / 0.725 |
| ST k16 flat400 | 0.100 / 0.464 | 0.100 / 0.647 | 0.200 / 0.659 | 0.133 / 0.615 | 0.233 / 0.663 | 0.433 / 0.754 |
| UNet BC | 0.100 / 0.485 | 0.133 / 0.535 | 0.200 / 0.583 | 0.167 / 0.528 | 0.133 / 0.604 | 0.467 / 0.849 |

**The unaided draw does not improve with n on any arm** — success wanders inside 0.100–0.233
with no trend, and coverage is equally flat (BC 0.485 → 0.604, k16 uniform 0.534 → 0.571, and
k16 uniform actually *falls* from its n=4 peak of 0.644). So on
these 30 episodes essentially **all** of best-of-N's gain is selection, not the search context
sharpening the policy's own draw. That is consistent with
[tgoal_moving_arms_2026-09-19.md](tgoal_moving_arms_2026-09-19.md) §1, which measured the same
thing on verifier value at 100k and found the *uniform* and BC arms flat-to-negative
(k16 uniform −0.02, BC −0.19) while only the ladder arms improved.

It also fixes the scale for section 4: at n=16 the ordering is
**`t_goal` 0.433–0.467 > `q` 0.200–0.267 > nothing-selected 0.133–0.233**. Q is a real ranker —
it beats not selecting at all on 5 of 6 rows — and still well short of the heuristic.

---

## 6. Why UNet BC drops at n=16 — two different answers

The BC row falls from n=8 to n=16 under both rankers, and they turn out to be unrelated. Both
Qs were re-run at seed 42 and at seed 1234, pinned to one GPU model, recording solved counts and
mean max goal coverage per n.

Cells are **episodes solved (of 30) / mean max goal coverage**. Coverage is now recorded raw
rather than inferred from reward: reward is `clip(coverage / 0.95, 0, 1)`
([pusht_env.py:132](diffusion_policy/env/pusht/pusht_env.py#L132)) and saturates at 1.0 from
0.95 coverage up, so a mean reward is right-censored exactly where the solved episodes are —
which is the half of the distribution this section is about. `PushTEnv` now keeps
`max_coverage` over the episode, read out the same way `reward` already is.

| BC, 30 episodes | n=1 | n=2 | n=4 | n=8 | n=16 |
|---|--:|--:|--:|--:|--:|
| `t_goal` seed 42 | 3 / 0.485 | 10 / 0.671 | 11 / 0.760 | **18** / 0.863 | **14** / 0.849 |
| `t_goal` seed 1234 | 6 / 0.631 | 8 / 0.703 | 12 / 0.698 | **14** / 0.794 | **16** / 0.839 |
| `q` sac206 seed 42 | 3 / 0.485 | 3 / 0.502 | 5 / 0.492 | **6** / 0.566 | **1** / 0.587 |
| `q` sac206 seed 1234 | 6 / 0.631 | 8 / 0.616 | 7 / 0.541 | **8** / 0.573 | **2** / 0.546 |
| `q` sac106 seed 42 | 3 / 0.485 | 2 / 0.617 | 10 / 0.658 | 6 / 0.608 | 8 / 0.739 |
| `q` sac106 seed 1234 | 6 / 0.631 | 5 / 0.602 | 6 / 0.671 | 7 / 0.665 | 9 / 0.715 |
| `final_pass` | 3 / 0.485 | 4 / 0.535 | 6 / 0.583 | 5 / 0.528 | 4 / 0.604 |

**Under `t_goal` it is noise.** The 18 → 14 fall at seed 42 becomes a 14 → **16 rise** at seed
1234, and coverage barely moves either way (0.863 → 0.849, 0.794 → 0.839). Recall that
[`_episode_seed`](eval_search_pusht.py#L342) puts `n` in the per-episode seed *deliberately* —
"the n=1 and n=16 columns should be independent draws rather than n=1 being a prefix of n=16" —
so consecutive columns are separate samples, not nested ones, and four episodes of thirty is
well inside that. **Nothing to explain.**

**Under `q` from sac206 it is real, and it is the selector.** 6 → 1 at seed 42 and 8 → 2 at seed
1234: it reproduces, and it is a 4–6× collapse, far outside the 1–2 episode noise floor. Three
facts locate the cause:

- **The candidate pool is not degrading.** `final_pass` is flat in n (3, 4, 6, 5, 4), so
  generating 16 chunks instead of 8 does not give BC worse chunks to choose from. What changed is
  the choosing.
- **The T does not end up further away — it stops crossing the line.** Across the n=8 → n=16
  fall, mean coverage **rises** at seed 42 (0.566 → 0.587) and barely moves at seed 1234
  (0.573 → 0.546), *while solves go 6 → 1 and 8 → 2*. The Q gets the block about as close as
  before, or closer, and converts almost none of it. Compare `t_goal` at the same n=16: coverage
  0.849 against `q`'s 0.587 — on **this** arm `t_goal` is better on both axes, which is why BC is
  where the collapse is extreme. §4 shows the other shape, where `q` leads on coverage and still
  loses on success.
- **It is specific to sac206.** sac106 shows no collapse at either seed (8 and 9 at n=16, above
  `final_pass`'s 4). The two Qs differ in demo seeding and in training length, and this result
  does not separate those.

**That is over-optimisation, not a weak ranker.** `argmax` over 16 independent BC draws is a
harder search against the Q's error surface than `argmax` over 8, so the wider the search the
more reliably it finds the candidate the Q most overrates. It is the best-of-N failure mode the
`q_spread_zero_frac` / abstention machinery in `sac/score.py` was built for, and
**at n=16 selecting by sac206's Q is worse than selecting nothing at all** — 1 solve against
`final_pass`'s 4.

---

## 7. What this report set running — status 2026-09-20

Both of the caveats below turned into work, and neither has results yet.

**The contamination caveat is being removed structurally, not argued around.** Two per-split SAC
arms are training with `--demo-episodes train`, each seeded only from its own manifest's train
episodes, so that manifest's test episodes are held out by construction rather than by picking a
manifest the buffer happens to miss:

| arm | manifest | train | steps so far |
|---|---|---|---|
| `sac_keypoint_blq137` | `pusht_blockquad_bottomleft_train137.json` | 137 | 4.24M / 10M |
| `sac_keypoint_brd100` | `pusht_blockborder_train100_core50.json` | 100 | 0.56M / 10M |

Readout: [sac_per_split_q_2026-09-20.md](sac_per_split_q_2026-09-20.md).

**The "`t_goal` is not a neutral opponent" confound (§4) is being removed too.** Seven policy
arms are training with the Q *as their verifier* — `verifier_tag: q_sac_all`, `search_context:
value`, so the search context they condition on is Q-shaped rather than `t_goal`-shaped:

`value_k{16,4}_ver-q_sac_all[_son-{flat400,ramp400to200}]` on `blq137`, plus
`value_k16_ver-q_sac_all` on `brd100`.

`q_sac_all` resolves to `sac_keypoint/model_8500000_steps.zip`
([pusht_verifier.py:370](diffusion_policy/env/pusht/pusht_verifier.py#L370)) — the same
checkpoint §1 and §4 used — with `q_sac_106` as the replay-buffer control. A policy trained under
Q and searched by Q is the comparison §4 could not make.

**§6 is a caution for those arms, not a prediction about them.** The over-optimisation measured
there was `argmax` at eval against a Q the policy was *not* trained under. Training the sampler
against the same Q changes the failure mode rather than obviously removing it, and the
diagnostic that would detect it is the `final_pass` control in §5 — cheap, and worth running
beside every one of those arms rather than after the fact.

---

## What this does not establish

- **It does not show that no learned Q can rank best-of-N.** It shows that *these two* Qs, from
  a keypoint SAC arm, ranked by `argmax` on the raw scalar, lose to the heuristic on the
  deployable metric while beating it on the expert-ranking one.
- **`t_goal` is the verifier all three policies were trained under**, so section 4 is not a
  neutral contest; see §4. The confound does not explain the result — Q loses hardest on the one
  arm that carries none of it — but no row here is free of it.
- **Why the two sections disagree is not established here.** The `block_still` stratum where Q
  wins and the `block_moving` stratum that decides episodes are the obvious suspects, but nothing
  in these runs isolates that. A per-decision diagnostic on the sweep's own trajectories would.
- **§6 does not separate demo seeding from training length.** sac206 collapses and sac106 does
  not, but they differ in both (all-206 vs 106, and 8.5M vs 3.3M steps). A 3.3M checkpoint of
  `sac_keypoint` would separate them and has not been run.
- **§5 is one seed on 30 episodes.** "The unaided draw does not improve with n" is a claim about
  these curves, which move 0.033 per episode; it agrees with the 100k measurement in the
  moving-arms report but is not independently powered.
- **Hold-out is of demo seeding only.** SAC also collects online rollouts from randomly-reset envs,
  so no episode's *state distribution* is unseen by either Q.
- **sac106 @3.3M is not its peak policy.** Its uniform eval reads 0.55 @3.31M, 0.80 @3.36M, then
  **0.0 at every eval from 3.41M to the 5M cancellation**. Checkpoints save every 100k, so none
  sits inside that spike, and 3.3M is the rising edge.
- **sac206's 1.00 is a 20-episode eval**, one episode above its 0.90–0.95 neighbours; the honest
  plateau is ~0.90. It is the uniform eval, parsed from the `eval/` block — `rollout/` carries a
  same-named key.
- **90 of the 100 episodes are in the policies' own training set.** That shapes the candidate
  distribution, not the Q-vs-`t_goal` comparison — both rank identical candidates. But `a*` here is
  on the policies' training distribution, not a held-out one for *them*.
- **Candidates are the policy's own draws, not uniform random actions.** The question answered is
  "does Q prefer the expert over what the policy would have done", which is what best-of-N ranks.
  It is not "does Q beat chance".
- **Three arms are not three independent trials.** They share one candidate-generation recipe, one
  episode set and one pair of Q checkpoints.

## Reproduce

```bash
SUBMIT=1 bash scripts/slurm/submit_q_rank_expert_non106.sh   # section 1, ~10 min/job
SUBMIT=1 bash scripts/slurm/submit_q_bon_mv.sh               # section 4, ~40 min/job

# section 5, the verifier-off control (the Q is inert, so one job per arm)
SUBMIT=1 SELECTION=final_pass TAG=_finalpass RANKERS=t_goal GPUS=rtx6k:1 \
    bash scripts/slurm/submit_q_bon_mv.sh
```

Both launchers read their arm and Q lists from `scripts/slurm/q_arms.sh`, so the two sections
cannot drift onto different checkpoints. Outputs are on
`/gscratch/robotics/harine/value_arms/{qeval_non106,bon_q_mv}/`, not under the repo.
