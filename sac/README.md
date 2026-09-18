# A learned Q as the PushT best-of-N verifier

Best-of-N search in this repo ranks candidate action chunks with a hand-written distance
heuristic, evaluated by rolling each candidate forward in a deterministic sim
(`PushTVerifier`, `VALUE_FNS`). It has a structural blind spot:

* `value_t_goal` **ignores `agent_pos` by construction** (`pusht_verifier.py:136`), so before
  the arm touches the block every candidate returns the identical value -- a measured 0.0000 px
  spread over 8 candidates -- and `argmax` degenerates to numpy's first-maximizer tie-break.
* Blind on **28-35% of all decisions**, rising to **68-73% during approach** (`docs/reports/archive/ASTAR_RECALL.md`).
  About a third of argmax picks are decided by the tie-break rather than by the verifier.
  **Re-measured 2026-09-17 on the six step-30k geometric-split checkpoints at n=16: 15-25%**
  (`docs/reports/ppo_sac_lstm-bc_eval_2026-09-17.md`, section 1). Different generation, different
  measurement -- not a contradiction, but the blind region is ~40% smaller than the figure above,
  so use the dated one when judging a new verifier against it.
* `armTn` patches this with an arm-to-T term, but that is a *proxy for* progress: the expert
  routinely swings the arm **around** the T to set up the next push, raising `d_arm_t` while
  lowering `d_t_goal`. Raw `armT` cost the UNet BC arm its whole best-of-n gain, 0.460 -> 0.060
  at n=8.

A Q learned under a **sparse** reward has no such blind spot in principle: `Q*(s, a)` is the
discounted time to solve, so an approach chunk that sets up a good push scores higher *because
the Bellman backup propagates the eventual success back to it* -- no hand-written approach term,
and no assumption that "closer to the T" means "better". Whether that holds in practice is the
experiment.

```
config.py   every default, and the flag that sets it
env.py      the task: the action codec, the chunked MDP, the demos as transitions
agent.py    the learner: ChunkSAC, the BON verifier head, the tau-carrying buffers
runner.py   running one: the loop, its diagnostics, and the entry point
score.py    the deployment shim -- PushTVerifier's signature, backed by the learned Q
eval.py     the evaluations, as subcommands
```

Six files, one subject each. An earlier layout had eighteen, several of them holding a single
class; the seams that survived are the ones where two halves must NOT drift -- `env.py` keeps
the codec beside the env and the demo loader because all three define what a transition IS, and
`agent.py` is separate from `env.py` because the learner and the task are genuinely independent.

`sac/` imports `recurrent_ppo.pusht_gym`, `.run_io` and `.corrupt_policy`, and keeps its own
defaults. One env definition, so the PPO and SAC arms measure the same task; but separate
defaults, because `reward`, `gamma`, `max_episode_steps` and `block_zero_coverage` all mean
something different here and sharing them would silently give this package the wrong task.

## Running

```bash
python sac/runner.py --obs keypoint
python sac/runner.py --obs image --num-envs 8
python sac/runner.py --obs keypoint --tau-ladder 0.95    # the single-head version
```

Runs land in `logs/sac/<obs>/<timestamp>/`, claimed exclusively, with `params/args.yaml`
recording what the run WAS. `--checkpoint` resumes in the checkpoint's own directory and
refuses any flag that would change what the checkpoint is (`IDENTITY_KEYS`).

## The four decisions, and the measurements behind them

### 1. The action is the chunk, and it is absolute

Q is a drop-in for `PushTVerifier.get_value(obs_dict, action)`, whose `action` is `(B, 8, 2)`
absolute pixel targets -- the window `_verifier_inputs` slices and the policy executes. So Q's
action is that chunk, and TD backs up with `gamma_chunk = gamma_base**8`.

**Invertibility is the requirement**, not a nicety: a verifier that cannot express the chunk it
is handed is not a verifier. Measured over all 24,208 eight-step demo windows -- fraction NOT
representable exactly:

| parameterisation | lost |
|---|---|
| **absolute over `[0, 512]`** | **0.00%** |
| absolute over `AGENT_BOUNDS` `[22, 490]` | 1.16% |
| increment chain at 64 px | 2.20% |
| increment chain at 33 px (`recurrent_ppo`'s measured `delta_scale`) | **26.01%** |

At the PPO default a quarter of real expert chunks could not be written down in Q's
coordinates. An increment chain anchors each target on the previous one, which makes the action
a *shape* rather than a location and buys translation equivariance -- a real argument, and the
reason it was implemented. It was then removed: at any scale small enough to be a useful
exploration prior it cannot express what the expert did, and a verifier that silently clips the
chunk it was asked about is worse than one that is merely coarse. Because absolute needs no
anchor, `encode`/`decode` are pure per-element maps -- no agent position, no mode, no scale.

Two deliberate divergences from `recurrent_ppo`, both forced by ST-compatibility: the clip is to
the **arena** `[0, 512]`, not `AGENT_BOUNDS` (demo targets reach 511 px and the ST eval path does
not clip), and the clip is part of the codec on both sides, so two candidates differing only
outside the arena -- which drive byte-identical trajectories -- receive the same Q.

### 2. Sparse reward, and the fact that reshapes it

`+1` on the first coverage > 0.95, terminate. Then `Q* ~ gamma^(chunks to solve)` and any
pre-contact signal is *earned* by the TD backup, so it cannot be dismissed as a re-encoded
distance heuristic. That is the whole argument for the package, so the other reward modes are
diagnostic arms and must never be reported as the headline Q.

**But no demonstration ever solves PushT by that criterion.** Replaying every demo state through
`PushTEnv`:

| max coverage in a demo episode | p5 | p50 | p95 | **max** |
|---|---|---|---|---|
| | 0.798 | 0.851 | 0.883 | **0.9018** |

| threshold | demo episodes reaching it |
|---|---|
| 0.80 | 93.7% |
| 0.85 | 51.0% |
| 0.90 | 1.0% |
| **0.95** | **0.0%** |

So a buffer seeded with demonstrations and a single 0.95 head carries **zero** positive reward,
and a Q regressed on it learns `Q = 0` -- the identical "same score for every candidate"
degeneracy as the heuristic being replaced, reached more expensively. That is not hypothetical:
an earlier PPO generation ran **18 arms to 10M steps and not one ever recorded a non-zero
evaluation success rate**. Those runs and their write-up were retired on 2026-09-16 when the
arms were retrained, so the record of them is in git history rather than on disk -- but the
finding is why this package exists, and it is the bar `bon/q_spread_zero_frac` is watched
against.

**The tau ladder** is the answer. Heads at {0.80, 0.85, 0.90, 0.95} share a trunk; rung `tau`
sees the terminate-at-tau MDP exactly -- its own `(reward, done)` and a `live` mask dropping
transitions from episodes that crossed `tau` earlier. `Q_0.95` is therefore **unchanged** by the
others' presence; the lower rungs only shape the trunk, and they have signal the demonstrations
already contain (658 and 226 terminals at 0.80 and 0.85, against **0** at 0.95).
`--tau-ladder 0.95` recovers the single-head version, and the demo loader prints
`<-- NO POSITIVE REWARD; this rung teaches Q = 0` at load rather than after a week.

### 3. The curriculum is the only source of a 0.95 terminal, and it had to be retuned

Measured coverage at reset under `block_near_goal_prob=1.0`:

| `block_goal_offset` | mean coverage | > 0.85 | **> 0.95** |
|---|---|---|---|
| **[30, 0.25]** (`recurrent_ppo`'s default) | 0.480 | 2% | **0%** |
| [8, 0.06] | 0.843 | 43% | 3% |
| **[3, 0.02]** (the default here) | 0.941 | 100% | 33% |

The inherited default sits at coverage 0.48 and never crosses the threshold it exists to reach.
The tight end is annealed out to `[20, 0.15]` over `--curriculum-anneal-frac` of training, so
the final Q is on the real start distribution and evaluation never uses the curriculum at all.

**An episode must not begin solved.** At `[3, 0.02]` a third of draws already exceed 0.95, and
`PushTEnv` then pays the sparse `+1` on the first step for doing nothing: measured, a do-nothing
policy returned **0.349**, most of the available reward, for free. `--max-reset-coverage`
rejects those draws; the same rule makes the ladder refuse to credit a rung the episode starts
above. With it, the do-nothing return falls to 0.082 and is earned.

Two notes on the inherited machinery. `block_zero_coverage` must be **False** here: it rejects
every near-goal draw, burns all 20 `SPAWN_TRIES`, warns, and then falls through to the last draw
anyway -- 20 wasted resets and warning spam rather than, as it first appears, a no-op.
And `max_episode_steps` is **304**, the nearest multiple of 8: a ragged final chunk would
bootstrap at `gamma**4` while the code says `gamma**8`, silently, and worst at the truncation
boundary where the critic has least data.

### 4. The verifier head is not SAC's critic

SAC learns the entropy-augmented `Q_soft^pi` -- the value of its own stochastic policy plus
`-alpha log pi` at every future step. With a sparse reward of magnitude 1.0 and ~38 chunks of
entropy accumulating against it, the soft value can be mostly entropy. Best-of-N does not
evaluate a stochastic policy; it takes an argmax over n candidates and executes it. So the
shipped verifier is a separate head whose backup is the operation the deployment performs:

```
y = r + gamma * (1 - d) * max_{a' in A_M(s')} mean_ensemble Q_target(s', a')
A_M(s') = { tanh(mu(s')) } u { M - 1 actor samples }
```

Built **after** the policy's own optimizers and with its features detached -- the construction
`recurrent_ppo.corrupt_policy._QHeadMixin` uses, so the isolation is a fact about construction
order rather than a promise, and it is tested. The SAC actor is a data-collection engine and an
`a'` proposer, not something the verifier co-adapts with.

## Off-distribution accuracy is the crux

Q is asked to rank **diffusion-policy** candidates, which this actor never proposes. A buffer
fed only by the actor yields a Q that is sharp exactly where it is not needed and flat where it
is. So `_sample_action` draws from a mixture -- actor 0.55, uniform 0.20, re-anchored demo chunk
0.20, roughened actor 0.05 -- kept flowing *past* `learning_starts`, not only during it.

The demo arm injects a chunk **shape** (targets minus their own start agent position) re-anchored
at the current agent position, so a push the expert performed in one corner is a usable proposal
anywhere; injecting the raw absolute encoding would only ever teach Q about the places the demos
happened to visit.

The override is in `_sample_action` and **not** in an env wrapper, deliberately: SB3 stores the
`buffer_action` that method returns, so a wrapper swapping the action afterwards would write one
action into the buffer and execute another. Nothing raises; the Q just regresses on labels
belonging to different actions.

A conservative (CQL-style) penalty is explicitly rejected: it pushes Q *down* on actions the
behaviour policy did not take, and best-of-N's entire job is to find the good candidate the
actor did not propose.

## What to watch, and the number to beat

| metric | says |
|---|---|
| `bon/q_spread_zero_frac` | fraction of states where Q scores every candidate identically. **The heuristic's own blind rate is 28-35% as first measured, 15-25% when re-measured on the step-30k geometric checkpoints** (see above). If this is not far below that, the new verifier has the old verifier's disease. `sac_keypoint` reads **0.0** at every probe through 1.4M steps. |
| `bon/q_resolution` | within-state spread / across-state spread. All n candidates share a state and differ by at most one chunk, so if this is tiny they sit inside the regression's noise floor and `--gamma` is too high. |
| `buffer/reward_rate_tau*` | per rung. **If the 0.95 row is 0, everything downstream is vacuous** -- and it says so on day one. |
| `bon/head_loss`, `bon/live_frac_tau*` | the head's own fit, and how much of the buffer each rung still owns. |
| `eval/success_rate`, `eval/max_coverage` | on the real start distribution, never the curriculum's. `max_coverage` is what `pusht_image_runner` reports, so it is the number comparable with the diffusion-policy arms. |

`--gamma` is quoted **per chunk**; the env discount is its 8th root. The tradeoff is explicit:

| `gamma_chunk` | per-chunk contrast | Q at episode start (38 chunks) |
|---|---|---|
| 0.9227 (`gamma_base` = 0.99) | 7.7% | 0.047 |
| **0.95** (default) | **5.0%** | **0.142** |
| 0.97 | 3.0% | 0.314 |

Lower and early-episode states are all ~0 and mutually indistinguishable; higher and the
per-chunk contrast drops below what MSE regression reliably resolves.

## Using the Q as a verifier

`sac/score.py` exposes `PushTQVerifier` with `PushTVerifier`'s signature:

```python
get_value(obs_dict, action) -> (B,)          # action (B, 8, 2) absolute pixel targets
rollout(obs_dict, action) -> (value, state)
```

The state is reconstructed from `agent_pos` + `block_pose_from_feedback(feedback)` -- the sim
verifier's own recipe, imported rather than re-derived, and **verified bit-identical** to
`PushTVerifier._reset_states_from_obs`.

**The keypoint arm needs no simulation at all.** The sim verifier forks a 32-process pool and
steps 8 base steps per candidate; this transforms 9 local keypoints by the block pose (an affine
map) and runs one forward pass. The image arm needs the current frame, which
`_verifier_inputs` currently drops (`_VERIFIER_OBS_KEYS = ('agent_pos', 'feedback')`); adding
`'image'` there is additive and safe, since `PushTVerifier` reads only those two keys.

**Only the ranking is replaced, never the search context.** `_score_candidates` returns
`(context, value, ...)`: `context` is what ST conditions its next candidate on, `value` is what
argmax ranks by. ST was *trained* with a `t_goal`-shaped context, so handing it a Q-shaped one
at eval would confound "Q ranks better" with "ST was given an input it has never seen". BC
(`PushTUNetSearchPolicy`) ignores the context entirely, which makes it the cleaner first read.

`rollout(render=True)` raises rather than guessing: this verifier does not simulate, so it has
no reached subgoal frame, and `search_context in {subgoal, subgoal_value}` must keep the sim
verifier.

## Not built yet

* The four eval scripts -- `rank_expert.py` (where the expert action ranks), `value_over_episode.py`
  (Q over candidates on frames where the arm is not touching the T), `bon_sweep.py` (ST k1 / BC
  30k at n = 1..64, Q vs `t_goal` vs `armTn`), and the ST-candidate bank the off-distribution
  probe needs.
* An 18-d subgoal regression head, which would let the Q serve `search_context in {subgoal,
  subgoal_value}` as well as ranking. Supervisable from the `s'` already in the replay buffer.
* A `ChunkDictReplayBuffer` storing obs once with a +1 index. `next_obs` of chunk *k* is `obs` of
  chunk *k+1* everywhere except episode boundaries, so the image arm's 54.0 KiB/transition is
  about twice what it needs; SB3's `DictReplayBuffer` asserts against `optimize_memory_usage`.

## The four evaluations

All four resolve their episodes through `eval_search_pusht.get_split_states`, i.e. the **50
held-out demo episodes** named by the committed split manifest -- ST and BC's own eval set, with
its checksum validation and `splits.json` cross-check. Not seeded env resets: `recurrent_ppo`'s
"fixed set" is fixed in ST's *style*, not ST's *set*.

```bash
# 2. where the expert action a* ranks, under BOTH verifiers on the SAME decisions
python sac/eval.py rank-expert -c <st_or_bc.ckpt> --q logs/sac/keypoint/<run>/model.zip

# 3. what the verifiers say while the arm is NOT touching the T
python sac/eval.py frames -c <ckpt> --q <sac.zip> --frames 50

# 4. the headline: best-of-N by ranker
python sac/eval.py bon-sweep -c <ckpt> --q <sac.zip> --rankers q,t_goal,armTn --max-n 16
```

`rank-expert` reuses `scripts/verifier_ranks_expert.py`'s `sample_points`, `build_batch`,
`classify` and `_stats` rather than reimplementing them -- including its mid-rank tie handling,
which matters here more than anywhere: on a blind decision a* ties all n candidates, and scoring
that with a strict `>` records the verifier's *absence of preference* as a* losing to
everything. The result is split by the existing blind/partial/informative classes, and the
**blind column is the headline** -- a Q that only matches the heuristic overall but ranks a*
well where the heuristic is silent is already the win.

`bon-sweep` monkeypatches `_score_candidates` **on the policy instance**, so nothing ST and
BC are trained and evaluated with changes on disk and an ordinary run stays bit-for-bit what it
was.

### Checkpoint compatibility, found the hard way

Not every checkpoint on this machine loads under current code. The `offline/` generation
(`stn1_demos-29-r8`, `bc_demos-30`) carries a `shape_meta` that declares `agent_pos` and
`feedback` as policy observations, which `PushTDiffusionSearchPolicy` now refuses -- `feedback`
is an exact transform of the block pose, so encoding it would hand the policy the ground-truth
T pose. Those are verifier-only keys. The `~/pusht_ckpts/` generation loads fine, and
`unetbc_ver-t_goal_enc-resnet18_*` is a `PushTUNetSearchPolicy` with `predict_action_best`.
`unetbc_demos-100_seed-42` is a plain `DiffusionUnetImagePolicy` and has no best-of-n at all.
