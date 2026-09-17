# BC vs ST candidate diagnostics — 2026-08-19

What was rendered and measured, where every file is, and how to re-derive it.

Division of labour, same as `aug9_analysis.md`: **the generated JSON/npz files are
authoritative for values, this document is authoritative for provenance.** Where a number
appears in both, regenerate rather than trust the copy here.

**Status: artifacts COMPLETE and verified** (300 videos, 9 dump cells, 450 per-step JSONs) —
`python scripts/check_aug19_artifacts.py` exits clean. The §5 findings below are written up
where the data settles a question and left open where it does not; §5b/5e/5f still need a
human read of the videos and the spread curves.

The full per-candidate report is `CANDIDATES_AUG19.md` (built from the same nine dumps).

The §2a and §2c tables are **transcribed from disk, not hand-maintained**: run
`python scripts/check_aug19_artifacts.py --tables` to regenerate them. Without `--tables` the
same script verifies every claim this doc makes — file counts, sidecar/file agreement, that
both arms rendered the same episodes, the per-step JSON schema, and that every artifact
carries the expected `verifier_value` — and exits non-zero if any of it is false.

---

## 0. The three questions

The 30-demo arms have full 10k–100k checkpoint sweeps and best-of-n success curves, but
every consumer of the search reduces the per-candidate signal to one number — the env runner
and `eval_search_pusht.py` take the argmax and drop the rest. Three things are therefore
unanswered on disk.

**1. What do the sampled actions actually look like?** Whether ST k=1's n candidates spread
differently from UNet BC's i.i.d. draws, and whether the argmax pick differs from the
last-generated one, is invisible in a success rate. → §2a.

**2. What are the candidates' distance values?** Specifically: when the argmax genuinely
discriminates, *which slot wins* — and is that distribution distinguishable from the i.i.d.
null that BC and ST k=1 provide by construction. → §2b, §2c.

**3. Can the per-slot loss weight be shaped?** Only a single geometric decay
(`slot_weight_decay`) was expressible. Linear, explicit-list and step-scheduled profiles were
not. → §2d.

Plus one piece of cleanup that was blocking honest checkpoint reporting: topk retention.
→ §2e.

---

## 1. Arms and checkpoints

Root: `/gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search`

| label | run dir | policy class | trunk | steps on disk |
|---|---|---|---|---|
| `unetbc` | `unet_bc/unetbc_demos-30_seed-42` | `PushTUNetSearchPolicy` | UNet, 293M | 10k…100k |
| `stk1` | `offline/bc_demos-30_seed-42` | `PushTDiffusionSearchPolicy` | 4/4/256, ~14M | 10k…100k |
| `stk16` | `outer_inner/value_k16_corrupt-False_demos-30_seed-42` | `PushTDiffusionSearchPolicy` | 4/4/256, ~14M | 10k…100k |

All three at `n_demos=30`, `seed=42`, split file
`diffusion_policy/config/splits/pusht_seed42_train30.json` (50 test / 30 val / 30 train).

**Why these three and not the big transformers.** The 6×8×1024 runs
(`offline/value_k1_arch-6x8x1024_...`, `outer_inner/value_k16_arch-6x8x1024_...`) exist, but
the big k=16 run stops at **step 40000** — so it cannot supply the 50k and 100k cells. Using
the 4/4/256 pair keeps ST k=1 vs ST k=16 a clean same-trunk comparison.

⚠️ **Trunk mismatch is not controlled.** UNet BC is 293M against ST's ~14M — a 20× parameter
gap. Any BC-vs-ST difference below confounds architecture, parameter count and search
mechanism. The comparison that *is* controlled is **ST k=1 vs ST k=16** (identical trunk,
identical trainer family, differing only in `n_candidates`).

**What k means.** `n_candidates` → `policy.max_actions` = the number of candidate slots
decoded per forward pass. One forward decodes all K slots against the *same* target (the
expert action); a staircase memory mask lets slot k attend to exactly the first k scored
candidates. So k=16 fits a family of conditionals from "no context" (slot 0) to "15 scored
candidates" (slot 15). At k=1 there is one slot, no context capacity, and the loss is a plain
BC denoising loss.

---

## 2. Artifacts

### 2a. Videos — objective 1

**Location** `videos/aug19_bcvSTk1_actions_viz/`
**Naming** `seed{NN}_n{n}_step{step}_{label}_{succ|trunc}.mp4`
**Sidecars** `episodes_{label}_step{step}_n{n}.json` — per-seed `episode_idx`, `success`,
`env_steps`, `max_reward`, `n_decisions`, `n_blind`, `blind_decisions`, video filename.

Arms `unetbc`, `stk1` × steps 10k/50k/100k × n ∈ {1, 2, 8, 16, 64} × 10 episodes = 300 mp4s.
Episodes are test-split positions 0–9, identical across both arms.

| label | step | n=1 | n=2 | n=8 | n=16 | n=64 |
|---|---|---|---|---|---|---|
| unetbc | 10000 | 10/10 · 0 succ · 0% blind | 10/10 · 3 succ · 23% blind | 10/10 · 4 succ · 10% blind | 10/10 · 7 succ · 7% blind | 10/10 · 7 succ · 5% blind |
| unetbc | 50000 | 10/10 · 0 succ · 0% blind | 10/10 · 5 succ · 31% blind | 10/10 · 2 succ · 28% blind | 10/10 · 6 succ · 21% blind | 10/10 · 4 succ · 17% blind |
| unetbc | 100000 | 10/10 · 2 succ · 0% blind | 10/10 · 2 succ · 31% blind | 10/10 · 6 succ · 27% blind | 10/10 · 2 succ · 29% blind | 10/10 · 5 succ · 23% blind |
| stk1 | 10000 | 10/10 · 0 succ · 0% blind | 10/10 · 0 succ · 19% blind | 10/10 · 2 succ · 7% blind | 10/10 · 3 succ · 6% blind | 10/10 · 4 succ · 1% blind |
| stk1 | 50000 | 10/10 · 3 succ · 0% blind | 10/10 · 3 succ · 28% blind | 10/10 · 3 succ · 27% blind | 10/10 · 5 succ · 19% blind | 10/10 · 4 succ · 18% blind |
| stk1 | 100000 | 10/10 · 0 succ · 0% blind | 10/10 · 3 succ · 32% blind | 10/10 · 2 succ · 26% blind | 10/10 · 1 succ · 21% blind | 10/10 · 2 succ · 15% blind |

**All 300 rendered and verified** (`scripts/check_aug19_artifacts.py`): every cell 10/10,
every sidecar agrees with the files on disk, both arms rendered the same episodes, and every
sidecar records `verifier_value: t_goal`. `succ` counts successes out of those 10 episodes —
these are 10 episodes, not the 50-episode eval split, so ±15pp; read success off
`SUCCESS_RATES_30_100.md`, not off this table.

⚠️ **n=1 is structurally 0% blind** and must not be averaged in with the rest: blindness means
the candidates disagreeing, which needs at least two of them. Excluding it, the blind rate is
8.8–11.9% at step 10k and 23.3–27.5% at 50k/100k — see §5a.

**On-frame legend**

| colour | meaning |
|---|---|
| green | the recorded demo's agent path — a static reference for the episode, not a per-state expert action |
| grey | losing candidates. At n>24 a positional subsample is drawn (never by score); the caption says `fan k/n shown` |
| **amber** | the **argmax** candidate — highest verifier value, and the one actually executed |
| **magenta** | the **final** candidate — slot n−1, the last generated |
| pink | the two coincide (always at n=1); caption says `chosen == final` |
| endpoint dots | every candidate's final waypoint, all n of them, always drawn |

One video frame per **decision** (every `n_action_steps` env steps), not per env step: a
decision is exactly one complete best-of-n search.

**Frame layout.** `[ scene | zoom ]` side by side, 776×460, caption above, nothing beneath.
An executed chunk spans ~8 waypoints a few pixels apart, so at scene scale the whole search is
a smudge beside the agent — the zoom panel is where the spread is actually legible, and it
draws **all** n candidates (only the scene panel subsamples).

Two layout switches, both recorded per-video in the sidecar (`closeup`, `value_strip`) so a
frame's layout is recoverable from provenance rather than inferred from its dimensions:

- `--closeup/--no-closeup` (default on). `--no-closeup` renders the scene alone at 384×460,
  for judging the trajectory in context. See `videos/aug19_stk16_nocloseup/`.
- `--value-strip/--no-value-strip` (**default off**). The per-candidate value bar chart that
  used to sit under the frame is gone. Its bar heights were rescaled per decision, so they
  showed spread *within* one frame and were not comparable *between* frames; the exact values
  are in the per-step JSON (§2b) and the caption still prints the executed value and the
  spread. Videos rendered before this carry the strip and are 776×576.

⚠️ **The zoom used to be pasted into the scene's bottom-left corner and covered a quarter of
it** — routinely the T block and the goal outline, i.e. exactly the context needed to
interpret the fan being magnified. It is a sibling panel now; nothing is occluded, and both
panels are full size (384² each) rather than the zoom being shrunk to 150–192px to limit the
damage. Videos rendered before 2026-08-19 evening have the old overlay layout.

The zoom's crop box uses a per-axis quantile (default 0.995) rather than min/max, because
min/max lets one stray candidate inflate the box until n=64 zooms *less* than n=8. The cost
is that a few waypoints fall outside and are clamped to the border; the panel prints how many
(`N clipped`). At the old 0.98 default that was ~8–9% of waypoints, which is why it is 0.995.

Caption lines are auto-fitted (shrink, then truncate with `..`) so a longer arm name or a
larger blind count can never silently run off the right edge.

⚠️ **The argmax path is AMBER, not cyan.** OpenCV draws in BGR and the frame is converted to
RGB at write time, so the constant `(60,200,255)` displays as `RGB(255,200,60)` — amber. The
original script's captions called that same value "cyan", which would send a reader looking
for the wrong colour. Verified by sampling pixels out of a rendered frame, not by reading the
constant. Amber is kept rather than switched to true cyan because PushT's scene is
grey/blue/red/green and cyan competes with the blue block.

⚠️ **`slot n−1` is only meaningful for `stk16`.** For `unetbc` (`predict_action` discards
the search context) and for `stk1` (`max_actions == 1`, so any n>1 uses a rolling window with
an empty history), the n candidates are i.i.d. and the magenta path is just "draw #n", not
"the most-conditioned draw". Those frames carry `slot n-1 = i.i.d. draw, no context`.

⚠️ **Only discriminating decisions are rendered.** See §4b. The rollout is unfiltered — the
argmax is still executed at every step — but blind decisions produce no frame, so the caption
carries the true decision index and a running skip count.

### 2b. Per-step candidate values — objective 2a

**Location** `analysis/aug19_candidate_scores/{label}_step{step}/per_step/ep{NN}_idx{M}.json`
**Schema** `pusht_candidate_values/v1` — see the `schema` key in each file.

One object per control step: all n candidate values in **generation order**, in both `value`
(verifier units, higher better) and `distance_px` (`-value`, lower better), plus
`argmax_first` / `argmax_last` / `n_argmax_tied` / `argmax_margin_px` / `degenerate` /
`executed` / `final_slot_value` and per-step min/max/mean/spread.

**The single episode designated for cross-arm comparison: split position 0 =
`episode_idx 3`**, i.e. `per_step/ep00_idx3.json` in every one of the nine dump directories.

Chosen because it is the most *discriminating* of the ten: 30 of its 38 control steps have a
unique maximizer and only 2 are blind, so the file actually shows candidate spread rather
than pages of identical values. It is also `seed00` in the videos, so the JSON and the
rendered episode are the same rollout under the same seed stream — the numbers can be read
against the picture.

Picked on a *coverage* property (how many steps discriminate), never on the values
themselves; picking the episode where an arm looks best would be choosing the example to fit
the story. All 50 episodes are dumped, so any other choice is one file away.

### 2c. Argmax-slot statistics — objective 2b

**Location** `analysis/aug19_candidate_scores/{label}_step{step}/candidate_scores_stats.json`
plus `candidate_scores.npz`, `candidate_scores_meta.json`, and the plot.
9 cells: 3 arms × 3 steps, `--n 16 --episodes 50 --split test --pair-seeds`.

| label | step | semantics | control steps | success | blind | argmax **unique** | argmax first | **perm null** | Δ vs null |
|---|---|---|---|---|---|---|---|---|---|
| unetbc *(null)* | 10000 | iid_unbounded | 1401 | 0.60 | 9.4% | 7.50 | 5.88 | 7.51 | -0.01 |
| unetbc *(null)* | 50000 | iid_unbounded | 1505 | 0.50 | 25.4% | 7.41 | 6.41 | 7.50 | -0.09 |
| unetbc *(null)* | 100000 | iid_unbounded | 1712 | 0.28 | 32.7% | 7.40 | 6.96 | 7.50 | -0.10 |
| stk1 *(null)* | 10000 | iid_no_context | 1703 | 0.30 | 7.6% | 7.74 | 6.12 | 7.50 | +0.24 |
| stk1 *(null)* | 50000 | iid_no_context | 1646 | 0.30 | 22.3% | 7.67 | 7.03 | 7.51 | +0.16 |
| stk1 *(null)* | 100000 | iid_no_context | 1728 | 0.22 | 28.2% | 7.41 | 6.83 | 7.49 | -0.09 |
| stk16 | 10000 | trained_staircase | 1625 | 0.34 | 16.9% | **6.31** | 5.52 | 7.49 | **-1.18** |
| stk16 | 50000 | trained_staircase | 1621 | 0.36 | 31.5% | **4.79** | 4.41 | 7.51 | **-2.72** |
| stk16 | 100000 | trained_staircase | 1744 | 0.20 | 33.7% | **4.13** | 4.02 | 7.51 | **-3.37** |

**All nine cells complete and verified** (`scripts/check_aug19_artifacts.py`), 50 test
episodes each at n=16, every one under `verifier_value: t_goal`. The six i.i.d. cells span
Δ = −0.10 … +0.24; the three `stk16` cells span −1.18 … −3.37. See §5d.

The ten episodes common to every dumped arm — and identical to the ten rendered as video —
are `[3, 7, 8, 9, 20, 24, 25, 26, 29, 33]`.

⚠️ **`unetbc` and `stk1` are not baselines *of* this statistic — they *are* its null
distribution.** Their candidates are i.i.d. by construction, so their argmax-slot spread is
what "no slot carries an advantage" looks like on real data. If `stk16` sits inside their
band, the search context does not move the argmax slot. Read every `stk16` cell against the
two arms above it and against `perm null`, never against the naive `(n-1)/2 = 7.5`.

⚠️ **Two conventions, two questions.** `first` (`np.argmax`, the first maximizer) is what the
deployed policy actually executed — `select_candidate` uses `scores.argmax(dim=1)`. `unique`
restricts to steps with a single maximizer and answers which slot genuinely wins. They differ
by several slots; see §4c.

### 2d. Slot weighting — objective 3

**Shipped** in `diffusion_policy/config/train_pusht_diffusion_search.yaml`:

```yaml
slot_weights:
  mode: uniform      # {uniform, geometric, linear, list, last_only}
  decay: null        # geometric: w_k ∝ decay^(K-1-k), 0 < decay < 1
  ratio: null        # linear:    w_last/w_first > 1
  weights: null      # list:      explicit K-length list of positives
  schedule: null     # {shape: linear|cosine|step, start_step, end_step}
  val: uniform       # {uniform, trained} -- what val_loss is computed under
sw_suffix: ''        # e.g. '_sw-lin4857'; goes into run_name
```

Properties that are asserted in `scripts/selection_smoke.py`, not just intended:

- Every profile is renormalized to **mean exactly 1**, so switching profiles cannot change
  the loss scale that `gradient_clip_norm` and the effective step size read.
- The curriculum blends in **weight space** (`w(t) = 1 + a(t)·(w_target − 1)`), so the mean
  stays exactly 1 at *every* point of the ramp — verified at start, midpoint and end.
- `slot_weight_decay: X` resolves to a vector bit-identical to
  `slot_weights: {mode: geometric, decay: X}`, so runs already trained under the scalar stay
  reproducible from their own config. It also keeps `val: trained`, the semantics those runs
  had.
- Nested keys are **whitelisted**: `slot_weights: {rato: 4}` raises. `_KNOWN_KWARGS` only
  guards the top-level name, so without this a typo would train uniform while the run name
  claimed a profile.
- `sw_suffix: ''` leaves `run_name` byte-identical to before, so no existing run directory
  moves; a non-uniform profile that fails to set it would otherwise resume a uniform run.
  Resolved example: `value_k16_ver-armT_sw-lin4857_corrupt-False_demos-30_seed-42`.

⚠️ **The matched linear ratio at K=16 is `4.857`, not 4.05** — `0.9^-15 = 4.857`. That is the
`ratio` which gives `linear` the same endpoint spread as `geometric(0.9)`, making the pair a
test of *curvature* alone. The smoke test computes it as `0.9 ** -(K-1)` rather than hard-coding.

Training runs launched: **none yet, deliberately.** The weighting worth running under
`selection: argmax` depends on what §2c shows — see §4d. The config surface lands first
because it is cheap and unblocks `final_pass` work either way.

| mode | params | `run_name` | status |
|---|---|---|---|
| — | — | — | gated on §2c |

### 2f. Full candidate report

`CANDIDATES_AUG19.md` — built by `dump_candidate_scores.py --report` over all nine dump
directories. Contains the mechanism table, the prefix (`mean` vs `max`) curves, the
record-rate permutation test, the per-slot rank/P(best) tables, the argmax-slot null-band
table (§3b) and the blindness table.

### 2e. Checkpoint retention change

**topk retention was removed on 2026-08-19.** `diffusion_policy/config/pusht_base.yaml` no
longer defines `checkpoint.topk`, and `TrainMLPImageWorkspace` no longer constructs a
`TopKCheckpointManager`. Every `checkpoints/step_*.ckpt` is retained permanently at
`checkpoint_every: 10000`; nothing is deleted; `latest.ckpt` (which `training.resume` reads)
is untouched.

It was wrong twice over:

1. **It selected on `val_loss`,** which is not the metric anything is chosen on. Checkpoint
   choice comes from `bon_search/success_curves.jsonl` with the step named explicitly.
2. **After a resume it was not top-k, and not capped at k.**
   `TopKCheckpointManager.path_value_map` is an in-memory dict that `save_checkpoint` never
   serializes (it is absent from `include_keys` and the class has no `state_dict`). On resume
   the map restarts empty, so the first k val-producing epochs are saved *unconditionally*
   via the under-capacity branch whatever their value, and pre-resume files are no longer
   deletion candidates. Measured: `offline/gaussian_k16_corrupt-False_demos-100_seed-42`
   holds **10** `epoch=*.ckpt` under `k: 5`.

`scripts/slurm/test_resume.sbatch` check 7 asserts no `epoch=*.ckpt` is ever written again.

**Legacy files left in place:** 160 `epoch=*.ckpt` files, **48 GB**, across the output tree.
Deleting checkpoints is irreversible, so this is left as an explicit decision rather than a
side effect. Of the three arms here, only `stk1` (`offline/bc_demos-30_seed-42`) carries any
— 5 files; `unetbc` and `stk16` have none, because neither workspace ever built a manager.

**Verified 2026-08-19** by `scripts/slurm/test_resume.sbatch` against
`offline/value_k1_arch-6x8x1024_...` (a run that carries 5 legacy `epoch=*.ckpt`), resuming
from step 100000: **all 15 assertions pass**, including `7a. no epoch=*.ckpt written` and
`7b. latest.ckpt written`, plus global_step continuing 100000 -> 100006, optimizer moments
restored, EMA neither reset nor frozen, normalizer preserved, and no eval block re-firing.

That run also exposed a flaw in the resume test itself: `latest.ckpt` is only rewritten on a
`checkpoint_every` boundary (10000), so a 6-step resume wrote no new checkpoint and every
state-survival assertion silently compared the copied-in file to itself. The sbatch now
shrinks `checkpoint_every` below `EXTRA_STEPS` and asserts (check 0) that a new `latest.ckpt`
actually appeared, so a vacuous run fails instead of passing.

First run trained after the change: `pending`.

---

## 3. How to re-derive

All commands from the repo root, in the `robodiff` env.

```bash
# --- 2a videos (one (label, step); the script sweeps every n internally)
python scripts/render_search_videos.py \
  -c /gscratch/robotics/harine/diffusion_policy_outputs/pusht_search/pusht_image_search/offline/bc_demos-30_seed-42/checkpoints/step_0100000.ckpt \
  --label stk1 --n 1 --n 2 --n 8 --n 16 --n 64 \
  --out-dir videos/aug19_bcvSTk1_actions_viz \
  --n-seeds 10 --no-subgoals --seed 42 --max-fan 24 --zoom 192

# all six cells
sbatch --array=0-5 scripts/slurm/render_bcvstk1.sbatch

# --- 2b/2c candidate values (one cell)
python scripts/dump_candidate_scores.py \
  -c <ckpt> --arm stk16 --out-dir analysis/aug19_candidate_scores/stk16_step0100000 \
  --n 16 --episodes 50 --split test --per-step-json --pair-seeds

# all nine cells, then the report
sbatch --array=0-8 scripts/slurm/dump_candidate_scores.sbatch
# --report is repeatable, so a bare glob binds only its FIRST match and the rest arrive as
# positional args ("Got unexpected extra arguments"). Build one flag per directory:
python scripts/dump_candidate_scores.py \
  $(for d in analysis/aug19_candidate_scores/*/; do printf ' --report %s' "$d"; done) \
  --out CANDIDATES_AUG19.md

# all six video cells / all nine dump cells
sbatch scripts/slurm/render_bcvstk1.sbatch          # array 0-5
sbatch scripts/slurm/dump_aug19_grid.sbatch         # array 0-8

# --- verify every artifact this doc claims, and re-emit the §2a / §2c tables from disk
python scripts/check_aug19_artifacts.py            # exits 1 on any problem
python scripts/check_aug19_artifacts.py --tables   # markdown only, for pasting back in

# --- 2d slot weights: unit + config assertions (no GPU rollouts needed for the profiles)
python scripts/selection_smoke.py

# --- 2e resume regression (must pass before any training launch).
# CKPT_EVERY must stay below EXTRA_STEPS or latest.ckpt is never rewritten and every
# assertion compares the copied-in file to itself; check 0 now fails loudly if so.
SRC=<run dir> CONFIG_NAME=<config> EXTRA_STEPS=6 CKPT_EVERY=3 \
  sbatch --account=gpu-a40-robotics --partition=gpu-a40 scripts/slurm/test_resume.sbatch \
  <config overrides>
```

Success rates these diagnostics are meant to explain: `SUCCESS_RATES_30_100.md`, built from
each run's `bon_search/success_curves.jsonl`.

---

## 4. Reading guide

### 4a. Which verifier these artifacts use — `t_goal`, the pre-2026-08-19 rule

The verifier value changed on 2026-08-19:

| tag | value | |
|---|---|---|
| `t_goal` | `-(T-to-goal distance)` | pre-cutover |
| `armT` | `-(T-to-goal + arm-to-T-centre distance)` | current |

**Everything in this document uses `t_goal`,** and that is deliberate, not inertia. All
three arms' checkpoints were trained and evaluated before the cutover, so their saved
configs carry no `verifier_value` key and resolve to `pusht_verifier.DEFAULT_VALUE_FN`,
which is `t_goal`. Each policy is therefore scored on exactly the value it was trained
against — and for `stk16` that matters twice over, because the verifier value is fed back
into the model as **search context**: scoring it with `armT` would condition it on a signal
it never saw. The existing `bon_search/success_curves.jsonl` numbers are also `t_goal`, so
this is what keeps the videos showing the trajectories the eval curves actually scored.

Every artifact records its own `verifier_value` (dump meta, per-step JSON, video sidecar,
and the on-frame caption), so a `t_goal` artifact can never be mistaken for an `armT` one.

⚠️ **Runs across the two are not comparable, and §4b below is a `t_goal` measurement.** The
approach term was added precisely because `t_goal` is flat across candidates before contact
— so under `armT` the blindness described next should largely disappear. Do not carry these
blindness numbers over to an `armT` run; re-measure.

### 4b. The verifier is blind whenever no candidate touches the T (`t_goal`)

The verifier value is `-mean_kp ||goal_kp − achieved_kp||`, a function of **where the T-block
ends up** after simulating a candidate's 8-step chunk. If no candidate contacts the block
during those 8 steps, the block does not move, every candidate returns the byte-identical
current distance, and the search carries **zero** information. Those steps are called
*degenerate* (`max − min <= 1e-9`) or *blind*.

Measured on `subgoal-only k16` step 76k, n=16, 612 live control steps, 20 episodes:

```
degenerate rate by control step:   t=0: 65%   t=1: 35%   t=2: 20%
                                   then ~15-25% throughout, late spikes (t=29,30: 42%)
overall degenerate rate                     20.9%
mean leading-degenerate run                 1.1 steps
mean total degenerate steps per episode     6.4
degenerate steps inside the leading run     17.2%
```

⚠️ **The intuitive reading — "the verifier is blind during the approach, then works" — is
wrong by about 5×.** Blindness does spike at the initial approach, but only ~17% of blind
steps live in that prefix. The other ~83% are **mid-episode** losses of contact: the agent
backs off, repositions, or circles the T to push from another face. Those are real control
decisions, often the pivotal ones, and the verifier cannot rank them.

This is a property of the verifier's design, not a measurement artifact, and it caps what
best-of-n can buy on this task no matter how good the candidate distribution becomes.

Consequences carried through every artifact here: **the rollout is never filtered** (the
policy executes the argmax at every step, blind ones included, so trajectories and success
rates are untouched); only rendering and reporting are scoped to discriminating steps; and
every artifact records its own blind-step count so a scoped statistic is never mistaken for
an unscoped one.

### 4c. Ties dominate the argmax-slot statistic, so the convention must be stated

Same reference dump, n=16:

```
fully degenerate (all 16 candidates identical)   20.9%
steps with >1 maximizer                          32.8%   (mean tie-set = 5.24)

argmax slot, FIRST maximizer (np.argmax)   mean 4.32   median  2
argmax slot, LAST maximizer                mean 8.92   median 11
argmax slot, RANDOM among the tied         mean 6.78   median  7
argmax slot, unique-maximizer steps only   mean 6.10   median  5
uniform null (n-1)/2                            7.5
```

`np.argmax` returns the **first** maximizer, so on a fully degenerate step it returns 0 —
100% of the time. About 21% of samples are therefore a hard 0 by construction, and partial
ties drag the rest. The median does not rescue it: a 21% point mass at 0 moves a median
through a 16-wide distribution (2 vs 5). **The answer ranges 4.32 → 8.92 on tiebreak
convention alone**, wider than any effect worth looking for.

Hence both conventions are reported (§2c) and each is quoted against its own permutation
null, never against `(n-1)/2`.

### 4d. Last-slot-heavy weighting is motivated by `final_pass`, not by `argmax`

The original `slot_weight_decay` rationale comes from `selection: final_pass`, where slot K−1
*is* the deployment condition and the earlier slots exist only to manufacture its context.
**These arms use `selection: argmax`,** where *every* slot is deployed: all n candidates come
from slots 0..K−1 and the executed action is the best of them. The objective is a good **max
over the pool**, not a good final conditional.

So under argmax, last-slot-heavy weighting is at best neutral and plausibly the wrong
direction — it deliberately degrades K−1 of the K candidates the selector chooses from. The
weighting that *is* justified under argmax is one proportional to how often a slot actually
wins, which is exactly what §2c measures. That is why §2d is gated.

### 4e. Cross-arm value levels are not comparable

Each arm is driven by its own selection rule and visits its own states, so raw per-slot value
*levels* are conditional on different state distributions. Only within-step statistics are
comparable: `centered_mean_by_index`, `rank_by_index`, `p_is_best_by_index`, and the
argmax-slot statistic. The same trap is documented at length in
`CANDIDATES_FROM_SUBGOAL.md` — a *lower* pooled mean can indicate a *better* policy, because
improving it converts long failing episodes (which contribute many late, high-value steps)
into short successful ones.

### 4f. What "paired" does and does not mean

`--pair-seeds` / the videos' `--seed` make each episode's diffusion noise a pure function of
`(episode index, draw index)` via `set_sample_seeds`, so an episode is scored on the same
seed stream under every arm and at every checkpoint, independent of batching.

⚠️ This pairs the arms on *(episode, draw index)*, **not** on the noise realization: the UNet
and the transformer denoise different-shaped tensors, so the actual Gaussian vectors differ.
The claim supported is "same initial state, same deterministic seed stream, same draw
ordering" — not "same noise".

---

## 5. Findings

*Filled in as results arrive.*

### 5.0 Already measured (reference dump, not one of the nine cells)

From `subgoal-only k16 cd0.9`, step 76k, n=16, 612 live control steps, 20 episodes — the
dump that existed before this work. Kept here because it is what the tooling was validated
against and it already answers part of 5d in the wrong direction.

| quantity | value |
|---|---|
| blind rate | 20.9% |
| blind steps inside the leading run | 17.2% (so ~83% are mid-episode) |
| argmax slot, unique-maximizer steps | **6.10** |
| permutation null, same convention | **7.49** |
| argmax slot, `first` (unscoped) / `last` | 4.32 / 8.92 |

⚠️ **The argmax lands on EARLIER slots than chance** (6.10 against a 7.49 null), not later.
If that survives on the nine cells, it is evidence *against* the premise of last-slot-heavy
weighting, and §2d should not be run as specified. This is one arm at one checkpoint under
`final_pass`, so it is a flag to check, not a conclusion.

### 5a. How often is the verifier blind, per arm and per checkpoint? — it RISES with training

| arm | 10k | 50k | 100k | success 10k → 100k |
|---|---|---|---|---|
| unetbc | 9.4% | 25.4% | 32.7% | 0.60 → 0.28 |
| stk1 | 7.6% | 22.3% | 28.2% | 0.30 → 0.22 |
| stk16 | 16.9% | 31.5% | 33.7% | 0.34 → 0.20 |

Blindness roughly triples on every arm between 10k and 100k, and success falls on every arm
over the same span. **The causal direction is the opposite of the obvious reading:** these
runs overfit, and a policy that fails to bring the agent to the T leaves more decisions where
no candidate can move the block, so nothing is rankable. Blind rate and success rate are two
views of one decline, not a cost of training longer.

Cross-checked against `unet_bc/.../bon_search/success_curves.jsonl`, written months earlier by
a different code path: 0.62 / 0.50 / 0.24 against this dump's 0.60 / 0.50 / 0.28.

`in leading run` stays at 21–35% across all nine cells, so **two-thirds to four-fifths of
blind steps are mid-episode**, not the opening approach — §4b's conclusion, now on
50 episodes × 9 cells rather than 20 × 1.

Blindness also **falls with search width** (unetbc 100k: 31% at n=2 → 23% at n=64, from the
video sidecars), so a wider search buys discrimination as well as selection — something the
success curves alone cannot show.

At the checkpoints of interest roughly a third of decisions have a verifier that separates
nothing. That caps what best-of-n can deliver regardless of candidate quality, and is the
concrete case for the `armT` approach term.

### 5b. Does the ST k=1 fan differ from the UNet BC fan?

Spread, multimodality, and how each drifts from 10k → 100k.

> pending

### 5c. Does the argmax pick differ from the final candidate?

Yes, and overwhelmingly. At `stk16` 100k the final candidate (slot 15, full context) wins
4.8% of discriminating steps against slot 0's 48.6% — so the executed action is ten times more
often the *unconditioned* draw than the most-conditioned one. See §5d for the full
distribution; this is the same result read from the other end.

> further notes:

### 5d. Is ST k=16's argmax-slot distribution outside the i.i.d. null band? — YES, and it is slot 0

**Answered. This was the decision gate for §2d / §4d, and it says do not run that sweep.**

The six i.i.d. cells land within ±0.25 slots of their own permutation null, across a 4x range
of blind rate and a 3x range of success — that is the noise floor. All three `stk16` cells sit
far outside it, and the gap grows monotonically with training:

| arm | step | semantics | succ | blind | argmax slot | perm null | Δ |
|---|---|---|---|---|---|---|---|
| unetbc | 10000 | iid_unbounded | 0.60 | 9.4% | 7.50 | 7.51 | −0.01 |
| unetbc | 50000 | iid_unbounded | 0.50 | 25.4% | 7.41 | 7.50 | −0.09 |
| unetbc | 100000 | iid_unbounded | 0.28 | 32.7% | 7.40 | 7.50 | −0.10 |
| stk1 | 10000 | iid_no_context | 0.30 | 7.6% | 7.74 | 7.50 | +0.24 |
| stk1 | 50000 | iid_no_context | 0.30 | 22.3% | 7.67 | 7.51 | +0.16 |
| stk1 | 100000 | iid_no_context | 0.22 | 28.2% | 7.41 | 7.49 | −0.09 |
| **stk16** | **10000** | trained_staircase | 0.34 | 16.9% | **6.31** | 7.49 | **−1.18** |
| **stk16** | **50000** | trained_staircase | 0.36 | 31.5% | **4.79** | 7.51 | **−2.72** |
| **stk16** | **100000** | trained_staircase | 0.20 | 33.7% | **4.13** | 7.51 | **−3.37** |

**It is not "early slots" — it is slot 0 specifically.** Restricting to steps with a unique
maximizer (uniform = 6.25% per slot):

| arm · step | slot 0 | slots 1–7 | slots 8–15 |
|---|---|---|---|
| unetbc · 100k *(null)* | 6.5% | 44.5% | 49.0% |
| stk16 · 10k | **15.8%** | 42.6% | 41.6% |
| stk16 · 50k | **37.7%** | 30.1% | 32.2% |
| stk16 · 100k | **48.6%** | 22.4% | 29.0% |

At 100k the **empty-context slot wins nearly half of all discriminating steps** — 7.8x
uniform — while slots 1–15 sit roughly uniform among themselves and all below chance. The
null arm's slot 0 is 6.5%, i.e. exactly uniform, so this is not a property of being first in
generation order.

**What it means.** Conditioning a draw on previously scored candidates makes it *worse*, and
training makes that worse-ness stronger (15.8% → 37.7% → 48.6%). The search context is not
merely uninformative under argmax; it is actively degrading the candidate pool. This agrees
with the "the search narrows rather than improves" finding in `CANDIDATES_FROM_SUBGOAL.md`,
measured there as a collapse in within-step spread.

⚠️ **Do not read `p_is_best_by_index` for this.** That statistic is tie-inclusive over all
live steps, so degenerate steps (where every slot counts as a maximizer) dilute it toward
uniform: it reports a max weight of 1.69 where the discriminating-step distribution reports
7.78. Same trap as §4c, one level deeper.

**Consequence for §2d.** Every profile built there — geometric, linear, `last_only`, and the
curriculum ramping toward them — weights slot K−1 heaviest. Under `selection: argmax` that
down-weights the slot producing half the executed actions. A weighting fitted to this
measurement would put ~7.8x on slot 0 and suppress the rest, which is a reductio: it says
train the unconditioned conditional, i.e. ST k=1. The config surface is built and tested, but
the sweep as specified would optimise against the data.

Scope: one arm (ST k=16, 4/4/256, 30 demos), three checkpoints, 50 test episodes each,
~1000–1100 discriminating steps per cell, verifier `t_goal`.

### 5e. Does the candidate value spread collapse with training?

Compare against the `centered_sd_by_index` collapse documented in
`CANDIDATES_FROM_SUBGOAL.md`, where the search was found to *narrow* rather than improve.

> pending

### 5f. Does n=64 justify its cost over n=16?

`prefix_max` vs `prefix_mean` across the two, per arm.

> pending

---

## Related

- `SUCCESS_RATES_30_100.md` — the success rates these diagnostics explain.
- `CANDIDATES_FROM_SUBGOAL.md` — the prior per-candidate analysis, on the 100-demo subgoal
  arms. Its methodology notes (permutation nulls, tie handling, episode-composition
  confounds) apply here verbatim.
- `aug9_analysis.md` — policy inventory and eval coverage.
- `AUDIT.md` — config-schema generations and the arm rename history.
