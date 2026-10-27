# Maze search experiments — every design choice, number and approximation (2026-10-07)

Covers the round-3 arms `dt_bon`, `dt_st_traj`, `dt_st_step` (fork `harine/maze` @ 3b04596, L2S root
`data/experiments/openloop11dt_pipeline_quadrant`), with the round-2 arms (`lstm_bon`, `lstm_st`, `dst64`) and the
`sriyash/l2s_online` reference for comparison. Every number below is read from the code, configs, zarr or run
artifacts named in brackets. Figure: `docs/dt_splits_quadrant.png`; results: `docs/maze_dt_search_results_2026-10-07.md`.

## 1. Environment (`L2S/l2s/maze.py`, `MazeEnv`)

- **Layout.** Procgen maze layout 0 of `data/procgen_mazes/layouts.npz`: an 11×11 grid with **71 free cells**, the rest walls.
  One layout for everything (train, val, both test sets). BFS diameter (longest shortest path between free cells) = **32**.
- **State.** The agent's integer cell `(x, y)` and the goal cell. No velocity, no orientation.
- **Observation** (`n_obs_steps = 1`, current frame only):
  `image` 3×11×11 — R = 255 on walls / 0 on free cells, G = 255 at the goal cell, B = 255 at the agent cell (uint8, fed as /255);
  `agent_pos` (2,), `goal_pos` (2,) in cell units. The policy flattens all of it to a **367-d** vector (363 + 2 + 2).
- **Action.** A continuous 2-d displacement. `rollout_delta` rounds `pos + delta` to a target cell and walks toward it one
  cell at a time, stopping at the first wall; the **realized** displacement is `new_pos − old_pos`. All round-2/3 policies
  run with `snap_cardinal`: the sampled action is snapped to the nearest of {(±1,0), (0,±1), (0,0)} before execution, so one
  decision = one BFS step or a stay, and a move into a wall realizes as (0,0). The expert's actions are exactly these unit moves.
- **Termination / success (harness rule, unchanged since round 2).** An attempt succeeds when `‖pos − goal‖₂ ≤ 1.0`, i.e. on
  reaching a cell **adjacent** to the goal (through-wall adjacency counts; `success_exact` additionally requires BFS distance ≤ 1
  and was identical to `success_rate` in every round-3 cell). The raw env's own `done` is `pos == goal`, but the harness and the
  training-time torch rollout both use the radius rule. Consequence: a successful attempt stops one step *before* where the expert
  demo ends (the expert ends on the goal), so the final expert move is never required at eval.
- **Verifier / feedback.** `value = −BFS(final cell, goal)` from a per-goal distance field (`MazeVerifier`, `unreachable = 122`).
  A successful attempt therefore always has value −1, the maximum attainable, which is why argmax@k == perf@k on this task.
- **Dynamics used in training.** The policies generate context attempts with `rollout_step_torch` / `build_images_torch`
  (`diffusion_policy/common/maze_verifier.py`), a batched torch copy of `rollout_delta` / `build_image`; `check_openloop.py`
  asserts the two produce identical trajectories, lengths, done steps and values from the same RNG seed.

## 2. Goal split, data, test sets (figure panels 1, 3, 4)

- **Quadrant split.** Centre (5,5). Test goal pool = free cells with x > 5 and y > 5 (**17 cells**, bottom-right); train goal pool =
  the other three quadrants excluding the centre row and column (**49 cells**). Starts: any free cell ≠ goal (71).
- **Expert data** (`data/procgen_maze_expert_quadrant_unpadded/data.zarr`, `l2s/maze.py --episodes 800 --steps 0 --seed 0`):
  800 episodes, (start, goal) drawn uniformly from (free cells × train goal pool); the demo is the BFS shortest path
  (ties broken by the BFS order over DIRS = right, left, down, up), **unpadded**: lengths 1..32, mean 12.4, 9 930 steps in total.
  800 episodes cover **712 unique pairs** (some pairs repeat), all 49 train goals, all 71 starts.
- **Test sets** (100 episodes each, `MazeEnv(seed = i)`, i = 0..99, identical across arms, built by `eval_openloop.build_episodes`):
  `eval` = goal from the **test** pool, (start, goal) ∉ the 800 training pairs (held-out goals — the distribution shift);
  `train_heldout` = goal from the **train** pool, pair ∉ the 800 (in-distribution, unseen pairs). Both sets exclude *all* 800
  pairs, i.e. also the validation pairs; the validation split is never a test set.

## 3. Train / validation split (round 3; figure panel 2)

- 80/20 **by unique (start, goal) pair**, `get_pair_val_mask(seed = 0)` (the launcher passes `task.dataset.seed = 0`; the config
  default 42 would give 640/160): **637 train episodes / 570 pairs, 163 val episodes / 142 pairs, 0 shared pairs.** All 49 goals
  and all starts occur on both sides — the split holds out *pairs*, not goals (goals are held out by the quadrant split instead).
- Round 2 used `val_ratio 0.05` by episode, which leaked 9 of 40 val pairs into training.
- The validation set is used only for `val_loss` (same loss as training, every 50 epochs) and the workspace's top-k bookkeeping;
  the eval grid is every 1 000-step checkpoint plus `latest`, so no val-based selection happens.
- **Normalizer** fit on train frames only: `action` limits → scale 1, offset 0 (actions ∈ {−1, 0, 1}, so identity);
  `agent_pos`, `goal_pos` gaussian (scale ≈ 0.45/0.34 and 0.29/0.31). Images are not normalized.

## 4. Model — `DTSearchPolicy` (`diffusion_policy/policy/dt_search_policy.py`), 3.39 M parameters (3.16 M in the encoder)

One causal transformer reads **one token stream per episode**: `[attempt 1 | attempt 2 | … | attempt k | current attempt]`.

**Token t** (sum of embeddings, d = 256):
`Linear(367→256)(obs_t)` + `Linear(2→256)(a_{t−1})` + `Linear(2→256)(fb_{t−1})` + `Linear(1→256)(d_{t−1})`
+ `Linear(SinPE_256(step within attempt))` + `Linear(SinPE_256(index in stream))`, dropout 0.1.
- `a_{t−1}` = the **realized** displacement of the previous step (normalized; = the expert action under teacher forcing).
- `fb_{t−1}` = 2 channels, both divided by the BFS diameter 32: `[−BFS(current cell), −BFS(final cell of the previous attempt)]`.
  `reward_mode = traj` zeroes channel 1 everywhere (feedback only once per attempt, on the boundary token);
  `reward_mode = step` keeps it, so the verifier's distance of the current cell is visible at **every** step, including attempt 1.
- `d_{t−1}` = 1 only on the first token of an attempt that follows an earlier attempt in the stream (the boundary); that token also
  carries the previous attempt's last realized action and its final feedback. At the start of the stream all three are zero.
- Positions are sinusoidal (no learned table, no maximum length). Nothing in the model has a horizon.

**Backbone.** 4 pre-norm `nn.TransformerEncoderLayer`s, d 256, 4 heads, FF 1 024, GELU, dropout 0.1, causal mask + key padding
over the padded batch, final LayerNorm. No KV cache: at rollout the whole stream is re-encoded at every step.

**Heads.** Action: `Linear(256→2)` mean with a **state-independent** `log_std` (init −1 → σ 0.37, floor −2 → σ ≥ 0.135);
sample → unnormalize → snap to cardinal. Stop: `Linear(256→1)`; the attempt ends when `σ(logit) > 0.5` (deterministic threshold).

**Attempt semantics (same code at train and eval).** An attempt ends when (a) the model stops, (b) the env reaches the radius-1
success, or (c) the **guard** of 71 steps (= number of free cells) trips. The guard is the only length constant and is not a
design parameter: it exists so a policy that never stops cannot run forever; how often it fires is measured (§7).

**Context window.** The stream keeps the last `max_actions − 1 = 7` earlier attempts; beyond 8 attempts the oldest is dropped and
the new first token's `(a, fb, d)` are zeroed so the stream looks like a fresh 7-attempt prefix. `dt_bon` has `max_actions = 1`:
every attempt starts an empty stream (i.i.d. attempts).

## 5. Training (`procgen_maze_dt_search.yaml` / `procgen_maze_dt_bc.yaml`, `TrainMLPImageWorkspace`)

| item | value |
|---|---|
| optimizer | AdamW, lr 1e-4, betas (0.95, 0.999), eps 1e-8, weight decay 1e-6 |
| schedule | cosine with 500 warm-up steps, sized to 10 batches × 1 000 epochs = **10 000 gradient steps** (`max_gradient_steps 10000`) |
| batch | 64 whole episodes, padded to the longest (`attention_mask`), 10 batches/epoch over 637 episodes |
| grad clip | 1.0; no EMA; dropout 0.1 on in train mode |
| seeds | `training.seed 0`, `task.dataset.seed 0` |
| checkpoints | every 1 000 steps (`step_0001001.ckpt` …: the workspace's `global_step` drifts +1 per epoch) + `latest.ckpt` |
| validation | `val_loss` every 50 epochs on the 163 val episodes |
| wall time | `dt_bon` 16 min; `dt_st_step` ≈ 17 h; `dt_st_traj` ≈ 20 h (A40-class, ckpt-all, requeued every 8 h and resumed) |

**Loss per batch (ST arms).** With the current weights, in eval mode (dropout off) and under `inference_mode`, generate K = 7
attempts per episode *sequentially* in the torch env from the expert's start (attempt k sees attempts < k). Then build
**all 8 prefixes** per episode, `[attempts 1..s | expert]` for s = 0..7, and run them as one batch of 8·64 sequences in train mode.
Loss = masked Gaussian NLL of the expert's actions (summed over the 2 dims) + BCE of the stop flag (target 1 on the expert's
last step, 0 elsewhere), averaged over the expert tokens of all prefixes. Context tokens get no loss. `dt_bon`: s = 0 only.
The expert's tokens are teacher-forced (its `a_{t−1}` is its own previous action; its step feedback comes from its own cells).

**Cost.** Early in training the untrained attempts run to the 71-step guard (7 × 71 steps × full re-encoding), ≈ 7–10 s/it on an
A40; as attempts shorten this drops to ≈ 4–6 s/it. `dt_bon` has no context generation: ≈ 0.1 s/it.

## 6. Evaluation (`L2S/l2s/eval_openloop.py`, `run_procgen_eval.py --eval-mode openloop`)

- 100 episodes per test set, all batched; **N = 64 sequential attempts** per episode, one shot: the k-th attempt is executed in the
  env as is (no re-planning, no chunking); the k ∈ {1, 2, 4, 8, 16, 32, 64} rows are prefixes of the same 64 attempts.
- The harness only passes observations: `policy.reset()` per episode, `policy.begin_attempt()` per attempt, then
  `predict_step(obs) → (action, active)` until no row is active; the policy decides stops and keeps its own context.
  `BatchedRollout` steps `MazeEnv.rollout_delta` and freezes a row at success; `check_openloop.py --algo dt_*` asserts the harness
  reproduces the training-time rollout exactly (lengths, done steps, values, trajectories).
- Sampling RNG seeded with the train seed (0); context attempts at eval are generated with dropout off, as in training.
- Metrics per row: `success_rate` = **perf@k** under strategy `any` (any of the first k attempts succeeded; the first success is the
  episode) and **argmax@k** under `max` (the attempt with the highest value among the first k); `success_exact`; `episode_length`
  (steps of the chosen attempt); `final_distance`; `n_unique_traj`, `mean_pairwise_endpoint_bfs`, `value_std` over the first k;
  `mean_first_success_k`. Every per-attempt trajectory, action, value and length is saved in `candidates.npz`.
- 36 checkpoint evals per arm (9 checkpoints × 2 splits × 2 strategies) plus the `latest` rows.

## 7. How attempts actually end (latest checkpoints, 100 episodes × 64 attempts, from `candidates.npz`)

| arm | split | reached goal | model stopped | guard (71 steps) | mean length | length of successes |
|---|---|---|---|---|---|---|
| dt_bon | eval | 0.19 | 0.73 | 0.08 | 15.4 | 5.9 |
| dt_st_traj | eval | 0.22 | 0.77 | 0.01 | 13.3 | 6.7 |
| dt_st_step | eval | 0.46 | 0.46 | 0.08 | 19.2 | 12.2 |
| all three | train_heldout | 0.96–0.97 | 0.02–0.04 | ≤ 0.01 | 11.5–11.7 | 11.2–11.5 |

The learned stop is what ends most failed attempts; the guard fires in ≤ 8 % of attempts on the held-out set and ≈ 0 in distribution.

## 8. Architectures across rounds

| | `dt_bon` / `dt_st_*` (round 3) | `lstm_bon` / `lstm_st` (round 2) | `dst64` (round 2) | `sriyash/l2s_online` (reference, PushT) |
|---|---|---|---|---|
| sequence model | causal transformer, 4×256, over one token stream of all attempts | 1-layer LSTM (256) over the current attempt | diffusion transformer (4×256) denoising a whole 64-step chunk | GPT-2, 4×256, over one token stream of all rollouts |
| token | obs + prev action + feedback + terminal + positions, one per env step | obs + prev realized action + pooled context vector, one per env step | one token per *whole candidate* (64×2 actions + value) | obs only (keypoint feedback folded into the obs), one per env step |
| cross-attempt context | attention over earlier attempts' tokens (window 7) | fixed 64×2+1 vector per candidate → Linear → mean/max pool → 128-d | attention over candidate tokens (≤ 7) | attention over earlier rollouts' tokens, KV cache kept, no window |
| execution | closed loop, one action per step | closed loop, one action per step | **open loop**: one sampled 64-chunk executed blind | closed loop, one action per step |
| action head | Gaussian (state-independent σ) + cardinal snap | same | DDPM (100 train / 8 inference steps) + snap | Gaussian with state-dependent σ, no snap |
| attempt length | model-predicted stop (+ goal, 71-step guard) | fixed 64-step horizon (stop at goal) | fixed 64-step chunk | fixed 300-step env limit |
| training of context | 7 on-policy attempts per batch, all 8 prefixes | same, pooled | realized/snapped chunk context | pool of 128 starts × 8 rollouts regenerated every 100 updates; random 0–8 of them as context |
| eval | sequential in-context, perf@k / argmax@k | sequential in-context, argmax / last | sequential in-context, argmax / last | **context-free** i.i.d. samples, best-of-n |

## 9. Approximations and design choices to keep in mind

1. **Success radius 1.0** (adjacent cell) instead of reaching the goal cell; a successful attempt ends one step before the expert's
   end, and through-wall adjacency counts (`success_exact` showed no such cases in round 3).
2. **Stop flag trained on the expert only** (target 1 at its last step; contexts carry no loss) and decoded by threshold 0.5, not sampled.
3. **Guard at 71 steps** on an attempt that neither stops nor reaches the goal (fired in ≤ 8 % of held-out attempts).
4. **Context attempts are generated with dropout off and no gradient**, while the loss runs with dropout on — a train-time
   mismatch only in dropout; the same no-dropout rollout is used at eval, so train-rollout == eval-rollout.
5. **`step` feedback is the verifier's distance of the current cell at every step** — information available *inside* attempt 1,
   so `dt_st_step`'s k = 1 gain is not evidence of cross-attempt search; `dt_bon` with `reward_mode = step` is the missing control.
6. **Feedback scale** = the layout's BFS diameter (32), computed from the layout, not tuned.
7. **Window slide approximation** beyond 8 attempts: the dropped attempt's final feedback is zeroed on the new first token so the
   stream looks like a training-time 7-attempt prefix (training never sees a slide).
8. **Teacher forcing** of the expert attempt; the expert's `a_{t−1}` equals the realized displacement because expert actions are unit
   moves that never hit walls.
9. **One layout, one seed**: layout 0 only, training seed 0 only, 100 episodes per test set (±5 % at best).
10. **Eval samples with the train seed**; the k-prefix rows share the same 64 attempts (not independent re-draws).
11. **Checkpoint-step drift** (+1 per epoch) — checkpoints are at 1 001, 2 001, … 9 002; reported as 1k … 9k.
12. **No KV cache**: ≈ 10–20 h per ST arm; a cache would change nothing in the model.
13. **Validation pairs are excluded from both test sets** together with the training pairs (the exclusion set is all 800 episodes).
14. **l2s_online reference** is PushT, not the maze; its eval is context-free; its one code change was the `use_ema` lookup fix.
