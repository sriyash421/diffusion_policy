# Maze full-trajectory search, round 3: decision-transformer token stream

## Context

Round 2 (`harine/maze` @ d995d8d, pipeline `openloop11snap_pipeline_quadrant`) made candidates full
open-loop trajectories scored by −BFS(endpoint), with three arms: `lstm_st`, `lstm_bon`, `dst64`.
User feedback: (1) attempts are executed one after another, so report perf@k (no argmax over N);
(2) the search model should be a decision-transformer over one token stream, like
`sriyash/l2s_online`; (3) no fixed-length padding anywhere; (4) token =
(current_obs, prev_action, prev_reward, prev_terminal); (5) train/val 80/20.

### Answer: was the search a transformer in every arm, with only the policy head varying? No.

| arm | within-attempt model | cross-attempt (search) context | candidate |
|---|---|---|---|
| `lstm_st` | LSTM, one action per step, closed loop | each earlier candidate flattened to a fixed **64×2+1** vector (realized actions + value) → Linear → mean/max **pool** over K → one 128-d vector appended to every LSTM input | step-by-step |
| `lstm_bon` | same LSTM | none | step-by-step |
| `dst64` | diffusion transformer denoising a whole **64-step chunk** at once | attention over K tokens, each token = one whole candidate (`action_value_emb` on 64×2+1) | one chunk, executed open loop |

So the LSTM arms had no transformer at all (search = pooled MLP summary), and `dst64` had a
transformer but over whole-candidate tokens and open-loop chunks. The arms differ in the search
mechanism, token granularity and closed- vs open-loop execution — not only in the action head.
None of them is the DT design asked for now.

### Why the padding started
1. The original L2S expert (`l2s/maze.py:get_expert_trajectory`) resampled every demo to a fixed
   `steps` (100) because DST/ST sample fixed-horizon windows (`SequenceSampler`).
2. Round 2 dropped the 100-step pad (`--steps 0`) but introduced **horizon 64** because `dst64`
   emits a fixed-length chunk and the LSTM's context slot was a fixed 64×2+1 vector. That was a
   modelling artefact. The per-episode collate (`sampler.py:get_collate_fn`, `attention_mask`)
   already handles variable length; a causal transformer over tokens needs no horizon at all.
   With a model-predicted terminal flag (user choice) there is no length parameter left at all —
   only the batch-level attention mask.

### What `sriyash/l2s_online` does (`origin/sriyash/l2s_online` on sriyash421/diffusion_policy, commit 86d5ef9, PushT)
`policy/online_search_policy.py::OnlineSearchPolicy` + `workspace/train_online_search_workspace.py`:
- Causal **GPT2** (hidden 256, 4 layers, 4 heads, `transformers.GPT2Model`), **one token per obs
  frame**: `obs_encoder(image, agent_pos, feedback)` → Linear. Feedback (goal-vs-achieved T keypoint
  displacement, 16-d) is a per-step obs channel; there is no action / reward / terminal token.
- Head: Gaussian, mean + state-dependent log_std (clamp −5..2), `rsample` at rollout.
- Context: `n_trajs=8` on-policy rollouts from the expert's start state in live PushT gym envs,
  generated autoregressively by the same transformer with the **KV cache kept across rollouts**
  (rollout k attends to rollouts <k); all rollout tokens are concatenated and the expert obs
  tokens appended: `[context | expert]`; loss = masked NLL on expert actions only.
- Training loop: outer/inner — rollouts for a pool of 128 starts once per outer step
  (`policy.eval()`), then 100 inner updates on minibatches of 8 with a random subset of
  0..8 of those (stale) rollouts as context (`_subsample_context`). No sliding window
  (`n_positions = 8·300 + 512`). Random context size ⇒ every slot 0..K is trained.
- Eval (`eval_bon.py`, runner): **context-free** `predict_action` sampled i.i.d., curves =
  E[max reward among n] — best-of-n, not in-context sequential. So that branch trains the in-context
  model but never evaluates it in context (train/eval mismatch we must not copy).
- Data: episode-level train/test split, normalizer fit on train frames only.

### Our version (maze) — same skeleton, these differences
1. **Token** = sum of embeddings of (obs features, prev_action, prev_reward, prev_terminal) + within-attempt
   timestep embedding (DT style), instead of obs-only with feedback folded into obs.
2. **Env** = the existing torch maze rollout (`maze_verifier.rollout_step_torch`, `snap_cardinal_torch`,
   stop at radius 1.0), no gym vector envs; rewards from the BFS verifier.
3. **Context generated fresh per batch** by the current weights (as `LSTMSearchPolicy.compute_loss`
   does), not a stale outer/inner pool — keeps train == eval generation exactly.
4. **Eval in context, sequentially**, perf@k; sliding window of the last `max_actions` attempts
   (train and eval); `max_actions=1` is the BoN arm.
5. 80/20 split by (start, goal) pair; normalizer on train frames.
6. Head stays Gaussian + `snap_cardinal` (user: continuous+snap is good).

## Design

### Token stream
Attempt k (k = 1..K−1 context, K = expert) is a sequence of tokens
`x_t = E_obs(obs_t) + E_a(a_{t−1}) + E_r(r_{t−1}) + E_d(d_{t−1}) + E_pos(t_within_attempt)`,
prediction target at `x_t` is `a_t`. At the first token of an attempt, `a_{t−1}, r_{t−1}, d_{t−1}` are the
**last step of the previous attempt** (so `d=1` marks the boundary and carries the terminal reward);
zeros for the very first attempt. All attempts of an episode are one causal stream (KV cache kept,
exactly as `_incontext_rollout_chunk`); the expert attempt is appended last; NLL on expert tokens only.

Reward arms (`policy.reward_mode`, user's two variants):
- `traj`: context = trajectory (actions) + one feedback per trajectory: `r_{t−1} = 0` inside an attempt,
  `r = −BFS(final cell)` on the boundary token.
- `step`: trajectory = (action, BFS feedback for that action) + trajectory feedback: every token carries
  `r_{t−1} = −BFS(cell after a_{t−1})`, and the boundary token additionally carries `−BFS(final cell)`
  (2-d reward input, second channel zero except at boundaries).

Lengths and termination: the model decides when an attempt ends. The head emits the action
(Gaussian, snapped) **and a Bernoulli terminal flag** `d_t` ("this is my last step"); supervision is the
expert's `done` array (already in the zarr, `done[-1] = True`). An attempt ends when the model emits
`d_t = 1` or the env reports the goal (radius 1.0; `success_exact` also logged); the next token then
carries that `d` and reward as `prev_terminal` / `prev_reward`. No step cap, no fixed horizon, no
`pad_after`, no fixed-size context slots: variable-length attempts are just more or fewer tokens.
Positions use `SinusoidalPosEmb` (`model/diffusion/positional_embedding.py`) for both the stream
position and the within-attempt timestep, so there is no embedding table and no maximum length; the
backbone is the repo's causal `nn.TransformerEncoder` stack (as `SearchTransformerForDiffusion`),
not GPT2's table-bound `wpe`. Batches use the existing `attention_mask` collate.
Guard only (not a design parameter): a rollout that has neither stopped nor reached the goal after
`n_free_cells` steps (the layout's number of free cells, i.e. longer than any simple path) is aborted
and counted as a failure, and the fraction of such attempts is logged (`frac_guard_hit`) so a
non-terminating policy is visible rather than silently truncated.

### Training (`compute_loss`, reusing `LSTMSearchPolicy`'s structure)
Each training sample is one expert episode. The model must learn to use 0, 1, …, 7 earlier attempts
(at eval, attempt j sees j−1 earlier ones). User choice: **all 8 prefixes per sample** (as the round-2
LSTM did). Per sample, generate 7 attempts sequentially under `inference_mode` with the same rollout code
the harness uses (snap, stop radius, model-predicted stop), then build the 8 sequences `[attempts 1..k | expert]` for
k = 0..7 and take masked NLL on the expert tokens of each (B·8 sequences per batch; one attempt
generation serves all 8 prefixes since attempt k is produced with attempts <k in context either way).
`max_actions=1` ⇒ k=0 only (BoN arm, plain BC). l2s_online's random-k rule stays available as a
config knob (`context_prefixes: all|random`) if compute is tight.

### Eval (`l2s/eval_openloop.py`, autoregressive mode)
Harness keeps one policy stream per episode: `policy.reset()` once, then
`predict_step(obs, prev_action, prev_reward, prev_terminal)` per env step; it supplies the same
reward/terminal values training used (shared helper). Beyond `max_actions` attempts the stream is
rebuilt from the last `max_actions−1` attempts (same rule in training's rollout helper).
Metrics per (split, ckpt), for k = 1..N:
- `perf@k`: any of the first k attempts reached the goal (attempts run sequentially, no selection);
- `argmax@k`: success of the attempt with the highest verifier value among the first k (what a
  verifier-picked episode would score); plus `success_exact` versions, per-episode `first_success_k`,
  best-so-far BFS, diversity columns unchanged. The old `max`/`last` strategy rows are replaced by these.

### Arms (all: decision-search transformer, Gaussian head + snap)
- `dt_st_traj`: context = earlier attempts' actions + one −BFS(final) per attempt (max_actions 8).
- `dt_st_step`: context = actions + per-step BFS feedback + −BFS(final) per attempt (max_actions 8).
- `dt_bon`: max_actions 1 (no context; the round-2 `lstm_bon` counterpart: i.i.d. rollouts, verifier picks).
- No LSTM arm (user: removed; in round 2 the LSTM *was* the policy, here the transformer replaces it).
  `dst64` / LSTM round-2 rows stay as reference, not retrained.

### Reference run: `sriyash/l2s_online` unchanged, on PushT
- Worktree of `origin/sriyash/l2s_online` (86d5ef9) under `/gscratch/robotics/harine/l2s_online_ref`
  (home quota is full); env `vae_pushT_l2s` (transformers 4.46.3 present; confirm gym/pymunk import).
- Data: [data/pusht_cchi_v7_replay.zarr](data/pusht_cchi_v7_replay.zarr) already has the derived
  `agent_pos`/`block_pos` keys the branch's dataset reads; point `zarr_path` at it via CLI override
  only (no code change).
- Train: `train_online_search.yaml` defaults (n_trajs 8, 100 outer × 100 inner, pool 128, 32 async
  envs) → one GPU + 16 CPUs, ckpt partition with requeue, finite `--time`; W&B project
  `pusht_online_search`. Then `eval_bon.py` (context-free best-of-n on its 50 test episodes) and the
  runner numbers. Report as given, with the note that its eval never uses context.

## Files

**Fork (`harine/maze`, this worktree)**
- NEW `diffusion_policy/policy/dt_search_policy.py` — `DTSearchPolicy`: token embeddings,
  causal `nn.TransformerEncoder` + `SinusoidalPosEmb` (same building blocks as
  `SearchTransformerForDiffusion`), Gaussian action head (reuse the LSTM arm's state-independent
  `log_std` + floor) + terminal logit, `reset/predict_step`, `rollout_candidates_torch`,
  `compute_loss`. Rollout/value/snap helpers come from `common/maze_verifier.py` and
  `lstm_search_policy.py` (import, don't copy). Comments terse.
- NEW `diffusion_policy/config/procgen_maze_dt_search.yaml` (+ 10-line `procgen_maze_dt_bc.yaml` override
  as `procgen_maze_lstm_bc.yaml` does): dataset `return_sequences: true, pad 0/0`, `val_ratio: 0.2`,
  `reward_mode`, `max_actions`, batch 64, AdamW 1e-4, cosine, 10k steps,
  `checkpoint_every: 1000`, `TrainMLPImageWorkspace` as the LSTM arm.
- `diffusion_policy/dataset/sim2real_image_dataset.py`: val mask by unique (start, goal) pair
  (the 800 episodes cover 712 pairs; episode-level `get_val_mask` leaks 9/40 val pairs into train),
  and `get_normalizer` fit on train frames only (pattern: l2s_online's `episode_frame_mask`).

**L2S (`/gscratch/robotics/harine/L2S`, branch per round-2 convention)**
- `l2s/eval_openloop.py`: reward/terminal fed to `predict_step`; perf@k metrics; drop strategies.
- `l2s/run_procgen_eval.py`, `l2s/run_procgen_train.py`, `l2s/procgen_pipeline.py` (`OPENLOOP_ALGOS`),
  `slurm/launch_openloop11.sh`: register `dt_st_traj`, `dt_st_step`, `dt_bon`; new `OL_ROOT`
  `openloop11dt_pipeline_quadrant`; W&B project unchanged.
- `scripts/plot_openloop_quadrant.py` / status script: perf@k curves instead of strategy rows.

## Verification
1. `python -m py_compile` on touched files (login node); pytest on a compute node.
2. Token-equality check (as round 2's N=12 check): generate 12 attempts with the training rollout and
   with the harness from the same start/seed; assert identical `(obs, prev_action, prev_reward,
   prev_terminal)` streams, identical sliding-window boundaries, identical values.
3. `bash slurm/launch_openloop11.sh smoke` with the new algos (2 min on ckpt); confirm
   `n_unique_traj > 1` for `dt_bon` and non-empty `success@k` curve.
4. Launch: `unset WANDB_API_KEY`; check `scontrol show reservation`; finite `--time`; ckpt evals at
   every 1k checkpoint as before; report success@k with coverage-style pairing (success + success_exact).
