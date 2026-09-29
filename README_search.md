# Learning to Search (L2S) — Procgen Maze Experiments

This document covers the search / test-time-compute branch of this repo: policies that
generate **multiple candidate action chunks**, score each one with a **verifier**, and
condition the next candidate on the `(action, value)` history. It is the counterpart to
plain behavior cloning (BC) and best-of-N (BoN) sampling on the same Procgen maze task.

Everything here lives under `diffusion_policy/`; the upstream Diffusion Policy README is
[`README.md`](README.md), and teleop docs are in [`README_teleop.md`](README_teleop.md).

---

## 1. What is in here

| Component | Path |
| --- | --- |
| Maze verifier (ground-truth value oracle) | `diffusion_policy/common/maze_verifier.py` |
| MLP/GPT2 search policy | `diffusion_policy/policy/search_policy.py` |
| Diffusion-transformer search policy | `diffusion_policy/policy/diffusion_transformer_search_policy.py` |
| Training workspace for both | `diffusion_policy/workspace/train_mlp_image_workspace.py` |
| Task definition (shapes, dataset) | `diffusion_policy/config/task/procgen_maze.yaml` |
| Configs | `diffusion_policy/config/procgen_maze_*.yaml` |

Four configs define the comparison:

| Config | Policy | Role |
| --- | --- | --- |
| `procgen_maze_bc.yaml` | `DiffusionUnetImagePolicy` | BC baseline; also the BoN backbone |
| `procgen_maze_diffusion_transformer.yaml` | `DiffusionTransformerHybridImagePolicy` | Transformer BC / BoN backbone |
| `procgen_maze_search.yaml` | `SearchPolicy` | GPT-2 trunk, Gaussian action head, verifier in the loop |
| `procgen_maze_diffusion_transformer_search.yaml` | `DiffusionTransformerSearchPolicy` | Diffusion search transformer, verifier in the loop |

The maze layout generation, expert data collection, evaluation and plotting scripts live in
the companion [L2S repo](https://github.com/sriyash421/L2S) (`l2s/generate_procgen_mazes.py`,
`l2s/maze.py`, `l2s/eval_bc.py`, `l2s/plot_test_time_compute.py`, `l2s/visualize_*.py`).
Only the verifier is vendored here so that training in this repo has no `l2s` dependency
— see [§5](#5-the-verifier).

---

## 2. Setup

```bash
conda env create -f conda_environment.yaml   # or reuse your existing env (e.g. `hitl`)
conda activate robodiff
pip install -e .
```

Training the search policies needs only this repo. Data generation and evaluation
additionally need the L2S repo on your `PYTHONPATH`:

```bash
git clone https://github.com/sriyash421/L2S && pip install -e L2S
```

### Data

```bash
# 1. Procgen layouts, disjoint train/eval splits
python l2s/generate_procgen_mazes.py \
  --out data/procgen_mazes/layouts.npz \
  --num-layouts 1000 \
  --train-ratio 0.8

# 2. Expert demonstrations from the *train* split only
python l2s/maze.py \
  --layouts data/procgen_mazes/layouts.npz \
  --split train \
  --maze-fraction 0.1 \
  --out data/procgen_maze_expert/data.zarr \
  --episodes 800 \
  --steps 100
```

Point training at the resulting directory with
`task.dataset.dataset_dir=data/procgen_maze_expert`.

---

## 3. Training

```bash
# Search transformer (GPT-2 trunk)
python diffusion_policy/train.py \
  --config-name procgen_maze_search.yaml \
  task.dataset.dataset_dir=data/procgen_maze_expert \
  training.seed=0 task.dataset.seed=0 \
  policy.verifier_noise=0.0

# Diffusion search transformer
python diffusion_policy/train.py \
  --config-name procgen_maze_diffusion_transformer_search.yaml \
  output_dir=data/procgen_maze_diffusion_transformer_search \
  name=procgen_maze_diffusion_transformer_search

# BC / BoN baselines
python diffusion_policy/train.py --config-name procgen_maze_bc.yaml
python diffusion_policy/train.py --config-name procgen_maze_diffusion_transformer.yaml
```

Standard Hydra overrides apply (`training.device=cpu`, `training.debug=True`,
`logging.mode=offline`, `training.max_train_steps=10` for smoke checks).

Both search policies are trained by `TrainMLPImageWorkspace`
(`diffusion_policy/workspace/train_mlp_image_workspace.py`). During training it calls the
verifier inside `compute_loss`, so **the verifier runs on every training step**, not just
at evaluation. The `sample_every` block logs `train_action_mse_error_min` /
`_avg` (best-of-K vs. mean-of-K chunk error) and `train_action_value` (mean verifier value)
whenever the policy exposes a `.verifier` attribute.

---

## 4. Hyperparameters

### Shared task settings (`config/task/procgen_maze.yaml`)

| Key | Value | Meaning |
| --- | --- | --- |
| `image_shape` | `[3, 11, 11]` | 11×11 maze grid; channel 0 is the wall/free map |
| `obs` | `image`, `agent_pos` (2), `goal_pos` (2) | verifier reads all three |
| `action` | `[2]` | continuous `(dx, dy)` delta per step |
| `horizon` | 16 | predicted chunk length |
| `n_obs_steps` | 2 | observation context |
| `n_action_steps` | 8 | steps actually executed per chunk |
| `val_ratio` | 0.05 | train/val split of the expert data |

### `procgen_maze_search.yaml` — `SearchPolicy`

| Key | Value | Notes |
| --- | --- | --- |
| `hidden_dim` | 256 | GPT-2 embedding width |
| `hidden_depth` | 4 | GPT-2 layers (`n_head=4`, dropout 0.1 on resid/embd/attn) |
| `max_actions` | 16 | candidates generated per observation (search width) |
| `corrupt_obs` | `False` | if true, obs features are diffused with a 100-step DDPM noise schedule |
| `mask_obs` / `concat_obs` | `False` | ablations: hide obs from later tokens / concat obs into every token |
| `verifier_noise` | 0.0 | multiplicative noise on verifier values (see §5) |
| optimizer | AdamW, `lr=1e-4`, betas `(0.95, 0.999)`, `wd=1e-6` | cosine schedule, 500 warmup steps |
| batch size | 64 | `max_gradient_steps=10000`, `max_train_steps=100`/epoch |
| loss | Gaussian NLL of the expert chunk under every candidate token; `log_std` clamped to `[-5, 2]` |

### `procgen_maze_diffusion_transformer_search.yaml` — `DiffusionTransformerSearchPolicy`

| Key | Value | Notes |
| --- | --- | --- |
| scheduler | DDIM, 100 train steps, `beta 1e-4 → 0.02`, `squaredcos_cap_v2`, `epsilon` | `clip_sample=True` |
| `num_inference_steps` | 8 | denoising steps per candidate |
| `n_layer` / `n_head` / `n_emb` | 4 / 4 / 256 | `n_cond_layers=2` encoder layers over the conditioning memory |
| `p_drop_emb` / `p_drop_attn` | 0.0 / 0.2 | |
| `causal_attn` | `False` | |
| `max_actions` | 16 | `max_context_actions = max_actions - 1 = 15` tokens of `(action, value)` history |
| `maze_path` | `null` | verifier reads the maze from the image observation instead |
| batch size | 256 | AdamW `lr=1e-4`, cosine, 500 warmup, 100 epochs |

### Baselines

| | `procgen_maze_bc` | `procgen_maze_diffusion_transformer` |
| --- | --- | --- |
| backbone | UNet, `down_dims=[256,512,1024]`, ResNet18 obs encoder | Transformer, 4 layers / 4 heads / 256 emb |
| scheduler | DDPM, 100 steps | DDIM, 100 train / 8 inference steps |
| batch size | 128 | 256 |
| steps | `max_gradient_steps=10000` | 100 epochs |

### Search-time knobs (evaluation)

| Flag (`l2s/eval_bc.py`, `l2s/plot_test_time_compute.py`) | Meaning |
| --- | --- |
| `--n-samples` | number of candidate chunks per decision (the test-time compute axis) |
| `--sample-strategy max\|last\|sample` | which candidate to execute |
| `--noise` | verifier noise at eval time |
| `--rollout-steps`, `--eval-episodes`, `--seeds`, `--env-seed` | rollout budget and seeding |

Note `max_actions=16` is the *trained* context width. `predict_n_actions` will go beyond it
by sliding the `(action, value)` context window, so `--n-samples 128` works with a policy
trained at `max_actions=16` — the model just never sees more than 15 past candidates at once.

---

## 5. The verifier

`diffusion_policy/common/maze_verifier.py` → `MazeVerifier`.

It is a **ground-truth oracle**, not a learned critic: given an observation and a candidate
action chunk, it simulates the chunk against the maze walls and returns the negative
shortest-path distance from the resulting position to the goal.

```python
from diffusion_policy.common.maze_verifier import MazeVerifier

verifier = MazeVerifier(env=None, maze_path=None, device='cuda:0',
                        success_radius=0.5, noise=0.0)
values = verifier.get_value(obs_dict, action)   # (B,) — higher is better
```

### Constructor arguments

| Arg | Default | Meaning |
| --- | --- | --- |
| `env` | `None` | optional live maze env; when given and `B == 1`, rollout is delegated to `env.rollout_actions` |
| `maze_path` | `None` | `.npy` maze to precompute an all-pairs distance table for |
| `device` | `"cpu"` | device for the distance fields; configs pass `${training.device}` |
| `success_radius` | 0.5 | goal tolerance |
| `noise` | 0.0 | value corruption: `value *= (1 + noise * N(0, 1))` — models an imperfect verifier |

### How `get_value` works

1. **Read state.** `agent_pos[:, -1]` and `goal_pos[:, -1]` from the obs dict (or `obs[:, -1, :2]`
   / `[:, -1, 2:4]` for the low-dim variant). A 4-D `action` is squeezed to its first candidate.
2. **Recover the maze.** Channel 0 of the last image frame is the wall map; `wall < 0.5` is free
   space (`_free_from_obs_torch`). No maze file is needed at eval time — each batch element may
   have a different layout.
3. **Roll the chunk out.** `_rollout_actions_torch` walks each `(dx, dy)` delta cell by cell along
   a Bresenham-style line, stopping at the first wall or out-of-bounds cell. Fully batched.
4. **Score the endpoint.** A BFS distance field to the goal is computed by iterated min-pooling
   over free cells (`_goal_distance_field_torch`), then sampled at the final position. Unreachable
   cells get `h*w + 1`. The value is the **negative** distance, so higher = closer to goal.
5. **Add noise** if `noise > 0`.

Fallbacks, in order: live-env rollout (batch of 1) → batched torch image path (the normal case)
→ per-maze NumPy rollout → naive `cumsum` of deltas ignoring walls.

### Caching

- `_ensure_table` / `get_shortest_path_table`: exact all-pairs BFS table, keyed by `maze.tobytes()`.
  Used by the `maze_path` / live-env paths.
- `_torch_distance_cache`: caches the last goal-distance field, keyed by the `(image, goal)` tensor
  storage pointers. Because the search loop calls `get_value` once per candidate with the *same*
  observation, this makes candidates 2…N nearly free — only the rollout is recomputed.

### Where it is called

- `SearchPolicy.predict_action` (`search_policy.py:241`) and `.compute_loss` (`:319`)
- `DiffusionTransformerSearchPolicy.predict_action` (`diffusion_transformer_search_policy.py:549`)
- `TrainMLPImageWorkspace` sampling block (`train_mlp_image_workspace.py:293`)

The loop is: sample candidate *k* conditioned on `(action, value)` pairs `1…k-1` → score it with
the verifier → append → repeat. `predict_n_actions` extends past `max_actions` by sliding that
context window.

### Caveats

- **It is an oracle.** It reads the true wall map and computes exact shortest paths. Results are
  upper bounds on what a learned verifier would give; use `verifier_noise` / `--noise` to study
  degradation.
- **Grid-snapped.** Positions are rounded to integer cells (`torch.round`), so sub-cell action
  precision is invisible to the value.
- `_goal_distance_field_torch` runs `h*w - 1` min-pool iterations — fine at 11×11, quadratic in
  grid area for larger mazes.
- This file is a **vendored copy** of `l2s/verifier.py` from the L2S repo. If you change one,
  mirror it in the other; the eval scripts there still import `l2s.verifier`.

---

## 6. Evaluation and plots

```bash
# single checkpoint + rollout panel
python l2s/eval_bc.py --checkpoint <ckpt> \
  --layouts data/procgen_mazes/layouts.npz --split eval \
  --eval-episodes 20 --n-samples 32 --rollout-steps 200 \
  --out data/eval_bc.png

# test-time compute curves
python l2s/plot_test_time_compute.py \
  --checkpoints <bc_ckpt> <search_ckpt> \
  --labels diffusion diffusion_search \
  --group-labels "Diffusion" "Diffusion Search" \
  --n-samples 1 2 4 8 16 32 64 128 \
  --sample-strategy sample --seeds 0 1 2 --noise 0.2
```

For paired comparisons keep `--seed` / `--env-seed` fixed across policies, and always evaluate on
`--split eval` — the expert data is generated from the train split only.

Checkpoints land in `data/outputs/<date>/<time>_<name>_<task>/checkpoints/` (`latest.ckpt` plus
per-step files); the top-k monitor is `val_loss` for `procgen_maze_search` and `train_loss` for the
diffusion search config.
