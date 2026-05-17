# Noising + Learning-to-Search (L2S) Policy — Maze

This document describes how to train the noising + Learning-to-Search (L2S)
policy that currently lives in this repo. Today the recipe is wired up for the
**maze** task; the same pipeline is being adapted to use the
[RoboMonkey](https://github.com/robomonkey-vla/RoboMonkey) HTTP verifier with
RGB observations (see [Adapting to RoboMonkey](#adapting-to-robomonkey-coming-next)
at the bottom).

## Overview

The policy combines two ideas:

1. **Noising** — at training time, encoded observation features are corrupted
   by a DDPM noise scheduler before being fed to the transformer trunk. This
   simulates noisy / partial context and is implemented in
   `SearchPolicy.corrupt_obs_features` in
   [`diffusion_policy/policy/search_policy.py`](diffusion_policy/policy/search_policy.py)
   (gated by `policy.corrupt_obs: True`).

2. **Learning to Search (L2S)** — a GPT-2 trunk consumes a sequence

   ```
   [obs_token, action_value_token_1, action_value_token_2, ..., action_value_token_K]
   ```

   At each position the model outputs a Normal over a full action chunk of
   shape `(horizon, action_dim)`. At inference (and during the inner rollout
   inside `compute_loss`), candidates are sampled one-by-one; each new
   candidate is scored by an external `verifier`, and the `(action, value)`
   pair is appended to the context for the next sample. See `predict_action`
   and `predict_n_actions` in
   [`diffusion_policy/policy/search_policy.py`](diffusion_policy/policy/search_policy.py).

Three input/output structural variants are supported via flags on the policy:

- **default** — the obs token is a separate prefix; subsequent tokens encode
  `(action, value)` pairs.
- **`mask_obs: True`** — the first action is predicted by a small MLP from the
  obs features; only later tokens go through the transformer trunk.
- **`concat_obs: True`** — obs features are concatenated into every token
  (no dedicated obs prefix).

`mask_obs` and `concat_obs` are mutually exclusive (asserted in
`SearchPolicy.__init__`).

## Files involved

| Concern    | File                                                                                                                                |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Policy     | [`diffusion_policy/policy/search_policy.py`](diffusion_policy/policy/search_policy.py) (`SearchPolicy`)                             |
| Workspace  | [`diffusion_policy/workspace/train_mlp_image_workspace.py`](diffusion_policy/workspace/train_mlp_image_workspace.py) (`TrainMLPImageWorkspace`) |
| Task       | [`diffusion_policy/config/task/maze.yaml`](diffusion_policy/config/task/maze.yaml) (low-dim maze, `FlattenObsEncoder`)                |
| Verifier   | `MazeVerifier` from the external `l2s` package (`from l2s.verifier import MazeVerifier`)                                            |

The L2S configs that ship today:

- [`maze_search.yaml`](diffusion_policy/config/maze_search.yaml) — vanilla L2S (no noising).
- [`maze_search_corruption.yaml`](diffusion_policy/config/maze_search_corruption.yaml) — L2S **with noising** (`corrupt_obs: True`).
- [`maze_search_mask_obs.yaml`](diffusion_policy/config/maze_search_mask_obs.yaml) / [`maze_search_mask_obs_corruption.yaml`](diffusion_policy/config/maze_search_mask_obs_corruption.yaml) — `mask_obs` variant ± noising.
- [`maze_search_concat_obs.yaml`](diffusion_policy/config/maze_search_concat_obs.yaml) / [`maze_search_concat_obs_corruption.yaml`](diffusion_policy/config/maze_search_concat_obs_corruption.yaml) — `concat_obs` variant ± noising.

For comparison, the plain MLP behavior-cloning baselines on the same task are
[`maze_bc_mlp.yaml`](diffusion_policy/config/maze_bc_mlp.yaml) and
[`maze_bc_mlp_corruption.yaml`](diffusion_policy/config/maze_bc_mlp_corruption.yaml).

## Data

The maze task uses the dataset class
`diffusion_policy.dataset.sim2real_image_dataset.Sim2RealImageMultiDataset` (see
`task.dataset` in [`task/maze.yaml`](diffusion_policy/config/task/maze.yaml)).
You need:

- **Expert trajectories** at `data/maze_expert/` — trajectories the policy is
  trained to imitate.
- **Maze layout** at `data/maze.npy` — the grid the in-process `MazeVerifier`
  uses to score candidate action chunks. Path is set per-config via
  `policy.maze_path`.

```
data/
├── maze_expert/   # demonstration dataset
└── maze.npy       # maze layout used by MazeVerifier
```

The `l2s` package (which provides `MazeVerifier`) must be importable:

```bash
pip install -e /path/to/l2s
```

## Train

The training entrypoint is the workspace itself:

```bash
# noising + L2S
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=maze_search_corruption \
    training.seed=42

# vanilla L2S (no noising)
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=maze_search

# mask_obs variants
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=maze_search_mask_obs_corruption

# concat_obs variants
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=maze_search_concat_obs_corruption
```

Outputs (checkpoints, logs, wandb artifacts) land under
`data/outputs/<date>/<time>_<name>_<task_name>/` per the `hydra.run.dir`
template at the bottom of each config.

## Key knobs

All on `policy:` in the config.

| Field                | Type    | Notes                                                                                                          |
| -------------------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| `corrupt_obs`        | `bool`  | If `True`, encoded obs features are noised every forward pass via a DDPM scheduler (the **noising** ingredient). |
| `mask_obs`           | `bool`  | First-action head is an MLP off the obs features; subsequent action tokens go through the trunk.                |
| `concat_obs`         | `bool`  | Obs features are concatenated into every action-value token instead of being a separate prefix.                  |
| `max_actions`        | `int`   | Search width — number of `(action, value)` candidates the model conditions on.                                  |
| `hidden_dim`         | `int`   | GPT-2 trunk width. (Note: the `concat_obs` config bumps this to 512 because each token is a concat of two halves.) |
| `hidden_depth`       | `int`   | Number of transformer layers.                                                                                   |
| `maze_path`          | `str`   | Path to the maze grid loaded by the in-process `MazeVerifier`.                                                  |
| `device`             | `str`   | Device for the `MazeVerifier` instance.                                                                         |

Top-level (outside `policy:`):

| Field            | Notes                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------- |
| `horizon`        | Length of an action chunk (the trunk predicts `(horizon, action_dim)` per token).           |
| `n_obs_steps`    | Number of past obs steps fed to the encoder.                                                |
| `n_action_steps` | How many of the predicted steps are actually executed at inference time.                    |

## What `compute_loss` does

For each batch:

1. Encode the obs window with `obs_encoder` and project to `hidden_dim`.
2. Under `torch.inference_mode()`, run `predict_action(obs_dict, verifier=self.verifier, n_actions=max_actions-1)`. This autoregressively samples `max_actions-1` candidate action chunks; each is scored by the in-process `MazeVerifier` via `verifier.get_value(obs_dict, action)`.
3. Pack `(actions, values)` into per-token features via `act_projection`.
4. Run a single forward pass through the GPT-2 trunk to get a `Normal(mean, std)` per token.
5. Loss = `-log_prob(target_action)` averaged over batch and over `max_actions` tokens. Each token is trained to put mass on the ground-truth action *given* the search context that preceded it.

The same `predict_action` path is used at sampling time (`sample_every`) to log
mean / min MSE across the candidate set as `train_action_mse_error_min` /
`train_action_mse_error_avg`.

## Inference / rollout

`predict_n_actions(obs_dict, verifier, n_actions)` is the public inference API.
If `n_actions <= max_actions` it just runs the autoregressive loop once. If you
need more candidates than the trunk's context length supports, it slides a
window of `(action, value)` history across additional rollouts. Output:
`(actions, values)` of shape `(B, n_actions, horizon, action_dim)` and
`(B, n_actions)` respectively — pick the argmax over `values` for greedy
deployment.
