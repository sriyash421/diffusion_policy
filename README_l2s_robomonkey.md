# README_L2S_RoboMonkey — Eggplant-in-basket, state-based L2S

End-to-end recipe for training the state-based RoboMonkey search policies on
the SIMPLER `widowx_put_eggplant_in_basket` task, scored by the RoboMonkey
verifier (`monkey-verifier`).

Two policy variants are available:

| Policy class | Action head | Config |
|---|---|---|
| `SearchPolicyRoboMonkey` | Gaussian (`Normal` per token) | `robomonkey_eggplant_search_state.yaml` |
| `SearchPolicyRoboMonkeyDiffusion` | Conditional diffusion (encoder-decoder transformer, joint K-candidate denoising) | `robomonkey_eggplant_search_state_diffusion.yaml` |

Both classes live in [diffusion_policy/policy/search_policy_robomonkey.py](diffusion_policy/policy/search_policy_robomonkey.py).

The diffusion variant is a port of Sriyash's maze
`DiffusionTransformerSearchPolicy` — same `SearchTransformerForDiffusion`
backbone, with the verifier injected via Hydra and a `_state_keys` filter
that keeps RGB out of the policy's obs encoder.

The **policy input is state only**. Images are used solely by the verifier
to score candidate actions during `compute_loss`. There are two ways to get
images into the verifier:

| Scenario                                  | Verifier class               | When to use                                                                |
| ----------------------------------------- | ---------------------------- | -------------------------------------------------------------------------- |
| **A.** Trajectories collected *without* images | `RoboMonkeyStateVerifier`    | Old shards that only have low-dim keys (e.g. `state0.zarr` under `~/data/eggplant_in_basket/state_only`). The verifier boots a SIMPLER env and renders an RGB frame from the saved state at training time. |
| **B.** Trajectories collected *with* images    | `RoboMonkeyVerifier`         | New shards under `~/data/eggplant_in_basket` whose zarr also contains `data/obs/agentview_image`. The verifier reads the image straight from the batch — no SIMPLER render at train time. Faster. |

Repos and locations:

- This repo: `~/RoboMonkey/diffusion_policy`
- RoboMonkey: `~/RoboMonkey`
- Verifier server (HTTP mode): `~/RoboMonkey/monkey-verifier/src/infer_server.py`
- State-only data: `~/data/eggplant_in_basket/state_only/state0.zarr`
- State + image data: `~/data/eggplant_in_basket/{state0,state1}.zarr`

Conda envs:

- `simpler_env` — SIMPLER + zarr + diffusion_policy training
- `sglang-vla` — OpenVLA action server (data collection only)
- `monkey-verifier` — RoboMonkey verifier (HTTP server or in-process model)

---

## 0. Prereqs (one-time)

```bash
cd ~/RoboMonkey
bash scripts/setup.sh   # creates simpler_env, sglang-vla, monkey-verifier envs
```

---

## 1. (Optional) Collect data

Skip unless you want fresh trajectories. Use **(a)** to add to the
no-image shard, **(b)** to produce a shard with images baked in.

### 1a. OpenVLA action server (terminal A — leave running)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang-vla
cd ~/RoboMonkey
bash scripts/run_openvla_server.sh
```

### 1b. Collect rollouts (terminal B)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env
cd ~/RoboMonkey

# state only (writes to ~/data/eggplant_in_basket/state_only/state0.zarr)
bash scripts/collect_eggplant_in_basket.sh

# state + per-step agentview RGB (uint8, T×H×W×3 under data/obs/agentview_image)
SAVE_IMAGES=True bash scripts/collect_eggplant_in_basket.sh
```

The "with images" output also lives under
`~/data/eggplant_in_basket/state*.zarr` for the scenario-B path below.

---

## 2. Verifier backend

The policy's `compute_loss` calls the verifier on every batch. Two ways to
run the reward model:

### 2a. HTTP server (default in the config)

```bash
# terminal C — leave running
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monkey-verifier
cd ~/RoboMonkey/monkey-verifier/src
python infer_server.py            # 0.0.0.0:3100

# Health check
curl http://127.0.0.1:3100/
```

### 2b. In-process (no HTTP, no JPEG round-trip — faster)

Append `policy.verifier.server_url=in_process` to the training command. The
verifier loads `RobotRewardModel` into the training python process and
scores all `B` `(image, action)` pairs in a single GPU forward. Requires
`monkey-verifier` deps importable from the training env (easiest: train
inside the `monkey-verifier` conda env, or set
`MONKEY_VERIFIER_SRC=$HOME/RoboMonkey/monkey-verifier/src`).

When in-process is enabled the HTTP server in §2a is unused; you can skip
starting it.

---

## 3. Train the L2S search policy

Files involved:

- Policy: [diffusion_policy/policy/search_policy_robomonkey.py](diffusion_policy/policy/search_policy_robomonkey.py)
- Verifier clients: [diffusion_policy/policy/verifiers.py](diffusion_policy/policy/verifiers.py)
- Task config: [diffusion_policy/config/task/robomonkey_eggplant_state.yaml](diffusion_policy/config/task/robomonkey_eggplant_state.yaml)
- Training config: [diffusion_policy/config/robomonkey_eggplant_search_state.yaml](diffusion_policy/config/robomonkey_eggplant_search_state.yaml)

`policy.corrupt_obs` toggles DDPM-style obs-feature noising. The flag is
baked into the run name (`..._corrupt` / `..._clean`), so it propagates to
the wandb run name, hydra run dir, and the saved `config.yaml`.

### 3a. Scenario A — trajectories *without* images (SIMPLER render at train time)

Verifier class: `RoboMonkeyStateVerifier`. Boots a SIMPLER env and renders
the agentview frame from saved arm qpos + asset poses for each batch
element, then scores via the (HTTP or in-process) reward model.

Override the task/verifier on the CLI to point at the state-only shard.
We also need the env vars the SIMPLER collector uses:

```bash
# terminal D
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

cd ~/RoboMonkey/diffusion_policy
export PRISMATIC_DATA_ROOT=$HOME/RoboMonkey/openvla-mini
export PYTHONPATH=$HOME/RoboMonkey/openvla-mini:$HOME/RoboMonkey/diffusion_policy:$HOME/RoboMonkey/monkey-verifier/src
export MONKEY_VERIFIER_SRC=$HOME/RoboMonkey/monkey-verifier/src
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True \
    task.dataset.dataset_dir=$HOME/data/eggplant_in_basket/state_only \
    'task.shape_meta.obs={arm_joint_pos:{shape:[6],type:low_dim},end_effector_pose:{shape:[7],type:low_dim},joint_vel:{shape:[6],type:low_dim},last_arm_action:{shape:[6],type:low_dim},last_gripper_action:{shape:[1],type:low_dim},insertive_asset_pose:{shape:[7],type:low_dim},receptive_asset_pose:{shape:[7],type:low_dim}}' \
    policy.verifier._target_=diffusion_policy.policy.verifiers.RoboMonkeyStateVerifier \
    +policy.verifier.simpler_task=widowx_put_eggplant_in_basket \
    +policy.verifier.arm_qpos_key=arm_joint_pos \
    +policy.verifier.src_pose_key=insertive_asset_pose \
    +policy.verifier.tgt_pose_key=receptive_asset_pose \
    +policy.verifier.resize_size=256 \
    ~policy.verifier.image_obs_key
```

`policy.corrupt_obs=False` for the un-noised variant. Add
`policy.verifier.server_url=in_process` to skip the HTTP server.

### 3b. Scenario B — trajectories *with* images (recommended, faster)

Verifier class: `RoboMonkeyVerifier`. Reads `agentview_image` straight from
the dataset; no SIMPLER render, no `PRISMATIC_DATA_ROOT`, no headless GL,
no `xvfb-run`.

This is what the checked-in
[robomonkey_eggplant_search_state.yaml](diffusion_policy/config/robomonkey_eggplant_search_state.yaml)
config already does (task points at `~/data/eggplant_in_basket`, verifier
is `RoboMonkeyVerifier` with `image_obs_key: agentview_image`).

```bash
# terminal D
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env       # or monkey-verifier if using in_process
cd ~/RoboMonkey/diffusion_policy
export MONKEY_VERIFIER_SRC=$HOME/RoboMonkey/monkey-verifier/src

# (1) Noised — DDPM obs-feature corruption ON  → name=..._corrupt
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True

# (2) Un-noised search policy             → name=..._clean
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=False

# (3) Either of the above + in-process verifier (no HTTP, single GPU forward
#     for the whole batch — fastest path):
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    policy.corrupt_obs=True \
    policy.verifier.server_url=in_process
```

Smoke test (no wandb, single train step):

```bash
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state \
    training.max_train_steps=1 logging.mode=disabled
```

### 3c. Conditional-diffusion variant — `SearchPolicyRoboMonkeyDiffusion`

Same task / dataset / verifier as §3b but the action head is a conditional
diffusion transformer instead of a Gaussian. The verifier defaults to
`in_process` in the config, so no HTTP server is needed — train inside the
`monkey-verifier` env (or set `MONKEY_VERIFIER_SRC`).

Files:

- Policy: [diffusion_policy/policy/search_policy_robomonkey.py](diffusion_policy/policy/search_policy_robomonkey.py) — `SearchPolicyRoboMonkeyDiffusion` (alongside the Gaussian `SearchPolicyRoboMonkey`)
- Backbone (shared with maze): [diffusion_policy/policy/diffusion_transformer_search_policy.py](diffusion_policy/policy/diffusion_transformer_search_policy.py) — `SearchTransformerForDiffusion`
- Training config: [diffusion_policy/config/robomonkey_eggplant_search_state_diffusion.yaml](diffusion_policy/config/robomonkey_eggplant_search_state_diffusion.yaml)

```bash
# terminal D
source ~/miniconda3/etc/profile.d/conda.sh
conda activate monkey-verifier         # in-process verifier needs these deps
cd ~/RoboMonkey/diffusion_policy
export MONKEY_VERIFIER_SRC=$HOME/RoboMonkey/monkey-verifier/src

# (1) Noised — DDPM obs-feature corruption ON  → name=..._diffusion_corrupt
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion

# (2) Un-noised                                → name=..._diffusion_clean
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion \
    policy.corrupt_obs=False

# (3) HTTP verifier instead of in-process (start §2a server first)
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion \
    policy.verifier.server_url=http://127.0.0.1:3100

# (4) Faster sampling during rollouts (fewer DDIM steps)
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion \
    policy.num_inference_steps=4
```

Smoke test:

```bash
python diffusion_policy/workspace/train_mlp_image_workspace.py \
    --config-name=robomonkey_eggplant_search_state_diffusion \
    training.max_train_steps=1 logging.mode=disabled
```

### Output dir per run

```
~/RoboMonkey/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS_robomonkey_eggplant_search_state_{corrupt,clean}_<task_name>>/
    ├── checkpoints/
    ├── .hydra/config.yaml         # frozen config snapshot (corrupt_obs baked in)
    └── wandb/
```

---

## 4. Evaluate in SIMPLER

Use the RoboMonkey eval driver — it loads checkpoints from
`~/RoboMonkey/diffusion_policy/data/outputs/...`:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env
cd ~/RoboMonkey

TASK=widowx_put_eggplant_in_basket \
bash scriptsv2/run_eval_diffusion_policy.sh \
    ~/RoboMonkey/diffusion_policy/data/outputs/<YYYY.MM.DD>/<HH.MM.SS_robomonkey_eggplant_search_state_corrupt_*>/checkpoints/latest.ckpt \
    100
```

For Best-of-N action verification at eval time, see §4d in
[~/RoboMonkey/README_L2S.md](file:///home/harine/RoboMonkey/README_L2S.md).
