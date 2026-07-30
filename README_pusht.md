# Push-T experiments (branch `harine/pushT`)

This branch adds a **feedback-conditioned** Push-T task, an **online-search** in-context
policy, and **best-of-N** evaluation/visualization on top of upstream Diffusion Policy.
The root [README.md](README.md) is still the upstream Columbia doc and does **not** cover
any of this — use this file for the Push-T work here.

What's new relative to upstream:

| Area | Files |
| --- | --- |
| Feedback obs (16-dim goal-vs-achieved keypoint displacement) | [`diffusion_policy/env/pusht/feedback_util.py`](diffusion_policy/env/pusht/feedback_util.py), [`pusht_feedback.py`](diffusion_policy/env/pusht/pusht_feedback.py) |
| Task configs | [`config/task/pusht_image.yaml`](diffusion_policy/config/task/pusht_image.yaml), [`config/task/pusht_image_online.yaml`](diffusion_policy/config/task/pusht_image_online.yaml) |
| Online-search policy + workspace | [`policy/online_search_policy.py`](diffusion_policy/policy/online_search_policy.py), [`workspace/train_online_search_workspace.py`](diffusion_policy/workspace/train_online_search_workspace.py), [`config/train_online_search.yaml`](diffusion_policy/config/train_online_search.yaml) |
| Best-of-N eval + video | [`eval_bon.py`](eval_bon.py), [`bon_video.py`](bon_video.py) |

---

## 1. Environment

Use the existing `robodiff` conda env (already has torch 2.4.0+cu121, diffusers,
hydra-core 1.2, zarr, pygame, pymunk, shapely, robomimic, wandb, dill — everything
these scripts import):

```bash
conda activate robodiff
```

If recreating from scratch: `mamba env create -f conda_environment.yaml`.

> **GPU required.** These are ~278M-param image UNets; a CPU run does not complete a
> single batch in practical time. Launch training/eval on a GPU node (`sbatch`/`srun`).

---

## 2. Dataset  ⚠️ needs a post-download patch

The Push-T dataset is `pusht_cchi_v7_replay.zarr` (206 episodes). It's the **same** file
whether you get it from Columbia or the "gym-pusht" folder the configs originally
referenced — there is no separate gym-pusht data release; that path was just where the
original author stored this file.

```bash
mkdir -p data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip   # ~31 MB
unzip pusht.zip
mv pusht/pusht_cchi_v7_replay.zarr .        # configs expect it at data/<zarr>, not data/pusht/<zarr>
rm -rf pusht pusht.zip && cd ..
```

**The raw zarr ships a single 5-dim `data/state` array, but this branch's dataset and
env-runner expect separate `data/agent_pos` (2d) and `data/block_pos` (3d) arrays**
([`pusht_image_dataset.py:88`](diffusion_policy/dataset/pusht_image_dataset.py#L88),
[`pusht_image_runner.py:54`](diffusion_policy/env_runner/pusht_image_runner.py#L54)).
Loading the unpatched file fails with `KeyError: 'agent_pos'`. Derive them once:

```bash
python - <<'PY'
import zarr, numpy as np
g = zarr.open('data/pusht_cchi_v7_replay.zarr', 'a')
d = g['data']; s = np.asarray(d['state'])   # [agent_x, agent_y, block_x, block_y, theta]
for name, arr in [('agent_pos', s[:, :2]), ('block_pos', s[:, 2:5])]:
    d.array(name, arr.astype(np.float32), chunks=(161, arr.shape[1]),
            compressor=d['state'].compressor, overwrite=True)
print('data keys:', list(d.keys()))
PY
```

This split is exact: resetting `PushTImageEnv(legacy=False)` to a recorded state
round-trips with 0.0 error, and `compute_feedback_from_pose(block_pos)` equals the live
`PushTFeedbackWrapper` feedback with 0.0 error. Re-downloading the dataset means redoing
this step.

All Push-T configs on this branch already point at `data/pusht_cchi_v7_replay.zarr`
(relative to repo root — hydra does not chdir here, so run commands from the repo root).

---

## 3. Behavior-cloning baseline (feedback-conditioned)

Standard diffusion-UNet BC on the modified Push-T task. The `task=pusht_image` override
is required (the workspace defaults to `lift_image_abs`).

```bash
python train.py \
  --config-name=train_diffusion_unet_image_workspace \
  task=pusht_image \
  training.device=cuda:0 training.seed=42
```

- Task adds a 16-dim `feedback` obs; splits into 50 held-out **test** episodes and 156
  train, of which `task.dataset.train_ratio=0.2` → 31 episodes are actually trained on
  (test is never subsampled). Override `train_ratio` to change the data budget.
- Rollouts run every `training.rollout_every_steps=5000` steps; success logged as
  `test/mean_score`. Checkpoints keep top-5 by `test_mean_score` plus `latest.ckpt`.
- Output: `data/outputs/<date>/<time>_train_diffusion_unet_image_pusht_image/`.
- W&B project defaults to `diffusion_policy_debug` (override `logging.project=...`).

---

## 4. Online-search policy (in-context)

Single-step policy that conditions on on-policy rollouts. Task is pinned to
`pusht_image_online` in the config's defaults — no `task=` override needed.

```bash
python train.py --config-name=train_online_search training.device=cuda:0
```

- `policy.n_trajs=8` context rollouts (built with `policy.n_envs=32` parallel Push-T
  envs, up to `policy.max_rollout_steps=300` each). Set **`policy.n_trajs=0`** for the
  pure-BC ablation (no env in the loop).
- Checkpoints keep top-5 by `train_action_mse_error`; W&B project `pusht_online_search`.

---

## 5. Evaluation

**Single rollout eval** (upstream script) — one rollout per held-out test reset:

```bash
python eval.py -c data/outputs/<run>/checkpoints/latest.ckpt \
  -o data/pusht_eval_output -d cuda:0
# -> eval_log.json (test/mean_score + rollout videos)
```

**Best-of-N** — roll each test reset out N times, report best-of-n and regret curves
([`eval_bon.py`](eval_bon.py)):

```bash
python eval_bon.py -c data/outputs/<run>/checkpoints/latest.ckpt \
  -o data/outputs/<run>/bon --n-samples 64 --n-envs 50
# -> bon_summary.json, bon_rewards.npz, bon_curves.png
```

Success = `max_reward >= 1.0` (coverage ≥ 95%). Options: `--n-resets` (default all
test resets), `--max-steps 300`, `--seed`.

**Best-of-N contact sheet** — one mp4, resets as rows × samples as columns, each cell
green-bordered on success ([`bon_video.py`](bon_video.py)):

```bash
python bon_video.py -c data/outputs/<run>/checkpoints/latest.ckpt \
  -o data/outputs/<run>/bon --n-resets 5 --n-samples 8
# -> bon_grid_5x8.mp4  (--reset-idxs 47,12,2,22,6 to pick specific resets)
```

---

## 6. SLURM

[`job.sh`](job.sh) has a usable preamble (`module load conda; conda activate robodiff2`)
but its body targets an unrelated `peg_insertion` experiment on another account/path —
swap in one of the `python train.py ...` commands above and set your own
`--account`/paths before submitting.
```
