# scripts/simpler — L2S data + eval pipeline

Scripts that move data from raw sources to trained diffusion-policy
checkpoints, run SimplerEnv evaluations on those checkpoints, and analyze
their action-variance / verifier-ranked performance.

Organized into sub-pipelines. Each directory is self-contained: the
shell wrapper inside it knows how to find its sibling Python file.

```
scripts/simpler/
├── plot_zarr/           # 3. plot a (multi-)zarr dataset
├── eval_diffusion/      # 4. evaluate a diffusion_policy ckpt in SimplerEnv
└── action_variance/     # 6. measure per-step action variance (motivation for BON)
```

The companion **[README_L2S.md](../../docs/simpler/README_L2S.md)** walks through these
sub-pipelines end-to-end with concrete examples.

---

## Data collection and filtering (not in this repo)

The collected datasets stay on gscratch; how they were collected (OpenVLA expert, seeds,
stored keys) and how the strict filters work is documented next to the data in
[`/gscratch/robotics/harine/data/README.md`](/gscratch/robotics/harine/data/README.md).
The collection / filter scripts (`data_collect/`, `bridge_to_zarr/`) and the LLaVA BoN
sweeps (`bon_eval/`) are at `github.com/harine/RoboMonkey` @ `2634271`, `scriptsv2/`.

## 3. `plot_zarr/` — Zarr dataset plots

| Script | What it does |
|---|---|
| `plot_zarr.py` | Multi-shard dashboard for any RoboMonkey/Bridge zarr (one or many `--shard` paths merged into a single figure). Supports `--align-to-arm-base` to overlay SIMPLER + Bridge shards in the same frame. |
| `plot_bridge_lerobot.py` | Pre-conversion preview: plot from raw LeRobot parquet + mp4 thumbnails before running `bridge_to_zarr.py`. |

```bash
# Single shard
python scripts/simpler/plot_zarr/plot_zarr.py --shard data/state0.zarr

# Multiple shards, merged into one plot
python scripts/simpler/plot_zarr/plot_zarr.py \
    --shard data/state0.zarr data/state1.zarr

# Overlay SIMPLER + Bridge V2 (auto-align frames)
python scripts/simpler/plot_zarr/plot_zarr.py \
    --shard data/state0.zarr data/bridge_v2_carrot.zarr \
    --align-to-arm-base
```

## 4. `eval_diffusion/` — SimplerEnv evaluation

| Script | What it does |
|---|---|
| `eval_diffusion.py` | Roll out a single diffusion_policy checkpoint (U-Net or MLP) in SimplerEnv for N episodes; logs per-episode JSONL + a summary `eval_log.json`. Supports BON via `--bon-k`. With `--seeds 17,50,3,9` runs an explicit seed list (one episode per seed, overriding `--num-episodes`). With `--viz-q` *and* BoN enabled, also dumps per-replan candidate actions + verifier rewards (Q-values) + scored frame to `<output_dir>/bon_q/ep<idx>_seed<seed>.npz`. |
| `eval_diffusion.sh` | Conda-env activation + headless `xvfb-run` wrapper around the Python eval. Forwards `SEEDS` and `VIZ_Q` env vars to the Python flags above. |
| `eval_diffusion_both.sh` | Convenience: evaluates BOTH U-Net and MLP checkpoints back-to-back for the configured task (`TASK=carrot` or `eggplant`). |
| `eval_summary.py` | Pretty-print one or more `eval_log.json` files as a comparison table. |

```bash
# Single checkpoint (eggplant)
bash scripts/simpler/eval_diffusion/eval_diffusion.sh \
    ~/diffusion_policy/.../checkpoints/latest.ckpt 100

# Both architectures, eggplant
bash scripts/simpler/eval_diffusion/eval_diffusion_both.sh

# Both architectures, carrot
TASK=carrot bash scripts/simpler/eval_diffusion/eval_diffusion_both.sh

# Compare existing logs
python scripts/simpler/eval_diffusion/eval_summary.py \
    data/eval/<run_a>/eval_log.json \
    data/eval/<run_b>/eval_log.json
```

## 6. `action_variance/` — Per-step action variance

| Script | What it does |
|---|---|
| `analyze_variance.py` | Sample N actions at each rollout step (or read MLP `log_std` directly), report empirical std per action dim, render an annotated video where each frame shows the per-dim std bars. |
| `analyze_variance.sh` | Runs the above for BOTH eggplant checkpoints back-to-back and prints a side-by-side mean-std comparison. |

```bash
bash scripts/simpler/action_variance/analyze_variance.sh
N_SAMPLES=64 bash scripts/simpler/action_variance/analyze_variance.sh
```

---

## Notes

- All `bash *.sh` scripts here assume the `simpler_env` conda env exists and
  the `diffusion_policy` repo is cloned at `$DIFFUSION_POLICY_ROOT`
  (default `~/diffusion_policy`).
