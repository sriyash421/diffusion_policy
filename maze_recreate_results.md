# L2S maze recreate: results

2026-09-28

## Results

(to fill)

## Rerun plan vs L2S + diffusion branch

The planned rerun departs from L2S (`sriyash/maze` @ `26c095f`) plus `diffusion_policy` (`sriyash/l2s-verifier-readme` @ `cbdd68a`) in the rows below. Everything else follows L2S as-is.

| | L2S + diffusion branch | Rerun plan |
| --- | --- | --- |
| `diffusion_policy` version | L2S pins `ac22548` | `cbdd68a`: adds the diffusion ST policy and configs, plus a vendored verifier identical to `l2s/verifier.py`. BC and the Gaussian ST are otherwise unchanged. |
| Mazes | 1000 procgen layouts: data from 80 train layouts, eval on 20 eval layouts | One layout (L2S procgen maze #0) for both train and eval |
| Start | Fixed per layout (first free cell from the bottom-left) | Uniform over free cells |
| Goal | Uniform over the free cells other than the start | Uniform over the split's goal pool |
| Train/test separation | Different mazes | Same maze; test (start, goal) pairs never appear in training; goal regions held out for quadrant and inner_outer |
| Datasets | One | Three: as_is, quadrant, inner_outer |
| `--maze-fraction` | 0.1 | Not used (one layout) |
| Eval `train` split | The 80 training mazes | Train goal pool on the same maze |
| Models | BoN (BC UNet); Gaussian ST at train noise 0.0 and 0.2 | BC UNet; Gaussian ST; diffusion ST; diffusion ST with `corrupt_obs=True` |
| Verifier noise | Train 0.0 / 0.2 (ST); eval 0.0 / 0.2 | 0 everywhere |
| Selection | `max` only; `last` raises an error for BoN | `max` and `last` for every model; for BoN, `last` is the N-th i.i.d. draw |
| Plots | `aggregate_procgen_pipeline.py` | Not used: its layout-overlap check rejects a single maze, so the `metrics.csv` files are combined instead |
| Conda envs | `hitl` for layouts; `robodiff2` in the examples | `procgen_maze` for layouts; `vae_pushT_l2s` for everything else |

Unchanged from L2S:
- the layout generator (1000 layouts, 0.8 split)
- the BFS expert: 800 episodes × 100 steps, zero-padded
- the image format, (x, y) positions, whole-cell motion and the verifier
- the training configs and train command
- seeds 0/1/2, with env seed = train seed
- eval: 100 episodes, 200 rollout steps, success radius 1.0, N = 1…256, `latest.ckpt`
