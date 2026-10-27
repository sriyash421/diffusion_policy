# L2S maze recreate on a 17x17 maze: results

2026-09-30

## Results

(to fill)

Combined metrics: `/gscratch/robotics/harine/L2S/data/experiments/maze17pad_splits_all_metrics.csv`. Regenerate it with:

```bash
python scripts/combine_maze_metrics.py data/experiments/maze17pad_splits_all_metrics.csv 'data/experiments/procgen17pad_pipeline_*'
```

## Rerun vs the 11x11 run

Everything follows the 11x11 run (`maze_recreate_results.md`) except the rows below.

| | 11x11 run | 17x17 run |
| --- | --- | --- |
| Procgen mode | `easy` (15x15 world); easy mode tops out at 15x15 mazes | `hard` (25x25 world) |
| Layouts | `--world-dim 10`, cropped to the 11x11 maze | `--world-dim 16`, stored uncropped as the 25x25 grid (`--keep-padding`) with the maze at rows/cols 4..20 |
| Layout file | `data/procgen_mazes/layouts.npz` | `data/procgen_mazes_17pad/layouts.npz` |
| Maze #0 | Level seed 1; 71 free cells | Level seed 4; 161 free cells; longest shortest path 58 steps |
| Observation image | `[3, 11, 11]` | `[3, 25, 25]`, set with Hydra overrides on `task.image_shape`, `task.shape_meta` and `task.dataset.shape_meta` |
| Positions | Maze coordinates | 25x25 grid coordinates (maze coordinate + 4) |
| Split centre | (5, 5) | (12, 12) in the grid = (8, 8) in maze coordinates; computed as the centre of the free cells' bounding box |
| Inner radius (inner_outer) | 2 | 4 |
| Goal pools, train/test | as_is 71/71, quadrant 49/17, inner_outer 59/12 | as_is 161/161, quadrant 101/35, inner_outer 116/45 |
| Data | `data/procgen_maze_expert_<split>` | `data/procgen17pad_maze_expert_<split>` |
| Experiment roots | `data/experiments/procgen_pipeline_<split>` | `data/experiments/procgen17pad_pipeline_<split>` |
| W&B project (training) | `maze_results` | `maze17pad_results` |

Unchanged:
- the expert: 800 episodes × 100 steps
- the models: BC UNet (BoN), Gaussian ST, diffusion ST, and diffusion ST with `corrupt_obs=True`
- seeds 0/1/2, with no verifier noise
- the training configs
- eval: `latest.ckpt`, N = 1…256, `eval` and `train` splits, `max` and `last`, 100 episodes, 200 steps, radius 1.0, env seed = train seed
