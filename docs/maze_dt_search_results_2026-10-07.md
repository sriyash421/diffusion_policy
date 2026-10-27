# Maze round 3: decision-search transformer — results (2026-10-07)

Fork `harine/maze` @ 3b04596; L2S root `data/experiments/openloop11dt_pipeline_quadrant`; plan `docs/plans/maze_dt_search_2026-10-06.md`.
11x11 layout 0, quadrant goal split, seed 0, 100 episodes per test set, N up to 64 sequential attempts, one shot per episode.
perf@k = any of the first k attempts reached the goal (strategy `any`); argmax@k (strategy `max`) was identical in every cell,
because a successful attempt always holds the maximal verifier value (-1). `uniq@8` = distinct realized trajectories among the first 8;
`first_k` = mean index of the first successful attempt among episodes solved by k=64. Measurements only; no checkpoint is nominated.

Arms (all: causal transformer over one token stream, Gaussian head + cardinal snap, model-predicted stop, no horizon):
- `dt_bon`: max_actions 1, i.i.d. attempts.
- `dt_st_traj`: context = earlier attempts' actions + one -BFS(final) per attempt.
- `dt_st_step`: as traj, plus -BFS(current cell) on every token (verifier distance visible within an attempt).

## Held-out quadrant goals (`eval`)

| arm | ckpt | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | uniq@8 | first_k | len@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dt_bon | 1001 | 0.14 | 0.16 | 0.19 | 0.24 | 0.32 | 0.38 | 0.43 | 6.8 | 12.0 | 14.2 |
| dt_bon | 2001 | 0.20 | 0.27 | 0.34 | 0.40 | 0.46 | 0.52 | 0.58 | 5.3 | 9.8 | 19.8 |
| dt_bon | 3001 | 0.22 | 0.29 | 0.34 | 0.40 | 0.46 | 0.57 | 0.66 | 4.5 | 12.8 | 18.4 |
| dt_bon | 4001 | 0.16 | 0.22 | 0.25 | 0.28 | 0.35 | 0.37 | 0.43 | 3.6 | 11.8 | 17.3 |
| dt_bon | 5001 | 0.24 | 0.29 | 0.34 | 0.37 | 0.46 | 0.50 | 0.55 | 3.3 | 10.0 | 16.3 |
| dt_bon | 6001 | 0.19 | 0.22 | 0.24 | 0.26 | 0.29 | 0.35 | 0.42 | 2.7 | 13.5 | 14.8 |
| dt_bon | 7001 | 0.19 | 0.20 | 0.22 | 0.24 | 0.28 | 0.31 | 0.35 | 2.3 | 10.0 | 14.2 |
| dt_bon | 8001 | 0.21 | 0.22 | 0.24 | 0.26 | 0.26 | 0.28 | 0.35 | 2.4 | 11.4 | 15.1 |
| dt_bon | 9001 | 0.20 | 0.20 | 0.22 | 0.24 | 0.27 | 0.31 | 0.37 | 2.3 | 13.0 | 13.9 |
| dt_bon | latest | 0.20 | 0.20 | 0.22 | 0.24 | 0.27 | 0.31 | 0.37 | 2.3 | 13.0 | 13.9 |
| dt_st_traj | 1001 | 0.18 | 0.21 | 0.23 | 0.32 | 0.44 | 0.50 | 0.55 | 6.9 | 11.0 | 16.8 |
| dt_st_traj | 2001 | 0.14 | 0.14 | 0.16 | 0.21 | 0.31 | 0.37 | 0.40 | 4.5 | 10.9 | 10.0 |
| dt_st_traj | 3001 | 0.18 | 0.26 | 0.35 | 0.41 | 0.46 | 0.50 | 0.55 | 4.1 | 8.3 | 14.3 |
| dt_st_traj | 4001 | 0.15 | 0.15 | 0.18 | 0.20 | 0.25 | 0.28 | 0.31 | 3.2 | 9.5 | 9.8 |
| dt_st_traj | 5001 | 0.22 | 0.30 | 0.35 | 0.42 | 0.42 | 0.47 | 0.47 | 3.3 | 4.7 | 14.9 |
| dt_st_traj | 6002 | 0.16 | 0.21 | 0.25 | 0.32 | 0.33 | 0.35 | 0.36 | 2.6 | 5.2 | 11.8 |
| dt_st_traj | 7002 | 0.15 | 0.21 | 0.23 | 0.27 | 0.31 | 0.36 | 0.39 | 2.8 | 9.3 | 14.2 |
| dt_st_traj | 8002 | 0.13 | 0.16 | 0.18 | 0.22 | 0.26 | 0.28 | 0.31 | 2.5 | 9.0 | 12.7 |
| dt_st_traj | 9002 | 0.14 | 0.19 | 0.23 | 0.29 | 0.34 | 0.37 | 0.39 | 2.6 | 7.7 | 14.7 |
| dt_st_traj | latest | 0.14 | 0.19 | 0.23 | 0.29 | 0.34 | 0.37 | 0.39 | 2.6 | 7.7 | 14.7 |
| dt_st_step | 1001 | 0.27 | 0.41 | 0.60 | 0.71 | 0.81 | 0.88 | 0.95 | 6.9 | 8.6 | 30.3 |
| dt_st_step | 2001 | 0.38 | 0.42 | 0.54 | 0.61 | 0.70 | 0.76 | 0.81 | 5.7 | 7.5 | 22.7 |
| dt_st_step | 3001 | 0.47 | 0.52 | 0.57 | 0.60 | 0.67 | 0.77 | 0.87 | 4.9 | 10.0 | 24.7 |
| dt_st_step | 4001 | 0.45 | 0.48 | 0.54 | 0.55 | 0.63 | 0.68 | 0.73 | 4.0 | 7.0 | 22.3 |
| dt_st_step | 5001 | 0.53 | 0.57 | 0.62 | 0.65 | 0.69 | 0.75 | 0.81 | 3.6 | 6.5 | 20.4 |
| dt_st_step | 6001 | 0.44 | 0.48 | 0.51 | 0.53 | 0.58 | 0.63 | 0.67 | 3.2 | 6.8 | 18.8 |
| dt_st_step | 7001 | 0.43 | 0.48 | 0.52 | 0.54 | 0.59 | 0.67 | 0.71 | 3.0 | 6.9 | 20.4 |
| dt_st_step | 8001 | 0.44 | 0.48 | 0.52 | 0.54 | 0.58 | 0.65 | 0.66 | 3.0 | 5.3 | 19.5 |
| dt_st_step | 9002 | 0.47 | 0.54 | 0.58 | 0.60 | 0.66 | 0.71 | 0.73 | 3.1 | 5.2 | 20.9 |
| dt_st_step | latest | 0.47 | 0.54 | 0.58 | 0.60 | 0.66 | 0.71 | 0.73 | 3.1 | 5.2 | 20.9 |

## Unseen pairs from the train pool (`train_heldout`)

| arm | ckpt | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | uniq@8 | first_k | len@1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dt_bon | 1001 | 0.58 | 0.78 | 0.84 | 0.91 | 0.98 | 0.99 | 0.99 | 6.3 | 2.7 | 18.2 |
| dt_bon | 2001 | 0.91 | 0.95 | 0.97 | 0.98 | 0.99 | 0.99 | 1.00 | 5.1 | 1.9 | 13.5 |
| dt_bon | 3001 | 0.94 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 3.9 | 1.1 | 13.9 |
| dt_bon | 4001 | 0.94 | 0.97 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 3.1 | 1.2 | 12.1 |
| dt_bon | 5001 | 0.96 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 2.6 | 1.2 | 11.8 |
| dt_bon | 6001 | 0.95 | 0.96 | 0.97 | 0.98 | 0.99 | 0.99 | 1.00 | 2.3 | 1.7 | 11.8 |
| dt_bon | 7001 | 0.96 | 0.96 | 0.96 | 0.97 | 0.97 | 0.99 | 0.99 | 2.1 | 1.6 | 11.7 |
| dt_bon | 8001 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 2.0 | 1.0 | 11.6 |
| dt_bon | 9001 | 0.96 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 2.1 | 1.1 | 12.0 |
| dt_bon | latest | 0.96 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 2.1 | 1.1 | 12.0 |
| dt_st_traj | 1001 | 0.52 | 0.72 | 0.85 | 0.93 | 0.97 | 1.00 | 1.00 | 6.1 | 3.0 | 14.5 |
| dt_st_traj | 2001 | 0.76 | 0.90 | 0.95 | 0.98 | 0.98 | 0.98 | 0.99 | 4.5 | 1.9 | 11.5 |
| dt_st_traj | 3001 | 0.90 | 0.95 | 0.97 | 0.97 | 0.98 | 0.99 | 0.99 | 3.7 | 1.4 | 13.2 |
| dt_st_traj | 4001 | 0.93 | 0.96 | 0.96 | 0.96 | 0.97 | 0.97 | 0.97 | 2.9 | 1.1 | 12.1 |
| dt_st_traj | 5001 | 0.94 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 | 2.4 | 1.4 | 12.0 |
| dt_st_traj | 6002 | 0.95 | 0.96 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 2.1 | 1.1 | 11.5 |
| dt_st_traj | 7002 | 0.95 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 1.9 | 1.1 | 12.1 |
| dt_st_traj | 8002 | 0.95 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 1.8 | 1.1 | 11.4 |
| dt_st_traj | 9002 | 0.95 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 1.8 | 1.1 | 11.7 |
| dt_st_traj | latest | 0.95 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 1.8 | 1.1 | 11.7 |
| dt_st_step | 1001 | 0.74 | 0.91 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 5.9 | 1.4 | 19.0 |
| dt_st_step | 2001 | 0.91 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 4.6 | 1.2 | 14.2 |
| dt_st_step | 3001 | 0.95 | 0.97 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 | 3.5 | 1.2 | 13.0 |
| dt_st_step | 4001 | 0.93 | 0.96 | 0.97 | 0.98 | 0.98 | 0.98 | 0.98 | 2.8 | 1.1 | 12.7 |
| dt_st_step | 5001 | 0.97 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 2.3 | 1.1 | 12.0 |
| dt_st_step | 6001 | 0.97 | 0.97 | 0.97 | 0.98 | 0.98 | 0.98 | 0.98 | 2.2 | 1.1 | 11.8 |
| dt_st_step | 7001 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | 0.98 | 1.8 | 1.5 | 11.4 |
| dt_st_step | 8001 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 1.9 | 1.1 | 11.4 |
| dt_st_step | 9002 | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 1.9 | 1.1 | 11.3 |
| dt_st_step | latest | 0.98 | 0.98 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 | 1.9 | 1.1 | 11.3 |

## Reference: sriyash/l2s_online (unchanged, PushT)

Worktree `/gscratch/robotics/harine/l2s_online_ref` @ 86d5ef9, default `train_online_search.yaml` (only overrides: zarr path, output dir,
`training.checkpoint_every=5`), 100 outer steps in 3.3 h; `eval_bon.py` (context-free samples, 50 test resets x 64):
single-sample success 0.00 / reward 0.23; success@n = 0.02 (2), 0.02 (4), 0.08 (8), 0.08 (16), 0.14 (32), 0.18 (64); best reward@64 0.51.
Its eval never feeds attempts back in context. One-line fix applied to its eval script (`cfg.training.get('use_ema', False)`).
W&B l2sml/pusht_online_search/runs/aav4qb3q; curve `docs/plans/l2s_online_ref_bon_curves.png`.

## Jobs

train: dt_bon 41493721_2, dt_st_traj 41493723_0, dt_st_step 41493725_1 (robotics/ckpt-all, requeue+resume). latest evals 41493722 / 41493724 / 41508218;
per-1k evals 41494435 / 41509508 / 41504531+41508219. l2s_online: train 41493210, eval 41497915.
