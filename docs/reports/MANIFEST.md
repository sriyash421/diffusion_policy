# What is NOT in this repo, and where it is

Generated 2026-09-16.

`docs/reports/summaries/` holds the small result files -- run provenance (`run.json`,
`splits.json`), selection curves, mode and attention summaries: 520 files, about 2 MB.

Its subdirectories are named `from-<source>/` rather than `<source>/` on purpose. `.gitignore`
matches `analysis`, `logs`, `ckpts`, `attn` and `modes24` **unanchored**, so a directory with
any of those names is excluded at ANY depth -- `docs/reports/summaries/analysis/` included.
Copying the files in under their original names left all 520 silently untracked, which is the
one outcome this folder exists to prevent. The `from-` prefix is what keeps them in the repo.
Everything below stayed out, and this page exists so that a number in `RESULTS.md` can be
traced to the artifact it came from even though the artifact is not here.

Two kinds of thing are excluded, for two different reasons:

* **Regenerable per-checkpoint artifacts** -- ~100 MB of `ep<NN>_idx<NNN>.json` candidate
  traces and 19 MB of `env_*.monitor.csv`. `.gitignore` already calls these out as
  "generated eval artifacts (JSON/PNG/NPZ per checkpoint) -- regenerable, 300M+". Tracking
  them would contradict that and bloat the history permanently.
* **Checkpoints, videos and replay buffers** -- too large, and none of them is a result.

## Locations

| path in repo | resolves to | size |
|---|---|---|
| `analysis/` | `/gscratch/robotics/harine/repo_offload/analysis_moved` | 314M |
| `attn/` | `/gscratch/robotics/harine/attention_analysis` | 63M |
| `ckpts/` | `/gscratch/robotics/harine/diffusion_policy_outputs` | 841G |
| `modes/` | `/gscratch/robotics/harine/mode_analysis` | 68M |
| `videos/` | `/gscratch/robotics/harine/repo_offload/videos_moved` | 365M |
| `logs/` | `(local directory)` | 22M |
| `modes24/` | `(local directory)` | 82M |

## The PPO grid is gone

`logs/grid/` held an 18-run recurrent-PPO / PPO grid, launched by hand -- no script in this
repo produced it, so it could be read but never re-launched as configured. Every one of those
runs recorded zero evaluation success, and on **2026-09-16** the tree and its write-ups were
deleted and the arms retrained. The record survives only in git history.

The lesson is kept: the replacement arms are launched by a script under `scripts/slurm/`, so
this generation is reproducible in a way the last one was not.

Note that `logs/` is on local disk rather than `/gscratch`, so it is not covered by whatever
backs the scratch tree up.
