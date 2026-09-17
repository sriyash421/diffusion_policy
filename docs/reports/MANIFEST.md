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
| `logs/` | `(local directory)` | 1.4G |
| `modes24/` | `(local directory)` | 82M |

## The one irreplaceable thing

`logs/grid/` holds the 18-run recurrent-PPO / PPO grid, and **no script in this repo
produces `logs/grid/`** -- the launcher that made those runs is gone. The runs can be read
(`recurrent_ppo/scripts/runs_doc.py`) but not re-launched as configured. That is recorded
here rather than reconstructed, because a rebuilt launcher would not be the one that
produced the numbers.

It is also on local disk rather than `/gscratch`, so it is not covered by whatever backs
the scratch tree up.
