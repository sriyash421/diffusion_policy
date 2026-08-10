# Archive

Superseded working documents and launchers. Nothing here is loaded by anything; it is kept
because these files are **untracked by git**, so deleting them would be unrecoverable.

## Round 7 (archived 2026-08-10)

`ROUND7_LAUNCH.md`, `launch_round7.sh`, `launch_round7_watchers.sh`.

Round 7 was the 8-run data-budget generation: BC@25, BC@100 and the six 100-demo argmax
search arms. All eight finished; their results are section 1 of `SUCCESS_RATES.md` and
their inventory rows are in `aug9_analysis.md`.

Why they are no longer usable as written:

- `ROUND7_LAUNCH.md` is a **"READY TO LAUNCH (paused)"** plan for runs that completed weeks
  ago, and its run-directory column uses the pre-rename `ctx-*` names (renamed 2026-08-05,
  `AUDIT.md` 9.9).
- `launch_round7.sh` names the five per-arm configs that moved to
  `diffusion_policy/config/archive_config/` on 2026-08-09, so it would fail at
  `--config-name`. The equivalent today is one base config plus three CLI overrides — see
  that directory's README.

What was **not** lost:

- The BC parity policy — what must stay shared with the search arms, why
  `num_inference_steps` stays at 8 (measured: n=1 success identical at 8/16/32/100 on two
  checkpoints, `scripts/diag_bc_n1.sbatch`), and why BC alone trains to 300k — lives in the
  header of `diffusion_policy/config/train_pusht_bc.yaml`, next to the config it governs.
- The SLURM account-vs-partition trap is documented in `scripts/launch_round8_29demo.sh`
  and `scripts/submit_selection_sweep.sh`.
- The open design gap (no search arm at 25 demos) was lifted into `aug9_analysis.md` 1a.

**`scripts/round7_status.sh` was NOT archived.** Despite the name it was never
Round-7-specific: it globs the run directory and reports whatever is on disk, including the
r8 runs and the `subgoal-only_k16_cd0.9` arm that no launcher knew about. It is now
`scripts/run_status.sh`.
