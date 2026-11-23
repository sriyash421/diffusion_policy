# Maze audits (code snapshot)

Code and small configs/results summaries copied from `/gscratch/robotics/harine/codex_audits/` (the working copies live there; scripts
use absolute paths to it). Runs, datasets, checkpoints, logs, figures and `*.before_*` backups are not included.

- `maze_quadrants_delta_20261019/`: maze settings (`datasets.json`, incl. `interp30` and the continuous `interp30c`), `setup_data.py`
  goal splits, `run_experiment.py` (BC UNet / DT training; `--horizon` for short-horizon closed-loop BC).
- `maze_rounds_20261103/`: the ST rounds policy and training (`rounds.py`, `asteroid.py`), best-of-n eval (`bon_eval.py`), plots and the
  rounds W&B report. `MAZE_SETTING` selects 11br (default), interp30 or interp30c.
- `maze_interp_20261105/`: interpolation split (30 random train goals): split setup, BC scan / test set, closed-loop BC (`bc_closed.py`),
  the continuous maze (`cmaze.py`, `make_cont_demos.py`), Slurm launchers, `plots.py` and `report/publish_report.py` (W&B report
  "Maze · interpolation"), and the JSON summaries behind the report figures.
- DAgger with a hidden goal (2026-11-22/23): continuous actions are now the default (`rounds.py` `MAZE_SETTING` = interp30c; every
  `datasets.json` entry labelled `actions: discrete | continuous`; `run_setting.py` takes the setting from the run). Observation modes
  in `maze_rounds_20261103/obs_modes.py` (`--obs window` = ASTEROID 7×7 view, `--obs ddpm --obs-t T` = DDPM-noised image); continuous
  expert labels in `rounds.py`; `asteroid.py --target relabel --no-context`, `evalctx`. Launchers `maze_interp_20261105/dag_run.sbatch`,
  `launch_dag.sh`, `launch_r3.sh` (3-round schedules); figures `plots_simple.py`, `plots_positions.py`; leakage `leak_audit.py`;
  W&B report `publish_dag_report.py`. New mazes: `maze_procgen_ast_20261121/pg_dagger.py`, `pg_run.sbatch`.
