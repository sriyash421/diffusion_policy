# SimplerEnv experiments (branch `harine/simpler`)

SimplerEnv (WidowX eggplant / stack / spoon / carrot) + diffusion policy, split out of
`RoboMonkey` the same way the maze (`harine/maze`) and Push-T (`harine/pushT`) work lives
in their own checkouts. This branch starts from `harine/improved_verifier`, so it has the
shared search transformer, the Veritas verifiers and the PushT/maze code as well.

Checked out at `/mmfs1/home/harine/diffusion_policy_standalone_simpler`, as a worktree of
the same `diffusion_policy` repo as `~/diffusion_policy_standalone_{maze,pushT}`.

## Layout

| Path | What |
| --- | --- |
| `diffusion_policy/` | the package: policies, workspaces, configs (`config/robomonkey_*`, `config/splits/simpler`, `config/veritas`) |
| `diffusion_policy/env/simpler/` | Simpler Veritas (camera, GSAM tracker, plans) and `wide_eggplant.py` |
| `SimplerEnv/` | submodule, **upstream** `simpler-env/SimplerEnv` @ `4ab7178` (ManiSkill2_real2sim @ `ef7a4d4`) |
| `dqc/` | submodule, `sriyash421/dqc` branch `harine-simpler` (DQC chunk-critic verifier) |
| `scripts/simpler/` | the old `RoboMonkey/scriptsv2` pipeline, same subdir names ([README](scripts/simpler/README.md), [Veritas](scripts/simpler/veritas/README.md)) |
| `tests/simpler/`, `unit_tests/test_veritas_*` | Simpler / Veritas tests |
| `docs/simpler/` | reports, `README_L2S.md`, `results_notes.md`, run/config lists from RoboMonkey |
| `data/`, `wandb`, `simpler_veritas_videos` | symlinks into gscratch (below); not tracked |

Fresh checkout: `git submodule update --init --recursive`.

## SimplerEnv: upstream submodule + one registered scene

Set up like `github.com/harine/veritas`: SimplerEnv is a plain submodule installed editable.
Our only change to it, the wide eggplant reset region used for the
`eggplant_reset_core_*` collection, lives here instead of in a fork:
[`diffusion_policy/env/simpler/wide_eggplant.py`](diffusion_policy/env/simpler/wide_eggplant.py)
registers `PutEggplantInBasketWideScene-v0` and the `widowx_put_eggplant_in_basket_wide`
task on import. Code that makes that task must import it first:

```python
import diffusion_policy.env.simpler.wide_eggplant  # noqa: F401
env = simpler_env.make("widowx_put_eggplant_in_basket_wide")
```

The standard tasks (eggplant, stack, spoon, carrot) are untouched upstream code. Reset parity against
the old RoboMonkey SimplerEnv was checked on both eggplant tasks (exact start poses).

## Environments

All conda envs live in `/gscratch/robotics/harine/miniconda3/envs`.

| Env | Used for |
| --- | --- |
| `simpler_env` | sim evals, Veritas prep, collection. Its `simpler_env` / `mani_skill2_real2sim` editable installs point at **this** checkout's `SimplerEnv/` |
| `monkey-verifier` | default `CONDA_ENV` for BC / search training and evals (just the env name; the LLaVA verifier itself is not used here). Its SimplerEnv editable still points at `RoboMonkey/SimplerEnv` (same code plus the wide scene) |
| `dqc_jax`, `dqc_repo` | DQC verifier training / BoN |
| `gsam2` | Grounded-SAM-2 tracking server for Veritas (`/gscratch/robotics/harine/Grounded-SAM-2`) |

To point another env at this SimplerEnv (in a Slurm job, not on a login node):

```bash
pip install --no-deps --no-build-isolation -e SimplerEnv/ManiSkill2_real2sim -e SimplerEnv
```

## Data

Nothing was moved. All datasets and runs stay on gscratch:

| Where | What |
| --- | --- |
| `/gscratch/robotics/harine/data/<task>/` | collected Simpler datasets (`strict_distance_success_split`, `stack_cube_strict_split`, `spoon_on_towel_strict`, `eggplant_reset_core_*`, `_archive/`, ...). See [dataset_registry.md](scripts/simpler/analysis/dataset_registry.md) |
| `/gscratch/robotics/harine/data/veritas_plans/` | Gemini Veritas plans |
| `data/{train,eval,viz,viz_q,veritas,models,plots}` | symlinks to `/gscratch/robotics/harine/RoboMonkey_data/home_data/*` (what `RoboMonkey/data` was) |
| `data/outputs` | symlink to `/gscratch/robotics/harine/robomonkey_outputs` (hydra outputs) |
| `/gscratch/robotics/harine/RoboMonkey_data/{train,experiments}` | absolute-path training runs and campaigns, referenced as-is |

## Coming from RoboMonkey

Paths in `scripts/simpler` were rewritten mechanically:

| RoboMonkey | here |
| --- | --- |
| `RoboMonkey/` (cwd / `SLURM_SUBMIT_DIR`) | repo root |
| `RoboMonkey/diffusion_policy/` | repo root (`DIFFUSION_POLICY_ROOT` defaults to it) |
| `RoboMonkey/scriptsv2/` | `scripts/simpler/` |
| `RoboMonkey/data/`, `RoboMonkey/diffusion_policy/data/` | `data/` |
| `RoboMonkey/{SimplerEnv,dqc}` | the submodules here |

Submit sbatch jobs from the repo root, e.g.
`sbatch --export=ALL,... scripts/simpler/train_search/train_search_veritas.sbatch`.

**Left behind in RoboMonkey** (still at `/mmfs1/home/harine/RoboMonkey`, branch
`harine/improved_verifier`, not deleted, because the maze/pushT/simpler worktrees keep their git metadata
in `RoboMonkey/.git/modules/diffusion_policy`):

- `monkey-verifier/` (LLaVA reward model) and `sglang-vla/`. Dropped scripts:
  `scriptsv2/{bon,bon_eval,robomonkey_repro,mse_q}/`, `q_fn_eval/eval_robomonkey.py`,
  `eval_search/sweep_bc_llava.sbatch`, `train_search/{train_diffusion_llava_na1,smoke_kvprefix}.sbatch`,
  `analysis/make_verifier_videos.sbatch`. The optional LLaVA verifier paths in
  `eval_search.py` / `eval_diffusion.py` still resolve to `RoboMonkey/monkey-verifier`.
- `openvla-mini/` (the OpenVLA expert). Collecting **new** data
  (`scripts/simpler/data_collect/collect_simpler.{sh,sbatch}`) still runs
  `RoboMonkey/openvla-mini/experiments/robot/simpler/collect_trajectories.py` against the
  sglang OpenVLA server. `collect_simpler.sh` registers the wide eggplant task before
  running it. The filter / subsample scripts in `data_collect/` don't need OpenVLA.
