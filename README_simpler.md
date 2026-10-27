# SimplerEnv experiments (branch `harine/simpler`)

SimplerEnv (WidowX eggplant / stack / spoon / carrot) + diffusion policy, split out of
`RoboMonkey` the same way the maze (`harine/maze`) and Push-T (`harine/pushT`) work lives
in their own checkouts. This branch starts from `harine/improved_verifier`, so it has the
shared search transformer, the Veritas verifiers and the PushT/maze code as well.

Checked out at `/mmfs1/home/harine/diffusion_policy_standalone_simpler`, as a worktree of
the same `diffusion_policy` repo as `~/diffusion_policy_standalone_{maze,pushT}`. That repo's
git data is the bare repo `/mmfs1/home/harine/diffusion_policy.git` (moved out of
`RoboMonkey/.git/modules/diffusion_policy` on 2026-10-27; RoboMonkey itself was then deleted).

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
| `simpler_env` | sim evals, Veritas prep |
| `monkey-verifier` | default `CONDA_ENV` for BC / search training and evals (just the env name; the LLaVA verifier itself is not used here) |
| `dqc_jax`, `dqc_repo` | DQC verifier training / BoN |

`simpler_env`, `monkey-verifier` and `dqc_jax` load `simpler_env` / `mani_skill2_real2sim`
editable from **this** checkout's `SimplerEnv/`. Their old `openvla` / `llava` editable
installs pointed into the deleted RoboMonkey checkout and no longer import; nothing on the
train / eval path uses them.
| `gsam2` | Grounded-SAM-2 tracking server for Veritas (`/gscratch/robotics/harine/Grounded-SAM-2`) |

To point another env at this SimplerEnv (in a Slurm job, not on a login node):

```bash
pip install --no-deps --no-build-isolation -e SimplerEnv/ManiSkill2_real2sim -e SimplerEnv
```

## Data

Nothing was moved. All datasets and runs stay on gscratch. How the data was collected
(OpenVLA expert, seeds, stored keys) and how the strict filters work:
[`/gscratch/robotics/harine/data/README.md`](/gscratch/robotics/harine/data/README.md).

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

**Not carried over.** The local RoboMonkey checkout was deleted on 2026-10-27; its history is
on GitHub (`github.com/harine/RoboMonkey`, branch `harine/improved_verifier` @ `2634271`).

- `monkey-verifier/` (LLaVA reward model), `sglang-vla/`, `openvla-mini/`. Dropped scripts:
  `scriptsv2/{bon,bon_eval,robomonkey_repro,mse_q,data_collect,bridge_to_zarr}/`,
  `q_fn_eval/eval_robomonkey.py`, `eval_search/sweep_bc_llava.sbatch`,
  `train_search/{train_diffusion_llava_na1,smoke_kvprefix}.sbatch`, `analysis/make_verifier_videos.sbatch`.
  The optional LLaVA / OpenVLA verifier paths in `eval_search.py`, `eval_diffusion.py` and
  `policy/verifiers.py` still name `/mmfs1/home/harine/RoboMonkey/...` and fail if selected.
- How the existing data was collected and filtered is written up next to it in
  `/gscratch/robotics/harine/data/README.md`; the OpenVLA collector and sglang server source
  (never pushed) are snapshotted in `/gscratch/robotics/harine/data/_collection_code/`.
