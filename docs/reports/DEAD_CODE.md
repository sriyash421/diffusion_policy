# Dead code, recorded rather than removed

Checked 2026-09-16. Nothing here is deleted and **no upstream file is edited** — this page
exists so that "is this reachable?" is answerable by grep instead of by tracing imports.

Vendored upstream code unrelated to PushT (`tests/`, `diffusion_policy/real_world/`, ray
launchers, the robomimic / kitchen / blockpush / maze / maniskill task families, the root
real-robot scripts) is out of scope and not inventoried here.

## Hydra targets that cannot be instantiated

| `_target_` | named by | why it fails |
|---|---|---|
| `diffusion_policy.model.ibc.global_avgpool.GlobalAvgpool` | `train_ibc_dfo_*_workspace.yaml` | `diffusion_policy/model/ibc/` does not exist |
| `diffusion_policy.model.obs_encoder.video_core.{VideoCore,VideoResNet}` | `train_diffusion_unet_video_workspace.yaml` | `diffusion_policy/model/obs_encoder/` does not exist |
| `train_search.TrainSearchWorkspace` | `config/archive_config/train_search.yaml` | not a package path; no top-level `train_search.py` |

## Modules with no live caller

Cross-referenced against every `.py`, `.yaml`, `.md`, `.sh` and `.sbatch` in the tree, matching
both the dotted module path and the file path — so hydra's string `_target_` references are
included.

| module | note |
|---|---|
| `diffusion_policy/policy/diffusion_unet_video_policy.py` | **unloadable**: module-level import of the missing `model.obs_encoder.temporal_aggregator` |
| `diffusion_policy/common/pymunk_override.py` | duplicate of `diffusion_policy/env/pusht/pymunk_override.py`, which IS live |
| `diffusion_policy/common/env_util.py` | |
| `diffusion_policy/common/pymunk_util.py` | |
| `diffusion_policy/model/diffusion/conditional_flow_matcher.py` | |

## Configs that raise on instantiation

`task/pusht_image.yaml` and `task/pusht_image_online.yaml` pass `train_ratio` and
`max_train_episodes` to `PushTImageDataset`, which accepts neither, and set no `split_file`,
which it requires. Both were left on the old API when the split manifest became the single
source of truth.

`task/pusht_image.yaml` is **upstream** (added 2023-03-07) and is left exactly as it is. It is
named only by `train_ibc_dfo_hybrid_workspace.yaml`, which is dead for a separate reason above,
and by one comment in `scripts/run_nopos_30demo.sh`. **No arm this repo supports goes near it**
— ST and BC-UNet both inherit `task: pusht_image_search_imgonly` from `pusht_base.yaml:16`.

`task/pusht_image_online.yaml` and `train_online_search.yaml` are this project's (2026-07-15)
and drive the online-search arm, which is outside the seven supported arms. That arm also
carries a private encoder block with `imagenet_norm: True` on a dataset already mapped to
[-1, 1] — a double normalisation diagnosed at `pusht_base.yaml:96-97` and left in place. Repair
or retire is an open decision.

## Two test directories

`tests/` (13 files, all 2023-07-23, upstream: realsense, robomimic, mello, block_pushing) is
dead — every file fails on an import or on absent hardware. It is left on disk; `pytest.ini`
now sets `testpaths = unit_tests` so a bare `pytest` no longer collects it.

`unit_tests/` is this project's and is the suite that runs.
