# pi0 + Veritas in SimplerEnv (official code, our logging)

Reproduction of the Veritas paper (arXiv 2606.18247) with its released code,
`princeton-prism/veritas`, cloned **unmodified** to `/gscratch/robotics/harine/veritas`
(`$VERITAS_ROOT`). Base policy: open-pi-zero Bridge-uniform
(`models/bridge_uniform_step19296_2024-12-26_22-31_42.pt`, a symlink into
`/gscratch/robotics/harine/veritas_models/`). Everything we add lives here and subclasses the
official `src.agent.eval.EvalAgent`.

| file | what |
| --- | --- |
| `setup_env.sh` | builds the checkout's uv env (`$VERITAS_ROOT/.venv`) inside a GPU job; falls back from the cu130 torch index to cu128 if the driver is too old |
| `run_pi0_eval.py` | `LoggedEvalAgent(EvalAgent)`: official config + hydra overrides, plus `episodes.jsonl`, resume (rows present are skipped), wandb (`l2sml/simpler_veritas`), `summary.json`, optional `+eval_seeds=<json>` for reset-by-seed |
| `eval_pi0.sbatch` | one arm per job; env vars `OUT_DIR`, `VERIFIER`, `N`, `TASK`, `N_EPISODES`, `ARM`, `EVAL_SEEDS`, `GEMINI_MODEL`, `EXTRA` |
| `submit_eggplant.sh` | submits `raw_n1` and `veritas_n5` (paper protocol: 24 variations x 10 trials) to a free non-preemptible partition; `PARTITION=ckpt` for the preemptible queue |
| `summarize.py` | table of arms with Wilson 95% CIs and per-variation rates |

Weights the official code expects under `$VERITAS_ROOT`:

- `transformers_cache/paligemma-3b-pt-224/` (gated on HF: `hf auth login` once, then
  `HF_HUB_DISABLE_XET=1 hf download google/paligemma-3b-pt-224 --local-dir $VERITAS_ROOT/transformers_cache/paligemma-3b-pt-224 --max-workers 2`
  on a login node; the xet downloader gets OOM-killed by the login cgroup),
- `transformers_cache/models--IDEA-Research--grounding-dino-base` -> symlink to the HF hub cache in
  `/gscratch/robotics/harine/huggingface/hub`,
- `Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt` -> symlink to
  `/gscratch/robotics/harine/Grounded-SAM-2/checkpoints/`.

Environment notes (klone):

- `setup_env.sh` makes one local edit to the checkout's `pyproject.toml`: `[tool.uv]
  build-constraint-dependencies = ["scikit-build-core<0.10"]`, because `ruckig==0.17.3`
  (ManiSkill2 dep, sdist only) does not build with newer scikit-build-core. uv re-locks only
  that manifest line; package versions are unchanged. Resolved env: torch 2.12.0+cu130
  (driver 580 on the GPU nodes supports CUDA 13), transformers 4.47.1, Python 3.10.
- `/usr/local/sbin` on the default PATH is unsearchable, so CPython reports any missing
  executable as `PermissionError`; torch 2.12's inductor shells out to `nvcc --version` and
  only tolerates "not found", which crashed `torch.compile`. The job script loads
  `cuda/12.8.1` so a real `nvcc` is on PATH; `USE_COMPILE=False` falls back to eager.

Gemini key: `~/.config/gemini/api_key` (or `GEMINI_KEY_FILE`), model `GEMINI_MODEL`. The
official default `gemini-2.5-flash` is retired on this key (404 on 2026-10-27), so the default
here is `gemini-3.8-flash`, the current flash model and the one the repo's own plan caches used:
a deviation from the paper, recorded in every wandb run's config. Plans are requested live per
episode, as upstream does.

```bash
# smoke (inside a GPU job): 2 episodes, no wandb
OUT_DIR=/gscratch/robotics/harine/_debug_runs/pi0_smoke N_EPISODES=2 WANDB_MODE=disabled \
  bash scripts/simpler/veritas_pi0/eval_pi0.sbatch
# the two step-1 arms
SUBMIT=1 bash scripts/simpler/veritas_pi0/submit_eggplant.sh
python scripts/simpler/veritas_pi0/summarize.py /gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_<date>/eggplant
```
