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

## Step 2: verifier feedback in pi0's context

Model code is in `diffusion_policy/pi0_context/` (`official.py`: config, checkpoint, adapter
and Bridge normalisation from the official checkout; `pizero_context.py`: `PiZeroContext`,
RICL-style `action: feedback` blocks between proprio and the action tokens, routed through
the action expert, LoRA on the expert, learned context-attention bias so an empty context
reproduces the base checkpoint bit for bit). `test_pizero_context.py` checks that on a GPU.

Data lives under `$STEP2_ROOT/eggplant/` (`/gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_step2`),
split `position_holdout_train100` (100 train / 20 val / 50 test at two held-out eggplant
positions), all jobs through `step2_job.sbatch` (`export CMD=...; sbatch --export=ALL`):

| step | script | output |
| --- | --- | --- |
| replay demos by seed in the official SimplerEnv; eval-resized frames, proprio with gripper openness, TCP error vs the zarr | `precompute_proprio_replay.py` | `replay/ep*.npz`, `replay_summary.json` |
| H=4 verifier targets and expert value stats (`scripts/simpler/veritas/build_veritas_targets.py --horizon 4`) | plan dir of symlinked records | `plans_h4/veritas_targets.{npy,json}` |
| 4 base-pi0 chunks per frame, verifier-scored (H=4, lookahead 3, alpha 0.5, z-scored) | `precompute_context_candidates.py` | `candidates/ep*.npz` |

Replay fidelity (536 episodes): resets reproduce the zarr exactly; the TCP stays within 1 mm
for 320 episodes and the rest diverge late (train: 14/100 exceed 5 mm, near the end), so
training masks frames by `tcp_err`. The zarr's own 224 images differ from the eval LANCZOS
resize by ~3 grey levels; the replay frames are the ones pi0 sees at eval.

### Cross-task protocol (step 2 revised)

Train the context model on eggplant + stack + carrot, test on spoon (unseen). Two targets:
arm A learns the verifier-argmax chunk from collected pi0+verifier rollouts
(`submit_step2_eval.sh collect`: `ContextEvalAgent +collect_decisions` stores every decision's
frame, candidates, `-cost/tol` values and the official pick, with up to 3 attempts per seed;
`rollout_dataset.RolloutFrames`, `train_pi0_context.py --dataset rollouts --tasks ...`); arm B
keeps expert-demo targets (`ContextFrames`, per-task replay/plans/targets/candidates;
`carrot_demo_pipeline.sbatch` builds carrot's from the raw shards). Context is in generation
order, the attention bias starts at 0. Evaluation: 50 seeds per task (`<task>/eval_seeds_50.json`,
seeds 300000+), shared Gemini cache, arms `raw`, `base_bon`, `mix_bon` (4 base + 1 fine-tuned
chunk, empty context: the budget-matched control), `ckpt_bon`, `ctx_seq[+interp]`,
`ctx_tworound`; `TASK=widowx_... bash submit_step2_eval.sh <mode> [ckpt] [lambda]`.

```bash
# smoke (inside a GPU job): 2 episodes, no wandb
OUT_DIR=/gscratch/robotics/harine/_debug_runs/pi0_smoke N_EPISODES=2 WANDB_MODE=disabled \
  bash scripts/simpler/veritas_pi0/eval_pi0.sbatch
# the two step-1 arms
SUBMIT=1 bash scripts/simpler/veritas_pi0/submit_eggplant.sh
python scripts/simpler/veritas_pi0/summarize.py /gscratch/robotics/harine/RoboMonkey_data/experiments/simpler_veritas_<date>/eggplant
```
