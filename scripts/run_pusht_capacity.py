"""Submit a paired capacity experiment; dry-run unless --submit is supplied.

Example: python scripts/run_pusht_capacity.py --submit
Both arms keep their original training recipe. Only denoiser width changes.
Four fresh models train on 87 episodes; evaluation covers ID test, OOD test and val.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
CONFIGS = ("train_pusht_unet_bc", "train_pusht_diffusion_search_single",
           "train_pusht_unet_bc_capacity", "train_pusht_st_k1_capacity")


def configs(split, seed):
    assert split == "blq137_holdout50"
    count, suffix, manifest = 87, "blq137-idholdout50", "pusht_blockquad_bottomleft_train87_idtest50_seed42.json"
    overrides = [f"n_demos={count}", f"split_suffix=_split-{suffix}",
                 f"task.dataset.split_file=diffusion_policy/config/splits/{manifest}",
                 "task.dataset.n_test_episodes=50", "task.dataset.n_val_episodes=19",
                 f"training.seed={seed}"]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(REPO / "diffusion_policy/config"), version_base=None):
        pairs = [(name, compose(config_name=name, overrides=overrides +
                  (["n_candidates=1"] if "diffusion_search_single" in name else []))) for name in CONFIGS]
        for _, cfg in pairs:
            OmegaConf.resolve(cfg)
    # Catch accidental changes to the shared parts of this comparison.
    a, b = [cfg for _, cfg in pairs[-2:]]
    for key in ("obs_encoder", "optimizer", "task.dataset", "horizon", "n_obs_steps",
                "n_action_steps", "policy.noise_scheduler", "policy.num_inference_steps",
                "training.max_gradient_steps", "training.ema_decay"):
        assert OmegaConf.select(a, key) == OmegaConf.select(b, key), key
    assert b.n_candidates == 1 and b.trainer == "offline"
    return overrides, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("blq137_holdout50",), default="blq137_holdout50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--account", default="robotics")
    parser.add_argument("--partition", default="gpu-a100")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    overrides, pairs = configs(args.split, args.seed)
    manifest = args.manifest or Path(pairs[0][1].output_root) / "capacity_experiments" / f"{args.split}_seed{args.seed}.json"
    if args.submit and manifest.exists():
        raise SystemExit(f"Submission record already exists: {manifest}; inspect it before resubmitting.")
    records = []
    env = dict(os.environ, DP_REPO=str(REPO))
    env.setdefault("TORCH_HOME", "/gscratch/robotics/harine/.cache/torch")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Match the geometric launcher: use the maintained netrc instead of a stale env key.
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text():
        env.pop("WANDB_API_KEY", None)
    for name, cfg in pairs:
        run = Path(cfg.output_root) / cfg.exp_name / cfg.task_name / cfg.trainer / cfg.run_name
        if args.submit and run.exists():
            raise SystemExit(f"Refusing to overwrite/resume an existing capacity run: {run}")
    for name, cfg in pairs:
        run = Path(cfg.output_root) / cfg.exp_name / cfg.task_name / cfg.trainer / cfg.run_name
        base = ["sbatch", "--parsable", f"--account={args.account}", f"--partition={args.partition}",
                "--export=ALL"]
        arm_overrides = overrides + (["n_candidates=1"] if "diffusion_search_single" in name else [])
        label = name.removeprefix("train_pusht_")
        train = base + ["--time=1-00:00:00", f"--job-name=capacity_{label}",
                        "scripts/slurm/train_pusht_search.sbatch", *arm_overrides]
        print(f"CONFIG_NAME={name} " + shlex.join(train), flush=True)
        rec = dict(config=name, run=str(run), overrides=arm_overrides)
        records.append(rec)
        if args.submit:
            rec["train_job"] = subprocess.check_output(train, env=dict(env, CONFIG_NAME=name), text=True).strip().split(";")[0]
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(records, indent=2) + "\n")
        dep = rec.get("train_job", "TRAIN_JOB")
        evaluate = base + [f"--dependency=afterok:{dep}", "--time=8:00:00",
            f"--job-name=capacity_eval_{label}",
            "scripts/slurm/eval_pusht_capacity.sbatch", str(run)]
        print(shlex.join(evaluate), flush=True)
        rec["eval_command"] = evaluate
        if args.submit:
            rec["eval_job"] = subprocess.check_output(evaluate, env=env, text=True).strip().split(";")[0]
            manifest.write_text(json.dumps(records, indent=2) + "\n")
    if args.submit:
        print(f"Submission record: {manifest}")


if __name__ == "__main__":
    main()
