"""Evaluate the capacity study at n=1 on validation, ID test and OOD test.

Uses the existing rollout, EMA loader, environment and per-episode noise seeding.
The 50 ID episodes must be excluded from the checkpoint's training manifest.
Select a checkpoint by validation mean reward only; report both tests at that step.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from eval_search_pusht import (load_policy, get_split_states, build_envs,
    _eval_split_at_n, override_terminal_scoring, ReplayBuffer, get_episode_init_states)
from diffusion_policy.dataset.pusht_image_dataset import load_split_manifest
from scripts.make_blq137_holdout import SOURCE, validate_holdout


def evaluate(checkpoint, run_dir, device, n_envs, weights="ema", inference_steps=8,
             include_train=False):
    policy, cfg = load_policy(str(checkpoint), device)
    env = None
    try:
        assert cfg.training.use_ema, "This comparison requires EMA weights"
        assert weights in ("ema", "live")
        if weights == "live":
            import dill
            payload = torch.load(str(checkpoint), map_location="cpu", pickle_module=dill)
            policy.load_state_dict(payload["state_dicts"]["model"])
            policy.to(device)
            del payload
        assert 1 <= inference_steps <= policy.noise_scheduler.config.num_train_timesteps
        policy.num_inference_steps = inference_steps
        assert cfg.policy.get("max_actions", 1) == 1
        policy.selection = "argmax"
        policy.corrupt_obs_eval = False
        override_terminal_scoring(policy, True)
        replay = ReplayBuffer.copy_from_path(cfg.task.dataset.zarr_path, keys=["agent_pos", "block_pos"])
        manifest = load_split_manifest(cfg.task.dataset.split_file, replay.episode_ends[:],
                                       {"train": 87, "val": 19, "test": 50})
        source = load_split_manifest(SOURCE, replay.episode_ends[:])
        validate_holdout(manifest, source)
        # This also checks the training-written splits.json against the checkpoint config.
        get_split_states(cfg, "train", run_dir=run_dir)
        states = {key: get_split_states(cfg, key, run_dir=run_dir) for key in ("val", "test")}
        mask = np.zeros(replay.n_episodes, dtype=bool)
        mask[manifest["id_test"]] = True
        states["id_test"] = (get_episode_init_states(replay, mask), np.flatnonzero(mask))
        if include_train:
            states["train"] = get_split_states(cfg, "train", run_dir=run_dir)
        env = build_envs(n_envs, cfg.policy.n_obs_steps, cfg.policy.n_action_steps, 300)
        result = dict(checkpoint=str(checkpoint), step=int(checkpoint.stem.split("_")[1]),
            holdout_checksum=manifest["holdout_checksum"], n=1, seed=42,
            max_steps=300, ema=(weights == "ema"), num_inference_steps=policy.num_inference_steps,
            selection="argmax", terminal_scoring=True, splits={})
        for key in (("train",) if include_train else ()) + ("val", "id_test", "test"):
            starts, indices = states[key]
            sr, ci, rewards = _eval_split_at_n(env, policy, starts, device, 1, n_envs, 42, key)
            result["splits"][key] = dict(episodes=indices.tolist(), success_rate=sr,
                success_ci=list(ci), mean_reward=float(rewards["max"].mean()),
                mean_coverage=float(rewards["coverage"].mean()),
                episode_max_rewards=rewards["max"].tolist(),
                episode_max_coverage=rewards["coverage"].tolist())
        return result
    finally:
        if env is not None:
            env.close()
        close = getattr(policy, "close", None)
        if close is not None:
            close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-envs", type=int, default=50)
    args = parser.parse_args()
    out = args.run_dir / "capacity_eval_id_ood"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted((args.run_dir / "checkpoints").glob("step_*.ckpt"))
    assert [int(c.stem.split("_")[1]) for c in checkpoints] == list(range(10000, 100001, 10000)), \
        "Expected the complete 10k..100k checkpoint grid; evaluation follows successful training."
    rows = []
    for checkpoint in checkpoints:
        dest = out / (checkpoint.stem + ".json")
        if dest.exists():
            row = json.loads(dest.read_text())
            assert row["checkpoint"] == str(checkpoint)
        else:
            row = evaluate(checkpoint, args.run_dir, args.device, args.n_envs)
            temp = dest.with_suffix(".tmp")
            temp.write_text(json.dumps(row, indent=2) + "\n")
            temp.replace(dest)
        rows.append(row)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    best = max(rows, key=lambda row: (row["splits"]["val"]["mean_reward"], -row["step"]))
    summary = dict(selection="maximum validation mean reward; earliest checkpoint breaks ties",
                   selected=best, final=rows[-1])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
