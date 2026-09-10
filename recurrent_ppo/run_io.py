"""Run-directory bookkeeping shared by train.py and play.py.

A run's directory is the record of what it was: `params/args.yaml` says how the environment
was shaped and how the policy was built, and play.py reads it rather than asking the user to
retype it. Retyping is how you end up evaluating a 300-step task with a 500-step checkpoint
and never noticing.
"""

import glob
import os
from pathlib import Path

import yaml

# the arguments that change what the checkpoint IS, rather than how a run is driven. Resuming
# with any of these altered would continue one experiment under another's name.
IDENTITY_KEYS = (
    "obs",
    "corrupt_obs",
    "corrupt_t_max",
    "reward",
    "occlusion",
    "occlusion_persistence",
    "max_episode_steps",
    "render_size",
    "keypoint_visible_rate",
    "action_mode",
    "delta_scale",
    "agent_start_range",
    "block_start_range",
    "lstm_hidden_size",
    "n_lstm_layers",
    "shared_lstm",
    "net_arch",
)


def dump_args(log_dir, args_dict):
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    with open(os.path.join(log_dir, "params", "args.yaml"), "w") as f:
        yaml.safe_dump(args_dict, f, sort_keys=True)


def load_args(run_dir):
    """The args.yaml of a run directory, or None if it has none (an older or hand-made run)."""
    path = Path(run_dir) / "params" / "args.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def check_conflicts(saved, current, keys=IDENTITY_KEYS):
    """Raise if any identity key differs between a saved run and what was asked for now."""
    if saved is None:
        return
    clashes = [(k, saved[k], current[k]) for k in keys
               if k in saved and k in current and saved[k] != current[k]]
    if clashes:
        lines = "\n".join(f"  {k}: checkpoint has {s!r}, you asked for {c!r}" for k, s, c in clashes)
        raise SystemExit(
            f"[ERROR] These arguments conflict with the checkpoint's own configuration:\n{lines}\n"
            "They define what the checkpoint is, so they cannot be changed on a resume. "
            "Drop the flags, or start a new run without --checkpoint."
        )


def vecnormalize_path_for(checkpoint_path):
    """The VecNormalize pickle SB3 saves beside a checkpoint.

    Anchored on the basename: a plain str.replace would also rewrite any 'model' earlier in the
    path, so a checkpoint under /data/models/ would resolve to /data/model_vecnormalizes/.
    Matches CheckpointCallback's naming -- model.zip -> model_vecnormalize.pkl, and
    model_1200000_steps.zip -> model_vecnormalize_1200000_steps.pkl.
    """
    p = Path(checkpoint_path)
    return p.with_name(p.stem.replace("model", "model_vecnormalize", 1) + ".pkl")


def get_checkpoint_path(log_root_path, use_last_checkpoint=False):
    """Newest run under log_root_path, then its final model (or its last periodic save)."""
    runs = sorted(d for d in glob.glob(os.path.join(log_root_path, "*")) if os.path.isdir(d))
    if not runs:
        raise FileNotFoundError(f"No runs found in {log_root_path}. Pass --checkpoint explicitly.")
    run_dir = runs[-1]
    if use_last_checkpoint:
        # model_<n>_steps.zip -- sorted by n, not by name, or 900k would beat 1.2M
        saves = glob.glob(os.path.join(run_dir, "model_*_steps.zip"))
        if not saves:
            raise FileNotFoundError(f"No periodic checkpoints in {run_dir}.")
        return max(saves, key=lambda p: int(os.path.basename(p).split("_")[-2]))
    return os.path.join(run_dir, "model.zip")
