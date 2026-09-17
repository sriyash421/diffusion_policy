"""The SAME episodes the diffusion-policy arms are scored on, for an RL arm.

The gap this closes: every offline arm resets to the recorded first frame of each held-out demo
episode, resolved through a committed split manifest. The RL arms sampled fresh starts from
`agent_start_range` / `block_start_range` instead -- reproducible within an arm, but a different
task distribution from every arm they were being compared against. `sac/README.md` named it: the
RL side's "fixed eval set" was fixed in ST's STYLE, not ST's SET.

Two pieces, and they are separate on purpose:

  `states_from_manifest`  WHICH episodes. An RL arm has no hydra cfg, so it cannot call
                          `eval_search_pusht.get_split_states` -- that reads `cfg.task.dataset`.
                          This names the manifest directly and goes through the same validated
                          loader, so the two paths cannot disagree about what "test" means.
  `score_on_states`       Rolling exactly those episodes, one result per episode.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.pusht_image_dataset import (
    get_episode_init_states, load_split_manifest, masks_from_manifest)
from recurrent_ppo.config import DEMO_ZARR

SPLITS = ("train", "val", "test")


def states_from_manifest(split_file, split="test", zarr_path=DEMO_ZARR):
    """(states (n,5), episode_idxs (n,)) for one split of a committed manifest.

    `load_split_manifest` is given the zarr's own episode_ends, so a manifest built against a
    different dataset raises here rather than silently indexing into different frames.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rb = ReplayBuffer.copy_from_path(zarr_path, keys=["agent_pos", "block_pos"])
    ends = np.asarray(rb.episode_ends[:])
    manifest = load_split_manifest(split_file, episode_ends=ends)
    masks = dict(zip(SPLITS, masks_from_manifest(manifest, len(ends))))
    mask = masks[split]
    return get_episode_init_states(rb, mask), np.nonzero(mask)[0]


def score_on_states(agent, venv, states, num_envs, deterministic=True, max_steps=300):
    """Roll out exactly `states`, one recorded result per state.

    IN WAVES of `num_envs`, not a rolling queue, and that is forced by SB3: a VecEnv auto-resets
    the instant an episode ends, so by the time `done` is visible the env has already restarted
    -- on whatever `reset_to_state` it still held. Assigning the next episode at that point would
    replay the previous one. A wave assigns each env one episode, resets, and runs until every
    env in the wave has finished; envs that finish early keep stepping and their extra
    completions are ignored. That is the same chunking the offline runner uses.

    Returns a list of per-episode dicts in the order of `states`, so results stay keyed to the
    episode index rather than to whichever env happened to finish.
    """
    out = []
    for start in range(0, len(states), num_envs):
        wave = states[start:start + num_envs]
        # every env gets an episode; the tail wave repeats earlier ones into the spare envs and
        # discards them, because a VecEnv cannot be stepped with fewer envs than it has
        assigned = [np.asarray(wave[i % len(wave)], dtype=np.float64) for i in range(num_envs)]
        _pin(venv, assigned)
        obs = venv.reset()
        lstm_states = None
        episode_starts = np.ones((num_envs,), dtype=bool)
        result = [None] * len(wave)
        steps = 0
        while any(r is None for r in result) and steps <= max_steps + 1:
            actions, lstm_states = agent.predict(
                obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic)
            obs, _, dones, infos = venv.step(actions)
            episode_starts = dones
            steps += 1
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and i < len(wave) and result[i] is None:
                    result[i] = {
                        "return": float(info["episode"]["r"]),
                        "is_success": bool(info["is_success"]),
                        "max_reward": float(info["max_reward"]),
                    }
        if any(r is None for r in result):
            raise RuntimeError(
                f"{sum(r is None for r in result)} of {len(wave)} episodes did not terminate "
                f"within {max_steps} steps. The env's max_episode_steps and this budget "
                f"disagree, so the wave would hang.")
        out.extend(result)

    # leave no pinned state behind: an env still holding one would replay that episode forever
    for i in range(num_envs):
        venv.env_method("set_wrapper_attr", "reset_to_state", None, indices=[i])
    return out


def _pin(venv, states):
    """Pin one state per env, then READ IT BACK and raise if it did not take.

    The read-back is not defensive padding. `VecEnv.set_attr` does a plain `setattr` on the
    outermost per-env wrapper, which here is `Monitor`, so setting `reset_to_state` that way
    puts it on the wrapper and `PushTGymEnv.reset` samples as usual -- an eval that runs on
    random starts while labelling its rows with manifest episode indices, with nothing raising.
    `set_wrapper_attr` walks the wrapper chain and assigns where the attribute already lives,
    which is the only spelling that works on gymnasium 1.x -- 1.0 removed the attribute
    forwarding that made `env_method("set_reset_state", ...)` viable, so that raises
    AttributeError on the Monitor. Measured here: gymnasium 1.1.1. Rather than trust any of
    this, set it, read it back, and fail loudly if the two disagree.
    """
    for i, st in enumerate(states):
        venv.env_method("set_wrapper_attr", "reset_to_state", st, indices=[i])
    back = [venv.env_method("get_wrapper_attr", "reset_to_state", indices=[i])[0]
            for i in range(len(states))]
    for i, (want, got) in enumerate(zip(states, back)):
        if got is None or not np.allclose(np.asarray(got, dtype=np.float64), want, atol=1e-9):
            raise RuntimeError(
                f"env {i}: reset state did not reach PushTGymEnv (read back {got!r}, wanted "
                f"{want!r}). The env stack forwards neither set_attr nor env_method to the "
                f"inner env, so a fixed-episode evaluation here would silently score random "
                f"starts. Check the wrapper chain in pusht_gym.make_env.")
