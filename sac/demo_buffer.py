"""The 206 demonstrations as chunk transitions, for every rung of the tau ladder.

WHY THE LADDER EXISTS, in one measurement: replaying every demo state through PushTEnv and
taking coverage exactly as `PushTGymEnv._block_coverage` does, the best any demonstration ever
achieves is **0.9018**, and the fraction reaching each threshold is

    0.80  93.7%        0.85  51.0%        0.90  1.0%        0.95  0.0%

`PushTEnv.success_threshold` is 0.95 and BON is scored on it. So a replay buffer seeded with
demonstrations and a single 0.95 head carries **zero** positive reward: every transition is
`r=0, done=False`, and a Q regressed on that learns `Q = 0` -- the identical "same score for
every candidate" degeneracy as the heuristic being replaced, reached more expensively. The
lower rungs are the same MDP at an easier threshold, and they have signal the demonstrations
already contain.

For head `tau` a transition is LIVE until the episode first exceeds `tau`; it is paid
`gamma_base**j` on the chunk containing that crossing (j = the within-chunk base step, so
crossing earlier is worth strictly more, matching the env) and is dropped from that head's
minibatch afterwards. That is exactly the terminate-at-tau MDP, so `Q_0.95` is unchanged by
the presence of the others.

Observations are read STRAIGHT from the zarr, not re-rendered. Verified byte-for-byte against
a live env reset to the same state: `data/img` (float32 in [0, 255]) maps to the uint8 CHW obs
`PushTGymEnv._convert_obs` produces with max difference **0**, and `data/keypoint` matches the
live 9 global keypoints to 1.3e-5 (float32 precision). So there is no re-render cost and, more
importantly, no distribution shift between demo-seeded and self-collected transitions.
"""

import hashlib
import os

import numpy as np

from recurrent_ppo.pusht_gym import PushTGymEnv
from sac.chunk_codec import encode
from sac.config import CHUNK, DEMO_ZARR, TAU_LADDER, WS, gamma_base

CACHE_DIR = os.path.expanduser("~/.cache/sac_pusht")


def _normalise(p):
    """Arena pixels -> [-1, 1], as PushTGymEnv._normalise does. One definition, imported shape."""
    return np.clip(np.asarray(p, dtype=np.float32) / (WS / 2) - 1.0, -1.0, 1.0)


def frame_coverage(zarr_path=DEMO_ZARR, cache=True):
    """Goal coverage at every demo frame, cached. ~25k shapely intersections, so not per-run."""
    import zarr

    root = zarr.open(str(zarr_path), "r")
    state = np.asarray(root["data/state"], dtype=np.float64)
    key = hashlib.md5(state.tobytes()).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"coverage_{key}.npy")
    if cache and os.path.exists(path):
        return np.load(path)

    from diffusion_policy.env.pusht.pusht_env import PushTEnv, pymunk_to_shapely

    env = PushTEnv(legacy=False, render_action=False)
    env.reset_to_state = state[0]
    env.seed(0)
    env.reset()
    goal_geom = pymunk_to_shapely(env._get_goal_pose_body(env.goal_pose), env.block.shapes)
    out = np.empty(len(state), dtype=np.float32)
    for i, s in enumerate(state):
        env._set_state(s)
        block = pymunk_to_shapely(env.block, env.block.shapes)
        out[i] = goal_geom.intersection(block).area / goal_geom.area
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, out)
    return out


def _obs_from_zarr(root, idx, obs_type):
    """The observation PushTGymEnv would emit at these demo frames, without re-simulating."""
    if obs_type == "keypoint":
        kps = np.asarray(root["data/keypoint"])[idx].reshape(len(idx), -1)   # 9 global kps
        agent = np.asarray(root["data/agent_pos"])[idx]
        flat = _normalise(np.concatenate([kps, agent], axis=-1))             # 20, in [-1, 1]
        # the demos are fully visible, so the mask half is all +1 -- the {-1,+1} encoding
        # PushTGymEnv._convert_obs uses, not {0,1}
        return np.concatenate([flat, np.ones_like(flat)], axis=-1)           # 40
    if obs_type == "image":
        # zarr img is float32 in [0, 255] HWC; the obs is uint8 CHW. Verified identical.
        return {
            "image": np.moveaxis(np.asarray(root["data/img"])[idx], -1, 1).astype(np.uint8),
            "agent_pos": _normalise(np.asarray(root["data/agent_pos"])[idx]),
        }
    if obs_type == "state":
        s = np.asarray(root["data/state"])[idx]
        return np.concatenate([_normalise(s[:, :4]),
                               np.stack([np.cos(s[:, 4]), np.sin(s[:, 4])], -1)], -1).astype(np.float32)
    raise ValueError(f"unknown obs_type {obs_type!r}")


def demo_transitions(zarr_path=DEMO_ZARR, obs_type="keypoint", chunk=CHUNK, stride=1,
                     mode="absolute", scale=64.0, gamma=0.95, tau_ladder=TAU_LADDER, frac=1.0):
    """Chunk transitions with a per-tau (reward, done, live) triple.

    Returns a dict with `obs`, `action`, `next_obs`, `reward` (N, n_tau), `done` (N, n_tau),
    `live` (N, n_tau) and `chunk_max_coverage` (N,).
    """
    import zarr

    root = zarr.open(str(zarr_path), "r")
    action = np.asarray(root["data/action"], dtype=np.float64)
    agent = np.asarray(root["data/agent_pos"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    cov = frame_coverage(zarr_path)
    gb = gamma_base(gamma, chunk)
    taus = np.asarray(tau_ladder, dtype=np.float32)

    t0, rew, done, live, cmax = [], [], [], [], []
    for s, e in zip(starts, ends):
        # first frame at which the episode exceeds each tau; len(cov) if it never does
        crossed = cov[s:e] > taus[:, None]                       # (n_tau, T)
        first = np.where(crossed.any(1), crossed.argmax(1), e - s)   # (n_tau,) episode-relative
        for t in range(s, e - chunk, stride):
            j = np.arange(1, chunk + 1)                          # states reached in this chunk
            rel = (t - s) + j                                    # episode-relative reached frames
            r = np.zeros(len(taus), np.float32)
            d = np.zeros(len(taus), bool)
            # live: the episode has not already crossed tau BEFORE this chunk's first reached
            # state. A transition after the crossing belongs to an MDP that has terminated.
            lv = first >= rel[0]
            for k, f in enumerate(first):
                if lv[k] and f <= rel[-1]:                       # the crossing is inside this chunk
                    r[k] = gb ** int(np.argmax(rel >= f))        # earlier crossing pays more
                    d[k] = True
            t0.append(t)
            rew.append(r)
            done.append(d)
            live.append(lv)
            cmax.append(cov[t + 1:t + 1 + chunk].max())

    t0 = np.asarray(t0)
    if frac < 1.0:
        keep = np.random.default_rng(0).permutation(len(t0))[:int(frac * len(t0))]
        t0, rew, done, live, cmax = t0[keep], np.asarray(rew)[keep], np.asarray(done)[keep], \
            np.asarray(live)[keep], np.asarray(cmax)[keep]

    chunks = action[t0[:, None] + np.arange(chunk)]              # (N, chunk, 2) absolute targets
    return {
        "obs": _obs_from_zarr(root, t0, obs_type),
        "next_obs": _obs_from_zarr(root, t0 + chunk, obs_type),
        "action": encode(chunks, agent[t0], mode=mode, scale=scale, chunk=chunk).astype(np.float32),
        "reward": np.asarray(rew, np.float32),
        "done": np.asarray(done, bool),
        "live": np.asarray(live, bool),
        "chunk_max_coverage": np.asarray(cmax, np.float32),
        "tau_ladder": tuple(float(t) for t in tau_ladder),
    }


def summarise(tr):
    """What each rung actually contributes -- printed at load, so a dead rung is loud."""
    lines = [f"[INFO] {len(tr['action'])} demo chunk transitions"]
    for k, tau in enumerate(tr["tau_ladder"]):
        live, pos = tr["live"][:, k], tr["done"][:, k]
        note = "" if pos.sum() else "   <-- NO POSITIVE REWARD; this rung teaches Q = 0"
        lines.append(f"       tau={tau:.2f}  live {live.mean():6.1%}  terminals {int(pos.sum()):5d}"
                     f"  ({pos.mean():.2%} of transitions){note}")
    return "\n".join(lines)
