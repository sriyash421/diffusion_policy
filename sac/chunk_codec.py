"""Between ST's absolute action chunks and the flat action a Q is defined on.

INVERTIBILITY IS THE REQUIREMENT. The Q exists to rank chunks a diffusion policy proposes, and
those arrive as absolute pixel targets `(B, 8, 2)` -- `PushTVerifier.get_value`'s second
argument. A codec that cannot express such a chunk cannot score it, and no amount of training
fixes that. Measured over all 24,208 eight-step demo windows, the fraction NOT representable:

    absolute over [0, WS]           0.00%      <- what this is
    absolute over AGENT_BOUNDS      1.16%      demo targets reach 511 px; AGENT_BOUNDS ends at 490
    increment chain at R = 64 px    2.20%
    increment chain at R = 33 px   26.01%      recurrent_ppo's measured delta_scale

An increment chain anchors each target on the previous one, which makes the action a SHAPE
rather than a location and buys translation equivariance. It was implemented, measured and
dropped: at any scale small enough to be a useful exploration prior it cannot express a quarter
of what the expert actually did, and a verifier that silently clips the chunk it was asked about
is worse than one that is merely coarse. `tests/test_sac.py` keeps the measurement, so that
number stays an executable fact rather than a claim in a comment.

Because absolute needs no anchor, `encode` and `decode` are pure per-element maps -- no agent
position, no mode, no scale. The anchor the demo-shape and smooth-walk samplers use is the
sampler's business (see `sac/sac.py`), not the codec's.

Clipping is part of the codec on BOTH sides, so `encode` is defined on `clip(a, 0, WS)`: two
candidates differing only outside the arena drive byte-identical trajectories and must
therefore receive the same Q.
"""

import numpy as np

from sac.config import CHUNK, WS


def action_dim(chunk=CHUNK):
    return 2 * chunk


def decode(u, chunk=CHUNK):
    """(..., 2*chunk) in [-1, 1] -> (..., chunk, 2) absolute pixel targets."""
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0)
    u = u.reshape(*u.shape[:-1], chunk, 2)
    return np.clip((u + 1.0) * 0.5 * WS, 0.0, WS)


def encode(action, chunk=CHUNK):
    """(..., chunk, 2) absolute pixel targets -> (..., 2*chunk) in [-1, 1]. Exact inverse."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    u = a / (WS / 2.0) - 1.0
    return u.reshape(*u.shape[:-2], 2 * chunk)


def assert_roundtrip(action, chunk=CHUNK, atol=1e-4):
    """Proof that a real chunk survives encode -> decode. Run at startup, not trusted."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    err = float(np.abs(decode(encode(a, chunk), chunk) - a).max())
    if err > atol:
        raise AssertionError(
            f"the chunk codec is not invertible: max error {err:.6g} px. A Q cannot score a "
            f"chunk it cannot express, so this is fatal rather than a warning.")
    return err


def demo_chunks(zarr_path, chunk=CHUNK, stride=1):
    """(actions (N, chunk, 2), agent_pos (N, 2)) over every in-episode window of the demos.

    Windowed WITHIN episodes: a window spanning an episode boundary joins two unrelated
    trajectories, and the chunk it produces is something no expert ever did. `agent_pos` is not
    needed by the codec; it is returned because the demo-shape sampler anchors on it.
    """
    import zarr

    root = zarr.open(str(zarr_path), "r")
    act = np.asarray(root["data/action"], dtype=np.float64)
    pos = np.asarray(root["data/agent_pos"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    idx = np.concatenate([np.arange(s, e - chunk + 1, stride) for s, e in zip(starts, ends)])
    return act[idx[:, None] + np.arange(chunk)], pos[idx]
