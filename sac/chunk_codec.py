"""Between ST's absolute action chunks and the flat action a Q is defined on.

INVERTIBILITY IS THE REQUIREMENT. The Q exists to rank chunks a diffusion policy proposes, and
those arrive as absolute pixel targets `(B, 8, 2)` -- `PushTVerifier.get_value`'s second
argument. If `encode` cannot express such a chunk, the Q cannot score it, and no amount of
training fixes that. So the codec is its own module with no gym or SB3 import, and
`assert_roundtrip` is run against the real demonstrations at startup rather than trusted.

Measured over all 24,208 eight-step demo windows -- fraction of chunks with any component
outside [-1, 1], i.e. NOT representable exactly:

    absolute over [0, WS]           0.00%      <- the default
    absolute over AGENT_BOUNDS      1.16%      demo targets reach 511 px; AGENT_BOUNDS ends at 490
    increment chain at R = 64 px    2.20%
    increment chain at R = 33 px   26.01%      recurrent_ppo's measured delta_scale

`increment` is kept, and is the only reason this file has a `mode` at all: anchoring on the
previous target makes the action a SHAPE rather than a location, so a chunk that works at one
arena position is the same vector at another, and Q need not relearn "push left" everywhere.
That is a real sample-efficiency argument. It is not the default because it is the one that can
fail to represent a candidate it is handed, and a 2.2% failure rate is 2.2% of rankings decided
by a clipped action rather than the one asked about.

Clipping is part of the codec on BOTH sides, so `encode` is defined on `clip(a, 0, WS)`: two
candidates differing only outside the arena produce byte-identical trajectories and must
therefore receive the same Q.
"""

import numpy as np

from sac.config import CHUNK, CHUNK_ACTION_MODES, WS


def action_dim(chunk=CHUNK):
    return 2 * chunk


def decode(u, agent_pos, mode="absolute", scale=64.0, chunk=CHUNK):
    """(..., 2*chunk) in [-1, 1] -> (..., chunk, 2) absolute pixel targets.

    `agent_pos` (..., 2) is the agent position at the START of the chunk, in pixels. It is
    unused by `absolute` and is the first anchor for `increment`.
    """
    u = np.clip(np.asarray(u, dtype=np.float64), -1.0, 1.0).reshape(*np.shape(u)[:-1], chunk, 2)
    if mode == "absolute":
        # [-1, 1] -> [0, WS]. Every demo target lands inside, so this is exact.
        return np.clip((u + 1.0) * 0.5 * WS, 0.0, WS)
    if mode == "increment":
        # a_0 = p_t + R*u_0 ;  a_k = a_{k-1} + R*u_k. Each anchor is either the observed agent
        # position or the PREVIOUS COMMANDED TARGET -- never the agent's realised position
        # mid-chunk, which would make encode require a simulation.
        out = np.empty_like(u)
        prev = np.asarray(agent_pos, dtype=np.float64)
        for k in range(chunk):
            prev = np.clip(prev + scale * u[..., k, :], 0.0, WS)
            out[..., k, :] = prev
        return out
    raise ValueError(f"unknown chunk_action_mode {mode!r}, expected one of {CHUNK_ACTION_MODES}")


def encode(action, agent_pos, mode="absolute", scale=64.0, chunk=CHUNK):
    """(..., chunk, 2) absolute pixel targets -> (..., 2*chunk) in [-1, 1].

    The inverse of `decode` on the clipped chunk. Values may fall outside [-1, 1] under
    `increment` when the chunk steps further than `scale`; they are NOT clipped here, so a
    caller can measure how often that happens (see `representable`) instead of silently
    scoring a different action.
    """
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    if mode == "absolute":
        u = a / (WS / 2.0) - 1.0
    elif mode == "increment":
        prev = np.concatenate([np.asarray(agent_pos, dtype=np.float64)[..., None, :],
                               a[..., :-1, :]], axis=-2)
        u = (a - prev) / scale
    else:
        raise ValueError(f"unknown chunk_action_mode {mode!r}, expected one of {CHUNK_ACTION_MODES}")
    return u.reshape(*u.shape[:-2], 2 * chunk)


def representable(action, agent_pos, mode="absolute", scale=64.0, chunk=CHUNK):
    """(chunks fully inside [-1, 1], components inside) -- what fraction this codec can express."""
    u = encode(action, agent_pos, mode=mode, scale=scale, chunk=chunk)
    inside = np.abs(u) <= 1.0 + 1e-9
    return float(inside.all(axis=-1).mean()), float(inside.mean())


def assert_roundtrip(action, agent_pos, mode="absolute", scale=64.0, chunk=CHUNK, atol=1e-4):
    """Proof that a real chunk survives encode -> decode. Run at startup, not trusted.

    Only the representable chunks are checked: an `increment` chunk that needs a step longer
    than `scale` is clipped by construction and cannot round-trip, which is what
    `representable` is for. Under `absolute` every chunk is representable, so this covers all
    of them.
    """
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, WS)
    u = encode(a, agent_pos, mode=mode, scale=scale, chunk=chunk)
    ok = (np.abs(u) <= 1.0 + 1e-9).all(axis=-1)
    if not ok.any():
        raise AssertionError(f"no chunk is representable under mode={mode!r} scale={scale}")
    back = decode(u[ok], np.asarray(agent_pos)[ok], mode=mode, scale=scale, chunk=chunk)
    err = float(np.abs(back - a[ok]).max())
    if err > atol:
        raise AssertionError(
            f"chunk codec is not invertible under mode={mode!r} scale={scale}: max error "
            f"{err:.6g} px over {int(ok.sum())} chunks. A Q cannot score a chunk it cannot "
            f"express, so this is fatal rather than a warning.")
    return err


def demo_chunks(zarr_path, chunk=CHUNK, stride=1):
    """(actions (N, chunk, 2), agent_pos (N, 2)) over every in-episode window of the demos.

    Windowed WITHIN episodes: a window spanning an episode boundary joins two unrelated
    trajectories, and the chunk it produces is something no expert ever did.
    """
    import zarr

    root = zarr.open(str(zarr_path), "r")
    act = np.asarray(root["data/action"], dtype=np.float64)
    pos = np.asarray(root["data/agent_pos"], dtype=np.float64)
    ends = np.asarray(root["meta/episode_ends"])
    starts = np.concatenate([[0], ends[:-1]])
    idx = np.concatenate([np.arange(s, e - chunk + 1, stride) for s, e in zip(starts, ends)])
    return act[idx[:, None] + np.arange(chunk)], pos[idx]
