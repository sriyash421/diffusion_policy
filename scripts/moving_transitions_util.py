"""Which demo transitions the `t_goal` verifier has signal on -- the shared definition.

`t_goal` scores a candidate by where the T ends up and nothing else, so on a decision where
no candidate moves the block every candidate scores identically and the verifier is blind.
The transitions worth training on under it are the ones where the DEMONSTRATOR's own T moves
over the executed window, which is a property of the state rather than of any policy.

ONE DEFINITION, THREE CONSUMERS. `expert_moved` is imported from astar_uniform_walk rather
than restated, because scripts/rank_arms_by_blockmotion.py and sac/eval.py already stratify
their results by it -- if the training filter and that stratum ever drift, the arms stop being
measured on the transitions they were trained on. The window enumeration comes from
`create_indices`, the numba routine SequenceSampler itself uses, for the same reason: a
hand-rolled copy would be a second answer to "what are this episode's windows".
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion_policy.common.sampler import create_indices          # noqa: E402
from diffusion_policy.dataset.pusht_image_dataset import decision_frame  # noqa: E402
from scripts.astar_uniform_walk import expert_moved                 # noqa: E402

# The window geometry every arm in this generation trains under. The counts below depend on
# it -- a ratio targeted at one horizon is not the same ratio at another -- so it is recorded
# in both manifests and checked on load rather than assumed.
HORIZON, N_OBS, N_ACT = 16, 2, 8
EPS_PX = 1e-3
# expert_moved hardcodes its rotation threshold; recorded for provenance, not passed.
EPS_THETA = 1e-4


def predicate_spec():
    return {'name': 'expert_moved', 'horizon': HORIZON, 'n_obs_steps': N_OBS,
            'n_action_steps': N_ACT, 'eps_px': EPS_PX, 'eps_theta': EPS_THETA}


def episode_rows(episode_ends, episode):
    """The (buffer_start, buffer_end, sample_start, sample_end) rows of ONE episode.

    Straight from `create_indices` with a single-episode mask, at the pad settings the task
    config sets (pad_before = n_obs_steps-1, pad_after = n_action_steps-1), so these are
    bit-for-bit the rows SequenceSampler will build for that episode.
    """
    ends = np.asarray(episode_ends, dtype=np.int64)
    mask = np.zeros(ends.shape, dtype=bool)
    mask[episode] = True
    return create_indices(ends, sequence_length=HORIZON, episode_mask=mask,
                          pad_before=N_OBS - 1, pad_after=N_ACT - 1, debug=False)


def moving_frames(block_pos, episode_ends, episode):
    """(kept, total) -- absolute decision frames of `episode` whose executed window moves the T.

    The executed window is block_pos[i : i+Ta+1], matching what astar_uniform_walk.build_batch
    hands `expert_moved`. It is clipped at buffer_end rather than padded: sample_sequence pads
    a short tail by repeating the last frame, and a repeated pose never moves, so a padded
    window would read as "still" on an artefact of padding instead of on the demonstration.
    """
    kept = []
    rows = episode_rows(episode_ends, episode)
    for bs, be, ss, _ in rows:
        i = decision_frame(int(bs), int(ss), N_OBS)
        hi = min(i + N_ACT + 1, int(be))
        if hi - i < 2:
            continue                       # window lies entirely inside the edge-repeat pad
        if bool(expert_moved([block_pos[i:hi]], N_ACT, eps_px=EPS_PX)[0]):
            kept.append(i)
    return kept, len(rows)


def moving_frames_for(block_pos, episode_ends, episodes):
    """(sorted kept frames, total windows) over a set of episodes."""
    kept, total = [], 0
    for e in sorted(int(x) for x in episodes):
        k, t = moving_frames(block_pos, episode_ends, e)
        kept += k
        total += t
    return sorted(kept), total
