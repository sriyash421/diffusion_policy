"""Per-step feedback for the full-trajectory PushT search policies.

One scalar per env step, read off the REALIZED state -- no simulation. It is the
veritas+heuristic pair the chunked arms rank with (`wp_v3_la_tgn`): the T-to-goal
distance and the pusher's lookahead pin cost against the episode's live waypoint
tracker. Each term is divided by its GEOMETRIC maximum (a function of the arena, not of
any measured run), so both are in [0, 1] and are summed 1:1:

    fb = -( d_T->goal / D_T_MAX  +  pin_cost / PIN_MAX )        in (-2, 0], 0 at the goal

`TGN_*_SPREAD` (pusht_verifier) are measured stds and are deliberately NOT reused here.
"""
import math

import numpy as np

from diffusion_policy.env.pusht.feedback_util import t_goal_distance
from diffusion_policy.env.pusht.pusht_verifier import (
    LOOKAHEAD_ALPHA, LOOKAHEAD_K, lookahead_cost)
from diffusion_policy.env.pusht.veritas.pusht_client import WORKSPACE_SIZE
from diffusion_policy.env.pusht.veritas.tracker import IMAGE_SIZE, agent_px_from_pos

# The T's keypoints and the goal's both lie inside the 512-px workspace, so no
# per-keypoint distance -- hence no mean of them -- can exceed its diagonal.
D_T_MAX = WORKSPACE_SIZE * math.sqrt(2.0)
# lookahead_cost sums <= LOOKAHEAD_K alpha-decayed pusher-to-waypoint distances in the
# 96-px tracker frame; each is bounded by that frame's diagonal.
PIN_MAX = IMAGE_SIZE * math.sqrt(2.0) * sum(LOOKAHEAD_ALPHA ** p for p in range(LOOKAHEAD_K))


def pin_cost(tracker, agent_pos) -> float:
    """Lookahead pin cost of one pusher position against one live DualTracker (0 if no plan)."""
    if tracker is None:
        return 0.0
    return lookahead_cost(tracker.pusher, agent_px_from_pos(agent_pos))


def step_feedback(agent_pos, feedback, trackers) -> np.ndarray:
    """(B,) feedback scalars from workspace ``agent_pos`` (B, 2), ``feedback`` (B, 16) and
    the B live ``DualTracker``s (``None`` entries contribute no pin term)."""
    agent_pos = np.asarray(agent_pos, dtype=np.float64).reshape(-1, 2)
    feedback = np.asarray(feedback, dtype=np.float32).reshape(agent_pos.shape[0], -1)
    assert len(trackers) == agent_pos.shape[0], (len(trackers), agent_pos.shape)
    d_t = t_goal_distance(feedback) / D_T_MAX
    pin = np.array([pin_cost(tr, p) for tr, p in zip(trackers, agent_pos)]) / PIN_MAX
    return -(d_t + pin).astype(np.float32)


def demo_step_feedback(plan, agent_pos, feedback) -> np.ndarray:
    """(T,) feedback along one recorded episode, replaying its tracker from the plan.

    Anchor at frame 0 with zero updates, one ``update`` per later frame -- the
    ``TrackerSnapshotStore`` / ``VeritasTrackerWrapper`` convention, so an expert token's
    feedback is what the live env would have reported at that frame.
    """
    from diffusion_policy.env.pusht.feedback_util import block_pose_from_feedback
    from diffusion_policy.env.pusht.veritas.tracker import DualTracker
    agent_pos = np.asarray(agent_pos, dtype=np.float64)
    feedback = np.asarray(feedback, dtype=np.float32)
    poses = block_pose_from_feedback(feedback)
    tracker = None if plan is None else DualTracker(plan, agent_pos[0], poses[0])
    out = np.empty(len(agent_pos), dtype=np.float32)
    for t in range(len(agent_pos)):
        if t > 0 and tracker is not None:
            tracker.update(agent_pos[t], poses[t])
        out[t] = step_feedback(agent_pos[t:t + 1], feedback[t:t + 1], [tracker])[0]
    return out


class EpisodePlans:
    """Plans for every episode of a zarr, looked up by an episode's RECORDED first frame.

    A whole-episode batch carries no episode index, but its first frame is the
    reset state the replay buffer stores for that episode (float32, exact), so the
    first (agent_pos, feedback) identifies the episode -- the same identification
    ``OnlineSearchPolicy.compute_loss`` makes when it rebuilds reset states from frame 0.
    """

    def __init__(self, zarr_path, plan_dir, value='wp_v3_la_tgn'):
        from diffusion_policy.common.replay_buffer import ReplayBuffer
        from diffusion_policy.env.pusht.feedback_util import compute_feedback_from_pose
        from diffusion_policy.env.pusht.veritas.tracker import load_split_plans
        rb = ReplayBuffer.copy_from_path(zarr_path, keys=['agent_pos', 'block_pos'])
        ends = rb.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        idxs = list(range(len(ends)))
        plans = load_split_plans(plan_dir, idxs, 'all', value)
        agent = np.asarray(rb['agent_pos'])[starts].astype(np.float32)
        fb = compute_feedback_from_pose(np.asarray(rb['block_pos'])[starts].astype(np.float32))
        self._by_key = {self._key(a, f): (e, p) for e, a, f, p in zip(idxs, agent, fb, plans)}
        assert len(self._by_key) == len(idxs), 'two episodes share a first frame'
        self.init_states = np.concatenate(
            [agent, np.asarray(rb['block_pos'])[starts].astype(np.float32)], axis=1)  # (E, 5)

    @staticmethod
    def _key(agent_pos, feedback):
        return (np.asarray(agent_pos, dtype=np.float32).round(3).tobytes()
                + np.asarray(feedback, dtype=np.float32).round(3).tobytes())

    def lookup(self, agent_pos0, feedback0):
        """(episode index, plan) for one recorded first frame."""
        return self._by_key[self._key(agent_pos0, feedback0)]
