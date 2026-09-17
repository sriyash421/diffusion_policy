"""The deployment shim: a learned Q with `PushTVerifier`'s signature.

`PushTVerifier.get_value(obs_dict, action) -> (B,)` is the one interface best-of-N ranks on
(`SearchProcedureMixin._score_candidates` -> `predict_action_best` -> argmax). Matching it
exactly is what lets a learned Q be swapped in for the hand-written distance heuristic with no
change to the search.

WHAT THIS REPLACES, and what it does not. `_score_candidates` returns `(context, value, ...)`:
`context` is what ST conditions its NEXT candidate on, `value` is what argmax ranks by. Only
`value` is replaced. ST was TRAINED with a t_goal-shaped search context, so feeding it a
Q-shaped context at eval would put it off its training distribution and confound "Q ranks
better" with "ST was handed an input it has never seen". BC ignores the context entirely, which
makes it the cleaner first read.

THE KEYPOINT ARM NEEDS NO SIMULATION. The sim verifier forks a 32-process PushT pool and steps
8 base steps per candidate; this reconstructs the block pose from `feedback` (Kabsch, exact),
transforms the 9 local keypoints by it (an affine map), and runs one forward pass. That is a
side benefit, not the point -- but it is the difference between a verifier that costs a
simulation per candidate and one that costs a matmul.

Two keypoint sets exist in this repo and must not be confused: `feedback_util` uses the T's
**8 polygon vertices** (the 16-d feedback), while `PushTKeypointsEnv` uses **9
farthest-point-sampled** keypoints (the 40-d obs). The block pose is the bridge between them.
"""

import numpy as np
import torch as th

from diffusion_policy.env.pusht.feedback_util import block_pose_from_feedback
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
from recurrent_ppo.corrupt_policy import ST_CROP
from recurrent_ppo.pusht_gym import AUG_CROP_KEY
from sac.config import WS

_KP_MANAGER = None


def keypoint_manager():
    """The 9 local block keypoints, built once. Deterministic (farthest-point, seed 0)."""
    global _KP_MANAGER
    if _KP_MANAGER is None:
        _KP_MANAGER = PymunkKeypointManager(**PushTKeypointsEnv.genenerate_keypoint_manager_params())
    return _KP_MANAGER


def _normalise(p):
    return np.clip(np.asarray(p, dtype=np.float32) / (WS / 2) - 1.0, -1.0, 1.0)


def state_from_obs(obs_dict):
    """(B, 5) [agent_x, agent_y, block_x, block_y, block_angle] -- the sim verifier's own recipe.

    Mirrors `PushTVerifier._reset_states_from_obs` rather than re-deriving it, so the learned
    and simulated verifiers are answering about the same state for the same observation.
    """
    agent = obs_dict["agent_pos"][:, -1]
    feedback = obs_dict["feedback"][:, -1]
    agent = agent.detach().cpu().numpy() if th.is_tensor(agent) else np.asarray(agent)
    feedback = feedback.detach().cpu().numpy() if th.is_tensor(feedback) else np.asarray(feedback)
    return np.concatenate([agent.astype(np.float64), block_pose_from_feedback(feedback)], axis=-1)


def obs_for_arm(obs_dict, obs_type):
    """The observation the trained Q expects, from the observation the POLICY was handed."""
    state = state_from_obs(obs_dict)                                   # (B, 5)
    if obs_type == "keypoint":
        kps = np.stack([keypoint_manager().get_keypoints_global(
            pose_map={"block": tuple(s[2:5])}, is_obj=False)["block"] for s in state])
        flat = _normalise(np.concatenate([kps.reshape(len(state), -1), state[:, :2]], axis=-1))
        return np.concatenate([flat, np.ones_like(flat)], axis=-1)     # mask half, all visible
    if obs_type == "image":
        image = obs_dict.get("image")
        if image is None:
            raise KeyError(
                "the image arm needs the current frame, but this obs_dict has no 'image'. "
                "`pusht_search_mixin._VERIFIER_OBS_KEYS` drops image keys because the SIM "
                "verifier resets from state and never re-renders; add 'image' to that tuple "
                "(additive -- PushTVerifier reads only agent_pos/feedback and ignores extras).")
        image = image.detach().cpu().numpy() if th.is_tensor(image) else np.asarray(image)
        image = image[:, -1] if image.ndim == 5 else image             # (B, To, C, H, W) -> (B, C, H, W)
        if image.dtype != np.uint8:                                    # policy obs is float [0, 1]
            image = (image * 255).astype(np.uint8)
        # IMAGE ONLY -- the pose is deliberately absent, so this Q sees no more than the ST/BC
        # policies whose candidates it ranks. `state` is still used above to rebuild the frame's
        # provenance and by the keypoint arm; it must not re-enter the image observation.
        #
        # THE CENTRE CROP IS SUPPLIED, not inferred. Training wraps the env in
        # VecAugmentationDraw, so the extractor reads its crop offset out of the observation
        # and would raise KeyError on a dict without one. Naming the centre explicitly also
        # makes deployment's crop a property of what we hand the Q, rather than a side effect
        # of whether `set_training_mode(False)` happened to have been called: the Q trains on
        # random crops and is deployed on the centre one, matching the ST/BC arms it ranks.
        height, width = image.shape[-2:]
        centre = np.array([(height - ST_CROP) // 2, (width - ST_CROP) // 2], dtype=np.float32)
        return {"image": image,
                AUG_CROP_KEY: np.broadcast_to(centre, (len(image), 2)).copy()}
    raise ValueError(f"unknown obs_type {obs_type!r}")


class PushTQVerifier:
    """A trained Q, with PushTVerifier's interface.

    `value_fn` is accepted and ignored: this verifier has exactly one value, the learned Q, and
    silently accepting `armTn` while returning something else is how an evaluation ends up
    labelled with a ranking it did not use. Anything but 'q' raises.
    """

    AGENT_DIM = 2
    STATE_DIM = 18

    def __init__(self, checkpoint, obs_type=None, rung=-1, device="auto", value_fn="q"):
        from recurrent_ppo.run_io import load_args
        from sac.config import DEFAULTS
        from sac.agent import ChunkSAC

        assert value_fn == "q", f"PushTQVerifier has one value, 'q'; got {value_fn!r}"
        import os

        saved = load_args(os.path.dirname(os.path.abspath(checkpoint))) or {}
        cfg = dict(DEFAULTS, **saved)
        self.obs_type = obs_type or cfg["obs"]
        self.rung = rung
        self.tau_ladder = tuple(cfg["tau_ladder"])
        self.agent = ChunkSAC.load(checkpoint, device=device)
        self.agent.policy.set_training_mode(False)
        self.value_fn = "q"
        print(f"[INFO] PushTQVerifier: {self.obs_type} arm, "
              f"ranking on tau={self.tau_ladder[self.rung]:.2f}")

    @th.no_grad()
    def get_value(self, obs_dict, action):
        """Scalar score per batch element. `action` is (B, H, 2) UNNORMALIZED pixel targets."""
        return self.rollout(obs_dict, action)[0]

    @th.no_grad()
    def rollout(self, obs_dict, action, render=False):
        """(value, state[, image]), matching PushTVerifier's contract.

        `state` is the CURRENT state in its [agent_pos (2), feedback (16)] layout, not a reached
        one: this verifier does not simulate, so there is no reached state to report. Callers
        that need the subgoal (search_context in {subgoal, subgoal_value}) must keep the sim
        verifier -- see the module docstring.
        """
        from sac.env import encode

        a = action.detach().cpu().numpy() if th.is_tensor(action) else np.asarray(action)
        a = a.astype(np.float64)
        state = state_from_obs(obs_dict)
        u = encode(a, chunk=a.shape[1]).astype(np.float32)
        obs = obs_for_arm(obs_dict, self.obs_type)
        value = self.agent.q_values(obs, u, rung=self.rung)

        ref = obs_dict["agent_pos"]
        device = ref.device if th.is_tensor(ref) else None
        value = value.to(device) if device is not None else value
        feedback = obs_dict["feedback"][:, -1]
        cur = th.cat([th.as_tensor(state[:, :2], dtype=feedback.dtype, device=feedback.device),
                      feedback], dim=-1) if th.is_tensor(feedback) else None
        if not render:
            return value, cur
        raise NotImplementedError(
            "PushTQVerifier does not simulate, so it has no reached subgoal frame to render. "
            "Use the sim verifier for search_context in {subgoal, subgoal_value}.")

    @th.no_grad()
    def spread(self, obs_dict, action):
        """Ensemble std of the score -- the honest uncertainty, for the abstention check.

        When this exceeds the spread ACROSS candidates, the verifier is not resolving the
        question it is being asked and a deployed BON should fall back to n=1 rather than rank
        on noise.
        """
        from sac.env import encode

        a = action.detach().cpu().numpy() if th.is_tensor(action) else np.asarray(action)
        u = encode(a.astype(np.float64), chunk=a.shape[1]).astype(np.float32)
        obs_t, _ = self.agent.policy.obs_to_tensor(obs_for_arm(obs_dict, self.obs_type))
        feats = self.agent._features(obs_t)
        u_t = th.as_tensor(u, device=self.agent.device)
        return self.agent.policy.bon_head.spread(feats, u_t)[:, self.rung]

    def close(self):
        pass
