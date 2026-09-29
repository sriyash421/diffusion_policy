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
        # 20-d, mask APPLIED not appended -- PushTGymEnv stopped carrying the visibility mask
        # as features, so the trained net's first layer is 20 wide. A 40-d vector here would be
        # a shape error at best and, if the widths ever coincided, a silently wrong score.
        # These frames are fully visible, so applying the mask is the identity.
        return _normalise(np.concatenate([kps.reshape(len(state), -1), state[:, :2]], axis=-1))
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

    `value_fn` names WHICH Q, and is otherwise inert: this verifier has exactly one value, the
    learned Q, and silently accepting `armTn` while returning something else is how an
    evaluation ends up labelled with a ranking it did not use. So only the bare `'q'` (the
    eval path, which is handed an explicit `--q <path>`) and the registered names of
    `pusht_verifier.Q_VERIFIERS` (the training path, where the name IS the path) are accepted;
    anything else raises.

    THE NAME IS NOT CHECKED AGAINST THE CHECKPOINT. `Q_VERIFIERS` is the one binding between
    them and `q_verifier_spec` is what reads it; re-deriving the mapping here would be a second
    copy of it. This assert only refuses a name that belongs to a DIFFERENT KIND of verifier.
    """

    AGENT_DIM = 2
    STATE_DIM = 18

    def __init__(self, checkpoint, obs_type=None, rung=-1, device="auto", value_fn="q"):
        from recurrent_ppo.run_io import load_args
        from sac.config import DEFAULTS
        from sac.agent import ChunkSAC

        from diffusion_policy.env.pusht.pusht_verifier import Q_VERIFIERS

        assert value_fn == "q" or value_fn in Q_VERIFIERS, (
            f"PushTQVerifier scores a learned Q; 'q' or one of {sorted(Q_VERIFIERS)} names "
            f"that, and {value_fn!r} does not.")
        import os

        saved = load_args(os.path.dirname(os.path.abspath(checkpoint))) or {}
        cfg = dict(DEFAULTS, **saved)
        self.obs_type = obs_type or cfg["obs"]
        self.rung = rung
        self.tau_ladder = tuple(cfg["tau_ladder"])
        self.agent = ChunkSAC.load(checkpoint, device=device)
        self.agent.policy.set_training_mode(False)
        # the name it was ASKED for, not a hardcoded 'q': the eval-side labelling reads
        # this back, and flattening `q_sac_106` to `q` here is how a control run gets filed
        # under the arm it was the control for.
        self.value_fn = value_fn
        print(f"[INFO] PushTQVerifier[{value_fn}]: {self.obs_type} arm, "
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


# ==================================================================== the hold-out guard
#
# A Q trained on demonstrations can be handed ANY checkpoint's held-out episodes, and until this
# existed nothing checked that the two manifests were disjoint. The failure is silent and in the
# flattering direction: the Q has seen the very transitions it is being asked to rank, and the
# heuristic it is compared against has no training set at all, so the comparison reads as a win.
#
# The three groups this is built for:
#   blq137 ckpts  <- the blq137 Q    0 overlap
#   brd100 ckpts  <- the brd100 Q    0 overlap
#   brd60  ckpts  <- the brd100 Q    0 overlap, because brd60.test IS brd100.test and
#                                    brd60.train is a strict subset of brd100.train
# and the two it must refuse: blq137 test under the brd100 Q (29 episodes), and brd60 VAL under
# the brd100 Q (all 40) -- the val case is why `--split test` is not merely a convention here.


def _run_dir_of(run_or_ckpt):
    """The run directory, whether given one or a checkpoint inside it.

    The same resolution `PushTQVerifier.__init__` uses, so the guard and the verifier can never
    disagree about which run a checkpoint belongs to.
    """
    import os

    p = os.path.abspath(run_or_ckpt)
    return p if os.path.isdir(p) else os.path.dirname(p)


def demo_train_episodes(run_or_ckpt):
    """Episodes that seeded this run's replay buffer. `None` means EVERY episode.

    Read off the RAW args.yaml, never off `dict(DEFAULTS, **saved)`: a PPO or BC run records no
    `demo_seed_frac`, and merging the SAC defaults over it would invent `demo_episodes="all"`
    and report a run that seeds no demonstrations at all as maximally contaminated. Key
    PRESENCE is the test for "is this a demo-seeded SAC run", because its absence is the only
    honest signal a foreign run gives.
    """
    from recurrent_ppo.run_io import load_args

    saved = load_args(_run_dir_of(run_or_ckpt)) or {}
    if "demo_seed_frac" not in saved:
        return set()                                   # not a demo-seeding run
    if float(saved.get("demo_seed_frac", 0.0)) <= 0.0:
        return set()                                   # seeding switched off
    if saved.get("demo_episodes", "all") == "all":
        return None                                    # every episode, including every eval one
    from recurrent_ppo.eval_episodes import states_from_manifest

    _, idxs = states_from_manifest(saved["split_file"], "train")
    return {int(i) for i in idxs}


def held_out_report(run_or_ckpt, episode_idxs):
    """What the guard knows, as a dict -- so it can ride in the output JSON beside the numbers.

    A result that cannot say whether it was held out is a result nobody can weigh later.
    """
    from recurrent_ppo.run_io import load_args

    saved = load_args(_run_dir_of(run_or_ckpt)) or {}
    seen = demo_train_episodes(run_or_ckpt)
    asked = [int(i) for i in episode_idxs]
    overlap = sorted(asked) if seen is None else sorted(set(asked) & seen)
    return {"run_dir": _run_dir_of(run_or_ckpt),
            "demo_episodes": saved.get("demo_episodes"),
            "demo_split_file": saved.get("split_file"),
            "seeded_from_all_episodes": seen is None,
            "n_scored": len(asked),
            "n_overlap": len(overlap),
            "overlap": overlap,
            "held_out": len(overlap) == 0}


def assert_held_out(run_or_ckpt, episode_idxs, allow=False):
    """Refuse to score `episode_idxs` with a Q whose buffer contains them. -> held_out_report.

    `allow` (the callers' --allow-contaminated) is the deliberate escape hatch, and it is
    recorded in the returned report rather than merely permitted: the all-206 `sac_keypoint` run
    can still be measured, and its numbers stay distinguishable from the held-out ones on a
    shared plot.
    """
    rep = dict(held_out_report(run_or_ckpt, episode_idxs), allowed_contaminated=bool(allow))
    if rep["held_out"] or allow:
        return rep
    where = ("seeded from ALL episodes" if rep["seeded_from_all_episodes"]
             else f"seeded from {rep['demo_split_file']}'s train split")
    raise SystemExit(
        f"[ERROR] {rep['n_overlap']} of the {rep['n_scored']} episodes being scored are in this "
        f"Q's replay buffer.\n"
        f"  Q run: {rep['run_dir']} ({where})\n"
        f"  overlapping episodes: {rep['overlap']}\n"
        "The Q has seen these transitions and the heuristic it is compared against has no "
        "training set at all, so the comparison would read as a win for the wrong reason. "
        "Score a manifest this Q holds out, or pass --allow-contaminated to record the number "
        "with the contamination declared beside it.")
