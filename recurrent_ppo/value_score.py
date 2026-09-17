"""A trained PPO V as the best-of-N verifier, with PushTVerifier's interface.

The heuristic being replaced, `t_goal`, ignores `agent_pos` by construction, so until a
candidate actually moves the block every candidate scores identically -- a blind rate the repo
measures at 28-35% of decisions. A learned V can express "this chunk leads somewhere good"
before contact, because its return telescopes to total progress under the delta reward.

V TAKES NO ACTION, so ranking with it is two steps, not one:

    1. the SIM verifier rolls the candidate chunk out and reports the state it REACHES
    2. V is evaluated at that state

Step 1 is not extra machinery -- `PushTVerifier.rollout` already returns the reached state
`(B, 18) = [agent_pos(2), feedback(16)]`, and with `render=True` the frame at it, byte-identical
to what `PushTImageEnv` would emit. So a V ranker costs exactly what the heuristic costs, plus
one forward pass. That is why `install_q_ranker`'s `--skip-context-sim` shortcut is valid for Q
and NOT for V: Q needs no rollout, V's input IS the rollout.

TWO THINGS THAT WOULD BE SILENTLY WRONG:

  * VecNormalize. `norm_reward` is on by default, so V is fitted in units of
    `reward / sqrt(ret_rms.var)` and means nothing until that factor is restored -- which is
    how a working critic looks broken. The saved `model_vecnormalize.pkl` is loaded and the
    factor is printed.
  * The crop key. A PPO image checkpoint emits no `aug_crop` and centre-crops internally, so
    adding one -- as `sac.score.obs_for_arm` must for SAC -- would be wrong here. The
    observation is built without it.

RECURRENT V IS QUERIED WITH NO HISTORY. A reached state is a one-off: there is no preceding
observation, so the LSTM starts from zeros and `V(history, s)` is evaluated off-distribution.
The feed-forward arms have no such problem. Record it beside any recurrent-V number.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch as th


def _normalise(p):
    """Arena pixels -> [-1, 1], the same map PushTGymEnv applies to its observation."""
    from recurrent_ppo.config import WS

    return (np.asarray(p, dtype=np.float32) / (WS / 2.0) - 1.0).astype(np.float32)


def obs_for_arm(state, image, obs_type):
    """The observation a PPO checkpoint expects, from a state the sim verifier REACHED.

    `state` is (B, 18) = [agent_pos(2), feedback(16)], the layout PushTVerifier.rollout returns.
    `image` is its rendered frame, or None on the keypoint arm.
    """
    from sac.score import keypoint_manager
    from diffusion_policy.env.pusht.feedback_util import block_pose_from_feedback

    state = np.asarray(state, dtype=np.float64)
    if obs_type == "keypoint":
        pose = np.asarray([block_pose_from_feedback(f) for f in state[:, 2:]], dtype=np.float64)
        kps = np.stack([keypoint_manager().get_keypoints_global(
            pose_map={"block": tuple(p)}, is_obj=False)["block"] for p in pose])
        # 20-d, mask APPLIED not appended -- these are fully visible, so applying it is the
        # identity. A 40-d vector would be a shape error against a 20-wide first layer.
        return _normalise(np.concatenate([kps.reshape(len(state), -1), state[:, :2]], axis=-1))
    if obs_type == "image":
        if image is None:
            raise ValueError(
                "the image arm needs the frame at the reached state; call "
                "PushTVerifier.rollout(..., render=True), which renders once per chunk")
        image = np.asarray(image)
        if image.dtype != np.uint8:                 # the verifier renders float [0, 1]
            image = (image * 255).astype(np.uint8)
        # NO aug_crop: a PPO checkpoint emits no crop key and centre-crops inside the
        # extractor. Supplying one is silently wrong rather than a crash.
        return {"image": image}
    raise ValueError(f"unknown obs_type {obs_type!r}")


class PushTVVerifier:
    """A trained PPO V, with PushTVerifier's interface.

    `value_fn` is accepted and must be 'v'. Silently accepting 't_goal' while returning
    something else is how an evaluation ends up labelled with a ranking it did not use, which is
    the same guard `PushTQVerifier` applies.
    """

    AGENT_DIM = 2
    STATE_DIM = 18

    def __init__(self, checkpoint, sim_verifier, obs_type=None, arch=None, device="auto",
                 value_fn="v"):
        from recurrent_ppo.arch import ARCHS
        from recurrent_ppo.config import DEFAULTS
        from recurrent_ppo.run_io import load_args, vecnormalize_path_for

        assert value_fn == "v", f"PushTVVerifier has one value, 'v'; got {value_fn!r}"
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        saved = load_args(run_dir) or {}
        cfg = dict(DEFAULTS, **saved, device=device)
        self.obs_type = obs_type or cfg["obs"]
        self.arch = ARCHS[arch or saved.get("arch", "lstm")]
        self.recurrent = self.arch.name == "lstm"

        # THE SIM VERIFIER IS A DEPENDENCY, not an implementation detail. V takes no action, so
        # the candidate chunk has to be rolled out before there is a state to evaluate -- this
        # object supplies that rollout, and is the same one the heuristic uses, so the two
        # rankers are compared on identical reached states.
        self.sim = sim_verifier
        self.agent = self.arch.load(checkpoint, None, cfg, print_system_info=False)
        self.agent.policy.set_training_mode(False)

        # V is fitted to VecNormalize-scaled returns, so it is in units of
        # reward / sqrt(ret_rms.var) and means nothing until the factor is restored -- which is
        # exactly how a working critic looks broken. Ranking is scale-invariant, so this does
        # not change an argmax; it makes the NUMBERS readable and comparable across runs.
        self.reward_scale = 1.0
        vec_path = vecnormalize_path_for(checkpoint)
        if vec_path and os.path.exists(vec_path):
            import pickle

            with open(vec_path, "rb") as f:
                vec = pickle.load(f)
            rms = getattr(vec, "ret_rms", None)
            if rms is not None:
                self.reward_scale = float(np.sqrt(rms.var + getattr(vec, "epsilon", 1e-8)))
        self.value_fn = "v"
        print(f"[INFO] PushTVVerifier: {self.obs_type} arm, {self.arch.name}, "
              f"reward_scale {self.reward_scale:.4g}"
              + ("  (recurrent V is queried with a ZERO LSTM state -- a reached state has no "
                 "history, so this is off-distribution)" if self.recurrent else ""))

    @th.no_grad()
    def get_value(self, obs_dict, action):
        return self.rollout(obs_dict, action)[0]

    @th.no_grad()
    def rollout(self, obs_dict, action, render=False):
        """Roll the chunk out, then evaluate V at the state it reaches.

        Returns `(value, state)` like PushTVerifier, so it drops into `_score_candidates`
        unchanged. The rollout is the sim's; only the scoring of the reached state is ours.
        """
        needs_image = self.obs_type == "image"
        out = self.sim.rollout(obs_dict, action, render=needs_image or render)
        state = out[1]
        image = out[2] if len(out) > 2 else None

        obs = obs_for_arm(np.asarray(state.cpu() if hasattr(state, "cpu") else state),
                          None if image is None else
                          (image.cpu().numpy() if hasattr(image, "cpu") else image),
                          self.obs_type)
        policy = self.agent.policy
        obs_t, _ = policy.obs_to_tensor(obs)
        if self.recurrent:
            # zero state: a reached state is a one-off with no preceding observation. Recorded
            # in the banner because it is a real caveat, not a detail.
            shape = (policy.lstm_critic.num_layers, len(state), policy.lstm_critic.hidden_size)
            zeros = (th.zeros(shape, device=policy.device), th.zeros(shape, device=policy.device))
            starts = th.ones(len(state), device=policy.device)
            values = policy.predict_values(obs_t, zeros, starts)
        else:
            values = policy.predict_values(obs_t)
        return values.flatten() * self.reward_scale, state

    def close(self):
        pass
