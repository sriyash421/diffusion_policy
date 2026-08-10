"""Offline conditional-diffusion search policy for PushT images.

Thin subclass of ``DiffusionTransformerSearchPolicy`` (the maze diffusion-search policy).
Everything about the transformer + candidate-search loop is inherited unchanged; only
three things differ, so the maze path is left untouched:

1. Verifier -- builds a feedback-based ``PushTVerifier`` instead of the maze-only
   ``MazeVerifier`` (see ``_build_verifier``).
2. Obs handling -- some runner paths add an ``attention_mask`` to the obs dict. It is in
   neither ``shape_meta`` nor the normalizer, so it is popped before encode/normalize --
   on a *copy*, leaving the caller's obs dict intact. (The block pose the verifier needs
   is reconstructed from the declared ``feedback`` key, so no reset carrier rides along.)
3. Search context (``search_context``) -- what feedback each already-generated candidate
   contributes to the context the next candidate is conditioned on:

   * ``'value'``   (default): the verifier scalar ``-mean_kp dist(goal, achieved)``.
     This is the original behaviour.
   * ``'state'``   : the **subgoal state** the candidate's action chunk actually reaches
     in the PushT sim -- ``[agent_pos (2), feedback (16)]``, normalizer-scaled. No scalar
     verdict at all, so the model must read the outcome off the state itself.
   * ``'state_value'`` : both, concatenated -- the subgoal state plus the scalar.
   * ``'subgoal'`` : the **rendered subgoal observation** the chunk lands on, pushed
     through this policy's own obs encoder -- i.e. the candidate reports back "here is
     what the world looks like afterwards", in the same feature space as the obs tokens.
   * ``'subgoal_value'`` : the encoded subgoal observation plus the scalar.

   All of them come out of the *same* simulated rollout, so the wider contexts cost no
   extra sim steps (the two subgoal modes add one render + one encoder pass per
   candidate). Candidate *ranking* uses the scalar verifier value in every mode -- but
   ranking is only what ``selection: argmax`` does with it. Under ``selection:
   final_pass`` the executed action is a further sample conditioned on all n scored
   candidates and no ranking happens at all, so the scalar reaches the model only through
   the context above. `selection` is defined on the base class; the two are orthogonal,
   and the arm labels name the pair:

     (value, argmax)         -> value
     (subgoal, argmax)       -> subgoal-chosen4value
     (subgoal_value, argmax) -> subgoal-value
     (subgoal, final_pass)   -> subgoal-only     <- the only arm with no oracle selection

The obs encoder is not changed here: the base class takes it as an injected arg, and the
PushT config injects ``MultiImageObsEncoder`` (ResNet18) instead of ``FlattenObsEncoder``.
"""
from typing import Dict, Optional

import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.diffusion_transformer_search_policy import (
    DiffusionTransformerSearchPolicy)
from diffusion_policy.env.pusht.pusht_verifier import PushTVerifier
from diffusion_policy.env.pusht.feedback_util import N_KEYPOINTS

# obs keys that ride in the obs dict but must never reach the encoder/normalizer
_NON_ENCODED_OBS_KEYS = ('attention_mask',)

# obs keys the verifier reads to reset the sim: agent_pos plus feedback, from which the
# block pose is reconstructed exactly (pusht_verifier.block_pose_from_feedback).
_VERIFIER_OBS_KEYS = ('agent_pos', 'feedback')

# per-candidate search-context feedback modes; see the module docstring for what each
# feeds back. The two 'subgoal*' modes render + encode the reached observation.
SEARCH_CONTEXTS = ('value', 'state', 'state_value', 'subgoal', 'subgoal_value')

# modes that need the sim to render the reached observation
_RENDER_CONTEXTS = ('subgoal', 'subgoal_value')


class PushTDiffusionSearchPolicy(DiffusionTransformerSearchPolicy):
    def _build_verifier(self, **kwargs):
        return PushTVerifier(
            n_envs=kwargs.get('verifier_n_envs', 32),
            legacy=kwargs.get('verifier_legacy', False),
            use_async=kwargs.get('verifier_use_async', True),
            verifier_steps=kwargs.get('verifier_steps', None),
        )

    @staticmethod
    def _search_context_mode(kwargs) -> str:
        mode = kwargs.get('search_context', 'value') or 'value'
        assert mode in SEARCH_CONTEXTS, \
            f"search_context must be one of {SEARCH_CONTEXTS}, got {mode!r}"
        return mode

    def _context_dim(self, obs_feature_dim: int, **kwargs) -> int:
        mode = self._search_context_mode(kwargs)
        return {
            'value': 1,
            'state': PushTVerifier.STATE_DIM,
            'state_value': PushTVerifier.STATE_DIM + 1,
            # the encoded subgoal obs is exactly as wide as an encoded obs step, since
            # it goes through the same obs_encoder (ResNet feature + the low_dim keys).
            'subgoal': obs_feature_dim,
            'subgoal_value': obs_feature_dim + 1,
        }[mode]

    def _verifier_inputs(self, obs_dict, action):
        """The current state and the action chunk that will actually be executed.

        Both halves need explicit indexing rather than the `[:, -1]` / "whole chunk"
        defaults, because the two callers hand in differently shaped windows:

        * obs: at EVAL the runner's MultiStepWrapper yields exactly ``n_obs_steps``
          steps, so ``[:, -1]`` is the current state. At TRAIN the dataset yields the
          full ``horizon`` (16) steps while the policy conditions on ``[:, :To]``, so
          ``[:, -1]`` would be the state ~14 control steps in the FUTURE. Index step
          ``To-1`` explicitly, keeping the time dim so the verifier's ``[:, -1]``
          still selects it.
        * action: ``predict_action`` / ``predict_action_best`` execute
          ``action_pred[:, To-1 : To-1+Ta]``, so that -- not ``action_pred[:, :Ta]`` --
          is the window the verifier must simulate, or it replays one already-past
          action and never simulates the last executed one.

        Only the keys the verifier reads are sliced; image/mask keys are dropped since
        the verifier resets the sim from state and never re-renders.
        """
        To, Ta = self.n_obs_steps, self.n_action_steps
        now = {k: v[:, To - 1:To] for k, v in obs_dict.items()
               if k in _VERIFIER_OBS_KEYS}
        return now, action[:, To - 1:To - 1 + Ta]

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Scale a raw rollout state ``[agent_pos (2), feedback (16)]``.

        The state comes out of the sim in raw pixel units (agent_pos in [0, 512],
        feedback a keypoint displacement in pixels). Both halves are exactly the two
        low_dim obs keys, so the policy's own normalizer already carries fitted stats for
        them -- reusing it puts the search context on the same scale as the obs the
        encoder sees, instead of dwarfing the embedding with hundreds-of-pixels inputs.
        """
        agent_dim = PushTVerifier.AGENT_DIM
        agent_pos = self.normalizer['agent_pos'].normalize(state[..., :agent_dim])
        feedback = self.normalizer['feedback'].normalize(state[..., agent_dim:])
        return torch.cat([agent_pos, feedback], dim=-1)

    def _normalize_value(self, state: torch.Tensor) -> torch.Tensor:
        """The verifier scalar rescaled onto the fitted feedback scale -> (B,).

        The raw scalar is ``-mean_kp ||feedback||`` in pixels (~0 to -300), which would sit
        about two orders of magnitude above the normalized state / subgoal embedding it is
        concatenated with, dominating ``action_value_emb``'s input.

        It is recomputed from SCALE-normalized feedback so the fitted statistics do the
        rescaling (no magic constant). The normalizer's *offset* is deliberately omitted:
        the value's defining property is that it is 0 exactly at the goal, and an offset
        would make ||feedback|| nonzero there and destroy that.

        Only the CONTEXT copy is rescaled -- the raw scalar still ranks candidates, so
        predict_action_best and the train_action_value* metrics are unchanged.
        """
        agent_dim = PushTVerifier.AGENT_DIM
        feedback = state[..., agent_dim:]              # (B, 2*N_KEYPOINTS), raw pixels
        scale = self.normalizer['feedback'].params_dict['scale'].to(feedback)
        kp = (feedback * scale).reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)
        return -kp.norm(dim=-1).mean(dim=-1)           # (B,)

    def _encode_subgoal(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Encode the reached observation -> (B, obs_feature_dim).

        Goes through the SAME normalize -> _encode_obs path as the observation
        conditioning (and as OnlineSearchPolicy's context tokens), so the subgoal
        embedding lands in the same feature space as the obs tokens the search is already
        conditioned on, reuses the fitted normalizer and obs encoder, and adds no
        parameters. MultiImageObsEncoder.forward requires every shape_meta obs key, which
        is exactly what the rollout returns: the rendered frame plus the two low_dim keys.

        The subgoal is a single step, so it is given a length-1 time axis for _encode_obs
        (which is shaped (B, T, ...) like online's) and squeezed back afterwards.
        """
        agent_dim = PushTVerifier.AGENT_DIM
        subgoal_obs = {
            'image': image,                      # (B, 3, 96, 96) in [0, 1]
            'agent_pos': state[..., :agent_dim],
            'feedback': state[..., agent_dim:],
        }
        nobs = self.normalizer.normalize(subgoal_obs)
        nobs = dict_apply(nobs, lambda x: x.unsqueeze(1))   # (B, 1, ...)
        return self._encode_obs(nobs)[:, 0]                 # (B, obs_feature_dim)

    def _score_candidates(self, verifier, obs_dict, action, want_subgoals: bool = False):
        """Simulate the candidate chunk once; build the context per ``search_context``.

        Ranking always uses the scalar verifier value, so ``predict_action_best`` picks
        the same way in every mode -- only what the *next* candidate gets to see changes.

        ``want_subgoals`` forces the render even in modes that would not otherwise need
        it, so a `value`-mode run can still log subgoal panels comparable to the others.
        """
        mode = self._search_context_mode(self.kwargs)
        render = want_subgoals or mode in _RENDER_CONTEXTS
        # simulate the EXECUTED chunk from the CURRENT state (see _verifier_inputs);
        # neither is the default the verifier would pick on its own.
        now, exec_action = self._verifier_inputs(obs_dict, action)

        # always rollout, never get_value: the two cost the same (get_value is literally
        # rollout(...)[0]) and rollout also returns the reached state, which every mode
        # now needs -- the context scalar is rescaled from it (see _normalize_value).
        out = verifier.rollout(now, exec_action, render=render)
        value, state = out[0], out[1]                # (B,), (B, STATE_DIM)
        image = out[2] if render else None           # (B, 3, 96, 96)

        # context gets the rescaled scalar; `value` stays raw and is returned as the score.
        nvalue = self._normalize_value(state)        # (B,)

        if mode == 'value':
            context = nvalue
        elif mode in ('state', 'state_value'):
            nstate = self._normalize_state(state)
            context = nstate if mode == 'state' else \
                torch.cat([nstate, nvalue.unsqueeze(-1)], dim=-1)
        else:
            embedding = self._encode_subgoal(image, state)
            context = embedding if mode == 'subgoal' else \
                torch.cat([embedding, nvalue.unsqueeze(-1)], dim=-1)

        subgoal = None
        if want_subgoals:
            # logging only -- never fed back into the search
            subgoal = {'image': image, 'value': value}
        return context, value, subgoal

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Filtering happens here rather than in encode_obs_cond because the search loop
        # calls _encode_obs_features directly (to encode once and reuse across
        # candidates); encode_obs_cond just adds corruption on top of this.
        #
        # Pop the mask key on a shallow copy so the caller's obs_dict is left intact
        # (verifier.get_value reads the same dict).
        clean = {k: v for k, v in obs_dict.items()
                 if k not in _NON_ENCODED_OBS_KEYS}
        return super()._encode_obs_features(clean)

    def close(self):
        """Shut down the verifier's sim worker pool.

        The pool is a set of subprocesses, so it does not go away on garbage collection;
        the training workspace calls this in a finally block. Not done via __del__ --
        interpreter-shutdown ordering makes that unreliable for subprocess pools.
        """
        verifier = getattr(self, 'verifier', None)
        if verifier is not None and hasattr(verifier, 'close'):
            verifier.close()

    # predict_action / predict_action_best / search_candidates are all inherited
    # unchanged from DiffusionTransformerSearchPolicy.
