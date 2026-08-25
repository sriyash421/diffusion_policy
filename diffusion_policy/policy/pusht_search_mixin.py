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

   * ``'value'``   (default): the verifier scalar, whichever of ``VALUE_FNS`` the run
     selected via ``verifier_value`` (see pusht_verifier). This is the original behaviour.
   * ``'subgoal'`` : the **rendered subgoal observation** the chunk lands on -- image plus
     the low_dim keys, pushed through this policy's own obs encoder -- i.e. the candidate
     reports back "here is what the world looks like afterwards", in the same feature
     space and of the same width as the obs tokens.
   * ``'subgoal_value'`` : the encoded subgoal observation plus the scalar.

   Both come out of the *same* simulated rollout, so the wider contexts cost no extra sim
   steps (the subgoal modes add one render + one encoder pass per candidate). Candidate
   *ranking* uses the scalar verifier value in every mode -- but
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
from diffusion_policy.env.pusht.pusht_verifier import (
    PushTVerifier, VALUE_FNS, DEFAULT_VALUE_FN, VERIFIER_VALUES,
    CROSS_CANDIDATE_VALUES, base_value_fn, value_terms_from_state,
    T_GOAL_SPREAD, ARM_T_SPREAD, ARM_TN_CONTEXT_SCALE)
from diffusion_policy.env.pusht.feedback_util import GOAL_KEYPOINTS, N_KEYPOINTS

# obs keys that ride in the obs dict but must never reach the encoder/normalizer
_NON_ENCODED_OBS_KEYS = ('attention_mask',)

# obs keys the verifier reads to reset the sim: agent_pos plus feedback, from which the
# block pose is reconstructed exactly (pusht_verifier.block_pose_from_feedback).
_VERIFIER_OBS_KEYS = ('agent_pos', 'feedback')

# per-candidate search-context feedback modes. 'subgoal' is the reached OBSERVATION --
# image plus the low_dim keys, encoded through the policy's own obs encoder -- not a raw
# state vector; the raw-state modes were removed because no arm ever used them.
SEARCH_CONTEXTS = ('value', 'subgoal', 'subgoal_value')

# modes that need the sim to render the reached observation
_RENDER_CONTEXTS = ('subgoal', 'subgoal_value')


class PushTSearchMixin:
    """Everything PushT-specific about the search: the sim verifier, what feedback the
    context carries, and how a candidate is scored. Shared by the diffusion and Gaussian
    arms, which differ only in how one candidate is generated.

    Requires of the host policy: ``normalizer``, ``obs_encoder``, ``n_obs_steps``,
    ``n_action_steps``, ``kwargs``, and ``_encode_obs`` (normalized obs dict -> features).
    """

    # Does this policy feed the verifier scalar back into the MODEL as search context?
    # True for the ST and Gaussian arms; PushTUNetSearchPolicy overrides it to False.
    #
    # It cannot be inferred from `search_context`: the UNet arm declares 'value' like the
    # others and then discards it (see its predict_action docstring). This attribute is the
    # only honest discriminator, and it gates the cross-candidate values, whose statistic
    # over all n candidates does not exist at the time a causal context is built.
    consumes_search_context = True

    def _check_cross_candidate_value(self, mode):
        """A cross-candidate value cannot coexist with a consumed search context.

        The context is CAUSAL: search_candidates generates candidate k conditioned on
        `values[:, :k]`, so when candidate k is scored the later candidates do not exist
        yet and no statistic over all n is available to put in its context. Approximating
        it with the base value would condition the model on a different ranking than the
        selector applies -- the exact failure _normalize_value's docstring warns about.

        So this raises rather than silently approximating. To relax it later, give the
        model a CAUSAL context -- z-scored over `values[:, :k]` rather than all n -- and
        gate on that instead.
        """
        if mode in CROSS_CANDIDATE_VALUES and self.consumes_search_context:
            raise ValueError(
                f'{type(self).__name__} feeds the verifier scalar back into the model as '
                f'search context, and that context is causal: candidate k is conditioned '
                f'on candidates < k, so a statistic over all n candidates does not exist '
                f'when k is scored. {mode!r} is therefore selection-only for now. Rank '
                f'with {base_value_fn(mode)!r}, or give this policy a prefix-standardized '
                f'context first.')

    def _fuses_scores(self) -> bool:
        return self._verifier_value_mode(self.kwargs) in CROSS_CANDIDATE_VALUES

    def _fuse_scores(self, scores, terms):
        """Re-rank the whole candidate set under a cross-candidate value. (B,n),(B,n,K)->(B,n)."""
        mode = self._verifier_value_mode(self.kwargs)
        # re-checked here, not only at construction: eval_search_pusht.py's
        # --verifier-value override swaps the value on an ALREADY-BUILT policy.
        self._check_cross_candidate_value(mode)
        return CROSS_CANDIDATE_VALUES[mode](terms)

    def _build_verifier(self, **kwargs):
        self._check_cross_candidate_value(self._verifier_value_mode(kwargs))
        return PushTVerifier(
            n_envs=kwargs.get('verifier_n_envs', 32),
            legacy=kwargs.get('verifier_legacy', False),
            use_async=kwargs.get('verifier_use_async', True),
            verifier_steps=kwargs.get('verifier_steps', None),
            value_fn=self._verifier_value_mode(kwargs),
        )

    @staticmethod
    def _verifier_value_mode(kwargs) -> str:
        """Which value the verifier scores with -- a key of ``VALUE_FNS``.

        Read from kwargs rather than off the built verifier because ``_normalize_value``
        needs it too, and on the UNet arm the verifier is built lazily (so asking it would
        fork the sim pool). The default is the pre-cutover value; see
        ``pusht_verifier.DEFAULT_VALUE_FN`` for why that direction.
        """
        mode = kwargs.get('verifier_value', DEFAULT_VALUE_FN) or DEFAULT_VALUE_FN
        assert mode in VERIFIER_VALUES, \
            f'verifier_value must be one of {sorted(VERIFIER_VALUES)}, got {mode!r}'
        return mode

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

    def _normalize_value(self, state: torch.Tensor) -> torch.Tensor:
        """The verifier scalar rescaled onto the fitted feedback scale -> (B,).

        MIRRORS ``verifier.value_fn`` and must branch with it -- the two are the same
        scalar at two scales, and if they disagree the model is conditioned on a different
        ranking than the selector applies.

        Under ``armTn`` the per-term normalization is already done, so the fitted PIXEL
        scale does not apply; that branch divides by ``ARM_TN_CONTEXT_SCALE`` instead to
        reach O(1). The other two are raw pixels -- ~0 to -800 (``armT``), ~0 to -300 (``t_goal``) -- which would sit
        orders of magnitude above the normalized state / subgoal embedding they are
        concatenated with, dominating ``action_value_emb``'s input, so those are rescaled.

        This RECOMPUTES the verifier's value in torch from the reached state rather than
        reading the scalar, so it must mirror ``PushTVerifier.rollout`` exactly -- which it
        can, because both terms are functions of ``[agent_pos, feedback]`` alone
        (``achieved_kp = GOAL_KEYPOINTS - feedback``, an exact identity; no pose fit).

        ONE shared scalar factor rescales the whole sum, deliberately. Scaling the two
        terms by different factors (e.g. the `feedback` scale for one and the `agent_pos`
        scale for the other) would reweight them against each other, so this context scalar
        would no longer be a monotone function of the raw scalar `argmax` ranks on -- the
        model and the selector would disagree about which candidate is better. With one
        factor ``context == raw * s``: ordering preserved, still exactly 0 at "T on goal,
        arm at the T's centre", still O(1).

        The factor is the MEAN of the fitted `feedback` scale, so the fitted statistics
        still do the rescaling (no magic constant). Its *offset* stays omitted: the value's
        defining property is that it is 0 at the goal, and an offset would destroy that.

        Only the CONTEXT copy is rescaled -- the raw scalar still ranks candidates, so
        predict_action_best and the train_action_value* metrics are unchanged.
        """
        agent_dim = PushTVerifier.AGENT_DIM
        feedback = state[..., agent_dim:]              # (B, 2*N_KEYPOINTS), raw pixels
        scale = self.normalizer['feedback'].params_dict['scale'].to(feedback)

        base = base_value_fn(self._verifier_value_mode(self.kwargs))
        if base == 't_goal':
            # THE PRE-2026-08-19 BODY, VERBATIM. Not re-derived in the armT form below with
            # the arm term dropped: a checkpoint from that era was trained on the context
            # this exact expression produced (per-dim scale applied BEFORE the norm, which
            # warps each keypoint's contribution by its own coordinate range), so anything
            # that changes the number -- however defensibly -- makes the eval unfaithful.
            kp = (feedback * scale).reshape(*feedback.shape[:-1], N_KEYPOINTS, 2)
            return -kp.norm(dim=-1).mean(dim=-1)       # (B,)

        # one shared torch mirror of the two numpy distance fns; also used by
        # _score_candidates to build the per-candidate `terms` a cross-candidate value fuses
        t_goal, arm_t = value_terms_from_state(state).unbind(dim=-1)  # (B,), (B,) px

        if base == 'armTn':
            # Per-term normalization is already done by value_arm_t_norm, so the fitted
            # feedback scale (a PIXEL scale) does not apply here -- but spread-normalized
            # units are not O(1) either: the value runs to about -32. Divide by the measured
            # dataset mean instead, which lands the context copy at ~-1 and, being a positive
            # constant, leaves the ranking identical to the raw scalar.
            return -(t_goal / T_GOAL_SPREAD
                     + arm_t / ARM_T_SPREAD) / ARM_TN_CONTEXT_SCALE  # (B,)

        # 'armT': the raw pixel sum, rescaled by one shared fitted factor.
        # Asserted, not left as a bare fallthrough: without this a NEWLY added value would
        # silently inherit the RETIRED armT body just by not matching a branch above.
        assert base == 'armT', \
            f'_normalize_value has no branch for {base!r}; add one rather than letting it ' \
            f'fall through to the retired armT rescale.'
        return -(t_goal + arm_t) * scale.mean()        # (B,)

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
        else:
            embedding = self._encode_subgoal(image, state)
            context = embedding if mode == 'subgoal' else \
                torch.cat([embedding, nvalue.unsqueeze(-1)], dim=-1)

        subgoal = None
        if want_subgoals:
            # logging only -- never fed back into the search
            subgoal = {'image': image, 'value': value}

        # The undecomposed components, for a CROSS-CANDIDATE value to re-weight once all n
        # candidates exist (SearchProcedureMixin._fuse_scores). Derived from `state`, which
        # rollout already sliced to the real rows, so the sim pool's tail-chunk padding can
        # never reach a statistic taken across candidates. Costs ~20 flops per candidate.
        terms = value_terms_from_state(state)        # (B, 2) raw px
        return context, value, subgoal, terms

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
