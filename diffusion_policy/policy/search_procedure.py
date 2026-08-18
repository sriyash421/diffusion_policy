"""The best-of-n search procedure, shared by every search policy.

The loop depends on exactly two things a subclass supplies -- ``predict_action`` (generate
one chunk, optionally conditioned on prior candidates) and ``_score_candidates`` (simulate
it) -- so the Gaussian and diffusion families run an identical procedure and differ only in
how a single candidate is produced.
"""
from typing import Dict, Optional
import contextlib

import torch

from diffusion_policy.common.selection_util import SELECTION_MODES, select_candidate


def _stack_subgoals(per_candidate):
    """[{k: (B, ...)}] (one dict per candidate) -> {k: (B, n, ...)}; None if empty."""
    if not per_candidate:
        return None
    return {k: torch.stack([d[k] for d in per_candidate], dim=1)
            for k in per_candidate[0]}


def _cat_subgoals(per_block):
    """[{k: (B, n_i, ...)}] -> {k: (B, sum n_i, ...)}; None if empty."""
    if not per_block:
        return None
    return {k: torch.cat([d[k] for d in per_block], dim=1) for k in per_block[0]}


class SearchProcedureMixin:
    """Requires of the host policy: ``predict_action``, ``normalizer``, ``verifier``,
    ``max_actions``, ``n_obs_steps``, ``n_action_steps``, ``selection``,
    ``selection_temperature``, ``corrupt_obs_features`` (see ObsCorruptionMixin), and
    ``_encode_obs_features``.
    """

    def _init_slot_weighting(self, slot_weight_decay=False):
        """Per-slot loss weighting is OFF by default.

        A float in (0, 1) enables it; 1.0 is rejected rather than silently accepted as
        "off", so a config that means uniform says so. The resolved weights are logged at
        construction -- under `final_pass` the deployed slot depends on the eval width, so
        the vector is the only honest record of what each slot was trained at.
        """
        if slot_weight_decay is False or slot_weight_decay is None:
            self.slot_weight_decay = None
            return
        decay = float(slot_weight_decay)
        if decay == 1.0:
            raise ValueError(
                'slot_weight_decay: 1.0 is a no-op; use False to mean uniform slots.')
        assert 0.0 < decay < 1.0, \
            f'slot_weight_decay must be False or in (0, 1), got {decay}'
        self.slot_weight_decay = decay
        w = self._slot_weights(torch.device('cpu'), torch.float32)
        if w is not None:
            print(f'{type(self).__name__}: slot_weight_decay={decay} -> per-slot weights '
                  f'[{", ".join(f"{x:.4f}" for x in w.tolist())}] (mean {w.mean():.6f})')

    def _init_selection(self, **kwargs):
        """Set the knobs predict_action_best reads. Call from the host's __init__."""
        self.selection = kwargs.get('selection', 'argmax') or 'argmax'
        assert self.selection in SELECTION_MODES, \
            f"selection must be one of {SELECTION_MODES}, got {self.selection!r}"
        # Temperature applies to the STANDARDIZED score, so one value means the same thing
        # across arms: T->0 reproduces argmax, T->inf a uniform pick among the n.
        self.selection_temperature = float(
            kwargs.get('selection_temperature', 1.0) or 1.0)
        assert self.selection_temperature > 0, 'selection_temperature must be > 0'

    @contextlib.contextmanager
    def _crop_scope(self):
        """No-op by default; policies that own image crops override it to pin one offset
        across the obs and every subgoal encoded inside the scope."""
        yield


    def _build_verifier(self, **kwargs):
        """Build this task's verifier. Abstract: there is no sensible default.

        Deliberately NOT defaulting to the maze verifier, which would put an `l2s` import
        on every subclass's path including the ones that never use it. See
        MazeDiffusionSearchPolicy / SearchPolicy (maze) and PushTDiffusionSearchPolicy.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement _build_verifier(**kwargs)')

    def _context_dim(self, obs_feature_dim: int, **kwargs) -> int:
        """Width of the per-candidate feedback in the search context.

        1 == the verifier scalar. Subclasses override to feed a richer signal (see
        PushTDiffusionSearchPolicy, which can feed the rollout state or the *encoded*
        subgoal observation instead of / along with the scalar -- hence obs_feature_dim).
        Called from __init__ to size the transformer's context embedding, so it must
        depend only on the config kwargs and the encoder width.
        """
        return 1

    def _score_candidates(self, verifier, obs_dict, action, want_subgoals: bool = False):
        """Evaluate one batch of candidates.

        Returns ``(context, score, subgoal)``:
          * ``context`` (B,) or (B, context_dim) -- the feedback fed back into the search
            context so the next candidate is conditioned on it.
          * ``score`` (B,) -- the scalar used to *rank* candidates (argmax at eval time).
          * ``subgoal`` -- dict of per-candidate debug tensors for logging, or None. Only
            populated when ``want_subgoals``; verifiers without a renderable outcome
            (e.g. the maze one) always return None.
        By default context and score are both the verifier value. Subclasses override to
        widen the context while keeping the scalar ranking signal.
        """
        value = verifier.get_value(obs_dict, action)
        return value, value, None

    def _normalize_context_actions(
            self, actions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        simulates them (``_verifier_inputs`` resets a sim from pixel coords), and the env
        runner executes them. But the model denoises a NORMALIZED trajectory against a
        normalized target, so feeding it raw actions puts the two halves of
        ``action_value_emb``'s input on scales orders of magnitude apart.

        Normalizing here -- at the model boundary, not in the search loop -- keeps one
        tensor from having to serve both boundaries. This mirrors the invariant
        OnlineSearchPolicy maintains: actions exist only in normalized space anywhere the
        model touches them, and are unnormalized strictly at the env/verifier boundary.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            obs_features: Optional[torch.Tensor] = None,
            return_subgoals: bool = False,
        ):
        """Generate n_actions candidates, each conditioned on the previous ones.

        Returns ``(actions, values)``, with ``scores`` appended when ``return_scores``
        and ``subgoals`` appended last when ``return_subgoals`` -- i.e. the tuple grows
        left-to-right: ``(actions, values[, scores][, subgoals])``.

        ``values`` is the search *context* feedback -- (B, n) for the scalar verifier
        value, (B, n, context_dim) for a wider context -- while ``scores`` is always the
        (B, n) scalar used to rank candidates. ``subgoals`` is a dict of stacked
        (B, n, ...) debug tensors (or None if the verifier has none); it is for logging
        only and is never fed back into the search.

        ``obs_features`` optionally supplies an already-encoded obs (see
        _encode_obs_features); otherwise it is encoded once here and shared by every
        candidate, since they all condition on the same observation.

        The crop scope is opened HERE rather than only in the callers, so the observation
        and every candidate's subgoal share one crop offset no matter who called -- this is
        the entry point TrainSearchOuterInnerWorkspace and the diagnostic scripts reach
        directly. It is reentrant, so the offline trainer's nested call is a no-op.
        """
        with self._crop_scope():
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            actions = None
            values = None
            scores = None
            subgoals = list()
            for _ in range(n_actions):
                new_action = self.predict_action(
                    obs_dict,
                    actions=actions,
                    values=values,
                    obs_features=obs_features,
                )['action_pred']
                new_value, new_score, new_subgoal = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)
                if actions is None:
                    actions = new_action.unsqueeze(1)
                    values = new_value.unsqueeze(1)
                    scores = new_score.unsqueeze(1)
                else:
                    actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                    values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
                    scores = torch.cat([scores, new_score.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    subgoals.append(new_subgoal)

        out = (actions, values)
        if return_scores:
            out = out + (scores,)
        if return_subgoals:
            out = out + (_stack_subgoals(subgoals),)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            return_subgoals: bool = False,
            obs_features: Optional[torch.Tensor] = None,
        ):
        """Search with a rolling context window; see search_candidates for the return shape.

        ``obs_features`` optionally supplies an already-encoded obs so a caller that needs
        the features for something else too (``predict_action_best`` in 'final_pass' mode
        draws one extra sample from them) pays for a single encoder pass rather than two.

        Opens its own (reentrant) crop scope. The rolling-window branch below encodes
        subgoals in a loop of its own rather than inside `search_candidates`, so wrapping
        only that method would leave every candidate past `max_actions` on an independent
        crop -- the exact split-crop defect, reappearing only at n > K.
        """
        with self._crop_scope():
            # encode once for the whole search, however many candidates it runs
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            if n_actions <= self.max_actions:
                return self.search_candidates(
                    obs_dict, verifier, n_actions, return_scores=return_scores,
                    obs_features=obs_features, return_subgoals=return_subgoals)

            # scores are always needed internally (the caller may only want values), but
            # subgoals are only rendered when actually asked for.
            head = self.search_candidates(
                obs_dict, verifier, self.max_actions, return_scores=True,
                obs_features=obs_features, return_subgoals=return_subgoals)
            actions, values, scores = head[0], head[1], head[2]
            subgoals = head[3] if return_subgoals else None
            all_actions = actions.clone()
            all_values = values.clone()
            all_scores = scores.clone()
            all_subgoals = [subgoals] if subgoals is not None else []

            action_history = actions[:, 1:]
            value_history = values[:, 1:]
            for _ in range(self.max_actions, n_actions):
                new_action = self.predict_action(
                    obs_dict,
                    actions=action_history,
                    values=value_history,
                    obs_features=obs_features,
                )['action_pred']
                new_value, new_score, new_subgoal = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)

                all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
                all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
                all_scores = torch.cat([all_scores, new_score.unsqueeze(1)], dim=1)
                action_history = torch.cat(
                    [action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
                value_history = torch.cat(
                    [value_history[:, 1:], new_value.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    # already (B, 1, ...) from the inner stack vs (B, ...) from the loop
                    all_subgoals.append({k: v.unsqueeze(1) for k, v in new_subgoal.items()})

        out = (all_actions, all_values)
        if return_scores:
            out = out + (all_scores,)
        if return_subgoals:
            out = out + (_cat_subgoals(all_subgoals),)
        return out

    @torch.inference_mode()
    def predict_action_best(
            self,
            obs_dict: Dict[str, torch.Tensor],
            n_actions: Optional[int] = None,
        ) -> Dict[str, torch.Tensor]:
        """Search readout, in the standard ``predict_action`` output format.

        Generates ``n_actions`` candidates via the sliding-window search
        (``predict_n_actions``), scores each with this policy's verifier, and returns one
        action chunk as ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon,
        Da), 'scores': (B, n)}`` -- the shape the env runners and MultiStepWrapper expect.
        (Tensors only: the runners push this dict straight through ``dict_apply``, so a
        string mode tag here would crash them. Read the mode off ``policy.selection``.)

        WHICH chunk depends on ``self.selection``:
          * 'argmax'     -- the argmax-verifier-value candidate. Best-of-n over an oracle.
          * 'final_pass' -- one MORE sample, conditioned on all n scored candidates, is
            drawn and returned. It is not simulated and not compared to anything, so the
            verifier scalar never touches selection; it reaches the model only as search
            context. ``scores`` still describes the n context candidates (so the caller can
            log the spread), but no longer describes the returned action.

        Cost at width n: 'argmax' is n samples + n sims, 'final_pass' is n+1 samples + n
        sims. Consumers that compare arms should compare at equal SAMPLES, not equal n --
        see the ``n_generations`` field eval_search_pusht.py records.
        """
        n = n_actions if n_actions is not None else self.max_actions
        # `scores` is the scalar verifier value in every search_context mode; `values` may
        # be a wider context (e.g. a subgoal state), which is not rankable.
        # The crop scope spans the whole search so the obs and every candidate's subgoal
        # share one offset, exactly as in training. (In eval mode that offset is the
        # deterministic center crop, so this is belt-and-braces rather than load-bearing.)
        with self._crop_scope():
            # encoded once and shared by the search AND (in 'final_pass') the extra sample
            obs_features = self._encode_obs_features(obs_dict)
            actions, values, scores = self.predict_n_actions(
                obs_dict, verifier=self.verifier, n_actions=n,
                return_scores=True, obs_features=obs_features)  # (B,n,H,Da), ctx, (B,n)

            if self.selection in ('argmax', 'softmax'):
                action_pred = select_candidate(                         # (B, H, Da)
                    actions, scores, self.selection, self.selection_temperature)
            else:
                # Condition on the last max_actions-1 candidates: that is the widest
                # context the model was ever trained at (the staircase memory mask tops out
                # at max_context_actions), so a longer one would index past cond_pos_emb.
                # `values`, not `scores` -- in the subgoal modes the context is the encoded
                # subgoal observation and the bare scalar would be the wrong width.
                keep = self.max_actions - 1
                action_pred = self.predict_action(
                    obs_dict,
                    actions=actions[:, -keep:],
                    values=values[:, -keep:],
                    obs_features=obs_features,
                )['action_pred']                                        # (B, H, Da)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],            # (B, n_action_steps, Da)
            'action_pred': action_pred,
            'scores': scores,
        }

