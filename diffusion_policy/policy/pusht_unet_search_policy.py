"""Best-of-n over the diffusion-UNet BC policy, by i.i.d. sampling.

The UNet baseline has no search context and no verifier -- it is a plain
``BaseImagePolicy``. Best-of-n for it therefore means exactly what the name says: draw n
INDEPENDENT samples, score each with the task's verifier, execute the highest. There is no
learned selection anywhere in the loop, which is what makes it the right baseline for the
search arms' test-time budget.

This class exists only to expose that. It adds the PushT verifier and the shared search
procedure to ``DiffusionUnetImagePolicy`` and overrides ``predict_action`` to ACCEPT AND
IGNORE the search context the loop threads through -- ignoring it is the point, not an
oversight, and it is what makes the n draws independent.

The training path is untouched: ``compute_loss`` is the UNet's own, and the verifier is
built lazily, so a training run never spawns its worker pool.
"""
from typing import Dict, Optional

import torch

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.pusht_search_mixin import PushTSearchMixin
from diffusion_policy.policy.search_procedure import SearchProcedureMixin

# search_candidates takes its simple branch while n <= max_actions; there is no trained
# staircase here to fall out of, so this is just "never take the rolling-window path".
_UNBOUNDED = 1 << 20


class PushTUNetSearchPolicy(PushTSearchMixin, SearchProcedureMixin, DiffusionUnetImagePolicy):
    # THE ONLY ARM where this is False: predict_action below takes no context input and
    # drops `values` on the floor, so the verifier scalar never reaches the model. That is
    # what makes the cross-candidate verifier values (armTd) usable here and nowhere else --
    # they have no per-candidate value to put in a causal context. See
    # PushTSearchMixin._check_cross_candidate_value.
    consumes_search_context = False

    def __init__(self, *args, **kwargs):
        search_kwargs = {k: kwargs.pop(k) for k in (
            'selection', 'selection_temperature', 'selection_seed', 'search_context',
            'verifier_n_envs', 'verifier_legacy', 'verifier_use_async', 'verifier_steps',
            'verifier_value',
        ) if k in kwargs}
        super().__init__(*args, **kwargs)
        # NOT self.kwargs -- DiffusionUnetImagePolicy already owns that name for the
        # scheduler.step() keyword arguments, and overwriting it passes 'selection' into
        # DDIM's step() and raises.
        self._search_kwargs = search_kwargs
        # only the scalar value mode: the wider modes feed an encoded subgoal into the
        # model as context, and this policy has no context input to feed.
        mode = search_kwargs.get('search_context', 'value') or 'value'
        assert mode == 'value', \
            f"{type(self).__name__} takes no search context, so search_context must be " \
            f"'value'; got {mode!r}"
        self._init_selection(**search_kwargs)
        self.max_actions = _UNBOUNDED
        self._verifier = None

    @property
    def verifier(self):
        """Built on first use, so training never spawns the 32-process sim pool."""
        if self._verifier is None:
            self._verifier = self._build_verifier(**self._search_kwargs)
        return self._verifier

    def _search_context_mode(self, kwargs=None) -> str:
        """Always 'value'. Overridden because the mixin reads it off ``self.kwargs``,
        which on this class is the UNet's scheduler kwargs, not the search knobs."""
        return 'value'

    def _verifier_value_mode(self, kwargs=None) -> str:
        """Which verifier value to score with, from the SEARCH kwargs.

        Overridden for the same reason as _search_context_mode above: the mixin reads this
        off ``self.kwargs``, which on this class is the DDIM scheduler kwargs. Without the
        override `_normalize_value` -- which `_score_candidates` calls in every mode, so it
        is live here -- would silently fall back to the default and could disagree with the
        verifier this policy actually built.
        """
        return PushTSearchMixin._verifier_value_mode(self._search_kwargs)

    def _encode_obs_features(self, obs_dict):
        """No-op: predict_action below ignores the cached features and re-encodes.

        The search loop caches obs features to avoid re-running the ResNet per candidate,
        but that hook belongs to the transformer policies, which take the features
        directly. The UNet's predict_action takes the raw obs dict, so there is nothing to
        hand it -- the encoder does run once per candidate here.
        """
        return None

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            obs_features: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """One sample, drawn independently of any previous candidate.

        ``actions``/``values``/``obs_features`` are the search loop's context and are
        deliberately discarded: this policy has no context input, so every call is an
        i.i.d. draw from the same conditional. That is what best-of-n means here.
        """
        return super().predict_action(obs_dict)
