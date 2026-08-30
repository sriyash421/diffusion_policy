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

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.crop_scope import CropScopeMixin
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.policy.pusht_search_mixin import PushTSearchMixin
from diffusion_policy.policy.search_procedure import SearchProcedureMixin


# BASE ORDER IS LOAD-BEARING: CropScopeMixin must precede SearchProcedureMixin, which
# defines `_crop_scope` as a NO-OP for hosts that own no crop. With the other order that
# no-op wins, the scope never clears `_crop_offsets`, and every batch after the first silently
# reuses the first batch's crops -- augmentation that looks alive and is not.
# DiffusionTransformerSearchPolicy orders them the same way, for the same reason.
class PushTUNetSearchPolicy(PushTSearchMixin, CropScopeMixin, SearchProcedureMixin,
                            DiffusionUnetImagePolicy):
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
        # Popped BEFORE super(): DiffusionUnetImagePolicy forwards every unrecognised kwarg
        # to scheduler.step(), so leaving these in raises there.
        crop_shape = kwargs.pop('crop_shape', None)
        random_crop = kwargs.pop('random_crop', True)
        shape_meta = kwargs.get('shape_meta', args[0] if args else None)
        super().__init__(*args, **kwargs)
        # The same attribute every other search host sets. It used to be `_search_kwargs`
        # here alone, because DiffusionUnetImagePolicy owns `kwargs` for its scheduler.step()
        # arguments and the mixin read the search knobs off `self.kwargs` -- so this class
        # needed a private alias plus two method overrides to redirect the mixin to it, and
        # every external caller needed a getattr dance to find the right bag. The hosts now
        # name it `search_kwargs`, which collides with nothing.
        self.search_kwargs = search_kwargs
        # only the scalar value mode: the wider modes feed an encoded subgoal into the
        # model as context, and this policy has no context input to feed.
        mode = search_kwargs.get('search_context', 'value') or 'value'
        assert mode == 'value', \
            f"{type(self).__name__} takes no search context, so search_context must be " \
            f"'value'; got {mode!r}"
        self._init_selection(**search_kwargs)
        # There is no trained staircase to fall out of here (consumes_search_context is
        # False), so the rolling window never applies. `None` says that; the 1<<20 sentinel
        # this replaces said it by arithmetic, and read as a search width in every message
        # and log line that printed it.
        self.max_actions = None
        self._verifier = None
        # Same crop contract as the search transformer: one offset per sample, shared by the
        # obs window. crop_shape/random_crop come from the policy block, interpolating the
        # same ${crop_shape} the encoder reads, so the two cannot disagree.
        self._init_crop(shape_meta, crop_shape, random_crop)

    @property
    def verifier(self):
        """Built on first use, so training never spawns the 32-process sim pool."""
        if self._verifier is None:
            self._verifier = self._build_verifier(**self.search_kwargs)
        return self._verifier

    def _encode_images(self, this_nobs, batch_size):
        """One crop offset per SAMPLE, shared by the observation window's frames.

        Without this the encoder's own CropRandomizer draws an offset per IMAGE, so at train
        time the two obs frames of one sample are cropped differently while at eval both take
        the same centre crop -- a train/eval mismatch in how the two frames are registered
        against each other, and more scene coverage per sample than the search transformer
        gets. The ST arms have shared per-sample offsets via CropScopeMixin; this gives the
        BC baseline the same observation so the comparison varies the algorithm, not the input.
        """
        # The scope is LOAD-BEARING, not decoration. `_crop_offsets_for` caches the drawn
        # offsets in `self._crop_offsets` and only the outermost `_crop_scope` exit clears
        # them -- so calling it with no scope open would hand every subsequent batch of the
        # same size the FIRST batch's crops, and the augmentation would quietly die. The
        # scope is reentrant, so when the search loop already opened one (every candidate of
        # a decision sharing an observation) this nests harmlessly and that outer scope
        # still owns the offsets.
        with self._crop_scope():
            offsets = self._crop_offsets_for(batch_size, repeat=self.n_obs_steps)
            if offsets is None:
                return self.obs_encoder(this_nobs)
            return self.obs_encoder(this_nobs, crop_offsets=offsets)

    def _encode_obs_features(self, obs_dict):
        """Encode the obs window ONCE per decision -> the (B, To*D) global conditioning.

        Returned rather than recomputed because the search loop hands it back to every
        candidate: all n condition on the SAME observation, and the encoder is frozen, so the
        n-1 repeat forwards were bit-identical waste. That is 34.2M parameters run 16 times
        per control step at n=16 instead of once -- the single largest avoidable cost in a
        best-of-n sweep on this arm. (It used to return None, which is what made
        predict_action re-encode.)
        """
        To = self.n_obs_steps
        # Sliced to To BEFORE normalizing; see DiffusionTransformerSearchPolicy's copy for
        # why the two orders are bit-identical and why this one is cheaper.
        obs = dict_apply(self._select_obs(obs_dict), lambda x: x[:, :To, ...])
        nobs = self.normalizer.normalize(obs)
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        return self._encode_images(this_nobs, B).reshape(B, -1)

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: Optional[torch.Tensor] = None,
            values: Optional[torch.Tensor] = None,
            obs_features: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """One sample, drawn independently of any previous candidate.

        ``actions``/``values`` are the search loop's context and are deliberately discarded:
        this policy has no context input, so every call is an i.i.d. draw from the same
        conditional. That is what best-of-n means here.

        ``obs_features`` is NOT discarded -- it is the already-encoded conditioning from
        `_encode_obs_features`, passed straight through so the backbone runs once per
        decision instead of once per candidate.
        """
        return super().predict_action(obs_dict, global_cond=obs_features)
