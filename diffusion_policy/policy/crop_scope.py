"""Policy-owned image cropping: one crop offset per SAMPLE, held for a whole forward pass.

Owned by the policy rather than the obs encoder's CropRandomizer because only the policy
knows WHICH IMAGES BELONG TO THE SAME SAMPLE: the offset drawn for the observation must
also be applied to every subgoal image that sample generates, or the model is asked to
compare two views of the same scene that are translated relative to each other at train
time and aligned at eval time. CropRandomizer cannot be handed an offset -- it draws
internally and sits inside an nn.Sequential, which cannot forward extra arguments.
"""
import contextlib

import torch


class CropScopeMixin:
    """Call `_init_crop(shape_meta, crop_shape, random_crop)` from the host's __init__."""

    def _init_crop(self, shape_meta, crop_shape, random_crop):
        self.crop_shape = None if crop_shape is None else tuple(crop_shape)
        self.random_crop = random_crop
        # uncropped image size, from shape_meta -- the offsets must be drawn against the
        # size the encoder actually receives
        self._crop_input_hw = (96, 96)
        for attr in shape_meta['obs'].values():
            if attr.get('type', 'low_dim') == 'rgb':
                self._crop_input_hw = tuple(attr['shape'][1:])
                break
        # Offsets come from a dedicated generator seeded from (seed, global_step), so the
        # crop is a pure function of those two: identical across restarts and machines, with
        # no RNG state to checkpoint, and no longer interleaved with the diffusion-noise
        # stream on the global RNG (which is what made it irreproducible after a resume).
        self._crop_generator = torch.Generator(device='cpu')
        self._crop_seed = 0
        self._crop_step = 0
        self._crop_offsets = None
        # `_crop_scope` is reentrant; only the outermost entry resets the offsets, so a
        # search entry point can guarantee a shared offset without disturbing a caller
        # that already opened one. See its docstring.
        self._crop_depth = 0

    def set_crop_step(self, seed: int, step: int):
        """Fix the (seed, step) the next training crop offsets are derived from.

        Called once per optimizer step by the workspace. Eval never needs it: outside
        train mode the crop is deterministic (center), so no offsets are drawn.
        """
        self._crop_seed = int(seed)
        self._crop_step = int(step)

    def _draw_crop_offsets(self, batch_size: int, height: int, width: int):
        """(B, 2) top-left crop offsets, one per SAMPLE.

        Valid range is [0, H-CH-1] x [0, W-CW-1], matching crop_image_from_indices'
        assertions and sample_random_image_crops' own bound.
        """
        ch, cw = self.crop_shape
        if not (self.training and self.random_crop):
            # center crop -- exactly CropRandomizer.forward_in's eval behaviour
            centre = torch.tensor([(height - ch) // 2, (width - cw) // 2],
                                  dtype=torch.long)
            return centre.unsqueeze(0).expand(batch_size, 2).clone()
        # Deterministic in (seed, step): two calls at the same optimizer step produce the
        # SAME crop, which is what lets the observation's offset be reused for the subgoals
        # generated from it, and what makes a resumed run reproduce an uninterrupted one.
        self._crop_generator.manual_seed(
            (self._crop_seed * 1_000_003 + self._crop_step) % (2 ** 31 - 1))
        dy = torch.randint(0, height - ch, (batch_size,), generator=self._crop_generator)
        dx = torch.randint(0, width - cw, (batch_size,), generator=self._crop_generator)
        return torch.stack([dy, dx], dim=-1)

    def _crop_offsets_for(self, batch_size: int, repeat: int = 1):
        """Per-image crop offsets for a flattened (B*repeat, C, H, W) batch, or None.

        One offset is drawn per SAMPLE and cached for the whole forward pass, then repeated
        across that sample's `repeat` images (the obs window's timesteps). Every later call
        inside the same `_crop_scope` -- notably each candidate's subgoal -- reuses the very
        same per-sample offsets, which is the point: the observation and the subgoals
        predicted from it stay spatially registered, exactly as they are at eval where both
        are center-cropped.
        """
        if self.crop_shape is None:
            return None
        offsets = self._crop_offsets
        if offsets is None or offsets.shape[0] != batch_size:
            offsets = self._draw_crop_offsets(batch_size, *self._crop_input_hw)
            self._crop_offsets = offsets
        if repeat > 1:
            offsets = offsets.repeat_interleave(repeat, dim=0)
        return offsets

    @contextlib.contextmanager
    def _crop_scope(self):
        """Hold one set of crop offsets for the duration of one forward pass.

        The obs encode and every subgoal encode inside the scope reuse the same per-sample
        offsets; the outermost exit clears them so they never leak into the next batch,
        whose batch size may differ.

        REENTRANT, and that is what makes the guarantee hold for every caller. The scope
        used to be opened only by `compute_loss` and `predict_action_best`, so the fix was
        a property of those two paths rather than of the search itself -- and
        `generate_search_context` / `search_candidates` are also called DIRECTLY, by
        TrainSearchOuterInnerWorkspace (which regenerates the context every outer pass) and
        by the diagnostic scripts. Those calls encoded the observation and each candidate's
        subgoal under independent per-image crops, i.e. exactly the split-crop defect the
        policy-level offset was introduced to remove (AUDIT.md P1-1), for any mode whose
        context contains a subgoal image.

        Every search entry point now opens a scope of its own. Nesting is a no-op: only
        depth 0 resets, so `compute_loss` -> `generate_search_context` -> `search_candidates`
        still shares the single offset set drawn at the top, and the offline trainer's
        behaviour is bit-for-bit unchanged.
        """
        outermost = self._crop_depth == 0
        if outermost:
            self._crop_offsets = None
        self._crop_depth += 1
        try:
            yield
        finally:
            self._crop_depth -= 1
            if self._crop_depth == 0:
                self._crop_offsets = None

