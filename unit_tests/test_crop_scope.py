"""Exercise CropScopeMixin: the crop contract both PushT arms now share.

    python unit_tests/test_crop_scope.py
    pytest unit_tests/test_crop_scope.py

Runs on a stub carrying nothing but `training` and a shape_meta, so it needs no model,
dataset or GPU.

WHAT THE CONTRACT IS, and why each half matters:

  * ONE OFFSET PER SAMPLE, repeated across the observation window. The two obs frames of a
    sample must be cropped identically or they are not registered against each other -- and
    at eval they always are (centre), so a per-image offset is a train/eval mismatch.
  * OFFSETS DO NOT LEAK ACROSS BATCHES. They are cached for the span of a `_crop_scope` and
    cleared on the outermost exit. Drawing with no scope open caches them forever, and every
    later batch of the same size silently reuses the first batch's crops -- augmentation that
    looks alive and is not.
  * EVAL IS THE CENTRE CROP, deterministically, matching CropRandomizer's own eval branch.
  * `train_crops()` asks for a training crop without asking for training MODE, which is what
    lets a trainer pre-encode a pool under eval() (dropout off) and still get the crop the
    inner updates will train on.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion_policy.policy.crop_scope import CropScopeMixin  # noqa: E402

SHAPE_META = {'obs': {'image': {'shape': [3, 96, 96], 'type': 'rgb'}}}
H = W = 96
CH = CW = 72
CENTRE = (H - CH) // 2


class _Stub(CropScopeMixin):
    def __init__(self, training=True, crop_shape=(CH, CW), random_crop=True):
        self.training = training
        self._init_crop(SHAPE_META, crop_shape, random_crop)


def test_one_offset_per_sample_repeated_across_the_window():
    s = _Stub()
    B, To = 5, 2
    with s._crop_scope():
        off = s._crop_offsets_for(B, repeat=To)
    assert off.shape == (B * To, 2), off.shape
    for i in range(B):
        assert torch.equal(off[i * To], off[i * To + 1]), \
            f'sample {i} got different offsets for its two frames: {off[i*To:i*To+2]}'
    # and different samples must not all collapse onto one offset
    assert len({tuple(off[i * To].tolist()) for i in range(B)}) > 1


def test_offsets_are_shared_inside_a_scope_and_cleared_on_exit():
    """The obs and every subgoal predicted from it share one crop; the next batch does not."""
    s = _Stub()
    with s._crop_scope():
        a = s._crop_offsets_for(4)
        b = s._crop_offsets_for(4)          # a subgoal encode, later in the same decision
        assert torch.equal(a, b), 'offsets must be reused inside one scope'
    assert s._crop_offsets is None, 'the outermost exit must clear the cache'


def test_scope_is_reentrant():
    s = _Stub()
    with s._crop_scope():
        outer = s._crop_offsets_for(4)
        with s._crop_scope():
            assert torch.equal(s._crop_offsets_for(4), outer)
        assert s._crop_offsets is not None, 'an inner exit must NOT clear the outer scope'
    assert s._crop_offsets is None


def test_no_leak_across_batches_when_the_step_advances():
    """The bug this test exists for: same batch size, new step -> new crops."""
    s = _Stub()
    s.set_crop_step(42, 0)
    with s._crop_scope():
        first = s._crop_offsets_for(8)
    s.set_crop_step(42, 1)
    with s._crop_scope():
        second = s._crop_offsets_for(8)
    assert not torch.equal(first, second), \
        'offsets did not change when the step advanced -- augmentation is dead'


def test_offsets_are_a_pure_function_of_seed_and_step():
    """Identical across restarts and machines: nothing to checkpoint."""
    a, b = _Stub(), _Stub()
    a.set_crop_step(7, 123)
    b.set_crop_step(7, 123)
    with a._crop_scope(), b._crop_scope():
        assert torch.equal(a._crop_offsets_for(6), b._crop_offsets_for(6))
    c = _Stub()
    c.set_crop_step(8, 123)
    with a._crop_scope(), c._crop_scope():
        assert not torch.equal(a._crop_offsets_for(6), c._crop_offsets_for(6))


def test_eval_returns_none_meaning_the_centre_crop():
    """None is not "no crop" -- it means the DETERMINISTIC CENTRE crop.

    `_crop_offsets_for` deliberately returns None outside training so CropRandomizer takes
    its own `ttf.center_crop` slice. Handing it explicit centre offsets gives the identical
    pixels, but by the forced-offset route, whose bounds assertions `.item()` device tensors
    and stall the pipeline -- ~4(n+1) syncs per control step in subgoal mode, for a crop that
    never varies. The offsets it declines to build are still the centre:
    """
    s = _Stub(training=False)
    with s._crop_scope():
        assert s._crop_offsets_for(4, repeat=2) is None
    assert bool((s._draw_crop_offsets(4, H, W) == CENTRE).all())


def test_random_crop_off_is_the_centre_crop_even_in_train():
    s = _Stub(training=True, random_crop=False)
    with s._crop_scope():
        assert s._crop_offsets_for(4) is None
    assert bool((s._draw_crop_offsets(4, H, W) == CENTRE).all())


def test_centre_shortcut_only_when_it_matches_center_crop():
    """The None shortcut is guarded: it is only valid where floor and round agree, i.e.
    where our centre is the same pixel as torchvision's. At an odd gap they differ, and the
    offsets must then be materialised rather than delegated."""
    even = _Stub(training=False, crop_shape=(72, 72))     # gap 24 -> 12 either way
    odd = _Stub(training=False, crop_shape=(73, 73))      # gap 23 -> 11 vs 12
    assert even._centre_is_default
    assert not odd._centre_is_default
    with odd._crop_scope():
        assert odd._crop_offsets_for(4) is not None


def test_train_crops_gives_training_offsets_without_training_mode():
    """What lets a pooled trainer pre-encode under eval() and still get the trained crop."""
    s = _Stub(training=False)
    s.set_crop_step(1, 1)
    with s._crop_scope():
        assert s._crop_offsets_for(4) is None, 'eval should delegate to the centre crop'
    with s.train_crops(), s._crop_scope():
        randomised = s._crop_offsets_for(4)
    assert randomised is not None, 'train_crops must materialise offsets'
    assert not bool((randomised == CENTRE).all()), 'train_crops did not randomise'
    assert not s.training, 'train_crops must not flip the module into training mode'
    # and it must restore
    with s._crop_scope():
        assert s._crop_offsets_for(4) is None


def test_offsets_are_in_range():
    s = _Stub()
    for step in range(20):
        s.set_crop_step(0, step)
        with s._crop_scope():
            off = s._crop_offsets_for(16)
        assert int(off.min()) >= 0
        assert int(off[:, 0].max()) < H - CH
        assert int(off[:, 1].max()) < W - CW


def test_no_crop_shape_means_no_offsets():
    s = _Stub(crop_shape=None)
    with s._crop_scope():
        assert s._crop_offsets_for(4) is None


def test_policies_resolve_the_real_crop_scope():
    """MRO guard. SearchProcedureMixin defines `_crop_scope` as a no-op for hosts that own no
    crop, so a search policy that lists it BEFORE CropScopeMixin silently gets the no-op --
    the scope never clears, and every batch after the first reuses the first batch's offsets.
    Cheap to assert, and invisible until you look at the drawn offsets."""
    from diffusion_policy.policy.crop_scope import CropScopeMixin as C
    from diffusion_policy.policy.diffusion_transformer_search_policy import (
        DiffusionTransformerSearchPolicy)
    from diffusion_policy.policy.pusht_unet_search_policy import PushTUNetSearchPolicy
    for cls in (DiffusionTransformerSearchPolicy, PushTUNetSearchPolicy):
        assert cls._crop_scope is C._crop_scope, (
            f'{cls.__name__}._crop_scope resolves to {cls._crop_scope.__qualname__}, not '
            f"CropScopeMixin's -- check the base-class order")


if __name__ == '__main__':
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); n += 1
            print(f'  ok  {name}')
    print(f'\n{n} tests passed')
