"""The paired measurement must agree with each policy's native objective."""
from unittest.mock import patch
import unittest

import hydra
import torch

from scripts.run_pusht_capacity import configs
from scripts.diagnose_pusht_capacity import predict_fixed_noise
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


def check_fixed_noise_matches_native_loss(model_index):
    _, pairs = configs('blq137_holdout50', 42)
    cfg = pairs[model_index][1]
    cfg.policy.obs_encoder.rgb_model.weights = None
    torch.manual_seed(123)
    policy = hydra.utils.instantiate(cfg.policy).eval()
    normalizer = LinearNormalizer()
    normalizer.fit({'action': torch.rand(64, 2) * 512})
    normalizer['image'] = get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    batch = {'action': torch.rand(2, 16, 2) * 512,
             'obs': {'image': torch.rand(2, 2, 3, 96, 96)}}
    noise = torch.randn(2, 16, 2)
    times = torch.tensor([10, 60])
    try:
        with torch.no_grad():
            measured = (predict_fixed_noise(policy, batch, noise, times) - noise).square().mean()
            is_st = not isinstance(policy.model, ConditionalUnet1D)
            with patch('torch.randn', return_value=noise), \
                 patch('torch.randn_like', return_value=noise.unsqueeze(1) if is_st else noise), \
                 patch('torch.randint', return_value=times.unsqueeze(1) if is_st else times):
                native = policy.compute_loss(batch)
        torch.testing.assert_close(measured, native, rtol=1e-5, atol=1e-6)
    finally:
        policy.close()


class TestCapacityDiagnosticLoss(unittest.TestCase):
    def test_unet(self):
        check_fixed_noise_matches_native_loss(-2)

    def test_st(self):
        check_fixed_noise_matches_native_loss(-1)
