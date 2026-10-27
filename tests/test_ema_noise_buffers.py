import torch
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.search_procedure import SearchProcedureMixin


def test_ema_copies_noise_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer('obs_feature_std', torch.ones(2))
            self.register_buffer('obs_feature_std_inited', torch.tensor(False))
    online, average = Model(), Model()
    online.obs_feature_std.copy_(torch.tensor([0.02, 7.0]))
    online.obs_feature_std_inited.fill_(True)
    EMAModel(average).step(online)
    torch.testing.assert_close(average.obs_feature_std, online.obs_feature_std)
    assert average.obs_feature_std_inited.item()


def test_observation_rng_independent_of_batch_and_action_stream():
    a, b = SearchProcedureMixin(), SearchProcedureMixin()
    a.set_sample_seeds([101, 202])
    b.set_sample_seeds([202])
    for _ in range(3):
        x = a._obs_random((2, 2, 4), torch.float32, 'cpu')
        torch.randn(100)  # Unrelated global draws cannot perturb observation noise.
        y = b._obs_random((1, 2, 4), torch.float32, 'cpu')
        torch.testing.assert_close(x[1], y[0], rtol=0, atol=0)
        tx = a._obs_random((2,), torch.long, 'cpu', levels=(2, 601))
        ty = b._obs_random((1,), torch.long, 'cpu', levels=(2, 601))
        assert tx[1] == ty[0]
    # Observation corruption must not shift action diffusion draws.
    z = SearchProcedureMixin()
    z.set_sample_seeds([202])
    torch.testing.assert_close(b._init_noise((1, 16, 2), torch.float32, 'cpu'),
                               z._init_noise((1, 16, 2), torch.float32, 'cpu'), rtol=0, atol=0)
