import torch
from diffusion_policy.model.diffusion.ema_model import EMAModel


def test_ema_copies_running_statistics_and_discrete_buffers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))
            self.register_buffer('obs_feature_std', torch.ones(2))
            self.register_buffer('obs_feature_std_inited', torch.tensor(False))
            self.child = torch.nn.BatchNorm1d(2)

    online, averaged = Model(), Model()
    ema = EMAModel(averaged, min_value=0.5, max_value=0.5)
    ema.optimization_step = 2
    with torch.no_grad():
        online.weight.fill_(3)
        online.obs_feature_std.copy_(torch.tensor([0.003, 0.1]))
        online.obs_feature_std_inited.fill_(True)
        online.child.running_mean.fill_(4)
        online.child.num_batches_tracked.fill_(7)
    ema.step(online)
    torch.testing.assert_close(averaged.weight, torch.tensor([2.0]))
    torch.testing.assert_close(averaged.obs_feature_std, online.obs_feature_std)
    assert averaged.obs_feature_std_inited.item()
    torch.testing.assert_close(averaged.child.running_mean, online.child.running_mean)
    assert averaged.child.num_batches_tracked.item() == 7
