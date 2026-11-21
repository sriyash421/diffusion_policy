"""Match the existing LSTM DDPM-100 feature noise, preserving clean environment feedback."""
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dt_base import DTSearchPolicy

class NoisyDTSearchPolicy(DTSearchPolicy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.obs_noise_scheduler=DDPMScheduler(
            num_train_timesteps=100,beta_start=.001,beta_end=.02,
            beta_schedule='linear',prediction_type='epsilon')

    def _obs_features(self,obs_dict):
        # Raw observations remain untouched: base _step/compute_loss use them for
        # passability, positions, true BFS feedback, termination and training labels.
        feat=super()._obs_features(obs_dict)
        noise=torch.randn_like(feat)
        timesteps=torch.randint(0,100,(feat.shape[0],),device=feat.device).long()
        # No training-mode guard: identical corruption in expert teacher forcing,
        # generated contexts and evaluation. Stream stores each noisy feature once.
        return self.obs_noise_scheduler.add_noise(feat,noise,timesteps)
