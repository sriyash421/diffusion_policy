"""Optional noising of encoded observation features, shared by every policy that offers it.

One implementation for what used to be five byte-identical copies (search / online / maze,
plus the MLP and hybrid BC policies).
"""
import torch


class ObsCorruptionMixin:
    """Call `_init_corruption(corrupt_obs, corrupt_obs_eval)` from the host's __init__.

    The host must also own `self.obs_noise_scheduler`.
    """

    def _init_corruption(self, corrupt_obs, corrupt_obs_eval=None):
        """Corruption is OFF by default, train and eval alike.

        Turning it on FORCES the eval question to be answered in the config: whether
        rollouts also see corrupted observations. It used to be answered implicitly and
        inconsistently -- online guarded on self.training at its call site, maze / PushT /
        the BC policies did not, so those corrupted at eval without ever saying so.
        """
        self.corrupt_obs = bool(corrupt_obs)
        if self.corrupt_obs and corrupt_obs_eval is None:
            raise ValueError(
                f'{type(self).__name__}: corrupt_obs: True requires an explicit '
                f'corrupt_obs_eval (True|False) -- whether eval rollouts also see '
                f'corrupted observations.')
        self.corrupt_obs_eval = bool(corrupt_obs_eval)

    def corrupt_obs_features(self, obs_features):
        if not self.corrupt_obs:
            return obs_features
        if not self.training and not self.corrupt_obs_eval:
            return obs_features

        obs_noise = torch.randn_like(obs_features)
        bsz = obs_features.shape[0]
        timesteps = torch.randint(
            0,
            self.obs_noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=obs_features.device,
        ).long()
        return self.obs_noise_scheduler.add_noise(obs_features, obs_noise, timesteps)
