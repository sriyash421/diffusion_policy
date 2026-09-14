"""Feature extractors that can corrupt the encoded observation, and the Q head.

CORRUPTION lives in a features extractor, passed through SB3's own
`features_extractor_class` / `features_extractor_kwargs`. Corrupting the encoded observation IS a
feature-extractor concern, and going through the documented extension point means SB3 builds it
itself, at the right time: nothing is mutated after `__init__`, so the optimizer is built over the
final module tree and the policies stay STOCK apart from the Q head.

The noise is the repo's: a DDPM forward marginal under the scheduler of
`config/train_pusht_diffusion_search.yaml:339`, matching
`ObsCorruptionMixin.corrupt_obs_features` operation for operation (`randn_like` BEFORE `randint`,
which is what makes them bit-identical under a shared seed -- see the test).

TWO THINGS THAT DIFFER FROM ST'S FLAT ARM, both deliberate and both measured:

  * The noise is SCALED BY A RUNNING FEATURE STD, as ST's per-slot ladder does and its flat arm
    does not. Unscaled, `sqrt(alpha_bar_t)` is not an SNR -- it rides on whatever magnitude the
    encoder's output happens to have, which for an end-to-end ResNet drifts by orders of
    magnitude during training (30 -> 9188 across one ST run's checkpoints). Scaled, `t` denotes a
    fixed SNR, stable across training and comparable across arms.
  * Timesteps are drawn from `U[0, t_max)` with t_max=200, not `U[0, 1000)`. Chosen by looking:
    scripts/render_noise_levels.py decodes the corruption through the SD VAE, and the block's
    ORIENTATION stops being readable around t=100. U[0,1000) put 90% of draws past that, i.e. the
    observation was absent rather than degraded.
"""



import torch as th
import torch.nn as nn
import torchvision
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.policies import (
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
)
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.vision.model_getter import get_resnet

DEFAULT_T_MAX = 200
ST_CROP = 76        # crop_shape in pusht_base.yaml


def make_obs_noise_scheduler():
    """The obs_noise_scheduler of train_pusht_diffusion_search.yaml, verbatim."""
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                         beta_schedule="linear", prediction_type="epsilon")


class STResNetExtractor(BaseFeaturesExtractor):
    """ST's image encoder: ResNet18 / IMAGENET1K_V1 / GroupNorm / 76px crop, trained end to end.

    Matches `obs_encoder` in config/pusht_base.yaml rather than SB3's NatureCNN, so the image arm
    learns from the same representation the diffusion-policy arms do. Not the frozen SD VAE: that
    was reverted on 2026-08-30 on measured evidence that freezing broke the policy, while the same
    encoder trained end to end recovered most of the gap.
    """

    def __init__(self, observation_space, crop=ST_CROP, use_group_norm=True):
        image_space = observation_space["image"]
        super().__init__(observation_space, features_dim=512 + observation_space["agent_pos"].shape[0])
        resnet = get_resnet("resnet18", weights="IMAGENET1K_V1")
        if use_group_norm:
            # batch statistics are wrong at these batch sizes; standard for diffusion policy
            resnet = replace_submodules(
                root_module=resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
        self.resnet = resnet
        self.train_crop = CropRandomizer(input_shape=image_space.shape, crop_height=crop,
                                         crop_width=crop, num_crops=1, pos_enc=False)
        self.eval_crop = torchvision.transforms.CenterCrop(size=(crop, crop))

    def forward(self, obs):
        # SB3 has already divided the uint8 image by 255; ST's normalizer then maps [0,1] -> [-1,1]
        img = obs["image"] * 2.0 - 1.0
        img = self.train_crop(img) if self.training else self.eval_crop(img)
        return th.cat([self.resnet(img), obs["agent_pos"]], dim=-1)


class CorruptingExtractor(BaseFeaturesExtractor):
    """An inner features extractor with the observation corruption on its output."""

    def __init__(self, observation_space, inner_class=FlattenExtractor, inner_kwargs=None,
                 enabled=True, t_max=DEFAULT_T_MAX, std_momentum=0.99):
        inner = inner_class(observation_space, **(inner_kwargs or {}))
        super().__init__(observation_space, features_dim=inner.features_dim)
        self.inner = inner
        self.enabled = bool(enabled)
        self.t_max = int(t_max)
        self.std_momentum = float(std_momentum)
        self.scheduler = make_obs_noise_scheduler()
        # ST's obs_feature_std, same EMA. Registered as buffers so it travels with the checkpoint:
        # a policy reloaded mid-training must keep the scale its noise was calibrated against.
        self.register_buffer("feature_std", th.ones(inner.features_dim))
        self.register_buffer("feature_std_inited", th.zeros(1, dtype=th.bool))

    @th.no_grad()
    def _update_feature_std(self, features):
        std = features.reshape(-1, features.shape[-1]).std(dim=0).clamp_min(1e-6)
        if not bool(self.feature_std_inited):
            self.feature_std.copy_(std)
            self.feature_std_inited.fill_(True)
        else:
            self.feature_std.mul_(self.std_momentum).add_(std, alpha=1.0 - self.std_momentum)

    def forward(self, obs):
        features = self.inner(obs)
        if not self.enabled:
            return features
        if self.training:
            self._update_feature_std(features)
        # randn BEFORE randint: the order ObsCorruptionMixin uses, and what makes the two
        # bit-identical under a shared seed
        noise = th.randn_like(features) * self.feature_std
        t = th.randint(0, self.t_max, (features.shape[0],), device=features.device).long()
        return self.scheduler.add_noise(features, noise, t)


class _QHeadMixin:
    """Q(s, a), trained alongside PPO but never inside its objective.

    Built AFTER `super().__init__()`, which is what keeps `q_net` out of `self.optimizer`: the
    policy optimizer was already constructed over the parameters that existed then. The Q head
    carries its own, so it provably cannot move the policy. See tests.

    Two subclasses supply the critic latent, because that is the only thing the architectures
    disagree about: under the LSTM it is Q(history, a), under frame stacking Q(stack, a).
    """

    def _build_q_head(self, q_net_arch=(128, 128), q_lr=1e-3):
        dim = self.mlp_extractor.latent_dim_vf + get_action_dim(self.action_space)
        layers, last = [], dim
        for hidden in q_net_arch:
            layers += [nn.Linear(last, hidden), nn.Tanh()]
            last = hidden
        self.q_net = nn.Sequential(*layers, nn.Linear(last, 1))
        self.q_optimizer = th.optim.Adam(self.q_net.parameters(), lr=q_lr)

    def q_values(self, obs, actions, **latent_kwargs):
        """Mirrors predict_values, then conditions on the action. Upstream is detached."""
        with th.no_grad():
            latent = self._q_latent(obs, **latent_kwargs)
        return self.q_net(th.cat([latent, actions], dim=-1)).flatten()

    def q_batch(self, data):
        """(prediction, target) for one rollout-buffer batch, so the callback needs no `if`."""
        raise NotImplementedError


class _RecurrentQHeadMixin(_QHeadMixin):
    def _q_latent(self, obs, lstm_states, episode_starts):
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        if self.lstm_critic is not None:
            latent, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent, _ = self._process_sequence(features, lstm_states, episode_starts, self.lstm_actor)
        else:
            latent = self.critic(features)
        return self.mlp_extractor.forward_critic(latent)

    def q_batch(self, data):
        # the recurrent buffer pads sequences to a common length; `mask` is which entries are real
        mask = data.mask > 1e-8
        q = self.q_values(data.observations, data.actions,
                          lstm_states=data.lstm_states.vf, episode_starts=data.episode_starts)
        return q[mask], data.returns[mask]


class _FeedForwardQHeadMixin(_QHeadMixin):
    def _q_latent(self, obs):
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)
        return self.mlp_extractor.forward_critic(features)

    def q_batch(self, data):
        # no padding without sequences, so every entry is real and there is no mask
        return self.q_values(data.observations, data.actions), data.returns


class QHeadRecurrentPolicy(_RecurrentQHeadMixin, RecurrentActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_q_head(q_net_arch, q_lr)


class QHeadRecurrentMultiInputPolicy(_RecurrentQHeadMixin, RecurrentMultiInputActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_q_head(q_net_arch, q_lr)


class QHeadPolicy(_FeedForwardQHeadMixin, ActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_q_head(q_net_arch, q_lr)


class QHeadMultiInputPolicy(_FeedForwardQHeadMixin, MultiInputActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_q_head(q_net_arch, q_lr)


def policy_for(obs_type, recurrent=True):
    """The policy class matching an observation type from pusht_gym.OBS_TYPES.

    Only the image arm is a Dict space; `keypoint` and `state` are both a flat Box.
    """
    if recurrent:
        return QHeadRecurrentMultiInputPolicy if obs_type == "image" else QHeadRecurrentPolicy
    return QHeadMultiInputPolicy if obs_type == "image" else QHeadPolicy


def features_extractor_kwargs(obs_type, corrupt_obs, t_max=DEFAULT_T_MAX):
    """The features_extractor_class / _kwargs pair for an arm, for policy_kwargs."""
    inner_class = STResNetExtractor if obs_type == "image" else FlattenExtractor
    if not corrupt_obs:
        return {"features_extractor_class": inner_class, "features_extractor_kwargs": {}}
    return {
        "features_extractor_class": CorruptingExtractor,
        "features_extractor_kwargs": {"inner_class": inner_class, "t_max": t_max},
    }


def corrupting_extractor(policy):
    """The CorruptingExtractor of a policy, or None if the arm is clean."""
    extractor = getattr(policy, "features_extractor", None)
    return extractor if isinstance(extractor, CorruptingExtractor) else None


def set_corruption(policy, enabled):
    """Turn corruption on/off on a loaded policy, with a message for a foreign checkpoint."""
    extractor = corrupting_extractor(policy)
    if extractor is None:
        if enabled:
            raise SystemExit(
                "[ERROR] This checkpoint has no CorruptingExtractor, so there is no observation "
                "corruption to enable. It was trained without --corrupt-obs.")
        return
    extractor.enabled = bool(enabled)
