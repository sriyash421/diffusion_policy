"""Feature extractors that can corrupt the encoded observation, and the Q head.

CORRUPTION lives in a features extractor, passed through SB3's own
`features_extractor_class` / `features_extractor_kwargs`. Corrupting the encoded observation IS a
feature-extractor concern, and going through the documented extension point means SB3 builds it
itself, at the right time: nothing is mutated after `__init__`, so the optimizer is built over the
final module tree and the policies stay STOCK apart from the Q head.

The noise is the repo's: a DDPM forward marginal under the scheduler of
`diffusion_policy/config/train_pusht_diffusion_search.yaml:339`, matching
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



import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
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
from recurrent_ppo.pusht_gym import AUG_CROP_KEY, AUG_NOISE_KEY, AUG_T_KEY, STATE_KEY

DEFAULT_T_MAX = 200
ST_CROP = 76
# 9 block keypoints x2 + agent xy, then the same again as a visibility mask
# flat observation widths, by arm; the image arm's is its encoder's output instead
OBS_DIMS = {"keypoint": 40, "state": 6}        # crop_shape in diffusion_policy/config/pusht_base.yaml


def snr_to_timestep(snr):
    """The DDPM step whose signal-to-noise ratio is closest to `snr`.

    alpha_bar_t / (1 - alpha_bar_t) IS the SNR in feature space, because the noise is scaled by
    the running feature std -- that is what the scaling buys, and what makes a level mean the
    same thing to a 40-d keypoint vector and a 514-d ResNet feature.
    """
    ab = make_obs_noise_scheduler().alphas_cumprod.numpy()
    return int(np.argmin(np.abs(ab / (1.0 - ab) - float(snr))))


def make_obs_noise_scheduler():
    """The obs_noise_scheduler of train_pusht_diffusion_search.yaml, verbatim."""
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                         beta_schedule="linear", prediction_type="epsilon")


def _widen_conv1(resnet, in_channels):
    """Let the pretrained stem read a stack of frames, not just one.

    VecFrameStack concatenates frames on the channel axis, so `--n-stack 4` hands the encoder
    12 channels and resnet18's 3-channel conv1 rejects them. The pretrained filters are worth
    keeping, so conv1 is rebuilt at the new width and its weights TILED across the stack and
    divided by the repeat count. That makes the stem's response to a static scene -- every
    frame identical, which is what the first steps of an episode and any motionless moment
    look like -- exactly what the 3-channel net would have produced.
    """
    old = resnet.conv1
    reps, remainder = divmod(in_channels, old.in_channels)
    assert remainder == 0, f"{in_channels} channels is not a whole number of {old.in_channels}-channel frames"
    conv = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                     stride=old.stride, padding=old.padding, bias=old.bias is not None)
    with th.no_grad():
        conv.weight.copy_(old.weight.repeat(1, reps, 1, 1) / reps)
        if old.bias is not None:
            conv.bias.copy_(old.bias)
    resnet.conv1 = conv


class STResNetExtractor(BaseFeaturesExtractor):
    """ST's image encoder: ResNet18 / IMAGENET1K_V1 / GroupNorm / 76px crop, trained end to end.

    Matches `obs_encoder` in diffusion_policy/config/pusht_base.yaml rather than SB3's
    NatureCNN, so the image arm learns from the same representation the diffusion-policy
    arms do. Not the frozen SD VAE: that
    was reverted on 2026-08-30 on measured evidence that freezing broke the policy, while the same
    encoder trained end to end recovered most of the gap.
    """

    def __init__(self, observation_space, crop=ST_CROP, use_group_norm=True):
        image_space = observation_space["image"]
        super().__init__(observation_space, features_dim=512 + observation_space["agent_pos"].shape[0])
        # the crop offset arrives with the observation when AugmentationDraw is in the stack
        self.replay_crop = AUG_CROP_KEY in observation_space.spaces
        resnet = get_resnet("resnet18", weights="IMAGENET1K_V1")
        if use_group_norm:
            # batch statistics are wrong at these batch sizes; standard for diffusion policy
            resnet = replace_submodules(
                root_module=resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
        if image_space.shape[0] != 3:
            _widen_conv1(resnet, image_space.shape[0])
        self.resnet = resnet
        # CropRandomizer switches on self.training by itself -- random crop in train, CENTRE crop
        # in eval. We override that in both directions: `_crop` pins the stored offset whenever
        # the observation carries one, which it always does on this arm. So this policy is
        # evaluated on a random crop, deliberately. See _crop.
        self.cropper = CropRandomizer(input_shape=image_space.shape, crop_height=crop,
                                      crop_width=crop, num_crops=1, pos_enc=False)

    def _crop(self, img, obs):
        """The stored crop if the observation carries one, else CropRandomizer's own choice.

        Pinning the stored offset is what keeps PPO's ratio a policy ratio on the image arm:
        without it the crop is redrawn on every epoch, so the update scores an action against
        a different view of the observation than the one it was chosen from.

        IT IS PINNED AT EVALUATION TOO, and that is a protocol choice rather than a side effect.
        `aug_for` emits `crop_span` for the image arm whether or not it is corrupted, so the key
        is always present and CropRandomizer's own eval branch -- a centre crop -- never runs.
        The evaluation is therefore true to training: the policy is scored on exactly the
        transform it learned under, not on a sharper view it never saw. Centre-cropping at eval
        would measure a different input distribution from the one the weights were fitted to,
        and the gap between the two would show up as an unexplained eval/train discrepancy.
        The eval env is re-seeded before each evaluation, so the crops are a FIXED set: every
        checkpoint sees the same ones, and the comparison between checkpoints is still paired.
        """
        if not self.replay_crop:
            return self.cropper(img)
        offsets = obs[AUG_CROP_KEY].long()
        # _forced_offsets is CropRandomizer's own extension point for exactly this
        self.cropper._forced_offsets = offsets
        try:
            return self.cropper(img)
        finally:
            self.cropper._forced_offsets = None

    def forward(self, obs):
        # SB3 has already divided the uint8 image by 255; ST's normalizer then maps [0,1] -> [-1,1]
        img = obs["image"] * 2.0 - 1.0
        return th.cat([self.resnet(self._crop(img, obs)), obs["agent_pos"]], dim=-1)


class KeypointExtractor(BaseFeaturesExtractor):
    """The keypoint observation, flattened, whether or not a draw is riding alongside it.

    A plain `FlattenExtractor` cannot be used once AugmentationDraw nests the observation in a
    Dict: it would flatten the draw into the features too.
    """

    def __init__(self, observation_space):
        space = (observation_space.spaces[STATE_KEY]
                 if isinstance(observation_space, spaces.Dict) else observation_space)
        super().__init__(observation_space, features_dim=int(np.prod(space.shape)))
        self.nested = isinstance(observation_space, spaces.Dict)
        self.flatten = nn.Flatten()

    def forward(self, obs):
        return self.flatten(obs[STATE_KEY] if self.nested else obs)


class CorruptingExtractor(BaseFeaturesExtractor):
    """An inner features extractor with the observation corruption on its output.

    The noise and its timestep are NOT drawn here: they are read off the observation, where
    `AugmentationDraw` put them. That is what makes the corruption replayable -- every one of
    PPO's `--n-epochs` passes over a stored transition reconstructs the same corrupted
    features, so the importance ratio stays a ratio of policies rather than of observations.
    """

    def __init__(self, observation_space, inner_class=FlattenExtractor, inner_kwargs=None,
                 enabled=True, t_max=None, std_momentum=0.99):
        inner = inner_class(observation_space, **(inner_kwargs or {}))
        super().__init__(observation_space, features_dim=inner.features_dim)
        self.inner = inner
        self.enabled = bool(enabled)
        # `t_max` is ACCEPTED AND IGNORED, and only so that checkpoints saved before it was
        # removed still load -- SB3 pickles features_extractor_kwargs into the zip and replays
        # them on load. The level is the env's alone: `aug_draw` draws t into the observation and
        # forward() reads back whatever it finds. Storing a cap here implied the extractor
        # enforced one, which it never did, and --corrupt-snr now deliberately draws a single t
        # that can sit well past any U[0, t_max) range.
        del t_max
        self.std_momentum = float(std_momentum)
        self.scheduler = make_obs_noise_scheduler()
        # The EMA may move only while the ROLLOUT is being collected, never during the update
        # epochs: a std that drifted between epoch 1 and epoch 2 would change the corruption
        # applied to an already-stored transition, undoing the replay. `self.training` cannot
        # gate this -- SB3 has it exactly backwards for this purpose, False while collecting and
        # True during the update -- so FeatureStdWindow toggles it around collection instead.
        self.update_std = False
        # ST's obs_feature_std, same EMA. Registered as buffers so it travels with the checkpoint:
        # a policy reloaded mid-training must keep the scale its noise was calibrated against.
        self.register_buffer("feature_std", th.ones(inner.features_dim))
        self.register_buffer("feature_std_inited", th.zeros(1, dtype=th.bool))

    @th.no_grad()
    def _update_feature_std(self, features):
        flat = features.reshape(-1, features.shape[-1])
        if flat.shape[0] < 2:
            # A std over ONE sample is NaN, and clamp_min does not repair a NaN -- it would
            # poison the buffer permanently and every action mean with it. Single-row batches
            # are routine: SB3 bootstraps a truncated episode by calling predict_values on one
            # terminal observation, and the eval env runs a single env. There is nothing to
            # estimate from one sample, so leave the EMA where it is.
            return
        std = flat.std(dim=0).clamp_min(1e-6)
        if not bool(self.feature_std_inited):
            self.feature_std.copy_(std)
            self.feature_std_inited.fill_(True)
        else:
            self.feature_std.mul_(self.std_momentum).add_(std, alpha=1.0 - self.std_momentum)

    def forward(self, obs):
        features = self.inner(obs)
        if not self.enabled:
            return features
        if self.update_std:
            self._update_feature_std(features)
        noise = obs[AUG_NOISE_KEY].to(features.dtype) * self.feature_std
        t = obs[AUG_T_KEY].reshape(-1).long()
        return self.scheduler.add_noise(features, noise, t)


class _QHeadMixin:
    """Q(s, a), trained alongside PPO but never inside its objective.

    Built AFTER `super().__init__()`, which is what keeps `q_net` out of `self.optimizer`: the
    policy optimizer was already constructed over the parameters that existed then. The Q head
    carries its own, so it provably cannot move the policy. See tests.

    Two subclasses supply the critic latent, because that is the only thing the architectures
    disagree about: under the LSTM it is Q(history, a), under frame stacking Q(stack, a).

    WHAT THIS Q IS. It is fitted by `train.QHead` to `rollout_buffer.returns`, the TD(lambda)
    return of the BEHAVIOUR policy, so it is Q^pi and not Q*: an evaluation of whatever PPO
    currently is, trustworthy only near that policy. Making it Q* means replacing that target
    with a Bellman backup -- r + gamma * max_a' Q_target(h', a') -- which needs a target network
    and a maximisation this head has no machinery for. `train.QHead._on_rollout_end` is the one
    place the target is chosen, so that is where the swap would go.
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


def policy_for(obs_type, recurrent=True, corrupt_obs=False):
    """The policy class matching an arm.

    The image arm is a Dict space; `keypoint` and `state` are a flat Box -- until corruption,
    which makes them Dict too, because the per-transition draw rides alongside the observation.
    """
    dict_space = obs_type == "image" or corrupt_obs
    if recurrent:
        return QHeadRecurrentMultiInputPolicy if dict_space else QHeadRecurrentPolicy
    return QHeadMultiInputPolicy if dict_space else QHeadPolicy


def features_extractor_kwargs(obs_type, corrupt_obs):
    """The features_extractor_class / _kwargs pair for an arm, for policy_kwargs.

    No noise level here: the level lives in the observation, so it is `aug_for`'s to set and the
    extractor never needs to be told it.
    """
    inner_class = STResNetExtractor if obs_type == "image" else KeypointExtractor
    if not corrupt_obs:
        return {"features_extractor_class": inner_class, "features_extractor_kwargs": {}}
    return {
        "features_extractor_class": CorruptingExtractor,
        "features_extractor_kwargs": {"inner_class": inner_class},
    }


def aug_for(obs_type, corrupt_obs, render_size=96, crop=ST_CROP, t_max=DEFAULT_T_MAX,
            n_stack=1, snr=None):
    """The AugmentationDraw kwargs an arm needs, or None if it needs no draw.

    The crop draw is needed by the image arm whether or not it is corrupted -- a redrawn crop
    biases the ratio exactly as a redrawn noise does, and it does so on the CLEAN arm too.
    """
    aug = {}
    if corrupt_obs:
        # The draw is applied AFTER frame stacking, so it has to match the width the extractor
        # actually produces from a stacked observation -- and the two arms widen differently.
        # A flat observation is concatenated whole, so its width scales with the stack. The
        # image arm does NOT: VecFrameStack stacks the frames on the CHANNEL axis, the ResNet
        # consumes all of them and still emits 512, and only agent_pos is concatenated. Scaling
        # the image arm by n_stack gave a 2056-wide noise for a 520-wide feature.
        stack = max(1, n_stack)
        aug["feature_dim"] = (512 + 2 * stack) if obs_type == "image" else OBS_DIMS[obs_type] * stack
        # a target SNR pins ONE level; without it the level is drawn from U[0, t_max)
        t = snr_to_timestep(snr) if snr is not None else None
        aug["t_min"], aug["t_max"] = (t, t + 1) if t is not None else (0, t_max)
    if obs_type == "image":
        aug["crop_span"] = render_size - crop
    return aug or None


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
