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
OBS_DIMS = {"keypoint": 20, "state": 6, "image": 512}   # 512 is the ResNet's own width, per frame


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


class STResNetExtractor(BaseFeaturesExtractor):
    """ST's image encoder: ResNet18 / IMAGENET1K_V1 / GroupNorm / 76px crop, trained end to end.

    IMAGE ONLY, 512-d per stacked frame: see __init__.

    Matches `obs_encoder` in diffusion_policy/config/pusht_base.yaml rather than SB3's
    NatureCNN, so the image arm learns from the same representation the diffusion-policy
    arms do. Not the frozen SD VAE: that
    was reverted on 2026-08-30 on measured evidence that freezing broke the policy, while the same
    encoder trained end to end recovered most of the gap.
    """

    def __init__(self, observation_space, crop=ST_CROP, use_group_norm=True):
        image_space = observation_space["image"]
        # VecFrameStack concatenates the stack on the CHANNEL axis, so a 4-stack arrives as
        # (B, 12, H, W) and this pretrained conv1 takes 3. Encode per frame through the one
        # shared encoder instead, exactly as ST does for its n_obs_steps
        # (diffusion_unet_hybrid_image_policy.py:239 reshapes (B, To, C, H, W) to (B*To, C, H, W),
        # runs the shared encoder once, and concatenates: global_cond_dim = obs_feature_dim * To).
        channels = image_space.shape[0]
        assert channels % 3 == 0, f"image channels {channels} is not a multiple of 3 (RGB)"
        self.n_frames = channels // 3
        # 512 per frame, the ResNet's own width: IMAGE ONLY. agent_pos used to be concatenated
        # here, which handed this arm the arm's exact position in closed form -- a strictly
        # stronger observation than task/pusht_image_search_imgonly.yaml gives the
        # diffusion-policy arms, and enough to make the two families' numbers non-comparable.
        super().__init__(observation_space, features_dim=512 * self.n_frames)
        # the crop offset arrives with the observation when AugmentationDraw is in the stack;
        # without one this arm centre-crops, which is what PPO now does in both phases
        self.replay_crop = AUG_CROP_KEY in observation_space.spaces
        resnet = get_resnet("resnet18", weights="IMAGENET1K_V1")
        if use_group_norm:
            # batch statistics are wrong at these batch sizes; standard for diffusion policy
            resnet = replace_submodules(
                root_module=resnet,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features))
        self.resnet = resnet
        # the cropper sees ONE frame at a time, so its input shape is 3 channels, not 3*n
        self.cropper = CropRandomizer(input_shape=(3,) + tuple(image_space.shape[1:]),
                                      crop_height=crop, crop_width=crop, num_crops=1, pos_enc=False)
        # The centre offset, precomputed. `_crop` ALWAYS supplies an offset, so
        # CropRandomizer's own `self.training` branch never runs -- see _crop for why.
        height, width = image_space.shape[1:]
        self._centre_offset = ((height - crop) // 2, (width - crop) // 2)

    def _crop(self, img, obs):
        """The stored crop if the observation carries one, else the CENTRE crop.

        `self.training` is never consulted. SB3 has it backwards for this purpose -- False
        while collecting, True during the update -- so an arm that let CropRandomizer branch on
        it would crop one way while acting and another while learning. Supplying an offset
        unconditionally is what makes the crop a property of the OBSERVATION rather than of
        whichever phase the algorithm happens to be in.

        That single rule gives all three behaviours the arms need:
          * PPO      `aug_for` emits no `crop_span` for image, so no key ever arrives and this
                     arm centre-crops at train AND eval. A deterministic crop cannot be redrawn
                     between rollout and update, so PPO's ratio stays a policy ratio for free.
          * SAC      wraps with VecAugmentationDraw, so the key is present and the crop is
                     random in both collection and update -- the augmentation the off-policy
                     arms keep.
          * verifier `sac.score.obs_for_arm` supplies the centre offset explicitly, so the Q is
                     deployed on a centre crop while having trained on random ones.
        """
        if not self.replay_crop:
            dy, dx = self._centre_offset
            offsets = th.tensor([dy, dx], device=img.device).expand(img.shape[0], 2)
        else:
            # ONE offset per transition, shared by every frame of the stack. The draw is per
            # stacked observation (VecAugmentationDraw sits outside arch.wrap and stores a
            # single (2,) offset), so repeating it is what keeps the replay exact: a per-frame
            # offset would need the draw itself to widen to (n_frames, 2), changing the space.
            offsets = obs[AUG_CROP_KEY].repeat_interleave(self.n_frames, dim=0)
        # _forced_offsets is CropRandomizer's own extension point for exactly this
        self.cropper._forced_offsets = offsets.long()
        try:
            return self.cropper(img)
        finally:
            self.cropper._forced_offsets = None

    def forward(self, obs):
        # SB3 has already divided the uint8 image by 255; ST's normalizer then maps [0,1] -> [-1,1]
        img = obs["image"] * 2.0 - 1.0
        batch = img.shape[0]
        # (B, 3*n, H, W) -> (B*n, 3, H, W); the stack is oldest-first and reshape preserves it
        img = img.reshape(batch * self.n_frames, 3, *img.shape[-2:])
        return self.resnet(self._crop(img, obs)).reshape(batch, 512 * self.n_frames)


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


def carries_pretrained_weights(policy_kwargs):
    """Does this policy's features extractor hold weights that must NOT be re-initialised?

    Keyed on the EXTRACTOR, not on the policy class: corruption makes the keypoint arm
    Dict-spaced too, so `MultiInput` does not mean `image`, and the keypoint arm has no
    pretrained weights to protect. `CorruptingExtractor` wraps the real one, so its
    `inner_class` is checked as well.
    """
    if policy_kwargs.get("features_extractor_class") is STResNetExtractor:
        return True
    inner = (policy_kwargs.get("features_extractor_kwargs") or {}).get("inner_class")
    return inner is STResNetExtractor


def _protect_pretrained(kwargs):
    """Turn SB3's orthogonal init off when the extractor is the pretrained ResNet.

    SB3 applies orthogonal init to the WHOLE features extractor. Over ResNet18/IMAGENET1K_V1
    that is destructive and slow in two separate ways, both measured:

      * conv1 does not survive it -- |w| 0.0762 -> 0.0936 -- so the image arm would train from
        orthogonal noise rather than the pretrained encoder that is the only reason it matches
        ST's.
      * `torch.nn.init.orthogonal_` runs a QR per layer and DEADLOCKS under multi-threaded BLAS.
        Policy construction takes 0.40s at one thread and has not returned after 120s at eight
        or at twenty-four. It is the reason `test_evaluate_actions_reproduces_the_collected_log_prob`
        could not run at all above one thread.

    This lives on the policy rather than in `arch._shared_policy_kwargs` so that it travels with
    the CLASS: anything constructing a policy directly -- the tests, a notebook, a future caller
    -- is covered, not just the training path. An explicit `ortho_init` from the caller still wins.
    """
    if "ortho_init" not in kwargs and carries_pretrained_weights(kwargs):
        kwargs["ortho_init"] = False
    return kwargs


class QHeadRecurrentPolicy(_RecurrentQHeadMixin, RecurrentActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **_protect_pretrained(kwargs))
        self._build_q_head(q_net_arch, q_lr)


class QHeadRecurrentMultiInputPolicy(_RecurrentQHeadMixin, RecurrentMultiInputActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **_protect_pretrained(kwargs))
        self._build_q_head(q_net_arch, q_lr)


class QHeadPolicy(_FeedForwardQHeadMixin, ActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **_protect_pretrained(kwargs))
        self._build_q_head(q_net_arch, q_lr)


class QHeadMultiInputPolicy(_FeedForwardQHeadMixin, MultiInputActorCriticPolicy):
    def __init__(self, *args, q_net_arch=(128, 128), q_lr=1e-3, **kwargs):
        super().__init__(*args, **_protect_pretrained(kwargs))
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
            n_stack=1, snr=None, random_crop=False):
    """The AugmentationDraw kwargs an arm needs, or None if it needs no draw.

    `random_crop` is OFF by default, which is what makes PPO centre-crop: with no `crop_span`
    the draw carries no offset, no `aug_crop` key reaches the observation, and
    STResNetExtractor falls back to the centre. PPO therefore sees the same deterministic
    transform while acting and while learning, and the eval it is scored on is the one it
    trained under.

    The off-policy arms pass `random_crop=True`. SAC keeps the augmentation because its
    updates replay a buffer rather than the trajectory just collected, so a redrawn crop
    costs it nothing -- and its verifier is deployed on a centre crop supplied explicitly by
    `sac.score.obs_for_arm`.
    """
    aug = {}
    if corrupt_obs:
        # The draw is applied AFTER frame stacking, so it has to match the width the extractor
        # actually produces from a stacked observation. Every arm scales with the stack: a flat
        # observation is concatenated whole, and the image arm encodes each frame through the
        # one shared ResNet and concatenates the 512-d results. Since that arm went image-only,
        # there is nothing concatenated alongside.
        aug["feature_dim"] = OBS_DIMS[obs_type] * max(1, n_stack)
        # a target SNR pins ONE level; without it the level is drawn from U[0, t_max)
        t = snr_to_timestep(snr) if snr is not None else None
        aug["t_min"], aug["t_max"] = (t, t + 1) if t is not None else (0, t_max)
    if obs_type == "image" and random_crop:
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
