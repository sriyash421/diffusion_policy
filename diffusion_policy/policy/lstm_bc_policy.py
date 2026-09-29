"""Behaviour cloning with recurrent PPO's LSTM, over whole demonstration episodes.

THE POINT OF THIS ARM. The repo has a recurrent policy (`recurrent_ppo`, sb3-contrib's
RecurrentPPO) and an offline one (the UNet BC arm), and no overlap: nothing measures what
recurrence buys when the data is fixed demonstrations rather than on-policy rollouts. So the
architecture here is PPO's, pinned key for key against `recurrent_ppo.config.DEFAULTS` -- that
module is the source of truth for every one of these values -- while the data, the split
manifest and the rollout episodes are the UNet BC arm's. What varies between this arm and UNet
BC is the architecture; what varies between this arm and the PPO LSTM is the learning signal.

WHAT IS PPO'S AND WHAT IS NOT:

  PPO's     features extractor -> nn.LSTM(128, 1 layer) -> MLP [128,128]/Tanh -> Gaussian
            with a STATE-INDEPENDENT log_std at -1.0 (SB3's DiagGaussianDistribution).
  NOT PPO's the action. PPO emits one delta action per step; this emits a 16-step chunk of
            ABSOLUTE target positions and executes 8, which is UNet BC's control cadence.
            Absolute because that is what the demonstrations contain -- PPO's delta space is
            not reachable from this data -- and chunked so the two offline arms are compared
            at the same control rate.
  DROPPED   the critic, the value head and the Q head. They are PPO's and have no role here.

ONE CLASS, TWO ARMS. The observation variant is decided entirely by which `obs_encoder` the
config instantiates: `MultiImageObsEncoder` (image-only, 512-d) or `FlattenObsEncoder`
(9 block-T keypoints + agent xy, 20-d). That mirrors `recurrent_ppo`, where the arms differ
only in `STResNetExtractor` vs `KeypointExtractor` and share one policy family.

TIME IS HANDLED IN TWO DIFFERENT SHAPES, and keeping them in correspondence is the whole
difficulty of a recurrent BC arm:

  training   one sample is one WHOLE episode (60-188 steps on the 30-demo train split).
             `compute_loss` runs the LSTM over all of it from a zero state and supervises a
             chunk at EVERY timestep. See `_chunk_targets` for why the targets are masked
             rather than padded.
  rollout    `MultiStepWrapper` hands over `n_action_steps` new frames per call.
             `predict_action` ticks the LSTM once per frame, carries (h, c) across calls, and
             emits the chunk from the LAST hidden state. See `predict_action` for the
             frame-accounting contract, which is the thing that makes the hidden-state
             trajectory at step t the same object in both shapes.
"""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.crop_scope import CropScopeMixin


class LSTMBCPolicy(CropScopeMixin, BaseImagePolicy):
    def __init__(self,
            shape_meta: Dict[str, Any],
            obs_encoder: nn.Module,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            lstm_hidden_size: int = 128,
            n_lstm_layers: int = 1,
            net_arch=(128, 128),
            log_std_init: float = -1.0,
            crop_shape=None,
            random_crop: bool = True,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # THE FRAME-ACCOUNTING INVARIANT, asserted rather than assumed. MultiStepWrapper's obs
        # deque is `maxlen=n_obs_steps+1`, so it can only ever hand back n_obs_steps+1 frames'
        # worth of history -- of which we need the n_action_steps most recent to be the ones
        # just stepped. At n_action_steps > n_obs_steps+1 the wrapper silently drops the
        # oldest of them and the LSTM loses a step per call, which nothing downstream would
        # report: the shapes still match and the loss still falls.
        assert n_action_steps <= n_obs_steps + 1, (
            f'n_action_steps={n_action_steps} exceeds n_obs_steps+1={n_obs_steps + 1}, so '
            f"MultiStepWrapper's obs deque cannot return every frame it just stepped and the "
            f'LSTM would skip {n_action_steps - n_obs_steps - 1} frame(s) per call. Raise '
            f'n_obs_steps to at least n_action_steps-1.')

        self.obs_encoder = obs_encoder
        self.horizon = horizon                  # the ACTION CHUNK length, not a sequence length
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs

        self.lstm = nn.LSTM(
            input_size=self.obs_feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True)

        # SB3's MlpExtractor over the LSTM output: net_arch "128,128" with Tanh, which is
        # sb3-contrib's default activation and is never overridden in recurrent_ppo/arch.py.
        layers, last = [], lstm_hidden_size
        for hidden in net_arch:
            layers += [nn.Linear(last, hidden), nn.Tanh()]
            last = hidden
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, horizon * action_dim)

        # STATE-INDEPENDENT, exactly SB3's DiagGaussianDistribution: one learned scalar per
        # action dimension, broadcast over the chunk. The repo's other BC heads
        # (MLPImagePolicy, TransformerImagePolicy) predict log_std from the state and clamp it
        # to [-5, 2); under an NLL objective that lets the model buy loss by inflating
        # variance instead of improving the mean. A fixed-shape scale makes the objective a
        # scaled MSE, which is what BC wants, and keeps the head the same object PPO learned.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        # One crop offset per SAMPLE, shared by every timestep of that episode. Load-bearing:
        # see `_encode`. crop_shape is None on the keypoint arm, which makes every crop call
        # a no-op.
        self._init_crop(shape_meta, crop_shape, random_crop)

        # rollout state; None is nn.LSTM's own "zeros" and needs no allocation
        self._state = None
        self._fresh = True

    # ------------------------------------------------------------------ shared forward parts
    def _encode(self, nobs, batch_size, n_steps):
        """(B, T, feature_dim) from a dict of (B, T, ...) observations.

        The crop scope is LOAD-BEARING, not decoration, and for a reason specific to this arm.
        Left to itself the encoder's CropRandomizer draws one offset per IMAGE, which for a
        whole-episode sample means ~130 independent +/-20px shifts WITHIN one episode at train
        time, against a single fixed centre crop at eval. The LSTM would be learning to
        integrate a jittering camera and then be evaluated on a still one. `_crop_offsets_for`
        with `repeat=n_steps` gives one offset per episode, repeated across its timesteps --
        the same contract pusht_unet_search_policy uses for its 2-frame window, applied at
        episode length.

        Beyond that, `_crop_offsets_for` caches into `self._crop_offsets` and only the
        outermost `_crop_scope` exit clears it, so calling it with no scope open would hand
        every later batch of the same size the FIRST batch's crops and the augmentation would
        quietly die.
        """
        # Extra keys are harmless: both encoders select by shape_meta, so the dataset's
        # verifier-only keys (agent_pos, feedback on the image arm) are normalized and
        # reshaped but never encoded.
        this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        with self._crop_scope():
            offsets = self._crop_offsets_for(batch_size, repeat=n_steps)
            if offsets is None:
                features = self.obs_encoder(this_nobs)
            else:
                features = self.obs_encoder(this_nobs, crop_offsets=offsets)
        return features.reshape(batch_size, n_steps, -1)

    def _chunk_dist(self, latent, *lead_dims):
        """Normal over an action chunk, from one or more leading (batch/time) dims."""
        mean = self.mean_head(latent).reshape(*lead_dims, self.horizon, self.action_dim)
        return Normal(mean, self.log_std.exp())

    def _chunk_targets(self, nactions, mask):
        """((B, T, horizon, Da) targets, (B, T, horizon) validity) from a padded batch.

        MASKED, NOT PADDED, and this is the one place a plausible shortcut is wrong. The
        chunk supervised at step t is actions[t : t+horizon], which for every t in the last
        `horizon-1` steps of an episode runs off its end. `collate_fn` has already padded the
        batch with 0.0, and 0.0 in normalized action space is the MIDDLE OF THE ARENA -- so
        training on those positions teaches the policy to drive to the centre during the final
        1.5 seconds of every episode, which is both wrong and exactly the kind of wrong that
        shows up as a mediocre success rate rather than as an error.

        `attention_mask` alone cannot express this: it is (B, T), one bit per timestep, and
        what is needed is per (t, j). So it is unfolded into the chunk's own shape. The
        returned targets at invalid positions are finite (the padded zeros) but contribute
        nothing, because the caller divides by `chunk_mask.sum()`.

        Note this deliberately differs from SequenceSampler's window padding, which
        edge-REPEATS the last frame. Repeating would fabricate a target -- "keep pushing
        towards wherever the demo stopped" -- that no demonstrator produced. Masking asserts
        only what the data says.
        """
        B, T, Da = nactions.shape
        H = self.horizon
        mask = mask.to(nactions.dtype)
        # gather indices t+j, clamped by zero-padding rather than by clamping the index (a
        # clamped index would silently alias the last real action into every tail position)
        idx = (torch.arange(T, device=nactions.device)[:, None]
               + torch.arange(H, device=nactions.device)[None, :])          # (T, H)
        target = F.pad(nactions, (0, 0, 0, H - 1))[:, idx]                   # (B, T, H, Da)
        chunk_mask = F.pad(mask, (0, H - 1))[:, idx]                         # (B, T, H)
        # redundant with the j=0 column, but states the intent: a chunk rooted at a padded
        # timestep is wholly invalid, however much of it happens to land on real actions
        return target, chunk_mask * mask[:, :, None]

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = nactions.shape
        assert Da == self.action_dim, f'expected action dim {self.action_dim}, got {Da}'
        # Present whenever the dataset runs in return_sequences mode (the workspace selects
        # sampler.get_collate_fn for it). All-ones is the un-padded case.
        mask = batch.get('attention_mask', None)
        if mask is None:
            mask = torch.ones((B, T), dtype=nactions.dtype, device=nactions.device)

        features = self._encode(nobs, B, T)
        # zero initial state: a demonstration episode starts at its own beginning, which is
        # also where the rollout's state is zeroed
        latent, _ = self.lstm(features)
        dist = self._chunk_dist(self.trunk(latent), B, T)

        target, chunk_mask = self._chunk_targets(nactions, mask)
        log_prob = dist.log_prob(target).sum(dim=-1)                 # (B, T, H)
        return -(log_prob * chunk_mask).sum() / chunk_mask.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ rollout
    def reset(self):
        """Zero the hidden state at an episode boundary.

        `BaseImagePolicy.reset` is a no-op stub and both rollout drivers already call it at
        the right granularity -- once per chunk of envs, immediately after `env.reset()`
        (pusht_search_image_runner.run, eval_search_pusht). So this needs no new hook.
        """
        self._state = None
        self._fresh = True

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One chunk from the current hidden state, having ingested the newly stepped frames.

        THE FRAME-ACCOUNTING CONTRACT. MultiStepWrapper's deque is `maxlen=n_obs_steps+1`, so
        after a step of `n_action_steps` actions it holds the previous frame plus the new ones
        and `_get_obs` returns exactly the `n_action_steps` new frames -- one LSTM tick per env
        step, which is what makes the hidden state at env step t the same object as the
        training state at sequence index t.

        THE EXCEPTION IS THE FIRST CALL. `MultiStepWrapper.reset` seeds the deque with a single
        observation and `stack_last_n_obs` PADS UP by repeating it, so the first call presents
        n_obs_steps identical copies of frame 0. Ingesting all of them would tick the LSTM 8
        times on one frame, while training reaches t=0 having consumed frame 0 once. Hence
        `_fresh`: one frame on the first call, `n_action_steps` after. The result is exact
        parity -- state@t=0 -> chunk actions[0:16], execute 0..7 -> next call ingests frames
        1..8 -> state@t=8 -> chunk actions[8:24].

        AFTER AN ENV TERMINATES the wrapper stops stepping it, so the frames it returns are
        ones already consumed and that env's state advances on duplicates. This is deliberately
        NOT corrected: hidden state is per batch element, a finished env's actions are
        discarded by the wrapper, and its score is a max over the episode that has stopped
        moving. Re-zeroing state mid-batch would corrupt the envs still running.
        """
        assert 'past_action' not in obs_dict
        obs_dict = dict(obs_dict)
        # The workspace's sample block injects this alongside the observation; the normalizer
        # iterates whatever keys it is handed and has no params for it.
        obs_dict.pop('attention_mask', None)

        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]

        n_new = 1 if self._fresh else self.n_action_steps
        assert n_new <= To, (
            f'need the {n_new} most recent frames but the observation carries only {To}; '
            f'the env runner must be built with n_obs_steps >= n_action_steps-1.')
        nobs = dict_apply(nobs, lambda x: x[:, -n_new:])

        features = self._encode(nobs, B, n_new)
        latent, self._state = self.lstm(features, self._state)
        self._fresh = False

        # the chunk belongs to the LAST hidden state, i.e. to the newest frame
        dist = self._chunk_dist(self.trunk(latent[:, -1]), B)
        # The MEAN, not a sample. The other two BC heads rsample() at eval, which at
        # log_std_init=-1.0 injects ~0.37 normalized units of noise into every executed
        # action; a BC rollout should report what the policy believes, not a draw around it.
        naction_pred = dist.mean
        action_pred = self.normalizer['action'].unnormalize(naction_pred)
        # start at 0, NOT n_obs_steps-1: this chunk is rooted at the CURRENT step, so slicing
        # from To-1 the way the window policies do would execute actions 7..14 for step t and
        # put a silent 7-step lag in the controller.
        return {
            'action': action_pred[:, :self.n_action_steps],
            'action_pred': action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
