from typing import Dict, Any
import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.common.obs_corruption import ObsCorruptionMixin
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from torch.distributions import Normal

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.gpt2_trunk import build_gpt2_trunk
from diffusion_policy.policy.crop_scope import CropScopeMixin
from diffusion_policy.policy.search_procedure import SearchProcedureMixin


class SearchPolicy(ObsCorruptionMixin, CropScopeMixin, SearchProcedureMixin, BaseImagePolicy):
    def __init__(self,
            shape_meta: dict[str, Any],
            obs_encoder: MultiImageObsEncoder,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            hidden_dim: int = 512,
            hidden_depth: int = 4,
            n_head: int = 4,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            corrupt_obs: bool = False,
            mask_obs: bool = False,
            concat_obs: bool = False,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]
        self.obs_encoder = obs_encoder
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.normalizer = LinearNormalizer()
        # `search_kwargs`, not `kwargs`: DiffusionUnetImagePolicy already owns `kwargs` for
        # its scheduler.step() keyword arguments, and PushTUNetSearchPolicy inherits both.
        # One unambiguous name means that class needs no aliases and no overrides.
        self.search_kwargs = kwargs
        self.verifier = self._build_verifier(**kwargs)
        self._init_selection(**kwargs)
        self._init_crop(shape_meta, kwargs.get('crop_shape'),
                        kwargs.get('random_crop', True))
        self.mask_obs = mask_obs
        self.concat_obs = concat_obs
        if self.concat_obs:
            assert not mask_obs, "Cannot use both concat_obs and mask_obs"
            print("Using concat_obs: concatenating obs features with action-value features at each step")

        if self.mask_obs:
            self.first_action_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.context_dim = self._context_dim(obs_feature_dim, **kwargs)
        ctx_in = horizon * action_dim + self.context_dim
        if self.concat_obs:
            self.obs_projection = nn.Linear(obs_feature_dim * n_obs_steps, hidden_dim // 2)
            self.act_projection = nn.Linear(ctx_in, hidden_dim // 2)
            self.hidden_dim = hidden_dim
        else:
            self.obs_projection = nn.Linear(obs_feature_dim * n_obs_steps, hidden_dim)
            self.act_projection = nn.Linear(ctx_in, hidden_dim)
        self.max_actions = kwargs['max_actions']

        # Shared trunk. n_positions covers the obs token plus max_actions candidates.
        self.transformer = build_gpt2_trunk(
            n_emb=hidden_dim, n_head=n_head, n_layer=hidden_depth,
            n_positions=1 + self.max_actions,
            p_drop_emb=p_drop_emb, p_drop_attn=p_drop_attn)
        
        output_dim = horizon * action_dim
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)
        
        self.log_std_limits = (-5.0, 2.0)

        self._init_corruption(corrupt_obs, kwargs.get('corrupt_obs_eval'))
        self.obs_noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
        if corrupt_obs:
            print("Corrupting obs with a separate noise scheduler")

    def _build_verifier(self, **kwargs):
        """`l2s` is imported here, not at module scope, so this module stays importable in
        environments that lack the maze-only package."""
        from l2s.verifier import MazeVerifier
        return MazeVerifier(
            maze_path=kwargs.get('maze_path', None),
            device=kwargs.get('device', 'cpu'),
            noise=kwargs.get('verifier_noise', 0.0),
        )

    def forward(
            self,
            obs_features: torch.Tensor, # B, hidden_dim
            action_value_features: torch.Tensor = None, # B, max_actions, hidden_dim
            # attention_mask: torch.Tensor = None, # B, max_actions
        ) -> Normal:
        """
        Returns a factorized Normal over an action trajectory of shape
        (batch, horizon, action_dim).
        """
            
        B = obs_features.shape[0]
        if self.mask_obs:
            h = self.first_action_mlp(obs_features) # B, hidden_dim
            h = h.unsqueeze(1) # B, 1, hidden_dim
            mean = self.mean_head(h).reshape(B, 1, self.horizon, self.action_dim)
            log_std = self.log_std_head(h).reshape(B, 1, self.horizon, self.action_dim)
            log_std = log_std.clamp(
                min=self.log_std_limits[0],
                max=self.log_std_limits[1]
            )
            std = torch.exp(log_std)
            first_action_mean = mean
            first_action_std = std

        if action_value_features is None:
             # B, 1, horizon, action_dim
            if self.mask_obs:
                return Normal(first_action_mean, first_action_std)
            if self.concat_obs:
                inputs = torch.cat([obs_features.unsqueeze(1), torch.zeros(B, 1, self.hidden_dim // 2, device=obs_features.device)], dim=-1) # B, 1, hidden_dim
            else:
                inputs = obs_features.unsqueeze(1) # B, 1, hidden_dim
            T = 1
        else:
            if self.mask_obs:
                inputs = action_value_features
                T = action_value_features.shape[1]
            else:
                if self.concat_obs:
                    first_token = torch.cat([obs_features.unsqueeze(1), torch.zeros(B, 1, self.hidden_dim // 2, device=obs_features.device)], dim=-1) # B, 1, hidden_dim
                    action_value_features = torch.cat([obs_features.unsqueeze(1).expand(-1, action_value_features.shape[1], -1), action_value_features], dim=-1) # B, max_actions, hidden_dim
                    inputs = torch.cat([first_token, action_value_features], dim=1) # B, max_actions+1, hidden_dim
                else:
                    inputs = torch.cat([obs_features.unsqueeze(1), action_value_features], dim=1)
                T = 1 + action_value_features.shape[1]
        
        attention_mask = torch.ones(B, T, device=obs_features.device)
        
        h = self.transformer(
            inputs_embeds=inputs,
            attention_mask=attention_mask
        ).last_hidden_state  # B, T+1, hidden_dim

        mean = self.mean_head(h).reshape(B, T, self.horizon, self.action_dim)
        log_std = self.log_std_head(h).reshape(B, T, self.horizon, self.action_dim)
        log_std = log_std.clamp(
            min=self.log_std_limits[0],
            max=self.log_std_limits[1]
        )
        std = torch.exp(log_std)
        if self.mask_obs:
            mean = torch.cat([first_action_mean, mean], dim=1)
            std = torch.cat([first_action_std, std], dim=1)
        return Normal(mean, std) # B, T, horizon, action_dim

    def _context_tokens(self, actions, values):
        """(B, K, H, Da) raw actions + (B, K[, context_dim]) feedback -> (B, K, hidden)."""
        B, K = actions.shape[:2]
        actions = self._normalize_context_actions(actions)
        actions = actions.reshape(B, K, -1).to(self.act_projection.weight.device)
        values = values.to(actions)
        if values.ndim == 2:
            # (B, K) scalar feedback -> (B, K, 1)
            values = values.unsqueeze(-1)
        assert values.shape[-1] == self.context_dim, \
            f"Got context feedback dim {values.shape[-1]}, expected {self.context_dim}"
        return self.act_projection(torch.cat([actions, values], dim=-1))

    def _encode_obs(self, nobs) -> torch.Tensor:
        """ALREADY-NORMALIZED obs dict (B, T, ...) -> (B, T, obs_feature_dim).

        Same contract as the diffusion policy's, so PushTSearchMixin's `_encode_subgoal`
        works against either family. The fused (B, obs_feature_dim*To) -> hidden
        projection stays in the caller: it is this policy's tokenization, not encoding.
        """
        if isinstance(nobs, dict):
            B, T = next(iter(nobs.values())).shape[:2]
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            B, T = nobs.shape[:2]
            this_nobs = nobs.reshape(-1, *nobs.shape[2:])
        offsets = self._crop_offsets_for(B, repeat=T)
        if offsets is None:
            return self.obs_encoder(this_nobs).reshape(B, T, -1)
        return self.obs_encoder(this_nobs, crop_offsets=offsets).reshape(B, T, -1)

    def _encode_obs_features(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize + encode the obs window -> (B, n_obs_steps, obs_feature_dim).

        Sliced to To BEFORE normalizing; see the diffusion policy's copy for why the two
        orders are bit-identical and why this one is cheaper.
        """
        To = self.n_obs_steps
        if isinstance(obs_dict, dict):
            obs_dict = dict_apply(obs_dict, lambda x: x[:, :To, ...])
        else:
            obs_dict = obs_dict[:, :To, ...]
        return self._encode_obs(self.normalizer.normalize(obs_dict))

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            actions: torch.Tensor = None,
            values: torch.Tensor = None,
            obs_features: torch.Tensor = None,
        ) -> Dict[str, torch.Tensor]:
        """One action chunk, optionally conditioned on prior candidates.

        ``obs_features`` optionally supplies an already-encoded observation, exactly as on
        the diffusion policy: every candidate of one search conditions on the SAME
        observation, so the search loop encodes once and hands the result to each call.

        The trunk emits one distribution per sequence position, i.e. one per candidate
        slot; the NEXT candidate is the last position, so the slice happens here rather
        than in the caller.
        """
        assert 'past_action' not in obs_dict
        To = self.n_obs_steps
        if obs_features is None:
            obs_features = self._encode_obs_features(obs_dict)
        B = obs_features.shape[0]
        obs_features = self.corrupt_obs_features(obs_features)
        obs_features = self.obs_projection(obs_features.reshape(B, -1))

        # Prepare action_value_features by concatenating actions and values, and projecting to hidden_dim
        action_value_features = (
            None if actions is None else self._context_tokens(actions, values))

        dist = self.forward(obs_features, action_value_features) # B, max_actions, horizon, action_dim
        naction_pred = dist.rsample() # B, max_actions, horizon, action_dim
        action_pred = self.normalizer['action'].unnormalize(naction_pred) # B, max_actions, horizon, action_dim

        action_pred = action_pred[:, -1]              # B, horizon, action_dim
        start = To - 1
        end = start + self.n_action_steps
        return {
            'action': action_pred[:, start:end],     # B, n_action_steps, action_dim
            'action_pred': action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        target_actions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = target_actions.shape
        To = self.n_obs_steps
        assert T == self.horizon, \
            f"Expected action horizon {self.horizon}, got {T}"
        assert Da == self.action_dim, \
            f"Expected action dim {self.action_dim}, got {Da}"

        # Encoded ONCE and shared with the context search below, the way the diffusion
        # policy's compute_loss already does it. The search used to encode the same
        # observation a second time; the two results were bit-identical, because
        # `_draw_crop_offsets` re-seeds from (seed, step) on every call and so hands both
        # encodes the same offsets. Keeping the raw features -- pre-corruption,
        # pre-projection -- is what makes them reusable: `predict_action` applies its own
        # corruption per candidate, exactly as before, so the global RNG stream is
        # unchanged.
        raw_obs_features = self._encode_obs_features(batch['obs'])
        obs_features = self.corrupt_obs_features(raw_obs_features)
        obs_features = self.obs_projection(obs_features.reshape(B, -1)) # B, hidden_dim

        with torch.inference_mode():
            actions, values = self.search_candidates(
                batch['obs'],
                verifier=self.verifier,
                n_actions=self.max_actions-1,
                obs_features=raw_obs_features,
            ) # B, max_actions, horizon*action_dim and B, max_actions

        action_value_features = self._context_tokens(actions, values) # B, max_actions, hidden
        dist = self.forward(obs_features, action_value_features)  # B, max_actions, horizon, action_dim
        target_actions = target_actions.unsqueeze(1).expand(-1, self.max_actions, -1, -1) # B, max_actions, horizon, action_dim
        log_prob = dist.log_prob(target_actions).sum(dim=(-1, -2)) # B, max_actions
        loss = -log_prob.mean()
        return loss
