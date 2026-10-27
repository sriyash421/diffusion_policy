"""Full-trajectory search transformer for PushT -- maze's DTSearchPolicy on PushT.

One causal transformer reads ONE token stream per episode: [attempt_1 | ... | attempt_k | current].
Token t = E_obs(obs_t) + E_a(a_{t-1}) + E_fb(fb_{t-1}) + E_d(d_{t-1}) + E_step(t in attempt) (+ stream
position); the head emits a Gaussian over the next action target a_t. An attempt ends when the env
reports done (coverage > 0.95 or its step cap) or the guard trips -- online_search's rule; there is
no stop head (round 1 trained one on "the demonstrator stopped here" and it never fired). The first
token of an attempt carries the previous attempt's last action / final feedback / d=1 (d is an INPUT
boundary marker only); zeros at the start of the stream. Beyond max_actions attempts the stream keeps
the last max_actions-1; max_actions=1 is best-of-N (every attempt starts an empty stream).

Feedback is `pusht_step_feedback.step_feedback` at the REALIZED state: -(T-to-goal + veritas pin
cost), each over its geometric maximum. reward_mode step: fb = [fb_t, 0] at every token plus the
boundary token [fb_final, X]; traj: the boundary token only. X is the attempt's TOTAL feedback:
`terminal_feedback: value` repeats fb_final (maze: -BFS(final) on both channels);
`terminal_feedback: coverage` is the env's ground-truth max coverage over the attempt (what
success is defined on, in [0, 1]) -- observed once the attempt has been executed, so it is a
legitimate signal for the next attempt and for ranking.

Backbones: `gpt2` (transformers.GPT2Model, learned positions, KV-cached stepping as
OnlineSearchPolicy -- the one that is trained) and `te` (nn.TransformerEncoder + sinusoidal stream
position, full recompute per step; kept for tests). Train == eval: `_step` is the single stepping
function behind predict_step (the eval harness) and rollout_candidates (the training context).

Differences from maze (diffusion_policy_standalone_maze/.../dt_search_policy.py): the env is the
live PushT gym env, so the harness supplies the per-step feedback (`fb`) and the env's `done`
flag with the observation; actions are continuous targets (no cardinal snap); no stop head.
"""
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.policy.crop_scope import CropScopeMixin
from diffusion_policy.env.pusht.pusht_step_feedback import demo_step_feedback, EpisodePlans

TOKEN_KEYS = ('feat', 'prev_a', 'fb', 'prev_d', 'step')
# keys the harness adds to the obs dict for predict_step; never normalized or encoded
STEP_KEYS = ('fb', 'done', 'coverage', 'attention_mask')
BACKBONES = ('gpt2', 'te')


def _expand_idx(idx, like):
    """(B, L) gather/scatter index broadcast to the trailing dims of `like`."""
    return idx.reshape(idx.shape + (1,) * (like.dim() - 2)).expand(-1, -1, *like.shape[2:])


class PushTDTSearchPolicy(CropScopeMixin, BaseImagePolicy):
    def __init__(self,
            shape_meta: Dict[str, Any],
            obs_encoder: nn.Module,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            max_actions: int = 8,
            reward_mode: str = 'step',
            max_attempt_steps: int = 300,
            terminal_feedback: str = 'value',
            backbone: str = 'gpt2',
            gpt2_n_positions: int = 2912,
            n_emb: int = 256,
            n_layer: int = 4,
            n_head: int = 4,
            p_drop: float = 0.1,
            log_std_init: float = -1.0,
            log_std_min: float = -2.0,
            crop_shape=None,
            random_crop: bool = True,
            zarr_path: str = None,
            plan_dir: str = None,
            verifier_value: str = 'wp_v3_la_tgn',
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        assert n_obs_steps == 1 and n_action_steps == 1, 'one token per env step'
        assert reward_mode in ('traj', 'step'), reward_mode
        assert backbone in BACKBONES, backbone
        assert terminal_feedback in ('value', 'coverage'), terminal_feedback
        self.terminal_feedback = terminal_feedback
        self.obs_encoder = obs_encoder
        self.horizon = horizon            # unused: attempts are variable length
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_shape[0]
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.max_attempt_steps = int(max_attempt_steps)
        self.backbone = backbone
        self.n_emb = n_emb
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.normalizer = LinearNormalizer()
        self.verifier_value = verifier_value
        self.search_kwargs = kwargs
        # training-time feedback for the expert tokens needs each demo's waypoint plan
        self._zarr_path, self._plan_dir = zarr_path, plan_dir
        self._plans = None

        self.obs_proj = nn.Linear(self.obs_feature_dim, n_emb)
        self.act_emb = nn.Linear(self.action_dim, n_emb)
        self.fb_emb = nn.Linear(2, n_emb)
        self.done_emb = nn.Linear(1, n_emb)
        self.pe = SinusoidalPosEmb(n_emb)   # no table, no maximum length
        self.step_proj = nn.Linear(n_emb, n_emb)
        self.drop = nn.Dropout(p_drop)
        if backbone == 'gpt2':
            from transformers import GPT2Config, GPT2Model
            # the same block as OnlineSearchPolicy; wpe supplies the stream position
            self.transformer = GPT2Model(GPT2Config(
                n_positions=int(gpt2_n_positions), n_embd=n_emb, n_layer=n_layer, n_head=n_head,
                resid_pdrop=p_drop, embd_pdrop=p_drop, attn_pdrop=p_drop))
        else:
            self.pos_proj = nn.Linear(n_emb, n_emb)
            layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=p_drop,
                activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, enable_nested_tensor=False)
            self.ln_f = nn.LayerNorm(n_emb)
        self.mean_head = nn.Linear(n_emb, self.action_dim)
        # state-independent log_std (as the LSTM arm); the floor keeps candidates distinct
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)

        # one crop offset per EPISODE, shared by all its frames (see LSTMBCPolicy._encode)
        self._init_crop(shape_meta, crop_shape, random_crop)
        self._st = None   # the episode stream used by predict_step / rollout_candidates

    # ------------------------------------------------------------------ pieces
    def _std(self):
        return self.log_std.clamp_min(self.log_std_min).exp()

    def _embed(self, S, keys=None):
        """Summed token embeddings (B, L, E) of a stream (or of the dict `keys` given)."""
        T = S if keys is None else keys
        B, L = T['step'].shape
        x = (self.obs_proj(T['feat']) + self.act_emb(T['prev_a']) + self.fb_emb(T['fb'])
             + self.done_emb(T['prev_d']))
        x = x + self.step_proj(self.pe(T['step'].reshape(-1).float()).reshape(B, L, -1))
        if self.backbone == 'te':
            x = x + self.pos_proj(self.pe(torch.arange(L, device=x.device).float()))[None]
        return self.drop(x)

    def _hidden(self, S):
        """(B, L, n_emb) causal hidden states of whole (padded) streams; positions >= len are padding."""
        x = self._embed(S)
        L = x.shape[1]
        valid = torch.arange(L, device=x.device)[None] < S['len'][:, None]
        if self.backbone == 'gpt2':
            return self.transformer(inputs_embeds=x, attention_mask=valid.to(x.dtype)).last_hidden_state
        causal = torch.ones((L, L), dtype=torch.bool, device=x.device).triu(1)
        return self.ln_f(self.encoder(x, mask=causal, src_key_padding_mask=~valid))

    def _hidden_last(self, S, tok, active):
        """Hidden state at each row's newest token after `tok` was appended for `active` rows.

        gpt2: ONE token through the KV cache (S['past']); inactive rows feed a masked dummy at a
        padding column so the cache stays batch-aligned, and explicit position_ids keep every
        row's positions contiguous -- identical to the compact streams the training forward sees.
        te: full recompute of the stream.
        """
        B = active.shape[0]
        device = active.device
        if self.backbone != 'gpt2':
            return self._hidden(S)[torch.arange(B, device=device), (S['len'] - 1).clamp_min(0)]
        x = self._embed(S, {k: v[:, None] for k, v in tok.items()})            # (B, 1, E)
        S['valid'] = torch.cat([S['valid'], active[:, None]], dim=1)            # (B, past+1)
        pos = (S['len'] - 1).clamp_min(0)[:, None]                              # token's own index
        out = self.transformer(inputs_embeds=x, past_key_values=S['past'], use_cache=True,
                               attention_mask=S['valid'].to(x.dtype), position_ids=pos)
        S['past'] = out.past_key_values
        S['h_last'] = torch.where(active[:, None], out.last_hidden_state[:, -1], S['h_last'])
        return S['h_last']

    def _rebuild_cache(self, S):
        """gpt2: recompute the KV cache from the compact stream (after a slide / for a prefix)."""
        if self.backbone != 'gpt2':
            return
        if S['step'].shape[1] == 0:
            S['past'], S['valid'] = None, torch.zeros((S['len'].shape[0], 0), dtype=torch.bool, device=S['len'].device)
            return
        x = self._embed(S)
        L = x.shape[1]
        valid = torch.arange(L, device=x.device)[None] < S['len'][:, None]
        out = self.transformer(inputs_embeds=x, attention_mask=valid.to(x.dtype), use_cache=True)
        S['past'], S['valid'] = out.past_key_values, valid

    def _heads(self, h):
        return Normal(self.mean_head(h), self._std())

    def _obs_features(self, obs, n_steps=1):
        """(B, n_steps, F) from raw obs of shape (B, n_steps, ...); extra keys are dropped."""
        # only the keys the normalizer was fitted on: the harness/env may carry extra ones
        obs = {k: v for k, v in obs.items() if k in self.normalizer.params_dict}
        nobs = self.normalizer.normalize(obs)
        B = next(iter(nobs.values())).shape[0]
        flat = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
        with self._crop_scope():
            offsets = self._crop_offsets_for(B, repeat=n_steps)
            feat = self.obs_encoder(flat) if offsets is None else \
                self.obs_encoder(flat, crop_offsets=offsets)
        return feat.reshape(B, n_steps, -1)

    def _mask_fb(self, fb):
        if self.reward_mode == 'traj':
            return torch.cat([torch.zeros_like(fb[..., :1]), fb[..., 1:]], dim=-1)
        return fb

    # ------------------------------------------------------------------ stream
    def _empty_stream(self, B, Fo, Da, device):
        z = lambda *shape, dtype=torch.float32: torch.zeros((B, 0) + shape, dtype=dtype, device=device)
        return {'feat': z(Fo), 'prev_a': z(Da), 'fb': z(2), 'prev_d': z(1), 'step': z(dtype=torch.long),
                'len': torch.zeros(B, dtype=torch.long, device=device), 'att_starts': [],
                'has_prev': False, 't': None,
                'last_a': torch.zeros(B, Da, device=device), 'last_fb': torch.zeros(B, 2, device=device),
                # gpt2 KV cache and its column mask; h_last = each row's newest hidden state
                'past': None, 'valid': torch.zeros((B, 0), dtype=torch.bool, device=device),
                'h_last': torch.zeros(B, self.n_emb, device=device)}

    @staticmethod
    def _append(S, rows, tok):
        """Write one token at S['len'] for the rows in the bool mask; the others get padding."""
        if not bool(rows.any()):
            return
        for k in TOKEN_KEYS:
            S[k] = torch.cat([S[k], S[k].new_zeros((S[k].shape[0], 1) + tuple(S[k].shape[2:]))], dim=1)
            S[k][rows, S['len'][rows]] = tok[k][rows]
        S['len'] = S['len'] + rows.long()

    def _slide(self, S, drop):
        """Drop the oldest `drop` attempts; the new first token gets zero prev_* (a fresh prefix)."""
        shift = S['att_starts'][drop]
        L = S['step'].shape[1]
        idx = (torch.arange(L, device=shift.device)[None] + shift[:, None]).clamp(max=L - 1)
        for k in TOKEN_KEYS:
            S[k] = torch.gather(S[k], 1, _expand_idx(idx, S[k]))
        for k in ('prev_a', 'fb', 'prev_d'):
            S[k][:, 0] = 0
        S['len'] = S['len'] - shift
        S['att_starts'] = [a - shift for a in S['att_starts'][drop:]]
        keep = int(S['len'].max())   # drop the now-stale padding columns, or the stream grows for ever
        for k in TOKEN_KEYS:
            S[k] = S[k][:, :keep]
        self._rebuild_cache(S)

    def reset(self):
        self._st = None

    def begin_attempt(self):
        """Mark an attempt boundary; keeps at most max_actions-1 earlier attempts in the stream."""
        S = self._st
        if S is None:
            return
        window = self.max_actions - 1
        if len(S['att_starts']) > window:
            if window == 0:
                self._st = None
                return
            self._slide(S, len(S['att_starts']) - window)
        S['has_prev'] = len(S['att_starts']) > 0
        S['att_starts'].append(S['len'].clone())
        S['t'] = None

    @torch.inference_mode()
    def predict_step(self, obs_dict):
        """One env step for a batch of rows: raw obs (B, 1, ...) plus ``fb`` (B,) -- the
        feedback at the state observed -- ``done`` (B,) bool from the env, and ``coverage`` (B,)
        -- the env's max coverage so far in this attempt (needed under
        terminal_feedback=coverage) -> (action (B, Da), active (B,)).

        Rows whose attempt has ended (env done, guard) get active=False and a zero action; call
        until no row is active so the last rows' feedback is settled.
        """
        return self._step(obs_dict)

    def _step(self, obs_dict):
        obs = {k: (v[:, -1] if k not in ('fb', 'done', 'coverage') else v) for k, v in obs_dict.items()
               if k != 'attention_mask'}
        fb = obs['fb'].float()
        done = obs['done'].bool()
        if self.terminal_feedback == 'coverage':
            assert 'coverage' in obs, 'terminal_feedback=coverage needs the env max coverage per row'
            total = obs['coverage'].float()
        else:
            total = fb
        B, device = fb.shape[0], fb.device
        feat = self._obs_features({k: v[:, None] for k, v in obs.items() if k not in STEP_KEYS})[:, 0]
        if self._st is None:
            self._st = self._empty_stream(B, feat.shape[1], self.action_dim, device)
            self._st['att_starts'].append(self._st['len'].clone())
        S = self._st
        first = S['t'] is None
        if first:
            S['t'] = torch.zeros(B, dtype=torch.long, device=device)
            S['active'] = torch.ones(B, dtype=torch.bool, device=device)
            S['prev_action'] = torch.zeros(B, self.action_dim, device=device)
            S['stop_reason'] = torch.zeros(B, dtype=torch.long, device=device)  # 0 running, 2 env done, 3 guard
        if not first:
            # settle the previous step: rows the env finished or that hit the guard are done
            guard = S['t'] >= self.max_attempt_steps
            ended = S['active'] & (done | guard)
            S['stop_reason'] = torch.where(ended, torch.where(done, 2, 3), S['stop_reason'])
            S['last_a'] = torch.where(ended[:, None], S['prev_action'], S['last_a'])
            S['last_fb'] = torch.where(ended[:, None], torch.stack([fb, total], dim=-1), S['last_fb'])
            S['active'] = S['active'] & ~ended
        active = S['active']
        zeros = torch.zeros(B, 1, device=device)
        if first and S['has_prev']:
            tok = {'prev_a': S['last_a'], 'fb': S['last_fb'], 'prev_d': torch.ones(B, 1, device=device)}
        elif first:
            tok = {'prev_a': torch.zeros(B, self.action_dim, device=device), 'fb': zeros.expand(-1, 2), 'prev_d': zeros}
        else:
            tok = {'prev_a': S['prev_action'], 'fb': torch.stack([fb, torch.zeros_like(fb)], dim=-1), 'prev_d': zeros}
        tok.update(feat=feat, fb=self._mask_fb(tok['fb']), step=S['t'])
        if not bool(active.any()):      # every row settled: nothing to append or predict
            return torch.zeros(B, self.action_dim, device=device), active.clone()
        self._append(S, active, tok)
        h = self._hidden_last(S, tok, active)
        naction = self._heads(h).sample()
        action = self.normalizer['action'].unnormalize(naction)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        S['prev_action'] = torch.where(active[:, None], naction, S['prev_action'])
        S['t'] = S['t'] + active.long()
        return action, active.clone()

    @property
    def attempt_stop_reason(self):
        """(B,) codes for the attempt just settled: 2 env done, 3 guard."""
        return None if self._st is None else self._st['stop_reason'].clone()

    # ------------------------------------------------------------------ rollout (live envs)
    def _get_vec_env(self):
        """The context-rollout env pool (as OnlineSearchPolicy._get_vec_env), built lazily."""
        if getattr(self, '_vec_env', None) is None:
            from diffusion_policy.env.pusht.pusht_feedback import build_parallel
            self._vec_env = build_parallel(
                n_envs=self.search_kwargs.get('rollout_n_envs', 32),
                use_async=self.search_kwargs.get('rollout_use_async', True),
                n_obs_steps=1, n_action_steps=1, max_episode_steps=self.max_attempt_steps,
                legacy=False, tracker=True)
        return self._vec_env

    def close(self):
        """Shut down the rollout worker pool (BaseWorkspace._close_worker_pools calls this)."""
        vec, self._vec_env = getattr(self, '_vec_env', None), None
        if vec is not None:
            vec.close()

    @torch.no_grad()
    def rollout_candidates(self, init_states, plans, n_candidates):
        """N sequential attempts per row in the live env from ``init_states`` (B, 5), the token
        logic of predict_step. Returns dict: values (B, N) = total feedback, success (B, N),
        n_steps (B, N); plus the final stream and per-attempt snapshots compute_loss builds its
        prefixes from. ``no_grad``, not inference_mode: the stream is fed to the training forward.
        """
        from diffusion_policy.env.pusht.pusht_feedback import set_reset_states, set_veritas_plans
        from diffusion_policy.env.pusht.pusht_step_feedback import step_feedback
        vec = self._get_vec_env()
        B = len(init_states)
        n_envs = vec.num_envs
        assert B <= n_envs, f'batch {B} exceeds the rollout pool ({n_envs}); raise rollout_n_envs'
        pad = n_envs - B
        states = list(np.asarray(init_states, dtype=np.float64)) + [np.asarray(init_states[0], dtype=np.float64)] * pad
        plans = list(plans) + [plans[0]] * pad
        device = self.device
        saved = self._st
        self._st = None
        values, success, n_steps, snaps = [], [], [], []
        for _ in range(n_candidates):
            self.begin_attempt()
            set_reset_states(vec, states)
            set_veritas_plans(vec, plans)
            obs = vec.reset()
            done = np.zeros(n_envs, dtype=bool)
            steps = np.zeros(n_envs, dtype=np.int64)
            hold = np.asarray(obs['agent_pos'][:, -1], dtype=np.float32)
            while True:
                fb = step_feedback(obs['agent_pos'][:, -1], obs['feedback'][:, -1],
                                   vec.call('get_attr', 'tracker'))
                obs_t = dict_apply(obs, lambda x: torch.from_numpy(np.asarray(x)).to(device))
                obs_t['fb'] = torch.from_numpy(fb).to(device)
                obs_t['done'] = torch.from_numpy(done).to(device)
                obs_t['coverage'] = torch.as_tensor(
                    np.asarray(vec.call('get_attr', 'max_coverage'), dtype=np.float32), device=device)
                action, active = self._step(obs_t)
                active = active.cpu().numpy()
                if not active.any():
                    break
                action = np.where(active[:, None], action.cpu().numpy().astype(np.float32), hold)
                hold = action
                steps += active
                obs, _, step_done, _ = vec.step(action[:, None, :])
                done |= np.asarray(step_done, dtype=bool)
            S = self._st
            snaps.append({'len': S['len'][:B].clone(), 'last_a': S['last_a'][:B].clone(),
                          'last_fb': S['last_fb'][:B].clone()})
            values.append(S['last_fb'][:B, 1].clone())      # the attempt's total feedback
            success.append(torch.from_numpy(done[:B]).to(device))
            n_steps.append(torch.from_numpy(steps[:B]).to(device))
        S = self._st
        self._st = saved
        return {'values': torch.stack(values, 1), 'success': torch.stack(success, 1),
                'n_steps': torch.stack(n_steps, 1), 'snaps': snaps,
                'stream': {k: S[k][:B] for k in TOKEN_KEYS}, 'n_in_stream': len(S['att_starts'])}

    def _init_states_and_plans(self, obs):
        """(B, 5) env reset states and the plans of the demos a whole-episode batch came from."""
        from diffusion_policy.env.pusht.feedback_util import block_pose_from_feedback
        agent = obs['agent_pos'][:, 0].detach().cpu().numpy()
        feedback = obs['feedback'][:, 0].detach().cpu().numpy()
        poses = block_pose_from_feedback(feedback)
        plans = [self._episode_plans().lookup(a, f)[1] for a, f in zip(agent, feedback)]
        return np.concatenate([agent, poses], axis=1), plans

    # ------------------------------------------------------------------ training
    def _episode_plans(self):
        if self._plans is None:
            assert self._zarr_path is not None, 'zarr_path is needed to look up the demos\' waypoint plans'
            from diffusion_policy.env.pusht.veritas.tracker import WAYPOINT_PLAN_DIRS
            plan_dir = self._plan_dir or WAYPOINT_PLAN_DIRS[self.verifier_value]
            self._plans = EpisodePlans(self._zarr_path, plan_dir, self.verifier_value)
        return self._plans

    def _expert_feedback(self, obs, mask):
        """(B, T) step feedback along each demo, replaying its tracker (padding rows get 0)."""
        agent = obs['agent_pos'].detach().cpu().numpy()
        feedback = obs['feedback'].detach().cpu().numpy()
        lens = mask.sum(1).tolist()
        plans = self._episode_plans()
        out = np.zeros(agent.shape[:2], dtype=np.float32)
        for b, n in enumerate(lens):
            _, plan = plans.lookup(agent[b, 0], feedback[b, 0])
            out[b, :n] = demo_step_feedback(plan, agent[b, :n], feedback[b, :n])
        return torch.as_tensor(out, device=obs['agent_pos'].device)

    def compute_loss(self, batch):
        """NLL of the expert's actions, for every prefix of 0..K context attempts."""
        obs = batch['obs']
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = nactions.shape
        device = nactions.device
        mask = batch.get('attention_mask', None)
        mask = torch.ones((B, T), dtype=torch.bool, device=device) if mask is None else mask.bool()
        feat = self._obs_features(obs, n_steps=T)
        fb_t = self._expert_feedback(obs, mask)
        # expert tokens: previous action / feedback within the attempt; the first token is the boundary.
        # token t carries the feedback measured at frame t (the state a_{t-1} produced); the
        # first token of the stream carries none
        prev_a = torch.cat([torch.zeros_like(nactions[:, :1]), nactions[:, :-1]], dim=1)
        fb = self._mask_fb(torch.stack([fb_t, torch.zeros_like(fb_t)], dim=-1))
        fb[:, 0] = 0
        prev_d = torch.zeros(B, T, 1, device=device)
        steps = torch.arange(T, device=device)[None].expand(B, -1)

        K = self.max_actions - 1
        cands = None
        if K > 0:
            # FRESH context by the current weights, as maze: eval() for the rollout (center crop,
            # no dropout), then back to the mode the caller left the policy in
            was_training = self.training
            self.eval()
            init_states, plans = self._init_states_and_plans(obs)
            cands = self.rollout_candidates(init_states, plans, K)
            assert cands['n_in_stream'] == K, 'the context window slid during training'
            self.train(was_training)
            self._last_context = {k: cands[k] for k in ('values', 'success', 'n_steps')}

        seqs, idxs = [], []
        for s in range(K + 1):
            S = self._empty_stream(B, feat.shape[-1], Da, device)
            e_prev_a, e_fb, e_prev_d = prev_a, fb, prev_d
            if s > 0:
                snap = cands['snaps'][s - 1]
                L = int(snap['len'].max())
                for k in TOKEN_KEYS:
                    S[k] = cands['stream'][k][:, :L].clone()
                S['len'] = snap['len'].clone()
                e_prev_a = torch.cat([snap['last_a'][:, None], prev_a[:, 1:]], dim=1)
                e_fb = torch.cat([self._mask_fb(snap['last_fb'])[:, None], fb[:, 1:]], dim=1)
                e_prev_d = torch.cat([torch.ones(B, 1, 1, device=device), prev_d[:, 1:]], dim=1)
            off = S['len']
            idx = off[:, None] + steps
            Ltot = int((off + T).max())
            for k, v in (('feat', feat), ('prev_a', e_prev_a), ('fb', e_fb), ('prev_d', e_prev_d), ('step', steps)):
                full = torch.zeros((B, Ltot) + tuple(v.shape[2:]), dtype=v.dtype, device=device)
                full[:, :S[k].shape[1]] = S[k]
                S[k] = full.scatter(1, _expand_idx(idx, v), v)
            S['len'] = off + mask.sum(1)
            seqs.append(S)
            idxs.append(idx)
        Lmax = max(S['step'].shape[1] for S in seqs)
        cat = {k: torch.cat([F.pad(S[k], (0, 0) * (S[k].dim() - 2) + (0, Lmax - S[k].shape[1])) for S in seqs])
               for k in TOKEN_KEYS}
        cat['len'] = torch.cat([S['len'] for S in seqs])
        h = self._hidden(cat)                                                        # (B*(K+1), Lmax, E)
        h = torch.gather(h, 1, _expand_idx(torch.cat(idxs), h))                      # expert positions
        dist = self._heads(h)
        n = K + 1
        m = mask.repeat(n, 1).float()
        return -(dist.log_prob(nactions.repeat(n, 1, 1)).sum(-1) * m).sum() / m.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ workspace hooks
    @torch.inference_mode()
    def predict_action(self, obs_dict):
        """Workspace sample block: the teacher-forced action MEAN at the demo's first frame.

        ``obs_dict`` is a whole-episode batch (B, T, ...) with ``attention_mask``; the
        workspace compares ``action`` (B, 1, Da) with the expert's first action.
        """
        obs = {k: v[:, :1] for k, v in obs_dict.items() if k not in STEP_KEYS}
        feat = self._obs_features(obs)
        B, device = feat.shape[0], feat.device
        S = self._empty_stream(B, feat.shape[-1], self.action_dim, device)
        tok = {'feat': feat[:, 0], 'prev_a': torch.zeros(B, self.action_dim, device=device),
               'fb': torch.zeros(B, 2, device=device), 'prev_d': torch.zeros(B, 1, device=device),
               'step': torch.zeros(B, dtype=torch.long, device=device)}
        self._append(S, torch.ones(B, dtype=torch.bool, device=device), tok)
        action = self.normalizer['action'].unnormalize(self._heads(self._hidden(S)[:, 0]).mean)
        return {'action': action[:, None], 'action_pred': action[:, None]}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
