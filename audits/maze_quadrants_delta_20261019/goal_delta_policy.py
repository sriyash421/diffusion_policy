"""ST with four feedback channels: step dx/dy and trajectory-end dx/dy.
Grid-normalized displacement; negative Euclidean endpoint distance ranks candidates.
"""
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.maze_verifier import (
    build_images_torch,
    rollout_step_torch,
    snap_cardinal_torch,
)

TOKEN_KEYS = ('feat', 'prev_a', 'fb', 'prev_d', 'step')


def _expand_idx(idx, like):
    """(B, L) gather/scatter index broadcast to the trailing dims of `like`."""
    return idx.reshape(idx.shape + (1,) * (like.dim() - 2)).expand(-1, -1, *like.shape[2:])


class GoalDeltaPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: Dict[str, Any],
            obs_encoder: nn.Module,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            max_actions: int = 8,
            reward_mode: str = 'traj',
            stop_radius: float = 1.0,
            n_emb: int = 256,
            n_layer: int = 4,
            n_head: int = 4,
            p_drop: float = 0.1,
            log_std_init: float = -1.0,
            log_std_min: float = -2.0,
            snap_cardinal: bool = True,
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        assert n_obs_steps == 1, 'one token per env step; use n_obs_steps=1'
        assert reward_mode in ('traj', 'step'), reward_mode
        self.obs_encoder = obs_encoder
        self.horizon = horizon            # unused: attempts are variable length
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_shape[0]
        self.max_actions = max_actions
        self.reward_mode = reward_mode
        self.stop_radius = stop_radius
        self.snap_cardinal = bool(snap_cardinal)
        self.n_emb = n_emb
        self.obs_feature_dim = obs_encoder.output_shape()[0]
        self.normalizer = LinearNormalizer()
        self.kwargs = kwargs
        self.obs_proj = nn.Linear(self.obs_feature_dim, n_emb)
        self.act_emb = nn.Linear(self.action_dim, n_emb)
        self.fb_emb = nn.Linear(4, n_emb)
        self.done_emb = nn.Linear(1, n_emb)
        self.pe = SinusoidalPosEmb(n_emb)   # no table, no maximum length
        self.step_proj = nn.Linear(n_emb, n_emb)
        self.pos_proj = nn.Linear(n_emb, n_emb)
        self.drop = nn.Dropout(p_drop)
        layer = nn.TransformerEncoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb, dropout=p_drop,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(n_emb)
        self.mean_head = nn.Linear(n_emb, self.action_dim)
        self.stop_head = nn.Linear(n_emb, 1)
        # state-independent log_std (as the LSTM arm); the floor keeps candidates distinct
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(log_std_init)))
        self.log_std_min = float(log_std_min)

        self._scale_cache = {}
        self._st = None   # the episode stream used by predict_step / the torch rollout

    # ------------------------------------------------------------------ pieces
    def _std(self):
        return self.log_std.clamp_min(self.log_std_min).exp()

    @staticmethod
    def goal_delta(pos, goal, free):
        scale = pos.new_tensor([free.shape[-1]-1, free.shape[-2]-1])
        return (goal - pos) / scale

    def _embed(self, S):
        B, L = S['step'].shape
        x = (self.obs_proj(S['feat']) + self.act_emb(S['prev_a']) + self.fb_emb(S['fb'])
             + self.done_emb(S['prev_d']))
        x = x + self.step_proj(self.pe(S['step'].reshape(-1).float()).reshape(B, L, -1))
        x = x + self.pos_proj(self.pe(torch.arange(L, device=x.device).float()))[None]
        return self.drop(x)

    def _hidden(self, S):
        """(B, L, n_emb) causal hidden states; positions >= len are padding."""
        x = self._embed(S)
        L = x.shape[1]
        causal = torch.ones((L, L), dtype=torch.bool, device=x.device).triu(1)
        pad = torch.arange(L, device=x.device)[None] >= S['len'][:, None]
        return self.ln_f(self.encoder(x, mask=causal, src_key_padding_mask=pad))

    def _heads(self, h):
        return Normal(self.mean_head(h), self._std()), self.stop_head(h)[..., 0]

    def _obs_features(self, obs_dict):
        return self.obs_encoder(self.normalizer.normalize(obs_dict))

    def _mask_fb(self, fb):
        if self.reward_mode == 'traj':
            return torch.cat([torch.zeros_like(fb[..., :2]), fb[..., 2:]], dim=-1)
        return fb

    # ------------------------------------------------------------------ stream
    @staticmethod
    def _empty_stream(B, Fo, Da, device):
        z = lambda *shape, dtype=torch.float32: torch.zeros((B, 0) + shape, dtype=dtype, device=device)
        return {'feat': z(Fo), 'prev_a': z(Da), 'fb': z(4), 'prev_d': z(1), 'step': z(dtype=torch.long),
                'len': torch.zeros(B, dtype=torch.long, device=device), 'att_starts': [],
                'has_prev': False, 't': None,
                'last_a': torch.zeros(B, Da, device=device), 'last_fb': torch.zeros(B, 4, device=device)}

    @staticmethod
    def _append(S, rows, tok):
        """Write one token at S['len'] for the rows in the bool mask; the others get padding."""
        if not bool(rows.any()):
            return
        for k in TOKEN_KEYS:
            S[k] = torch.cat([S[k], S[k].new_zeros((S[k].shape[0], 1) + tuple(S[k].shape[2:]))], dim=1)
            S[k][rows, S['len'][rows]] = tok[k][rows]
        S['len'] = S['len'] + rows.long()

    @staticmethod
    def _slide(S, drop):
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

    def predict_step(self, obs_dict):
        """One env step for a batch of rows: raw obs (B, To, ...) -> (action (B, Da), active (B,)).

        Rows whose attempt has ended (model stop, goal within stop_radius, guard) get active=False
        and a zero action; call until no row is active so the last rows' feedback is settled.
        """
        obs = {k: v[:, -1] for k, v in obs_dict.items() if k != 'attention_mask'}
        return self._step(obs, obs['image'][:, 0] < 0.5)

    def _step(self, obs, free):
        pos = obs['agent_pos'].float()
        goal = obs['goal_pos'].float()
        B = pos.shape[0]
        device = pos.device
        feat = self._obs_features(obs)
        if self._st is None:
            self._st = self._empty_stream(B, feat.shape[1], self.action_dim, device)
            self._st['att_starts'].append(self._st['len'].clone())
        S = self._st
        first = S['t'] is None
        if first:
            S['guard'] = free.flatten(1).sum(1)
            S['t'] = torch.zeros(B, dtype=torch.long, device=device)
            S['active'] = torch.ones(B, dtype=torch.bool, device=device)
            S['pos'] = pos
            S['stop'] = torch.zeros(B, dtype=torch.bool, device=device)
        delta = self.goal_delta(pos, goal, free)
        realized = self.normalizer['action'].normalize(pos - S['pos'])
        S['pos'] = pos
        if not first:
            # settle the previous step: rows that reached the goal, stopped or hit the guard are done
            reached = (pos - goal).norm(dim=-1) <= self.stop_radius
            ended = S['active'] & (reached | S['stop'] | (S['t'] >= S['guard']))
            S['last_a'] = torch.where(ended[:, None], realized, S['last_a'])
            S['last_fb'] = torch.where(ended[:, None], torch.cat([delta, delta], dim=-1), S['last_fb'])
            S['active'] = S['active'] & ~ended
        active = S['active']
        zeros = torch.zeros(B, 1, device=device)
        if first and S['has_prev']:
            tok = {'prev_a': S['last_a'], 'fb': S['last_fb'], 'prev_d': torch.ones(B, 1, device=device)}
        elif first:
            tok = {'prev_a': torch.zeros(B, self.action_dim, device=device), 'fb': zeros.expand(-1, 4), 'prev_d': zeros}
        else:
            tok = {'prev_a': realized, 'fb': torch.cat([delta, torch.zeros_like(delta)], dim=-1), 'prev_d': zeros}
        tok.update(feat=feat, fb=self._mask_fb(tok['fb']), step=S['t'])
        self._append(S, active, tok)
        h = self._hidden(S)[torch.arange(B, device=device), (S['len'] - 1).clamp_min(0)]
        dist, stop_logit = self._heads(h)
        action = self.normalizer['action'].unnormalize(dist.sample())
        if self.snap_cardinal:
            action = snap_cardinal_torch(action)
        action = torch.where(active[:, None], action, torch.zeros_like(action))
        S['stop'] = torch.sigmoid(stop_logit) > 0.5
        S['t'] = S['t'] + active.long()
        return action, active.clone()

    # ------------------------------------------------------------------ rollout (torch env)
    @torch.inference_mode()
    def rollout_candidates_torch(self, free, start, goal, n_candidates, iid=False):
        """N attempts per row in the torch env, sequential (one stream) unless `iid`.

        Returns dict: traj (B,N,T+1,2) frozen after the end, realized (B,N,T,Da), n_steps (B,N),
        done_step (B,N) (-1 if the goal was not reached), values (B,N) = -Euclidean(final, goal); plus the
        final stream and per-attempt snapshots that compute_loss builds its prefixes from.
        """
        if iid:
            N = n_candidates
            rep = lambda x: x.repeat_interleave(N, dim=0)
            out = self.rollout_candidates_torch(rep(free), rep(start), rep(goal), 1)
            return {k: v.reshape(start.shape[0], N, *v.shape[2:]) for k, v in out.items()
                    if k not in ('snaps', 'stream', 'n_in_stream')}
        B = start.shape[0]
        device = start.device
        saved = self._st
        self._st = None
        trajs, n_steps, done_step, snaps = [], [], [], []
        for _ in range(n_candidates):
            self.begin_attempt()
            pos = start.float()
            traj = [pos]
            ds = torch.full((B,), -1, dtype=torch.long, device=device)
            t = 0
            while True:
                obs = {'image': build_images_torch(free, pos, goal), 'agent_pos': pos, 'goal_pos': goal}
                action, active = self._step(obs, free)
                if not bool(active.any()):   # this call settled the last rows
                    break
                pos = torch.where(active[:, None], rollout_step_torch(free, pos, action), pos)
                traj.append(pos)
                t += 1
                reached = active & (ds < 0) & ((pos - goal).norm(dim=-1) <= self.stop_radius)
                ds = torch.where(reached, torch.full_like(ds, t), ds)
            S = self._st
            snaps.append({'len': S['len'].clone(), 'last_a': S['last_a'].clone(), 'last_fb': S['last_fb'].clone()})
            trajs.append(torch.stack(traj, dim=1))
            n_steps.append(S['t'].clone())
            done_step.append(ds)
        T = max(tr.shape[1] for tr in trajs) - 1
        traj = torch.stack([torch.cat([tr, tr[:, -1:].expand(-1, T + 1 - tr.shape[1], -1)], dim=1)
                            for tr in trajs], dim=1)
        values = -(traj[:, :, -1] - goal[:, None]).norm(dim=-1)
        S = self._st
        self._st = saved
        return {'traj': traj, 'realized': traj[:, :, 1:] - traj[:, :, :-1], 'n_steps': torch.stack(n_steps, 1),
                'done_step': torch.stack(done_step, 1), 'values': values,
                'snaps': snaps, 'stream': {k: S[k] for k in TOKEN_KEYS}, 'n_in_stream': len(S['att_starts'])}

    # ------------------------------------------------------------------ training
    def compute_loss(self, batch):
        """NLL of the expert's actions + BCE of its stop flag, for every prefix of 0..K context attempts."""
        obs = batch['obs']
        nactions = self.normalizer['action'].normalize(batch['action'])
        B, T, Da = nactions.shape
        device = nactions.device
        mask = batch.get('attention_mask', None)
        mask = torch.ones((B, T), dtype=torch.bool, device=device) if mask is None else mask.bool()
        feat = self._obs_features(dict_apply(obs, lambda x: x.reshape(-1, *x.shape[2:]))).reshape(B, T, -1)
        free = obs['image'][:, 0, 0] < 0.5
        start, goal = obs['agent_pos'][:, 0], obs['goal_pos'][:, 0]
        delta = self.goal_delta(obs['agent_pos'], goal[:, None], free)
        # expert tokens: previous action / feedback within the attempt; the first token is the boundary
        prev_a = torch.cat([torch.zeros_like(nactions[:, :1]), nactions[:, :-1]], dim=1)
        fb = self._mask_fb(torch.cat([delta, torch.zeros_like(delta)], dim=-1))
        fb[:, 0] = 0
        prev_d = torch.zeros(B, T, 1, device=device)
        steps = torch.arange(T, device=device)[None].expand(B, -1)
        stop_target = mask & ~torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)

        K = self.max_actions - 1
        if K > 0:
            self.eval()
            cands = self.rollout_candidates_torch(free, start, goal, K)
            assert cands['n_in_stream'] == K, 'the context window slid during training'
        self.train(torch.is_grad_enabled())   # the workspace leaves the policy in eval() after validation

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
        dist, stop_logit = self._heads(h)
        n = K + 1
        m = mask.repeat(n, 1).float()
        nll = -(dist.log_prob(nactions.repeat(n, 1, 1)).sum(-1) * m).sum() / m.sum().clamp_min(1.0)
        bce = (F.binary_cross_entropy_with_logits(stop_logit, stop_target.repeat(n, 1).float(), reduction='none')
               * m).sum() / m.sum().clamp_min(1.0)
        return nll + bce

    # ------------------------------------------------------------------ workspace hooks
    @torch.inference_mode()
    def predict_action(self, obs_dict, verifier=None, n_actions=None):
        """Workspace sample block: (realized (B, N, T, Da), values (B, N)) of N torch-env attempts."""
        n_actions = self.max_actions if n_actions is None else n_actions
        free = obs_dict['image'][:, 0, 0] < 0.5
        cands = self.rollout_candidates_torch(free, obs_dict['agent_pos'][:, 0], obs_dict['goal_pos'][:, 0],
                                              n_actions, iid=self.max_actions == 1)
        return cands['realized'], cands['values']

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
