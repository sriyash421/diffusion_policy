"""Eval agent for PiZeroContext checkpoints inside the official Veritas loop.

Overrides only ``_sample_actions``: the candidates it returns go through the official
``select_best_action_chunk`` argmax and execution unchanged. Modes (``+ctx_mode=``):

  base_bon      N chunks from the frozen base (= the official verifier arm; the control)
  ckpt_bon      N chunks from the loaded checkpoint with no context (fine-tune-only control)
  ctx_seq       N-1 chunks from the frozen base, verifier-scored, then ONE chunk from the
                checkpoint conditioned on them (best value first): budget N, argmax over N
  ctx_tworound  N base chunks, then one conditioned chunk: budget N+1, argmax over N+1
  ctx_sliding   all N chunks from the checkpoint one after another, chunk i conditioned on
                the previous min(i, Kc) chunks and their live verifier values (generation
                order, a sliding window of Kc = 4): the search-transformer sweep over N.
                Without ``+ctx_ckpt`` the context model is untrained (LoRA at identity,
                random context encoders, attention bias ``+ctx_bias_init`` default 0): the
                "same context, no fine-tune" control.

``+interp_lambda=L`` (RICL technique 6, off at 0): the conditioned chunk becomes
``w * best_context_chunk + (1 - w) * model_chunk`` with ``w = exp(-L * c)`` and ``c`` the
best context chunk's verifier cost in units of the active waypoint's tol_px.

Context values are the step-2 verifier (``VeritasPixelVerifier``, H=4, lookahead 3, alpha
0.5) on the live plan's upcoming waypoints as ``-cost / tol_px`` (or the older per-task
z-score with ``+value_stats=``), the same quantity the model saw in training. Per episode the
row records how often the conditioned chunk won the argmax.

Collection (``+collect_decisions=<dir>``, usually with ``base_bon``): every decision's frame
(as pi0 saw it), raw proprio, TCP, targets vector, the N candidates (model and env space),
their values/costs and the official argmax's pick are stored per attempt as
``<dir>/ep{seed:06d}_a{attempt}.npz`` with the attempt's success. ``+attempts=k`` re-runs a
failed seed up to k times (same reset), so later attempts can be used as trajectory context.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

DP_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(DP_ROOT, "scripts", "simpler", "veritas_pi0")
for p in (DP_ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_pi0_eval import LoggedEvalAgent  # noqa: E402  (installs the Gemini backoff/cache)
import src.agent.eval as _eval_mod  # noqa: E402
from diffusion_policy.pi0_context.official import clip_value  # noqa: E402
from diffusion_policy.pi0_context.pizero_context import build_pizero_context  # noqa: E402
from diffusion_policy.policy.veritas_verifier import VeritasPixelVerifier  # noqa: E402

MODES = ("base_bon", "ckpt_bon", "ctx_seq", "ctx_tworound", "ctx_sliding")


class ContextEvalAgent(LoggedEvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ctx_mode = cfg.get("ctx_mode", "base_bon")
        assert self.ctx_mode in MODES, self.ctx_mode
        self.interp_lambda = float(cfg.get("interp_lambda", 0.0))
        # +force_n_ctx=0: the conditioned chunk sees an empty context (budget-matched mixed
        # control: 4 base + 1 fine-tuned chunk without context)
        self.force_n_ctx = cfg.get("force_n_ctx", None)
        self.ctx_model = None
        if self.ctx_mode != "base_bon":
            from omegaconf import open_dict
            with open_dict(cfg):   # the trainer's default; overwritten by a loaded checkpoint
                cfg.context_attn_bias_init = float(cfg.get("ctx_bias_init", 0.0))
            self.ctx_model = build_pizero_context(cfg, cfg.checkpoint_path, lora=True,
                                                  device=self.device, dtype=self.dtype)
            ckpt = cfg.get("ctx_ckpt", None)
            if ckpt:
                ck = torch.load(ckpt, map_location="cpu", weights_only=False)
                res = self.ctx_model.load_state_dict(ck["trainable"], strict=False)
                assert not res.unexpected_keys, res.unexpected_keys
                self.ctx_step = int(ck.get("step", -1))
            else:
                self.ctx_step = 0   # untrained (smoke only)
            self.ctx_model.eval()
            self.ctx_blocks = self.ctx_model.num_context_blocks
        # feedback scalar: task-agnostic -cost/tol_px (pixel error in units of the active
        # waypoint's tolerance); +value_stats=<json> switches to the older per-task z-score
        self.value_stats = json.load(open(cfg.value_stats)) if cfg.get("value_stats") else None
        self.ctx_verifier = VeritasPixelVerifier(
            camera_file=cfg.get("camera_file", os.path.join(
                DP_ROOT, "diffusion_policy/config/veritas", f"camera_{cfg.env.task}.json")),
            lookahead_k=self.lookahead_k, lookahead_alpha=self.lookahead_alpha, default_tol_px=60.0,
            horizon=4, normalize="none", plan_done_mode="zero", health_check=False)
        # the official loop keeps obs local; the adapter sees it right before sampling
        self._obs = None
        adapter, orig_pre = self.env_adapter, self.env_adapter.preprocess

        def preprocess(env, obs, instruction):
            self._obs = obs
            return orig_pre(env, obs, instruction)
        adapter.preprocess = preprocess
        # record which candidate the official argmax picks
        orig_select = _eval_mod.select_best_action_chunk

        def select(*a, **k):
            out = orig_select(*a, **k)
            self._last_best = int(out[0])
            self._ep["n_decisions"] += 1
            if self._last_candidates and self._last_best == self._last_candidates - 1 \
                    and self.ctx_mode in ("ctx_seq", "ctx_tworound"):
                self._ep["n_ctx_chosen"] += 1
            if self._pending is not None:
                self._pending["best_idx"] = self._last_best
                self._decisions.append(self._pending)
                self._pending = None
            return out
        _eval_mod.select_best_action_chunk = select
        self._last_candidates = 0
        self._reset_ep()
        # collection
        self.collect_dir = cfg.get("collect_decisions", None)
        self.attempts = int(cfg.get("attempts", 1))
        if self.collect_dir:
            os.makedirs(self.collect_dir, exist_ok=True)
        self._pending = None
        self._decisions = []
        self._attempt = 1
        self._attempt_results = []
        print(f"[ContextEvalAgent] mode={self.ctx_mode} N={self.verifier_samples} "
              f"interp_lambda={self.interp_lambda} ckpt={cfg.get('ctx_ckpt')}", flush=True)
        self.wandb.config.update(dict(ctx_mode=self.ctx_mode, interp_lambda=self.interp_lambda,
                                      ctx_ckpt=cfg.get("ctx_ckpt"), ctx_step=getattr(self, "ctx_step", None)),
                                 allow_val_change=True)

    def _reset_ep(self):
        self._ep = dict(n_decisions=0, n_ctx_chosen=0, base_value_sum=0.0, ctx_value_sum=0.0,
                        best_base_value_sum=0.0, n_scored=0)

    def _extra_row_fields(self):
        e, self._ep = self._ep, None
        n = max(1, e["n_scored"])
        out = dict(ctx_mode=self.ctx_mode, interp_lambda=self.interp_lambda,
                   n_decisions=e["n_decisions"], n_ctx_chosen=e["n_ctx_chosen"],
                   mean_base_value=e["base_value_sum"] / n, mean_best_base_value=e["best_base_value_sum"] / n,
                   mean_ctx_value=e["ctx_value_sum"] / n)
        if self.attempts > 1:
            out["attempt_successes"] = list(self._attempt_results)
            out["success_within_attempts"] = bool(any(self._attempt_results))
        self._reset_ep()
        return out

    # ------------------------------------------------------------------ collection
    def _record_decision(self, inputs, raws, chunks, values, costs, tol):
        pv = inputs["pixel_values"][0].float().cpu().numpy()                 # (3,224,224), (x/255-0.5)/0.5
        image = np.clip(np.round((pv * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8).transpose(1, 2, 0)
        tv, _ = self._targets_vector()
        self._pending = dict(image=image, proprio=np.asarray(self.env_adapter.preprocess_proprio(self._obs), np.float32),
                             tcp=np.asarray(self._obs["extra"]["tcp_pose"], np.float32)[:3], targets=tv, tol=np.float32(tol),
                             cand_norm=raws.astype(np.float32), cand_env=np.stack(chunks).astype(np.float32),
                             values=values.astype(np.float32), costs=costs.astype(np.float32))

    def _write_attempt(self, seed, success):
        if not self.collect_dir or seed is None:
            return
        path = os.path.join(self.collect_dir, f"ep{int(seed):06d}_a{self._attempt}.npz")
        d = self._decisions
        if not d:
            return
        np.savez_compressed(path, seed=int(seed), attempt=self._attempt, success=bool(success),
                            steps=int(self.frame_idx), task=self.cfg.env.task,
                            instruction=self.env.get_language_instruction(), n_decisions=len(d),
                            **{k: np.stack([x[k] for x in d]) for k in d[0]})
        self._decisions = []
        self._pending = None

    def _after_episode(self, cnt, seed, success):
        self._attempt_results = [bool(success)]
        self._write_attempt(seed, success)
        while not success and self._attempt < self.attempts and seed is not None:
            self._attempt += 1
            self._seeded_reset(seed)
            success = bool(self._run_episode(cnt))
            self._attempt_results.append(success)
            self._write_attempt(seed, success)
        self._attempt = 1

    # ------------------------------------------------------------------ scoring
    def _targets_vector(self):
        upcoming, tol = self._veritas_active_waypoint()
        v = np.full(2 + 2 * self.lookahead_k, -1.0, dtype=np.float32)
        if upcoming:
            v[0] = len(upcoming)
            v[1] = tol if tol and tol > 0 else 60.0
            for k, uv in enumerate(upcoming[:self.lookahead_k]):
                v[2 + 2 * k:4 + 2 * k] = uv
        else:
            v[0] = 0
        return v, (tol if tol and tol > 0 else 60.0)

    def _score(self, env_chunks):
        """Normalised verifier values and raw costs of env-format chunks on the live plan."""
        tv, tol = self._targets_vector()
        n = len(env_chunks)
        if not getattr(self, "_camera_checked", False):
            self.ctx_verifier.camera.assert_matches(self._obs)   # raises if the live camera differs
            self._camera_checked = True
        obs = dict(end_effector_pose=torch.as_tensor(np.repeat(np.asarray(self._obs["extra"]["tcp_pose"])[None, :3], n, 0)),
                   veritas_targets=torch.as_tensor(np.repeat(tv[None], n, 0)))
        act = torch.as_tensor(np.stack(env_chunks), dtype=torch.float32)
        cost = self.ctx_verifier.cost(obs, act).numpy()
        if self.value_stats:
            value = (-cost - self.value_stats["value_mean"]) / self.value_stats["value_std"]
        else:
            value = clip_value(-cost / tol)   # same clipping as the training datasets
        return value, cost, tol

    # ------------------------------------------------------------------ sampling
    def _base_chunks(self, inputs, n):
        batched = {k: (v.expand(n, *v.shape[1:]).contiguous() if torch.is_tensor(v) and v.size(0) == 1 else v)
                   for k, v in inputs.items()}
        with torch.inference_mode():
            raws = self.model(**batched).float().cpu().numpy()
        return raws, [self.env_adapter.postprocess(raws[i]) for i in range(n)]

    def _ckpt_chunks(self, inputs, n, ctx_actions=None, ctx_values=None, n_ctx=0, prefix=None):
        attention_mask = (inputs["input_ids"] != self.ctx_model.pad_token_id).long()
        rep = lambda t: t.expand(n, *t.shape[1:]).contiguous()
        with torch.inference_mode():
            out = self.ctx_model.infer_action_with_context(
                rep(inputs["input_ids"]), rep(inputs["pixel_values"]), rep(attention_mask), rep(inputs["proprios"]),
                None if ctx_actions is None else ctx_actions.to(self.device, self.dtype),
                None if ctx_values is None else ctx_values.to(self.device, self.dtype),
                torch.full((n,), int(n_ctx), device=self.device, dtype=torch.long), prefix_caches=prefix)
        raws = out.float().cpu().numpy()
        return raws, [self.env_adapter.postprocess(raws[i]) for i in range(n)]

    def _sliding_chunks(self, inputs, N):
        """N chunks from the checkpoint, chunk i conditioned on the previous min(i, Kc)."""
        Kc, H, A = self.ctx_blocks, self.ctx_model.horizon_steps, self.ctx_model.action_dim
        attention_mask = (inputs["input_ids"] != self.ctx_model.pad_token_id).long()
        with torch.inference_mode():
            prefix = self.ctx_model.prefix_caches(inputs["input_ids"], inputs["pixel_values"],
                                                  attention_mask, inputs["proprios"])
        raws, chunks, values, costs = [], [], [], []
        tol = None
        for i in range(N):
            n_ctx = min(i, Kc) if self.force_n_ctx is None else min(int(self.force_n_ctx), i)
            ctx_a = np.zeros((1, Kc, H, A), np.float32)
            ctx_v = np.zeros((1, Kc), np.float32)
            if n_ctx:
                ctx_a[0, :n_ctx] = np.stack(raws[i - n_ctx:i])
                ctx_v[0, :n_ctx] = values[i - n_ctx:i]
            raw, chunk = self._ckpt_chunks(inputs, 1, torch.as_tensor(ctx_a), torch.as_tensor(ctx_v), n_ctx, prefix)
            v, c, tol = self._score(chunk)
            raws.append(raw[0]); chunks.append(chunk[0]); values.append(float(v[0])); costs.append(float(c[0]))
        return np.stack(raws), chunks, np.array(values, np.float32), np.array(costs, np.float32), tol

    def _sample_actions(self, inputs):
        N = self.verifier_samples
        if self.ctx_mode in ("base_bon", "ckpt_bon"):
            raws, chunks = (self._base_chunks if self.ctx_mode == "base_bon" else self._ckpt_chunks)(inputs, N)
            self._last_candidates = N
            if self.collect_dir:
                values, costs, tol = self._score(chunks)
                self._record_decision(inputs, raws, chunks, values, costs, tol)
            return chunks
        if self.ctx_mode == "ctx_sliding":
            raws, chunks, values, costs, tol = self._sliding_chunks(inputs, N)
            if self.collect_dir:
                self._record_decision(inputs, raws, chunks, values, costs, tol)
            e = self._ep
            e["ctx_value_sum"] += float(values.mean()); e["n_scored"] += 1
            self._last_candidates = N
            return chunks
        n_base = N - 1 if self.ctx_mode == "ctx_seq" else N
        raws, chunks = self._base_chunks(inputs, n_base)
        values, costs, tol = self._score(chunks)
        if self.collect_dir:
            self._record_decision(inputs, raws, chunks, values, costs, tol)
        Kc = self.ctx_blocks
        # context in generation order (search-transformer convention: slot k sees the first k)
        n_ctx = min(n_base, Kc)
        if self.force_n_ctx is not None:
            n_ctx = int(self.force_n_ctx)
        ctx_a = np.zeros((1, Kc, raws.shape[1], raws.shape[2]), np.float32)
        ctx_v = np.zeros((1, Kc), np.float32)
        ctx_a[0, :n_ctx] = raws[:n_ctx]
        ctx_v[0, :n_ctx] = values[:n_ctx]
        raw_ctx, _ = self._ckpt_chunks(inputs, 1, torch.as_tensor(ctx_a), torch.as_tensor(ctx_v), n_ctx)
        raw_ctx = raw_ctx[0]
        if self.interp_lambda > 0:
            best = int(np.argmax(values))
            w = float(np.exp(-self.interp_lambda * costs[best] / tol))
            raw_ctx = w * raws[best] + (1 - w) * raw_ctx
        ctx_chunk = self.env_adapter.postprocess(raw_ctx)
        cv, _, _ = self._score([ctx_chunk])
        e = self._ep
        e["base_value_sum"] += float(values.mean()); e["best_base_value_sum"] += float(values.max())
        e["ctx_value_sum"] += float(cv[0]); e["n_scored"] += 1
        self._last_candidates = n_base + 1
        return chunks + [ctx_chunk]
