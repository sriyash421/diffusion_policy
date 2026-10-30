#!/usr/bin/env python
"""open-pi-zero in SimplerEnv through the official Veritas EvalAgent, plus per-episode JSONL,
resume and wandb.

Thin subclass of ``src.agent.eval.EvalAgent`` from the princeton-prism/veritas checkout at
``$VERITAS_ROOT`` (default /gscratch/robotics/harine/veritas). That checkout is never edited.
The config is the official ``config/eval/bridge.yaml`` composed with the hydra overrides given
on the command line, the same way ``scripts/run.py`` there does, so every official knob
(``env.task``, ``checkpoint_path``, ``+use_veritas_verifier``, ``+verifier_samples``,
``+collect_demos``, ``use_bf16``, ...) applies unchanged.

Extra overrides (``+key=value``):
  arm            wandb run name; default ``raw_n1`` / ``veritas_n5`` from the verifier flags
  eval_seeds     JSON list of env seeds: reset by seed instead of the 24-variation loop
                 (the seeded step-2 protocol); n_eval_episode becomes len(seeds)
  wandb_project  default ``simpler_veritas``; wandb_entity default ``l2sml``
  wandb_mode     ``online`` (default) | ``offline`` | ``disabled``
  wandb_id       stable run id; default derived from (arm, log_dir) so a requeued job resumes

Output in ``log_dir``: ``episodes.jsonl`` (one row per finished episode; rows already present
are skipped on restart), the official ``videos/`` and ``verified_data/`` and ``summary.json``.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

VERITAS_ROOT = os.environ.get("VERITAS_ROOT", "/gscratch/robotics/harine/veritas")
if VERITAS_ROOT not in sys.path:
    sys.path.insert(0, VERITAS_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from src.agent.eval import EvalAgent  # noqa: E402
from src.agent.utils import gemini_client as _gemini_client  # noqa: E402


# The official retry loop sleeps only on 429/quota errors; a 503 "high demand" burst gets
# three back-to-back attempts and then silently falls back to an EMPTY plan, which would turn
# verifier episodes into unguided ones. Wrap the raw call with exponential backoff instead.
_TRANSIENT = ("503", "UNAVAILABLE", "high demand", "429", "RESOURCE_EXHAUSTED",
              "500 INTERNAL", "502", "504", "DEADLINE_EXCEEDED")
_orig_generate = _gemini_client.GeminiClient._generate


def _generate_with_backoff(self, parts, generation_config, _orig=_orig_generate,
                           _max=8, _delay0=5.0):
    delay = _delay0
    for attempt in range(_max):
        try:
            return _orig(self, parts, generation_config)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if attempt == _max - 1 or not any(k in msg for k in _TRANSIENT):
                raise
            print(f"[gemini backoff] {msg[:90]!r} -> retry {attempt + 1}/{_max} in {delay:.0f}s",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120.0)



# Content-addressed cache of Gemini responses (GEMINI_CACHE_DIR): the seeded protocol resets
# to identical first frames, so every arm gets identical object lists and plans. Only
# JSON-parseable responses are stored, so a bad reply is retried live, never replayed.
_GEMINI_CACHE_DIR = os.environ.get("GEMINI_CACHE_DIR")


class _CachedResponse:
    def __init__(self, text):
        self.text = text


def _generate_cached(self, parts, generation_config):
    if not _GEMINI_CACHE_DIR:
        return _generate_with_backoff(self, parts, generation_config)
    h = hashlib.sha1(str(self.model_name).encode())
    for p in parts:
        if isinstance(p, str):
            h.update(p.encode())
        else:  # PIL image
            h.update(np.asarray(p).tobytes())
    h.update(json.dumps(generation_config, sort_keys=True, default=str).encode())
    path = os.path.join(_GEMINI_CACHE_DIR, h.hexdigest()[:24] + ".json")
    if os.path.exists(path):
        with open(path) as f:
            return _CachedResponse(json.load(f)["text"])
    resp = _generate_with_backoff(self, parts, generation_config)
    try:
        json.loads(resp.text)
    except Exception:
        return resp
    os.makedirs(_GEMINI_CACHE_DIR, exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(dict(model=str(self.model_name), text=resp.text,
                       prompt=[p for p in parts if isinstance(p, str)]), f)
    os.replace(tmp, path)
    return resp


_gemini_client.GeminiClient._generate = _generate_cached


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def _git_head(path: str) -> str:
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return 'unknown'


class LoggedEvalAgent(EvalAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.eval_seeds = None
        self.seed_split = {}
        seeds_file = cfg.get("eval_seeds", None)
        if seeds_file:
            with open(seeds_file) as f:
                seeds = json.load(f)
            if isinstance(seeds, dict):   # {"test": [...], "train20": [...]}
                self.seed_split = {int(s): name for name, lst in seeds.items() for s in lst}
                seeds = [s for lst in seeds.values() for s in lst]
            self.eval_seeds = [int(s) for s in seeds]
            self.n_eval_episode = len(self.eval_seeds)
        kind = "veritas" if self.use_veritas_verifier else "raw"
        self.arm = cfg.get("arm", None) or f"{kind}_n{self.verifier_samples}"
        self.episodes_path = os.path.join(self.log_dir, "episodes.jsonl")
        self.rows = self._read_rows()
        self._init_wandb()

    # ------------------------------------------------------------------ bookkeeping
    def _read_rows(self) -> dict:
        rows = {}
        if os.path.exists(self.episodes_path):
            with open(self.episodes_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        r = json.loads(line)
                        rows[int(r["episode"])] = r
        return rows

    def _init_wandb(self):
        import wandb
        cfg = self.cfg
        mode = cfg.get("wandb_mode", "online")
        run_id = cfg.get("wandb_id", None) or hashlib.sha1(
            f"{self.arm}|{self.log_dir}".encode()).hexdigest()[:16]
        config = OmegaConf.to_container(cfg, resolve=True)
        config.update(dict(
            arm=self.arm,
            gemini_model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
            veritas_commit=_git_head(VERITAS_ROOT),
            n_eval_episode=self.n_eval_episode,
            seeded=self.eval_seeds is not None,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        ))
        tags = [cfg.env.task, "verifier" if self.use_veritas_verifier else "raw",
                f"n{self.verifier_samples}"]
        self.wandb = wandb.init(
            entity=cfg.get("wandb_entity", "l2sml"),
            project=cfg.get("wandb_project", "simpler_veritas"),
            name=self.arm, id=run_id, resume="allow", mode=mode,
            group=cfg.get("wandb_group", cfg.env.task), tags=tags,
            config=config, dir=self.log_dir)
        if self.rows:
            print(f"[LoggedEvalAgent] resuming: {len(self.rows)} episodes already in "
                  f"{self.episodes_path}")

    def _append_row(self, row: dict):
        with open(self.episodes_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rows[row["episode"]] = row

    # ------------------------------------------------------------------ seeded resets
    def _seeded_reset(self, seed: int):
        """Make the next env.reset() use ``seed`` (ignoring the official episode_id options)."""
        env = self.env
        orig = env.reset

        def reset(*args, **kwargs):
            env.reset = orig
            return orig(seed=int(seed))

        env.reset = reset

    # ------------------------------------------------------------------ loop
    def run(self):
        n_var = self.n_variations
        for cnt in range(self.n_eval_episode):
            if cnt in self.rows:
                continue
            seed = None
            if self.eval_seeds is not None:
                seed = self.eval_seeds[cnt]
                self._seeded_reset(seed)
            t0 = time.time()
            success = bool(self._run_episode(cnt))
            self._after_episode(cnt, seed, success)
            # plan actually used (the verifier keeps it until the next episode starts);
            # 0 waypoints on a verifier run means Gemini fell back to an empty plan.
            plan = getattr(getattr(self, "veritas_verifier", None), "plan", None)
            n_waypoints = len(plan.waypoints) if plan is not None and plan.waypoints else 0
            row = dict(
                episode=cnt,
                variation=None if seed is not None else cnt % n_var,
                trial=None if seed is not None else cnt // n_var,
                seed=seed,
                success=success,
                steps=int(self.frame_idx),
                wall_s=round(time.time() - t0, 1),
                instruction=self.env.get_language_instruction(),
                task=self.cfg.env.task,
                verifier=bool(self.use_veritas_verifier),
                n_samples=int(self.verifier_samples),
                n_waypoints=n_waypoints,
                split=self.seed_split.get(seed) if seed is not None else None,
                time=_dt.datetime.now().isoformat(timespec="seconds"),
            )
            row.update(self._extra_row_fields())
            self._append_row(row)
            done = sorted(self.rows)
            k = sum(self.rows[i]["success"] for i in done)
            self.wandb.log({
                "episode/success": int(success),
                "episode/steps": row["steps"],
                "episode/wall_s": row["wall_s"],
                "episode/n_waypoints": n_waypoints,
                "success_rate": k / len(done),
                "n_done": len(done),
            }, step=cnt)
            print(f"[{self.arm}] ep {cnt}/{self.n_eval_episode} success={success} "
                  f"steps={row['steps']} running SR={k/len(done):.3f} ({k}/{len(done)})", flush=True)
        self._finish()

    def _extra_row_fields(self) -> dict:
        """Subclasses add per-episode fields here (called once per finished episode)."""
        return {}

    def _after_episode(self, cnt: int, seed, success: bool) -> None:
        """Hook right after an episode, before its row is written (extra attempts, dumps)."""

    def _finish(self):
        import wandb
        done = sorted(self.rows)
        n = len(done)
        k = sum(self.rows[i]["success"] for i in done)
        lo, hi = wilson_ci(k, n)
        summary = dict(arm=self.arm, task=self.cfg.env.task, n_episodes=n, n_success=k,
                       success_rate=k / n if n else float("nan"), ci95_low=lo, ci95_high=hi,
                       mean_steps=float(np.mean([self.rows[i]["steps"] for i in done])) if n else None,
                       n_empty_plan=(sum(1 for i in done if self.rows[i].get("n_waypoints", -1) == 0)
                                     if self.use_veritas_verifier else None),
                       verifier=bool(self.use_veritas_verifier), n_samples=int(self.verifier_samples))
        if self.seed_split:
            for name in sorted(set(self.seed_split.values())):
                sub = [i for i in done if self.rows[i].get("split") == name]
                ks = sum(self.rows[i]["success"] for i in sub)
                summary[f"{name}_n"] = len(sub)
                summary[f"{name}_success_rate"] = ks / len(sub) if sub else None
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        for key, val in summary.items():
            self.wandb.summary[f"summary/{key}"] = val
        if self.eval_seeds is None:
            per_var = {}
            for i in done:
                v = self.rows[i]["variation"]
                per_var.setdefault(v, []).append(self.rows[i]["success"])
            table = wandb.Table(columns=["variation", "n", "success_rate"])
            for v in sorted(per_var):
                table.add_data(v, len(per_var[v]), float(np.mean(per_var[v])))
            self.wandb.log({"per_variation": table})
        print("============ Evaluation Summary ============")
        print(json.dumps(summary, indent=2))
        self.wandb.finish()


def main(argv=None, agent_cls=None):
    from hydra import compose, initialize_config_dir
    agent_cls = agent_cls or LoggedEvalAgent
    argv = sys.argv[1:] if argv is None else argv
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
    OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
    with initialize_config_dir(config_dir=os.path.join(VERITAS_ROOT, "config", "eval"),
                               version_base=None):
        cfg = compose(config_name="bridge.yaml", overrides=argv)
    OmegaConf.resolve(cfg)
    with open_dict(cfg):
        cfg.gpu_id = 0
        cfg.multi_gpu = False
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    with open(os.path.join(cfg.log_dir, "resolved_config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    agent = agent_cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
