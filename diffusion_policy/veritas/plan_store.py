"""On-disk plan records and Gemini cost accounting.

A plan record is the ``ep{idx}.json`` the planning scripts write: the raw Gemini
response plus the clipped ``plan``. A record without ``plan`` is a failed call and is
refused rather than fallen back from (Veritas's own fallback -- one waypoint at the
image centre -- would silently steer scoring toward the goal).
"""

from __future__ import annotations

import json
import os
import random
import time
import pathlib
from typing import Callable, Dict, Iterable, Optional, Type

from diffusion_policy.veritas.schemas import VeritasPlan

# Paid-tier standard $ per 1M tokens (input, output incl. thinking), ai.google.dev pricing
# as of 2026-09-28; 3.8-flash doubles on 2027-01-01. Batch API is half.
_JITTER = random.Random()   # retry jitter; private so the global RNG is untouched

PRICE_PER_M = {'gemini-3.8-flash': (0.75, 3.75), 'gemini-3.5-flash': (1.50, 9.00)}


def cost_usd(usage, model) -> Optional[float]:
    """Standard paid-tier cost of one call, or None if the model or usage is unknown."""
    if not usage or model not in PRICE_PER_M:
        return None
    p_in, p_out = PRICE_PER_M[model]
    out = (usage.get('candidates_token_count') or 0) + (usage.get('thoughts_token_count') or 0)
    return ((usage.get('prompt_token_count') or 0) * p_in + out * p_out) / 1e6


def read_record(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def has_plan(path) -> bool:
    path = pathlib.Path(path)
    return path.exists() and 'plan' in read_record(path)


def load_plan(path, plan_cls: Type[VeritasPlan] = VeritasPlan) -> VeritasPlan:
    """The clipped plan out of a record, refusing a record without one."""
    rec = read_record(path)
    if 'plan' not in rec:
        raise ValueError(f'{path}: no clipped `plan` recorded -- the Gemini call failed; '
                         f're-run the planning script for this episode.')
    return plan_cls(**rec['plan'])


def write_plan_record(path, plan: Optional[VeritasPlan], **fields) -> dict:
    """Merge ``plan`` (clipped) and ``fields`` into the record at ``path``.

    The Gemini client logs the raw response to the same path first (``_save_vlm_log``);
    this adds the parsed plan and provenance beside it. Returns the record.
    """
    path = pathlib.Path(path)
    rec = read_record(path) if path.exists() else {}
    rec.update(fields)
    if plan is not None:
        rec['plan'] = plan.model_dump()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))
    return rec


def usage_summary(out, called: Iterable[int], model: str) -> str:
    """Gemini usage + cost of THIS run's calls, and of every plan the folder holds."""
    out = pathlib.Path(out)

    def tally(paths):
        n, tok, usd = 0, {'in': 0, 'out': 0, 'thought': 0}, 0.0
        for p in paths:
            rec = read_record(p)
            u = rec.get('usage') or {}
            if not u:
                continue
            n += 1
            tok['in'] += u.get('prompt_token_count') or 0
            tok['out'] += u.get('candidates_token_count') or 0
            tok['thought'] += u.get('thoughts_token_count') or 0
            usd += cost_usd(u, rec.get('model')) or 0.0
        return n, tok, usd

    def line(label, n, tok, usd):
        return (f'{label}: {n} Gemini call(s), tokens in {tok["in"]} / out {tok["out"]} / '
                f'thought {tok["thought"]}, ${usd:.3f}')

    this = tally(out / f'ep{i}.json' for i in called if (out / f'ep{i}.json').exists())
    every = tally(sorted(out.glob('ep*.json')))
    return (f'GEMINI USAGE ({model})\n  ' + line('this run', *this) + '\n  '
            + line(f'{out}/ total', *every))


class CachedPlanProvider:
    """Plans keyed by env seed: ``<plan_dir>/seed{seed}.json`` (a symlink to the
    episode record the planning script wrote).

    On a miss the provider calls ``planner(seed, **kwargs)`` -- a live Gemini call --
    only when ``allow_live`` is set, and writes the result to the same store so later
    runs hit the cache. Otherwise a miss raises: a rate computed over the episodes that
    happen to have plans is not comparable to anything (same rule as
    ``env/pusht/veritas/tracker.load_split_plans``).
    """

    def __init__(self, plan_dir, planner: Optional[Callable[..., dict]] = None,
                 allow_live: bool = False, plan_cls: Type[VeritasPlan] = VeritasPlan):
        self.plan_dir = pathlib.Path(plan_dir)
        self.planner = planner
        self.allow_live = allow_live
        self.plan_cls = plan_cls
        self.n_hits = 0
        self.n_live = 0

    @staticmethod
    def seed_name(seed: int) -> str:
        return f'seed{int(seed)}.json'

    def path_for_seed(self, seed: int) -> pathlib.Path:
        return self.plan_dir / self.seed_name(seed)

    def __call__(self, seed: int, **live_kwargs) -> VeritasPlan:
        path = self.path_for_seed(seed)
        if has_plan(path):
            self.n_hits += 1
            return load_plan(path, self.plan_cls)
        if not (self.allow_live and self.planner is not None):
            raise FileNotFoundError(
                f'no plan for seed {seed} at {path}. Generate it offline with '
                f'scriptsv2/veritas/plan_simpler_episodes.py, or set '
                f'VERITAS_ALLOW_LIVE=1 to call Gemini at reset.')
        # Gemini overload (503) spikes outlast the client's own retries; keep trying for
        # up to VERITAS_PLAN_MAX_WAIT_S before giving up (nothing has run yet at reset)
        max_wait = float(os.environ.get('VERITAS_PLAN_MAX_WAIT_S', '3600'))
        t0, wait = time.time(), 60.0
        while True:
            try:
                rec = self.planner(seed, **live_kwargs)
            except Exception as e:  # transport errors the client did not absorb
                print(f'[plan_store] live planning for seed {seed} raised: {e!r}', flush=True)
                rec = None
            if rec and 'plan' in rec:
                break
            if time.time() - t0 + wait > max_wait:
                raise RuntimeError(f'live Gemini planning failed for seed {seed}')
            sleep_s = wait * _JITTER.uniform(0.75, 1.25)
            print(f'[plan_store] no plan for seed {seed}; retrying in {sleep_s:.0f}s', flush=True)
            time.sleep(sleep_s)
            wait = min(2 * wait, 600.0)
        self.n_live += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2))
        return self.plan_cls(**rec['plan'])


def link_seed(plan_dir, seed: int, ep_json_name: str) -> None:
    """``seed{seed}.json -> ep{idx}.json`` inside ``plan_dir`` (relative symlink,
    replaced if present)."""
    plan_dir = pathlib.Path(plan_dir)
    link = plan_dir / CachedPlanProvider.seed_name(seed)
    if link.is_symlink() or link.exists():
        link.unlink()
    os.symlink(ep_json_name, link)
