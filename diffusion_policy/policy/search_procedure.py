"""The best-of-n search procedure, shared by every search policy.

The loop depends on exactly two things a subclass supplies -- ``predict_action`` (generate
one chunk, optionally conditioned on prior candidates) and ``_score_candidates`` (simulate
it) -- so the Gaussian and diffusion families run an identical procedure and differ only in
how a single candidate is produced.
"""
from typing import Dict, Optional
import contextlib
import math

import torch
import torch.nn.functional as F

from diffusion_policy.common.selection_util import SELECTION_MODES, select_candidate


def _stack_subgoals(per_candidate):
    """[{k: (B, ...)}] (one dict per candidate) -> {k: (B, n, ...)}; None if empty."""
    if not per_candidate:
        return None
    return {k: torch.stack([d[k] for d in per_candidate], dim=1)
            for k in per_candidate[0]}


def _cat_subgoals(per_block):
    """[{k: (B, n_i, ...)}] -> {k: (B, sum n_i, ...)}; None if empty."""
    if not per_block:
        return None
    return {k: torch.cat([d[k] for d in per_block], dim=1) for k in per_block[0]}


class SearchProcedureMixin:
    """Requires of the host policy: ``predict_action``, ``normalizer``, ``verifier``,
    ``max_actions``, ``n_obs_steps``, ``n_action_steps``, ``selection``,
    ``selection_temperature``, ``corrupt_obs_features`` (see ObsCorruptionMixin), and
    ``_encode_obs_features``.
    """

    def _init_slot_weighting(self, slot_weight_decay=False, slot_weights=None):
        """Resolve the per-candidate-slot loss weighting. OFF (uniform) by default.

        Two config surfaces, and they are mutually exclusive:

          slot_weights: {mode, decay, ratio, weights, schedule, val}   -- the general one
          slot_weight_decay: <float>                                   -- LEGACY scalar,
              exactly equivalent to {mode: geometric, decay: X, val: trained}

        Both accepted so the runs that already trained under the scalar keep resolving to
        the identical weight vector; setting both to non-defaults raises rather than
        silently letting one win.

        The nested dict is key-checked against a whitelist for the same reason
        DiffusionTransformerSearchPolicy._validate_kwargs exists: a typo inside a config
        block is otherwise silently ignored, which is how an ablation arm ends up secretly
        identical to its sibling. _KNOWN_KWARGS only guards the TOP-level name, so without
        this `slot_weights.rato: 4` would train uniform and say nothing.
        """
        spec = self._resolve_slot_weight_spec(slot_weight_decay, slot_weights)
        self.slot_weight_spec = spec
        # Kept as a plain float for `geometric` (else None) because two external readers
        # index it by name: scripts/selection_smoke.py and dump_candidate_scores.py's
        # _slot_weight_table.
        self.slot_weight_decay = spec['decay'] if spec['mode'] == 'geometric' else None
        # Curriculum position. A plain int, not a buffer: it is re-derived from global_step
        # every optimizer step, so a resume reproduces the schedule with nothing to restore
        # -- the same property CropScopeMixin._crop_step relies on.
        self._slot_weight_step = 0
        if spec['mode'] == 'uniform':
            return
        w = self._slot_weights(torch.device('cpu'), torch.float32)
        if w is None:
            return
        def _fmt(v):
            return '[' + ', '.join(f'{x:.4f}' for x in v.tolist()) + f'] (mean {v.mean():.6f})'
        name = type(self).__name__
        if spec['mode'] == 'curriculum':
            # Print EVERY waypoint: one vector is not an honest record of a run whose
            # profile moves, and the log is the only place the schedule is recorded.
            print(f'{name}: slot_weights curriculum, {len(spec["waypoints"])} waypoints, '
                  f'interp={spec["interp"]}')
            for wp in spec['waypoints']:
                v = self._slot_profile(wp, self.max_actions,
                                       torch.device('cpu'), torch.float32)
                tag = wp['mode'] + (' reversed' if wp.get('reverse') else '')
                print(f'  step {wp["step"]:>7,}  {tag:<16} {_fmt(v)}')
        elif spec['schedule'] is None:
            print(f'{name}: slot_weights {spec["mode"]} -> per-slot weights {_fmt(w)}')
        else:
            # Under a curriculum ONE printed vector is not an honest record of what the run
            # trained at, so both endpoints are logged.
            sch = spec['schedule']
            a = self._slot_weights(torch.device('cpu'), torch.float32, step=sch['start_step'])
            b = self._slot_weights(torch.device('cpu'), torch.float32, step=sch['end_step'])
            print(f'{name}: slot_weights {spec["mode"]} + {sch["shape"]} schedule '
                  f'{sch["start_step"]}..{sch["end_step"]}\n'
                  f'  at start {_fmt(a)}\n  at end   {_fmt(b)}')

    # Every key the slot_weights block may contain. See _init_slot_weighting for why this
    # whitelist exists rather than a permissive .get().
    _SLOT_WEIGHT_KEYS = frozenset(
        {'mode', 'decay', 'ratio', 'weights', 'schedule', 'val', 'reverse', 'waypoints',
         'interp'})
    _SLOT_SCHEDULE_KEYS = frozenset({'shape', 'start_step', 'end_step'})
    # One waypoint of a curriculum: a profile spec plus the step it is reached at. The same
    # profile keys mean the same thing here as at the top level -- that is what lets a
    # curriculum reuse every shape without a second implementation.
    _SLOT_WAYPOINT_KEYS = frozenset({'step', 'mode', 'decay', 'ratio', 'weights', 'reverse'})
    _SLOT_MODES = ('uniform', 'geometric', 'linear', 'tent', 'list', 'last_only',
                   'curriculum')
    # Shapes a profile can take on its own, i.e. everything a curriculum waypoint may be.
    # 'curriculum' is excluded: waypoints do not nest.
    _SLOT_PROFILE_MODES = ('uniform', 'geometric', 'linear', 'tent', 'list', 'last_only')
    _SLOT_SHAPES = ('linear', 'cosine', 'step')

    @classmethod
    def _fill_profile(cls, src, spec, where):
        """Validate one profile's shape parameters from `src` into `spec`. Pure.

        Shared by the top-level slot_weights block and by every curriculum waypoint, so a
        shape means exactly the same thing wherever it is written.
        """
        mode = spec['mode']
        if mode == 'geometric':
            if src.get('decay') is None:
                raise ValueError(f"{where}.mode: geometric needs a `decay` in (0, 1)")
            d = float(src['decay'])
            if d == 1.0:
                raise ValueError(f'{where}.decay: 1.0 is a no-op; use mode: uniform.')
            assert 0.0 < d < 1.0, f'{where}.decay must be in (0, 1), got {d}'
            spec['decay'] = d
        elif mode in ('linear', 'tent'):
            if src.get('ratio') is None:
                raise ValueError(f"{where}.mode: {mode} needs a `ratio` > 1 "
                                 f"(w_last/w_first for linear, w_peak/w_edge for tent)")
            r = float(src['ratio'])
            if r == 1.0:
                raise ValueError(f'{where}.ratio: 1.0 is a no-op; use mode: uniform.')
            assert r > 1.0, f'{where}.ratio must be > 1, got {r}'
            spec['ratio'] = r
        elif mode == 'list':
            if src.get('weights') is None:
                raise ValueError(f"{where}.mode: list needs an explicit `weights` list")
            w = [float(x) for x in src['weights']]
            assert w and all(x > 0 for x in w), \
                f'{where}.weights must all be > 0 (they are renormalized to mean 1)'
            spec['weights'] = w
        # `reverse` mirrors the finished vector, so a DESCENDING profile is expressible
        # without loosening the `ratio > 1` / `0 < decay < 1` asserts above -- those rule out
        # the degenerate and no-op cases and are worth keeping.
        spec['reverse'] = bool(src.get('reverse', False))
        return spec

    @classmethod
    def _resolve_waypoint(cls, wp, i):
        """Validate one curriculum waypoint into a profile spec carrying its `step`."""
        try:
            from omegaconf import OmegaConf
            if OmegaConf.is_config(wp):
                wp = OmegaConf.to_container(wp, resolve=True)
        except ImportError:
            pass
        if not isinstance(wp, dict):
            raise TypeError(f'slot_weights.waypoints[{i}] must be a mapping, got {wp!r}')
        where = f'slot_weights.waypoints[{i}]'
        bad = sorted(set(wp) - cls._SLOT_WAYPOINT_KEYS)
        if bad:
            raise TypeError(f'{where} got unknown key(s) {bad}; known: '
                            f'{sorted(cls._SLOT_WAYPOINT_KEYS)}')
        if wp.get('step') is None:
            raise ValueError(f'{where} needs a `step`')
        mode = wp.get('mode')
        if mode not in cls._SLOT_PROFILE_MODES:
            raise ValueError(f'{where}.mode must be one of {cls._SLOT_PROFILE_MODES}, '
                             f'got {mode!r}')
        spec = {'mode': mode, 'decay': None, 'ratio': None, 'weights': None,
                'step': int(wp['step'])}
        return cls._fill_profile(wp, spec, where)

    @classmethod
    def _resolve_slot_weight_spec(cls, slot_weight_decay=False, slot_weights=None):
        """Validate and normalize both surfaces into one dict. Pure; no self state."""
        try:                                    # hydra hands over DictConfig/ListConfig
            from omegaconf import OmegaConf
            if slot_weights is not None and OmegaConf.is_config(slot_weights):
                slot_weights = OmegaConf.to_container(slot_weights, resolve=True)
        except ImportError:
            pass

        legacy_on = slot_weight_decay is not False and slot_weight_decay is not None
        sw = dict(slot_weights or {})
        unknown = sorted(set(sw) - cls._SLOT_WEIGHT_KEYS)
        if unknown:
            raise TypeError(
                f'slot_weights got unknown key(s) {unknown}. These would be silently '
                f'ignored, leaving the run on uniform weights while the config claims '
                f'otherwise. Known keys: {sorted(cls._SLOT_WEIGHT_KEYS)}')
        mode = sw.get('mode', 'uniform') or 'uniform'
        if mode not in cls._SLOT_MODES:
            raise ValueError(f'slot_weights.mode must be one of {cls._SLOT_MODES}, got {mode!r}')
        if legacy_on and mode != 'uniform':
            raise ValueError(
                'slot_weight_decay and slot_weights are two spellings of the same knob; '
                'set exactly one. slot_weight_decay: X == slot_weights: '
                '{mode: geometric, decay: X, val: trained}.')

        if legacy_on:
            decay = float(slot_weight_decay)
            if decay == 1.0:
                raise ValueError(
                    'slot_weight_decay: 1.0 is a no-op; use False to mean uniform slots.')
            assert 0.0 < decay < 1.0, \
                f'slot_weight_decay must be False or in (0, 1), got {decay}'
            # val: 'trained' preserves the semantics every run under the scalar had --
            # val_loss was computed WITH the weighting.
            return {'mode': 'geometric', 'decay': decay, 'ratio': None, 'weights': None,
                    'schedule': None, 'val': 'trained', 'legacy_scalar': True}

        spec = {'mode': mode, 'decay': None, 'ratio': None, 'weights': None,
                'schedule': None, 'reverse': False, 'waypoints': None,
                'interp': 'linear',
                # Default 'uniform': with topk retention removed val_loss no longer selects
                # anything, but it is still the cross-arm comparison signal and a
                # non-stationary reweighting would make it move for reasons unrelated to
                # fit. A fixed yardstick keeps a curriculum's val curve comparable with
                # itself. See _slot_weights.
                'val': sw.get('val', 'uniform') or 'uniform',
                'legacy_scalar': False}
        if spec['val'] not in ('uniform', 'trained'):
            raise ValueError(f"slot_weights.val must be 'uniform' or 'trained', "
                             f"got {spec['val']!r}")
        if mode == 'curriculum':
            wps = sw.get('waypoints')
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(wps):
                    wps = OmegaConf.to_container(wps, resolve=True)
            except ImportError:
                pass
            if not wps or len(wps) < 2:
                raise ValueError('slot_weights.mode: curriculum needs at least 2 '
                                 '`waypoints`; one waypoint is a fixed profile, so use '
                                 'that profile directly instead.')
            spec['waypoints'] = [cls._resolve_waypoint(w, i) for i, w in enumerate(wps)]
            # How the profile moves BETWEEN waypoints.
            #   'step'   -- hold each waypoint's profile until the next one's step. "20k
            #               steps on slot-0-heavy, then 20k on the tent" is this one: each
            #               profile is actually trained on for its whole stretch.
            #   'linear' -- interpolate, so a waypoint's exact profile is touched for an
            #               instant and the objective drifts continuously.
            # Both keep mean 1 at every step, so neither moves the loss scale.
            spec['interp'] = sw.get('interp', 'linear') or 'linear'
            if spec['interp'] not in ('linear', 'step'):
                raise ValueError(f"slot_weights.interp must be 'linear' or 'step', "
                                 f"got {spec['interp']!r}")
            steps = [w['step'] for w in spec['waypoints']]
            if steps != sorted(steps) or len(set(steps)) != len(steps):
                raise ValueError(f'slot_weights.waypoints steps must be strictly '
                                 f'increasing, got {steps}')
            if sw.get('schedule') is not None:
                raise ValueError('slot_weights.schedule with mode: curriculum is two '
                                 'schedules at once -- the waypoints ARE the schedule. '
                                 'Drop one.')
        else:
            if sw.get('waypoints') is not None:
                raise ValueError(f'slot_weights.waypoints only means anything with '
                                 f'mode: curriculum, not mode: {mode}')
            if sw.get('interp') is not None:
                raise ValueError(f'slot_weights.interp only means anything with '
                                 f'mode: curriculum, not mode: {mode}')
            cls._fill_profile(sw, spec, 'slot_weights')

        sch = sw.get('schedule')
        if sch is not None:
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(sch):
                    sch = OmegaConf.to_container(sch, resolve=True)
            except ImportError:
                pass
            sch = dict(sch)
            bad = sorted(set(sch) - cls._SLOT_SCHEDULE_KEYS)
            if bad:
                raise TypeError(f'slot_weights.schedule got unknown key(s) {bad}; known: '
                                f'{sorted(cls._SLOT_SCHEDULE_KEYS)}')
            shape = sch.get('shape', 'linear') or 'linear'
            if shape not in cls._SLOT_SHAPES:
                raise ValueError(f'slot_weights.schedule.shape must be one of '
                                 f'{cls._SLOT_SHAPES}, got {shape!r}')
            start, end = int(sch.get('start_step', 0) or 0), int(sch.get('end_step', 0) or 0)
            if end <= start:
                raise ValueError(f'slot_weights.schedule.end_step ({end}) must exceed '
                                 f'start_step ({start})')
            if mode == 'uniform':
                raise ValueError('slot_weights.schedule with mode: uniform ramps toward '
                                 'uniform from uniform, i.e. it does nothing. Set a mode.')
            spec['schedule'] = {'shape': shape, 'start_step': start, 'end_step': end}
        return spec

    @staticmethod
    def _slot_ramp(step, schedule):
        """Curriculum position in [0, 1]. Clamped at both ends."""
        s, e = schedule['start_step'], schedule['end_step']
        a = (float(step) - s) / max(e - s, 1)
        a = min(max(a, 0.0), 1.0)
        shape = schedule['shape']
        if shape == 'linear':
            return a
        if shape == 'cosine':
            return 0.5 * (1.0 - math.cos(math.pi * a))
        return 1.0 if float(step) >= e else 0.0        # 'step'

    def set_slot_weight_step(self, step: int):
        """Fix the training step the curriculum profile is derived from.

        Called once per optimizer step by the workspace, next to set_crop_step, so the two
        share one notion of "the current step". A plain int and NOT a buffer: it is
        re-derived from global_step every step, so a resumed run reproduces the schedule
        with nothing to checkpoint.
        """
        self._slot_weight_step = int(step)

    @staticmethod
    def _slot_profile(spec, K, device, dtype) -> torch.Tensor:
        """One mean-1 weight vector from a profile spec. Pure; no self state.

        `spec` is either the slot_weights block itself or one curriculum waypoint -- the
        keys mean the same thing in both, which is what lets a curriculum reuse every shape.
        """
        k = torch.arange(K, device=device, dtype=dtype)
        mode = spec['mode']
        if mode == 'uniform':
            # Flat. Only reachable as a curriculum WAYPOINT -- at the top level 'uniform'
            # short-circuits to None in _slot_weights, which is the bit-identical
            # unweighted path. Here it has to be a real vector so it can be blended.
            base = torch.ones(K, device=device, dtype=dtype)
        elif mode == 'geometric':
            base = spec['decay'] ** (K - 1 - k)
        elif mode == 'linear':
            # w_k proportional to 1 + (ratio-1)*k/(K-1): same endpoint ratio as a geometric
            # decay of ratio^(-1/(K-1)), differing only in curvature. That pairing is what
            # makes geometric-vs-linear an experiment about SHAPE rather than about spread.
            base = 1.0 + (spec['ratio'] - 1.0) * k / max(K - 1, 1)
        elif mode == 'tent':
            # Symmetric peak at the centre, `ratio` = peak/edge. The middle-heavy waypoint
            # of a curriculum: mass on the slots that are neither context-free nor fully
            # conditioned. At even K the peak straddles the two central slots.
            mid = (K - 1) / 2.0
            base = 1.0 + (spec['ratio'] - 1.0) * (1.0 - (k - mid).abs() / max(mid, 1e-9))
        elif mode == 'last_only':
            base = (k == K - 1).to(dtype)
        elif mode == 'list':
            w = spec['weights']
            if len(w) != K:
                raise ValueError(
                    f'slot_weights.weights has {len(w)} entries but max_actions is {K}; '
                    f'an explicit profile must name every slot.')
            base = torch.tensor(w, device=device, dtype=dtype)
        else:
            raise ValueError(f'unhandled slot_weights mode {mode!r}')
        w = base / base.mean()
        # Mirror LAST, so `reverse` flips the finished profile rather than its parameter --
        # reversing a mean-1 vector is still mean-1, so the loss scale is untouched.
        if spec.get('reverse'):
            w = torch.flip(w, dims=(0,))
        return w

    def _slot_curriculum(self, waypoints, step, K, device, dtype,
                         interp='linear') -> torch.Tensor:
        """Interpolate between ordered waypoint profiles, in WEIGHT space.

        Held flat before the first waypoint's step and after the last one's. In between,
        `interp='linear'` interpolates and `interp='step'` holds each waypoint's profile
        until the next one's step -- the latter is what "N steps on this profile" means. Every waypoint is mean 1 and a convex combination of mean-1 vectors is
        mean 1, so the loss SCALE is constant across the whole curriculum -- the same
        invariant the single-target `schedule` blend relies on, and the reason
        gradient_clip_norm and val_loss stay comparable while the profile moves.
        """
        ws = [self._slot_profile(p, K, device, dtype) for p in waypoints]
        steps = [p['step'] for p in waypoints]
        t = float(step)
        if t <= steps[0]:
            return ws[0]
        if t >= steps[-1]:
            return ws[-1]
        for i in range(len(steps) - 1):
            # HALF-OPEN [steps[i], steps[i+1]): the waypoint's own step belongs to the
            # interval that STARTS there. With an inclusive upper bound the earlier
            # interval claimed the boundary, so under interp='step' the new profile did
            # not take effect until one interval later.
            if steps[i] <= t < steps[i + 1]:
                if interp == 'step':
                    return ws[i]        # hold this profile until the next waypoint's step
                a = (t - steps[i]) / max(steps[i + 1] - steps[i], 1)
                return ws[i] + a * (ws[i + 1] - ws[i])
        return ws[-1]           # unreachable: t is bracketed by the two guards above

    def _slot_weights(self, device, dtype, step=None) -> Optional[torch.Tensor]:
        """Per-slot loss weights: a length-K vector (K = max_actions) with mean 1, or None
        for the uniform path.

        One forward decodes all K candidate slots against the same expert action, slot k
        attending to the first k scored context candidates; these weights say how much of
        the gradient each of those conditionals gets.

          geometric    w_k ∝ decay^(K-1-k)
          linear       w_k ∝ 1 + (ratio-1)*k/(K-1)
          tent         symmetric peak at the centre, ratio = peak/edge
          list         an explicit K-length profile
          last_only    slot K-1 only -- an extreme probe, not a recipe
          curriculum   interpolate between `waypoints`, each of the above, over training

        `reverse: true` mirrors any of them, which is how a slot-0-heavy profile is written.
        `schedule` ramps a single profile up from uniform instead. The mean stays exactly 1
        throughout, so changing profile never moves the loss SCALE that gradient_clip_norm,
        the effective step size, and val_loss all read. Which profile to use and why: the
        slot_weights block in train_pusht_diffusion_search.yaml.
        """
        # None == uniform; width 1 has a single slot, so there is nothing to weight
        spec = self.slot_weight_spec
        if spec['mode'] == 'uniform' or self.max_actions in (None, 1):
            return None
        K = self.max_actions
        t = self._slot_weight_step if step is None else step
        if spec['mode'] == 'curriculum':
            return self._slot_curriculum(spec['waypoints'], t, K, device, dtype,
                                         spec['interp'])
        w = self._slot_profile(spec, K, device, dtype)

        sch = spec['schedule']
        if sch is not None:
            a = self._slot_ramp(t, sch)
            # Blend in WEIGHT space, not by interpolating the profile's parameter. Because
            # mean(w) == 1 exactly, mean(w - 1) == 0, so mean(1 + a*(w-1)) == 1 for EVERY a
            # -- the loss scale is constant across the whole ramp. Interpolating `decay`
            # instead would also have to pass through decay == 1.0, which the config
            # validator rejects as a no-op.
            w = 1.0 + a * (w - 1.0)
        return w

    # ---------------------------------------------------------------- per-slot loss norm

    _SLOT_NORM_KEYS = frozenset({'mode'})
    _SLOT_NORM_MODES = ('l2', 'l1', 'l2tol1')

    @classmethod
    def _resolve_slot_loss_norm(cls, slot_loss_norm=None):
        """Validate the slot_loss_norm block into {'mode': ...}. Pure; no self state."""
        try:                                    # hydra hands over DictConfig
            from omegaconf import OmegaConf
            if slot_loss_norm is not None and OmegaConf.is_config(slot_loss_norm):
                slot_loss_norm = OmegaConf.to_container(slot_loss_norm, resolve=True)
        except ImportError:
            pass
        sn = dict(slot_loss_norm or {})
        # Whitelisted for the same reason slot_weights is: a silently ignored typo would
        # train the default objective while the config claims otherwise.
        unknown = sorted(set(sn) - cls._SLOT_NORM_KEYS)
        if unknown:
            raise TypeError(f'slot_loss_norm got unknown key(s) {unknown}; known: '
                            f'{sorted(cls._SLOT_NORM_KEYS)}')
        mode = sn.get('mode', 'l2') or 'l2'
        if mode not in cls._SLOT_NORM_MODES:
            raise ValueError(f'slot_loss_norm.mode must be one of {cls._SLOT_NORM_MODES}, '
                             f'got {mode!r}')
        return {'mode': mode}

    def _init_slot_loss_norm(self, slot_loss_norm=None):
        """Resolve which norm each slot's loss term uses. L2 everywhere by default."""
        self.slot_loss_norm_spec = self._resolve_slot_loss_norm(slot_loss_norm)
        a = self._slot_norm_alphas(torch.device('cpu'), torch.float32)
        if a is None:
            return
        print(f'{type(self).__name__}: slot_loss_norm '
              f'{self.slot_loss_norm_spec["mode"]} -> per-slot L1 fraction ['
              + ', '.join(f'{x:.4f}' for x in a.tolist()) + ']')

    def _slot_norm_alphas(self, device, dtype) -> Optional[torch.Tensor]:
        """Per-slot L1 fraction alpha_k, or None for the plain-MSE path.

        Slot k's loss term is ``(1-a_k)*(pred-target)**2 + a_k*|pred-target|``, so a_k = 0
        is pure L2 and a_k = 1 is pure L1.

          l2      None -- every slot MSE, bit-identical to the objective that predates this
          l1      a_k = 1
          l2tol1  a_k = k/(K-1): slot 0 pure L2, slot K-1 pure L1, linear in between

        NOT renormalized, unlike _slot_weights: these pick a norm, they do not split a fixed
        loss budget. The two norms are within ~20% at unit residual (the target is
        eps ~ N(0,1): E[e^2] = 1.00 vs E[|e|] = 0.80), so blending them does not
        appreciably reweight one slot against another.
        """
        mode = self.slot_loss_norm_spec['mode']
        K = self.max_actions
        if mode == 'l2':
            return None
        if mode == 'l1':
            return torch.ones(K, device=device, dtype=dtype)
        if K == 1:                     # nothing to interpolate; slot 0 is the pure-L2 end
            return None
        return torch.arange(K, device=device, dtype=dtype) / (K - 1)

    def _slot_norm_loss(self, pred, target, slot_weighting=True):
        """Elementwise loss per slot, (B, K, H, Da): plain MSE, or the per-slot L1/L2 blend.

        ``slot_weighting=False`` forces plain MSE whatever the config says -- see
        compute_loss, which passes it for val_loss.
        """
        alphas = self._slot_norm_alphas(pred.device, pred.dtype) if slot_weighting else None
        if alphas is None:
            return F.mse_loss(pred, target, reduction='none')
        err = pred - target
        a = alphas.view(1, -1, 1, 1)
        return (1.0 - a) * err.pow(2) + a * err.abs()

    # ------------------------------------------------------ per-slot observation noise

    _SLOT_OBS_NOISE_KEYS = frozenset(
        {'mode', 'decay', 'timesteps', 'shape', 'base_range'})
    _SLOT_OBS_NOISE_MODES = ('uniform', 'linear_t', 'geometric', 'linear_signal', 'list',
                             'random_base')
    # Which profiles `random_base` may borrow its spacing from. `list` is excluded: an
    # explicit K-length profile already names absolute timesteps, so shifting it is a
    # contradiction; use it directly if you want fixed levels.
    _SLOT_OBS_SHAPES = ('linear_t', 'geometric', 'linear_signal')

    @classmethod
    def _resolve_slot_obs_noise(cls, slot_obs_noise=None):
        """Validate and normalize the slot_obs_noise block. Pure; no self state.

        Key-checked against a whitelist for the same reason slot_weights is: a typo inside a
        config block is otherwise silently ignored, which is how an ablation arm ends up
        secretly identical to its sibling. Without this `slot_obs_noise.mod: linear_t` would
        train uncorrupted and say nothing.
        """
        try:                                    # hydra hands over DictConfig/ListConfig
            from omegaconf import OmegaConf
            if slot_obs_noise is not None and OmegaConf.is_config(slot_obs_noise):
                slot_obs_noise = OmegaConf.to_container(slot_obs_noise, resolve=True)
        except ImportError:
            pass

        sn = dict(slot_obs_noise or {})
        unknown = sorted(set(sn) - cls._SLOT_OBS_NOISE_KEYS)
        if unknown:
            raise TypeError(
                f'slot_obs_noise got unknown key(s) {unknown}. These would be silently '
                f'ignored, leaving the run uncorrupted while the config claims otherwise. '
                f'Known keys: {sorted(cls._SLOT_OBS_NOISE_KEYS)}')
        mode = sn.get('mode', 'uniform') or 'uniform'
        if mode not in cls._SLOT_OBS_NOISE_MODES:
            raise ValueError(f'slot_obs_noise.mode must be one of '
                             f'{cls._SLOT_OBS_NOISE_MODES}, got {mode!r}')

        spec = {'mode': mode, 'decay': None, 'timesteps': None,
                'shape': None, 'base_range': None}
        if mode == 'random_base':
            # The SHAPE supplies the profile; the base -- the timestep slot 0 sits at -- is
            # redrawn per sample, and the profile is rescaled into [0, base]. So slot 0's
            # corruption is random and the schedule runs from it down to clean at slot K-1,
            # instead of the model seeing the same K levels for its whole life.
            shape = sn.get('shape') or 'linear_signal'
            if shape not in cls._SLOT_OBS_SHAPES:
                raise ValueError(f'slot_obs_noise.shape must be one of '
                                 f'{cls._SLOT_OBS_SHAPES}, got {shape!r}')
            spec['shape'] = shape
            if shape == 'geometric':
                if sn.get('decay') is None:
                    raise ValueError('slot_obs_noise.mode: random_base with shape: '
                                     'geometric needs a `decay` in (0, 1)')
                d = float(sn['decay'])
                assert 0.0 < d < 1.0, \
                    f'slot_obs_noise.decay must be in (0, 1), got {d}'
                spec['decay'] = d
            elif sn.get('decay') is not None:
                raise ValueError(f'slot_obs_noise.decay only means something under '
                                 f'shape: geometric, not shape: {shape}')
            # [lo, hi] the per-sample base is drawn from. hi == T-1 gives the full fixed
            # ladder; lo == 0 gives an entirely clean observation.
            br = sn.get('base_range')
            if br is not None:
                br = [int(x) for x in br]
                if len(br) != 2 or br[0] > br[1] or br[0] < 0:
                    raise ValueError('slot_obs_noise.base_range must be [lo, hi] with '
                                     f'0 <= lo <= hi, got {br}')
                spec['base_range'] = br
            return spec
        if sn.get('shape') is not None or sn.get('base_range') is not None:
            raise ValueError(f'slot_obs_noise.shape / base_range only mean something under '
                             f'mode: random_base, not mode: {mode}')
        if mode == 'geometric':
            if sn.get('decay') is None:
                raise ValueError("slot_obs_noise.mode: geometric needs a `decay` in (0, 1)")
            d = float(sn['decay'])
            if d == 1.0:
                raise ValueError('slot_obs_noise.decay: 1.0 holds every slot at the noisiest '
                                 'timestep; use mode: uniform to mean no ladder.')
            assert 0.0 < d < 1.0, f'slot_obs_noise.decay must be in (0, 1), got {d}'
            spec['decay'] = d
        elif mode == 'list':
            if sn.get('timesteps') is None:
                raise ValueError("slot_obs_noise.mode: list needs an explicit `timesteps` list")
            spec['timesteps'] = [int(x) for x in sn['timesteps']]
        return spec

    def _init_slot_obs_noise(self, slot_obs_noise=None):
        """Resolve the per-slot observation-corruption ladder. OFF by default.

        Call AFTER `_init_corruption` and after `self.obs_noise_scheduler` exists: the
        mutual-exclusion check reads `self.corrupt_obs`, and `linear_signal` inverts the
        scheduler's `alphas_cumprod`.
        """
        spec = self._resolve_slot_obs_noise(slot_obs_noise)
        self.slot_obs_noise_spec = spec
        if spec['mode'] == 'uniform':
            return
        # N-5: both surfaces noise the SAME tensor (obs_cond), so enabling both applies the
        # corruption twice at two unrelated levels. Same treatment slot_weight_decay gets
        # against slot_weights: refuse rather than let one silently win.
        if getattr(self, 'corrupt_obs', False):
            raise ValueError(
                'corrupt_obs and slot_obs_noise both corrupt obs_cond, so enabling both '
                'noises it twice. Set exactly one: corrupt_obs for the flat single-level '
                'arm, slot_obs_noise for the per-slot ladder.')
        name = type(self).__name__
        detail = spec['mode'] + (f" decay={spec['decay']}" if spec['decay'] is not None else '')
        T = self.obs_noise_scheduler.config.num_train_timesteps
        t = self._slot_obs_timesteps()
        if t is None:
            shape = self._slot_obs_shape()
            if shape is None:
                return
            lo, hi = self.slot_obs_base_range()
            detail += f" shape={spec['shape']} base_range=[{lo}, {hi}]"
            # There is no single ladder to print: the extent moves per sample. Print the one
            # at the MIDPOINT base, which is the average case, and say so.
            t = self.rescale_slot_timesteps(shape, (lo + hi) // 2)
            detail += ' (levels below are at the midpoint base)'
        abar = self.obs_noise_scheduler.alphas_cumprod.to(t.device)
        sig = abar[t].sqrt()
        print(f'{name}: slot_obs_noise {detail} -> per-slot (timestep, sqrt(alpha_bar))')
        print('  ' + ', '.join(f'{i}:({int(ti)}, {si:.3f})'
                               for i, (ti, si) in enumerate(zip(t.tolist(), sig.tolist()))))
        # The convention every shape obeys: slot 0 is the MOST corrupted, because slot k
        # conditions on the first k scored candidates and the slot with no context is the
        # one that should be least sure of what it is looking at. `list` is the only mode
        # that can violate it -- the formulas cannot -- and a reversed profile trains a
        # ladder pointing the wrong way while every label still says otherwise.
        if bool((t[1:] > t[:-1]).any()):
            print(f'  WARNING: the timestep profile is not non-increasing, so slot 0 is not '
                  f'the most corrupted slot. Every other shape puts the noisiest end at '
                  f'slot 0 (no context); check the order of slot_obs_noise.timesteps.')
        # A ladder whose slots are not distinguishable is a silently degenerate arm --
        # geometric does this readily (see the plan's ladder comparison), so say so here
        # rather than let the run finish before anyone notices.
        dup = int((sig[1:] - sig[:-1]).abs().lt(5e-3).sum())
        if dup:
            print(f'  WARNING: {dup}/{len(sig) - 1} adjacent slot pairs differ by < 0.005 in '
                  f'sqrt(alpha_bar) -- those slots are near-duplicate corruption levels.')

    def _slot_obs_timesteps(self, device=None) -> Optional[torch.Tensor]:
        """Per-slot DDPM timestep for the obs ladder: a length-K long tensor, or None.

        Slot 0 is the MOST corrupted (highest t) and slot K-1 the least, because slot k
        conditions on the first k scored context candidates -- the slot with no context is
        the one that should be least sure of what it is looking at.

          linear_t       t_k = (K-1-k)/(K-1) * (T-1)          -- even in the timestep index
          geometric      t_k = (T-1) * decay^k                -- even in log t
          linear_signal  t_k = argmin |sqrt(abar) - target_k| -- even in retained signal
          list           an explicit K-length profile

        `linear_signal` is spaced evenly in `sqrt(alpha_bar)`, the factor the observation is
        actually scaled by, because `alpha_bar` is a cumulative product: equal steps in t are
        NOT equal steps in corruption. It is inverted by lookup rather than in closed form for
        the same reason.
        """
        spec = self.slot_obs_noise_spec
        # None == no FIXED ladder. random_base has no fixed timesteps by construction: its
        # levels are drawn per sample, so it registers its SHAPE instead (_slot_obs_shape).
        if (spec['mode'] in ('uniform', 'random_base')) or self.max_actions in (None, 1):
            return None
        return self._slot_obs_profile(spec['mode'], spec, device=device)

    def _slot_obs_shape(self, device=None) -> Optional[torch.Tensor]:
        """The borrowed shape profile for ``mode: random_base``. Length K, in timesteps.

        This is the ladder AT FULL EXTENT -- exactly what the same-named fixed mode would
        produce, running from ``T-1`` at slot 0 down to 0 at slot K-1. The decision-time
        ladder rescales it into ``[0, t_base]``::

            t_k = round(shape_k * t_base / (T - 1))

        so slot 0 sits at the drawn level, slot K-1 stays clean, and the ladder's EXTENT is
        what varies. (A rigid translate cannot work here: the fixed shapes already span the
        whole timestep range, so a block of that width has nowhere to slide.)

        Computed once at construction and registered as a buffer, so the ladder a checkpoint
        trained under cannot differ from the one that evaluates it.
        """
        spec = self.slot_obs_noise_spec
        if spec['mode'] != 'random_base' or self.max_actions in (None, 1):
            return None
        return self._slot_obs_profile(spec['shape'], spec, device=device)

    def slot_obs_base_range(self):
        """``(lo, hi)`` the random base is drawn from, inclusive. random_base only.

        Default ``(0, T-1)``: at ``hi`` the ladder is the full fixed one, at ``lo`` every
        slot is clean. Raise ``lo`` to keep a floor of corruption on slot 0 -- at a very low
        base the rescaled levels round together and most slots see the same near-clean
        observation, which is legitimate but uninformative.
        """
        spec = self.slot_obs_noise_spec
        T = self.obs_noise_scheduler.config.num_train_timesteps
        if spec['base_range'] is None:
            return 0, T - 1
        lo, hi = spec['base_range']
        return int(lo), int(min(hi, T - 1))

    def _slot_obs_profile(self, mode, spec, device=None) -> torch.Tensor:
        """The length-K timestep profile for one shape. Shared by the fixed ladders and by
        `random_base`, which borrows a shape for its spacing -- so the three formulas exist
        exactly once and a `random_base` arm cannot drift from its fixed-ladder counterpart.
        """
        K = self.max_actions
        T = self.obs_noise_scheduler.config.num_train_timesteps
        denom = max(K - 1, 1)
        k = torch.arange(K, dtype=torch.float64)
        if mode == 'linear_t':
            t = torch.round((K - 1 - k) / denom * (T - 1))
        elif mode == 'geometric':
            t = torch.round((T - 1) * spec['decay'] ** k)
        elif mode == 'linear_signal':
            sig = self.obs_noise_scheduler.alphas_cumprod.to(torch.float64).sqrt()  # (T,)
            # sig is decreasing in t: sig[0] ~ 1 (clean), sig[T-1] the floor (noisiest).
            target = sig[-1] + (sig[0] - sig[-1]) * k / denom
            t = (sig[None, :] - target[:, None]).abs().argmin(dim=1)
        elif mode == 'list':
            w = spec['timesteps']
            if len(w) != K:
                raise ValueError(
                    f'slot_obs_noise.timesteps has {len(w)} entries but max_actions is {K}; '
                    f'an explicit profile must name every slot.')
            t = torch.tensor(w, dtype=torch.float64)
        else:
            raise ValueError(f'unhandled slot_obs_noise mode {mode!r}')
        return t.to(dtype=torch.long, device=device).clamp_(0, T - 1)

    def rescale_slot_timesteps(self, shape, base):
        """Fit the full-extent ``shape`` profile into ``[0, base]``. Shared by the startup
        print and by the corruption itself, so what is printed is what is applied.

        ``base`` may be a scalar or a ``(B,)`` tensor; the result is ``(K,)`` or ``(B, K)``.
        """
        T = self.obs_noise_scheduler.config.num_train_timesteps
        shape = shape.to(torch.float64)
        if torch.is_tensor(base):
            scaled = shape.unsqueeze(0) * (base.to(torch.float64).unsqueeze(1) / (T - 1))
        else:
            scaled = shape * (float(base) / (T - 1))
        return scaled.round().to(dtype=torch.long).clamp_(0, T - 1)

    def _init_selection(self, **kwargs):
        """Set the knobs predict_action_best reads. Call from the host's __init__."""
        self.selection = kwargs.get('selection', 'argmax') or 'argmax'
        assert self.selection in SELECTION_MODES, \
            f"selection must be one of {SELECTION_MODES}, got {self.selection!r}"
        # Temperature applies to the STANDARDIZED score, so one value means the same thing
        # across arms: T->0 reproduces argmax, T->inf a uniform pick among the n.
        self.selection_temperature = float(
            kwargs.get('selection_temperature', 1.0) or 1.0)
        assert self.selection_temperature > 0, 'selection_temperature must be > 0'
        # 1-based candidate to execute under selection 'index' (negatives count from
        # the end). Eval-only: set by --selection/--selection-index on a trained
        # checkpoint, never trained under. None everywhere else.
        self.selection_index = kwargs.get('selection_index', None)
        assert self.selection != 'index' or self.selection_index is not None, \
            "selection 'index' needs selection_index"
        # Selection draws from its OWN stream, not the global one the diffusion sampler
        # uses, so turning softmax on does not perturb the trajectories themselves. Same
        # reason CropScopeMixin owns _crop_generator. Not a buffer: it holds no learnable
        # state and a checkpoint should not pin the eval noise.
        self._selection_generator = torch.Generator()
        self._selection_generator.manual_seed(
            int(kwargs.get('selection_seed', 0) or 0) + 20250818)

        # Per-EPISODE sampling seeds, set by the eval harness (see set_sample_seeds).
        # None means "use the global RNG", which is what training does.
        self._sample_seeds = None
        self._sample_draw = 0

    def set_sample_seeds(self, seeds):
        """Make the diffusion noise a pure function of (episode, draw index).

        WHY. Eval rolls episodes out in chunks of ``n_envs`` and draws one
        ``torch.randn(size=(B, ...))`` for the whole chunk, so an episode's noise depended
        on how many envs preceded it in its chunk -- i.e. on ``--n-envs``, on where the
        chunk boundary fell, and on how many padding slots were appended. Two evals of the
        SAME checkpoint at ``--n-envs 16`` and ``--n-envs 50`` therefore scored different
        trajectories, and an ST curve was never paired with a BC curve episode-by-episode.

        With per-row seeds each episode owns its own generator, so its noise is identical
        whatever the batching -- and padding rows cannot perturb real ones, because they no
        longer share a stream. Both policies use DDIM at eta=0, whose ``step`` draws no
        noise at all, so this initial draw is the ONLY stochastic input to a rollout: fixing
        it fixes the whole trajectory.

        ``seeds`` is one integer per row of the batch, or None to restore global-RNG
        behaviour. The draw counter resets here, so it must be called once per chunk.
        """
        self._sample_seeds = None if seeds is None else [int(x) for x in seeds]
        self._sample_draw = 0

    def _init_noise(self, shape, dtype, device, generator=None):
        """Initial denoising noise: per-row streams when seeded, else the global RNG.

        Generators are CPU-side and the result is moved to `device`, so a run is
        reproducible across CPU/GPU as well as across batch shapes. The draw counter
        advances once per call, which is what gives the k candidates of one search step
        (and successive control steps) independent noise while keeping each episode's
        sequence a pure function of its own seed.
        """
        if self._sample_seeds is None:
            return torch.randn(size=shape, dtype=dtype, device=device, generator=generator)
        batch = shape[0]
        assert len(self._sample_seeds) == batch, (
            f'set_sample_seeds got {len(self._sample_seeds)} seeds but the batch is {batch}')
        draw = self._sample_draw
        self._sample_draw += 1
        rows = []
        for base in self._sample_seeds:
            g = torch.Generator()
            g.manual_seed((base * 1_000_003 + draw) % (2 ** 63 - 1))
            rows.append(torch.randn(size=tuple(shape[1:]), dtype=dtype, generator=g))
        return torch.stack(rows).to(device)

    @contextlib.contextmanager
    def _crop_scope(self):
        """No-op by default; policies that own image crops override it to pin one offset
        across the obs and every subgoal encoded inside the scope."""
        yield

    @contextlib.contextmanager
    def _corrupt_scope(self):
        """No-op by default; policies that own a per-slot obs ladder override it to pin one
        corruption sample across every candidate of a decision."""
        yield

    @property
    def context_capacity(self) -> int:
        """How many scored candidates this policy can be CONDITIONED on.

        `max_actions - 1` for the search transformers -- the staircase memory mask tops out
        at that, so a longer context would index past `cond_pos_emb`. **0** for a policy that
        consumes no search context at all (`max_actions is None`, i.e. the UNet BC arm),
        which is the honest answer: its `predict_action` drops `actions` and `values` on the
        floor. The 1<<20 sentinel this replaces made `max_actions - 1` evaluate to a number
        so large that every slice became the whole tensor, which then got discarded --
        arriving at the same behaviour by accident.
        """
        return 0 if self.max_actions is None else self.max_actions - 1

    def _slot_kwargs(self, slot):
        """``{'slot': k}`` when the per-slot obs ladder is active, ``{}`` otherwise.

        The other hosts of this mixin (maze and online search) have no ladder and their
        ``predict_action`` does not take the argument, so it is passed only where it means
        something.
        """
        on = (getattr(self, 'slot_obs_t', None) is not None
              or getattr(self, 'slot_obs_shape', None) is not None)
        return {'slot': slot} if on else {}


    def _build_verifier(self, **kwargs):
        """Build this task's verifier. Abstract: there is no sensible default.

        Deliberately NOT defaulting to the maze verifier, which would put an `l2s` import
        on every subclass's path including the ones that never use it. See
        MazeDiffusionSearchPolicy / SearchPolicy (maze) and PushTDiffusionSearchPolicy.
        """
        raise NotImplementedError(
            f'{type(self).__name__} must implement _build_verifier(**kwargs)')

    def _context_dim(self, obs_feature_dim: int, **kwargs) -> int:
        """Width of the per-candidate feedback in the search context.

        1 == the verifier scalar. Subclasses override to feed a richer signal (see
        PushTDiffusionSearchPolicy, which can feed the rollout state or the *encoded*
        subgoal observation instead of / along with the scalar -- hence obs_feature_dim).
        Called from __init__ to size the transformer's context embedding, so it must
        depend only on the config kwargs and the encoder width.
        """
        return 1

    def _score_candidates(self, verifier, obs_dict, action, want_subgoals: bool = False):
        """Evaluate one batch of candidates.

        Returns ``(context, score, subgoal, terms)``:
          * ``context`` (B,) or (B, context_dim) -- the feedback fed back into the search
            context so the next candidate is conditioned on it.
          * ``score`` (B,) -- the scalar used to *rank* candidates (argmax at eval time).
          * ``subgoal`` -- dict of per-candidate debug tensors for logging, or None. Only
            populated when ``want_subgoals``; verifiers without a renderable outcome
            (e.g. the maze one) always return None.
          * ``terms`` (B, K) -- the raw components the score is built from, kept
            undecomposed so a CROSS-CANDIDATE value can re-weight them once all n
            candidates exist (see ``_fuse_scores``). None when the verifier has no
            decomposition, which is every verifier but PushT's.
        By default context and score are both the verifier value. Subclasses override to
        widen the context while keeping the scalar ranking signal.
        """
        value = verifier.get_value(obs_dict, action)
        return value, value, None, None

    def _fuses_scores(self) -> bool:
        """Does this policy's ranking score need the WHOLE candidate set?

        False for every per-candidate value, which is all of them by default. True only for
        a cross-candidate rule (PushT's ``armTd``), where the score of candidate k depends
        on the other n-1 candidates and therefore cannot be computed as they are generated.
        """
        return False

    def _fuse_scores(self, scores, terms):
        """``(B, n)``, ``(B, n, K)`` -> ``(B, n)``. Identity unless ``_fuses_scores()``."""
        return scores

    def _normalize_context_actions(
            self, actions: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Put context actions into the space the model works in.

        The search loop carries candidates in RAW action units because the verifier
        simulates them (``_verifier_inputs`` resets a sim from pixel coords), and the env
        runner executes them. But the model denoises a NORMALIZED trajectory against a
        normalized target, so feeding it raw actions puts the two halves of
        ``action_value_emb``'s input on scales orders of magnitude apart.

        Normalizing here -- at the model boundary, not in the search loop -- keeps one
        tensor from having to serve both boundaries. This mirrors the invariant
        OnlineSearchPolicy maintains: actions exist only in normalized space anywhere the
        model touches them, and are unnormalized strictly at the env/verifier boundary.
        """
        if actions is None:
            return None
        return self.normalizer['action'].normalize(actions)

    def search_candidates(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            obs_features: Optional[torch.Tensor] = None,
            return_subgoals: bool = False,
            return_terms: bool = False,
        ):
        """Generate n_actions candidates, each conditioned on the previous ones.

        Returns ``(actions, values)``, with ``scores`` appended when ``return_scores``,
        ``subgoals`` when ``return_subgoals``, and ``terms`` last when ``return_terms``
        -- i.e. the tuple grows left-to-right:
        ``(actions, values[, scores][, subgoals][, terms])``. Every optional element is
        APPENDED, never inserted, because ~10 sites unpack this tuple positionally.

        ``terms`` is (B, n, K): the raw components each score was built from, kept so a
        CROSS-CANDIDATE value can re-weight them once all n candidates exist. It is None
        unless the verifier decomposes its value (only PushT's does).

        ``values`` is the search *context* feedback -- (B, n) for the scalar verifier
        value, (B, n, context_dim) for a wider context -- while ``scores`` is always the
        (B, n) scalar used to rank candidates. ``subgoals`` is a dict of stacked
        (B, n, ...) debug tensors (or None if the verifier has none); it is for logging
        only and is never fed back into the search.

        ``obs_features`` optionally supplies an already-encoded obs (see
        _encode_obs_features); otherwise it is encoded once here and shared by every
        candidate, since they all condition on the same observation.

        The crop scope is opened HERE rather than only in the callers, so the observation
        and every candidate's subgoal share one crop offset no matter who called -- this is
        the entry point TrainSearchOuterInnerWorkspace and the diagnostic scripts reach
        directly. It is reentrant, so the offline trainer's nested call is a no-op.
        """
        with self._crop_scope(), self._corrupt_scope():
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            actions = None
            values = None
            scores = None
            terms = None
            subgoals = list()
            for i in range(n_actions):
                # candidate i conditions on exactly i scored context entries, so i IS the
                # slot index the training loss decodes -- see _slot_obs_timesteps.
                new_action = self.predict_action(
                    obs_dict,
                    actions=actions,
                    values=values,
                    obs_features=obs_features,
                    **self._slot_kwargs(i),
                )['action_pred']
                new_value, new_score, new_subgoal, new_terms = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)
                if actions is None:
                    actions = new_action.unsqueeze(1)
                    values = new_value.unsqueeze(1)
                    scores = new_score.unsqueeze(1)
                    terms = None if new_terms is None else new_terms.unsqueeze(1)
                else:
                    actions = torch.cat([actions, new_action.unsqueeze(1)], dim=1)
                    values = torch.cat([values, new_value.unsqueeze(1)], dim=1)
                    scores = torch.cat([scores, new_score.unsqueeze(1)], dim=1)
                    if new_terms is not None:
                        terms = torch.cat([terms, new_terms.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    subgoals.append(new_subgoal)

        out = (actions, values)
        if return_scores:
            out = out + (scores,)
        if return_subgoals:
            out = out + (_stack_subgoals(subgoals),)
        if return_terms:
            out = out + (terms,)
        return out

    @torch.inference_mode()
    def predict_n_actions(
            self,
            obs_dict: Dict[str, torch.Tensor],
            verifier,
            n_actions,
            return_scores: bool = False,
            return_subgoals: bool = False,
            obs_features: Optional[torch.Tensor] = None,
            return_terms: bool = False,
        ):
        """Search with a rolling context window; see search_candidates for the return shape.

        ``obs_features`` optionally supplies an already-encoded obs so a caller that needs
        the features for something else too (``predict_action_best`` in 'final_pass' mode
        draws one extra sample from them) pays for a single encoder pass rather than two.

        Opens its own (reentrant) crop scope. The rolling-window branch below encodes
        subgoals in a loop of its own rather than inside `search_candidates`, so wrapping
        only that method would leave every candidate past `max_actions` on an independent
        crop -- the exact split-crop defect, reappearing only at n > K.

        THIS is where a cross-candidate value is applied (``_fuse_scores``), rather than in
        `predict_action_best`, because both branches below produce the complete (B, n) set
        and several callers -- scripts/render_search_videos.py, scripts/dump_candidate_
        scores.py -- take the candidates straight from here and run their own
        `scores.argmax(dim=1)`. Fusing downstream of them would leave those ranking on the
        BASE value while their output claimed otherwise. Callers of `search_candidates`
        directly (the training workspaces) deliberately do NOT get fusion: a cross-candidate
        value has no training semantics and is rejected as a `verifier_tag`.
        """
        # a cross-candidate value needs the terms; nothing else pays for them
        need_terms = return_scores and self._fuses_scores()
        with self._crop_scope(), self._corrupt_scope():
            # encode once for the whole search, however many candidates it runs
            if obs_features is None:
                obs_features = self._encode_obs_features(obs_dict)
            # `max_actions is None` == this policy has no trained staircase to fall out of
            # (consumes_search_context False), so the rolling window never applies however
            # large n gets. The alternative was a 1<<20 sentinel on max_actions, which read
            # as a search width everywhere it was printed.
            if self.max_actions is None or n_actions <= self.max_actions:
                head = self.search_candidates(
                    obs_dict, verifier, n_actions, return_scores=return_scores,
                    obs_features=obs_features, return_subgoals=return_subgoals,
                    return_terms=True)
                # rebind to the rolling branch's names so both fall through to ONE exit,
                # which is what keeps fusion from being applied per-window at n > K.
                # Walk the tuple rather than indexing fixed slots: the optional elements
                # are appended, so `subgoals` sits at 2 or 3 depending on return_scores.
                all_terms = head[-1]
                head = head[:-1]
                all_actions, all_values = head[0], head[1]
                i = 2
                all_scores = None
                if return_scores:
                    all_scores, i = head[i], i + 1
                all_subgoals_stacked = head[i] if return_subgoals else None
                return self._finish_n_actions(
                    all_actions, all_values, all_scores, all_subgoals_stacked, all_terms,
                    return_scores, return_subgoals, return_terms, need_terms)

            # scores are always needed internally (the caller may only want values), but
            # subgoals are only rendered when actually asked for.
            head = self.search_candidates(
                obs_dict, verifier, self.max_actions, return_scores=True,
                obs_features=obs_features, return_subgoals=return_subgoals,
                return_terms=True)
            terms = head[-1]
            actions, values, scores = head[0], head[1], head[2]
            subgoals = head[3] if return_subgoals else None
            all_actions = actions.clone()
            all_values = values.clone()
            all_scores = scores.clone()
            all_terms = None if terms is None else terms.clone()
            all_subgoals = [subgoals] if subgoals is not None else []

            action_history = actions[:, 1:]
            value_history = values[:, 1:]
            for _ in range(self.max_actions, n_actions):
                # The rolling window holds the context at its widest, so every generation
                # past max_actions sits at the ladder's last (cleanest) slot.
                new_action = self.predict_action(
                    obs_dict,
                    actions=action_history,
                    values=value_history,
                    obs_features=obs_features,
                    **self._slot_kwargs(self.max_actions - 1),
                )['action_pred']
                new_value, new_score, new_subgoal, new_terms = self._score_candidates(
                    verifier, obs_dict, new_action, want_subgoals=return_subgoals)

                all_actions = torch.cat([all_actions, new_action.unsqueeze(1)], dim=1)
                all_values = torch.cat([all_values, new_value.unsqueeze(1)], dim=1)
                all_scores = torch.cat([all_scores, new_score.unsqueeze(1)], dim=1)
                if new_terms is not None:
                    all_terms = torch.cat([all_terms, new_terms.unsqueeze(1)], dim=1)
                action_history = torch.cat(
                    [action_history[:, 1:], new_action.unsqueeze(1)], dim=1)
                value_history = torch.cat(
                    [value_history[:, 1:], new_value.unsqueeze(1)], dim=1)
                if new_subgoal is not None:
                    # already (B, 1, ...) from the inner stack vs (B, ...) from the loop
                    all_subgoals.append({k: v.unsqueeze(1) for k, v in new_subgoal.items()})

        return self._finish_n_actions(
            all_actions, all_values, all_scores, _cat_subgoals(all_subgoals), all_terms,
            return_scores, return_subgoals, return_terms, need_terms)

    def _finish_n_actions(self, actions, values, scores, subgoals, terms,
                          return_scores, return_subgoals, return_terms, need_terms):
        """The single exit both predict_n_actions branches fall through to.

        Fusing here rather than in either branch is what guarantees a cross-candidate value
        sees the WHOLE candidate set: at n > max_actions the rolling branch builds the stack
        in two pieces, and standardizing each piece separately would silently give a
        different -- and wrong -- ranking than the same n at max_actions.
        """
        if need_terms:
            assert terms is not None, (
                f'{type(self).__name__} ranks with a cross-candidate verifier value but its '
                f'verifier returned no per-term decomposition; _score_candidates must supply '
                f'`terms` (B, K) for the fusion in _fuse_scores.')
            # dim 1 is the CANDIDATE axis. Never dim 0 -- that is the parallel-episode
            # batch (50 independent test episodes at eval), and standardizing across it
            # would let one episode's candidate spread decide another episode's action.
            scores = self._fuse_scores(scores, terms)

        out = (actions, values)
        if return_scores:
            out = out + (scores,)
        if return_subgoals:
            out = out + (subgoals,)
        if return_terms:
            out = out + (terms,)
        return out

    @torch.inference_mode()
    def predict_action_best(
            self,
            obs_dict: Dict[str, torch.Tensor],
            n_actions: Optional[int] = None,
        ) -> Dict[str, torch.Tensor]:
        """Search readout, in the standard ``predict_action`` output format.

        Generates ``n_actions`` candidates via the sliding-window search
        (``predict_n_actions``), scores each with this policy's verifier, and returns one
        action chunk as ``{'action': (B, n_action_steps, Da), 'action_pred': (B, horizon,
        Da), 'scores': (B, n)}`` -- the shape the env runners and MultiStepWrapper expect.
        (Tensors only: the runners push this dict straight through ``dict_apply``, so a
        string mode tag here would crash them. Read the mode off ``policy.selection``.)

        ``n_actions`` is the TOTAL number of generations, identically under every
        selection rule, so curves from different rules share one x axis and one cost.

        WHICH chunk depends on ``self.selection``:
          * 'argmax'     -- the argmax-verifier-value candidate. Best-of-n over an oracle.
          * 'index'      -- a FIXED candidate by generation order (``selection_index``,
            1-based), scores ignored. What one slot of the search is worth with no
            selection on top; requires n >= that slot.
          * 'final_pass' -- only n-1 candidates are searched and scored; the n'th
            generation is conditioned on them and returned. It is not simulated and not
            compared to anything, so the verifier scalar never touches selection; it
            reaches the model only as search context. ``scores`` describes the n-1 context
            candidates (so the caller can log the spread), not the returned action, and is
            absent entirely at n=1.

        That split matches training exactly: ``generate_search_context`` runs at
        ``max_actions - 1`` and the loss covers slot ``max_actions - 1``, i.e. K
        generations in total. It also makes 'final_pass' at n=1 the empty-context
        conditional -- the same action 'argmax' and 'softmax' return there, since with one
        candidate there is nothing to select.

        Cost at width n is n samples under every rule; 'argmax'/'softmax' additionally run
        n verifier sims and 'final_pass' n-1.
        """
        # max_actions is None on a policy that consumes no context; it is not a width, so
        # there is no default n to fall back to there.
        n = n_actions if n_actions is not None else self.max_actions
        assert n is not None, ('n_actions must be given for a policy with no search width '
                              '(max_actions is None)')
        final_pass = self.selection == 'final_pass'
        # Under 'final_pass' the last of the n generations IS the returned action, so only
        # n-1 of them are searched.
        n_search = n - 1 if final_pass else n
        assert n_search >= 0, f'n_actions must be >= 1, got {n}'
        # `scores` is the scalar verifier value in every search_context mode; `values` may
        # be a wider context (e.g. a subgoal state), which is not rankable.
        # The crop scope spans the whole search so the obs and every candidate's subgoal
        # share one offset, exactly as in training. (In eval mode that offset is the
        # deterministic center crop, so this is belt-and-braces rather than load-bearing.)
        with self._crop_scope(), self._corrupt_scope():
            # encoded once and shared by the search AND (in 'final_pass') the final sample
            obs_features = self._encode_obs_features(obs_dict)
            actions = values = scores = None
            if n_search > 0:
                actions, values, scores = self.predict_n_actions(
                    obs_dict, verifier=self.verifier, n_actions=n_search,
                    return_scores=True, obs_features=obs_features)  # (B,n,H,Da), ctx, (B,n)

            if not final_pass:
                action_pred = select_candidate(                         # (B, H, Da)
                    actions, scores, self.selection, self.selection_temperature,
                    generator=self._selection_generator,
                    index=getattr(self, 'selection_index', None))
            else:
                # Condition on the last max_actions-1 candidates: that is the widest
                # context the model was ever trained at (the staircase memory mask tops out
                # at max_context_actions), so a longer one would index past cond_pos_emb.
                # `values`, not `scores` -- in the subgoal modes the context is the encoded
                # subgoal observation and the bare scalar would be the wrong width.
                # keep may be 0 -- at n=1 (nothing generated yet) or at max_actions=1 (the
                # ST k=1 arm, which has no context capacity at all). `actions[:, -0:]` is the
                # WHOLE tensor, so the empty case must pass None rather than a slice.
                keep = min(n_search, self.context_capacity)
                action_pred = self.predict_action(
                    obs_dict,
                    actions=actions[:, -keep:] if keep else None,
                    values=values[:, -keep:] if keep else None,
                    obs_features=obs_features,
                    # conditioned on `keep` context entries, so it is slot `keep` -- the
                    # ladder's cleanest reachable slot at this width.
                    **self._slot_kwargs(keep),
                )['action_pred']                                        # (B, H, Da)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        out = {
            'action': action_pred[:, start:end],            # (B, n_action_steps, Da)
            'action_pred': action_pred,
        }
        # Omitted rather than None when nothing was scored: the env runners push this dict
        # straight through dict_apply, which would call .detach() on a None.
        if scores is not None:
            out['scores'] = scores
        return out

