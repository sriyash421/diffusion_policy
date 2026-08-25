"""Smoke test for `selection: final_pass` -- the subgoal-only arm.

Checks the four things that would silently produce a wrong run rather than an exception:

  1. The existing arms are untouched: every pre-existing config still resolves to
     selection=argmax / slot_weight_decay=False / slot_weights.mode=uniform, and the
     uniform path returns EXACTLY the unweighted mean (so the live runs' loss curve does
     not shift under them if they are ever resumed against this code). 1.0 is rejected
     rather than silently meaning "off".
  2. The kwarg whitelist still bites: a typo'd key must raise, or an ablation arm can end
     up secretly identical to its sibling (which is what _KNOWN_KWARGS exists to prevent).
  3. predict_action_best under final_pass returns an action that is NOT any of the n
     candidates -- i.e. it really is a fresh conditioned sample and not a relabelled argmax.
  4. The returned dict is tensor-only, because the env runners push it through dict_apply.

Run on a compute node: it builds a ResNet18 and a pool of PushT sims, which the login
node's shared ~10GB cgroup will not tolerate.

  python scripts/selection_smoke.py
"""
if __name__ == "__main__":
    # `python scripts/selection_smoke.py` puts scripts/ on sys.path, not the repo root, so
    # `import diffusion_policy` failed with ModuleNotFoundError however the CWD was set.
    # Same preamble the other scripts in here use.
    import sys, os, pathlib
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import math
import numpy as np
import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

import hydra
from hydra import compose, initialize_config_dir
import os

CFG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'diffusion_policy', 'config'))

# The six argmax arms, as (label, base config, overrides). They no longer have a config
# file each -- one base plus three overrides reproduces every cell of the 3x2 matrix, which
# is the form launch_round8_29demo.sh already uses (see config/archive/README.md). The
# override line is now the thing that can be got wrong, so it is what this asserts against.
ARGMAX_ARMS = [
    (f'{arm}/{"corrupt" if corrupt else "clean"}',
     ['search_context=' + ctx, 'arm=' + arm, 'corrupt_obs=' + str(corrupt)])
    for ctx, arm in (('value', 'value'),
                     ('subgoal', 'subgoal-chosen4value'),
                     ('subgoal_value', 'subgoal-value'))
    for corrupt in (False, True)
]


def _tiny(cfg, n_envs=4):
    """Shrink a config so the smoke test is seconds, not minutes."""
    cfg.policy.max_actions = 4
    cfg.policy.num_inference_steps = 2
    cfg.policy.verifier_n_envs = n_envs
    cfg.policy.n_layer = 1
    cfg.policy.n_cond_layers = 1
    cfg.policy.n_emb = 64
    return cfg


def _fake_normalizer(policy, shape_meta):
    """Fit the normalizer on random data -- the smoke test never touches the real dataset."""
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    nz = LinearNormalizer()
    data = {'action': torch.rand(64, 2) * 512}
    for key, attr in shape_meta['obs'].items():
        if attr.get('type', 'low_dim') == 'low_dim':
            data[key] = torch.rand(64, attr['shape'][0]) * 512
    nz.fit(data, last_n_dims=1, mode='limits')
    for key, attr in shape_meta['obs'].items():
        if attr.get('type', 'low_dim') == 'rgb':
            nz[key] = get_image_range_normalizer()
    policy.set_normalizer(nz)
    return nz


def _batch(shape_meta, policy, B=2, T=16):
    obs = {}
    for key, attr in shape_meta['obs'].items():
        shape = tuple(attr['shape'])
        if attr.get('type', 'low_dim') == 'rgb':
            obs[key] = torch.rand(B, T, *shape)
        else:
            obs[key] = torch.rand(B, T, *shape) * 512
    return {'obs': obs, 'action': torch.rand(B, T, 2) * 512}


def _check_context_decay(policy, lam=0.9):
    """The memory mask must carry lambda^(distance from the LATEST visible entry).

    This is the whole contract of context_decay and it is easy to get subtly wrong -- the
    weight is a function of the (slot, entry) PAIR, because the staircase mask gives slot c
    exactly c context entries, so "the latest" differs per slot. A mask that decayed by
    absolute entry index instead would look plausible and be wrong.
    """
    model = policy.model
    old = model.context_decay
    K = policy.max_actions
    Kc = model.max_context_actions
    off = model.n_obs_steps
    try:
        model.context_decay = lam
        mask, _ = model._build_memory_masks(
            batch_size=1, n_candidates=K, n_context_actions=Kc,
            context_lengths=None, device=torch.device('cpu'))
        # (B*n_head, K*horizon, S) -> take head 0, the first decoder row of each candidate
        m = mask.reshape(1, model.n_head, K, model.horizon, -1)[0, 0, :, 0, :]   # (K, S)
        neg = torch.finfo(m.dtype).min
        bad = []
        for c in range(K):
            length = c                      # staircase: slot c sees entries 0..c-1
            for j in range(Kc):
                got = m[c, off + j].item()
                if j >= length:
                    if got != neg:
                        bad.append(f'slot {c} entry {j} should be masked, got {got:.4f}')
                else:
                    want = (length - 1 - j) * math.log(lam)
                    if abs(got - want) > 1e-5:
                        bad.append(f'slot {c} entry {j}: got {got:.4f} want {want:.4f}')
            if abs(m[c, 0].item()) > 1e-9:
                bad.append(f'slot {c}: obs position biased ({m[c,0].item():.4f}), must be 0')
        if bad:
            raise AssertionError('context_decay mask wrong:\n  ' + '\n  '.join(bad[:6]))
        # the latest visible entry always gets weight exactly 1 (bias 0), at every length
        latest = [m[c, off + c - 1].item() for c in range(1, K)]
        assert max(abs(v) for v in latest) < 1e-9, latest
        print(f'[ok] context_decay: latest entry bias 0 at every context length 1..{K-1}; '
              f'entry j gets {lam}^(m-1-j)')

        # decay=1.0 must return the original bool mask, i.e. byte-identical behaviour
        model.context_decay = 1.0
        b, _ = model._build_memory_masks(1, K, Kc, None, torch.device('cpu'))
        assert b.dtype == torch.bool, f'decay=1.0 must keep the bool mask, got {b.dtype}'
        print('[ok] context_decay=1.0 keeps the original bool mask (arms unaffected)')
    finally:
        model.context_decay = old


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    failures = []

    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        # ---- 1a. existing arms still declare the old behaviour
        for name, overrides in ARGMAX_ARMS:
            cfg = compose(config_name='train_pusht_diffusion_search',
                          overrides=overrides)
            assert cfg.selection == 'argmax', f'{name} selection={cfg.selection}'
            # False, not 1.0: 'off' is explicit now (a no-op float is rejected), and the
            # uniform path it selects is numerically what 1.0 always did.
            assert cfg.slot_weight_decay is False, f'{name} decay={cfg.slot_weight_decay}'
            # Post-rename (AUDIT.md 9.9) the run dir is keyed on `arm`, not search_context.
            # Assert it resolves to THIS arm's label: a config whose directory disagreed with
            # its mechanism would file a whole column of SUCCESS_RATES.md under the wrong arm,
            # and `training.resume` would either resume the wrong run or start a fresh one.
            assert cfg.slot_weights.mode == 'uniform', \
                f'{name} slot_weights.mode={cfg.slot_weights.mode}'
            assert cfg.sw_suffix == '', f'{name} sw_suffix={cfg.sw_suffix!r}'
            rn = OmegaConf.to_container(cfg, resolve=True)['run_name']
            # run_name carries every axis that makes two runs incomparable, so none of them
            # can collide on one checkpoint directory (which `training.resume` would then
            # silently continue): k{n_candidates}, ver-{verifier_tag} (the scoring rule
            # changed on 2026-08-19), and sw_suffix (the slot-weight profile). sw_suffix is
            # EMPTY at the defaults, so this assertion also pins that adding slot_weights
            # did not move any existing run's directory.
            assert rn.startswith(
                f'{cfg.arm}_k{cfg.n_candidates}_ver-{cfg.verifier_tag}_'
                f'corrupt-{cfg.corrupt_obs}_'), \
                f'{name}: run_name {rn!r} does not match arm {cfg.arm!r}'
            assert cfg.context_decay == 1.0, f'{name} context_decay={cfg.context_decay}'

        # a non-uniform profile MUST be able to name itself in the directory
        cfg = compose(config_name='train_pusht_diffusion_search',
                      overrides=ARGMAX_ARMS[0][1] + [
                          'slot_weights.mode=linear', 'slot_weights.ratio=4.857',
                          'sw_suffix=_sw-lin4857'])
        rn = OmegaConf.to_container(cfg, resolve=True)['run_name']
        assert '_sw-lin4857_' in rn, rn
        print(f'[ok] sw_suffix reaches run_name: {rn}')
        print(f'[ok] {len(ARGMAX_ARMS)} existing arms: selection=argmax, '
              f'slot_weight_decay=False, context_decay=1.0, run_name keyed on arm')

        cfg = _tiny(compose(config_name='train_pusht_diffusion_search_subgoal_only'))
        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)

        # ---- 2. the kwarg whitelist still bites
        from diffusion_policy.policy.diffusion_transformer_search_policy import (
            DiffusionTransformerSearchPolicy)
        try:
            DiffusionTransformerSearchPolicy._validate_kwargs(
                {'max_actions': 4, 'selectionn': 'final_pass'})
            failures.append('typo\'d kwarg `selectionn` did NOT raise')
        except TypeError as e:
            assert 'selectionn' in str(e)
            print('[ok] typo\'d config key still raises')

        # ---- build the final_pass policy
        policy = hydra.utils.instantiate(cfg.policy)
        _fake_normalizer(policy, shape_meta)
        policy.eval()
        assert policy.selection == 'final_pass'
        assert policy.slot_weight_decay == 0.9
        print(f'[ok] policy built: selection={policy.selection} '
              f'slot_weight_decay={policy.slot_weight_decay} max_actions={policy.max_actions}')

        try:
            B, n = 2, 3
            batch = _batch(shape_meta, policy, B=B)
            obs_eval = {k: v[:, :policy.n_obs_steps] for k, v in batch['obs'].items()}

            # ---- 3. the executed action is a FRESH sample, not one of the candidates
            # n is the TOTAL generation count, so final_pass searches at n-1 and the n'th
            # sample is the one returned; `scores` therefore covers n-1 candidates.
            torch.manual_seed(1)
            out = policy.predict_action_best(obs_eval, n_actions=n)
            act, pred, scores = out['action'], out['action_pred'], out['scores']
            assert act.shape == (B, policy.n_action_steps, 2), act.shape
            assert scores.shape == (B, n - 1), scores.shape
            print(f'[ok] final_pass shapes: action={tuple(act.shape)} scores={tuple(scores.shape)}')

            # ---- 4. tensor-only dict (the runners dict_apply over it)
            bad = [k for k, v in out.items() if not torch.is_tensor(v)]
            if bad:
                failures.append(f'non-tensor values in predict_action_best output: {bad} '
                                f'-- dict_apply in the env runner will crash on these')
            else:
                print('[ok] predict_action_best returns tensors only')

            torch.manual_seed(1)
            cands = policy.predict_n_actions(
                obs_eval, verifier=policy.verifier, n_actions=n - 1, return_scores=True)[0]
            # same seed => the n-1 context candidates are reproduced; the deployed action
            # is the n'th draw, so it must differ from every one of them
            dists = [(pred - cands[:, k]).abs().max().item() for k in range(n - 1)]
            if min(dists) < 1e-6:
                failures.append(
                    f'final_pass action coincides with candidate {int(np.argmin(dists))} '
                    f'-- it is not a fresh conditioned sample')
            else:
                print(f'[ok] deployed action differs from all {n - 1} candidates '
                      f'(min |diff| = {min(dists):.3f} px)')

            # ---- 3b. n counts GENERATIONS, so at n=1 there is nothing to search: every
            # rule returns the empty-context conditional, and no scores are produced.
            outs = {}
            for mode in ('argmax', 'softmax', 'final_pass'):
                policy.selection = mode
                torch.manual_seed(7)
                outs[mode] = policy.predict_action_best(obs_eval, n_actions=1)
            policy.selection = 'final_pass'
            if 'scores' in outs['final_pass']:
                failures.append('final_pass at n=1 still returned scores -- it should '
                                'search at n-1 = 0 and score nothing')
            ref = outs['argmax']['action_pred']
            for mode in ('softmax', 'final_pass'):
                d = (outs[mode]['action_pred'] - ref).abs().max().item()
                if d > 1e-6:
                    failures.append(
                        f'{mode} at n=1 differs from argmax by {d:.4g} -- with one '
                        f'generation there is nothing to select, so all rules must agree')
            if not failures:
                print('[ok] all three selection rules agree at n=1 (nothing to select)')

            # ---- 3c. selection must not consume the SAMPLER's randomness. softmax used
            # to draw from the global stream via Categorical.sample(), which shifted every
            # subsequent torch.randn in conditional_sample -- so turning softmax on
            # reseeded the rollout rather than only changing the pick.
            states = {}
            for mode in ('argmax', 'softmax'):
                policy.selection = mode
                torch.manual_seed(11)
                policy.predict_action_best(obs_eval, n_actions=n)
                states[mode] = torch.random.get_rng_state()
            policy.selection = 'final_pass'
            if not torch.equal(states['argmax'], states['softmax']):
                failures.append('softmax left the global RNG in a different state than '
                                'argmax -- selection is still stealing sampler draws')
            else:
                print('[ok] softmax draws from its own generator (global RNG untouched)')

            # ---- 1b. the slot weight PROFILES have the intended shape
            K = policy.max_actions
            cpu, f32 = torch.device('cpu'), torch.float32

            def profile(**spec):
                """Resolve a slot_weights block on this policy and return the vector."""
                policy._init_slot_weighting(False, spec)
                return policy._slot_weights(cpu, f32)

            # geometric is the legacy shape; keep asserting exactly what it always did
            w = profile(mode='geometric', decay=0.9)
            assert w is not None and w.shape == (K,), w
            assert torch.allclose(w.mean(), torch.tensor(1.0), atol=1e-6), \
                f'weights must average 1 or the loss scale shifts: mean={w.mean():.6f}'
            assert torch.all(w[1:] > w[:-1]), f'weights must rise with context length: {w}'
            ratio = (w[-1] / w[0]).item()
            assert abs(ratio - 0.9 ** -(K - 1)) < 1e-4, ratio
            print(f'[ok] geometric: w_0={w[0]:.3f} .. w_{K-1}={w[-1]:.3f}, '
                  f'mean={w.mean():.6f}, last/first={ratio:.2f}x')

            # BACK-COMPAT: the legacy scalar must resolve to the identical vector, or every
            # run trained under it is no longer reproducible from its own config.
            policy._init_slot_weighting(0.9, None)
            assert torch.allclose(policy._slot_weights(cpu, f32), w, atol=1e-7)
            assert policy.slot_weight_decay == 0.9, \
                'the .slot_weight_decay attribute is read by selection_smoke and by ' \
                'dump_candidate_scores._slot_weight_table; it must survive'
            assert policy.slot_weight_spec['val'] == 'trained', \
                'the legacy scalar had val_loss computed WITH the weighting; keep it'
            print('[ok] legacy slot_weight_decay=0.9 == slot_weights geometric 0.9')

            # linear at ratio decay^-(K-1) has the SAME endpoints as geometric(decay) and
            # differs only in curvature -- that pairing is the only one that varies shape
            # without also varying spread, so it is the comparison worth running.
            match = 0.9 ** -(K - 1)
            lin = profile(mode='linear', ratio=match)
            assert abs((lin[-1] / lin[0]).item() - match) < 1e-3, lin
            assert torch.allclose(lin.mean(), torch.tensor(1.0), atol=1e-6)
            assert not torch.allclose(lin, w), 'linear and geometric must differ in between'
            print(f'[ok] linear ratio={match:.4f} matches geometric endpoints, '
                  f'differs mid-profile ({lin[K//2]:.3f} vs {w[K//2]:.3f})')

            for spec in (dict(mode='last_only'),
                         dict(mode='list', weights=[1.0] * (K - 1) + [3.0])):
                p = profile(**spec)
                assert p.shape == (K,) and torch.allclose(p.mean(), torch.tensor(1.0), atol=1e-6), \
                    (spec, p)
            print('[ok] last_only and explicit list profiles resolve, mean 1')

            # ---- 1b'. rejections. Each of these is a config that LOOKS like it asks for a
            # weighting and would otherwise train uniform while the run name claims a profile.
            def rejects(label, **spec):
                try:
                    policy._init_slot_weighting(False, spec)
                except (ValueError, AssertionError, TypeError):
                    return
                failures.append(f'slot_weights {label} was accepted but must raise')

            rejects('mode=geometric with no decay', mode='geometric')
            rejects('decay=1.0 (a silent no-op)', mode='geometric', decay=1.0)
            rejects('ratio=1.0 (a silent no-op)', mode='linear', ratio=1.0)
            rejects('unknown mode', mode='quadratic')
            # THE IMPORTANT ONE: _KNOWN_KWARGS only guards the top-level name, so without a
            # nested whitelist `rato: 4` trains uniform and says nothing.
            rejects('typo in a nested key', mode='linear', rato=4.0)
            rejects('list of the wrong length', mode='list', weights=[1.0] * (K + 1))
            rejects('schedule that ends before it starts', mode='linear', ratio=2.0,
                    schedule={'start_step': 10, 'end_step': 5})
            rejects('schedule on mode=uniform', mode='uniform',
                    schedule={'start_step': 0, 'end_step': 10})
            rejects('unknown val', mode='linear', ratio=2.0, val='sometimes')
            try:
                policy._init_slot_weighting(0.9, {'mode': 'linear', 'ratio': 2.0})
            except ValueError:
                pass
            else:
                failures.append('both slot_weight_decay and slot_weights were accepted; '
                                'one silently wins and the run name cannot say which')
            print('[ok] every malformed slot_weights spec raises (incl. nested typos)')

            # uniform is the default and must give the None (unweighted) path
            policy._init_slot_weighting(False, None)
            assert policy._slot_weights(cpu, f32) is None
            print('[ok] default -> uniform path (None)')

            # ---- 1b''. curriculum: ramps from uniform to the target, mean 1 throughout
            policy._init_slot_weighting(False, dict(
                mode='geometric', decay=0.9,
                schedule={'shape': 'linear', 'start_step': 0, 'end_step': 100}))
            w0 = policy._slot_weights(cpu, f32, step=0)
            w50 = policy._slot_weights(cpu, f32, step=50)
            w100 = policy._slot_weights(cpu, f32, step=100)
            w999 = policy._slot_weights(cpu, f32, step=999)
            assert torch.allclose(w0, torch.ones(K), atol=1e-6), w0
            assert torch.allclose(w100, w, atol=1e-6), w100
            assert torch.allclose(w999, w100, atol=1e-6), 'past end_step must clamp'
            assert ((w50 - w0).abs() > 1e-6).any(), 'the midpoint must actually move'
            for name, v in (('start', w0), ('mid', w50), ('end', w100)):
                # THE load-bearing property: the blend is done in weight space, so mean(w)
                # is 1 at EVERY point of the ramp and gradient_clip_norm sees a constant
                # loss scale. Interpolating the decay parameter would not give this.
                assert torch.allclose(v.mean(), torch.tensor(1.0), atol=1e-6), (name, v.mean())
            policy.set_slot_weight_step(100)
            assert torch.allclose(policy._slot_weights(cpu, f32), w100, atol=1e-6), \
                'set_slot_weight_step must drive the profile the loss sees'
            print('[ok] curriculum: uniform -> target, mean 1 at every point, '
                  'set_slot_weight_step drives it')

            # ---- 1c. the weighting actually reaches the loss
            policy.train()
            policy.set_crop_step(42, 0)
            policy._init_slot_weighting(False, None)
            torch.manual_seed(7)
            uniform = policy.compute_loss(batch)
            policy._init_slot_weighting(False, dict(mode='geometric', decay=0.9))
            torch.manual_seed(7)
            decayed = policy.compute_loss(batch)
            torch.manual_seed(7)
            forced_uniform = policy.compute_loss(batch, slot_weighting=False)
            print(f'[ok] loss uniform -> {uniform.item():.6f} | '
                  f'geometric 0.9 -> {decayed.item():.6f}')
            if abs(uniform.item() - decayed.item()) < 1e-9:
                failures.append('slot_weights geometric changed nothing -- the weighting is '
                                'not reaching the loss')
            if abs(uniform.item() - forced_uniform.item()) > 1e-9:
                failures.append('compute_loss(slot_weighting=False) did not reproduce the '
                                'uniform loss -- val_loss would not be a fixed yardstick')
            else:
                print('[ok] compute_loss(slot_weighting=False) == the uniform loss exactly')

            # a curriculum at step 0 must be the uniform loss to float noise, or the ramp
            # starts from something other than where it claims
            policy._init_slot_weighting(False, dict(
                mode='geometric', decay=0.9,
                schedule={'shape': 'linear', 'start_step': 0, 'end_step': 100}))
            policy.set_slot_weight_step(0)
            torch.manual_seed(7)
            curr0 = policy.compute_loss(batch)
            if abs(curr0.item() - uniform.item()) > 1e-9:
                failures.append(f'curriculum at step 0 ({curr0.item():.9f}) != uniform '
                                f'({uniform.item():.9f}); the ramp does not start uniform')
            else:
                print('[ok] curriculum at step 0 == uniform loss exactly')

            # a mean-1 weighting of EQUAL per-slot losses must equal the flat mean, which is
            # what keeps val_loss on the same scale across arms
            from einops import reduce
            equal = torch.ones(5, K, 3, 2) * 0.37
            policy._init_slot_weighting(False, dict(mode='geometric', decay=0.9))
            wq = policy._slot_weights(cpu, f32)
            flat = reduce(equal, 'b ... -> b (...)', 'mean').mean()
            wtd = (reduce(equal, 'b k ... -> b k', 'mean') * wq).mean()
            assert torch.allclose(flat, wtd, atol=1e-6), (flat, wtd)
            print('[ok] mean-1 normalization preserves the loss scale')
            policy._init_slot_weighting(False, None)

            # ---- 5. context recency decay: the mask carries lambda^(dist from latest)
            _check_context_decay(policy)
        finally:
            policy.close()

    if failures:
        print('\nFAILURES:')
        for f in failures:
            print(' -', f)
        raise SystemExit(1)
    print('\nall selection smoke checks passed')


if __name__ == '__main__':
    main()
