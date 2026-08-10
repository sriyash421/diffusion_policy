"""Smoke test for `selection: final_pass` -- the subgoal-only arm.

Checks the four things that would silently produce a wrong run rather than an exception:

  1. The existing arms are untouched: every pre-existing config still resolves to
     selection=argmax / slot_weight_decay=1.0, and _compute_loss at decay 1.0 returns
     EXACTLY the unweighted mean (so the six live runs' loss curve does not shift under
     them if they are ever resumed against this code).
  2. The kwarg whitelist still bites: a typo'd key must raise, or an ablation arm can end
     up secretly identical to its sibling (which is what _KNOWN_KWARGS exists to prevent).
  3. predict_action_best under final_pass returns an action that is NOT any of the n
     candidates -- i.e. it really is a fresh conditioned sample and not a relabelled argmax.
  4. The returned dict is tensor-only, because the env runners push it through dict_apply.

Run on a compute node: it builds a ResNet18 and a pool of PushT sims, which the login
node's shared ~10GB cgroup will not tolerate.
"""
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
            assert cfg.slot_weight_decay == 1.0, f'{name} decay={cfg.slot_weight_decay}'
            # Post-rename (AUDIT.md 9.9) the run dir is keyed on `arm`, not search_context.
            # Assert it resolves to THIS arm's label: a config whose directory disagreed with
            # its mechanism would file a whole column of SUCCESS_RATES.md under the wrong arm,
            # and `training.resume` would either resume the wrong run or start a fresh one.
            rn = OmegaConf.to_container(cfg, resolve=True)['run_name']
            assert rn.startswith(f'{cfg.arm}_corrupt-{cfg.corrupt_obs}_'), \
                f'{name}: run_name {rn!r} does not match arm {cfg.arm!r}'
            assert cfg.context_decay == 1.0, f'{name} context_decay={cfg.context_decay}'
        print(f'[ok] {len(ARGMAX_ARMS)} existing arms: selection=argmax, '
              f'slot_weight_decay=1.0, context_decay=1.0, run_name keyed on arm')

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
            torch.manual_seed(1)
            out = policy.predict_action_best(obs_eval, n_actions=n)
            act, pred, scores = out['action'], out['action_pred'], out['scores']
            assert act.shape == (B, policy.n_action_steps, 2), act.shape
            assert scores.shape == (B, n), scores.shape
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
                obs_eval, verifier=policy.verifier, n_actions=n, return_scores=True)[0]
            # same seed => the n candidates are reproduced; the deployed action is the
            # (n+1)-th draw, so it must differ from every one of them
            dists = [(pred - cands[:, k]).abs().max().item() for k in range(n)]
            if min(dists) < 1e-6:
                failures.append(
                    f'final_pass action coincides with candidate {int(np.argmin(dists))} '
                    f'-- it is not a fresh conditioned sample')
            else:
                print(f'[ok] deployed action differs from all {n} candidates '
                      f'(min |diff| = {min(dists):.3f} px)')

            # ---- 1b. the slot weights have the intended shape
            w = policy._slot_weights(torch.device('cpu'), torch.float32)
            K = policy.max_actions
            assert w is not None and w.shape == (K,), w
            assert torch.allclose(w.mean(), torch.tensor(1.0), atol=1e-6), \
                f'weights must average 1 or the loss scale shifts: mean={w.mean():.6f}'
            assert torch.all(w[1:] > w[:-1]), f'weights must rise with context length: {w}'
            ratio = (w[-1] / w[0]).item()
            expected = policy.slot_weight_decay ** -(K - 1)
            assert abs(ratio - expected) < 1e-4, (ratio, expected)
            print(f'[ok] slot weights: w_0={w[0]:.3f} .. w_{K-1}={w[-1]:.3f}, '
                  f'mean={w.mean():.6f}, last/first={ratio:.2f}x')

            # decay=1.0 must return None -- the uniform path, byte-identical to the arms
            # that predate this key
            policy.slot_weight_decay = 1.0
            assert policy._slot_weights(torch.device('cpu'), torch.float32) is None
            print('[ok] decay=1.0 takes the uniform path (returns None)')

            # ---- 1c. the weighting actually reaches the loss, and 1.0 is the plain mean
            policy.train()
            policy.set_crop_step(42, 0)
            torch.manual_seed(7)
            uniform = policy.compute_loss(batch)
            policy.slot_weight_decay = 0.9
            torch.manual_seed(7)
            decayed = policy.compute_loss(batch)
            print(f'[ok] loss decay=1.0 -> {uniform.item():.6f} | '
                  f'decay=0.9 -> {decayed.item():.6f}')
            if abs(uniform.item() - decayed.item()) < 1e-9:
                failures.append('slot_weight_decay=0.9 changed nothing -- the weighting is '
                                'not reaching the loss')

            # a mean-1 weighting of EQUAL per-slot losses must equal the flat mean, which is
            # what keeps val_loss on the same scale across arms
            from einops import reduce
            equal = torch.ones(5, K, 3, 2) * 0.37
            wq = policy._slot_weights(torch.device('cpu'), torch.float32)
            flat = reduce(equal, 'b ... -> b (...)', 'mean').mean()
            wtd = (reduce(equal, 'b k ... -> b k', 'mean') * wq).mean()
            assert torch.allclose(flat, wtd, atol=1e-6), (flat, wtd)
            print('[ok] mean-1 normalization preserves the loss scale')

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
