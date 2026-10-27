"""BCSearchWrapper's eval-only state noise (eval_search --eval-obs-noise-level)."""
import torch

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.search_policy_robomonkey import BCSearchWrapper


class _FakeBC(torch.nn.Module):
    """Records the obs it is asked to act on; returns zero chunks."""

    def __init__(self):
        super().__init__()
        self.obs_keys = ('ee', 'obj')
        self.n_obs_steps, self.n_action_steps = 1, 4
        self.normalizer = LinearNormalizer()
        g = torch.Generator().manual_seed(0)
        self.normalizer.fit({'ee': torch.randn(500, 7, generator=g) * 0.1 + 0.3,
                             'obj': torch.randn(500, 6, generator=g) * 0.05 - 0.1},
                            mode='limits')
        self.seen = []

    def predict_action(self, obs):
        self.seen.append({k: v.clone() for k, v in obs.items() if torch.is_tensor(v)})
        B = obs['ee'].shape[0]
        return {'action': torch.zeros(B, 4, 7), 'action_pred': torch.zeros(B, 8, 7)}


class _FakeVerifier:
    image_obs_key = 'rgb'

    def __init__(self):
        self.seen = []

    def get_value(self, obs, actions):
        self.seen.append({k: v.clone() for k, v in obs.items() if torch.is_tensor(v)})
        return torch.zeros(actions.shape[0])


def _obs():
    return {'ee': torch.full((1, 1, 7), 0.3), 'obj': torch.full((1, 1, 6), -0.1),
            'rgb': torch.rand(1, 1, 3, 8, 8)}


def _run(level, seed, n=4, batch='1', monkeypatch=None):
    monkeypatch.setenv('BC_VERIFIER_SCORE', 'full')
    monkeypatch.setenv('BC_BATCH_CANDIDATES', batch)
    base, ver = _FakeBC(), _FakeVerifier()
    w = BCSearchWrapper(base, ver, max_actions=n)
    w.eval_obs_noise_level = level
    obs = _obs()
    w.predict_n_actions(obs, ver, n, obs_noise_seed=seed)
    return base, ver, obs


def test_off_is_identity(monkeypatch):
    base, _, obs = _run(None, 7, monkeypatch=monkeypatch)
    assert torch.equal(base.seen[0]['ee'][0], obs['ee'][0])


def test_shared_across_candidates_and_verifier_clean(monkeypatch):
    for batch in ('1', '0'):
        base, ver, obs = _run(0.25, 7, batch=batch, monkeypatch=monkeypatch)
        rows = torch.cat([s['ee'] for s in base.seen])            # (N, 1, 7)
        assert not torch.allclose(rows[0], obs['ee'][0])           # noised
        assert torch.equal(rows, rows[:1].expand_as(rows))         # one draw for all N
        assert torch.equal(base.seen[0]['rgb'][:1], obs['rgb'])    # images untouched
        assert all(torch.equal(s['ee'][:1], obs['ee']) for s in ver.seen)  # verifier clean


def test_deterministic_per_replan_seed_and_magnitude(monkeypatch):
    a = _run(0.25, 7, monkeypatch=monkeypatch)[0].seen[0]['ee']
    b = _run(0.25, 7, monkeypatch=monkeypatch)[0].seen[0]['ee']
    c = _run(0.25, 8, monkeypatch=monkeypatch)[0].seen[0]['ee']
    assert torch.equal(a, b) and not torch.equal(a, c)
    # normalized-space noise std at level 0.25 (t=25): sqrt(1 - alphas_cumprod[25])
    base = _FakeBC()
    w = BCSearchWrapper(base, _FakeVerifier())
    w.eval_obs_noise_level = 0.25
    abar = float(w.obs_noise_scheduler.alphas_cumprod[25])
    x = {'ee': torch.zeros(1, 4000, 7), 'obj': torch.zeros(1, 4000, 6)}
    nx = base.normalizer['ee'].normalize(x['ee'])
    out = base.normalizer['ee'].normalize(w.noisy_policy_obs(x, 3)['ee'])
    resid = out - abar ** 0.5 * nx
    assert abs(resid.std().item() - (1 - abar) ** 0.5) < 0.01
    assert 0.25 < (1 - abar) ** 0.5 < 0.32
