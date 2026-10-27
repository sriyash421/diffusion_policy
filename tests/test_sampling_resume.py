import random

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.search_procedure import SearchProcedureMixin
from eval_search_pusht import _episode_seed, merge_curves, rollout_max_rewards


class ToyWorkspace(BaseWorkspace):
    include_keys = ('next_inner_idx',)

    def __init__(self, folder):
        super().__init__(OmegaConf.create({}), str(folder))
        self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.4))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.next_inner_idx = 2

    def update(self):
        x = torch.randn(4, 3) + random.random() + np.random.random()
        self.optimizer.zero_grad()
        loss = self.model(x).square().mean()
        loss.backward()
        self.optimizer.step()
        return loss.detach()


def test_checkpoint_restores_update_rng_and_cursor(tmp_path):
    uninterrupted = ToyWorkspace(tmp_path)
    uninterrupted.update()
    checkpoint = uninterrupted.save_checkpoint(use_thread=False)
    expected = [uninterrupted.update() for _ in range(3)]
    resumed = ToyWorkspace(tmp_path)
    resumed.load_checkpoint(checkpoint)
    # Initialization after loading is allowed; the trainer restores at loop entry.
    torch.randn(20)
    random.random()
    np.random.random()
    resumed.restore_training_rng()
    assert resumed.next_inner_idx == 2
    actual = [resumed.update() for _ in range(3)]
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for a, b in zip(uninterrupted.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_seed_is_paired_across_search_widths():
    seeds = [_episode_seed(42, n, 7) for n in (1, 2, 4, 8, 16, 32, 64)]
    assert len(set(seeds)) == 1
    assert seeds[0] != _episode_seed(42, 1, 8)


def test_eval_does_not_merge_different_candidate_rng():
    old = {'n': [1], 'candidate_rng': None}
    new = {'n': [2], 'candidate_rng': 'paired_decision_v3'}
    assert merge_curves(old, new) == new


def test_wider_search_does_not_shift_later_decision_noise():
    class Env:
        def call_each(self, *args, **kwargs):
            pass

        def reset(self):
            self.step_i = 0
            return {'obs': np.zeros((1, 2), dtype=np.float32)}

        def step(self, action):
            self.step_i += 1
            return {'obs': np.zeros((1, 2), dtype=np.float32)}, None, self.step_i == 2, {}

        def call(self, _, name):
            return [[0.1, 0.2]] if name == 'reward' else [0.2]

    class Policy(SearchProcedureMixin):
        def __init__(self):
            self.first_draws = []
            self.set_sample_seeds([_episode_seed(42, 1, 7)])

        def reset(self):
            pass

        def predict_action_best(self, obs, n_actions):
            draws = [self._init_noise((1, 8, 2), torch.float32, 'cpu')
                     for _ in range(n_actions)]
            self.first_draws.append(draws[0])
            return {'action': draws[0]}

    policies = [Policy(), Policy()]
    for policy, n in zip(policies, (1, 4)):
        rollout_max_rewards(Env(), policy, [np.zeros(5)], torch.device('cpu'), n)
    for a, b in zip(policies[0].first_draws, policies[1].first_draws):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert not torch.equal(*policies[0].first_draws)
