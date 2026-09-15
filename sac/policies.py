"""The verifier head: a HARD, max-over-candidates Q, trained beside SAC but never inside it.

WHY NOT SAC'S OWN CRITIC. SAC learns the entropy-augmented `Q_soft^pi` -- the value of its own
stochastic policy, plus `-alpha log pi` at every future step. With a sparse reward whose entire
magnitude is 1.0 and ~38 chunks of entropy accumulating against it, the soft value can be mostly
entropy. Best-of-N does not evaluate the actor's stochastic policy; it picks the argmax over n
candidates and executes it. So the backup here is the operation the deployment performs:

    y = r + gamma * (1 - d) * max_{a' in A_M(s')} mean_ensemble Q_target(s', a')
    A_M(s') = { tanh(mu(s')) } u { M - 1 actor samples }

ISOLATION. Built AFTER the policy's own `__init__`, carrying its own optimizer, with the
features detached -- the same construction `recurrent_ppo.corrupt_policy._QHeadMixin` uses and
tests. It provably cannot move the policy, so the SAC run is unchanged by its presence and the
actor remains a data-collection engine rather than something this head is co-adapting with.

THE LADDER. One head per rung, over a shared trunk. Rung `tau` sees the terminate-at-tau MDP:
its own `(reward, done)`, and a `live` mask that drops transitions from episodes that crossed
tau earlier. `Q_0.95` is therefore the SAME function it would be if trained alone -- the lower
rungs only shape the trunk, and they have signal the demonstrations already contain (658 and
226 terminals at 0.80 and 0.85, against ZERO at 0.95).
"""

import copy

import torch as th
from torch import nn


def _mlp(in_dim, net_arch, out_dim):
    layers, last = [], in_dim
    for hidden in net_arch:
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    return nn.Sequential(*layers, nn.Linear(last, out_dim))


class BONHead(nn.Module):
    """An ensemble of `n_critics` nets, each emitting one value per rung."""

    def __init__(self, feature_dim, action_dim, n_tau, n_critics=2, net_arch=(256, 256)):
        super().__init__()
        self.n_tau, self.n_critics = int(n_tau), int(n_critics)
        self.nets = nn.ModuleList([_mlp(feature_dim + action_dim, net_arch, n_tau)
                                   for _ in range(self.n_critics)])

    def forward(self, features, actions):
        """-> (n_critics, B, n_tau)."""
        x = th.cat([features, actions], dim=-1)
        return th.stack([net(x) for net in self.nets], dim=0)

    def mean(self, features, actions):
        """The point estimate used to RANK. `min` controls overestimation in a backup; for
        ranking the mean is the better estimator and the spread is the honest uncertainty."""
        return self.forward(features, actions).mean(0)

    def spread(self, features, actions):
        """Ensemble std -> (B, n_tau). When this exceeds the spread ACROSS candidates, the
        verifier is not resolving the question it is being asked and BON should fall back to
        n=1 rather than rank on noise."""
        return self.forward(features, actions).std(0)


def attach_bon_head(policy, n_tau, n_critics=2, net_arch=(256, 256), lr=3e-4):
    """Build the head and its optimizer on an already-constructed SAC policy.

    After `policy.__init__`, so `policy.optimizer`/`policy.critic.optimizer` were built over the
    parameters that existed then and cannot see these. That is what makes the isolation a fact
    about the construction order rather than a promise.
    """
    feature_dim = policy.critic.features_extractor.features_dim
    action_dim = policy.action_space.shape[0]
    head = BONHead(feature_dim, action_dim, n_tau, n_critics, net_arch).to(policy.device)
    policy.bon_head = head
    policy.bon_head_target = copy.deepcopy(head).requires_grad_(False)
    policy.bon_optimizer = th.optim.Adam(head.parameters(), lr=lr)
    return head
