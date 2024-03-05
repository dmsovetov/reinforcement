import math
from functools import partial
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical
from torch.nn import Module, Sequential, Linear, ReLU, Tanh


class BasicPositionalEncoding(Module):
    def __init__(self, in_features: int, k: int):
        super(BasicPositionalEncoding, self).__init__()
        self.k = k
        self.in_features = in_features

    @property
    def out_features(self) -> int:
        return self.k * 2 * self.in_features + self.in_features

    def forward(self, x):
        out = [x]

        for j in range(self.k):
            out.append(torch.sin(2 ** j * x))
            out.append(torch.cos(2 ** j * x))

        return torch.cat(out, dim=1)


class SimpleDQN(Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 64):
        super(SimpleDQN, self).__init__()

        # encoder = BasicPositionalEncoding(in_features, 10)

        self.net = Sequential(
            # encoder,
            # Linear(encoder.out_features, hidden),
            Linear(in_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, out_features)
        )

    def forward(self, state: Tensor):
        return self.net(state)


class Policy(Module):
    def __init__(self, net: Module):
        super(Policy, self).__init__()
        self.net = net

    def forward(self, state):
        logits = self.net(state)
        probs = F.softmax(logits, dim=1)
        m = Categorical(probs)
        action = m.sample()
        return action.tolist(), m.log_prob(action), probs

    def logits(self, state):
        return self.net(state)


class MLPOld(Module):
    def __init__(self, in_features, out_features, hidden: int = 64, layers: int = 1):
        super(MLPOld, self).__init__()

        nodes = []

        for i in range(layers):
            nodes += [Linear(in_features if i == 0 else hidden, hidden), ReLU()]

        nodes.append(Linear(hidden, out_features))

        self.net = Sequential(*nodes)

    def forward(self, state):
        return self.net(state)


class MLP(Module):
    def __init__(self, in_features: int, out_features: int = 0, hidden: List[int] = None, activation=None):
        super(MLP, self).__init__()

        if hidden is None:
            hidden = [64]
        if activation is None:
            activation = ReLU

        layers = []
        for i, n in enumerate(hidden):
            layers += [Linear(in_features if i == 0 else hidden[i - 1], n), activation()]

        if out_features > 0:
            layers.append(Linear(hidden[-1], out_features))

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def initialize_ortho(module_gains: Dict[Module, float]):
    def init_weights(m: Module, g: float = 1):
        if isinstance(m, Linear):
            nn.init.orthogonal_(m.weight, gain=g)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    for module, gain in module_gains.items():
        module.apply(partial(init_weights, g=gain))


class AC(Module):
    def __init__(self, state_dim, action_dim, hidden=None, ortho_init: bool = True):
        super(AC, self).__init__()

        if hidden is None:
            hidden = [64]

        self.policy_hidden = MLP(state_dim, hidden=hidden, activation=Tanh)
        self.value_hidden = MLP(state_dim, hidden=hidden, activation=Tanh)

        self.policy_net = Linear(hidden[-1], action_dim)
        self.value_net = Linear(hidden[-1], 1)

        if ortho_init:
            initialize_ortho({
                self.policy_hidden: math.sqrt(2),
                self.value_hidden: math.sqrt(2),
                self.policy_net: 0.01,
                self.value_net: 1,
            })

    def state_value(self, state: Tensor):
        vh = self.value_hidden(state)
        return self.value_net(vh)

    def action(self, state):
        ah = self.policy_hidden(state)
        logits = self.policy_net(ah)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return action.tolist(), distribution.log_prob(action), distribution.entropy()

    def forward(self, state):
        v = self.state_value(state)
        action, log_prob, entropy = self.action(state)
        return action, log_prob, entropy, v.squeeze(-1)

