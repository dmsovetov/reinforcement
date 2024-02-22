import torch
from torch import Tensor
from torch.distributions import Categorical
from torch.nn import Module, Sequential, Linear, ReLU
import torch.nn.functional as F


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

        #encoder = BasicPositionalEncoding(in_features, 10)

        self.net = Sequential(
            #encoder,
            #Linear(encoder.out_features, hidden),
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


class MLP(Module):
    def __init__(self, in_features, out_features, hidden: int = 64, layers: int = 1):
        super(MLP, self).__init__()

        nodes = []

        for i in range(layers):
            nodes += [Linear(in_features if i == 0 else hidden, hidden), ReLU()]

        nodes.append(Linear(hidden, out_features))

        self.net = Sequential(*nodes)

    def forward(self, state):
        return self.net(state)
