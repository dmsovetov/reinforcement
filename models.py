import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU


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

        encoder = BasicPositionalEncoding(in_features, 10)

        self.net = Sequential(
            encoder,
            Linear(encoder.out_features, hidden),
            #Linear(in_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, out_features)
        )

    def forward(self, state: Tensor):
        return self.net(state)
