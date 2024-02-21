import gymnasium
import numpy as np
import torch
from gym.wrappers import TransformObservation

from dqn import DQNOptions, DQN

#env_id = 'CartPole-v1'
env_id = 'LunarLander-v2'
max_v = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1.])


def normalize_obs(obs: np.ndarray) -> np.ndarray:
    if env_id == 'LunarLander-v2':
        return obs / max_v
    return obs


test_env = gymnasium.make(env_id, render_mode="human")
test_env = TransformObservation(test_env, normalize_obs)

net = torch.load('checkpoint.pt')
ops = DQNOptions(max_steps=1_000_000, batch_size=32, gamma=0.99)
dqn = DQN(net, net, options=ops, device='cuda')
dqn.evaluate(test_env)
