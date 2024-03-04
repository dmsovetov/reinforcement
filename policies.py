import random
from typing import List

import numpy as np

from valuefunctions import ActionValueFunction


class LinearDecayEpsilon:
    def __init__(self, min_value: float = 0.05, max_value: float = 1.0, decay: int = 10000):
        self.min_value = min_value
        self.max_value = max_value
        self.decay = decay
        self.index = 0

    def __next__(self) -> float:
        # result = self.min_value + (self.max_value - self.min_value) * math.exp(-self.decay * self.index)
        result = np.interp(self.index, [0, self.decay], [self.max_value, self.min_value])
        self.index += 1
        return float(result)


class Policy:
    def act(self, obs):
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return self.env.action_space.sample()


class GreedyPolicy(Policy):
    def __init__(self, env, action_values: ActionValueFunction, possible_actions: List):
        self.env = env
        self.possible_actions = possible_actions
        self.action_values = action_values

    def act(self, state):
        return self.action_values.greedy_action(state, self.possible_actions)


class EpsilonGreedyPolicy(GreedyPolicy):
    def __init__(self, env, action_values: ActionValueFunction, possible_actions: List, eps: float = 1.0):
        super(EpsilonGreedyPolicy, self).__init__(env, action_values, possible_actions)
        self.eps = eps

    @property
    def epsilon(self):
        return self.eps

    @epsilon.setter
    def epsilon(self, value):
        self.eps = value

    def act(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        return super().act(state)


def evaluate_policy(env, policy: Policy, n: int = 100) -> float:
    rewards = []

    for i in range(n):
        state, _ = env.reset()
        done, terminated = False, False
        episode_reward = 0.0

        while not (done or terminated):
            action = policy.act(state)
            state, reward, done, terminated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    return sum(rewards) / len(rewards)


def render_policy(env, policy: Policy):
    state, _ = env.reset()
    episode_reward = 0

    while True:
        action = policy.act(state)
        state, r, done, truncated, _ = env.step(action)
        env.render()
        episode_reward += r

        if done or truncated:
            print('Reward: %2.2f' % episode_reward)
            episode_reward = 0
            state, _ = env.reset()
