import copy
import random
from typing import List

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import TransformObservation
from numpy import ndarray
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.nn import Module, HuberLoss
from torch.optim import Adam
from tqdm import tqdm

from environments import TrainingEnvironment, ExperienceRecorder
from models import SimpleDQN


class DQNOptions:
    def __init__(self,
                 max_steps: int = 100_000,
                 batch_size: int = 256,
                 gamma: float = 0.99,
                 epsilon_decay: int = 10000,
                 epochs: int = 1,
                 learning_rate: float = 5e-4,
                 replay_buffer_size: int = 50000):
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epochs = epochs
        self.learning_rate = learning_rate


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


class DQNLoss(Module):
    def __init__(self, net: Module, target_net: Module, gamma: float, loss=None):
        super(DQNLoss, self).__init__()
        self.loss = HuberLoss() if loss is None else loss
        self.net = net
        self.target_net = target_net
        self.gamma = gamma

    def forward(self, states, actions, rewards, done, next_states):
        target_q_values = self.target_net(next_states)
        max_target_q_values = target_q_values.max(-1, keepdim=True)[0]
        targets = rewards + self.gamma * max_target_q_values * (1.0 - done)

        # Compute loss
        q_values = self.net(states)
        actions = torch.gather(q_values, dim=1, index=actions)
        return self.loss(actions, targets)


class DQN:
    def __init__(self, net: Module, options: DQNOptions = DQNOptions(), device: str = 'cpu'):
        self.net = net.to(device)
        self.target_net = copy.deepcopy(net).to(device)
        self.target_net.load_state_dict(net.state_dict())
        self.options = options
        self.device = device

    def fit(self, env: TrainingEnvironment):
        decaying_epsilon = LinearDecayEpsilon(self.options.epsilon_min, self.options.epsilon_max,
                                              int(self.options.epsilon_decay / len(env.unwrapped.envs)))
        loss = DQNLoss(self.net, self.target_net, self.options.gamma)
        optimizer = Adam(self.net.parameters(), lr=self.options.learning_rate)
        progress = tqdm(range(self.options.max_steps))
        env = ExperienceRecorder(env, self.options.replay_buffer_size)

        state = env.reset()
        best_avg_reward = -1000000

        for step in progress:
            # Select an action
            epsilon = next(decaying_epsilon)
            actions = self.epsilon_greedy_action(env, state, epsilon)

            # Execute selected action
            next_state, reward, terminated, infos = env.step(actions)
            state = next_state

            for i in range(self.options.epochs):
                # Sample next batch of experience
                batch = env.sample(self.device, self.options.batch_size)

                if batch is None:
                    break

                # Compute loss
                loss_value = loss(batch.states, batch.actions, batch.rewards, batch.done, batch.next_states)

                # Update
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()

            # Update target network
            if step % 1000 == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            if env.mean_episode_reward:
                progress.set_description('avg. reward=%2.4f' % env.mean_episode_reward)

                if env.mean_episode_reward > best_avg_reward:
                    torch.save(self.net, 'checkpoint.pt')
                    best_avg_reward = env.mean_episode_reward

    @torch.no_grad()
    def epsilon_greedy_action(self, env, state: ndarray, epsilon: float) -> List[int]:
        if random.random() < epsilon:
            return [env.action_space.sample() for _ in range(state.shape[0])]

        return self.greedy_action(state)

    @torch.no_grad()
    def greedy_action(self, state: ndarray) -> List[int]:
        obs = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        q_values = self.net(obs)
        return torch.argmax(q_values, dim=-1).tolist()

    def evaluate(self, env):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            action = self.greedy_action(np.expand_dims(state, 0))
            state, r, done, truncated, _ = env.step(action[0])
            env.render()
            episode_reward += r

            if done or truncated:
                print('Reward: %2.2f' % episode_reward)
                episode_reward = 0
                state, _ = env.reset()


if __name__ == '__main__':
    max_v = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1.])


    def make_env(name):
        def fn():
            def normalize_obs(obs: ndarray) -> ndarray:
                #    return obs / max_v
                return obs

            result = gymnasium.make(name)
            return TransformObservation(result, normalize_obs)

        return fn


    # test_env = gymnasium.make('LunarLander-v2')
    # test_env = TransformObservation(test_env, normalize_obs)
    test_env = DummyVecEnv([make_env('CartPole-v1') for _ in range(2)])
    test_env = TrainingEnvironment('CartPole-v1', test_env)
    # test_env = make_env('LunarLander-v2')()
    obs_n = test_env.observation_space.shape[0]
    act_n = test_env.action_space.n
    ops = DQNOptions(max_steps=1_000_000,
                     batch_size=64,
                     gamma=0.99,
                     epsilon_decay=10000,
                     epochs=1,
                     # learning_rate=1e-4,
                     # replay_buffer_size=200_000
                     )
    dqn = DQN(SimpleDQN(obs_n, act_n), options=ops, device='cuda')
    dqn.fit(test_env)

    # net = torch.load('checkpoint.pt')
    # ops = DQNOptions(max_steps=1_000_000, batch_size=32, gamma=0.99)
    # dqn = DQN(net, net, options=ops, device='cuda')
    # dqn.evaluate(gymnasium.make('CartPole-v1', render_mode="human"))

    # test_env = gymnasium.make('CartPole-v1')
    # obs_n = test_env.observation_space.shape[0]
    # act_n = test_env.action_space.n
    # ops = DQNOptions(max_steps=1_000_000, batch_size=32, gamma=0.99)
    # dqn = DQN(SimpleDQN(obs_n, act_n), SimpleDQN(obs_n, act_n), options=ops, device='cuda')
    # dqn.fit(test_env)

    # net = torch.load('checkpoint.pt')
    # ops = DQNOptions(max_steps=1_000_000, batch_size=32, gamma=0.99)
    # dqn = DQN(net, net, options=ops, device='cuda')
    # dqn.evaluate(gymnasium.make('CartPole-v1', render_mode="human"))
