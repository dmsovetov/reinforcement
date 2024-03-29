import copy
import random
from typing import List

import numpy as np
import torch
from numpy import ndarray
from torch.nn import Module, HuberLoss
from torch.optim import Adam
from tqdm import tqdm

from environments import TrainingEnvironment, ExperienceRecorder
from policies import LinearDecayEpsilon


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
        env.track_gradients(self.net)
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
