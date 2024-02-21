import collections
import random
from datetime import datetime

import gymnasium
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Training parameters
n_training_episodes = 10000  # Total training episodes
n_replay_buffer_size = 1_000_000
n_batch_size = 32

# Environment parameters
max_steps = 10000

# Exploration parameters
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.05  # Minimum exploration probability
decay_rate = 0.0005  # Exponential decay rate for exploration prob


#gamma = 0.99


class Agent:
    def __init__(self, net, device: str = 'cpu'):
        self.net = net.to(device)
        self.device = device
        self.experience = ReplayBuffer(n_replay_buffer_size, device)

    @torch.no_grad()
    def step(self, env, state, epsilon: float):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            scores = self.net(torch.from_numpy(state).to(self.device))
            action = torch.argmax(scores).item()

        next_state, reward, terminated, truncated, info = env.step(action)
        self.experience.append(state, action, reward, next_state, terminated)

        return next_state, reward, terminated or truncated


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.items = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device

    def append(self, state, action, reward, next_state, terminal):
        self.items.append((
            torch.from_numpy(state).float(),
            torch.tensor(action).unsqueeze(-1),
            torch.tensor(reward).float(),
            torch.from_numpy(next_state).float(),
            terminal
        ))

    def sample(self, batch_size: int):
        if len(self.items) < batch_size:
            return None

        items = random.sample(self.items, batch_size)

        states, actions, rewards, next_states, terminals = zip(*items)

        return (torch.stack(states).to(self.device),
                torch.stack(actions).to(self.device),
                torch.stack(rewards).to(self.device),
                torch.stack(next_states).to(self.device),
                torch.BoolTensor(terminals).to(self.device))


class Brain(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden: int = 256):
        super(Brain, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, s: Tensor) -> Tensor:
        return self.net(s)


class QLoss(nn.Module):
    def __init__(self, net, tgt_net, gamma):
        super(QLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.net = net
        self.tgt_net = tgt_net
        self.gamma = gamma

    def forward(self, states, actions, rewards, next_states, terminal):
        # q_target = self.tgt_net(next_states).detach().max(axis=1)[0].unsqueeze(1)
        # #y_j = rewards + self.gamma * q_target #* (1 - dones)          # target, if terminal then y_j = rewards
        # y_j = self.gamma * q_target
        # y_j[terminal] = 0.0
        # y_j = rewards.unsqueeze(-1) + y_j
        # q_eval = self.net(states).gather(1, actions)
        #
        # return self.mse(q_eval, y_j)

        state_action_values = self.net(states).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.tgt_net(next_states).max(1)[0]
            next_state_values[terminal] = 0.0
            next_state_values = next_state_values
            expected_state_action_values = next_state_values * self.gamma + rewards
            expected_state_action_values = expected_state_action_values.detach()

        return self.mse(expected_state_action_values, state_action_values)


def train(env):
    net = Brain(env.observation_space.shape[0], env.action_space.n, hidden=256)
    target_net = Brain(env.observation_space.shape[0], env.action_space.n, hidden=256).cuda()
    target_net.load_state_dict(net.state_dict())
    agent = Agent(net, device='cuda')
    loss = QLoss(net, target_net, 0.99)

    optimizer = Adam(net.parameters(), lr=1e-4)
    progress = tqdm(range(n_training_episodes), unit="ep")
    frame_idx = 0

    now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    writer = SummaryWriter('runs/dqn-' + now)

    for episode in progress:
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reset the environment
        state, info = env.reset()
        loss_total = 0
        reward_total = 0
        steps_total = 0

        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            next_state, reward, terminated = agent.step(env, state, epsilon)
            reward_total += reward

            if terminated:
                break

            batch = agent.experience.sample(n_batch_size)

            if batch is None:
                continue

            optimizer.zero_grad()
            loss_value = loss(*batch)
            loss_value.backward()
            optimizer.step()

            frame_idx += 1
            if frame_idx % 1000 == 0:
                target_net.load_state_dict(net.state_dict())

            steps_total += 1
            loss_total += loss_value

        loss_avg = loss_total / steps_total
        progress.set_description('Epoch #%d: reward=%2.2f, loss=%2.2f, eps=%2.2f' % (episode, reward_total, loss_avg, epsilon))

        if frame_idx % 10 == 0:
            writer.add_scalar("reward", reward_total, frame_idx)
            writer.add_scalar("loss", loss_avg, frame_idx)
            writer.flush()


if __name__ == '__main__':
    agent_env = gymnasium.make("LunarLander-v2", render_mode="rgb_array")
    train(agent_env)
