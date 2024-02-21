import collections
import random
from datetime import datetime
from typing import List

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import TransformObservation
from numpy import ndarray
from timeit import default_timer as timer

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


ExperienceBatch = collections.namedtuple('ExperienceBatch', ['states', 'actions', 'rewards', 'next_states', 'done'])


class DQN(Module):
    def __init__(self, net: Module, target_net: Module, options: DQNOptions = DQNOptions(), device: str = 'cpu'):
        super(DQN, self).__init__()

        self.net = net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(net.state_dict())
        self.options = options
        self.device = device
        self.experience = collections.deque(maxlen=options.replay_buffer_size)

    def fit(self, env):
        decaying_epsilon = LinearDecayEpsilon(self.options.epsilon_min, self.options.epsilon_max,
                                              int(self.options.epsilon_decay / len(env.unwrapped.envs)))
        loss = MSELoss()
        optimizer = Adam(self.net.parameters(), lr=self.options.learning_rate)
        episode_infos = collections.deque(maxlen=100)
        episode_count = 0
        progress = tqdm(range(self.options.max_steps))

        state = env.reset()
        episode_reward = 0
        best_avg_reward = -1000000

        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        writer = SummaryWriter('runs/%s-%s' % (env.unwrapped.envs[0].spec.id, now))
        start_time = timer()
        act_time = 0
        upd_time = 0
        smp_time = 0

        for step in progress:
            s1 = timer()

            # Select an action
            epsilon = next(decaying_epsilon)
            actions = self.epsilon_greedy_action(env, state, epsilon)

            # Execute selected action
            next_state, reward, terminated, infos = env.step(actions)
            for (s, a, r, t, ns, info) in zip(state, actions, reward, terminated, next_state, infos):
                self.experience.append((s, a, r, t, ns))

                if t:
                    episode_infos.append(info['episode'])
                    episode_count += 1
            state = next_state
            episode_reward += sum(reward) / len(reward)
            act_time += timer() - s1

            #if terminated: #or info['TimeLimit.truncated']:
            #    #state, _ = env.reset()
           #     episode_rewards.append(episode_reward)
           #     episode_reward = 0
           #     continue

            if len(self.experience) < self.options.batch_size:
                continue

            for i in range(self.options.epochs):
                # Sample next batch of experience
                s2 = timer()
                batch = self.sample_experience()
                smp_time += timer() - s2

                s3 = timer()
                target_q_values = self.target_net(batch.next_states)
                max_target_q_values = target_q_values.max(-1, keepdim=True)[0]
                targets = batch.rewards + self.options.gamma * (1.0 - batch.done) * max_target_q_values

                # Compute loss
                q_values = self.net(batch.states)
                actions = torch.gather(q_values, dim=1, index=batch.actions)
                loss_value = loss(actions, targets)

                # Update
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                upd_time += timer() - s3

            # Update target network
            if step % 1000 == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            if len(episode_infos):
                rew_mean = np.mean([e['r'] for e in episode_infos]) or 0
                len_mean = np.mean([e['l'] for e in episode_infos]) or 0

                time_elapsed = timer() - start_time
                at = 100.0 * act_time / time_elapsed
                ut = 100.0 * upd_time / time_elapsed
                st = 100.0 * smp_time / time_elapsed
                #avg_reward = sum(episode_infos) / float(len(episode_infos))
                progress.set_description('avg. reward=%2.4f, act=%2.2f%%, upd=%2.2f%%, smp=%2.2f%%' % (rew_mean, at, ut, st))

                if step % 100 == 0:
                    writer.add_scalar("rew_mean", rew_mean, step)
                    writer.add_scalar("len_mean", len_mean, step)
                    writer.flush()

                if rew_mean > best_avg_reward:
                    torch.save(self.net, 'checkpoint.pt')
                    best_avg_reward = rew_mean

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

    def sample_experience(self) -> ExperienceBatch:
        items = random.sample(self.experience, self.options.batch_size)
        states, actions, rewards, done, next_states = zip(*items)
        return ExperienceBatch(
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=self.device).unsqueeze(-1),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device),
            torch.as_tensor(np.asarray(done), dtype=torch.float32, device=self.device).unsqueeze(-1)
        )

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
                return obs / max_v
            #    return obs

            result = gymnasium.make(name)
            return Monitor(TransformObservation(result, normalize_obs), allow_early_resets=True)

        return fn


    #test_env = gymnasium.make('LunarLander-v2')
    #test_env = TransformObservation(test_env, normalize_obs)
    test_env = DummyVecEnv([make_env('LunarLander-v2') for _ in range(8)])
    #test_env = make_env('LunarLander-v2')()
    obs_n = test_env.observation_space.shape[0]
    act_n = test_env.action_space.n
    ops = DQNOptions(max_steps=1_000_000,
                     batch_size=64,
                     gamma=0.99,
                     epsilon_decay=10000,
                     epochs=1,
                     #learning_rate=1e-4,
                     #replay_buffer_size=200_000
                     )
    dqn = DQN(SimpleDQN(obs_n, act_n), SimpleDQN(obs_n, act_n), options=ops, device='cuda')
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
