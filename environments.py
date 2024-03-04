import collections
import os.path
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
from gymnasium.core import ActType
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnv
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

ExperienceBatch = collections.namedtuple('ExperienceBatch', ['states', 'actions', 'rewards', 'next_states', 'done'])


def get_possible_actions(env):
    return [a for a in range(env.action_space.start, env.action_space.start + env.action_space.n)]


class TrainingEnvironment(VecEnvWrapper):

    def __init__(self, env_id, env: VecEnv, log_steps: int = 100, log_dir: str = 'runs'):
        super().__init__(VecMonitor(env))
        self.episode_infos = collections.deque(maxlen=100)
        self.episode_count = 0
        self.env_id = env_id
        self.log_steps = log_steps
        self.step_count = 0
        now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.writer = SummaryWriter(os.path.join(log_dir, '%s-%s' % (env_id, now)))
        self.net = None

    @property
    def total_envs(self) -> int:
        return self.venv.num_envs

    @property
    def mean_episode_reward(self):
        return np.mean([e['r'] for e in self.episode_infos]) if self.episode_infos else None

    @property
    def mean_episode_length(self):
        return np.mean([e['l'] for e in self.episode_infos]) if self.episode_infos else None

    def step(self, action):
        next_state, reward, terminated, infos = self.venv.step(action)

        for info, t in zip(infos, terminated):
            if t:
                self.episode_infos.append(info['episode'])
                self.episode_count += 1

        self.step_count += self.total_envs

        if self.step_count % self.log_steps == 0 and self.step_count > 0:
            if self.net is not None:
                grad_max = 0.0
                grad_means = 0.0
                grad_count = 0
                for p in self.net.parameters():
                    if p.grad is not None:
                        grad_max = max(grad_max, p.grad.abs().max().item())
                        grad_means += (p.grad ** 2).mean().sqrt().item()
                        grad_count += 1

                if grad_count > 0:
                    self.writer.add_scalar("Gradients/L2", grad_means / grad_count, self.step_count)
                    self.writer.add_scalar("Gradients/Max", grad_max, self.step_count)

            if self.mean_episode_reward:
                self.writer.add_scalar("Agent/Reward Mean", self.mean_episode_reward, self.step_count)
                self.writer.add_scalar("Agent/Length Mean", self.mean_episode_length, self.step_count)
            self.writer.flush()

        return next_state, reward, terminated, infos

    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.step_count)

    def track_gradients(self, net: Module):
        self.net = net

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()


class ExperienceRecorder(VecEnvWrapper):

    def __init__(self, env: TrainingEnvironment, buffer_size: int):
        super().__init__(env)
        self.experience = collections.deque(maxlen=buffer_size)
        self.state = None

    def step(self, actions: List[ActType]):
        next_state, reward, terminated, infos = self.venv.step(actions)

        for (s, a, r, t, ns, info) in zip(self.state, actions, reward, terminated, next_state, infos):
            self.experience.append((s, a, r, t, ns))

        self.state = next_state

        return next_state, reward, terminated, infos

    def reset(self) -> VecEnvObs:
        self.state = self.venv.reset()
        return self.state

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def sample(self, device: str, batch_size: int) -> [ExperienceBatch, None]:
        if len(self.experience) < batch_size:
            return None

        items = random.sample(self.experience, batch_size)
        states, actions, rewards, done, next_states = zip(*items)
        return ExperienceBatch(
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(actions), dtype=torch.int64, device=device).unsqueeze(-1),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(done), dtype=torch.float32, device=device).unsqueeze(-1)
        )


class QuantizationObservationTransformer(TransformObservation):
    def __init__(self, env, bins: int):
        super(QuantizationObservationTransformer, self).__init__(env, lambda o: self.quantize(o))

        if isinstance(env.observation_space, Box):
            box_space = env.observation_space
        else:
            raise TypeError("The observation space is not of type Box.")

        self.bins = [
            np.linspace(low, high, num=bins + 1)[:-1]
            for low, high in zip(box_space.low, box_space.high)
        ]

        num_dimensions = env.observation_space.shape[0]
        self.observation_space = Discrete(bins ** num_dimensions)
        self.num_bins_per_dimension = bins

    def quantize(self, obs) -> int:
        discrete_obs = sum(
            np.digitize(obs[i], self.bins[i]) * (self.num_bins_per_dimension ** i)
            for i in range(len(obs))
        )
        return discrete_obs

    @staticmethod
    def quantize_value(value: float, quants: int, box_space: Box, index: int) -> int:
        s = (value - box_space.low[index]) / (box_space.high[index] - box_space.low[index])
        return round(quants * s)
