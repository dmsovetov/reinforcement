import collections
import os.path
import random
from datetime import datetime
from typing import List

import gymnasium
import numpy as np
import torch
from gymnasium.core import ActType
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecMonitor
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnv
from torch.utils.tensorboard import SummaryWriter


ExperienceBatch = collections.namedtuple('ExperienceBatch', ['states', 'actions', 'rewards', 'next_states', 'done'])


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

        self.step_count += 1

        if self.step_count % self.log_steps == 0:
            self.writer.add_scalar("rew_mean", self.mean_episode_reward, self.step_count)
            self.writer.add_scalar("len_mean", self.mean_episode_length, self.step_count)
            self.writer.flush()

        return next_state, reward, terminated, infos

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


def make_env(name):
    def fn():
        result = gymnasium.make(name)
        #return Monitor(result, allow_early_resets=True)
        return result

    return fn


def test():
    test_env = DummyVecEnv([make_env('CartPole-v1') for _ in range(2)])
    train_env = TrainingEnvironment('test', test_env)
    train_env.reset()

    while True:
        next_state, reward, terminated, infos = train_env.step([0, 0])


if __name__ == '__main__':
    test()
