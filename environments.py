import collections
import os.path
import random
import time
import uuid
from typing import List

import gymnasium
import numpy as np
import torch
from gymnasium.core import ActType
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import TransformObservation, RecordVideo
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn, VecEnv
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

ExperienceBatch = collections.namedtuple('ExperienceBatch', ['states', 'actions', 'rewards', 'next_states', 'done'])


def get_possible_actions(env):
    return [a for a in range(env.action_space.start, env.action_space.start + env.action_space.n)]


def get_state_n(env):
    return env.observation_space.shape[0]


def get_action_n(env):
    return env.action_space.n


def get_env_dims(env):
    return get_state_n(env), get_action_n(env)


def get_observation_range(env):
    if isinstance(env.observation_space, Box):
        box_space = env.observation_space
    else:
        raise TypeError("The observation space is not of type Box.")

    return [(low, high) for low, high in zip(box_space.low, box_space.high)]


def get_envs() -> List[str]:
    return [
        'CartPole-v1',
        'LunarLander-v2',
        'MountainCar-v0',
        'Acrobot-v1'
    ]


def make_env(env_id, n: int):
    def fn():
        return gymnasium.make(env_id)

    return SubprocVecEnv([fn for _ in range(n)])


def make_training_env(env_id: str, name: str, n: int):
    return TrainingEnvironment(env_id, '%s-%s' % (env_id, name), make_env(env_id, n), log_dir='runs')


class TrainingEnvironmentInfo:
    def __init__(self, index: int, truncated_state):
        self.index = index
        self.truncated_state = truncated_state


class VideoRecordingCallback:
    def __init__(self, env_id: str, policy, directory: str = 'video', prefix: str = 'episode'):
        env = gymnasium.make(env_id, render_mode="rgb_array")
        self.env = RecordVideo(env, directory, name_prefix=prefix, step_trigger=lambda s: True, disable_logger=True)
        self.policy = policy

    def __call__(self, episode: int):
        state, _ = self.env.reset()
        self.env.start_video_recorder()

        for i in range(1000):
            action = self.policy(self.env, state)
            next_state, reward, done, _, _ = self.env.step(action)
            state = next_state
            self.env.render()
            if done:
                break

        self.env.close()
        self.env.close_video_recorder()


EpisodeCallback = collections.namedtuple('EpisodeCallback', ['interval', 'fn', 'last_episode'])


class TrainingEnvironment(VecEnvWrapper):

    def __init__(self, env_id: str, env_name: str, env: VecEnv, log_steps: int = 100, log_dir: str = 'runs'):
        super().__init__(VecMonitor(env))
        self.episode_infos = collections.deque(maxlen=100)
        self.episode_count = 0
        self.env_name = env_name
        self.env_id = env_id
        self.log_steps = log_steps
        self.run_id = str(uuid.uuid4()).split('-')[0]
        self.step_count = 0
        now = round(time.time())
        self.writer = SummaryWriter(os.path.join(log_dir, '%s-%d-%s' % (env_name, now, self.run_id)))
        self.net = None
        self.episode_callbacks = []

    @property
    def total_envs(self) -> int:
        return self.venv.num_envs

    @property
    def total_episodes(self):
        return self.episode_count

    @property
    def mean_episode_reward(self):
        return np.mean([e['r'] for e in self.episode_infos]) if self.episode_infos else None

    @property
    def mean_episode_length(self):
        return np.mean([e['l'] for e in self.episode_infos]) if self.episode_infos else None

    def add_episode_callback(self, fn, interval: int = 1):
        self.episode_callbacks.append(EpisodeCallback(interval, fn, last_episode=0))

    def add_video_recording(self, policy, interval: int = 100):
        directory = 'video/%s-%s' % (self.env_name, self.run_id)
        self.add_episode_callback(VideoRecordingCallback(self.env_id, policy, directory=directory), interval)

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

                self.writer.add_scalar("rollout/ep_rew_mean", self.mean_episode_reward, self.step_count)
                self.writer.add_scalar("rollout/ep_len_mean", self.mean_episode_length, self.step_count)
            self.writer.flush()

        transformed_infos = []

        for i, info in enumerate(infos):
            truncated_state = None

            if info["TimeLimit.truncated"]:
                truncated_state = info["terminal_observation"]

            transformed_infos.append(TrainingEnvironmentInfo(i, truncated_state))

        for i, cb in enumerate(self.episode_callbacks):
            episodes_since_last = self.episode_count - cb.last_episode

            if episodes_since_last >= cb.interval:
                cb.fn(self.episode_count)
                self.episode_callbacks[i] = EpisodeCallback(cb.interval, cb.fn, self.episode_count)

        return next_state, reward, terminated, transformed_infos

    def add_scalar(self, name, value):
        self.writer.add_scalar(name, value, self.step_count)

    def track_gradients(self, net: Module):
        self.net = net

    def add_hyperparameters(self, args):
        text = "\n".join([f"|{key}|{value}|" for key, value in vars(args).items() if key != 'func'])
        self.writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % text)

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()

    def reset(self) -> VecEnvObs:
        return self.venv.reset()


class DeviceEnv(VecEnvWrapper):
    def __init__(self, env, device, inverse_done=False):
        super(DeviceEnv, self).__init__(env)
        self.device = device
        self.inverse_done = inverse_done

    def step(self, actions: List[ActType]):
        return self.apply_transform(*self.venv.step(actions))

    def step_wait(self):
        return self.apply_transform(*self.venv.step_wait())

    def reset(self):
        state = self.venv.reset()
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        return state

    def apply_transform(self, state, reward, terminated, infos):
        state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
        reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)

        if self.inverse_done:
            terminated = torch.as_tensor(1 - terminated, dtype=torch.float32, device=self.device)
        else:
            terminated = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)

        for info in infos:
            if info.truncated_state is not None:
                info.truncated_state = torch.as_tensor(info.truncated_state, dtype=torch.float32, device=self.device)

        return state, reward, terminated, infos


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
