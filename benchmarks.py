import argparse

import gymnasium
import numpy as np
import torch
from numpy import ndarray
from gymnasium.wrappers import TransformObservation
from stable_baselines3.common.vec_env import DummyVecEnv

from dqn import DQNOptions, DQN
from environments import TrainingEnvironment
from models import Policy, MLP, SimpleDQN
from reinforce import Reinforce

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='benchmark')

    parser.add_argument('env', type=str, choices=['CartPole-v1', 'LunarLander-v2'])
    parser.add_argument('model', type=str, choices=['reinforce', 'dqn'])
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--hidden', type=int, default=16)
    args = parser.parse_args()

    def make_env(n: int):
        max_v = np.array([1.5, 1.5, 5., 5., 3.14, 5., 1., 1.])

        def normalize_obs(obs: ndarray) -> ndarray:
            return obs / max_v

        def fn():
            if args.env == 'LunarLander-v2':
                return TransformObservation(gymnasium.make(args.env), normalize_obs)

            return gymnasium.make(args.env)

        return DummyVecEnv([fn for _ in range(n)])

    for i in range(3):
        env = TrainingEnvironment('%s-%s-%d' % (args.env, args.model, args.hidden), make_env(1))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        obs_n = env.observation_space.shape[0]
        act_n = env.action_space.n

        if args.model == 'reinforce':
            policy = Policy(MLP(obs_n, act_n, hidden=args.hidden, layers=2))
            r = Reinforce(policy, gamma=args.gamma, n_training_episodes=4000, n_max_episode_steps=1_000_000, device='cuda')
            r.fit(env)
        if args.model == 'dqn':
            ops = DQNOptions(max_steps=100_000,
                             batch_size=64,
                             gamma=args.gamma,
                             epsilon_decay=10000,
                             epochs=1,
                             # learning_rate=1e-4,
                             replay_buffer_size=500_000
                             )
            dqn = DQN(SimpleDQN(obs_n, act_n, hidden=args.hidden), options=ops, device='cuda')
            dqn.fit(env)
