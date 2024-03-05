import argparse

import numpy as np
from torch.nn import Tanh

import actorcritic
import models
from actorcritic import ActorCritic
from dqn import DQNOptions, DQN
from environments import TrainingEnvironment, make_env
from models import Policy, SimpleDQN, AC, MLP
from reinforce import Reinforce


def benchmark(args):
    env = TrainingEnvironment('%s-%s-%d' % (args.env, args.model, args.hidden), make_env(args.env, args.envs), log_dir='runs')

    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n

    if args.model == 'reinforce':
        policy = Policy(MLP(obs_n, act_n, hidden=[args.hidden], activation=Tanh))
        models.initialize_ortho({policy.net: np.sqrt(2)})
        r = Reinforce(policy,
                      gamma=args.gamma,
                      n_training_episodes=15000,
                      learning_rate=7e-4,
                      n_max_episode_steps=1_000_000,
                      device='cuda')
        r.fit(env)
    if args.model == 'ac':
        policy = AC(obs_n, act_n, hidden=[args.hidden, args.hidden])
        ac = ActorCritic(policy,
                         n_training_steps=8_000_000,
                         device='cuda',
                         normalize_advantage=False,
                         n_steps=100,
                         ent_coef=1e-05,
                         clip=0.5,
                         learning_rate=7e-4,
                         gamma=args.gamma)
        ac.fit(env)
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


def main():
    parser = argparse.ArgumentParser(prog='benchmark')

    subparsers = parser.add_subparsers(required=True)
    actorcritic.command_line(subparsers)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
