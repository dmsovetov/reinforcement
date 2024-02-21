import gymnasium
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

env = DummyVecEnv([lambda: gymnasium.make('LunarLander-v2', render_mode='human') for _ in range(1)])
state = env.reset()
episode_reward = 0

net = torch.load('reinforce.pt').cpu()

while True:
    action, _, _ = net(torch.as_tensor(state))
    state, r, done, truncated = env.step([action])
    env.render()
    episode_reward += r

    if done:
        print('Reward: %2.2f' % episode_reward)
        episode_reward = 0
        state = env.reset()
