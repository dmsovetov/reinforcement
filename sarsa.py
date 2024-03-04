import time
from collections import deque

import gymnasium
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

from environments import get_possible_actions, QuantizationObservationTransformer
from playground import WindyGridworldEnv, MountainCarQuantization
from policies import GreedyPolicy, evaluate_policy, LinearDecayEpsilon, EpsilonGreedyPolicy, render_policy
from valuefunctions import DiscreteActionValueFunction


def sarsa_max(env, gamma: float = 1.0, max_steps: int = 1000, min_eps: float = 0.05, alpha: float = 0.01):
    possible_actions = get_possible_actions(env)
    action_values = DiscreteActionValueFunction()
    progress = tqdm(range(max_steps))
    greedy_policy = GreedyPolicy(env, action_values, possible_actions)
    policy = EpsilonGreedyPolicy(env, action_values, possible_actions)
    epsilon = LinearDecayEpsilon(min_value=min_eps, decay=max_steps)

    reward_sum = 0.0
    last_rewards = deque(maxlen=100)

    state, _ = env.reset()

    for i in progress:
        action = policy.act(state)
        next_state, reward, done, terminated, _ = env.step(action)
        reward_sum += reward

        if done or terminated:
            target = reward
            next_state, _ = env.reset()
            last_rewards.append(reward_sum)
            reward_sum = 0.0
        else:
            next_action = greedy_policy.act(next_state)
            target = reward + gamma * action_values.get(next_state, next_action)

        action_values.update(state, action, target, alpha)

        state = next_state
        policy.epsilon = next(epsilon)

        if i % 1000 == 0 and len(last_rewards):
            avg_reward = sum(last_rewards) / len(last_rewards)
            progress.set_description('reward=%2.2f' % avg_reward)

    return GreedyPolicy(env, action_values, possible_actions)


def sarsa(env, gamma: float = 1.0, max_steps: int = 1000, min_eps: float = 0.05, alpha: float = 0.01):
    possible_actions = get_possible_actions(env)
    action_values = DiscreteActionValueFunction()
    progress = tqdm(range(max_steps))
    policy = EpsilonGreedyPolicy(env, action_values, possible_actions)
    epsilon = LinearDecayEpsilon(min_value=min_eps, decay=max_steps)

    state, _ = env.reset()
    action = policy.act(state)
    reward_sum = 0.0
    last_rewards = deque(maxlen=100)

    for i in progress:
        next_state, reward, done, terminated, _ = env.step(action)
        reward_sum += reward

        if done or terminated:
            target = reward
            next_state, _ = env.reset()
            next_action = policy.act(state)
            last_rewards.append(reward_sum)
            reward_sum = 0.0
        else:
            next_action = policy.act(next_state)
            target = reward + gamma * action_values.get(next_state, next_action)

        action_values.update(state, action, target, alpha)

        state = next_state
        action = next_action
        policy.epsilon = next(epsilon)

        if i % 1000 == 0 and len(last_rewards):
            avg_reward = sum(last_rewards) / len(last_rewards)
            progress.set_description('reward=%2.2f' % avg_reward)

    return GreedyPolicy(env, action_values, possible_actions)


def test_sarsa_windy_gridworld():
    env = TimeLimit(WindyGridworldEnv(), max_episode_steps=500)
    policy = sarsa(env, max_steps=100_000, alpha=0.05)
    time.sleep(0.1)
    print('[Sarsa] WindyGridworld: %2.2f [Expected -15.00]' % evaluate_policy(env, policy))


def test_sarsa_taxi_v3():
    env = gymnasium.make('Taxi-v3')
    policy = sarsa(env, max_steps=2_000_000, alpha=0.05)
    time.sleep(0.1)
    print('[Sarsa] Taxi-v3: %2.2f [Expected ~8]' % evaluate_policy(env, policy))


def test_sarsa_max_windy_gridworld():
    env = TimeLimit(WindyGridworldEnv(), max_episode_steps=500)
    policy = sarsa_max(env, max_steps=100_000, alpha=0.05)
    time.sleep(0.1)
    print('[SarsaMax] WindyGridworld: %2.2f [Expected -15.00]' % evaluate_policy(env, policy))


def test_sarsa_max_taxi_v3():
    env = gymnasium.make('Taxi-v3')
    policy = sarsa_max(env, max_steps=2_000_000, alpha=0.05)
    time.sleep(0.1)
    print('[SarsaMax] Taxi-v3: %2.2f [Expected ~8]' % evaluate_policy(env, policy))


def test_sarsa_mountain_car():
    env = MountainCarQuantization(gymnasium.make('MountainCar-v0'))
    policy = sarsa(env, max_steps=10_000_000, alpha=0.05)
    time.sleep(0.1)
    render_policy(MountainCarQuantization(gymnasium.make('MountainCar-v0', render_mode='human')), policy)
    print('[Sarsa] MountainCar-v0: %2.2f' % evaluate_policy(env, policy))


def test_sarsa_lunar_lander():
    env = QuantizationObservationTransformer(gymnasium.make('LunarLander-v2'), bins=50)
    policy = sarsa(env, max_steps=10_000_000, alpha=0.05)
    time.sleep(0.1)
    render_policy(QuantizationObservationTransformer(gymnasium.make('LunarLander-v2', render_mode='human'), bins=50), policy)
    print('[Sarsa] LunarLander-v2: %2.2f' % evaluate_policy(env, policy))


if __name__ == '__main__':
    # test_sarsa_max_windy_gridworld()
    # test_sarsa_windy_gridworld()
    # test_sarsa_max_taxi_v3()
    # test_sarsa_taxi_v3()
    # test_sarsa_mountain_car()
    # test_sarsa_lunar_lander()
    pass
