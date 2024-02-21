import gymnasium
import numpy as np
from tqdm import tqdm

# Training parameters
n_training_episodes = 1000000  # Total training episodes
learning_rate = 0.7           # Learning rate

# Evaluation parameters
n_eval_episodes = 100        # Total number of test episodes

# Environment parameters
env_id = "FrozenLake-v1"     # Name of the environment
max_steps = 99               # Max steps per episode
gamma = 0.95                 # Discounting rate
eval_seed = []               # The evaluation seed of the environment

# Exploration parameters
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.05            # Minimum exploration probability
decay_rate = 0.0005            # Exponential decay rate for exploration prob


def greedy_policy(q, s):
    return np.argmax(q[s])


def evaluate(env, q, seed):
    episode_rewards = []

    for episode in tqdm(range(n_eval_episodes)):
        if seed:
            state, info = env.reset(seed=seed[episode])
        else:
            state, info = env.reset()

        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = greedy_policy(q, state)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def train(env):
    env.reset()

    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def epsilon_greedy_policy(s, eps):
        if np.random.random() < eps:
            return greedy_policy(q_table, s)

        return env.action_space.sample()

    for episode in tqdm(range(n_training_episodes)):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

        # Reset the environment
        state, info = env.reset()

        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, terminated, truncated, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            current_value = q_table[state][action]
            target_value = reward + gamma * np.max(q_table[new_state])
            q_table[state][action] = current_value + learning_rate * (target_value - current_value)

            # If terminated or truncated finish the episode
            if terminated or truncated:
                break

            # Our next state is the new state
            state = new_state

    return q_table


if __name__ == '__main__':
    #agent_env = gymnasium.make(env_id, map_name="4x4", is_slippery=False, render_mode="rgb_array")
    agent_env = gymnasium.make("Taxi-v3", render_mode="rgb_array")
    result = train(agent_env)
    mean_reward, std_reward = evaluate(agent_env, result, eval_seed)
    print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
