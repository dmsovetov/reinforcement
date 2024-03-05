import copy

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Adam, RMSprop
from tqdm import tqdm

from environments import TrainingEnvironment


class Reinforce:
    def __init__(
            self,
            net: Module,
            n_training_episodes: int = 1000,
            n_max_episode_steps: int = 1000,
            gamma: float = 1.0,
            learning_rate: float = 1e-3,
            clip: float = None,
            device: str = 'cpu'
    ):
        self.net = net.to(device)
        self.gamma = gamma
        self.n_training_episodes = n_training_episodes
        self.n_max_episode_steps = n_max_episode_steps
        self.learning_rate = learning_rate
        self.device = device
        self.clip = clip

    def fit(self, env: TrainingEnvironment):
        progress = tqdm(range(self.n_training_episodes))
        #optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        optimizer = RMSprop(self.net.parameters(), lr=self.learning_rate, eps=1e-5, alpha=0.99)
        env.track_gradients(self.net)

        best_avg_reward = -1000000

        for _ in progress:
            # Sample episode
            log_probs, rewards, probs, states, episode_length = self.sample_episode(env)
            returns = self.calculate_cumulative_returns(env.total_envs, rewards, episode_length)

            # Calculate loss
            loss = []
            for log_prob, g in zip(log_probs, returns):
                loss.append(-log_prob * g)
            loss = torch.cat(loss).sum() / env.total_envs

            # Calculate entropy loss
            #probs = torch.cat(probs)
            #entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
            #loss = loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip)
            optimizer.step()

            # Calculate KL
            #new_logits_v = self.net.logits(torch.as_tensor(np.array(states).squeeze(1), device=self.device))
            #new_prob_v = F.softmax(new_logits_v, dim=1)
            #kl_div_v = -((new_prob_v / probs).log() * probs).sum(dim=1).mean()
            #env.add_scalar("Policy/KL", kl_div_v.item())
            #env.add_scalar('Policy/Entropy', entropy.item())

            progress.set_description('reward=%2.2f' % env.mean_episode_reward)

            # Save model
            if env.mean_episode_reward > best_avg_reward:
                torch.save(copy.deepcopy(self.net), 'reinforce.pt')
                best_avg_reward = env.mean_episode_reward

    def calculate_cumulative_returns(self, n, rewards, episode_length):
        eps = np.finfo(np.float32).eps.item()
        returns = np.zeros((max(episode_length), n))
        mean_return = 0.0

        for i in range(n):
            cumulative_reward = 0

            for t in reversed(range(episode_length[i])):
                cumulative_reward = cumulative_reward * self.gamma + rewards[t][i]
                returns[t, i] = cumulative_reward

            mean_return += returns[:episode_length[i], i].sum() / sum(episode_length)

        std_return = 0.0

        for i in range(n):
            std_return += ((returns[:episode_length[i], i] - mean_return) ** 2).sum() / sum(episode_length)

        std_return = np.sqrt(std_return)

        for i in range(n):
            returns[:episode_length[i], i] -= mean_return
            returns[:episode_length[i], i] /= std_return + eps

        return torch.as_tensor(returns, device=self.device)

        # returns = deque(maxlen=len(rewards))
        # n_steps = len(rewards)
        #
        # for t in reversed(range(n_steps)):
        #     r = (returns[0] if len(returns) > 0 else 0)
        #     returns.appendleft(self.gamma * r + rewards[t])
        #
        # returns = torch.as_tensor(returns, device=self.device)
        # returns = (returns - returns.mean()) / (returns.std() + eps)

    def sample_episode(self, env):
        state = env.reset()
        log_probs = []
        rewards = []
        probs = []
        s = []

        episode_length = [-1] * env.total_envs
        environments_left = env.total_envs

        for t in range(self.n_max_episode_steps):
            action, log_prob, p = self.net(torch.as_tensor(state, device=self.device, dtype=torch.float32))
            state, reward, done, infos = env.step(action)

            for i, info in enumerate(infos):
                if 'episode' in info and episode_length[i] == -1:
                    episode_length[i] = info['episode']['l']
                    environments_left -= 1

            rewards.append(reward)
            log_probs.append(log_prob)
            probs.append(p)
            s.append(state)

            if environments_left == 0:
                break

        return log_probs, rewards, probs, s, episode_length


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """
    Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state, _ = env.reset()
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, truncated, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward
