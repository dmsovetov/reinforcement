import copy
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam
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
            device: str = 'cpu'
    ):
        self.net = net.to(device)
        self.gamma = gamma
        self.n_training_episodes = n_training_episodes
        self.n_max_episode_steps = n_max_episode_steps
        self.learning_rate = learning_rate
        self.device = device

    def fit(self, env: TrainingEnvironment):
        progress = tqdm(range(self.n_training_episodes))
        eps = np.finfo(np.float32).eps.item()
        optimizer = Adam(self.net.parameters(), lr=self.learning_rate)
        env.track_gradients(self.net)

        best_avg_reward = -1000000

        for _ in progress:
            # Sample episode
            log_probs, rewards, probs, states = self.sample_episode(env)

            # Calculate cumulative returns
            returns = deque(maxlen=len(rewards))
            n_steps = len(rewards)

            for t in reversed(range(n_steps)):
                r = (returns[0] if len(returns) > 0 else 0)
                returns.appendleft(self.gamma * r + rewards[t])

            returns = torch.as_tensor(np.array(returns), device=self.device)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # Calculate loss
            loss = []
            for log_prob, g in zip(log_probs, returns):
                loss.append(-log_prob * g)
            loss = torch.cat(loss).sum()

            # Calculate entropy loss
            probs = torch.cat(probs)
            entropy = -(probs * torch.log(probs)).sum(dim=1).mean()
            #loss = loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimizer.step()

            # Calculate KL
            new_logits_v = self.net.logits(torch.as_tensor(np.array(states).squeeze(1), device=self.device))
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / probs).log() * probs).sum(dim=1).mean()
            env.add_scalar("Policy/KL", kl_div_v.item())
            env.add_scalar('Policy/Entropy', entropy.item())

            progress.set_description('reward=%2.2f' % (np.mean(env.mean_episode_reward)))

            # Save model
            if env.mean_episode_reward > best_avg_reward:
                torch.save(copy.deepcopy(self.net), 'reinforce.pt')
                best_avg_reward = env.mean_episode_reward

    def sample_episode(self, env):
        state = env.reset()
        log_probs = []
        rewards = []
        probs = []
        s = []

        for t in range(self.n_max_episode_steps):
            action, log_prob, p = self.net(torch.as_tensor(state, device=self.device))
            state, reward, done, infos = env.step([action])
            rewards.append(reward)
            log_probs.append(log_prob)
            probs.append(p)
            s.append(state)

            if done:
                break

        return log_probs, rewards, probs, s


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
