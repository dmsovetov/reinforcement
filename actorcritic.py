import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim import RMSprop
from tqdm import tqdm

from commandline import add_default_args, get_device
from environments import TrainingEnvironment, DeviceEnv, make_training_env, get_env_dims
from models import AC


class Rollout:
    def __init__(self, num_envs: int, steps: int, device: str):
        self.entropy = torch.zeros((steps, num_envs), device=device)
        self.log_prob = torch.zeros((steps, num_envs), device=device)
        self.values = torch.zeros((steps, num_envs), device=device)
        self.rewards = torch.zeros((steps, num_envs), device=device)
        self.done_mask = torch.zeros((steps, num_envs), device=device)
        self.step = 0

    def append(self, state_values: Tensor, rewards: Tensor, entropy: Tensor, log_probs: Tensor, done_mask: Tensor):
        self.log_prob[self.step] = log_probs
        self.entropy[self.step] = entropy
        self.values[self.step] = state_values
        self.rewards[self.step] = rewards
        self.done_mask[self.step] = done_mask
        self.step += 1


class RolloutCollector:
    def __init__(self, policy: Module, steps: int, gamma: float, device: str):
        self.policy = policy
        self.steps = steps
        self.gamma = gamma
        self.device = device

    def __call__(self, env, state):
        rollout = Rollout(env.num_envs, self.steps, self.device)

        for t in range(self.steps):
            actions, log_prob, entropy, values = self.policy(state)
            state, rewards, done_mask, infos = env.step(actions)

            for info in [info for info in infos if info.truncated_state is not None]:
                with torch.no_grad():
                    terminal_value = self.policy.state_value(info.truncated_state)[0]
                    rewards[info.index] += self.gamma * terminal_value

            rollout.append(values, rewards, entropy, log_prob, done_mask)

        return state, rollout


class NStepAdvantageEstimation:
    def __init__(self, net: Module, steps: int, gamma: float):
        self.net = net
        self.steps = steps
        self.gamma = gamma

    def __call__(self, rollout, next_state):
        returns = torch.zeros_like(rollout.rewards)

        with torch.no_grad():
            cumulative_return = self.net.state_value(next_state).transpose(0, 1)

        for t in reversed(range(self.steps)):
            cumulative_return = rollout.rewards[t] + self.gamma * cumulative_return * rollout.done_mask[t]
            returns[t] = cumulative_return

        return returns, returns - rollout.values


class GeneralizedAdvantageEstimation:
    def __init__(self, net: Module, steps: int, gamma: float, gae: float, advantage_returns: bool = True):
        self.net = net
        self.steps = steps
        self.gamma = gamma
        self.gae = gae
        self.advantage_returns = advantage_returns

    def __call__(self, rollout, next_state):
        advantages = torch.zeros_like(rollout.rewards)
        returns = torch.zeros_like(rollout.rewards)

        with torch.no_grad():
            last_state_value = self.net.state_value(next_state).transpose(0, 1)

        cumulative = 0

        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_values = last_state_value
            else:
                next_values = rollout.values[t + 1]

            non_terminal = rollout.done_mask[t]

            delta = rollout.rewards[t] + self.gamma * next_values * non_terminal - rollout.values[t]
            cumulative = delta + self.gamma * self.gae * non_terminal * cumulative
            advantages[t] = cumulative

        if self.advantage_returns:
            return (advantages + rollout.values).detach(), advantages

        # Calculate cumulative returns otherwise
        cumulative_return = last_state_value

        for t in reversed(range(self.steps)):
            cumulative_return = rollout.rewards[t] + self.gamma * cumulative_return * rollout.done_mask[t]
            returns[t] = cumulative_return

        return returns, advantages


class ActorCritic:
    def __init__(
            self,
            net: Module,
            n_training_steps: int = 1000,
            gamma: float = 1.0,
            gae: float = 1.0,
            learning_rate: float = 1e-3,
            clip: float = None,
            n_steps: int = 5,
            ent_coef: float = 0,
            vf_coef: float = 0.5,
            normalize_advantage: bool = False,
            device: str = 'cpu'):
        self.net = net.to(device)
        self.n_training_steps = n_training_steps
        self.gamma = gamma
        self.gae = gae
        self.learning_rate = learning_rate
        self.device = device
        self.clip = clip
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.n_steps = n_steps
        self.normalize_advantage = normalize_advantage

    def fit(self, env: TrainingEnvironment):
        progress = tqdm(range(self.n_training_steps // (env.num_envs * self.n_steps)))
        env = DeviceEnv(env, device=self.device, inverse_done=True)
        env.track_gradients(self.net)

        optimizer = RMSprop(self.net.parameters(), lr=self.learning_rate, eps=1e-5, alpha=0.99)

        if self.gae < 1e-6:
            advantage_estimation = NStepAdvantageEstimation(self.net, self.n_steps, self.gamma)
        else:
            advantage_estimation = GeneralizedAdvantageEstimation(self.net, self.n_steps, self.gamma, self.gae)
        rollout_collector = RolloutCollector(self.net, self.n_steps, self.gamma, self.device)

        state = env.reset()

        def select_action(_, s):
            a, _, _ = self.net.action(torch.as_tensor(s, dtype=torch.float32, device=self.device))
            return a

        env.add_video_recording(select_action, 1000)

        for _ in progress:
            # Collect next rollout
            state, rollout = rollout_collector(env, state)

            # Estimate advantage
            returns, advantage = advantage_estimation(rollout, next_state=state)

            if self.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            # Calculate loss
            policy_loss = (-rollout.log_prob * advantage.detach()).mean()
            critic_loss = F.mse_loss(rollout.values, returns)
            entropy_loss = -rollout.entropy.mean()
            loss = policy_loss + self.vf_coef * critic_loss + self.ent_coef * entropy_loss

            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=self.clip)

            optimizer.step()

            # Update stats
            if env.mean_episode_reward:
                progress.set_description('episodes=%d, reward=%2.2f' % (env.total_episodes, env.mean_episode_reward))

            env.add_scalar('Training/Policy Loss', -policy_loss.item())
            env.add_scalar('Training/Value Loss', critic_loss.item())
            env.add_scalar('Training/Entropy Loss', -entropy_loss.item())

            env.add_scalar('train/policy_loss', policy_loss.item())
            env.add_scalar('train/value_loss', critic_loss.item())
            env.add_scalar('train/entropy_loss', entropy_loss.item())


def command_line(in_args):
    def run(args):
        env = make_training_env(args.env, 'AC', args.envs)
        env.add_hyperparameters(args)
        obs_n, act_n = get_env_dims(env)
        policy = AC(obs_n, act_n, hidden=[args.hidden, args.hidden])
        ac = ActorCritic(policy,
                         n_training_steps=args.iterations,
                         device=get_device(args),
                         normalize_advantage=False,
                         n_steps=args.steps,
                         ent_coef=args.ent_coef,
                         clip=args.grad_clip,
                         learning_rate=args.learning_rate,
                         gamma=args.gamma,
                         gae=args.gae)
        ac.fit(env)

    cmd = in_args.add_parser('ac')
    add_default_args(cmd)
    cmd.add_argument('--iterations', type=int, default=1000)
    cmd.add_argument('--steps', type=int, default=5)
    cmd.add_argument('--ent-coef', type=float, default=1e-05)
    cmd.add_argument('--vf-coef', type=float, default=0.5)
    cmd.add_argument('--grad-clip', type=float, default=0.5)
    cmd.add_argument('--learning-rate', type=float, default=7e-4)
    cmd.add_argument('--hidden', type=int, default=64)
    cmd.add_argument('--gae', type=float, default=1.0)
    cmd.set_defaults(func=run)
