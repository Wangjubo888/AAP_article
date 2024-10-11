import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import Config

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, epsilon=0.2):
        self.actor = self._build_actor(state_dim, action_dim)
        self.critic = self._build_critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=10000)

    def _build_actor(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def _build_critic(self, state_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_mean = self.actor(state)
        action = action_mean.detach().numpy() + np.random.normal(0, 0.1, size=action_mean.size())
        return action

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def update(self):
        if len(self.memory) < Config.batch_size:
            return
        batch = random.sample(self.memory, Config.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        returns = self._calculate_returns(rewards)
        values = self.critic(states).squeeze()
        advantage = returns - values.detach()

        # 更新策略网络
        action_means = self.actor(states)
        dist = torch.distributions.Normal(action_means, 0.1)
        old_log_probs = dist.log_prob(actions).sum(dim=-1)

        new_dist = torch.distributions.Normal(self.actor(states), 0.1)
        new_log_probs = new_dist.log_prob(actions).sum(dim=-1)

        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -torch.min(surr1, surr2).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # 更新价值网络
        value_loss = (returns - values).pow(2).mean()
        self.optimizer_critic.zero_grad()
        value_loss.backward()
        self.optimizer_critic.step()

    def _calculate_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns)