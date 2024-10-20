# agent.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


# 神经网络定义
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.max_action


class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(total_state_dim + total_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = self.l3(x)
        return x


# 经验回放池
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        self.memory.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# MADDPG智能体
class MADDPGAgent:
    def __init__(self, num_agents, agent_index, state_dim, action_dim, max_action):
        self.num_agents = num_agents
        self.agent_index = agent_index
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(num_agents * state_dim, num_agents * action_dim)
        self.critic_target = Critic(num_agents * state_dim, num_agents * action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = ReplayMemory(1000)
        self.gamma = 0.95
        self.tau = 0.01

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        return action

    def update(self, agents, batch_size):
        if len(self.memory) < batch_size:
            return

        # Sample a batch from memory
        samples = self.memory.sample(batch_size)
        batch = list(zip(*samples))
        states = np.array(batch[0])  # [batch_size, num_agents, state_dim]
        actions = np.array(batch[1])  # [batch_size, num_agents, action_dim]
        rewards = np.array(batch[2])  # [batch_size, num_agents]
        next_states = np.array(batch[3])  # [batch_size, num_agents, state_dim]
        dones = np.array(batch[4])  # [batch_size, num_agents]

        # Convert to torch tensors
        states = torch.FloatTensor(states.reshape(batch_size, -1))  # [batch_size, num_agents * state_dim]
        actions = torch.FloatTensor(actions.reshape(batch_size, -1))  # [batch_size, num_agents * action_dim]
        rewards = torch.FloatTensor(rewards[:, self.agent_index].reshape(batch_size, 1))  # [batch_size, 1]
        next_states = torch.FloatTensor(next_states.reshape(batch_size, -1))  # [batch_size, num_agents * state_dim]
        dones = torch.FloatTensor(dones[:, self.agent_index].reshape(batch_size, 1))  # [batch_size, 1]

        # Compute target Q values
        with torch.no_grad():
            next_actions = []
            for agent in agents:
                idx = agent.agent_index
                next_state_i = next_states[:, idx * self.state_dim:(idx +1)*self.state_dim]
                next_a_i = agent.actor_target(next_state_i)
                next_actions.append(next_a_i)
            next_actions = torch.cat(next_actions, dim=1)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * target_q * (1 - dones)

        # Compute current Q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        # Optimize Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute Actor loss
        actions_pred = []
        for agent in agents:
            idx = agent.agent_index
            state_i = states[:, idx * self.state_dim:(idx + 1)*self.state_dim]
            if agent.agent_index == self.agent_index:
                a_i = self.actor(state_i)
            else:
                # Use the current actions stored in the batch
                a_i = actions[:, idx * self.action_dim:(idx + 1)*self.action_dim]
            actions_pred.append(a_i)
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic(states, actions_pred).mean()

        # Optimize Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
