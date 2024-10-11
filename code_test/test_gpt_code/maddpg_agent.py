import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer1(state))
        a = torch.relu(self.layer2(a))
        return self.max_action * torch.tanh(self.layer3(a))

# Critic网络
class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(total_state_dim + total_action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        q = torch.relu(self.layer1(torch.cat([states, actions], 1)))
        q = torch.relu(self.layer2(q))
        return self.layer3(q)


class MADDPGAgent:
    def __init__(self, num_agents, state_dim, action_dim, max_action):
        self.num_agents = num_agents
        self.action_dim = action_dim

        # 创建每个智能体的Actor
        self.actors = [Actor(state_dim, action_dim, max_action).cuda() for _ in range(num_agents)]
        self.actor_targets = [Actor(state_dim, action_dim, max_action).cuda() for _ in range(num_agents)]
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=1e-3) for actor in self.actors]

        for i in range(num_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())

        # 共享的Critic
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents
        self.critic = Critic(total_state_dim, total_action_dim).cuda()
        self.critic_target = Critic(total_state_dim, total_action_dim).cuda()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

    def select_action(self, state, agent_idx):
        state = torch.FloatTensor(state.reshape(1, -1)).cuda()
        return self.actors[agent_idx](state).cpu().data.numpy().flatten()

    def train(self):
        # 检查缓冲区是否有足够的经验
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)

        # 将数据转换为torch张量
        state = torch.FloatTensor(np.array(state)).cuda()
        action = torch.FloatTensor(np.array(action)).cuda()
        reward = torch.FloatTensor(np.array(reward)).reshape(self.batch_size, self.num_agents, 1).cuda()
        next_state = torch.FloatTensor(np.array(next_state)).cuda()
        done = torch.FloatTensor(np.array(done)).reshape(self.batch_size, self.num_agents, 1).cuda()

        # 更新Critic
        with torch.no_grad():
            next_actions = torch.cat([self.actor_targets[i](next_state[:, i]) for i in range(self.num_agents)], dim=1)
            target_q = self.critic_target(next_state.view(self.batch_size, -1), next_actions)
            target_q = reward.sum(1) + (1 - done.sum(1)) * self.gamma * target_q

        current_q = self.critic(state.view(self.batch_size, -1), action.view(self.batch_size, -1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新每个Actor
        for i in range(self.num_agents):
            current_actions = [self.actors[j](state[:, j]) if j == i else action[:, j] for j in range(self.num_agents)]
            current_actions = torch.cat(current_actions, dim=1)
            actor_loss = -self.critic(state.view(self.batch_size, -1), current_actions).mean()

            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()

        # 软更新目标网络
        for i in range(self.num_agents):
            for param, target_param in zip(self.actors[i].parameters(), self.actor_targets[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
