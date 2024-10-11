import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Actor(nn.Module):
    """Actor 网络，用于生成连续动作"""

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出动作在 [-1, 1] 之间


class Critic(nn.Module):
    """Critic 网络，用于评估状态-动作的值"""

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DDPGAgent:
    """基于 DDPG 算法的智能体"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.tau = 0.001
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.memory = []
        self.batch_size = 64

        # 初始化目标网络
        self.update_target_models(soft_update=False)

    def update_target_models(self, soft_update=True):
        """软更新目标网络"""
        if soft_update:
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
            # 硬更新：直接复制参数
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state)
        return action.detach().numpy().flatten()

    def remember(self, state, action, reward, next_state, done):
        """保存经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        """经验回放，进行训练"""
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # 计算目标 Q 值
            target_action = self.target_actor(torch.FloatTensor(next_state))
            target_q_value = self.target_critic(torch.FloatTensor(next_state), target_action)
            target = reward + (1 - done) * self.gamma * target_q_value.item()

            # 更新 Critic 网络
            current_q_value = self.critic(torch.FloatTensor(state), torch.FloatTensor(action))
            critic_loss = nn.MSELoss()(current_q_value, torch.FloatTensor([target]))
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # 更新 Actor 网络
            predicted_action = self.actor(torch.FloatTensor(state))
            actor_loss = -self.critic(torch.FloatTensor(state), predicted_action).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        self.update_target_models()

    def load(self, name):
        """加载模型"""
        self.actor.load_state_dict(torch.load(f"{name}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{name}_critic.pth"))

    def save(self, name):
        """保存模型"""
        torch.save(self.actor.state_dict(), f"{name}_actor.pth")
        torch.save(self.critic.state_dict(), f"{name}_critic.pth")
