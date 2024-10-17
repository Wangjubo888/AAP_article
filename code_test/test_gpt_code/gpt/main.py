# main.py

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from env import UrbanUAVEnv
from agent import MADDPGAgent
from definitions import MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# 超参数
BATCH_SIZE = 1024
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200

SAVE_MODEL_PATH = './models'
LOG_DIR = './runs/experiment_maddpg'

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if __name__ == "__main__":
    env = UrbanUAVEnv()
    num_agents = env.num_drones  # 固定无人机数量
    state_dim = env.observation_space.shape[0]  # 13
    action_dim = env.action_space.shape[0]  # 3
    agents = []

    for i in range(num_agents):
        agent = MADDPGAgent(
            num_agents=num_agents,
            agent_index=i,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=1.0  # 动作输出经过 tanh，取值范围 [-1, 1]
        )
        agents.append(agent)

    writer = SummaryWriter(log_dir=LOG_DIR)

    plt.ion()

    for episode in tqdm(range(NUM_EPISODES), desc='Training'):
        obs = env.reset()
        total_rewards = np.zeros(num_agents)
        episode_loss = 0.0

        for step in range(MAX_STEPS_PER_EPISODE):
            actions = []
            observations = []

            for i, agent in enumerate(agents):
                state = obs[i]
                action = agent.select_action(state)
                # 添加探索噪声
                noise = np.random.normal(0, 0.1, size=action_dim)
                action = action + noise
                # 限制动作范围
                action = np.clip(action, -1.0, 1.0)
                # 将动作映射到实际范围
                real_action = action * np.array([MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE])
                actions.append(real_action)
                observations.append(state)

            # 确保 actions 列表长度与无人机数量一致
            if len(actions) < num_agents:
                for _ in range(num_agents - len(actions)):
                    actions.append(np.zeros(action_dim))

            next_obs, rewards, dones, _ = env.step(actions)

            # 存储经验到所有智能体的回放池
            for i, agent in enumerate(agents):
                agent.memory.push(
                    obs,      # 所有智能体的观测
                    actions,  # 所有智能体的动作
                    rewards,  # 所有智能体的奖励
                    next_obs,
                    dones     # 每个智能体的完成状态
                )

            obs = next_obs
            total_rewards += rewards

            # 更新所有智能体
            for agent in agents:
                agent.update(agents, BATCH_SIZE)

            # 渲染可视化
            if step % 10 == 0:
                env.render()

            if all(dones):
                break

        avg_reward = np.mean(total_rewards)
        writer.add_scalar('Average Reward', avg_reward, episode)
        writer.add_scalar('Total Reward', np.sum(total_rewards), episode)

        print(f'Episode {episode}, Average Reward: {avg_reward:.2f}, Total Reward: {np.sum(total_rewards):.2f}')

        # 每隔100轮保存模型
        if episode % 100 == 0:
            for idx, agent in enumerate(agents):
                torch.save(agent.actor.state_dict(), f'{SAVE_MODEL_PATH}/agent_{idx}_actor_episode_{episode}.pth')
                torch.save(agent.critic.state_dict(), f'{SAVE_MODEL_PATH}/agent_{idx}_critic_episode_{episode}.pth')

    env.close()
    writer.close()

    plt.ioff()
    plt.show()
