import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from env import UrbanUAVEnv
from agent import DQNAgent
from definitions import MAX_ACCELERATION, DT
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random

# 超参数设置
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 5000
TARGET_UPDATE_FREQUENCY = 10
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 200

# 设置保存模型和日志的路径
SAVE_MODEL_PATH = './models'
LOG_DIR = './runs/experiment1'

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

if __name__ == "__main__":
    env = UrbanUAVEnv()
    num_agents = env.num_drones  # 无人机数量
    state_dim = env.observation_space.shape[0]
    action_dim = 125  # 动作维度：5 * 5 * 5

    agents = [DQNAgent(state_dim, action_dim, learning_rate=LEARNING_RATE, gamma=GAMMA,
                       epsilon_start=EPSILON_START, epsilon_end=EPSILON_END,
                       epsilon_decay=EPSILON_DECAY, target_update_frequency=TARGET_UPDATE_FREQUENCY,
                       memory_capacity=MEMORY_CAPACITY) for _ in range(num_agents)]
    writer = SummaryWriter(log_dir=LOG_DIR)

    plt.ion()  # 开启交互式模式

    # 预填充经验回放池
    PRE_FILL_STEPS = MEMORY_CAPACITY // 10  # 预填充 10% 的容量

    print("Filling Replay Memory...")
    for _ in tqdm(range(PRE_FILL_STEPS), desc='Filling Replay Memory'):
        states = env.reset()
        done = False
        while not done:
            actions = []
            for i in range(num_agents):
                action_index = random.randint(0, action_dim - 1)
                idx_speed = action_index % 5
                idx_heading = (action_index // 5) % 5
                idx_altitude = (action_index // 25) % 5

                delta_speed = np.linspace(-MAX_ACCELERATION, MAX_ACCELERATION, num=5)[idx_speed]
                delta_heading = np.linspace(-math.pi / 4, math.pi / 4, num=5)[idx_heading]
                delta_altitude = np.linspace(-5.0, 5.0, num=5)[idx_altitude]
                env_action = {'delta_speed': delta_speed,
                              'delta_heading': delta_heading,
                              'delta_altitude': delta_altitude}
                actions.append(env_action)
            next_states, rewards, done, _ = env.step(actions)
            for i, agent in enumerate(agents):
                agent.memory.push(states[i], action_index, rewards[i], next_states[i], done)
            states = next_states

    for episode in tqdm(range(NUM_EPISODES), desc='Training'):
        states = env.reset()
        total_rewards = np.zeros(num_agents)
        episode_loss = 0.0
        for step in range(MAX_STEPS_PER_EPISODE):
            actions = []
            for i, agent in enumerate(agents):
                state = states[i]
                action_index = agent.select_action(state)
                # 动作索引转换为实际动作
                idx_speed = action_index % 5
                idx_heading = (action_index // 5) % 5
                idx_altitude = (action_index // 25) % 5

                delta_speed = np.linspace(-MAX_ACCELERATION, MAX_ACCELERATION, num=5)[idx_speed]
                delta_heading = np.linspace(-math.pi / 4, math.pi / 4, num=5)[idx_heading]
                delta_altitude = np.linspace(-5.0, 5.0, num=5)[idx_altitude]
                env_action = {'delta_speed': delta_speed,
                              'delta_heading': delta_heading,
                              'delta_altitude': delta_altitude}
                actions.append(env_action)

            next_states, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.memory.push(states[i], action_index, rewards[i], next_states[i], done)
                loss = agent.optimize_model(BATCH_SIZE)
                if loss is not None:
                    episode_loss += loss.item()
                total_rewards[i] += rewards[i]

            states = next_states

            if step % 10 == 0:
                env.render()

            if done:
                break

        # 更新目标网络
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            for agent in agents:
                agent.update_target_network()

        # 计算平均奖励和损失
        avg_reward = np.mean(total_rewards)
        avg_loss = episode_loss / MAX_STEPS_PER_EPISODE

        # 使用 TensorBoard 记录指标
        writer.add_scalar('Average Reward', avg_reward, episode)
        writer.add_scalar('Total Reward', np.sum(total_rewards), episode)
        writer.add_scalar('Average Loss', avg_loss, episode)

        # 输出日志
        print(f'Episode {episode}, Average Reward: {avg_reward:.2f}, Total Reward: {np.sum(total_rewards):.2f}, Average Loss: {avg_loss:.4f}')

        # 保存模型
        if episode % 50 == 0:
            for idx, agent in enumerate(agents):
                torch.save(agent.policy_net.state_dict(), f'{SAVE_MODEL_PATH}/agent_{idx}_episode_{episode}.pth')

    env.close()
    writer.close()

    plt.ioff()
    plt.show()
