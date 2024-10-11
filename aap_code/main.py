import numpy as np
import torch
import matplotlib.pyplot as plt
import Config
from environment import SimpleUAVEnv
from agent import PPOAgent
import wandb
import os

if __name__ == "__main__":
    config = Config()
    env = SimpleUAVEnv(config.num_agents, config.R1, config.R2, config.R3)
    agents = [PPOAgent(state_dim=config.state_dim, action_dim=config.action_dim, lr=config.lr, gamma=config.gamma,
                       epsilon=config.epsilon) for _ in range(config.num_agents)]

    # 使用WandB进行实验记录
    if config.use_wandb:
        wandb.init(project="UAV-MARL", config=config.__dict__)

    # 日志记录
    rewards_log = []
    success_rate_log = []
    positions_log = [[] for _ in range(config.num_agents)]

    for ep in range(config.max_episodes):
        states = env.reset()
        trajectories = [[] for _ in range(config.num_agents)]
        done = [False] * config.num_agents
        total_rewards = [0] * config.num_agents
        success_count = 0

        for step in range(config.max_steps):
            actions = [agents[i].select_action(states[i]) for i in range(config.num_agents)]
            next_states, rewards, dones = env.step(actions)
            for i in range(config.num_agents):
                if not done[i]:
                    agents[i].store_transition(states[i], actions[i], rewards[i], next_states[i])
                    total_rewards[i] += rewards[i]
                    positions_log[i].append(states[i][:3])  # 记录位置数据
                    if dones[i]:
                        success_count += 1 if rewards[i] > 0 else 0
            states = next_states
            done = dones
            if all(done):
                break

        # 更新智能体
        for agent in agents:
            agent.update()

        # 日志记录
        avg_reward = np.mean(total_rewards)
        success_rate = success_count / config.num_agents
        rewards_log.append(avg_reward)
        success_rate_log.append(success_rate)

        if config.use_wandb:
            wandb.log({"Average Reward": avg_reward, "Success Rate": success_rate})

        print(f"Episode {ep + 1}/{config.max_episodes}, Average Reward: {avg_reward}, Success Rate: {success_rate}")

    # 保存模型
    if not os.path.exists("models"):
        os.makedirs("models")
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), f"models/actor_agent_{i}.pth")
        torch.save(agent.critic.state_dict(), f"models/critic_agent_{i}.pth")

    # 可视化训练结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_log)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Training Progress - Reward')

    plt.subplot(1, 2, 2)
    plt.plot(success_rate_log)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.title('Training Progress - Success Rate')

    plt.tight_layout()
    plt.show()

    # 三维可视化无人机轨迹
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y', 'm']

    for i in range(config.num_agents):
        positions = np.array(positions_log[i])
        if len(positions) > 0:
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[i % len(colors)],
                    label=f'Drone {i}')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectory of UAVs')
    ax.legend()
    plt.show()