import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
from uav_env import UAVEnv
from maddpg_agent import MADDPGAgent


def train_with_visualization():
    # 初始化环境和MADDPG代理
    env = UAVEnv()
    num_agents = env.num_coop_agents + env.num_noncoop_agents
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    agent = MADDPGAgent(num_agents, state_dim, action_dim, max_action)

    num_episodes = 500
    max_steps = 1000
    episode_rewards = []

    # 设置三维可视化
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create 3D axis
    colors = ['b', 'r', 'g', 'c', 'm', 'y']

    def plot_landing_zones(ax):
        """绘制起降场"""
        for runway in env.vertiport['takeoff']:
            ax.scatter(runway['center'][0], runway['center'][1], runway['center'][2], color='blue', s=100,
                       label='Takeoff Zone')
        for runway in env.vertiport['landing']:
            ax.scatter(runway['center'][0], runway['center'][1], runway['center'][2], color='green', s=100,
                       label='Landing Zone')

    def plot_cylinder(ax):
        """绘制圆柱形空域"""
        # 创建圆柱体侧面
        z = np.linspace(0, env.cylinder_height, 50)  # 高度方向
        theta = np.linspace(0, 2 * np.pi, 50)  # 角度方向
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = env.cylinder_radius * np.cos(theta_grid)
        y_grid = env.cylinder_radius * np.sin(theta_grid)

        # 绘制圆柱体侧面
        ax.plot_surface(x_grid, y_grid, z_grid, color='gray', alpha=0.2, edgecolor='none')  # 设置透明度

    def update_visualization(frame):
        ax.clear()
        nonlocal states

        # 绘制圆柱形空域
        plot_cylinder(ax)
        # 绘制起降场
        plot_landing_zones(ax)

        # 选择动作并执行一步
        actions = []
        for i in range(len(states)):  # 使用当前的states数量
            action = agent.select_action(np.array(states[i]), i)
            action = action + np.random.normal(0, 0.05, size=env.action_space.shape[0])  # 减小噪声
            action = np.clip(action, env.action_space.low, env.action_space.high)
            actions.append(action)

        # 执行环境步
        next_states, rewards, done, _ = env.step(actions)

        # 确保 rewards 的长度和 current states 的长度一致
        rewards = rewards[:len(states)]

        # 绘制无人机的位置
        for i in range(len(states)):  # 使用当前的states数量
            ax.scatter(states[i][0], states[i][1], states[i][2], c=colors[i % len(colors)], label=f'UAV {i}', s=50)
            # 绘制无人机的速度方向
            ax.quiver(states[i][0], states[i][1], states[i][2],
                      states[i][3], states[i][4], states[i][5],
                      length=2, color=colors[i % len(colors)], arrow_length_ratio=0.1)  # 缩短箭头的长度

        # 设置三维空间的显示范围
        ax.set_xlim([-env.cylinder_radius, env.cylinder_radius])
        ax.set_ylim([-env.cylinder_radius, env.cylinder_radius])
        ax.set_zlim([0, env.cylinder_height])
        ax.legend()
        ax.set_title(f'Episode: {current_episode}, Step: {frame}, Reward: {sum(rewards):.2f}')

        # 更新总奖励
        episode_rewards[-1] += sum(rewards)

        states = next_states  # 更新 states

        return ax

    # 训练主循环
    for current_episode in range(num_episodes):
        states = env.reset()
        episode_rewards.append(0)

        # 动画设置
        ani = FuncAnimation(fig, update_visualization, frames=range(max_steps), interval=50, repeat=False)
        plt.show(block=False)
        plt.pause(1)  # 动画刷新时间

        # 保存每个智能体的策略
        for i in range(num_agents):
            torch.save(agent.actors[i].state_dict(), f'maddpg_actor_{i}.pth')

        print(f"Episode: {current_episode + 1}, Total Reward: {episode_rewards[-1]:.2f}")

    plt.close()


if __name__ == "__main__":
    train_with_visualization()
