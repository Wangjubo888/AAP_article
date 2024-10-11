import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from uav_env import UAVEnv

def visualize():
    env = UAVEnv()
    states = env.reset()
    num_agents = env.num_coop_agents + env.num_noncoop_agents

    fig, ax = plt.subplots()
    colors = ['b', 'r', 'g', 'c', 'm', 'y']

    def update(frame):
        ax.clear()
        next_states, _, _, _ = env.step([np.zeros(env.action_space.shape[0])] * num_agents)
        for i in range(num_agents):
            ax.scatter(next_states[i][0], next_states[i][1], c=colors[i % len(colors)], label=f'UAV {i}')
            ax.quiver(next_states[i][0], next_states[i][1], next_states[i][3], next_states[i][4])

        ax.set_xlim(-env.cylinder_radius, env.cylinder_radius)
        ax.set_ylim(-env.cylinder_radius, env.cylinder_radius)
        ax.legend()
        ax.set_title(f'Step: {frame}')
        return ax

    ani = FuncAnimation(fig, update, frames=range(100), interval=200)
    plt.show()

if __name__ == "__main__":
    visualize()
