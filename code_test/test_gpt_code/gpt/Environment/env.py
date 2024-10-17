# env.py

import gym
import numpy as np
import random
from typing import List, Dict
from definitions import UAV, MIN_SAFE_DISTANCE_MATRIX, UAV_TYPES, MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE, DT, \
    R1, R2, R3, H1, H2, MAX_ALTITUDE
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class UrbanUAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UrbanUAVEnv, self).__init__()
        self.num_drones = 5  # 调整为与起飞垫数量一致
        self.drones = []
        self.time_step = 0
        self.max_time_steps = 200

        # 定义动作空间为连续空间
        action_dim = 3  # [delta_speed, delta_turn, delta_climb]
        self.action_space = gym.spaces.Box(
            low=np.array([-MAX_ACCELERATION, -MAX_TURN_RATE, -MAX_CLIMB_RATE]),
            high=np.array([MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE]),
            dtype=np.float32
        )

        # 定义观测空间
        state_dim = 13  # [position(3), velocity(3), energy(1), type_info(1), cooperative_flag(1), priority_info(1), neighbors(3)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # 空域定义
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.H1 = H1
        self.H2 = H2

        # 起飞和降落场
        self.takeoff_pads = {
            0: {'position': np.array([-R1, -R1, 0.0]), 'status': 'available'},
            1: {'position': np.array([R1, -R1, 0.0]), 'status': 'available'},
            2: {'position': np.array([-R1, R1, 0.0]), 'status': 'available'},
            3: {'position': np.array([R1, R1, 0.0]), 'status': 'available'},
            4: {'position': np.array([0.0, 0.0, 0.0]), 'status': 'available'},
        }
        self.landing_pads = {
            0: {'position': np.array([-R1 / 2, R1 / 2, 0.0]), 'status': 'available'},
            1: {'position': np.array([R1 / 2, -R1 / 2, 0.0]), 'status': 'available'},
        }

        # 环和中心
        self.takeoff_ring_center = np.array([0.0, 0.0, self.H1])
        self.landing_ring_center = np.array([0.0, 0.0, self.H2])

        # 进近区域和空域
        self.approach_area_radius = self.R2
        self.airspace_radius = self.R3

        # 降落序号计数器
        self.landing_sequence = 0

        self.done_drones = set()
        self.reset()

        # 初始化渲染标志和图形对象
        self.viewer_initialized = False
        self.fig = None
        self.ax = None

    def reset(self):
        self.drones = []
        self.done_drones = set()
        self.landing_sequence = 0

        for pad in self.takeoff_pads.values():
            pad['status'] = 'available'
        for pad in self.landing_pads.values():
            pad['status'] = 'available'

        # 分配任务：优先分配 'takeoff' 任务，确保所有需要起飞的无人机都能获得起飞垫
        for i in range(self.num_drones):
            if i < len(self.takeoff_pads):
                task_type = 'takeoff'
            else:
                task_type = random.choice(['landing', 'cruise'])
            drone = UAV.random_uav(drone_id=i, task_type=task_type, environment=self)
            if drone:
                self.drones.append(drone)

        self.time_step = 0
        return self._get_observation()

    def step(self, actions: List[np.ndarray]):
        rewards = []
        dones = []
        infos = {}

        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                if i < len(actions):
                    action = actions[i]
                else:
                    action = np.zeros(self.action_space.shape[0])  # 默认动作
                other_drones = [d for idx, d in enumerate(self.drones) if idx != i]
                drone.step(action, other_drones)
                if drone.task_finished or drone.energy <= 0:
                    self.done_drones.add(i)

        self.time_step += 1
        self._update_pads_status()
        self._update_collisions()
        rewards = self._compute_rewards()
        obs = self._get_observation()

        # 为每个智能体生成单独的完成状态
        dones = [drone.task_finished or drone.energy <= 0 for drone in self.drones]

        done = False  # 环境的全局完成状态，如果需要可以保持为False
        return obs, rewards, dones, infos

    def _compute_rewards(self):
        rewards = []
        for i, drone in enumerate(self.drones):
            reward = 0.0
            if drone.r_task_completion:
                reward += 100.0  # 任务完成奖励
            if drone.avoided_collision:
                reward += 5.0  # 成功避让奖励
            if drone.r_collision:
                reward -= 100.0  # 碰撞惩罚
            if drone.violation:
                reward -= 50.0  # 违规惩罚
            reward -= drone.energy_consumption * 0.1  # 能量消耗惩罚
            rewards.append(reward)
        return rewards

    def _get_observation(self):
        obs = []
        for i, drone in enumerate(self.drones):
            position = drone.position
            velocity = drone.speed * drone.heading
            energy = np.array([drone.energy])
            type_index = UAV_TYPES.index(drone.type)
            type_info = np.array([type_index])
            cooperative_flag = np.array([1.0 if drone.cooperative else 0.0])
            priority_info = np.array([drone.priority])
            neighbors = self._get_neighbors(drone)
            observation = np.concatenate(
                [position, velocity, energy, type_info, cooperative_flag, priority_info, neighbors])
            obs.append(observation)
        return obs

    def _get_neighbors(self, drone):
        closest_distance = float('inf')
        neighbor_info = np.zeros(3)
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < 100.0 and distance < closest_distance:
                    closest_distance = distance
                    relative_position = other_drone.position - drone.position
                    neighbor_info = relative_position
        return neighbor_info

    def _update_pads_status(self):
        # 更新起飞垫
        for drone in self.drones:
            if drone.task_type == 'takeoff' and drone.taking_off:
                pad_id = drone.assigned_pad
                if pad_id is not None and self.takeoff_pads[pad_id]['status'] == 'occupied':
                    if drone.altitude > 10.0:
                        self.takeoff_pads[pad_id]['status'] = 'available'

        # 更新降落垫
        for drone in self.drones:
            if drone.task_type == 'landing' and drone.task_finished:
                pad_id = drone.assigned_pad
                if pad_id is not None and self.landing_pads[pad_id]['status'] == 'occupied':
                    self.landing_pads[pad_id]['status'] = 'available'

    def _update_collisions(self):
        for drone in self.drones:
            drone.r_collision = 0

        for i, drone in enumerate(self.drones):
            for j, other_drone in enumerate(self.drones):
                if j > i:
                    if self.check_collision(drone, other_drone):
                        drone.r_collision = 1
                        other_drone.r_collision = 1
                        self.done_drones.add(i)
                        self.done_drones.add(j)

    def check_collision(self, drone1: UAV, drone2: UAV) -> bool:
        type_pair = (drone1.type, drone2.type)
        min_distance = MIN_SAFE_DISTANCE_MATRIX.get(type_pair, 10.0)
        distance = np.linalg.norm(drone1.position - drone2.position)
        return distance < min_distance

    def get_idle_takeoff_pads(self):
        return [pad_id for pad_id, pad in self.takeoff_pads.items() if pad['status'] == 'available']

    def get_idle_landing_pads(self):
        return [pad_id for pad_id, pad in self.landing_pads.items() if pad['status'] == 'available']

    def assign_landing_pad(self, drone):
        idle_pads = self.get_idle_landing_pads()
        if idle_pads:
            pad_id = idle_pads[0]
            self.landing_pads[pad_id]['status'] = 'occupied'
            return pad_id
        else:
            return None

    def has_idle_landing_pad(self):
        return bool(self.get_idle_landing_pads())

    def get_next_landing_sequence(self):
        self.landing_sequence += 1
        return self.landing_sequence

    def approach_area_center(self):
        return np.array([0.0, 0.0, self.H2])

    def render(self, mode='human'):
        if not self.viewer_initialized:
            self.fig = plt.figure(figsize=(12, 10))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.viewer_initialized = True

        self.ax.cla()  # 清空之前的绘图内容

        # 绘制外空域（R3）
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = self.R3 * np.outer(np.cos(u), np.sin(v))
        y = self.R3 * np.outer(np.sin(u), np.sin(v))
        z = self.R3 * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_wireframe(x, y, z, color='grey', alpha=0.1)

        # 绘制进近空域（R2）
        x_inner = self.R2 * np.outer(np.cos(u), np.sin(v))
        y_inner = self.R2 * np.outer(np.sin(u), np.sin(v))
        z_inner = self.R2 * np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax.plot_wireframe(x_inner, y_inner, z_inner, color='green', alpha=0.1)

        # 绘制起飞环
        theta = np.linspace(0, 2 * np.pi, 100)
        x_takeoff = self.R1 * np.cos(theta)
        y_takeoff = self.R1 * np.sin(theta)
        z_takeoff = np.ones_like(x_takeoff) * self.H1
        self.ax.plot(x_takeoff, y_takeoff, z_takeoff, color='blue', linestyle='--', label='Takeoff Ring')

        # 绘制降落环
        z_landing = np.ones_like(x_takeoff) * self.H2
        self.ax.plot(x_takeoff, y_takeoff, z_landing, color='orange', linestyle='--', label='Landing Ring')

        # 绘制起飞垫
        for pad_id, pad in self.takeoff_pads.items():
            pos = pad['position']
            self.ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='s', s=100,
                            label='Takeoff Pad' if pad_id == 0 else "")

        # 绘制降落垫
        for pad_id, pad in self.landing_pads.items():
            pos = pad['position']
            self.ax.scatter(pos[0], pos[1], pos[2], color='orange', marker='s', s=100,
                            label='Landing Pad' if pad_id == 0 else "")

        # 绘制无人机和轨迹
        for drone in self.drones:
            positions = np.array(drone.position_history)
            if positions.shape[0] > 1:
                if drone.task_type == 'takeoff':
                    color = 'cyan'
                    marker = 'o'
                elif drone.task_type == 'landing':
                    color = 'magenta'
                    marker = '^'
                else:
                    color = 'red' if not drone.cooperative else 'green'
                    marker = 's'
                self.ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=color, linewidth=1)
            # 绘制无人机当前位置
            if drone.task_type == 'takeoff':
                color = 'cyan'
                marker = 'o'
            elif drone.task_type == 'landing':
                color = 'magenta'
                marker = '^'
            else:
                color = 'red' if not drone.cooperative else 'green'
                marker = 's'
            self.ax.scatter(drone.position[0], drone.position[1], drone.position[2], color=color, marker=marker, s=50)

        # 设置图形参数
        self.ax.set_xlim([-self.R3, self.R3])
        self.ax.set_ylim([-self.R3, self.R3])
        self.ax.set_zlim([0, MAX_ALTITUDE])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title(f'Time Step: {self.time_step}')

        # 创建图例，避免重复
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys())

        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.viewer_initialized:
            plt.close(self.fig)
            self.viewer_initialized = False
