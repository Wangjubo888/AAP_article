import gym
import numpy as np
import random
from typing import List, Dict
from definitions import UAV, MAX_DRONES, MIN_DISTANCE, SENSING_RANGE, MAX_EPISODE_LEN, MAX_ACCELERATION, DT, NUM_MOVE
import matplotlib.pyplot as plt


class UrbanUAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UrbanUAVEnv, self).__init__()
        self.num_drones = MAX_DRONES
        self.drones = []  # 无人机列表
        self.time_step = 0
        self.max_time_steps = MAX_EPISODE_LEN
        self.num_move = NUM_MOVE
        self.cylinder_radius = 5000
        self.cylinder_height = 1500
        # 起飞场和降落场的配置
        self.takeoff_pads_positions = {
            0: np.array([-150.0, -150.0, 0.0]),
            1: np.array([150.0, -150.0, 0.0]),
        }
        self.landing_pads_positions = {
            0: np.array([-150.0, 150.0, 0.0]),
            1: np.array([150.0, 150.0, 0.0]),
        }
        self.takeoff_pads_status = {0: 'available', 1: 'available'}
        self.landing_pads_status = {0: 'available', 1: 'available'}
        self.takeoff_pad_cooldown = 10
        self.landing_pad_cooldown = 10
        self.takeoff_pad_cooldown_timers = {0: 0, 1: 0}
        self.landing_pad_cooldown_timers = {0: 0, 1: 0}

        # 起飞环参数
        self.takeoff_ring_center = np.array([0.0, -150.0, 20.0])
        self.takeoff_ring_altitude = 20.0

        # 降落环参数
        self.landing_ring_center = np.array([0.0, 150.0, 20.0])
        self.landing_ring_altitude = 20.0
        self.landing_ring_radius = 50.0

        # 降落进近点（降落环外部）
        self.landing_approach_point = np.array([0.0, 200.0, 20.0])

        # 降落排序序号计数器
        self.landing_sequence_counter = 0

        # 定义动作空间和观测空间
        # 动作空间：delta_speed, delta_heading, delta_altitude
        self.action_space = gym.spaces.Dict({
            'delta_speed': gym.spaces.Box(-MAX_ACCELERATION, MAX_ACCELERATION, shape=(1,), dtype=np.float32),
            'delta_heading': gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
            'delta_altitude': gym.spaces.Box(-5.0, 5.0, shape=(1,), dtype=np.float32),
        })

        # 观测空间：自身状态 + 邻居信息
        # 自身状态：位置（x, y, z），速度（vx, vy, vz），剩余能量
        # 邻居信息：相对位置和速度
        # 观测空间
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        # 定义空域范围
        self.airspace_bounds = (-200.0, -200.0, 200.0, 200.0)
        self.done_drones = set()
        self.reset()

    def get_idle_landing_pads(self):
        """获取空闲的降落场列表"""
        return [pad_id for pad_id, status in self.landing_pads_status.items() if status == 'available']

    def _spawn_on_surface(self):
        """
        在圆柱体侧表面生成一点
        :return:侧表面np.array([x,y,z]) \
        [ 983.23415065 -182.34748423  132.68898602]
        """
        theta = random.uniform(0, 2 * np.pi)
        x = self.cylinder_radius * np.cos(theta)
        y = self.cylinder_radius * np.sin(theta)
        z = np.random.uniform(0, self.cylinder_height)
        return np.array([x, y, z])

    def reset(self):
        self.drones = []
        self.done_drones = set()
        # 重置起飞场和降落场状态
        self.takeoff_pads_status = {0: 'available', 1: 'available'}
        self.landing_pads_status = {0: 'available', 1: 'available'}
        self.takeoff_pad_cooldown_timers = {0: 0, 1: 0}
        self.landing_pad_cooldown_timers = {0: 0, 1: 0}
        self.landing_sequence_counter = 0

        for i in range(self.num_drones):
            task_type = random.choice(['takeoff', 'landing'])
            cooperative = random.choice([True, False])
            emergency = (i == 0)
            drone = UAV.random_uav(self.airspace_bounds, drone_id=i, task_type=task_type,
                                   cooperative=cooperative, environment=self, emergency=emergency)
            self.drones.append(drone)
        print(self.drones)
        self.time_step = 0
        return self._get_observation()

    def get_idle_takeoff_pads(self):
        """获取空闲的起飞场列表"""
        return [pad_id for pad_id, status in self.takeoff_pads_status.items() if status == 'available']

    def get_next_landing_sequence(self):
        """获取下一个降落排序序号"""
        self.landing_sequence_counter += 1
        return self.landing_sequence_counter

    def step(self, actions: List[Dict[str, np.ndarray]]):
        rewards = []
        dones = []
        infos = {}

        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                action = actions[i]
                drone.step(action)

        # 更新状态
        self.time_step += 1

        # 检查碰撞和任务完成
        self._update_collisions()
        self._update_done()

        # 计算奖励
        rewards = self._compute_rewards()
        # 获取观测
        obs = self._get_observation()
        # 检查是否结束
        done = self.time_step >= self.max_time_steps or len(self.done_drones) == self.num_drones
        self._update_pads_status()
        return obs, rewards, done, infos

    def _update_pads_status(self):
        """更新起飞场和降落场的状态和冷却计时器"""
        # 更新起飞场
        for pad_id, status in self.takeoff_pads_status.items():
            if status == 'occupied':
                self.takeoff_pad_cooldown_timers[pad_id] += 1
                if self.takeoff_pad_cooldown_timers[pad_id] >= self.takeoff_pad_cooldown:
                    self.takeoff_pads_status[pad_id] = 'available'
                    self.takeoff_pad_cooldown_timers[pad_id] = 0

        # 更新降落场
        for pad_id, status in self.landing_pads_status.items():
            if status == 'occupied':
                self.landing_pad_cooldown_timers[pad_id] += 1
                if self.landing_pad_cooldown_timers[pad_id] >= self.landing_pad_cooldown:
                    self.landing_pads_status[pad_id] = 'available'
                    self.landing_pad_cooldown_timers[pad_id] = 0

    def _compute_rewards(self):
        rewards = []
        for i, drone in enumerate(self.drones):
            if i in self.done_drones:
                rewards.append(0.0)
            else:
                reward = 0.0
                # 碰撞惩罚
                if drone.r_collision:
                    reward -= 100.0
                # 任务完成奖励
                if drone.r_task_completion:
                    reward += 100.0
                # 能量消耗惩罚
                reward -= drone.compute_energy_consumption(0, np.array([0.0, 0.0, 0.0]), 0.0) * 0.1
                rewards.append(reward)
        return rewards

    def _get_observation(self):
        obs = []
        for i, drone in enumerate(self.drones):
            if i in self.done_drones:
                obs.append(np.zeros(self.observation_space.shape))
            else:
                position = drone.position
                velocity = drone.components
                energy = np.array([drone.energy])
                neighbors = self._get_neighbors(drone)
                observation = np.concatenate([position, velocity, energy, neighbors])
                obs.append(observation)
        return obs

    def _get_neighbors(self, drone):
        closest_distance = float('inf')
        neighbor_info = np.zeros(3)
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < SENSING_RANGE and distance < closest_distance:
                    closest_distance = distance
                    relative_position = other_drone.position - drone.position
                    neighbor_info = relative_position
        return neighbor_info

    def _update_collisions(self):
        for drone in self.drones:
            drone.r_collision = 0

        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                for j, other_drone in enumerate(self.drones):
                    if j != i and j not in self.done_drones:
                        if drone.check_collision(other_drone):
                            drone.r_collision = 1
                            other_drone.r_collision = 1
                            self.done_drones.add(i)
                            self.done_drones.add(j)

    def _update_done(self):
        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                if drone.has_reached_target() or drone.energy <= 0:
                    drone.r_task_completion = 1
                    self.done_drones.add(i)

    def render(self, mode='human'):
        # 使用 matplotlib 的 3D 绘图功能
        if not hasattr(self, 'fig'):
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            plt.ion()
        else:
            self.ax.cla()

        self.ax.set_xlim(self.airspace_bounds[0], self.airspace_bounds[2])
        self.ax.set_ylim(self.airspace_bounds[1], self.airspace_bounds[3])
        self.ax.set_zlim(0, 100)
        self.ax.set_title('Urban UAV Environment')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Altitude')

        for drone in self.drones:
            if drone.drone_id not in self.done_drones:
                x, y, z = drone.position
                self.ax.scatter(x, y, z, c='blue', marker='o')
                # 绘制无人机的轨迹
                if not hasattr(drone, 'trajectory'):
                    drone.trajectory = [drone.position.copy()]
                else:
                    drone.trajectory.append(drone.position.copy())
                trajectory = np.array(drone.trajectory)
                self.ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='blue', linewidth=0.5)
                # 绘制目标位置
                tx, ty, tz = drone.target
                self.ax.scatter(tx, ty, tz, c='red', marker='x')

                # 绘制连线
                self.ax.plot([x, tx], [y, ty], [z, tz], 'k--', linewidth=0.5)

        plt.pause(0.001)

    def close(self):
        pass
