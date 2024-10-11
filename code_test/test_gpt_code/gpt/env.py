import gym
import numpy as np
import random
import math
from typing import List, Dict
from shapely.geometry import Point, Polygon, LineString
from gym import spaces
from definitions import UAV, MAX_DRONES, MIN_DISTANCE, SENSING_RANGE, MAX_EPISODE_LEN, MAX_ACCELERATION, DT, NUM_MOVE
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.animation as animation


class UrbanUAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UrbanUAVEnv, self).__init__()
        self.num_drones = MAX_DRONES
        self.drones = []  # 无人机列表
        self.time_step = 0
        self.max_time_steps = MAX_EPISODE_LEN
        self.num_move = NUM_MOVE

        # 定义动作空间和观测空间
        # 动作空间：delta_speed, delta_heading, delta_altitude
        self.action_space = spaces.Box(low=np.array([-MAX_ACCELERATION, -math.pi / 4, -5.0]),
                                       high=np.array([MAX_ACCELERATION, math.pi / 4, 5.0]),
                                       dtype=np.float32)

        # 观测空间：自身状态 + 邻居信息
        # 自身状态：位置（x, y, z），速度（vx, vy, vz），剩余能量
        # 邻居信息：相对位置和速度
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # 创建空域，多边形表示
        self.airspace = self.create_airspace()

        # 记录完成的无人机
        self.done_drones = set()

        # 初始化
        self.reset()

    def create_airspace(self) -> Polygon:
        # 定义一个简单的矩形空域，您可以根据需要进行修改
        minx, miny = -200.0, -200.0
        maxx, maxy = 200.0, 200.0
        airspace_polygon = Polygon([(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny)])
        return airspace_polygon

    def reset(self):
        self.drones = []
        self.done_drones = set()
        for i in range(self.num_drones):
            task_type = random.choice(['takeoff', 'landing'])
            cooperative = random.choice([True, False])
            emergency = (i == 0)  # 假设第一个无人机为紧急状态
            drone = UAV.random_uav(self.airspace, drone_id=i, task_type=task_type,
                                   cooperative=cooperative, emergency=emergency)
            self.drones.append(drone)
        self.time_step = 0
        return self._get_observation()

    def step(self, actions: List[Dict[str, float]]):
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

        return obs, rewards, done, infos

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
                reward -= drone.compute_energy_consumption(0, 0, 0) * 0.1
                rewards.append(reward)
        return rewards

    def _get_observation(self):
        obs = []
        for i, drone in enumerate(self.drones):
            if i in self.done_drones:
                obs.append(np.zeros(self.observation_space.shape))
            else:
                # 自身状态
                position = np.array([drone.position.x, drone.position.y, drone.altitude])
                velocity = np.array(drone.components + (0.0,))  # 暂时忽略垂直速度
                energy = np.array([drone.energy])
                # 邻居信息
                neighbors = self._get_neighbors(drone)
                observation = np.concatenate([position, velocity, energy, neighbors])
                obs.append(observation)
        return obs

    def _get_neighbors(self, drone):
        # 简化处理，仅考虑最近的一个邻居
        closest_distance = float('inf')
        neighbor_info = np.zeros(3)
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = drone.position.distance(other_drone.position)
                if distance < SENSING_RANGE and distance < closest_distance:
                    closest_distance = distance
                    relative_position = np.array([other_drone.position.x - drone.position.x,
                                                  other_drone.position.y - drone.position.y,
                                                  other_drone.altitude - drone.altitude])
                    neighbor_info = relative_position
        return neighbor_info

    def _update_collisions(self):
        # 重置碰撞状态
        for drone in self.drones:
            drone.r_collision = 0

        # 检查碰撞
        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                for j, other_drone in enumerate(self.drones):
                    if j != i and j not in self.done_drones:
                        if drone.check_collision(other_drone):
                            drone.r_collision = 1
                            other_drone.r_collision = 1
                            # 标记为完成，停止移动
                            self.done_drones.add(i)
                            self.done_drones.add(j)

    def _update_done(self):
        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                if drone.has_reached_target() or drone.energy <= 0:
                    drone.r_task_completion = 1
                    self.done_drones.add(i)

    def render(self, mode='human'):
        # 使用 matplotlib 可视化无人机的位置
        plt.clf()
        plt.xlim(-250, 250)
        plt.ylim(-250, 250)
        plt.title('Urban UAV Environment')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')

        # 绘制无人机位置和目标
        for drone in self.drones:
            if drone.drone_id not in self.done_drones:
                x, y = drone.position.x, drone.position.y
                plt.scatter(x, y, c='blue', marker='o')
                plt.text(x, y, f'ID:{drone.drone_id}')
                # 绘制目标位置
                tx, ty = drone.target.x, drone.target.y
                plt.scatter(tx, ty, c='red', marker='x')

                # 绘制连线
                plt.plot([x, tx], [y, ty], 'k--', linewidth=0.5)

        plt.pause(0.001)

    def close(self):
        pass
