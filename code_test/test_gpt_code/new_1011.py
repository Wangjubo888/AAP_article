import numpy as np
import random
import gym
from gym import spaces

# 定义全局参数
MAX_DRONES = 20
MAX_VELOCITY = 10.0  # 最大速度，单位：m/s
MAX_ACCELERATION = 5.0  # 最大加速度，单位：m/s²
SENSING_RANGE = 50.0  # 感知范围，单位：m
DT = 0.1  # 时间步长，单位：s


# 定义无人机类
class Drone:
    def __init__(self, drone_id, task_type, cooperative, emergency=False):
        self.drone_id = drone_id
        self.task_type = task_type  # 'takeoff' or 'landing'
        self.cooperative = cooperative
        self.emergency = emergency
        self.position = np.array([0.0, 0.0, 0.0])  # 位置
        self.velocity = np.array([0.0, 0.0, 0.0])  # 速度
        self.destination = self._generate_destination()
        self.energy = 100.0  # 能量水平

    def _generate_destination(self):
        # 生成目的地，根据任务类型
        if self.task_type == 'takeoff':
            return np.array([random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(50, 100)])
        elif self.task_type == 'landing':
            return np.array([0.0, 0.0, 0.0])

    def step(self, action):
        # 动作为加速度控制
        acceleration = np.clip(action, -MAX_ACCELERATION, MAX_ACCELERATION)

        # 更新速度，考虑速度限制
        self.velocity += acceleration * DT
        self.velocity = np.clip(self.velocity, -MAX_VELOCITY, MAX_VELOCITY)

        # 更新位置
        self.position += self.velocity * DT

        # 能量消耗，假设与加速度大小相关
        self.energy -= np.linalg.norm(acceleration) * DT

    def get_state(self):
        return self.position, self.velocity


# 定义环境类
class MultiAgentEnv(gym.Env):
    def __init__(self):
        super(MultiAgentEnv, self).__init__()
        self.drones = []
        self.time_step = 0
        self.max_time_steps = 1000

        # 动作空间：加速度控制，连续空间
        self.action_space = spaces.Box(low=-MAX_ACCELERATION, high=MAX_ACCELERATION, shape=(3,), dtype=np.float32)

        # 观测空间：自身状态 + 邻居信息 + 目的地
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self):
        self.drones = []
        for i in range(MAX_DRONES):
            task_type = random.choice(['takeoff', 'landing'])
            cooperative = random.choice([True, False])
            emergency = (i == 0)  # 假设第一个无人机为紧急状态
            drone = Drone(drone_id=i, task_type=task_type, cooperative=cooperative, emergency=emergency)
            self.drones.append(drone)
        self.time_step = 0
        return self._get_observation()

    def step(self, actions):
        rewards = []
        dones = []
        infos = {}
        for i, drone in enumerate(self.drones):
            action = actions[i]
            reward, done, info = self._take_action(drone, action)
            rewards.append(reward)
            dones.append(done)
        self.time_step += 1
        obs = self._get_observation()
        return obs, rewards, dones, infos

    def _take_action(self, drone, action):
        # 执行动作并计算奖励
        drone.step(action)

        reward = 0.0
        done = False
        info = {}

        # 检查碰撞
        if self._check_collision(drone):
            reward -= 100.0  # 碰撞惩罚
            done = True

        # 检查任务完成
        if self._check_task_completion(drone):
            reward += 100.0  # 任务完成奖励
            done = True

        # 考虑能量消耗
        if drone.energy <= 0:
            done = True

        # 紧急状态奖励
        if drone.emergency:
            reward += 50.0

        return reward, done, info

    def _get_observation(self):
        obs = []
        for drone in self.drones:
            obs.append(self._observe_drone(drone))
        return obs

    def _observe_drone(self, drone):
        # 自身状态
        position = drone.position
        velocity = drone.velocity

        # 目的地
        destination = drone.destination

        # 邻居信息
        neighbors = self._get_neighbors(drone)

        # 将信息拼接为固定长度的观测向量
        observation = np.concatenate([position, velocity, destination, neighbors])
        return observation

    def _get_neighbors(self, drone):
        # 获取感知范围内的邻居无人机的位置
        neighbors = []
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < SENSING_RANGE:
                    relative_position = other_drone.position - drone.position
                    neighbors.append(relative_position)
        # 填充或截断邻居信息
        max_neighbors = 1  # 简化为只考虑最近的邻居
        if len(neighbors) == 0:
            neighbors.append(np.array([0.0, 0.0, 0.0]))
        neighbors = neighbors[:max_neighbors]
        return neighbors[0]

    def _check_collision(self, drone):
        # 检查是否与其他无人机碰撞
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                if distance < 2.0:  # 碰撞距离阈值
                    return True
        return False

    def _check_task_completion(self, drone):
        # 检查是否到达目的地
        if np.linalg.norm(drone.position - drone.destination) < 5.0:
            return True
        return False
