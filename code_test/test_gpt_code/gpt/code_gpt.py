import gym
import numpy as np
import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from shapely.geometry import Point, Polygon, LineString
from gym import spaces

# 定义常量
MAX_DRONES = 20  # 最大无人机数量
MAX_SPEED = 15.0  # 最大速度，单位：m/s
MIN_SPEED = 5.0   # 最小速度，单位：m/s
MAX_ACCELERATION = 5.0  # 最大加速度，单位：m/s²
SENSING_RANGE = 50.0  # 感知范围，单位：m
DT = 1.0  # 时间步长，单位：s
NUM_MOVE = 5  # 每个动作的移动步数
MAX_EPISODE_LEN = 500  # 最大回合长度
NUM_DETECTION_SECTORS = 12  # 检测区域划分的扇区数
RANGE_DETECTION = 50.0  # 检测区域半径，单位：m
MIN_DISTANCE = 5.0  # 无人机之间的最小安全距离，单位：m

# 定义无人机类
@dataclass
class UAV:
    drone_id: int
    position: Point
    target: Point
    optimal_speed: float  # 最优速度，单位：m/s
    task_type: str  # 'takeoff' 或 'landing'
    cooperative: bool  # 是否为合作无人机
    emergency: bool = False  # 是否为紧急状态
    max_altitude: float = 100.0  # 最大高度，单位：m
    min_altitude: float = 0.0    # 最小高度，单位：m

    speed: float = field(init=False)
    heading: float = field(init=False)
    altitude: float = field(init=False)
    optimal_path_length: float = field(init=False)

    action: Optional[int] = field(init=False)
    intention_speed: float = field(init=False)
    intention_heading: float = field(init=False)
    intention_altitude: float = field(init=False)
    last_intention_speed: float = 0.0
    last_intention_heading: float = 0.0
    last_altitude: float = field(init=False)

    closest_distance: float = float('inf')
    path_length: float = 0.0
    energy: float = 100.0  # 能量水平（百分比）

    # 强化学习的奖励组件
    r_collision: int = 0
    r_task_completion: int = 0
    r_energy_efficiency: int = 0
    r_safety: int = 0

    def __post_init__(self) -> None:
        """
        初始化无人机的航向、速度和高度
        """
        self.heading = self.calculate_bearing()
        self.speed = self.optimal_speed
        if self.task_type == 'takeoff':
            self.altitude = self.min_altitude  # 起飞任务从最低高度开始
        elif self.task_type == 'landing':
            self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        else:
            self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        self.last_altitude = self.altitude
        self.optimal_path_length = self.position.distance(self.target)

    def calculate_bearing(self) -> float:
        """
        计算当前位姿到目标的航向角
        """
        dx = self.target.x - self.position.x
        dy = self.target.y - self.position.y
        bearing = math.atan2(dy, dx)
        return bearing % (2 * math.pi)

    def predict_position(self, dt: float = DT) -> Point:
        """
        预测在 dt 秒后的未来位置，保持当前速度和航向
        """
        dx = self.speed * math.cos(self.heading) * dt
        dy = self.speed * math.sin(self.heading) * dt
        return Point(self.position.x + dx, self.position.y + dy)

    @property
    def components(self) -> Tuple[float, float]:
        """
        更新速度在 X 和 Y 方向的分量
        """
        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        return vx, vy

    def distance_to_target(self) -> float:
        """
        计算到目标的当前距离
        """
        return self.position.distance(self.target)

    def heading_drift(self) -> float:
        """
        计算当前航向与目标航向之间的偏差角度
        """
        bearing = self.calculate_bearing()
        drift = bearing - self.heading
        if drift > math.pi:
            drift -= 2 * math.pi
        elif drift < -math.pi:
            drift += 2 * math.pi
        return drift

    @classmethod
    def random_uav(cls, airspace_polygon: Polygon, drone_id: int, task_type: str,
                   cooperative: bool, emergency: bool = False):
        """
        在指定空域内创建一个随机无人机
        """
        def random_point_in_polygon(polygon: Polygon) -> Point:
            minx, miny, maxx, maxy = polygon.bounds
            while True:
                point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
                if polygon.contains(point):
                    return point

        # 随机起始位置
        position = random_point_in_polygon(airspace_polygon)

        # 随机目标位置（在边界上）
        boundary = airspace_polygon.boundary
        while True:
            d = random.uniform(0, boundary.length)
            target = boundary.interpolate(d)
            if target.distance(position) > 10.0:  # 确保目标位置不太接近起始位置
                break

        # 随机最优速度
        optimal_speed = random.uniform(MIN_SPEED, MAX_SPEED)

        return cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                   task_type=task_type, cooperative=cooperative, emergency=emergency)

    def step(self, action: dict):
        """
        根据给定的动作更新无人机的状态
        """
        # 动作包含对速度、航向和高度的改变
        delta_speed = action.get('delta_speed', 0.0)
        delta_heading = action.get('delta_heading', 0.0)
        delta_altitude = action.get('delta_altitude', 0.0)

        # 更新速度、航向和高度，确保在允许的范围内
        self.speed = np.clip(self.speed + delta_speed, MIN_SPEED, MAX_SPEED)
        self.heading = (self.heading + delta_heading) % (2 * math.pi)
        self.altitude = np.clip(self.altitude + delta_altitude, self.min_altitude, self.max_altitude)

        # 更新位置
        dt = DT  # 时间步长，单位：秒
        dx = self.speed * math.cos(self.heading) * dt
        dy = self.speed * math.sin(self.heading) * dt
        self.position = Point(self.position.x + dx, self.position.y + dy)

        # 更新能量消耗
        self.energy -= self.compute_energy_consumption(delta_speed, delta_heading, delta_altitude)

        # 更新路径长度
        self.path_length += math.hypot(dx, dy)

    def compute_energy_consumption(self, delta_speed, delta_heading, delta_altitude):
        """
        根据速度、航向和高度的变化计算能量消耗
        """
        # 简单模型：能量消耗与速度和高度成正比
        energy = abs(delta_speed) * 0.1 + abs(delta_heading) * 0.05 + abs(delta_altitude) * 0.2
        energy += self.speed * 0.01  # 与速度成正比
        return energy

    def check_collision(self, other_uav, collision_distance=MIN_DISTANCE):
        """
        检查是否与另一架无人机发生碰撞
        """
        if self.position.distance(other_uav.position) < collision_distance and abs(self.altitude - other_uav.altitude) < 5.0:
            return True
        else:
            return False

    def has_reached_target(self, tolerance=5.0):
        """
        检查无人机是否到达目标位置
        """
        if self.distance_to_target() <= tolerance:
            return True
        else:
            return False

# 定义环境类
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

    def step(self, actions: List[dict]):
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
        # 简化的渲染方法，可以使用 matplotlib 或其他可视化工具实现
        pass

    def close(self):
        pass
