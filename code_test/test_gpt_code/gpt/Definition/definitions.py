# definitions.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import random
import gym
import math
from scipy.stats import burr

# 无人机类型
ALL_UAV_TYPES = ['multirotor', 'light_hybrid_wing', 'medium_hybrid_wing', 'heavy_hybrid_wing']

# 不同类型无人机的最优速度（单位：m/s）
ALL_UAV_OPTIMAL_SPEED = {
    'multirotor': 10.0,
    'light_hybrid_wing': 15.0,
    'medium_hybrid_wing': 20.0,
    'heavy_hybrid_wing': 25.0,
}
# 各无人机类型的性能参数
ALL_UAV_PERFORMANCE = {
    'multirotor': {
        'max_acceleration': 5.0,
        'max_turn_rate': 45.0,    # 度/秒
        'max_climb_rate': 5.0,
    },
    'light_hybrid_wing': {
        'max_acceleration': 6.0,
        'max_turn_rate': 40.0,
        'max_climb_rate': 6.0,
    },
    'medium_hybrid_wing': {
        'max_acceleration': 7.0,
        'max_turn_rate': 35.0,
        'max_climb_rate': 7.0,
    },
    'heavy_hybrid_wing': {
        'max_acceleration': 8.0,
        'max_turn_rate': 30.0,
        'max_climb_rate': 8.0,
    }
}

# 不同类型无人机之间的最小安全距离矩阵（单位：米）
MIN_SAFE_DISTANCE_MATRIX = {
    ('multirotor', 'multirotor'): 10.0,
    ('multirotor', 'light_hybrid_wing'): 15.0,
    ('multirotor', 'medium_hybrid_wing'): 20.0,
    ('multirotor', 'heavy_hybrid_wing'): 25.0,
    ('light_hybrid_wing', 'light_hybrid_wing'): 15.0,
    ('light_hybrid_wing', 'medium_hybrid_wing'): 20.0,
    ('light_hybrid_wing', 'heavy_hybrid_wing'): 25.0,
    ('medium_hybrid_wing', 'medium_hybrid_wing'): 20.0,
    ('medium_hybrid_wing', 'heavy_hybrid_wing'): 25.0,
    ('heavy_hybrid_wing', 'heavy_hybrid_wing'): 25.0,
    # 对称填充矩阵
}
# 填充对称的矩阵值
for (type1, type2), distance in list(MIN_SAFE_DISTANCE_MATRIX.items()):
    if (type2, type1) not in MIN_SAFE_DISTANCE_MATRIX:
        MIN_SAFE_DISTANCE_MATRIX[(type2, type1)] = distance

# 空域参数
R1 = 100.0    # 起飞/降落环的半径（米）
R2 = 1000.0    # 进近区域的半径（米），R2 > R1
R3 = 5000.0    # 外部空域的半径（米）
H1 = 150.0    # 起飞环的高度（米）
H2 = 100.0     # 降落环的高度（米）
MAX_ALTITUDE = 1500.0   # 空域的最大高度（米）
DT = 1
TAKEOFF_RING_CAPACITY = 5   # 起飞环容量（同时容纳的无人机数量）
LANDING_RING_CAPACITY = 5   # 降落环容量

PAD_COOLDOWN_TIME = 10  # 停机坪冷却时间（秒）


@dataclass
class UAV:
    drone_id: int
    optimal_speed: float
    task_type: str
    cooperative: bool
    type: str  # 无人机类型
    emergency: bool = False  # 是否为紧急无人机
    position: np.ndarray = field(init=False)
    target: Optional[np.ndarray] = field(default=None)  # 目标位置

    # 飞行参数
    speed: float = field(init=False)
    heading: np.ndarray = field(init=False)
    heading_angle: float = field(init=False)  # 航向角（度）
    altitude: float = field(init=False)
    energy: float = field(init=False)
    path_length: float = field(init=False)
    environment: Optional[object] = field(default=None, init=False)
    # 性能参数
    max_acceleration: float = field(init=False)
    max_turn_rate: float = field(init=False)
    max_climb_rate: float = field(init=False)
    # 任务状态
    sequence_number: Optional[int] = field(default=None, init=False)
    in_takeoff_ring: bool = field(default=False, init=False)
    in_landing_ring: bool = field(default=False, init=False)
    assigned_pad: Optional[int] = field(default=None, init=False)
    taking_off: bool = field(default=False, init=False)
    landing: bool = field(default=False, init=False)
    task_finished: bool = field(default=False, init=False)
    r_collision: int = field(default=0, init=False)
    r_task_completion: int = field(default=0, init=False)
    violation: bool = field(default=False, init=False)
    avoided_collision: bool = field(default=False, init=False)
    energy_consumption: float = field(default=0.0, init=False)
    priority: int = field(default=0, init=False)  # 避让优先级，数值越高优先级越高
    position_history: List[np.ndarray] = field(default_factory=list, init=False)
    entered_approach_time: Optional[int] = field(default=None, init=False)  # 进入进近空域的时间
    assigned_ring_entry_point: Optional[np.ndarray] = field(default=None, init=False)  # 分配的环入口点
    target_point: Optional[np.ndarray] = field(default=None, init=False)  # 最终目标点
    action_space: gym.spaces.Box = field(init=False)
    heading_error_scale = 2.0
    airspeed_error_scale = 1.0
    max_wind_speed = 10.0  # 最大风速 10 m/s（适用于低空飞行环境）
    wind_speed_range = 2.0  # 风速波动范围 ±2 m/s
    reference_wind_heading = random.uniform(0, 2 * math.pi)  # 随机生成风向，单位：弧度 [0, 2π]
    reference_wind_speed = np.clip(burr.rvs(c=4.089, d=0.814, loc=-0.042, size=1, scale=17.47),
                                   wind_speed_range, max_wind_speed - wind_speed_range)  # 风速的生成
    wait_time: int = field(default=0)
    permission: str = field(default="unrestricted")  # 初始权限为无限制

    def __post_init__(self):
        # 设置初始位置
        self.position = np.zeros(3) if self.position is None else self.position
        self.altitude = self.position[2].item()
        self.energy = 100.0  # 初始能量
        self.path_length = 0.0
        self.environment = None
        self.position_history.append(self.position.copy())

        # 选择无人机类型
        uav_type = self.type
        self.optimal_speed = ALL_UAV_OPTIMAL_SPEED[uav_type]
        performance = ALL_UAV_PERFORMANCE[uav_type]
        self.max_acceleration = performance['max_acceleration']
        self.max_turn_rate = performance['max_turn_rate']
        self.max_climb_rate = performance['max_climb_rate']

        # 根据任务类型设置初始位置和目标位置
        if self.task_type == 'takeoff':
            self.position = np.array([0, 0, H1])  # 起飞环中心
            theta_target = np.random.uniform(0, 2 * np.pi)
            self.target = np.array(
                [R3 * np.cos(theta_target), R3 * np.sin(theta_target), np.random.uniform(0, MAX_ALTITUDE)])
            self.speed = 0.0
            self.heading = self.calculate_direction()
            self.heading_angle = self.bearing
        elif self.task_type == 'landing':
            theta = np.random.uniform(0, 2 * np.pi)
            self.position = np.array([R3 * np.cos(theta), R3 * np.sin(theta), np.random.uniform(H2, MAX_ALTITUDE)])
            self.target = np.array([0.0, 0.0, H2])  # 降落环中心
            self.speed = self.optimal_speed
            self.heading = self.calculate_direction()
            self.heading_angle = self.bearing

        # 初始化动作空间
        self.action_space = gym.spaces.Box(
            low=np.array([-self.max_acceleration * DT, -self.max_turn_rate * DT, -self.max_climb_rate * DT]),
            high=np.array([self.max_acceleration * DT, self.max_turn_rate * DT, self.max_climb_rate * DT]),
            dtype=np.float32,
            shape=(3,)
        )

    def calculate_direction(self) -> np.ndarray:
        direction = self.target - self.position
        norm = np.linalg.norm(direction[:2])
        return direction / norm if norm != 0 else np.array([0.0, 0.0, 0.0])

    @property
    def bearing(self) -> float:
        """
        Bearing from current position to target
        :return:
        """
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        compass = math.atan2(dy, dx)
        return (compass + 2 * math.pi) % math.pi

    def step(self, action: np.ndarray, other_drones: List['UAV']):

        # Uncertainty: Adding random heading error (航向误差)
        heading_error = np.random.normal(0, self.heading_error_scale)  # 生成航向误差
        speed_error = np.random.normal(0, self.airspeed_error_scale)  # 生成速度误差

        # 限制速度更新
        delta_speed = action[0]  # 控制加速度
        self.speed = np.clip(self.speed + delta_speed + speed_error, 0.0,
                             self.optimal_speed * 1.5).item()  # 更新速度并限制最大速度

        # 更新航向角，并限制最大转向角
        delta_turn = np.clip(action[1] + heading_error, -self.max_turn_rate * DT, self.max_turn_rate * DT)  # 限制航向变化速率
        self.heading_angle += delta_turn
        self.heading_angle = self.heading_angle % 360  # 保持航向角在 [0, 360) 范围内

        # 更新高度
        delta_climb = np.clip(action[2], -self.max_climb_rate * DT, self.max_climb_rate * DT)  # 限制爬升速率
        self.altitude = np.clip(self.altitude + delta_climb, 0.0, MAX_ALTITUDE).item()  # 更新高度并限制最大高度

        # 计算新的航向
        self.heading = np.array([
            np.cos(np.deg2rad(self.heading_angle)),
            np.sin(np.deg2rad(self.heading_angle)),
            0.0
        ])

        # **风速对位置的影响**（风速误差）
        wind_heading = self.reference_wind_heading
        wind_speed = self.reference_wind_speed + np.clip(np.random.normal(0, 1), -1,
                                                         1) * self.wind_speed_range  # 生成风速误差
        wind_dx = math.cos(wind_heading) * wind_speed
        wind_dy = math.sin(wind_heading) * wind_speed

        # 更新位置：将风速和无人机速度一起考虑
        self.position += (self.speed * self.heading[:2] + np.array([wind_dx, wind_dy])) * DT  # 风速对位置的影响
        self.position[2] = self.altitude  # 更新高度

        # 能量消耗更新，调用能量消耗计算函数
        self.energy_consumption += self.compute_energy_consumption(delta_speed, delta_turn, delta_climb)

        # 更新路径长度
        self.path_length += np.linalg.norm(self.speed * self.heading[:2] * DT + np.array([wind_dx, wind_dy]) * DT)

        # 更新位置历史
        self.position_history.append(self.position.copy())

        # 检查任务状态（如是否完成任务）
        self.check_task_status()

        # 检查是否有违规行为（如进入禁止区域、与其他无人机发生冲突等）
        self.check_violation()

        # 碰撞检测和避免（如果有多个无人机的环境）
        self.avoid_collisions(other_drones)

        # 返回新的状态
        return self.position, self.speed, self.heading_angle, self.energy_consumption, self.path_length, self.task_finished

    def check_violation(self):
        # 检查是否超出空域边界
        if self.environment and hasattr(self.environment, 'airspace_radius'):
            if np.linalg.norm(self.position[:2]) > self.environment.airspace_radius:
                self.violation = True

    def compute_energy_consumption(self, delta_speed, delta_heading, delta_altitude):
        """
        根据速度、航向和高度的变化计算能量消耗
        """
        energy = abs(delta_speed) * 0.1 + np.linalg.norm(delta_heading) * 0.05 + abs(delta_altitude) * 0.2
        energy += self.speed * 0.01
        return energy

    def check_task_status(self):
        if self.task_type == 'takeoff':
            # 判断是否离开空域
            if np.linalg.norm(self.position[:2]) > self.environment.airspace_radius:
                self.task_finished = True
                # 起飞环进入冷却
                self.environment.takeoff_ring_available = False
                self.environment.takeoff_cooldown = PAD_COOLDOWN_TIME

        elif self.task_type == 'landing':
            # 判断是否到达降落环的 XY 平面区域
            distance_to_target_xy = np.linalg.norm(self.position[:2] - self.target[:2])
            if distance_to_target_xy < 5.0:  # 允许的误差范围
                self.task_finished = True
                # 降落环进入冷却
                self.environment.landing_ring_available = False
                self.environment.landing_cooldown = PAD_COOLDOWN_TIME

    def handle_takeoff_task(self):
        if not self.taking_off:
            # 检查是否可以垂直上升到起飞环高度
            if self.environment.can_ascend_to_takeoff_ring():
                self.taking_off = True
                self.target = self.position.copy()
                self.target[2] = self.environment.H1  # 起飞环高度
                self.heading = np.array([0.0, 0.0, 1.0])  # 垂直上升
                self.speed = ALL_UAV_OPTIMAL_SPEED[self.type]
            else:
                # 停留在原地等待
                self.speed = 0.0
        elif not self.in_takeoff_ring:
            # 检查是否到达起飞环高度
            if abs(self.altitude - self.environment.H1) < 1.0:
                self.in_takeoff_ring = True
                self.environment.enter_takeoff_ring(self)
                self.assign_ring_entry_point()
            else:
                # 继续垂直上升
                self.heading = np.array([0.0, 0.0, 1.0])  # 垂直上升
        else:
            # 在起飞环上飞行
            if self.reach_exit_point():
                # 计算飞向目标点的航向角
                self.target = self.target_point
                self.update_heading_towards_target()
            else:
                # 沿起飞环飞行
                self.fly_along_ring(self.environment.takeoff_ring_center, self.environment.R1)

            # 检查是否到达目标点
            if self.has_reached_target():
                self.task_finished = True
                self.r_task_completion = 1

    def handle_landing_task(self):
        if not self.entered_approach_time:
            # 向进近空域飞行
            self.target = self.environment.get_approach_area_center()
            self.update_heading_towards_target()
            if self.in_approach_area():
                self.entered_approach_time = self.environment.time_step
                self.environment.add_to_approach_queue(self)
        elif not self.in_landing_ring:
            # 检查是否可以进入降落环
            if self.environment.can_enter_landing_ring(self):
                self.in_landing_ring = True
                self.environment.enter_landing_ring(self)
                self.assign_ring_entry_point()
            else:
                # 在进近空域等待或缓慢靠近
                self.speed = ALL_UAV_OPTIMAL_SPEED[self.type] * 0.5
                self.update_heading_towards_target()
        else:
            # 在降落环上飞行
            if self.environment.can_land(self):
                # 飞向降落区域内的目标点
                self.target = self.environment.get_landing_point(self)
                self.update_heading_towards_target()
                if np.linalg.norm(self.position[:2] - self.target[:2]) < 5.0:
                    # 垂直下降
                    if self.altitude > 0.5:
                        self.heading = np.array([0.0, 0.0, -1.0])
                        self.speed = ALL_UAV_OPTIMAL_SPEED[self.type]
                    else:
                        self.task_finished = True
                        self.r_task_completion = 1
                        self.environment.finish_landing(self)
            else:
                # 沿降落环飞行
                self.fly_along_ring(self.environment.landing_ring_center, self.environment.R1)

    def update_heading_towards_target(self):
        direction = self.target - self.position
        norm = np.linalg.norm(direction)
        if norm != 0:
            self.heading = direction / norm
            self.heading_angle = np.rad2deg(np.arctan2(self.heading[1], self.heading[0]))

    def fly_along_ring(self, center, radius):
        # 沿环飞行，顺时针方向
        relative_pos = self.position - center
        angle = np.arctan2(relative_pos[1], relative_pos[0])
        angle -= np.deg2rad(10.0) * DT  # 每秒转动 10 度
        new_x = center[0] + radius * np.cos(angle)
        new_y = center[1] + radius * np.sin(angle)
        self.target = np.array([new_x, new_y, self.altitude])
        self.update_heading_towards_target()

    def reach_exit_point(self):
        # 检查是否到达起飞环的出口点
        return np.linalg.norm(self.position - self.assigned_ring_entry_point) < 5.0

    def assign_ring_entry_point(self):
        # 根据环内无人机数量，分配进入环的点
        if self.task_type == 'takeoff':
            num_uavs_in_ring = len(self.environment.takeoff_ring_uavs)
            capacity = TAKEOFF_RING_CAPACITY
            center = self.environment.takeoff_ring_center
            altitude = self.environment.H1
        else:
            num_uavs_in_ring = len(self.environment.landing_ring_uavs)
            capacity = LANDING_RING_CAPACITY
            center = self.environment.landing_ring_center
            altitude = self.environment.H2
        angle = (2 * np.pi / capacity) * num_uavs_in_ring
        radius = self.environment.R1
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        self.assigned_ring_entry_point = np.array([x, y, altitude])
        self.target = self.assigned_ring_entry_point
        self.update_heading_towards_target()

        if self.task_type == 'takeoff':
            # 计算起飞后目标点
            theta = np.random.uniform(0, 2 * np.pi)
            x_outer = self.environment.airspace_radius * np.cos(theta)
            y_outer = self.environment.airspace_radius * np.sin(theta)
            z_outer = np.random.uniform(0, MAX_ALTITUDE)
            self.target_point = np.array([x_outer, y_outer, z_outer])

    def in_approach_area(self):
        distance_to_app = np.linalg.norm(self.position[:2] - self.environment.get_approach_area_center()[:2])
        return distance_to_app <= self.environment.approach_area_radius

    def has_reached_target(self, tolerance=3.0):
        """
        检查无人机是否到达目标位置
        """
        if np.linalg.norm(self.position - self.target) <= tolerance:
            return True
        else:
            return False

    @classmethod
    def random_uav(cls, drone_id: int, task_type: str, environment):
        cooperative = random.choice([True, False])
        uav_type = random.choice(environment.UAV_TYPES)
        optimal_speed = ALL_UAV_OPTIMAL_SPEED[uav_type]

        # 确定是否为紧急无人机
        if environment.has_emergency_uav():
            emergency = False
        else:
            emergency = random.choice([True, False])  # 随机决定是否为紧急无人机
            if emergency:
                environment.set_emergency_uav_exists(True)
            else:
                emergency = False

        if task_type == 'takeoff':
            # 在起飞停机坪中心生成位置
            pad = random.choice(environment.takeoff_pads)
            x = pad['center'][0]
            y = pad['center'][1]
            position = np.array([x, y, 0.0])
            drone = cls(drone_id=drone_id, position=position, target=None, optimal_speed=optimal_speed,
                        task_type=task_type, cooperative=cooperative, type=uav_type, emergency=emergency)
            drone.environment = environment
            return drone
        elif task_type == 'landing':
            # 从空域外部侧面随机生成
            theta = random.uniform(0, 2 * np.pi)
            x = environment.airspace_radius * np.cos(theta)
            y = environment.airspace_radius * np.sin(theta)
            z = random.uniform(0, MAX_ALTITUDE)
            position = np.array([x, y, z])
            drone = cls(drone_id=drone_id, position=position, target=None, optimal_speed=optimal_speed,
                        task_type=task_type, cooperative=cooperative, type=uav_type, emergency=emergency)
            drone.environment = environment
            return drone
