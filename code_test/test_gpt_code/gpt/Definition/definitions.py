# definitions.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
import random

# 常量和参数
UAV_TYPES = ['multirotor', 'light_hybrid_wing', 'medium_hybrid_wing', 'heavy_hybrid_wing']

# 不同类型无人机的最优速度（根据需求设定）
UAV_OPTIMAL_SPEED = {
    'multirotor': 10.0,
    'light_hybrid_wing': 15.0,
    'medium_hybrid_wing': 20.0,
    'heavy_hybrid_wing': 25.0,
}

MIN_SAFE_DISTANCE_MATRIX = {
    ('multirotor', 'multirotor'): 10.0,
    ('multirotor', 'light_hybrid_wing'): 15.0,
    ('multirotor', 'medium_hybrid_wing'): 20.0,
    ('multirotor', 'heavy_hybrid_wing'): 25.0,
    # ... （其他组合，您可以根据需要补充）
}

MAX_ACCELERATION = 5.0
MAX_TURN_RATE = 30.0  # 每秒最大转向角度（度）
MAX_CLIMB_RATE = 5.0  # 每秒最大爬升/下降速度
DT = 1.0

# 空域参数
R1 = 100.0   # 起飞/降落场和环的半径
R2 = 150.0   # 进近区域的半径（R2 > R1）
R3 = 300.0   # 外部空域的半径
H1 = 120.0   # 起飞环的高度
H2 = 80.0    # 降落环的高度
MAX_ALTITUDE = 1500.0  # 空域的最大高度

TAKEOFF_RING_CAPACITY = 5  # 起飞环容量
LANDING_RING_CAPACITY = 5  # 降落环容量

PAD_COOLDOWN_TIME = 10  # 停机坪冷却时间

@dataclass
class UAV:
    drone_id: int
    position: np.ndarray
    target: Optional[np.ndarray]
    optimal_speed: float
    task_type: str
    cooperative: bool
    type: str  # 无人机类型
    emergency: bool = False  # 是否为紧急无人机

    # 飞行参数
    speed: float = field(init=False)
    heading: np.ndarray = field(init=False)
    heading_angle: float = field(init=False)  # 航向角（度）
    altitude: float = field(init=False)
    energy: float = field(init=False)
    path_length: float = field(init=False)
    environment: Optional[object] = field(default=None, init=False)

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

    # 新增属性
    position_history: List[np.ndarray] = field(default_factory=list, init=False)
    entered_approach_time: Optional[int] = field(default=None, init=False)  # 进入进近空域的时间
    assigned_ring_entry_point: Optional[np.ndarray] = field(default=None, init=False)  # 分配的环入口点
    target_point: Optional[np.ndarray] = field(default=None, init=False)  # 最终目标点

    def __post_init__(self):
        self.speed = 0.0  # 初始化速度为0
        self.heading = np.array([0.0, 0.0, 0.0])
        self.heading_angle = 0.0
        self.altitude = self.position[2]
        self.energy = 100.0
        self.path_length = 0.0
        self.environment = None  # 将在环境中设置
        # 初始化位置历史
        self.position_history.append(self.position.copy())

    def calculate_direction(self) -> np.ndarray:
        direction = self.target - self.position
        norm = np.linalg.norm(direction[:2])
        return direction / norm if norm != 0 else np.array([0.0, 0.0, 0.0])

    def step(self, action: np.ndarray, other_drones: List['UAV']):
        # action 是一个连续的向量，例如 [delta_speed, delta_turn, delta_climb]
        delta_speed = action[0]
        delta_turn = action[1]
        delta_climb = action[2]

        # 更新无人机速度
        self.speed = np.clip(self.speed + delta_speed * DT, 0.0, UAV_OPTIMAL_SPEED[self.type] * 1.5)  # 限制最大速度为最优速度的1.5倍

        # 更新无人机航向角
        self.heading_angle += delta_turn * DT
        self.heading_angle = self.heading_angle % 360  # 保持在 [0, 360) 范围内

        # 更新无人机高度
        self.altitude = np.clip(self.altitude + delta_climb * DT, 0.0, MAX_ALTITUDE)

        # 根据新的航向和速度更新位置
        self.heading = np.array([
            np.cos(np.deg2rad(self.heading_angle)),
            np.sin(np.deg2rad(self.heading_angle)),
            0.0
        ])
        self.position += self.speed * self.heading * DT
        self.position[2] = self.altitude

        # 能量消耗和路径长度更新
        self.energy_consumption += self.compute_energy_consumption(delta_speed, delta_turn, delta_climb)
        self.path_length += self.speed * DT

        # 更新位置历史
        self.position_history.append(self.position.copy())

        # 检查是否完成任务
        self.check_task_status()

        # 检查违规
        self.check_violation()

    def compute_energy_consumption(self, delta_speed, delta_turn, delta_climb) -> float:
        return abs(delta_speed) + abs(delta_turn) + abs(delta_climb)

    def check_task_status(self):
        # 根据任务类型检查任务状态
        if self.task_type == 'takeoff':
            self.handle_takeoff_task()
        elif self.task_type == 'landing':
            self.handle_landing_task()

    def handle_takeoff_task(self):
        if not self.taking_off:
            # 检查是否可以垂直上升到起飞环高度
            if self.environment.can_ascend_to_takeoff_ring():
                self.taking_off = True
                self.target = self.position.copy()
                self.target[2] = self.environment.H1  # 起飞环高度
                self.heading = np.array([0.0, 0.0, 1.0])  # 垂直上升
                self.speed = UAV_OPTIMAL_SPEED[self.type]
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
            self.target = self.environment.approach_area_center()
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
                self.speed = UAV_OPTIMAL_SPEED[self.type] * 0.5
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
                        self.speed = UAV_OPTIMAL_SPEED[self.type]
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
        angle -= np.deg2rad(10.0) * DT  # 每秒转动10度
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
        distance = np.linalg.norm(self.position[:2] - self.environment.approach_area_center()[:2])
        return distance <= self.environment.approach_area_radius

    def has_reached_target(self) -> bool:
        distance = np.linalg.norm(self.position - self.target)
        return distance < 5.0  # 目标半径范围内视为到达

    def check_violation(self):
        # 检查是否违反空域限制等规则
        pass

    @classmethod
    def random_uav(cls, drone_id: int, task_type: str, environment):
        cooperative = random.choice([True, False])
        uav_type = random.choice(UAV_TYPES)
        optimal_speed = UAV_OPTIMAL_SPEED[uav_type]

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
            # 在可用的起飞区域内随机生成位置
            area = random.choice(environment.takeoff_areas)
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, area['radius'])
            x = area['center'][0] + radius * np.cos(angle)
            y = area['center'][1] + radius * np.sin(angle)
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
        else:
            # 不再创建巡航无人机
            return None
