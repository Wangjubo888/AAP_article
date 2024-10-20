# definitions.py

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import random
import matplotlib.pyplot as plt

# 常量和参数
UAV_TYPES = ['multirotor', 'light_hybrid_wing', 'medium_hybrid_wing', 'heavy_hybrid_wing']

MIN_SAFE_DISTANCE_MATRIX = {
    ('multirotor', 'multirotor'): 10.0,
    ('multirotor', 'light_hybrid_wing'): 15.0,
    ('multirotor', 'medium_hybrid_wing'): 20.0,
    ('multirotor', 'heavy_hybrid_wing'): 25.0,
    # ... （其他组合，您可以根据需要补充）
}

MAX_SPEED = 80.0
MIN_SPEED = 10.0
Cruise_SPEED = 10.0
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

@dataclass
class UAV:
    drone_id: int
    position: np.ndarray
    target: np.ndarray
    optimal_speed: float
    task_type: str
    cooperative: bool
    type: str  # 无人机类型
    emergency: bool = False

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

    def __post_init__(self):
        self.speed = self.optimal_speed
        self.heading = self.calculate_direction()
        self.heading_angle = np.rad2deg(np.arctan2(self.heading[1], self.heading[0]))
        self.altitude = self.position[2]
        self.energy = 100.0
        self.path_length = 0.0
        self.environment = None  # 将在环境中设置
        # 初始化位置历史
        self.position_history.append(self.position.copy())

    def calculate_direction(self) -> np.ndarray:
        direction = self.target - self.position
        norm = np.linalg.norm(direction)
        return direction / norm if norm != 0 else np.array([0.0, 0.0, 0.0])

    def step(self, action: np.ndarray, other_drones: List['UAV']):
        # action 是一个连续的向量，例如 [delta_speed, delta_turn, delta_climb]
        delta_speed = action[0]
        delta_turn = action[1]
        delta_climb = action[2]

        # 更新无人机速度
        self.speed = np.clip(self.speed + delta_speed * DT, MIN_SPEED, MAX_SPEED)

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
        if self.has_reached_target():
            self.task_finished = True
            self.r_task_completion = 1

        # 检查避让
        if self.cooperative:
            self.avoid_collision(other_drones)

        # 检查违规
        self.check_violation()

    def avoid_collision(self, other_drones: List['UAV']):
        for other in other_drones:
            if other.drone_id != self.drone_id:
                distance_vector = self.position - other.position
                distance = np.linalg.norm(distance_vector)
                min_distance = MIN_SAFE_DISTANCE_MATRIX.get((self.type, other.type), 10.0)
                if distance < min_distance * 1.5:
                    if self.priority < other.priority:
                        self.perform_avoidance_maneuver(distance_vector)
                    else:
                        continue

    def perform_avoidance_maneuver(self, distance_vector):
        avoidance_direction = distance_vector / (np.linalg.norm(distance_vector) + 1e-6)
        # 调整航向角基于避让方向
        angle_adjustment = np.rad2deg(np.arctan2(avoidance_direction[1], avoidance_direction[0]))
        self.heading_angle += angle_adjustment
        self.heading_angle = self.heading_angle % 360

        # 调整高度（如果需要）
        if abs(distance_vector[2]) < 10.0:
            self.altitude += np.sign(avoidance_direction[2]) * MAX_CLIMB_RATE * DT
            self.altitude = np.clip(self.altitude, 0.0, MAX_ALTITUDE)
            self.position[2] = self.altitude

        self.avoided_collision = True

    def check_violation(self):
        self.violation = False
        if not self.cooperative:
            if self.in_no_fly_zone():
                self.violation = True

    def in_no_fly_zone(self):
        distance_to_center = np.linalg.norm(self.position[:2])
        if distance_to_center < self.environment.approach_area_radius and self.task_type != 'landing':
            return True
        return False

    def compute_energy_consumption(self, delta_speed, delta_turn, delta_climb):
        energy = abs(delta_speed) * 0.1 + abs(delta_turn) * 0.05 + abs(delta_climb) * 0.2
        energy += self.speed * 0.01
        return energy

    def has_reached_target(self, tolerance=5.0):
        return np.linalg.norm(self.position - self.target) <= tolerance

    @classmethod
    def random_uav(cls, drone_id: int, task_type: str, environment):
        cooperative = random.choice([True, False])
        emergency = False
        uav_type = random.choice(UAV_TYPES)
        optimal_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        priority = random.randint(1, 10)  # 随机分配优先级

        if task_type == 'takeoff':
            idle_pads = environment.get_idle_takeoff_pads()
            if not idle_pads:
                return None
            pad_id = random.choice(idle_pads)
            position = environment.takeoff_pads[pad_id]['position'].copy()
            position[2] = 0.0
            target = environment.takeoff_ring_center.copy()
            drone = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                        task_type=task_type, cooperative=cooperative, type=uav_type, emergency=emergency)
            drone.assigned_pad = pad_id
            drone.environment = environment
            drone.priority = priority
            environment.takeoff_pads[pad_id]['status'] = 'occupied'
            return drone
        elif task_type == 'landing':
            theta = random.uniform(0, 2 * np.pi)
            x = environment.airspace_radius * np.cos(theta)
            y = environment.airspace_radius * np.sin(theta)
            z = random.uniform(0, MAX_ALTITUDE)
            position = np.array([x, y, z])
            target = environment.approach_area_center()
            drone = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                        task_type=task_type, cooperative=cooperative, type=uav_type, emergency=emergency)
            drone.environment = environment
            drone.priority = priority
            return drone
        else:
            # 巡航无人机
            minx, miny, maxx, maxy = -environment.airspace_radius, -environment.airspace_radius, \
                                     environment.airspace_radius, environment.airspace_radius
            position = np.array([random.uniform(minx, maxx),
                                 random.uniform(miny, maxy),
                                 random.uniform(10.0, MAX_ALTITUDE)])
            target = np.array([random.uniform(minx, maxx),
                               random.uniform(miny, maxy),
                               random.uniform(10.0, MAX_ALTITUDE)])
            drone = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                        task_type=task_type, cooperative=cooperative, type=uav_type, emergency=emergency)
            drone.environment = environment
            drone.priority = priority
            return drone

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
            self.ax.scatter(pos[0], pos[1], pos[2], color='blue', marker='s', s=100, label='Takeoff Pad' if pad_id == 0 else "")

        # 绘制降落垫
        for pad_id, pad in self.landing_pads.items():
            pos = pad['position']
            self.ax.scatter(pos[0], pos[1], pos[2], color='orange', marker='s', s=100, label='Landing Pad' if pad_id == 0 else "")

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
