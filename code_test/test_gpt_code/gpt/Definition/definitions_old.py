import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import random

# 常量定义
MAX_SPEED = 20.0
MIN_SPEED = 5.0
MAX_TURN_RATE = 30.0  # 每秒最大转向角度（度）
MAX_CLIMB_RATE = 5.0  # 每秒最大爬升/下降速度
MAX_ACCELERATION = 5.0  # 最大加速度，单位：m/s²
SENSING_RANGE = 100.0  # 感知范围，单位：m
RANGE_DETECTION = 100.0  # 检测区域半径，单位：m
DT = 0.1  # 时间步长，单位：s
NUM_MOVE = 5  # 每个动作的移动步数
MAX_EPISODE_LEN = 500  # 最大回合长度
NUM_DETECTION_SECTORS = 12  # 检测区域划分的扇区数
MIN_DISTANCE = 5.0  # 无人机之间的最小安全距离，单位：m

# 空域参数
R1 = 100.0   # 起飞/降落场和环的半径
R2 = 150.0   # 进近区域的半径（R2 > R1）
R3 = 300.0   # 外部空域的半径
H1 = 120.0   # 起飞环的高度
H2 = 80.0    # 降落环的高度
MAX_ALTITUDE = 1500.0  # 空域的最大高度

# 定义无人机类型
UAV_TYPES = ['multirotor', 'light_hybrid_wing', 'medium_hybrid_wing', 'heavy_hybrid_wing']
UAV_TYPE_MAPPING = {
    'multirotor': 0,
    'light_hybrid_wing': 1,
    'medium_hybrid_wing': 2,
    'heavy_hybrid_wing': 3
}
# 定义不同类型组合之间的最小安全距离矩阵（单位：米）
MIN_SAFE_DISTANCE_MATRIX = {
    ('multirotor', 'multirotor'): 10.0,
    ('multirotor', 'light_hybrid_wing'): 15.0,
    ('multirotor', 'medium_hybrid_wing'): 20.0,
    ('multirotor', 'heavy_hybrid_wing'): 25.0,
    ('light_hybrid_wing', 'light_hybrid_wing'): 15.0,
    ('light_hybrid_wing', 'medium_hybrid_wing'): 20.0,
    ('light_hybrid_wing', 'heavy_hybrid_wing'): 25.0,
    ('medium_hybrid_wing', 'medium_hybrid_wing'): 25.0,
    ('medium_hybrid_wing', 'heavy_hybrid_wing'): 30.0,
    ('heavy_hybrid_wing', 'heavy_hybrid_wing'): 35.0,
    # 对称性
    ('light_hybrid_wing', 'multirotor'): 15.0,
    ('medium_hybrid_wing', 'multirotor'): 20.0,
    ('heavy_hybrid_wing', 'multirotor'): 25.0,
    ('medium_hybrid_wing', 'light_hybrid_wing'): 20.0,
    ('heavy_hybrid_wing', 'light_hybrid_wing'): 25.0,
    ('heavy_hybrid_wing', 'medium_hybrid_wing'): 30.0,
}


# 定义无人机类
@dataclass
class UAV:
    drone_id: int
    type: str
    position: np.ndarray  # 三维坐标，形状为 (3,)
    target: np.ndarray  # 三维坐标，形状为 (3,)
    task_type: str  # 'takeoff' 或 'landing'
    cooperative: bool  # 是否为合作无人机
    optimal_speed: float  # 最优速度，单位：m/s
    emergency: bool = False  # 是否为紧急状态
    max_altitude: float = 1500.0  # 最大高度，单位：m，同样也是研究空域最大高度
    min_altitude: float = 0.0    # 最小高度，单位：m
    energy_consumption: float = field(default=0.0, init=False)

    speed: float = field(init=False)
    heading: np.ndarray = field(init=False)  # 三维方向向量
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
    priority: int = field(default=0, init=False)  # 避让优先级，数值越高优先级越高
    # 强化学习的奖励组件
    r_collision: int = 0
    r_task_completion: int = 0
    r_energy_efficiency: int = 0
    r_safety: int = 0
    environment: Optional[object] = field(default=None, init=False)  # 环境引用

    # 起飞降落环属性、起降场冷却
    landing_ring_radius: float = 30.0  # 降落（在下面）排序环的半径
    landing_ring_altitude: float = 20.0  # 降落排序环的高度
    in_approach_ring: bool = field(default=False, init=False)  # 是否进入降落排序环
    landing: bool = field(default=False, init=False)  # 是否正在降落
    landing_complete: bool = field(default=False, init=False)  # 是否完成起飞

    # 起飞相关属性
    takeoff_ring_radius: float = 50.0  # 起飞(上面）排序环的半径
    takeoff_takeoff_complete: bool = field(default=False, init=False)  # 是否完成起飞ring_altitude: float = 50.0  # 起飞排序环的高度
    in_takeoff_ring: bool = field(default=False, init=False)  # 是否进入起飞排序环
    taking_off: bool = field(default=False, init=False)  # 是否正在起飞
    takeoff_complete: bool = field(default=False, init=False)  # 是否完成起飞

    takeoff_pad_assigned: Optional[int] = field(default=None, init=False)  # 分配的起飞场ID
    landing_pad_assigned: Optional[int] = field(default=None, init=False)  # 分配的降落场ID
    sequence_number: Optional[int] = field(default=None, init=False)  # 降落排序序号

    position_history: List[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """
        初始化无人机的航向、速度和高度
        """
        # if self.task_type == 'takeoff':
        #     self.altitude = self.min_altitude  # 起飞任务从最低高度开始
        # elif self.task_type == 'landing':
        #     self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        # else:
        #     self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        self.heading = self.calculate_direction()
        self.heading_angle = np.rad2deg(np.arctan2(self.heading[1], self.heading[0]))
        self.speed = random.uniform(MIN_SPEED, MAX_SPEED)
        self.altitude = self.position[2]
        self.last_altitude = self.altitude
        self.optimal_path_length = np.linalg.norm(self.position - self.target)
        self.energy = 100.0  # 初始能量
        self.path_length = 0.0
        self.r_collision = 0
        self.r_task_completion = 0
        self.environment = None  # 将在环境中设置

    def calculate_direction(self) -> np.ndarray:
        """
        计算当前位姿到目标的方向向量（单位向量）
        """
        direction = self.target - self.position
        norm = np.linalg.norm(direction)
        return direction / norm if norm != 0 else np.array([0.0, 0.0, 0.0])

    def step(self, action: np.ndarray, other_drones: List['UAV']):
        """
                根据给定的动作更新无人机的状态
                """
        # 获取动作指令中的速度、航向及高度变化量
        delta_speed = action[0]
        delta_turn = action[1]
        delta_climb = action[2]
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

        # 检查是否完成任务
        if self.has_reached_target():
            self.task_finished = True
            self.r_task_completion = 1

        # 检查避让
        if self.cooperative:
            self.avoid_collision(other_drones)

        # 检查违规
        self.check_violation()
        self.position_history.append(self.position.copy())

    def avoid_collision(self, other_drones: List['UAV']):
        avoidance_vector = np.array([0.0, 0.0, 0.0])
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
        # 调整航向角
        angle_adjustment = np.rad2deg(np.arctan2(avoidance_direction[1], avoidance_direction[0]))
        self.heading_angle += angle_adjustment
        self.heading_angle = self.heading_angle % 360

        # 调整高度
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


