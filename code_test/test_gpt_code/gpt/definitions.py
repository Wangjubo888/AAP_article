import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Optional
from shapely.geometry import Point, Polygon
import numpy as np

# 常量定义
MAX_DRONES = 20  # 最大无人机数量
MAX_SPEED = 15.0  # 最大速度，单位：m/s
MIN_SPEED = 5.0   # 最小速度，单位：m/s
MAX_ACCELERATION = 5.0  # 最大加速度，单位：m/s²
SENSING_RANGE = 100.0  # 感知范围，单位：m
DT = 0.1  # 时间步长，单位：s
NUM_MOVE = 5  # 每个动作的移动步数
MAX_EPISODE_LEN = 500  # 最大回合长度
NUM_DETECTION_SECTORS = 12  # 检测区域划分的扇区数
RANGE_DETECTION = 100.0  # 检测区域半径，单位：m
MIN_DISTANCE = 5.0  # 无人机之间的最小安全距离，单位：m


# 定义无人机类
@dataclass
class UAV:
    drone_id: int
    position: np.ndarray  # 三维坐标，形状为 (3,)
    target: np.ndarray  # 三维坐标，形状为 (3,)
    task_type: str  # 'takeoff' 或 'landing'
    cooperative: bool  # 是否为合作无人机
    emergency: bool = False  # 是否为紧急状态
    max_altitude: float = 1500.0  # 最大高度，单位：m，同样也是研究空域最大高度
    min_altitude: float = 0.0    # 最小高度，单位：m
    optimal_speed: float = 10.0  # 最优速度，单位：m/s

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

    # 强化学习的奖励组件
    r_collision: int = 0
    r_task_completion: int = 0
    r_energy_efficiency: int = 0
    r_safety: int = 0

    # 起飞降落环属性、起降场冷却
    landing_ring_radius: float = 30.0  # 降落（在下面）排序环的半径
    landing_ring_altitude: float = 20.0  #
    in_approach_ring: bool = field(default=False, init=False)  # 是否进入降落排序环
    landing: bool = field(default=False, init=False)  # 是否正在降落
    landing_pad_assigned: Optional[int] = field(default=None, init=False)  # 分配的降落场ID
    # 起飞相关属性
    takeoff_ring_radius: float = 50.0  # 起飞(上面）排序环的半径
    takeoff_ring_altitude: float = 50.0  # 起飞排序环的高度
    in_takeoff_ring: bool = field(default=False, init=False)  # 是否进入起飞排序环
    taking_off: bool = field(default=False, init=False)  # 是否正在起飞
    takeoff_complete: bool = field(default=False, init=False)  # 是否完成起飞

    def __post_init__(self) -> None:
        """
        初始化无人机的航向、速度和高度
        """
        self.heading = self.calculate_direction()
        self.speed = self.optimal_speed
        if self.task_type == 'takeoff':
            self.altitude = self.min_altitude  # 起飞任务从最低高度开始
        elif self.task_type == 'landing':
            self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        # else:
        #     self.altitude = random.uniform(self.min_altitude, self.max_altitude)
        self.last_altitude = self.altitude
        self.optimal_path_length = np.linalg.norm(self.position - self.target)

    def calculate_direction(self) -> np.ndarray:
        """
        计算当前位姿到目标的方向向量（单位向量）
        """
        direction = self.target - self.position
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])
        return direction / norm

    def predict_position(self, dt: float = DT) -> np.ndarray:
        """
        预测在 dt 秒后的未来位置，保持当前速度和航向
        """
        return self.position + self.speed * self.heading * dt

    @property
    def components(self) -> np.ndarray:
        """
        获取速度在 X、Y、Z 方向的分量
        """
        return self.speed * self.heading

    def distance_to_target(self) -> float:
        """
        计算到目标的当前距离
        """
        return np.linalg.norm(self.position - self.target)

    def heading_drift(self) -> np.ndarray:
        """
        计算当前航向与目标航向之间的偏差向量
        """
        desired_direction = self.calculate_direction()
        return desired_direction - self.heading

    @classmethod
    def random_uav(cls, airspace_bounds: Tuple[float, float, float, float],
                   drone_id: int, task_type: str,
                   cooperative: bool, emergency: bool = False):
        """
        在指定空域内创建一个随机无人机
        """
        minx, miny, maxx, maxy = airspace_bounds
        position = np.array([random.uniform(minx, maxx),
                             random.uniform(miny, maxy),
                             random.uniform(0.0, 100.0)])
        target = np.array([random.uniform(minx, maxx),
                           random.uniform(miny, maxy),
                           random.uniform(0.0, 100.0)])
        while np.linalg.norm(position - target) < 10.0:
            target = np.array([random.uniform(minx, maxx),
                               random.uniform(miny, maxy),
                               random.uniform(0.0, 100.0)])
        optimal_speed = random.uniform(MIN_SPEED, MAX_SPEED)
        return cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                   task_type=task_type, cooperative=cooperative, emergency=emergency)

    def _generate_destination(self, uav_id):
        # 生成目的地，根据任务类型
        if self.task_type == 'takeoff':
            return self._spawn_on_surface()
        elif self.task_type == 'landing':
            return self.assign_landing_target(uav_id)

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

    def step(self, action: dict):
        """
        根据给定的动作更新无人机的状态
        """
        # 获取动作指令中的速度、航向及高度变化量
        delta_speed = action.get('delta_speed', 0.0)
        delta_heading = action.get('delta_heading', np.array([0.0, 0.0, 0.0]))
        delta_altitude = action.get('delta_altitude', 0.0)

        # 更新当前对象的速度、航向及高度，并限制在合理范围内
        self.speed = np.clip(self.speed + delta_speed, MIN_SPEED, MAX_SPEED)
        new_heading = self.heading + delta_heading
        norm = np.linalg.norm(new_heading)
        if norm != 0:
            self.heading = new_heading / norm
        self.altitude = np.clip(self.altitude + delta_altitude, self.min_altitude, self.max_altitude)

        # 根据时间步长DT更新位置
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude

        # 减少能量并累加路径长度
        self.energy -= self.compute_energy_consumption(delta_speed, delta_heading, delta_altitude)
        self.path_length += self.speed * dt

        # 着陆逻辑
        if self.task_type == 'landing':
            distance_to_ring_center = np.linalg.norm(self.position[:2] - self.approach_ring_center[:2])

            if not self.in_approach_ring and distance_to_ring_center <= self.approach_ring_radius:
                self.in_approach_ring = True
                self.target = self.landing_point
                self.heading = self.calculate_direction()

            if self.in_approach_ring and not self.landing:
                self.heading[2] = 0.0  # 保持水平飞行
                distance_to_landing_point = np.linalg.norm(self.position[:2] - self.landing_point[:2])

                if distance_to_landing_point <= 5.0:
                    self.landing = True

            if self.landing:
                if self.environment.landing_pad_status == 'available':
                    self.vertical_landing()
                else:
                    self.speed = 0.0  # 悬停
        elif self.task_type == 'takeoff':
            self.takeoff_logic()

    def takeoff_logic(self):
        """
        执行起飞任务的无人机的起飞逻辑
        """
        if not self.taking_off:
            # 检查起飞场是否可用
            if self.environment.takeoff_pad_status == 'available':
                # 起飞场可用，开始起飞
                self.taking_off = True
                self.environment.takeoff_pad_status = 'occupied'
                self.environment.takeoff_pad_cooldown_timer = 0
            else:
                # 起飞场不可用，等待
                pass
        elif self.taking_off and not self.in_takeoff_ring:
            # 垂直上升至起飞排序环高度
            ascent_speed = np.clip(self.speed, MIN_SPEED, MAX_SPEED)
            self.altitude += ascent_speed * DT
            self.position[2] = self.altitude
            if self.altitude >= self.takeoff_ring_altitude:
                self.in_takeoff_ring = True
                # 设置航向为沿起飞排序环飞行
                self.heading = self.calculate_takeoff_ring_heading()
        elif self.in_takeoff_ring and not self.takeoff_complete:
            # 沿着起飞排序环维持高度飞行
            self.altitude = self.takeoff_ring_altitude
            self.position[2] = self.altitude
            # 检查是否可以驶出起飞排序环
            if self.can_depart_takeoff_ring():
                # 更新航向，朝向目标
                self.heading = self.calculate_direction()
                self.takeoff_complete = True
            else:
                # 继续沿着起飞排序环飞行
                self.move_along_takeoff_ring()
        elif self.takeoff_complete:
            # 起飞完成，按照正常逻辑飞行
            # ...（可以调用原有的移动逻辑或进行调整）
            self.move_towards_target()
        else:
            pass  # 其他情况

    def calculate_takeoff_ring_heading(self):
        """
        计算沿着起飞排序环飞行的航向
        """
        # 计算当前位置在起飞排序环上的切线方向
        # 起飞排序环中心
        center = self.environment.takeoff_pad_position.copy()
        center[2] = self.takeoff_ring_altitude
        direction = np.cross(np.array([0, 0, 1]), self.position - center)
        norm = np.linalg.norm(direction)
        if norm != 0:
            return direction / norm
        else:
            return np.array([0.0, 0.0, 0.0])

    def can_depart_takeoff_ring(self):
        """
        判断无人机是否可以驶出起飞排序环
        """
        # 当无人机的速度方向与目的地方向连线的投影与起飞环相切时
        to_target = self.target - self.position
        to_target[2] = 0.0  # 投影到水平面
        heading_proj = self.heading.copy()
        heading_proj[2] = 0.0
        angle = np.arccos(np.clip(np.dot(heading_proj, to_target) /
                                  (np.linalg.norm(heading_proj) * np.linalg.norm(to_target)), -1.0, 1.0))
        # 当夹角接近90度，即切线方向
        if np.abs(angle - np.pi / 2) < 0.1:
            return True
        else:
            return False

    def move_along_takeoff_ring(self):
        """
        沿着起飞排序环飞行
        """
        # 速度和航向保持不变，更新位置
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude

    def move_towards_target(self):
        """
        朝向目标飞行，保持高度层飞行
        """
        # 更新航向
        self.heading = self.calculate_bearing()
        self.heading[2] = 0.0  # 保持高度不变
        # 更新位置
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude

    def calculate_bearing(self):
        """
        计算当前位姿到目标的方向向量（单位向量）
        """
        direction = self.target - self.position
        direction[2] = 0.0  # 投影到水平面，保持高度不变
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0])
        return direction / norm

    def compute_energy_consumption(self, delta_speed, delta_heading, delta_altitude):
        """
        根据速度、航向和高度的变化计算能量消耗
        """
        energy = abs(delta_speed) * 0.1 + np.linalg.norm(delta_heading) * 0.05 + abs(delta_altitude) * 0.2
        energy += self.speed * 0.01
        return energy

    def check_collision(self, other_uav, collision_distance=MIN_DISTANCE):
        """
        检查是否与另一架无人机发生碰撞
        """
        distance = np.linalg.norm(self.position - other_uav.position)
        if distance < collision_distance:
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

    def vertical_landing(self):
        """
        垂直下降至地面
        """
        if self.altitude > self.min_altitude:
            descent_speed = np.clip(self.speed, MIN_SPEED, MAX_SPEED)
            self.altitude -= descent_speed * DT
            self.position[2] = self.altitude
        else:
            # 完成降落
            self.altitude = self.min_altitude
            self.position[2] = self.altitude
            self.r_task_completion = 1
            self.landing = False  # 标记为降落完成
            self.environment.landing_pad_status = 'occupied'  # 标记降落场为占用状态
            self.environment.landing_pad_cooldown_timer = 0  # 重置冷却计时器

    @property
    def approach_ring_center(self) -> np.ndarray:
        """
        获取降落排序环的中心点（降落场的位置）
        """
        return self.environment.landing_pad_position

    @property
    def landing_point(self) -> np.ndarray:
        """
        获取降落场的位置（用于垂直下降）
        """
        return self.environment.landing_pad_position