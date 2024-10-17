import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import random


def takeoff_logic(self):
    """
    起飞任务的无人机逻辑
    """
    if not self.taking_off:
        # 检查起飞场是否可用
        if self.environment.takeoff_pads_status[self.takeoff_pad_assigned] == 'available':
            # 起飞场可用，开始起飞
            self.taking_off = True
            self.environment.takeoff_pads_status[self.takeoff_pad_assigned] = 'occupied'
            self.environment.takeoff_pad_cooldown_timers[self.takeoff_pad_assigned] = 0
        else:
            # 起飞场不可用，等待
            pass
    elif self.taking_off and not self.in_takeoff_ring:
        # 垂直上升至起飞环高度
        ascent_speed = np.clip(self.speed, MIN_SPEED, MAX_SPEED)
        self.altitude += ascent_speed * DT
        self.position[2] = self.altitude
        if self.altitude >= self.environment.takeoff_ring_altitude:
            self.in_takeoff_ring = True
            # 设置航向为沿起飞排序环飞行
            self.heading = self.calculate_takeoff_ring_heading()
    elif self.in_takeoff_ring and not self.takeoff_complete:
        # 沿着起飞排序环飞行
        self.altitude = self.environment.takeoff_ring_altitude
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
        # 更新位置
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude
    else:
        pass  # 其他情况


def landing_logic(self):
    """
    降落任务的无人机逻辑
    """
    if not self.in_approach_ring:
        # 前往降落进近点
        self.heading = self.calculate_direction()
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude
        if np.linalg.norm(self.position - self.target) < 5.0:
            self.in_approach_ring = True
            # 更新目标为降落环
            self.target = self.environment.landing_ring_center.copy()
    elif self.in_approach_ring and not self.landing:
        # 按序号进入降落环
        # 简化逻辑，直接进入降落环
        self.altitude = self.environment.landing_ring_altitude
        self.position[2] = self.altitude
        # 绕降落环飞行
        self.heading = self.calculate_landing_ring_heading()
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude
        # 检查是否有空闲的降落场
        idle_landing_pads = self.environment.get_idle_landing_pads()
        if idle_landing_pads:
            self.landing_pad_assigned = idle_landing_pads[0]  # 按序号分配
            self.environment.landing_pads_status[self.landing_pad_assigned] = 'occupied'
            self.environment.landing_pad_cooldown_timers[self.landing_pad_assigned] = 0
            self.landing = True
            # 更新目标为降落场的投影圆心位置
            self.target = self.environment.landing_pads_positions[self.landing_pad_assigned].copy()
            self.target[2] = self.altitude
    elif self.landing:
        # 从降落环驶向降落场的投影圆心位置
        self.heading = self.calculate_direction()
        dt = DT
        self.position += self.speed * self.heading * dt
        self.position[2] = self.altitude
        if np.linalg.norm(self.position - self.target) < 5.0:
            # 垂直下降
            descent_speed = np.clip(self.speed, MIN_SPEED, MAX_SPEED)
            self.altitude -= descent_speed * DT
            self.position[2] = self.altitude
            if self.altitude <= self.min_altitude:
                self.altitude = self.min_altitude
                self.position[2] = self.altitude
                self.r_task_completion = 1
                # 标记降落场为占用状态，重置冷却计时器
                self.environment.landing_pads_status[self.landing_pad_assigned] = 'occupied'
                self.environment.landing_pad_cooldown_timers[self.landing_pad_assigned] = 0
    else:
        pass  # 其他情况


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
def random_uav(cls, drone_id: int, task_type: str, environment):
    """
    根据任务类型，按照指定的逻辑生成无人机
    """
    # 如果没有传入 uav_type，则随机选择
    if uav_type is None:
        uav_type = random.choice(UAV_TYPES)
    cooperative = random.choice([True, False])
    optimal_speed = random.uniform(MIN_SPEED, MAX_SPEED)

    if task_type == 'takeoff':
        # 选择一个空闲的起飞场
        idle_takeoff_pads = environment.get_idle_takeoff_pads()
        if not idle_takeoff_pads:
            raise Exception("没有可用的起飞场")
        takeoff_pad = random.choice(idle_takeoff_pads)
        position = environment.takeoff_pads_positions[takeoff_pad].copy()
        position[2] = 0.0  # 高度为零

        # 目标是起飞环的投影圆心位置
        target = environment.takeoff_ring_center.copy()
        target[2] = environment.takeoff_ring_altitude

        # 创建无人机实例，将 uav_type 传递给实例
        uav = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                  task_type=task_type, cooperative=cooperative, uav_type=uav_type, emergency=emergency)

        uav.takeoff_pad_assigned = takeoff_pad
        uav.environment = environment
        return uav

    elif task_type == 'landing':
        # 在空域侧面生成出生点
        maxr, cylinder_height = airspace_bounds
        theta = random.uniform(0, 2 * np.pi)
        x = maxr * np.cos(theta)
        y = maxr * np.sin(theta)
        z = np.random.uniform(0, cylinder_height)
        position = np.array([x, y, z])

        # 目标是降落环外部的进近空域
        target = environment.landing_approach_point.copy()

        # 创建无人机实例
        uav = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                  task_type=task_type, cooperative=cooperative, uav_type=uav_type, emergency=emergency)
        # 分配降落排序序号
        uav.sequence_number = environment.get_next_landing_sequence()
        uav.environment = environment
        return uav

    else:
        # 巡航任务，按照之前的逻辑
        maxx, cylinder_height = airspace_bounds
        maxy = maxx
        minx = -maxx
        miny = -maxy
        position = environment._spawn_on_surface()
        target = environment._spawn_on_surface()
        while np.linalg.norm(position - target) < 10.0:
            target = np.array([random.uniform(minx, maxx),
                               random.uniform(miny, maxy),
                               random.uniform(environment.min_altitude, environment.max_altitude)])
        uav = cls(drone_id=drone_id, position=position, target=target, optimal_speed=optimal_speed,
                  task_type=task_type, cooperative=cooperative, uav_type=uav_type, emergency=emergency)
        uav.environment = environment
        return uav


def _generate_destination(self, uav_id):
    # 生成目的地，根据任务类型
    pass


def calculate_takeoff_ring_heading(self):
    """
    计算沿着起飞排序环飞行的航向
    """
    # 计算当前位置在起飞排序环上的切线方向
    center = self.environment.takeoff_ring_center.copy()
    direction = np.cross(np.array([0, 0, 1]), self.position - center)
    norm = np.linalg.norm(direction)
    if norm != 0:
        return direction / norm
    else:
        return np.array([0.0, 0.0, 0.0])


def calculate_landing_ring_heading(self):
    """
    计算沿着降落排序环飞行的航向
    """
    # 计算当前位置在降落排序环上的切线方向
    center = self.environment.landing_ring_center.copy()
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