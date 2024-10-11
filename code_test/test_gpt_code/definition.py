import numpy as np
import time
import random
from shapely.geometry import Point, Polygon


class Drone:
    trajectory_length: float = 0
    r_conflict: int = 0
    r_warning: int = 0
    r_eco_airspeed: int = 0
    r_eco_heading: int = 0
    r_eco_level: int = 0
    r_smo_airspeed: int = 0
    r_smo_heading: int = 0
    r_smo_level: int = 0
    r_over_max_level: int = 0
    r_over_min_level: int = 0

    def __init__(self, drone_id, task_type, cooperative, emergency=False):
        self.drone_id = drone_id
        self.task_type = task_type  # 'takeoff' or 'landing'
        self.cooperative = cooperative
        self.emergency = emergency
        self.position = np.array([0.0, 0.0, 0.0])  # 位置
        self.velocity = np.array([0.0, 0.0, 0.0])  # 速度
        self.destination: Point
        self.energy = 100.0  # 能量水平
        self.MAX_VELOCITY = 10.0  # 最大速度，单位：m/s
        self.MAX_ACCELERATION = 5.0  # 最大加速度，单位：m/s²
        self.SENSING_RANGE = 50.0  # 感知范围，单位：m
        self.dt = 0.1

    def step(self, action):
        # 动作为加速度控制
        acceleration = np.clip(action, -self.MAX_ACCELERATION, self.MAX_ACCELERATION)

        # 更新速度，考虑速度限制
        self.velocity += acceleration * self.dt
        self.velocity = np.clip(self.velocity, -self.MAX_VELOCITY, self.MAX_VELOCITY)

        # 更新位置
        self.position += self.velocity * self.dt

        # 能量消耗，假设与加速度大小相关
        self.energy -= np.linalg.norm(acceleration) * self.dt

    def get_state(self):
        return self.position, self.velocity


