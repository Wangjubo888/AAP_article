# env.py

import gym
import numpy as np
import random
from typing import List
from definitions import UAV, MIN_SAFE_DISTANCE_MATRIX, UAV_TYPES, UAV_OPTIMAL_SPEED, MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE, DT, R1, R2, R3, H1, H2, MAX_ALTITUDE, TAKEOFF_RING_CAPACITY, LANDING_RING_CAPACITY
import math
from vispy import app, scene
from vispy.scene import visuals
from vispy.visuals import MeshVisual
from vispy.color import get_colormap
import time

class UrbanUAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UrbanUAVEnv, self).__init__()
        self.num_drones = 5  # 调整为所需的无人机数量
        self.drones = []
        self.time_step = 0
        self.max_time_steps = 200

        # 定义动作空间为连续空间
        action_dim = 3  # [delta_speed, delta_turn, delta_climb]
        self.action_space = gym.spaces.Box(
            low=np.array([-MAX_ACCELERATION, -MAX_TURN_RATE, -MAX_CLIMB_RATE]),
            high=np.array([MAX_ACCELERATION, MAX_TURN_RATE, MAX_CLIMB_RATE]),
            dtype=np.float32
        )

        # 定义观测空间
        num_nearest = 3
        self.num_nearest = num_nearest
        state_dim = 10 + num_nearest * 6  # [原来的13维 + 邻居信息]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # 空域定义
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.H1 = H1
        self.H2 = H2
        self.airspace_radius = self.R3
        self.approach_area_radius = self.R2

        # 起飞和降落区域（四个关于原点对称的圆形区域）
        self.takeoff_areas = [
            {'center': np.array([-R1, -R1, 0.0]), 'radius': R1 / 2},
            {'center': np.array([R1, -R1, 0.0]), 'radius': R1 / 2},
            {'center': np.array([-R1, R1, 0.0]), 'radius': R1 / 2},
            {'center': np.array([R1, R1, 0.0]), 'radius': R1 / 2},
        ]

        self.landing_areas = [
            {'center': np.array([-R1, 0.0, 0.0]), 'radius': R1 / 2},
            {'center': np.array([R1, 0.0, 0.0]), 'radius': R1 / 2},
            {'center': np.array([0.0, -R1, 0.0]), 'radius': R1 / 2},
            {'center': np.array([0.0, R1, 0.0]), 'radius': R1 / 2},
        ]

        # 环和中心
        self.takeoff_ring_center = np.array([0.0, 0.0, self.H1])
        self.landing_ring_center = np.array([0.0, 0.0, self.H2])

        # 进近队列
        self.approach_queue: List[UAV] = []

        # 紧急无人机标志
        self.emergency_uav_exists = False

        self.done_drones = set()

        # 环内的无人机列表
        self.takeoff_ring_uavs: List[UAV] = []
        self.landing_ring_uavs: List[UAV] = []

        self.reset()

        # 初始化 VisPy 画布
        self.canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=30)
        self.view.camera.fov = 60
        self.view.camera.distance = self.R3 * 2.0

        # 添加标记器用于无人机
        self.markers = visuals.Markers()
        self.view.add(self.markers)

        # 添加轨迹线
        self.traces = []
        for _ in range(self.num_drones):
            line = visuals.Line(color='black', width=1)
            self.view.add(line)
            self.traces.append(line)

        # 设置颜色映射
        self.colormap = get_colormap('viridis')

        # 添加起飞环和降落环的可视化
        self.draw_rings()

        # 添加空域的可视化
        self.draw_airspace()

        # 添加起飞和降落区域的可视化
        self.draw_areas()

        # 启动事件循环
        self.app = app.Application()
        self.timer = app.Timer(0.1, connect=self.on_timer, start=True)

    def on_timer(self, event):
        pass  # 需要处理时可以添加逻辑

    def reset(self):
        self.drones = []
        self.done_drones = set()
        self.approach_queue = []
        self.emergency_uav_exists = False
        self.time_step = 0
        self.takeoff_ring_uavs = []
        self.landing_ring_uavs = []

        # 只生成执行起飞或降落任务的无人机
        for i in range(self.num_drones):
            task_type = random.choice(['takeoff', 'landing'])
            drone = UAV.random_uav(drone_id=i, task_type=task_type, environment=self)
            if drone:
                self.drones.append(drone)

        return self._get_observation()

    def step(self, actions: List[np.ndarray]):
        rewards = []
        dones = []
        infos = {}

        for i, drone in enumerate(self.drones):
            if i not in self.done_drones:
                if i < len(actions):
                    action = actions[i]
                else:
                    action = np.zeros(self.action_space.shape[0])  # 默认动作
                drone.step(action, self.drones)
                if drone.task_finished or drone.energy <= 0:
                    self.done_drones.add(i)

        self.time_step += 1
        self._apply_collision_avoidance()

        rewards = self._compute_rewards()
        obs = self._get_observation()

        # 为每个智能体生成单独的完成状态
        dones = [drone.task_finished or drone.energy <= 0 for drone in self.drones]

        done = False  # 环境的全局完成状态，如果需要可以保持为False
        return obs, rewards, dones, infos

    def _apply_collision_avoidance(self):
        # 在这里实现避撞逻辑
        for i, drone in enumerate(self.drones):
            if i in self.done_drones:
                continue
            # 获取附近的无人机
            neighbors = self._get_nearest_neighbors(drone, self.num_nearest)
            for neighbor in neighbors:
                if neighbor.drone_id in self.done_drones:
                    continue
                distance_vector = neighbor.position - drone.position
                distance = np.linalg.norm(distance_vector)
                min_distance = MIN_SAFE_DISTANCE_MATRIX.get((drone.type, neighbor.type), 10.0)
                if distance < min_distance:
                    # 发生碰撞
                    drone.r_collision = 1
                    neighbor.r_collision = 1
                    self.done_drones.add(i)
                    self.done_drones.add(self.drones.index(neighbor))
                elif distance < min_distance + 10.0:
                    # 接近碰撞，需要避让
                    if drone.cooperative:
                        # 调整航向角远离邻近无人机
                        avoidance_direction = -distance_vector / (distance + 1e-6)
                        angle_adjustment = np.rad2deg(np.arctan2(avoidance_direction[1], avoidance_direction[0])) - drone.heading_angle
                        angle_adjustment = np.clip(angle_adjustment, -MAX_TURN_RATE * DT, MAX_TURN_RATE * DT)
                        drone.heading_angle += angle_adjustment
                        drone.heading_angle = drone.heading_angle % 360
                        drone.avoided_collision = True
                    else:
                        # 非合作无人机不采取避让动作
                        pass

    def _compute_rewards(self):
        rewards = []
        for i, drone in enumerate(self.drones):
            reward = 0.0
            if drone.r_task_completion:
                reward += 100.0  # 任务完成奖励

            # 奖励或惩罚无人机保持最优速度
            speed_diff = abs(drone.speed - UAV_OPTIMAL_SPEED[drone.type])
            reward -= speed_diff * 0.5  # 惩罚速度偏离

            if drone.avoided_collision:
                reward += 10.0   # 成功避让奖励
            if drone.r_collision:
                reward -= 200.0 # 碰撞惩罚
            if drone.violation:
                reward -= 50.0  # 违规惩罚
            reward -= drone.energy_consumption * 0.1  # 能量消耗惩罚
            rewards.append(reward)
        return rewards

    def _get_observation(self):
        obs = []
        for i, drone in enumerate(self.drones):
            position = drone.position
            velocity = drone.speed * drone.heading
            energy = np.array([drone.energy])
            type_index = UAV_TYPES.index(drone.type)
            type_info = np.array([type_index])
            cooperative_flag = np.array([1.0 if drone.cooperative else 0.0])
            priority_info = np.array([drone.priority])

            # 获取最近的邻居信息
            neighbors_info = self._get_neighbors_info(drone)

            observation = np.concatenate([position, velocity, energy, type_info, cooperative_flag, priority_info, neighbors_info])
            obs.append(observation)
        return obs

    def _get_neighbors_info(self, drone):
        neighbors = self._get_nearest_neighbors(drone, self.num_nearest)
        info = []
        for neighbor in neighbors:
            relative_position = neighbor.position - drone.position
            relative_velocity = (neighbor.speed * neighbor.heading) - (drone.speed * drone.heading)
            info.extend(relative_position)
            info.extend(relative_velocity)
        # 如果邻居数量不足，填充0
        if len(neighbors) < self.num_nearest:
            for _ in range(self.num_nearest - len(neighbors)):
                info.extend([0.0]*6)
        return np.array(info)

    def _get_nearest_neighbors(self, drone, num_neighbors):
        distances = []
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                distances.append((distance, other_drone))
        distances.sort(key=lambda x: x[0])
        neighbors = [item[1] for item in distances[:num_neighbors]]
        return neighbors

    def can_ascend_to_takeoff_ring(self):
        # 检查起飞环容量是否未达到上限
        return len(self.takeoff_ring_uavs) < TAKEOFF_RING_CAPACITY

    def enter_takeoff_ring(self, drone: UAV):
        self.takeoff_ring_uavs.append(drone)

    def can_enter_landing_ring(self, drone: UAV):
        # 检查降落环容量是否未达到上限，并根据优先级顺序允许进入
        if len(self.landing_ring_uavs) < LANDING_RING_CAPACITY:
            if self.approach_queue and self.approach_queue[0] == drone:
                return True
        return False

    def enter_landing_ring(self, drone: UAV):
        self.landing_ring_uavs.append(drone)
        if drone in self.approach_queue:
            self.approach_queue.remove(drone)

    def can_land(self, drone: UAV):
        # 检查是否可以降落
        return True  # 这里不再限制降落停机坪，可直接降落在降落区域内

    def get_landing_point(self, drone: UAV):
        # 在降落区域内随机选择一个点
        area = random.choice(self.landing_areas)
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0, area['radius'])
        x = area['center'][0] + radius * np.cos(angle)
        y = area['center'][1] + radius * np.sin(angle)
        z = 0.0
        return np.array([x, y, z])

    def finish_landing(self, drone: UAV):
        if drone in self.landing_ring_uavs:
            self.landing_ring_uavs.remove(drone)

    def has_emergency_uav(self):
        return self.emergency_uav_exists

    def set_emergency_uav_exists(self, exists: bool):
        self.emergency_uav_exists = exists

    def approach_area_center(self):
        return np.array([0.0, 0.0, self.H2])

    def add_to_approach_queue(self, drone: UAV):
        # 紧急无人机插入队列最前面
        if drone.emergency:
            self.approach_queue.insert(0, drone)
            drone.priority = 999  # 紧急无人机最高优先级
        else:
            self.approach_queue.append(drone)
            # 优先级根据进入进近空域的时间分配，进入越早，优先级越高
            drone.priority = -drone.entered_approach_time  # 时间越小，值越大

    def draw_rings(self):
        # 绘制起飞环
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.takeoff_ring_center[0] + self.R1 * np.cos(theta)
        y = self.takeoff_ring_center[1] + self.R1 * np.sin(theta)
        z = np.full_like(x, self.H1)
        takeoff_ring = np.vstack((x, y, z)).T
        line1 = visuals.Line(pos=takeoff_ring, color='blue', width=2)
        self.view.add(line1)

        # 绘制降落环
        x = self.landing_ring_center[0] + self.R1 * np.cos(theta)
        y = self.landing_ring_center[1] + self.R1 * np.sin(theta)
        z = np.full_like(x, self.H2)
        landing_ring = np.vstack((x, y, z)).T
        line2 = visuals.Line(pos=landing_ring, color='green', width=2)
        self.view.add(line2)

    def draw_airspace(self):
        # 绘制外部空域（仅绘制底部圆圈作为参考）
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.airspace_radius * np.cos(theta)
        y = self.airspace_radius * np.sin(theta)
        z = np.zeros_like(x)
        airspace_circle = np.vstack((x, y, z)).T
        line = visuals.Line(pos=airspace_circle, color='gray', width=1)
        self.view.add(line)

    def draw_areas(self):
        # 绘制起飞区域
        for area in self.takeoff_areas:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = area['center'][0] + area['radius'] * np.cos(theta)
            y = area['center'][1] + area['radius'] * np.sin(theta)
            z = np.full_like(x, 0.0)
            pos = np.vstack((x, y, z)).T
            line = visuals.Line(pos=pos, color='blue', width=1)
            self.view.add(line)

        # 绘制降落区域
        for area in self.landing_areas:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = area['center'][0] + area['radius'] * np.cos(theta)
            y = area['center'][1] + area['radius'] * np.sin(theta)
            z = np.full_like(x, 0.0)
            pos = np.vstack((x, y, z)).T
            line = visuals.Line(pos=pos, color='green', width=1)
            self.view.add(line)

    def render(self, mode='human'):
        # 准备无人机的位置和颜色
        positions = []
        colors = []
        for drone in self.drones:
            positions.append(drone.position)
            if drone.task_type == 'takeoff':
                colors.append([0, 1, 1, 1])  # 青色
            elif drone.task_type == 'landing':
                if drone.emergency:
                    colors.append([1, 0, 0, 1])  # 红色表示紧急无人机
                else:
                    colors.append([1, 0, 1, 1])  # 品红色
            else:
                colors.append([0.5, 0.5, 0.5, 1])  # 灰色

        positions = np.array(positions)
        colors = np.array(colors)

        # 更新无人机标记
        self.markers.set_data(positions[:, :3], face_color=colors, size=10)

        # 更新轨迹
        max_trace_length = 100  # 仅绘制最近的100个位置点
        for idx, drone in enumerate(self.drones):
            if len(drone.position_history) > 1:
                trace = np.array(drone.position_history[-max_trace_length:])
                self.traces[idx].set_data(pos=trace[:, :3], width=1, color='black')  # X, Y, Z

        # 更新画布
        self.canvas.update()
        time.sleep(0.01)  # 控制渲染速度

    def close(self):
        if hasattr(self, 'canvas'):
            self.canvas.close()
