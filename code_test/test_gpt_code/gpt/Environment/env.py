# env.py

import gym
import numpy as np
import random
from typing import List, Optional, Dict
from definitions import (
    UAV, ALL_UAV_TYPES, ALL_UAV_OPTIMAL_SPEED, ALL_UAV_PERFORMANCE,
    MIN_SAFE_DISTANCE_MATRIX,
    DT, R1, R2, R3,
    H1, H2, MAX_ALTITUDE, TAKEOFF_RING_CAPACITY, LANDING_RING_CAPACITY
)
import plotly.graph_objs as go
import plotly.offline as pyo


class RiskAssessment:
    def __init__(self, drones, collision_threshold=50.0):
        self.drones = drones
        self.collision_threshold = collision_threshold

    def assess_risk(self):
        density = self.calculate_density()
        risk_level = self.evaluate_risk_level(density)
        return risk_level

    def calculate_density(self):
        positions = np.array([drone.position for drone in self.drones])
        density = np.sum(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1) < self.collision_threshold) / len(self.drones)
        return density

    def evaluate_risk_level(self, density):
        if density < 0.2:
            return "low"
        elif density < 0.5:
            return "medium"
        else:
            return "high"


class UrbanUAVEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(UrbanUAVEnv, self).__init__()
        self.num_drones = random.randint(10, 30)  # 无人机总数量随机
        self.drones = []
        self.time_step = 0
        self.max_time_steps = 200
        self.risk_assessment = RiskAssessment(self.drones)
        self.current_demand = "low"  # 初始需求设为低
        # 为每个无人机生成独立的动作空间
        self.action_spaces = {}

        for uav_type in self.UAV_TYPES:
            max_acceleration = self.UAV_PERFORMANCE[uav_type]['max_acceleration']
            max_turn_rate = self.UAV_PERFORMANCE[uav_type]['max_turn_rate']
            max_climb_rate = self.UAV_PERFORMANCE[uav_type]['max_climb_rate']

            # 生成每个无人机的动作空间
            self.action_spaces[uav_type] = gym.spaces.Box(
                low=np.array([-max_acceleration, -max_turn_rate, -max_climb_rate]),
                high=np.array([max_acceleration, max_turn_rate, max_climb_rate]),
                dtype=np.float32,
                shape=(3,)  # [delta_speed, delta_turn, delta_climb]
            )

        self.num_nearest = 5  # 最近邻数量
        self.state_dim = self._get_observation_space_dimension()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
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

        # 起飞和降落停机坪（分别两个），内切于起飞/降落环，关于原点对称分布
        # 起飞停机坪
        self.takeoff_pads = [
            {'center': np.array([-R1 / 2, 0.0, 0.0]), 'radius': R1 / 4},
            {'center': np.array([R1 / 2, 0.0, 0.0]), 'radius': R1 / 4},
        ]
        # 降落停机坪
        self.landing_pads = [
            {'center': np.array([0.0, -R1 / 2, 0.0]), 'radius': R1 / 4},
            {'center': np.array([0.0, R1 / 2, 0.0]), 'radius': R1 / 4},
        ]
        # 环和中心
        self.takeoff_ring_center = np.array([0.0, 0.0, self.H1])
        self.landing_ring_center = np.array([0.0, 0.0, self.H2])
        self.takeoff_ring_available = True
        self.landing_ring_available = True
        self.takeoff_cooldown = 0
        self.landing_cooldown = 0
        # 进近队列
        self.approach_queue: List[UAV] = []
        self.landing_queue: List[UAV] = []

        # 紧急无人机标志
        self.emergency_uav_exists = False

        self.done_drones = set()

        # 环内的无人机列表
        self.takeoff_ring_uavs: List[UAV] = []
        self.landing_ring_uavs: List[UAV] = []

        # 可视化设置
        self.fig = None  # Plotly 图表对象
        self.episode_data = []  # 存储每个时间步的无人机位置信息

        # 初始化环境
        self.reset()

    def reset(self):
        self.drones = []
        self.done_drones = set()
        self.approach_queue = []
        self.emergency_uav_exists = False
        self.time_step = 0
        self.takeoff_ring_uavs = []
        self.landing_ring_uavs = []
        # 重置冷却时间和可用状态
        self.takeoff_ring_available = True
        self.landing_ring_available = True
        self.takeoff_cooldown = 0
        self.landing_cooldown = 0
        # 生成执行起飞或降落任务的无人机
        for i in range(self.num_drones):
            task_type = random.choice(['takeoff', 'landing'])
            drone = UAV.random_uav(drone_id=i, task_type=task_type, environment=self)
            if drone:
                self.drones.append(drone)
        # 重置 episode_data
        self.episode_data = []

        return self._get_observation()

    def step(self, actions: List[np.ndarray]):
        risk_level = self.risk_assessment.assess_risk()
        self.adjust_permissions(risk_level)
        # 根据当前需求动态调整权限
        self.dynamic_priority_adjustment()
        rewards = []
        dones = []
        infos = {}
        # 更新冷却时间
        if not self.takeoff_ring_available:
            self.takeoff_cooldown -= 1
            if self.takeoff_cooldown <= 0:
                self.takeoff_ring_available = True

        if not self.landing_ring_available:
            self.landing_cooldown -= 1
            if self.landing_cooldown <= 0:
                self.landing_ring_available = True

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

        # 收集当前时间步的无人机位置信息
        self.collect_episode_data()

        done = False  # 环境的全局完成状态，如果需要可以保持为 False
        return obs, rewards, dones, infos

    def adjust_permissions(self, risk_level):
        for drone in self.drones:
            if drone.is_non_cooperative:
                if risk_level == "high":
                    drone.wait_time += 5
                elif risk_level == "medium":
                    drone.wait_time += 2
                else:
                    drone.wait_time = max(drone.wait_time - 1, 0)

    def dynamic_priority_adjustment(self):
        for drone in self.drones:
            if not drone.cooperative:
                if self.current_demand == "high":
                    drone.permission = "restricted"  # 限制非合作无人机
                elif self.current_demand == "medium":
                    drone.permission = "limited"  # 部分限制
                else:
                    drone.permission = "unrestricted"  # 允许运行

    def collect_episode_data(self):
        # 收集当前时间步的无人机位置信息
        time_step_data = []
        for drone in self.drones:
            time_step_data.append({
                'drone_id': drone.drone_id,
                'position': drone.position.copy(),
                'color': self.get_drone_color(drone)
            })
        self.episode_data.append(time_step_data)

    def render_episode(self, episode_num):
        # 在回合结束时生成动画
        frames = []
        drone_ids = [drone.drone_id for drone in self.drones]

        # 为每个无人机初始化轨迹
        data_dict = {}
        for drone_id in drone_ids:
            data_dict[drone_id] = {
                'x': [],
                'y': [],
                'z': [],
                'color': None
            }

        # 获取环境可视化元素
        environment_traces = self.get_environment_traces()

        # 遍历每个时间步的数据，构建动画帧
        for t, time_step_data in enumerate(self.episode_data):
            frame_data = []
            # 添加环境元素
            frame_data.extend(environment_traces)
            for drone_data in time_step_data:
                drone_id = drone_data['drone_id']
                x, y, z = drone_data['position']
                color = drone_data['color']

                # 更新轨迹数据
                data_dict[drone_id]['x'].append(x)
                data_dict[drone_id]['y'].append(y)
                data_dict[drone_id]['z'].append(z)
                data_dict[drone_id]['color'] = color

                # 创建无人机位置的 Scatter3d
                frame_data.append(go.Scatter3d(
                    x=[x],
                    y=[y],
                    z=[z],
                    mode='markers',
                    marker=dict(size=5, color=color),
                    name=f'Drone {drone_id}'
                ))

                # 创建无人机轨迹的 Scatter3d
                frame_data.append(go.Scatter3d(
                    x=data_dict[drone_id]['x'],
                    y=data_dict[drone_id]['y'],
                    z=data_dict[drone_id]['z'],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'Drone {drone_id} Path'
                ))

            frames.append(go.Frame(data=frame_data, name=str(t)))

        # 创建初始帧
        initial_data = []
        # 添加环境元素
        initial_data.extend(environment_traces)
        for drone_id in drone_ids:
            # 无人机初始位置
            initial_data.append(go.Scatter3d(
                x=[data_dict[drone_id]['x'][0]],
                y=[data_dict[drone_id]['y'][0]],
                z=[data_dict[drone_id]['z'][0]],
                mode='markers',
                marker=dict(size=5, color=data_dict[drone_id]['color']),
                name=f'Drone {drone_id}'
            ))
            # 无人机初始轨迹（为空）
            initial_data.append(go.Scatter3d(
                x=[],
                y=[],
                z=[],
                mode='lines',
                line=dict(color=data_dict[drone_id]['color'], width=2),
                name=f'Drone {drone_id} Path'
            ))

        # 定义滑块和播放按钮
        sliders = [dict(
            steps=[dict(method='animate',
                        args=[[str(k)],
                              dict(mode='immediate',
                                   frame=dict(duration=100, redraw=True),
                                   transition=dict(duration=0))],
                        label=str(k)) for k in range(len(frames))],
            active=0,
            transition=dict(duration=0),
            x=0,  # slider 的水平位置
            y=0,  # slider 的垂直位置
            currentvalue=dict(font=dict(size=12), prefix='Time Step: ', visible=True, xanchor='center'),
            len=1.0  # slider 的长度
        )]

        updatemenus = [dict(type='buttons',
                            buttons=[dict(label='Play',
                                          method='animate',
                                          args=[None,
                                                dict(frame=dict(duration=100, redraw=True),
                                                     transition=dict(duration=0),
                                                     fromcurrent=True,
                                                     mode='immediate')])],
                            direction='left',
                            pad=dict(r=10, t=85),
                            showactive=False,
                            x=0.1,
                            y=0,
                            xanchor='right',
                            yanchor='top')]

        # 创建 Figure
        self.fig = go.Figure(data=initial_data, frames=frames)

        # 更新布局
        self.fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=10, range=[-self.R3, self.R3]),
                yaxis=dict(nticks=10, range=[-self.R3, self.R3]),
                zaxis=dict(nticks=10, range=[0, MAX_ALTITUDE]),
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=800,
            height=600,
            title=f'Urban UAV Environment - Episode {episode_num} Animation',
            updatemenus=updatemenus,
            sliders=sliders
        )

        # 显示动画
        pyo.plot(self.fig, filename=f'uav_episode_{episode_num}_animation.html', auto_open=True)

    def get_environment_traces(self):
        # 生成环境元素的可视化轨迹（空域、进近区域、停机坪、环等）
        traces = []

        # 定义角度数组
        theta = np.linspace(0, 2 * np.pi, 100)

        # 最外部的圆柱空域（在 z=0 平面上表示为圆）
        x = self.R3 * np.cos(theta)
        y = self.R3 * np.sin(theta)
        z = np.zeros_like(x)
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='gray', width=2),
            name='Outer Airspace'
        ))

        # 内部进近区域
        x = self.R2 * np.cos(theta)
        y = self.R2 * np.sin(theta)
        z = np.zeros_like(x)
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='orange', width=2),
            name='Approach Area'
        ))

        # 起飞停机坪
        for pad in self.takeoff_pads:
            center = pad['center']
            radius = pad['radius']
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = np.full_like(x, center[2])
            traces.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(color='blue', width=2),
                name='Takeoff Pad'
            ))

        # 降落停机坪
        for pad in self.landing_pads:
            center = pad['center']
            radius = pad['radius']
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            z = np.full_like(x, center[2])
            traces.append(go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='lines',
                line=dict(color='green', width=2),
                name='Landing Pad'
            ))

        # 起飞环
        x = self.takeoff_ring_center[0] + self.R1 * np.cos(theta)
        y = self.takeoff_ring_center[1] + self.R1 * np.sin(theta)
        z = np.full_like(x, self.H1)
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Takeoff Ring'
        ))

        # 降落环
        x = self.landing_ring_center[0] + self.R1 * np.cos(theta)
        y = self.landing_ring_center[1] + self.R1 * np.sin(theta)
        z = np.full_like(x, self.H2)
        traces.append(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            line=dict(color='green', width=2),
            name='Landing Ring'
        ))

        return traces

    def get_drone_color(self, drone):
        if drone.task_type == 'takeoff':
            return 'cyan'  # 青色表示起飞无人机
        elif drone.task_type == 'landing':
            if drone.emergency:
                return 'red'  # 红色表示紧急无人机
            else:
                return 'magenta'  # 品红色表示降落无人机
        else:
            return 'gray'  # 灰色

    def _apply_collision_avoidance(self):
        # 使用空间划分优化邻居查找
        drone_positions = np.array([drone.position for drone in self.drones])
        grid_size = 50.0  # 定义网格大小
        grid_indices = np.floor(drone_positions[:, :2] / grid_size).astype(int)
        grid_dict = {}
        for idx, grid_idx in enumerate(grid_indices):
            key = tuple(grid_idx)
            if key not in grid_dict:
                grid_dict[key] = []
            grid_dict[key].append(idx)

        for i, drone in enumerate(self.drones):
            if i in self.done_drones:
                continue

            # 获取当前无人机所在网格及周围网格的无人机列表
            current_grid = tuple(grid_indices[i])
            neighbor_grids = [current_grid]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_grids.append((current_grid[0] + dx, current_grid[1] + dy))

            potential_neighbors = []
            for grid in neighbor_grids:
                potential_neighbors.extend(grid_dict.get(grid, []))

                # 遍历潜在邻居，进行避撞判断
                for idx in potential_neighbors:
                    if idx == i or idx in self.done_drones:
                        continue
                    neighbor = self.drones[idx]
                    distance_vector = neighbor.position - drone.position
                    distance = np.linalg.norm(distance_vector)
                    min_distance = MIN_SAFE_DISTANCE_MATRIX.get((drone.type, neighbor.type), 10.0)
                    if distance < min_distance:
                        # 发生碰撞
                        drone.r_collision = 1
                        neighbor.r_collision = 1
                        self.done_drones.add(i)
                        self.done_drones.add(idx)
                    elif distance < min_distance + 10.0:
                        # 接近碰撞，需要避让
                        if drone.cooperative:
                            # 预测未来位置，进行避让
                            future_position = drone.position + drone.speed * drone.heading * DT
                            neighbor_future_position = neighbor.position + neighbor.speed * neighbor.heading * DT
                            future_distance = np.linalg.norm(future_position - neighbor_future_position)
                            if future_distance < min_distance:
                                # 调整航向角远离邻近无人机
                                avoidance_direction = -distance_vector / (distance + 1e-6)
                                angle_to_neighbor = np.rad2deg(
                                    np.arctan2(avoidance_direction[1], avoidance_direction[0]))
                                angle_diff = (angle_to_neighbor - drone.heading_angle + 360) % 360
                                if angle_diff > 180:
                                    angle_diff -= 360
                                angle_adjustment = np.clip(angle_diff, -drone.max_turn_rate * DT,
                                                           drone.max_turn_rate * DT)
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

            # 1. 碰撞惩罚
            min_distance = self.get_min_distance(drone)
            safe_distance = MIN_SAFE_DISTANCE_MATRIX.get((drone.type, drone.type), 10.0)
            if min_distance < safe_distance:
                reward -= 10.0  # 严重碰撞
            elif min_distance < safe_distance + 5.0:
                reward -= 5.0   # 接近碰撞
            else:
                # 距离越近，惩罚越大
                reward -= 1.0 * np.exp(-min_distance / (2 * safe_distance))

            # 2. 任务完成奖励
            if drone.task_finished:
                reward += 100.0  # 任务完成奖励

            # 3. 合作无人机的额外奖励
            if drone.cooperative:
                # 成功避让非合作无人机，给予奖励
                if drone.avoided_collision:
                    reward += 5.0

                # 遵守空域规则的奖励
                if not drone.violation:
                    reward += 2.0
            else:
                # 非合作无人机可能不遵守规则，无额外奖励
                pass
            # 4. 能量消耗惩罚
            reward -= drone.energy_consumption * 0.1

            # 5. 违规惩罚
            if drone.violation:
                reward -= 50.0

            rewards.append(reward)
        return rewards

    def _get_observation(self):
        obs = []
        for i, drone in enumerate(self.drones):
            # 自身状态
            own_state = np.concatenate([drone.position, drone.velocity, [drone.energy], [drone.type_index],
                                        [1.0 if drone.cooperative else 0.0], [drone.priority]])

            # 初始化邻居信息列表
            neighbors_info = []
            # 获取邻居列表
            neighbors = self._get_nearest_neighbors(drone, self.num_nearest)
            for neighbor in neighbors:
                relative_position = neighbor.position - drone.position
                relative_velocity = neighbor.velocity - drone.velocity
                neighbor_cooperative_flag = np.array([1.0 if neighbor.cooperative else 0.0])
                if drone.cooperative and neighbor.cooperative:
                    # 可以获取目标信息
                    target_relative_position = neighbor.target - drone.position
                    neighbor_info = np.concatenate(
                        [relative_position, relative_velocity, neighbor_cooperative_flag, target_relative_position])
                else:
                    # 无法获取目标信息，用零填充
                    target_relative_position = np.zeros(3)
                    neighbor_info = np.concatenate(
                        [relative_position, relative_velocity, neighbor_cooperative_flag, target_relative_position])
                neighbors_info.extend(neighbor_info)
            # 填充不足的邻居信息
            expected_length = self.num_nearest * (3 + 3 + 1 + 3)  # 每个邻居信息的长度
            if len(neighbors_info) < expected_length:
                neighbors_info.extend([0.0] * (expected_length - len(neighbors_info)))
            # 组合自身状态和邻居信息
            observation = np.concatenate([own_state, neighbors_info])
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
        # 在降落停机坪内随机选择一个点
        pad = random.choice(self.landing_pads)
        angle = random.uniform(0, 2 * np.pi)
        radius = random.uniform(0, pad['radius'])
        x = pad['center'][0] + radius * np.cos(angle)
        y = pad['center'][1] + radius * np.sin(angle)
        z = 0.0
        return np.array([x, y, z])

    def finish_landing(self, drone: UAV):
        if drone in self.landing_ring_uavs:
            self.landing_ring_uavs.remove(drone)

    def has_emergency_uav(self):
        return self.emergency_uav_exists

    def set_emergency_uav_exists(self, exists: bool):
        self.emergency_uav_exists = exists

    def get_approach_area_center(self):
        return self.approach_area_center

    def add_to_approach_queue(self, drone: UAV):
        # 紧急无人机插入队列最前面
        if drone.emergency:
            self.approach_queue.insert(0, drone)
            drone.priority = 999  # 紧急无人机最高优先级
        else:
            self.approach_queue.append(drone)
            # 优先级根据进入进近空域的时间分配，进入越早，值越大
            drone.priority = -drone.entered_approach_time  # 时间越小，值越大

    def get_min_distance(self, drone):
        distances = []
        for other_drone in self.drones:
            if other_drone.drone_id != drone.drone_id:
                distance = np.linalg.norm(drone.position - other_drone.position)
                distances.append(distance)
        if distances:
            return min(distances)
        else:
            return np.inf

    def close(self):
        pass
