import gym
import numpy as np
from gym import spaces
import random
import math
import time

# 0: 起飞, 1: 降落


class UAVEnv(gym.Env):
    """多智能体无人机环境"""

    def __init__(self,
                 max_tol_agents=15,  # 最大无人机数量
                 max_airspace_area=1000,
                 max_steps=1000,
                 time_step=0.1,
                 max_speed=20,  # 最大速度 m/s
                 max_turn_rate=np.pi / 6,  # 最大转向速率（每步） radians
                 max_x_drate=2,  # 最大爬升/下降速率 m/s
                 max_y_drate=2,  # 最大爬升/下降速率 m/s
                 max_z_drate=2,  # 最大爬升/下降速率 m/s
                 landing_pad_cooldown=10,  # 起降场使用后冷却时间s
                 max_wind_speed=10,
                 wind_speed_range=2,
                 range_detection=100,   # 检测距离
                 min_distance=25):
        super(UAVEnv, self).__init__()
        self.start_time = time.time()  # 开始时间
        self.conflict_count = 0  # 冲突计数
        self.total_energy_consumption = 0  # 总能量消耗
        self.tasks_completed = 0  # 完成的任务数
        self.total_distance_traveled = 0  # 无人机总飞行距离

        self.num_coop_agents = random.randint(1, max_tol_agents)
        self.num_noncoop_agents = random.randint(0, max_tol_agents - self.num_coop_agents)
        self.max_airspace_area = max_airspace_area
        self.max_steps = max_steps
        self.time_step = time_step  # 时间步长
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.max_x_rate = max_x_drate
        self.max_y_rate = max_y_drate
        self.max_z_rate = max_z_drate
        self.landing_pad_cooldown = landing_pad_cooldown

        self.range_detection = range_detection
        self.range_warning = min_distance * 2  # radius of the warning area
        self.range_protection = min_distance  # radius of the protection area

        # 环境空间定义
        self.cylinder_radius = 1000  # 圆柱体空域的半径,m
        self.cylinder_height = 1500  # 圆柱体空域的高度,m
        self.waiting_radius = 300  # 等待区半径,m
        self.landing_radius = 10  # 降落区半径,m
        self.min_distance = 25  # 无人机之间的最小安全距离,m
        self.assigned_targets = {}

        # 提前初始化 vertiport_center，给一个默认值
        self.vertiport_center = np.array([0, 0, 0])
        self.vertiport = self._define_vertiports()

        # 状态空间：位置(x, y, z)，速度，电量，任务类型(起飞: 0 或降落: 1)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -5, -5, -5, 0, 0]),
            high=np.array([self.cylinder_radius * 2, self.cylinder_radius * 2, self.cylinder_height, 5, 5, 5, 100, 1]),
            dtype=np.float32
        )

        # 动作空间：控制加速度和角度变化
        self.action_space = spaces.Box(
            low=np.array([-self.max_x_rate, -self.max_y_rate, -self.max_z_rate, -self.max_turn_rate]),
            high=np.array([self.max_x_rate, self.max_y_rate, self.max_z_rate, self.max_turn_rate]),
            dtype=np.float32
        )

        # 风速和不确定性
        self.max_wind_speed = max_wind_speed
        self.wind_speed_range = wind_speed_range
        self.reference_wind_heading = random.uniform(0, 2 * math.pi)
        self.reference_wind_speed = random.uniform(0, self.max_wind_speed)

        self.intention_matrix = np.array([_ for _ in range(27)]).reshape([3, 3, 3])
        self.reset()

    def reset(self):
        """重置环境"""
        self.step_count = 0
        # 随机生成起降场中心
        self.vertiport_center = self._get_random_point_in_cylinder()
        self.uav_states = []  # 存储所有无人机的初始状态
        for _ in range(self.num_coop_agents + self.num_noncoop_agents):
            uav_state = self.init_uav_state()
            self.uav_states.append(uav_state)

        # 初始化起降场状态
        self.runway_status = {'takeoff': [0, 0], 'landing': [0, 0]}  # 记录每个起降场的冷却时间

        # 清空冲突和警告集合
        self.conflicts = set()
        self.warnings = set()
        self.done = set()

        # 初始化风速
        self.update_wind()
        return self.uav_states

    def update_wind(self):
        """更新风速和风向"""
        self.reference_wind_heading = random.uniform(0, 2 * math.pi)
        self.reference_wind_speed = np.clip(random.normalvariate(0, 1) * self.wind_speed_range, 0, self.max_wind_speed)
        self.wind_speed = np.array([self.reference_wind_speed * math.cos(self.reference_wind_heading),
                                    self.reference_wind_speed * math.sin(self.reference_wind_heading),
                                    0])  # 假设风速只在XY平面

    def init_uav_state(self):
        """初始化无人机状态"""
        position = self._spawn_on_surface()
        velocity = np.random.uniform(0, self.max_speed, 3)
        battery = 100
        task_type = 1
        return np.concatenate([position, velocity, [battery, task_type]])

    def step(self, actions):
        """更新环境"""
        self.step_count += 1
        new_uav_states = []

        # 更新起降场的冷却状态
        for pad_type in self.runway_status:
            self.runway_status[pad_type] = [max(0, cooldown - 1) for cooldown in self.runway_status[pad_type]]

        for i in range(len(self.uav_states)):
            new_state, task_completed = self.update_uav_state(self.uav_states[i], actions[i])
            # 只保留未完成任务的无人机
            if not task_completed:
                new_uav_states.append(new_state)

        self.uav_states = new_uav_states

        # 更新冲突状态
        self.update_conflicts()

        done = self.step_count >= self.max_steps or len(self.uav_states) == 0
        rewards = self.calculate_rewards()
        return self.uav_states, rewards, done, {}

    def update_uav_state(self, state, action):
        """根据动作更新无人机状态"""
        position, velocity, battery, task_type = state[:3], state[3:6], state[6], state[7]
        acceleration, angle_change = action[:3], action[3]

        # 限制加速度（更严格的限制）
        max_acceleration = 2  # 限制最大加速度
        acceleration_magnitude = np.linalg.norm(acceleration)
        if acceleration_magnitude > max_acceleration:
            acceleration = acceleration / acceleration_magnitude * max_acceleration

        # 限制转向变化
        angle_change = np.clip(angle_change, -self.max_turn_rate, self.max_turn_rate)

        # 限制爬升/下降速率
        acceleration[2] = np.clip(acceleration[2], -self.max_climb_rate, self.max_climb_rate)

        # 计算相对速度（考虑风速的扰动）
        relative_velocity = velocity - self.wind_speed  # 无人机相对风速的速度

        # 更新速度（使用无人机的相对速度，并加上加速度的影响）
        new_velocity = relative_velocity + acceleration * self.time_step  # 新速度

        # 限制速度
        max_velocity = 10  # 限制最大速度
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > max_velocity:
            new_velocity = new_velocity / velocity_magnitude * max_velocity

        # 重新计算实际速度（加入风速）
        velocity = new_velocity + self.wind_speed  # 最终的速度

        # 更新位置（基于新的速度）
        position += velocity * self.time_step  # 位置更新
        trajectory_length = ((velocity[0]*self.time_step)**2 + (velocity[1]*self.time_step)**2
                             + (velocity[2]*self.time_step)**2)**0.5
        # 电量消耗
        power_consumption = 0.1 * np.linalg.norm(acceleration) ** 2 + 0.05 * np.linalg.norm(velocity) ** 2
        battery -= power_consumption * self.time_step  # 电量消耗更新

        # 检查任务完成条件
        task_completed = self.execute_task(task_type, position)

        # 位置保持在圆柱体范围内
        position[:2] = np.clip(position[:2], -self.cylinder_radius, self.cylinder_radius)
        position[2] = np.clip(position[2], 0, self.cylinder_height)
        battery = max(0, battery)  # 电量不低于0

        return np.concatenate([position, velocity, [battery, task_type]]), task_completed

    def execute_task(self, task_type, position):
        # 自主学习选择合适的起降区域
        task_completed = False

        if task_type == 0:  # 起飞任务
            # 自主选择一个起飞区域
            assigned_takeoff_zone = self.select_takeoff_zone(position)  # 假设有一个方法自主选择起飞区域
            if np.linalg.norm(position[:2] - assigned_takeoff_zone['center'][:2]) > assigned_takeoff_zone['radius']:
                task_completed = True
                self.close_runway('takeoff')

        elif task_type == 1:  # 降落任务
            # 自主选择一个降落区域
            assigned_landing_zone = self.select_landing_zone(position)
            if assigned_landing_zone is not None:
                landing_center = assigned_landing_zone['center']
                landing_radius = assigned_landing_zone['radius']

                # 降落任务完成条件
                if np.linalg.norm(position[:2] - landing_center[:2]) < landing_radius and position[2] <= 0:
                    task_completed = True
                    self.close_runway('landing')
                    assigned_landing_zone['occupied'] = False  # 任务完成后标记该降落区域为空闲

        return task_completed

    def select_takeoff_zone(self, current_position):
        # 遍历所有起飞区域，选择一个最适合的，可以设置策略
        available_takeoff_zones = [zone for zone in self.vertiport['takeoff'] if not zone['occupied']]

        if len(available_takeoff_zones) == 0:
            raise ValueError("No available takeoff zones.")

        # 假设当前无人机的位置存储在 self.current_position
        current_position = current_position[:2]

        # 按照离当前位置的距离选择最近的起飞区域
        selected_zone = min(available_takeoff_zones,
                            key=lambda zone: np.linalg.norm(current_position - zone['center'][:2]))

        # 标记该区域为占用状态
        selected_zone['occupied'] = True

        return selected_zone

    def select_landing_zone(self, current_position):
        # 遍历所有降落区域，选择一个最适合的
        available_landing_zones = [zone for zone in self.vertiport['landing'] if not zone['occupied']]

        if len(available_landing_zones) == 0:
            raise ValueError("No available landing zones.")

        # 假设当前无人机的位置存储在 self.current_position
        current_position = current_position[:2]

        # 按照离当前位置的距离选择最近的降落区域
        selected_zone = min(available_landing_zones,
                            key=lambda zone: np.linalg.norm(current_position - zone['center'][:2]))

        # 标记该区域为占用状态
        selected_zone['occupied'] = True

        return selected_zone

    def _define_vertiports(self):
        """定义四个内切小圆（起飞和降落场地）"""
        self.vertiport_center = (0, 0, 0)
        zones = {
            'takeoff': [
                {'center': self.vertiport_center + np.array([-2 * self.landing_radius, 2 * self.landing_radius, 0])
                    , 'radius': self.landing_radius, 'occupied': False},
                {'center': self.vertiport_center + np.array([2 * self.landing_radius, 2 * self.landing_radius, 0])
                    , 'radius': self.landing_radius, 'occupied': False},
            ],
            'landing': [
                {'center': self.vertiport_center + np.array([-2 * self.landing_radius, -2 * self.landing_radius, 0])
                    , 'radius': self.landing_radius, 'occupied': False},
                {'center': self.vertiport_center + np.array([2 * self.landing_radius, -2 * self.landing_radius, 0])
                    , 'radius': self.landing_radius, 'occupied': False},
            ]
        }
        return zones

    def close_runway(self, runway_type):
        """关闭起降场一定时间"""
        for i in range(len(self.runway_status[runway_type])):
            if self.runway_status[runway_type][i] == 0:
                self.runway_status[runway_type][i] = self.landing_pad_cooldown
                break

    def generate_target_for_takeoff(self, center, radius, height):
        """
        为无人机生成圆柱空域侧面上的随机目的地坐标。
        参数:
            center (tuple): 圆柱的底面中心坐标 (x, y)。
            radius (float): 圆柱的半径。
            height (float): 圆柱的高度。
        返回:
            tuple: 随机生成的目的地坐标 (x, y, z)。
        """
        # 随机生成角度，范围是 [0, 2π]，以决定点在圆周上的位置
        theta = np.random.uniform(0, 2 * np.pi)

        # 计算圆柱侧面上的 (x, y) 坐标，圆心位于 center
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # 随机生成高度 z 坐标，范围在 [0, height] 之间
        z = np.random.uniform(0, height)

        return x, y, z

    def assign_takeoff_target(self, uav_id):
        """
        为每个执行起飞任务的无人机分配目标点。
        参数:
            uav_id (int): 无人机的唯一标识符
        """
        # 如果该无人机还没有被分配目标点，生成目标并存储
        if uav_id not in self.assigned_targets:
            target = self.generate_target_for_takeoff(self.vertiport_center,
                                                      self.cylinder_radius,
                                                      self.cylinder_height)
            self.assigned_targets[uav_id] = target
            print(f"为无人机 {uav_id} 分配的目标点: {target}")
        else:
            print(f"无人机 {uav_id} 已有目标点: {self.assigned_targets[uav_id]}")

    def calculate_rewards(self):
        """计算奖励函数"""
        rewards = []
        for i, uav_state in enumerate(self.uav_states):
            position = uav_state[:3]
            battery = uav_state[6]
            task_type = uav_state[7]
            reward = 0

            # 冲突惩罚
            if i in self.conflicts:
                reward -= 100

            # 起飞任务成功奖励
            if task_type == 0:
                # 确保每架无人机有一个分配的目标
                self.assign_takeoff_target(i)

                # 获取分配给该无人机的目标点
                target = self.assigned_targets[i]
                distance_to_target = np.linalg.norm(np.array(target[:2]) - np.array(position[:2]))  # 使用 2D 距离
                reward += (self.cylinder_radius - distance_to_target)

            # 降落任务成功奖励
            elif task_type == 1:
                distance_to_landing = np.linalg.norm(np.array(position[:2]) - np.array(self.vertiport_center[:2]))
                if distance_to_landing < self.landing_radius and position[2] <= 0:
                    reward += 100
                else:
                    # 增加距离奖励，鼓励靠近降落区域
                    reward += (self.landing_radius - distance_to_landing) * 10

            # 电量奖励
            if battery > 20:
                reward += battery * 0.5  # 电量越高，奖励越多
            else:
                reward -= (20 - battery) * 2.5  # 电量越低，惩罚越重

            rewards.append(reward)
        return rewards

    def calculate_efficiency_metrics(self):
        elapsed_time = time.time() - self.start_time  # 计算运行时间
        avg_energy_consumption = self.total_energy_consumption / (self.tasks_completed + 1e-5)  # 防止除零
        avg_distance_traveled = self.total_distance_traveled / (self.tasks_completed + 1e-5)

        efficiency_metrics = {
            "Elapsed Time (s)": elapsed_time,
            "Tasks Completed": self.tasks_completed,
            "Conflict Count": self.conflict_count,
            "Average Energy Consumption": avg_energy_consumption,
            "Average Distance Traveled": avg_distance_traveled
        }

        return efficiency_metrics

    def update_conflicts(self):
        """更新冲突和警告状态"""
        # self.conflicts = set()  # set of flights that are in conflict in a step
        # self.warning = set()  # set of warning that are in conflict in a step
        # self.period_conflicts = set()  # set of flights that are in conflict in an action cycle
        # self.period_warning = set()  # set of flights that are in warning in an action cycle
        self.conflicts = set()

        for i in range(len(self.uav_states)):
            for j in range(i + 1, len(self.uav_states)):
                distance = np.linalg.norm(self.uav_states[i][:3] - self.uav_states[j][:3])
                if distance < self.min_distance:
                    self.conflicts.update([i, j])

    def _get_random_point_in_cylinder(self):
        """在圆柱体内随机生成一点"""
        theta = random.uniform(0, 2 * np.pi)
        r = (self.cylinder_radius - 5) * np.sqrt(random.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = random.uniform(0, self.cylinder_height)
        return np.array([x, y, z])

    def _get_random_point_for_runway(self):
        """在圆柱体内随机生成一点"""
        theta = random.uniform(0, 2 * np.pi)
        r = self.cylinder_radius * np.sqrt(random.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = 0
        return np.array([x, y, z])
    def _spawn_on_surface(self):
        """在圆柱体表面生成一点"""
        theta = random.uniform(0, 2 * np.pi)
        x = self.cylinder_radius * np.cos(theta)
        y = self.cylinder_radius * np.sin(theta)
        z = random.choice([random.uniform(0, self.cylinder_height), 0])  # 顶面或底面
        return np.array([x, y, z])

