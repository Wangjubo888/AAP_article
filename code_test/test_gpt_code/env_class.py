import numpy as np
import time
import random
from shapely.geometry import Point, Polygon
from definition import Drone


class UAVEnvironment:
    def __init__(self, max_drones=20):
        # 环境参数
        self.takeoff_fields = [np.array([50, 50, 0]), np.array([-50, -50, 0])]  # 用于起飞的两个起降场
        self.landing_fields = [np.array([50, -50, 0]), np.array([-50, 50, 0])]  # 用于降落的两个起降场
        self.field_cooldown = {0: 0, 1: 0, 2: 0, 3: 0}  # 每个起降场的冷却时间
        self.field_cooldown_time = 60  # 冷却时间为60秒
        self.field_radius = 25
        self.cylinder_radius = 1000
        self.cylinder_height = 1500
        self.vertiport_center = np.array([0, 0, 0])
        self.vertiport_four_center = [np.array([25, 25, 0]), np.array([-25, -25, 0])
            , np.array([25, -25, 0]), np.array([-25, 25, 0])]
        # 进近空域
        self.approach_zone_radius = 250
        self.approach_queue = []
        # 起飞和降落环配置
        self.takeoff_ring_radius = 150
        self.landing_ring_radius = 200
        self.takeoff_ring_height = 300
        self.landing_ring_height = 150
        self.takeoff_ring_capacity = 6
        self.landing_ring_capacity = 8
        self.takeoff_ring_queue = []
        self.landing_ring_queue = []
        # 初始化无人机数量
        self.num_drones = max_drones
        self.num_cooperative = np.random.randint(0, max_drones + 1)
        self.num_non_cooperative = self.num_drones - self.num_cooperative

        # 无人机状态
        self.uav_states = []
        self.assigned_targets = {}
        self.cooperative_uavs = list(range(self.num_cooperative))
        self.non_cooperative_uavs = list(range(self.num_cooperative, self.num_drones))
        self.conflicts = []
        # 状态空间和动作空间
        self.observation_space = 6 + self.num_drones * 3  # 自己的位置和速度 + 其他无人机的相对位置
        self.action_space = 3  # 加速度 (ax, ay, az)

        self.step_count = 0
        self.conflict_count = 0
        self.uav = Drone()

    def reset(self):
        """重置环境，初始化无人机的状态和目标
        :return: self.uav_states
        """
        self.uav_states = []
        for _ in range(self.num_cooperative + self.num_non_cooperative):
            uav_state = self.init_uav_state()
            self.uav_states.append(uav_state)
        self.step_count = 0
        self.conflict_count = 0
        self.conflicts = []
        self.assigned_targets = {}
        self.takeoff_ring_queue = []
        self.landing_ring_queue = []
        self.approach_queue = []
        return self.uav_states

    def init_uav_state(self):
        """初始化无人机状态
        :return: [896.41298795 443.21975929 189.49440078   9.19848037   9.45412977 \
        3.01983434 100. 1.]
        """
        position = self._spawn_on_surface()
        velocity = np.random.uniform(0, self.max_speed, 3)
        battery = 100
        task_type = 1
        return np.concatenate([position, velocity, [battery, task_type]])

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

    def _generate_destination(self, uav_id):
        # 生成目的地，根据任务类型
        if self.task_type == 'takeoff':
            return self._spawn_on_surface()
        elif self.task_type == 'landing':
            return self.assign_landing_target(uav_id)

    def update(self, state, control_input, dt):
        # 状态更新，包括位置、速度、姿态等
        position, velocity, orientation = state
        thrust, tilt = control_input

        # 计算加速度
        acceleration = np.array([
            thrust * np.sin(tilt) / self.mass,
            thrust * np.cos(tilt) / self.mass - 9.81,  # 考虑重力
            0  # 简化的二维模型
        ]) - self.drag_coefficient * velocity

        # 更新速度和位置
        velocity += acceleration * dt
        position += velocity * dt

        # 更新姿态
        orientation += tilt * dt

        return position, velocity, orientation

    def get_observation(self, uav_id):
        """
        获取指定无人机的观测信息
        对合作无人机返回更多信息（目标位置等）
        对非合作无人机仅返回相对位置和速度
        """
        uav = self.uav_states[uav_id]
        observation = list(uav['position']) + list(uav['velocity'])

        for i in range(self.num_uavs):
            if i != uav_id:
                relative_position = self.uav_states[i]['position'] - uav['position']
                relative_velocity = self.uav_states[i]['velocity'] - uav['velocity']
                observation.extend(relative_position)
                observation.extend(relative_velocity)

                # 如果是合作无人机，能获取对方的目标位置
                if uav_id in self.cooperative_uavs and i in self.cooperative_uavs:
                    target_position = self.assigned_targets[i]
                    observation.extend(target_position)

        return np.array(observation)

    def check_field_cooldown(self, field_id):
        """检查起降场的冷却时间"""
        return time.time() > self.field_cooldown[field_id]

    def assign_takeoff_target(self, uav_id):
        """为起飞任务的无人机分配目标点，并进入起飞环"""
        task_type = self.uav_states[uav_id]['task_type']
        if task_type != 0:
            return

        if uav_id not in self.assigned_targets and len(self.takeoff_ring_queue) < self.takeoff_ring_capacity:
            target = self.generate_target_for_takeoff(self.takeoff_fields[np.random.choice([0, 1])]
                                                      , self.takeoff_ring_radius, self.cylinder_height)
            self.assigned_targets[uav_id] = target
            self.takeoff_ring_queue.append(uav_id)
            print(f"为无人机 {uav_id} 分配的目标点: {target}")

    def generate_target_for_takeoff(self, radius, height):
        """为起飞任务生成随机目标"""
        center = self.vertiport_center
        theta = np.random.uniform(0, 2 * np.pi)
        x = center + radius * np.cos(theta)
        y = center + radius * np.sin(theta)
        z = np.random.uniform(0, height)
        return np.array([x, y, z])

    def assign_landing_target(self, uav_id):
        """为降落任务的无人机分配降落点，并进入降落环"""
        task_type = self.uav_states[uav_id]['task_type']
        if task_type != 1:
            return

        if uav_id not in self.assigned_targets and len(self.landing_ring_queue) < self.landing_ring_capacity:
            target = self.landing_fields[np.random.choice([0, 1])]
            self.assigned_targets[uav_id] = target
            self.landing_ring_queue.append(uav_id)
            print(f"为无人机 {uav_id} 分配的降落点: {target}")

    def enter_approach_zone(self, uav_id):
        """检查无人机是否进入进近空域，进入则分配序号"""
        task_type = self.uav_states[uav_id]['task_type']
        if task_type != 1:
            return

        position = self.uav_states[uav_id]['position']
        distance_to_center = np.linalg.norm(position - np.array([0, 0, 0]))

        if distance_to_center < self.approach_zone_radius and uav_id not in self.approach_queue:
            self.approach_queue.append(uav_id)
            print(f"无人机 {uav_id} 进入进近空域，分配序号 {len(self.approach_queue)}")

    def step(self, actions):
        """执行无人机的动作（加速度）"""
        for i, action in enumerate(actions):
            acceleration = np.clip(action, -self.max_acceleration, self.max_acceleration)
            velocity = self.uav_states[i]['velocity'] + acceleration * self.dt
            velocity = np.clip(velocity, -self.max_speed, self.max_speed)
            self.uav_states[i]['position'] += velocity * self.dt
            self.uav_states[i]['velocity'] = velocity
            self.uav_states[i]['battery'] -= np.linalg.norm(velocity) * self.dt

            if self.uav_states[i]['task_type'] == 1:
                self.enter_approach_zone(i)

        self.check_conflicts()

        rewards = [self.calculate_reward(i) for i in range(self.num_uavs)]
        done = all([uav['battery'] <= 0 for uav in self.uav_states])
        return [self.get_observation(i) for i in range(self.num_uavs)], rewards, done, {}

    def check_conflicts(self):
        """检查是否有无人机之间的冲突（距离过近）"""
        conflict_threshold = 5.0
        self.conflicts = []
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                distance = np.linalg.norm(self.uav_states[i]['position'] - self.uav_states[j]['position'])
                if distance < conflict_threshold:
                    self.conflicts.append((i, j))

    def calculate_reward(self, uav_id):
        """计算奖励函数"""
        uav_state = self.uav_states[uav_id]
        position = uav_state['position']
        battery = uav_state['battery']
        task_type = uav_state['task_type']
        reward = 0

        if any(uav_id in pair for pair in self.conflicts):
            reward -= 100

        if task_type == 0:
            target = self.assigned_targets[uav_id]
            distance_to_target = np.linalg.norm(target - position)
            if distance_to_target < 5.0:
                reward += 100
                self.takeoff_ring_queue.remove(uav_id)
                print(f"无人机 {uav_id} 完成起飞任务。")
            else:
                reward += (self.cylinder_radius - distance_to_target)

        if task_type == 1:
            target = self.assigned_targets[uav_id]
            distance_to_target = np.linalg.norm(target - position)
            if distance_to_target < 5.0:
                reward += 100
                self.landing_ring_queue.remove(uav_id)
                print(f"无人机 {uav_id} 完成降落任务。")

        reward += max(0, battery) * 0.1
        return reward
