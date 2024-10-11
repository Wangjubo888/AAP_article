import numpy as np
import time
import Config

class SimpleUAVEnv:
    def __init__(self, num_agents, R1, R2, R3):
        self.num_agents = num_agents
        self.R1, self.R2, self.R3 = R1, R2, R3
        self.state_dim = 12  # 定义状态维度: 位置（3），速度（3），航向（1），任务状态（1），距离（4）
        self.action_dim = 4  # 动作空间：速度调节(1)，方向(3)
        self.agents = [self._init_agent(i) for i in range(num_agents)]
        self.obstacles = self._init_obstacles()
        self.last_landing_time = [0] * num_agents  # 记录每个无人机上次着陆的时间

    def _init_agent(self, agent_id):
        return {
            'position': np.random.uniform(low=-self.R3, high=self.R3, size=(3,)),  # 初始位置随机生成
            'velocity': np.array([0.0, 0.0, 0.0]),
            'heading': 0.0,
            'task': 'takeoff',
            'name': f"drone_{agent_id}"
        }

    def _init_obstacles(self):
        # 初始化静态和动态障碍物
        return [np.random.uniform(low=-self.R3, high=self.R3, size=(3,)) for _ in range(10)]

    def step(self, actions):
        rewards, dones, next_states = [], [], []
        current_time = time.time()
        for i, action in enumerate(actions):
            reward, done, next_state = self._update_agent(i, action, current_time)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
        return next_states, rewards, dones

    def _update_agent(self, agent_idx, action, current_time):
        agent = self.agents[agent_idx]

        # 更新位置和速度
        vel_cmd = action['velocity']
        agent['position'] += vel_cmd * 1  # 模拟1秒内的动作
        agent['velocity'] = vel_cmd
        agent['heading'] = action['heading']

        reward, done = self._check_agent_status(agent, agent_idx, current_time)
        return reward, done, agent

    def _check_agent_status(self, agent, agent_idx, current_time):
        # 计算任务完成奖励、避障奖励和碰撞惩罚
        position = agent['position']
        if np.linalg.norm(position) > self.R3:
            return -100, True  # 出界
        reward = -0.1 * np.linalg.norm(agent['velocity'])  # 消耗能量惩罚

        # 检查是否可以降落，考虑起降场的容量问题
        if agent['task'] == 'land' and np.linalg.norm(position[:2]) < self.R1:
            if current_time - self.last_landing_time[agent_idx] >= Config.landing_interval:
                self.last_landing_time[agent_idx] = current_time
                return 100, True  # 成功降落
            else:
                reward -= 50  # 如果不能降落，施加惩罚
        return reward, False

    def reset(self):
        self.agents = [self._init_agent(i) for i in range(self.num_agents)]
        self.last_landing_time = [0] * self.num_agents  # 重置每个无人机的上次着陆时间
        return self.agents
