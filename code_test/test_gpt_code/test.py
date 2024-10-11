import unittest
from env_class import UAVEnvironment
import numpy as np

class TestUAVEnvironment(unittest.TestCase):

    def setUp(self):
        # 假设的初始化环境设置
        self.env = UAVEnvironment(2, 0)
        self.env.uav_states = {
            0: {'position': np.array([0, 0, 0]), 'velocity': np.array([1, 1, 1])},
            1: {'position': np.array([10, 10, 10]), 'velocity': np.array([2, 2, 2])},
            # 增加更多无人机状态，如果需要
        }
        self.env.num_uavs = 2
        self.env.cooperative_uavs = {0, 1}  # 假设所有无人机都是合作的
        self.env.assigned_targets = {
            0: np.array([100, 100, 100]),
            1: np.array([200, 200, 200]),
            # 增加更多目标位置，如果需要
        }

    def test_get_observation(self):
        # 测试无人机0的观测值
        observation1 = self.env.get_observation(0)
        print(f"1 is {observation1}")
        expected_observation = np.array([
            0, 0, 0,  # 无人机0的位置
            1, 1, 1,  # 无人机0的速度
            10, 10, 10,  # 无人机1相对于无人机0的位置
            1, 1, 1,  # 无人机1相对于无人机0的速度
            200, 200, 200,  # 无人机1的目标位置
        ])
        np.testing.assert_array_equal(observation1, expected_observation, "Observation for UAV 0 is incorrect")

        # 测试无人机1的观测值
        observation2 = self.env.get_observation(1)
        print(observation2)
        expected_observation = np.array([
            10, 10, 10,  # 无人机1的位置
            2, 2, 2,  # 无人机1的速度
            -10, -10, -10,  # 无人机0相对于无人机1的位置
            -1, -1, -1,  # 无人机0相对于无人机1的速度
            100, 100, 100,  # 无人机1的目标位置
        ])
        np.testing.assert_array_equal(observation2, expected_observation, "Observation for UAV 1 is incorrect")

    def tearDown(self):
        self.env = None

if __name__ == '__main__':
    unittest.main()
