import unittest
from unittest.mock import patch
import numpy as np
from gpt.definitions import UAV  # 假设类定义在这个模块中
import random

UAV_TYPES = ['multirotor', 'light_hybrid_wing']
MIN_SPEED = 50.0
MAX_SPEED = 60.0

class MockEnvironment:
    def __init__(self):
        self.min_altitude = 10.0
        self.max_altitude = 100.0
        self.takeoff_pads_positions = {'pad1': np.array([0, 0, 0]), 'pad2': np.array([10, 10, 0])}
        self.takeoff_ring_center = np.array([50, 50, 0])
        self.takeoff_ring_altitude = 50.0
        self.landing_approach_point = np.array([100, 100, 50])
        self.landing_sequence = 1

    def get_idle_takeoff_pads(self):
        return list(self.takeoff_pads_positions.keys())

    def get_next_landing_sequence(self):
        sequence_number = self.landing_sequence
        self.landing_sequence += 1
        return sequence_number

def mock_random_choice(choices):
    # 修正：根据场景返回不同的 mock 值
    if 'pad1' in choices:
        return 'pad1'  # 起飞场选择
    return 'multirotor'  # 无人机类型选择

def mock_random_uniform(a, b):
    return (a + b) / 2.0  # 返回中间值

class TestUAV(unittest.TestCase):
    def setUp(self):
        self.environment = MockEnvironment()

    @patch('gpt.definitions.random.choice', side_effect=mock_random_choice)
    @patch('gpt.definitions.random.uniform', side_effect=mock_random_uniform)
    def test_random_uav_takeoff(self, mock_uniform, mock_choice):
        uav = UAV.random_uav((100, 100), 1, 'takeoff', True, self.environment, uav_type='multirotor')

        self.assertIsNotNone(uav)
        self.assertEqual(uav.task_type, 'takeoff')
        self.assertEqual(uav.uav_type, 'multirotor')
        self.assertEqual(uav.optimal_speed, 10.0)  # 检查速度
        self.assertTrue(np.array_equal(uav.position, self.environment.takeoff_pads_positions['pad1']))

        expected_target = self.environment.takeoff_ring_center.copy()
        expected_target[2] = self.environment.takeoff_ring_altitude
        self.assertTrue(np.array_equal(uav.target, expected_target))

        self.assertEqual(uav.takeoff_pad_assigned, 'pad1')

    @patch('gpt.definitions.random.choice', side_effect=mock_random_choice)
    @patch('gpt.definitions.random.uniform', side_effect=mock_random_uniform)
    def test_random_uav_landing(self, mock_uniform, mock_choice):
        uav = UAV.random_uav((100, 100), 2, 'landing', True, self.environment, uav_type='light_hybrid_wing')
        print(uav)
        self.assertIsNotNone(uav)
        self.assertEqual(uav.task_type, 'landing')
        self.assertEqual(uav.uav_type, 'light_hybrid_wing')
        self.assertEqual(uav.optimal_speed, 10.0)

        position = uav.position
        target = uav.target

        self.assertTrue(np.linalg.norm(position) <= 100)  # 位置是否在合理范围内
        self.assertTrue(np.array_equal(target, self.environment.landing_approach_point))

        self.assertEqual(uav.sequence_number, 1)

if __name__ == '__main__':
    unittest.main()
