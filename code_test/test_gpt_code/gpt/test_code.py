import unittest
import numpy as np
from code_gpt import UAV  # 假设UAV类在code_gpt模块中


class TestUAV(unittest.TestCase):
    def test_components(self):
        # 创建一个UAV实例
        uav = UAV()
        # 假设初始化时设置了以下属性值
        uav.speed = 10  # 假设速度为10单位
        uav.heading = np.array([1, 1, 1]) / np.sqrt(3)  # 假设航向为等分量的X、Y、Z方向

        # 调用components方法
        components = uav.components()

        # 验证components方法的返回是否为NumPy数组
        self.assertIsInstance(components, np.ndarray)

        # 验证返回的数组大小是否正确
        self.assertEqual(components.size, 3)

        # 验证返回的数组内容是否正确，这里预期是[10/sqrt(3), 10/sqrt(3), 10/sqrt(3)]
        expected_components = np.array([10 / np.sqrt(3), 10 / np.sqrt(3), 10 / np.sqrt(3)])
        np.testing.assert_array_almost_equal(components, expected_components, decimal=6)


if __name__ == '__main__':
    unittest.main()
