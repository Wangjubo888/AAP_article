from dataclasses import dataclass, field
import numpy as np


@dataclass
class Destination:
    dt: float  # 时间步长
    radius: float = 100.0  # 起降场区域半径
    t_close: float = 60.0  # 关闭时间（秒）
    t_nxt_open: float = field(default=0.0, init=False)  # 距离下一次开放的时间
    t_open_since: float = field(default=0.0, init=False)  # 已经开放的持续时间
    is_open: bool = field(default=True, init=False)  # 目的地是否开放
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # 目的地位置



    def reset(self):
        self.open()

    def step(self):
        if not self.is_open:
            self.t_nxt_open -= self.dt
            if self.t_nxt_open <= 0:
                self.open()
        else:
            self.t_open_since += self.dt

    def open(self):
        self.t_open_since = 0.0
        self.t_nxt_open = 0.0
        self.is_open = True

    def close(self):
        self.t_open_since = 0.0
        self.is_open = False
        self.t_nxt_open = self.t_close