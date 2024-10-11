import numpy as np


def select_takeoff_zone(self):
    # 遍历所有起飞区域，选择一个最适合的
    available_takeoff_zones = [zone for zone in self.runways['takeoff'] if not zone['occupied']]

    if len(available_takeoff_zones) == 0:
        raise ValueError("No available takeoff zones.")

    # 假设当前无人机的位置存储在 self.current_position
    current_position = self.current_position[:2]

    # 按照离当前位置的距离选择最近的起飞区域
    selected_zone = min(available_takeoff_zones,
                        key=lambda zone: np.linalg.norm(current_position - zone['center'][:2]))

    # 标记该区域为占用状态
    selected_zone['occupied'] = True

    return selected_zone


def select_landing_zone(self):
    # 遍历所有降落区域，选择一个最适合的
    available_landing_zones = [zone for zone in self.runways['landing'] if not zone['occupied']]

    if len(available_landing_zones) == 0:
        raise ValueError("No available landing zones.")

    # 假设当前无人机的位置存储在 self.current_position
    current_position = self.current_position[:2]

    # 按照离当前位置的距离选择最近的降落区域
    selected_zone = min(available_landing_zones,
                        key=lambda zone: np.linalg.norm(current_position - zone['center'][:2]))

    # 标记该区域为占用状态
    selected_zone['occupied'] = True

    return selected_zone

