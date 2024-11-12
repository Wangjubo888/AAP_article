import numpy as np


class RiskAssessment:
    def __init__(self, drones, collision_threshold=50.0):
        self.drones = drones
        self.collision_threshold = collision_threshold

    def assess_risk(self):
        density = self.calculate_density()
        risk_level = self.evaluate_risk_level(density)
        return risk_level

    def calculate_density(self):
        # 根据无人机位置计算空域密度
        positions = np.array([drone.position for drone in self.drones])
        density = np.sum(np.linalg.norm(positions[:, None] - positions[None, :], axis=-1) < self.collision_threshold) / len(self.drones)
        return density

    def evaluate_risk_level(self, density):
        # 根据密度评估风险级别
        if density < 0.2:
            return "low"
        elif density < 0.5:
            return "medium"
        else:
            return "high"
