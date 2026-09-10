"""Mac CrewAI 管理端与 CARLA 计算机之间的执行协议。"""

from .client import CarlaWorkerClient, CarlaWorkerConfig

__all__ = ["CarlaWorkerClient", "CarlaWorkerConfig"]
