"""区分前进和倒车参数的拖拉机纯追踪横向控制器。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agri_carla.utils.geometry import wrap_pi


@dataclass
class PurePursuitConfig:
    """轴距、最大转角以及前进/倒车前视距离参数。"""
    wheelbase: float = 2.5
    max_steer_angle_deg: float = 48.0
    fwd_lookahead_min: float = 1.0
    fwd_lookahead_gain: float = 0.2
    rev_lookahead_min: float = 0.5
    rev_lookahead_gain: float = 0.1


class PurePursuitController:
    """根据运动方向选择参数并计算归一化转向指令。"""

    def __init__(self, trajectory, offset_x: float = 0.0, offset_y: float = 0.0,
                 config: PurePursuitConfig | None = None):
        """绑定标准轨迹、场景偏移和控制参数。"""
        self.trajectory = trajectory
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.config = config or PurePursuitConfig()
        self.index = 0

    def point(self, index: int) -> tuple[float, float]:
        """返回叠加场景坐标偏移后的参考路径点。"""
        return self.trajectory.x[index] + self.offset_x, self.trajectory.y[index] + self.offset_y

    def lookahead_point(
        self,
        x: float,
        y: float,
        actual_speed_ms: float,
        is_forward: bool,
    ) -> tuple[float, float, int, float]:
        """在当前同向轨迹段内寻找满足动态前视距离的目标点。"""
        cfg = self.config
        lookahead_min = cfg.fwd_lookahead_min if is_forward else cfg.rev_lookahead_min
        lookahead_gain = cfg.fwd_lookahead_gain if is_forward else cfg.rev_lookahead_gain
        lookahead = max(lookahead_min, lookahead_min + lookahead_gain * actual_speed_ms)

        lx, ly = self.point(self.trajectory.num_points - 1)
        li = self.trajectory.num_points - 1
        # 搜索到换向边界即停止，防止前视点跨越前进/倒车分段。
        current_direction = bool(self.trajectory.is_forward[self.index])

        for i in range(self.index, self.trajectory.num_points):
            if bool(self.trajectory.is_forward[i]) != current_direction:
                lx, ly = self.point(i)
                li = i
                break
            tx, ty = self.point(i)
            if math.hypot(x - tx, y - ty) >= lookahead:
                lx, ly = tx, ty
                li = i
                break
        return lx, ly, li, lookahead

    def compute_steer(
        self,
        x: float,
        y: float,
        yaw_deg: float,
        actual_speed_ms: float,
        is_forward: bool,
    ) -> tuple[float, float, int, float]:
        """按纯追踪几何关系计算转向、航向误差和前视点信息。"""
        lx, ly, li, lookahead_dist = self.lookahead_point(x, y, actual_speed_ms, is_forward)
        target_heading = math.atan2(ly - y, lx - x)
        vehicle_heading = math.radians(yaw_deg)
        # 倒车时车辆有效运动航向与车头相差 pi，最终还需修正转向符号。
        if not is_forward:
            vehicle_heading += math.pi

        alpha = wrap_pi(target_heading - vehicle_heading)
        cfg = self.config
        delta = math.atan2(2.0 * cfg.wheelbase * math.sin(alpha), max(lookahead_dist, 0.3))
        steer = delta / math.radians(cfg.max_steer_angle_deg)
        if not is_forward:
            steer = -steer
        return float(np.clip(steer, -1.0, 1.0)), alpha, li, lookahead_dist
