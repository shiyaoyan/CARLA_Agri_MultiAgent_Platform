"""拖拉机纵向速度 PID 基线控制器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PIDConfig:
    """PID 系数、积分限幅、前馈油门和制动阈值配置。"""
    kp: float = 0.95
    ki: float = 0.22
    kd: float = 0.03
    integral_limit: float = 4.0
    min_throttle: float = 0.28
    max_ff_speed_ms: float = 2.2
    ff_limit: float = 0.58
    brake_deadband_ms: float = 0.35
    brake_gain: float = 0.9


class PIDController:
    """将目标速度与当前速度之差转换为油门和制动指令。"""

    def __init__(self, config: PIDConfig | None = None):
        """初始化参数及误差积分、上一时刻误差。"""
        self.config = config or PIDConfig()
        self.integral = 0.0
        self.last_error = 0.0

    def reset(self) -> None:
        """在换向或停车后清除积分和微分历史，防止旧误差延续。"""
        self.integral = 0.0
        self.last_error = 0.0

    def step(self, target_ms: float, actual_ms: float, dt: float) -> tuple[float, float, float]:
        """计算一个控制周期的油门、制动和原始 PID 输出。"""
        cfg = self.config
        if target_ms < 0.03:
            self.reset()
            return 0.0, 1.0 if actual_ms > 0.12 else 0.0, 0.0

        error = target_ms - actual_ms
        self.integral = float(np.clip(self.integral + error * dt, -cfg.integral_limit, cfg.integral_limit))
        derivative = (error - self.last_error) / max(dt, 1e-6)
        self.last_error = error

        pid = cfg.kp * error + cfg.ki * self.integral + cfg.kd * derivative

        # 加速阶段叠加速度前馈，减速阶段超过死区才施加制动，避免油门制动抖动。
        if error >= 0.0:
            feedforward = float(np.clip(target_ms / cfg.max_ff_speed_ms, 0.0, cfg.ff_limit))
            throttle = float(np.clip(cfg.min_throttle + feedforward + pid, 0.0, 1.0))
            return throttle, 0.0, float(pid)

        if actual_ms > target_ms + cfg.brake_deadband_ms:
            brake = float(np.clip((actual_ms - target_ms - cfg.brake_deadband_ms) * cfg.brake_gain, 0.0, 1.0))
            return 0.0, brake, float(pid)

        return 0.0, 0.0, float(pid)
