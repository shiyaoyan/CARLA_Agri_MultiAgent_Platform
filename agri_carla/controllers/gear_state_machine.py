"""模拟拖拉机停车、空挡等待和前进/倒车切换的换挡状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict


class GearState(TypedDict):
    """一次换挡状态更新返回的稳定字段。"""

    mode: Literal["neutral", "brake_to_shift", "enter_neutral", "drive"]
    forward: bool
    reverse: bool
    gear: int
    event: str | None


@dataclass
class GearStateConfig:
    """允许换向速度、空挡持续时间和可探测前进挡位配置。"""
    shift_speed_ms: float = 0.18
    shift_neutral_s: float = 0.6
    forward_gears: tuple[int, ...] = (1, 2, 3)
    reverse_manual_gear: int = -1
    gear_probe_interval_s: float = 6.0


class GearStateMachine:
    """为农业路径中的频繁换向提供最小但明确的机械过程状态。"""

    def __init__(self, initial_forward: bool = True, config: GearStateConfig | None = None):
        """按首段运动方向初始化当前挡位和待切换状态。"""
        self.config = config or GearStateConfig()
        self.current_forward = initial_forward
        self.pending_forward: bool | None = None
        self.neutral_until = -1.0
        self.forward_gear_index = 0

    def reset(self, initial_forward: bool = True) -> None:
        """恢复初始方向并清除尚未完成的换挡过程。"""
        self.current_forward = initial_forward
        self.pending_forward = None
        self.neutral_until = -1.0
        self.forward_gear_index = 0

    def update(self, target_forward: bool, actual_speed_ms: float, sim_time: float) -> GearState:
        """根据目标方向、车速和仿真时间推进换挡状态。"""
        cfg = self.config
        neutral_active = sim_time < self.neutral_until
        event = None

        if self.pending_forward is not None and not neutral_active:
            self.current_forward = self.pending_forward
            self.pending_forward = None
            self.forward_gear_index = 0
            event = "PHASE_SHIFT"

        if neutral_active:
            return {
                "mode": "neutral",
                "forward": self.current_forward,
                "reverse": False,
                "gear": 0,
                "event": "SHIFT_NEUTRAL",
            }

        # 方向变化时必须先减速至阈值以下，再进入定时空挡，最后完成挂挡。
        if target_forward != self.current_forward:
            if actual_speed_ms > cfg.shift_speed_ms:
                return {
                    "mode": "brake_to_shift",
                    "forward": self.current_forward,
                    "reverse": not self.current_forward,
                    "gear": self.current_gear(),
                    "event": "WAIT_SHIFT_STOP",
                }
            self.pending_forward = target_forward
            self.neutral_until = sim_time + cfg.shift_neutral_s
            return {
                "mode": "enter_neutral",
                "forward": self.current_forward,
                "reverse": False,
                "gear": 0,
                "event": "SHIFT_NEUTRAL",
            }

        return {
            "mode": "drive",
            "forward": self.current_forward,
            "reverse": not self.current_forward,
            "gear": self.current_gear(),
            "event": event,
        }

    def current_gear(self) -> int:
        """返回当前前进挡或倒挡编号。"""
        if self.current_forward:
            gears = self.config.forward_gears or (1,)
            return gears[self.forward_gear_index % len(gears)]
        return self.config.reverse_manual_gear

    def next_forward_gear(self) -> int:
        """循环试探下一前进挡，用于高油门低车速诊断后的恢复尝试。"""
        gears = self.config.forward_gears or (1,)
        self.forward_gear_index = (self.forward_gear_index + 1) % len(gears)
        return self.current_gear()
