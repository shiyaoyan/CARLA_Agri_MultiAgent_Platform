"""控制器接口智能体（CIA）的在线控制与可选参数自优化入口。"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .gear_state_machine import GearStateConfig, GearStateMachine
from .pid import PIDConfig, PIDController
from .pure_pursuit import PurePursuitConfig, PurePursuitController
from agri_carla.utils.geometry import wrap_pi


@dataclass
class ControllerInterfaceConfig:
    """控制周期、位置同步增益、限速及三个控制子模块配置。"""
    dt: float = 0.05
    k_pos: float = 0.8
    max_speed_kmh: float = 8.0
    pid: PIDConfig | None = None
    pure_pursuit: PurePursuitConfig | None = None
    gear: GearStateConfig | None = None


class ControllerInterfaceAgent:
    """组合纯追踪、速度 PID 和换挡状态机，并提供可选参数自优化。

    CIA 输出与 CARLA 无关的控制字典；EEA 读取该字典后才创建并施加
    ``carla.VehicleControl``，因此 CIA 是独立的并列智能体服务。
    """

    def __init__(self, trajectory, offset_x: float, offset_y: float, config: ControllerInterfaceConfig | None = None):
        """根据标准轨迹创建纵向、横向和换挡控制状态。"""
        self.config = config or ControllerInterfaceConfig()
        self.trajectory = trajectory
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.pid = PIDController(self.config.pid)
        self.tracker = PurePursuitController(trajectory, offset_x, offset_y, self.config.pure_pursuit)
        self.gear = GearStateMachine(bool(trajectory.is_forward[0]), self.config.gear)
        self.max_speed_ms = self.config.max_speed_kmh / 3.6
        self.high_throttle_low_speed_time = 0.0
        self.last_gear_probe_time = 0.0

    def reset(self) -> None:
        """在新实验开始前清除所有控制器历史状态。"""
        self.pid.reset()
        self.gear.reset(bool(self.trajectory.is_forward[0]))
        self.high_throttle_low_speed_time = 0.0
        self.last_gear_probe_time = 0.0

    @staticmethod
    def self_optimization_config(data: dict[str, Any] | None):
        """解析 CIA 内部的贝叶斯优化开关和参数搜索配置。"""
        from .controller_optimization import ControllerSelfOptimizationConfig

        return ControllerSelfOptimizationConfig.from_dict(data)

    @classmethod
    def run_self_optimization(
        cls,
        *,
        config,
        parameter_optimization_csvs: Sequence[str | Path],
        independent_test_csvs: Sequence[str | Path] = (),
        experiment_config,
        run_trial: Callable[[Any, str | Path], dict[str, Any]],
        progress_callback=None,
    ) -> dict[str, Any]:
        """在优化集上搜索控制参数，并在独立测试集上评价最优参数。"""
        from .controller_optimization import ControllerSelfOptimizationSession

        return ControllerSelfOptimizationSession(config).run(
            parameter_optimization_csvs=parameter_optimization_csvs,
            independent_test_csvs=independent_test_csvs,
            experiment_config=experiment_config,
            run_trial=run_trial,
            progress_callback=progress_callback,
        )

    @classmethod
    def prepare_experiment_configuration(
        cls,
        task_type: str,
        experiment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """校验控制相关配置并生成 EEA 可读取的控制器清单。

        该方法不连接 CARLA，也不运行控制循环。它使 CIA 能作为 CrewAI Flow
        中的独立同层智能体，在 EEA 启动仿真前检查控制方式、参数优化开关和
        关键控制配置。实际逐帧控制仍由 EEA 在仿真运行时调用 CIA 完成。
        """
        experiment = dict(experiment or {})
        config_data = dict(experiment.get("config", {}))
        if task_type == "trajectory_tracking":
            from agri_carla.simulation.tracking_experiment import (
                TrackingExperimentConfig,
            )

            tracking_config = TrackingExperimentConfig.from_dict(config_data)
            controller = tracking_config.controller or ControllerInterfaceConfig(
                dt=tracking_config.dt
            )
            optimization = cls.self_optimization_config(
                tracking_config.controller_optimization
            )
            return {
                "task_type": task_type,
                "validated": True,
                "control_pipeline": [
                    "direction_aware_pure_pursuit",
                    "pid_speed_control",
                    "gear_shifting_state_machine",
                    "carla_vehicle_control",
                ],
                "controller": dataclasses.asdict(controller),
                "self_optimization": dataclasses.asdict(optimization),
            }

        if task_type == "full_process_navigation":
            from agri_carla.simulation.full_process_experiment import (
                FullProcessExperimentConfig,
            )

            full_process_config = FullProcessExperimentConfig(**config_data)
            return {
                "task_type": task_type,
                "validated": True,
                "control_pipeline": [
                    "road_navigation_control",
                    "field_and_headland_geometric_control",
                    "carla_vehicle_control",
                ],
                "road_speed_kmh": full_process_config.road_speed_kmh,
                "field_line_speed_kmh": full_process_config.field_line_speed_kmh,
                "apply_front_wheel_steering": (
                    full_process_config.apply_front_wheel_steering
                ),
            }

        return {
            "task_type": task_type,
            "validated": True,
            "control_pipeline": [],
            "message": "No simulation controller is required for this task type.",
        }

    def time_reference(self, sim_time: float):
        """按仿真时间插值得到参考位置、速度和相邻轨迹索引。"""
        import numpy as np

        traj = self.trajectory
        t_target = min(sim_time, traj.t[-1])
        ref_x = np.interp(t_target, traj.t, traj.x) + self.offset_x
        ref_y = np.interp(t_target, traj.t, traj.y) + self.offset_y
        ref_speed = np.interp(t_target, traj.t, traj.speed_ms)
        time_idx = int(np.searchsorted(traj.t, t_target))
        if time_idx == 0:
            time_idx = 1
        if time_idx >= traj.num_points:
            time_idx = traj.num_points - 1
        return t_target, ref_x, ref_y, ref_speed, time_idx

    def compute(self, vehicle_state: dict[str, float], sim_time: float) -> dict[str, Any]:
        """返回不依赖 CARLA 类型的控制和诊断字典。

        ``vehicle_state`` 必须包含位置、车头航向和实际速度。结果同时给出油门、
        制动、转向、挡位以及路径误差，供 EEA 执行和 PES 记录。
        """
        import numpy as np

        traj = self.trajectory
        t_target, ref_x, ref_y, ref_speed, time_idx = self.time_reference(sim_time)
        self.tracker.index = time_idx
        target_forward = bool(traj.is_forward[time_idx])
        actual_speed = vehicle_state["actual_speed_ms"]
        x = vehicle_state["x"]
        y = vehicle_state["y"]

        dx = traj.x[time_idx] - traj.x[time_idx - 1]
        dy = traj.y[time_idx] - traj.y[time_idx - 1]
        length = math.hypot(dx, dy)
        if length > 1e-5:
            tx, ty = dx / length, dy / length
        else:
            yaw_rad = math.radians(vehicle_state["yaw_deg"])
            tx, ty = math.cos(yaw_rad), math.sin(yaw_rad)
            if not target_forward:
                tx, ty = -tx, -ty

        # 将时间同步位置误差分解到路径切向和法向；最近路径误差另由轨迹对象计算，
        # 两者分别用于速度同步和最终跟踪性能评价。
        pos_err = (ref_x - x) * tx + (ref_y - y) * ty
        ex = ref_x - x
        ey = ref_y - y
        geometric_dist = math.hypot(ex, ey)
        nx, ny = -ty, tx
        time_sync_cte = ex * nx + ey * ny
        ref_heading = math.atan2(ty, tx)
        vehicle_motion_heading = math.radians(vehicle_state["yaw_deg"])
        if not target_forward:
            vehicle_motion_heading += math.pi
        time_sync_heading_error = wrap_pi(vehicle_motion_heading - ref_heading)
        path_error = traj.nearest_path_error(x, y, self.offset_x, self.offset_y)
        path_heading_error = wrap_pi(vehicle_motion_heading - float(path_error["path_heading_rad"]))

        target_speed_cmd = float(np.clip(ref_speed + self.config.k_pos * pos_err, 0.0, self.max_speed_ms))
        gear_state = self.gear.update(target_forward, actual_speed, sim_time)

        diagnostic = []
        event = gear_state.get("event")
        if event:
            diagnostic.append(str(event))
            if event == "PHASE_SHIFT":
                self.pid.reset()

        # 换挡状态优先于普通控制：空挡时撤销控制，未停车时强制制动。
        if gear_state["mode"] in {"neutral", "enter_neutral"}:
            throttle, brake, steer = 0.0, 0.0, 0.0
            heading_error = 0.0
            lookahead_idx = time_idx
            lookahead_dist = 0.0
        elif gear_state["mode"] == "brake_to_shift":
            throttle, brake, steer = 0.0, 1.0, 0.0
            heading_error = 0.0
            lookahead_idx = time_idx
            lookahead_dist = 0.0
        else:
            throttle, brake, _pid_out = self.pid.step(target_speed_cmd, actual_speed, self.config.dt)
            steer, heading_error, lookahead_idx, lookahead_dist = self.tracker.compute_steer(
                x, y, vehicle_state["yaw_deg"], actual_speed, self.gear.current_forward
            )

        # 持续高油门但速度过低时记录诊断，并按时间间隔试探另一前进挡。
        if throttle > 0.85 and actual_speed < max(0.35, target_speed_cmd * 0.35) and target_speed_cmd > 0.6:
            self.high_throttle_low_speed_time += self.config.dt
            if self.high_throttle_low_speed_time > 2.0:
                diagnostic.append("SPEED_LOW_HIGH_THROTTLE")
                gear_cfg = self.gear.config
                if self.gear.current_forward:
                    if sim_time - self.last_gear_probe_time >= gear_cfg.gear_probe_interval_s:
                        self.gear.next_forward_gear()
                        self.last_gear_probe_time = sim_time
                        self.pid.reset()
                        diagnostic.append("GEAR_PROBE")
                self.high_throttle_low_speed_time = 0.0
        else:
            self.high_throttle_low_speed_time = 0.0

        return {
            "throttle": float(throttle),
            "brake": float(brake),
            "steer": float(steer),
            "reverse": bool(gear_state["reverse"]),
            "current_forward": bool(gear_state["forward"]),
            "gear": int(self.gear.current_gear()) if gear_state["mode"] == "drive" else int(gear_state["gear"]),
            "manual_gear_shift": True,
            "target_idx": time_idx,
            "lookahead_idx": lookahead_idx,
            "is_forward": target_forward,
            "ref_x": ref_x,
            "ref_y": ref_y,
            "target_abs_kmh": ref_speed * 3.6,
            "cmd_abs_kmh": target_speed_cmd * 3.6,
            "pos_err_m": pos_err,
            "geometric_err_m": geometric_dist,
            "cte_m": path_error["path_cte_m"],
            "path_cte_m": path_error["path_cte_m"],
            "path_distance_m": path_error["path_distance_m"],
            "nearest_path_idx": path_error["nearest_path_idx"],
            "nearest_path_u": path_error["nearest_path_u"],
            "nearest_path_x": path_error["nearest_path_x"],
            "nearest_path_y": path_error["nearest_path_y"],
            "time_sync_cte_m": time_sync_cte,
            "time_sync_heading_error_rad": time_sync_heading_error,
            "heading_error_rad": heading_error,
            "path_heading_error_rad": path_heading_error,
            "lookahead_dist_m": lookahead_dist,
            "diagnostic": "|".join(diagnostic),
        }
