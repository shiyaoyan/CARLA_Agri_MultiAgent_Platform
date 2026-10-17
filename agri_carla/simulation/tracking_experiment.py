"""真实拖拉机轨迹在 CARLA 中的闭环跟踪实验。"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from agri_carla.controllers.gear_state_machine import GearStateConfig
from agri_carla.controllers.interface_agent import ControllerInterfaceAgent, ControllerInterfaceConfig
from agri_carla.controllers.pid import PIDConfig
from agri_carla.controllers.pure_pursuit import PurePursuitConfig
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.trajectory.reference_path import ReferenceTrajectory
from agri_carla.utils.carla_import import import_carla
from agri_carla.utils.io import ensure_dir


@dataclass
class TrackingExperimentConfig:
    """轨迹跟踪实验的连接、地图、车辆、评价和控制器配置。"""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 60.0
    carla_map: str = ""
    vehicle_bp: str = "vehicle.tractor.tractor"
    vehicle_type: str = "tractor"
    dt: float = 0.05
    map_offset_x: float = 50.0
    map_offset_y: float = -10.0
    spawn_z: float = 2.0
    debug_life_time: float = 1200.0
    output_dir: str = "results/trajectory_tracking"
    max_extra_time_s: float = 15.0
    finish_speed_ms: float = 0.35
    finish_position_error_m: float = 1.0
    apply_agri_physics: bool = True
    trajectory_flip_y: bool = True
    metric_min_speed_ms: float = 0.2
    metric_skip_start_s: float = 5.0
    metric_skip_end_s: float = 5.0
    controller: ControllerInterfaceConfig | None = None
    controller_optimization: dict[str, Any] | None = None
    vegetation_request: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrackingExperimentConfig":
        """把 JSON 字典递归转换为实验及其嵌套控制器配置对象。"""
        values = dict(data or {})
        controller = values.get("controller")
        if isinstance(controller, dict):
            controller_values = dict(controller)
            pid = controller_values.get("pid")
            pure_pursuit = controller_values.get("pure_pursuit")
            gear = controller_values.get("gear")
            if isinstance(pid, dict):
                controller_values["pid"] = _dataclass_from_dict(PIDConfig, pid)
            if isinstance(pure_pursuit, dict):
                controller_values["pure_pursuit"] = _dataclass_from_dict(
                    PurePursuitConfig,
                    pure_pursuit,
                )
            if isinstance(gear, dict):
                gear_values = dict(gear)
                if isinstance(gear_values.get("forward_gears"), list):
                    gear_values["forward_gears"] = tuple(gear_values["forward_gears"])
                controller_values["gear"] = _dataclass_from_dict(GearStateConfig, gear_values)
            values["controller"] = _dataclass_from_dict(
                ControllerInterfaceConfig,
                controller_values,
            )
        return _dataclass_from_dict(cls, values)


class TrackingExperimentRunner:
    """运行论文第 3.1 节所用的真实轨迹复现与跟踪实验。"""

    def __init__(
        self,
        config: TrackingExperimentConfig | None = None,
        controller_agent_class=ControllerInterfaceAgent,
        performance_evaluation_subagent_class=PerformanceEvaluationSubagent,
    ):
        """建立 CARLA 连接，并按配置选择当前世界或加载指定地图。"""
        self.config = config or TrackingExperimentConfig()
        self.controller_agent_class = controller_agent_class
        self.performance_evaluation_subagent_class = (
            performance_evaluation_subagent_class
        )
        self.carla = import_carla()
        self.client = self.carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.load_world(self.config.carla_map) if self.config.carla_map else self.client.get_world()
        if self.config.vegetation_request:
            from agri_carla.environment.vegetation import restore_experiment_vegetation

            restore_experiment_vegetation(self.config)

    def run(self, trajectory_csv: str | Path) -> dict[str, str]:
        """在同步仿真中跟踪一条参考轨迹并保存逐帧记录与评价结果。

        实验结束后无论成功或异常都会销毁车辆并恢复原世界设置，防止本轮
        同步模式和固定步长影响后续 CARLA 实验。
        """
        trajectory = ReferenceTrajectory(trajectory_csv, flip_y=self.config.trajectory_flip_y)
        trajectory.print_data_check()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_dir(Path(self.config.output_dir) / f"{trajectory.csv_name}_{timestamp}")
        recorder = self.performance_evaluation_subagent_class(output_dir)

        # 固定时间步可使相同输入和参数下的控制与评价过程可重复。
        original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.dt
        self.world.apply_settings(settings)

        vehicle = None
        try:
            vehicle = self._spawn_vehicle(trajectory)
            self._apply_optional_agri_physics(vehicle)
            self._draw_target_points(trajectory)
            for _ in range(10):
                self.world.tick()

            controller = self.controller_agent_class(
                trajectory,
                self.config.map_offset_x,
                self.config.map_offset_y,
                self.config.controller or ControllerInterfaceConfig(dt=self.config.dt),
            )
            controller.reset()

            max_sim_time = trajectory.t[-1] + self.config.max_extra_time_s
            max_steps = int(max_sim_time / self.config.dt)

            for step in range(max_steps):
                # 每个仿真步读取车辆状态，经 CIA 计算后再转换为 CARLA 控制量。
                self.world.tick()
                sim_time = step * self.config.dt
                transform = vehicle.get_transform()
                loc = transform.location
                vel = vehicle.get_velocity()
                actual_speed = math.hypot(vel.x, vel.y)

                state = {
                    "x": loc.x,
                    "y": loc.y,
                    "yaw_deg": transform.rotation.yaw,
                    "actual_speed_ms": actual_speed,
                }
                command = controller.compute(state, sim_time)
                carla_control = self._to_carla_control(command)
                vehicle.apply_control(carla_control)

                signed_actual = actual_speed if command["current_forward"] else -actual_speed
                target_signed = command["target_abs_kmh"] if command["is_forward"] else -command["target_abs_kmh"]
                cmd_signed = command["cmd_abs_kmh"] if command["is_forward"] else -command["cmd_abs_kmh"]

                # 起步、停车、换挡和参考轨迹结束后的数据保留用于诊断，但不纳入
                # 稳态误差统计，避免瞬态过程掩盖路径跟踪本身的表现。
                time_idx = int(command["target_idx"])
                in_reference_time = sim_time <= trajectory.t[-1] + 1e-9
                metric_reasons = []
                if not in_reference_time:
                    metric_reasons.append("POST_ENDPOINT")
                if sim_time < self.config.metric_skip_start_s:
                    metric_reasons.append("START_TRANSIENT")
                if sim_time > trajectory.t[-1] - self.config.metric_skip_end_s:
                    metric_reasons.append("END_TRANSIENT")
                if actual_speed < self.config.metric_min_speed_ms:
                    metric_reasons.append("LOW_SPEED")
                diagnostic_text = str(command["diagnostic"])
                if "SHIFT_NEUTRAL" in diagnostic_text:
                    metric_reasons.append("SHIFT_TRANSIENT")
                metric_valid = len(metric_reasons) == 0

                recorder.add_row({
                    # 同时记录时间同步误差和最近路径几何误差，便于复核指标来源。
                    "time_s": sim_time,
                    "x": loc.x,
                    "y": loc.y,
                    "yaw_deg": transform.rotation.yaw,
                    "target_idx": time_idx,
                    "lookahead_idx": command["lookahead_idx"],
                    "phase_id": int(trajectory.segment_id[time_idx]),
                    "is_forward": command["is_forward"],
                    "target_abs_kmh": command["target_abs_kmh"],
                    "cmd_abs_kmh": command["cmd_abs_kmh"],
                    "actual_abs_kmh": actual_speed * 3.6,
                    "target_signed_kmh": float(target_signed),
                    "cmd_signed_kmh": float(cmd_signed),
                    "actual_signed_kmh": signed_actual * 3.6,
                    "pos_err_m": command["pos_err_m"],
                    "geometric_err_m": command["geometric_err_m"],
                    "ref_x": command["ref_x"],
                    "ref_y": command["ref_y"],
                    "throttle": command["throttle"],
                    "brake": command["brake"],
                    "steer": command["steer"],
                    "cte_m": command["cte_m"],
                    "path_cte_m": command["path_cte_m"],
                    "path_distance_m": command["path_distance_m"],
                    "nearest_path_idx": command["nearest_path_idx"],
                    "nearest_path_u": command["nearest_path_u"],
                    "nearest_path_x": command["nearest_path_x"],
                    "nearest_path_y": command["nearest_path_y"],
                    "time_sync_cte_m": command["time_sync_cte_m"],
                    "time_sync_heading_error_deg": math.degrees(float(command["time_sync_heading_error_rad"])),
                    "heading_error_deg": math.degrees(float(command["heading_error_rad"])),
                    "path_heading_error_deg": math.degrees(float(command["path_heading_error_rad"])),
                    "in_reference_time": in_reference_time,
                    "metric_valid": metric_valid,
                    "metric_filter_reason": "|".join(metric_reasons),
                    "map_offset_x": self.config.map_offset_x,
                    "map_offset_y": self.config.map_offset_y,
                    "diagnostic": command["diagnostic"],
                })

                if command["diagnostic"]:
                    for code in str(command["diagnostic"]).split("|"):
                        if code == "PHASE_SHIFT":
                            recorder.add_event(sim_time, code, "Controller state-machine event")

                is_finished = sim_time >= trajectory.t[-1]
                # 只有参考时间结束、车辆基本停止且位置接近终点时才提前结束。
                if is_finished and actual_speed < self.config.finish_speed_ms and abs(float(command["pos_err_m"])) < self.config.finish_position_error_m:
                    recorder.add_event(sim_time, "FINISHED", "The trajectory replay finished with stable position and speed.")
                    break

            return recorder.save(trajectory)
        finally:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
                self.world.tick()
            self.world.apply_settings(original_settings)

    def _spawn_vehicle(self, trajectory: ReferenceTrajectory):
        """按轨迹首段方向计算初始航向并生成拖拉机。"""
        bp = self.world.get_blueprint_library().find(self.config.vehicle_bp)
        x0 = trajectory.x[0] + self.config.map_offset_x
        y0 = trajectory.y[0] + self.config.map_offset_y
        next_i = min(1, trajectory.num_points - 1)
        dx = trajectory.x[next_i] - trajectory.x[0]
        dy = trajectory.y[next_i] - trajectory.y[0]
        yaw = math.degrees(math.atan2(dy, dx))
        if not trajectory.is_forward[0]:
            # 首段为倒车时，车头方向与轨迹运动方向相反。
            yaw = (yaw + 180.0) % 360.0
        spawn = self.carla.Transform(
            self.carla.Location(x=x0, y=y0, z=self.config.spawn_z),
            self.carla.Rotation(yaw=yaw),
        )
        vehicle = self.world.try_spawn_actor(bp, spawn)
        if vehicle is None:
            raise RuntimeError("Vehicle spawning failed. Check the spawn point and blueprint.")
        return vehicle

    def _to_carla_control(self, command: dict[str, Any]):
        """把 CIA 的通用控制字典转换为 CARLA VehicleControl。"""
        control = self.carla.VehicleControl()
        control.throttle = float(command["throttle"])
        control.brake = float(command["brake"])
        control.steer = float(command["steer"])
        control.reverse = bool(command["reverse"])
        control.hand_brake = False
        control.manual_gear_shift = bool(command.get("manual_gear_shift", True))
        control.gear = int(command["gear"])
        return control

    def _draw_target_points(self, trajectory: ReferenceTrajectory) -> None:
        """在 CARLA 调试视图中以不同颜色标出前进和倒车参考点。"""
        for i in range(0, trajectory.num_points, 2):
            color = self.carla.Color(0, 220, 255) if trajectory.is_forward[i] else self.carla.Color(255, 130, 0)
            self.world.debug.draw_point(
                self.carla.Location(
                    x=trajectory.x[i] + self.config.map_offset_x,
                    y=trajectory.y[i] + self.config.map_offset_y,
                    z=1.5,
                ),
                size=0.08,
                color=color,
                life_time=self.config.debug_life_time,
            )

    def _apply_optional_agri_physics(self, vehicle) -> None:
        """按开关应用农机物理模板；不可用时保留当前 CARLA 参数。"""
        if not self.config.apply_agri_physics:
            return
        try:
            from agri_carla.machinery.physics import AgriPhysicsConfig, apply_agri_physics
        except Exception:
            print("Agricultural physics helper not available; running with the current CARLA vehicle physics.")
            return
        apply_agri_physics(vehicle, AgriPhysicsConfig(self.config.vehicle_type))


def _dataclass_from_dict(dataclass_type, data: dict[str, Any]):
    """仅保留目标数据类声明的字段，忽略 JSON 中的扩展键。"""
    valid_keys = {field.name for field in fields(dataclass_type)}
    return dataclass_type(**{
        key: value
        for key, value in data.items()
        if key in valid_keys
    })
