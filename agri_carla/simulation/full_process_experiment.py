"""停车区、机耕道、田块作业和返回组成的全流程导航实验。"""

from __future__ import annotations

from dataclasses import dataclass

from agri_carla.controllers.task_controls import (
    enforce_front_wheel_steering,
    get_linear_control,
    get_offroad_control,
    limit_road_speed,
    set_agent_target_speed,
    should_advance_point_task,
)
from agri_carla.evaluation.path_tracker import PathTracker
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.simulation.task_planner import FullProcessTaskConfig, FullProcessTaskPlanner
from agri_carla.utils.carla_import import import_carla
from agri_carla.utils.io import ensure_dir


def _collapse_consecutive_stages(stages) -> list[str]:
    """合并连续重复的任务类型，得到用于评价的阶段顺序。"""
    collapsed: list[str] = []
    for stage in stages:
        if not collapsed or stage != collapsed[-1]:
            collapsed.append(stage)
    return collapsed


def _stage_transitions_completed(
    expected_stages: list[str],
    observed_stages: list[str],
    task_completed: bool,
) -> bool:
    """仅在任务完成且实际阶段序列完全符合计划时判定切换成功。"""
    return task_completed and observed_stages == expected_stages


@dataclass
class FullProcessExperimentConfig:
    """全流程案例的 CARLA 连接、速度、田块和关键位置参数。"""
    host: str = "localhost"
    port: int = 2000
    timeout: float = 30.0
    carla_map: str = ""
    vehicle_bp: str = "vehicle.tractor.tractor"
    vehicle_type: str = "tractor"
    output_dir: str = "results/full_process_navigation"
    fixed_delta_seconds: float = 0.05
    road_speed_kmh: float = 13.5
    field_line_speed_kmh: float = 7.5
    max_steps: int = 60000
    apply_front_wheel_steering: bool = True
    apply_agri_physics: bool = False
    harvest_width_m: float = 3.0
    garage_spawn: tuple[float, float, float, float] = (74.0, -8.0, 2.0, 90.0)
    road_entry: tuple[float, float, float] = (74.0, -8.0, 2.0)
    field_top_left: tuple[float, float, float] = (150.0, -80.0, 0.5)
    field_top_right: tuple[float, float, float] = (150.0, -40.0, 0.5)
    field_bottom_right: tuple[float, float, float] = (50.0, -40.0, 0.5)
    field_bottom_left: tuple[float, float, float] = (50.0, -80.0, 0.5)
    field_entry: tuple[float, float, float] = (50.0, -80.0, 0.5)
    field_exit: tuple[float, float, float] = (50.0, -80.0, 0.5)
    field_exit_road: tuple[float, float, float] = (50.0, -98.0, 0.5)


class FullProcessNavigationExperiment:
    """运行论文第 3.2 节所用的全流程农业导航案例。"""

    def __init__(
        self,
        config: FullProcessExperimentConfig | None = None,
        performance_evaluation_subagent_class=PerformanceEvaluationSubagent,
    ):
        """连接 CARLA，选择地图，并保存 PES 类型供实验结束时评价。"""
        self.config = config or FullProcessExperimentConfig()
        self.performance_evaluation_subagent_class = (
            performance_evaluation_subagent_class
        )
        self.carla = import_carla()
        self.client = self.carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.load_world(self.config.carla_map) if self.config.carla_map else self.client.get_world()
        self.carla_map = self.world.get_map()

    def run(self) -> dict[str, str]:
        """按任务队列运行各导航阶段并输出路径与任务级评价表。"""
        # BehaviorAgent 仅用于 OpenDRIVE 可驾驶道路上的机耕道导航。
        from agents.navigation.behavior_agent import BehaviorAgent as CarlaBehaviorAgent

        # 同步模式和固定时间步保证距离、时长及阶段记录具有一致的时间基准。
        output_dir = ensure_dir(self.config.output_dir)
        original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.fixed_delta_seconds
        self.world.apply_settings(settings)

        vehicle = None
        collision_sensor = None
        tracker = PathTracker()
        collision_count = 0
        task_completed = False
        forward_reverse_completed = True
        stage_transition_completed = False
        sim_time_s = 0.0

        try:
            vehicle = self._spawn_vehicle()
            collision_count_box = {"count": 0}
            collision_sensor = self._attach_collision_sensor(vehicle, collision_count_box)
            self._apply_optional_agri_physics(vehicle)
            if self.config.apply_front_wheel_steering:
                enforce_front_wheel_steering(vehicle)
            vehicle.set_autopilot(False)
            for _ in range(20):
                self.world.tick()

            task_queue = self._build_task_queue()
            # 计划阶段序列用于事后检查是否存在漏执行、乱序或提前结束。
            expected_stage_sequence = _collapse_consecutive_stages(
                task["type"] for task in task_queue
            )
            observed_stage_sequence: list[str] = []
            road_navigation_controller = CarlaBehaviorAgent(vehicle, behavior="normal")
            set_agent_target_speed(road_navigation_controller, self.config.road_speed_kmh)
            current_task_idx = 0
            agent_destination_set = False

            for step in range(self.config.max_steps):
                self.world.tick()
                sim_time_s = step * self.config.fixed_delta_seconds
                veh_loc = vehicle.get_location()
                current_task = task_queue[current_task_idx] if current_task_idx < len(task_queue) else None
                task_type = current_task["type"] if current_task else "IDLE"
                tracker.update(veh_loc, task_type)
                if current_task is not None and (
                    not observed_stage_sequence or task_type != observed_stage_sequence[-1]
                ):
                    observed_stage_sequence.append(task_type)

                if current_task is None:
                    # 队列全部消费后才标记任务完成并驻车。
                    task_completed = True
                    vehicle.apply_control(self.carla.VehicleControl(hand_brake=True))
                    break

                mode = current_task["type"]
                if mode == "HARVEST_LINE":
                    # 田内直线作业采用面向线段的几何控制。
                    control, distance_to_target, _cte = get_linear_control(
                        self.carla,
                        vehicle,
                        current_task["start"],
                        current_task["target"],
                        target_speed=self.config.field_line_speed_kmh,
                    )
                    vehicle.apply_control(control)
                    if distance_to_target < 2.0:
                        current_task_idx += 1

                elif mode in ["GARAGE_EXIT", "OFFROAD_APPROACH", "TURN_BULB", "FIELD_EXIT", "GARAGE_RETURN"]:
                    # 无道路拓扑区域按当前阶段选择点目标几何控制和到达阈值。
                    agent_destination_set = False
                    target_loc = current_task["target"]
                    geo_mode = "GARAGE" if mode in ["GARAGE_EXIT", "GARAGE_RETURN"] else (
                        "TURN" if mode == "TURN_BULB" else (
                            "FIELD_ENTRY" if mode in ["OFFROAD_APPROACH", "FIELD_EXIT"] else "NORMAL"
                        )
                    )
                    threshold = 4.0 if mode == "TURN_BULB" else (
                        1.8 if mode == "OFFROAD_APPROACH" else (
                            3.0 if mode == "FIELD_EXIT" else 4.0
                        )
                    )
                    control, distance_to_target = get_offroad_control(self.carla, vehicle, target_loc, geo_mode)
                    vehicle.apply_control(control)
                    if should_advance_point_task(current_task, distance_to_target, threshold):
                        current_task_idx += 1

                elif mode in ["ROAD_NAV", "ROAD_RETURN"]:
                    # 道路阶段调用 CARLA BehaviorAgent，并额外限制拖拉机目标速度。
                    target_loc = current_task["target"]
                    if not agent_destination_set:
                        set_agent_target_speed(road_navigation_controller, self.config.road_speed_kmh)
                        road_navigation_controller.set_destination(target_loc)
                        agent_destination_set = True
                    if not road_navigation_controller.done():
                        control = road_navigation_controller.run_step()
                        control = limit_road_speed(vehicle, control, target_speed_kmh=self.config.road_speed_kmh)
                        control.manual_gear_shift = False
                        vehicle.apply_control(control)
                        if veh_loc.distance(target_loc) < 5.0:
                            current_task_idx += 1
                            agent_destination_set = False
                    else:
                        current_task_idx += 1
                        agent_destination_set = False

            path_csv = tracker.save_csv(output_dir / "full_process_path.csv")
            path_png = tracker.save_plot(output_dir / "full_process_path.png")
            collision_count = int(collision_count_box["count"])
            # 阶段切换必须同时满足任务完成和观测顺序与计划顺序一致。
            stage_transition_completed = _stage_transitions_completed(
                expected_stage_sequence,
                observed_stage_sequence,
                task_completed,
            )
            summary = self.performance_evaluation_subagent_class.task_level_summary(
                task_completed=task_completed,
                stage_transition_completed=stage_transition_completed,
                forward_reverse_completed=forward_reverse_completed,
                collision_count=collision_count,
                total_distance_m=tracker.total_distance,
                operation_path_length_m=tracker.work_distance,
                task_duration_s=sim_time_s,
            )
            summary_csv = output_dir / "full_process_task_evaluation.csv"
            summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
            return {
                "path_csv": str(path_csv),
                "path_plot": str(path_png) if path_png else "",
                "task_evaluation": str(summary_csv),
            }
        finally:
            if collision_sensor is not None and collision_sensor.is_alive:
                collision_sensor.destroy()
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
            self.world.apply_settings(original_settings)

    def _spawn_vehicle(self):
        """使用停车区位姿和指定蓝图生成实验车辆。"""
        bp = self.world.get_blueprint_library().find(self.config.vehicle_bp)
        x, y, z, yaw = self.config.garage_spawn
        spawn = self.carla.Transform(
            self.carla.Location(x=x, y=y, z=z + 0.5),
            self.carla.Rotation(yaw=yaw),
        )
        vehicle = self.world.try_spawn_actor(bp, spawn)
        if vehicle is None:
            raise RuntimeError("Vehicle spawning failed. Check garage_spawn and vehicle_bp.")
        return vehicle

    def _attach_collision_sensor(self, vehicle, collision_count_box: dict[str, int]):
        """挂载碰撞传感器，并通过可变字典累计回调事件次数。"""
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, self.carla.Transform(), attach_to=vehicle)

        def on_collision(_event):
            collision_count_box["count"] += 1

        sensor.listen(on_collision)
        return sensor

    def _apply_optional_agri_physics(self, vehicle) -> None:
        """按开关应用农机物理模板；全流程案例默认使用蓝图现有参数。"""
        if not self.config.apply_agri_physics:
            return
        try:
            from agri_carla.machinery.physics import AgriPhysicsConfig, apply_agri_physics
        except Exception:
            print("Agricultural physics helper not available; running with the current CARLA vehicle physics.")
            return
        apply_agri_physics(vehicle, AgriPhysicsConfig(self.config.vehicle_type))

    def _loc(self, value: tuple[float, float, float]):
        """把配置文件中的三元组转换为 CARLA Location。"""
        return self.carla.Location(x=value[0], y=value[1], z=value[2])

    def _build_task_queue(self) -> list[dict]:
        """将配置中的关键位置交给任务规划器生成有序执行队列。"""
        corners = [
            self._loc(self.config.field_top_left),
            self._loc(self.config.field_top_right),
            self._loc(self.config.field_bottom_right),
            self._loc(self.config.field_bottom_left),
        ]
        planner = FullProcessTaskPlanner(
            self.carla,
            self.carla_map,
            FullProcessTaskConfig(harvest_width_m=self.config.harvest_width_m),
        )
        return planner.build(
            road_entry_loc=self._loc(self.config.road_entry),
            field_entry_point=self._loc(self.config.field_entry),
            field_exit_point=self._loc(self.config.field_exit),
            field_exit_road_point=self._loc(self.config.field_exit_road),
            garage_return_loc=self._loc(self.config.garage_spawn[:3]),
            field_corners=corners,
        )
