"""Satellite-based garage, road, two-pass field operation and return."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import csv
import hashlib
import math

from agri_carla.controllers.full_process_control import MissionPathFollower, parked
from agri_carla.controllers.task_controls import enforce_front_wheel_steering, limit_road_speed, set_agent_target_speed
from agri_carla.evaluation.path_tracker import PathTracker
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.machinery.physics_audit import physics_report
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig
from agri_carla.simulation.task_planner import FullProcessTaskPlanner, lane_key, serializable_tasks, xyz
from agri_carla.utils.carla_import import import_carla
from agri_carla.utils.io import ensure_dir, write_json


def _collapse_consecutive_stages(stages):
    result = []
    for stage in stages:
        if not result or stage != result[-1]:
            result.append(stage)
    return result


def _stage_transitions_completed(expected_stages, observed_stages, task_completed):
    return task_completed and observed_stages == expected_stages


def vehicle_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    return dict(position_m=xyz(transform.location), yaw_deg=float(transform.rotation.yaw),
                pitch_deg=float(getattr(transform.rotation, "pitch", 0)),
                roll_deg=float(getattr(transform.rotation, "roll", 0)),
                speed_ms=math.hypot(velocity.x, velocity.y))


def collision_records(count):
    """Every recorded contact, tagged with the stage that was running."""
    return [{**event, "stage": count["frame_stages"].get(event["frame"], "UNMAPPED_FRAME")}
            for event in count["events"]]


def collision_reason(count):
    fatal = count.get("fatal_events") or []
    if not fatal:
        return "Collision detected; event details unavailable"
    first = min(fatal, key=lambda event: event["frame"])
    stage = count["frame_stages"].get(first["frame"], "UNMAPPED_FRAME")
    return (f"Collision detected: stage={stage}, frame={first['frame']}, "
            f"other={first['other_actor_type']} (id={first['other_actor_id']}), "
            f"impulse={first['normal_impulse_magnitude']:.3f}. See collision_events.json")


def agent_route_with_lookahead(task):
    """Extend control lookahead, not the recorded mission gate or stage goal."""
    route = list(task["route"])
    wp, option = route[-1]
    for _ in range(10):
        following = [w for w in wp.next(0.5) if lane_key(w) == lane_key(wp)]
        if not following:
            break
        wp = following[0]
        route.append((wp, option))
    return route


class FullProcessRunFailed(RuntimeError):
    """Artifacts were saved, but the physical mission did not complete."""


class FullProcessNavigationExperiment:
    """Run the multi-stage agricultural navigation experiment (paper Section 3.3)."""

    def __init__(self, config=None, performance_evaluation_subagent_class=PerformanceEvaluationSubagent):
        self.config = config or FullProcessExperimentConfig()
        self.performance_evaluation_subagent_class = performance_evaluation_subagent_class
        self.carla = import_carla()
        self.client = self.carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.get_world()
        if self.config.reload_world:
            self.world = self.client.load_world(self.config.carla_map)
        actual_name = self.world.get_map().name.rsplit("/", 1)[-1]
        if actual_name != self.config.carla_map.rsplit("/", 1)[-1]:
            raise ValueError(f"Expected {self.config.carla_map}, current map is {actual_name}. "
                             "Open the intended map or explicitly set reload_world=true.")
        if self.config.vegetation_request:
            from agri_carla.environment.vegetation import restore_experiment_vegetation
            restore_experiment_vegetation(self.config)
        self.carla_map = self.world.get_map()

    def plan(self):
        from agents.navigation.global_route_planner import GlobalRoutePlanner
        planner = GlobalRoutePlanner(self.carla_map, self.config.path_spacing_m)
        tasks = FullProcessTaskPlanner(self.carla, self.carla_map, self.config, planner).build()
        out = ensure_dir(self.config.output_dir)
        write_json(out / "planned_mission.json", {
            "status": "PLANNED_ONLY", "carla_simulation_executed": False,
            "coordinate_frame": "CARLA world metres (Y south)",
            "map": self.carla_map.name, "config": asdict(self.config),
            "xodr_sha256": hashlib.sha256(self.carla_map.to_opendrive().encode()).hexdigest(),
            "tasks": serializable_tasks(tasks),
        })
        return tasks

    def run(self):
        from agents.navigation.behavior_agent import BehaviorAgent
        run_name = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]
        out = ensure_dir(Path(self.config.output_dir) / run_name)
        self.config = replace(self.config, output_dir=str(out))
        c = self.config
        original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = c.fixed_delta_seconds
        self.world.apply_settings(settings)
        vehicle = sensor = None
        tracker = PathTracker()
        count = {"count": 0, "fatal": 0, "events": [], "fatal_events": [],
                 "startup_contacts": [], "frame_stages": {}, "armed": False}
        startup = {"configured_spawn": list(c.garage_spawn), "settling_states": []}
        completed = False
        failure = ""
        sim_time = 0.0
        expected, observed, events = [], [], []
        logs = []
        control_geometry = {}
        physics_evidence = {"custom_physics_applied": False, "readback_matches": False,
                            "note": "Vehicle physics has not yet been read back."}
        try:
            tasks = self.plan()
            expected = _collapse_consecutive_stages(t["type"] for t in tasks)
            vehicle = self._spawn_vehicle()
            startup["spawn_state"] = vehicle_state(vehicle)
            sensor = self._attach_collision_sensor(vehicle, count)
            requested_physics = self._apply_optional_agri_physics(vehicle)
            if c.apply_front_wheel_steering:
                enforce_front_wheel_steering(vehicle)
            physics = vehicle.get_physics_control()
            physics.use_gear_autobox = bool(c.use_gear_autobox)
            vehicle.apply_physics_control(physics)
            vehicle.set_autopilot(False)
            vehicle.apply_control(self.carla.VehicleControl(throttle=0, brake=1))
            for _ in range(20):
                frame = self.world.tick()
                count["frame_stages"][frame] = "INITIALIZATION"
                startup["settling_states"].append(dict(frame=frame, **vehicle_state(vehicle)))
            physics_evidence = physics_report(vehicle, requested_physics, c.apply_front_wheel_steering,
                                              expect_autobox=bool(c.use_gear_autobox))
            write_json(out / "vehicle_physics.json", physics_evidence)
            if not physics_evidence["readback_matches"]:
                raise ValueError("Physics read-back mismatch: " + "; ".join(physics_evidence["differences"]))
            print(f"Vehicle physics: custom={bool(requested_physics)}, read-back verified; results: {out}")
            control_geometry = self._check_vehicle_geometry(vehicle)
            road_controller = BehaviorAgent(vehicle, behavior="normal",
                                            opt_dict={"sampling_resolution": c.path_spacing_m,
                                                      "base_min_distance": 0.5,
                                                      "distance_ratio": 0.5})
            set_agent_target_speed(road_controller, c.road_speed_kmh)
            index, active_index = 0, -1
            parked_ticks = 0
            progress_mark, progress_time = 0.0, 0.0
            for step in range(c.max_steps):
                frame = self.world.tick()
                count["frame_stages"][frame] = tasks[index]["type"]
                sim_time = (step + 1) * c.fixed_delta_seconds
                task = tasks[index]
                trans = vehicle.get_transform()
                position, yaw = xyz(trans.location), trans.rotation.yaw
                velocity = vehicle.get_velocity()
                speed = math.hypot(velocity.x, velocity.y)
                if index != active_index:
                    follower = MissionPathFollower(task, c)
                    active_index = index
                    progress_mark, progress_time = 0.0, sim_time
                    observed.append(task["type"])
                    events.append(dict(event="stage_started", index=index, stage=task["type"], time_s=sim_time))
                    print(f"[{sim_time:7.2f}s] {index+1}/{len(tasks)} {task['type']}")
                    if "route" in task:
                        road_controller.set_global_plan(agent_route_with_lookahead(task),
                                                        stop_waypoint_creation=True, clean_queue=True)
                obs = follower.observe(position, yaw)
                tracker.update(trans.location, task["type"])
                road_failure = False
                if "route" in task:
                    if road_controller.done() and not obs["arrived"]:
                        road_failure = True
                        control = self.carla.VehicleControl(brake=1.0)
                    else:
                        control = road_controller.run_step()
                    control = limit_road_speed(vehicle, control, c.road_speed_kmh)
                    control.manual_gear_shift = False
                    control.gear = 1
                else:
                    control = follower.control(self.carla, position, yaw, speed, obs)
                vehicle.apply_control(control)
                self._on_tick(vehicle, task, step)
                logs.append(dict(time_s=sim_time, task_index=index, stage=task["type"],
                                 x=position[0], y=position[1], z=position[2], yaw_deg=yaw,
                                 speed_ms=speed, throttle=control.throttle, brake=control.brake,
                                 steer=control.steer, reverse=control.reverse, gear=vehicle.get_control().gear,
                                 collision_count=count["count"], fatal_collision_count=count["fatal"], **obs))
                if road_failure:
                    raise RuntimeError("Road controller finished before measured arrival at the stage gate")
                if not count["armed"] and obs["progress_m"] >= c.collision_arm_distance_m:
                    count["armed"] = True
                if c.stop_on_collision and count["fatal"]:
                    raise RuntimeError(collision_reason(count))
                if obs["cross_track_m"] > c.max_cross_track_m:
                    raise RuntimeError("Cross-track safety limit exceeded")
                if obs["progress_m"] > progress_mark + 0.2:
                    progress_mark, progress_time = obs["progress_m"], sim_time
                if sim_time-progress_time > c.stall_timeout_s:
                    raise RuntimeError("No path progress before stall timeout")
                if index == len(tasks)-1:
                    parked_ticks = parked_ticks+1 if parked(obs, speed, c) else 0
                    if parked(obs, speed, c):
                        vehicle.apply_control(self.carla.VehicleControl(brake=1.0, hand_brake=True))
                    if parked_ticks * c.fixed_delta_seconds >= c.parking_hold_s:
                        events.append(dict(event="stage_completed", index=index, stage=task["type"], time_s=sim_time))
                        completed = True
                        break
                elif obs["arrived"]:
                    events.append(dict(event="stage_completed", index=index, stage=task["type"], time_s=sim_time))
                    index += 1
            if not completed:
                failure = "Maximum simulation steps reached"
        except (Exception, KeyboardInterrupt) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if vehicle is not None and vehicle.is_alive:
                vehicle.apply_control(self.carla.VehicleControl(brake=1, hand_brake=True))
            if sensor is not None and sensor.is_alive:
                sensor.destroy()
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
            self.world.apply_settings(original_settings)

        path_csv = tracker.save_csv(out / "full_process_path.csv")
        path_png = tracker.save_plot(out / "full_process_path.png")
        transitions = _stage_transitions_completed(expected, observed, completed)
        summary = self.performance_evaluation_subagent_class.task_level_summary(
            task_completed=completed, stage_transition_completed=transitions,
            forward_reverse_completed=None, collision_count=count["count"],
            total_distance_m=tracker.total_distance, operation_path_length_m=tracker.work_distance,
            task_duration_s=sim_time)
        summary_path = out / "full_process_task_evaluation.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        raw_path = out / "per_tick_states.csv"
        with raw_path.open("w", newline="", encoding="utf-8-sig") as stream:
            fields = list(logs[0]) if logs else ["time_s", "task_index", "stage", "x", "y", "yaw_deg", "speed_ms"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(logs)
        write_json(out / "stage_events.json", events)
        contacts = collision_records(count)
        write_json(out / "collision_events.json", contacts)
        write_json(out / "startup_diagnostics.json", startup)
        write_json(out / "vehicle_physics.json", physics_evidence)
        status = dict(status="COMPLETED" if completed else "FAILED", task_completed=completed,
                      stage_transition_completed=transitions, failure_reason=failure,
                      expected_stages=expected, observed_stages=observed, map_name=c.carla_map,
                      collision_count=count["count"], fatal_collision_count=count["fatal"],
                      startup_contact_count=len(count["startup_contacts"]), collision_events=contacts,
                      startup_diagnostics=str(out / "startup_diagnostics.json"),
                      collision_policy=f"static map contacts are non-fatal until {c.collision_arm_distance_m} m of stage progress; all other contacts are always fatal",
                      total_ticks=len(logs), duration_s=sim_time,
                      forward_reverse_operation="NOT_APPLICABLE_FORWARD_ONLY",
                      control_geometry=control_geometry,
                      custom_physics_applied=physics_evidence["custom_physics_applied"],
                      physics_readback_matches=physics_evidence["readback_matches"],
                      distance_metric="Unchanged PathTracker; work_distance counts HARVEST_LINE only",
                      initialization_ticks_excluded=20)
        write_json(out / "run_status.json", status)
        result = dict(path_csv=str(path_csv), path_plot=str(path_png) if path_png else "",
                      task_evaluation=str(summary_path), raw_states=str(raw_path),
                      vehicle_physics=str(out / "vehicle_physics.json"),
                      collision_events=str(out / "collision_events.json"),
                      startup_diagnostics=str(out / "startup_diagnostics.json"),
                      run_status=str(out / "run_status.json"), planned_mission=str(out / "planned_mission.json"))
        if not completed:
            raise FullProcessRunFailed(f"{failure}. Full logs saved in {out}")
        return result

    def _spawn_vehicle(self):
        bp = self.world.get_blueprint_library().find(self.config.vehicle_bp)
        x, y, z, yaw = self.config.garage_spawn
        vehicle = self.world.try_spawn_actor(bp, self.carla.Transform(
            self.carla.Location(x=x, y=y, z=z+self.config.spawn_height_offset_m),
            self.carla.Rotation(yaw=yaw)))
        if vehicle is None:
            raise RuntimeError("Vehicle spawning failed. Check the garage and blueprint.")
        return vehicle

    def _attach_collision_sensor(self, vehicle, count):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        sensor = self.world.spawn_actor(bp, self.carla.Transform(), attach_to=vehicle)
        def on_collision(event):
            impulse = xyz(event.normal_impulse)
            other = event.other_actor
            row = dict(frame=int(event.frame), timestamp_s=float(event.timestamp),
                       other_actor_id=int(getattr(other, "id", -1)),
                       other_actor_type=str(getattr(other, "type_id", "unknown")),
                       other_semantic_tags=list(getattr(other, "semantic_tags", [])),
                       sensor_position_m=xyz(event.transform.location), normal_impulse=impulse,
                       normal_impulse_magnitude=math.sqrt(sum(x*x for x in impulse)))
            count["events"].append(row)
            count["count"] += 1
            if count["armed"] or not row["other_actor_type"].startswith("static."):
                count["fatal"] += 1
                count["fatal_events"].append(row)
            else:
                count["startup_contacts"].append(row)
            print(f"[collision] frame={row['frame']} other={row['other_actor_type']} "
                  f"id={row['other_actor_id']} impulse={row['normal_impulse_magnitude']:.3f} "
                  f"{('FATAL' if count['armed'] or not row['other_actor_type'].startswith('static.') else 'startup')}",
                  flush=True)
        sensor.listen(on_collision)
        return sensor

    def _apply_optional_agri_physics(self, vehicle):
        if self.config.apply_agri_physics:
            from agri_carla.machinery.physics import AgriPhysicsConfig, apply_agri_physics
            config = AgriPhysicsConfig(self.config.vehicle_type)
            allowed = config.to_dict()
            for key, value in (self.config.physics_overrides or {}).items():
                if key not in allowed:
                    raise ValueError(f"Unknown tractor physics field: {key}")
                setattr(config, key, value)
            apply_agri_physics(vehicle, config)
            return config.to_dict()
        if self.config.physics_overrides:
            raise ValueError("Physics overrides require apply_agri_physics=true")
        return None

    def _on_tick(self, vehicle, task, step):
        """Optional visualization hook; measurements and controls remain in run()."""

    def _check_vehicle_geometry(self, vehicle):
        wheels = vehicle.get_physics_control().wheels
        if len(wheels) < 4:
            raise ValueError("The experiment requires a four-wheel tractor")
        if any(abs(w.max_steer_angle) > 1e-3 for w in wheels[2:4]):
            raise ValueError("Rear-wheel steering is enabled; front-steered case cannot run")
        max_steer = min(w.max_steer_angle for w in wheels[:2])
        if not 0 < max_steer < 90:
            raise ValueError("Invalid front-wheel maximum steering angle")
        # Wheel positions are in world centimetres: projecting onto world X
        # would incorrectly yield zero for the north-facing garage spawn.
        axle_delta = [(sum(getattr(w.position, axis) for w in wheels[:2])
                       - sum(getattr(w.position, axis) for w in wheels[2:4])) / 2
                      for axis in ("x", "y", "z")]
        wheelbase = math.sqrt(sum(d*d for d in axle_delta)) / 100
        if not math.isfinite(wheelbase) or wheelbase < 0.5:
            raise ValueError("Cannot infer the wheelbase from wheel positions")
        minimum = wheelbase / math.tan(math.radians(max_steer))
        if self.config.min_turn_radius_m < minimum:
            raise ValueError(f"Designed turning radius is below the blueprint limit {minimum:.2f} m")
        self.config.wheelbase_m = wheelbase
        self.config.max_steer_angle_deg = max_steer
        return dict(wheelbase_m=wheelbase, front_max_steer_deg=max_steer,
                    kinematic_min_radius_m=minimum, designed_radius_m=self.config.min_turn_radius_m)
