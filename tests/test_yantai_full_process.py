"""Mission contracts and low-speed kinematic checks, not CARLA integration."""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from agri_carla.controllers.full_process_control import MissionPathFollower, parked
from agri_carla.controllers.task_controls import limit_road_speed
from agri_carla.evaluation.metrics import task_level_summary
from agri_carla.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig
from agri_carla.simulation.full_process_experiment import FullProcessNavigationExperiment, FullProcessRunFailed
from agri_carla.simulation.full_process_geometry import angle_error, distance
from agri_carla.simulation.task_planner import select_gate, checked_road_route
from scripts.preview_yantai_full_process import build_preview
from agri_carla.paper_cases import PaperCaseCatalog

ROOT = Path(__file__).resolve().parents[1]
XODR = ROOT / "data/maps/yantai_satellite/YantaiSatelliteRebuilt.xodr"


def control(**kwargs):
    values = dict(throttle=0.0, brake=0.0, steer=0.0, reverse=False, manual_gear_shift=False,
                  hand_brake=False, gear=1)
    values.update(kwargs)
    return NS(**values)


class YantaiFullProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = FullProcessExperimentConfig()
        cls.tasks, cls.road = build_preview(cls.config, XODR)

    def test_two_pass_contract_and_continuous_stages(self):
        self.assertEqual([t["type"] for t in self.tasks], ["GARAGE_EXIT", "ROAD_NAV", "OFFROAD_APPROACH",
            "HARVEST_LINE", "TURN_BULB", "HARVEST_LINE", "FIELD_EXIT", "ROAD_RETURN", "GARAGE_RETURN"])
        for a, b in zip(self.tasks, self.tasks[1:]):
            self.assertLess(distance(a["points"][-1], b["points"][0]), 1e-8)
        self.assertAlmostEqual(self.tasks[3]["length_m"], math.hypot(7, 65), places=6)
        self.assertNotEqual(self.tasks[1]["route"][0][0].lane_id, self.tasks[7]["route"][0][0].lane_id)

    def test_authoritative_config_and_registry_match_defaults(self):
        request = json.loads((ROOT / "configs/yantai_full_process_agent_request.json").read_text())
        registry = json.loads((ROOT / "configs/map_task_registry/yantai_full_process.json").read_text())
        defaults = json.loads(json.dumps(asdict(self.config)))
        self.assertEqual(request["experiment"]["config"], defaults)
        self.assertEqual(registry["experiment_config"], defaults)

    def test_paper_case_rejects_wrong_map_before_execution(self):
        request = {"task_type": "full_process_navigation", "experiment": {"config": asdict(self.config)}}
        PaperCaseCatalog.validate_case_request("section_3_3_full_process", request)
        request["experiment"]["config"]["carla_map"] = "WrongMap"
        with self.assertRaises(ValueError):
            PaperCaseCatalog.validate_case_request("section_3_3_full_process", request)

    def test_unrequested_world_reload_is_not_performed(self):
        for map_name in ("YantaiSatelliteRebuilt", "WrongMap"):
            world = Mock()
            world.get_map.return_value.name = "/Game/Maps/" + map_name
            client = Mock()
            client.get_world.return_value = world
            with patch("agri_carla.simulation.full_process_experiment.import_carla",
                       return_value=NS(Client=Mock(return_value=client))):
                if map_name == "WrongMap":
                    with self.assertRaises(ValueError):
                        FullProcessNavigationExperiment(replace(self.config))
                else:
                    FullProcessNavigationExperiment(replace(self.config))
            client.load_world.assert_not_called()

    def test_unsupported_or_empty_map_rejected(self):
        for map_name in ("", "YantaiTrackingField"):
            with self.assertRaises(ValueError):
                replace(self.config, carla_map=map_name)

    def test_invalid_configuration(self):
        for kwargs in ({"harvest_passes": 34}, {"harvest_width_m": 0}, {"min_turn_radius_m": 1},
                       {"garage_spawn": [math.nan, 1, 2, 3]}, {"max_steps": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.config, **kwargs)

    def test_waypoint_projection_checks_direction_and_distance(self):
        east = select_gate(self.road.waypoints, self.config.road_entry, -2.5, self.config)
        west = select_gate(self.road.waypoints, self.config.road_return, 177.5, self.config)
        self.assertNotEqual(east.lane_id, west.lane_id)
        with self.assertRaises(ValueError):
            select_gate(self.road.waypoints, (1000, 1000, 0), 0, self.config)

    def test_disconnected_route_rejected(self):
        route = self.tasks[1]["route"]
        with self.assertRaises(ValueError):
            checked_road_route(NS(trace_route=lambda *a: []), route[0][0], route[-1][0], .25)

    def test_safety_braking_is_never_replaced_by_throttle(self):
        vehicle = NS(get_velocity=lambda: NS(x=0, y=0))
        c = limit_road_speed(vehicle, control(brake=1), 6)
        self.assertEqual((c.brake, c.throttle), (1, 0))

    def test_bulb_proximity_cannot_skip_turn(self):
        task = self.tasks[4]
        follower = MissionPathFollower(task, self.config)
        obs = follower.observe(task["points"][-1], task["end_yaw"])
        self.assertFalse(obs["arrived"])
        self.assertGreater(obs["remaining_m"], 20)

    def test_parking_requires_position_progress_heading_and_speed(self):
        task = self.tasks[-1]
        follower = MissionPathFollower(task, self.config)
        for p in task["points"]:
            obs = follower.observe(p, task["end_yaw"])
        self.assertTrue(parked(obs, 0, self.config))
        self.assertFalse(parked(obs, 1, self.config))
        self.assertFalse(parked(follower.observe(task["points"][-1], task["end_yaw"]+90), 0, self.config))
        far = (task["points"][-1][0]+5, task["points"][-1][1], 0)
        self.assertFalse(follower.observe(far, task["end_yaw"])["arrived"])

    def test_reference_following_in_kinematic_model(self):
        # This deliberately excludes CARLA dynamics and BehaviorAgent. It tests
        # the continuous off-road controller on the complete reference geometry.
        c = self.config
        x, y, _, yaw = c.garage_spawn
        yaw = math.radians(yaw)
        speed = 0.0
        max_error = 0.0
        for task in self.tasks:
            follower = MissionPathFollower(task, c)
            for _ in range(6000):
                obs = follower.observe((x,y,0), math.degrees(yaw))
                max_error = max(max_error, obs["cross_track_m"])
                if obs["arrived"] and (task["type"] != "GARAGE_RETURN" or speed <= c.stop_speed_ms):
                    break
                cmd = follower.control(NS(VehicleControl=control), (x,y,0), math.degrees(yaw), speed, obs)
                speed = max(0, speed+(1.5*cmd.throttle-3*cmd.brake-.15*speed)*c.fixed_delta_seconds)
                x += speed*math.cos(yaw)*c.fixed_delta_seconds
                y += speed*math.sin(yaw)*c.fixed_delta_seconds
                yaw += speed/c.wheelbase_m*math.tan(cmd.steer*math.radians(c.max_steer_angle_deg))*c.fixed_delta_seconds
            else:
                self.fail(f"Kinematic model did not finish {task['type']}: {obs}")
        self.assertLess(max_error, 1.2)
        self.assertLess(distance((x,y,0), c.garage_return), c.gate_tolerance_m)

    def test_forward_only_metric_is_not_a_fake_reverse_success(self):
        summary = task_level_summary(False, False, None, 0, 1, 1, 1)
        row = summary[summary.evaluation_item == "Forward/reverse operation"].iloc[0]
        self.assertIn("Not applicable", row.result)

    def test_wheelbase_is_invariant_to_world_yaw(self):
        for yaw in (0, 90, 177):
            with self.subTest(yaw=yaw):
                angle = math.radians(yaw)
                wheels = []
                for x, y, steer in ((125,-80,35),(125,80,35),(-125,-80,0),(-125,80,0)):
                    wheels.append(NS(max_steer_angle=steer, position=NS(
                        x=1000+x*math.cos(angle)-y*math.sin(angle),
                        y=2400+x*math.sin(angle)+y*math.cos(angle), z=50)))
                runner = object.__new__(FullProcessNavigationExperiment)
                runner.config = replace(self.config)
                info = runner._check_vehicle_geometry(NS(get_physics_control=lambda: NS(wheels=wheels)))
                self.assertAlmostEqual(info["wheelbase_m"], 2.5)

    def test_timeout_preserves_failure_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = object.__new__(FullProcessNavigationExperiment)
            runner.config = replace(self.config, max_steps=2, output_dir=tmp)
            runner.performance_evaluation_subagent_class = PerformanceEvaluationSubagent
            runner.carla = NS(VehicleControl=control)
            runner.world = Mock()
            runner.world.get_settings.side_effect = lambda: NS()
            runner.world.tick.side_effect = iter(range(1, 30))
            pos = NS(x=0.,y=0.,z=0.)
            vehicle = Mock(is_alive=True)
            vehicle.get_transform.return_value = NS(location=pos,rotation=NS(yaw=0))
            vehicle.get_velocity.return_value = NS(x=0,y=0)
            vehicle.get_control.return_value = control()
            runner._spawn_vehicle = Mock(return_value=vehicle)
            runner._attach_collision_sensor = Mock(return_value=None)
            runner._apply_optional_agri_physics = Mock(return_value=None)
            runner._check_vehicle_geometry = Mock(return_value={})
            runner.plan = Mock(return_value=[dict(type="HARVEST_LINE", points=[(i*.25,0,0) for i in range(41)],
                                                 end_yaw=0,speed_kmh=4,length_m=10)])
            module = NS(BehaviorAgent=Mock())
            with patch.dict("sys.modules", {"agents.navigation.behavior_agent": module}), patch(
                    "agri_carla.simulation.full_process_experiment.enforce_front_wheel_steering"), patch(
                    "agri_carla.simulation.full_process_experiment.physics_report",
                    return_value={"custom_physics_applied": False, "readback_matches": True}):
                with self.assertRaises(FullProcessRunFailed):
                    runner.run()
            saved = Path(runner.config.output_dir)
            status = json.loads((saved/"run_status.json").read_text())
            self.assertEqual(status["status"], "FAILED")
            self.assertEqual(status["total_ticks"], 2)
            self.assertTrue((saved/"per_tick_states.csv").is_file())
            vehicle.destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
