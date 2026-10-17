"""Regression checks against the author's preserved 2026-10-15 CARLA evidence."""
import ast
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

from agri_carla.controllers.full_process_control import MissionPathFollower
from agri_carla.machinery.physics import AgriPhysicsConfig
from agri_carla.machinery.physics_audit import physics_report
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig, yantai_tractor_physics
from agri_carla.simulation.full_process_experiment import FullProcessNavigationExperiment, collision_records
from scripts.audit_yantai_full_process_run import audit
from tests.test_yantai_standalone import fake_physics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/"data/experiment_results/section_3_3/yantai_satellite"


def embedded_sources(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "MODULE_SOURCES":
            result[ast.literal_eval(target.slice)] = ast.literal_eval(node.value)
    return result


class WithoutDocstrings(ast.NodeTransformer):
    def visit_Module(self, node):
        return self._strip(node)

    def visit_ClassDef(self, node):
        return self._strip(node)

    def visit_FunctionDef(self, node):
        return self._strip(node)

    def _strip(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


class ReconciliationTests(unittest.TestCase):
    def test_archived_run_reconciles_without_editing_raw_files(self):
        run = EVIDENCE/"run_20261015T010039Z_f27a13ca"
        before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(run.iterdir()) if path.is_file()}
        result = audit(run)
        self.assertTrue(result["audit_passed"], result["errors"])
        self.assertEqual(before, result["raw_sha256"])
        self.assertAlmostEqual(result["total_distance_m"], 281.6183423043411)
        self.assertAlmostEqual(result["harvest_line_distance_m"], 130.29925277007627)
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(run.iterdir()) if path.is_file()}
        self.assertEqual(before, after, "the reconciliation must not rewrite raw results")

    def test_executable_core_matches_submitted_runner(self):
        submitted = embedded_sources(EVIDENCE/"source/standalone_yantai_full_process.py")
        generated = embedded_sources(ROOT/"standalone_yantai_full_process.py")
        # The case now pins the recorded physics explicitly. Global defaults in
        # the legacy project retain their pre-existing gearbox, outside this case.
        exceptions = {"_yantai_embedded.simulation.full_process_config",
                      "_yantai_embedded.machinery.physics"}
        self.assertEqual(set(submitted), set(generated))
        for key in submitted.keys() - exceptions:
            with self.subTest(module=key):
                left = ast.dump(WithoutDocstrings().visit(ast.parse(submitted[key])))
                right = ast.dump(WithoutDocstrings().visit(ast.parse(generated[key])))
                self.assertEqual(left, right)

    def test_global_torque_and_stiffness_updated(self):
        actual, expected = AgriPhysicsConfig().to_dict(), yantai_tractor_physics()
        self.assertEqual(actual["torque_curve"], expected["torque_curve"])
        self.assertEqual(actual["long_stiff_value"], 5000)

    def test_config_snapshots_do_not_share_mutable_values(self):
        first, second = FullProcessExperimentConfig(), FullProcessExperimentConfig()
        first.physics_overrides["torque_curve"].append((9999, 0))
        self.assertEqual(second.physics_overrides, yantai_tractor_physics())

    def test_throttle_ramp_and_manual_option(self):
        config = replace(FullProcessExperimentConfig(), use_gear_autobox=False)
        task = dict(type="HARVEST_LINE", points=[(0, 0, 0), (1, 0, 0)], end_yaw=0, speed_kmh=4)
        follower = MissionPathFollower(task, config)
        follower.pid.step = Mock(return_value=(1.0, 0.0, 0.0))
        obs = follower.observe((0, 0, 0), 0)
        for expected in (0.15, 0.30, 0.45):
            control = follower.control(NS(VehicleControl=NS), (0, 0, 0), 0, 0, obs)
            self.assertAlmostEqual(control.throttle, expected)
            self.assertTrue(control.manual_gear_shift)
            self.assertEqual(control.gear, 1)
        follower.pid.step.return_value = (0.0, 1.0, 0.0)
        control = follower.control(NS(VehicleControl=NS), (0, 0, 0), 0, 1, obs)
        self.assertEqual((control.throttle, control.brake), (0, 1))

    def test_all_contacts_are_counted_even_when_not_fatal(self):
        runner = object.__new__(FullProcessNavigationExperiment)
        runner.world = Mock()
        runner.carla = NS(Transform=lambda: None)
        count = dict(count=0, fatal=0, events=[], fatal_events=[], startup_contacts=[],
                     frame_stages={1: "INITIALIZATION", 2: "GARAGE_EXIT", 3: "ROAD_RETURN"}, armed=False)
        sensor = runner._attach_collision_sensor(object(), count)
        callback = sensor.listen.call_args.args[0]

        def event(frame, actor):
            return NS(frame=frame, timestamp=frame*.05,
                      normal_impulse=NS(x=0, y=0, z=1),
                      transform=NS(location=NS(x=0, y=0, z=0)),
                      other_actor=NS(id=2, type_id=actor, semantic_tags=[]))

        callback(event(1, "static.road"))
        callback(event(2, "vehicle.other"))
        count["armed"] = True
        callback(event(3, "static.wall"))
        self.assertEqual((count["count"], count["fatal"], len(count["startup_contacts"])), (3, 2, 1))
        self.assertEqual([r["stage"] for r in collision_records(count)],
                         ["INITIALIZATION", "GARAGE_EXIT", "ROAD_RETURN"])

    def test_requested_gearbox_is_checked(self):
        request = yantai_tractor_physics()
        physics = fake_physics(request)
        for wheel in physics.wheels[2:]:
            wheel.max_steer_angle = 0
        vehicle = NS(get_physics_control=lambda: physics)
        self.assertFalse(physics_report(vehicle, request, True, expect_autobox=False)["readback_matches"])
        physics.use_gear_autobox = False
        self.assertTrue(physics_report(vehicle, request, True, expect_autobox=False)["readback_matches"])


if __name__ == "__main__":
    unittest.main()
