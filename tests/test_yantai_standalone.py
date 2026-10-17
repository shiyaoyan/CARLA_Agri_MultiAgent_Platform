"""Offline packaging and physics-application contracts, not CARLA validation."""
import ast
import copy
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from agri_carla.controllers.task_controls import enforce_front_wheel_steering
from agri_carla.machinery.physics import AgriPhysicsConfig
from agri_carla.machinery.physics_audit import physics_report, SCALARS, WHEEL_FIELDS
from agri_carla.simulation.full_process_config import FullProcessExperimentConfig, yantai_tractor_physics
from agri_carla.simulation.full_process_experiment import FullProcessNavigationExperiment
from scripts.build_yantai_standalone import build

ROOT = Path(__file__).resolve().parents[1]


def fake_physics(requested):
    return NS(**{k: requested[k] for k in SCALARS},
              center_of_mass=NS(x=0, y=0, z=requested["center_of_mass"]),
              torque_curve=[NS(x=x, y=y*requested["torque_multiplier"]) for x,y in requested["torque_curve"]],
              forward_gears=[NS(ratio=r*requested["gear_ratio_multiplier"], down_ratio=d, up_ratio=u)
                             for r,d,u in requested["forward_gears"]],
              use_gear_autobox=True,
              wheels=[NS(**{k: requested[k] for k in WHEEL_FIELDS}, radius=50) for _ in range(4)])


class StandaloneTests(unittest.TestCase):
    def test_file_is_generated_from_current_sources(self):
        self.assertEqual((ROOT/"standalone_yantai_full_process.py").read_text(), build(ROOT))

    def test_embedded_sources_are_python38_parseable(self):
        namespace = runpy.run_path(str(ROOT/"standalone_yantai_full_process.py"))
        for name, text in namespace["MODULE_SOURCES"].items():
            with self.subTest(module=name):
                ast.parse(text, feature_version=(3, 8))
        self.assertEqual(namespace["TRACTOR_PHYSICS"], yantai_tractor_physics())

    def test_single_file_imports_without_project_or_crewai(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = shutil.copy2(ROOT/"standalone_yantai_full_process.py", tmp)
            result = subprocess.run([sys.executable, "-I", script, "--check"], cwd=tmp,
                                    env={**os.environ, "PYTHONPATH": ""},
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout[result.stdout.index("{"):])
            self.assertFalse(output["carla_executed"])
            self.assertFalse(output["crew_ai_used"])
            self.assertEqual(output["status"], "OFFLINE_IMPORT_CONFIG_OK")

    def test_navigation_path_finds_sibling_from_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            api = base/"home/ysy/carla/PythonAPI"
            task = api/"yantai_quanliucheng"
            navigation = api/"carla/agents/navigation"
            task.mkdir(parents=True)
            navigation.mkdir(parents=True)
            (navigation/"behavior_agent.py").write_text("class BehaviorAgent: pass\n")
            script = shutil.copy2(ROOT/"standalone_yantai_full_process.py", task)
            program = (
                "import runpy; "
                f"ns = runpy.run_path({script!r}); "
                "found = ns['configure_carla_paths'](''); "
                "from agents.navigation.behavior_agent import BehaviorAgent; "
                f"assert found == {str(api/'carla')!r}; "
                "assert BehaviorAgent.__name__ == 'BehaviorAgent'"
            )
            result = subprocess.run([sys.executable, "-I", "-c", program], cwd=base,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_navigation_path_explicit_root_has_priority(self):
        namespace = runpy.run_path(str(ROOT/"standalone_yantai_full_process.py"))
        with tempfile.TemporaryDirectory() as tmp:
            api = Path(tmp).resolve()/"PythonAPI/carla"
            nav = api/"agents/navigation"
            nav.mkdir(parents=True)
            (nav/"behavior_agent.py").touch()
            with patch.object(sys, "path", list(sys.path)):
                for root in (tmp, str(Path(tmp)/"PythonAPI"), str(api)):
                    self.assertEqual(namespace["configure_carla_paths"](root), str(api))
                    self.assertEqual(sys.path[0], str(api))
                    self.assertEqual(sys.path.count(str(api)), 1)

    def test_navigation_path_missing_does_not_insert_guessed_directories(self):
        namespace = runpy.run_path(str(ROOT/"standalone_yantai_full_process.py"))
        before = list(sys.path)
        with patch.object(Path, "is_file", return_value=False):
            self.assertIsNone(namespace["configure_carla_paths"](""))
        self.assertEqual(sys.path, before)

    def test_front_wheel_setting_reassigns_carla_wheel_list(self):
        requested = AgriPhysicsConfig().to_dict()
        initial = fake_physics(requested)

        class CopyOnRead:
            def __init__(self):
                self._wheels = initial.wheels
            @property
            def wheels(self):
                return copy.deepcopy(self._wheels)
            @wheels.setter
            def wheels(self, value):
                self._wheels = value

        physics = CopyOnRead()
        vehicle = NS(get_physics_control=lambda: physics, apply_physics_control=lambda p: None)
        enforce_front_wheel_steering(vehicle)
        self.assertEqual([w.max_steer_angle for w in physics.wheels], [48,48,0,0])

    def test_readback_accepts_expected_and_flags_wrong_values(self):
        request = AgriPhysicsConfig().to_dict()
        physics = fake_physics(request)
        for wheel in physics.wheels[2:]:
            wheel.max_steer_angle = 0
        vehicle = NS(get_physics_control=lambda: physics)
        self.assertTrue(physics_report(vehicle, request, True)["readback_matches"])
        physics.mass = 1
        report = physics_report(vehicle, request, True)
        self.assertFalse(report["readback_matches"])
        self.assertIn("physics.mass", report["differences"][0])

    def test_runner_applies_configured_physics_by_default(self):
        runner = object.__new__(FullProcessNavigationExperiment)
        runner.config = FullProcessExperimentConfig()
        with patch("agri_carla.machinery.physics.apply_agri_physics") as apply:
            result = runner._apply_optional_agri_physics(object())
        self.assertEqual(result, yantai_tractor_physics())
        apply.assert_called_once()

    def test_runner_honors_explicit_snapshot_and_rejects_typo(self):
        runner = object.__new__(FullProcessNavigationExperiment)
        runner.config = FullProcessExperimentConfig(physics_overrides={"mass": 6000.0})
        with patch("agri_carla.machinery.physics.apply_agri_physics"):
            self.assertEqual(runner._apply_optional_agri_physics(object())["mass"], 6000)
        runner.config.physics_overrides = {"mas": 6000}
        with self.assertRaisesRegex(ValueError, "Unknown"):
            runner._apply_optional_agri_physics(object())


if __name__ == "__main__":
    unittest.main()
