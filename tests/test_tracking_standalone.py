"""Packaging and output equivalence on a fixed mocked CARLA state trace."""
import ast
import copy
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from agri_carla.agents.experiment_execution import ExperimentExecutionAgent
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig
from agri_carla.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent
from agri_carla.trajectory.reference_path import ReferenceTrajectory
from scripts.build_tracking_standalone import build

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"standalone_trajectory_tracking.py"


def write_fixture(path):
    metres = np.r_[np.arange(9), np.arange(7, -1, -1)]*.8
    latitude = 37.43317
    pd.DataFrame({
        "Longitude (°)": 121.512558 + np.degrees(metres/(6378137*math.cos(math.radians(latitude)))),
        "Latitude (°)": latitude,
        "Time": pd.date_range("2026-10-15 08:00:00", periods=len(metres), freq="1s"),
        "Speed": [2.88]*(len(metres)-1)+[0],
        "State": [1]*9+[0]*8,
    }).to_csv(path, index=False)


def fake_carla(trajectory, config):
    """Replay identical observations; this is not a vehicle dynamics simulator."""
    world = NS(ticks=0, debug=NS(draw_point=Mock()), apply_settings=Mock())
    world.get_settings = lambda: NS(synchronous_mode=False, fixed_delta_seconds=None)

    def tick():
        world.ticks += 1
        return world.ticks

    world.tick = tick
    car = NS(is_alive=True, physics=NS(wheels=[NS() for _ in range(4)]),
             apply_control=Mock(), get_physics_control=lambda: car.physics)

    def state():
        time = max(0, world.ticks-11)*config.dt
        return NS(
            location=NS(x=float(np.interp(time, trajectory.t, trajectory.x))+config.map_offset_x,
                        y=config.map_offset_y+.12*math.sin(time), z=config.spawn_z),
            rotation=NS(yaw=0))

    def velocity():
        time = max(0, world.ticks-11)*config.dt
        speed = 0 if time < 1 or 8 <= time <= 9.2 or time >= trajectory.t[-1] else .8
        return NS(x=speed, y=0, z=0)

    def destroy():
        car.is_alive = False

    car.get_transform = state
    car.get_velocity = velocity
    car.apply_physics_control = lambda physics: setattr(car, "physics", physics)
    car.destroy = destroy
    world.get_blueprint_library = lambda: NS(find=lambda _: NS())
    world.try_spawn_actor = lambda *_: car
    client = NS(set_timeout=Mock(), get_world=lambda: world, load_world=lambda _: world)
    return NS(Client=lambda *_: client, Location=lambda **values: NS(**values),
              Rotation=lambda **values: NS(**values),
              Transform=lambda location, rotation: NS(location=location, rotation=rotation),
              Color=lambda *args: args, VehicleControl=NS,
              Vector2D=lambda **values: NS(**values), Vector3D=lambda x,y,z: NS(x=x,y=y,z=z),
              GearPhysicsControl=NS)


class TrackingStandaloneTests(unittest.TestCase):
    def test_generated_file_and_embedded_source_hashes(self):
        self.assertEqual(SCRIPT.read_text(), build(ROOT))
        ns = runpy.run_path(str(SCRIPT))
        ast.parse(SCRIPT.read_text(), feature_version=(3, 8))
        for name, source in ns["MODULE_SOURCES"].items():
            with self.subTest(module=name):
                ast.parse(source, feature_version=(3, 8))
        for name, digest in ns["SOURCE_SHA256"].items():
            path = ROOT/"agri_carla"/(name.replace(".", "/")+".py")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_offline_check_has_no_project_carla_or_crewai_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = shutil.copy2(SCRIPT, tmp)
            result = subprocess.run([sys.executable, "-I", script, "--check"], cwd=tmp,
                                    env={**os.environ, "PYTHONPATH": ""},
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertFalse(data["carla_executed"])
            self.assertFalse(data["crew_ai_used"])
            self.assertEqual(data["tractor_physics"]["long_stiff_value"], 5000)

    def test_unfilled_input_gets_an_actionable_error(self):
        result = subprocess.run([sys.executable, "-I", str(SCRIPT)], capture_output=True,
                                text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Fill TRAJECTORY_CSV", result.stderr)
        self.assertNotIn("CARLA Python API is not available", result.stderr)

    def test_one_file_prepares_relative_input_without_carla(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            script = shutil.copy2(SCRIPT, base)
            raw = base/"measured.csv"
            write_fixture(raw)
            before = raw.read_bytes()
            result = subprocess.run(
                [sys.executable, "-I", script, "--trajectory", "measured.csv",
                 "--output-dir", "results", "--prepare-only"],
                cwd=ROOT, env={**os.environ, "PYTHONPATH": ""},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            processed = base/"results/processed_trajectory/measured_standard.csv"
            self.assertTrue(processed.is_file())
            self.assertEqual(raw.read_bytes(), before)
            self.assertIn("PREPARED_ONLY", result.stdout)
            self.assertEqual(len(pd.read_csv(processed)), 17)

    def test_project_and_standalone_produce_identical_files(self):
        ns = runpy.run_path(str(SCRIPT))
        ns["install_embedded_modules"]()
        embedded = importlib.import_module("_tracking_embedded.simulation.tracking_experiment")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = base/"measured.csv"
            write_fixture(raw)
            standard = base/"reference.csv"
            TrajectoryPreprocessingSubagent().process(raw, standard)
            trajectory = ReferenceTrajectory(standard)
            results = []
            configs = [TrackingExperimentConfig(output_dir=str(base/"project")),
                       embedded.TrackingExperimentConfig.from_dict({
                           **copy.deepcopy(ns["EXPERIMENT_CONFIG"]),
                           "output_dir": str(base/"standalone"),
                       })]
            for i, config in enumerate(configs):
                carla = fake_carla(trajectory, config)
                processed = base/("project_input" if i == 0 else "standalone_input")/"standard.csv"
                with patch.dict("sys.modules", {"carla": carla}):
                    if i == 0:
                        result = ExperimentExecutionAgent().run_trajectory_tracking(
                            raw, processed_csv=processed, config=config)
                    else:
                        result = ns["run_tracking"](raw, config, processed_csv=processed)
                results.append(result)
            self.assertEqual(set(results[0]), set(results[1]))
            left = Path(results[0]["tick_records"]).parent
            right = Path(results[1]["tick_records"]).parent
            expected = {"tick_records.csv", "diagnostic_events.csv", "phase_summary.csv",
                        "tracking_metrics_summary.csv", "01_tracking_error_metrics.png",
                        "01_time_sync_diagnostics.png", "02_spatial_trajectory.png",
                        "README_diagnostics.txt"}
            self.assertEqual({p.name for p in left.iterdir()}, expected)
            self.assertEqual({p.name for p in right.iterdir()}, expected)
            for name in expected:
                with self.subTest(file=name):
                    self.assertEqual((left/name).read_bytes(), (right/name).read_bytes())
            self.assertEqual((base/"project_input/standard.csv").read_bytes(),
                             (base/"standalone_input/standard.csv").read_bytes())
            metrics = pd.read_csv(left/"tracking_metrics_summary.csv").iloc[0]
            self.assertGreater(metrics["total_sample_count"], metrics["valid_sample_count"])
            ticks = pd.read_csv(left/"tick_records.csv")
            self.assertTrue(ticks["is_forward"].any())
            self.assertTrue((~ticks["is_forward"]).any())
            self.assertIn("PHASE_SHIFT", (left/"diagnostic_events.csv").read_text(encoding="utf-8-sig"))

    def test_preprocessing_cannot_overwrite_input(self):
        ns = runpy.run_path(str(SCRIPT))
        ns["install_embedded_modules"]()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"input.csv"
            write_fixture(path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "overwrite"):
                ns["prepare_trajectory"](path, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
