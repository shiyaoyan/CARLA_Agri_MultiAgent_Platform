"""Package the project's fixed-parameter tracking experiment into one Python file."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
from pprint import pformat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_yantai_standalone import LICENSE_TEXT

PREFIX = "_tracking_embedded"
MODULES = (
    "utils.io", "utils.geometry", "utils.carla_import",
    "trajectory.schema", "trajectory.preprocessing_subagent", "trajectory.reference_path",
    "controllers.pid", "controllers.pure_pursuit", "controllers.gear_state_machine",
    "controllers.interface_agent", "machinery.physics", "evaluation.metrics",
    "evaluation.performance_evaluation_subagent", "simulation.tracking_experiment",
)


class IsolateImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("agri_carla."):
            node.module = PREFIX + node.module[len("agri_carla"):]
        return node


RUNTIME = r'''
class EmbeddedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULE_SOURCES or fullname in PACKAGE_NAMES:
            return importlib.util.spec_from_loader(
                fullname, self, is_package=fullname in PACKAGE_NAMES)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(Path(__file__).resolve()) + "::" + module.__name__
        if module.__name__ in MODULE_SOURCES:
            exec(compile(MODULE_SOURCES[module.__name__], module.__file__, "exec"), module.__dict__)


def install_embedded_modules():
    if not any(isinstance(finder, EmbeddedModules) for finder in sys.meta_path):
        sys.meta_path.insert(0, EmbeddedModules())


def script_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def prepare_trajectory(source, destination=None):
    from _tracking_embedded.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent
    if destination is None:
        return Path(source)
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("The standardized CSV must not overwrite the input CSV")
    return TrajectoryPreprocessingSubagent().process(source, destination)


def run_tracking(source, config, processed_csv=None):
    """Same fixed-parameter TPS -> tracking runner -> PES path as the project EEA."""
    from _tracking_embedded.simulation.tracking_experiment import TrackingExperimentRunner
    if (config.controller_optimization or {}).get("enabled"):
        raise ValueError("This file runs fixed controller parameters. Disable optimization "
                         "and enter the locked controller values in EXPERIMENT_CONFIG.")
    csv_to_run = prepare_trajectory(source, processed_csv)
    return TrackingExperimentRunner(config).run(csv_to_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default=TRAJECTORY_CSV)
    parser.add_argument("--output-dir", default=EXPERIMENT_CONFIG["output_dir"])
    parser.add_argument("--host", default=EXPERIMENT_CONFIG["host"])
    parser.add_argument("--port", type=int, default=EXPERIMENT_CONFIG["port"])
    parser.add_argument("--check", action="store_true", help="Offline imports/config only; no CSV/CARLA required")
    parser.add_argument("--prepare-only", action="store_true", help="TPS and input checks only; no CARLA")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    install_embedded_modules()
    from _tracking_embedded.machinery.physics import AgriPhysicsConfig
    from _tracking_embedded.simulation.tracking_experiment import TrackingExperimentConfig
    from _tracking_embedded.trajectory.reference_path import ReferenceTrajectory
    values = copy.deepcopy(EXPERIMENT_CONFIG)
    values.update(host=args.host, port=args.port, output_dir=str(script_path(args.output_dir)))
    if values.get("vegetation_request"):
        raise ValueError("This standalone file tracks on an existing map; set vegetation_request=None")
    if (values.get("controller_optimization") or {}).get("enabled"):
        raise ValueError("Use locked controller values; this file does not run Bayesian optimization")
    config = TrackingExperimentConfig.from_dict(values)
    if config.controller is not None and abs(config.controller.dt-config.dt) > 1e-12:
        raise ValueError("controller.dt must equal experiment dt")
    if args.check:
        print(json.dumps({
            "status": "OFFLINE_IMPORT_CONFIG_OK", "carla_executed": False,
            "crew_ai_used": False, "experiment_config": asdict(config),
            "tractor_physics": AgriPhysicsConfig(config.vehicle_type).to_dict(),
            "source_sha256": SOURCE_SHA256,
        }, indent=2))
        return
    if not args.trajectory or args.trajectory == "FILL_IN_TRAJECTORY.csv":
        parser.error("Fill TRAJECTORY_CSV at the top of this file, or use --trajectory /path/to/data.csv")
    source = script_path(args.trajectory)
    if not source.is_file():
        parser.error("Input CSV not found: " + str(source))
    destination = None
    if PREPROCESS_INPUT:
        destination = script_path(PROCESSED_CSV) if PROCESSED_CSV else (
            Path(config.output_dir) / "processed_trajectory" / (source.stem + "_standard.csv"))
    csv_to_run = prepare_trajectory(source, destination)
    trajectory = ReferenceTrajectory(csv_to_run, flip_y=config.trajectory_flip_y)
    import numpy as np
    if trajectory.num_points < 2 or not np.all(np.isfinite(trajectory.t)) or trajectory.t[-1] <= 0:
        raise ValueError("At least two trajectory points with a positive duration are required")
    if np.any(np.diff(trajectory.t) < 0):
        raise ValueError("Trajectory timestamps must not go backwards")
    if not all(np.all(np.isfinite(value)) for value in (trajectory.x, trajectory.y, trajectory.speed_ms)):
        raise ValueError("Trajectory contains non-finite coordinates or speed")
    if args.prepare_only:
        trajectory.print_data_check()
        print(json.dumps({"status": "PREPARED_ONLY", "carla_executed": False,
                          "trajectory_csv": str(csv_to_run)}, indent=2))
        return
    print("Tracking on:", config.carla_map or "currently open CARLA world",
          "| custom tractor physics:", config.apply_agri_physics)
    physics = AgriPhysicsConfig(config.vehicle_type)
    if config.apply_agri_physics:
        print("Tractor physics:", json.dumps(physics.to_dict()))
        print("Project steering: max_steer_angle applied to all four wheels; "
              "manual gear state machine. No extra rear-wheel lock is added.")
        gears = (config.controller.gear.forward_gears
                 if config.controller and config.controller.gear else (1, 2, 3))
        if any(gear > len(physics.forward_gears) for gear in gears):
            print("Configuration note: the project's controller allows gears", gears,
                  "but the configured physics has", len(physics.forward_gears),
                  "forward gear(s). Keep controller.gear consistent when choosing experiment settings.")
    result = run_tracking(csv_to_run, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
'''


def build(root=ROOT):
    root = Path(root)
    sys.path.insert(0, str(root))
    from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig
    from agri_carla.controllers.interface_agent import ControllerInterfaceConfig
    from agri_carla.controllers.pid import PIDConfig
    from agri_carla.controllers.pure_pursuit import PurePursuitConfig
    from agri_carla.controllers.gear_state_machine import GearStateConfig
    config = TrackingExperimentConfig()
    config.controller = ControllerInterfaceConfig(
        dt=config.dt, pid=PIDConfig(), pure_pursuit=PurePursuitConfig(), gear=GearStateConfig())
    sources, hashes = {}, {}
    for name in MODULES:
        path = root / "agri_carla" / (name.replace(".", "/")+".py")
        source = path.read_text(encoding="utf-8")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        tree = IsolateImports().visit(ast.parse(source))
        sources[PREFIX+"."+name] = ast.unparse(ast.fix_missing_locations(tree))+"\n"
    packages = {PREFIX} | {PREFIX+"."+name.split(".")[0] for name in MODULES}
    header = '''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone paper Section 3.2 trajectory-tracking experiment.

Fill TRAJECTORY_CSV below, open the CARLA map, then run this file.
Requires Python >= 3.8, matching CARLA Python API, numpy, pandas and matplotlib.
No CrewAI, LLM API, navigation agents, external JSON or project installation.
Project TPS, controller, physics, recording, metrics and plots are embedded.
Fixed controller parameters only; no Bayesian search is triggered.
Relative paths are resolved next to this script.
"""
import argparse
import copy
from dataclasses import asdict
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

'''
    header += "".join("# "+line+"\n" for line in LICENSE_TEXT.splitlines())+"\n"
    header += '# Fill this with your measured CSV path or filename.\nTRAJECTORY_CSV = r"FILL_IN_TRAJECTORY.csv"\n'
    header += '# True matches the project TPS preprocessing workflow. False accepts a standard CSV directly.\n'
    header += 'PREPROCESS_INPUT = True\n'
    header += '# Empty: automatically create <output_dir>/processed_trajectory/<name>_standard.csv.\n'
    header += 'PROCESSED_CSV = r""\n\n'
    header += '# Same effective defaults as TrackingExperimentConfig and ControllerInterfaceAgent.\n'
    header += '# carla_map="" uses the current world. A nonempty map name invokes load_world, as in the project.\n'
    header += '# Speeds in the input CSV are km/h; offsets/spawn height are CARLA world metres.\n'
    header += 'EXPERIMENT_CONFIG = '+pformat(asdict(config), width=100, sort_dicts=False)+"\n\n"
    header += 'SOURCE_SHA256 = '+pformat(hashes, width=110)+"\n\n"
    header += 'PACKAGE_NAMES = '+repr(sorted(packages))+"\nMODULE_SOURCES = {}\n"
    for name, source in sources.items():
        literal = source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        header += f"\n# Bundled source: {name}\nMODULE_SOURCES[{name!r}] = '''{literal}'''\n"
    return header+RUNTIME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"standalone_trajectory_tracking.py")
    args = parser.parse_args()
    args.output.write_text(build(), encoding="utf-8")
    print(args.output)
