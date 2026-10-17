"""Build a readable single-file runner from the tested project modules."""
import argparse
import ast
import hashlib
import importlib.util
from pathlib import Path
from pprint import pformat
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "utils.io", "utils.carla_import", "utils.geometry",
    "controllers.pid", "controllers.pure_pursuit", "controllers.task_controls",
    "controllers.full_process_control", "machinery.physics", "machinery.physics_audit",
    "evaluation.metrics", "evaluation.path_tracker", "evaluation.performance_evaluation_subagent",
    "simulation.full_process_geometry", "simulation.full_process_config",
    "simulation.task_planner", "simulation.full_process_experiment",
)
PREFIX = "_yantai_embedded"
LICENSE_TEXT = """MIT License

Copyright (c) 2026 shiyaoyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


class IsolateImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith("agri_carla."):
            node.module = PREFIX + node.module[len("agri_carla"):]
        return node


RUNTIME = r'''
class EmbeddedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULE_SOURCES or fullname in PACKAGE_NAMES:
            return importlib.util.spec_from_loader(fullname, self,
                                                  is_package=fullname in PACKAGE_NAMES)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__name__
        module.__file__ = str(Path(__file__).resolve()) + "::" + name
        if name in MODULE_SOURCES:
            exec(compile(MODULE_SOURCES[name], module.__file__, "exec"), module.__dict__)


def configure_carla_paths(root):
    candidates = [Path(root).expanduser().resolve()] if root else []
    # Support a test folder next to PythonAPI/carla, regardless of the IDE cwd.
    for start in (Path(__file__).resolve().parent, Path.cwd().resolve()):
        candidates.extend([start, *start.parents])
    checked = set()
    for base in candidates:
        for api in (base / "PythonAPI/carla", base / "carla", base):
            if api in checked:
                continue
            checked.add(api)
            if (api / "agents/navigation/behavior_agent.py").is_file():
                value = str(api)
                if value in sys.path:
                    sys.path.remove(value)
                sys.path.insert(0, value)
                return value
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=CASE_CONFIG["host"])
    parser.add_argument("--port", type=int, default=CASE_CONFIG["port"])
    parser.add_argument("--carla-root", default=os.environ.get("CARLA_ROOT", ""))
    parser.add_argument("--output-dir", default=CASE_CONFIG["output_dir"])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="Offline import/config check; no CARLA run")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-debug", action="store_true")
    args = parser.parse_args()
    navigation_path = configure_carla_paths(args.carla_root)
    if navigation_path:
        print("CARLA navigation path:", navigation_path)
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.meta_path.insert(0, EmbeddedModules())
    from _yantai_embedded.simulation.full_process_config import FullProcessExperimentConfig
    from _yantai_embedded.simulation.full_process_experiment import FullProcessNavigationExperiment
    from _yantai_embedded.machinery.physics import AgriPhysicsConfig
    values = dict(CASE_CONFIG)
    values.update(host=args.host, port=args.port, output_dir=args.output_dir,
                  apply_agri_physics=True, physics_overrides=TRACTOR_PHYSICS,
                  reload_world=False, vegetation_request=None)
    config = FullProcessExperimentConfig(**values)
    if set(TRACTOR_PHYSICS) != set(AgriPhysicsConfig().to_dict()):
        raise ValueError("TRACTOR_PHYSICS must contain all supported physics fields")
    print("Map:", config.carla_map, "| custom tractor physics: enabled | front steering only")
    if args.check:
        print(json.dumps({"status": "OFFLINE_IMPORT_CONFIG_OK", "carla_executed": False,
                          "crew_ai_used": False, "source_sha256": SOURCE_SHA256,
                          "tractor_physics": TRACTOR_PHYSICS}, indent=2))
        return
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError(f"CARLA Python API import failed in {sys.executable}: {exc}. "
                           "Use the Python environment with the matching CARLA API.") from exc
    try:
        from agents.navigation.behavior_agent import BehaviorAgent
    except ModuleNotFoundError as exc:
        if exc.name == "agents" or (exc.name or "").startswith("agents."):
            raise RuntimeError(
                "CARLA's official navigation agents were not found. The installation must "
                "contain PythonAPI/carla/agents/navigation/behavior_agent.py. "
                "Use --carla-root /path/to/CARLA (not the .venv path), or add "
                "PythonAPI/carla to PYTHONPATH. Do not install an unrelated PyPI agents "
                f"package. Detected navigation path: {navigation_path!r}.") from exc
        raise RuntimeError(f"CARLA navigation import requires the missing dependency "
                           f"{exc.name!r} in {sys.executable}. Original error: {exc}") from exc

    class VisualExperiment(FullProcessNavigationExperiment):
        def _on_tick(self, vehicle, task, step):
            if getattr(self, "_visualization_failed", False):
                return
            try:
                transform = vehicle.get_transform()
                if not args.no_camera and step % 4 == 0:
                    yaw = math.radians(transform.rotation.yaw)
                    location = transform.location
                    self.world.get_spectator().set_transform(carla.Transform(
                        carla.Location(x=location.x-10*math.cos(yaw),
                                       y=location.y-10*math.sin(yaw), z=location.z+9),
                        carla.Rotation(pitch=-35, yaw=transform.rotation.yaw)))
                if not args.no_debug and getattr(self, "_shown_task", None) is not task:
                    self._shown_task = task
                    for a, b in zip(task["points"][::4], task["points"][4::4]):
                        self.world.debug.draw_line(carla.Location(x=a[0], y=a[1], z=a[2]+.3),
                                                   carla.Location(x=b[0], y=b[1], z=b[2]+.3),
                                                   thickness=.06, color=carla.Color(40,160,240),
                                                   life_time=120.0)
                if not args.no_debug and step % 10 == 0:
                    p = transform.location
                    self.world.debug.draw_point(carla.Location(x=p.x, y=p.y, z=p.z+.2),
                                                size=.08, color=carla.Color(230,60,60),
                                                life_time=120.0)
            except Exception as exc:
                self._visualization_failed = True
                print("Visualization disabled; control is unchanged:", exc)

    runner = VisualExperiment(config)
    spectator = runner.world.get_spectator()
    previous_view = spectator.get_transform()
    try:
        if args.plan_only:
            tasks = runner.plan()
            print("PLANNED_ONLY:", len(tasks), "stages; no tractor was spawned.")
        else:
            print(json.dumps(runner.run(), indent=2))
    finally:
        if not args.no_camera:
            spectator.set_transform(previous_view)


if __name__ == "__main__":
    main()
'''


def build(root=ROOT):
    root = Path(root)
    sys.path.insert(0, str(root))
    from dataclasses import asdict
    from agri_carla.simulation.full_process_config import FullProcessExperimentConfig
    # Load this root's physics file directly, even when rebuilding a second project.
    spec = importlib.util.spec_from_file_location("_build_physics", root / "agri_carla/machinery/physics.py")
    physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(physics)
    sources, hashes = {}, {}
    for name in MODULES:
        text = (root / "agri_carla" / (name.replace(".", "/") + ".py")).read_text(encoding="utf-8")
        hashes[name] = hashlib.sha256(text.encode()).hexdigest()
        tree = IsolateImports().visit(ast.parse(text))
        sources[PREFIX + "." + name] = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    packages = {PREFIX} | {PREFIX + "." + name.split(".")[0] for name in MODULES}
    header = '''#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""YantaiSatelliteRebuilt standalone full-process CARLA test.

Run: python standalone_yantai_full_process.py
Requires Python >= 3.8, numpy, pandas, matplotlib, shapely, the matching CARLA
Python API and official CARLA navigation agents/dependencies. No CrewAI, model
API, project installation, external JSON or trajectory CSV is used.
Open YantaiSatelliteRebuilt and start the simulation first. Existing actors
are not cleared. Blue lines are reference paths; red points are actual motion.
Edit CASE_CONFIG and TRACTOR_PHYSICS below; do not edit the generated modules.
This file is generated by scripts/build_yantai_standalone.py from project code.
It has offline tests, not a claimed completed CARLA experiment.
"""
import argparse
import importlib.abc
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

'''
    header += "".join("# " + line + "\n" for line in LICENSE_TEXT.splitlines()) + "\n"
    header += "# Editable experiment settings (CARLA world metres, degrees, km/h).\n"
    header += "CASE_CONFIG = " + pformat(asdict(FullProcessExperimentConfig()), width=100, sort_dicts=False) + "\n\n"
    header += "# Section 3.3 physics snapshot; other experiments retain their own defaults.\n"
    snapshot = physics.AgriPhysicsConfig().to_dict()
    snapshot.update(FullProcessExperimentConfig().physics_overrides or {})
    header += "TRACTOR_PHYSICS = " + pformat(snapshot, width=100, sort_dicts=False) + "\n\n"
    header += "SOURCE_SHA256 = " + pformat(hashes, width=110) + "\n\n"
    header += "PACKAGE_NAMES = " + repr(sorted(packages)) + "\nMODULE_SOURCES = {}\n"
    for name, source in sources.items():
        literal = source.replace("\\", "\\\\").replace("'''", "\\'\\'\\'")
        header += f"\n# Bundled source: {name}\nMODULE_SOURCES[{name!r}] = '''{literal}'''\n"
    return header + RUNTIME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "standalone_yantai_full_process.py")
    options = parser.parse_args()
    options.output.write_text(build(), encoding="utf-8")
    print(options.output)
