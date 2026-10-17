"""Run or preflight the new satellite-based full-process navigation case."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agri_carla.simulation.full_process_experiment import FullProcessExperimentConfig, FullProcessNavigationExperiment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "configs/yantai_full_process_agent_request.json"))
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--plan-only", action="store_true", help="Check gates and routes in the loaded CARLA map; do not spawn a tractor")
    p.add_argument("--load-map", action="store_true", help="Explicitly allow a world reload, which removes runtime actors")
    p.add_argument("--output-dir")
    args = p.parse_args()
    request = json.loads(Path(args.config).read_text())
    data = dict(request.get("experiment", {}).get("config", request))
    for key in ("host", "port"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    if args.load_map:
        data["reload_world"] = True
    if args.output_dir:
        data["output_dir"] = args.output_dir
    config = FullProcessExperimentConfig(**data)
    runner = FullProcessNavigationExperiment(config)
    if args.plan_only:
        tasks = runner.plan()
        print(json.dumps({"status": "PLANNED_ONLY", "task_count": len(tasks), "output_dir": config.output_dir}))
    else:
        print(json.dumps(runner.run(), indent=2))


if __name__ == "__main__":
    main()
