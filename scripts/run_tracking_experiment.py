"""用于单独调试 EEA 轨迹跟踪能力的命令行入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.agents import ExperimentExecutionAgent
from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig


def main() -> None:
    """构造跟踪实验配置，并按开关选择固定参数或贝叶斯优化模式。"""
    parser = argparse.ArgumentParser(description="Run trajectory tracking validation in CARLA.")
    parser.add_argument("--trajectory", required=True, help="Standard trajectory CSV.")
    parser.add_argument("--processed-csv", default="", help="Optional path for standardized CSV generated before running.")
    parser.add_argument("--vehicle-bp", default="vehicle.tractor.tractor")
    parser.add_argument("--carla-map", default="", help="Optional existing CARLA map name. If omitted, the current world is used.")
    parser.add_argument("--output-dir", default="results/trajectory_tracking")
    parser.add_argument("--map-offset-x", type=float, default=50.0)
    parser.add_argument("--map-offset-y", type=float, default=-10.0)
    parser.add_argument("--no-flip-y", action="store_true", help="Use this for trajectories generated from CARLA-world road-scene coordinates.")
    parser.add_argument(
        "--auto-optimize-controller",
        action="store_true",
        help="Enable repeated error-driven controller parameter optimization.",
    )
    parser.add_argument(
        "--optimization-trials",
        type=int,
        default=24,
        help="Number of controller candidates, including the baseline.",
    )
    args = parser.parse_args()
    config = TrackingExperimentConfig(
        carla_map=args.carla_map,
        vehicle_bp=args.vehicle_bp,
        output_dir=args.output_dir,
        map_offset_x=args.map_offset_x,
        map_offset_y=args.map_offset_y,
        trajectory_flip_y=not args.no_flip_y,
        controller_optimization={
            # 开关关闭时 max_trials 等优化字段不会触发多轮实验。
            "enabled": args.auto_optimize_controller,
            "method": "bayesian_optimization",
            "max_trials": args.optimization_trials,
            "confirm_best": True,
        },
    )
    result = ExperimentExecutionAgent().run_trajectory_tracking(
        args.trajectory,
        processed_csv=args.processed_csv or None,
        config=config,
    )
    print(result)


if __name__ == "__main__":
    main()
