"""由农业道路场景请求生成标准跟踪轨迹 CSV。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.trajectory import generate_route_csv_from_scene_request


def main() -> None:
    """按指定采样间距和目标速度离散道路中心线。"""
    parser = argparse.ArgumentParser(description="Generate a standard trajectory CSV from a road-scene request YAML.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacing-m", type=float, default=1.0)
    parser.add_argument("--speed-kmh", type=float, default=4.0)
    args = parser.parse_args()
    output = generate_route_csv_from_scene_request(
        args.request,
        args.output,
        sample_spacing_m=args.spacing_m,
        speed_kmh=args.speed_kmh,
    )
    print(f"Route CSV written to: {output}")


if __name__ == "__main__":
    main()
