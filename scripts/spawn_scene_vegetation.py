"""在已导入 CARLA 的农业地图中规划或生成树木和草地区域。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.agents.scene_generation import SceneGenerationAgent, vegetation_with_road_context
from agri_carla.environment.vegetation import dump_available_static_blueprints
from agri_carla.utils.io import read_yaml


def main() -> None:
    """处理蓝图查询、仅规划预览和实际生成三种运行方式。"""
    parser = argparse.ArgumentParser(description="Place tree and grass regions in an imported CARLA agricultural map.")
    parser.add_argument("--request", required=True, help="JSON/YAML vegetation placement request.")
    parser.add_argument(
        "--map",
        default="",
        help="Override carla_map with the exact map path reported by CARLA.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only generate placement plan and preview; do not connect to CARLA.")
    parser.add_argument("--spawn", action="store_true", help="Connect to CARLA and place vegetation even if the request has dry_run=true.")
    parser.add_argument("--list-static-blueprints", default="", help="Write available CARLA static blueprints to this JSON file.")
    args = parser.parse_args()

    payload = read_yaml(args.request)
    if not isinstance(payload, dict):
        raise ValueError("Vegetation request must be a JSON/YAML object.")
    # SGA 的完整结构化任务文件可直接作为输入，脚本会自动取出其中的
    # vegetation_request；同时继续兼容只包含植被参数的旧请求文件。
    request = payload.get("vegetation_request", payload)
    if not isinstance(request, dict):
        raise ValueError("vegetation_request must be a JSON/YAML object.")
    request = dict(request)
    if payload.get("road_scene_request"):
        derived = Path(payload.get("road_scene_output_dir", "outputs/road_scene")) / "vegetation_runtime_request.json"
        if derived.is_file():
            request = json.loads(derived.read_text(encoding="utf-8"))
        else:
            request = vegetation_with_road_context(request, payload["road_scene_request"])
    if args.map:
        # 自定义地图在不同 CARLA 包中的注册路径可能不同，允许用户直接使用
        # get_available_maps 返回的完整名称，而无须修改结构化任务文件。
        request["carla_map"] = args.map
    if args.list_static_blueprints:
        # 先列出当前 CARLA 版本可用静态蓝图，便于在请求中选择真实资产名。
        path = dump_available_static_blueprints(
            request.get("host", "localhost"),
            int(request.get("port", 2000)),
            float(request.get("timeout", 30.0)),
            args.list_static_blueprints,
        )
        print(path)
        return

    if args.spawn:
        # 命令行显式选项优先于请求文件中的 dry_run 值。
        request["dry_run"] = False
    elif args.dry_run:
        request["dry_run"] = True
    result = SceneGenerationAgent().run_vegetation(request)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
