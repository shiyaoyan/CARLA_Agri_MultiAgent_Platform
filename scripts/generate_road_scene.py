"""独立调用 SGA 生成 CARLA 农业道路场景的命令行脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.agents import SceneGenerationAgent
from agri_carla.utils.io import read_yaml


def main() -> None:
    """读取道路场景请求并输出 XODR、FBX、纹理及验证结果。"""
    parser = argparse.ArgumentParser(description="Generate CARLA-ready agricultural road scene artifacts.")
    parser.add_argument("--request", required=True, help="Road scene request YAML.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    request = read_yaml(args.request)
    result = SceneGenerationAgent().run(request, args.output_dir)
    print(result)


if __name__ == "__main__":
    main()
