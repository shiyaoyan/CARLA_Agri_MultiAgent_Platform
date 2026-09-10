"""独立调用 MCA 生成农机参数模板的命令行脚本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.agents import MachineryConfigurationAgent


def main() -> None:
    """解析农机类型、模型文件和输出路径并生成 JSON 模板。"""
    parser = argparse.ArgumentParser(description="Generate agricultural machinery configuration templates.")
    parser.add_argument("--type", choices=["tractor"], required=True)
    parser.add_argument("--model-file", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    template = MachineryConfigurationAgent().run(args.type, model_file=args.model_file, output_file=args.output)
    print(f"Template for {template.machinery_type} written to: {args.output}")


if __name__ == "__main__":
    main()
