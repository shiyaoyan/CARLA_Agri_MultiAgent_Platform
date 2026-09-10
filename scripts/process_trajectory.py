"""单独调用 TPS 完成真实拖拉机轨迹标准化与方向识别。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agri_carla.agents import ExperimentExecutionAgent


def main() -> None:
    """将原始轨迹 CSV 转换为平台统一轨迹格式。"""
    parser = argparse.ArgumentParser(description="Process tractor trajectory into the standard CSV format.")
    parser.add_argument("--input", required=True, help="Raw trajectory CSV.")
    parser.add_argument("--output", required=True, help="Standardized output CSV.")
    args = parser.parse_args()
    output = ExperimentExecutionAgent().preprocess_trajectory(
        args.input,
        args.output,
    )
    print(f"Processed trajectory written to: {output}")


if __name__ == "__main__":
    main()
