"""轨迹跟踪和全流程导航使用的基础评价指标。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TrackingMetrics:
    """一组横向误差的 MAE、RMSE、最大值和有效样本数。"""
    mae_m: float
    rmse_m: float
    max_error_m: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        """转换为便于写入表格的键值字典。"""
        return asdict(self)


def compute_tracking_metrics(errors) -> TrackingMetrics:
    """过滤非有限值后计算一维跟踪误差统计量。"""
    values = np.asarray(errors, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return TrackingMetrics(mae_m=float("nan"), rmse_m=float("nan"), max_error_m=float("nan"), samples=0)
    abs_values = np.abs(values)
    return TrackingMetrics(
        mae_m=float(np.mean(abs_values)),
        rmse_m=float(np.sqrt(np.mean(values ** 2))),
        max_error_m=float(np.max(abs_values)),
        samples=int(values.size),
    )


def summarize_tick_records(record_csv: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    """从逐帧记录中的 ``cte_m`` 列生成简要误差表。"""
    df = pd.read_csv(record_csv)
    metrics = compute_tracking_metrics(df["cte_m"].values)
    rows = [metrics.to_dict()]
    summary = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return summary


def task_level_summary(
    task_completed: bool,
    stage_transition_completed: bool,
    forward_reverse_completed: bool | None,
    collision_count: int,
    total_distance_m: float,
    operation_path_length_m: float,
    task_duration_s: float,
) -> pd.DataFrame:
    """生成论文全流程导航案例所用的任务级评价表。

    完成性项目以 Completed/Failed 表示；碰撞、距离和时长保留为可直接
    报告的数值结果。该函数不推断任务状态，只整理运行器提供的观测值。
    """
    return pd.DataFrame([
        {
            "evaluation_item": "Task completion",
            "description": "Whether the vehicle completed departure, road driving, field entry, field operation and return",
            "result": "Completed" if task_completed else "Failed",
        },
        {
            "evaluation_item": "Stage transition",
            "description": "Whether different navigation stages were switched successfully",
            "result": "Completed" if stage_transition_completed else "Failed",
        },
        {
            "evaluation_item": "Forward/reverse operation",
            "description": "Whether forward and reverse segments were executed correctly",
            "result": ("Not applicable (forward-only task)" if forward_reverse_completed is None
                       else "Completed" if forward_reverse_completed else "Failed"),
        },
        {
            "evaluation_item": "Collision event",
            "description": "Whether collision occurred during the task",
            "result": collision_count,
        },
        {
            "evaluation_item": "Total travel distance",
            "description": "Simulated travel distance during the task",
            "result": f"{total_distance_m:.3f} m",
        },
        {
            "evaluation_item": "Operation path length",
            "description": "Path length inside the field operation area",
            "result": f"{operation_path_length_m:.3f} m",
        },
        {
            "evaluation_item": "Task duration",
            "description": "Total simulation time",
            "result": f"{task_duration_s:.3f} s",
        },
    ])
