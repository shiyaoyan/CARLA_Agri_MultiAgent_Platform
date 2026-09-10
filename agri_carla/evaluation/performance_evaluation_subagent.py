"""实验评价智能体（PES）的逐帧记录、统计和绘图实现。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from agri_carla.utils.io import ensure_dir


class PerformanceEvaluationSubagent:
    """收集 CARLA 逐帧数据并生成与原始实验脚本口径一致的报告。"""

    def __init__(self, output_dir: str | Path):
        """创建输出目录，并初始化逐帧记录和诊断事件缓冲区。"""
        self.output_dir = ensure_dir(output_dir)
        self.rows: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    def add_event(self, sim_time: float, code: str, message: str) -> None:
        """记录换挡、结束等离散事件，同时输出到终端。"""
        row = {"time": sim_time, "code": code, "message": message}
        self.events.append(row)
        print(f"[event {sim_time:7.2f}s] {code}: {message}")

    def add_row(self, row: dict[str, object]) -> None:
        """追加一个仿真步的车辆状态、控制量及误差信息。"""
        self.rows.append(row)

    @staticmethod
    def task_level_summary(**values):
        """转发全流程任务评价请求，保持 PES 为统一评价入口。"""
        from .metrics import task_level_summary

        return task_level_summary(**values)

    def save(self, trajectory=None) -> dict[str, str]:
        """一次性保存原始记录、事件、分段统计、总体指标和诊断图。"""
        records = pd.DataFrame(self.rows)
        events = pd.DataFrame(self.events)
        record_path = self.output_dir / "tick_records.csv"
        event_path = self.output_dir / "diagnostic_events.csv"
        summary_path = self.output_dir / "phase_summary.csv"
        metrics_path = self.output_dir / "tracking_metrics_summary.csv"
        records.to_csv(record_path, index=False, encoding="utf-8-sig")
        events.to_csv(event_path, index=False, encoding="utf-8-sig")

        summary = self.build_phase_summary(records)
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        metrics = self.build_overall_metrics(records)
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        if not records.empty and trajectory is not None:
            self.plot_all(records, trajectory, metrics)
        self.write_readme(metrics)
        return {
            "tick_records": str(record_path),
            "diagnostic_events": str(event_path),
            "phase_summary": str(summary_path),
            "tracking_metrics_summary": str(metrics_path),
        }

    @staticmethod
    def metric_values(records: pd.DataFrame, prefix: str) -> dict[str, float | int]:
        """计算最近路径横向误差和航向误差的常用统计量。

        横向误差取自 ``path_cte_m``，航向误差取自
        ``path_heading_error_deg``；两者均基于最近参考路径线段，而不是
        简单比较同一时间戳的参考点与车辆位置。
        """
        if records.empty:
            return {}
        lateral_error = records["path_cte_m"].astype(float).values
        abs_lateral_error = np.abs(lateral_error)
        heading_error = records["path_heading_error_deg"].astype(float).values
        abs_heading_error = np.abs(heading_error)
        return {
            f"{prefix}_sample_count": len(records),
            f"{prefix}_lateral_rmse_m": math.sqrt(np.mean(lateral_error ** 2)),
            f"{prefix}_lateral_max_abs_m": np.max(abs_lateral_error),
            f"{prefix}_lateral_p95_abs_m": np.percentile(abs_lateral_error, 95),
            f"{prefix}_lateral_std_m": np.std(lateral_error),
            f"{prefix}_lateral_mean_abs_m": np.mean(abs_lateral_error),
            f"{prefix}_heading_rmse_deg": math.sqrt(np.mean(heading_error ** 2)),
            f"{prefix}_heading_max_abs_deg": np.max(abs_heading_error),
            f"{prefix}_heading_p95_abs_deg": np.percentile(abs_heading_error, 95),
            f"{prefix}_heading_std_deg": np.std(heading_error),
            f"{prefix}_heading_mean_abs_deg": np.mean(abs_heading_error),
        }

    def build_overall_metrics(self, records: pd.DataFrame) -> pd.DataFrame:
        """同时汇总全参考时段样本和排除瞬态后的有效样本。

        ``all`` 指参考轨迹有效时间内的全部记录；``valid`` 进一步排除起步、
        结束、低速及换挡瞬态。两套结果并存，便于论文统计和问题诊断。
        """
        if records.empty:
            return pd.DataFrame()
        eval_records = records[records["in_reference_time"].astype(bool)] if "in_reference_time" in records else records
        valid = eval_records[eval_records["metric_valid"].astype(bool)] if "metric_valid" in eval_records else eval_records
        row: dict[str, float | int] = {
            "recorded_sample_count": len(records),
            "total_sample_count": len(eval_records),
            "valid_sample_count": len(valid),
            "post_endpoint_sample_count": len(records) - len(eval_records),
            "valid_ratio": len(valid) / len(eval_records) if len(eval_records) > 0 else 0.0,
        }
        row.update(self.metric_values(valid, "valid"))
        row.update(self.metric_values(eval_records, "all"))
        return pd.DataFrame([row])

    def build_phase_summary(self, records: pd.DataFrame) -> pd.DataFrame:
        """按前进/倒车轨迹段统计误差、速度误差和最大转向输入。"""
        if records.empty:
            return pd.DataFrame()
        if "in_reference_time" in records:
            records = records[records["in_reference_time"].astype(bool)]
        if records.empty:
            return pd.DataFrame()
        rows = []
        for phase_id, group in records.groupby("phase_id", as_index=False):
            metric_row = self.metric_values(group, "all")
            rows.append({
                "phase_id": phase_id,
                "direction": "forward" if bool(group["is_forward"].iloc[0]) else "reverse",
                "start_time_s": group["time_s"].iloc[0],
                "end_time_s": group["time_s"].iloc[-1],
                "mean_pos_err_m": group["pos_err_m"].mean(),
                "max_pos_err_m": group["pos_err_m"].max(),
                "metric_sample_scope": "all",
                "metric_sample_count": len(group),
                "lateral_rmse_m": metric_row["all_lateral_rmse_m"],
                "lateral_max_abs_m": metric_row["all_lateral_max_abs_m"],
                "lateral_p95_abs_m": metric_row["all_lateral_p95_abs_m"],
                "lateral_std_m": metric_row["all_lateral_std_m"],
                "lateral_mean_abs_m": metric_row["all_lateral_mean_abs_m"],
                "heading_rmse_deg": metric_row["all_heading_rmse_deg"],
                "heading_max_abs_deg": metric_row["all_heading_max_abs_deg"],
                "heading_p95_abs_deg": metric_row["all_heading_p95_abs_deg"],
                "heading_std_deg": metric_row["all_heading_std_deg"],
                "heading_mean_abs_deg": metric_row["all_heading_mean_abs_deg"],
                "speed_rmse_kmh": math.sqrt(np.mean((group["actual_abs_kmh"] - group["target_abs_kmh"]) ** 2)),
                "max_abs_steer": np.abs(group["steer"]).max(),
            })
        return pd.DataFrame(rows)

    @staticmethod
    def shade_reverse_spans(ax, time_axis, reverse_mask) -> None:
        """在时序图中以浅色背景标出连续倒车区间。"""
        if len(time_axis) == 0 or not np.any(reverse_mask):
            return
        start = None
        for index, is_reverse in enumerate(reverse_mask):
            if is_reverse and start is None:
                start = time_axis[index]
            if start is not None and (not is_reverse or index == len(reverse_mask) - 1):
                end = time_axis[index] if is_reverse else time_axis[index - 1]
                ax.axvspan(start, end, color="orange", alpha=0.12)
                start = None

    def plot_all(self, records: pd.DataFrame, trajectory, metrics: pd.DataFrame) -> None:
        """绘制误差时序、统计柱状图和参考/实际空间轨迹。"""
        import matplotlib.pyplot as plt

        if "in_reference_time" in records:
            # 参考轨迹结束后的停车过程不进入论文中的误差曲线。
            plot_records = records[records["in_reference_time"].astype(bool)]
        else:
            plot_records = records
        if plot_records.empty:
            return

        time_axis = plot_records["time_s"].values
        reverse_mask = ~plot_records["is_forward"].astype(bool).values
        lateral_error = plot_records["path_cte_m"].astype(float).values
        heading_error = plot_records["path_heading_error_deg"].astype(float).values
        metric_row = metrics.iloc[0] if not metrics.empty else None

        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        axes[0, 0].plot(time_axis, lateral_error, "g-", linewidth=1.1, label="All samples")
        self.shade_reverse_spans(axes[0, 0], time_axis, reverse_mask)
        axes[0, 0].axhline(0, color="black", linewidth=0.8)
        axes[0, 0].set_xlabel("Simulation time (s)")
        axes[0, 0].set_ylabel("Nearest-path lateral error (m)")
        axes[0, 0].legend(loc="upper right")
        axes[0, 0].grid(True, alpha=0.35)

        axes[0, 1].plot(time_axis, heading_error, "b-", linewidth=1.1, label="All samples")
        self.shade_reverse_spans(axes[0, 1], time_axis, reverse_mask)
        axes[0, 1].axhline(0, color="black", linewidth=0.8)
        axes[0, 1].set_xlabel("Simulation time (s)")
        axes[0, 1].set_ylabel("Nearest-path heading error (deg)")
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.35)

        if metric_row is not None:
            names = ["RMSE", "MAE", "P95", "Max", "Std"]
            colors = ["#4daf4a", "#984ea3", "#ff7f00", "#e41a1c", "#377eb8"]
            lateral_values = [
                metric_row["all_lateral_rmse_m"],
                metric_row["all_lateral_mean_abs_m"],
                metric_row["all_lateral_p95_abs_m"],
                metric_row["all_lateral_max_abs_m"],
                metric_row["all_lateral_std_m"],
            ]
            heading_values = [
                metric_row["all_heading_rmse_deg"],
                metric_row["all_heading_mean_abs_deg"],
                metric_row["all_heading_p95_abs_deg"],
                metric_row["all_heading_max_abs_deg"],
                metric_row["all_heading_std_deg"],
            ]
            axes[1, 0].bar(names, lateral_values, color=colors)
            axes[1, 0].set_ylabel("Meters")
            axes[1, 0].set_title("Full-run lateral error metrics")
            axes[1, 0].grid(True, axis="y", alpha=0.35)
            for index, value in enumerate(lateral_values):
                axes[1, 0].text(index, value, f"{value:.3f}", ha="center", va="bottom")

            axes[1, 1].bar(names, heading_values, color=colors)
            axes[1, 1].set_ylabel("Degrees")
            axes[1, 1].set_title("Full-run heading error metrics")
            axes[1, 1].grid(True, axis="y", alpha=0.35)
            for index, value in enumerate(heading_values):
                axes[1, 1].text(index, value, f"{value:.2f}", ha="center", va="bottom")

        fig.suptitle("Nearest-Path Tracking Metrics", y=0.98)
        fig.tight_layout()
        fig.savefig(self.output_dir / "01_tracking_error_metrics.png", dpi=150)
        fig.savefig(self.output_dir / "01_time_sync_diagnostics.png", dpi=150)
        plt.close(fig)

        target_x = trajectory.x + records["map_offset_x"].iloc[0]
        target_y = trajectory.y + records["map_offset_y"].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(target_x[trajectory.is_forward], target_y[trajectory.is_forward], s=12, c="cyan", label="Target Fwd")
        ax.scatter(target_x[~trajectory.is_forward], target_y[~trajectory.is_forward], s=12, c="orange", label="Target Rev")
        ax.plot(records["x"], records["y"], "r-", linewidth=1.8, label="Actual Path")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(self.output_dir / "02_spatial_trajectory.png", dpi=150)
        plt.close(fig)

    def write_readme(self, metrics: pd.DataFrame) -> None:
        """写出指标定义和本轮关键结果，便于独立复核输出文件。"""
        metric_lines = []
        if not metrics.empty:
            row = metrics.iloc[0]
            metric_lines = [
                "",
                "Overall nearest-path metrics:",
                f"- Total samples: {int(row['total_sample_count'])}",
                f"- Post-endpoint samples: {int(row['post_endpoint_sample_count'])}",
                f"- Lateral RMSE: {row['all_lateral_rmse_m']:.4f} m",
                f"- Lateral MAE: {row['all_lateral_mean_abs_m']:.4f} m",
                f"- Lateral P95: {row['all_lateral_p95_abs_m']:.4f} m",
                f"- Maximum lateral error: {row['all_lateral_max_abs_m']:.4f} m",
                f"- Lateral standard deviation: {row['all_lateral_std_m']:.4f} m",
                f"- Heading RMSE: {row['all_heading_rmse_deg']:.4f} deg",
                f"- Heading MAE: {row['all_heading_mean_abs_deg']:.4f} deg",
                f"- Heading P95: {row['all_heading_p95_abs_deg']:.4f} deg",
                f"- Maximum heading error: {row['all_heading_max_abs_deg']:.4f} deg",
                f"- Heading standard deviation: {row['all_heading_std_deg']:.4f} deg",
            ]
        lines = [
            "Trajectory tracking diagnostic notes",
            "This report follows the nearest-path evaluation logic used by result_time.py.",
            "",
            "1. pos_err_m is the longitudinal timing error projected onto the local trajectory tangent.",
            "2. cmd_abs_kmh is the catch-up speed command generated from the timing error.",
            "3. path_cte_m / cte_m is the signed lateral error to the nearest reference path segment.",
            "4. time_sync_cte_m is retained as a diagnostic value for same-time reference-point deviation.",
            "5. path_heading_error_deg is the difference between vehicle motion direction and nearest path heading.",
            "6. metric_valid marks samples after excluding start, end, low-speed and shifting transients.",
        ] + metric_lines
        (self.output_dir / "README_diagnostics.txt").write_text("\n".join(lines), encoding="utf-8")
