"""轨迹处理智能体（TPS）的数据清洗、格式统一与运动方向识别。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ALIASES, STANDARD_TRAJECTORY_COLUMNS


@dataclass
class TrajectoryPreprocessingConfig:
    """位置平滑、状态滤波和前进/倒车识别阈值。"""
    smoothing_window: int = 5
    state_filter_window: int = 7
    motion_threshold_m: float = 0.25
    forward_angle_threshold_deg: float = 100.0
    default_speed_kmh: float = 0.0
    y_axis_flip_for_carla: bool = False


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    """按别名表查找某个标准字段对应的原始 CSV 列。"""
    for candidate in ALIASES[canonical]:
        if candidate in df.columns:
            return candidate
    return None


def _angle_diff_deg(a: float, b: float) -> float:
    """计算归一化到正负 180 度内的航向差。"""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


def _rolling_majority(values: list[int], window: int) -> list[int]:
    """用中心滑动窗口多数表决抑制孤立的换向标签。"""
    series = pd.Series(values)
    filtered = series.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: 1 if sum(x) >= len(x) / 2 else 0
    )
    return filtered.astype(int).tolist()


class TrajectoryPreprocessingSubagent:
    """准备拖拉机轨迹并识别前进、倒车作业区段。"""

    def __init__(self, config: TrajectoryPreprocessingConfig | None = None):
        """保存轨迹平滑和方向判定配置。"""
        self.config = config or TrajectoryPreprocessingConfig()

    def process(self, input_file: str | Path, output_file: str | Path) -> Path:
        """依次完成列名规范、方向标签生成和标准 CSV 输出。"""
        df = pd.read_csv(input_file)
        normalized = self.normalize_columns(df)
        labeled = self.ensure_forward_reverse_labels(normalized)
        optimized = self.optimize_standard_format(labeled)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        optimized.to_csv(output_file, index=False, encoding="utf-8-sig")
        return output_file

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射不同数据源表头，并为非核心缺失字段提供可追踪默认值。"""
        out = pd.DataFrame()
        for canonical in STANDARD_TRAJECTORY_COLUMNS:
            source = _find_column(df, canonical)
            if source is not None:
                out[canonical] = df[source]

        missing_core = [c for c in ("longitude", "latitude") if c not in out.columns]
        if missing_core:
            raise ValueError(f"Trajectory CSV lacks required coordinates: {missing_core}")

        if "time" not in out.columns:
            out["time"] = pd.date_range("1970-01-01", periods=len(out), freq="1s").astype(str)
        if "speed" not in out.columns:
            out["speed"] = self.config.default_speed_kmh
        if "direction" not in out.columns:
            out["direction"] = np.nan
        return out

    def ensure_forward_reverse_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """保留已有有效标签，否则根据位移方向与估计车头方向自动识别。

        车辆发生有效位移时，用相邻平滑位置得到运动方向；若它与当前估计车头
        方向差小于阈值则判为前进，否则判为倒车并反向更新车头方向。
        """
        out = df.copy()
        if "forward_reverse_label" in out.columns and out["forward_reverse_label"].notna().any():
            out["forward_reverse_label"] = pd.to_numeric(out["forward_reverse_label"], errors="coerce")
            out["forward_reverse_label"] = out["forward_reverse_label"].ffill().fillna(1).astype(int)
            return out

        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        valid_idx = out.dropna(subset=["longitude", "latitude"]).index
        valid = out.loc[valid_idx].copy()
        if valid.empty:
            raise ValueError("No valid longitude/latitude rows found.")

        lon0 = valid["longitude"].iloc[0]
        lat0 = valid["latitude"].iloc[0]
        radius = 6378137.0
        valid["local_x_m"] = radius * np.radians(valid["longitude"] - lon0) * math.cos(math.radians(lat0))
        valid["local_y_m"] = radius * np.radians(valid["latitude"] - lat0)

        window = self.config.smoothing_window
        valid["x_smooth"] = valid["local_x_m"].rolling(window=window, center=True, min_periods=1).mean()
        valid["y_smooth"] = valid["local_y_m"].rolling(window=window, center=True, min_periods=1).mean()
        valid["dx"] = valid["x_smooth"].diff()
        valid["dy"] = valid["y_smooth"].diff()
        valid["dist"] = np.hypot(valid["dx"], valid["dy"])

        states: list[int] = []
        current_state = 1
        tractor_heading = None

        # 静止或位移过小时沿用上一状态，避免 GNSS 抖动被识别为短暂换向。
        for _, row in valid.iterrows():
            dx = row["dx"]
            dy = row["dy"]
            dist = row["dist"]
            if pd.isna(dx) or pd.isna(dy) or dist < self.config.motion_threshold_m:
                states.append(current_state)
                continue

            theta = math.degrees(math.atan2(dy, dx)) % 360.0
            if tractor_heading is None:
                tractor_heading = theta
                states.append(current_state)
                continue

            diff = _angle_diff_deg(theta, tractor_heading)
            if abs(diff) < self.config.forward_angle_threshold_deg:
                current_state = 1
                tractor_heading = theta
            else:
                current_state = 0
                tractor_heading = (theta + 180.0) % 360.0
            states.append(current_state)

        states_final = _rolling_majority(states, self.config.state_filter_window)
        if states_final.count(0) > states_final.count(1):
            states_final = [1 if s == 0 else 0 for s in states_final]

        out["forward_reverse_label"] = np.nan
        out.loc[valid_idx, "forward_reverse_label"] = states_final
        out["forward_reverse_label"] = out["forward_reverse_label"].ffill().fillna(1).astype(int)
        return out

    def optimize_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数值、时间和航向字段，并按统一列顺序返回。"""
        out = df.copy()
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        out = out.dropna(subset=["longitude", "latitude"]).copy()

        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        if out["time"].isna().any():
            out["time"] = pd.date_range("1970-01-01", periods=len(out), freq="1s")
        out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        out["speed"] = pd.to_numeric(out["speed"], errors="coerce").fillna(self.config.default_speed_kmh)

        direction = pd.to_numeric(out["direction"], errors="coerce")
        missing_direction = direction.isna()
        if missing_direction.any():
            lon = out["longitude"].astype(float).values
            lat = out["latitude"].astype(float).values
            dx = np.diff(lon, prepend=lon[0])
            dy = np.diff(lat, prepend=lat[0])
            inferred = np.degrees(np.arctan2(dy, dx)) % 360.0
            direction = direction.fillna(pd.Series(inferred, index=out.index))
        out["direction"] = direction

        out["forward_reverse_label"] = pd.to_numeric(out["forward_reverse_label"], errors="coerce")
        out["forward_reverse_label"] = out["forward_reverse_label"].ffill().fillna(1).astype(int)
        return out[STANDARD_TRAJECTORY_COLUMNS]


def process_tractor_trajectory(input_file_path: str | Path, output_file_path: str | Path) -> Path:
    """使用默认 TPS 配置处理单个轨迹文件的便捷函数。"""
    return TrajectoryPreprocessingSubagent().process(input_file_path, output_file_path)
