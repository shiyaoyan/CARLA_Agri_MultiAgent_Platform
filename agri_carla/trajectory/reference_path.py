"""CIA 和 PES 共用的标准参考轨迹表示及最近路径误差计算。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from agri_carla.utils.io import read_csv_fallback


class ReferenceTrajectory:
    """CARLA 轨迹跟踪实验使用的米制参考路径。"""

    def __init__(self, csv_path: str | Path, use_gps_speed_for_reverse: bool = False, flip_y: bool = True):
        """读取标准 CSV，并计算相对时间、局部坐标、速度和方向分段。"""
        self.csv_path = Path(csv_path)
        self.csv_name = self.csv_path.name
        self.df = read_csv_fallback(self.csv_path)

        required = {"longitude", "latitude", "time"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV lacks required columns: {sorted(missing)}")

        self.df["time"] = pd.to_datetime(self.df["time"])
        self.t = (self.df["time"] - self.df["time"].iloc[0]).dt.total_seconds().values

        # 将经纬度换算为以首点为原点的局部米制坐标；按地图坐标约定可翻转 Y 轴。
        earth_radius = 6378137.0
        lat = np.radians(self.df["latitude"].astype(float).values)
        lon = np.radians(self.df["longitude"].astype(float).values)
        self.x = earth_radius * lon * np.cos(lat[0])
        self.y = earth_radius * lat
        self.x -= self.x[0]
        self.y -= self.y[0]
        if flip_y:
            self.y = -self.y

        if "speed" in self.df.columns:
            self.speed_ms = np.maximum(self.df["speed"].astype(float).values / 3.6, 0.0)
        else:
            self.speed_ms = np.zeros(len(self.x))

        dx = np.diff(self.x, prepend=self.x[0])
        dy = np.diff(self.y, prepend=self.y[0])
        self.gps_speed_ms = np.hypot(dx, dy)

        if "forward_reverse_label" in self.df.columns:
            self.is_forward = self.df["forward_reverse_label"].astype(int).values == 1
        elif "state" in self.df.columns:
            self.is_forward = self.df["state"].astype(int).values == 1
        else:
            self.is_forward = np.ones(len(self.x), dtype=bool)

        if use_gps_speed_for_reverse and "speed" in self.df.columns:
            reverse_mask = ~self.is_forward
            self.speed_ms[reverse_mask] = self.gps_speed_ms[reverse_mask]

        self.num_points = len(self.x)
        self.segment_id = self._build_segment_ids()
        # 预计算各线段向量和长度，后续每个仿真周期可向量化求最近投影点。
        self.seg_x0 = self.x[:-1]
        self.seg_y0 = self.y[:-1]
        self.seg_dx = np.diff(self.x)
        self.seg_dy = np.diff(self.y)
        self.seg_len2 = self.seg_dx ** 2 + self.seg_dy ** 2
        self.seg_len = np.sqrt(self.seg_len2)

    def _build_segment_ids(self):
        """在每次前进/倒车状态改变处创建新的轨迹段编号。"""
        segment_id = np.zeros(self.num_points, dtype=int)
        sid = 0
        for i in range(1, self.num_points):
            if self.is_forward[i] != self.is_forward[i - 1]:
                sid += 1
            segment_id[i] = sid
        return segment_id

    def print_data_check(self) -> None:
        """打印轨迹时长以及前进、倒车点数和速度统计。"""
        print(f"\nTrajectory file: {self.csv_name}")
        print(f"Duration: {self.t[-1]:.2f} s")
        for direction, name in [(True, "forward"), (False, "reverse")]:
            mask = self.is_forward == direction
            if not np.any(mask):
                continue
            csv_kmh = self.speed_ms[mask] * 3.6
            print(
                f"  {name}: points={np.sum(mask)} | "
                f"speed mean/max={np.mean(csv_kmh):.2f}/{np.max(csv_kmh):.2f} km/h"
            )

    def to_dataframe(self, offset_x: float = 0.0, offset_y: float = 0.0) -> pd.DataFrame:
        """转换为叠加地图偏移后的分析数据表。"""
        return pd.DataFrame({
            "time_s": self.t,
            "x": self.x + offset_x,
            "y": self.y + offset_y,
            "speed_ms": self.speed_ms,
            "is_forward": self.is_forward,
            "segment_id": self.segment_id,
        })

    def nearest_path_error(self, x: float, y: float, map_offset_x: float = 0.0, map_offset_y: float = 0.0) -> dict[str, float | int]:
        """计算车辆到整条参考折线的最近投影、带符号横向误差和航向。

        该误差按最近线段而非最近离散点计算，可减少采样间距对评价结果的影响。
        """
        if self.num_points < 2:
            return {
                "path_cte_m": 0.0,
                "path_distance_m": 0.0,
                "path_heading_rad": 0.0,
                "nearest_path_idx": 0,
                "nearest_path_u": 0.0,
                "nearest_path_x": float(self.x[0] + map_offset_x),
                "nearest_path_y": float(self.y[0] + map_offset_y),
            }

        px = x - map_offset_x
        py = y - map_offset_y
        wx = px - self.seg_x0
        wy = py - self.seg_y0
        denom = np.where(self.seg_len2 > 1e-8, self.seg_len2, 1.0)
        u = np.clip((wx * self.seg_dx + wy * self.seg_dy) / denom, 0.0, 1.0)
        proj_x = self.seg_x0 + u * self.seg_dx
        proj_y = self.seg_y0 + u * self.seg_dy
        err_x = px - proj_x
        err_y = py - proj_y
        dist2 = err_x ** 2 + err_y ** 2
        idx = int(np.argmin(dist2))

        seg_len = self.seg_len[idx]
        if seg_len > 1e-5:
            signed_cte = (self.seg_dx[idx] * err_y[idx] - self.seg_dy[idx] * err_x[idx]) / seg_len
            heading = math.atan2(self.seg_dy[idx], self.seg_dx[idx])
        else:
            signed_cte = math.copysign(math.sqrt(dist2[idx]), err_y[idx])
            heading = 0.0

        return {
            "path_cte_m": float(signed_cte),
            "path_distance_m": float(math.sqrt(dist2[idx])),
            "path_heading_rad": float(heading),
            "nearest_path_idx": idx,
            "nearest_path_u": float(u[idx]),
            "nearest_path_x": float(proj_x[idx] + map_offset_x),
            "nearest_path_y": float(proj_y[idx] + map_offset_y),
        }
