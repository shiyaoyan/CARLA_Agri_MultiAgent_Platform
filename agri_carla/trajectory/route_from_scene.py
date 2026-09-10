"""从 SGA 道路场景请求生成可用于轨迹跟踪的标准路线 CSV。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from agri_carla.utils.io import read_yaml


def _interp_polyline(points: list[tuple[float, float]], spacing_px: float) -> list[tuple[float, float]]:
    """按给定像素间距对道路中心折线进行线性重采样。"""
    if len(points) < 2:
        return points
    sampled: list[tuple[float, float]] = []
    for start, end in zip(points[:-1], points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(length / max(spacing_px, 1e-6))))
        for i in range(steps):
            ratio = i / steps
            sampled.append((start[0] + dx * ratio, start[1] + dy * ratio))
    sampled.append(points[-1])
    deduped: list[tuple[float, float]] = []
    for point in sampled:
        if not deduped or math.hypot(point[0] - deduped[-1][0], point[1] - deduped[-1][1]) > 1e-6:
            deduped.append(point)
    return deduped


def _pixel_to_lonlat(point, anchor, meters_per_pixel: float, north_up: bool):
    """依据地理控制点和比例尺将像素坐标近似反算为经纬度。"""
    radius = 6378137.0
    px, py = point
    dx_m = (px - float(anchor["pixel_x"])) * meters_per_pixel
    dy_img_m = (py - float(anchor["pixel_y"])) * meters_per_pixel
    dy_m = -dy_img_m if north_up else dy_img_m
    lat0 = float(anchor["latitude"])
    lon0 = float(anchor["longitude"])
    lat = lat0 + math.degrees(dy_m / radius)
    lon = lon0 + math.degrees(dx_m / (radius * math.cos(math.radians(lat0))))
    return lon, lat


def generate_route_csv_from_scene_request(
    request_yaml: str | Path,
    output_csv: str | Path,
    sample_spacing_m: float = 1.0,
    speed_kmh: float = 4.0,
    start_time: str = "2026-01-01 00:00:00",
) -> Path:
    """按指定道路顺序连接中心线并输出标准前进路线。

    ``route_road_ids`` 可显式控制经过道路的顺序；未提供时按请求中的道路顺序
    连接。该函数生成的是测试路线，不替代复杂路网的最短路径规划。
    """
    request = read_yaml(request_yaml)
    meters_per_pixel = float(request["meters_per_pixel"])
    spacing_px = sample_spacing_m / meters_per_pixel
    anchor = request["geo_anchor"]
    north_up = bool(request.get("north_up", True))
    base_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")

    rows = []
    sequence: list[tuple[float, float]] = []
    route_road_ids = request.get("route_road_ids")
    roads = request["roads"]
    if route_road_ids:
        road_by_id = {road["road_id"]: road for road in roads}
        roads = [road_by_id[road_id] for road_id in route_road_ids]

    # 相邻道路共享端点时移除重复采样点，避免生成零长度轨迹段。
    for road in roads:
        polyline = [tuple(point) for point in road["pixel_polyline"]]
        sampled = _interp_polyline(polyline, spacing_px)
        if sequence and sampled and math.hypot(sequence[-1][0] - sampled[0][0], sequence[-1][1] - sampled[0][1]) < 1e-6:
            sampled = sampled[1:]
        sequence.extend(sampled)

    if len(sequence) < 2:
        raise ValueError("The road-scene request does not contain enough polyline points to generate a route.")

    for i, point in enumerate(sequence):
        lon, lat = _pixel_to_lonlat(point, anchor, meters_per_pixel, north_up)
        if i < len(sequence) - 1:
            nxt = sequence[i + 1]
            dx = nxt[0] - point[0]
            dy = nxt[1] - point[1]
        else:
            prev = sequence[i - 1]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
        # Direction is expressed as a planar heading angle in degrees.
        direction = float(np.degrees(np.arctan2(-dy if north_up else dy, dx)) % 360.0)
        rows.append({
            "longitude": lon,
            "latitude": lat,
            "time": (base_time + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S"),
            "speed": speed_kmh,
            "direction": direction,
            "forward_reverse_label": 1,
        })

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False, encoding="utf-8-sig")
    return output_csv
