#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Standalone paper Section 3.2 trajectory-tracking experiment.

Fill TRAJECTORY_CSV below, open the CARLA map, then run this file.
Requires Python >= 3.8, matching CARLA Python API, numpy, pandas and matplotlib.
No CrewAI, LLM API, navigation agents, external JSON or project installation.
Project TPS, controller, physics, recording, metrics and plots are embedded.
Fixed controller parameters only; no Bayesian search is triggered.
Relative paths are resolved next to this script.
"""
import argparse
import copy
from dataclasses import asdict
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys

# MIT License
# 
# Copyright (c) 2026 shiyaoyan
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Fill this with your measured CSV path or filename.
TRAJECTORY_CSV = r"FILL_IN_TRAJECTORY.csv"
# True matches the project TPS preprocessing workflow. False accepts a standard CSV directly.
PREPROCESS_INPUT = True
# Empty: automatically create <output_dir>/processed_trajectory/<name>_standard.csv.
PROCESSED_CSV = r""

# Same effective defaults as TrackingExperimentConfig and ControllerInterfaceAgent.
# carla_map="" uses the current world. A nonempty map name invokes load_world, as in the project.
# Speeds in the input CSV are km/h; offsets/spawn height are CARLA world metres.
EXPERIMENT_CONFIG = {'host': 'localhost',
 'port': 2000,
 'timeout': 60.0,
 'carla_map': '',
 'vehicle_bp': 'vehicle.tractor.tractor',
 'vehicle_type': 'tractor',
 'dt': 0.05,
 'map_offset_x': 50.0,
 'map_offset_y': -10.0,
 'spawn_z': 2.0,
 'debug_life_time': 1200.0,
 'output_dir': 'results/trajectory_tracking',
 'max_extra_time_s': 15.0,
 'finish_speed_ms': 0.35,
 'finish_position_error_m': 1.0,
 'apply_agri_physics': True,
 'trajectory_flip_y': True,
 'metric_min_speed_ms': 0.2,
 'metric_skip_start_s': 5.0,
 'metric_skip_end_s': 5.0,
 'controller': {'dt': 0.05,
                'k_pos': 0.8,
                'max_speed_kmh': 8.0,
                'pid': {'kp': 0.95,
                        'ki': 0.22,
                        'kd': 0.03,
                        'integral_limit': 4.0,
                        'min_throttle': 0.28,
                        'max_ff_speed_ms': 2.2,
                        'ff_limit': 0.58,
                        'brake_deadband_ms': 0.35,
                        'brake_gain': 0.9},
                'pure_pursuit': {'wheelbase': 2.5,
                                 'max_steer_angle_deg': 48.0,
                                 'fwd_lookahead_min': 1.0,
                                 'fwd_lookahead_gain': 0.2,
                                 'rev_lookahead_min': 0.5,
                                 'rev_lookahead_gain': 0.1},
                'gear': {'shift_speed_ms': 0.18,
                         'shift_neutral_s': 0.6,
                         'forward_gears': (1, 2, 3),
                         'reverse_manual_gear': -1,
                         'gear_probe_interval_s': 6.0}},
 'controller_optimization': None,
 'vegetation_request': None}

SOURCE_SHA256 = {'controllers.gear_state_machine': 'e0ac4f6c6dc77be98b1651a27f84b10e7075ffc91e9a8f1058afec9bc3a009a7',
 'controllers.interface_agent': '23b5bd362ed28fdfc29c09ec984ff5488b050e38aedb0e5703f49be22d5a0dbb',
 'controllers.pid': 'bb14f79f4f4d10de13278f3b77726c8cfc8a6b2cd4efc65af99077e08f8bf905',
 'controllers.pure_pursuit': 'c38680099de6faa7a97f0e54b242744b7f97572c4a36d6d42a58708e0ebeb112',
 'evaluation.metrics': '0123b064f606158036eddee0202e307e0e92c49897f6cc4352a6d3e62b182ec4',
 'evaluation.performance_evaluation_subagent': '822ee9c396ee1cdb0bd9235f18804d76a01664cfef1551fd08730ecd4135e837',
 'machinery.physics': '99ad43ee84266972014ed2fed017c0c48af86dc1c04e25fd43fec6eb5b057b04',
 'simulation.tracking_experiment': 'd6d9c6fe607b39db9eeb88fb1eafcd6bd35d73802bc20e83b7786fe528fc8ae2',
 'trajectory.preprocessing_subagent': '7749f0367f65545fe07728001d0c65a2cde7f10740df90d87ef4ca58cb92f076',
 'trajectory.reference_path': 'a85da46aff787d5a83d1ca03662b16d76957bbdcf7e6e404808a2c765bb831d1',
 'trajectory.schema': 'c2b946d3e9f2bdc7d18aec942960789e87a80514c5b028608bd2d180b9f5fefc',
 'utils.carla_import': '2d2a1086a9d6bf763b2cfcf8f28ae29faa9222e3317280a33686ca4b78db970c',
 'utils.geometry': '1e833632725ad8c8bc48ddcb545ad02438d8c71b888e710bedd643f7fc557227',
 'utils.io': 'e733357d313eaddb4cf56fa932838597fcce5ba4809cac3f608e0dc91a49d594'}

PACKAGE_NAMES = ['_tracking_embedded', '_tracking_embedded.controllers', '_tracking_embedded.evaluation', '_tracking_embedded.machinery', '_tracking_embedded.simulation', '_tracking_embedded.trajectory', '_tracking_embedded.utils']
MODULE_SOURCES = {}

# Bundled source: _tracking_embedded.utils.io
MODULE_SOURCES['_tracking_embedded.utils.io'] = '''"""项目配置、轨迹数据和结果文件的统一读写工具。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回标准化后的 ``Path`` 对象。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_csv_fallback(path: str | Path):
    """读取可能采用 UTF-8 或 GBK 编码的农业轨迹 CSV。

    实车采集软件导出的 CSV 编码并不统一，因此先尝试兼容 Excel 的
    ``utf-8-sig``，仅在解码失败时回退到常见中文编码 GBK。
    """
    import pandas as pd
    try:
        return pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='gbk')

def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    """以可读格式写入 JSON，并保留中文字符。"""
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return path

def read_yaml(path: str | Path) -> dict[str, Any]:
    """读取 JSON 或 YAML 配置，并统一返回字典。

    JSON 不依赖额外软件包；只有读取 YAML 时才延迟导入 PyYAML，使仅使用
    JSON 配置的 CARLA 计算机不必安装不需要的依赖。
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        return json.loads(text)
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('PyYAML is required to read .yaml files. Install project requirements or provide the same request as a .json file.') from exc
    data = yaml.safe_load(text)
    return data or {}

def write_yaml(path: str | Path, data: dict[str, Any]) -> Path:
    """根据扩展名写入 JSON 或 YAML 格式的结构化数据。"""
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == '.json':
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8')
        return path
    import yaml
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path
'''

# Bundled source: _tracking_embedded.utils.geometry
MODULE_SOURCES['_tracking_embedded.utils.geometry'] = '''"""轨迹处理与车辆控制共用的二维几何计算。"""
import math

def wrap_pi(angle: float) -> float:
    """将弧度角归一化到 ``[-pi, pi)``，避免航向跨越正负 180 度时突变。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def distance_xy(a, b) -> float:
    """计算具有 ``x``、``y`` 字段的 CARLA 风格对象之间的平面距离。"""
    return math.hypot(a.x - b.x, a.y - b.y)

def local_xy_from_lonlat(longitude, latitude):
    """将经纬度序列转换为以首个采样点为原点的局部米制坐标。

    该近似适合本文尺度较小的田间轨迹，可避免直接用角度单位进行路径跟踪；
    大范围或跨投影带数据应改用正式地图投影。
    """
    import numpy as np
    earth_radius = 6378137.0
    lon = np.radians(longitude.astype(float))
    lat = np.radians(latitude.astype(float))
    x = earth_radius * lon * np.cos(lat[0])
    y = earth_radius * lat
    return (x - x[0], y - y[0])
'''

# Bundled source: _tracking_embedded.utils.carla_import
MODULE_SOURCES['_tracking_embedded.utils.carla_import'] = '''"""CARLA Python API 的可选导入工具。

项目中的场景生成、轨迹预处理和结果分析可以在没有安装 CARLA 的计算机上运行。
只有真正执行仿真实验时才调用本模块，从而避免包导入阶段因缺少 CARLA egg 而失败。
"""

def import_carla():
    """导入 CARLA API，并将底层导入错误转换为更明确的运行提示。"""
    try:
        import carla
    except ImportError as exc:
        raise RuntimeError('CARLA Python API is not available. Install CARLA or add the CARLA egg path before running simulation experiments.') from exc
    return carla
'''

# Bundled source: _tracking_embedded.trajectory.schema
MODULE_SOURCES['_tracking_embedded.trajectory.schema'] = '''"""平台统一轨迹字段及常见实车数据列名别名。"""
STANDARD_TRAJECTORY_COLUMNS = ['longitude', 'latitude', 'time', 'speed', 'direction', 'forward_reverse_label']
ALIASES = {'longitude': ['longitude', 'Longitude (°)', 'lon', 'Lon', '经度'], 'latitude': ['latitude', 'Latitude (°)', 'lat', 'Lat', '纬度'], 'time': ['time', 'Time', 'timestamp', 'Timestamp', '时间'], 'speed': ['speed', 'Speed', 'speed_kmh', '速度'], 'direction': ['direction', 'Direction', 'heading', 'Heading', '航向'], 'forward_reverse_label': ['forward_reverse_label', 'State', 'state', 'direction_label']}
'''

# Bundled source: _tracking_embedded.trajectory.preprocessing_subagent
MODULE_SOURCES['_tracking_embedded.trajectory.preprocessing_subagent'] = '''"""轨迹处理智能体（TPS）的数据清洗、格式统一与运动方向识别。"""
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
    filtered = series.rolling(window=window, center=True, min_periods=1).apply(lambda x: 1 if sum(x) >= len(x) / 2 else 0)
    return filtered.astype(int).tolist()

class TrajectoryPreprocessingSubagent:
    """准备拖拉机轨迹并识别前进、倒车作业区段。"""

    def __init__(self, config: TrajectoryPreprocessingConfig | None=None):
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
        optimized.to_csv(output_file, index=False, encoding='utf-8-sig')
        return output_file

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射不同数据源表头，并为非核心缺失字段提供可追踪默认值。"""
        out = pd.DataFrame()
        for canonical in STANDARD_TRAJECTORY_COLUMNS:
            source = _find_column(df, canonical)
            if source is not None:
                out[canonical] = df[source]
        missing_core = [c for c in ('longitude', 'latitude') if c not in out.columns]
        if missing_core:
            raise ValueError(f'Trajectory CSV lacks required coordinates: {missing_core}')
        if 'time' not in out.columns:
            out['time'] = pd.date_range('1970-01-01', periods=len(out), freq='1s').astype(str)
        if 'speed' not in out.columns:
            out['speed'] = self.config.default_speed_kmh
        if 'direction' not in out.columns:
            out['direction'] = np.nan
        return out

    def ensure_forward_reverse_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """保留已有有效标签，否则根据位移方向与估计车头方向自动识别。

        车辆发生有效位移时，用相邻平滑位置得到运动方向；若它与当前估计车头
        方向差小于阈值则判为前进，否则判为倒车并反向更新车头方向。
        """
        out = df.copy()
        if 'forward_reverse_label' in out.columns and out['forward_reverse_label'].notna().any():
            out['forward_reverse_label'] = pd.to_numeric(out['forward_reverse_label'], errors='coerce')
            out['forward_reverse_label'] = out['forward_reverse_label'].ffill().fillna(1).astype(int)
            return out
        out['longitude'] = pd.to_numeric(out['longitude'], errors='coerce')
        out['latitude'] = pd.to_numeric(out['latitude'], errors='coerce')
        valid_idx = out.dropna(subset=['longitude', 'latitude']).index
        valid = out.loc[valid_idx].copy()
        if valid.empty:
            raise ValueError('No valid longitude/latitude rows found.')
        lon0 = valid['longitude'].iloc[0]
        lat0 = valid['latitude'].iloc[0]
        radius = 6378137.0
        valid['local_x_m'] = radius * np.radians(valid['longitude'] - lon0) * math.cos(math.radians(lat0))
        valid['local_y_m'] = radius * np.radians(valid['latitude'] - lat0)
        window = self.config.smoothing_window
        valid['x_smooth'] = valid['local_x_m'].rolling(window=window, center=True, min_periods=1).mean()
        valid['y_smooth'] = valid['local_y_m'].rolling(window=window, center=True, min_periods=1).mean()
        valid['dx'] = valid['x_smooth'].diff()
        valid['dy'] = valid['y_smooth'].diff()
        valid['dist'] = np.hypot(valid['dx'], valid['dy'])
        states: list[int] = []
        current_state = 1
        tractor_heading = None
        for _, row in valid.iterrows():
            dx = row['dx']
            dy = row['dy']
            dist = row['dist']
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
        out['forward_reverse_label'] = np.nan
        out.loc[valid_idx, 'forward_reverse_label'] = states_final
        out['forward_reverse_label'] = out['forward_reverse_label'].ffill().fillna(1).astype(int)
        return out

    def optimize_standard_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数值、时间和航向字段，并按统一列顺序返回。"""
        out = df.copy()
        out['longitude'] = pd.to_numeric(out['longitude'], errors='coerce')
        out['latitude'] = pd.to_numeric(out['latitude'], errors='coerce')
        out = out.dropna(subset=['longitude', 'latitude']).copy()
        out['time'] = pd.to_datetime(out['time'], errors='coerce')
        if out['time'].isna().any():
            out['time'] = pd.date_range('1970-01-01', periods=len(out), freq='1s')
        out['time'] = out['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        out['speed'] = pd.to_numeric(out['speed'], errors='coerce').fillna(self.config.default_speed_kmh)
        direction = pd.to_numeric(out['direction'], errors='coerce')
        missing_direction = direction.isna()
        if missing_direction.any():
            lon = out['longitude'].astype(float).values
            lat = out['latitude'].astype(float).values
            dx = np.diff(lon, prepend=lon[0])
            dy = np.diff(lat, prepend=lat[0])
            inferred = np.degrees(np.arctan2(dy, dx)) % 360.0
            direction = direction.fillna(pd.Series(inferred, index=out.index))
        out['direction'] = direction
        out['forward_reverse_label'] = pd.to_numeric(out['forward_reverse_label'], errors='coerce')
        out['forward_reverse_label'] = out['forward_reverse_label'].ffill().fillna(1).astype(int)
        return out[STANDARD_TRAJECTORY_COLUMNS]

def process_tractor_trajectory(input_file_path: str | Path, output_file_path: str | Path) -> Path:
    """使用默认 TPS 配置处理单个轨迹文件的便捷函数。"""
    return TrajectoryPreprocessingSubagent().process(input_file_path, output_file_path)
'''

# Bundled source: _tracking_embedded.trajectory.reference_path
MODULE_SOURCES['_tracking_embedded.trajectory.reference_path'] = '''"""CIA 和 PES 共用的标准参考轨迹表示及最近路径误差计算。"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from _tracking_embedded.utils.io import read_csv_fallback

class ReferenceTrajectory:
    """CARLA 轨迹跟踪实验使用的米制参考路径。"""

    def __init__(self, csv_path: str | Path, use_gps_speed_for_reverse: bool=False, flip_y: bool=True):
        """读取标准 CSV，并计算相对时间、局部坐标、速度和方向分段。"""
        self.csv_path = Path(csv_path)
        self.csv_name = self.csv_path.name
        self.df = read_csv_fallback(self.csv_path)
        required = {'longitude', 'latitude', 'time'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f'CSV lacks required columns: {sorted(missing)}')
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.t = (self.df['time'] - self.df['time'].iloc[0]).dt.total_seconds().values
        earth_radius = 6378137.0
        lat = np.radians(self.df['latitude'].astype(float).values)
        lon = np.radians(self.df['longitude'].astype(float).values)
        self.x = earth_radius * lon * np.cos(lat[0])
        self.y = earth_radius * lat
        self.x -= self.x[0]
        self.y -= self.y[0]
        if flip_y:
            self.y = -self.y
        if 'speed' in self.df.columns:
            self.speed_ms = np.maximum(self.df['speed'].astype(float).values / 3.6, 0.0)
        else:
            self.speed_ms = np.zeros(len(self.x))
        dx = np.diff(self.x, prepend=self.x[0])
        dy = np.diff(self.y, prepend=self.y[0])
        self.gps_speed_ms = np.hypot(dx, dy)
        if 'forward_reverse_label' in self.df.columns:
            self.is_forward = self.df['forward_reverse_label'].astype(int).values == 1
        elif 'state' in self.df.columns:
            self.is_forward = self.df['state'].astype(int).values == 1
        else:
            self.is_forward = np.ones(len(self.x), dtype=bool)
        if use_gps_speed_for_reverse and 'speed' in self.df.columns:
            reverse_mask = ~self.is_forward
            self.speed_ms[reverse_mask] = self.gps_speed_ms[reverse_mask]
        self.num_points = len(self.x)
        self.segment_id = self._build_segment_ids()
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
        print(f'\\nTrajectory file: {self.csv_name}')
        print(f'Duration: {self.t[-1]:.2f} s')
        for direction, name in [(True, 'forward'), (False, 'reverse')]:
            mask = self.is_forward == direction
            if not np.any(mask):
                continue
            csv_kmh = self.speed_ms[mask] * 3.6
            print(f'  {name}: points={np.sum(mask)} | speed mean/max={np.mean(csv_kmh):.2f}/{np.max(csv_kmh):.2f} km/h')

    def to_dataframe(self, offset_x: float=0.0, offset_y: float=0.0) -> pd.DataFrame:
        """转换为叠加地图偏移后的分析数据表。"""
        return pd.DataFrame({'time_s': self.t, 'x': self.x + offset_x, 'y': self.y + offset_y, 'speed_ms': self.speed_ms, 'is_forward': self.is_forward, 'segment_id': self.segment_id})

    def nearest_path_error(self, x: float, y: float, map_offset_x: float=0.0, map_offset_y: float=0.0) -> dict[str, float | int]:
        """计算车辆到整条参考折线的最近投影、带符号横向误差和航向。

        该误差按最近线段而非最近离散点计算，可减少采样间距对评价结果的影响。
        """
        if self.num_points < 2:
            return {'path_cte_m': 0.0, 'path_distance_m': 0.0, 'path_heading_rad': 0.0, 'nearest_path_idx': 0, 'nearest_path_u': 0.0, 'nearest_path_x': float(self.x[0] + map_offset_x), 'nearest_path_y': float(self.y[0] + map_offset_y)}
        px = x - map_offset_x
        py = y - map_offset_y
        wx = px - self.seg_x0
        wy = py - self.seg_y0
        denom = np.where(self.seg_len2 > 1e-08, self.seg_len2, 1.0)
        u = np.clip((wx * self.seg_dx + wy * self.seg_dy) / denom, 0.0, 1.0)
        proj_x = self.seg_x0 + u * self.seg_dx
        proj_y = self.seg_y0 + u * self.seg_dy
        err_x = px - proj_x
        err_y = py - proj_y
        dist2 = err_x ** 2 + err_y ** 2
        idx = int(np.argmin(dist2))
        seg_len = self.seg_len[idx]
        if seg_len > 1e-05:
            signed_cte = (self.seg_dx[idx] * err_y[idx] - self.seg_dy[idx] * err_x[idx]) / seg_len
            heading = math.atan2(self.seg_dy[idx], self.seg_dx[idx])
        else:
            signed_cte = math.copysign(math.sqrt(dist2[idx]), err_y[idx])
            heading = 0.0
        return {'path_cte_m': float(signed_cte), 'path_distance_m': float(math.sqrt(dist2[idx])), 'path_heading_rad': float(heading), 'nearest_path_idx': idx, 'nearest_path_u': float(u[idx]), 'nearest_path_x': float(proj_x[idx] + map_offset_x), 'nearest_path_y': float(proj_y[idx] + map_offset_y)}
'''

# Bundled source: _tracking_embedded.controllers.pid
MODULE_SOURCES['_tracking_embedded.controllers.pid'] = '''"""拖拉机纵向速度 PID 基线控制器。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class PIDConfig:
    """PID 系数、积分限幅、前馈油门和制动阈值配置。"""
    kp: float = 0.95
    ki: float = 0.22
    kd: float = 0.03
    integral_limit: float = 4.0
    min_throttle: float = 0.28
    max_ff_speed_ms: float = 2.2
    ff_limit: float = 0.58
    brake_deadband_ms: float = 0.35
    brake_gain: float = 0.9

class PIDController:
    """将目标速度与当前速度之差转换为油门和制动指令。"""

    def __init__(self, config: PIDConfig | None=None):
        """初始化参数及误差积分、上一时刻误差。"""
        self.config = config or PIDConfig()
        self.integral = 0.0
        self.last_error = 0.0

    def reset(self) -> None:
        """在换向或停车后清除积分和微分历史，防止旧误差延续。"""
        self.integral = 0.0
        self.last_error = 0.0

    def step(self, target_ms: float, actual_ms: float, dt: float) -> tuple[float, float, float]:
        """计算一个控制周期的油门、制动和原始 PID 输出。"""
        cfg = self.config
        if target_ms < 0.03:
            self.reset()
            return (0.0, 1.0 if actual_ms > 0.12 else 0.0, 0.0)
        error = target_ms - actual_ms
        self.integral = float(np.clip(self.integral + error * dt, -cfg.integral_limit, cfg.integral_limit))
        derivative = (error - self.last_error) / max(dt, 1e-06)
        self.last_error = error
        pid = cfg.kp * error + cfg.ki * self.integral + cfg.kd * derivative
        if error >= 0.0:
            feedforward = float(np.clip(target_ms / cfg.max_ff_speed_ms, 0.0, cfg.ff_limit))
            throttle = float(np.clip(cfg.min_throttle + feedforward + pid, 0.0, 1.0))
            return (throttle, 0.0, float(pid))
        if actual_ms > target_ms + cfg.brake_deadband_ms:
            brake = float(np.clip((actual_ms - target_ms - cfg.brake_deadband_ms) * cfg.brake_gain, 0.0, 1.0))
            return (0.0, brake, float(pid))
        return (0.0, 0.0, float(pid))
'''

# Bundled source: _tracking_embedded.controllers.pure_pursuit
MODULE_SOURCES['_tracking_embedded.controllers.pure_pursuit'] = '''"""区分前进和倒车参数的拖拉机纯追踪横向控制器。"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
from _tracking_embedded.utils.geometry import wrap_pi

@dataclass
class PurePursuitConfig:
    """轴距、最大转角以及前进/倒车前视距离参数。"""
    wheelbase: float = 2.5
    max_steer_angle_deg: float = 48.0
    fwd_lookahead_min: float = 1.0
    fwd_lookahead_gain: float = 0.2
    rev_lookahead_min: float = 0.5
    rev_lookahead_gain: float = 0.1

class PurePursuitController:
    """根据运动方向选择参数并计算归一化转向指令。"""

    def __init__(self, trajectory, offset_x: float=0.0, offset_y: float=0.0, config: PurePursuitConfig | None=None):
        """绑定标准轨迹、场景偏移和控制参数。"""
        self.trajectory = trajectory
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.config = config or PurePursuitConfig()
        self.index = 0

    def point(self, index: int) -> tuple[float, float]:
        """返回叠加场景坐标偏移后的参考路径点。"""
        return (self.trajectory.x[index] + self.offset_x, self.trajectory.y[index] + self.offset_y)

    def lookahead_point(self, x: float, y: float, actual_speed_ms: float, is_forward: bool) -> tuple[float, float, int, float]:
        """在当前同向轨迹段内寻找满足动态前视距离的目标点。"""
        cfg = self.config
        lookahead_min = cfg.fwd_lookahead_min if is_forward else cfg.rev_lookahead_min
        lookahead_gain = cfg.fwd_lookahead_gain if is_forward else cfg.rev_lookahead_gain
        lookahead = max(lookahead_min, lookahead_min + lookahead_gain * actual_speed_ms)
        lx, ly = self.point(self.trajectory.num_points - 1)
        li = self.trajectory.num_points - 1
        current_direction = bool(self.trajectory.is_forward[self.index])
        for i in range(self.index, self.trajectory.num_points):
            if bool(self.trajectory.is_forward[i]) != current_direction:
                lx, ly = self.point(i)
                li = i
                break
            tx, ty = self.point(i)
            if math.hypot(x - tx, y - ty) >= lookahead:
                lx, ly = (tx, ty)
                li = i
                break
        return (lx, ly, li, lookahead)

    def compute_steer(self, x: float, y: float, yaw_deg: float, actual_speed_ms: float, is_forward: bool) -> tuple[float, float, int, float]:
        """按纯追踪几何关系计算转向、航向误差和前视点信息。"""
        lx, ly, li, lookahead_dist = self.lookahead_point(x, y, actual_speed_ms, is_forward)
        target_heading = math.atan2(ly - y, lx - x)
        vehicle_heading = math.radians(yaw_deg)
        if not is_forward:
            vehicle_heading += math.pi
        alpha = wrap_pi(target_heading - vehicle_heading)
        cfg = self.config
        delta = math.atan2(2.0 * cfg.wheelbase * math.sin(alpha), max(lookahead_dist, 0.3))
        steer = delta / math.radians(cfg.max_steer_angle_deg)
        if not is_forward:
            steer = -steer
        return (float(np.clip(steer, -1.0, 1.0)), alpha, li, lookahead_dist)
'''

# Bundled source: _tracking_embedded.controllers.gear_state_machine
MODULE_SOURCES['_tracking_embedded.controllers.gear_state_machine'] = '''"""模拟拖拉机停车、空挡等待和前进/倒车切换的换挡状态机。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, TypedDict

class GearState(TypedDict):
    """一次换挡状态更新返回的稳定字段。"""
    mode: Literal['neutral', 'brake_to_shift', 'enter_neutral', 'drive']
    forward: bool
    reverse: bool
    gear: int
    event: str | None

@dataclass
class GearStateConfig:
    """允许换向速度、空挡持续时间和可探测前进挡位配置。"""
    shift_speed_ms: float = 0.18
    shift_neutral_s: float = 0.6
    forward_gears: tuple[int, ...] = (1, 2, 3)
    reverse_manual_gear: int = -1
    gear_probe_interval_s: float = 6.0

class GearStateMachine:
    """为农业路径中的频繁换向提供最小但明确的机械过程状态。"""

    def __init__(self, initial_forward: bool=True, config: GearStateConfig | None=None):
        """按首段运动方向初始化当前挡位和待切换状态。"""
        self.config = config or GearStateConfig()
        self.current_forward = initial_forward
        self.pending_forward: bool | None = None
        self.neutral_until = -1.0
        self.forward_gear_index = 0

    def reset(self, initial_forward: bool=True) -> None:
        """恢复初始方向并清除尚未完成的换挡过程。"""
        self.current_forward = initial_forward
        self.pending_forward = None
        self.neutral_until = -1.0
        self.forward_gear_index = 0

    def update(self, target_forward: bool, actual_speed_ms: float, sim_time: float) -> GearState:
        """根据目标方向、车速和仿真时间推进换挡状态。"""
        cfg = self.config
        neutral_active = sim_time < self.neutral_until
        event = None
        if self.pending_forward is not None and (not neutral_active):
            self.current_forward = self.pending_forward
            self.pending_forward = None
            self.forward_gear_index = 0
            event = 'PHASE_SHIFT'
        if neutral_active:
            return {'mode': 'neutral', 'forward': self.current_forward, 'reverse': False, 'gear': 0, 'event': 'SHIFT_NEUTRAL'}
        if target_forward != self.current_forward:
            if actual_speed_ms > cfg.shift_speed_ms:
                return {'mode': 'brake_to_shift', 'forward': self.current_forward, 'reverse': not self.current_forward, 'gear': self.current_gear(), 'event': 'WAIT_SHIFT_STOP'}
            self.pending_forward = target_forward
            self.neutral_until = sim_time + cfg.shift_neutral_s
            return {'mode': 'enter_neutral', 'forward': self.current_forward, 'reverse': False, 'gear': 0, 'event': 'SHIFT_NEUTRAL'}
        return {'mode': 'drive', 'forward': self.current_forward, 'reverse': not self.current_forward, 'gear': self.current_gear(), 'event': event}

    def current_gear(self) -> int:
        """返回当前前进挡或倒挡编号。"""
        if self.current_forward:
            gears = self.config.forward_gears or (1,)
            return gears[self.forward_gear_index % len(gears)]
        return self.config.reverse_manual_gear

    def next_forward_gear(self) -> int:
        """循环试探下一前进挡，用于高油门低车速诊断后的恢复尝试。"""
        gears = self.config.forward_gears or (1,)
        self.forward_gear_index = (self.forward_gear_index + 1) % len(gears)
        return self.current_gear()
'''

# Bundled source: _tracking_embedded.controllers.interface_agent
MODULE_SOURCES['_tracking_embedded.controllers.interface_agent'] = '''"""控制器接口智能体（CIA）的在线控制与可选参数自优化入口。"""
from __future__ import annotations
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from .gear_state_machine import GearStateConfig, GearStateMachine
from .pid import PIDConfig, PIDController
from .pure_pursuit import PurePursuitConfig, PurePursuitController
from _tracking_embedded.utils.geometry import wrap_pi

@dataclass
class ControllerInterfaceConfig:
    """控制周期、位置同步增益、限速及三个控制子模块配置。"""
    dt: float = 0.05
    k_pos: float = 0.8
    max_speed_kmh: float = 8.0
    pid: PIDConfig | None = None
    pure_pursuit: PurePursuitConfig | None = None
    gear: GearStateConfig | None = None

class ControllerInterfaceAgent:
    """组合纯追踪、速度 PID 和换挡状态机，并提供可选参数自优化。

    CIA 输出与 CARLA 无关的控制字典；EEA 读取该字典后才创建并施加
    ``carla.VehicleControl``，因此 CIA 是独立的并列智能体服务。
    """

    def __init__(self, trajectory, offset_x: float, offset_y: float, config: ControllerInterfaceConfig | None=None):
        """根据标准轨迹创建纵向、横向和换挡控制状态。"""
        self.config = config or ControllerInterfaceConfig()
        self.trajectory = trajectory
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.pid = PIDController(self.config.pid)
        self.tracker = PurePursuitController(trajectory, offset_x, offset_y, self.config.pure_pursuit)
        self.gear = GearStateMachine(bool(trajectory.is_forward[0]), self.config.gear)
        self.max_speed_ms = self.config.max_speed_kmh / 3.6
        self.high_throttle_low_speed_time = 0.0
        self.last_gear_probe_time = 0.0

    def reset(self) -> None:
        """在新实验开始前清除所有控制器历史状态。"""
        self.pid.reset()
        self.gear.reset(bool(self.trajectory.is_forward[0]))
        self.high_throttle_low_speed_time = 0.0
        self.last_gear_probe_time = 0.0

    @staticmethod
    def self_optimization_config(data: dict[str, Any] | None):
        """解析 CIA 内部的贝叶斯优化开关和参数搜索配置。"""
        from .controller_optimization import ControllerSelfOptimizationConfig
        return ControllerSelfOptimizationConfig.from_dict(data)

    @classmethod
    def run_self_optimization(cls, *, config, optimization_csvs: Sequence[str | Path], validation_csvs: Sequence[str | Path], experiment_config, run_trial: Callable[[Any, str | Path], dict[str, Any]], progress_callback=None) -> dict[str, Any]:
        """在优化集上搜索控制参数，并在独立验证集上评价锁定的参数。"""
        from .controller_optimization import ControllerSelfOptimizationSession
        return ControllerSelfOptimizationSession(config).run(optimization_csvs=optimization_csvs, validation_csvs=validation_csvs, experiment_config=experiment_config, run_trial=run_trial, progress_callback=progress_callback)

    @classmethod
    def prepare_experiment_configuration(cls, task_type: str, experiment: dict[str, Any] | None) -> dict[str, Any]:
        """校验控制相关配置并生成 EEA 可读取的控制器清单。

        该方法不连接 CARLA，也不运行控制循环。它使 CIA 能作为 CrewAI Flow
        中的独立同层智能体，在 EEA 启动仿真前检查控制方式、参数优化开关和
        关键控制配置。实际逐帧控制仍由 EEA 在仿真运行时调用 CIA 完成。
        """
        experiment = dict(experiment or {})
        config_data = dict(experiment.get('config', {}))
        if task_type == 'trajectory_tracking':
            from _tracking_embedded.simulation.tracking_experiment import TrackingExperimentConfig
            tracking_config = TrackingExperimentConfig.from_dict(config_data)
            controller = tracking_config.controller or ControllerInterfaceConfig(dt=tracking_config.dt)
            optimization = cls.self_optimization_config(tracking_config.controller_optimization)
            return {'task_type': task_type, 'validated': True, 'control_pipeline': ['direction_aware_pure_pursuit', 'pid_speed_control', 'gear_shifting_state_machine', 'carla_vehicle_control'], 'controller': dataclasses.asdict(controller), 'self_optimization': dataclasses.asdict(optimization)}
        if task_type == 'full_process_navigation':
            from _tracking_embedded.simulation.full_process_experiment import FullProcessExperimentConfig
            full_process_config = FullProcessExperimentConfig(**config_data)
            return {'task_type': task_type, 'validated': True, 'control_pipeline': ['road_navigation_control', 'field_and_headland_geometric_control', 'carla_vehicle_control'], 'road_speed_kmh': full_process_config.road_speed_kmh, 'field_line_speed_kmh': full_process_config.field_line_speed_kmh, 'apply_front_wheel_steering': full_process_config.apply_front_wheel_steering}
        return {'task_type': task_type, 'validated': True, 'control_pipeline': [], 'message': 'No simulation controller is required for this task type.'}

    def time_reference(self, sim_time: float):
        """按仿真时间插值得到参考位置、速度和相邻轨迹索引。"""
        import numpy as np
        traj = self.trajectory
        t_target = min(sim_time, traj.t[-1])
        ref_x = np.interp(t_target, traj.t, traj.x) + self.offset_x
        ref_y = np.interp(t_target, traj.t, traj.y) + self.offset_y
        ref_speed = np.interp(t_target, traj.t, traj.speed_ms)
        time_idx = int(np.searchsorted(traj.t, t_target))
        if time_idx == 0:
            time_idx = 1
        if time_idx >= traj.num_points:
            time_idx = traj.num_points - 1
        return (t_target, ref_x, ref_y, ref_speed, time_idx)

    def compute(self, vehicle_state: dict[str, float], sim_time: float) -> dict[str, Any]:
        """返回不依赖 CARLA 类型的控制和诊断字典。

        ``vehicle_state`` 必须包含位置、车头航向和实际速度。结果同时给出油门、
        制动、转向、挡位以及路径误差，供 EEA 执行和 PES 记录。
        """
        import numpy as np
        traj = self.trajectory
        t_target, ref_x, ref_y, ref_speed, time_idx = self.time_reference(sim_time)
        self.tracker.index = time_idx
        target_forward = bool(traj.is_forward[time_idx])
        actual_speed = vehicle_state['actual_speed_ms']
        x = vehicle_state['x']
        y = vehicle_state['y']
        dx = traj.x[time_idx] - traj.x[time_idx - 1]
        dy = traj.y[time_idx] - traj.y[time_idx - 1]
        length = math.hypot(dx, dy)
        if length > 1e-05:
            tx, ty = (dx / length, dy / length)
        else:
            yaw_rad = math.radians(vehicle_state['yaw_deg'])
            tx, ty = (math.cos(yaw_rad), math.sin(yaw_rad))
            if not target_forward:
                tx, ty = (-tx, -ty)
        pos_err = (ref_x - x) * tx + (ref_y - y) * ty
        ex = ref_x - x
        ey = ref_y - y
        geometric_dist = math.hypot(ex, ey)
        nx, ny = (-ty, tx)
        time_sync_cte = ex * nx + ey * ny
        ref_heading = math.atan2(ty, tx)
        vehicle_motion_heading = math.radians(vehicle_state['yaw_deg'])
        if not target_forward:
            vehicle_motion_heading += math.pi
        time_sync_heading_error = wrap_pi(vehicle_motion_heading - ref_heading)
        path_error = traj.nearest_path_error(x, y, self.offset_x, self.offset_y)
        path_heading_error = wrap_pi(vehicle_motion_heading - float(path_error['path_heading_rad']))
        target_speed_cmd = float(np.clip(ref_speed + self.config.k_pos * pos_err, 0.0, self.max_speed_ms))
        gear_state = self.gear.update(target_forward, actual_speed, sim_time)
        diagnostic = []
        event = gear_state.get('event')
        if event:
            diagnostic.append(str(event))
            if event == 'PHASE_SHIFT':
                self.pid.reset()
        if gear_state['mode'] in {'neutral', 'enter_neutral'}:
            throttle, brake, steer = (0.0, 0.0, 0.0)
            heading_error = 0.0
            lookahead_idx = time_idx
            lookahead_dist = 0.0
        elif gear_state['mode'] == 'brake_to_shift':
            throttle, brake, steer = (0.0, 1.0, 0.0)
            heading_error = 0.0
            lookahead_idx = time_idx
            lookahead_dist = 0.0
        else:
            throttle, brake, _pid_out = self.pid.step(target_speed_cmd, actual_speed, self.config.dt)
            steer, heading_error, lookahead_idx, lookahead_dist = self.tracker.compute_steer(x, y, vehicle_state['yaw_deg'], actual_speed, self.gear.current_forward)
        if throttle > 0.85 and actual_speed < max(0.35, target_speed_cmd * 0.35) and (target_speed_cmd > 0.6):
            self.high_throttle_low_speed_time += self.config.dt
            if self.high_throttle_low_speed_time > 2.0:
                diagnostic.append('SPEED_LOW_HIGH_THROTTLE')
                gear_cfg = self.gear.config
                if self.gear.current_forward:
                    if sim_time - self.last_gear_probe_time >= gear_cfg.gear_probe_interval_s:
                        self.gear.next_forward_gear()
                        self.last_gear_probe_time = sim_time
                        self.pid.reset()
                        diagnostic.append('GEAR_PROBE')
                self.high_throttle_low_speed_time = 0.0
        else:
            self.high_throttle_low_speed_time = 0.0
        return {'throttle': float(throttle), 'brake': float(brake), 'steer': float(steer), 'reverse': bool(gear_state['reverse']), 'current_forward': bool(gear_state['forward']), 'gear': int(self.gear.current_gear()) if gear_state['mode'] == 'drive' else int(gear_state['gear']), 'manual_gear_shift': True, 'target_idx': time_idx, 'lookahead_idx': lookahead_idx, 'is_forward': target_forward, 'ref_x': ref_x, 'ref_y': ref_y, 'target_abs_kmh': ref_speed * 3.6, 'cmd_abs_kmh': target_speed_cmd * 3.6, 'pos_err_m': pos_err, 'geometric_err_m': geometric_dist, 'cte_m': path_error['path_cte_m'], 'path_cte_m': path_error['path_cte_m'], 'path_distance_m': path_error['path_distance_m'], 'nearest_path_idx': path_error['nearest_path_idx'], 'nearest_path_u': path_error['nearest_path_u'], 'nearest_path_x': path_error['nearest_path_x'], 'nearest_path_y': path_error['nearest_path_y'], 'time_sync_cte_m': time_sync_cte, 'time_sync_heading_error_rad': time_sync_heading_error, 'heading_error_rad': heading_error, 'path_heading_error_rad': path_heading_error, 'lookahead_dist_m': lookahead_dist, 'diagnostic': '|'.join(diagnostic)}
'''

# Bundled source: _tracking_embedded.machinery.physics
MODULE_SOURCES['_tracking_embedded.machinery.physics'] = '''"""面向低速农业作业的 CARLA 农机物理参数及应用过程。"""
from __future__ import annotations

class AgriPhysicsConfig:
    """CARLA 农机宏观运动参数模型。

    参数用于获得与拖拉机低速、较大转弯半径和换挡响应相符的宏观表现，并非
    完整轮胎—土壤高精度动力学模型。
    """

    def __init__(self, vehicle_type: str='tractor'):
        """设置本文已验证的拖拉机质量、轮胎、发动机和传动系统默认值。"""
        if vehicle_type.lower() != 'tractor':
            raise ValueError(f'Only the validated tractor physics template is currently supported: {vehicle_type}')
        self.mass = 5760.0
        self.center_of_mass = -0.2
        self.drag_coefficient = 1.0
        self.tire_friction = 3.5
        self.damping_rate = 1.0
        self.max_steer_angle = 48.0
        self.long_stiff_value = 5000.0
        self.lat_stiff_max_load = 4.0
        self.lat_stiff_value = 20.0
        self.max_brake_torque = 1400.0
        self.max_rpm = 2400.0
        self.moi = 1.0
        self.damping_rate_full_throttle = 0.15
        self.damping_rate_zero_throttle_clutch_engaged = 2.0
        self.torque_curve = [(0, 300), (700, 300), (1400, 500), (2100, 500), (2700, 500)]
        self.torque_multiplier = 1.0
        self.gear_switch_time = 0.3
        self.final_ratio = 5.28
        self.forward_gears = [(15.2, 0.2, 0.6)]
        self.gear_ratio_multiplier = 1.0

    def to_dict(self) -> dict[str, object]:
        """导出当前全部可配置物理字段。"""
        return {'mass': self.mass, 'center_of_mass': self.center_of_mass, 'drag_coefficient': self.drag_coefficient, 'tire_friction': self.tire_friction, 'damping_rate': self.damping_rate, 'max_steer_angle': self.max_steer_angle, 'long_stiff_value': self.long_stiff_value, 'lat_stiff_max_load': self.lat_stiff_max_load, 'lat_stiff_value': self.lat_stiff_value, 'max_brake_torque': self.max_brake_torque, 'max_rpm': self.max_rpm, 'moi': self.moi, 'damping_rate_full_throttle': self.damping_rate_full_throttle, 'damping_rate_zero_throttle_clutch_engaged': self.damping_rate_zero_throttle_clutch_engaged, 'torque_curve': self.torque_curve, 'torque_multiplier': self.torque_multiplier, 'gear_switch_time': self.gear_switch_time, 'final_ratio': self.final_ratio, 'forward_gears': self.forward_gears, 'gear_ratio_multiplier': self.gear_ratio_multiplier}

def apply_agri_physics(vehicle, config: AgriPhysicsConfig) -> None:
    """将农机参数写入已生成的 CARLA 车辆 actor。

    该函数会覆盖 actor 的质量、发动机、挡位和车轮参数；全流程导航若需使用
    CARLA 默认车辆响应，可由实验配置选择不调用本函数。
    """
    import carla
    physics_control = vehicle.get_physics_control()
    physics_control.mass = config.mass
    physics_control.drag_coefficient = config.drag_coefficient
    physics_control.center_of_mass = carla.Vector3D(0, 0, config.center_of_mass)
    physics_control.max_rpm = config.max_rpm
    physics_control.moi = config.moi
    physics_control.damping_rate_full_throttle = config.damping_rate_full_throttle
    physics_control.damping_rate_zero_throttle_clutch_engaged = config.damping_rate_zero_throttle_clutch_engaged
    physics_control.torque_curve = [carla.Vector2D(x=rpm, y=torque * config.torque_multiplier) for rpm, torque in config.torque_curve]
    physics_control.final_ratio = config.final_ratio
    physics_control.gear_switch_time = config.gear_switch_time
    gear_list = []
    for ratio, down, up in config.forward_gears:
        gear = carla.GearPhysicsControl()
        gear.ratio = ratio * config.gear_ratio_multiplier
        gear.down_ratio = down
        gear.up_ratio = up
        gear_list.append(gear)
    physics_control.forward_gears = gear_list
    new_wheels = []
    for wheel in physics_control.wheels:
        wheel.tire_friction = config.tire_friction
        wheel.damping_rate = config.damping_rate
        wheel.max_steer_angle = config.max_steer_angle
        wheel.long_stiff_value = config.long_stiff_value
        wheel.lat_stiff_max_load = config.lat_stiff_max_load
        wheel.lat_stiff_value = config.lat_stiff_value
        wheel.max_brake_torque = config.max_brake_torque
        new_wheels.append(wheel)
    physics_control.wheels = new_wheels
    vehicle.apply_physics_control(physics_control)
'''

# Bundled source: _tracking_embedded.evaluation.metrics
MODULE_SOURCES['_tracking_embedded.evaluation.metrics'] = '''"""轨迹跟踪和全流程导航使用的基础评价指标。"""
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
        return TrackingMetrics(mae_m=float('nan'), rmse_m=float('nan'), max_error_m=float('nan'), samples=0)
    abs_values = np.abs(values)
    return TrackingMetrics(mae_m=float(np.mean(abs_values)), rmse_m=float(np.sqrt(np.mean(values ** 2))), max_error_m=float(np.max(abs_values)), samples=int(values.size))

def summarize_tick_records(record_csv: str | Path, output_csv: str | Path | None=None) -> pd.DataFrame:
    """从逐帧记录中的 ``cte_m`` 列生成简要误差表。"""
    df = pd.read_csv(record_csv)
    metrics = compute_tracking_metrics(df['cte_m'].values)
    rows = [metrics.to_dict()]
    summary = pd.DataFrame(rows)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return summary

def task_level_summary(task_completed: bool, stage_transition_completed: bool, forward_reverse_completed: bool | None, collision_count: int, total_distance_m: float, operation_path_length_m: float, task_duration_s: float) -> pd.DataFrame:
    """生成论文全流程导航案例所用的任务级评价表。

    完成性项目以 Completed/Failed 表示；碰撞、距离和时长保留为可直接
    报告的数值结果。该函数不推断任务状态，只整理运行器提供的观测值。
    """
    return pd.DataFrame([{'evaluation_item': 'Task completion', 'description': 'Whether the vehicle completed departure, road driving, field entry, field operation and return', 'result': 'Completed' if task_completed else 'Failed'}, {'evaluation_item': 'Stage transition', 'description': 'Whether different navigation stages were switched successfully', 'result': 'Completed' if stage_transition_completed else 'Failed'}, {'evaluation_item': 'Forward/reverse operation', 'description': 'Whether forward and reverse segments were executed correctly', 'result': 'Not applicable (forward-only task)' if forward_reverse_completed is None else 'Completed' if forward_reverse_completed else 'Failed'}, {'evaluation_item': 'Collision event', 'description': 'Whether collision occurred during the task', 'result': collision_count}, {'evaluation_item': 'Total travel distance', 'description': 'Simulated travel distance during the task', 'result': f'{total_distance_m:.3f} m'}, {'evaluation_item': 'Operation path length', 'description': 'Path length inside the field operation area', 'result': f'{operation_path_length_m:.3f} m'}, {'evaluation_item': 'Task duration', 'description': 'Total simulation time', 'result': f'{task_duration_s:.3f} s'}])
'''

# Bundled source: _tracking_embedded.evaluation.performance_evaluation_subagent
MODULE_SOURCES['_tracking_embedded.evaluation.performance_evaluation_subagent'] = '''"""实验评价智能体（PES）的逐帧记录、统计和绘图实现。"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from _tracking_embedded.utils.io import ensure_dir

class PerformanceEvaluationSubagent:
    """收集 CARLA 逐帧数据并生成与原始实验脚本口径一致的报告。"""

    def __init__(self, output_dir: str | Path):
        """创建输出目录，并初始化逐帧记录和诊断事件缓冲区。"""
        self.output_dir = ensure_dir(output_dir)
        self.rows: list[dict[str, object]] = []
        self.events: list[dict[str, object]] = []

    def add_event(self, sim_time: float, code: str, message: str) -> None:
        """记录换挡、结束等离散事件，同时输出到终端。"""
        row = {'time': sim_time, 'code': code, 'message': message}
        self.events.append(row)
        print(f'[event {sim_time:7.2f}s] {code}: {message}')

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
        record_path = self.output_dir / 'tick_records.csv'
        event_path = self.output_dir / 'diagnostic_events.csv'
        summary_path = self.output_dir / 'phase_summary.csv'
        metrics_path = self.output_dir / 'tracking_metrics_summary.csv'
        records.to_csv(record_path, index=False, encoding='utf-8-sig')
        events.to_csv(event_path, index=False, encoding='utf-8-sig')
        summary = self.build_phase_summary(records)
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        metrics = self.build_overall_metrics(records)
        metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        if not records.empty and trajectory is not None:
            self.plot_all(records, trajectory, metrics)
        self.write_readme(metrics)
        return {'tick_records': str(record_path), 'diagnostic_events': str(event_path), 'phase_summary': str(summary_path), 'tracking_metrics_summary': str(metrics_path)}

    @staticmethod
    def metric_values(records: pd.DataFrame, prefix: str) -> dict[str, float | int]:
        """计算最近路径横向误差和航向误差的常用统计量。

        横向误差取自 ``path_cte_m``，航向误差取自
        ``path_heading_error_deg``；两者均基于最近参考路径线段，而不是
        简单比较同一时间戳的参考点与车辆位置。
        """
        if records.empty:
            return {}
        lateral_error = records['path_cte_m'].astype(float).values
        abs_lateral_error = np.abs(lateral_error)
        heading_error = records['path_heading_error_deg'].astype(float).values
        abs_heading_error = np.abs(heading_error)
        return {f'{prefix}_sample_count': len(records), f'{prefix}_lateral_rmse_m': math.sqrt(np.mean(lateral_error ** 2)), f'{prefix}_lateral_max_abs_m': np.max(abs_lateral_error), f'{prefix}_lateral_p95_abs_m': np.percentile(abs_lateral_error, 95), f'{prefix}_lateral_std_m': np.std(lateral_error), f'{prefix}_lateral_mean_abs_m': np.mean(abs_lateral_error), f'{prefix}_heading_rmse_deg': math.sqrt(np.mean(heading_error ** 2)), f'{prefix}_heading_max_abs_deg': np.max(abs_heading_error), f'{prefix}_heading_p95_abs_deg': np.percentile(abs_heading_error, 95), f'{prefix}_heading_std_deg': np.std(heading_error), f'{prefix}_heading_mean_abs_deg': np.mean(abs_heading_error)}

    def build_overall_metrics(self, records: pd.DataFrame) -> pd.DataFrame:
        """同时汇总全参考时段样本和排除瞬态后的有效样本。

        ``all`` 指参考轨迹有效时间内的全部记录；``valid`` 进一步排除起步、
        结束、低速及换挡瞬态。两套结果并存，便于论文统计和问题诊断。
        """
        if records.empty:
            return pd.DataFrame()
        eval_records = records[records['in_reference_time'].astype(bool)] if 'in_reference_time' in records else records
        valid = eval_records[eval_records['metric_valid'].astype(bool)] if 'metric_valid' in eval_records else eval_records
        row: dict[str, float | int] = {'recorded_sample_count': len(records), 'total_sample_count': len(eval_records), 'valid_sample_count': len(valid), 'post_endpoint_sample_count': len(records) - len(eval_records), 'valid_ratio': len(valid) / len(eval_records) if len(eval_records) > 0 else 0.0}
        row.update(self.metric_values(valid, 'valid'))
        row.update(self.metric_values(eval_records, 'all'))
        return pd.DataFrame([row])

    def build_phase_summary(self, records: pd.DataFrame) -> pd.DataFrame:
        """按前进/倒车轨迹段统计误差、速度误差和最大转向输入。"""
        if records.empty:
            return pd.DataFrame()
        if 'in_reference_time' in records:
            records = records[records['in_reference_time'].astype(bool)]
        if records.empty:
            return pd.DataFrame()
        rows = []
        for phase_id, group in records.groupby('phase_id', as_index=False):
            metric_row = self.metric_values(group, 'all')
            rows.append({'phase_id': phase_id, 'direction': 'forward' if bool(group['is_forward'].iloc[0]) else 'reverse', 'start_time_s': group['time_s'].iloc[0], 'end_time_s': group['time_s'].iloc[-1], 'mean_pos_err_m': group['pos_err_m'].mean(), 'max_pos_err_m': group['pos_err_m'].max(), 'metric_sample_scope': 'all', 'metric_sample_count': len(group), 'lateral_rmse_m': metric_row['all_lateral_rmse_m'], 'lateral_max_abs_m': metric_row['all_lateral_max_abs_m'], 'lateral_p95_abs_m': metric_row['all_lateral_p95_abs_m'], 'lateral_std_m': metric_row['all_lateral_std_m'], 'lateral_mean_abs_m': metric_row['all_lateral_mean_abs_m'], 'heading_rmse_deg': metric_row['all_heading_rmse_deg'], 'heading_max_abs_deg': metric_row['all_heading_max_abs_deg'], 'heading_p95_abs_deg': metric_row['all_heading_p95_abs_deg'], 'heading_std_deg': metric_row['all_heading_std_deg'], 'heading_mean_abs_deg': metric_row['all_heading_mean_abs_deg'], 'speed_rmse_kmh': math.sqrt(np.mean((group['actual_abs_kmh'] - group['target_abs_kmh']) ** 2)), 'max_abs_steer': np.abs(group['steer']).max()})
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
                ax.axvspan(start, end, color='orange', alpha=0.12)
                start = None

    def plot_all(self, records: pd.DataFrame, trajectory, metrics: pd.DataFrame) -> None:
        """绘制误差时序、统计柱状图和参考/实际空间轨迹。"""
        import matplotlib.pyplot as plt
        if 'in_reference_time' in records:
            plot_records = records[records['in_reference_time'].astype(bool)]
        else:
            plot_records = records
        if plot_records.empty:
            return
        time_axis = plot_records['time_s'].values
        reverse_mask = ~plot_records['is_forward'].astype(bool).values
        lateral_error = plot_records['path_cte_m'].astype(float).values
        heading_error = plot_records['path_heading_error_deg'].astype(float).values
        metric_row = metrics.iloc[0] if not metrics.empty else None
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        axes[0, 0].plot(time_axis, lateral_error, 'g-', linewidth=1.1, label='All samples')
        self.shade_reverse_spans(axes[0, 0], time_axis, reverse_mask)
        axes[0, 0].axhline(0, color='black', linewidth=0.8)
        axes[0, 0].set_xlabel('Simulation time (s)')
        axes[0, 0].set_ylabel('Nearest-path lateral error (m)')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.35)
        axes[0, 1].plot(time_axis, heading_error, 'b-', linewidth=1.1, label='All samples')
        self.shade_reverse_spans(axes[0, 1], time_axis, reverse_mask)
        axes[0, 1].axhline(0, color='black', linewidth=0.8)
        axes[0, 1].set_xlabel('Simulation time (s)')
        axes[0, 1].set_ylabel('Nearest-path heading error (deg)')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.35)
        if metric_row is not None:
            names = ['RMSE', 'MAE', 'P95', 'Max', 'Std']
            colors = ['#4daf4a', '#984ea3', '#ff7f00', '#e41a1c', '#377eb8']
            lateral_values = [metric_row['all_lateral_rmse_m'], metric_row['all_lateral_mean_abs_m'], metric_row['all_lateral_p95_abs_m'], metric_row['all_lateral_max_abs_m'], metric_row['all_lateral_std_m']]
            heading_values = [metric_row['all_heading_rmse_deg'], metric_row['all_heading_mean_abs_deg'], metric_row['all_heading_p95_abs_deg'], metric_row['all_heading_max_abs_deg'], metric_row['all_heading_std_deg']]
            axes[1, 0].bar(names, lateral_values, color=colors)
            axes[1, 0].set_ylabel('Meters')
            axes[1, 0].set_title('Full-run lateral error metrics')
            axes[1, 0].grid(True, axis='y', alpha=0.35)
            for index, value in enumerate(lateral_values):
                axes[1, 0].text(index, value, f'{value:.3f}', ha='center', va='bottom')
            axes[1, 1].bar(names, heading_values, color=colors)
            axes[1, 1].set_ylabel('Degrees')
            axes[1, 1].set_title('Full-run heading error metrics')
            axes[1, 1].grid(True, axis='y', alpha=0.35)
            for index, value in enumerate(heading_values):
                axes[1, 1].text(index, value, f'{value:.2f}', ha='center', va='bottom')
        fig.suptitle('Nearest-Path Tracking Metrics', y=0.98)
        fig.tight_layout()
        fig.savefig(self.output_dir / '01_tracking_error_metrics.png', dpi=150)
        fig.savefig(self.output_dir / '01_time_sync_diagnostics.png', dpi=150)
        plt.close(fig)
        target_x = trajectory.x + records['map_offset_x'].iloc[0]
        target_y = trajectory.y + records['map_offset_y'].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(target_x[trajectory.is_forward], target_y[trajectory.is_forward], s=12, c='cyan', label='Target Fwd')
        ax.scatter(target_x[~trajectory.is_forward], target_y[~trajectory.is_forward], s=12, c='orange', label='Target Rev')
        ax.plot(records['x'], records['y'], 'r-', linewidth=1.8, label='Actual Path')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.35)
        fig.tight_layout()
        fig.savefig(self.output_dir / '02_spatial_trajectory.png', dpi=150)
        plt.close(fig)

    def write_readme(self, metrics: pd.DataFrame) -> None:
        """写出指标定义和本轮关键结果，便于独立复核输出文件。"""
        metric_lines = []
        if not metrics.empty:
            row = metrics.iloc[0]
            metric_lines = ['', 'Overall nearest-path metrics:', f"- Total samples: {int(row['total_sample_count'])}", f"- Post-endpoint samples: {int(row['post_endpoint_sample_count'])}", f"- Lateral RMSE: {row['all_lateral_rmse_m']:.4f} m", f"- Lateral MAE: {row['all_lateral_mean_abs_m']:.4f} m", f"- Lateral P95: {row['all_lateral_p95_abs_m']:.4f} m", f"- Maximum lateral error: {row['all_lateral_max_abs_m']:.4f} m", f"- Lateral standard deviation: {row['all_lateral_std_m']:.4f} m", f"- Heading RMSE: {row['all_heading_rmse_deg']:.4f} deg", f"- Heading MAE: {row['all_heading_mean_abs_deg']:.4f} deg", f"- Heading P95: {row['all_heading_p95_abs_deg']:.4f} deg", f"- Maximum heading error: {row['all_heading_max_abs_deg']:.4f} deg", f"- Heading standard deviation: {row['all_heading_std_deg']:.4f} deg"]
        lines = ['Trajectory tracking diagnostic notes', 'This report follows the nearest-path evaluation logic used by result_time.py.', '', '1. pos_err_m is the longitudinal timing error projected onto the local trajectory tangent.', '2. cmd_abs_kmh is the catch-up speed command generated from the timing error.', '3. path_cte_m / cte_m is the signed lateral error to the nearest reference path segment.', '4. time_sync_cte_m is retained as a diagnostic value for same-time reference-point deviation.', '5. path_heading_error_deg is the difference between vehicle motion direction and nearest path heading.', '6. metric_valid marks samples after excluding start, end, low-speed and shifting transients.'] + metric_lines
        (self.output_dir / 'README_diagnostics.txt').write_text('\\n'.join(lines), encoding='utf-8')
'''

# Bundled source: _tracking_embedded.simulation.tracking_experiment
MODULE_SOURCES['_tracking_embedded.simulation.tracking_experiment'] = '''"""真实拖拉机轨迹在 CARLA 中的闭环跟踪实验。"""
from __future__ import annotations
import datetime
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any
from _tracking_embedded.controllers.gear_state_machine import GearStateConfig
from _tracking_embedded.controllers.interface_agent import ControllerInterfaceAgent, ControllerInterfaceConfig
from _tracking_embedded.controllers.pid import PIDConfig
from _tracking_embedded.controllers.pure_pursuit import PurePursuitConfig
from _tracking_embedded.evaluation.performance_evaluation_subagent import PerformanceEvaluationSubagent
from _tracking_embedded.trajectory.reference_path import ReferenceTrajectory
from _tracking_embedded.utils.carla_import import import_carla
from _tracking_embedded.utils.io import ensure_dir

@dataclass
class TrackingExperimentConfig:
    """轨迹跟踪实验的连接、地图、车辆、评价和控制器配置。"""
    host: str = 'localhost'
    port: int = 2000
    timeout: float = 60.0
    carla_map: str = ''
    vehicle_bp: str = 'vehicle.tractor.tractor'
    vehicle_type: str = 'tractor'
    dt: float = 0.05
    map_offset_x: float = 50.0
    map_offset_y: float = -10.0
    spawn_z: float = 2.0
    debug_life_time: float = 1200.0
    output_dir: str = 'results/trajectory_tracking'
    max_extra_time_s: float = 15.0
    finish_speed_ms: float = 0.35
    finish_position_error_m: float = 1.0
    apply_agri_physics: bool = True
    trajectory_flip_y: bool = True
    metric_min_speed_ms: float = 0.2
    metric_skip_start_s: float = 5.0
    metric_skip_end_s: float = 5.0
    controller: ControllerInterfaceConfig | None = None
    controller_optimization: dict[str, Any] | None = None
    vegetation_request: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> 'TrackingExperimentConfig':
        """把 JSON 字典递归转换为实验及其嵌套控制器配置对象。"""
        values = dict(data or {})
        controller = values.get('controller')
        if isinstance(controller, dict):
            controller_values = dict(controller)
            pid = controller_values.get('pid')
            pure_pursuit = controller_values.get('pure_pursuit')
            gear = controller_values.get('gear')
            if isinstance(pid, dict):
                controller_values['pid'] = _dataclass_from_dict(PIDConfig, pid)
            if isinstance(pure_pursuit, dict):
                controller_values['pure_pursuit'] = _dataclass_from_dict(PurePursuitConfig, pure_pursuit)
            if isinstance(gear, dict):
                gear_values = dict(gear)
                if isinstance(gear_values.get('forward_gears'), list):
                    gear_values['forward_gears'] = tuple(gear_values['forward_gears'])
                controller_values['gear'] = _dataclass_from_dict(GearStateConfig, gear_values)
            values['controller'] = _dataclass_from_dict(ControllerInterfaceConfig, controller_values)
        return _dataclass_from_dict(cls, values)

class TrackingExperimentRunner:
    """运行论文第 3.1 节所用的真实轨迹复现与跟踪实验。"""

    def __init__(self, config: TrackingExperimentConfig | None=None, controller_agent_class=ControllerInterfaceAgent, performance_evaluation_subagent_class=PerformanceEvaluationSubagent):
        """建立 CARLA 连接，并按配置选择当前世界或加载指定地图。"""
        self.config = config or TrackingExperimentConfig()
        self.controller_agent_class = controller_agent_class
        self.performance_evaluation_subagent_class = performance_evaluation_subagent_class
        self.carla = import_carla()
        self.client = self.carla.Client(self.config.host, self.config.port)
        self.client.set_timeout(self.config.timeout)
        self.world = self.client.load_world(self.config.carla_map) if self.config.carla_map else self.client.get_world()
        if self.config.vegetation_request:
            from _tracking_embedded.environment.vegetation import restore_experiment_vegetation
            restore_experiment_vegetation(self.config)

    def run(self, trajectory_csv: str | Path) -> dict[str, str]:
        """在同步仿真中跟踪一条参考轨迹并保存逐帧记录与评价结果。

        实验结束后无论成功或异常都会销毁车辆并恢复原世界设置，防止本轮
        同步模式和固定步长影响后续 CARLA 实验。
        """
        trajectory = ReferenceTrajectory(trajectory_csv, flip_y=self.config.trajectory_flip_y)
        trajectory.print_data_check()
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = ensure_dir(Path(self.config.output_dir) / f'{trajectory.csv_name}_{timestamp}')
        recorder = self.performance_evaluation_subagent_class(output_dir)
        original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config.dt
        self.world.apply_settings(settings)
        vehicle = None
        try:
            vehicle = self._spawn_vehicle(trajectory)
            self._apply_optional_agri_physics(vehicle)
            self._draw_target_points(trajectory)
            for _ in range(10):
                self.world.tick()
            controller = self.controller_agent_class(trajectory, self.config.map_offset_x, self.config.map_offset_y, self.config.controller or ControllerInterfaceConfig(dt=self.config.dt))
            controller.reset()
            max_sim_time = trajectory.t[-1] + self.config.max_extra_time_s
            max_steps = int(max_sim_time / self.config.dt)
            for step in range(max_steps):
                self.world.tick()
                sim_time = step * self.config.dt
                transform = vehicle.get_transform()
                loc = transform.location
                vel = vehicle.get_velocity()
                actual_speed = math.hypot(vel.x, vel.y)
                state = {'x': loc.x, 'y': loc.y, 'yaw_deg': transform.rotation.yaw, 'actual_speed_ms': actual_speed}
                command = controller.compute(state, sim_time)
                carla_control = self._to_carla_control(command)
                vehicle.apply_control(carla_control)
                signed_actual = actual_speed if command['current_forward'] else -actual_speed
                target_signed = command['target_abs_kmh'] if command['is_forward'] else -command['target_abs_kmh']
                cmd_signed = command['cmd_abs_kmh'] if command['is_forward'] else -command['cmd_abs_kmh']
                time_idx = int(command['target_idx'])
                in_reference_time = sim_time <= trajectory.t[-1] + 1e-09
                metric_reasons = []
                if not in_reference_time:
                    metric_reasons.append('POST_ENDPOINT')
                if sim_time < self.config.metric_skip_start_s:
                    metric_reasons.append('START_TRANSIENT')
                if sim_time > trajectory.t[-1] - self.config.metric_skip_end_s:
                    metric_reasons.append('END_TRANSIENT')
                if actual_speed < self.config.metric_min_speed_ms:
                    metric_reasons.append('LOW_SPEED')
                diagnostic_text = str(command['diagnostic'])
                if 'SHIFT_NEUTRAL' in diagnostic_text:
                    metric_reasons.append('SHIFT_TRANSIENT')
                metric_valid = len(metric_reasons) == 0
                recorder.add_row({'time_s': sim_time, 'x': loc.x, 'y': loc.y, 'yaw_deg': transform.rotation.yaw, 'target_idx': time_idx, 'lookahead_idx': command['lookahead_idx'], 'phase_id': int(trajectory.segment_id[time_idx]), 'is_forward': command['is_forward'], 'target_abs_kmh': command['target_abs_kmh'], 'cmd_abs_kmh': command['cmd_abs_kmh'], 'actual_abs_kmh': actual_speed * 3.6, 'target_signed_kmh': float(target_signed), 'cmd_signed_kmh': float(cmd_signed), 'actual_signed_kmh': signed_actual * 3.6, 'pos_err_m': command['pos_err_m'], 'geometric_err_m': command['geometric_err_m'], 'ref_x': command['ref_x'], 'ref_y': command['ref_y'], 'throttle': command['throttle'], 'brake': command['brake'], 'steer': command['steer'], 'cte_m': command['cte_m'], 'path_cte_m': command['path_cte_m'], 'path_distance_m': command['path_distance_m'], 'nearest_path_idx': command['nearest_path_idx'], 'nearest_path_u': command['nearest_path_u'], 'nearest_path_x': command['nearest_path_x'], 'nearest_path_y': command['nearest_path_y'], 'time_sync_cte_m': command['time_sync_cte_m'], 'time_sync_heading_error_deg': math.degrees(float(command['time_sync_heading_error_rad'])), 'heading_error_deg': math.degrees(float(command['heading_error_rad'])), 'path_heading_error_deg': math.degrees(float(command['path_heading_error_rad'])), 'in_reference_time': in_reference_time, 'metric_valid': metric_valid, 'metric_filter_reason': '|'.join(metric_reasons), 'map_offset_x': self.config.map_offset_x, 'map_offset_y': self.config.map_offset_y, 'diagnostic': command['diagnostic']})
                if command['diagnostic']:
                    for code in str(command['diagnostic']).split('|'):
                        if code == 'PHASE_SHIFT':
                            recorder.add_event(sim_time, code, 'Controller state-machine event')
                is_finished = sim_time >= trajectory.t[-1]
                if is_finished and actual_speed < self.config.finish_speed_ms and (abs(float(command['pos_err_m'])) < self.config.finish_position_error_m):
                    recorder.add_event(sim_time, 'FINISHED', 'The trajectory replay finished with stable position and speed.')
                    break
            return recorder.save(trajectory)
        finally:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
                self.world.tick()
            self.world.apply_settings(original_settings)

    def _spawn_vehicle(self, trajectory: ReferenceTrajectory):
        """按轨迹首段方向计算初始航向并生成拖拉机。"""
        bp = self.world.get_blueprint_library().find(self.config.vehicle_bp)
        x0 = trajectory.x[0] + self.config.map_offset_x
        y0 = trajectory.y[0] + self.config.map_offset_y
        next_i = min(1, trajectory.num_points - 1)
        dx = trajectory.x[next_i] - trajectory.x[0]
        dy = trajectory.y[next_i] - trajectory.y[0]
        yaw = math.degrees(math.atan2(dy, dx))
        if not trajectory.is_forward[0]:
            yaw = (yaw + 180.0) % 360.0
        spawn = self.carla.Transform(self.carla.Location(x=x0, y=y0, z=self.config.spawn_z), self.carla.Rotation(yaw=yaw))
        vehicle = self.world.try_spawn_actor(bp, spawn)
        if vehicle is None:
            raise RuntimeError('Vehicle spawning failed. Check the spawn point and blueprint.')
        return vehicle

    def _to_carla_control(self, command: dict[str, Any]):
        """把 CIA 的通用控制字典转换为 CARLA VehicleControl。"""
        control = self.carla.VehicleControl()
        control.throttle = float(command['throttle'])
        control.brake = float(command['brake'])
        control.steer = float(command['steer'])
        control.reverse = bool(command['reverse'])
        control.hand_brake = False
        control.manual_gear_shift = bool(command.get('manual_gear_shift', True))
        control.gear = int(command['gear'])
        return control

    def _draw_target_points(self, trajectory: ReferenceTrajectory) -> None:
        """在 CARLA 调试视图中以不同颜色标出前进和倒车参考点。"""
        for i in range(0, trajectory.num_points, 2):
            color = self.carla.Color(0, 220, 255) if trajectory.is_forward[i] else self.carla.Color(255, 130, 0)
            self.world.debug.draw_point(self.carla.Location(x=trajectory.x[i] + self.config.map_offset_x, y=trajectory.y[i] + self.config.map_offset_y, z=1.5), size=0.08, color=color, life_time=self.config.debug_life_time)

    def _apply_optional_agri_physics(self, vehicle) -> None:
        """按开关应用农机物理模板；不可用时保留当前 CARLA 参数。"""
        if not self.config.apply_agri_physics:
            return
        try:
            from _tracking_embedded.machinery.physics import AgriPhysicsConfig, apply_agri_physics
        except Exception:
            print('Agricultural physics helper not available; running with the current CARLA vehicle physics.')
            return
        apply_agri_physics(vehicle, AgriPhysicsConfig(self.config.vehicle_type))

def _dataclass_from_dict(dataclass_type, data: dict[str, Any]):
    """仅保留目标数据类声明的字段，忽略 JSON 中的扩展键。"""
    valid_keys = {field.name for field in fields(dataclass_type)}
    return dataclass_type(**{key: value for key, value in data.items() if key in valid_keys})
'''

class EmbeddedModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in MODULE_SOURCES or fullname in PACKAGE_NAMES:
            return importlib.util.spec_from_loader(
                fullname, self, is_package=fullname in PACKAGE_NAMES)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(Path(__file__).resolve()) + "::" + module.__name__
        if module.__name__ in MODULE_SOURCES:
            exec(compile(MODULE_SOURCES[module.__name__], module.__file__, "exec"), module.__dict__)


def install_embedded_modules():
    if not any(isinstance(finder, EmbeddedModules) for finder in sys.meta_path):
        sys.meta_path.insert(0, EmbeddedModules())


def script_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


def prepare_trajectory(source, destination=None):
    from _tracking_embedded.trajectory.preprocessing_subagent import TrajectoryPreprocessingSubagent
    if destination is None:
        return Path(source)
    if Path(source).resolve() == Path(destination).resolve():
        raise ValueError("The standardized CSV must not overwrite the input CSV")
    return TrajectoryPreprocessingSubagent().process(source, destination)


def run_tracking(source, config, processed_csv=None):
    """Same fixed-parameter TPS -> tracking runner -> PES path as the project EEA."""
    from _tracking_embedded.simulation.tracking_experiment import TrackingExperimentRunner
    if (config.controller_optimization or {}).get("enabled"):
        raise ValueError("This file runs fixed controller parameters. Disable optimization "
                         "and enter the locked controller values in EXPERIMENT_CONFIG.")
    csv_to_run = prepare_trajectory(source, processed_csv)
    return TrackingExperimentRunner(config).run(csv_to_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default=TRAJECTORY_CSV)
    parser.add_argument("--output-dir", default=EXPERIMENT_CONFIG["output_dir"])
    parser.add_argument("--host", default=EXPERIMENT_CONFIG["host"])
    parser.add_argument("--port", type=int, default=EXPERIMENT_CONFIG["port"])
    parser.add_argument("--check", action="store_true", help="Offline imports/config only; no CSV/CARLA required")
    parser.add_argument("--prepare-only", action="store_true", help="TPS and input checks only; no CARLA")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    install_embedded_modules()
    from _tracking_embedded.machinery.physics import AgriPhysicsConfig
    from _tracking_embedded.simulation.tracking_experiment import TrackingExperimentConfig
    from _tracking_embedded.trajectory.reference_path import ReferenceTrajectory
    values = copy.deepcopy(EXPERIMENT_CONFIG)
    values.update(host=args.host, port=args.port, output_dir=str(script_path(args.output_dir)))
    if values.get("vegetation_request"):
        raise ValueError("This standalone file tracks on an existing map; set vegetation_request=None")
    if (values.get("controller_optimization") or {}).get("enabled"):
        raise ValueError("Use locked controller values; this file does not run Bayesian optimization")
    config = TrackingExperimentConfig.from_dict(values)
    if config.controller is not None and abs(config.controller.dt-config.dt) > 1e-12:
        raise ValueError("controller.dt must equal experiment dt")
    if args.check:
        print(json.dumps({
            "status": "OFFLINE_IMPORT_CONFIG_OK", "carla_executed": False,
            "crew_ai_used": False, "experiment_config": asdict(config),
            "tractor_physics": AgriPhysicsConfig(config.vehicle_type).to_dict(),
            "source_sha256": SOURCE_SHA256,
        }, indent=2))
        return
    if not args.trajectory or args.trajectory == "FILL_IN_TRAJECTORY.csv":
        parser.error("Fill TRAJECTORY_CSV at the top of this file, or use --trajectory /path/to/data.csv")
    source = script_path(args.trajectory)
    if not source.is_file():
        parser.error("Input CSV not found: " + str(source))
    destination = None
    if PREPROCESS_INPUT:
        destination = script_path(PROCESSED_CSV) if PROCESSED_CSV else (
            Path(config.output_dir) / "processed_trajectory" / (source.stem + "_standard.csv"))
    csv_to_run = prepare_trajectory(source, destination)
    trajectory = ReferenceTrajectory(csv_to_run, flip_y=config.trajectory_flip_y)
    import numpy as np
    if trajectory.num_points < 2 or not np.all(np.isfinite(trajectory.t)) or trajectory.t[-1] <= 0:
        raise ValueError("At least two trajectory points with a positive duration are required")
    if np.any(np.diff(trajectory.t) < 0):
        raise ValueError("Trajectory timestamps must not go backwards")
    if not all(np.all(np.isfinite(value)) for value in (trajectory.x, trajectory.y, trajectory.speed_ms)):
        raise ValueError("Trajectory contains non-finite coordinates or speed")
    if args.prepare_only:
        trajectory.print_data_check()
        print(json.dumps({"status": "PREPARED_ONLY", "carla_executed": False,
                          "trajectory_csv": str(csv_to_run)}, indent=2))
        return
    print("Tracking on:", config.carla_map or "currently open CARLA world",
          "| custom tractor physics:", config.apply_agri_physics)
    physics = AgriPhysicsConfig(config.vehicle_type)
    if config.apply_agri_physics:
        print("Tractor physics:", json.dumps(physics.to_dict()))
        print("Project steering: max_steer_angle applied to all four wheels; "
              "manual gear state machine. No extra rear-wheel lock is added.")
        gears = (config.controller.gear.forward_gears
                 if config.controller and config.controller.gear else (1, 2, 3))
        if any(gear > len(physics.forward_gears) for gear in gears):
            print("Configuration note: the project's controller allows gears", gears,
                  "but the configured physics has", len(physics.forward_gears),
                  "forward gear(s). Keep controller.gear consistent when choosing experiment settings.")
    result = run_tracking(csv_to_run, config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
