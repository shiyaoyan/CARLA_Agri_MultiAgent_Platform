"""轨迹处理与车辆控制共用的二维几何计算。"""

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

    # 采用 WGS84 长半轴作为地球半径，并用首点纬度修正经度方向尺度。
    earth_radius = 6378137.0
    lon = np.radians(longitude.astype(float))
    lat = np.radians(latitude.astype(float))
    x = earth_radius * lon * np.cos(lat[0])
    y = earth_radius * lat
    return x - x[0], y - y[0]
