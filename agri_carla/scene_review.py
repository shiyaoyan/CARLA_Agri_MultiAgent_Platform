"""PGA 场景候选结果的可视化复核工具。

该模块不负责识别道路、农田或树带。几何语义由多模态模型给出，本地代码只把
结构化候选叠加到原始卫星图上，供用户在 SGA 执行前确认。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def render_scene_candidate_preview(
    image_path: str | Path,
    structured_request: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """在原始卫星图上绘制候选道路、农田和树带并返回输出路径。

    颜色含义保持固定：黄色为道路中心线，黑色为道路端点或交叉口，蓝色为
    农田/草地区域，绿色为树带区域。预览图只用于人工复核，不参与后续几何计算。
    """
    source = Path(image_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as opened:
        image = opened.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    vegetation = structured_request.get("vegetation_request")
    if isinstance(vegetation, dict):
        for region in vegetation.get("grass_regions", []):
            polygon = _pixel_polygon(region)
            if polygon:
                draw.polygon(
                    polygon,
                    fill=(33, 113, 181, 62),
                    outline=(20, 82, 150, 235),
                    width=3,
                )
        for region in vegetation.get("tree_regions", []):
            polygon = _pixel_polygon(region)
            if polygon:
                draw.polygon(
                    polygon,
                    fill=(36, 148, 84, 58),
                    outline=(22, 120, 69, 245),
                    width=4,
                )

    road_request = structured_request.get("road_scene_request")
    if isinstance(road_request, dict):
        meters_per_pixel = _positive_float(
            road_request.get("meters_per_pixel"),
            default=1.0,
        )
        for road in road_request.get("roads", []):
            points = _polyline(road.get("pixel_polyline"))
            if len(points) < 2:
                continue
            width_m = _positive_float(road.get("width_m"), default=3.0)
            # 中心线宽度按实际道路宽度适度缩放，同时限制上限以免遮住底图。
            line_width = max(4, min(14, int(round(width_m / meters_per_pixel * 0.18))))
            draw.line(
                points,
                fill=(255, 190, 0, 245),
                width=line_width,
                joint="curve",
            )

        for node in road_request.get("nodes", []):
            point = _point(node.get("pixel")) if isinstance(node, dict) else None
            if point is None:
                continue
            radius = 5
            draw.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill=(20, 20, 20, 255),
                outline=(255, 255, 255, 245),
                width=2,
            )

    image.alpha_composite(overlay)
    image.convert("RGB").save(destination, quality=95)
    return destination


def _pixel_polygon(region: Any) -> list[tuple[float, float]]:
    """仅返回像素坐标多边形；世界坐标区域不能直接叠加到卫星图。"""
    if not isinstance(region, dict):
        return []
    if str(region.get("coordinate_mode", "pixel")).lower() != "pixel":
        return []
    points = _polyline(region.get("polygon"))
    return points if len(points) >= 3 else []


def _polyline(value: Any) -> list[tuple[float, float]]:
    """把 JSON 二维点列表转换为 Pillow 可绘制的点序列。"""
    if not isinstance(value, list):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        point = _point(item)
        if point is None:
            return []
        points.append(point)
    return points


def _point(value: Any) -> tuple[float, float] | None:
    """读取一个有限的二维数值点。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _positive_float(value: Any, *, default: float) -> float:
    """读取正数，异常或非正值时使用不会影响预览生成的默认值。"""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default
