"""农业场景生成智能体（SGA）的顶层接口。"""

from __future__ import annotations

from pathlib import Path
import copy
import json
from dataclasses import fields
from typing import Any

from agri_carla.environment import VegetationPlacementConfig, VegetationPlacer
from agri_carla.road_generation.agent import AgriculturalRoadGenerator, RoadSceneGenerationResult, RoadSceneRequest
from agri_carla.utils.io import write_json


def vegetation_with_road_context(request, road_request=None):
    """Use one coordinate frame for roads, field boundaries and tree belts."""
    result = copy.deepcopy(request)
    if road_request:
        for key in ("image_width_px", "image_height_px", "meters_per_pixel", "north_up", "geo_anchor"):
            value = road_request.get(key, True if key == "north_up" else None)
            if value is not None:
                if key in result and result[key] != value:
                    raise ValueError(f"Road and vegetation coordinate metadata disagree: {key}")
                result[key] = value
        result.setdefault("carla_map", road_request.get("map_name", ""))
        result["road_scene_request"] = copy.deepcopy(road_request)
    return result


class SceneGenerationAgent:
    """根据 PGA 校验后的结构化几何生成道路文件和植被放置方案。

    道路节点、中心线和环境区域候选由多模态模型在 PGA 中生成，并经复核、校验
    与用户确认后传入；SGA 不读取图像，也不执行颜色阈值、像素分类或自然场景
    语义识别。道路生成与植被放置由不同本地组件完成，但统一通过同一个顶层
    智能体向编排器报告进度。
    """

    def __init__(
        self,
        generator: AgriculturalRoadGenerator | None = None,
        vegetation_placer: VegetationPlacer | None = None,
    ):
        """注入可替换的道路生成器和植被放置器，便于测试。"""
        self.generator = generator or AgriculturalRoadGenerator()
        self.vegetation_placer = vegetation_placer or VegetationPlacer()

    def run(
        self,
        request: dict,
        output_dir: str | Path,
        context=None,
        vegetation_request=None,
    ) -> RoadSceneGenerationResult:
        """过滤无关字段、构造道路请求并生成 CARLA 场景文件。"""
        if context is not None:
            context.progress(
                "SGA",
                "sga_scene",
                "Validating the agricultural road-scene request.",
            )
        # 只接受 RoadSceneRequest 明确定义的字段，避免模型输出的解释性文本被
        # 意外传入底层道路生成代码。
        if "road_scene_request" in request:
            vegetation_request = request.get("vegetation_request", vegetation_request)
            request = request["road_scene_request"]
        request = dict(request)
        vegetation = None
        if vegetation_request is not None:
            vegetation = vegetation_with_road_context(vegetation_request, request)
            config = VegetationPlacementConfig.from_dict(vegetation)
            from agri_carla.environment.vegetation import _region_from_dict

            if config.grass_regions:
                request["farmland_polygons"] = [
                    self.vegetation_placer._polygon_to_world(config, _region_from_dict(region, "grass"))
                    for region in config.grass_regions
                ]
        valid_keys = {field.name for field in fields(RoadSceneRequest)}
        structured = RoadSceneRequest(**{key: value for key, value in request.items() if key in valid_keys})
        result = self.generator.generate(structured, output_dir)
        if vegetation is not None:
            finalized = json.loads(Path(result.spec_path).read_text(encoding="utf-8"))
            vegetation["road_scene_request"] = finalized
            vegetation["road_xodr_path"] = str(Path(result.xodr_path).resolve())
            vegetation.setdefault("output_dir", str(Path(output_dir) / "vegetation"))
            # Default mesh surfaces are already present in this map's FBX.
            vegetation["grass_regions"] = [r for r in vegetation.get("grass_regions", [])
                                           if r.get("visual_mode", "mesh") != "mesh"]
            result.vegetation_request_path = str(write_json(
                Path(output_dir) / "vegetation_runtime_request.json", vegetation
            ))
        if context is not None:
            context.progress(
                "SGA",
                "sga_scene",
                "Agricultural road-scene files generated and topology validated.",
            )
        return result

    def run_vegetation(
        self,
        request: dict[str, Any],
        context=None,
    ) -> dict[str, Any]:
        """把多边形植被请求转换为 CARLA 对象放置计划或实际生成对象。"""
        if context is not None:
            context.progress(
                "SGA",
                "sga_vegetation",
                "Generating vegetation placement plan.",
            )
        config = VegetationPlacementConfig.from_dict(request)
        result = self.vegetation_placer.run(config)
        if context is not None:
            context.progress(
                "SGA",
                "sga_vegetation",
                "Vegetation placement plan generated.",
            )
        return result
