"""农业场景生成智能体（SGA）的顶层接口。"""

from __future__ import annotations

from pathlib import Path
from dataclasses import fields
from typing import Any

from agri_carla.environment import VegetationPlacementConfig, VegetationPlacer
from agri_carla.road_generation.agent import AgriculturalRoadGenerator, RoadSceneGenerationResult, RoadSceneRequest


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
        valid_keys = {field.name for field in fields(RoadSceneRequest)}
        structured = RoadSceneRequest(**{key: value for key, value in request.items() if key in valid_keys})
        result = self.generator.generate(structured, output_dir)
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
