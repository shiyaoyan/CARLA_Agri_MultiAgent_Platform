"""SGA 与本地 RoadRunner/OpenDRIVE 生成工具之间的适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agri_carla.road_generation.topology import (
    RoadTopologyValidator,
    TopologyValidationConfig,
)
from agri_carla.utils.io import write_json


@dataclass
class RoadSceneRequest:
    """PGA 产生并由 CrewAI Flow 传给 SGA 的结构化道路场景请求。"""

    map_name: str
    image_width_px: int
    image_height_px: int
    meters_per_pixel: float
    geo_anchor: dict[str, float]
    nodes: list[dict[str, Any]]
    roads: list[dict[str, Any]]
    # 原始卫星图只用于生成道路叠加预览。PGA 场景解析仍要求提供原图；当
    # SGA 复用已验证的结构化道路几何时，可以不再依赖该图像文件。
    image_path: str = ""
    north_up: bool = True
    speed_kph: float = 20.0
    scene_profile: str = "agricultural"
    flat_terrain: bool = True
    description: str = ""
    topology_validation: dict[str, Any] | None = None
    farmland_polygons: list[list[tuple[float, float]]] | None = None
    junction_clearance_m: float | None = None
    carla_0915_boundary_segment_m: float = 0.5


@dataclass
class RoadSceneGenerationResult:
    """一次场景生成返回的文件路径、拓扑规模和自检结果。"""
    xodr_path: str
    fbx_path: str
    texture_path: str
    texture_paths: list[str]
    spec_path: str
    preview_svg_path: str
    import_bundle_dir: str
    road_count: int
    junction_count: int
    connection_count: int
    topology_report_path: str = ""
    topology_valid: bool = True
    regeneration_count: int = 0
    repair_action_count: int = 0
    vegetation_request_path: str = ""


class AgriculturalRoadGenerator:
    """封装本地道路生成及拓扑自查自纠流程。"""

    def generate(self, request: RoadSceneRequest, output_dir: str | Path) -> RoadSceneGenerationResult:
        """生成 XODR、FBX、纹理和导入包，并按配置验证或修复拓扑。"""
        from roadrunner_mcp.realmap import (
            GeoAnchor,
            RealMapTraceSpec,
            TraceNode,
            TraceRoad,
            generate_realmap_artifacts,
        )

        if not request.flat_terrain:
            raise ValueError(
                "The satellite-trace SGA currently supports flat terrain only. "
                "Provide flat_terrain=true or add a verified elevation profile."
            )

        # 将平台通用字典转换为底层工具使用的强类型地理锚点、节点和道路对象。
        anchor = GeoAnchor(**request.geo_anchor)
        nodes = [TraceNode(node_id=item["node_id"], pixel=tuple(item["pixel"])) for item in request.nodes]
        roads = [
            TraceRoad(
                road_id=item["road_id"],
                start_node=item["start_node"],
                end_node=item["end_node"],
                pixel_polyline=[tuple(point) for point in item["pixel_polyline"]],
                width_m=float(item.get("width_m", 3.0)),
                surface=item.get("surface", "dirt"),
            )
            for item in request.roads
        ]
        spec = RealMapTraceSpec(
            map_name=request.map_name,
            image_path=request.image_path,
            image_width_px=request.image_width_px,
            image_height_px=request.image_height_px,
            meters_per_pixel=request.meters_per_pixel,
            north_up=request.north_up,
            geo_anchor=anchor,
            nodes=nodes,
            roads=roads,
            speed_kph=request.speed_kph,
            scene_profile=request.scene_profile,
            description=request.description,
            farmland_polygons=request.farmland_polygons,
            junction_clearance_m=request.junction_clearance_m,
            carla_0915_boundary_segment_m=request.carla_0915_boundary_segment_m,
        )
        output_dir = Path(output_dir)
        validation_config = TopologyValidationConfig.from_dict(request.topology_validation)
        # 关闭自检时保持最短生成路径；开启后先在内存中修复规格，再生成文件并
        # 校验 OpenDRIVE。这样无道路、无效节点引用等输入不会先留下半成品。
        if not validation_config.enabled:
            artifact_result = generate_realmap_artifacts(spec, output_dir)
            return self._result(artifact_result)

        validator = RoadTopologyValidator(validation_config)
        current_spec = spec.normalized()
        attempt_records: list[dict[str, Any]] = []
        pending_actions: list[dict[str, Any]] = []
        repair_rounds = 0
        max_repair_rounds = max(0, int(validation_config.max_regeneration_attempts))

        while True:
            spec_check = validator.validate_spec(current_spec)
            attempt_records.append({
                "phase": "pre_generation",
                "attempt": len(attempt_records) + 1,
                "repairs_applied_before_validation": pending_actions,
                "spec_check": spec_check.to_dict(),
                "valid": spec_check.valid,
            })
            if spec_check.valid:
                break
            if (
                not validation_config.auto_repair
                or repair_rounds >= max_repair_rounds
            ):
                break

            repaired_spec, repair_actions = validator.repair_spec(current_spec)
            if not repair_actions:
                break
            current_spec = repaired_spec
            pending_actions = repair_actions
            repair_rounds += 1

        preflight_valid = spec_check.valid
        result: Any = None
        xodr_check = None
        if preflight_valid or not validation_config.fail_on_unresolved:
            try:
                result = generate_realmap_artifacts(current_spec, output_dir)
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                attempt_records.append({
                    "phase": "artifact_generation",
                    "attempt": len(attempt_records) + 1,
                    "valid": False,
                    "error": str(exc),
                })
            else:
                xodr_check = validator.validate_xodr(result.xodr_path)
                attempt_records.append({
                    "phase": "post_generation",
                    "attempt": len(attempt_records) + 1,
                    "xodr_check": xodr_check.to_dict(),
                    "valid": xodr_check.valid,
                })

        final_valid = bool(
            preflight_valid
            and xodr_check is not None
            and xodr_check.valid
        )

        repair_action_count = sum(
            len(record.get("repairs_applied_before_validation", []))
            for record in attempt_records
        )
        report = {
            "map_name": current_spec.map_name,
            "enabled": True,
            "auto_repair": validation_config.auto_repair,
            "valid": final_valid,
            "regeneration_count": repair_rounds,
            "repair_action_count": repair_action_count,
            "configuration": {
                field: getattr(validation_config, field)
                for field in validation_config.__dataclass_fields__
            },
            "attempts": attempt_records,
        }
        if result is not None and final_valid:
            from .carla_diagnostics import editor_route_coverage
            # Open boundaries are valid. Report the stock editor limitation
            # separately, without inserting artificial roads or failing topology.
            report["editor_route_coverage"] = editor_route_coverage(
                Path(result.xodr_path).read_text(encoding="utf-8")
            )
        report_path = write_json(
            output_dir / f"{current_spec.map_name}.topology_report.json",
            report,
        )

        if not final_valid and validation_config.fail_on_unresolved:
            raise RuntimeError(
                "Agricultural road topology remains invalid after repair attempts. "
                f"See {report_path}."
            )
        if result is None:
            raise RuntimeError(
                "Agricultural road artifacts could not be generated from the request. "
                f"See {report_path}."
            )
        generated = self._result(
            result,
            topology_report_path=str(report_path),
            topology_valid=final_valid,
            regeneration_count=repair_rounds,
            repair_action_count=repair_action_count,
        )
        return generated

    @staticmethod
    def _result(
        result,
        topology_report_path: str = "",
        topology_valid: bool = True,
        regeneration_count: int = 0,
        repair_action_count: int = 0,
    ) -> RoadSceneGenerationResult:
        """把底层工具返回对象整理为平台稳定的结果数据类。"""
        return RoadSceneGenerationResult(
            xodr_path=str(result.xodr_path),
            fbx_path=str(result.fbx_path),
            texture_path=str(result.texture_path),
            texture_paths=[str(path) for path in result.texture_paths],
            spec_path=str(result.spec_path),
            preview_svg_path=str(result.preview_svg_path),
            import_bundle_dir=str(result.import_bundle_dir),
            road_count=result.road_count,
            junction_count=result.junction_count,
            connection_count=result.connection_count,
            topology_report_path=topology_report_path,
            topology_valid=topology_valid,
            regeneration_count=regeneration_count,
            repair_action_count=repair_action_count,
        )
