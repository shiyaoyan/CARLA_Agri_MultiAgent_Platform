"""任务规划智能体（PGA）：输入解析、请求校验和执行计划生成。

PGA 是用户请求进入多智能体系统后的统一处理入口。它允许直接使用结构化请求，
也可调用支持图像输入的大模型解析自然语言及附件。首次规划使用原始卫星图和
用户的空间描述，由多模态模型提出道路中心线、农田及树带候选区域；PGA 只进行
字段、坐标和拓扑约束校验，不执行颜色阈值、骨架化或像素分类。候选结果先生成
叠加预览，用户确认后才能交给 SGA；若不符合要求，可补充描述或提供修订标注图。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

from agri_carla.llm import JSONLLMClient, LLMConfig, build_llm_client
from agri_carla.orchestration.errors import (
    InputValidationError,
    NeedsUserInputError,
)
from agri_carla.orchestration.models import ExecutionPlan, PlanningResult, TaskStep
from agri_carla.orchestration.registry import AgentRegistry, build_default_registry
from agri_carla.scene_review import render_scene_candidate_preview
from agri_carla.utils.io import read_yaml


SUPPORTED_TASK_TYPES = {
    "trajectory_tracking",
    "full_process_navigation",
    "configuration_only",
    "scene_enrichment",
}


class PlanGenerationAgent:
    """将多源用户输入转换为已校验请求和有依赖关系的执行计划。"""

    def __init__(
        self,
        llm_client: JSONLLMClient | None = None,
        *,
        project_root: str | Path | None = None,
        registry: AgentRegistry | None = None,
    ):
        """注入可选大模型客户端，并确定项目根目录和智能体能力白名单。"""
        self.llm_client = llm_client
        self.project_root = Path(project_root or ".").resolve()
        self.registry = registry or build_default_registry()
        self._llm_trace: list[dict[str, Any]] = []

    def run(self, payload: dict[str, Any]) -> PlanningResult:
        """完成一次任务规划并返回 CrewAI Flow 可执行的规划结果。

        处理顺序为：获得候选结构化请求、补全附件路径、复核模型生成的道路几何、
        校验字段与文件、必要时要求大模型修复，最后生成执行步骤及依赖关系。
        """
        self._llm_trace = []
        source, structured_request = self._obtain_structured_request(payload)
        llm_config = LLMConfig.from_dict(payload.get("llm"))
        structured_request = self._apply_attachment_defaults(
            structured_request,
            payload,
        )
        geometry_verified = False
        if source == "llm" and self._has_visual_geometry(structured_request):
            structured_request = self._verify_geometry_with_llm(
                payload,
                structured_request,
                llm_config,
            )
            structured_request = self._apply_attachment_defaults(
                structured_request,
                payload,
            )
            geometry_verified = True
        strict_paths = bool(
            payload.get("validation", {}).get(
                "strict_paths",
                source == "llm",
            )
        )
        errors, warnings = self.validate_request(
            structured_request,
            strict_paths=strict_paths,
            require_execution_identifiers=source == "llm",
        )

        repair_attempts = 0
        # 只有大模型生成的请求允许自动修复；用户直接提交的结构化文件若有错误，
        # 应原样报告，避免平台悄悄改写用户已经明确指定的实验配置。
        while errors and source == "llm" and repair_attempts < llm_config.max_repair_attempts:
            repair_attempts += 1
            structured_request = self._repair_with_llm(
                payload,
                structured_request,
                errors,
                llm_config,
            )
            structured_request = self._apply_attachment_defaults(
                structured_request,
                payload,
            )
            if not geometry_verified and self._has_visual_geometry(
                structured_request
            ):
                structured_request = self._verify_geometry_with_llm(
                    payload,
                    structured_request,
                    llm_config,
                )
                structured_request = self._apply_attachment_defaults(
                    structured_request,
                    payload,
                )
                geometry_verified = True
            errors, warnings = self.validate_request(
                structured_request,
                strict_paths=strict_paths,
                require_execution_identifiers=True,
            )

        if errors:
            missing = [item for item in errors if item.startswith("Missing ")]
            error_class = NeedsUserInputError if missing else InputValidationError
            raise error_class(errors)

        scene_review: dict[str, Any] = {}
        if source == "llm" and self._has_visual_geometry(structured_request):
            structured_request["multimodal_geometry"] = {
                "recognition_source": "multimodal_llm",
                "independent_verification": geometry_verified,
                "model": llm_config.model,
                "input_mode": "raw_satellite_image_with_text_guidance",
                "correction_images_used": bool(
                    self._correction_image_paths(payload)
                ),
                "geometry_types": [
                    name
                    for name, key in (
                        ("road_network", "road_scene_request"),
                        ("vegetation_regions", "vegetation_request"),
                    )
                    if isinstance(structured_request.get(key), dict)
                ],
            }
            structured_request, scene_review = self._attach_scene_review(
                payload,
                structured_request,
            )

        plan = self.build_execution_plan(structured_request)
        plan.validate_dependencies()
        return PlanningResult(
            structured_request=structured_request,
            execution_plan=plan,
            validation_report={
                "valid": True,
                "errors": [],
                "warnings": warnings,
                "strict_paths": strict_paths,
                "llm_repair_attempts": repair_attempts,
                "llm_geometry_verified": geometry_verified,
                "scene_review": scene_review,
                "llm_trace": list(self._llm_trace),
                "registered_agents": self.registry.describe(),
            },
            source=source,
        )

    def _obtain_structured_request(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """按优先级获得结构化请求，并标记其来源。

        优先级依次为内嵌字典、结构化文件、完整任务请求和大模型解析。来源标记
        会决定是否要求严格路径校验以及是否允许大模型自动修复。
        """
        if isinstance(payload.get("structured_request"), dict):
            return "provided", copy.deepcopy(payload["structured_request"])

        structured_file = payload.get("structured_request_file")
        if structured_file:
            return "provided_file", read_yaml(self._resolve_path(structured_file))

        if payload.get("task_type") in SUPPORTED_TASK_TYPES:
            return "provided_request", copy.deepcopy(payload)

        natural_request = str(payload.get("natural_language_request", "")).strip()
        if not natural_request:
            raise NeedsUserInputError(
                ["Missing natural_language_request or structured_request."]
            )

        llm_config = LLMConfig.from_dict(payload.get("llm"))
        if not llm_config.enabled and self.llm_client is None:
            raise NeedsUserInputError(
                [
                    "Missing enabled LLM configuration. Provide llm.enabled=true "
                    "or submit an existing structured request."
                ]
            )
        # 大模型仅返回 JSON 候选，不直接调用智能体、执行脚本或连接 CARLA。
        client = self.llm_client or build_llm_client(llm_config)
        system_prompt, user_prompt = self._build_prompts(payload)
        candidate = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=self._resolved_image_paths(payload),
        )
        if isinstance(candidate.get("structured_request"), dict):
            candidate = candidate["structured_request"]
        self._record_llm_candidate("initial", candidate, llm_config)
        if candidate.get("needs_user_input"):
            questions = candidate.get("questions") or [
                "The model reported that required information is missing."
            ]
            raise NeedsUserInputError(questions)
        return "llm", self._merge_request_template(payload, candidate)

    def _build_prompts(
        self,
        payload: dict[str, Any],
    ) -> tuple[str, str]:
        """构造约束大模型输出的系统提示词和用户提示词。

        提示词显式提供 JSON Schema、可用智能体和附件摘要，并禁止模型虚构地图、
        蓝图或实验结果。首次规划要求模型直接理解原始卫星图；如果用户对候选
        结果进行了修订，附加标注图只作为第二轮校正依据。
        """
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "llm"
            / "task_request.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        template = self._request_template(payload)
        attachment_summary = self._attachment_summary(payload)
        geometry_system_instruction = (
            "The primary image is an original, unannotated satellite image or orthoimage. "
            "Interpret it together with the user's spatial description and propose only "
            "the agricultural roads, farmland/grass areas and tree belts requested by the "
            "user. For roads, follow the center of the visible road surface, use text cues "
            "such as left/right/top/bottom, orientation, shape and connectivity, represent "
            "endpoints and intersections as nodes, and split roads at every shared "
            "intersection. For farmland and tree belts, infer conservative polygon "
            "boundaries from both the visible land-cover boundary and the user's location "
            "description. Do not classify every field, canopy or green area in the image; "
            "include only regions needed by the requested scene. If a correction image is "
            "provided, it is a second-round user correction and overrides the initial "
            "interpretation: closed semi-transparent polygons denote farmland/grass and "
            "elongated polygons or marked bands denote tree belts. Never treat an isolated "
            "point as an exact individual-tree coordinate. Return final pixel geometry, not "
            "color thresholds, masks, skeletons or other image-processing parameters."
        )
        geometry_user_instruction = (
            "When a new agricultural scene is requested, return a complete "
            "road_scene_request. Preserve map metadata supplied by the user or locked "
            "template. Use the attached image's actual dimensions with a top-left pixel "
            "origin; x increases rightward and y increases downward. Create nodes only "
            "for endpoints and intersections of the road network designated by the user. "
            "Each road must have a unique "
            "road_id, valid start_node and end_node, positive width_m, surface, and a "
            "sparse pixel_polyline. The first and last polyline points must equal the "
            "corresponding node pixels. A road passing through an intersection must be "
            "split into separate road records sharing the same junction node. Do not "
            "add roads or vegetation outside the user's requested scene. If the request "
            "selects one road from a larger visible network, ignore the remaining roads. "
            "For vegetation_request, use grass_regions for user-described farmland or "
            "non-colliding ground fill and tree_regions for user-described tree belts. "
            "Boundaries must be visually supportable and spatially consistent with the "
            "description. When a correction image exists, trace its valid correction "
            "polygons or bands, preserve concave corners, and ignore unrelated marks. "
            "Do not convert a single point into an individual tree. "
            "Preserve the requested tree/grass class. Use grass visual_mode=mesh for "
            "actual boundary-conforming farmland surfaces; debug_area is only an "
            "explicitly requested diagnostic overlay, never a rendered farmland substitute. "
            "For automatic roadside trees, set roadside_trees.enabled=true with "
            "band_width_m, density_per_100m2, grid_spacing_m and min_spacing_m. "
            "Their placement uses the finalized road geometry. CARLA tree props must "
            "exist in the target blueprint library; do not claim assets were imported. "
            "Do not output color thresholds, masks, skeletons or intermediate "
            "pixel-extraction settings."
        )
        system_prompt = (
            "You are the Plan Generation Agent of a CARLA agricultural simulation "
            "platform. Convert the user's request and attachments into one JSON "
            "task request. Return JSON only. Never invent file paths, CARLA map "
            "names, blueprint identifiers, measured values or experimental results. "
            "Pixel coordinates for scene geometry must be grounded in the attached image "
            "and the user's description; never fabricate geometry without visual support. "
            "Preserve units. Use only these task types: "
            "trajectory_tracking, full_process_navigation, configuration_only, "
            "scene_enrichment. If required information is unavailable, return "
            '{"needs_user_input": true, "questions": ["..."]}. '
            "The CrewAI Flow, not the language model, will execute agents. "
            + geometry_system_instruction
        )
        user_prompt = (
            "USER REQUEST:\n%s\n\n"
            "ATTACHMENTS AND FILE PREVIEWS:\n%s\n\n"
            "AVAILABLE AGENTS:\n%s\n\n"
            "OPTIONAL REQUEST TEMPLATE:\n%s\n\n"
            "TARGET JSON SCHEMA:\n%s\n\n"
            "EXECUTION FIELD CONTRACT:\n%s\n\n"
            "For an existing CARLA scene, do not add road_scene_request. "
            "For a supplied trajectory CSV, retain its exact path. "
            "Use road_scene_request only when a new agricultural scene must be generated. "
            "%s"
            % (
                str(payload.get("natural_language_request", "")),
                attachment_summary,
                json.dumps(self.registry.describe(), ensure_ascii=False, indent=2),
                json.dumps(template, ensure_ascii=False, indent=2),
                json.dumps(schema, ensure_ascii=False, indent=2),
                json.dumps(
                    self._execution_field_contract(),
                    ensure_ascii=False,
                    indent=2,
                ),
                geometry_user_instruction,
            )
        )
        return system_prompt, user_prompt

    @staticmethod
    def _execution_field_contract() -> dict[str, Any]:
        """返回领域执行器实际接受的字段，约束大模型不要臆造配置名。"""
        from agri_carla.controllers.controller_optimization import (
            ControllerSelfOptimizationConfig,
            TrackingObjectiveConfig,
        )
        from agri_carla.controllers.gear_state_machine import GearStateConfig
        from agri_carla.controllers.interface_agent import ControllerInterfaceConfig
        from agri_carla.controllers.pid import PIDConfig
        from agri_carla.controllers.pure_pursuit import PurePursuitConfig
        from agri_carla.simulation.full_process_experiment import (
            FullProcessExperimentConfig,
        )
        from agri_carla.simulation.tracking_experiment import TrackingExperimentConfig

        names = lambda cls: [item.name for item in fields(cls)]
        return {
            "machinery": ["type", "model_file", "output_file"],
            "trajectory_tracking": {
                "experiment": [
                    "trajectory_csv",
                    "processed_csv",
                    "parameter_optimization_set_dir",
                    "independent_validation_set_dir",
                    "processed_dataset_dir",
                    "config",
                ],
                "config": names(TrackingExperimentConfig),
                "controller": names(ControllerInterfaceConfig),
                "controller.pid": names(PIDConfig),
                "controller.pure_pursuit": names(PurePursuitConfig),
                "controller.gear": names(GearStateConfig),
                "controller_optimization": names(
                    ControllerSelfOptimizationConfig
                ),
                "controller_optimization.objective": names(
                    TrackingObjectiveConfig
                ),
            },
            "full_process_navigation": {
                "experiment": ["config"],
                "config": names(FullProcessExperimentConfig),
            },
        }

    @staticmethod
    def _has_visual_geometry(request: dict[str, Any]) -> bool:
        """判断请求是否包含需要独立视觉复核的道路或植被几何。"""
        return any(
            isinstance(request.get(key), dict)
            for key in ("road_scene_request", "vegetation_request")
        )

    def _verify_geometry_with_llm(
        self,
        payload: dict[str, Any],
        candidate: dict[str, Any],
        llm_config: LLMConfig,
    ) -> dict[str, Any]:
        """用第二次独立视觉调用检查道路和植被几何并返回完整请求。"""
        client = self.llm_client or build_llm_client(llm_config)
        system_prompt = (
            "You are the independent geometry verifier for a CARLA agricultural "
            "scene request. Compare the candidate JSON with the attached satellite "
            "image and the user's spatial scene description. The primary image is the "
            "original unannotated image; any separately identified correction image is "
            "a second-round user correction. Return "
            "only one complete "
            "corrected task-request JSON. "
            "Keep user-supplied map metadata unchanged. Correct only visually "
            "unsupported or geometrically inconsistent road centerlines, nodes and "
            "scene-region polygons. Do not add unrequested roads, vegetation, vehicles "
            "or experiment results."
        )
        user_prompt = (
            "ORIGINAL USER REQUEST:\n%s\n\n"
            "LOCKED REQUEST TEMPLATE:\n%s\n\n"
            "CANDIDATE TASK REQUEST:\n%s\n\n"
            "IMAGE INFORMATION:\n%s\n\n"
            "Verify that every user-designated road centerline is represented exactly "
            "once; each required intersection is one shared node; roads are split at "
            "that node; endpoints and bends follow the visible road-surface "
            "center; coordinates "
            "remain inside the image; and road widths and surfaces match the user's "
            "description. For vegetation, verify that every proposed farmland/grass or "
            "tree-belt polygon agrees with the location described by the user and has a "
            "visually supportable land-cover boundary. Do not expand recognition to all "
            "fields or canopy visible in the image. If a correction image is present, "
            "follow its closed farmland polygons and elongated tree bands and disregard "
            "the superseded first-pass boundary. Never interpret an isolated point as "
            "an individual tree. "
            "Grass intended "
            "as non-colliding ground fill must use visual_mode=debug_area. Return no "
            "intermediate image-processing configuration."
            % (
                str(payload.get("natural_language_request", "")),
                json.dumps(self._request_template(payload), ensure_ascii=False, indent=2),
                json.dumps(candidate, ensure_ascii=False, indent=2),
                self._attachment_summary(payload),
            )
        )
        verified = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=self._resolved_image_paths(payload),
        )
        if isinstance(verified.get("structured_request"), dict):
            verified = verified["structured_request"]
        self._record_llm_candidate("geometry_verification", verified, llm_config)
        if verified.get("needs_user_input"):
            raise NeedsUserInputError(
                verified.get("questions")
                or ["The geometry verifier requires additional user information."]
            )
        return self._merge_request_template(payload, verified)

    def _repair_with_llm(
        self,
        payload: dict[str, Any],
        candidate: dict[str, Any],
        errors: list[str],
        llm_config: LLMConfig,
    ) -> dict[str, Any]:
        """将校验错误反馈给大模型，生成完整的修正候选请求。"""
        client = self.llm_client or build_llm_client(llm_config)
        geometry_repair_instructions: list[str] = []
        if isinstance(candidate.get("road_scene_request"), dict):
            geometry_repair_instructions.append(
                "Inspect the attached image and correct road_scene_request directly. "
                "Keep one "
                "node for each designated endpoint or required intersection, split every "
                "road at a shared intersection, and make each sparse pixel_polyline "
                "follow the visible annotation or user-designated road-surface center. "
                "Its first and last points must "
                "equal the referenced node pixels. Keep all coordinates inside the "
                "image and preserve the stated widths and surfaces. Return no color "
                "thresholds, masks, skeletons or intermediate extraction settings. "
            )
        if isinstance(candidate.get("vegetation_request"), dict):
            geometry_repair_instructions.append(
                "Inspect the attached image and correct vegetation_request directly. "
                "Use the original satellite image and the user's spatial description to "
                "represent only requested grass/farmland and tree-belt regions. If a "
                "correction image is present, its closed farmland polygons and elongated "
                "tree bands override the first-pass geometry. Represent each requested "
                "region exactly once with a non-degenerate, visually grounded polygon. "
                "Keep pixel coordinates inside the image, never create an individual "
                "tree from one point, "
                "and use visual_mode=debug_area for non-colliding grass fill. "
            )
        geometry_repair_instruction = "".join(geometry_repair_instructions)
        system_prompt = (
            "Repair a CARLA agricultural task-request JSON object. Return only "
            "the complete corrected JSON object. Do not invent missing user data. "
            + geometry_repair_instruction
            + "If a correction requires unknown information, return "
            '{"needs_user_input": true, "questions": ["..."]}.'
        )
        user_prompt = (
            "ORIGINAL USER REQUEST:\n%s\n\n"
            "CANDIDATE JSON:\n%s\n\n"
            "VALIDATION ERRORS:\n%s\n\n"
            "ATTACHMENTS:\n%s"
            % (
                str(payload.get("natural_language_request", "")),
                json.dumps(candidate, ensure_ascii=False, indent=2),
                "\n".join("- " + item for item in errors),
                self._attachment_summary(payload),
            )
        )
        repaired = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=self._resolved_image_paths(payload),
        )
        self._record_llm_candidate(
            "repair",
            repaired.get("structured_request", repaired),
            llm_config,
        )
        if repaired.get("needs_user_input"):
            raise NeedsUserInputError(
                repaired.get("questions")
                or ["Additional user information is required."]
            )
        return self._merge_request_template(
            payload,
            repaired.get("structured_request", repaired),
        )

    def validate_request(
        self,
        request: dict[str, Any],
        *,
        strict_paths: bool,
        require_execution_identifiers: bool = True,
    ) -> tuple[list[str], list[str]]:
        """校验任务字段、输入路径及执行所需标识符。

        错误会阻止执行；警告只说明例如地图需要预先导入 CARLA 等前置假设。
        ``strict_paths`` 用于控制是否检查本地文件实际存在。
        """
        errors: list[str] = []
        warnings: list[str] = []
        task_type = request.get("task_type")
        if not task_type:
            errors.append("Missing task_type.")
        elif task_type not in SUPPORTED_TASK_TYPES:
            errors.append("Unsupported task_type: %s." % task_type)

        if task_type == "trajectory_tracking":
            experiment = request.get("experiment")
            if not isinstance(experiment, dict):
                errors.append("Missing experiment object for trajectory_tracking.")
            else:
                has_csv = bool(experiment.get("trajectory_csv"))
                config = experiment.get("config")
                if not isinstance(config, dict):
                    errors.append("Missing experiment.config.")
                else:
                    optimization = config.get("controller_optimization")
                    optimization_enabled = bool(
                        isinstance(optimization, dict)
                        and optimization.get("enabled")
                    )
                    if optimization_enabled:
                        for key in (
                            "parameter_optimization_set_dir",
                            "independent_validation_set_dir",
                        ):
                            if not experiment.get(key):
                                errors.append(f"Missing experiment.{key}.")
                    elif not has_csv:
                        errors.append("Missing experiment.trajectory_csv.")
                    if require_execution_identifiers and not config.get("carla_map"):
                        errors.append("Missing experiment.config.carla_map.")
                    if require_execution_identifiers and not config.get("vehicle_bp"):
                        errors.append("Missing experiment.config.vehicle_bp.")

        if task_type == "full_process_navigation":
            experiment = request.get("experiment")
            config = experiment.get("config") if isinstance(experiment, dict) else None
            if not isinstance(config, dict):
                errors.append(
                    "Missing experiment.config for full_process_navigation."
                )
            else:
                if require_execution_identifiers and not config.get("carla_map"):
                    errors.append("Missing experiment.config.carla_map.")
                if require_execution_identifiers and not config.get("vehicle_bp"):
                    errors.append("Missing experiment.config.vehicle_bp.")

        errors.extend(self._execution_contract_errors(request))

        road_request = request.get("road_scene_request")
        if road_request is not None:
            if not isinstance(road_request, dict):
                errors.append("road_scene_request must be an object.")
            else:
                # 原始卫星图是首次多模态解析的输入，由 _obtain_structured_request
                # 在调用模型前检查；已经确认并保存的结构化道路几何可以脱离原图
                # 复跑，因此此处只校验生成 CARLA 文件真正必需的字段。
                for key in ("map_name", "meters_per_pixel", "nodes", "roads"):
                    if road_request.get(key) in (None, "", []):
                        errors.append("Missing road_scene_request.%s." % key)
                for index, road in enumerate(road_request.get("roads", [])):
                    if not isinstance(road, dict):
                        errors.append(
                            "road_scene_request.roads[%d] must be an object." % index
                        )
                        continue
                    for key in (
                        "road_id",
                        "start_node",
                        "end_node",
                        "width_m",
                        "pixel_polyline",
                    ):
                        if road.get(key) in (None, "", []):
                            errors.append(
                                "Missing road_scene_request.roads[%d].%s."
                                % (index, key)
                            )
                errors.extend(self._road_scene_geometry_errors(road_request))

        vegetation_request = request.get("vegetation_request")
        if vegetation_request is not None:
            if not isinstance(vegetation_request, dict):
                errors.append("vegetation_request must be an object.")
            else:
                errors.extend(
                    self._vegetation_geometry_errors(vegetation_request)
                )

        if strict_paths:
            for label, path, kind in self._input_paths(request):
                resolved = self._resolve_path(path)
                exists = resolved.is_dir() if kind == "directory" else resolved.is_file()
                if not exists:
                    errors.append("Input %s does not exist: %s." % (label, path))
            if task_type == "trajectory_tracking":
                experiment = request.get("experiment", {})
                if not isinstance(experiment, dict):
                    experiment = {}
                config = experiment.get("config", {})
                if not isinstance(config, dict):
                    config = {}
                optimization = config.get("controller_optimization", {})
                if isinstance(optimization, dict) and optimization.get("enabled"):
                    for key in (
                        "parameter_optimization_set_dir",
                        "independent_validation_set_dir",
                    ):
                        if not experiment.get(key):
                            continue
                        dataset_dir = self._resolve_path(experiment[key])
                        if dataset_dir.is_dir():
                            csv_count = sum(
                                1
                                for path in dataset_dir.glob("*.csv")
                                if path.is_file()
                            )
                            if csv_count < 2:
                                errors.append(
                                    f"experiment.{key} must contain at least 2 "
                                    f"trajectory CSV files; found {csv_count}."
                                )

        if request.get("machinery") is None:
            warnings.append(
                "No machinery configuration step was requested; the existing "
                "CARLA blueprint parameters will be used."
            )
        if task_type in {"trajectory_tracking", "full_process_navigation"}:
            config = request.get("experiment", {}).get("config", {})
            if config.get("carla_map") and request.get("road_scene_request") is None:
                warnings.append(
                    "The task assumes that CARLA map '%s' is already imported."
                    % config["carla_map"]
                )
            if config.get("carla_map") and request.get("road_scene_request") is not None:
                warnings.append(
                    "SGA generates CARLA-ready files but does not import the map. "
                    "EEA can run only if '%s' has already been imported into CARLA."
                    % config["carla_map"]
                )
        return errors, warnings

    @classmethod
    def _execution_contract_errors(
        cls,
        request: dict[str, Any],
    ) -> list[str]:
        """在规划阶段拒绝领域执行器不会读取或会导致构造失败的字段。"""
        errors: list[str] = []
        contract = cls._execution_field_contract()

        machinery = request.get("machinery")
        if machinery is not None:
            if not isinstance(machinery, dict):
                errors.append("machinery must be an object.")
            else:
                errors.extend(
                    cls._unknown_field_errors(
                        "machinery",
                        machinery,
                        contract["machinery"],
                    )
                )
                if not str(machinery.get("type", "")).strip():
                    errors.append("Missing machinery.type.")
                if not str(machinery.get("output_file", "")).strip():
                    errors.append("Missing machinery.output_file.")

        task_type = request.get("task_type")
        if task_type not in {"trajectory_tracking", "full_process_navigation"}:
            return errors

        experiment = request.get("experiment")
        if not isinstance(experiment, dict):
            return errors
        task_contract = contract[task_type]
        errors.extend(
            cls._unknown_field_errors(
                "experiment",
                experiment,
                task_contract["experiment"],
            )
        )
        config = experiment.get("config")
        if not isinstance(config, dict):
            return errors
        errors.extend(
            cls._unknown_field_errors(
                "experiment.config",
                config,
                task_contract["config"],
            )
        )

        if task_type == "full_process_navigation":
            try:
                from agri_carla.simulation.full_process_experiment import (
                    FullProcessExperimentConfig,
                )

                FullProcessExperimentConfig(**config)
            except (TypeError, ValueError) as exc:
                errors.append(
                    "Invalid full-process execution configuration: %s" % exc
                )
            return errors

        controller = config.get("controller")
        if controller is not None:
            if not isinstance(controller, dict):
                errors.append("experiment.config.controller must be an object.")
            else:
                errors.extend(
                    cls._unknown_field_errors(
                        "experiment.config.controller",
                        controller,
                        task_contract["controller"],
                    )
                )
                for key in ("pid", "pure_pursuit", "gear"):
                    nested = controller.get(key)
                    if nested is None:
                        continue
                    if not isinstance(nested, dict):
                        errors.append(
                            "experiment.config.controller.%s must be an object."
                            % key
                        )
                        continue
                    errors.extend(
                        cls._unknown_field_errors(
                            "experiment.config.controller.%s" % key,
                            nested,
                            task_contract["controller.%s" % key],
                        )
                    )

        optimization = config.get("controller_optimization")
        if optimization is not None:
            if not isinstance(optimization, dict):
                errors.append(
                    "experiment.config.controller_optimization must be an object."
                )
            else:
                errors.extend(
                    cls._unknown_field_errors(
                        "experiment.config.controller_optimization",
                        optimization,
                        task_contract["controller_optimization"],
                    )
                )
                objective = optimization.get("objective")
                if objective is not None:
                    if not isinstance(objective, dict):
                        errors.append(
                            "experiment.config.controller_optimization.objective "
                            "must be an object."
                        )
                    else:
                        errors.extend(
                            cls._unknown_field_errors(
                                "experiment.config.controller_optimization.objective",
                                objective,
                                task_contract[
                                    "controller_optimization.objective"
                                ],
                            )
                        )

        try:
            from agri_carla.controllers.interface_agent import (
                ControllerInterfaceAgent,
            )
            from agri_carla.simulation.tracking_experiment import (
                TrackingExperimentConfig,
            )

            TrackingExperimentConfig.from_dict(config)
            ControllerInterfaceAgent.self_optimization_config(optimization)
        except (TypeError, ValueError) as exc:
            errors.append("Invalid tracking execution configuration: %s" % exc)
        return errors

    @staticmethod
    def _unknown_field_errors(
        label: str,
        data: dict[str, Any],
        allowed: list[str],
    ) -> list[str]:
        unknown = sorted(set(data) - set(allowed))
        if not unknown:
            return []
        return [
            "%s contains unsupported execution fields: %s."
            % (label, ", ".join(unknown))
        ]

    @staticmethod
    def _road_scene_geometry_errors(
        road_request: dict[str, Any],
    ) -> list[str]:
        """校验模型生成道路几何的坐标、引用、端点和图连通性。"""
        errors: list[str] = []
        for key in ("image_width_px", "image_height_px", "geo_anchor"):
            if road_request.get(key) in (None, "", []):
                errors.append("Missing road_scene_request.%s." % key)

        try:
            width = float(road_request.get("image_width_px", 0))
            height = float(road_request.get("image_height_px", 0))
        except (TypeError, ValueError):
            width = 0.0
            height = 0.0
        if width <= 0.0 or height <= 0.0:
            errors.append(
                "road_scene_request image dimensions must be positive numbers."
            )

        def valid_point(value: Any) -> bool:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return False
            try:
                x = float(value[0])
                y = float(value[1])
            except (TypeError, ValueError):
                return False
            if not math.isfinite(x) or not math.isfinite(y):
                return False
            if width > 0.0 and height > 0.0:
                return 0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0
            return True

        nodes = road_request.get("nodes") or []
        node_map: dict[str, tuple[float, float]] = {}
        for index, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(
                    "road_scene_request.nodes[%d] must be an object." % index
                )
                continue
            node_id = str(node.get("node_id", "")).strip()
            if not node_id:
                errors.append(
                    "Missing road_scene_request.nodes[%d].node_id." % index
                )
            elif node_id in node_map:
                errors.append("Duplicate road node identifier: %s." % node_id)
            if not valid_point(node.get("pixel")):
                errors.append(
                    "road_scene_request.nodes[%d].pixel must be an in-image "
                    "[pixel_x, pixel_y] point." % index
                )
            elif node_id:
                node_map[node_id] = (
                    float(node["pixel"][0]),
                    float(node["pixel"][1]),
                )

        roads = road_request.get("roads") or []
        road_ids = set()
        graph_edges: list[tuple[str, str]] = []
        referenced_nodes: set[str] = set()
        endpoint_tolerance = float(
            (road_request.get("geometry_validation") or {}).get(
                "endpoint_tolerance_px",
                25.0,
            )
        )
        for index, road in enumerate(roads):
            if not isinstance(road, dict):
                continue
            road_id = str(road.get("road_id", "")).strip()
            if road_id in road_ids:
                errors.append("Duplicate road identifier: %s." % road_id)
            elif road_id:
                road_ids.add(road_id)
            start_node = str(road.get("start_node", ""))
            end_node = str(road.get("end_node", ""))
            for label, node_id in (("start_node", start_node), ("end_node", end_node)):
                if node_id and node_id not in node_map:
                    errors.append(
                        "road_scene_request.roads[%d].%s refers to undefined node %s."
                        % (index, label, node_id)
                    )
            if start_node in node_map and end_node in node_map:
                graph_edges.append((start_node, end_node))
                referenced_nodes.update((start_node, end_node))
            try:
                width_m = float(road.get("width_m", 0.0))
            except (TypeError, ValueError):
                width_m = 0.0
            if width_m <= 0.0:
                errors.append(
                    "road_scene_request.roads[%d].width_m must be positive."
                    % index
                )
            if not str(road.get("surface", "")).strip():
                errors.append(
                    "Missing road_scene_request.roads[%d].surface." % index
                )
            polyline = road.get("pixel_polyline") or []
            if not isinstance(polyline, list) or len(polyline) < 2:
                errors.append(
                    "road_scene_request.roads[%d].pixel_polyline requires at "
                    "least two points." % index
                )
                continue
            invalid_points = [
                point_index
                for point_index, point in enumerate(polyline)
                if not valid_point(point)
            ]
            if invalid_points:
                errors.append(
                    "road_scene_request.roads[%d].pixel_polyline has invalid "
                    "in-image points at indices %s."
                    % (index, invalid_points)
                )
                continue
            if start_node in node_map:
                gap = math.hypot(
                    float(polyline[0][0]) - node_map[start_node][0],
                    float(polyline[0][1]) - node_map[start_node][1],
                )
                if gap > endpoint_tolerance:
                    errors.append(
                        "road_scene_request.roads[%d] starts %.1f px away from "
                        "its declared node %s." % (index, gap, start_node)
                    )
            if end_node in node_map:
                gap = math.hypot(
                    float(polyline[-1][0]) - node_map[end_node][0],
                    float(polyline[-1][1]) - node_map[end_node][1],
                )
                if gap > endpoint_tolerance:
                    errors.append(
                        "road_scene_request.roads[%d] ends %.1f px away from "
                        "its declared node %s." % (index, gap, end_node)
                    )

        require_single = bool(
            (road_request.get("topology_validation") or {}).get(
                "require_single_component",
                True,
            )
        )
        if require_single and graph_edges:
            adjacency: dict[str, set[str]] = {
                node_id: set() for node_id in referenced_nodes
            }
            for left, right in graph_edges:
                adjacency[left].add(right)
                adjacency[right].add(left)
            remaining = set(adjacency)
            components = 0
            while remaining:
                components += 1
                stack = [remaining.pop()]
                while stack:
                    current = stack.pop()
                    neighbors = adjacency[current] & remaining
                    remaining -= neighbors
                    stack.extend(neighbors)
            if components > 1:
                errors.append(
                    "road_scene_request contains %d disconnected road-network "
                    "components; marked intersections must be represented by a "
                    "shared node." % components
                )
        return errors

    @staticmethod
    def _vegetation_geometry_errors(
        vegetation_request: dict[str, Any],
    ) -> list[str]:
        """校验模型直接给出的树木与草地区域多边形。"""
        errors: list[str] = []
        region_count = 0
        try:
            image_width = float(vegetation_request.get("image_width_px", 0))
            image_height = float(vegetation_request.get("image_height_px", 0))
        except (TypeError, ValueError):
            image_width = 0.0
            image_height = 0.0

        if image_width <= 0.0 or image_height <= 0.0:
            errors.append(
                "vegetation_request image dimensions must be positive numbers."
            )
        try:
            meters_per_pixel = float(
                vegetation_request.get("meters_per_pixel", 0)
            )
        except (TypeError, ValueError):
            meters_per_pixel = 0.0
        if meters_per_pixel <= 0.0:
            errors.append(
                "vegetation_request.meters_per_pixel must be a positive number."
            )

        for collection_name in ("tree_regions", "grass_regions"):
            regions = vegetation_request.get(collection_name, [])
            if not isinstance(regions, list):
                errors.append("vegetation_request.%s must be an array." % collection_name)
                continue
            region_count += len(regions)
            for index, region in enumerate(regions):
                prefix = "vegetation_request.%s[%d]" % (
                    collection_name,
                    index,
                )
                if not isinstance(region, dict):
                    errors.append("%s must be an object." % prefix)
                    continue
                coordinate_mode = str(
                    region.get("coordinate_mode", "pixel")
                ).lower()
                if coordinate_mode not in {"pixel", "world"}:
                    errors.append(
                        "%s.coordinate_mode must be 'pixel' or 'world'."
                        % prefix
                    )
                polygon = region.get("polygon")
                if not isinstance(polygon, list) or len(polygon) < 3:
                    errors.append(
                        "%s.polygon must contain at least three points."
                        % prefix
                    )
                    continue
                for point_index, point in enumerate(polygon):
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        errors.append(
                            "%s.polygon[%d] must be [x, y]."
                            % (prefix, point_index)
                        )
                        continue
                    try:
                        x = float(point[0])
                        y = float(point[1])
                    except (TypeError, ValueError):
                        errors.append(
                            "%s.polygon[%d] must contain numeric coordinates."
                            % (prefix, point_index)
                        )
                        continue
                    if not math.isfinite(x) or not math.isfinite(y):
                        errors.append(
                            "%s.polygon[%d] contains a non-finite coordinate."
                            % (prefix, point_index)
                        )
                    elif (
                        coordinate_mode == "pixel"
                        and image_width > 0.0
                        and image_height > 0.0
                        and not (
                            0.0 <= x <= image_width - 1.0
                            and 0.0 <= y <= image_height - 1.0
                        )
                    ):
                        errors.append(
                            "%s.polygon[%d] lies outside the image."
                            % (prefix, point_index)
                        )
                valid_polygon = []
                for point in polygon:
                    try:
                        valid_polygon.append((float(point[0]), float(point[1])))
                    except (IndexError, TypeError, ValueError):
                        valid_polygon = []
                        break
                if len(valid_polygon) >= 3:
                    doubled_area = abs(
                        sum(
                            left[0] * right[1] - right[0] * left[1]
                            for left, right in zip(
                                valid_polygon,
                                valid_polygon[1:] + valid_polygon[:1],
                            )
                        )
                    )
                    if doubled_area <= 2.0:
                        errors.append(
                            "%s.polygon must enclose a non-zero area." % prefix
                        )
        if region_count == 0 and not vegetation_request.get("roadside_trees", {}).get("enabled", False):
            errors.append(
                "vegetation_request requires at least one tree or grass region."
            )
        return errors

    def build_execution_plan(
        self,
        request: dict[str, Any],
    ) -> ExecutionPlan:
        """根据任务内容生成顶层智能体步骤及其依赖关系。

        场景和植被步骤先执行，MCA 在场景准备后生成参数模板，CIA 随后校验
        控制配置，EEA 最后运行实验。CIA 与 EEA 同层，EEA 在运行过程中
        继续调用 CIA 提供逐帧控制服务。
        """
        task_type = str(request["task_type"])
        steps: list[TaskStep] = []
        scene_step_ids: list[str] = []

        # 仅在请求包含对应功能时加入步骤，使用户可以单独调用部分智能体。
        if request.get("road_scene_request"):
            steps.append(
                TaskStep(
                    "sga_scene",
                    "SGA",
                    "generate_scene",
                    max_attempts=2,
                )
            )
            scene_step_ids.append("sga_scene")

        if request.get("vegetation_request"):
            steps.append(
                TaskStep(
                    "sga_vegetation",
                    "SGA",
                    "place_vegetation",
                    depends_on=list(scene_step_ids),
                    max_attempts=1,
                )
            )
            scene_step_ids.append("sga_vegetation")

        machinery_dependencies = list(scene_step_ids)
        if request.get("machinery"):
            steps.append(
                TaskStep(
                    "mca_configuration",
                    "MCA",
                    "generate_machinery_template",
                    depends_on=machinery_dependencies,
                )
            )

        experiment_dependencies = list(scene_step_ids)
        if request.get("machinery"):
            experiment_dependencies.append("mca_configuration")

        if task_type in {"trajectory_tracking", "full_process_navigation"}:
            steps.append(
                TaskStep(
                    "cia_configuration",
                    "CIA",
                    "prepare_controller",
                    depends_on=list(experiment_dependencies),
                )
            )
            experiment_dependencies.append("cia_configuration")

        if task_type == "trajectory_tracking":
            steps.append(
                TaskStep(
                    "eea_tracking",
                    "EEA",
                    "run_trajectory_tracking",
                    depends_on=experiment_dependencies,
                    max_attempts=2,
                )
            )
        elif task_type == "full_process_navigation":
            steps.append(
                TaskStep(
                    "eea_full_process",
                    "EEA",
                    "run_full_process_navigation",
                    depends_on=experiment_dependencies,
                    max_attempts=2,
                )
            )

        return ExecutionPlan(task_type=task_type, steps=steps)

    def _apply_attachment_defaults(
        self,
        request: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """在含义唯一时，用附件路径补全大模型遗漏的文件字段。"""
        request = copy.deepcopy(request)
        data_files = self._data_paths(payload)
        # road_scene_request.image_path 始终指向原始卫星图，而不是第二轮可选的
        # 修订标注图。修订图只供多模态模型校正候选几何。
        image_paths = self._primary_image_paths(payload)
        if request.get("task_type") == "trajectory_tracking":
            experiment = request.setdefault("experiment", {})
            if not experiment.get("trajectory_csv"):
                csv_paths = [path for path in data_files if Path(path).suffix.lower() == ".csv"]
                if len(csv_paths) == 1:
                    experiment["trajectory_csv"] = csv_paths[0]
        road_request = request.get("road_scene_request")
        if isinstance(road_request, dict) and not road_request.get("image_path"):
            if len(image_paths) == 1:
                road_request["image_path"] = image_paths[0]
        return request

    def _attach_scene_review(
        self,
        payload: dict[str, Any],
        request: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """生成场景候选叠加图并写入确认状态。

        首次多模态规划产生 ``candidate`` 状态。只有用户随后提交保存的结构化
        请求并显式确认，CrewAI Flow 才会调用 SGA。
        """
        request = copy.deepcopy(request)
        primary_images = self._primary_image_paths(payload)
        road_request = request.get("road_scene_request")
        image_path = ""
        if isinstance(road_request, dict):
            image_path = str(road_request.get("image_path", ""))
        if not image_path and primary_images:
            image_path = primary_images[0]

        map_name = "agricultural_scene"
        if isinstance(road_request, dict) and road_request.get("map_name"):
            map_name = str(road_request["map_name"])
        else:
            vegetation = request.get("vegetation_request")
            if isinstance(vegetation, dict) and vegetation.get("carla_map"):
                map_name = str(vegetation["carla_map"])

        review_options = payload.get("scene_review")
        review_options = review_options if isinstance(review_options, dict) else {}
        preview_path = str(review_options.get("preview_path", "")).strip()
        if preview_path:
            preview = self._resolve_path(preview_path)
        else:
            safe_name = "".join(
                char if char.isalnum() or char in "_.-" else "_"
                for char in map_name
            )
            geometry_digest = hashlib.sha256(
                json.dumps(
                    {
                        "roads": request.get("road_scene_request"),
                        "vegetation": request.get("vegetation_request"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()[:10]
            preview = (
                self.project_root
                / "outputs"
                / "scene_reviews"
                / (safe_name + "_" + geometry_digest + "_candidate_preview.png")
            )

        preview_error = ""
        if image_path:
            try:
                render_scene_candidate_preview(
                    self._resolve_path(image_path),
                    request,
                    preview,
                )
            except (OSError, ValueError) as exc:
                preview_error = str(exc)
        else:
            preview_error = "No primary satellite image is available."

        review = {
            "status": "candidate",
            "requires_confirmation": True,
            "revision_round": 2 if self._correction_image_paths(payload) else 1,
            "original_image_path": image_path,
            "correction_image_paths": self._correction_image_paths(payload),
            "preview_path": self._portable_path(preview) if not preview_error else "",
            "preview_error": preview_error,
            "revision_guidance": (
                "If the candidate is inaccurate, rerun PGA with a more precise spatial "
                "description and optionally a correction image. If it is correct, execute "
                "this saved structured request with --confirm-scene."
            ),
        }
        request["scene_review"] = review
        return request, review

    def _portable_path(self, path: Path) -> str:
        """项目内产物使用相对路径，项目外路径保留绝对形式。"""
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path.resolve())

    def _request_template(self, payload: dict[str, Any]) -> dict[str, Any]:
        """读取内嵌或文件形式的请求模板。"""
        template = payload.get("request_template")
        if isinstance(template, dict):
            return template
        if isinstance(template, str) and template:
            return read_yaml(self._resolve_path(template))
        return {}

    def _merge_request_template(
        self,
        payload: dict[str, Any],
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """按 locked 或 defaults 策略合并模板与大模型候选值。"""
        template = self._request_template(payload)
        if not template:
            return candidate
        policy = str(payload.get("template_policy", "locked")).lower()
        if policy == "locked":
            return self._deep_merge(candidate, template)
        if policy == "defaults":
            return self._deep_merge(template, candidate)
        raise ValueError(
            "template_policy must be 'locked' or 'defaults', not %s." % policy
        )

    @classmethod
    def _deep_merge(
        cls,
        base: dict[str, Any],
        overlay: dict[str, Any],
    ) -> dict[str, Any]:
        """递归合并嵌套字典，并通过深拷贝避免修改原输入。"""
        merged = copy.deepcopy(base)
        for key, value in overlay.items():
            if (
                isinstance(value, dict)
                and isinstance(merged.get(key), dict)
            ):
                merged[key] = cls._deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    def _attachment_summary(self, payload: dict[str, Any]) -> str:
        """生成提供给大模型的附件存在性和 CSV 小样摘要。"""
        summaries: list[str] = []
        primary_images = set(self._primary_image_paths(payload))
        correction_images = set(self._correction_image_paths(payload))
        for image_path in self._image_paths(payload):
            path = self._resolve_path(image_path)
            role = "PRIMARY_RAW_SATELLITE_IMAGE"
            if image_path in correction_images:
                role = "OPTIONAL_SECOND_ROUND_CORRECTION_IMAGE"
            elif image_path not in primary_images:
                role = "SUPPORTING_IMAGE"
            summary = "IMAGE role=%s path=%s exists=%s" % (
                role,
                image_path,
                path.is_file(),
            )
            if path.is_file():
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        summary += (
                            " width_px=%d height_px=%d coordinate_origin=top-left"
                            % (image.width, image.height)
                        )
                except (ModuleNotFoundError, OSError):
                    pass
            summaries.append(summary)
        for data_path in self._data_paths(payload):
            path = self._resolve_path(data_path)
            summary = "DATA path=%s exists=%s" % (data_path, path.is_file())
            if path.is_file() and path.suffix.lower() == ".csv":
                summary += "\n" + self._csv_preview(path)
            summaries.append(summary)
        return "\n\n".join(summaries) if summaries else "(none)"

    @staticmethod
    def _csv_preview(path: Path, max_rows: int = 5) -> str:
        """读取少量 CSV 行供字段识别，不把完整实测数据发送给模型。"""
        for encoding in ("utf-8-sig", "gbk"):
            try:
                with path.open("r", encoding=encoding, newline="") as stream:
                    rows = []
                    reader = csv.reader(stream)
                    for index, row in enumerate(reader):
                        rows.append(row)
                        if index >= max_rows:
                            break
                return "CSV preview:\n" + "\n".join(
                    ",".join(str(value) for value in row) for row in rows
                )
            except UnicodeDecodeError:
                continue
        return "CSV preview unavailable because the file encoding is unsupported."

    def _input_paths(
        self,
        request: dict[str, Any],
    ) -> list[tuple[str, str, str]]:
        """收集结构化请求中需要进行存在性校验的文件和目录。"""
        paths: list[tuple[str, str, str]] = []
        road_request = request.get("road_scene_request")
        if isinstance(road_request, dict) and road_request.get("image_path"):
            paths.append(
                ("road_scene_request.image_path", str(road_request["image_path"]), "file")
            )
        experiment = request.get("experiment")
        if isinstance(experiment, dict):
            for key in ("trajectory_csv",):
                if experiment.get(key):
                    paths.append(("experiment." + key, str(experiment[key]), "file"))
            for key in (
                "parameter_optimization_set_dir",
                "independent_validation_set_dir",
            ):
                if experiment.get(key):
                    paths.append(("experiment." + key, str(experiment[key]), "directory"))
        if request.get("map_task_registry"):
            paths.append(
                ("map_task_registry", str(request["map_task_registry"]), "file")
            )
        return paths

    def _resolve_path(self, path: str | Path) -> Path:
        """将相对输入路径解析到项目根目录。"""
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    def _resolved_image_paths(self, payload: dict[str, Any]) -> list[str]:
        """返回供多模态 API 读取的绝对图像路径。"""
        return [
            str(self._resolve_path(path))
            for path in self._image_paths(payload)
        ]

    def _record_llm_candidate(
        self,
        stage: str,
        candidate: dict[str, Any],
        config: LLMConfig,
    ) -> None:
        """记录模型、温度和响应摘要，不保存 API 密钥。"""
        serialized = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        self._llm_trace.append(
            {
                "stage": stage,
                "provider": config.provider,
                "model": config.model,
                "temperature": config.temperature,
                "response_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )

    @staticmethod
    def _image_paths(payload: dict[str, Any]) -> list[str]:
        """按原始图、普通附件、修订图的顺序返回所有视觉输入。"""
        paths = PlanGenerationAgent._primary_image_paths(payload)
        paths.extend(PlanGenerationAgent._correction_image_paths(payload))
        return list(dict.fromkeys(paths))

    @staticmethod
    def _primary_image_paths(payload: dict[str, Any]) -> list[str]:
        """返回首次场景理解使用的原始卫星图。"""
        return [str(path) for path in payload.get("image_paths", [])]

    @staticmethod
    def _correction_image_paths(payload: dict[str, Any]) -> list[str]:
        """返回候选结果不满意时由用户补充的第二轮修订标注图。"""
        return [str(path) for path in payload.get("correction_image_paths", [])]

    @staticmethod
    def _data_paths(payload: dict[str, Any]) -> list[str]:
        """返回轨迹和参数等非图像输入文件。"""
        return [str(path) for path in payload.get("data_files", [])]
